"""DINOv3 + Cascade Mask R-CNN family model construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .contracts import InstanceSegmentationSettings
from .preprocessing import unpack_size


class ChannelsLastPyramid(torch.nn.Module):
    """Preserve the validated channels-last SimpleFeaturePyramid execution."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for stage in getattr(self.backbone, "stages", []):
            stage.to(memory_format=torch.channels_last)

    @property
    def size_divisibility(self) -> int:
        return int(getattr(self.backbone, "size_divisibility", 0))

    @property
    def padding_constraints(self) -> dict[str, int]:
        return getattr(self.backbone, "padding_constraints", {})

    def output_shape(self):
        return self.backbone.output_shape()

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        bottom_up = self.backbone.net(value)
        features = bottom_up[self.backbone.in_feature]
        if features.is_cuda and features.ndim == 4:
            features = features.contiguous(memory_format=torch.channels_last)
        results = [stage(features) for stage in self.backbone.stages]
        if self.backbone.top_block is not None:
            in_feature = self.backbone.top_block.in_feature
            if in_feature in bottom_up:
                top_block_input = bottom_up[in_feature]
            else:
                index = self.backbone._out_features.index(in_feature)
                top_block_input = results[index]
            results.extend(self.backbone.top_block(top_block_input))
        return dict(zip(self.backbone._out_features, results))


def _patch_inference_thresholds(
    config: Any, settings: InstanceSegmentationSettings
) -> None:
    if hasattr(config.model.roi_heads, "box_predictors"):
        for predictor in config.model.roi_heads.box_predictors:
            predictor.test_score_thresh = settings.score_threshold
            predictor.test_nms_thresh = settings.nms_threshold
            predictor.test_topk_per_image = settings.topk_per_image
    if hasattr(config.model, "proposal_generator"):
        config.model.proposal_generator.pre_nms_topk = (
            20_000,
            int(settings.rpn_pre_nms_topk_test),
        )
        config.model.proposal_generator.post_nms_topk = (
            2_000,
            int(settings.rpn_post_nms_topk_test),
        )
        config.model.proposal_generator.nms_thresh = float(settings.rpn_nms_threshold)


def build_instance_segmenter(
    settings: InstanceSegmentationSettings,
    *,
    device: str,
) -> torch.nn.Module:
    """Build the audited fixed-batch TRT family variant as one cohesive unit."""

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyCall as L
    from detectron2.config import LazyConfig, instantiate
    from detectron2.modeling import DINOv3Backbone

    from .atss import ATSSRPN
    from .trt.runtime_adapter import TensorRTBackboneAdapter

    config_path = settings.config_path or Path(__file__).with_name("config.py")
    for path, label in (
        (settings.checkpoint, "detector checkpoint"),
        (config_path, "Detectron config"),
        (settings.trt_backbone_engine, "TensorRT backbone engine"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    config = LazyConfig.load(str(config_path))
    target_h, target_w = unpack_size(settings.target_size)
    config.model.roi_heads.num_classes = 1
    config.model.backbone.square_pad = target_h if target_h == target_w else 0
    config.model.backbone.net = L(DINOv3Backbone)(
        img_size=max(target_h, target_w),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layers_to_use=1,
        out_feature="last_feat",
        weights=settings.backbone_weights,
        pretrained=True,
    )
    config.model.backbone.in_feature = "last_feat"

    original_proposal = config.model.proposal_generator
    config.model.proposal_generator = L(ATSSRPN)(
        topk=20,
        center_radius=2.0,
        in_features=original_proposal.in_features,
        head=original_proposal.head,
        anchor_generator=original_proposal.anchor_generator,
        anchor_matcher=None,
        box2box_transform=original_proposal.box2box_transform,
        batch_size_per_image=256,
        positive_fraction=0.7,
        pre_nms_topk=(20_000, int(settings.rpn_pre_nms_topk_test)),
        post_nms_topk=(2_000, int(settings.rpn_post_nms_topk_test)),
        nms_thresh=float(settings.rpn_nms_threshold),
        min_box_size=0,
        anchor_boundary_thresh=getattr(original_proposal, "anchor_boundary_thresh", -1),
        loss_weight=getattr(original_proposal, "loss_weight", 1.0),
        box_reg_loss_type=getattr(original_proposal, "box_reg_loss_type", "smooth_l1"),
        smooth_l1_beta=getattr(original_proposal, "smooth_l1_beta", 0.0),
    )
    config.model.roi_heads.batch_size_per_image = 512
    config.model.roi_heads.positive_fraction = 0.7
    config.model.roi_heads.proposal_append_gt = True
    if hasattr(config.model.roi_heads, "proposal_matchers"):
        for index, threshold in enumerate((0.45, 0.55, 0.65)):
            if index < len(config.model.roi_heads.proposal_matchers):
                config.model.roi_heads.proposal_matchers[index].thresholds = [threshold]
    _patch_inference_thresholds(config, settings)

    model = instantiate(config.model).to(device).eval()
    DetectionCheckpointer(model).load(str(settings.checkpoint))
    if not hasattr(model, "roi_heads") or not hasattr(
        model.roi_heads, "num_cascade_stages"
    ):
        raise RuntimeError("model.roi_heads.num_cascade_stages not found")
    original_stages = int(model.roi_heads.num_cascade_stages)
    if settings.cascade_stages > original_stages:
        raise ValueError(
            f"cascade_stages must be in [1, {original_stages}], "
            f"got {settings.cascade_stages}"
        )
    model.roi_heads.num_cascade_stages = int(settings.cascade_stages)

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
        raise RuntimeError("model.backbone.net not found")
    model.backbone.net = TensorRTBackboneAdapter(
        str(settings.trt_backbone_engine.resolve())
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if settings.channels_last_pyramid:
        if not hasattr(model.backbone, "stages"):
            raise RuntimeError("SimpleFeaturePyramid stages not found")
        model.backbone = ChannelsLastPyramid(model.backbone)
    return model


__all__ = ["ChannelsLastPyramid", "build_instance_segmenter"]
