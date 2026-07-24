"""Construct the production-shaped EVA02 Cascade segmenter."""

from __future__ import annotations

import contextlib
import io
import warnings

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate

from .contracts import InstanceSegmentationSettings
from .pruning import drop_blocks
from .runtime_support import install_legacy_checkpoint_loading, install_sdpa_attention


def _configure_compile_heads(model, mode: str, topk_per_image: int):
    compile_proposals = False
    compile_roi_heads = False
    resolved_mode = mode
    if mode == "none":
        return model
    if mode.startswith("proposal-only:"):
        compile_proposals = True
        resolved_mode = mode.split(":", 1)[1]
    elif mode.startswith("roi-only:"):
        compile_roi_heads = True
        resolved_mode = mode.split(":", 1)[1]
    else:
        compile_proposals = True
        compile_roi_heads = True

    if compile_roi_heads and hasattr(model, "roi_heads"):
        model.roi_heads.fixed_num_proposals = int(topk_per_image)
        if hasattr(model.roi_heads, "box_pooler"):
            model.roi_heads.box_pooler.fixed_boxes_per_image = int(topk_per_image)
    if compile_proposals and hasattr(model, "proposal_generator"):
        model.proposal_generator = torch.compile(
            model.proposal_generator, mode=resolved_mode
        )
    if compile_roi_heads and hasattr(model, "roi_heads"):
        model.roi_heads = torch.compile(model.roi_heads, mode=resolved_mode)
    return model


def build_segmenter(settings: InstanceSegmentationSettings, *, device: str):
    """Load the configured model once; composition supplies resolved paths."""

    if not settings.config_path.is_file():
        raise FileNotFoundError(f"Config not found: {settings.config_path}")
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint}")
    with contextlib.redirect_stderr(io.StringIO()):
        config = LazyConfig.load(str(settings.config_path))

    backbone = config.model.backbone
    if hasattr(backbone, "net") and hasattr(backbone.net, "img_size"):
        backbone.net.img_size = settings.target_size
    if hasattr(backbone, "square_pad"):
        backbone.square_pad = settings.target_size
    if (
        settings.disable_activation_checkpointing
        and hasattr(backbone, "net")
        and hasattr(backbone.net, "use_act_checkpoint")
    ):
        backbone.net.use_act_checkpoint = False

    config.model.roi_heads.num_classes = settings.num_classes
    if hasattr(config.model.roi_heads, "box_predictors"):
        for predictor in config.model.roi_heads.box_predictors:
            predictor.test_score_thresh = settings.score_threshold
            predictor.test_nms_thresh = settings.nms_threshold
            predictor.test_topk_per_image = settings.topk_per_image
    if hasattr(config.model, "proposal_generator"):
        pre_nms = max(
            settings.topk_per_image,
            settings.topk_per_image * settings.rpn_pre_nms_multiplier,
        )
        config.model.proposal_generator.pre_nms_topk = (pre_nms, pre_nms)
        config.model.proposal_generator.post_nms_topk = (
            settings.topk_per_image,
            settings.topk_per_image,
        )

    if settings.prefer_sdpa:
        install_sdpa_attention()
    model = instantiate(config.model).to(device).eval()
    install_legacy_checkpoint_loading()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DetectionCheckpointer(model).load(str(settings.checkpoint))
    model, _ = drop_blocks(model, settings.drop_block_indices)

    if settings.model_half and device.startswith("cuda"):
        model = model.half()
    elif settings.backbone_half and hasattr(model.backbone, "net"):
        model.backbone.net = model.backbone.net.half()
    if settings.compile_backbone != "none":
        model.backbone.net = torch.compile(
            model.backbone.net, mode=settings.compile_backbone
        )
    return _configure_compile_heads(
        model, settings.compile_heads, settings.topk_per_image
    )


__all__ = ["build_segmenter"]
