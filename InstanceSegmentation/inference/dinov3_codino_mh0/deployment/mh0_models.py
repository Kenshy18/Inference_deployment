"""Production-trainable SC-BALANCED detector and mask head registrations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.core import bbox2result
from mmdet.models.builder import BACKBONES, DETECTORS, HEADS
from mmdet.models.necks.sfp import SFP
from mmdet.models.roi_heads.mask_heads import refine_mask_head as refine_mask_module
from mmdet.models.roi_heads.mask_heads.refine_mask_head import (
    MultiBranchFusion,
    SimpleRefineMaskHead,
)
from projects.models.co_detr import CoDETR
from dinov3.hub.backbones import dinov3_vits16plus


class FlexibleMultiBranchFusionAvg(MultiBranchFusion):
    """MultiBranchFusionAvg supporting one, two, or three dilation branches."""

    def forward(self, features):
        branches = [
            module(features)
            for name, module in self.named_children()
            if name.startswith("dilation_conv_")
        ]
        pooled = F.avg_pool2d(features, features.shape[-1])
        return self.merge_conv(sum(branches, pooled))


# SimpleSFMStage resolves this class from the module global namespace.
refine_mask_module.MultiBranchFusionAvg = FlexibleMultiBranchFusionAvg


@BACKBONES.register_module(force=True)
class DINOv3ViTSPlus(BaseModule):
    """MMDetection wrapper for the final normalized DINOv3 ViT-S+/16 map."""

    def __init__(
        self,
        weights,
        pretrained=True,
        layers_to_use=1,
        embed_dim=384,
        patch_size=16,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.layers_to_use = int(layers_to_use)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.backbone = dinov3_vits16plus(
            pretrained=pretrained,
            weights=weights,
            check_hash=False,
        )
        if self.backbone.embed_dim != self.embed_dim:
            raise ValueError(
                f"ViT-S+ embed_dim mismatch: {self.backbone.embed_dim} != {self.embed_dim}"
            )

    def init_weights(self):
        pass

    def forward(self, inputs):
        features = self.backbone.get_intermediate_layers(
            inputs,
            n=self.layers_to_use,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        if self.layers_to_use == 1:
            return [features[-1]]
        return [torch.cat(features, dim=1)]


@HEADS.register_module(force=True)
class TrainableExplicit112SimpleRefineMaskHead(SimpleRefineMaskHead):
    """Train on native stages and append parameter-free 112 logits for eval."""

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        # MMCV's FP16 training loop supplies autocast, while its standalone
        # evaluation loop does not. Keep this head valid in both call paths.
        with torch.autocast(
            device_type=instance_feats.device.type,
            dtype=torch.float16,
            enabled=instance_feats.is_cuda,
        ):
            predictions, hidden_states = super().forward(
                instance_feats, semantic_feat, rois, roi_labels
            )
        if not self.training and predictions[-1].shape[-2:] != (112, 112):
            predictions.append(
                F.interpolate(
                    predictions[-1],
                    size=(112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        return predictions, hidden_states


@DETECTORS.register_module(force=True)
class TrainableSCBalancedCoDETR(CoDETR):
    """SC-BALANCED P3/P4/P5 query route with a trainable P2 mask bypass."""

    def __init__(
        self,
        query_level_indices=(1, 2, 3),
        neck_level_channels=(128, 256, 256, 256, 256),
        query_channels=256,
        mask_channels=192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_level_indices = tuple(int(index) for index in query_level_indices)
        self.neck_level_channels = tuple(int(value) for value in neck_level_channels)
        self.query_channels = int(query_channels)
        self.mask_channels = int(mask_channels)
        if len(self.neck_level_channels) != 5:
            raise ValueError("SC-BALANCED requires P2-P6 neck channel declarations")

        self._specialize_heterogeneous_neck()
        query_sources = [self.neck_level_channels[index] for index in self.query_level_indices]
        self.sc_query_adapters = nn.ModuleList(
            nn.Identity()
            if source == self.query_channels
            else nn.Conv2d(source, self.query_channels, 1, bias=False)
            for source in query_sources
        )
        mask_sources = [self.neck_level_channels[0], self.query_channels, self.query_channels, self.query_channels]
        self.sc_mask_adapters = nn.ModuleList(
            nn.Identity()
            if source == self.mask_channels
            else nn.Conv2d(source, self.mask_channels, 1, bias=False)
            for source in mask_sources
        )

    def _specialize_heterogeneous_neck(self):
        channels = self.neck_level_channels
        if len(set(channels)) == 1:
            return
        templates = {
            width: SFP(in_channels=[384], out_channels=width, num_outs=5, use_p2=True)
            for width in set(channels)
        }
        for index, name in enumerate(("p2", "p3", "p4", "p5", "p6")):
            setattr(self.neck, name, getattr(templates[channels[index]], name))
        self.neck.out_channels = list(channels)

    def _query_features(self, pyramid):
        if len(pyramid) != 5:
            raise RuntimeError(f"Expected SFP P2-P6, got {len(pyramid)} levels")
        return [
            adapter(pyramid[index])
            for adapter, index in zip(self.sc_query_adapters, self.query_level_indices)
        ]

    def _mask_features(self, pyramid, encoded):
        if len(encoded) < 3:
            raise RuntimeError(f"Expected at least 3 encoded levels, got {len(encoded)}")
        sources = [pyramid[0], encoded[0], encoded[1], encoded[2]]
        return [adapter(feature) for adapter, feature in zip(self.sc_mask_adapters, sources)]

    def _mask_forward(self, x, rois, roi_labels):
        # SingleRoIExtractor force-casts its feature tuple to FP32. CoDETR's
        # evaluation path casts RoIs to the pre-extractor feature dtype, which
        # is FP16, so normalize the coordinate tensor before mmcv RoIAlign.
        rois = rois.float()
        ins_feats = self.mask_roi_extractor(
            x[: self.mask_roi_extractor.num_inputs], rois
        )
        stage_instance_preds, hidden_states = self.mask_head(
            ins_feats, x[0], rois, roi_labels
        )
        return dict(
            stage_instance_preds=stage_instance_preds,
            hidden_states=hidden_states,
            mask_feats=ins_feats,
        )

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs,
    ):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        if not self.with_attn_mask:
            for img_meta in img_metas:
                input_h, input_w = img_meta["batch_input_shape"]
                img_meta["img_shape"] = [input_h, input_w, 3]

        pyramid = self.extract_feat(img, img_metas)
        query_features = self._query_features(pyramid)
        query_results = self.query_head.forward_train(
            query_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        losses, encoded = query_results[:2]
        if len(query_results) != 3:
            raise RuntimeError("CoDINO query head must return sampled proposal results during training")
        results_list = query_results[2]
        mask_features = self._mask_features(pyramid, encoded)

        # F227 intentionally routes P3-P5 to the query path and P2-P5 to the
        # mask path. P6 and CoDINOHead's synthetic downsample level are still
        # registered modules, so attach a zero-valued graph edge to keep DDP's
        # static reduction contract without changing any loss value.
        route_anchor = pyramid[4].sum() * 0.0
        if len(encoded) > 3:
            route_anchor = route_anchor + encoded[3].sum() * 0.0
        first_loss_name = next(iter(losses))
        losses[first_loss_name] = losses[first_loss_name] + route_anchor

        if not hasattr(self, "mask_head"):
            return losses
        if gt_masks is None:
            raise ValueError("SC-BALANCED instance training requires gt_masks")
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in img_metas]
        sampling_results = []
        for image_index in range(len(img_metas)):
            assign_result = self.bbox_assigner.assign(
                results_list[image_index],
                gt_bboxes[image_index],
                gt_bboxes_ignore[image_index],
                gt_labels[image_index],
            )
            sampling_results.append(
                self.bbox_sampler.sample(
                    assign_result,
                    results_list[image_index],
                    gt_bboxes[image_index],
                    gt_labels[image_index],
                    feats=[feature[image_index][None] for feature in mask_features],
                )
            )
        mask_results = self._mask_forward_train(
            mask_features, sampling_results, gt_masks, img_metas
        )
        losses.update(mask_results["loss_mask"])
        if "loss_mask_iou" in mask_results:
            losses.update(mask_results["loss_mask_iou"])
        return losses

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        pyramid = self.extract_feat(img, img_metas)
        query_features = self._query_features(pyramid)
        results_list, encoded = self.query_head.simple_test(
            query_features, img_metas, rescale=rescale, return_encoder_output=True
        )
        bbox_results = [
            bbox2result(boxes, labels, self.query_head.num_classes)
            for boxes, labels in results_list
        ]
        if not hasattr(self, "mask_head"):
            return bbox_results
        mask_features = self._mask_features(pyramid, encoded)
        det_bboxes = [boxes for boxes, _ in results_list]
        det_labels = [labels for _, labels in results_list]
        segm_results = self.simple_test_mask(
            mask_features, img_metas, det_bboxes, det_labels, rescale=rescale
        )
        return list(zip(bbox_results, segm_results))
