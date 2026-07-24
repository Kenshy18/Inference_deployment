"""Inference-only compatibility required by the DINOv3 Cascade runtime.

This module contains only the ATSS proposal-generator registration and trusted
local-checkpoint loading behavior that the validated runtime previously gained
as side effects of importing the full training entrypoint.  Keep this module
free of dataset, trainer, working-directory, and process-environment setup.

Extracted without algorithm changes from
``training/dinov3/train_dinov3_cascade_unified.py`` in the validated source:

* imports / logger dependencies: lines 16, 117, 119-120, 132-133;
* checkpoint compatibility: lines 31-53;
* ``pairwise_iou`` / ``atss_assign_single_image``: lines 532-626;
* ``ATSSRPN``: lines 628-750.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.structures import Boxes
from detectron2.utils import comm


# Allowlist the same OmegaConf objects as the validated training-module import.
try:  # pragma: no cover - depends on the installed PyTorch/OmegaConf versions
    from omegaconf import DictConfig, ListConfig  # type: ignore
    from omegaconf.base import ContainerMetadata  # type: ignore
    from typing import Any as TypingAny

    torch.serialization.add_safe_globals(
        [DictConfig, ListConfig, ContainerMetadata, TypingAny]
    )
except Exception:
    pass


# Trust repository-managed checkpoints, preserving the validated PyTorch>=2.6
# behavior. Explicit caller choices still win because setdefault is used.
_torch_load_orig = torch.load


def _torch_load_full(*args, **kwargs):  # pragma: no cover - I/O wrapper
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_full


logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)


@torch.no_grad()
def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def atss_assign_single_image(
    anchors_per_level: List[torch.Tensor],
    gt_boxes: torch.Tensor,
    strides: List[int],
    topk: int = 9,
    center_radius: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = gt_boxes.device
    anchors = torch.cat(anchors_per_level, dim=0)
    total_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]

    labels = anchors.new_full((total_anchors,), 0, dtype=torch.int8)
    matched = anchors.new_full((total_anchors,), -1, dtype=torch.int64)

    if num_gt == 0:
        return labels, matched

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    ious = pairwise_iou(gt_boxes, anchors)

    level_offsets: List[Tuple[int, int]] = []
    start = 0
    for per_level in anchors_per_level:
        end = start + per_level.shape[0]
        level_offsets.append((start, end))
        start = end

    positives = torch.zeros((num_gt, total_anchors), dtype=torch.bool, device=device)

    for gt_idx in range(num_gt):
        gx = (gt_boxes[gt_idx, 0] + gt_boxes[gt_idx, 2]) * 0.5
        gy = (gt_boxes[gt_idx, 1] + gt_boxes[gt_idx, 3]) * 0.5

        candidates: List[torch.Tensor] = []
        for level, (st, ed) in enumerate(level_offsets):
            if st >= ed:
                continue
            idx = torch.arange(st, ed, device=device)
            dist = (ax[idx] - gx).abs() + (ay[idx] - gy).abs()
            topk_level = min(topk, idx.numel())
            if topk_level == 0:
                continue
            order = dist.topk(k=topk_level, largest=False).indices
            candidates.append(idx[order])

        if not candidates:
            continue

        cand_inds = torch.cat(candidates, dim=0)
        candidate_ious = ious[gt_idx, cand_inds]
        threshold = candidate_ious.mean() + candidate_ious.std()

        if center_radius > 0:
            center_mask = torch.zeros_like(cand_inds, dtype=torch.bool)
            for level, (st, ed) in enumerate(level_offsets):
                level_mask = (cand_inds >= st) & (cand_inds < ed)
                if not level_mask.any():
                    continue
                radius = strides[level] * center_radius
                level_inds = cand_inds[level_mask]
                cx_ok = (ax[level_inds] >= gx - radius) & (
                    ax[level_inds] <= gx + radius
                )
                cy_ok = (ay[level_inds] >= gy - radius) & (
                    ay[level_inds] <= gy + radius
                )
                center_mask[level_mask] = cx_ok & cy_ok
        else:
            center_mask = torch.ones_like(cand_inds, dtype=torch.bool)

        pos_inds = cand_inds[(ious[gt_idx, cand_inds] >= threshold) & center_mask]
        positives[gt_idx, pos_inds] = True

    any_pos = positives.any(dim=0)
    if any_pos.any():
        pos_indices = any_pos.nonzero(as_tuple=False).squeeze(1)
        matched_iou, matched_gt = ious[:, pos_indices].max(dim=0)
        labels[pos_indices] = 1
        matched[pos_indices] = matched_gt

    return labels, matched


@PROPOSAL_GENERATOR_REGISTRY.register()
class ATSSRPN(RPN):
    def __init__(self, strides=None, topk=9, center_radius=0.0, **kwargs):
        super().__init__(**kwargs)
        if strides is None and hasattr(self.anchor_generator, "strides"):
            strides = list(self.anchor_generator.strides)
        self.strides = strides if strides is not None else []
        self.topk = topk
        self.center_radius = center_radius

        if comm.is_main_process():
            logger.info(
                "Using ATSS RPN (topk=%s, center_radius=%s, strides=%s)",
                self.topk,
                self.center_radius,
                self.strides,
            )

    @classmethod
    def from_config(cls, cfg, input_shape):
        params = super().from_config(cfg, input_shape)
        in_features = params.get("in_features", cfg.MODEL.RPN.IN_FEATURES)
        params["strides"] = [input_shape[f].stride for f in in_features]
        params["topk"] = getattr(cfg.MODEL.RPN, "ATSS_TOPK", 9)
        params["center_radius"] = getattr(cfg.MODEL.RPN, "ATSS_CENTER_RADIUS", 0.0)
        if comm.is_main_process():
            logger.info(
                "ATSS from_config: features=%s, strides=%s",
                in_features,
                params["strides"],
            )
        return params

    def set_atss_params(self, topk: int, center_radius: float) -> None:
        self.topk = topk
        self.center_radius = center_radius

    def label_and_sample_anchors(self, anchors, gt_instances):
        anchors_cat = Boxes.cat(anchors)
        num_per_level = [len(level.tensor) for level in anchors]
        gt_boxes = [instance.gt_boxes for instance in gt_instances]
        image_sizes = [instance.image_size for instance in gt_instances]
        gt_labels: List[torch.Tensor] = []
        matched_boxes: List[torch.Tensor] = []

        for image_size, gt_boxes_i in zip(image_sizes, gt_boxes):
            anchors_per_level = []
            start = 0
            for count in num_per_level:
                end = start + count
                anchors_per_level.append(anchors_cat.tensor[start:end])
                start = end

            if len(gt_boxes_i) > 0:
                gt_tensor = gt_boxes_i.tensor
            else:
                gt_tensor = anchors_cat.tensor.new_zeros((0, 4))

            if anchors_per_level and len(gt_tensor) > 0:
                usable_levels = min(len(anchors_per_level), len(self.strides))
                labels_atss, matched_atss = atss_assign_single_image(
                    anchors_per_level[:usable_levels],
                    gt_tensor,
                    strides=self.strides[:usable_levels],
                    topk=self.topk,
                    center_radius=self.center_radius,
                )

                if usable_levels < len(anchors_per_level):
                    remaining = torch.cat(anchors_per_level[usable_levels:], dim=0)
                    from detectron2.modeling.matcher import Matcher
                    from detectron2.structures import pairwise_iou as d2_pairwise_iou

                    matrix = d2_pairwise_iou(gt_boxes_i, Boxes(remaining))
                    match_ids, labels_remain = self.anchor_matcher(matrix)
                    labels_i = torch.cat([labels_atss, labels_remain])
                    matched_ids = torch.cat([matched_atss, match_ids])
                else:
                    labels_i, matched_ids = labels_atss, matched_atss
            else:
                labels_i = torch.zeros(
                    len(anchors_cat), dtype=torch.long, device=anchors_cat.tensor.device
                )
                matched_ids = torch.zeros(
                    len(anchors_cat), dtype=torch.long, device=anchors_cat.tensor.device
                )

            if self.anchor_boundary_thresh >= 0:
                inside = anchors_cat.inside_box(image_size, self.anchor_boundary_thresh)
                labels_i[~inside] = -1

            if len(gt_boxes_i) == 0:
                matched_boxes_i = torch.zeros_like(anchors_cat.tensor)
            else:
                clamped = matched_ids.clamp(min=0, max=len(gt_boxes_i) - 1)
                matched_boxes_i = gt_boxes_i[clamped].tensor
                matched_boxes_i[labels_i != 1] = 0

            labels_i = super()._subsample_labels(labels_i)
            gt_labels.append(labels_i)
            matched_boxes.append(matched_boxes_i)

        return gt_labels, matched_boxes

    def _subsample_labels(self, labels, matched_gt_boxes):
        batch_size = self.batch_size_per_image
        positive_fraction = self.positive_fraction

        sampled_labels: List[torch.Tensor] = []
        sampled_boxes: List[torch.Tensor] = []

        for labels_per_image, boxes_per_image in zip(labels, matched_gt_boxes):
            pos_idx = (labels_per_image == 1).nonzero(as_tuple=False).squeeze(1)
            neg_idx = (labels_per_image == 0).nonzero(as_tuple=False).squeeze(1)

            num_pos = min(int(batch_size * positive_fraction), pos_idx.numel())
            if num_pos > 0:
                pos_idx = pos_idx[
                    torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]
                ]

            num_neg = min(batch_size - num_pos, neg_idx.numel())
            if num_neg > 0:
                neg_idx = neg_idx[
                    torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
                ]

            sampled = labels_per_image.new_full(labels_per_image.shape, -1)
            sampled[pos_idx] = 1
            sampled[neg_idx] = 0

            sampled_labels.append(sampled)
            sampled_boxes.append(boxes_per_image)

        return sampled_labels, sampled_boxes


__all__ = ["ATSSRPN", "atss_assign_single_image", "pairwise_iou"]
