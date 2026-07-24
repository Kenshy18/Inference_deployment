"""Minimal RT-DETR detection filtering; temporal processing is out of scope."""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms


def filter_detections(
    labels,
    boxes,
    scores,
    *,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
    max_area_ratio: float,
    class_filter: set[int] | None,
    frame_area: float,
):
    keep = scores >= score_threshold
    if class_filter is not None:
        class_keep = torch.zeros_like(keep)
        for class_id in class_filter:
            class_keep |= labels == class_id
        keep &= class_keep
    area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0)
    keep &= area <= frame_area * max_area_ratio
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    if len(scores):
        selected = batched_nms(
            boxes, scores, labels, nms_threshold
        )[:max_detections]
        labels, boxes, scores = (
            labels[selected],
            boxes[selected],
            scores[selected],
        )
    return labels, boxes, scores


__all__ = ["filter_detections"]
