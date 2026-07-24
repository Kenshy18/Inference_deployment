"""Dependency-free construction settings for DINOv3 Cascade segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstanceSegmentationSettings:
    checkpoint: Path
    backbone_weights: str
    trt_backbone_engine: Path
    config_path: Path | None = None
    target_size: int | tuple[int, int] = (720, 1280)
    score_threshold: float = 0.3
    nms_threshold: float = 0.4
    topk_per_image: int = 200
    rpn_pre_nms_topk_test: int = 100
    rpn_post_nms_topk_test: int = 40
    rpn_nms_threshold: float = 0.9
    cascade_stages: int = 1
    channels_last_pyramid: bool = True

    def __post_init__(self) -> None:
        target_h, target_w = (
            (self.target_size, self.target_size)
            if isinstance(self.target_size, int)
            else self.target_size
        )
        if int(target_h) <= 0 or int(target_w) <= 0:
            raise ValueError(f"target_size must be positive: {self.target_size}")
        if self.topk_per_image <= 0:
            raise ValueError("topk_per_image must be positive")
        if self.rpn_pre_nms_topk_test <= 0 or self.rpn_post_nms_topk_test <= 0:
            raise ValueError("RPN top-k settings must be positive")
        if self.cascade_stages <= 0:
            raise ValueError("cascade_stages must be positive")


__all__ = ["InstanceSegmentationSettings"]
