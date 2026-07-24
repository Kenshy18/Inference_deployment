"""Dependency-free construction settings for EVA02 Cascade segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COMPILE_MODES = frozenset(
    {"none", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
)

NATIVE_CONFIG_PATH = Path(__file__).with_name("lazy_config.py")


@dataclass(frozen=True, slots=True)
class InstanceSegmentationSettings:
    config_path: Path
    checkpoint: Path
    target_size: int = 1280
    num_classes: int = 1
    score_threshold: float = 0.1
    nms_threshold: float = 0.5
    topk_per_image: int = 80
    rpn_pre_nms_multiplier: int = 10
    compile_backbone: str = "max-autotune"
    compile_heads: str = "none"
    model_half: bool = False
    backbone_half: bool = True
    disable_activation_checkpointing: bool = True
    prefer_sdpa: bool = True
    drop_block_indices: tuple[int, ...] = (19, 21, 22)

    def __post_init__(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if self.num_classes <= 0 or self.topk_per_image <= 0:
            raise ValueError("num_classes and topk_per_image must be positive")
        if self.rpn_pre_nms_multiplier <= 0:
            raise ValueError("rpn_pre_nms_multiplier must be positive")
        for name, value in (
            ("score_threshold", self.score_threshold),
            ("nms_threshold", self.nms_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.compile_backbone not in COMPILE_MODES:
            raise ValueError(f"unsupported compile_backbone: {self.compile_backbone!r}")
        head_parts = self.compile_heads.split(":", 1)
        if len(head_parts) == 2 and head_parts[0] not in {"proposal-only", "roi-only"}:
            raise ValueError(f"unsupported compile_heads scope: {head_parts[0]!r}")
        if head_parts[-1] not in COMPILE_MODES:
            raise ValueError(f"unsupported compile_heads: {self.compile_heads!r}")
        if any(index < 0 for index in self.drop_block_indices):
            raise ValueError("drop_block_indices must be non-negative")
        if len(set(self.drop_block_indices)) != len(self.drop_block_indices):
            raise ValueError("drop_block_indices must be unique")


__all__ = ["COMPILE_MODES", "InstanceSegmentationSettings", "NATIVE_CONFIG_PATH"]
