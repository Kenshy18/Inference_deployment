"""DINOv3 Cascade model execution settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoInferenceSettings:
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bf16"
    score_threshold: float = 0.3
    classifier_pad_to: int = 0
    classifier_pad_multiple: int = 0

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError(f"unsupported amp_dtype: {self.amp_dtype}")


__all__ = ["VideoInferenceSettings"]
