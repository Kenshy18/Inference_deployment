"""Configuration contracts for the repository-level inference command."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class InferenceMode(str, Enum):
    SEGMENTATION = "segmentation"
    FACE = "face"
    SEGMENTATION_FACE = "segmentation-face"

    @property
    def uses_segmentation(self) -> bool:
        return self in {self.SEGMENTATION, self.SEGMENTATION_FACE}

    @property
    def uses_face_detection(self) -> bool:
        return self in {self.FACE, self.SEGMENTATION_FACE}


@dataclass(frozen=True, slots=True)
class OrchestrationRequest:
    input_path: Path
    output_path: Path
    mode: InferenceMode
    segmentation_model: str | None = None
    segmentation_backend: str = "auto"
    face_model: str = "rtdetr_head_face"
    face_classes: tuple[str, ...] = ("Face", "Head")
    runtime_python: Path = Path(sys.executable)
    device: str = "cuda:0"
    max_frames: int | None = None
    warmup_frames: int = 0
    face_warmup_iterations: int = 3
    overwrite: bool = False
    fast_sqlite: bool = False

    def __post_init__(self) -> None:
        if self.mode.uses_segmentation and not self.segmentation_model:
            raise ValueError(
                "segmentation_model is required for the selected mode"
            )
        if not self.mode.uses_segmentation and self.segmentation_model is not None:
            raise ValueError(
                "segmentation_model is not valid for face-only mode"
            )
        if not self.face_model.strip():
            raise ValueError("face_model must not be empty")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if self.face_warmup_iterations < 0:
            raise ValueError("face_warmup_iterations must be non-negative")
        if not self.device.strip():
            raise ValueError("device must not be empty")


__all__ = ["InferenceMode", "OrchestrationRequest"]
