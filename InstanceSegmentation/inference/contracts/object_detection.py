"""Object-detection task input/output rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .classification import Classification
from .common import FrameBatch, FrameReference, ModelDescriptor


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Source-image pixel coordinates using half-open XYXY semantics."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box must satisfy x2 >= x1 and y2 >= y1")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    bbox: BoundingBox
    classification: Classification | None = None
    track_id: int | None = None
    source: str = "detection"

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("detection class_id must be non-negative")
        if not self.class_name:
            raise ValueError("detection class_name must not be empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("detection score must be in [0, 1]")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if not self.source:
            raise ValueError("detection source must not be empty")


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    model: ModelDescriptor
    frame: FrameReference
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        from .common import TaskType

        if self.model.task is not TaskType.OBJECT_DETECTION:
            raise ValueError("DetectionFrame requires an object-detection model")


@runtime_checkable
class ObjectDetectionAdapter(Protocol):
    descriptor: ModelDescriptor

    def predict(self, batch: FrameBatch) -> Sequence[DetectionFrame]:
        ...

    def synchronize(self) -> None:
        ...

    def close(self) -> None:
        ...


__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionFrame",
    "ObjectDetectionAdapter",
]
