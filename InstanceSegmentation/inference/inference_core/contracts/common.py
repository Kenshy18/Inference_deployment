"""Framework-neutral frame and model contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

CONTRACT_VERSION = "1.0"


class TaskType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    CLASSIFICATION = "classification"


class ColorSpace(str, Enum):
    BGR = "bgr"
    RGB = "rgb"


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded source frame before model-specific preprocessing."""

    index: int
    timestamp_sec: float
    image: np.ndarray
    color_space: ColorSpace = ColorSpace.BGR

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be non-negative")
        if not math.isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("frame timestamp must be finite and non-negative")
        if not isinstance(self.image, np.ndarray):
            raise TypeError("frame image must be a numpy.ndarray")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("frame image must have shape HxWx3")
        if self.image.shape[0] <= 0 or self.image.shape[1] <= 0:
            raise ValueError("frame image dimensions must be positive")
        if self.image.dtype != np.uint8:
            raise ValueError("frame image must use uint8 pixels")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class FrameBatch:
    """Source-order frames passed to one model adapter call."""

    frames: tuple[Frame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frame batch must not be empty")
        indices = [frame.index for frame in self.frames]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("frame indices must be unique and source ordered")

    @classmethod
    def from_sequence(cls, frames: Sequence[Frame]) -> "FrameBatch":
        return cls(tuple(frames))

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def images(self) -> list[np.ndarray]:
        return [frame.image for frame in self.frames]


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Image-free frame identity attached to persisted inference results."""

    index: int
    timestamp_sec: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be non-negative")
        if not math.isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("frame timestamp must be finite and non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")

    @classmethod
    def from_frame(cls, frame: Frame) -> "FrameReference":
        return cls(
            index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            width=frame.width,
            height=frame.height,
        )


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    task: TaskType
    implementation: str
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not isinstance(self.task, TaskType):
            raise TypeError("task must be a TaskType")
        if not self.implementation.strip():
            raise ValueError("implementation must not be empty")
        if not self.contract_version.strip():
            raise ValueError("contract_version must not be empty")


@runtime_checkable
class VisionAdapter(Protocol):
    """Minimum boundary implemented by every inference backend."""

    descriptor: ModelDescriptor

    def predict(self, batch: FrameBatch) -> Sequence[object]:
        ...

    def synchronize(self) -> None:
        ...

    def close(self) -> None:
        ...


__all__ = [
    "CONTRACT_VERSION",
    "ColorSpace",
    "Frame",
    "FrameBatch",
    "FrameReference",
    "ModelDescriptor",
    "TaskType",
    "VisionAdapter",
]
