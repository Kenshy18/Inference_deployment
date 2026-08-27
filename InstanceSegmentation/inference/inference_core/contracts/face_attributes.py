"""Typed rich face geometry attached to an object detection."""

from __future__ import annotations

import math
from dataclasses import dataclass


KEYPOINT_CLASS_NAMES = {
    0: "Background",
    1: "Eye",
    2: "Nose",
    3: "Mouth",
}
KEYPOINT_STATE_NAMES = {
    0: "absent",
    1: "occluded",
    2: "visible",
}


def _validate_probability(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FaceEllipse:
    """Face ellipse in source-image pixels and radians."""

    cx: float
    cy: float
    major_radius: float
    minor_radius: float
    theta_radians: float

    def __post_init__(self) -> None:
        values = (
            self.cx,
            self.cy,
            self.major_radius,
            self.minor_radius,
            self.theta_radians,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("face ellipse values must be finite")
        if self.major_radius < 0 or self.minor_radius < 0:
            raise ValueError("face ellipse radii must be non-negative")


@dataclass(frozen=True, slots=True)
class FaceKeypoint:
    """One semantic face keypoint with validity and probability details."""

    point_index: int
    class_id: int
    x: float
    y: float
    state: int
    confidence: float
    valid: bool
    class_probabilities: tuple[float, ...]
    state_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.point_index < 0:
            raise ValueError("face keypoint index must be non-negative")
        if self.class_id not in KEYPOINT_CLASS_NAMES:
            raise ValueError(f"unsupported face keypoint class {self.class_id}")
        if self.state not in KEYPOINT_STATE_NAMES:
            raise ValueError(f"unsupported face keypoint state {self.state}")
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("face keypoint coordinates must be finite")
        _validate_probability(self.confidence, "face keypoint confidence")
        for value in self.class_probabilities:
            _validate_probability(value, "face keypoint class probability")
        for value in self.state_probabilities:
            _validate_probability(value, "face keypoint state probability")
        if len(self.class_probabilities) != 4:
            raise ValueError("face keypoint class probabilities must have 4 values")
        if len(self.state_probabilities) != 2:
            raise ValueError("face keypoint state probabilities must have 2 values")

    @property
    def class_name(self) -> str:
        return KEYPOINT_CLASS_NAMES[self.class_id]

    @property
    def state_name(self) -> str:
        return KEYPOINT_STATE_NAMES[self.state]


@dataclass(frozen=True, slots=True)
class FaceMask:
    """Uncompressed quantized face probability mask in inference memory."""

    width: int
    height: int
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    data: bytes
    encoding: str = "u8-probability-v1"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("face mask dimensions must be positive")
        box = (self.box_x1, self.box_y1, self.box_x2, self.box_y2)
        if any(not math.isfinite(value) for value in box):
            raise ValueError("face mask box must be finite")
        if self.box_x2 < self.box_x1 or self.box_y2 < self.box_y1:
            raise ValueError("face mask box must satisfy xyxy ordering")
        if not self.data:
            raise ValueError("face mask data must not be empty")
        if self.encoding != "u8-probability-v1":
            raise ValueError(f"unsupported face mask encoding {self.encoding!r}")
        if len(self.data) != self.width * self.height:
            raise ValueError("face mask data size does not match its dimensions")


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """Rich face attributes predicted for one detected head."""

    score: float
    present: bool
    ellipse: FaceEllipse | None
    keypoints: tuple[FaceKeypoint, ...]
    mask: FaceMask | None = None

    def __post_init__(self) -> None:
        _validate_probability(self.score, "face observation score")
        indices = [point.point_index for point in self.keypoints]
        if sorted(indices) != list(range(5)):
            raise ValueError("face observation requires keypoint indices 0 through 4")
        if self.present and self.ellipse is None:
            raise ValueError("present face observation requires an ellipse")
        if not self.present and self.ellipse is not None:
            raise ValueError("absent face observation must not have an ellipse")
        if not self.present and self.mask is not None:
            raise ValueError("absent face observation must not have a mask")


__all__ = [
    "KEYPOINT_CLASS_NAMES",
    "KEYPOINT_STATE_NAMES",
    "FaceEllipse",
    "FaceKeypoint",
    "FaceMask",
    "FaceObservation",
]
