"""Canonical geometry for face and eye privacy masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ALGORITHM_VERSION = "face-privacy-geometry-v1"
Point = tuple[float, float]
Polygon = tuple[Point, ...]
FaceEllipse = tuple[float, float, float, float, float]
MaskTarget = Literal["face", "eyes"]
EyeShape = Literal["ellipse", "rectangle"]

# Adopted dimensions. Ratios are half-extents relative to complete face size.
EYE_HALF_WIDTH_MIN_FACE_RATIO = 0.38
EYE_HALF_WIDTH_MAX_FACE_RATIO = 0.59
EYE_HALF_WIDTH_DISTANCE_RATIO = 0.97
EYE_HALF_HEIGHT_MIN_FACE_RATIO = 0.112
EYE_HALF_HEIGHT_MAX_FACE_RATIO = 0.235
EYE_HALF_HEIGHT_DISTANCE_RATIO = 0.405
FALLBACK_HALF_WIDTH_FACE_RATIO = 0.495
FALLBACK_HALF_HEIGHT_FACE_RATIO = 0.162
FALLBACK_HALF_HEIGHT_MAX_FACE_RATIO = 0.31


@dataclass(frozen=True)
class FaceKeypoint:
    x: float
    y: float
    class_name: str
    confidence: float
    valid: bool


@dataclass(frozen=True)
class PrivacyMask:
    target: MaskTarget
    shape: EyeShape
    polygon: Polygon
    center: Point
    half_width: float
    half_height: float
    theta_radians: float
    derivation: str
    confidence: float


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _valid_ellipse(ellipse: FaceEllipse | None) -> FaceEllipse | None:
    if ellipse is None or not _finite(*ellipse):
        return None
    if ellipse[2] <= 0.0 or ellipse[3] <= 0.0:
        return None
    return ellipse


def _ellipse_polygon(
    center: Point,
    half_width: float,
    half_height: float,
    angle: float,
    *,
    points: int,
) -> Polygon:
    count = max(12, int(points))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    cx, cy = center
    return tuple(
        (
            cx
            + half_width * math.cos(phase) * cosine
            - half_height * math.sin(phase) * sine,
            cy
            + half_width * math.cos(phase) * sine
            + half_height * math.sin(phase) * cosine,
        )
        for phase in (2.0 * math.pi * index / count for index in range(count))
    )


def _rectangle_polygon(
    center: Point,
    half_width: float,
    half_height: float,
    angle: float,
) -> Polygon:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    cx, cy = center
    return tuple(
        (
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    )


def _face_axes(
    ellipse: FaceEllipse,
    keypoints: tuple[FaceKeypoint, ...],
) -> tuple[Point, Point]:
    cx, cy, _major, _minor, theta = ellipse
    down = (math.cos(theta), math.sin(theta))
    lower = [
        point
        for point in keypoints
        if point.valid
        and point.class_name.casefold() in {"nose", "mouth"}
        and _finite(point.x, point.y)
    ]
    if lower:
        mean_x = sum(point.x for point in lower) / len(lower)
        mean_y = sum(point.y for point in lower) / len(lower)
        if (mean_x - cx) * down[0] + (mean_y - cy) * down[1] < 0.0:
            down = (-down[0], -down[1])
    elif down[1] < 0.0:
        down = (-down[0], -down[1])
    across = (down[1], -down[0])
    return across, down


def _candidate_eyes(
    keypoints: tuple[FaceKeypoint, ...],
    minimum_confidence: float,
) -> list[FaceKeypoint]:
    return sorted(
        (
            point
            for point in keypoints
            if point.valid
            and point.class_name.casefold() == "eye"
            and point.confidence >= minimum_confidence
            and _finite(point.x, point.y, point.confidence)
        ),
        key=lambda point: point.confidence,
        reverse=True,
    )


def _usable_eye_pair(
    ellipse: FaceEllipse,
    keypoints: tuple[FaceKeypoint, ...],
    minimum_confidence: float,
) -> tuple[FaceKeypoint, FaceKeypoint] | None:
    cx, cy, major, minor, _theta = ellipse
    across, down = _face_axes(ellipse, keypoints)
    candidates = _candidate_eyes(keypoints, minimum_confidence)
    if len(candidates) < 2:
        return None
    first, second = candidates[:2]
    dx = second.x - first.x
    dy = second.y - first.y
    distance = math.hypot(dx, dy)
    face_width = 2.0 * minor
    if distance < 0.15 * face_width or distance > 1.10 * face_width:
        return None
    if abs((dx * across[0] + dy * across[1]) / distance) < 0.50:
        return None
    for point in (first, second):
        relative_x = point.x - cx
        relative_y = point.y - cy
        local_x = (relative_x * across[0] + relative_y * across[1]) / minor
        local_y = (relative_x * down[0] + relative_y * down[1]) / major
        if local_x * local_x + local_y * local_y > 1.45**2:
            return None
    return first, second


def _face_mask(ellipse: FaceEllipse) -> PrivacyMask:
    cx, cy, major, minor, theta = ellipse
    return PrivacyMask(
        target="face",
        shape="ellipse",
        polygon=_ellipse_polygon((cx, cy), major, minor, theta, points=96),
        center=(cx, cy),
        half_width=major,
        half_height=minor,
        theta_radians=theta,
        derivation="face-ellipse",
        confidence=1.0,
    )


def _eye_mask(
    ellipse: FaceEllipse,
    keypoints: tuple[FaceKeypoint, ...],
    *,
    shape: EyeShape,
    minimum_confidence: float,
) -> PrivacyMask:
    cx, cy, major, minor, _theta = ellipse
    face_width = 2.0 * minor
    face_height = 2.0 * major
    pair = _usable_eye_pair(ellipse, keypoints, minimum_confidence)
    if pair is not None:
        first, second = pair
        dx = second.x - first.x
        dy = second.y - first.y
        distance = math.hypot(dx, dy)
        center = ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
        angle = math.atan2(dy, dx)
        half_width = min(
            EYE_HALF_WIDTH_MAX_FACE_RATIO * face_width,
            max(
                EYE_HALF_WIDTH_MIN_FACE_RATIO * face_width,
                EYE_HALF_WIDTH_DISTANCE_RATIO * distance,
            ),
        )
        half_height = min(
            EYE_HALF_HEIGHT_MAX_FACE_RATIO * face_height,
            max(
                EYE_HALF_HEIGHT_MIN_FACE_RATIO * face_height,
                EYE_HALF_HEIGHT_DISTANCE_RATIO * distance,
            ),
        )
        derivation = "eye-keypoints"
        confidence = min(first.confidence, second.confidence)
    else:
        across, down = _face_axes(ellipse, keypoints)
        local_eyes: list[tuple[float, float]] = []
        for point in _candidate_eyes(keypoints, minimum_confidence):
            relative_x = point.x - cx
            relative_y = point.y - cy
            local_x = (relative_x * across[0] + relative_y * across[1]) / minor
            local_y = (relative_x * down[0] + relative_y * down[1]) / major
            if local_x * local_x + local_y * local_y <= 1.45**2:
                local_eyes.append((local_x, local_y))
        center_x = (
            max(-0.35, min(0.35, sum(x for x, _y in local_eyes) / len(local_eyes)))
            if local_eyes
            else 0.0
        )
        center_y = (
            max(-0.65, min(0.40, sum(y for _x, y in local_eyes) / len(local_eyes)))
            if local_eyes
            else -0.32
        )
        center = (
            cx + center_x * minor * across[0] + center_y * major * down[0],
            cy + center_x * minor * across[1] + center_y * major * down[1],
        )
        angle = math.atan2(across[1], across[0])
        half_width = FALLBACK_HALF_WIDTH_FACE_RATIO * face_width
        half_height = FALLBACK_HALF_HEIGHT_FACE_RATIO * face_height
        if local_eyes:
            half_width = min(
                1.08 * minor,
                max(
                    half_width,
                    max(abs(x - center_x) * minor for x, _y in local_eyes)
                    + 0.11 * face_width,
                ),
            )
            half_height = min(
                FALLBACK_HALF_HEIGHT_MAX_FACE_RATIO * face_height,
                max(
                    half_height,
                    max(abs(y - center_y) * major for _x, y in local_eyes)
                    + 0.06 * face_height,
                ),
            )
            if shape == "ellipse":
                required_scale = max(
                    math.sqrt(
                        ((x - center_x) * minor / half_width) ** 2
                        + ((y - center_y) * major / half_height) ** 2
                    )
                    for x, y in local_eyes
                )
                if required_scale > 0.88:
                    growth = required_scale / 0.88
                    half_width = min(1.08 * minor, half_width * growth)
                    half_height = min(
                        FALLBACK_HALF_HEIGHT_MAX_FACE_RATIO * face_height,
                        half_height * growth,
                    )
        derivation = "ellipse-fallback"
        confidence = 0.0
    polygon = (
        _ellipse_polygon(center, half_width, half_height, angle, points=64)
        if shape == "ellipse"
        else _rectangle_polygon(center, half_width, half_height, angle)
    )
    return PrivacyMask(
        target="eyes",
        shape=shape,
        polygon=polygon,
        center=center,
        half_width=half_width,
        half_height=half_height,
        theta_radians=angle,
        derivation=derivation,
        confidence=confidence,
    )


def derive_privacy_mask(
    target: MaskTarget,
    ellipse: FaceEllipse | None,
    keypoints: tuple[FaceKeypoint, ...],
    *,
    eye_shape: EyeShape = "ellipse",
    minimum_eye_confidence: float = 0.35,
) -> PrivacyMask | None:
    resolved = _valid_ellipse(ellipse)
    if resolved is None:
        return None
    if target not in {"face", "eyes"}:
        raise ValueError("target must be face or eyes")
    if eye_shape not in {"ellipse", "rectangle"}:
        raise ValueError("eye_shape must be ellipse or rectangle")
    if not 0.0 <= minimum_eye_confidence <= 1.0:
        raise ValueError("minimum_eye_confidence must be between 0 and 1")
    if target == "face":
        return _face_mask(resolved)
    return _eye_mask(
        resolved,
        keypoints,
        shape=eye_shape,
        minimum_confidence=minimum_eye_confidence,
    )
