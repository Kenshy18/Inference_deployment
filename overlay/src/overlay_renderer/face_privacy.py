"""Derive deterministic face/eye privacy masks from rich face observations.

The inference SQLite remains the immutable source of model observations.  This
module turns one exact face ellipse plus its semantic keypoints into a polygon
that can be rendered, exported, or consumed by a later mosaic implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .models import Ellipse, FaceKeypointOverlay, Point, Polygon


PrivacyTarget = Literal["face", "eyes"]
EyeMaskShape = Literal["ellipse", "rectangle"]
PrivacyDerivation = Literal["face-ellipse", "eye-keypoints", "ellipse-fallback"]

# Eye masks intentionally include eyelids, brows, and the outer eye corners.
# These ratios are half-extents relative to the complete face width/height.
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
class FacePrivacyMask:
    """One concrete privacy region in original-video pixel coordinates."""

    target: PrivacyTarget
    shape: EyeMaskShape
    polygon: Polygon
    derivation: PrivacyDerivation
    confidence: float


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _valid_ellipse(ellipse: Ellipse | None) -> Ellipse | None:
    if ellipse is None or not _finite(ellipse):
        return None
    cx, cy, major, minor, theta = ellipse
    if major <= 0.0 or minor <= 0.0:
        return None
    return cx, cy, major, minor, theta


def _ellipse_polygon(
    center: Point,
    half_width: float,
    half_height: float,
    angle: float,
    *,
    points: int,
) -> Polygon:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    cx, cy = center
    return tuple(
        (
            cx
            + half_width * math.cos(step) * cosine
            - half_height * math.sin(step) * sine,
            cy
            + half_width * math.cos(step) * sine
            + half_height * math.sin(step) * cosine,
        )
        for step in (
            2.0 * math.pi * index / max(12, points)
            for index in range(max(12, points))
        )
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
    output: list[Point] = []
    for local_x, local_y in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        output.append(
            (
                cx + local_x * cosine - local_y * sine,
                cy + local_x * sine + local_y * cosine,
            )
        )
    return tuple(output)


def derive_face_privacy_mask(
    ellipse: Ellipse | None,
    *,
    points: int = 96,
) -> FacePrivacyMask | None:
    """Use the detector's exact ellipse as the complete face privacy mask."""

    resolved = _valid_ellipse(ellipse)
    if resolved is None:
        return None
    cx, cy, major, minor, theta = resolved
    return FacePrivacyMask(
        target="face",
        shape="ellipse",
        polygon=_ellipse_polygon(
            (cx, cy),
            major,
            minor,
            theta,
            points=points,
        ),
        derivation="face-ellipse",
        confidence=1.0,
    )


def _face_axes(
    ellipse: Ellipse,
    keypoints: tuple[FaceKeypointOverlay, ...],
) -> tuple[Point, Point]:
    """Return unit vectors pointing across the face and toward the lower face."""

    cx, cy, _major, _minor, theta = ellipse
    major_axis = (math.cos(theta), math.sin(theta))
    lower_points = [
        point
        for point in keypoints
        if point.valid
        and point.class_name.casefold() in {"nose", "mouth"}
        and _finite((point.x, point.y))
    ]
    if lower_points:
        mean_x = sum(point.x for point in lower_points) / len(lower_points)
        mean_y = sum(point.y for point in lower_points) / len(lower_points)
        projection = (
            (mean_x - cx) * major_axis[0] + (mean_y - cy) * major_axis[1]
        )
        if projection < 0.0:
            major_axis = (-major_axis[0], -major_axis[1])
    elif major_axis[1] < 0.0:
        # Ellipse orientation has no direction.  For the fallback, choose the
        # image-down direction when no semantic lower-face point is usable.
        major_axis = (-major_axis[0], -major_axis[1])
    across_axis = (major_axis[1], -major_axis[0])
    return across_axis, major_axis


def _usable_eye_pair(
    ellipse: Ellipse,
    keypoints: tuple[FaceKeypointOverlay, ...],
    *,
    minimum_confidence: float,
) -> tuple[FaceKeypointOverlay, FaceKeypointOverlay] | None:
    cx, cy, major, minor, _theta = ellipse
    across_axis, down_axis = _face_axes(ellipse, keypoints)
    candidates = sorted(
        (
            point
            for point in keypoints
            if point.valid
            and point.class_name.casefold() == "eye"
            and point.confidence >= minimum_confidence
            and _finite((point.x, point.y, point.confidence))
        ),
        key=lambda point: point.confidence,
        reverse=True,
    )
    if len(candidates) < 2:
        return None
    first, second = candidates[:2]
    delta_x = second.x - first.x
    delta_y = second.y - first.y
    distance = math.hypot(delta_x, delta_y)
    face_width = 2.0 * minor
    if distance < 0.15 * face_width or distance > 1.10 * face_width:
        return None
    alignment = abs(
        (delta_x * across_axis[0] + delta_y * across_axis[1]) / distance
    )
    if alignment < 0.50:
        return None
    for point in (first, second):
        relative_x = point.x - cx
        relative_y = point.y - cy
        across = (
            relative_x * across_axis[0] + relative_y * across_axis[1]
        ) / minor
        down = (
            relative_x * down_axis[0] + relative_y * down_axis[1]
        ) / major
        if across * across + down * down > 1.45 * 1.45:
            return None
    return first, second


def derive_eye_privacy_mask(
    ellipse: Ellipse | None,
    keypoints: tuple[FaceKeypointOverlay, ...],
    *,
    shape: EyeMaskShape = "ellipse",
    minimum_confidence: float = 0.35,
    points: int = 64,
) -> FacePrivacyMask | None:
    """Derive one rotated eye band from eyes, with an ellipse-only fallback.

    The two-eye path follows the detected eye line while bounding the region by
    face scale.  If the pair is missing or geometrically implausible, the mask
    falls back to a conservative upper-face band.  The fallback still uses
    valid nose/mouth points to disambiguate which end of the ellipse is lower.
    """

    resolved = _valid_ellipse(ellipse)
    if resolved is None:
        return None
    if shape not in {"ellipse", "rectangle"}:
        raise ValueError("eye mask shape must be ellipse or rectangle")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum eye confidence must be between 0 and 1")

    cx, cy, major, minor, _theta = resolved
    face_width = 2.0 * minor
    face_height = 2.0 * major
    eye_pair = _usable_eye_pair(
        resolved,
        keypoints,
        minimum_confidence=minimum_confidence,
    )
    if eye_pair is not None:
        first, second = eye_pair
        delta_x = second.x - first.x
        delta_y = second.y - first.y
        distance = math.hypot(delta_x, delta_y)
        center = ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
        angle = math.atan2(delta_y, delta_x)
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
        derivation: PrivacyDerivation = "eye-keypoints"
        confidence = min(first.confidence, second.confidence)
    else:
        across_axis, down_axis = _face_axes(resolved, keypoints)
        individual_eyes = [
            point
            for point in keypoints
            if point.valid
            and point.class_name.casefold() == "eye"
            and point.confidence >= minimum_confidence
            and _finite((point.x, point.y, point.confidence))
        ]
        local_eyes: list[tuple[float, float]] = []
        for point in individual_eyes:
            relative_x = point.x - cx
            relative_y = point.y - cy
            local_x = (
                relative_x * across_axis[0] + relative_y * across_axis[1]
            ) / minor
            local_y = (
                relative_x * down_axis[0] + relative_y * down_axis[1]
            ) / major
            if local_x * local_x + local_y * local_y <= 1.45 * 1.45:
                local_eyes.append((local_x, local_y))
        if local_eyes:
            local_center_x = max(
                -0.35,
                min(0.35, sum(value[0] for value in local_eyes) / len(local_eyes)),
            )
            local_center_y = max(
                -0.65,
                min(0.40, sum(value[1] for value in local_eyes) / len(local_eyes)),
            )
        else:
            local_center_x = 0.0
            local_center_y = -0.32
        center = (
            cx
            + local_center_x * minor * across_axis[0]
            + local_center_y * major * down_axis[0],
            cy
            + local_center_x * minor * across_axis[1]
            + local_center_y * major * down_axis[1],
        )
        angle = math.atan2(across_axis[1], across_axis[0])
        half_width = FALLBACK_HALF_WIDTH_FACE_RATIO * face_width
        half_height = FALLBACK_HALF_HEIGHT_FACE_RATIO * face_height
        if local_eyes:
            half_width = min(
                1.08 * minor,
                max(
                    half_width,
                    max(
                        abs(value[0] - local_center_x) * minor
                        for value in local_eyes
                    )
                    + 0.11 * face_width,
                ),
            )
            half_height = min(
                FALLBACK_HALF_HEIGHT_MAX_FACE_RATIO * face_height,
                max(
                    half_height,
                    max(
                        abs(value[1] - local_center_y) * major
                        for value in local_eyes
                    )
                    + 0.06 * face_height,
                ),
            )
            # Bounding extents alone do not guarantee containment by an
            # ellipse at a corner.  Grow both axes together when necessary.
            if shape == "ellipse":
                required_scale = max(
                    math.sqrt(
                        (
                            (value[0] - local_center_x)
                            * minor
                            / half_width
                        )
                        ** 2
                        + (
                            (value[1] - local_center_y)
                            * major
                            / half_height
                        )
                        ** 2
                    )
                    for value in local_eyes
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
        _ellipse_polygon(
            center,
            half_width,
            half_height,
            angle,
            points=points,
        )
        if shape == "ellipse"
        else _rectangle_polygon(center, half_width, half_height, angle)
    )
    return FacePrivacyMask(
        target="eyes",
        shape=shape,
        polygon=polygon,
        derivation=derivation,
        confidence=confidence,
    )


def derive_privacy_mask(
    target: Literal["none", "face", "eyes"],
    ellipse: Ellipse | None,
    keypoints: tuple[FaceKeypointOverlay, ...],
    *,
    eye_shape: EyeMaskShape = "ellipse",
    minimum_eye_confidence: float = 0.35,
) -> FacePrivacyMask | None:
    """Dispatch helper shared by rendering and future mask exporters."""

    if target == "none":
        return None
    if target == "face":
        return derive_face_privacy_mask(ellipse)
    if target == "eyes":
        return derive_eye_privacy_mask(
            ellipse,
            keypoints,
            shape=eye_shape,
            minimum_confidence=minimum_eye_confidence,
        )
    raise ValueError("face privacy target must be none, face, or eyes")
