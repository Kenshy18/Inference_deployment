"""Convert binary masks to bounded-size source-coordinate polygons."""

from __future__ import annotations

import cv2
import numpy as np


DEFAULT_MAX_MASK_POINTS = 150
_MIN_POLYGON_POINTS = 3


def _allocate_contour_limits(
    contours: list[np.ndarray],
    max_points: int,
) -> list[int]:
    """Allocate a hard total point budget while keeping every selected contour."""

    minimum_total = _MIN_POLYGON_POINTS * len(contours)
    if minimum_total > max_points:
        raise ValueError("selected contours exceed the minimum point budget")
    needs = np.asarray(
        [max(0, len(contour) - _MIN_POLYGON_POINTS) for contour in contours],
        dtype=np.int64,
    )
    remaining = max_points - minimum_total
    if int(needs.sum()) <= remaining:
        return [len(contour) for contour in contours]
    if remaining == 0:
        return [_MIN_POLYGON_POINTS] * len(contours)

    raw = remaining * needs.astype(np.float64) / float(needs.sum())
    extras = np.floor(raw).astype(np.int64)
    unused = remaining - int(extras.sum())
    if unused:
        fractions = raw - extras
        order = np.argsort(-fractions, kind="stable")
        for index in order:
            if unused == 0:
                break
            if extras[index] < needs[index]:
                extras[index] += 1
                unused -= 1
    return [
        _MIN_POLYGON_POINTS + int(extra)
        for extra in extras
    ]


def _simplify_contour(contour: np.ndarray, point_limit: int) -> np.ndarray:
    """Use the smallest Douglas-Peucker epsilon that satisfies a point limit."""

    if len(contour) <= point_limit:
        return contour
    perimeter = float(cv2.arcLength(contour, closed=True))
    low = 0.0
    high = max(perimeter, 1.0)
    best: np.ndarray | None = None
    for _ in range(32):
        epsilon = (low + high) * 0.5
        candidate = cv2.approxPolyDP(contour, epsilon, closed=True)
        count = len(candidate)
        if count > point_limit:
            low = epsilon
        else:
            high = epsilon
            if count >= _MIN_POLYGON_POINTS:
                best = candidate
    if best is not None:
        return best

    # Degenerate contours can jump directly from more than the limit to fewer
    # than three points. Ordered sampling is a deterministic last-resort bound.
    indices = np.linspace(
        0,
        len(contour),
        num=point_limit,
        endpoint=False,
        dtype=np.int64,
    )
    return contour[indices]


def mask_to_polygons(
    mask: np.ndarray,
    *,
    max_points: int = DEFAULT_MAX_MASK_POINTS,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> list[list[float]]:
    """Extract simple contours with at most ``max_points`` vertices per mask."""

    if max_points < _MIN_POLYGON_POINTS:
        raise ValueError("max_points must be at least 3")
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    result = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    raw_contours = [
        contour
        for contour in result[-2]
        if len(contour) >= _MIN_POLYGON_POINTS
    ]
    if not raw_contours:
        return []

    maximum_contours = max_points // _MIN_POLYGON_POINTS
    if len(raw_contours) > maximum_contours:
        ranked = sorted(
            range(len(raw_contours)),
            key=lambda index: abs(cv2.contourArea(raw_contours[index])),
            reverse=True,
        )
        selected = set(ranked[:maximum_contours])
        raw_contours = [
            contour
            for index, contour in enumerate(raw_contours)
            if index in selected
        ]

    limits = _allocate_contour_limits(raw_contours, max_points)
    polygons: list[list[float]] = []
    coordinate_offset = np.asarray(
        [float(x_offset) + 0.5, float(y_offset) + 0.5],
        dtype=np.float32,
    )
    for contour, limit in zip(raw_contours, limits):
        simplified = _simplify_contour(contour, limit)
        points = simplified.reshape(-1, 2).astype(np.float32)
        if len(points) >= _MIN_POLYGON_POINTS:
            polygons.append((points + coordinate_offset).reshape(-1).tolist())
    return polygons
