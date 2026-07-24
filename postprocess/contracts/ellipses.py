"""Shared representation helpers for the public ellipse artifact format."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import numpy as np


def canonicalize_ellipse(values: Iterable[float]) -> list[float]:
    cx, cy, a, b, theta = [float(value) for value in values]
    if b > a:
        a, b = b, a
        theta += 90.0
    theta = (theta + 90.0) % 180.0 - 90.0
    return [cx, cy, max(a, 1e-6), max(b, 1e-6), theta]


def angle_distance_degrees(first: float, second: float) -> float:
    difference = abs((first - second + 90.0) % 180.0 - 90.0)
    return min(difference, 180.0 - difference)


def ellipse_pair_cost(
    left: list[float],
    right: list[float],
    center_weight: float,
    size_weight: float,
    angle_weight: float,
) -> float:
    left_center = np.asarray(left[:2], dtype=np.float64)
    right_center = np.asarray(right[:2], dtype=np.float64)
    left_a, left_b = float(left[2]), float(left[3])
    right_a, right_b = float(right[2]), float(right[3])
    center_scale = max(
        math.sqrt(max(left_a * left_b, 1e-6)),
        math.sqrt(max(right_a * right_b, 1e-6)),
        1.0,
    )
    center_term = float(np.linalg.norm(left_center - right_center) / center_scale)
    size_term = abs(math.log(max(left_a, 1e-6) / max(right_a, 1e-6)))
    size_term += abs(math.log(max(left_b, 1e-6) / max(right_b, 1e-6)))
    eccentricity = max(
        0.0,
        0.5
        * (
            1.0
            - min(left_a, left_b) / max(left_a, left_b)
            + 1.0
            - min(right_a, right_b) / max(right_a, right_b)
        ),
    )
    angle_term = (
        angle_distance_degrees(float(left[4]), float(right[4]))
        / 45.0
        * max(0.1, eccentricity)
    )
    return (
        center_weight * center_term
        + size_weight * size_term
        + angle_weight * angle_term
    )


def ellipse_to_polygon(
    ellipse: Iterable[float], *, points: int = 96
) -> list[list[float]]:
    cx, cy, a, b, theta = canonicalize_ellipse(ellipse)
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    cos_theta = math.cos(math.radians(theta))
    sin_theta = math.sin(math.radians(theta))
    x = cx + a * cosine * cos_theta - b * sine * sin_theta
    y = cy + a * cosine * sin_theta + b * sine * cos_theta
    return np.column_stack((x, y)).tolist()


def ellipses_to_polygons(
    ellipses: Iterable[Iterable[float]], *, points: int = 96
) -> list[list[list[float]]]:
    return [ellipse_to_polygon(ellipse, points=points) for ellipse in ellipses]


def ellipses_to_polygons_json(
    ellipses: Iterable[Iterable[float]], *, points: int = 96
) -> str:
    return json.dumps(
        ellipses_to_polygons(ellipses, points=points),
        ensure_ascii=False,
        separators=(",", ":"),
    )
