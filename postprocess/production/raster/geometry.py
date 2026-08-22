"""Boundary generators shared by CPU or CUDA exact-raster evaluators.

The raster contract consumes closed boundary samples.  Shape-specific code is
kept here so polygon, ellipse and Catmull--Rom metrics all use the same mask
and Recall/IoU implementation.
"""

from __future__ import annotations

import numpy as np

from contracts.ellipses import canonicalize_ellipse
from production.curve.runtime.model import sampling_matrix


def ellipse_boundaries(
    ellipses: np.ndarray,
    *,
    points: int = 96,
) -> np.ndarray:
    """Return the canonical Production polygonization of rotated ellipses.

    ``ellipses`` has rows ``(cx, cy, major_radius, minor_radius, degrees)``.
    Trigonometric basis values are calculated once in float64.  This is the
    same parameterization as :func:`contracts.ellipses.ellipse_to_polygon`.
    """

    values = np.asarray(ellipses, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("ellipses must have shape (cases, 5)")
    count = int(points)
    if count < 3:
        raise ValueError("points must be at least three")
    canonical = np.asarray(
        [canonicalize_ellipse(row) for row in values], dtype=np.float64
    )
    angle = np.arange(count, dtype=np.float64) * (2.0 * np.pi / float(count))
    cosine = np.cos(angle)[None, :]
    sine = np.sin(angle)[None, :]
    theta = np.deg2rad(canonical[:, 4])[:, None]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    major = canonical[:, 2, None]
    minor = canonical[:, 3, None]
    x = canonical[:, 0, None] + major * cosine * cos_theta - minor * sine * sin_theta
    y = canonical[:, 1, None] + major * cosine * sin_theta + minor * sine * cos_theta
    return np.ascontiguousarray(np.stack((x, y), axis=2), dtype=np.float64)


def catmull_rom_boundaries(
    control_points: np.ndarray,
    *,
    samples_per_segment: int = 16,
) -> np.ndarray:
    """Sample closed uniform Catmull--Rom controls in float64.

    The fixed 1/6 Bezier handles are already represented by the shared linear
    sampling matrix.  Keeping this function linear is important: endpoint
    boundary interpolation is then identical to interpolating editable P and
    sampling the intermediate curve.
    """

    values = np.asarray(control_points, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("control_points must have shape (cases, points, 2)")
    if values.shape[1] < 3:
        raise ValueError("Catmull--Rom requires at least three control points")
    if not np.all(np.isfinite(values)):
        raise ValueError("control points must be finite")
    matrix = sampling_matrix(values.shape[1], int(samples_per_segment))
    return np.ascontiguousarray(
        np.einsum("sp,cpd->csd", matrix, values, optimize=True),
        dtype=np.float64,
    )
