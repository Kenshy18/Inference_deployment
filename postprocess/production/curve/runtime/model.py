"""Exact curve model requested by the editor contract.

This is a closed, uniform Catmull--Rom spline with tension 1.0, represented as
piecewise cubic Bezier segments.  Users edit only ``P[i]``.  No API accepts
free Bezier handles, so an optimizer cannot accidentally leave this model.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


_DUPLICATE_EPSILON = 1e-6
_HANDLE_FACTOR = 1.0 / 6.0


@lru_cache(maxsize=32)
def sampling_matrix(
    control_point_count: int,
    samples_per_segment: int,
) -> np.ndarray:
    """Return the exact linear map from interpolation points to curve samples."""
    count = int(control_point_count)
    samples = int(samples_per_segment)
    if count < 3:
        raise ValueError("control_point_count must be at least three")
    if samples < 1:
        raise ValueError("samples_per_segment must be positive")
    output = np.zeros((count * samples, count), dtype=np.float64)
    for segment in range(count):
        for sample in range(samples):
            parameter = float(sample) / float(samples)
            inverse = 1.0 - parameter
            w0 = inverse**3
            w1 = 3.0 * inverse**2 * parameter
            w2 = 3.0 * inverse * parameter**2
            w3 = parameter**3
            row = segment * samples + sample
            output[row, (segment - 1) % count] -= w1 * _HANDLE_FACTOR
            output[row, segment] += w0 + w1 + w2 * _HANDLE_FACTOR
            output[row, (segment + 1) % count] += w1 * _HANDLE_FACTOR + w2 + w3
            output[row, (segment + 2) % count] -= w2 * _HANDLE_FACTOR
    output.setflags(write=False)
    return output


def normalize_control_points(points: np.ndarray) -> np.ndarray:
    """Validate a non-duplicated closed-loop control-point sequence."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if (
        len(value) >= 2
        and float(np.linalg.norm(value[0] - value[-1])) <= _DUPLICATE_EPSILON
    ):
        value = value[:-1]
    if len(value) < 3:
        raise ValueError("a closed Catmull-Rom curve requires at least three points")
    if not np.all(np.isfinite(value)):
        raise ValueError("control points must be finite")
    return np.ascontiguousarray(value, dtype=np.float64)


def bezier_segments(points: np.ndarray) -> np.ndarray:
    """Return ``(N, 4, 2)`` fixed-handle cubic Bezier segments.

    For segment ``P[i] -> P[i+1]`` the returned values are exactly::

        B0 = P[i]
        B1 = P[i] + (P[i+1] - P[i-1]) / 6
        B2 = P[i+1] - (P[i+2] - P[i]) / 6
        B3 = P[i+1]
    """
    value = normalize_control_points(points)
    p0 = np.roll(value, 1, axis=0)
    p1 = value
    p2 = np.roll(value, -1, axis=0)
    p3 = np.roll(value, -2, axis=0)
    return np.ascontiguousarray(
        np.stack(
            (
                p1,
                p1 + (p2 - p0) * _HANDLE_FACTOR,
                p2 - (p3 - p1) * _HANDLE_FACTOR,
                p2,
            ),
            axis=1,
        ),
        dtype=np.float64,
    )


def evaluate_cubic(segment: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate one cubic Bezier segment with the stated Bernstein formula."""
    control = np.asarray(segment, dtype=np.float64).reshape(4, 2)
    t = np.asarray(parameters, dtype=np.float64).reshape(-1, 1)
    if np.any(t < 0.0) or np.any(t > 1.0):
        raise ValueError("Bezier parameters must be in [0, 1]")
    one_minus = 1.0 - t
    return (
        one_minus**3 * control[0]
        + 3.0 * one_minus**2 * t * control[1]
        + 3.0 * one_minus * t**2 * control[2]
        + t**3 * control[3]
    )


def sample_closed_curve(
    points: np.ndarray,
    samples_per_segment: int = 16,
) -> np.ndarray:
    """Sample all closed segments without duplicating their shared endpoints."""
    samples = int(samples_per_segment)
    if samples < 1:
        raise ValueError("samples_per_segment must be positive")
    value = normalize_control_points(points)
    return np.ascontiguousarray(
        sampling_matrix(len(value), samples) @ value,
        dtype=np.float64,
    )


def sample_curve_sequence(
    control_points: np.ndarray,
    samples_per_segment: int = 16,
) -> np.ndarray:
    """Vectorized convenience wrapper for a ``(T, N, 2)`` sequence."""
    value = np.asarray(control_points, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 2:
        raise ValueError("control_points must have shape (frames, points, 2)")
    if value.shape[1] < 3:
        raise ValueError("a closed Catmull-Rom curve requires at least three points")
    if not np.all(np.isfinite(value)):
        raise ValueError("control points must be finite")
    return np.ascontiguousarray(
        sampling_matrix(value.shape[1], int(samples_per_segment)) @ value,
        dtype=np.float64,
    )
