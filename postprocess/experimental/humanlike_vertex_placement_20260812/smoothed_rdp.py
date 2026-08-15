"""Spatial RDP vertices with robust temporal arclength smoothing."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _EPS,
    orient_ccw,
)

from .spatial import align_polygon_sequence, rdp_fixed_count


def _arclength_fractions(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    following = np.roll(contour, -1, axis=0)
    lengths = np.linalg.norm(following - contour, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return cumulative / max(float(cumulative[-1]), _EPS), lengths


def _point_fractions(contour: np.ndarray, points: np.ndarray) -> np.ndarray:
    cumulative, _lengths = _arclength_fractions(contour)
    squared = np.sum((points[:, None, :] - contour[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared, axis=1)
    fractions = cumulative[indices].astype(np.float64)
    output = np.empty_like(fractions)
    output[0] = fractions[0]
    for position in range(1, len(fractions)):
        step = (fractions[position] - fractions[position - 1]) % 1.0
        output[position] = output[position - 1] + max(step, 1e-6)
    return output


def _temporal_unwrap(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for frame in range(1, len(output)):
        shift = round(float(output[frame - 1, 0] - output[frame, 0]))
        output[frame] += float(shift)
    return output


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=window,
        axis=0,
    )
    return np.median(windows, axis=-1)


def _project_order(values: np.ndarray, minimum_gap: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for frame in range(len(output)):
        start = float(output[frame, 0])
        for position in range(1, output.shape[1]):
            output[frame, position] = max(
                output[frame, position],
                output[frame, position - 1] + minimum_gap,
            )
        maximum = start + 1.0 - minimum_gap
        if output[frame, -1] > maximum:
            span = max(output[frame, -1] - start, _EPS)
            output[frame] = start + (output[frame] - start) * (
                (1.0 - minimum_gap) / span
            )
    return output


def _sample_fraction(contour: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    cumulative, lengths = _arclength_fractions(contour)
    normalized = np.mod(fractions, 1.0)
    segments = np.searchsorted(cumulative, normalized, side="right") - 1
    segments = np.clip(segments, 0, len(contour) - 1)
    local_start = cumulative[segments]
    local_width = lengths[segments] / max(float(np.sum(lengths)), _EPS)
    alpha = (normalized - local_start) / np.maximum(local_width, _EPS)
    following = np.roll(contour, -1, axis=0)
    return (
        (1.0 - alpha[:, None]) * contour[segments]
        + alpha[:, None] * following[segments]
    )


def smoothed_rdp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    temporal_window: int = 5,
    smooth_blend: float = 0.50,
    minimum_gap_ratio: float = 0.08,
) -> np.ndarray:
    contours = [orient_ccw(value) for value in polygons]
    spatial = [rdp_fixed_count(value, int(target)) for value in contours]
    aligned = align_polygon_sequence(spatial)
    fractions = np.asarray(
        [
            _point_fractions(contour, points)
            for contour, points in zip(contours, aligned, strict=True)
        ],
        dtype=np.float64,
    )
    fractions = _temporal_unwrap(fractions)
    smoothed = _median_smooth(fractions, temporal_window)
    blend = float(np.clip(smooth_blend, 0.0, 1.0))
    mixed = (1.0 - blend) * fractions + blend * smoothed
    mixed = _project_order(
        mixed,
        minimum_gap=max(float(minimum_gap_ratio) / max(int(target), 1), 1e-6),
    )
    return np.asarray(
        [
            _sample_fraction(contour, frame_fractions)
            for contour, frame_fractions in zip(contours, mixed, strict=True)
        ],
        dtype=np.float64,
    )
