"""Fast curvature-density polygon sampling with temporal stabilization."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _EPS,
    align_temporal_dense,
)


def _circular_smooth(values: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return values.copy()
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(radius / 2.0, 0.75)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= np.sum(weights)
    output = np.zeros_like(values, dtype=np.float64)
    for offset, weight in zip(offsets.astype(np.int32), weights, strict=True):
        output += float(weight) * np.roll(values, int(offset), axis=1)
    return output


def curvature_saliency(
    dense: np.ndarray,
    *,
    spatial_radius: int = 2,
    temporal_window: int = 5,
) -> np.ndarray:
    value = np.asarray(dense, dtype=np.float64)
    scales = []
    for step in (1, 2, 4, 8):
        before = value - np.roll(value, step, axis=1)
        after = np.roll(value, -step, axis=1) - value
        before /= np.maximum(np.linalg.norm(before, axis=2, keepdims=True), _EPS)
        after /= np.maximum(np.linalg.norm(after, axis=2, keepdims=True), _EPS)
        cross = before[:, :, 0] * after[:, :, 1] - before[:, :, 1] * after[:, :, 0]
        dot = np.sum(before * after, axis=2)
        # Divide by scale so broad smooth bending is not mistaken for a corner.
        scales.append(np.abs(np.arctan2(cross, dot)) / np.sqrt(float(step)))
    saliency = np.maximum.reduce(scales)
    saliency = _circular_smooth(saliency, spatial_radius)
    temporal_window = max(1, int(temporal_window))
    if temporal_window > 1 and len(saliency) > 1:
        radius = temporal_window // 2
        padded = np.pad(saliency, ((radius, radius), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=temporal_window,
            axis=0,
        )
        saliency = np.median(windows, axis=-1)
    scale = np.quantile(saliency, 0.90, axis=1, keepdims=True)
    saliency = saliency / np.maximum(scale, 1e-6)
    return np.clip(saliency, 0.0, 4.0)


def sample_density(
    dense: np.ndarray,
    saliency: np.ndarray,
    target: int,
    *,
    curvature_weight: float,
    curvature_power: float = 1.0,
) -> np.ndarray:
    value = np.asarray(dense, dtype=np.float64)
    saliency = np.asarray(saliency, dtype=np.float64)
    target = max(3, int(target))
    output = np.empty((len(value), target, 2), dtype=np.float64)
    for frame in range(len(value)):
        following = np.roll(value[frame], -1, axis=0)
        lengths = np.linalg.norm(following - value[frame], axis=1)
        local = 0.5 * (saliency[frame] + np.roll(saliency[frame], -1))
        density = 1.0 + float(curvature_weight) * np.power(
            np.maximum(local, 0.0), float(curvature_power)
        )
        weighted = lengths * density
        cumulative = np.concatenate([[0.0], np.cumsum(weighted)])
        total = float(cumulative[-1])
        positions = np.linspace(0.0, total, target, endpoint=False)
        segments = np.searchsorted(cumulative, positions, side="right") - 1
        segments = np.clip(segments, 0, len(value[frame]) - 1)
        alpha = (positions - cumulative[segments]) / np.maximum(
            weighted[segments], _EPS
        )
        output[frame] = (
            (1.0 - alpha[:, None]) * value[frame, segments]
            + alpha[:, None] * following[segments]
        )
    return output


def curvature_density_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 256,
    curvature_weight: float = 2.0,
    curvature_power: float = 1.0,
    spatial_radius: int = 2,
    temporal_window: int = 5,
) -> np.ndarray:
    dense = align_temporal_dense(polygons, int(dense_vertices))
    saliency = curvature_saliency(
        dense,
        spatial_radius=spatial_radius,
        temporal_window=temporal_window,
    )
    return sample_density(
        dense,
        saliency,
        target,
        curvature_weight=curvature_weight,
        curvature_power=curvature_power,
    )
