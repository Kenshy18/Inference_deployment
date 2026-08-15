"""Track-wise worst-case boundary-error vertex hierarchy."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    align_temporal_dense,
    signed_area,
)

from .spatial import _diameter_pair


def _forward_indices(start: int, end: int, count: int) -> np.ndarray:
    distance = (int(end) - int(start)) % int(count)
    if distance <= 1:
        return np.empty((0,), dtype=np.int32)
    return (int(start) + np.arange(1, distance, dtype=np.int32)) % int(count)


def _track_segment_split(
    dense: np.ndarray,
    start: int,
    end: int,
    radii: np.ndarray,
    temporal_quantile: float,
) -> tuple[float, int | None]:
    interior = _forward_indices(start, end, dense.shape[1])
    if not len(interior):
        return 0.0, None
    a = dense[:, int(start), :]
    b = dense[:, int(end), :]
    points = dense[:, interior, :]
    delta = b - a
    denominator = np.sum(delta * delta, axis=1)
    alpha = np.sum((points - a[:, None, :]) * delta[:, None, :], axis=2)
    alpha = alpha / np.maximum(denominator[:, None], 1e-9)
    alpha = np.clip(alpha, 0.0, 1.0)
    projection = a[:, None, :] + alpha[:, :, None] * delta[:, None, :]
    distances = np.linalg.norm(points - projection, axis=2)
    normalized = distances / np.maximum(radii[:, None], 1.0)
    per_index = np.quantile(
        normalized,
        float(np.clip(temporal_quantile, 0.0, 1.0)),
        axis=0,
    )
    maximum = float(np.max(per_index))
    candidates = interior[np.flatnonzero(np.isclose(per_index, maximum, atol=1e-12))]
    return maximum, int(candidates[0])


def trackwise_worst_rdp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 128,
    temporal_quantile: float = 1.0,
) -> np.ndarray:
    dense = align_temporal_dense(polygons, int(dense_vertices))
    return trackwise_worst_rdp_from_dense(
        dense,
        target,
        temporal_quantile=temporal_quantile,
    )


def trackwise_worst_rdp_from_dense(
    dense: np.ndarray,
    target: int,
    *,
    temporal_quantile: float = 1.0,
) -> np.ndarray:
    dense = np.asarray(dense, dtype=np.float64)
    if len(dense) == 0:
        return dense
    target = int(np.clip(int(target), 3, dense.shape[1]))
    radii = np.asarray(
        [math.sqrt(max(abs(signed_area(frame)), 1.0) / math.pi) for frame in dense],
        dtype=np.float64,
    )
    first, second = _diameter_pair(dense[len(dense) // 2])
    selected = {int(first), int(second)}
    heap: list[tuple[float, int, int, int]] = []

    def push(start: int, end: int) -> None:
        error, split = _track_segment_split(
            dense,
            start,
            end,
            radii,
            temporal_quantile,
        )
        if split is not None:
            heapq.heappush(heap, (-error, int(start), int(end), int(split)))

    push(first, second)
    push(second, first)
    while len(selected) < target and heap:
        _error, start, end, split = heapq.heappop(heap)
        if split in selected:
            continue
        selected.add(split)
        push(start, split)
        push(split, end)
    indices = np.asarray(sorted(selected), dtype=np.int32)
    if len(indices) != target:
        raise RuntimeError(f"track hierarchy produced {len(indices)} != {target}")
    return dense[:, indices]
