"""Track persistent, non-overlapping corner slots with a temporal Viterbi DP."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    align_current_equal_arc,
    signed_area,
)

from .shared_area_dp import _best_cycle, _edge_costs


def _forward_indices(start: int, end: int, count: int) -> np.ndarray:
    distance = (int(end) - int(start)) % int(count)
    return (int(start) + np.arange(distance + 1, dtype=np.int32)) % int(count)


def _edge_frame_cost(
    dense: np.ndarray,
    start: int,
    end: int,
    areas: np.ndarray,
    radii_squared: np.ndarray,
    *,
    distance_weight: float,
    missing_area_weight: float,
    excess_area_weight: float,
) -> np.ndarray:
    indices = _forward_indices(start, end, dense.shape[1])
    if len(indices) <= 2:
        return np.zeros((dense.shape[0],), dtype=np.float64)
    points = dense[:, indices, :]
    crosses = points[:, :-1, 0] * points[:, 1:, 1] - points[:, :-1, 1] * points[:, 1:, 0]
    chord = points[:, -1, 0] * points[:, 0, 1] - points[:, -1, 1] * points[:, 0, 0]
    local = 0.5 * (np.sum(crosses, axis=1) + chord)
    segment = points[:, -1, :] - points[:, 0, :]
    denominator = np.maximum(np.sum(segment * segment, axis=1), 1e-12)
    offset = points[:, 1:-1, :] - points[:, :1, :]
    alpha = np.clip(
        np.sum(offset * segment[:, None, :], axis=2) / denominator[:, None],
        0.0,
        1.0,
    )
    projection = points[:, :1, :] + alpha[:, :, None] * segment[:, None, :]
    maximum_squared = np.max(np.sum((points[:, 1:-1, :] - projection) ** 2, axis=2), axis=1)
    return (
        float(missing_area_weight) * np.maximum(local, 0.0)
        + float(excess_area_weight) * np.maximum(-local, 0.0)
    ) / areas + float(distance_weight) * maximum_squared / radii_squared


def _viterbi(unary: np.ndarray, positions: np.ndarray, temporal_weight: float) -> np.ndarray:
    frames, states = unary.shape
    if states == 1:
        return np.zeros((frames,), dtype=np.int32)
    span = max(float(positions[-1] - positions[0]), 1.0)
    delta = (positions[:, None] - positions[None, :]) / span
    transition = float(temporal_weight) * delta * delta
    costs = np.empty_like(unary)
    parents = np.empty((frames, states), dtype=np.int32)
    costs[0] = unary[0]
    parents[0] = -1
    for frame in range(1, frames):
        values = costs[frame - 1, :, None] + transition
        parents[frame] = np.argmin(values, axis=0)
        costs[frame] = unary[frame] + values[parents[frame], np.arange(states)]
    path = np.empty((frames,), dtype=np.int32)
    path[-1] = int(np.argmin(costs[-1]))
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = parents[frame, path[frame]]
    return path


def persistent_slot_dp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 128,
    allocation_tail_weight: float = 0.5,
    allocation_worst_weight: float = 1.0,
    allocation_distance_weight: float = 0.0,
    temporal_weight: float = 0.05,
    distance_weight: float = 2.0,
    missing_area_weight: float = 1.0,
    excess_area_weight: float = 1.0,
) -> np.ndarray:
    """Place a fixed number of persistent, human-like contour vertices.

    First, an exact track-level DP allocates non-overlapping contour sections.
    Then each section tracks its best local corner with a temporal Viterbi DP.
    A vertex can never enter another vertex's section, so identity swaps and
    full-perimeter drift are impossible by construction.
    """
    dense = align_current_equal_arc(polygons, int(dense_vertices))
    if not len(dense):
        return np.empty((0, int(target), 2), dtype=np.float64)
    aggregate = _edge_costs(
        dense,
        tail_weight=allocation_tail_weight,
        worst_weight=allocation_worst_weight,
        missing_area_weight=missing_area_weight,
        excess_area_weight=excess_area_weight,
        distance_weight=allocation_distance_weight,
    )
    base = np.sort(_best_cycle(aggregate, int(target)))
    samples = int(dense.shape[1])
    areas = np.asarray(
        [max(abs(signed_area(frame)), 1.0) for frame in dense], dtype=np.float64
    )
    radii_squared = np.maximum(areas / math.pi, 1.0)
    output_indices = np.empty((len(dense), len(base)), dtype=np.int32)
    for slot, center in enumerate(base):
        previous = int(base[(slot - 1) % len(base)])
        following = int(base[(slot + 1) % len(base)])
        left_gap = (int(center) - previous) % samples
        right_gap = (following - int(center)) % samples
        # Midpoint boundaries partition the contour.  Endpoints are shared
        # only conceptually; candidate interiors never overlap.
        left = (int(center) - max(left_gap // 2, 1)) % samples
        right = (int(center) + max(right_gap // 2, 1)) % samples
        candidates = _forward_indices(left, right, samples)[1:-1]
        if not len(candidates):
            candidates = np.asarray([int(center)], dtype=np.int32)
        unary = np.empty((len(dense), len(candidates)), dtype=np.float64)
        for state, candidate in enumerate(candidates):
            unary[:, state] = _edge_frame_cost(
                dense,
                left,
                int(candidate),
                areas,
                radii_squared,
                distance_weight=distance_weight,
                missing_area_weight=missing_area_weight,
                excess_area_weight=excess_area_weight,
            ) + _edge_frame_cost(
                dense,
                int(candidate),
                right,
                areas,
                radii_squared,
                distance_weight=distance_weight,
                missing_area_weight=missing_area_weight,
                excess_area_weight=excess_area_weight,
            )
        # Express states in unwrapped slot coordinates for a meaningful
        # temporal distance even when the slot crosses contour index zero.
        unwrapped = np.arange(len(candidates), dtype=np.float64)
        path = _viterbi(unary, unwrapped, temporal_weight)
        output_indices[:, slot] = candidates[path]
    return np.asarray(
        dense[np.arange(len(dense))[:, None], output_indices, :], dtype=np.float64
    )
