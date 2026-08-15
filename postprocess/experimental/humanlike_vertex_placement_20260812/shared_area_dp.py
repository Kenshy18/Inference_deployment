"""Persistent polygon identities selected by a track-level area DP.

The old Production correspondence is retained: every vertex is one fixed
arclength identity over the whole contiguous track.  Unlike equal-arclength
sampling, however, the identities are allocated to contour sections according
to their reconstruction cost across time.  No video pixels are used.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    align_current_equal_arc,
    signed_area,
)


def _edge_costs(
    dense: np.ndarray,
    *,
    tail_weight: float,
    worst_weight: float,
    missing_area_weight: float,
    excess_area_weight: float,
    distance_weight: float = 0.0,
    maximum_distance_frames: int = 96,
) -> np.ndarray:
    """Return cost[start, forward_distance] for one shared contour edge."""
    frames, samples, _ = dense.shape
    doubled = np.concatenate([dense, dense[:, :1]], axis=1)
    crosses = (
        doubled[:, :-1, 0] * doubled[:, 1:, 1]
        - doubled[:, :-1, 1] * doubled[:, 1:, 0]
    )
    # Two copies make every cyclic arc a normal prefix interval.
    cycle_cross = np.concatenate([crosses, crosses], axis=1)
    prefix = np.concatenate(
        [np.zeros((frames, 1), dtype=np.float64), np.cumsum(cycle_cross, axis=1)],
        axis=1,
    )
    # ``dense`` is already an open, ordered contour.  Compute the same
    # shoelace sum directly and avoid repeating normalization/allclose checks
    # for every frame.
    areas = np.asarray(
        [
            max(
                abs(
                    0.5
                    * float(
                        np.sum(
                            frame[:, 0] * np.roll(frame[:, 1], -1)
                            - np.roll(frame[:, 0], -1) * frame[:, 1]
                        )
                    )
                ),
                1.0,
            )
            for frame in dense
        ],
        dtype=np.float64,
    )
    radii_squared = np.maximum(areas / np.pi, 1.0)
    costs = np.full((samples, samples + 1), np.inf, dtype=np.float64)
    costs[:, 1] = 0.0
    starts = np.arange(samples, dtype=np.int64)
    selected_frames: np.ndarray | None = None
    sampled_dense: np.ndarray | None = None
    sampled_radii_squared: np.ndarray | None = None
    if distance_weight > 0.0:
        if frames > int(maximum_distance_frames):
            selected_frames = np.unique(
                np.linspace(
                    0,
                    frames - 1,
                    int(maximum_distance_frames),
                    dtype=np.int64,
                )
            )
        else:
            selected_frames = np.arange(frames, dtype=np.int64)
        sampled_dense = dense[selected_frames]
        sampled_radii_squared = radii_squared[selected_frames]
    for distance in range(2, samples + 1):
        ends = starts + distance
        arc_twice = prefix[:, ends] - prefix[:, starts]
        left = dense[:, starts, :]
        right = dense[:, ends % samples, :]
        chord_twice = right[:, :, 0] * left[:, :, 1] - right[:, :, 1] * left[:, :, 0]
        local = 0.5 * (arc_twice + chord_twice)
        normalized = (
            float(missing_area_weight) * np.maximum(local, 0.0)
            + float(excess_area_weight) * np.maximum(-local, 0.0)
        ) / areas[:, None]
        mean = np.mean(normalized, axis=0)
        tail = np.quantile(normalized, 0.95, axis=0)
        worst = np.max(normalized, axis=0)
        costs[:, distance] = mean + float(tail_weight) * tail + float(worst_weight) * worst
        if distance_weight > 0.0:
            assert sampled_dense is not None
            assert sampled_radii_squared is not None
            interior_offsets = np.arange(1, distance, dtype=np.int64)
            interior_indices = (starts[:, None] + interior_offsets[None, :]) % samples
            interior = sampled_dense[:, interior_indices, :]
            sampled_left = sampled_dense[:, starts, :]
            sampled_right = sampled_dense[:, ends % samples, :]
            segment = sampled_right - sampled_left
            denominator = np.maximum(np.sum(segment * segment, axis=2), 1e-12)
            point_offset = interior - sampled_left[:, :, None, :]
            alpha = np.clip(
                np.sum(point_offset * segment[:, :, None, :], axis=3)
                / denominator[:, :, None],
                0.0,
                1.0,
            )
            projection = sampled_left[:, :, None, :] + alpha[:, :, :, None] * segment[:, :, None, :]
            maximum_squared = np.max(
                np.sum((interior - projection) ** 2, axis=3), axis=2
            )
            distance_normalized = maximum_squared / sampled_radii_squared[:, None]
            distance_mean = np.mean(distance_normalized, axis=0)
            distance_tail = np.quantile(distance_normalized, 0.95, axis=0)
            distance_worst = np.max(distance_normalized, axis=0)
            costs[:, distance] += float(distance_weight) * (
                distance_mean
                + float(tail_weight) * distance_tail
                + float(worst_weight) * distance_worst
            )
    return costs


def _best_cycle(costs: np.ndarray, target: int) -> np.ndarray:
    samples = int(costs.shape[0])
    target = int(np.clip(int(target), 3, samples))
    best_cost = float("inf")
    best_indices: np.ndarray | None = None
    infinity = float("inf")
    # Enumerating the first identity avoids privileging an arbitrary phase.
    # M<=256 and K<=20 in this experiment, so the exact enumeration is small.
    for anchor in range(samples):
        previous = np.full((samples + 1,), infinity, dtype=np.float64)
        previous[0] = 0.0
        parents = np.full((target + 1, samples + 1), -1, dtype=np.int32)
        for edges in range(1, target + 1):
            current = np.full((samples + 1,), infinity, dtype=np.float64)
            lower = edges
            upper = samples - (target - edges)
            for end in range(lower, upper + 1):
                starts = np.arange(edges - 1, end, dtype=np.int64)
                finite = np.isfinite(previous[starts])
                if not np.any(finite):
                    continue
                starts = starts[finite]
                distances = end - starts
                values = previous[starts] + costs[(anchor + starts) % samples, distances]
                selected = int(np.argmin(values))
                current[end] = float(values[selected])
                parents[edges, end] = int(starts[selected])
            previous = current
        value = float(previous[samples])
        if value >= best_cost:
            continue
        positions = np.empty((target,), dtype=np.int32)
        end = samples
        for edges in range(target, 0, -1):
            start = int(parents[edges, end])
            if start < 0:
                raise RuntimeError("broken shared-area DP parent chain")
            positions[edges - 1] = start
            end = start
        best_cost = value
        best_indices = (anchor + positions) % samples
    if best_indices is None:
        raise RuntimeError("shared-area DP has no solution")
    return np.asarray(best_indices, dtype=np.int32)


def shared_area_dp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 128,
    tail_weight: float = 0.5,
    worst_weight: float = 1.0,
    missing_area_weight: float = 1.0,
    excess_area_weight: float = 1.0,
    distance_weight: float = 0.0,
) -> np.ndarray:
    """Select ``target`` persistent contour identities for a whole track."""
    dense = align_current_equal_arc(polygons, int(dense_vertices))
    if not len(dense):
        return np.empty((0, int(target), 2), dtype=np.float64)
    costs = _edge_costs(
        dense,
        tail_weight=tail_weight,
        worst_weight=worst_weight,
        missing_area_weight=missing_area_weight,
        excess_area_weight=excess_area_weight,
        distance_weight=distance_weight,
    )
    indices = _best_cycle(costs, int(target))
    return np.asarray(dense[:, indices, :], dtype=np.float64)
