"""Band-limited temporal DP for corner-aware, fixed-count vertices."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _EPS,
    _best_phase,
    orient_ccw,
    resample_closed,
    signed_area,
)

from .spatial import _point_segment_distances, rdp_fixed_count


def _nearest_indices(contour: np.ndarray, points: np.ndarray) -> np.ndarray:
    squared = np.sum(
        (points[:, None, :] - contour[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared, axis=1).astype(np.int32)


def _ordered_unwrap(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    output = np.empty_like(values)
    output[0] = int(values[0])
    for position in range(1, len(values)):
        step = (int(values[position]) - int(values[position - 1])) % int(count)
        output[position] = output[position - 1] + max(step, 1)
    return output


def _similarity_prediction(
    previous_contour: np.ndarray,
    current_contour: np.ndarray,
    previous_vertices: np.ndarray,
    samples: int = 64,
) -> np.ndarray:
    left = resample_closed(previous_contour, int(samples))
    right = resample_closed(current_contour, int(samples))
    right = _best_phase(left, right, allow_reverse=False, procrustes=True)
    left_center = np.mean(left, axis=0)
    right_center = np.mean(right, axis=0)
    left_zero = left - left_center
    right_zero = right - right_center
    covariance = left_zero.T @ right_zero
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left_zero * left_zero), _EPS))
    return (
        scale * ((previous_vertices - left_center[None, :]) @ rotation)
        + right_center[None, :]
    )


class _EdgeCosts:
    def __init__(self, contour: np.ndarray, distance_weight: float) -> None:
        self.contour = np.asarray(contour, dtype=np.float64)
        self.count = len(self.contour)
        self.doubled = np.concatenate([self.contour, self.contour], axis=0)
        self.area = max(abs(signed_area(self.contour)), 1.0)
        self.radius = math.sqrt(self.area / math.pi)
        self.distance_weight = float(distance_weight)
        self.cache: dict[tuple[int, int], float] = {}

    def __call__(self, start: int, end: int) -> float:
        start = int(start)
        end = int(end)
        if end <= start or end - start >= self.count:
            return float("inf")
        key = (start, end)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        arc = self.doubled[start : end + 1]
        if len(arc) <= 2:
            self.cache[key] = 0.0
            return 0.0
        cross_sum = float(
            np.sum(
                arc[:-1, 0] * arc[1:, 1]
                - arc[1:, 0] * arc[:-1, 1]
            )
            + arc[-1, 0] * arc[0, 1]
            - arc[0, 0] * arc[-1, 1]
        )
        local_area = 0.5 * abs(cross_sum) / self.area
        maximum_distance = float(
            np.max(_point_segment_distances(arc[1:-1], arc[0], arc[-1]))
        )
        normalized_distance = maximum_distance / max(self.radius, 1.0)
        value = float(
            local_area + self.distance_weight * normalized_distance * normalized_distance
        )
        self.cache[key] = value
        return value


def _propagate_one(
    previous_contour: np.ndarray,
    previous_vertices: np.ndarray,
    previous_indices: np.ndarray,
    current_contour: np.ndarray,
    *,
    band: int,
    temporal_weight: float,
    distance_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    previous_contour = orient_ccw(previous_contour)
    current_contour = orient_ccw(current_contour)
    previous_indices = _ordered_unwrap(previous_indices, len(previous_contour))
    vertex_count = len(previous_vertices)
    current_count = len(current_contour)
    prediction = _similarity_prediction(
        previous_contour,
        current_contour,
        previous_vertices,
    )
    anchor = int(_nearest_indices(current_contour, prediction[:1])[0])
    previous_gaps = np.diff(
        np.concatenate(
            [previous_indices, [previous_indices[0] + len(previous_contour)]]
        )
    ).astype(np.float64)
    cumulative_fraction = np.concatenate(
        [[0.0], np.cumsum(previous_gaps[:-1]) / len(previous_contour)]
    )
    expected = anchor + np.rint(cumulative_fraction * current_count).astype(np.int64)
    doubled = np.concatenate([current_contour, current_contour], axis=0)
    radius = math.sqrt(max(abs(signed_area(current_contour)), 1.0) / math.pi)
    edge_cost = _EdgeCosts(current_contour, distance_weight)
    offsets = np.arange(-max(1, int(band)), max(1, int(band)) + 1, dtype=np.int64)
    best_total = float("inf")
    best_path: list[int] | None = None

    for start in (anchor + offsets):
        start = int(start)
        layers: list[np.ndarray] = [np.asarray([start], dtype=np.int64)]
        valid = True
        for position in range(1, vertex_count):
            centre = int(start + (expected[position] - anchor))
            candidates = centre + offsets
            candidates = candidates[(candidates > start) & (candidates < start + current_count)]
            candidates = np.unique(candidates)
            if not len(candidates):
                valid = False
                break
            layers.append(candidates)
        if not valid:
            continue

        costs = np.asarray(
            [
                temporal_weight
                * float(
                    np.sum(
                        (doubled[start % current_count] - prediction[0]) ** 2
                    )
                )
                / max(radius * radius, 1.0)
            ],
            dtype=np.float64,
        )
        parents: list[np.ndarray] = []
        for position in range(1, vertex_count):
            previous_layer = layers[position - 1]
            current_layer = layers[position]
            next_costs = np.full((len(current_layer),), np.inf, dtype=np.float64)
            next_parents = np.full((len(current_layer),), -1, dtype=np.int32)
            for right_pos, right in enumerate(current_layer):
                node_cost = (
                    temporal_weight
                    * float(
                        np.sum(
                            (
                                doubled[int(right) % current_count]
                                - prediction[position]
                            )
                            ** 2
                        )
                    )
                    / max(radius * radius, 1.0)
                )
                for left_pos, left in enumerate(previous_layer):
                    if int(left) >= int(right) or not np.isfinite(costs[left_pos]):
                        continue
                    value = (
                        costs[left_pos]
                        + edge_cost(int(left), int(right))
                        + node_cost
                    )
                    if value < next_costs[right_pos]:
                        next_costs[right_pos] = value
                        next_parents[right_pos] = int(left_pos)
            costs = next_costs
            parents.append(next_parents)
        for last_pos, last in enumerate(layers[-1]):
            total = costs[last_pos] + edge_cost(int(last), start + current_count)
            if total >= best_total:
                continue
            path = [int(last)]
            cursor = int(last_pos)
            for position in range(vertex_count - 1, 0, -1):
                cursor = int(parents[position - 1][cursor])
                if cursor < 0:
                    path = []
                    break
                path.append(int(layers[position - 1][cursor]))
            if not path:
                continue
            path.reverse()
            best_total = float(total)
            best_path = path
    if best_path is None:
        raise RuntimeError("band DP found no ordered vertex placement")
    indices = np.asarray([value % current_count for value in best_path], dtype=np.int32)
    return current_contour[indices], indices


def temporal_band_dp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    band: int = 8,
    temporal_weight: float = 0.20,
    distance_weight: float = 1.0,
) -> np.ndarray:
    contours = [orient_ccw(value) for value in polygons]
    if not contours:
        return np.empty((0, 0, 2), dtype=np.float64)
    center = len(contours) // 2
    center_vertices = rdp_fixed_count(contours[center], int(target))
    center_indices = _nearest_indices(contours[center], center_vertices)
    output = np.empty((len(contours), int(target), 2), dtype=np.float64)
    indices: list[np.ndarray | None] = [None] * len(contours)
    output[center] = center_vertices
    indices[center] = center_indices
    for frame in range(center + 1, len(contours)):
        output[frame], indices[frame] = _propagate_one(
            contours[frame - 1],
            output[frame - 1],
            np.asarray(indices[frame - 1]),
            contours[frame],
            band=band,
            temporal_weight=temporal_weight,
            distance_weight=distance_weight,
        )
    for frame in range(center - 1, -1, -1):
        output[frame], indices[frame] = _propagate_one(
            contours[frame + 1],
            output[frame + 1],
            np.asarray(indices[frame + 1]),
            contours[frame],
            band=band,
            temporal_weight=temporal_weight,
            distance_weight=distance_weight,
        )
    return output
