"""Deterministic, fixed-count, corner-aware polygon approximations.

The routines select actual source-contour points.  Unlike equal-arclength
sampling, vertices naturally accumulate at corners and deep concavities.
Unlike the first temporal decimator, the selected positions are free to differ
per frame; only their count and cyclic correspondence are fixed.
"""

from __future__ import annotations

import heapq
from typing import Callable, Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _best_phase,
    has_self_intersection,
    orient_ccw,
    resample_closed,
)


_EPS = 1e-9


def _point_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(end - start, dtype=np.float64)
    denominator = float(np.dot(delta, delta))
    if denominator <= _EPS:
        return np.linalg.norm(points - start[None, :], axis=1)
    alpha = np.clip(((points - start[None, :]) @ delta) / denominator, 0.0, 1.0)
    projection = start[None, :] + alpha[:, None] * delta[None, :]
    return np.linalg.norm(points - projection, axis=1)


def _forward_indices(start: int, end: int, count: int) -> np.ndarray:
    distance = (int(end) - int(start)) % int(count)
    if distance <= 1:
        return np.empty((0,), dtype=np.int32)
    return (int(start) + np.arange(1, distance, dtype=np.int32)) % int(count)


def _segment_split(
    points: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, int | None]:
    interior = _forward_indices(start, end, len(points))
    if not len(interior):
        return 0.0, None
    distances = _point_segment_distances(
        points[interior], points[int(start)], points[int(end)]
    )
    maximum = float(np.max(distances))
    # Stable tie breaking is important for temporal reproducibility.
    candidates = interior[np.flatnonzero(np.isclose(distances, maximum, atol=1e-12))]
    return maximum, int(candidates[0])


def _diameter_pair(points: np.ndarray) -> tuple[int, int]:
    delta = points[:, None, :] - points[None, :, :]
    squared = np.sum(delta * delta, axis=2)
    first, second = np.unravel_index(int(np.argmax(squared)), squared.shape)
    if first == second:
        second = (first + len(points) // 2) % len(points)
    return int(first), int(second)


def rdp_fixed_count(points: np.ndarray, target: int) -> np.ndarray:
    """Closed-curve Douglas-Peucker hierarchy stopped at an exact count."""
    contour = orient_ccw(points)
    target = max(3, int(target))
    if len(contour) < target:
        contour = resample_closed(contour, max(target * 4, 32))
    target = int(np.clip(target, 3, len(contour)))
    if target >= len(contour):
        return contour.copy()
    first, second = _diameter_pair(contour)
    selected = {first, second}
    heap: list[tuple[float, int, int, int]] = []

    def push(start: int, end: int) -> None:
        error, split = _segment_split(contour, start, end)
        if split is not None:
            heapq.heappush(heap, (-error, int(start), int(end), int(split)))

    push(first, second)
    push(second, first)
    while len(selected) < target and heap:
        _negative_error, start, end, split = heapq.heappop(heap)
        if split in selected:
            continue
        selected.add(split)
        push(start, split)
        push(split, end)
    ordered = np.asarray(sorted(selected), dtype=np.int32)
    result = contour[ordered]
    if len(result) != target:
        raise RuntimeError(f"RDP hierarchy produced {len(result)} != {target} vertices")
    return result


def _triangle_area(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
    return 0.5 * abs(
        float(
            (current[0] - previous[0]) * (following[1] - previous[1])
            - (current[1] - previous[1]) * (following[0] - previous[0])
        )
    )


def visvalingam_fixed_count(points: np.ndarray, target: int) -> np.ndarray:
    """Remove the smallest local area while retaining an exact count."""
    contour = orient_ccw(points)
    count = len(contour)
    target = int(np.clip(int(target), 3, count))
    if target >= count:
        return contour.copy()
    previous = np.arange(count, dtype=np.int32) - 1
    previous[0] = count - 1
    following = (np.arange(count, dtype=np.int32) + 1) % count
    active = np.ones((count,), dtype=bool)
    versions = np.zeros((count,), dtype=np.int32)
    heap: list[tuple[float, int, int]] = []

    def push(index: int) -> None:
        if not active[index]:
            return
        area = _triangle_area(
            contour[int(previous[index])], contour[index], contour[int(following[index])]
        )
        heapq.heappush(heap, (float(area), int(index), int(versions[index])))

    for index in range(count):
        push(index)
    remaining = count
    while remaining > target and heap:
        _area, index, version = heapq.heappop(heap)
        if not active[index] or int(versions[index]) != version:
            continue
        left = int(previous[index])
        right = int(following[index])
        active[index] = False
        following[left] = right
        previous[right] = left
        versions[left] += 1
        versions[right] += 1
        push(left)
        push(right)
        remaining -= 1
    result = contour[np.flatnonzero(active)]
    if len(result) != target:
        raise RuntimeError(
            f"Visvalingam hierarchy produced {len(result)} != {target} vertices"
        )
    return result


def align_polygon_sequence(
    polygons: Iterable[np.ndarray],
    *,
    allow_reverse: bool = False,
    phase_mode: str = "translation",
) -> np.ndarray:
    """Assign cyclic vertex IDs from the temporal centre outwards.

    ``translation`` removes only centroid motion.  Rotation deliberately
    remains observable: removing it made approximately symmetric masks
    ambiguous and occasionally changed the phase by half a perimeter.
    ``similarity`` is retained only as an ablation of the rejected behaviour.
    """
    values = [orient_ccw(value) for value in polygons]
    if not values:
        return np.empty((0, 0, 2), dtype=np.float64)
    count = len(values[0])
    if any(len(value) != count for value in values):
        raise ValueError("all frames must have the same vertex count")
    center = len(values) // 2
    aligned = np.empty((len(values), count, 2), dtype=np.float64)
    aligned[center] = values[center]

    def best(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if phase_mode == "similarity":
            return _best_phase(
                reference,
                candidate,
                allow_reverse=allow_reverse,
                procrustes=True,
            )
        if phase_mode != "translation":
            raise ValueError(f"unknown phase_mode: {phase_mode}")
        reference_zero = reference - np.mean(reference, axis=0, keepdims=True)
        variants = [candidate]
        if allow_reverse:
            variants.append(candidate[::-1])
        best_value = variants[0]
        best_cost = float("inf")
        for variant in variants:
            variant_zero = variant - np.mean(variant, axis=0, keepdims=True)
            for shift in range(count):
                rolled_zero = np.roll(variant_zero, -shift, axis=0)
                cost = float(np.mean(np.sum((reference_zero - rolled_zero) ** 2, axis=1)))
                if cost < best_cost:
                    best_cost = cost
                    best_value = np.roll(variant, -shift, axis=0)
        return np.asarray(best_value, dtype=np.float64).copy()

    for frame in range(center + 1, len(values)):
        aligned[frame] = best(aligned[frame - 1], values[frame])
    for frame in range(center - 1, -1, -1):
        aligned[frame] = best(aligned[frame + 1], values[frame])
    return aligned


def approximate_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    method: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    approximated = [method(value, int(target)) for value in polygons]
    aligned = align_polygon_sequence(approximated)
    if any(has_self_intersection(value) for value in aligned):
        raise ValueError("approximation created a self-intersecting polygon")
    return aligned
