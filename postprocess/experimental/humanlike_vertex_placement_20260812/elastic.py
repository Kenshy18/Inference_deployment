"""Curvature-DTW registration followed by track-wise boundary simplification."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _EPS,
    align_temporal_dense,
    signed_area,
)

from .trackwise import trackwise_worst_rdp_from_dense


def _normalize_similarity(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    lc = np.mean(left, axis=0)
    rc = np.mean(right, axis=0)
    left0 = left - lc
    right0 = right - rc
    u, singular, vt = np.linalg.svd(left0.T @ right0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left0 * left0), _EPS))
    aligned = scale * (left0 @ rotation) + rc
    radius = math.sqrt(max(abs(signed_area(right)), 1.0) / math.pi)
    return (aligned - rc[None, :]) / max(radius, 1.0)


def _curvature_features(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    radius = math.sqrt(max(abs(signed_area(value)), 1.0) / math.pi)
    features = []
    for step in (1, 2, 4, 8):
        previous = value - np.roll(value, step, axis=0)
        following = np.roll(value, -step, axis=0) - value
        previous /= np.maximum(np.linalg.norm(previous, axis=1, keepdims=True), _EPS)
        following /= np.maximum(np.linalg.norm(following, axis=1, keepdims=True), _EPS)
        cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
        dot = np.sum(previous * following, axis=1)
        features.append(np.arctan2(cross, dot))
    center = np.mean(value, axis=0)
    radial = np.linalg.norm(value - center[None, :], axis=1) / max(radius, 1.0)
    features.append(radial - np.mean(radial))
    return np.asarray(features, dtype=np.float64).T


def _dtw_mapping(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    reference_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    band: int,
    curvature_weight: float,
    position_weight: float,
) -> np.ndarray:
    count = len(reference_features)
    infinity = float("inf")
    accumulated = np.full((count, count), infinity, dtype=np.float64)
    parent = np.full((count, count), -1, dtype=np.int8)
    for left in range(count):
        low = max(0, left - int(band))
        high = min(count, left + int(band) + 1)
        feature_delta = target_features[low:high] - reference_features[left]
        position_delta = target_positions[low:high] - reference_positions[left]
        local = (
            float(curvature_weight) * np.sum(feature_delta * feature_delta, axis=1)
            + float(position_weight) * np.sum(position_delta * position_delta, axis=1)
        )
        for offset, right in enumerate(range(low, high)):
            cost = float(local[offset])
            if left == 0 and right == 0:
                accumulated[left, right] = cost
                continue
            choices = (
                accumulated[left - 1, right - 1]
                if left > 0 and right > 0
                else infinity,
                accumulated[left - 1, right] if left > 0 else infinity,
                accumulated[left, right - 1] if right > 0 else infinity,
            )
            move = int(np.argmin(choices))
            best = float(choices[move])
            if np.isfinite(best):
                accumulated[left, right] = cost + best
                parent[left, right] = move
    left = count - 1
    right = count - 1
    path: list[tuple[int, int]] = []
    while left >= 0 and right >= 0:
        path.append((left, right))
        if left == 0 and right == 0:
            break
        move = int(parent[left, right])
        if move == 0:
            left -= 1
            right -= 1
        elif move == 1:
            left -= 1
        elif move == 2:
            right -= 1
        else:
            # A disconnected endpoint means the band was too narrow.
            return np.arange(count, dtype=np.float64)
    path.reverse()
    buckets: list[list[int]] = [[] for _ in range(count)]
    for source, destination in path:
        buckets[int(source)].append(int(destination))
    known_x = []
    known_y = []
    for index, bucket in enumerate(buckets):
        if bucket:
            known_x.append(index)
            known_y.append(float(np.median(bucket)))
    mapping = np.interp(
        np.arange(count, dtype=np.float64),
        np.asarray(known_x, dtype=np.float64),
        np.asarray(known_y, dtype=np.float64),
    )
    mapping[0] = 0.0
    mapping[-1] = float(count - 1)
    mapping = np.maximum.accumulate(mapping)
    return mapping


def _sample_mapping(points: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    count = len(points)
    lower = np.floor(mapping).astype(np.int32)
    upper = np.minimum(lower + 1, count - 1)
    alpha = mapping - lower
    return (1.0 - alpha[:, None]) * points[lower] + alpha[:, None] * points[upper]


def elastic_registered_dense(
    polygons: Iterable[np.ndarray],
    *,
    dense_vertices: int = 128,
    band: int = 16,
    curvature_weight: float = 1.0,
    position_weight: float = 0.20,
) -> np.ndarray:
    dense = align_temporal_dense(polygons, int(dense_vertices))
    if len(dense) <= 1:
        return dense
    center = len(dense) // 2
    reference = dense[center]
    reference_features = _curvature_features(reference)
    reference_positions = _normalize_similarity(reference, reference)
    registered = np.empty_like(dense)
    registered[center] = reference
    for frame in range(len(dense)):
        if frame == center:
            continue
        target_features = _curvature_features(dense[frame])
        target_positions = _normalize_similarity(reference, dense[frame])
        mapping = _dtw_mapping(
            reference_features,
            target_features,
            reference_positions,
            target_positions,
            band=band,
            curvature_weight=curvature_weight,
            position_weight=position_weight,
        )
        registered[frame] = _sample_mapping(dense[frame], mapping)
    return registered


def elastic_trackwise_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 128,
    band: int = 16,
    curvature_weight: float = 1.0,
    position_weight: float = 0.20,
    temporal_quantile: float = 1.0,
) -> np.ndarray:
    registered = elastic_registered_dense(
        polygons,
        dense_vertices=dense_vertices,
        band=band,
        curvature_weight=curvature_weight,
        position_weight=position_weight,
    )
    return trackwise_worst_rdp_from_dense(
        registered,
        target,
        temporal_quantile=temporal_quantile,
    )
