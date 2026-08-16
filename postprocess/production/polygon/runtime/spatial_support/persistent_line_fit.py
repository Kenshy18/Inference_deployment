"""Persistent section-wise boundary-line polygon approximation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .optimizer import (
    align_current_equal_arc,
    has_self_intersection,
    signed_area,
)

from .shared_area_dp import _best_cycle, _edge_costs


def _arc_indices(start: int, end: int, count: int) -> np.ndarray:
    distance = (int(end) - int(start)) % int(count)
    return (int(start) + np.arange(distance + 1, dtype=np.int32)) % int(count)


def _fit_line(points: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    chord = points[-1] - points[0]
    if float(np.dot(direction, chord)) < 0.0:
        direction = -direction
    # For a CCW contour the interior is on the left; this is the outward
    # normal.  A high projection quantile moves the fitted boundary outwards.
    normal = np.asarray([direction[1], -direction[0]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    projections = points @ normal
    offset = float(np.quantile(projections, float(np.clip(quantile, 0.0, 1.0))))
    return normal, offset


def _fit_lines(points: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit one supporting line per frame while batching the quantiles.

    The eigensystem is intentionally evaluated frame by frame so its floating
    point path stays identical to :func:`_fit_line`.  The expensive Python
    dispatch around thousands of independent ``np.quantile`` calls is removed
    by evaluating the row-wise quantiles in one NumPy call.
    """
    values = np.asarray(points, dtype=np.float64)
    normals = np.empty((len(values), 2), dtype=np.float64)
    for frame, frame_points in enumerate(values):
        center = np.mean(frame_points, axis=0)
        centered = frame_points - center
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        chord = frame_points[-1] - frame_points[0]
        if float(np.dot(direction, chord)) < 0.0:
            direction = -direction
        normal = np.asarray([direction[1], -direction[0]], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        normals[frame] = normal
    projections = np.sum(values * normals[:, None, :], axis=2)
    offsets = np.quantile(
        projections,
        float(np.clip(quantile, 0.0, 1.0)),
        axis=1,
    )
    return normals, np.asarray(offsets, dtype=np.float64)


def _intersection(
    left_normal: np.ndarray,
    left_offset: float,
    right_normal: np.ndarray,
    right_offset: float,
    fallback: np.ndarray,
    maximum_distance: float,
    regularization: float,
) -> np.ndarray:
    matrix = np.stack([left_normal, right_normal], axis=0)
    # A literal line intersection is ill-conditioned when adjacent fitted
    # edges become almost parallel and makes a persistent vertex jump.  This
    # ridge solution changes continuously and tends to the old Production
    # contour identity in precisely that ambiguous case.
    ridge = max(float(regularization), 0.0)
    system = matrix.T @ matrix + ridge * np.eye(2, dtype=np.float64)
    right_hand = (
        matrix.T @ np.asarray([left_offset, right_offset], dtype=np.float64)
        + ridge * fallback
    )
    point = np.linalg.solve(system, right_hand)
    delta = point - fallback
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance):
        return fallback.copy()
    if distance > maximum_distance:
        point = fallback + delta * (maximum_distance / max(distance, 1e-12))
    return point


def persistent_line_fit_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    dense_vertices: int = 128,
    coverage_quantile: float = 0.65,
    maximum_intersection_radius: float = 0.35,
    intersection_regularization: float = 0.01,
    allocation_tail_weight: float = 0.5,
    allocation_worst_weight: float = 1.0,
    allocation_distance_weight: float = 0.5,
    missing_area_weight: float = 1.0,
    excess_area_weight: float = 1.0,
    repair_self_intersections: bool = True,
) -> np.ndarray:
    """Approximate persistent contour sections by robust supporting lines."""
    # Vertex IDs must remain in the image-coordinate gauge used by the old
    # Production implementation.  Procrustes phase alignment is suitable for
    # comparing shapes, but it silently rotates the global vertex numbering
    # during real object rotation and destroys keyframe interpolation.
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
    anchors = np.sort(_best_cycle(aggregate, int(target)))
    frames = len(dense)
    lines_normal = np.empty((frames, len(anchors), 2), dtype=np.float64)
    lines_offset = np.empty((frames, len(anchors)), dtype=np.float64)
    for edge, start in enumerate(anchors):
        end = int(anchors[(edge + 1) % len(anchors)])
        indices = _arc_indices(int(start), end, dense.shape[1])
        normals, offsets = _fit_lines(dense[:, indices], coverage_quantile)
        lines_normal[:, edge] = normals
        lines_offset[:, edge] = offsets
    output = np.empty((frames, len(anchors), 2), dtype=np.float64)
    for frame in range(frames):
        radius = math.sqrt(max(abs(signed_area(dense[frame])), 1.0) / math.pi)
        maximum = float(maximum_intersection_radius) * radius
        for vertex, anchor in enumerate(anchors):
            previous_edge = (vertex - 1) % len(anchors)
            current_edge = vertex
            output[frame, vertex] = _intersection(
                lines_normal[frame, previous_edge],
                lines_offset[frame, previous_edge],
                lines_normal[frame, current_edge],
                lines_offset[frame, current_edge],
                dense[frame, int(anchor)],
                maximum,
                intersection_regularization,
            )
        if repair_self_intersections and has_self_intersection(output[frame]):
            fallback = dense[frame, anchors, :]
            displacement = output[frame] - fallback
            repaired = fallback
            # Keep as much of the fitted geometry as possible.  The fallback
            # is a simple ordered contour subset, so this deterministic search
            # always terminates without changing vertex identities.
            for alpha in np.linspace(63.0 / 64.0, 0.0, 64):
                candidate = fallback + float(alpha) * displacement
                if not has_self_intersection(candidate):
                    repaired = candidate
                    break
            output[frame] = repaired
    return output
