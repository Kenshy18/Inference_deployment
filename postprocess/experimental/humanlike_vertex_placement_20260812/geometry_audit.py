"""Geometry and semantic-correspondence metrics for fixed-count polygons."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    orient_ccw,
    resample_closed,
    signed_area,
)


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return {"count": 0}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "q50": float(np.quantile(data, 0.50)),
        "q95": float(np.quantile(data, 0.95)),
        "q99": float(np.quantile(data, 0.99)),
        "maximum": float(np.max(data)),
    }


def _point_segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    delta = end - start
    denominator = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    offset = points[:, None, :] - start[None, :, :]
    alpha = np.clip(np.sum(offset * delta[None, :, :], axis=2) / denominator[None, :], 0.0, 1.0)
    projection = start[None, :, :] + alpha[:, :, None] * delta[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - projection, axis=2), axis=1)


def boundary_reconstruction_metrics(
    references: Iterable[np.ndarray],
    candidates: Iterable[np.ndarray],
) -> dict[str, object]:
    normalized: list[float] = []
    pixels: list[float] = []
    frame_mean: list[float] = []
    frame_p95: list[float] = []
    frame_max: list[float] = []
    frames = 0
    for reference, candidate in zip(references, candidates, strict=True):
        ref = orient_ccw(reference)
        out = orient_ccw(candidate)
        sample = resample_closed(out, max(len(ref), 64))
        distances = np.concatenate([
            _point_segment_distances(ref, out),
            _point_segment_distances(sample, ref),
        ])
        radius = math.sqrt(max(abs(signed_area(ref)), 1.0) / math.pi)
        norm = distances / max(radius, 1.0)
        pixels.extend(distances.tolist())
        normalized.extend(norm.tolist())
        frame_mean.append(float(np.mean(norm)))
        frame_p95.append(float(np.quantile(norm, 0.95)))
        frame_max.append(float(np.max(norm)))
        frames += 1
    return {
        "frames": frames,
        "point_distance_pixels": _distribution(pixels),
        "point_distance_in_equivalent_radius": _distribution(normalized),
        "frame_mean_in_equivalent_radius": _distribution(frame_mean),
        "frame_p95_in_equivalent_radius": _distribution(frame_p95),
        "frame_hausdorff_in_equivalent_radius": _distribution(frame_max),
    }


def _contour_fractions(contour: np.ndarray, points: np.ndarray) -> np.ndarray:
    contour = orient_ccw(contour)
    end = np.roll(contour, -1, axis=0)
    delta = end - contour
    lengths = np.linalg.norm(delta, axis=1)
    total = max(float(np.sum(lengths)), 1e-12)
    denominator = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    offset = points[:, None, :] - contour[None, :, :]
    alpha = np.clip(np.sum(offset * delta[None, :, :], axis=2) / denominator[None, :], 0.0, 1.0)
    projection = contour[None, :, :] + alpha[:, :, None] * delta[None, :, :]
    distances = np.sum((points[:, None, :] - projection) ** 2, axis=2)
    segment = np.argmin(distances, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    fractions = (cumulative[segment] + alpha[np.arange(len(points)), segment] * lengths[segment]) / total
    return np.mod(fractions, 1.0)


def tangential_correspondence_metrics(
    references_by_frame: dict[int, np.ndarray],
    candidates_by_frame: dict[int, np.ndarray],
) -> dict[str, object]:
    common = sorted(set(references_by_frame) & set(candidates_by_frame))
    local_steps: list[float] = []
    global_steps: list[float] = []
    cumulative_ranges: list[float] = []
    allocation_events = 0
    transitions = 0
    cumulative = None
    minimum = maximum = None
    previous_frame = None
    previous = None
    for frame in common:
        points = np.asarray(candidates_by_frame[frame], dtype=np.float64)
        fractions = _contour_fractions(references_by_frame[frame], points)
        if previous is not None and len(previous) == len(fractions):
            gap = max(int(frame) - int(previous_frame), 1)
            delta = (fractions - previous + 0.5) % 1.0 - 0.5
            global_delta = float(np.median(delta))
            local = (delta - global_delta) * len(fractions) / float(gap)
            local_steps.extend(np.abs(local).tolist())
            global_steps.append(abs(global_delta) * len(fractions) / float(gap))
            allocation_events += int(np.sum(np.abs(local) >= 0.5))
            cumulative = local.copy() if cumulative is None else cumulative + local
            minimum = cumulative.copy() if minimum is None else np.minimum(minimum, cumulative)
            maximum = cumulative.copy() if maximum is None else np.maximum(maximum, cumulative)
            transitions += 1
        previous_frame = frame
        previous = fractions
    if minimum is not None:
        cumulative_ranges = (maximum - minimum).tolist()
    return {
        "transitions": int(transitions),
        "local_tangential_step_in_vertex_spacings_per_frame": _distribution(local_steps),
        "global_phase_step_in_vertex_spacings_per_frame": _distribution(global_steps),
        "local_steps_over_half_spacing": int(allocation_events),
        "cumulative_local_drift_range_in_vertex_spacings": _distribution(cumulative_ranges),
    }
