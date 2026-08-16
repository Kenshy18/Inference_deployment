#!/usr/bin/env python3
"""Deterministic role-based polygon candidates for the 0809 experiment.

The implementation intentionally consumes only the already decoded polygon
geometry in ``InstanceRun``.  It never opens video pixels.  Raster work is
limited to small per-component ROIs and is converted back to the run's fixed
vertex representation before entering the existing CUDA interval evaluator.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable

import cv2
import numpy as np

from experimental.polygon_recall_optimizer.temporal_candidates import _align_order
from overlay_renderer.keyframe_cache import _numpy_resample


ROLE_IDS = (
    "A2", "A4", "D6", "B3", "C1", "C6", "E2", "F3", "G3", "Z1",
    "A07", "A06", "G02", "G04", "C02", "E02",
    "C02_115", "C02_125",
    "C02_120", "C02_130",
    "A06_K3", "A06_K4",
    "G02_H3", "G04_H3", "G02_H8", "G04_H8",
    "GF8_K2_135", "GB8_K2_135", "GF8_K2_150", "GB8_K2_150",
    "GF8_K3_150", "GB8_K3_150", "GF12_K2_150", "GB12_K2_150",
    "GF8_K2_200", "GB8_K2_200", "GFT8_K2_200", "GBT8_K2_200",
    "GF8_K1_200", "GB8_K1_200", "GF8_K1_300", "GB8_K1_300",
    "F3_Q65", "F3_Q75", "D6_R5",
    "CTR4_125", "CTR4_150", "CTR8_150",
    "VF6", "VB6", "VF8", "VB8", "VF10", "VB10", "VF12", "VB12",
    "VFR8", "VBR8",
    "LSF8_110", "LSB8_110", "LSF8_115", "LSB8_115",
    "IVF8_110", "IVB8_110", "IVF8_115", "IVB8_115", "IVF8_120", "IVB8_120",
    "A2_P1", "A4_P1", "D6_P1", "B3_P1", "C1_P1", "C6_P1",
    "E2_P1", "F3_P1", "G3_P1", "Z1_P1",
    "S30_R2_A105", "S30_R2_A125",
    "S30_R5_A105", "S30_R5_A125",
    "S30_R10_A105", "S30_R10_A125",
)


_ROLE_FRAME_LOCAL = threading.local()


def _centre(points: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(points, dtype=np.float64), axis=0)


def _translation_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ordered = _align_order(reference, candidate)
    return ordered + (_centre(reference) - _centre(ordered))


def _similarity_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Procrustes translation/rotation/isotropic-scale alignment."""
    ordered = _align_order(reference, candidate)
    rc = _centre(reference)
    cc = _centre(ordered)
    left = ordered - cc
    right = reference - rc
    covariance = left.T @ right
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left * left), 1e-9))
    scale = float(np.clip(scale, 0.65, 1.55))
    return scale * (left @ rotation) + rc


def _polygon_area(points: np.ndarray) -> float:
    value = np.asarray(points, dtype=np.float64)
    return abs(float(cv2.contourArea(value.astype(np.float32))))


def _run_frame_numbers(run) -> np.ndarray:
    """Return the immutable frame-number vector without rebuilding it per role."""
    cached = getattr(run, "_orthogonal_role_frame_numbers", None)
    if cached is None:
        cached = np.asarray(run.frame_numbers, dtype=np.int64)
        setattr(run, "_orthogonal_role_frame_numbers", cached)
    return cached


def _frame_neighbours(run, frame_index: int, radius: int) -> np.ndarray:
    cache = _frame_cache(run, frame_index)
    key = ("frame_neighbours", int(radius))
    if key in cache:
        return cache[key]
    frames = _run_frame_numbers(run)
    current = int(frames[int(frame_index)])
    lo = int(np.searchsorted(frames, current - int(radius), side="left"))
    hi = int(np.searchsorted(frames, current + int(radius), side="right"))
    value = np.arange(lo, hi, dtype=np.int32)
    cache[key] = value
    return value


def _aligned_window(
    run, frame_index: int, slot: int, radius: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    cache = _frame_cache(run, frame_index)
    key = ("aligned_window", int(slot), int(radius), str(mode))
    if key in cache:
        return cache[key]

    # A larger window aligned to the same current-frame reference contains
    # byte-identical values for every smaller window.  A06 is evaluated before
    # F3 in the fixed baseline, so its radius-5 translation alignment can feed
    # the radius-2 local-normal candidate without repeating vertex matching.
    reusable = []
    for cached_key, cached_value in cache.items():
        if (
            isinstance(cached_key, tuple)
            and len(cached_key) == 4
            and cached_key[0] == "aligned_window"
            and int(cached_key[1]) == int(slot)
            and str(cached_key[3]) == str(mode)
            and int(cached_key[2]) >= int(radius)
        ):
            reusable.append((int(cached_key[2]), cached_value))
    if reusable:
        _cached_radius, (cached_indices, cached_aligned) = min(
            reusable, key=lambda item: item[0]
        )
        frames = _run_frame_numbers(run)
        current = int(frames[int(frame_index)])
        keep = np.abs(frames[np.asarray(cached_indices, dtype=np.int32)] - current) <= int(
            radius
        )
        value = (
            np.asarray(cached_indices, dtype=np.int32)[keep],
            np.asarray(cached_aligned)[keep],
        )
        cache[key] = value
        return value

    reference = np.asarray(run.anchors[int(frame_index)][int(slot)], dtype=np.float64)
    indices = _frame_neighbours(run, frame_index, radius)
    aligner = _translation_align if mode == "T" else _similarity_align
    aligned = np.stack(
        [
            aligner(
                reference,
                np.asarray(run.anchors[int(index)][int(slot)], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    value = (indices, aligned)
    cache[key] = value
    return value


def _frame_cache(run, frame_index: int) -> dict:
    """Small cache shared by all role rules for the current candidate frame."""
    cache = getattr(_ROLE_FRAME_LOCAL, "cache", None)
    if (
        cache is None
        or cache.get("run") is not run
        or int(cache.get("frame_index", -1)) != int(frame_index)
    ):
        cache = {"run": run, "frame_index": int(frame_index), "values": {}}
        _ROLE_FRAME_LOCAL.cache = cache
    return cache["values"]


def _shared_window(run, frame_index: int, slot: int, radius: int, mode: str):
    indices, aligned, masks, origin = _shared_raster_window(
        run, frame_index, slot, radius, mode
    )
    cache = _frame_cache(run, frame_index)
    key = ("tsdf", int(slot), int(radius), str(mode))
    if key not in cache:
        cache[key] = _tsdf_stack(masks)
    return indices, aligned, masks, origin, cache[key]


def _shared_raster_window(
    run, frame_index: int, slot: int, radius: int, mode: str
):
    """Cache alignment and rasterization independently from optional TSDF."""

    cache = _frame_cache(run, frame_index)
    key = ("raster_window", int(slot), int(radius), str(mode))
    if key not in cache:
        indices, aligned = _aligned_window(run, frame_index, slot, radius, mode)
        masks, origin = _raster_stack(aligned)
        cache[key] = (indices, aligned, masks, origin)
    return cache[key]


def _shared_support_polygon(
    run,
    frame_index: int,
    slot: int,
    radius: int,
    support_fraction: float = 0.30,
) -> np.ndarray:
    """Translation-aligned temporal support shape shared by size variants.

    The expensive alignment, rasterization, support count and contour
    extraction are performed once per (frame, component, radius).  Area
    variants only rescale this cached polygon, so six states require three
    temporal aggregations rather than six.

    ``ceil(fraction * observations)`` gives the literal robust-union rule:
    a pixel must be present in at least three of ten observations, four of
    eleven, and so on.  The current raw mask is not unioned into the result;
    endpoint feasibility remains the responsibility of the hard Recall gate.
    """

    cache = _frame_cache(run, frame_index)
    key = (
        "support_polygon",
        int(slot),
        int(radius),
        round(float(support_fraction), 6),
    )
    if key in cache:
        return np.asarray(cache[key], dtype=np.float64)

    reference = np.asarray(
        run.anchors[int(frame_index)][int(slot)], dtype=np.float64
    )
    _indices, aligned = _aligned_window(
        run, int(frame_index), int(slot), int(radius), "T"
    )
    masks, origin = _raster_stack(aligned)
    required = max(
        1,
        int(math.ceil(float(support_fraction) * float(len(masks)) - 1e-12)),
    )
    supported = np.sum(masks, axis=0, dtype=np.int16) >= int(required)
    polygon = _mask_to_polygon(supported, origin, reference)
    result = np.asarray(reference if polygon is None else polygon, dtype=np.float64)
    cache[key] = result
    return result


def _support_area_variant(
    run,
    frame_index: int,
    slot: int,
    *,
    radius: int,
    area_factor: float,
) -> np.ndarray:
    """Scale a shared temporal support shape by a literal area multiplier."""

    polygon = _shared_support_polygon(run, frame_index, slot, radius, 0.30)
    center = _centre(polygon)
    coordinate_factor = math.sqrt(float(area_factor))
    return center + coordinate_factor * (polygon - center)


def _s30_r2_a105(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=2, area_factor=1.05
    )


def _s30_r2_a125(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=2, area_factor=1.25
    )


def _s30_r5_a105(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=5, area_factor=1.05
    )


def _s30_r5_a125(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=5, area_factor=1.25
    )


def _s30_r10_a105(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=10, area_factor=1.05
    )


def _s30_r10_a125(run, frame_index: int, slot: int) -> np.ndarray:
    return _support_area_variant(
        run, frame_index, slot, radius=10, area_factor=1.25
    )


def _raster_stack(polygons: np.ndarray, padding: int = 7):
    all_points = np.concatenate([np.asarray(value, dtype=np.float64) for value in polygons])
    low = np.floor(np.min(all_points, axis=0) - int(padding)).astype(np.int32)
    high = np.ceil(np.max(all_points, axis=0) + int(padding)).astype(np.int32)
    width, height = np.maximum(high - low + 1, 3).tolist()
    masks = np.zeros((len(polygons), int(height), int(width)), dtype=np.uint8)
    for index, points in enumerate(polygons):
        local = np.rint(np.asarray(points) - low[None, :]).astype(np.int32)
        cv2.fillPoly(masks[index], [local], 1)
    return masks, low.astype(np.float64)


def _tsdf_stack(masks: np.ndarray) -> np.ndarray:
    values = []
    for mask in masks:
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        radius = math.sqrt(max(float(mask.sum()), 1.0) / math.pi)
        limit = max(2.0, 0.15 * radius)
        values.append(np.clip(outside - inside, -limit, limit))
    return np.asarray(values, dtype=np.float32)


def _mask_to_polygon(
    mask: np.ndarray, origin: np.ndarray, reference: np.ndarray
) -> np.ndarray | None:
    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 3 or cv2.contourArea(contour.astype(np.float32)) < 1.0:
        return None
    contour += np.asarray(origin, dtype=np.float64)[None, :]
    contour = _numpy_resample(contour, len(reference))
    return _align_order(np.asarray(reference, dtype=np.float64), contour)


def _aggregate_polygon(
    reference: np.ndarray,
    aligned: np.ndarray,
    *,
    quantile: float = 0.5,
) -> np.ndarray:
    masks, origin = _raster_stack(aligned)
    if float(quantile) == 0.5 and len(masks) % 2 == 1:
        # For an odd stack the median signed distance is <= 0 exactly when a
        # strict majority of masks contains the pixel.  Avoiding two OpenCV
        # distance transforms per mask preserves the zero level set bit for
        # bit.  Even boundary windows retain the former TSDF path below.
        selected = np.sum(masks, axis=0, dtype=np.int16) >= (len(masks) // 2 + 1)
    else:
        phi = np.quantile(_tsdf_stack(masks), float(quantile), axis=0)
        selected = phi <= 0.0
    polygon = _mask_to_polygon(selected, origin, reference)
    return np.asarray(reference if polygon is None else polygon, dtype=np.float64)


def _area_zscores(run, slot: int) -> np.ndarray:
    cache = getattr(run, "_phase2_role_area_zscores", None)
    if cache is None:
        cache = {}
        setattr(run, "_phase2_role_area_zscores", cache)
    if int(slot) in cache:
        return cache[int(slot)]
    areas = np.asarray(
        [_polygon_area(frame[int(slot)]) for frame in run.anchors], dtype=np.float64
    )
    log_area = np.log(np.maximum(areas, 1.0))
    median = float(np.median(log_area))
    mad = float(np.median(np.abs(log_area - median)))
    result = (log_area - median) / max(1.4826 * mad, 0.08)
    cache[int(slot)] = result
    return result


def _a2(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "T")
    return _aggregate_polygon(reference, aligned)


def _a4(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "S")
    return _aggregate_polygon(reference, aligned)


def _d6_with_radius(run, frame_index: int, slot: int, radius: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    indices, aligned = _aligned_window(run, frame_index, slot, radius, "S")
    latent = _aggregate_polygon(reference, aligned)
    centres = np.asarray([_centre(run.anchors[int(i)][slot]) for i in indices])
    areas = np.asarray([_polygon_area(run.anchors[int(i)][slot]) for i in indices])
    target_center = np.median(centres, axis=0)
    target_area = float(np.median(areas))
    scale = math.sqrt(target_area / max(_polygon_area(latent), 1e-9))
    scale = float(np.clip(scale, 0.90, 1.12))
    return target_center + scale * (latent - _centre(latent))


def _d6(run, frame_index: int, slot: int) -> np.ndarray:
    return _d6_with_radius(run, frame_index, slot, 2)


def _d6_r5(run, frame_index: int, slot: int) -> np.ndarray:
    return _d6_with_radius(run, frame_index, slot, 5)


def _b3(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    frames = _run_frame_numbers(run)
    current_frame = int(frames[frame_index])
    indices = _frame_neighbours(run, frame_index, 8)
    zscores = _area_zscores(run, slot)
    reliable = [int(i) for i in indices if abs(float(zscores[int(i)])) <= 2.5]
    before = [i for i in reliable if int(frames[i]) < current_frame]
    after = [i for i in reliable if int(frames[i]) > current_frame]
    if not before or not after:
        return reference
    left, right = before[-1], after[0]
    left_points = np.asarray(run.anchors[left][slot], dtype=np.float64)
    right_points = _align_order(
        left_points, np.asarray(run.anchors[right][slot], dtype=np.float64)
    )
    alpha = (current_frame - int(frames[left])) / max(
        int(frames[right]) - int(frames[left]), 1
    )
    return (1.0 - alpha) * left_points + alpha * right_points


def _c1(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "T")
    masks, origin = _raster_stack(np.concatenate((aligned, reference[None]), axis=0))
    temporal, raw = masks[:-1], masks[-1]
    phi = _tsdf_stack(temporal)
    median = np.quantile(phi, 0.50, axis=0) <= 0.0
    envelope = (np.quantile(phi, 0.35, axis=0) <= 0.0).astype(np.uint8)
    radius = max(1, int(round(math.sqrt(max(float(raw.sum()), 1.0) / math.pi) * 0.03)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    cap = cv2.dilate(envelope, kernel)
    result = np.logical_and(np.logical_or(raw, median), cap)
    polygon = _mask_to_polygon(result, origin, reference)
    return reference if polygon is None else polygon


def _c6(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "T")
    masks, origin = _raster_stack(np.concatenate((aligned, reference[None]), axis=0))
    temporal, raw = masks[:-1], masks[-1].astype(bool)
    support = np.mean(temporal, axis=0)
    result = np.logical_or(raw, support >= 0.60)
    result = np.logical_and(result, np.logical_or(~raw, support > 0.15))
    result = cv2.morphologyEx(
        result.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    polygon = _mask_to_polygon(result, origin, reference)
    return reference if polygon is None else polygon


def _e2(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "T")
    masks, origin = _raster_stack(aligned)
    support = np.mean(masks, axis=0)
    core = (support >= 0.70).astype(np.uint8)
    envelope = (support >= 0.25).astype(np.uint8)
    count, labels = cv2.connectedComponents(envelope)
    result = np.zeros_like(envelope)
    for component in range(1, count):
        region = labels == component
        if np.any(core[region]):
            result[region] = 1
    polygon = _mask_to_polygon(result, origin, reference)
    return reference if polygon is None else polygon


def _f3_with_quantile(
    run, frame_index: int, slot: int, quantile: float
) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    # Translation-only alignment is deliberate: a similarity alignment would
    # normalize away exactly the local scale/axis deficit that this outward
    # normal candidate is meant to repair.
    _indices, aligned = _aligned_window(run, frame_index, slot, 2, "T")
    previous = np.roll(reference, 1, axis=0)
    following = np.roll(reference, -1, axis=0)
    tangent = following - previous
    normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    center_direction = reference - _centre(reference)
    flip = np.sum(normal * center_direction, axis=1) < 0.0
    normal[flip] *= -1.0
    offsets = np.sum((aligned - reference[None]) * normal[None], axis=2)
    outward = np.maximum(np.quantile(offsets, float(quantile), axis=0), 0.0)
    outward = (
        np.roll(outward, 1) + 2.0 * outward + np.roll(outward, -1)
    ) / 4.0
    cap = max(2.0, 0.12 * math.sqrt(max(_polygon_area(reference), 1.0) / math.pi))
    return reference + np.minimum(outward, cap)[:, None] * normal


def _f3(run, frame_index: int, slot: int) -> np.ndarray:
    return _f3_with_quantile(run, frame_index, slot, 0.50)


def _f3_q65(run, frame_index: int, slot: int) -> np.ndarray:
    return _f3_with_quantile(run, frame_index, slot, 0.65)


def _f3_q75(run, frame_index: int, slot: int) -> np.ndarray:
    return _f3_with_quantile(run, frame_index, slot, 0.75)


def _frame_size() -> tuple[float | None, float | None]:
    width = float(os.environ["MASK_PIPELINE_PHASE2_FRAME_WIDTH"]) if os.environ.get(
        "MASK_PIPELINE_PHASE2_FRAME_WIDTH"
    ) else None
    height = float(os.environ["MASK_PIPELINE_PHASE2_FRAME_HEIGHT"]) if os.environ.get(
        "MASK_PIPELINE_PHASE2_FRAME_HEIGHT"
    ) else None
    return width, height


def _touches_edge(points: np.ndarray, margin: float = 4.0) -> bool:
    width, height = _frame_size()
    x, y = np.asarray(points)[:, 0], np.asarray(points)[:, 1]
    return bool(
        np.min(x) <= margin
        or np.min(y) <= margin
        or (width is not None and np.max(x) >= width - 1.0 - margin)
        or (height is not None and np.max(y) >= height - 1.0 - margin)
    )


def _g3(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    if not _touches_edge(reference):
        return _f3(run, frame_index, slot)
    indices = _frame_neighbours(run, frame_index, 12)
    candidates = [
        int(i)
        for i in indices
        if not _touches_edge(np.asarray(run.anchors[int(i)][slot]))
    ]
    if not candidates:
        return reference
    nearest = min(candidates, key=lambda i: abs(i - frame_index))
    source = np.asarray(run.anchors[nearest][slot], dtype=np.float64)
    return _translation_align(reference, source)


def _z1(run, frame_index: int, slot: int) -> np.ndarray:
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    indices = _frame_neighbours(run, frame_index, 4)
    target = float(
        np.median([_polygon_area(run.anchors[int(i)][slot]) for i in indices])
    )
    current = max(_polygon_area(reference), 1e-9)
    factor = math.sqrt(max(target, current) / current)
    factor = float(np.clip(factor, 1.0, 1.16))
    center = _centre(reference)
    return center + factor * (reference - center)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if int(radius) <= 0:
        return np.asarray(mask, dtype=np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(radius) + 1, 2 * int(radius) + 1)
    )
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel)


def _cap_union_area(
    candidate: np.ndarray, raw: np.ndarray, max_ratio: float = 1.35
) -> np.ndarray:
    """Keep raw plus the closest supported additions under an area hard cap."""
    candidate_mask = np.asarray(candidate, dtype=bool)
    raw_mask = np.asarray(raw, dtype=bool)
    raw_area = int(np.count_nonzero(raw_mask))
    limit = max(raw_area, int(math.floor(float(max_ratio) * raw_area)))
    if int(np.count_nonzero(candidate_mask)) <= limit:
        return candidate_mask.astype(np.uint8)
    output = raw_mask.copy()
    additions = np.logical_and(candidate_mask, ~raw_mask)
    allowance = max(0, limit - raw_area)
    ys, xs = np.nonzero(additions)
    if allowance <= 0 or not len(xs):
        return output.astype(np.uint8)
    distance = cv2.distanceTransform((~raw_mask).astype(np.uint8), cv2.DIST_L2, 3)
    order = np.lexsort((xs, ys, distance[ys, xs]))
    chosen = order[:allowance]
    output[ys[chosen], xs[chosen]] = True
    return output.astype(np.uint8)


def _orthogonal_order_stat(
    run,
    frame_index: int,
    slot: int,
    *,
    kth: int | None,
    union_raw: bool,
) -> np.ndarray:
    """A06/A07: shared +/-5 translation-aligned temporal tube."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    if kth is None:
        _indices, _aligned, masks, origin, phi = _shared_window(
            run, frame_index, slot, 5, "T"
        )
        candidate_mask = np.median(phi, axis=0) <= 0.0
    else:
        _indices, _aligned, masks, origin = _shared_raster_window(
            run, frame_index, slot, 5, "T"
        )
        needed = min(max(int(kth), 1), len(masks))
        # The window is at most 11 binary masks, so int16 is exact. NumPy's
        # default uint8 reduction promotes to uint64 and was the dominant A06
        # candidate-generation cost on long tracks.
        candidate_mask = np.sum(masks, axis=0, dtype=np.int16) >= needed
    if union_raw:
        # The shared ROI contains the current aligned mask already.  Use its
        # exact raster rather than remapping a second origin.
        current_index = int(np.argmin(np.abs(_indices - int(frame_index))))
        raw_mask = masks[current_index]
        candidate_mask = np.logical_or(candidate_mask, raw_mask)
    candidate_mask = _dilate(candidate_mask, 1)
    if union_raw:
        candidate_mask = _cap_union_area(candidate_mask, raw_mask)
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _a07(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(
        run, frame_index, slot, kth=None, union_raw=False
    )


def _a06(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(
        run, frame_index, slot, kth=2, union_raw=True
    )


def _a06_k3(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(
        run, frame_index, slot, kth=3, union_raw=True
    )


def _a06_k4(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(
        run, frame_index, slot, kth=4, union_raw=True
    )


def _directional_endpoint_envelope(
    run,
    frame_index: int,
    slot: int,
    *,
    forward: bool,
    horizon: int = 5,
) -> np.ndarray:
    """G02/G04 endpoint envelope from two persistent one-sided shapes."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    length = int(len(run.frame_numbers))
    direction = 1 if forward else -1
    horizon = max(2, int(horizon))
    offsets = (horizon, horizon - 1)
    indices = [frame_index + direction * value for value in offsets]
    if any(index < 0 or index >= length for index in indices):
        return reference
    aligned = np.stack(
        [
            _similarity_align(
                reference,
                np.asarray(run.anchors[int(index)][slot], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    masks, origin = _raster_stack(np.concatenate((aligned, reference[None]), axis=0))
    persistent = np.logical_and(masks[0], masks[1])
    candidate_mask = np.logical_or(masks[-1], persistent)
    candidate_mask = _dilate(candidate_mask, 1)
    candidate_mask = _cap_union_area(candidate_mask, masks[-1])
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _g02(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=True
    )


def _g04(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=False
    )


def _g02_h3(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=True, horizon=3
    )


def _g04_h3(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=False, horizon=3
    )


def _g02_h8(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=True, horizon=8
    )


def _g04_h8(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=False, horizon=8
    )


def _directional_support_envelope(
    run,
    frame_index: int,
    slot: int,
    *,
    forward: bool,
    horizon: int,
    support_count: int,
    max_ratio: float,
    alignment: str = "S",
) -> np.ndarray:
    """Pose-normalized one-sided persistent non-rigid shape envelope."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    direction = 1 if bool(forward) else -1
    indices = [
        frame_index + direction * offset
        for offset in range(1, int(horizon) + 1)
        if 0 <= frame_index + direction * offset < len(run.frame_numbers)
    ]
    if len(indices) < int(support_count):
        return reference
    aligner = _translation_align if str(alignment) == "T" else _similarity_align
    aligned = np.stack(
        [
            aligner(
                reference,
                np.asarray(run.anchors[int(index)][slot], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    masks, origin = _raster_stack(
        np.concatenate((aligned, reference[None]), axis=0), padding=5
    )
    raw = masks[-1]
    persistent = np.sum(masks[:-1], axis=0, dtype=np.int16) >= int(support_count)
    candidate_mask = _dilate(np.logical_or(raw, persistent), 1)
    candidate_mask = _cap_union_area(
        candidate_mask, raw, max_ratio=float(max_ratio)
    )
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _gf8_k2_135(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=2, max_ratio=1.35,
    )


def _gb8_k2_135(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=2, max_ratio=1.35,
    )


def _gf8_k2_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=2, max_ratio=1.50,
    )


def _gb8_k2_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=2, max_ratio=1.50,
    )


def _gf8_k3_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=3, max_ratio=1.50,
    )


def _gb8_k3_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=3, max_ratio=1.50,
    )


def _gf12_k2_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=12,
        support_count=2, max_ratio=1.50,
    )


def _gb12_k2_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=12,
        support_count=2, max_ratio=1.50,
    )


def _gf8_k2_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=2, max_ratio=2.00,
    )


def _gb8_k2_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=2, max_ratio=2.00,
    )


def _gft8_k2_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=2, max_ratio=2.00, alignment="T",
    )


def _gbt8_k2_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=2, max_ratio=2.00, alignment="T",
    )


def _gf8_k1_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=1, max_ratio=2.00,
    )


def _gb8_k1_200(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=1, max_ratio=2.00,
    )


def _gf8_k1_300(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=True, horizon=8,
        support_count=1, max_ratio=3.00,
    )


def _gb8_k1_300(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_support_envelope(
        run, frame_index, slot, forward=False, horizon=8,
        support_count=1, max_ratio=3.00,
    )


def _c02_with_cap(
    run, frame_index: int, slot: int, *, max_ratio: float
) -> np.ndarray:
    """Motion-direction Minkowski sweep over a five-frame horizon."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    indices = _frame_neighbours(run, frame_index, 2)
    if len(indices) < 2:
        return reference
    frames = np.asarray(_run_frame_numbers(run)[indices], dtype=np.float64)
    centres = np.asarray([_centre(run.anchors[int(i)][slot]) for i in indices])
    design = np.stack((frames - np.mean(frames), np.ones(len(frames))), axis=1)
    velocity = np.linalg.lstsq(design, centres, rcond=None)[0][0]
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or speed < 0.20:
        return reference
    half_sweep = min(2.5 * speed, 0.30 * math.sqrt(max(_polygon_area(reference), 1.0)))
    direction = velocity / max(speed, 1e-9)
    shifts = np.linspace(-half_sweep, half_sweep, 9)
    swept = np.stack(
        [reference + float(value) * direction[None, :] for value in shifts], axis=0
    )
    masks, origin = _raster_stack(swept, padding=4)
    candidate_mask = _dilate(np.any(masks, axis=0), 1)
    candidate_mask = _cap_union_area(
        candidate_mask, masks[len(masks) // 2], max_ratio=max_ratio
    )
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _c02(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.35)


def _c02_115(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.15)


def _c02_125(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.25)


def _c02_120(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.20)


def _c02_130(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.30)


def _c02_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.50)


def _c02_175(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.75)


def _trajectory_residual_sweep(
    run,
    frame_index: int,
    slot: int,
    *,
    radius: int,
    max_ratio: float,
) -> np.ndarray:
    """Sweep only deviations from a locally fitted linear centroid path."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    indices = _frame_neighbours(run, frame_index, int(radius))
    if len(indices) < 3:
        return reference
    frames = np.asarray(_run_frame_numbers(run)[indices], dtype=np.float64)
    centres = np.asarray(
        [_centre(run.anchors[int(index)][slot]) for index in indices],
        dtype=np.float64,
    )
    centered_times = frames - float(run.frame_numbers[int(frame_index)])
    design = np.stack((centered_times, np.ones(len(indices))), axis=1)
    trend = design @ np.linalg.lstsq(design, centres, rcond=None)[0]
    residuals = centres - trend
    current_pos = int(np.argmin(np.abs(indices - int(frame_index))))
    shifts = residuals - residuals[current_pos]
    # The residual tube should model curvature, not duplicate C02's velocity
    # sweep.  Bound extreme centroid outliers before the area-aware raster cap.
    radial_cap = 0.35 * math.sqrt(
        max(_polygon_area(reference), 1.0) / math.pi
    )
    lengths = np.linalg.norm(shifts, axis=1)
    scale = np.minimum(1.0, radial_cap / np.maximum(lengths, 1e-9))
    shifts = shifts * scale[:, None]
    swept = np.stack(
        [reference + shift[None, :] for shift in shifts], axis=0
    )
    masks, origin = _raster_stack(swept, padding=5)
    raw = masks[current_pos]
    candidate_mask = _dilate(np.any(masks, axis=0), 1)
    candidate_mask = _cap_union_area(
        candidate_mask, raw, max_ratio=float(max_ratio)
    )
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _ctr4_125(run, frame_index: int, slot: int) -> np.ndarray:
    return _trajectory_residual_sweep(
        run, frame_index, slot, radius=4, max_ratio=1.25
    )


def _ctr4_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _trajectory_residual_sweep(
        run, frame_index, slot, radius=4, max_ratio=1.50
    )


def _ctr8_150(run, frame_index: int, slot: int) -> np.ndarray:
    return _trajectory_residual_sweep(
        run, frame_index, slot, radius=8, max_ratio=1.50
    )


def _vertex_linefit_endpoint(
    run,
    frame_index: int,
    slot: int,
    *,
    forward: bool,
    horizon: int,
    robust: bool = False,
) -> np.ndarray:
    """Least-squares endpoint for a one-sided aligned vertex trajectory."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    direction = 1 if bool(forward) else -1
    indices = [
        frame_index + direction * offset
        for offset in range(0, int(horizon) + 1)
        if 0 <= frame_index + direction * offset < len(run.frame_numbers)
    ]
    if len(indices) < 3:
        return reference
    current_frame = float(run.frame_numbers[int(frame_index)])
    times = np.asarray(
        [float(run.frame_numbers[int(index)]) - current_frame for index in indices],
        dtype=np.float64,
    )
    aligned = np.stack(
        [
            _align_order(
                reference,
                np.asarray(run.anchors[int(index)][slot], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    design = np.stack((times, np.ones(len(times))), axis=1)
    targets = aligned.reshape(len(times), -1)
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    if bool(robust):
        for _iteration in range(2):
            residual = targets - design @ coefficients
            frame_error = np.sqrt(np.mean(np.square(residual), axis=1))
            median = float(np.median(frame_error))
            scale = max(
                1.4826 * float(np.median(np.abs(frame_error - median))), 1e-6
            )
            threshold = median + 1.5 * scale
            weights = np.minimum(1.0, threshold / np.maximum(frame_error, 1e-9))
            weighted = np.sqrt(weights)[:, None]
            coefficients = np.linalg.lstsq(
                design * weighted, targets * weighted, rcond=None
            )[0]
    return coefficients[1].reshape(reference.shape)


def _vf6(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=True, horizon=6
    )


def _vb6(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=False, horizon=6
    )


def _vf8(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=True, horizon=8
    )


def _vb8(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=False, horizon=8
    )


def _vf10(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=True, horizon=10
    )


def _vb10(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=False, horizon=10
    )


def _vf12(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=True, horizon=12
    )


def _vb12(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=False, horizon=12
    )


def _vfr8(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=True, horizon=8, robust=True
    )


def _vbr8(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(
        run, frame_index, slot, forward=False, horizon=8, robust=True
    )


def _linear_interval_endpoint(
    run,
    frame_index: int,
    slot: int,
    *,
    forward: bool,
    horizon: int,
    max_ratio: float,
) -> np.ndarray:
    """Endpoint shape fitted directly to a fixed-length linear interpolation.

    The opposite endpoint is fixed to the observed shape at ``horizon``.  For
    every vertex, solve the least-squares endpoint which makes linear polygon
    interpolation reproduce all intermediate observations.  The observations
    are similarity-aligned to the current pose first, so the candidate models
    only the residual non-rigid shape change; ordinary translation/rotation/
    scale remains the responsibility of interpolation between keyframes.

    A fitted endpoint is never emitted directly.  Only its locally supported
    additions to the raw endpoint are retained and those additions are capped
    to ``max_ratio`` of raw area.  This makes excessive whole-mask expansion
    impossible by construction while preserving raw endpoint Recall.
    """
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    direction = 1 if bool(forward) else -1
    indices = [frame_index + direction * offset for offset in range(horizon + 1)]
    if any(index < 0 or index >= len(run.frame_numbers) for index in indices):
        return reference
    # Do not bridge a temporal discontinuity hidden inside a sparse track.
    frame_numbers = np.asarray(_run_frame_numbers(run)[indices], dtype=np.int64)
    expected = direction * np.arange(horizon + 1, dtype=np.int64)
    if not np.array_equal(frame_numbers - frame_numbers[0], expected):
        return reference

    aligned = np.stack(
        [
            _similarity_align(
                reference,
                np.asarray(run.anchors[int(index)][slot], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    opposite = aligned[-1]
    fractions = np.arange(horizon + 1, dtype=np.float64) / float(horizon)
    own_weights = 1.0 - fractions
    opposite_weights = fractions
    # Downweight the far middle slightly: it is the least identifiable part
    # of the endpoint inverse problem and most vulnerable to a single raw-mask
    # outlier.  Both actual endpoints retain unit confidence.
    temporal_weights = 0.75 + 0.25 * np.abs(2.0 * fractions - 1.0)
    numerator = np.sum(
        (
            temporal_weights * own_weights
        )[:, None, None]
        * (aligned - opposite_weights[:, None, None] * opposite[None]),
        axis=0,
    )
    denominator = float(np.sum(temporal_weights * own_weights * own_weights))
    fitted = numerator / max(denominator, 1e-9)

    # Bound pathological vertex excursions before rasterization.  The cap is
    # relative to equivalent raw radius, hence resolution-independent.
    displacement = fitted - reference
    radius = math.sqrt(max(_polygon_area(reference), 1.0) / math.pi)
    displacement_cap = 0.35 * radius
    lengths = np.linalg.norm(displacement, axis=1)
    displacement *= np.minimum(
        1.0, displacement_cap / np.maximum(lengths, 1e-9)
    )[:, None]
    fitted = reference + displacement

    masks, origin = _raster_stack(np.stack((fitted, reference)), padding=5)
    candidate_mask = np.logical_or(masks[0], masks[1])
    candidate_mask = _cap_union_area(
        candidate_mask, masks[1], max_ratio=float(max_ratio)
    )
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _lsf8_110(run, frame_index: int, slot: int) -> np.ndarray:
    return _linear_interval_endpoint(
        run, frame_index, slot, forward=True, horizon=8, max_ratio=1.10
    )


def _lsb8_110(run, frame_index: int, slot: int) -> np.ndarray:
    return _linear_interval_endpoint(
        run, frame_index, slot, forward=False, horizon=8, max_ratio=1.10
    )


def _lsf8_115(run, frame_index: int, slot: int) -> np.ndarray:
    return _linear_interval_endpoint(
        run, frame_index, slot, forward=True, horizon=8, max_ratio=1.15
    )


def _lsb8_115(run, frame_index: int, slot: int) -> np.ndarray:
    return _linear_interval_endpoint(
        run, frame_index, slot, forward=False, horizon=8, max_ratio=1.15
    )


def _cap_supported_union_area(
    candidate_masks: np.ndarray,
    raw: np.ndarray,
    *,
    max_ratio: float,
) -> np.ndarray:
    """Select recurrent local additions before distant one-off additions."""
    masks = np.asarray(candidate_masks, dtype=bool)
    raw_mask = np.asarray(raw, dtype=bool)
    union = np.any(masks, axis=0)
    raw_area = int(np.sum(raw_mask))
    limit = max(raw_area, int(math.floor(float(max_ratio) * raw_area)))
    if int(np.sum(np.logical_or(union, raw_mask))) <= limit:
        return np.logical_or(union, raw_mask).astype(np.uint8)
    output = raw_mask.copy()
    additions = np.logical_and(union, ~raw_mask)
    allowance = max(0, limit - raw_area)
    ys, xs = np.nonzero(additions)
    if allowance <= 0 or not len(xs):
        return output.astype(np.uint8)
    support = np.sum(masks, axis=0, dtype=np.int16)
    distance = cv2.distanceTransform((~raw_mask).astype(np.uint8), cv2.DIST_L2, 3)
    # Primary key: recurring support (descending); secondary: distance from
    # raw (ascending).  Coordinate ties keep the result deterministic.
    order = np.lexsort((xs, ys, distance[ys, xs], -support[ys, xs]))
    chosen = order[:allowance]
    output[ys[chosen], xs[chosen]] = True
    return output.astype(np.uint8)


def _inverse_interval_envelope(
    run,
    frame_index: int,
    slot: int,
    *,
    forward: bool,
    horizon: int,
    max_ratio: float,
) -> np.ndarray:
    """Invert midpoint observations into the required linear endpoint shape.

    For a linear polygon interpolation ``R_k ~= a P + b Q`` with the opposite
    endpoint ``Q`` fixed, every midpoint implies ``P_k=(R_k-bQ)/a``.  The
    recurrent union of these inverse projections is therefore a direct shape
    proposal for making an interval-8 edge Recall-feasible.  Only the half of
    the interval controlled strongly by this endpoint is used; the paired
    forward/backward state handles the other half without the unstable
    ``1/a`` amplification near the opposite endpoint.
    """
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    direction = 1 if bool(forward) else -1
    indices = [frame_index + direction * offset for offset in range(horizon + 1)]
    if any(index < 0 or index >= len(run.frame_numbers) for index in indices):
        return reference
    frame_numbers = np.asarray(_run_frame_numbers(run)[indices], dtype=np.int64)
    expected = direction * np.arange(horizon + 1, dtype=np.int64)
    if not np.array_equal(frame_numbers - frame_numbers[0], expected):
        return reference
    aligned = np.stack(
        [
            _similarity_align(
                reference,
                np.asarray(run.anchors[int(index)][slot], dtype=np.float64),
            )
            for index in indices
        ],
        axis=0,
    )
    opposite = aligned[-1]
    inverse_shapes = [reference]
    for offset in range(1, horizon // 2 + 1):
        fraction = float(offset) / float(horizon)
        own_weight = 1.0 - fraction
        inferred = (aligned[offset] - fraction * opposite) / own_weight
        displacement = inferred - reference
        radius = math.sqrt(max(_polygon_area(reference), 1.0) / math.pi)
        cap = 0.30 * radius
        lengths = np.linalg.norm(displacement, axis=1)
        displacement *= np.minimum(1.0, cap / np.maximum(lengths, 1e-9))[:, None]
        inverse_shapes.append(reference + displacement)
    masks, origin = _raster_stack(np.asarray(inverse_shapes), padding=5)
    candidate_mask = _cap_supported_union_area(
        masks, masks[0], max_ratio=float(max_ratio)
    )
    polygon = _mask_to_polygon(candidate_mask, origin, reference)
    return reference if polygon is None else polygon


def _ivf8_115(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=True, horizon=8, max_ratio=1.15
    )


def _ivf8_110(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=True, horizon=8, max_ratio=1.10
    )


def _ivb8_110(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=False, horizon=8, max_ratio=1.10
    )


def _ivb8_115(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=False, horizon=8, max_ratio=1.15
    )


def _ivf8_120(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=True, horizon=8, max_ratio=1.20
    )


def _ivb8_120(run, frame_index: int, slot: int) -> np.ndarray:
    return _inverse_interval_envelope(
        run, frame_index, slot, forward=False, horizon=8, max_ratio=1.20
    )


def _local_area_zscores(run, slot: int, radius: int = 5) -> np.ndarray:
    cache = getattr(run, "_orthogonal_local_area_zscores", None)
    if cache is None:
        cache = {}
        setattr(run, "_orthogonal_local_area_zscores", cache)
    key = (int(slot), int(radius))
    if key in cache:
        return cache[key]
    areas = np.asarray(
        [_polygon_area(frame[int(slot)]) for frame in run.anchors], dtype=np.float64
    )
    values = np.log(np.maximum(areas, 1.0))
    output = np.zeros_like(values)
    for index in range(len(values)):
        lo, hi = max(0, index - radius), min(len(values), index + radius + 1)
        window = values[lo:hi]
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        output[index] = (values[index] - median) / max(1.4826 * mad, 0.08)
    cache[key] = output
    return output


def _e02(run, frame_index: int, slot: int) -> np.ndarray:
    """Forward finite-horizon peak hold with 1.5 px/frame release."""
    reference = np.asarray(run.anchors[frame_index][slot], dtype=np.float64)
    zscores = _local_area_zscores(run, slot)
    past = [index for index in range(max(0, frame_index - 6), frame_index + 1)]
    aligned = []
    erosion = []
    for index in past:
        if index != frame_index and float(zscores[index]) > 2.5:
            continue
        aligned.append(
            _translation_align(
                reference,
                np.asarray(run.anchors[index][slot], dtype=np.float64),
            )
        )
        erosion.append(int(round(1.5 * (frame_index - index))))
    if not aligned:
        return reference
    masks, origin = _raster_stack(np.asarray(aligned), padding=max(7, max(erosion) + 2))
    held = np.zeros_like(masks[0])
    for mask, radius in zip(masks, erosion, strict=True):
        value = mask
        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            value = cv2.erode(value, kernel)
        held = np.logical_or(held, value)
    # Under the current hard raw-Recall experiment E02 cannot prune a flagged
    # current expansion.  It therefore remains a one-sided shrink/occlusion
    # repair state; weighted Recall is a separate future experiment.
    current_pos = len(masks) - 1
    held = np.logical_or(held, masks[current_pos])
    held = _dilate(held, 1)
    held = _cap_union_area(held, masks[current_pos])
    polygon = _mask_to_polygon(held, origin, reference)
    return reference if polygon is None else polygon


_GENERATORS: dict[str, Callable] = {
    "A2": _a2,
    "A4": _a4,
    "D6": _d6,
    "D6_R5": _d6_r5,
    "B3": _b3,
    "C1": _c1,
    "C6": _c6,
    "E2": _e2,
    "F3": _f3,
    "F3_Q65": _f3_q65,
    "F3_Q75": _f3_q75,
    "G3": _g3,
    "Z1": _z1,
    "A07": _a07,
    "A06": _a06,
    "A06_K3": _a06_k3,
    "A06_K4": _a06_k4,
    "G02": _g02,
    "G04": _g04,
    "G02_H3": _g02_h3,
    "G04_H3": _g04_h3,
    "G02_H8": _g02_h8,
    "G04_H8": _g04_h8,
    "GF8_K2_135": _gf8_k2_135,
    "GB8_K2_135": _gb8_k2_135,
    "GF8_K2_150": _gf8_k2_150,
    "GB8_K2_150": _gb8_k2_150,
    "GF8_K3_150": _gf8_k3_150,
    "GB8_K3_150": _gb8_k3_150,
    "GF12_K2_150": _gf12_k2_150,
    "GB12_K2_150": _gb12_k2_150,
    "GF8_K2_200": _gf8_k2_200,
    "GB8_K2_200": _gb8_k2_200,
    "GFT8_K2_200": _gft8_k2_200,
    "GBT8_K2_200": _gbt8_k2_200,
    "GF8_K1_200": _gf8_k1_200,
    "GB8_K1_200": _gb8_k1_200,
    "GF8_K1_300": _gf8_k1_300,
    "GB8_K1_300": _gb8_k1_300,
    "C02": _c02,
    "C02_115": _c02_115,
    "C02_125": _c02_125,
    "C02_120": _c02_120,
    "C02_130": _c02_130,
    "C02_150": _c02_150,
    "C02_175": _c02_175,
    "CTR4_125": _ctr4_125,
    "CTR4_150": _ctr4_150,
    "CTR8_150": _ctr8_150,
    "VF6": _vf6,
    "VB6": _vb6,
    "VF8": _vf8,
    "VB8": _vb8,
    "VF10": _vf10,
    "VB10": _vb10,
    "VF12": _vf12,
    "VB12": _vb12,
    "VFR8": _vfr8,
    "VBR8": _vbr8,
    "LSF8_110": _lsf8_110,
    "LSB8_110": _lsb8_110,
    "LSF8_115": _lsf8_115,
    "LSB8_115": _lsb8_115,
    "IVF8_115": _ivf8_115,
    "IVB8_115": _ivb8_115,
    "IVF8_110": _ivf8_110,
    "IVB8_110": _ivb8_110,
    "IVF8_120": _ivf8_120,
    "IVB8_120": _ivb8_120,
    "E02": _e02,
    "S30_R2_A105": _s30_r2_a105,
    "S30_R2_A125": _s30_r2_a125,
    "S30_R5_A105": _s30_r5_a105,
    "S30_R5_A125": _s30_r5_a125,
    "S30_R10_A105": _s30_r10_a105,
    "S30_R10_A125": _s30_r10_a125,
}


def build_role_candidate(run, frame_index: int, role_id: str) -> np.ndarray:
    """Build every component for one named candidate role."""
    union_raw = str(role_id).endswith("_P1")
    base_role_id = str(role_id)[:-3] if union_raw else str(role_id)
    if base_role_id not in _GENERATORS:
        raise KeyError(f"unknown role candidate: {role_id}")
    generator = _GENERATORS[base_role_id]
    slots = []
    for slot in range(int(run.contour_count)):
        reference = np.asarray(
            run.anchors[int(frame_index)][slot], dtype=np.float64
        )
        candidate = np.asarray(
            generator(run, int(frame_index), slot), dtype=np.float64
        )
        if union_raw:
            masks, origin = _raster_stack(
                np.stack((candidate, reference), axis=0), padding=5
            )
            candidate_mask = _dilate(np.logical_or(masks[0], masks[1]), 1)
            candidate_mask = _cap_union_area(candidate_mask, masks[1])
            converted = _mask_to_polygon(candidate_mask, origin, reference)
            candidate = reference if converted is None else converted
        slots.append(np.asarray(candidate, dtype=np.float32))
    return np.asarray(slots, dtype=np.float32)


__all__ = ["ROLE_IDS", "build_role_candidate"]
