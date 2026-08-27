#!/usr/bin/env python3
"""Deterministic temporal-shape candidates used by Production DP.

Only roles referenced by the frozen polygon and Catmull--Rom palettes live
here. Historical search candidates belong in experiments and must not grow
the deployed runtime surface.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable

import cv2
import numpy as np

from production.polygon.runtime.geometry import (
    align_order as _align_order,
    resample_closed as _numpy_resample,
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
        keep = np.abs(
            frames[np.asarray(cached_indices, dtype=np.int32)] - current
        ) <= int(radius)
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


def _shared_raster_window(run, frame_index: int, slot: int, radius: int, mode: str):
    """Cache alignment and rasterization independently from optional TSDF."""

    cache = _frame_cache(run, frame_index)
    key = ("raster_window", int(slot), int(radius), str(mode))
    if key not in cache:
        indices, aligned = _aligned_window(run, frame_index, slot, radius, mode)
        masks, origin = _raster_stack(aligned)
        cache[key] = (indices, aligned, masks, origin)
    return cache[key]


def _raster_stack(polygons: np.ndarray, padding: int = 7):
    all_points = np.concatenate(
        [np.asarray(value, dtype=np.float64) for value in polygons]
    )
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


def _f3_with_quantile(run, frame_index: int, slot: int, quantile: float) -> np.ndarray:
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
    outward = (np.roll(outward, 1) + 2.0 * outward + np.roll(outward, -1)) / 4.0
    cap = max(2.0, 0.12 * math.sqrt(max(_polygon_area(reference), 1.0) / math.pi))
    return reference + np.minimum(outward, cap)[:, None] * normal


def _f3(run, frame_index: int, slot: int) -> np.ndarray:
    return _f3_with_quantile(run, frame_index, slot, 0.50)


def _f3_q75(run, frame_index: int, slot: int) -> np.ndarray:
    return _f3_with_quantile(run, frame_index, slot, 0.75)


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


def _a06(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(run, frame_index, slot, kth=2, union_raw=True)


def _a06_k3(run, frame_index: int, slot: int) -> np.ndarray:
    return _orthogonal_order_stat(run, frame_index, slot, kth=3, union_raw=True)


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
    return _directional_endpoint_envelope(run, frame_index, slot, forward=True)


def _g04(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(run, frame_index, slot, forward=False)


def _g02_h3(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=True, horizon=3
    )


def _g04_h3(run, frame_index: int, slot: int) -> np.ndarray:
    return _directional_endpoint_envelope(
        run, frame_index, slot, forward=False, horizon=3
    )


def _c02_with_cap(run, frame_index: int, slot: int, *, max_ratio: float) -> np.ndarray:
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


def _c02_125(run, frame_index: int, slot: int) -> np.ndarray:
    return _c02_with_cap(run, frame_index, slot, max_ratio=1.25)


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
            scale = max(1.4826 * float(np.median(np.abs(frame_error - median))), 1e-6)
            threshold = median + 1.5 * scale
            weights = np.minimum(1.0, threshold / np.maximum(frame_error, 1e-9))
            weighted = np.sqrt(weights)[:, None]
            coefficients = np.linalg.lstsq(
                design * weighted, targets * weighted, rcond=None
            )[0]
    return coefficients[1].reshape(reference.shape)


def _vf8(run, frame_index: int, slot: int) -> np.ndarray:
    return _vertex_linefit_endpoint(run, frame_index, slot, forward=True, horizon=8)


_GENERATORS: dict[str, Callable] = {
    "C02_125": _c02_125,
    "G02": _g02,
    "G04": _g04,
    "A06": _a06,
    "F3": _f3,
    "D6": _d6,
    "G02_H3": _g02_h3,
    "G04_H3": _g04_h3,
    "A06_K3": _a06_k3,
    "D6_R5": _d6_r5,
    "F3_Q75": _f3_q75,
    "VF8": _vf8,
}

PRODUCTION_ROLE_IDS = tuple(_GENERATORS)


def build_role_candidate(run, frame_index: int, role_id: str) -> np.ndarray:
    """Build every component for one named candidate role."""
    union_raw = str(role_id).endswith("_P1")
    base_role_id = str(role_id)[:-3] if union_raw else str(role_id)
    if base_role_id not in _GENERATORS:
        raise KeyError(f"unknown role candidate: {role_id}")
    generator = _GENERATORS[base_role_id]
    slots = []
    for slot in range(int(run.contour_count)):
        reference = np.asarray(run.anchors[int(frame_index)][slot], dtype=np.float64)
        candidate = np.asarray(generator(run, int(frame_index), slot), dtype=np.float64)
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


__all__ = ["PRODUCTION_ROLE_IDS", "build_role_candidate"]
