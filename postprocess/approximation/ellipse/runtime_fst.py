from __future__ import annotations

import sys
import types
from pathlib import Path

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parent


def _register_inline_module(
    module_name: str, export_map: dict[str, str]
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(SELF_PATH)
    for public_name, global_name in export_map.items():
        value = globals()[global_name]
        setattr(module, public_name, value)
        setattr(module, global_name, value)
    sys.modules[module_name] = module
    return module


import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import multiprocessing
import os
import pickle
import sqlite3
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

fst_WIDTH = 1920
fst_HEIGHT = 1080
fst_POLYGON_POINTS = 96
fst_MASK_COLOR = np.array([0, 0, 255], dtype=np.float32)
fst_ELLIPSE_COLOR = (0, 0, 0)
fst__ANGLE_TABLE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
fst__KERNEL_CACHE: dict[int, np.ndarray] = {}
fst__GRID_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
fst__ROW_POLYGONS_JSONS: list[str] = []
fst__ROW_LOCAL_RASTER_PAYLOADS: list[
    tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]
] = []
fst__ROW_GT_POLYGONS: list[list[np.ndarray]] = []


def fst__get_unit_circle(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    cached = fst__ANGLE_TABLE.get(n_points)
    if cached is None:
        angles = np.linspace(
            0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float32
        )
        cached = (np.cos(angles), np.sin(angles))
        fst__ANGLE_TABLE[n_points] = cached
    return cached


def fst_ellipse_to_polygon_array(
    cx: float,
    cy: float,
    a: float,
    b: float,
    theta_deg: float,
    n_points: int = fst_POLYGON_POINTS,
) -> np.ndarray:
    unit_cos, unit_sin = fst__get_unit_circle(n_points)
    cos_t = math.cos(math.radians(theta_deg))
    sin_t = math.sin(math.radians(theta_deg))
    xs = a * unit_cos
    ys = b * unit_sin
    pts = np.empty((n_points, 2), dtype=np.float32)
    pts[:, 0] = cx + xs * cos_t - ys * sin_t
    pts[:, 1] = cy + xs * sin_t + ys * cos_t
    return pts


def fst_ellipse_to_polygon(
    cx: float,
    cy: float,
    a: float,
    b: float,
    theta_deg: float,
    n_points: int = fst_POLYGON_POINTS,
) -> list[list[float]]:
    return (
        fst_ellipse_to_polygon_array(cx, cy, a, b, theta_deg, n_points=n_points)
        .astype(np.float64)
        .tolist()
    )


def fst_parse_polygons(polygons_json: str) -> list[np.ndarray]:
    polygons = json.loads(polygons_json)
    return [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in polygons]


def fst_make_polygons_json(
    ellipses: list[tuple[float, float, float, float, float]]
) -> str:
    polygons = [
        fst_ellipse_to_polygon(cx, cy, a, b, angle, n_points=fst_POLYGON_POINTS)
        for cx, cy, a, b, angle in ellipses
    ]
    return json.dumps(polygons)


def fst_ellipses_to_polygon_arrays(
    ellipses: list[tuple[float, float, float, float, float]]
) -> list[np.ndarray]:
    return [
        fst_ellipse_to_polygon_array(
            cx, cy, a, b, angle, n_points=fst_POLYGON_POINTS
        ).astype(np.float32)
        for cx, cy, a, b, angle in ellipses
    ]


def fst_rasterize_full(
    polygons_json: str, height: int = fst_HEIGHT, width: int = fst_WIDTH
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in fst_parse_polygons(polygons_json):
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def fst_rasterize_full_from_polygons(
    polygons: list[np.ndarray], height: int = fst_HEIGHT, width: int = fst_WIDTH
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def fst_prepare_local_raster_payload_from_polygons(
    pts: list[np.ndarray],
) -> tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]:
    all_pts = np.concatenate(pts, axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    bbox_w = float(max_xy[0] - min_xy[0] + 1.0)
    bbox_h = float(max_xy[1] - min_xy[1] + 1.0)
    pad = max(24, int(max(bbox_w, bbox_h) * 0.45))
    x0 = max(0, int(math.floor(min_xy[0])) - pad)
    y0 = max(0, int(math.floor(min_xy[1])) - pad)
    x1 = min(fst_WIDTH - 1, int(math.ceil(max_xy[0])) + pad)
    y1 = min(fst_HEIGHT - 1, int(math.ceil(max_xy[1])) + pad)
    shifted = [
        np.round(poly - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        for poly in pts
    ]
    return ((y1 - y0 + 1, x1 - x0 + 1), (x0, y0), shifted)


def fst_prepare_local_raster_payload(
    polygons_json: str,
) -> tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]:
    return fst_prepare_local_raster_payload_from_polygons(
        fst_parse_polygons(polygons_json)
    )


def fst_rasterize_local_mask_from_payload(
    payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]
) -> tuple[np.ndarray, tuple[int, int]]:
    shape, origin, shifted = payload
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, shifted, 1)
    return (mask, origin)


def fst_rasterize_polygons_to_local_mask(
    polygons_json: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    return fst_rasterize_local_mask_from_payload(
        fst_prepare_local_raster_payload(polygons_json)
    )


def fst_set_row_local_raster_cache(
    polygons_jsons: list[str],
    payloads: list[tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]],
    gt_polygons: list[list[np.ndarray]],
) -> None:
    global fst__ROW_POLYGONS_JSONS, fst__ROW_LOCAL_RASTER_PAYLOADS, fst__ROW_GT_POLYGONS
    fst__ROW_POLYGONS_JSONS = polygons_jsons
    fst__ROW_LOCAL_RASTER_PAYLOADS = payloads
    fst__ROW_GT_POLYGONS = gt_polygons


def fst_normalize_ellipse(
    ellipse: tuple[float, float, float, float, float]
) -> tuple[float, float, float, float, float]:
    cx, cy, a, b, angle = ellipse
    if a < b:
        a, b = (b, a)
        angle += 90.0
    angle %= 180.0
    return (float(cx), float(cy), float(max(a, 1.0)), float(max(b, 1.0)), float(angle))


def fst_fit_ellipse_from_points(
    points_xy: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    if len(points_xy) < 5:
        return None
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    return fst_normalize_ellipse(
        (
            float(cx),
            float(cy),
            max(float(w) / 2.0, 1.0),
            max(float(h) / 2.0, 1.0),
            float(angle),
        )
    )


def fst_fit_ellipse_from_mask(
    mask: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = np.concatenate(contours, axis=0).reshape(-1, 2)
    if len(points) < 5:
        ys, xs = np.where(mask > 0)
        points = np.column_stack([xs, ys]).astype(np.float32)
    return fst_fit_ellipse_from_points(points)


def fst_render_ellipses(
    shape: tuple[int, int],
    ellipses: list[tuple[float, float, float, float, float]],
    scales: list[float],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for (cx, cy, a, b, angle), scale in zip(ellipses, scales):
        poly = np.round(
            fst_ellipse_to_polygon_array(cx, cy, a * scale, b * scale, angle)
        ).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
    return mask


def fst_compute_mask_metrics(
    gt_mask: np.ndarray, sub_mask: np.ndarray, gt_area: int | None = None
) -> tuple[float, float]:
    if gt_area is None:
        gt_area = int(np.count_nonzero(gt_mask))
    intersection = int(np.count_nonzero(gt_mask & sub_mask))
    union = int(np.count_nonzero(gt_mask | sub_mask))
    recall = intersection / gt_area if gt_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return (iou, recall)


def fst_compute_exact_metrics(
    gt_polygons_json: str, pred_polygons_json: str
) -> dict[str, float]:
    gt_polys = fst_parse_polygons(gt_polygons_json)
    pred_polys = fst_parse_polygons(pred_polygons_json)
    return fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys)


def fst_compute_exact_metrics_from_gt_polys(
    gt_polys: list[np.ndarray], pred_polygons_json: str
) -> dict[str, float]:
    return fst_compute_exact_metrics_from_polygons(
        gt_polys, fst_parse_polygons(pred_polygons_json)
    )


def fst_compute_exact_metrics_from_polygons(
    gt_polys: list[np.ndarray], pred_polys: list[np.ndarray]
) -> dict[str, float]:
    rounded_polys = [np.round(poly).astype(np.int32) for poly in gt_polys + pred_polys]
    all_pts = np.concatenate(rounded_polys, axis=0)
    points_in_bounds = (
        int(all_pts[:, 0].min()) >= 0
        and int(all_pts[:, 1].min()) >= 0
        and (int(all_pts[:, 0].max()) < fst_WIDTH)
        and (int(all_pts[:, 1].max()) < fst_HEIGHT)
    )
    if points_in_bounds:
        x0 = int(all_pts[:, 0].min())
        y0 = int(all_pts[:, 1].min())
        x1 = int(all_pts[:, 0].max())
        y1 = int(all_pts[:, 1].max())
        shift = np.array([x0, y0], dtype=np.int32)
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        gt_mask = np.zeros(shape, dtype=np.uint8)
        pred_mask = np.zeros(shape, dtype=np.uint8)
        gt_rounded = rounded_polys[: len(gt_polys)]
        pred_rounded = rounded_polys[len(gt_polys) :]
        for poly in gt_rounded:
            cv2.fillPoly(gt_mask, [poly - shift], 1)
        for poly in pred_rounded:
            cv2.fillPoly(pred_mask, [poly - shift], 1)
    else:
        gt_mask = fst_rasterize_full_from_polygons(gt_polys)
        pred_mask = fst_rasterize_full_from_polygons(pred_polys)
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = int((gt_mask | pred_mask).sum())
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def fst_compute_weighted_error(metrics: dict[str, float]) -> int:
    fn_pixels = int(metrics["gt_area"] - metrics["intersection"])
    fp_pixels = int(metrics["pred_area"] - metrics["intersection"])
    return int(2 * fn_pixels + fp_pixels)


def fst_candidate_score(iou: float, recall: float, recall_target: float) -> float:
    return iou - 4.0 * max(0.0, recall_target - recall)


def fst_binary_search_scale(
    gt_mask: np.ndarray,
    ellipses: list[tuple[float, float, float, float, float]],
    fixed_scales: list[float] | None,
    scale_index: int | None,
    low: float,
    high: float,
    recall_target: float,
    iterations: int,
    gt_area: int | None = None,
) -> float:
    if gt_area is None:
        gt_area = int(np.count_nonzero(gt_mask))
    lo = low
    hi = high
    best = hi
    base_scales = [1.0] * len(ellipses) if fixed_scales is None else list(fixed_scales)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        scales = list(base_scales)
        if scale_index is None:
            scales = [mid] * len(ellipses)
        else:
            scales[scale_index] = mid
        sub_mask = fst_render_ellipses(gt_mask.shape, ellipses, scales)
        intersection = int(np.count_nonzero(gt_mask & sub_mask))
        recall = intersection / gt_area if gt_area > 0 else 1.0
        if recall >= recall_target:
            best = mid
            hi = mid
        else:
            lo = mid
    return best


def fst_optimize_candidate_scales(
    gt_mask: np.ndarray,
    ellipses: list[tuple[float, float, float, float, float]],
    recall_target: float,
    min_scale: float = 0.35,
    max_scale: float = 3.0,
) -> tuple[list[float], float, float]:
    gt_area = int(np.count_nonzero(gt_mask))
    current_high = max_scale
    for _ in range(6):
        sub_mask = fst_render_ellipses(
            gt_mask.shape, ellipses, [current_high] * len(ellipses)
        )
        intersection = int(np.count_nonzero(gt_mask & sub_mask))
        recall = intersection / gt_area if gt_area > 0 else 1.0
        if recall >= recall_target:
            break
        current_high *= 1.4
    shared_scale = fst_binary_search_scale(
        gt_mask,
        ellipses,
        fixed_scales=None,
        scale_index=None,
        low=min_scale,
        high=current_high,
        recall_target=recall_target,
        iterations=12,
        gt_area=gt_area,
    )
    scales = [shared_scale] * len(ellipses)
    for _ in range(2):
        changed = False
        for idx in range(len(scales)):
            improved = fst_binary_search_scale(
                gt_mask,
                ellipses,
                fixed_scales=scales,
                scale_index=idx,
                low=min_scale,
                high=scales[idx],
                recall_target=recall_target,
                iterations=10,
                gt_area=gt_area,
            )
            if improved < scales[idx] - 0.001:
                scales[idx] = improved
                changed = True
        if not changed:
            break
    final_mask = fst_render_ellipses(gt_mask.shape, ellipses, scales)
    iou, recall = fst_compute_mask_metrics(gt_mask, final_mask, gt_area=gt_area)
    return (scales, iou, recall)


def fst_apply_scales_to_ellipses(
    ellipses: list[tuple[float, float, float, float, float]], scales: list[float]
) -> list[tuple[float, float, float, float, float]]:
    return [
        fst_normalize_ellipse((cx, cy, a * scale, b * scale, angle))
        for (cx, cy, a, b, angle), scale in zip(ellipses, scales)
    ]


def fst_refine_ellipses_locally(
    gt_mask: np.ndarray,
    ellipses: list[tuple[float, float, float, float, float]],
    recall_target: float,
    max_rounds: int = 6,
) -> tuple[list[tuple[float, float, float, float, float]], float, float]:
    current = [list(fst_normalize_ellipse(ellipse)) for ellipse in ellipses]
    gt_area = int(np.count_nonzero(gt_mask))

    def evaluate(candidate: list[list[float]]) -> tuple[float, float, float]:
        norm = [fst_normalize_ellipse(tuple(ellipse)) for ellipse in candidate]
        sub_mask = fst_render_ellipses(gt_mask.shape, norm, [1.0] * len(norm))
        iou, recall = fst_compute_mask_metrics(gt_mask, sub_mask, gt_area=gt_area)
        return (fst_candidate_score(iou, recall, recall_target), iou, recall)

    best_score, best_iou, best_recall = evaluate(current)
    height, width = gt_mask.shape
    step_pos = max(2.0, 0.04 * max(width, height))
    step_rad = max(2.0, 0.05 * max(width, height))
    step_angle = 12.0
    for _ in range(max_rounds):
        improved = False
        for ellipse_idx in range(len(current)):
            for param_idx in range(5):
                base_value = current[ellipse_idx][param_idx]
                deltas = (
                    (-step_pos, step_pos)
                    if param_idx in (0, 1)
                    else (-step_rad, step_rad)
                    if param_idx in (2, 3)
                    else (-step_angle, step_angle)
                )
                local_best = None
                for delta in deltas:
                    trial = [ellipse[:] for ellipse in current]
                    trial[ellipse_idx][param_idx] = base_value + delta
                    if param_idx in (2, 3) and trial[ellipse_idx][param_idx] < 1.0:
                        continue
                    score, iou, recall = evaluate(trial)
                    if local_best is None or score > local_best[0]:
                        local_best = (score, iou, recall, trial)
                if local_best is not None and local_best[0] > best_score + 1e-06:
                    best_score, best_iou, best_recall, current = local_best
                    improved = True
        if not improved:
            step_pos *= 0.5
            step_rad *= 0.5
            step_angle *= 0.5
            if step_pos < 0.5 and step_rad < 0.5 and (step_angle < 1.0):
                break
    refined = [fst_normalize_ellipse(tuple(ellipse)) for ellipse in current]
    return (refined, best_iou, best_recall)


def fst_build_component_mask(
    shape: tuple[int, int], points_xy: np.ndarray, kernel_size: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts_int = points_xy.astype(np.int32)
    mask[pts_int[:, 1], pts_int[:, 0]] = 1
    kernel = fst__KERNEL_CACHE.get(kernel_size)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        fst__KERNEL_CACHE[kernel_size] = kernel
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def fst__fit_axis_split_candidates(
    gt_mask: np.ndarray, points: np.ndarray, projection: np.ndarray, quantile: float
) -> list[tuple[float, float, float, float, float]] | None:
    threshold = float(np.quantile(projection, quantile))
    ellipses: list[tuple[float, float, float, float, float]] = []
    for component in (projection <= threshold, projection > threshold):
        frac = float(component.mean())
        if frac < 0.12 or frac > 0.88:
            return None
        component_points = points[component]
        component_mask = fst_build_component_mask(
            gt_mask.shape, component_points, kernel_size=5
        )
        ellipse = fst_fit_ellipse_from_mask(component_mask)
        if ellipse is None:
            return None
        ellipses.append(ellipse)
    return ellipses


def fst_generate_principal_axis_candidates(
    gt_mask: np.ndarray,
) -> dict[tuple[str, float], list[tuple[float, float, float, float, float]]]:
    ys, xs = np.where(gt_mask > 0)
    if len(xs) < 32:
        return {}
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    denom = max(len(points) - 1, 1)
    cov = centered.T @ centered / denom
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    results: dict[
        tuple[str, float], list[tuple[float, float, float, float, float]]
    ] = {}
    for quantile in (0.35, 0.5, 0.65):
        candidate = fst__fit_axis_split_candidates(
            gt_mask, points, centered @ axes[:, 0], quantile
        )
        if candidate is not None:
            results["major", quantile] = candidate
    minor = fst__fit_axis_split_candidates(gt_mask, points, centered @ axes[:, 1], 0.5)
    if minor is not None:
        results["minor", 0.5] = minor
    return results


def fst_select_distance_transform_peaks(
    distance_map: np.ndarray,
) -> list[tuple[int, int]]:
    max_value = float(distance_map.max())
    if max_value <= 0.0:
        return []
    local_max = distance_map == cv2.dilate(
        distance_map, np.ones((9, 9), dtype=np.float32)
    )
    peak_mask = local_max & (distance_map >= max_value * 0.28)
    coords = np.argwhere(peak_mask)
    if len(coords) == 0:
        return []
    values = distance_map[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    min_sep = max(12.0, 0.08 * min(distance_map.shape))
    peaks: list[tuple[int, int]] = []
    for idx in order:
        y, x = coords[idx]
        if not peaks:
            peaks.append((int(x), int(y)))
            continue
        if all((math.hypot(x - px, y - py) >= min_sep for px, py in peaks)):
            peaks.append((int(x), int(y)))
        if len(peaks) == 2:
            break
    return peaks


def fst_distance_transform_candidate(
    gt_mask: np.ndarray,
) -> list[tuple[float, float, float, float, float]] | None:
    distance_map = cv2.distanceTransform(gt_mask, cv2.DIST_L2, 5)
    peaks = fst_select_distance_transform_peaks(distance_map)
    if len(peaks) < 2:
        return None
    ys, xs = np.where(gt_mask > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    seeds = np.array(peaks, dtype=np.float32)
    sq_dists = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(sq_dists, axis=1)
    ellipses = []
    for label in (0, 1):
        component_points = points[labels == label]
        if len(component_points) < 16:
            return None
        component_mask = fst_build_component_mask(
            gt_mask.shape, component_points, kernel_size=7
        )
        ellipse = fst_fit_ellipse_from_mask(component_mask)
        if ellipse is None:
            return None
        ellipses.append(ellipse)
    return ellipses


def fst_shift_ellipses_to_local(
    absolute_ellipses: list[tuple[float, float, float, float, float]],
    origin: tuple[int, int],
) -> list[tuple[float, float, float, float, float]]:
    ox, oy = origin
    return [
        fst_normalize_ellipse((cx - ox, cy - oy, a, b, angle))
        for cx, cy, a, b, angle in absolute_ellipses
    ]


def fst_shift_ellipses_to_absolute(
    local_ellipses: list[tuple[float, float, float, float, float]],
    origin: tuple[int, int],
) -> list[tuple[float, float, float, float, float]]:
    ox, oy = origin
    return [
        fst_normalize_ellipse((cx + ox, cy + oy, a, b, angle))
        for cx, cy, a, b, angle in local_ellipses
    ]


def fst_ensure_two_ellipses(
    ellipses: list[tuple[float, float, float, float, float]]
) -> list[tuple[float, float, float, float, float]]:
    if len(ellipses) == 2:
        return ellipses
    cx, cy, _, _, angle = ellipses[0]
    return [ellipses[0], (cx, cy, 2.0, 2.0, angle)]


def fst_downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.copy()
    height, width = mask.shape
    target_h = max(8, int(round(height / factor)))
    target_w = max(8, int(round(width / factor)))
    return cv2.resize(
        mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_AREA
    )


def fst_detect_edge_touches(gt_mask: np.ndarray) -> dict[str, bool]:
    return {
        "left": bool(np.any(gt_mask[:, 0] > 0)),
        "right": bool(np.any(gt_mask[:, -1] > 0)),
        "top": bool(np.any(gt_mask[0, :] > 0)),
        "bottom": bool(np.any(gt_mask[-1, :] > 0)),
    }


def fst_build_initial_single_ellipse(
    gt_mask: np.ndarray,
) -> tuple[float, float, float, float, float]:
    ellipse = fst_fit_ellipse_from_mask(gt_mask)
    if ellipse is None:
        ys, xs = np.where(gt_mask > 0)
        if len(xs) == 0:
            ellipse = (0.5, 0.5, 1.0, 1.0, 0.0)
        else:
            x0 = float(xs.min())
            x1 = float(xs.max())
            y0 = float(ys.min())
            y1 = float(ys.max())
            ellipse = (
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                max(1.0, (x1 - x0 + 1.0) / 2.0),
                max(1.0, (y1 - y0 + 1.0) / 2.0),
                0.0,
            )
    return fst_normalize_ellipse(ellipse)


def fst_reflect_points_across_sides(
    points_xy: np.ndarray, shape: tuple[int, int], touches: dict[str, bool]
) -> list[tuple[str, np.ndarray]]:
    if len(points_xy) == 0:
        return []
    height, width = shape
    x_last = float(width - 1)
    y_last = float(height - 1)
    candidates: list[tuple[str, np.ndarray]] = []

    def reflected_points(active_sides: tuple[str, ...]) -> np.ndarray:
        pts = points_xy.copy()
        for side in active_sides:
            if side == "left":
                pts[:, 0] = -pts[:, 0]
            elif side == "right":
                pts[:, 0] = 2.0 * x_last - pts[:, 0]
            elif side == "top":
                pts[:, 1] = -pts[:, 1]
            elif side == "bottom":
                pts[:, 1] = 2.0 * y_last - pts[:, 1]
        return pts

    for side in ("left", "right", "top", "bottom"):
        if touches[side]:
            candidates.append((f"reflect_{side}", reflected_points((side,))))
    for pair in (
        ("left", "top"),
        ("left", "bottom"),
        ("right", "top"),
        ("right", "bottom"),
    ):
        if all((touches[side] for side in pair)):
            candidates.append((f"reflect_{pair[0]}_{pair[1]}", reflected_points(pair)))
    return candidates


def fst_mask_contour_points(gt_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        return np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
    ys, xs = np.where(gt_mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


def fst_build_edge_aware_initial_candidates(
    gt_mask: np.ndarray, base_ellipse: tuple[float, float, float, float, float]
) -> list[tuple[str, tuple[float, float, float, float, float]]]:
    points_xy = fst_mask_contour_points(gt_mask)
    touches = fst_detect_edge_touches(gt_mask)
    if not any(touches.values()) or len(points_xy) < 5:
        return [("base", fst_normalize_ellipse(base_ellipse))]
    candidates: list[tuple[str, tuple[float, float, float, float, float]]] = [
        ("base", fst_normalize_ellipse(base_ellipse))
    ]
    seen = {tuple((round(v, 3) for v in fst_normalize_ellipse(base_ellipse)))}

    def push(
        name: str, ellipse: tuple[float, float, float, float, float] | None
    ) -> None:
        if ellipse is None:
            return
        norm = fst_normalize_ellipse(ellipse)
        key = tuple((round(v, 3) for v in norm))
        if key in seen:
            return
        seen.add(key)
        candidates.append((name, norm))

    for name, reflected in fst_reflect_points_across_sides(
        points_xy, gt_mask.shape, touches
    ):
        push(
            name,
            fst_fit_ellipse_from_points(np.concatenate([points_xy, reflected], axis=0)),
        )
    cx, cy, a, b, angle = fst_normalize_ellipse(base_ellipse)
    outward_dx = 0.0
    outward_dy = 0.0
    if touches["left"]:
        outward_dx -= max(2.0, a * 0.12)
    if touches["right"]:
        outward_dx += max(2.0, a * 0.12)
    if touches["top"]:
        outward_dy -= max(2.0, b * 0.12)
    if touches["bottom"]:
        outward_dy += max(2.0, b * 0.12)
    if outward_dx != 0.0 or outward_dy != 0.0:
        for factor in (1.0, 1.8, 2.6):
            push(
                f"outward_shift_{factor:.1f}",
                (cx + outward_dx * factor, cy + outward_dy * factor, a, b, angle),
            )
            push(
                f"outward_shift_{factor:.1f}_wider",
                (
                    cx + outward_dx * factor,
                    cy + outward_dy * factor,
                    a * 1.08,
                    b * 1.04,
                    angle,
                ),
            )
            push(
                f"outward_shift_{factor:.1f}_thinner",
                (
                    cx + outward_dx * factor,
                    cy + outward_dy * factor,
                    max(1.0, a * 0.95),
                    max(1.0, b * 0.98),
                    angle,
                ),
            )
    return candidates


def fst_evaluate_single_ellipse(
    gt_mask: np.ndarray,
    ellipse: tuple[float, float, float, float, float],
    gt_area: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    pred_mask = fst_render_ellipses(gt_mask.shape, [ellipse], [1.0])
    if gt_area is None:
        gt_area = int(cv2.countNonZero(gt_mask))
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection = int(cv2.countNonZero(cv2.bitwise_and(gt_mask, pred_mask)))
    union = gt_area + pred_area - intersection
    iou = intersection / union if union > 0 else 1.0
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    return (
        {
            "iou": iou,
            "recall": recall,
            "precision": precision,
            "intersection": float(intersection),
            "union": float(union),
            "pred_area": float(pred_area),
            "gt_area": float(gt_area),
        },
        pred_mask,
    )


def fst_refine_edge_outward(
    gt_mask: np.ndarray,
    ellipse: tuple[float, float, float, float, float],
    recall_target: float,
    touches: dict[str, bool],
    max_rounds: int = 12,
    gt_area: int | None = None,
) -> tuple[tuple[float, float, float, float, float], dict[str, float]]:
    current = list(fst_normalize_ellipse(ellipse))
    if gt_area is None:
        gt_area = int(cv2.countNonZero(gt_mask))

    def evaluate(candidate: list[float]) -> tuple[float, dict[str, float]]:
        norm = fst_normalize_ellipse(tuple(candidate))
        metrics, _ = fst_evaluate_single_ellipse(gt_mask, norm, gt_area=gt_area)
        score = fst_candidate_score(metrics["iou"], metrics["recall"], recall_target)
        fn_pixels = int(metrics["gt_area"] - metrics["intersection"])
        fp_pixels = int(metrics["pred_area"] - metrics["intersection"])
        score -= 1e-06 * float(2 * fn_pixels + fp_pixels)
        return (score, metrics)

    best_score, best_metrics = evaluate(current)
    step_shift_x = max(2.0, current[2] * 0.18)
    step_shift_y = max(2.0, current[3] * 0.18)
    step_radius = max(1.0, max(current[2], current[3]) * 0.08)
    x_dir = (-1.0 if touches["left"] else 0.0) + (1.0 if touches["right"] else 0.0)
    y_dir = (-1.0 if touches["top"] else 0.0) + (1.0 if touches["bottom"] else 0.0)
    for _ in range(max_rounds):
        improved = False
        trials: list[list[float]] = []
        if x_dir != 0.0:
            trials.extend(
                [
                    [
                        current[0] + x_dir * step_shift_x,
                        current[1],
                        current[2],
                        current[3],
                        current[4],
                    ],
                    [
                        current[0] + x_dir * step_shift_x,
                        current[1],
                        current[2] + step_radius,
                        current[3],
                        current[4],
                    ],
                    [
                        current[0] + x_dir * step_shift_x,
                        current[1],
                        max(1.0, current[2] - step_radius),
                        current[3],
                        current[4],
                    ],
                ]
            )
        if y_dir != 0.0:
            trials.extend(
                [
                    [
                        current[0],
                        current[1] + y_dir * step_shift_y,
                        current[2],
                        current[3],
                        current[4],
                    ],
                    [
                        current[0],
                        current[1] + y_dir * step_shift_y,
                        current[2],
                        current[3] + step_radius,
                        current[4],
                    ],
                    [
                        current[0],
                        current[1] + y_dir * step_shift_y,
                        current[2],
                        max(1.0, current[3] - step_radius),
                        current[4],
                    ],
                ]
            )
        if x_dir != 0.0 and y_dir != 0.0:
            trials.append(
                [
                    current[0] + x_dir * step_shift_x,
                    current[1] + y_dir * step_shift_y,
                    current[2],
                    current[3],
                    current[4],
                ]
            )
        for trial in trials:
            score, metrics = evaluate(trial)
            if score > best_score + 1e-07:
                current = trial
                best_score = score
                best_metrics = metrics
                improved = True
        if not improved:
            step_shift_x *= 0.5
            step_shift_y *= 0.5
            step_radius *= 0.5
            if step_shift_x < 0.5 and step_shift_y < 0.5 and (step_radius < 0.5):
                break
    return (fst_normalize_ellipse(tuple(current)), best_metrics)


def fst_solve_single_ellipse(
    gt_mask: np.ndarray, recall_target: float, refinement_rounds: int = 4
) -> tuple[tuple[float, float, float, float, float], dict[str, float]]:
    gt_area = int(cv2.countNonZero(gt_mask))
    base_ellipse = fst_build_initial_single_ellipse(gt_mask)
    touches = fst_detect_edge_touches(gt_mask)
    initial_candidates = fst_build_edge_aware_initial_candidates(gt_mask, base_ellipse)
    pre_ranked: list[
        tuple[float, str, tuple[float, float, float, float, float], dict[str, float]]
    ] = []
    for candidate_name, ellipse in initial_candidates:
        scales, _, _ = fst_optimize_candidate_scales(
            gt_mask, [ellipse], recall_target=recall_target
        )
        baked = fst_apply_scales_to_ellipses([ellipse], scales)[0]
        metrics, _ = fst_evaluate_single_ellipse(gt_mask, baked, gt_area=gt_area)
        pre_ranked.append(
            (
                fst_candidate_score(metrics["iou"], metrics["recall"], recall_target),
                candidate_name,
                baked,
                metrics,
            )
        )
    pre_ranked.sort(key=lambda item: item[0], reverse=True)
    shortlisted = pre_ranked[: min(4, len(pre_ranked))]
    best_ellipse = shortlisted[0][2]
    best_metrics = shortlisted[0][3]
    best_score = shortlisted[0][0]
    for _, _, ellipse, _ in shortlisted:
        refined = ellipse
        refined_metrics, _ = fst_evaluate_single_ellipse(
            gt_mask, refined, gt_area=gt_area
        )
        if refinement_rounds > 0:
            refined_list, _, _ = fst_refine_ellipses_locally(
                gt_mask,
                [ellipse],
                recall_target=recall_target,
                max_rounds=refinement_rounds,
            )
            refined = refined_list[0]
            refined_metrics, _ = fst_evaluate_single_ellipse(
                gt_mask, refined, gt_area=gt_area
            )
        if any(touches.values()):
            edge_refined, edge_metrics = fst_refine_edge_outward(
                gt_mask,
                refined,
                recall_target=recall_target,
                touches=touches,
                max_rounds=12,
                gt_area=gt_area,
            )
            if (
                fst_candidate_score(
                    edge_metrics["iou"], edge_metrics["recall"], recall_target
                )
                > fst_candidate_score(
                    refined_metrics["iou"], refined_metrics["recall"], recall_target
                )
                + 1e-09
            ):
                refined = edge_refined
                refined_metrics = edge_metrics
        if refined_metrics["recall"] < recall_target:
            refined_scales, _, _ = fst_optimize_candidate_scales(
                gt_mask, [refined], recall_target=recall_target
            )
            refined = fst_apply_scales_to_ellipses([refined], refined_scales)[0]
            refined_metrics, _ = fst_evaluate_single_ellipse(
                gt_mask, refined, gt_area=gt_area
            )
        refined_score = fst_candidate_score(
            refined_metrics["iou"], refined_metrics["recall"], recall_target
        )
        if refined_score > best_score + 1e-09:
            best_ellipse = refined
            best_metrics = refined_metrics
            best_score = refined_score
    return (best_ellipse, best_metrics)


def fst_solve_k1_row(
    polygons_json: str,
    recall_target: float,
    exact_refine_rounds: int,
    prepared_payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]
    | None = None,
    gt_polys: list[np.ndarray] | None = None,
) -> tuple[str, dict[str, float], str, list[tuple[float, float, float, float, float]]]:
    if prepared_payload is None:
        gt_mask, origin = fst_rasterize_polygons_to_local_mask(polygons_json)
    else:
        gt_mask, origin = fst_rasterize_local_mask_from_payload(prepared_payload)
    touches = fst_detect_edge_touches(gt_mask)
    if any(touches.values()):
        gt_area = int(cv2.countNonZero(gt_mask))
        ellipse, _ = fst_solve_single_ellipse(
            gt_mask, recall_target=recall_target, refinement_rounds=4
        )
        ellipse, metrics = fst_refine_edge_outward(
            gt_mask,
            ellipse,
            recall_target=recall_target,
            touches=touches,
            max_rounds=12,
            gt_area=gt_area,
        )
        if metrics["recall"] < recall_target:
            scales, _, _ = fst_optimize_candidate_scales(
                gt_mask, [ellipse], recall_target=recall_target
            )
            ellipse = fst_apply_scales_to_ellipses([ellipse], scales)[0]
        absolute = fst_shift_ellipses_to_absolute([ellipse], origin)
        pred_polys = fst_ellipses_to_polygon_arrays(absolute)
        pred_json = json.dumps(
            [poly.astype(np.float64).tolist() for poly in pred_polys]
        )
        exact = (
            fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys)
            if gt_polys is not None
            else fst_compute_exact_metrics(polygons_json, pred_json)
        )
        exact["weighted_error"] = float(fst_compute_weighted_error(exact))
        return (pred_json, exact, "edge_aggressive", absolute)
    single = fst_fit_ellipse_from_mask(gt_mask)
    if single is None:
        ys, xs = np.where(gt_mask > 0)
        if len(xs) == 0:
            ellipse = (0.0, 0.0, 1.0, 1.0, 0.0)
        else:
            ellipse = (float(xs.mean()), float(ys.mean()), 1.0, 1.0, 0.0)
        ellipses = [ellipse]
        candidate_name = "fallback_point"
    else:
        scales, _, _ = fst_optimize_candidate_scales(
            gt_mask, [single], recall_target=recall_target
        )
        ellipses = fst_apply_scales_to_ellipses([single], scales)
        candidate_name = "single_fit"
    if exact_refine_rounds > 0:
        refined, refined_iou, refined_recall = fst_refine_ellipses_locally(
            gt_mask,
            ellipses,
            recall_target=recall_target,
            max_rounds=exact_refine_rounds,
        )
        base_mask = fst_render_ellipses(gt_mask.shape, ellipses, [1.0])
        base_iou, base_recall = fst_compute_mask_metrics(gt_mask, base_mask)
        if refined_recall >= base_recall and refined_iou >= base_iou:
            ellipses = refined
            candidate_name = "single_fit_refined"
    absolute = fst_shift_ellipses_to_absolute(ellipses, origin)
    pred_polys = fst_ellipses_to_polygon_arrays(absolute)
    pred_json = json.dumps([poly.astype(np.float64).tolist() for poly in pred_polys])
    exact = (
        fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys)
        if gt_polys is not None
        else fst_compute_exact_metrics(polygons_json, pred_json)
    )
    exact["weighted_error"] = float(fst_compute_weighted_error(exact))
    return (pred_json, exact, candidate_name, absolute)


def fst__k1_pool_init() -> None:
    cv2.setNumThreads(1)


def fst__solve_k1_row_worker(
    task: tuple[int, int, str, float, int]
) -> tuple[
    int,
    tuple[int, str, str],
    dict[str, object],
    tuple[tuple[int, str], dict[str, object]],
]:
    idx, frame, track_id, recall_target, exact_refine_rounds = task
    polygons_json = fst__ROW_POLYGONS_JSONS[idx]
    prepared_payload = fst__ROW_LOCAL_RASTER_PAYLOADS[idx]
    gt_polys = fst__ROW_GT_POLYGONS[idx]
    pred_json, exact, candidate_name, ellipses = fst_solve_k1_row(
        polygons_json,
        recall_target=recall_target,
        exact_refine_rounds=exact_refine_rounds,
        prepared_payload=prepared_payload,
        gt_polys=gt_polys,
    )
    weighted_error = int(exact["weighted_error"])
    metric_row = {
        "frame": frame,
        "track_id": track_id,
        "candidate_name": candidate_name,
        "gt_area": int(exact["gt_area"]),
        "pred_area": int(exact["pred_area"]),
        "intersection": int(exact["intersection"]),
        "union": int(exact["union"]),
        "recall": float(exact["recall"]),
        "precision": float(exact["precision"]),
        "iou": float(exact["iou"]),
        "weighted_error": weighted_error,
        "ellipse_params": json.dumps(fst_serialize_ellipses(ellipses)),
    }
    solution = {
        "pred_json": pred_json,
        "ellipses": ellipses,
        "metrics": dict(exact),
        "candidate_name": candidate_name,
    }
    return (
        idx,
        (frame, track_id, pred_json),
        metric_row,
        ((frame, track_id), solution),
    )


def fst__solve_k1_payload_worker(
    task: tuple[
        int,
        int,
        str,
        str,
        tuple[tuple[int, int], tuple[int, int], list[np.ndarray]],
        list[np.ndarray],
        float,
        int,
    ]
) -> tuple[
    int,
    tuple[int, str, str],
    dict[str, object],
    tuple[tuple[int, str], dict[str, object]],
]:
    """Spawn-safe K1 worker that does not rely on fork-inherited caches."""

    (
        idx,
        frame,
        track_id,
        polygons_json,
        prepared_payload,
        gt_polys,
        recall_target,
        exact_refine_rounds,
    ) = task
    pred_json, exact, candidate_name, ellipses = fst_solve_k1_row(
        polygons_json,
        recall_target=recall_target,
        exact_refine_rounds=exact_refine_rounds,
        prepared_payload=prepared_payload,
        gt_polys=gt_polys,
    )
    weighted_error = int(exact["weighted_error"])
    metric_row = {
        "frame": frame,
        "track_id": track_id,
        "candidate_name": candidate_name,
        "gt_area": int(exact["gt_area"]),
        "pred_area": int(exact["pred_area"]),
        "intersection": int(exact["intersection"]),
        "union": int(exact["union"]),
        "recall": float(exact["recall"]),
        "precision": float(exact["precision"]),
        "iou": float(exact["iou"]),
        "weighted_error": weighted_error,
        "ellipse_params": json.dumps(fst_serialize_ellipses(ellipses)),
    }
    solution = {
        "pred_json": pred_json,
        "ellipses": ellipses,
        "metrics": dict(exact),
        "candidate_name": candidate_name,
    }
    return (
        idx,
        (frame, track_id, pred_json),
        metric_row,
        ((frame, track_id), solution),
    )


def fst_determine_k1_workers(requested_workers: int, row_count: int) -> int:
    if row_count < 32:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested_workers > 0:
        return max(1, min(requested_workers, cpu_count))
    return max(1, min(cpu_count, row_count))


def fst__precompute_k2_ranked_candidate_worker(
    task: tuple[int, tuple[int, str], float]
) -> tuple[
    int,
    tuple[int, str],
    list[tuple[float, str, list[tuple[float, float, float, float, float]]]],
]:
    idx, key, recall_target = task
    gt_mask, _ = fst_rasterize_local_mask_from_payload(
        fst__ROW_LOCAL_RASTER_PAYLOADS[idx]
    )
    ranked = build_k2_ranked_candidates(gt_mask, recall_target=recall_target)
    return (idx, key, ranked)


def fst_determine_k2_precompute_workers(
    requested_workers: int, selected_count: int
) -> int:
    if selected_count < 32:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested_workers > 0:
        return max(1, min(requested_workers, cpu_count))
    return max(1, min(cpu_count, selected_count))


def fst_solve_k2_selected_rows(
    track_rows: list[tuple[int, str, str, int]],
    selected_keys: set[tuple[int, str]],
    device: torch.device,
    recall_target: float,
    downsample_factor: int,
    steps: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_min_steps: int,
    max_candidates: int,
    max_prev_gap: int,
    ranked_candidates_lookup: dict[
        tuple[int, str],
        list[tuple[float, str, list[tuple[float, float, float, float, float]]]],
    ]
    | None = None,
    prepared_context_lookup: dict[tuple[int, str], dict[str, object]] | None = None,
    reverse: bool = False,
) -> dict[tuple[int, str], dict[str, object]]:
    ordered_rows = list(reversed(track_rows)) if reverse else list(track_rows)
    previous_solution: list[tuple[float, float, float, float, float]] | None = None
    previous_frame: int | None = None
    solved: dict[tuple[int, str], dict[str, object]] = {}
    for frame, track_id, polygons_json, row_idx in ordered_rows:
        key = (int(frame), str(track_id))
        if key not in selected_keys:
            continue
        cached_context = (
            None
            if prepared_context_lookup is None
            else prepared_context_lookup.get(key)
        )
        if cached_context is None:
            gt_mask, origin = fst_rasterize_local_mask_from_payload(
                fst__ROW_LOCAL_RASTER_PAYLOADS[row_idx]
            )
        else:
            gt_mask = np.asarray(cached_context["gt_mask"], dtype=np.uint8)
            origin = tuple(cached_context["origin"])
        previous = None
        if (
            previous_solution is not None
            and previous_frame is not None
            and (abs(int(frame) - previous_frame) <= max_prev_gap)
        ):
            previous = previous_solution
        if ranked_candidates_lookup is None or key not in ranked_candidates_lookup:
            candidates = build_k2_initial_candidates(
                gt_mask,
                prev_absolute_ellipses=previous,
                origin=origin,
                recall_target=recall_target,
                max_candidates=max_candidates,
            )
        else:
            ranked = list(ranked_candidates_lookup[key])
            if previous is not None:
                prev_local = fst_ensure_two_ellipses(
                    fst_shift_ellipses_to_local(previous, origin)
                )
                ranked.append(
                    score_k2_candidate(
                        gt_mask, "prev_track", prev_local, recall_target=recall_target
                    )
                )
                ranked.sort(key=lambda item: item[0], reverse=True)
            candidates = [
                (name, ellipses) for _, name, ellipses in ranked[:max_candidates]
            ]
        ellipses, candidate_name, iou, recall = optimize_k2_candidates_gpu(
            gt_mask,
            candidates,
            recall_target=recall_target,
            device=device,
            downsample_factor=downsample_factor,
            steps=steps,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_min_steps=early_stop_min_steps,
            prepared_context=None
            if cached_context is None
            else dict(cached_context["opt"]),
        )
        absolute_ellipses = fst_shift_ellipses_to_absolute(ellipses, origin)
        solved[key] = {
            "ellipses": absolute_ellipses,
            "candidate_name": candidate_name,
            "local_iou": iou,
            "local_recall": recall,
        }
        previous_solution = absolute_ellipses
        previous_frame = int(frame)
    return solved


def fst_load_rows(sqlite_path: Path) -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    rows = [
        (int(frame), str(track_id), str(polygons))
        for frame, track_id, polygons in conn.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id"
        )
    ]
    conn.close()
    return rows


def fst_load_k1_cost_lookup(csv_path: Path) -> dict[tuple[int, str], int]:
    lookup: dict[tuple[int, str], int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            track_id = str(row["track_id"])
            lookup[(frame, track_id)] = int(float(row["weighted_error"]))
    return lookup


def fst_draw_outlines(
    img: np.ndarray, polygons_json: str, color: tuple[int, int, int], thickness: int
) -> None:
    for pts in fst_parse_polygons(polygons_json):
        cv2.polylines(
            img,
            [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
            True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def fst_blend_mask(
    img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float
) -> None:
    idx = mask > 0
    if np.any(idx):
        img[idx] = (img[idx].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(
            np.uint8
        )


def fst_get_annotation_anchor(
    polygons_json: str, width: int, height: int
) -> tuple[int, int]:
    polygons = fst_parse_polygons(polygons_json)
    all_pts = np.concatenate(polygons, axis=0)
    min_xy = all_pts.min(axis=0)
    x = int(np.clip(math.floor(float(min_xy[0])), 8, max(8, width - 8)))
    y = int(np.clip(math.floor(float(min_xy[1])) - 10, 20, max(20, height - 8)))
    return x, y


def fst_open_nvenc_writer(
    output_video: str, width: int, height: int, fps: float
) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-rc",
        "vbr",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
        output_video,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def fst_load_rows_by_track(sqlite_path: Path) -> dict[str, list[tuple[int, str, str]]]:
    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute(
        "SELECT frame, track_id, polygons FROM masks ORDER BY track_id, frame"
    ).fetchall()
    conn.close()
    by_track: dict[str, list[tuple[int, str, str]]] = {}
    for frame, track_id, polygons in rows:
        by_track.setdefault(str(track_id), []).append(
            (int(frame), str(track_id), str(polygons))
        )
    return by_track


def fst_load_sqlite_mask_metadata(
    reference_sqlite: Path | None,
) -> tuple[
    dict[tuple[int, str], dict[str, object]], dict[str, dict[str, object]], list[int]
]:
    frame_track_meta: dict[tuple[int, str], dict[str, object]] = {}
    track_meta: dict[str, dict[str, object]] = {}
    cut_frames: list[int] = []
    if reference_sqlite is None:
        return frame_track_meta, track_meta, cut_frames
    ref_path = Path(reference_sqlite)
    if not ref_path.exists():
        return frame_track_meta, track_meta, cut_frames
    conn = sqlite3.connect(str(ref_path))
    conn.row_factory = sqlite3.Row
    try:
        mask_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='masks'"
        ).fetchone()
        if mask_tables is not None:
            mask_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(masks)").fetchall()
            }
            if {"frame", "track_id", "polygons"}.issubset(mask_columns):
                select_parts = [
                    "frame",
                    "track_id",
                    (
                        "shape_type"
                        if "shape_type" in mask_columns
                        else "'polygon' AS shape_type"
                    ),
                    ("dilate_px" if "dilate_px" in mask_columns else "0 AS dilate_px"),
                    (
                        "feather_px"
                        if "feather_px" in mask_columns
                        else "0 AS feather_px"
                    ),
                    (
                        "mosaic_block"
                        if "mosaic_block" in mask_columns
                        else "0 AS mosaic_block"
                    ),
                    (
                        "mosaic_alias"
                        if "mosaic_alias" in mask_columns
                        else "0.0 AS mosaic_alias"
                    ),
                    ("label" if "label" in mask_columns else "NULL AS label"),
                    (
                        "is_endpoint_extrapolated"
                        if "is_endpoint_extrapolated" in mask_columns
                        else "0 AS is_endpoint_extrapolated"
                    ),
                ]
                for row in conn.execute(
                    f"SELECT {', '.join(select_parts)} FROM masks"
                ).fetchall():
                    track_id = str(row["track_id"])
                    meta = {
                        "shape_type": str(row["shape_type"])
                        if row["shape_type"] is not None
                        else "polygon",
                        "dilate_px": int(row["dilate_px"])
                        if row["dilate_px"] is not None
                        else 0,
                        "feather_px": int(row["feather_px"])
                        if row["feather_px"] is not None
                        else 0,
                        "mosaic_block": int(row["mosaic_block"])
                        if row["mosaic_block"] is not None
                        else 0,
                        "mosaic_alias": float(row["mosaic_alias"])
                        if row["mosaic_alias"] is not None
                        else 0.0,
                        "label": str(row["label"])
                        if row["label"] is not None
                        else None,
                        "is_endpoint_extrapolated": int(row["is_endpoint_extrapolated"])
                        if row["is_endpoint_extrapolated"] is not None
                        else 0,
                    }
                    frame_track_meta[(int(row["frame"]), track_id)] = meta
                    if track_id not in track_meta:
                        track_defaults = dict(meta)
                        track_defaults["is_endpoint_extrapolated"] = 0
                        track_meta[track_id] = track_defaults
                    elif (
                        track_meta[track_id].get("label") is None
                        and meta.get("label") is not None
                    ):
                        track_meta[track_id]["label"] = meta.get("label")
        if (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            ).fetchone()
            is not None
        ):
            track_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(tracks)").fetchall()
            }
            if "track_id" in track_columns:
                select_sql = "SELECT track_id, {} FROM tracks".format(
                    "label" if "label" in track_columns else "NULL AS label"
                )
                for row in conn.execute(select_sql).fetchall():
                    track_id = str(row["track_id"])
                    info = track_meta.setdefault(track_id, {})
                    if row["label"] is not None:
                        info["label"] = str(row["label"])
        if (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cuts'"
            ).fetchone()
            is not None
        ):
            cut_frames = [
                int(row["frame"])
                for row in conn.execute(
                    "SELECT frame FROM cuts ORDER BY frame"
                ).fetchall()
            ]
    finally:
        conn.close()
    return frame_track_meta, track_meta, cut_frames


def fst_write_sqlite(
    rows: list[tuple[int, str, str]],
    output_path: Path,
    reference_sqlite: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    frame_track_meta, track_meta, cut_frames = fst_load_sqlite_mask_metadata(
        reference_sqlite
    )
    conn = sqlite3.connect(str(output_path))
    try:
        conn.execute(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                is_endpoint_extrapolated INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(frame, track_id)
            )
            """
        )
        conn.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)")
        conn.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
        cur = conn.cursor()
        seen_tracks: dict[str, str | None] = {}
        for frame, track_id, polygons_json in rows:
            key = (int(frame), str(track_id))
            meta = frame_track_meta.get(key)
            if meta is None:
                meta = track_meta.get(str(track_id), {})
            shape_type = str(meta.get("shape_type") or "polygon")
            dilate_px = int(meta.get("dilate_px") or 0)
            feather_px = int(meta.get("feather_px") or 0)
            mosaic_block = int(meta.get("mosaic_block") or 0)
            mosaic_alias = float(meta.get("mosaic_alias") or 0.0)
            label = meta.get("label")
            is_endpoint_extrapolated = int(meta.get("is_endpoint_extrapolated") or 0)
            cur.execute(
                "INSERT INTO masks(frame, track_id, polygons, shape_type, dilate_px, feather_px, mosaic_block, mosaic_alias, label, is_endpoint_extrapolated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(frame),
                    str(track_id),
                    str(polygons_json),
                    shape_type,
                    dilate_px,
                    feather_px,
                    mosaic_block,
                    mosaic_alias,
                    label,
                    is_endpoint_extrapolated,
                ),
            )
            if str(track_id) not in seen_tracks:
                seen_tracks[str(track_id)] = str(label) if label is not None else None
        if seen_tracks:
            cur.executemany(
                "INSERT OR REPLACE INTO tracks(track_id, label) VALUES (?, ?)",
                [(track_id, label) for track_id, label in sorted(seen_tracks.items())],
            )
        if cut_frames:
            cur.executemany(
                "INSERT OR IGNORE INTO cuts(frame) VALUES (?)",
                [(int(frame),) for frame in cut_frames],
            )
        conn.commit()
    finally:
        conn.close()


def fst_evaluate_submission(
    gt_rows: list[tuple[int, str, str]], submission_rows: list[tuple[int, str, str]]
) -> dict[str, float]:
    sub_lookup = {
        (frame, track_id): polygons for frame, track_id, polygons in submission_rows
    }
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    k1_intersection = k1_union = k1_gt_area = k1_pred_area = 0
    k2_intersection = k2_union = k2_gt_area = k2_pred_area = 0
    recall_below_090 = 0
    recall_below_095 = 0
    missing_rows = 0
    mean_recall = []
    mean_precision = []
    mean_iou = []
    k1_count = 0
    k2_count = 0
    for frame, track_id, gt_json in gt_rows:
        pred_json = sub_lookup.get((frame, track_id))
        if pred_json is None:
            missing_rows += 1
            continue
        metrics = fst_compute_exact_metrics(gt_json, pred_json)
        total_intersection += int(metrics["intersection"])
        total_union += int(metrics["union"])
        total_gt_area += int(metrics["gt_area"])
        total_pred_area += int(metrics["pred_area"])
        mean_recall.append(metrics["recall"])
        mean_precision.append(metrics["precision"])
        mean_iou.append(metrics["iou"])
        poly_count = len(json.loads(pred_json))
        if poly_count >= 2:
            k2_count += 1
            k2_intersection += int(metrics["intersection"])
            k2_union += int(metrics["union"])
            k2_gt_area += int(metrics["gt_area"])
            k2_pred_area += int(metrics["pred_area"])
        else:
            k1_count += 1
            k1_intersection += int(metrics["intersection"])
            k1_union += int(metrics["union"])
            k1_gt_area += int(metrics["gt_area"])
            k1_pred_area += int(metrics["pred_area"])
        if metrics["recall"] < 0.9:
            recall_below_090 += 1
        if metrics["recall"] < 0.95:
            recall_below_095 += 1
    return {
        "global_recall": total_intersection / total_gt_area if total_gt_area else 1.0,
        "global_precision": total_intersection / total_pred_area
        if total_pred_area
        else 1.0,
        "global_iou": total_intersection / total_union if total_union else 1.0,
        "mean_recall": float(np.mean(mean_recall)) if mean_recall else 1.0,
        "mean_precision": float(np.mean(mean_precision)) if mean_precision else 1.0,
        "mean_iou": float(np.mean(mean_iou)) if mean_iou else 1.0,
        "recall_below_090": int(recall_below_090),
        "recall_below_095": int(recall_below_095),
        "missing_rows": int(missing_rows),
        "total_gt_rows": len(gt_rows),
        "total_sub_rows": len(submission_rows),
        "k1_count": int(k1_count),
        "k2_count": int(k2_count),
        "k1_recall": k1_intersection / k1_gt_area if k1_gt_area else 0.0,
        "k1_iou": k1_intersection / k1_union if k1_union else 0.0,
        "k2_recall": k2_intersection / k2_gt_area if k2_gt_area else 0.0,
        "k2_iou": k2_intersection / k2_union if k2_union else 0.0,
    }


def fst_evaluate_k1_metric_rows(
    metric_rows: list[dict[str, object]],
    total_gt_rows: int | None = None,
    total_sub_rows: int | None = None,
) -> dict[str, float]:
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    recall_below_090 = 0
    recall_below_095 = 0
    mean_recall: list[float] = []
    mean_precision: list[float] = []
    mean_iou: list[float] = []
    for row in metric_rows:
        intersection = int(row["intersection"])
        union = int(row["union"])
        gt_area = int(row["gt_area"])
        pred_area = int(row["pred_area"])
        recall = float(row["recall"])
        precision = float(row["precision"])
        iou = float(row["iou"])
        total_intersection += intersection
        total_union += union
        total_gt_area += gt_area
        total_pred_area += pred_area
        mean_recall.append(recall)
        mean_precision.append(precision)
        mean_iou.append(iou)
        if recall < 0.9:
            recall_below_090 += 1
        if recall < 0.95:
            recall_below_095 += 1
    if total_gt_rows is None:
        total_gt_rows = len(metric_rows)
    if total_sub_rows is None:
        total_sub_rows = len(metric_rows)
    return {
        "global_recall": total_intersection / total_gt_area if total_gt_area else 1.0,
        "global_precision": total_intersection / total_pred_area
        if total_pred_area
        else 1.0,
        "global_iou": total_intersection / total_union if total_union else 1.0,
        "mean_recall": float(np.mean(mean_recall)) if mean_recall else 1.0,
        "mean_precision": float(np.mean(mean_precision)) if mean_precision else 1.0,
        "mean_iou": float(np.mean(mean_iou)) if mean_iou else 1.0,
        "recall_below_090": int(recall_below_090),
        "recall_below_095": int(recall_below_095),
        "missing_rows": int(max(0, int(total_gt_rows) - len(metric_rows))),
        "total_gt_rows": int(total_gt_rows),
        "total_sub_rows": int(total_sub_rows),
        "k1_count": int(len(metric_rows)),
        "k2_count": 0,
        "k1_recall": total_intersection / total_gt_area if total_gt_area else 0.0,
        "k1_iou": total_intersection / total_union if total_union else 0.0,
        "k2_recall": 0.0,
        "k2_iou": 0.0,
    }


def fst_summarize_weighted_errors(
    values: list[int], thresholds: list[int]
) -> dict[str, object]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "threshold_counts": {}}

    def percentile(p: float) -> int:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return int(ordered[idx])

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": int(ordered[-1]),
        "threshold_counts": {
            str(t): int(sum((v >= t for v in ordered))) for t in thresholds
        },
    }


def fst_serialize_ellipses(
    ellipses: list[tuple[float, float, float, float, float]]
) -> list[list[float]]:
    return [
        [float(cx), float(cy), float(a), float(b), float(angle)]
        for cx, cy, a, b, angle in ellipses
    ]


def fst_deserialize_ellipses(
    serialized: list[list[float]] | list[tuple[float, float, float, float, float]]
) -> list[tuple[float, float, float, float, float]]:
    return [
        (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
        )
        for values in serialized
    ]


def fst_ellipse_area(ellipse: tuple[float, float, float, float, float]) -> float:
    _, _, a, b, _ = ellipse
    return float(max(a, 1.0) * max(b, 1.0))


def fst_composite_center_and_scale(
    ellipses: list[tuple[float, float, float, float, float]]
) -> tuple[float, float, float]:
    if not ellipses:
        return (0.0, 0.0, 1.0)
    areas = np.asarray(
        [max(fst_ellipse_area(ellipse), 1.0) for ellipse in ellipses], dtype=np.float64
    )
    weight_sum = float(areas.sum())
    cx = float(
        sum((ellipse[0] * area for ellipse, area in zip(ellipses, areas))) / weight_sum
    )
    cy = float(
        sum((ellipse[1] * area for ellipse, area in zip(ellipses, areas))) / weight_sum
    )
    scale = float(np.sqrt(weight_sum / max(len(ellipses), 1)))
    return (cx, cy, max(scale, 1.0))


def fst_angle_distance_deg(angle_a: float, angle_b: float) -> float:
    diff = abs((angle_a - angle_b) % 180.0)
    return float(min(diff, 180.0 - diff))


def fst_compute_local_metrics_for_local_ellipses(
    gt_mask: np.ndarray, local_ellipses: list[tuple[float, float, float, float, float]]
) -> dict[str, float]:
    pred_mask = fst_render_ellipses(
        gt_mask.shape, local_ellipses, [1.0] * len(local_ellipses)
    )
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = int((gt_mask | pred_mask).sum())
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    metrics = {
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }
    metrics["weighted_error"] = float(fst_compute_weighted_error(metrics))
    return metrics


def fst_compute_local_metrics_for_absolute_ellipses(
    gt_mask: np.ndarray,
    origin: tuple[int, int],
    absolute_ellipses: list[tuple[float, float, float, float, float]],
) -> dict[str, float]:
    return fst_compute_local_metrics_for_local_ellipses(
        gt_mask, fst_shift_ellipses_to_local(absolute_ellipses, origin)
    )


def fst_build_k2_solve_band(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    k1_ellipses_lookup: dict[
        tuple[int, str], list[tuple[float, float, float, float, float]]
    ],
    threshold: int,
    radius: int,
    error_percentile: float,
    instability_percentile: float,
    instability_floor: float,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        errors = np.asarray(
            [float(k1_metrics_lookup[key]["weighted_error"]) for key in keys],
            dtype=np.float64,
        )
        seed_indices = {
            idx for idx, err in enumerate(errors) if err >= float(threshold)
        }
        high_error_cut = (
            float(np.percentile(errors, error_percentile))
            if len(errors) > 0
            else float(threshold)
        )
        instability_scores = np.zeros(len(track_rows), dtype=np.float64)
        for idx in range(1, len(track_rows)):
            prev_key = keys[idx - 1]
            curr_key = keys[idx]
            prev_ellipse = k1_ellipses_lookup[prev_key][0]
            curr_ellipse = k1_ellipses_lookup[curr_key][0]
            _, _, prev_scale = fst_composite_center_and_scale([prev_ellipse])
            _, _, curr_scale = fst_composite_center_and_scale([curr_ellipse])
            ref_scale = max(prev_scale, curr_scale, 8.0)
            center_jump = (
                math.hypot(
                    curr_ellipse[0] - prev_ellipse[0], curr_ellipse[1] - prev_ellipse[1]
                )
                / ref_scale
            )
            area_jump = abs(
                math.log(max(fst_ellipse_area(curr_ellipse), 1.0))
                - math.log(max(fst_ellipse_area(prev_ellipse), 1.0))
            )
            angle_jump = fst_angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
            instability_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
        instability_cut = (
            float(np.percentile(instability_scores, instability_percentile))
            if len(instability_scores) > 0
            else float("inf")
        )
        extra_indices = {
            idx
            for idx, err in enumerate(errors)
            if err >= max(high_error_cut, float(threshold) * 0.6)
        }
        extra_indices |= {
            idx
            for idx, score in enumerate(instability_scores)
            if score >= max(instability_cut, instability_floor)
        }
        source_indices = seed_indices | extra_indices
        for src_idx in source_indices:
            src_frame = int(track_rows[src_idx][0])
            for frame, track_id_value, _, _ in track_rows:
                if abs(int(frame) - src_frame) <= radius:
                    selected.add((int(frame), str(track_id_value)))
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "seed_count": len(seed_indices),
                "expanded_count": int(sum((1 for key in keys if key in selected))),
                "error_cut": float(high_error_cut),
                "instability_cut": float(instability_cut)
                if np.isfinite(instability_cut)
                else None,
            }
        )
    summary = {
        "threshold": int(threshold),
        "radius": int(radius),
        "selected_count": len(selected),
        "tracks": summary_tracks,
    }
    return (selected, summary)


fst = _register_inline_module(
    "standalone_runtime_fst",
    {
        "WIDTH": "fst_WIDTH",
        "HEIGHT": "fst_HEIGHT",
        "POLYGON_POINTS": "fst_POLYGON_POINTS",
        "MASK_COLOR": "fst_MASK_COLOR",
        "ELLIPSE_COLOR": "fst_ELLIPSE_COLOR",
        "_ANGLE_TABLE": "fst__ANGLE_TABLE",
        "_KERNEL_CACHE": "fst__KERNEL_CACHE",
        "_GRID_CACHE": "fst__GRID_CACHE",
        "_ROW_POLYGONS_JSONS": "fst__ROW_POLYGONS_JSONS",
        "_ROW_LOCAL_RASTER_PAYLOADS": "fst__ROW_LOCAL_RASTER_PAYLOADS",
        "_ROW_GT_POLYGONS": "fst__ROW_GT_POLYGONS",
        "_get_unit_circle": "fst__get_unit_circle",
        "ellipse_to_polygon_array": "fst_ellipse_to_polygon_array",
        "ellipse_to_polygon": "fst_ellipse_to_polygon",
        "parse_polygons": "fst_parse_polygons",
        "make_polygons_json": "fst_make_polygons_json",
        "ellipses_to_polygon_arrays": "fst_ellipses_to_polygon_arrays",
        "rasterize_full": "fst_rasterize_full",
        "rasterize_full_from_polygons": "fst_rasterize_full_from_polygons",
        "prepare_local_raster_payload_from_polygons": "fst_prepare_local_raster_payload_from_polygons",
        "prepare_local_raster_payload": "fst_prepare_local_raster_payload",
        "rasterize_local_mask_from_payload": "fst_rasterize_local_mask_from_payload",
        "rasterize_polygons_to_local_mask": "fst_rasterize_polygons_to_local_mask",
        "set_row_local_raster_cache": "fst_set_row_local_raster_cache",
        "normalize_ellipse": "fst_normalize_ellipse",
        "fit_ellipse_from_points": "fst_fit_ellipse_from_points",
        "fit_ellipse_from_mask": "fst_fit_ellipse_from_mask",
        "render_ellipses": "fst_render_ellipses",
        "compute_mask_metrics": "fst_compute_mask_metrics",
        "compute_exact_metrics": "fst_compute_exact_metrics",
        "compute_exact_metrics_from_gt_polys": "fst_compute_exact_metrics_from_gt_polys",
        "compute_exact_metrics_from_polygons": "fst_compute_exact_metrics_from_polygons",
        "compute_weighted_error": "fst_compute_weighted_error",
        "candidate_score": "fst_candidate_score",
        "binary_search_scale": "fst_binary_search_scale",
        "optimize_candidate_scales": "fst_optimize_candidate_scales",
        "apply_scales_to_ellipses": "fst_apply_scales_to_ellipses",
        "refine_ellipses_locally": "fst_refine_ellipses_locally",
        "build_component_mask": "fst_build_component_mask",
        "_fit_axis_split_candidates": "fst__fit_axis_split_candidates",
        "generate_principal_axis_candidates": "fst_generate_principal_axis_candidates",
        "select_distance_transform_peaks": "fst_select_distance_transform_peaks",
        "distance_transform_candidate": "fst_distance_transform_candidate",
        "shift_ellipses_to_local": "fst_shift_ellipses_to_local",
        "shift_ellipses_to_absolute": "fst_shift_ellipses_to_absolute",
        "ensure_two_ellipses": "fst_ensure_two_ellipses",
        "downsample_mask": "fst_downsample_mask",
        "detect_edge_touches": "fst_detect_edge_touches",
        "build_initial_single_ellipse": "fst_build_initial_single_ellipse",
        "reflect_points_across_sides": "fst_reflect_points_across_sides",
        "mask_contour_points": "fst_mask_contour_points",
        "build_edge_aware_initial_candidates": "fst_build_edge_aware_initial_candidates",
        "evaluate_single_ellipse": "fst_evaluate_single_ellipse",
        "refine_edge_outward": "fst_refine_edge_outward",
        "solve_single_ellipse": "fst_solve_single_ellipse",
        "solve_k1_row": "fst_solve_k1_row",
        "_k1_pool_init": "fst__k1_pool_init",
        "_solve_k1_row_worker": "fst__solve_k1_row_worker",
        "_solve_k1_payload_worker": "fst__solve_k1_payload_worker",
        "determine_k1_workers": "fst_determine_k1_workers",
        "_precompute_k2_ranked_candidate_worker": "fst__precompute_k2_ranked_candidate_worker",
        "determine_k2_precompute_workers": "fst_determine_k2_precompute_workers",
        "solve_k2_selected_rows": "fst_solve_k2_selected_rows",
        "draw_outlines": "fst_draw_outlines",
        "blend_mask": "fst_blend_mask",
        "get_annotation_anchor": "fst_get_annotation_anchor",
        "open_nvenc_writer": "fst_open_nvenc_writer",
        "load_rows": "fst_load_rows",
        "load_k1_cost_lookup": "fst_load_k1_cost_lookup",
        "load_rows_by_track": "fst_load_rows_by_track",
        "write_sqlite": "fst_write_sqlite",
        "evaluate_submission": "fst_evaluate_submission",
        "evaluate_k1_metric_rows": "fst_evaluate_k1_metric_rows",
        "summarize_weighted_errors": "fst_summarize_weighted_errors",
        "serialize_ellipses": "fst_serialize_ellipses",
        "deserialize_ellipses": "fst_deserialize_ellipses",
        "ellipse_area": "fst_ellipse_area",
        "composite_center_and_scale": "fst_composite_center_and_scale",
        "angle_distance_deg": "fst_angle_distance_deg",
        "compute_local_metrics_for_local_ellipses": "fst_compute_local_metrics_for_local_ellipses",
        "compute_local_metrics_for_absolute_ellipses": "fst_compute_local_metrics_for_absolute_ellipses",
        "build_k2_solve_band": "fst_build_k2_solve_band",
    },
)

sys.modules["final_standalone_t5000"] = fst
