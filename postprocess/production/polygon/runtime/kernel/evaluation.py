"""Vector rasterization, cached frame evaluation, and raw candidates."""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

from .geometry import compute_exact_metrics_from_polygons, similarity_residuals
from .types import FrameEvalContext, InstanceRun, ShapeCandidate


def flatten_contours(contours: np.ndarray) -> np.ndarray:
    return np.asarray(contours, dtype=np.float32).reshape(-1, 2)


def split_vector_to_polygons(
    vector: np.ndarray, contour_count: int, anchors_per_contour: int
) -> list[np.ndarray]:
    vec = np.asarray(vector, dtype=np.float32).reshape(
        contour_count, anchors_per_contour, 2
    )
    return [np.asarray(vec[idx], dtype=np.float32) for idx in range(contour_count)]


def vector_proxy_stats(
    vector: np.ndarray, contour_count: int, anchors_per_contour: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    arr = np.asarray(vector, dtype=np.float32).reshape(
        contour_count, anchors_per_contour, 2
    )
    pts = arr.reshape(-1, 2)
    if pts.size <= 0:
        return (
            0.0,
            np.zeros((2,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            1.0,
        )
    center = np.mean(pts, axis=0).astype(np.float32)
    radii = np.linalg.norm(pts - center[None, :], axis=1).astype(np.float32)
    mean_radius = float(max(np.mean(radii, dtype=np.float64), 1e-6))
    x = arr[..., 0].astype(np.float64, copy=False)
    y = arr[..., 1].astype(np.float64, copy=False)
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    area = float(0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1)).sum())
    return area, center, radii, mean_radius


def scale_vector_about_centroid(vector: np.ndarray, scale_mul: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim == 3:
        out = np.zeros_like(arr, dtype=np.float64)
        for idx in range(arr.shape[0]):
            center = np.asarray(np.mean(arr[idx], axis=0), dtype=np.float64)
            out[idx] = center + float(scale_mul) * (arr[idx] - center)
        return out.astype(np.float32)
    pts = arr.reshape(-1, 2)
    center = np.asarray(np.mean(pts, axis=0), dtype=np.float64)
    return (center + float(scale_mul) * (pts - center)).astype(np.float32)


def rasterize_mask_with_context(
    polygons: list[np.ndarray],
    context: FrameEvalContext,
    out_mask: np.ndarray | None = None,
) -> np.ndarray:
    if out_mask is None:
        mask = np.zeros(context.shape_hw, dtype=np.uint8)
    else:
        mask = np.asarray(out_mask, dtype=np.uint8)
        mask.fill(0)
    pts_list: list[np.ndarray] = []
    for poly in polygons:
        pts = (np.asarray(poly, dtype=np.float32) - context.shift_xy[None, :]) * float(
            context.scale_factor
        )
        pts = np.round(pts).astype(np.int32)
        if len(pts) >= 3:
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)
    return mask


def rasterize_interpolated_mask_with_context(
    start_polygons: list[np.ndarray],
    end_polygons: list[np.ndarray],
    alpha: float,
    context: FrameEvalContext,
    out_mask: np.ndarray | None = None,
) -> np.ndarray:
    if out_mask is None:
        mask = np.zeros(context.shape_hw, dtype=np.uint8)
    else:
        mask = np.asarray(out_mask, dtype=np.uint8)
        mask.fill(0)
    pts_list: list[np.ndarray] = []
    alpha32 = np.float32(alpha)
    beta32 = np.float32(1.0) - alpha32
    for start_poly, end_poly in zip(start_polygons, end_polygons):
        start_pts = np.asarray(start_poly, dtype=np.float32)
        end_pts = np.asarray(end_poly, dtype=np.float32)
        pts = (
            beta32 * start_pts + alpha32 * end_pts - context.shift_xy[None, :]
        ) * float(context.scale_factor)
        pts = np.round(pts).astype(np.int32)
        if len(pts) >= 3:
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)
    return mask


def build_frame_eval_contexts(
    run: InstanceRun, args: argparse.Namespace
) -> list[FrameEvalContext]:
    contexts: list[FrameEvalContext] = []
    scale_factor = float(np.clip(float(args.dp_eval_scale), 0.1, 1.0))
    pad = int(max(0, int(args.dp_eval_pad)))
    for frame_idx in range(len(run.frame_numbers)):
        raw_vector = flatten_contours(run.anchors[frame_idx])
        gt_polygon_area, gt_center, gt_radii, gt_mean_radius = vector_proxy_stats(
            raw_vector, run.contour_count, run.anchors_per_contour
        )
        raw_polys = split_vector_to_polygons(
            flatten_contours(run.anchors[frame_idx]),
            run.contour_count,
            run.anchors_per_contour,
        )
        all_polys = [
            np.asarray(poly, dtype=np.float32)
            for poly in run.gt_polygons[frame_idx] + raw_polys
            if len(poly) >= 3
        ]
        if all_polys:
            all_pts = np.concatenate(all_polys, axis=0)
            min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32) - pad
            max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32) + pad
        else:
            min_xy = np.asarray([0, 0], dtype=np.int32)
            max_xy = np.asarray([4, 4], dtype=np.int32)
        shift_xy = min_xy.astype(np.float32)
        width = int(max_xy[0] - min_xy[0] + 1)
        height = int(max_xy[1] - min_xy[1] + 1)
        shape_hw = (
            max(1, int(math.ceil(height * scale_factor))),
            max(1, int(math.ceil(width * scale_factor))),
        )
        context = FrameEvalContext(
            gt_mask=np.zeros(shape_hw, dtype=np.uint8),
            gt_area=0,
            shift_xy=shift_xy,
            shape_hw=shape_hw,
            scale_factor=scale_factor,
            gt_center=np.asarray(gt_center, dtype=np.float32),
            gt_radii=np.asarray(gt_radii, dtype=np.float32),
            gt_mean_radius=float(gt_mean_radius),
            gt_polygon_area=float(gt_polygon_area),
        )
        gt_mask = rasterize_mask_with_context(run.gt_polygons[frame_idx], context)
        contexts.append(
            FrameEvalContext(
                gt_mask=gt_mask,
                gt_area=int(gt_mask.sum()),
                shift_xy=shift_xy,
                shape_hw=shape_hw,
                scale_factor=scale_factor,
                gt_center=np.asarray(gt_center, dtype=np.float32),
                gt_radii=np.asarray(gt_radii, dtype=np.float32),
                gt_mean_radius=float(gt_mean_radius),
                gt_polygon_area=float(gt_polygon_area),
                scratch_pred_mask=np.zeros(shape_hw, dtype=np.uint8),
                scratch_intersection_mask=np.zeros(shape_hw, dtype=np.uint8),
            )
        )
    return contexts


def compute_cached_metrics_from_polygons(
    gt_context: FrameEvalContext, pred_polys: list[np.ndarray]
) -> dict[str, float]:
    pred_mask = rasterize_mask_with_context(
        pred_polys,
        gt_context,
        out_mask=gt_context.scratch_pred_mask,
    )
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection_mask = gt_context.scratch_intersection_mask
    if intersection_mask is None:
        intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
    cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    union = int(gt_context.gt_area + pred_area - intersection)
    recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_context.gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def compute_cached_metrics_from_interpolated_polygons(
    gt_context: FrameEvalContext,
    start_polys: list[np.ndarray],
    end_polys: list[np.ndarray],
    alpha: float,
) -> dict[str, float]:
    pred_mask = rasterize_interpolated_mask_with_context(
        start_polys,
        end_polys,
        alpha,
        gt_context,
        out_mask=gt_context.scratch_pred_mask,
    )
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection_mask = gt_context.scratch_intersection_mask
    if intersection_mask is None:
        intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
    cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    union = int(gt_context.gt_area + pred_area - intersection)
    recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_context.gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def evaluate_frame_vector_loss_budget(
    run: InstanceRun,
    frame_idx: int,
    vector: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, float]:
    pred_polys = split_vector_to_polygons(
        vector, run.contour_count, run.anchors_per_contour
    )
    if eval_contexts is not None:
        metrics = compute_cached_metrics_from_polygons(
            eval_contexts[int(frame_idx)], pred_polys
        )
    else:
        metrics = compute_exact_metrics_from_polygons(
            run.gt_polygons[int(frame_idx)], pred_polys
        )
    return float(frame_accuracy_loss(metrics, args)), float(
        recall_budget_from_metrics(metrics)
    )


def recall_budget_from_metrics(metrics: dict[str, float]) -> float:
    return max(0.0, 1.0 - float(metrics["recall"]))


def recall_budget_limit(frame_count: int, args: argparse.Namespace) -> float:
    recall_min = float(np.clip(float(args.recall_min), 0.0, 1.0))
    return float(max(frame_count, 0)) * max(0.0, 1.0 - recall_min)


def recall_violation(
    total_budget: float, frame_count: int, args: argparse.Namespace
) -> float:
    return max(float(total_budget) - float(recall_budget_limit(frame_count, args)), 0.0)


def frame_accuracy_loss(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return float(args.interval_iou_weight) * (1.0 - float(metrics["iou"]))


def adaptive_shape_penalty_scales(
    frame_loss_mean: float, args: argparse.Namespace
) -> tuple[float, float]:
    mean_loss = max(float(frame_loss_mean), 0.0)
    gain = max(float(args.shape_penalty_adapt_gain), 0.0)
    if gain <= 0.0:
        return 1.0, 1.0
    base = 1.0 + gain * mean_loss
    distance_scale = 1.0 / max(
        base ** max(float(args.shape_distance_relief), 0.0), 1e-6
    )
    switch_scale = 1.0 / max(base ** max(float(args.shape_switch_relief), 0.0), 1e-6)
    distance_scale = max(float(args.shape_distance_min_scale), float(distance_scale))
    switch_scale = max(float(args.shape_switch_min_scale), float(switch_scale))
    return float(distance_scale), float(switch_scale)


def build_frame_candidates(
    run: InstanceRun,
    _contexts: list[object],
    eval_contexts: list[FrameEvalContext],
    args: argparse.Namespace,
) -> list[list[ShapeCandidate]]:
    candidates_by_frame: list[list[ShapeCandidate]] = []
    for idx in range(len(run.frame_numbers)):
        raw_vector = flatten_contours(run.anchors[idx])
        raw_metrics = compute_cached_metrics_from_polygons(
            eval_contexts[idx],
            split_vector_to_polygons(
                raw_vector, run.contour_count, run.anchors_per_contour
            ),
        )
        raw_frame_loss = frame_accuracy_loss(raw_metrics, args)
        raw_area, raw_center, raw_radii, raw_mean_radius = vector_proxy_stats(
            raw_vector, run.contour_count, run.anchors_per_contour
        )
        raw_candidate = ShapeCandidate(
            label="raw",
            vector=np.asarray(raw_vector, dtype=np.float32),
            polygons=split_vector_to_polygons(
                raw_vector, run.contour_count, run.anchors_per_contour
            ),
            frame_loss=float(raw_frame_loss),
            objective=float(raw_frame_loss),
            recall_budget=float(recall_budget_from_metrics(raw_metrics)),
            area=float(raw_area),
            center=np.asarray(raw_center, dtype=np.float32),
            radii=np.asarray(raw_radii, dtype=np.float32),
            mean_radius=float(raw_mean_radius),
        )
        candidates_by_frame.append([raw_candidate])
    return candidates_by_frame


def shape_distance(vector_a: np.ndarray, vector_b: np.ndarray, scale: float) -> float:
    residual, _ = similarity_residuals(
        np.asarray(vector_a, dtype=np.float32), np.asarray(vector_b, dtype=np.float32)
    )
    norms = np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=1)
    return float(np.mean(norms) / max(float(scale), 1.0))


__all__ = (
    "adaptive_shape_penalty_scales",
    "build_frame_candidates",
    "build_frame_eval_contexts",
    "compute_cached_metrics_from_interpolated_polygons",
    "compute_cached_metrics_from_polygons",
    "evaluate_frame_vector_loss_budget",
    "flatten_contours",
    "frame_accuracy_loss",
    "rasterize_interpolated_mask_with_context",
    "rasterize_mask_with_context",
    "recall_budget_from_metrics",
    "recall_budget_limit",
    "recall_violation",
    "scale_vector_about_centroid",
    "shape_distance",
    "split_vector_to_polygons",
    "vector_proxy_stats",
)
