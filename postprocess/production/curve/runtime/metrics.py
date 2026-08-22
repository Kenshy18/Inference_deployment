"""Raster and temporal metrics for the Production Catmull--Rom runtime."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from production.polygon.runtime.spatial_support.optimizer import (
    orient_ccw,
    temporal_residuals,
)

from .topology import has_strict_self_intersection


@dataclass(frozen=True, slots=True)
class CurveMetrics:
    frames: int
    mean_iou: float
    minimum_iou: float
    q01_iou: float
    q05_iou: float
    mean_recall: float
    minimum_recall: float
    recall_violations: int
    temporal_residual: float
    temporal_q95: float
    self_intersections: int


def frame_raster_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    padding: int = 2,
) -> tuple[float, float]:
    """Return OpenCV-raster IoU and Recall on one common local canvas."""
    left = orient_ccw(reference)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
    minimum = np.floor(np.minimum(np.min(left, axis=0), np.min(right, axis=0))).astype(
        np.int64
    ) - int(padding)
    maximum = np.ceil(np.maximum(np.max(left, axis=0), np.max(right, axis=0))).astype(
        np.int64
    ) + int(padding)
    width = max(1, int(maximum[0] - minimum[0] + 1))
    height = max(1, int(maximum[1] - minimum[1] + 1))
    reference_mask = np.zeros((height, width), dtype=np.uint8)
    candidate_mask = np.zeros_like(reference_mask)
    cv2.fillPoly(
        reference_mask,
        [np.rint(left - minimum[None, :]).astype(np.int32)],
        1,
    )
    cv2.fillPoly(
        candidate_mask,
        [np.rint(right - minimum[None, :]).astype(np.int32)],
        1,
    )
    intersection = int(
        cv2.countNonZero(cv2.bitwise_and(reference_mask, candidate_mask))
    )
    reference_area = int(cv2.countNonZero(reference_mask))
    candidate_area = int(cv2.countNonZero(candidate_mask))
    union = reference_area + candidate_area - intersection
    iou = float(intersection / union) if union else 1.0
    recall = float(intersection / reference_area) if reference_area else 1.0
    return iou, recall


def sequence_raster_metrics(
    references: list[np.ndarray],
    sampled_curves: np.ndarray,
    control_points: np.ndarray,
    *,
    recall_floor: float,
) -> CurveMetrics:
    if len(references) != len(sampled_curves):
        raise ValueError("references and sampled curves must have equal length")
    values = [
        frame_raster_metrics(reference, curve)
        for reference, curve in zip(references, sampled_curves)
    ]
    ious = np.asarray([row[0] for row in values], dtype=np.float64)
    recalls = np.asarray([row[1] for row in values], dtype=np.float64)
    residuals = temporal_residuals(np.asarray(control_points, dtype=np.float64))
    flat = residuals.reshape(-1) if residuals.size else np.zeros((1,), np.float64)
    return CurveMetrics(
        frames=int(len(references)),
        mean_iou=float(np.mean(ious)),
        minimum_iou=float(np.min(ious)),
        q01_iou=float(np.quantile(ious, 0.01)),
        q05_iou=float(np.quantile(ious, 0.05)),
        mean_recall=float(np.mean(recalls)),
        minimum_recall=float(np.min(recalls)),
        recall_violations=int(np.count_nonzero(recalls + 1e-12 < float(recall_floor))),
        temporal_residual=float(np.mean(flat)),
        temporal_q95=float(np.quantile(flat, 0.95)),
        self_intersections=int(
            sum(has_strict_self_intersection(curve) for curve in sampled_curves)
        ),
    )


__all__ = ("CurveMetrics", "frame_raster_metrics", "sequence_raster_metrics")
