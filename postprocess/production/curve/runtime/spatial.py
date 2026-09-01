"""Spatial-control builders for Production curve fitting and parity review."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from production.polygon.runtime.spatial_support.quality_repair import (
    persistent_line_fit_quality_guarded,
)

from .keyframe_dp import BoundaryRenderer, _frame_metrics
from .topology import has_strict_self_intersection


@dataclass(frozen=True, slots=True)
class SpatialRepairResult:
    controls: np.ndarray
    scales: np.ndarray
    minimum_recall: float
    mean_iou: float
    recall_violations: int
    repaired_frames: int
    unresolved_frames: tuple[int, ...]
    invalid_candidate_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompletionEnvelopeResult:
    controls: np.ndarray
    minimum_recall: float
    mean_iou: float
    maximum_area_ratio: float
    replaced_frames: tuple[int, ...]


def repair_spatial_controls(
    references: list[np.ndarray],
    controls: np.ndarray,
    renderer: BoundaryRenderer,
    *,
    recall_floor: float,
    maximum_scale: float = 1.08,
    scale_step: float = 0.002,
) -> SpatialRepairResult:
    """Use the smallest high-IoU isotropic P repair under exact Recall."""
    source = np.asarray(controls, dtype=np.float64)
    output = source.copy()
    scales = np.ones((len(source),), dtype=np.float64)
    selected_ious = np.zeros((len(source),), dtype=np.float64)
    selected_recalls = np.zeros_like(selected_ious)
    unresolved: list[int] = []
    invalid: list[int] = []
    candidates = np.arange(
        1.0,
        float(maximum_scale) + float(scale_step) * 0.5,
        float(scale_step),
    )
    for frame, (reference, points) in enumerate(zip(references, source, strict=True)):
        center = np.mean(points, axis=0, keepdims=True)
        best = None
        highest_recall = None
        for scale in candidates:
            trial = center + float(scale) * (points - center)
            boundary = renderer(trial)
            if has_strict_self_intersection(boundary):
                continue
            iou, recall, _precision, _area_ratio = _frame_metrics(reference, boundary)
            fallback_score = (float(recall), float(iou), -float(scale))
            if highest_recall is None or fallback_score > highest_recall[0]:
                highest_recall = (
                    fallback_score,
                    trial.copy(),
                    float(scale),
                    iou,
                    recall,
                )
            if recall + 1e-12 < float(recall_floor):
                continue
            score = (float(iou), -float(scale))
            if best is None or score > best[0]:
                best = (score, trial.copy(), float(scale), iou, recall)
        chosen = best if best is not None else highest_recall
        if chosen is None:
            # Scaling preserves the topology of P.  If the fitted curve is
            # self-intersecting, every scale is therefore invalid and the
            # conservative completion-envelope path must replace this frame.
            # Keep a finite placeholder here instead of aborting a multi-hour
            # batch before that deterministic fallback can run.
            output[frame] = points
            scales[frame] = 1.0
            selected_ious[frame] = 0.0
            selected_recalls[frame] = 0.0
            unresolved.append(int(frame))
            invalid.append(int(frame))
            continue
        _score, trial, scale, iou, recall = chosen
        output[frame] = trial
        scales[frame] = scale
        selected_ious[frame] = iou
        selected_recalls[frame] = recall
        if best is None:
            unresolved.append(int(frame))
    return SpatialRepairResult(
        controls=np.ascontiguousarray(output, dtype=np.float64),
        scales=scales,
        minimum_recall=float(np.min(selected_recalls)),
        mean_iou=float(np.mean(selected_ious)),
        recall_violations=int(
            np.count_nonzero(selected_recalls + 1e-12 < float(recall_floor))
        ),
        repaired_frames=int(np.count_nonzero(scales > 1.0 + 1e-12)),
        unresolved_frames=tuple(unresolved),
        invalid_candidate_frames=tuple(invalid),
    )


def complete_spatial_recall_with_envelopes(
    references: list[np.ndarray],
    controls: np.ndarray,
    renderer: BoundaryRenderer,
    *,
    unresolved_frames: tuple[int, ...],
    recall_floor: float,
    binary_steps: int = 20,
) -> CompletionEnvelopeResult:
    """Replace only otherwise-infeasible frames with a bounded ellipse.

    A malformed or extremely concave raw contour can occasionally be
    impossible to cover by scaling its fitted Catmull--Rom controls. Failing
    an eight-hour batch for that ordinary approximation limitation is worse
    than using a conservative mask on the affected frame. This fallback
    remains inside the exact Production representation: it has the same P
    count, derives all handles with the fixed 1/6 rule, and is accepted only
    after the authoritative raster Recall and topology checks pass.

    The axis-aligned ellipse starts large enough to enclose the raw contour's
    bounding box. A deterministic binary search then removes excess area
    while preserving the hard Recall floor.
    """

    source = np.asarray(controls, dtype=np.float64)
    output = source.copy()
    replaced = tuple(sorted(set(int(value) for value in unresolved_frames)))
    if any(index < 0 or index >= len(source) for index in replaced):
        raise ValueError("completion envelope frame is outside the sequence")
    point_count = int(source.shape[1])
    if point_count < 3:
        raise ValueError("completion envelope requires at least three P points")
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))

    for frame in replaced:
        reference = np.asarray(references[frame], dtype=np.float64).reshape(-1, 2)
        if len(reference) < 3 or not np.all(np.isfinite(reference)):
            raise ValueError(f"frame {frame}: invalid reference contour")
        minimum = np.min(reference, axis=0)
        maximum = np.max(reference, axis=0)
        center = 0.5 * (minimum + maximum)
        half_extent = np.maximum(0.5 * (maximum - minimum) + 2.0, 2.0)

        def candidate(scale: float) -> tuple[np.ndarray, float, float, float]:
            points = center + unit * half_extent * float(scale)
            boundary = renderer(points)
            if has_strict_self_intersection(boundary):
                return points, 0.0, 0.0, float("inf")
            iou, recall, _precision, area_ratio = _frame_metrics(
                reference,
                boundary,
            )
            return points, float(iou), float(recall), float(area_ratio)

        low = 0.0
        high = float(np.sqrt(2.0) * 1.10)
        best: tuple[tuple[float, float], np.ndarray] | None = None
        for _attempt in range(8):
            points, iou, recall, area_ratio = candidate(high)
            if recall + 1e-12 >= float(recall_floor):
                best = ((iou, -area_ratio), points.copy())
                break
            high *= 1.5
        if best is None:
            # This indicates invalid raster geometry rather than an ordinary
            # approximation miss. Keep corrupt input from being published.
            raise RuntimeError(
                f"frame {frame}: conservative Catmull--Rom envelope could "
                "not satisfy exact Recall"
            )
        for _step in range(max(1, int(binary_steps))):
            middle = 0.5 * (low + high)
            points, iou, recall, area_ratio = candidate(middle)
            if recall + 1e-12 >= float(recall_floor):
                high = middle
                score = (iou, -area_ratio)
                if score > best[0]:
                    best = (score, points.copy())
            else:
                low = middle
        output[frame] = best[1]

    recalls = np.empty((len(output),), dtype=np.float64)
    ious = np.empty_like(recalls)
    area_ratios = np.empty_like(recalls)
    for frame, (reference, points) in enumerate(zip(references, output, strict=True)):
        boundary = renderer(points)
        if has_strict_self_intersection(boundary):
            raise RuntimeError(f"frame {frame}: completion produced an invalid curve")
        iou, recall, _precision, area_ratio = _frame_metrics(reference, boundary)
        recalls[frame] = recall
        ious[frame] = iou
        area_ratios[frame] = area_ratio
    violations = np.flatnonzero(recalls + 1e-12 < float(recall_floor))
    if len(violations):
        raise RuntimeError(
            "completion envelope exact audit failed at frames "
            + ",".join(str(int(value)) for value in violations[:16])
        )
    return CompletionEnvelopeResult(
        controls=np.ascontiguousarray(output, dtype=np.float64),
        minimum_recall=float(np.min(recalls)),
        mean_iou=float(np.mean(ious)),
        maximum_area_ratio=float(np.max(area_ratios)),
        replaced_frames=replaced,
    )


def build_production_polygon_controls(
    references: list[np.ndarray],
    point_count: int,
    *,
    recall_floor: float = 0.97,
    iou_floor: float = 0.95,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run Production's current persistent line-fit placement at one count."""
    controls, stats = persistent_line_fit_quality_guarded(
        references,
        int(point_count),
        recall_floor=float(recall_floor),
        iou_floor=float(iou_floor),
        dense_vertices=64,
        coverage_quantile=0.65,
        maximum_intersection_radius=0.20,
        intersection_regularization=0.01,
    )
    return np.asarray(controls, dtype=np.float64), {
        "frames": int(stats.frames),
        "quality_repaired_frames": int(stats.repaired_frames),
        "quality_fallback_frames": int(stats.fallback_frames),
        "quality_tested_blends": int(stats.tested_blends),
    }


__all__ = (
    "CompletionEnvelopeResult",
    "SpatialRepairResult",
    "build_production_polygon_controls",
    "complete_spatial_recall_with_envelopes",
    "repair_spatial_controls",
)
