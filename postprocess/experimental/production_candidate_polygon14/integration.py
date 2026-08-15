"""Narrow adapter from the prepared polygon run to polygon14_keyframe_v1."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from experimental.humanlike_vertex_placement_20260812.spatial import (
    rdp_fixed_count,
)
from experimental.temporal_vertex_decimation_20260812.optimizer import (
    _best_phase,
    has_self_intersection,
)

from .config import CANDIDATE, Polygon14CandidateConfig
from .spatial import build_spatial_track


_EPSILON = 1e-12


def _metrics(
    endpoint_evaluator, frame_index: int, anchors: np.ndarray
) -> tuple[float, float]:
    components, vertices, _coordinates = anchors.shape
    values = endpoint_evaluator.exact_frame_metrics(
        int(frame_index),
        np.ascontiguousarray(anchors.reshape(-1), dtype=np.float32),
        int(components),
        int(vertices),
    )
    return float(values[4]), float(values[6])


def _scaled(anchors: np.ndarray, scale: float) -> np.ndarray:
    value = np.asarray(anchors, dtype=np.float64)
    centers = np.mean(value, axis=1, keepdims=True)
    return np.asarray(centers + float(scale) * (value - centers), dtype=np.float32)


def _direct_rdp_frame(
    references,
    aligned_to: np.ndarray,
    vertices: int,
) -> np.ndarray:
    output = np.empty_like(aligned_to, dtype=np.float32)
    for slot, reference in enumerate(references):
        candidate = rdp_fixed_count(
            np.asarray(reference, dtype=np.float64), int(vertices)
        )
        candidate = _best_phase(
            np.asarray(aligned_to[slot], dtype=np.float64),
            candidate,
            allow_reverse=False,
            procrustes=False,
        )
        output[slot] = np.asarray(candidate, dtype=np.float32)
    return output


def _repair_frame_exact_recall(
    endpoint_evaluator,
    frame_index: int,
    references,
    anchors: np.ndarray,
    *,
    recall_floor: float,
) -> tuple[np.ndarray | None, float, float, float]:
    """Return the highest-IoU exact-feasible local repair for one frame."""
    vertices = int(anchors.shape[1])
    bases = [np.asarray(anchors, dtype=np.float32)]
    direct = _direct_rdp_frame(references, anchors, vertices)
    if not np.array_equal(direct, bases[0]):
        bases.append(direct)
    best = None
    # Fine near 1.0 where almost all observed repairs live, then a bounded
    # emergency range.  The final native exact gate, not scale monotonicity,
    # decides feasibility.
    coarse_scales = [1.0]
    coarse_scales.extend(float(value) for value in np.arange(1.001, 1.051, 0.001))
    coarse_scales.extend(float(value) for value in np.arange(1.052, 1.201, 0.002))
    for base in bases:
        first_feasible_scale = None
        for scale in coarse_scales:
            candidate = base if scale == 1.0 else _scaled(base, scale)
            if any(has_self_intersection(polygon) for polygon in candidate):
                continue
            recall, iou = _metrics(endpoint_evaluator, frame_index, candidate)
            if recall + _EPSILON < float(recall_floor):
                continue
            first_feasible_scale = float(scale)
            score = (float(iou), -float(scale))
            if best is None or score > best[0]:
                best = (score, candidate.copy(), recall, iou, float(scale))
            break
        if first_feasible_scale is None or first_feasible_scale <= 1.0:
            continue
        lower = max(1.0, first_feasible_scale - 0.002)
        for scale in np.arange(lower, first_feasible_scale + 0.00005, 0.0001):
            candidate = _scaled(base, float(scale))
            if any(has_self_intersection(polygon) for polygon in candidate):
                continue
            recall, iou = _metrics(endpoint_evaluator, frame_index, candidate)
            if recall + _EPSILON < float(recall_floor):
                continue
            score = (float(iou), -float(scale))
            if best is None or score > best[0]:
                best = (score, candidate.copy(), recall, iou, float(scale))
    if best is None:
        return None, 0.0, 0.0, 0.0
    return best[1], float(best[2]), float(best[3]), float(best[4])


def _repair_sequence_exact_recall(
    run,
    endpoint_evaluator,
    anchors: np.ndarray,
    *,
    recall_floor: float,
) -> tuple[np.ndarray, int, float]:
    output = np.asarray(anchors, dtype=np.float32).copy()
    repaired = 0
    maximum_scale = 1.0
    for frame_index in range(int(output.shape[0])):
        recall, _iou = _metrics(endpoint_evaluator, frame_index, output[frame_index])
        if recall + _EPSILON >= float(recall_floor):
            continue
        candidate, repaired_recall, _repaired_iou, scale = _repair_frame_exact_recall(
            endpoint_evaluator,
            frame_index,
            run.gt_polygons[frame_index],
            output[frame_index],
            recall_floor=recall_floor,
        )
        if candidate is None or repaired_recall + _EPSILON < float(recall_floor):
            continue
        output[frame_index] = candidate
        repaired += 1
        maximum_scale = max(maximum_scale, float(scale))
    return output, int(repaired), float(maximum_scale)


def apply_spatial_candidate(
    run,
    profile: dict[str, float | int],
    endpoint_evaluator=None,
    config: Polygon14CandidateConfig = CANDIDATE,
) -> None:
    """Select the smallest exact-feasible track-wide vertex count.

    A run keeps one vertex count for its entire lifetime so indexed linear
    interpolation and vertex identity remain well-defined.  The unchanged
    tracked masks remain the Recall reference.  Counts are attempted in the
    frozen 14/16/18/20 order and a run is rejected if even 20 points cannot
    satisfy the native exact Recall floor on every source frame.
    """
    if bool(getattr(run, "_polygon14_candidate_applied", False)):
        return
    if endpoint_evaluator is None:
        raise RuntimeError("adaptive vertex selection requires native exact evaluator")
    fallback_counts = tuple(int(value) for value in config.vertex_fallbacks)
    if fallback_counts != (14, 16, 18, 20):
        raise RuntimeError(f"unexpected vertex fallback contract: {fallback_counts}")
    # run.gt_polygons deliberately remains untouched. Both DP edge feasibility
    # and pair-vote use it as the exact source-mask reference.
    started = time.perf_counter()
    attempts: list[tuple[int, np.ndarray, object, float, float, int, float]] = []
    selected = None
    recall_only = None
    exact_repaired_total = 0
    exact_repair_maximum_scale = 1.0
    for vertices in fallback_counts:
        attempt_config = replace(config, vertices_per_component=int(vertices))
        anchors, stats = build_spatial_track(run.gt_polygons, attempt_config)
        anchors, exact_repaired, repair_scale = _repair_sequence_exact_recall(
            run,
            endpoint_evaluator,
            anchors,
            recall_floor=float(config.spatial_recall_floor),
        )
        exact_repaired_total += int(exact_repaired)
        exact_repair_maximum_scale = max(
            exact_repair_maximum_scale, float(repair_scale)
        )
        minimum_recall = 1.0
        minimum_iou = 1.0
        for frame_index in range(int(anchors.shape[0])):
            vector = np.ascontiguousarray(
                anchors[frame_index].reshape(-1), dtype=np.float32
            )
            values = endpoint_evaluator.exact_frame_metrics(
                int(frame_index),
                vector,
                int(stats.components),
                int(vertices),
            )
            minimum_recall = min(minimum_recall, float(values[4]))
            minimum_iou = min(minimum_iou, float(values[6]))
        attempt = (
            int(vertices),
            np.asarray(anchors, dtype=np.float32),
            stats,
            float(minimum_recall),
            float(minimum_iou),
            int(exact_repaired),
            float(repair_scale),
        )
        attempts.append(attempt)
        recall_feasible = minimum_recall + _EPSILON >= float(
            config.spatial_recall_floor
        )
        quality_feasible = recall_feasible and (
            minimum_iou + _EPSILON >= float(config.spatial_iou_floor)
        )
        if recall_feasible and recall_only is None:
            recall_only = attempt
        if quality_feasible:
            selected = attempt
            break
    if selected is None:
        selected = recall_only
    if selected is None:
        last = attempts[-1]
        profile["polygon_vertex_fallback_failed_runs"] = (
            int(profile.get("polygon_vertex_fallback_failed_runs", 0)) + 1
        )
        raise RuntimeError(
            "no exact Recall-feasible spatial polygon through 20 vertices: "
            f"stream={getattr(run, 'stream_id', '<unknown>')!r} "
            f"minimum_recall_at_20={last[3]:.9f} "
            f"floor={config.spatial_recall_floor:.9f}"
        )
    (
        vertices,
        anchors,
        stats,
        minimum_recall,
        minimum_iou,
        selected_exact_repaired,
        selected_repair_scale,
    ) = selected
    run.anchors = np.ascontiguousarray(anchors, dtype=np.float32)
    run.anchors_per_contour = int(vertices)
    run.run_target_total_points = int(stats.components * stats.vertices_per_component)
    run._polygon14_candidate_applied = True
    run._spatial_vertex_count = int(vertices)
    run._spatial_minimum_recall = float(minimum_recall)
    run._spatial_minimum_iou = float(minimum_iou)
    profile["polygon14_spatial_seconds"] = float(
        profile.get("polygon14_spatial_seconds", 0.0)
    ) + float(time.perf_counter() - started)
    profile["polygon14_frames"] = int(profile.get("polygon14_frames", 0)) + int(
        stats.frames
    )
    profile["polygon14_components"] = int(profile.get("polygon14_components", 0)) + int(
        stats.components
    )
    profile["polygon14_repaired_component_frames"] = int(
        profile.get("polygon14_repaired_component_frames", 0)
    ) + int(stats.repaired_component_frames)
    profile["polygon14_fallback_component_frames"] = int(
        profile.get("polygon14_fallback_component_frames", 0)
    ) + int(stats.fallback_component_frames)
    profile["polygon_vertex_attempts"] = int(
        profile.get("polygon_vertex_attempts", 0)
    ) + len(attempts)
    profile[f"polygon_vertex_selected_{vertices}_runs"] = (
        int(profile.get(f"polygon_vertex_selected_{vertices}_runs", 0)) + 1
    )
    profile["polygon_vertex_fallback_runs"] = int(
        profile.get("polygon_vertex_fallback_runs", 0)
    ) + int(vertices != fallback_counts[0])
    profile["polygon_vertex_recall_only_runs"] = int(
        profile.get("polygon_vertex_recall_only_runs", 0)
    ) + int(minimum_iou + _EPSILON < float(config.spatial_iou_floor))
    profile["polygon_vertex_exact_repaired_frames"] = int(
        profile.get("polygon_vertex_exact_repaired_frames", 0)
    ) + int(selected_exact_repaired)
    profile["polygon_vertex_exact_repair_maximum_scale"] = max(
        float(profile.get("polygon_vertex_exact_repair_maximum_scale", 1.0)),
        float(selected_repair_scale),
    )
    profile["polygon_vertex_exact_repair_attempted_frames"] = int(
        profile.get("polygon_vertex_exact_repair_attempted_frames", 0)
    ) + int(exact_repaired_total)
    profile["polygon_vertex_exact_repair_attempted_maximum_scale"] = max(
        float(profile.get("polygon_vertex_exact_repair_attempted_maximum_scale", 1.0)),
        float(exact_repair_maximum_scale),
    )
    profile["polygon_vertex_selected_max"] = max(
        int(profile.get("polygon_vertex_selected_max", 0)), int(vertices)
    )
    profile["polygon_vertex_selected_min"] = min(
        int(profile.get("polygon_vertex_selected_min", vertices)), int(vertices)
    )
