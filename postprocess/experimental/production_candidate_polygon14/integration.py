"""Narrow adapter from the prepared polygon run to polygon14_keyframe_v1."""

from __future__ import annotations

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
    """Build an independent RDP candidate aligned to persistent vertex IDs."""
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
    maximum_scale: float = 1.05,
) -> tuple[np.ndarray | None, float, float, float]:
    """Choose the highest-IoU exact-feasible fixed-14 local repair."""
    vertices = int(anchors.shape[1])
    bases = [np.asarray(anchors, dtype=np.float32)]
    direct = _direct_rdp_frame(references, anchors, vertices)
    if not np.array_equal(direct, bases[0]):
        bases.append(direct)
    best = None
    # Keep repair intentionally small. The final native exact measurement,
    # not assumed scale monotonicity, decides feasibility.
    coarse_scales = [1.0]
    coarse_scales.extend(
        float(value)
        for value in np.arange(1.001, float(maximum_scale) + 0.0005, 0.001)
    )
    for base in bases:
        best_scale_for_base = None
        best_score_for_base = None
        for scale in coarse_scales:
            candidate = base if scale == 1.0 else _scaled(base, scale)
            if any(has_self_intersection(polygon) for polygon in candidate):
                continue
            recall, iou = _metrics(endpoint_evaluator, frame_index, candidate)
            if recall + _EPSILON < float(recall_floor):
                continue
            score = (float(iou), -float(scale))
            if best_score_for_base is None or score > best_score_for_base:
                best_score_for_base = score
                best_scale_for_base = float(scale)
            if best is None or score > best[0]:
                best = (score, candidate.copy(), recall, iou, float(scale))
        if best_scale_for_base is None:
            continue
        lower = max(1.0, best_scale_for_base - 0.001)
        upper = min(float(maximum_scale), best_scale_for_base + 0.001)
        for scale in np.arange(lower, upper + 0.00005, 0.0001):
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
    maximum_scale: float = 1.05,
) -> tuple[np.ndarray, int, float]:
    output = np.asarray(anchors, dtype=np.float32).copy()
    repaired = 0
    repair_limit = float(maximum_scale)
    maximum_applied_scale = 1.0
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
            maximum_scale=repair_limit,
        )
        if candidate is None or repaired_recall + _EPSILON < float(recall_floor):
            continue
        output[frame_index] = candidate
        repaired += 1
        maximum_applied_scale = max(maximum_applied_scale, float(scale))
    return output, int(repaired), float(maximum_applied_scale)


def apply_spatial_candidate(
    run,
    profile: dict[str, float | int],
    endpoint_evaluator=None,
    config: Polygon14CandidateConfig = CANDIDATE,
) -> None:
    """Build fixed 14-point anchors and apply a best-effort exact Recall repair."""
    if bool(getattr(run, "_polygon14_candidate_applied", False)):
        return
    if endpoint_evaluator is None:
        raise RuntimeError("fixed-14 Recall repair requires native exact evaluator")
    vertices = int(config.vertices_per_component)
    if vertices != 14:
        raise RuntimeError(f"unexpected fixed vertex contract: {vertices}")
    # run.gt_polygons deliberately remains untouched. Both DP edge feasibility
    # and pair-vote use it as the exact source-mask reference.
    started = time.perf_counter()
    anchors, stats = build_spatial_track(run.gt_polygons, config)
    anchors, exact_repaired, repair_scale = _repair_sequence_exact_recall(
        run,
        endpoint_evaluator,
        anchors,
        recall_floor=float(config.spatial_recall_floor),
        maximum_scale=float(config.spatial_recall_repair_max_scale),
    )
    minimum_recall = 1.0
    minimum_iou = 1.0
    unresolved_recall_frames = 0
    for frame_index in range(int(anchors.shape[0])):
        values = endpoint_evaluator.exact_frame_metrics(
            int(frame_index),
            np.ascontiguousarray(anchors[frame_index].reshape(-1), dtype=np.float32),
            int(stats.components),
            vertices,
        )
        recall = float(values[4])
        minimum_recall = min(minimum_recall, recall)
        minimum_iou = min(minimum_iou, float(values[6]))
        unresolved_recall_frames += int(
            recall + _EPSILON < float(config.spatial_recall_floor)
        )
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
    profile["polygon14_exact_repaired_frames"] = int(
        profile.get("polygon14_exact_repaired_frames", 0)
    ) + int(exact_repaired)
    profile["polygon14_exact_repair_maximum_scale"] = max(
        float(profile.get("polygon14_exact_repair_maximum_scale", 1.0)),
        float(repair_scale),
    )
    profile["polygon14_unresolved_recall_frames"] = int(
        profile.get("polygon14_unresolved_recall_frames", 0)
    ) + int(unresolved_recall_frames)
    profile["polygon14_spatial_minimum_recall"] = min(
        float(profile.get("polygon14_spatial_minimum_recall", 1.0)),
        float(minimum_recall),
    )
    profile["polygon14_spatial_minimum_iou"] = min(
        float(profile.get("polygon14_spatial_minimum_iou", 1.0)),
        float(minimum_iou),
    )
