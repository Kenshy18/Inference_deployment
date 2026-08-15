"""Track-independent optimization against asymmetric temporal references."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from .fixed_budget import FrameEvaluation, RawMask, Segment, evaluate_segments
from .pareto_dp import (
    LocalParetoPoint,
    _SegmentTask,
    _optimize_segment_task,
)
from .temporal_consensus import TemporalConsensusResult


@dataclass(frozen=True)
class IndependentSegmentSelection:
    track_id: str
    segment_id: int
    source_frame_count: int
    frontier_size: int
    selected_point_index: int
    keyframe_count: int
    mean_key_interval: float
    min_recall: float
    min_iou: float
    mean_iou: float
    quality_sum: float
    target_deviation: float


@dataclass(frozen=True)
class TrustedOptimizationResult:
    segments: dict[str, list[Segment]]
    selections: tuple[IndependentSegmentSelection, ...]
    frontiers: dict[tuple[str, int], tuple[LocalParetoPoint, ...]]
    edge_evaluations: int
    feasible_edges: int
    anchor_state_total: int
    elapsed_seconds: float


def _point_interval(point: LocalParetoPoint) -> float:
    if len(point.keyframes) < 2:
        return math.inf
    return (point.keyframes[-1].frame - point.keyframes[0].frame) / (
        len(point.keyframes) - 1
    )


def _select_local_point(
    frontier: list[LocalParetoPoint],
    *,
    target_mean_key_interval: float,
) -> int:
    """Select independently with the requested interval as an effort target.

    The target defines the starting key budget, not a reason to select a bad
    minimum-key endpoint.  From that budget onward, choose the knee between
    extra keys and robust quality.  This captures large early quality gains
    while avoiding the trivial all-frames-as-keys solution.
    """

    span = max(
        frontier[-1].keyframes[-1].frame - frontier[-1].keyframes[0].frame,
        0,
    )
    requested_keys = max(2, int(math.ceil(span / target_mean_key_interval)) + 1)
    start_index = next(
        (
            index
            for index, point in enumerate(frontier)
            if point.keyframe_count >= requested_keys
        ),
        len(frontier) - 1,
    )
    candidates = frontier[start_index:]
    if len(candidates) <= 1:
        return start_index
    window = min(5, max(1, (len(candidates) - 1) // 3))

    def gain_per_key(left: int, right: int) -> float:
        key_delta = max(
            candidates[right].keyframe_count - candidates[left].keyframe_count,
            1,
        )
        return (
            candidates[right].quality_sum - candidates[left].quality_sum
        ) / key_delta

    initial_gain = gain_per_key(0, window)
    if initial_gain <= 1e-12:
        return start_index
    # Keep adding keys while their rolling robust-quality gain retains at
    # least 35% of the first post-target gain. This local diminishing-return
    # rule is stable across long and short tracks, unlike a globally normalized
    # knee that drifted to 319 keys on a 971-frame segment.
    minimum_retained_gain = 0.35 * initial_gain
    local_index = window
    while local_index + window < len(candidates):
        if gain_per_key(local_index, local_index + window) < minimum_retained_gain:
            # A mean utility curve can flatten immediately before a discrete
            # repair of the worst frame. Look ahead a few keys and do not stop
            # one key before a material lower-tail jump (the exact failure
            # seen in segment 105 at 54 -> 55 keys).
            current_min = candidates[local_index].min_iou
            lookahead_end = min(len(candidates), local_index + 11)
            jump_index = next(
                (
                    index
                    for index in range(local_index + 1, lookahead_end)
                    if candidates[index].min_iou >= current_min + 0.03
                ),
                None,
            )
            if jump_index is None:
                break
            local_index = jump_index
            continue
        local_index += 1
    return start_index + local_index


def optimize_segments_independently(
    segments: dict[str, list[Segment]],
    trusted: TemporalConsensusResult,
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    target_mean_key_interval: float,
    quality_mode: str = "tail_harmonic",
    max_edge_span_frames: int = 30,
    point_count: int = 23,
    anchor_point_strategy: str = "simplify_budget",
    max_anchor_scale: float = 1.25,
    anchor_state_count: int = 1,
    anchor_expansion: float = 0.0,
    anchor_relative_iou_margin: float | None = 0.15,
    edge_threads: int = 1,
    edge_processes: int = 1,
    workers: int = 1,
) -> TrustedOptimizationResult:
    """Build and select one Pareto front per scene/track segment."""

    started = time.perf_counter()
    selected: dict[tuple[str, int], tuple] = {}
    selections: list[IndependentSegmentSelection] = []
    frontiers: dict[tuple[str, int], tuple[LocalParetoPoint, ...]] = {}
    edge_evaluations = 0
    feasible_edges = 0
    anchor_state_total = 0
    task_metadata: list[tuple[str, Segment]] = []
    tasks: list[_SegmentTask] = []
    for track_id, values in segments.items():
        for segment in values:
            local_raw = {
                identity: raw
                for identity, raw in trusted.trusted_masks.items()
                if identity[1] == track_id
                and segment.first_frame <= identity[0] <= segment.last_frame
                and start_frame <= identity[0] <= end_frame
            }
            if not local_raw:
                continue
            task_metadata.append((track_id, segment))
            tasks.append(
                _SegmentTask(
                    segment=segment,
                    raw_masks=local_raw,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    recall_floor=recall_floor,
                    anchor_iou_floor=0.0,
                    anchor_relative_iou_margin=anchor_relative_iou_margin,
                    frame_iou_floor=0.0,
                    anchor_point_strategy=anchor_point_strategy,
                    max_frame_hausdorff_px=None,
                    max_edge_span_frames=max_edge_span_frames,
                    point_count=point_count,
                    max_anchor_scale=max_anchor_scale,
                    anchor_state_count=anchor_state_count,
                    anchor_expansion=anchor_expansion,
                    edge_threads=edge_threads,
                    edge_processes=edge_processes,
                    dominance_epsilon=1e-10,
                    quality_mode=quality_mode,
                )
            )
    effective_workers = min(max(1, int(workers)), max(len(tasks), 1))
    if effective_workers > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            task_results = list(executor.map(_optimize_segment_task, tasks))
    else:
        task_results = [_optimize_segment_task(task) for task in tasks]

    for (track_id, segment), task_result in zip(task_metadata, task_results):
        frontier, evaluated, feasible, states = task_result
        if not frontier:
            continue
        frontiers[(track_id, segment.segment_id)] = tuple(frontier)
        point_index = _select_local_point(
            frontier,
            target_mean_key_interval=target_mean_key_interval,
        )
        point = frontier[point_index]
        selected[(track_id, segment.segment_id)] = point.keyframes
        interval = _point_interval(point)
        selections.append(
            IndependentSegmentSelection(
                track_id=track_id,
                segment_id=segment.segment_id,
                source_frame_count=point.frame_count,
                frontier_size=len(frontier),
                selected_point_index=point_index,
                keyframe_count=point.keyframe_count,
                mean_key_interval=interval,
                min_recall=point.min_recall,
                min_iou=point.min_iou,
                mean_iou=point.mean_iou,
                quality_sum=point.quality_sum,
                target_deviation=abs(interval - target_mean_key_interval),
            )
        )
        edge_evaluations += evaluated
        feasible_edges += feasible
        anchor_state_total += states

    output = {
        track_id: [
            replace(
                segment,
                keyframes=selected.get(
                    (track_id, segment.segment_id), segment.keyframes
                ),
            )
            for segment in values
        ]
        for track_id, values in segments.items()
    }
    return TrustedOptimizationResult(
        segments=output,
        selections=tuple(selections),
        frontiers=frontiers,
        edge_evaluations=edge_evaluations,
        feasible_edges=feasible_edges,
        anchor_state_total=anchor_state_total,
        elapsed_seconds=time.perf_counter() - started,
    )


def tail_quality_summary(evaluations: list[FrameEvaluation]) -> dict[str, float | int]:
    """Report lower-tail quality explicitly instead of hiding it in a mean."""

    ious = np.asarray([item.iou for item in evaluations], dtype=np.float64)
    recalls = np.asarray([item.recall for item in evaluations], dtype=np.float64)
    precisions = np.asarray([item.precision for item in evaluations], dtype=np.float64)
    area_ratios = np.asarray(
        [item.area_ratio for item in evaluations], dtype=np.float64
    )

    def cvar(values: np.ndarray, fraction: float) -> float:
        if not len(values):
            return 0.0
        count = max(1, int(math.ceil(len(values) * fraction)))
        return float(np.mean(np.partition(values, count - 1)[:count]))

    def q(values: np.ndarray, probability: float) -> float:
        return float(np.quantile(values, probability)) if len(values) else 0.0

    return {
        "frame_count": int(len(evaluations)),
        "recall_min": float(np.min(recalls)) if len(recalls) else 1.0,
        "recall_q01": q(recalls, 0.01),
        "recall_mean": float(np.mean(recalls)) if len(recalls) else 1.0,
        "iou_min": float(np.min(ious)) if len(ious) else 1.0,
        "iou_cvar01": cvar(ious, 0.01),
        "iou_cvar05": cvar(ious, 0.05),
        "iou_q01": q(ious, 0.01),
        "iou_q05": q(ious, 0.05),
        "iou_mean": float(np.mean(ious)) if len(ious) else 1.0,
        "precision_q01": q(precisions, 0.01),
        "precision_mean": float(np.mean(precisions)) if len(precisions) else 1.0,
        "area_ratio_q01": q(area_ratios, 0.01),
        "area_ratio_q99": q(area_ratios, 0.99),
    }


def evaluate_against_both_references(
    original_raw: dict[tuple[int, str], RawMask],
    trusted: TemporalConsensusResult,
    segments: dict[str, list[Segment]],
) -> dict[str, dict[str, float | int]]:
    return {
        "trusted_reference": tail_quality_summary(
            evaluate_segments(trusted.trusted_masks, segments)
        ),
        "raw_observation": tail_quality_summary(
            evaluate_segments(original_raw, segments)
        ),
    }
