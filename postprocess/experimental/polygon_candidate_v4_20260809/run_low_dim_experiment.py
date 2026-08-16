#!/usr/bin/env python3
"""Run C1 affine and C2 smooth-normal refinement around one exact-DP path."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from shapely.geometry import box

from experimental.polygon_candidate_v4_20260809.low_dim_refinement import (
    refine_path_sequential,
    refine_targets_low_dim,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from experimental.polygon_recall_optimizer.pareto_dp import (
    canonicalize_selected_path,
    optimize_segment_pareto,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    audit_border_safety,
    build_border_safety_constraints,
    evaluate_direct,
    expand_border_constraints,
    video_dimensions,
)


def _select(frontier, target: float):
    return min(
        frontier,
        key=lambda value: (
            abs((value.frame_count - 1) / max(value.keyframe_count - 1, 1) - target),
            -value.mean_iou,
        ),
    )


def _tail(rows) -> dict[str, float | int]:
    iou = np.asarray([row.iou for row in rows], dtype=np.float64)
    area = np.asarray([row.area_ratio for row in rows], dtype=np.float64)
    return {
        "iou_q01": float(np.quantile(iou, 0.01)),
        "iou_q05": float(np.quantile(iou, 0.05)),
        "iou_min": float(np.min(iou)),
        "area_q99": float(np.quantile(area, 0.99)),
        "area_max": float(np.max(area)),
        "area_over_2": int(np.sum(area > 2.0)),
    }


def _stage(
    name,
    segment,
    constraints,
    quality,
    borders,
    visible,
    *,
    width,
    height,
    start_frame,
    end_frame,
    target_interval,
    edge_processes,
    candidate_mode,
    temporal_recall_quantile,
    extra_states=None,
):
    started = time.perf_counter()
    frontier, edges, feasible, anchors = optimize_segment_pareto(
        segment,
        constraints,
        quality_masks=quality,
        border_constraints=borders,
        visible_bounds=(0.0, 0.0, float(width), float(height)),
        start_frame=start_frame,
        end_frame=end_frame,
        recall_floor=0.97,
        max_edge_span_frames=30,
        point_count=23,
        max_anchor_scale=1.25,
        anchor_state_count=4,
        anchor_expansion=0.30,
        edge_processes=edge_processes,
        stored_vertex_contract=True,
        pair_vote_states=True,
        candidate_mode=candidate_mode,
        temporal_recall_quantile=temporal_recall_quantile,
        extra_anchor_states=extra_states,
    )
    point = _select(frontier, target_interval)
    reconstructed = replace(
        segment,
        interpolation_method="linear_polygon_index_v1",
        keyframes=canonicalize_selected_path(point.keyframes),
    )
    rows = evaluate_direct(
        quality,
        {segment.track_id: [reconstructed]},
        visible_rectangle=visible,
        border_constraints=borders,
    )
    overlay_rows = evaluate_segments(
        quality,
        {segment.track_id: [reconstructed]},
        visible_rectangle=visible,
        border_constraints=borders,
    )
    summary = summarize(
        rows,
        {segment.track_id: [reconstructed]},
        start_frame=start_frame,
        end_frame=end_frame,
    )
    border = audit_border_safety(borders, {segment.track_id: [reconstructed]})
    agreement = max(
        (
            float(left.predicted_geometry.symmetric_difference(right.predicted_geometry).area)
            for left, right in zip(rows, overlay_rows, strict=True)
        ),
        default=0.0,
    )
    if summary["recall_min"] + 1e-8 < 0.97 or not border["passed"]:
        raise RuntimeError(f"{name} violates a hard constraint")
    if agreement > 1e-7:
        raise RuntimeError(f"{name} editor/Overlay paths disagree")
    return reconstructed, rows, {
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "anchor_states": anchors,
        "edge_evaluations": edges,
        "feasible_edges": feasible,
        "feasible_edge_ratio": feasible / max(edges, 1),
        "keyframe_count": point.keyframe_count,
        "mean_key_interval": (point.frame_count - 1)
        / max(point.keyframe_count - 1, 1),
        "mean_iou": point.mean_iou,
        "min_recall": point.min_recall,
        "tail": _tail(rows),
        "audit": summary,
        "border": border,
        "path_agreement_max_area": agreement,
    }


def _rank_key_targets_scored(
    segment, rows, limit: int
) -> list[tuple[float, int, str]]:
    by_frame = {row.frame: row for row in rows}
    keys = list(segment.keyframes)
    ranked = []
    for index, key in enumerate(keys):
        left = keys[max(0, index - 1)].frame
        right = keys[min(len(keys) - 1, index + 1)].frame
        local = [by_frame[frame] for frame in range(left, right + 1) if frame in by_frame]
        if not local:
            continue
        losses = np.asarray([1.0 - row.iou for row in local], dtype=np.float64)
        area = np.asarray([row.area_ratio for row in local], dtype=np.float64)
        score = float(np.mean(losses) + np.quantile(losses, 0.8))
        score += 0.10 * max(0.0, float(np.max(area)) - 1.5)
        ranked.append((score, key.frame, segment.track_id))
    ranked.sort(reverse=True)
    return ranked[: max(1, limit)]


def _rank_key_targets(segment, rows, limit: int) -> list[tuple[int, str]]:
    return [
        (frame, track_id)
        for _score, frame, track_id in _rank_key_targets_scored(segment, rows, limit)
    ]


def _rank_problem_targets(segment, rows, limit: int) -> list[tuple[int, str]]:
    if limit <= 0:
        return []
    key_frames = {key.frame for key in segment.keyframes}
    frames = sorted(key_frames)
    ranked = sorted(
        (row for row in rows if row.frame not in key_frames),
        key=lambda row: (row.iou, -row.area_ratio),
    )
    output: list[tuple[int, str]] = []
    used_intervals: set[tuple[int, int]] = set()
    for row in ranked:
        left = max((frame for frame in frames if frame < row.frame), default=frames[0])
        right = min((frame for frame in frames if frame > row.frame), default=frames[-1])
        interval = (left, right)
        if interval in used_intervals:
            continue
        used_intervals.add(interval)
        output.append((row.frame, row.track_id))
        if len(output) >= limit:
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--target-interval", type=float, default=5.0)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--max-problem-targets", type=int, default=12)
    parser.add_argument("--normal-controls", type=int, default=6)
    parser.add_argument(
        "--candidate-mode", default="legacy_temporal_recall"
    )
    parser.add_argument("--temporal-recall-quantile", type=float, default=0.97)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--skip-redp",
        action="store_true",
        help="evaluate only stage1 plus sequential fixed-key refinement",
    )
    parser.add_argument("--edge-processes", type=int, default=5)
    parser.add_argument("--label", default="男性器")
    args = parser.parse_args()

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    quality = {item: raw for item, raw in quality.items() if item[1] == args.track_id}
    loaded = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    segment = next(
        value
        for value in loaded[args.track_id]
        if value.first_frame <= args.start_frame and value.last_frame >= args.end_frame
    )
    width, height = video_dimensions(args.source_sqlite)
    visible = box(0.0, 0.0, float(width), float(height))
    config = BorderExpansionConfig()
    constraints, expansion = expand_border_constraints(
        quality, width=width, height=height, config=config
    )
    borders, border_preparation = build_border_safety_constraints(
        quality,
        constraints,
        width=width,
        height=height,
        config=config,
        local_recall_floor=0.97,
    )
    common = dict(
        width=width,
        height=height,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_interval=args.target_interval,
        edge_processes=args.edge_processes,
        candidate_mode=args.candidate_mode,
        temporal_recall_quantile=args.temporal_recall_quantile,
    )
    stage1_segment, stage1_rows, stage1 = _stage(
        "stage1", segment, constraints, quality, borders, visible, **common
    )
    key_targets = _rank_key_targets(stage1_segment, stage1_rows, args.max_targets)
    problem_targets = _rank_problem_targets(
        stage1_segment, stage1_rows, args.max_problem_targets
    )
    targets = key_targets + problem_targets
    refinement = None
    if not args.skip_redp:
        refinement = refine_targets_low_dim(
            {args.track_id: [stage1_segment]},
            quality,
            constraints,
            borders,
            targets,
            recall_floor=0.97,
            width=width,
            height=height,
            normal_control_count=args.normal_controls,
            rounds=args.rounds,
        )
    stages = {"stage1": stage1}
    sequential = refine_path_sequential(
        {args.track_id: [stage1_segment]},
        quality,
        constraints,
        borders,
        key_targets,
        recall_floor=0.97,
        width=width,
        height=height,
        normal_control_count=args.normal_controls,
        rounds=args.rounds,
    )
    sequential_segment = sequential.segments[args.track_id][0]
    sequential_rows = evaluate_direct(
        quality,
        {args.track_id: [sequential_segment]},
        visible_rectangle=visible,
        border_constraints=borders,
    )
    sequential_overlay = evaluate_segments(
        quality,
        {args.track_id: [sequential_segment]},
        visible_rectangle=visible,
        border_constraints=borders,
    )
    sequential_summary = summarize(
        sequential_rows,
        {args.track_id: [sequential_segment]},
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    sequential_border = audit_border_safety(
        borders, {args.track_id: [sequential_segment]}
    )
    sequential_agreement = max(
        (
            float(left.predicted_geometry.symmetric_difference(right.predicted_geometry).area)
            for left, right in zip(sequential_rows, sequential_overlay, strict=True)
        ),
        default=0.0,
    )
    if sequential_summary["recall_min"] + 1e-8 < 0.97 or not sequential_border["passed"]:
        raise RuntimeError("sequential refinement violates a hard constraint")
    if sequential_agreement > 1e-7:
        raise RuntimeError("sequential editor/Overlay paths disagree")
    stages["sequential_fixed_keys"] = {
        "wall_seconds": sequential.elapsed_seconds,
        "pipeline_seconds_including_stage1": (
            stage1["wall_seconds"] + sequential.elapsed_seconds
        ),
        "accepted_keys": sequential.accepted_keys,
        "keyframe_count": len(sequential_segment.keyframes),
        "mean_key_interval": (args.end_frame - args.start_frame)
        / max(len(sequential_segment.keyframes) - 1, 1),
        "mean_iou": sequential_summary["iou_mean"],
        "min_recall": sequential_summary["recall_min"],
        "tail": _tail(sequential_rows),
        "audit": sequential_summary,
        "border": sequential_border,
        "path_agreement_max_area": sequential_agreement,
    }
    model_maps = (
        {}
        if refinement is None
        else {
            "affine": refinement.states_by_model.get("affine", {}),
            f"normal{args.normal_controls}": refinement.states_by_model.get(
                f"normal{args.normal_controls}", {}
            ),
            "combined": refinement.extra_states,
        }
    )
    for name, extras in (() if args.skip_redp else model_maps.items()):
        if not extras:
            continue
        _segment, _rows, summary = _stage(
            name,
            segment,
            constraints,
            quality,
            borders,
            visible,
            extra_states=extras,
            **common,
        )
        stages[name] = summary
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "track_id": args.track_id,
        "range": [args.start_frame, args.end_frame],
        "configuration": vars(args) | {"source_sqlite": str(args.source_sqlite), "output": str(args.output)},
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "target_frames": {
            "selected_keys": key_targets,
            "problem_insertions": problem_targets,
        },
        "refinement": (
            None
            if refinement is None
            else {
                "elapsed_seconds": refinement.elapsed_seconds,
                "objective_evaluations": refinement.objective_evaluations,
                "accepted_states": refinement.accepted_states,
                "records": refinement.records,
            }
        ),
        "sequential_refinement": {
            "elapsed_seconds": sequential.elapsed_seconds,
            "accepted_keys": sequential.accepted_keys,
            "records": sequential.records,
        },
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: {"iou": value["mean_iou"], "seconds": value["wall_seconds"]} for name, value in stages.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
