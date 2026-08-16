#!/usr/bin/env python3
"""Run the Production-preserving, editor-contract polygon Pareto solver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shapely.geometry import box

from .fixed_budget import evaluate_segments, load_raw_masks, load_segments, summarize
from .pareto_dp import optimize_pareto_frontier
from .run_pareto import _filter_segments, _keyframe_rows
from .sqlite_export import export_selected_sqlite
from .superior import (
    BorderExpansionConfig,
    audit_border_safety,
    build_border_safety_constraints,
    compare_geometry_paths,
    evaluate_direct,
    expand_border_constraints,
    summarize_minimal,
    supported_single_component_segments,
    video_dimensions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--segment-id", type=int)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--target-mean-key-interval", type=float, required=True)
    parser.add_argument(
        "--solver-mode",
        choices=("full", "target-only"),
        default="full",
        help="complete Pareto frontier or the exact requested key-count point only",
    )
    parser.add_argument("--max-edge-span-frames", type=int, default=30)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--max-anchor-scale", type=float, default=1.25)
    parser.add_argument("--anchor-state-count", type=int, default=4)
    parser.add_argument("--anchor-expansion", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--edge-processes", type=int, default=0)
    parser.add_argument(
        "--edge-threads",
        type=int,
        default=0,
        help="use threads for edge work; positive values disable edge processes",
    )
    parser.add_argument(
        "--edge-batch-size",
        type=int,
        default=32,
        help="anchor-pair GEOS batch size; use 1 for the scalar reference path",
    )
    parser.add_argument("--no-border-expansion", action="store_true")
    parser.add_argument("--no-pair-vote-states", action="store_true")
    parser.add_argument("--border-trigger-px", type=float, default=10.0)
    parser.add_argument("--border-expand-ratio", type=float, default=0.10)
    parser.add_argument("--border-min-expand-px", type=float, default=6.0)
    parser.add_argument("--border-max-expand-px", type=float, default=40.0)
    parser.add_argument("--border-influence-px", type=float, default=24.0)
    parser.add_argument(
        "--border-local-recall-floor",
        type=float,
        help="defaults to --recall-floor",
    )
    return parser.parse_args()


def _segment_ids(segments) -> set[int]:
    return {
        int(segment.segment_id) for values in segments.values() for segment in values
    }


def _filter_ids(segments, ids: set[int]):
    return {
        track_id: [segment for segment in values if int(segment.segment_id) in ids]
        for track_id, values in segments.items()
        if any(int(segment.segment_id) in ids for segment in values)
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.recall_floor <= 1.0:
        raise ValueError("recall-floor must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_count = max(1, int(os.cpu_count() or 1))
    # A light 25-process oversubscription (five segments by five edge workers)
    # hides the mixed Python/GEOS stalls best on the 24-core deployment CPU.
    # Explicit CLI values remain available for other hosts.
    workers = (
        args.workers
        if args.workers > 0
        else min(5, max(1, int(round(cpu_count / 5))))
    )
    edge_threads = max(0, int(args.edge_threads))
    edge_processes = (
        1
        if edge_threads > 0
        else (
            args.edge_processes
            if args.edge_processes > 0
            else max(1, int(round(cpu_count / workers)))
        )
    )
    quality_raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    baseline_all = _filter_segments(
        load_segments(
            args.baseline_sqlite,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        ),
        args.track_id,
        args.segment_id,
    )
    if args.track_id is not None:
        quality_raw = {
            identity: value
            for identity, value in quality_raw.items()
            if identity[1] == args.track_id
        }
    if not quality_raw or not baseline_all:
        raise SystemExit("no matching raw masks and Production segments")
    baseline, topology_fallbacks = supported_single_component_segments(baseline_all)
    if not baseline:
        raise RuntimeError("no single-component polygon segment can be optimized")
    selected_tracks = set(baseline)
    quality_raw = {
        identity: value
        for identity, value in quality_raw.items()
        if identity[1] in selected_tracks
    }
    width, height = video_dimensions(args.source_sqlite)
    border_config = BorderExpansionConfig(
        enabled=not args.no_border_expansion,
        trigger_px=args.border_trigger_px,
        expand_ratio=args.border_expand_ratio,
        min_expand_px=args.border_min_expand_px,
        max_expand_px=args.border_max_expand_px,
        influence_px=args.border_influence_px,
    )
    constraints, border_summary = expand_border_constraints(
        quality_raw,
        width=width,
        height=height,
        config=border_config,
    )
    border_constraints, border_safety_summary = build_border_safety_constraints(
        quality_raw,
        constraints,
        width=width,
        height=height,
        config=border_config,
        local_recall_floor=(
            args.recall_floor
            if args.border_local_recall_floor is None
            else args.border_local_recall_floor
        ),
    )
    visible_rectangle = box(0.0, 0.0, float(width), float(height))
    baseline_quality_rows = evaluate_segments(
        quality_raw,
        baseline,
        visible_rectangle=visible_rectangle,
        border_constraints=border_constraints,
    )
    baseline_summary = summarize(
        baseline_quality_rows,
        baseline,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    result = optimize_pareto_frontier(
        baseline,
        constraints,
        quality_masks=quality_raw,
        border_constraints=border_constraints,
        visible_bounds=(0.0, 0.0, float(width), float(height)),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        max_edge_span_frames=args.max_edge_span_frames,
        point_count=args.point_count,
        max_anchor_scale=args.max_anchor_scale,
        anchor_state_count=args.anchor_state_count,
        anchor_expansion=args.anchor_expansion,
        workers=workers,
        edge_threads=max(1, edge_threads),
        edge_processes=edge_processes,
        selection="target_interval_quality_floor",
        target_mean_key_interval=args.target_mean_key_interval,
        minimum_mean_iou=float(baseline_summary["iou_mean"]),
        stored_vertex_contract=True,
        pair_vote_states=not args.no_pair_vote_states,
        edge_batch_size=max(1, args.edge_batch_size),
        solver_mode=args.solver_mode.replace("-", "_"),
    )

    selected_quality_overlay = evaluate_segments(
        quality_raw,
        result.segments,
        visible_rectangle=visible_rectangle,
        border_constraints=border_constraints,
    )
    selected_quality_direct = evaluate_direct(
        quality_raw,
        result.segments,
        visible_rectangle=visible_rectangle,
        border_constraints=border_constraints,
    )
    selected_constraint_overlay = evaluate_segments(
        constraints,
        result.segments,
        visible_rectangle=visible_rectangle,
        border_constraints=border_constraints,
    )
    selected_constraint_direct = evaluate_direct(
        constraints,
        result.segments,
        visible_rectangle=visible_rectangle,
        border_constraints=border_constraints,
    )
    border_safety_audit = audit_border_safety(
        border_constraints, result.segments
    )
    selected_summary = summarize(
        selected_quality_overlay,
        result.segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    direct_quality_summary = summarize_minimal(selected_quality_direct)
    constraint_overlay_summary = summarize_minimal(selected_constraint_overlay)
    constraint_direct_summary = summarize_minimal(selected_constraint_direct)
    path_agreement = compare_geometry_paths(
        selected_quality_overlay, selected_quality_direct
    )

    tolerance = 1e-8
    if constraint_direct_summary["recall_min"] + tolerance < args.recall_floor:
        raise RuntimeError(
            "stored point_index interpolation violated minimum Recall: "
            f"{constraint_direct_summary['recall_min']:.12f}"
        )
    if direct_quality_summary["recall_min"] + tolerance < args.recall_floor:
        raise RuntimeError(
            "stored point_index interpolation violated original-mask minimum "
            f"Recall: {direct_quality_summary['recall_min']:.12f}"
        )
    if constraint_overlay_summary["recall_min"] + tolerance < args.recall_floor:
        raise RuntimeError("Production/Overlay interpolation violated minimum Recall")
    if path_agreement["symmetric_difference_max_area"] > 1e-6:
        raise RuntimeError(
            "stored and aligned interpolation paths differ: "
            f"{path_agreement['symmetric_difference_max_area']:.12g}"
        )
    if not border_safety_audit["passed"]:
        raise RuntimeError(
            "stored point_index interpolation violated border safety: "
            f"{border_safety_audit}"
        )
    if abs(selected_summary["iou_mean"] - result.selected.mean_iou) > 1e-9:
        raise RuntimeError("independent IoU does not match the Pareto objective")
    quality_non_regression = (
        selected_summary["iou_mean"] + tolerance >= baseline_summary["iou_mean"]
    )
    if not quality_non_regression and not border_constraints:
        raise RuntimeError(
            "selected Pareto point has lower mean IoU than Production baseline: "
            f"selected={selected_summary['iou_mean']:.12f}, "
            f"production={baseline_summary['iou_mean']:.12f}"
        )

    export = None
    exported_audit = None
    if args.output_sqlite is not None:
        export = export_selected_sqlite(
            args.baseline_sqlite,
            args.output_sqlite,
            result.segments,
            quality_raw,
            label=args.label,
            target_mean_key_interval=args.target_mean_key_interval,
            recall_floor=args.recall_floor,
            selection_reason="superior_pareto_recall_constrained",
            algorithm="experimental.polygon_recall_optimizer.superior_v1",
        )
        optimized_ids = _segment_ids(result.segments)
        exported_segments = _filter_ids(
            load_segments(
                args.output_sqlite,
                label=args.label,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            ),
            optimized_ids,
        )
        exported_constraint_rows = evaluate_direct(
            constraints,
            exported_segments,
            visible_rectangle=visible_rectangle,
            border_constraints=border_constraints,
        )
        exported_quality_rows = evaluate_direct(
            quality_raw,
            exported_segments,
            visible_rectangle=visible_rectangle,
            border_constraints=border_constraints,
        )
        exported_border_safety = audit_border_safety(
            border_constraints, exported_segments
        )
        exported_audit = {
            "constraint": summarize_minimal(exported_constraint_rows),
            "quality": summarize_minimal(exported_quality_rows),
            "path_agreement": compare_geometry_paths(
                selected_quality_direct, exported_quality_rows
            ),
            "border_safety": exported_border_safety,
        }
        if exported_audit["constraint"]["recall_min"] + tolerance < args.recall_floor:
            raise RuntimeError("exported SQLite violated minimum Recall")
        if exported_audit["path_agreement"]["symmetric_difference_max_area"] > 1e-7:
            raise RuntimeError("exported SQLite changed selected geometry")
        if not exported_border_safety["passed"]:
            raise RuntimeError("exported SQLite violated border safety")

    frontier = [
        {
            "keyframe_count": point.keyframe_count,
            "key_frequency": point.key_frequency,
            "mean_key_interval": point.mean_key_interval,
            "mean_iou": point.mean_iou,
            "min_recall": point.min_recall,
            "local_point_indices": list(point.local_point_indices),
        }
        for point in result.frontier
    ]
    feasible_intervals = [point.mean_key_interval for point in result.frontier]
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "algorithm": (
            "superior_target_only_v1"
            if args.solver_mode == "target-only"
            else "superior_pareto_v1"
        ),
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "recall_floor": args.recall_floor,
        "target_mean_key_interval": args.target_mean_key_interval,
        "target_status": {
            "requested": args.target_mean_key_interval,
            "actual": result.selected.mean_key_interval,
            "absolute_deviation": abs(
                result.selected.mean_key_interval - args.target_mean_key_interval
            ),
            "feasible_range": [min(feasible_intervals), max(feasible_intervals)],
            "recall_floor_relaxed": False,
        },
        "production_features": {
            "border_expansion": border_summary,
            "border_safety_constraints": border_safety_summary,
            "border_safety_audit": border_safety_audit,
            "segment_topology_preserved": True,
            "cut_bounded_segments_inherited": True,
            "unsupported_topology_fallbacks": topology_fallbacks,
        },
        "vertex_contract": {
            "point_count": args.point_count,
            "stored_pairwise_alignment": True,
            "overlay_vs_point_index": path_agreement,
        },
        "optimizer": {
            "seconds": result.elapsed_seconds,
            "worker_count": result.worker_count,
            "edge_processes": edge_processes,
            "edge_threads": max(1, edge_threads),
            "edge_evaluations": result.edge_evaluations,
            "feasible_edges": result.feasible_edges,
            "anchor_state_total": result.anchor_state_total,
            "frontier_size": len(result.frontier),
            "selected_index": result.selected_index,
            "pair_vote_states": not args.no_pair_vote_states,
            "edge_batch_size": max(1, args.edge_batch_size),
            "solver_mode": args.solver_mode,
        },
        "production_baseline": baseline_summary,
        "selected_quality_overlay": selected_summary,
        "selected_quality_point_index": direct_quality_summary,
        "selected_constraint_overlay": constraint_overlay_summary,
        "selected_constraint_point_index": constraint_direct_summary,
        "quality_non_regression": quality_non_regression,
        "quality_tradeoff_reason": (
            None
            if quality_non_regression
            else "hard_border_recall_and_offcanvas_safety"
        ),
        "sqlite_export": export,
        "exported_audit": exported_audit,
        "frontier": frontier,
    }
    (args.output_dir / "superior_pareto_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selected_keyframes.json").write_text(
        json.dumps(
            _keyframe_rows(result.segments, args.start_frame, args.end_frame),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "keys": selected_summary["keyframe_count"],
                "actual_interval": result.selected.mean_key_interval,
                "minimum_recall": constraint_direct_summary["recall_min"],
                "mean_iou": selected_summary["iou_mean"],
                "production_mean_iou": baseline_summary["iou_mean"],
                "seconds": result.elapsed_seconds,
                "output_sqlite": (
                    None if args.output_sqlite is None else str(args.output_sqlite)
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
