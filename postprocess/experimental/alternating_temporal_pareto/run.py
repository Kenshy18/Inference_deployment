#!/usr/bin/env python3
"""Run and profile the Production-independent alternating temporal Pareto."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import resource
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box

from ..polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from ..polygon_recall_optimizer.pareto_dp import optimize_pareto_frontier
from ..polygon_recall_optimizer.sqlite_export import export_selected_sqlite
from ..polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    audit_border_safety,
    compare_geometry_paths,
    evaluate_direct,
    supported_single_component_segments,
    video_dimensions,
)
from .independent_border import build_independent_border_constraints
from .refinement import refine_selected_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--border-recall-floor", type=float, default=0.97)
    parser.add_argument("--target-mean-key-interval", type=float, default=10.0)
    parser.add_argument("--report-targets", default="1,3,5,8,10,15")
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--window-radii", default="2,5,10")
    parser.add_argument("--temporal-recall-quantile", type=float, default=0.90)
    parser.add_argument("--max-anchor-scale", type=float, default=1.25)
    parser.add_argument("--max-edge-span-frames", type=int, default=30)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--edge-processes", type=int, default=5)
    parser.add_argument(
        "--solver-mode", choices=("full", "target-only"), default="full"
    )
    parser.add_argument("--max-problem-frames", type=int, default=96)
    parser.add_argument("--max-selected-refinements", type=int, default=0)
    parser.add_argument("--reference-sqlite", type=Path)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="skip SQLite export for partial-segment smoke tests",
    )
    parser.add_argument("--border-trigger-px", type=float, default=10.0)
    parser.add_argument("--border-expand-ratio", type=float, default=0.10)
    parser.add_argument("--border-min-expand-px", type=float, default=6.0)
    parser.add_argument("--border-max-expand-px", type=float, default=40.0)
    parser.add_argument("--border-influence-px", type=float, default=24.0)
    return parser.parse_args()


def _parse_triple(value: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError("window-radii must contain three positive integers")
    return values


def _profile_call(name: str, output_dir: Path, function, *args, **kwargs):
    profile = cProfile.Profile()
    started = time.perf_counter()
    value = profile.runcall(function, *args, **kwargs)
    elapsed = time.perf_counter() - started
    profile.dump_stats(output_dir / f"{name}.prof")
    with (output_dir / f"{name}_profile.txt").open("w", encoding="utf-8") as stream:
        stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        )
        stats.print_stats(80)
    return value, elapsed


def _segment_ids(segments) -> set[int]:
    return {
        int(segment.segment_id)
        for values in segments.values()
        for segment in values
    }


def _filter_ids(segments, ids: set[int]):
    return {
        track_id: [
            segment for segment in values if int(segment.segment_id) in ids
        ]
        for track_id, values in segments.items()
        if any(int(segment.segment_id) in ids for segment in values)
    }


def _expansion_summary(rows) -> dict[str, object]:
    area = np.asarray([row.area_ratio for row in rows], dtype=np.float64)
    excess = np.asarray([row.excess_area_ratio for row in rows], dtype=np.float64)
    ranked = sorted(
        rows,
        key=lambda row: (row.area_ratio, row.excess_area_ratio),
        reverse=True,
    )
    return {
        "area_ratio_q95": float(np.quantile(area, 0.95)),
        "area_ratio_q99": float(np.quantile(area, 0.99)),
        "area_ratio_max": float(np.max(area)),
        "excess_area_q95": float(np.quantile(excess, 0.95)),
        "excess_area_q99": float(np.quantile(excess, 0.99)),
        "excess_area_max": float(np.max(excess)),
        "frames_area_ratio_over_2": int(np.sum(area > 2.0)),
        "frames_area_ratio_over_3": int(np.sum(area > 3.0)),
        "worst_frames": [
            {
                "frame": int(row.frame),
                "track_id": row.track_id,
                "area_ratio": float(row.area_ratio),
                "excess_area_ratio": float(row.excess_area_ratio),
                "iou": float(row.iou),
                "recall": float(row.recall),
            }
            for row in ranked[:20]
        ],
    }


def _frontier_targets(result, targets: list[float]) -> dict[str, object]:
    return {
        str(target): {
            "keyframe_count": point.keyframe_count,
            "mean_key_interval": point.mean_key_interval,
            "mean_iou": point.mean_iou,
            "min_recall": point.min_recall,
        }
        for target in targets
        for point in [
            min(
                result.frontier,
                key=lambda value: (
                    abs(value.mean_key_interval - target),
                    -value.mean_iou,
                ),
            )
        ]
    }


def _audit_stage(
    name,
    result,
    quality_raw,
    constraints,
    border_constraints,
    visible,
    *,
    start_frame,
    end_frame,
    recall_floor,
):
    quality_direct = evaluate_direct(
        quality_raw,
        result.segments,
        visible_rectangle=visible,
        border_constraints=border_constraints,
    )
    quality_overlay = evaluate_segments(
        quality_raw,
        result.segments,
        visible_rectangle=visible,
        border_constraints=border_constraints,
    )
    constraint_direct = evaluate_direct(
        constraints,
        result.segments,
        visible_rectangle=visible,
        border_constraints=border_constraints,
    )
    quality_summary = summarize(
        quality_direct,
        result.segments,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    constraint_summary = summarize(
        constraint_direct,
        result.segments,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    border = audit_border_safety(border_constraints, result.segments)
    agreement = compare_geometry_paths(quality_direct, quality_overlay)
    if quality_summary["recall_min"] + 1e-8 < recall_floor:
        raise RuntimeError(f"{name} violates original-mask Recall")
    if constraint_summary["recall_min"] + 1e-8 < recall_floor:
        raise RuntimeError(f"{name} violates expanded-mask Recall")
    if not border["passed"]:
        raise RuntimeError(f"{name} violates border safety")
    if agreement["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError(f"{name} editor/Overlay paths differ")
    return {
        "quality": quality_summary,
        "constraint": constraint_summary,
        "border": border,
        "expansion": _expansion_summary(quality_direct),
        "path_agreement": agreement,
    }, quality_direct


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = _parse_triple(args.window_radii)
    targets = [float(item) for item in args.report_targets.split(",")]
    quality_raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    topology_all = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    topology, topology_fallbacks = supported_single_component_segments(topology_all)
    selected_tracks = set(topology)
    quality_raw = {
        identity: raw
        for identity, raw in quality_raw.items()
        if identity[1] in selected_tracks
    }
    width, height = video_dimensions(args.source_sqlite)
    border_config = BorderExpansionConfig(
        enabled=True,
        trigger_px=args.border_trigger_px,
        expand_ratio=args.border_expand_ratio,
        min_expand_px=args.border_min_expand_px,
        max_expand_px=args.border_max_expand_px,
        influence_px=args.border_influence_px,
    )
    constraints, border_constraints, border_preparation = (
        build_independent_border_constraints(
            quality_raw,
            width=width,
            height=height,
            config=border_config,
            local_recall_floor=args.border_recall_floor,
        )
    )
    expansion_preparation = {
        "enabled": False,
        "algorithm": "none_raw_mask_is_hard_recall_reference",
        "production_transform_used": False,
    }
    visible = box(0.0, 0.0, float(width), float(height))
    common = dict(
        quality_masks=quality_raw,
        border_constraints=border_constraints,
        visible_bounds=(0.0, 0.0, float(width), float(height)),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        max_edge_span_frames=args.max_edge_span_frames,
        point_count=args.point_count,
        max_anchor_scale=args.max_anchor_scale,
        workers=args.workers,
        edge_processes=args.edge_processes,
        selection="target_interval",
        target_mean_key_interval=args.target_mean_key_interval,
        stored_vertex_contract=True,
        pair_vote_states=False,
        solver_mode=args.solver_mode.replace("-", "_"),
        candidate_mode="temporal7",
        temporal_window_radii=windows,
        temporal_recall_quantile=args.temporal_recall_quantile,
    )

    stage1, stage1_wall = _profile_call(
        "stage1_dp", args.output_dir, optimize_pareto_frontier, topology, constraints, **common
    )
    stage1_audit, stage1_rows = _audit_stage(
        "stage1",
        stage1,
        quality_raw,
        constraints,
        border_constraints,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )
    stage1_sqlite = args.output_dir / "stage1_temporal7.sqlite"
    stage1_export = None
    if not args.skip_export:
        stage1_export = export_selected_sqlite(
            args.source_sqlite,
            stage1_sqlite,
            stage1.segments,
            quality_raw,
            label=args.label,
            target_mean_key_interval=args.target_mean_key_interval,
            recall_floor=args.recall_floor,
            selection_reason="temporal7_stage1",
            algorithm="experimental.alternating_temporal_pareto.stage1",
        )
    checkpoint = {
        "stage": "stage1_complete",
        "stage1": {
            "wall_seconds": stage1_wall,
            "optimizer_seconds": stage1.elapsed_seconds,
            "anchor_states": stage1.anchor_state_total,
            "edge_evaluations": stage1.edge_evaluations,
            "feasible_edges": stage1.feasible_edges,
            "selected": stage1_audit,
            "frontier_targets": _frontier_targets(stage1, targets),
            "sqlite_export": stage1_export,
        },
    }
    (args.output_dir / "checkpoint_stage1.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    refinement, refinement_wall = _profile_call(
        "shape_refinement",
        args.output_dir,
        refine_selected_path,
        stage1.segments,
        quality_raw,
        constraints,
        border_constraints,
        stage1_rows,
        recall_floor=args.recall_floor,
        point_count=args.point_count,
        window_radii=windows,
        recall_quantile=args.temporal_recall_quantile,
        max_anchor_scale=args.max_anchor_scale,
        width=width,
        height=height,
        max_problem_frames=args.max_problem_frames,
        max_selected_keys=args.max_selected_refinements,
    )
    stage2, stage2_wall = _profile_call(
        "stage2_dp",
        args.output_dir,
        optimize_pareto_frontier,
        topology,
        constraints,
        **common,
        extra_anchor_states=refinement.extra_states,
    )
    stage2_audit, _stage2_rows = _audit_stage(
        "stage2",
        stage2,
        quality_raw,
        constraints,
        border_constraints,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )
    stage2_sqlite = args.output_dir / "stage2_alternating_temporal7.sqlite"
    stage2_export = None
    if not args.skip_export:
        stage2_export = export_selected_sqlite(
            args.source_sqlite,
            stage2_sqlite,
            stage2.segments,
            quality_raw,
            label=args.label,
            target_mean_key_interval=args.target_mean_key_interval,
            recall_floor=args.recall_floor,
            selection_reason="alternating_temporal7_stage2",
            algorithm="experimental.alternating_temporal_pareto.stage2",
        )

    reference = None
    if args.reference_sqlite is not None:
        reference_segments = _filter_ids(
            load_segments(
                args.reference_sqlite,
                label=args.label,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            ),
            _segment_ids(stage2.segments),
        )
        reference_rows = evaluate_direct(
            quality_raw,
            reference_segments,
            visible_rectangle=visible,
            border_constraints=border_constraints,
        )
        reference = {
            "path": str(args.reference_sqlite.resolve()),
            "quality": summarize(
                reference_rows,
                reference_segments,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            ),
            "expansion": _expansion_summary(reference_rows),
            "border": audit_border_safety(
                border_constraints, reference_segments
            ),
        }
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "algorithm": "alternating_temporal7_pareto_v1",
        "production_shape_dependency": False,
        "topology_only_source": str(args.source_sqlite.resolve()),
        "configuration": {
            "initial_shape_count": 7,
            "shape_names": [
                "initial_raw",
                "short_iou",
                "short_recall",
                "medium_iou",
                "medium_recall",
                "long_iou",
                "long_recall",
            ],
            "window_radii": windows,
            "temporal_recall_quantile": args.temporal_recall_quantile,
            "recall_floor": args.recall_floor,
            "border_recall_floor": args.border_recall_floor,
            "target_mean_key_interval": args.target_mean_key_interval,
            "target_is_selection_preference": args.solver_mode == "full",
            "point_count": args.point_count,
            "workers": args.workers,
            "edge_processes": args.edge_processes,
        },
        "data": {
            "raw_observations": len(quality_raw),
            "segments": sum(len(values) for values in topology.values()),
            "topology_fallbacks": topology_fallbacks,
            "width": width,
            "height": height,
        },
        "stage1": {
            "wall_seconds": stage1_wall,
            "optimizer_seconds": stage1.elapsed_seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "anchor_states": stage1.anchor_state_total,
            "edge_evaluations": stage1.edge_evaluations,
            "feasible_edges": stage1.feasible_edges,
            "selected": stage1_audit,
            "frontier_targets": _frontier_targets(stage1, targets),
            "sqlite_export": stage1_export,
        },
        "shape_refinement": {
            "wall_seconds": refinement_wall,
            "internal_seconds": refinement.elapsed_seconds,
            "selected_key_targets": refinement.selected_key_targets,
            "problem_frame_targets": refinement.problem_frame_targets,
            "optimized_targets": refinement.optimized_targets,
            "objective_evaluations": refinement.objective_evaluations,
            "accepted_states": refinement.accepted_states,
            "baseline_loss_sum": refinement.baseline_loss_sum,
            "refined_loss_sum": refinement.refined_loss_sum,
            "records": refinement.records,
        },
        "stage2": {
            "wall_seconds": stage2_wall,
            "optimizer_seconds": stage2.elapsed_seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "anchor_states": stage2.anchor_state_total,
            "edge_evaluations": stage2.edge_evaluations,
            "feasible_edges": stage2.feasible_edges,
            "selected": stage2_audit,
            "frontier_targets": _frontier_targets(stage2, targets),
            "sqlite_export": stage2_export,
        },
        "reference_new_pareto": reference,
        "border_preparation": border_preparation,
        "expansion_preparation": expansion_preparation,
        "artifacts": {
            "stage1_sqlite": (
                None if args.skip_export else str(stage1_sqlite.resolve())
            ),
            "stage2_sqlite": (
                None if args.skip_export else str(stage2_sqlite.resolve())
            ),
            "profiles": [
                "stage1_dp.prof",
                "stage1_dp_profile.txt",
                "shape_refinement.prof",
                "shape_refinement_profile.txt",
                "stage2_dp.prof",
                "stage2_dp_profile.txt",
            ],
        },
    }
    report_path = args.output_dir / "alternating_temporal_pareto_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage1": stage1_audit["quality"],
                "refined_states": refinement.accepted_states,
                "stage2": stage2_audit["quality"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
