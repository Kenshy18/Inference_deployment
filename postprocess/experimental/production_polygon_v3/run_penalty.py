#!/usr/bin/env python3
"""Prune a dual-safe V3 path with a soft Production-style key penalty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import box

from ..polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
)
from ..polygon_recall_optimizer.sqlite_export import export_selected_sqlite
from ..polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    audit_border_safety,
    build_border_safety_constraints,
    compare_geometry_paths,
    evaluate_direct,
    expand_border_constraints,
    video_dimensions,
)
from .penalty import optimize_production_v3_penalty
from .run import _key_stats, _normal_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--safe-v3-sqlite", type=Path, required=True)
    parser.add_argument("--production-sqlite", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--normal-recall-floor", type=float, default=0.97)
    parser.add_argument("--border-recall-floor", type=float, default=0.97)
    parser.add_argument(
        "--constraint-mode",
        choices=("both", "normal-only", "border-only"),
        default="both",
        help=(
            "Experimental ablation switch. 'both' is the intended V3 mode; "
            "the other values isolate which hard constraint drives key count."
        ),
    )
    parser.add_argument("--target-mean-key-interval", type=float, default=10.0)
    parser.add_argument("--max-edge-span-frames", type=int, default=30)
    parser.add_argument("--penalty-search-steps", type=int, default=36)
    parser.add_argument(
        "--candidate-mode",
        choices=("all-observations", "safe-keys"),
        default="all-observations",
    )
    parser.add_argument("--max-anchor-scale", type=float, default=1.25)
    parser.add_argument("--border-trigger-px", type=float, default=10.0)
    parser.add_argument("--border-expand-ratio", type=float, default=0.10)
    parser.add_argument("--border-min-expand-px", type=float, default=6.0)
    parser.add_argument("--border-max-expand-px", type=float, default=40.0)
    parser.add_argument("--border-influence-px", type=float, default=24.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    safe = load_segments(
        args.safe_v3_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    if not raw or not safe:
        raise RuntimeError("no matching source masks or safe V3 segments")
    production = None
    if args.production_sqlite is not None:
        production = load_segments(
            args.production_sqlite,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    width, height = video_dimensions(args.source_sqlite)
    visible = box(0.0, 0.0, float(width), float(height))
    config = BorderExpansionConfig(
        enabled=True,
        trigger_px=args.border_trigger_px,
        expand_ratio=args.border_expand_ratio,
        min_expand_px=args.border_min_expand_px,
        max_expand_px=args.border_max_expand_px,
        influence_px=args.border_influence_px,
    )
    expanded, expansion_summary = expand_border_constraints(
        raw, width=width, height=height, config=config
    )
    prepared_border, border_summary = build_border_safety_constraints(
        raw,
        expanded,
        width=width,
        height=height,
        config=config,
        local_recall_floor=args.border_recall_floor,
    )
    normal_recall_floor = (
        0.0 if args.constraint_mode == "border-only" else args.normal_recall_floor
    )
    border = {} if args.constraint_mode == "normal-only" else prepared_border
    candidate_segments = (
        production
        if args.candidate_mode == "all-observations" and production is not None
        else safe
    )
    result = optimize_production_v3_penalty(
        candidate_segments,
        raw,
        expanded,
        border,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        normal_recall_floor=normal_recall_floor,
        target_mean_key_interval=args.target_mean_key_interval,
        max_edge_span_frames=args.max_edge_span_frames,
        penalty_search_steps=args.penalty_search_steps,
        candidate_mode=args.candidate_mode.replace("-", "_"),
        max_anchor_scale=args.max_anchor_scale,
    )
    direct = evaluate_direct(raw, result.segments, visible_rectangle=visible)
    overlay = evaluate_segments(raw, result.segments)
    normal = _normal_summary(direct, normal_recall_floor)
    border_audit = audit_border_safety(border, result.segments)
    agreement = compare_geometry_paths(direct, overlay)
    if int(normal["recall_violations"]) != 0:
        raise RuntimeError(f"penalty path violates normal Recall: {normal}")
    if not border_audit["passed"]:
        raise RuntimeError(f"penalty path violates border Recall: {border_audit}")
    if agreement["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError("penalty path differs between editor and Overlay")

    export = export_selected_sqlite(
        args.safe_v3_sqlite,
        args.output_sqlite,
        result.segments,
        raw,
        label=args.label,
        target_mean_key_interval=args.target_mean_key_interval,
        recall_floor=normal_recall_floor,
        selection_reason="production_v3_soft_key_penalty",
        algorithm="experimental.production_polygon_v3.penalty",
    )
    exported = load_segments(
        args.output_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    exported_direct = evaluate_direct(raw, exported, visible_rectangle=visible)
    exported_overlay = evaluate_segments(raw, exported)
    exported_normal = _normal_summary(exported_direct, normal_recall_floor)
    exported_border = audit_border_safety(border, exported)
    exported_agreement = compare_geometry_paths(exported_direct, exported_overlay)
    if int(exported_normal["recall_violations"]) != 0 or not exported_border["passed"]:
        raise RuntimeError("exported penalty SQLite violates a hard Recall constraint")
    if exported_agreement["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError("exported penalty SQLite changed interpolation geometry")

    safe_direct = evaluate_direct(raw, safe, visible_rectangle=visible)
    production_summary = None
    if production is not None:
        production_direct = evaluate_direct(raw, production, visible_rectangle=visible)
        production_summary = {
            **_key_stats(
                production,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            ),
            **_normal_summary(production_direct, normal_recall_floor),
            "border": audit_border_safety(border, production),
        }
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "algorithm": "production_polygon_v3_soft_key_penalty",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "safe_v3_sqlite": str(args.safe_v3_sqlite.resolve()),
        "output_sqlite": str(args.output_sqlite.resolve()),
        "constraints": {
            "mode": args.constraint_mode,
            "normal_minimum_recall": normal_recall_floor,
            "screen_edge_minimum_recall": (
                None
                if args.constraint_mode == "normal-only"
                else args.border_recall_floor
            ),
            "screen_edge_strip_px": args.border_influence_px,
        },
        "selection": {
            "objective": "sum(1 - frame IoU) + lambda * keyframe_count",
            "target_is_soft": True,
            "target_mean_key_interval": args.target_mean_key_interval,
            "target_keyframe_count": result.target_keyframe_count,
            "actual_keyframe_count": result.keyframe_count,
            "actual_mean_key_interval": result.actual_mean_key_interval,
            "selected_lambda": result.selected_penalty,
            "quality_loss": result.quality_loss,
            "candidate_mode": args.candidate_mode,
            "decoded_lambda_optimal_candidates": result.decoded_candidate_count,
        },
        "optimizer": {
            "seconds": result.elapsed_seconds,
            "evaluated_edges": result.evaluated_edges,
            "feasible_edges": result.feasible_edges,
            "max_edge_span_frames": args.max_edge_span_frames,
        },
        "production": production_summary,
        "safe_v3": {
            **_key_stats(
                safe, start_frame=args.start_frame, end_frame=args.end_frame
            ),
            **_normal_summary(safe_direct, normal_recall_floor),
            "border": audit_border_safety(border, safe),
        },
        "penalty_v3": {
            **_key_stats(
                exported, start_frame=args.start_frame, end_frame=args.end_frame
            ),
            **exported_normal,
            "border": exported_border,
        },
        "border_preparation": expansion_summary,
        "border_constraints": {
            **border_summary,
            "active": args.constraint_mode != "normal-only",
        },
        "path_agreement": exported_agreement,
        "sqlite_export": export,
    }
    (args.output_dir / "production_v3_penalty_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target_interval": args.target_mean_key_interval,
                "actual_interval": result.actual_mean_key_interval,
                "safe_keys": len([key for values in safe.values() for segment in values for key in segment.keyframes]),
                "selected_keys": result.keyframe_count,
                "normal_recall_min": exported_normal["recall_min"],
                "border_recall_min": exported_border["minimum_local_recall"],
                "mean_iou": exported_normal["iou_mean"],
                "lambda": result.selected_penalty,
                "seconds": result.elapsed_seconds,
                "output_sqlite": str(args.output_sqlite),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
