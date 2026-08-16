#!/usr/bin/env python3
"""Run and independently audit the Production-based polygon V3 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    supported_single_component_segments,
    video_dimensions,
)
from .optimizer import guard_production_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--normal-recall-floor", type=float, default=0.97)
    parser.add_argument("--border-recall-floor", type=float, default=0.97)
    parser.add_argument(
        "--point-count",
        type=int,
        default=0,
        help="0 preserves each Production segment's original vertex count",
    )
    parser.add_argument("--max-anchor-scale", type=float, default=1.25)
    parser.add_argument("--border-trigger-px", type=float, default=10.0)
    parser.add_argument("--border-expand-ratio", type=float, default=0.10)
    parser.add_argument("--border-min-expand-px", type=float, default=6.0)
    parser.add_argument("--border-max-expand-px", type=float, default=40.0)
    parser.add_argument("--border-influence-px", type=float, default=24.0)
    return parser.parse_args()


def _key_stats(segments, *, start_frame: int, end_frame: int) -> dict[str, float | int]:
    key_count = 0
    total_span = 0
    interval_count = 0
    for values in segments.values():
        for segment in values:
            frames = [
                key.frame
                for key in segment.keyframes
                if start_frame <= key.frame <= end_frame
            ]
            key_count += len(frames)
            if len(frames) >= 2:
                total_span += frames[-1] - frames[0]
                interval_count += len(frames) - 1
    return {
        "keyframe_count": key_count,
        "mean_key_interval": total_span / max(interval_count, 1),
    }


def _normal_summary(rows, floor: float) -> dict[str, float | int]:
    recalls = np.asarray([row.recall for row in rows], dtype=np.float64)
    ious = np.asarray([row.iou for row in rows], dtype=np.float64)
    precisions = np.asarray([row.precision for row in rows], dtype=np.float64)
    area_ratios = np.asarray([row.area_ratio for row in rows], dtype=np.float64)
    return {
        "frame_count": len(rows),
        "recall_min": float(np.min(recalls)) if len(recalls) else 1.0,
        "recall_mean": float(np.mean(recalls)) if len(recalls) else 1.0,
        "recall_q01": float(np.quantile(recalls, 0.01)) if len(recalls) else 1.0,
        "recall_violations": int(np.sum(recalls + 1e-12 < floor)),
        "iou_min": float(np.min(ious)) if len(ious) else 1.0,
        "iou_mean": float(np.mean(ious)) if len(ious) else 1.0,
        "iou_q01": float(np.quantile(ious, 0.01)) if len(ious) else 1.0,
        "precision_mean": float(np.mean(precisions)) if len(precisions) else 1.0,
        "area_ratio_mean": float(np.mean(area_ratios)) if len(area_ratios) else 1.0,
        "area_ratio_q99": float(np.quantile(area_ratios, 0.99)) if len(area_ratios) else 1.0,
    }


def _variant_summary(
    rows,
    segments,
    border_audit,
    *,
    start_frame: int,
    end_frame: int,
    normal_floor: float,
) -> dict[str, object]:
    value: dict[str, object] = _normal_summary(rows, normal_floor)
    value.update(_key_stats(segments, start_frame=start_frame, end_frame=end_frame))
    value["border"] = border_audit
    return value


def main() -> int:
    args = parse_args()
    for name, value in (
        ("normal-recall-floor", args.normal_recall_floor),
        ("border-recall-floor", args.border_recall_floor),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    baseline_all = load_segments(
        args.baseline_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    if not raw or not baseline_all:
        raise RuntimeError("no matching AI masks or Production segments")
    baseline, unsupported = supported_single_component_segments(baseline_all)
    if unsupported:
        raise RuntimeError(
            "Production polygon V3 currently requires one polygon component per "
            f"segment; unsupported={unsupported[:5]}"
        )

    width, height = video_dimensions(args.source_sqlite)
    visible = box(0.0, 0.0, float(width), float(height))
    border_config = BorderExpansionConfig(
        enabled=True,
        trigger_px=args.border_trigger_px,
        expand_ratio=args.border_expand_ratio,
        min_expand_px=args.border_min_expand_px,
        max_expand_px=args.border_max_expand_px,
        influence_px=args.border_influence_px,
    )
    expanded, expansion_summary = expand_border_constraints(
        raw, width=width, height=height, config=border_config
    )
    border_constraints, border_constraint_summary = build_border_safety_constraints(
        raw,
        expanded,
        width=width,
        height=height,
        config=border_config,
        local_recall_floor=args.border_recall_floor,
    )

    baseline_direct = evaluate_direct(raw, baseline, visible_rectangle=visible)
    baseline_border = audit_border_safety(border_constraints, baseline)
    result = guard_production_v3(
        baseline,
        raw,
        expanded,
        border_constraints,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        normal_recall_floor=args.normal_recall_floor,
        point_count=args.point_count,
        max_anchor_scale=args.max_anchor_scale,
    )
    memory_direct = evaluate_direct(raw, result.segments, visible_rectangle=visible)
    memory_overlay = evaluate_segments(raw, result.segments)
    memory_border = audit_border_safety(border_constraints, result.segments)
    path_agreement = compare_geometry_paths(memory_direct, memory_overlay)
    memory_normal = _normal_summary(memory_direct, args.normal_recall_floor)
    if int(memory_normal["recall_violations"]) != 0:
        raise RuntimeError(f"in-memory V3 violates normal Recall: {memory_normal}")
    if not memory_border["passed"]:
        raise RuntimeError(f"in-memory V3 violates border Recall: {memory_border}")
    if path_agreement["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError(
            "editor point-index and Overlay paths differ: "
            f"{path_agreement['symmetric_difference_max_area']}"
        )

    baseline_keys = _key_stats(
        baseline, start_frame=args.start_frame, end_frame=args.end_frame
    )
    export = export_selected_sqlite(
        args.baseline_sqlite,
        args.output_sqlite,
        result.segments,
        raw,
        label=args.label,
        target_mean_key_interval=float(baseline_keys["mean_key_interval"]),
        recall_floor=args.normal_recall_floor,
        selection_reason="production_v3_dual_minimum_recall",
        algorithm="experimental.production_polygon_v3",
    )
    exported = load_segments(
        args.output_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    exported_direct = evaluate_direct(raw, exported, visible_rectangle=visible)
    exported_overlay = evaluate_segments(raw, exported)
    exported_border = audit_border_safety(border_constraints, exported)
    exported_path = compare_geometry_paths(exported_direct, exported_overlay)
    exported_normal = _normal_summary(exported_direct, args.normal_recall_floor)
    if int(exported_normal["recall_violations"]) != 0:
        raise RuntimeError("exported SQLite violates normal minimum Recall")
    if not exported_border["passed"]:
        raise RuntimeError("exported SQLite violates screen-edge minimum Recall")
    if exported_path["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError("exported editor and Overlay interpolation paths differ")

    baseline_summary = _variant_summary(
        baseline_direct,
        baseline,
        baseline_border,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        normal_floor=args.normal_recall_floor,
    )
    v3_summary = _variant_summary(
        exported_direct,
        exported,
        exported_border,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        normal_floor=args.normal_recall_floor,
    )
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "algorithm": "production_polygon_v3_dual_minimum_recall",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "output_sqlite": str(args.output_sqlite.resolve()),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "constraints": {
            "normal": {
                "semantics": "minimum per observed frame",
                "floor": args.normal_recall_floor,
                "reference": "original AI mask",
            },
            "screen_edge": {
                "semantics": "minimum per touched side per observed frame",
                "floor": args.border_recall_floor,
                "strip_width_px": args.border_influence_px,
                "offcanvas_extent_required": True,
            },
        },
        "production_preservation": {
            "all_production_key_positions_retained": True,
            "key_count_is_not_fixed": True,
            "optional_large_expansion_states": False,
            "post_decode_shape_mutation": False,
            "adjusted_production_keys": result.adjusted_production_keys,
            "added_constraint_keys": result.added_constraint_keys,
        },
        "border_preparation": expansion_summary,
        "border_constraints": border_constraint_summary,
        "optimizer": {
            "seconds": result.elapsed_seconds,
            "evaluated_edges": result.evaluated_edges,
            "feasible_edges": result.feasible_edges,
            "selection": "fewest additions, then maximum accumulated raw-mask IoU",
        },
        "production": baseline_summary,
        "v3": v3_summary,
        "path_agreement": exported_path,
        "sqlite_export": export,
        "segments": list(result.segment_diagnostics),
    }
    (args.output_dir / "production_v3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "production_keys": baseline_summary["keyframe_count"],
                "v3_keys": v3_summary["keyframe_count"],
                "normal_recall_min": v3_summary["recall_min"],
                "border_recall_min": v3_summary["border"]["minimum_local_recall"],
                "mean_iou": v3_summary["iou_mean"],
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
