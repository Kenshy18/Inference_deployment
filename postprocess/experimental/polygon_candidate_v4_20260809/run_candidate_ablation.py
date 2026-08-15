#!/usr/bin/env python3
"""Compare additive candidate families on one track/range using exact DP."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from shapely.geometry import box

from experimental.polygon_recall_optimizer.fixed_budget import (
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


DEFAULT_MODES = (
    "legacy",
    "legacy_temporal_short",
    "legacy_temporal_medium",
    "legacy_temporal_long",
    "legacy_temporal_iou",
    "legacy_temporal_recall",
    "legacy_temporal_union",
)


def _point(frontier, target: float):
    return min(
        frontier,
        key=lambda value: (
            abs(
                (value.frame_count - 1) / max(value.keyframe_count - 1, 1)
                - target
            ),
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
        "area_ratio_mean": float(np.mean(area)),
        "area_ratio_q95": float(np.quantile(area, 0.95)),
        "area_ratio_q99": float(np.quantile(area, 0.99)),
        "area_ratio_max": float(np.max(area)),
        "area_ratio_over_2": int(np.sum(area > 2.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--target-interval", type=float, default=5.0)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--edge-processes", type=int, default=5)
    parser.add_argument("--temporal-window-radii", default="2,5,10")
    parser.add_argument("--temporal-recall-quantile", type=float, default=0.90)
    parser.add_argument(
        "--quality-mode",
        choices=("mean_iou", "log_iou", "tail_harmonic", "tail_boundary"),
        default="mean_iou",
    )
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    args = parser.parse_args()
    window_radii = tuple(
        int(item.strip())
        for item in args.temporal_window_radii.split(",")
        if item.strip()
    )
    if len(window_radii) != 3:
        raise ValueError("temporal-window-radii must contain three integers")

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    quality = {
        identity: value
        for identity, value in quality.items()
        if identity[1] == args.track_id
    }
    loaded = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    segment = next(
        value
        for value in loaded[args.track_id]
        if value.first_frame <= args.start_frame
        and value.last_frame >= args.end_frame
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
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "track_id": args.track_id,
        "range": [args.start_frame, args.end_frame],
        "raw_observations": len(quality),
        "target_interval": args.target_interval,
        "recall_floor": 0.97,
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in (item.strip() for item in args.modes.split(",") if item.strip()):
        print(f"START {mode}", flush=True)
        started = time.perf_counter()
        frontier, edge_count, feasible_count, anchor_count = optimize_segment_pareto(
            segment,
            constraints,
            quality_masks=quality,
            border_constraints=borders,
            visible_bounds=(0.0, 0.0, float(width), float(height)),
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            recall_floor=0.97,
            max_edge_span_frames=30,
            point_count=23,
            max_anchor_scale=1.25,
            anchor_state_count=4,
            anchor_expansion=0.30,
            edge_processes=args.edge_processes,
            stored_vertex_contract=True,
            pair_vote_states=True,
            candidate_mode=mode,
            temporal_window_radii=window_radii,
            temporal_recall_quantile=args.temporal_recall_quantile,
            quality_mode=args.quality_mode,
        )
        selected = _point(frontier, args.target_interval)
        reconstructed = replace(
            segment,
            interpolation_method="linear_polygon_index_v1",
            keyframes=canonicalize_selected_path(selected.keyframes),
        )
        rows = evaluate_direct(
            quality,
            {args.track_id: [reconstructed]},
            visible_rectangle=visible,
            border_constraints=borders,
        )
        summary = summarize(
            rows,
            {args.track_id: [reconstructed]},
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        border_audit = audit_border_safety(borders, {args.track_id: [reconstructed]})
        if summary["recall_min"] + 1e-8 < 0.97 or not border_audit["passed"]:
            raise RuntimeError(f"{mode} failed a hard constraint")
        result = {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "anchor_states": anchor_count,
            "edge_evaluations": edge_count,
            "feasible_edges": feasible_count,
            "feasible_edge_ratio": feasible_count / max(edge_count, 1),
            "frontier_size": len(frontier),
            "keyframe_count": selected.keyframe_count,
            "mean_key_interval": (selected.frame_count - 1)
            / max(selected.keyframe_count - 1, 1),
            "mean_iou": selected.mean_iou,
            "min_recall": selected.min_recall,
            "audit": summary,
            "tail": _tail(rows),
            "border": border_audit,
        }
        report["modes"][mode] = result
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"DONE {mode} iou={selected.mean_iou:.9f} "
            f"q01={result['tail']['iou_q01']:.9f} "
            f"seconds={result['wall_seconds']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
