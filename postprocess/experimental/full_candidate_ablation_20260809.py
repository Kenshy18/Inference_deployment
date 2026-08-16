#!/usr/bin/env python3
"""Whole-range candidate ablation for the interval-5 Pareto point.

This is an offline geometry-only diagnostic.  It does not open video pixels and
does not modify production SQLite files or production pipeline code.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

from experimental.polygon_recall_optimizer.fixed_budget import (
    _raw_keyframe,
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.pareto_dp import optimize_pareto_frontier
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    build_border_safety_constraints,
    expand_border_constraints,
    supported_single_component_segments,
    video_dimensions,
)


def _raw_topology(segments, quality, point_count: int):
    output = {}
    for track_id, values in segments.items():
        converted = []
        for segment in values:
            keys = []
            for key in segment.keyframes:
                raw = quality.get((key.frame, track_id))
                keys.append(
                    key if raw is None else _raw_keyframe(raw, point_count=point_count)
                )
            converted.append(replace(segment, keyframes=tuple(keys)))
        output[track_id] = converted
    return output


def _selected_summary(result) -> dict[str, object]:
    point = result.frontier[result.selected_index]
    return {
        "keyframe_count": point.keyframe_count,
        "mean_key_interval": point.mean_key_interval,
        "mean_iou": point.mean_iou,
        "min_recall": point.min_recall,
        "anchor_states": result.anchor_state_total,
        "edge_evaluations": result.edge_evaluations,
        "feasible_edges": result.feasible_edges,
        "feasible_edge_ratio": result.feasible_edges
        / max(result.edge_evaluations, 1),
        "optimizer_seconds": result.elapsed_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--target-interval", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--edge-processes", type=int, default=5)
    parser.add_argument(
        "--variants",
        default=(
            "legacy_full,raw_topology,no_pair,pair_no_expansion,"
            "raw_candidate_historical_border"
        ),
    )
    args = parser.parse_args()

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    all_segments = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    segments, topology_fallbacks = supported_single_component_segments(all_segments)
    quality = {
        identity: raw for identity, raw in quality.items() if identity[1] in segments
    }
    raw_segments = _raw_topology(segments, quality, 23)
    width, height = video_dimensions(args.source_sqlite)
    config = BorderExpansionConfig()
    historical_constraints, expansion_preparation = expand_border_constraints(
        quality, width=width, height=height, config=config
    )
    historical_borders, border_preparation = build_border_safety_constraints(
        quality,
        historical_constraints,
        width=width,
        height=height,
        config=config,
        local_recall_floor=0.97,
    )

    variants = {
        "legacy_full": {
            "segments": segments,
            "constraints": historical_constraints,
            "states": 4,
            "expansion": 0.30,
            "pair": True,
            "meaning": "complete historical candidate set",
        },
        "raw_topology": {
            "segments": raw_segments,
            "constraints": historical_constraints,
            "states": 4,
            "expansion": 0.30,
            "pair": True,
            "meaning": "remove Production keyframe shapes only",
        },
        "no_pair": {
            "segments": segments,
            "constraints": historical_constraints,
            "states": 4,
            "expansion": 0.30,
            "pair": False,
            "meaning": "remove pair-vote endpoint candidates",
        },
        "pair_no_expansion": {
            "segments": segments,
            "constraints": historical_constraints,
            "states": 2,
            "expansion": 0.0,
            "pair": True,
            "meaning": "remove deliberate staged expansion states",
        },
        "raw_candidate_historical_border": {
            "segments": segments,
            "constraints": quality,
            "states": 4,
            "expansion": 0.30,
            "pair": True,
            "meaning": (
                "remove historical directionally border-expanded raw anchor "
                "while retaining historical border audit"
            ),
        },
    }
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(variants))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "range": [args.start_frame, args.end_frame],
        "label": args.label,
        "target_mean_key_interval": args.target_interval,
        "recall_floor": 0.97,
        "segment_count": sum(len(values) for values in segments.values()),
        "raw_observation_count": len(quality),
        "topology_fallbacks": topology_fallbacks,
        "historical_expansion_preparation": expansion_preparation,
        "historical_border_preparation": border_preparation,
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in requested:
        variant = variants[name]
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        try:
            result = optimize_pareto_frontier(
                variant["segments"],
                variant["constraints"],
                quality_masks=quality,
                border_constraints=historical_borders,
                visible_bounds=(0.0, 0.0, float(width), float(height)),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                recall_floor=0.97,
                max_edge_span_frames=30,
                point_count=23,
                max_anchor_scale=1.25,
                anchor_state_count=variant["states"],
                anchor_expansion=variant["expansion"],
                selection="target_interval",
                target_mean_key_interval=args.target_interval,
                workers=args.workers,
                edge_processes=args.edge_processes,
                stored_vertex_contract=True,
                pair_vote_states=variant["pair"],
                solver_mode="target_only",
                candidate_mode="legacy",
            )
        except Exception as error:  # diagnostic must retain failure evidence
            report["variants"][name] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
                "meaning": variant["meaning"],
            }
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"FAILED {name}: {type(error).__name__}: {error}", flush=True
            )
            continue
        summary = _selected_summary(result)
        summary.update(
            {
                "wall_seconds": time.perf_counter() - started,
                "meaning": variant["meaning"],
                "production_key_shapes_available": variant["segments"] is segments,
                "historical_expanded_raw_anchor_available": (
                    variant["constraints"] is historical_constraints
                ),
                "anchor_state_count": variant["states"],
                "anchor_expansion": variant["expansion"],
                "pair_vote_states": variant["pair"],
            }
        )
        report["variants"][name] = summary
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"DONE {name} {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
