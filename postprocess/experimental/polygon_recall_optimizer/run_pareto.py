#!/usr/bin/env python3
"""Build and validate a recall-constrained keyframe/IoU Pareto front."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from .pareto_dp import optimize_pareto_frontier
from .sqlite_export import export_selected_sqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize polygon keyframes over the non-dominated key-count/IoU "
            "front while enforcing a fixed minimum per-frame recall."
        )
    )
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        help=(
            "Optional keyframe-primary V3 SQLite copy containing the selected "
            "solution; the schema is fingerprinted and must remain unchanged"
        ),
    )
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--segment-id", type=int)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument(
        "--anchor-iou-floor",
        type=float,
        default=0.0,
        help="Hard minimum IoU at every selected keyframe shape",
    )
    parser.add_argument(
        "--frame-iou-floor",
        type=float,
        default=0.0,
        help="Hard minimum IoU at every reconstructed raw-observation frame",
    )
    parser.add_argument(
        "--anchor-point-strategy",
        choices=("uniform", "simplify_budget"),
        default="uniform",
        help="How raw keyframe polygons spend their vertex budget",
    )
    parser.add_argument(
        "--max-frame-hausdorff-px",
        type=float,
        help="Hard maximum symmetric polygon boundary deviation in pixels",
    )
    parser.add_argument("--max-edge-span-frames", type=int, default=60)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--max-anchor-scale", type=float, default=1.25)
    parser.add_argument(
        "--anchor-state-count",
        type=int,
        default=1,
        help="Shape states retained at each candidate frame (1 is legacy)",
    )
    parser.add_argument(
        "--anchor-expansion",
        type=float,
        default=0.04,
        help="Maximum mild expansion added to feasible anchor shape states",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Independent segment processes; 0 selects a CPU-aware default",
    )
    parser.add_argument(
        "--edge-threads",
        type=int,
        default=1,
        help="Shared-memory threads for edges inside each long segment",
    )
    parser.add_argument(
        "--edge-processes",
        type=int,
        default=0,
        help="Forked edge processes per segment; 0 uses remaining CPU cores",
    )
    parser.add_argument(
        "--selection",
        choices=(
            "knee",
            "preference",
            "min_keys",
            "max_iou",
            "key_budget",
            "target_frequency",
            "target_interval",
        ),
        default="knee",
    )
    parser.add_argument(
        "--quality-preference",
        type=float,
        default=0.50,
        help="0 favors fewer keys; 1 favors IoU (selection=preference only)",
    )
    parser.add_argument("--key-budget", type=int)
    parser.add_argument(
        "--target-key-frequency",
        type=float,
        help="Soft target keyframes/raw-mask-frames ratio",
    )
    parser.add_argument(
        "--target-mean-key-interval",
        type=float,
        help="Soft target mean number of frames between consecutive keys",
    )
    return parser.parse_args()


def _filter_segments(segments, track_id: str | None, segment_id: int | None):
    output = {}
    for candidate_track, values in segments.items():
        if track_id is not None and candidate_track != track_id:
            continue
        kept = [
            segment
            for segment in values
            if segment_id is None or segment.segment_id == segment_id
        ]
        if kept:
            output[candidate_track] = kept
    return output


def _keyframe_rows(segments, start_frame: int, end_frame: int) -> list[dict]:
    rows = []
    for track_id, values in segments.items():
        for segment in values:
            for keyframe in segment.keyframes:
                if not start_frame <= keyframe.frame <= end_frame:
                    continue
                rows.append(
                    {
                        "track_id": track_id,
                        "segment_id": segment.segment_id,
                        "frame": keyframe.frame,
                        "components": [
                            {
                                "slot": slot,
                                "kind": component.kind,
                                "values": component.values,
                            }
                            for slot, component in keyframe.components
                        ],
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    if args.selection == "target_interval" and args.target_mean_key_interval is None:
        raise SystemExit(
            "--selection target_interval requires --target-mean-key-interval"
        )
    if args.selection == "target_frequency" and args.target_key_frequency is None:
        raise SystemExit("--selection target_frequency requires --target-key-frequency")
    cpu_count = max(1, int(os.cpu_count() or 1))
    workers = (
        int(args.workers) if int(args.workers) > 0 else (2 if cpu_count >= 4 else 1)
    )
    edge_processes = (
        int(args.edge_processes)
        if int(args.edge_processes) > 0
        else max(1, cpu_count // workers)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    baseline = _filter_segments(
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
        raw = {
            identity: mask
            for identity, mask in raw.items()
            if identity[1] == args.track_id
        }
    if not raw or not baseline:
        raise SystemExit("no matching raw masks and track segments")

    result = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        anchor_iou_floor=args.anchor_iou_floor,
        frame_iou_floor=args.frame_iou_floor,
        anchor_point_strategy=args.anchor_point_strategy,
        max_frame_hausdorff_px=args.max_frame_hausdorff_px,
        max_edge_span_frames=args.max_edge_span_frames,
        point_count=args.point_count,
        max_anchor_scale=args.max_anchor_scale,
        anchor_state_count=args.anchor_state_count,
        anchor_expansion=args.anchor_expansion,
        workers=workers,
        edge_threads=args.edge_threads,
        edge_processes=edge_processes,
        selection=args.selection,
        preference=args.quality_preference,
        key_budget=args.key_budget,
        target_key_frequency=args.target_key_frequency,
        target_mean_key_interval=args.target_mean_key_interval,
    )
    baseline_summary = summarize(
        evaluate_segments(raw, baseline),
        baseline,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    selected_summary = summarize(
        evaluate_segments(raw, result.segments),
        result.segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    # This is intentionally an independent reconstruction pass.  A mismatch
    # means the edge objective no longer matches the reader/overlay semantics.
    if selected_summary["recall_min"] + 1e-10 < args.recall_floor:
        raise RuntimeError(
            "independent validation violated recall floor: "
            f"{selected_summary['recall_min']:.12f} < {args.recall_floor:.12f}"
        )
    if abs(selected_summary["iou_mean"] - result.selected.mean_iou) > 1e-10:
        raise RuntimeError("independent IoU does not match the DP objective")

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
    target_status = None
    if args.selection == "target_interval":
        requested = float(args.target_mean_key_interval)
        feasible_min = min(point.mean_key_interval for point in result.frontier)
        feasible_max = max(point.mean_key_interval for point in result.frontier)
        target_status = {
            "metric": "mean_key_interval",
            "requested": requested,
            "actual": result.selected.mean_key_interval,
            "absolute_deviation": abs(result.selected.mean_key_interval - requested),
            "inside_pareto_range": feasible_min <= requested <= feasible_max,
            "pareto_range": [feasible_min, feasible_max],
            "recall_floor_relaxed": False,
        }
    elif args.selection == "target_frequency":
        requested = float(args.target_key_frequency)
        feasible_min = min(point.key_frequency for point in result.frontier)
        feasible_max = max(point.key_frequency for point in result.frontier)
        target_status = {
            "metric": "key_frequency",
            "requested": requested,
            "actual": result.selected.key_frequency,
            "absolute_deviation": abs(result.selected.key_frequency - requested),
            "inside_pareto_range": feasible_min <= requested <= feasible_max,
            "pareto_range": [feasible_min, feasible_max],
            "recall_floor_relaxed": False,
        }
    sqlite_export = None
    if args.output_sqlite is not None:
        sqlite_export = export_selected_sqlite(
            args.baseline_sqlite,
            args.output_sqlite,
            result.segments,
            raw,
            label=args.label,
            target_mean_key_interval=args.target_mean_key_interval,
            recall_floor=args.recall_floor,
        )
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "track_id": args.track_id,
        "segment_id": args.segment_id,
        "recall_floor": args.recall_floor,
        "anchor_iou_floor": args.anchor_iou_floor,
        "frame_iou_floor": args.frame_iou_floor,
        "anchor_point_strategy": args.anchor_point_strategy,
        "max_frame_hausdorff_px": args.max_frame_hausdorff_px,
        "max_edge_span_frames": args.max_edge_span_frames,
        "anchor_state_count": args.anchor_state_count,
        "anchor_expansion": args.anchor_expansion,
        "anchor_state_total": result.anchor_state_total,
        "worker_count": result.worker_count,
        "edge_threads": args.edge_threads,
        "edge_processes": edge_processes,
        "logical_cpu_count": cpu_count,
        "selection": args.selection,
        "quality_preference": args.quality_preference,
        "key_budget": args.key_budget,
        "target_key_frequency": args.target_key_frequency,
        "target_mean_key_interval": args.target_mean_key_interval,
        "target_status": target_status,
        "selected_index": result.selected_index,
        "edge_evaluations": result.edge_evaluations,
        "feasible_edges": result.feasible_edges,
        "optimizer_seconds": result.elapsed_seconds,
        "baseline": baseline_summary,
        "selected": selected_summary,
        "sqlite_export": sqlite_export,
        "frontier": frontier,
    }
    (args.output_dir / "pareto_frontier.json").write_text(
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
        f"front={len(frontier)} selected={result.selected_index} "
        f"keys={selected_summary['keyframe_count']} "
        f"min_recall={selected_summary['recall_min']:.6f} "
        f"mean_iou={selected_summary['iou_mean']:.6f} "
        f"seconds={result.elapsed_seconds:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
