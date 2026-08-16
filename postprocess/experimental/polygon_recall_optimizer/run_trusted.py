#!/usr/bin/env python3
"""Run track-independent keyframe optimization on trusted temporal masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .fixed_budget import evaluate_segments, load_raw_masks, load_segments, summarize
from .sqlite_export import export_selected_sqlite
from .temporal_consensus import build_segment_bounded_temporal_consensus
from .trusted_optimizer import (
    evaluate_against_both_references,
    optimize_segments_independently,
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
    parser.add_argument("--target-mean-key-interval", type=float, default=10.0)
    parser.add_argument(
        "--quality-mode",
        choices=("mean_iou", "log_iou", "tail_harmonic", "tail_boundary"),
        default="tail_harmonic",
    )
    parser.add_argument("--consensus-radius", type=int, default=2)
    parser.add_argument("--support-fraction", type=float, default=0.50)
    parser.add_argument("--max-edge-span-frames", type=int, default=30)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--anchor-state-count", type=int, default=1)
    parser.add_argument("--anchor-expansion", type=float, default=0.0)
    parser.add_argument("--anchor-relative-iou-margin", type=float, default=0.15)
    parser.add_argument("--edge-processes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
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


def main() -> int:
    args = parse_args()
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
    consensus = build_segment_bounded_temporal_consensus(
        raw,
        baseline,
        radius=args.consensus_radius,
        support_fraction=args.support_fraction,
    )
    result = optimize_segments_independently(
        baseline,
        consensus,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        target_mean_key_interval=args.target_mean_key_interval,
        quality_mode=args.quality_mode,
        max_edge_span_frames=args.max_edge_span_frames,
        point_count=args.point_count,
        anchor_state_count=args.anchor_state_count,
        anchor_expansion=args.anchor_expansion,
        anchor_relative_iou_margin=args.anchor_relative_iou_margin,
        edge_processes=args.edge_processes,
        workers=args.workers,
    )
    baseline_dual = evaluate_against_both_references(raw, consensus, baseline)
    selected_dual = evaluate_against_both_references(raw, consensus, result.segments)
    trusted_dense = summarize(
        evaluate_segments(consensus.trusted_masks, result.segments),
        result.segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    if trusted_dense["recall_min"] + 1e-10 < args.recall_floor:
        raise RuntimeError(f"trusted Recall violation: {trusted_dense['recall_min']}")
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
    classification_counts = Counter(
        item.classification for item in consensus.diagnostics.values()
    )
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "frame_range": [args.start_frame, args.end_frame],
        "track_id": args.track_id,
        "segment_id": args.segment_id,
        "recall_floor": args.recall_floor,
        "target_mean_key_interval": args.target_mean_key_interval,
        "quality_mode": args.quality_mode,
        "consensus": {
            "radius": args.consensus_radius,
            "support_fraction": args.support_fraction,
            "classification_counts": dict(sorted(classification_counts.items())),
        },
        "optimizer": {
            "elapsed_seconds": result.elapsed_seconds,
            "edge_evaluations": result.edge_evaluations,
            "feasible_edges": result.feasible_edges,
            "anchor_state_total": result.anchor_state_total,
            "max_edge_span_frames": args.max_edge_span_frames,
            "point_count": args.point_count,
            "anchor_state_count": args.anchor_state_count,
            "anchor_expansion": args.anchor_expansion,
            "anchor_relative_iou_margin": args.anchor_relative_iou_margin,
            "selection_scope": "independent scene_id/track_id segment",
            "segment_workers": args.workers,
        },
        "baseline": baseline_dual,
        "selected": selected_dual,
        "selections": [asdict(item) for item in result.selections],
        "local_frontiers": {
            f"{track_id}:{segment_id}": [
                {
                    "keyframe_count": point.keyframe_count,
                    "mean_key_interval": (
                        (point.keyframes[-1].frame - point.keyframes[0].frame)
                        / max(point.keyframe_count - 1, 1)
                    ),
                    "min_recall": point.min_recall,
                    "min_iou": point.min_iou,
                    "mean_iou": point.mean_iou,
                    "quality_sum": point.quality_sum,
                }
                for point in frontier
            ]
            for (track_id, segment_id), frontier in result.frontiers.items()
        },
        "sqlite_export": sqlite_export,
    }
    (args.output_dir / "trusted_optimization.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostic_rows = [asdict(item) for item in consensus.diagnostics.values()]
    (args.output_dir / "temporal_diagnostics.json").write_text(
        json.dumps(diagnostic_rows, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    selected_rows = [
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
        for track_id, values in result.segments.items()
        for segment in values
        for keyframe in segment.keyframes
        if args.start_frame <= keyframe.frame <= args.end_frame
    ]
    (args.output_dir / "selected_keyframes.json").write_text(
        json.dumps(selected_rows, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"segments={len(result.selections)} "
        f"keys={trusted_dense['keyframe_count']} "
        f"trusted_recall_min={trusted_dense['recall_min']:.6f} "
        f"trusted_iou_mean={trusted_dense['iou_mean']:.6f} "
        f"seconds={result.elapsed_seconds:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
