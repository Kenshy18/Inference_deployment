#!/usr/bin/env python3
"""Compare fixed-key-budget polygon optimizers on one V3 SQLite pair."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .fixed_budget import (
    adaptive_add_recall_keys,
    adaptive_split_recall_keys,
    blend_keys_toward_raw,
    evaluate_segments,
    floor_area_positions,
    load_raw_masks,
    load_segments,
    lexicographic_recall_stability_optimize,
    minimax_recall_positions,
    pair_vote_refine,
    projected_temporal_smooth,
    refine_to_key_budget,
    raw_anchor_at_existing_positions,
    repair_interval_recall_with_scale,
    scale_all_keys,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.90)
    parser.add_argument(
        "--repair-max-scale",
        type=float,
        default=1.25,
        help=(
            "Maximum linear scale allowed for interval recall repair. "
            "Smaller values preserve area but may require more adaptive keys."
        ),
    )
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
    if not raw:
        raise SystemExit("no raw masks matched the requested label and frame range")
    baseline = load_segments(
        args.baseline_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    variants = {"production_baseline": baseline}
    variants["raw_anchor_same_positions"] = raw_anchor_at_existing_positions(
        baseline, raw
    )
    for weight in (0.25, 0.50, 0.75):
        variants[f"production_anchor_blend_{weight:.2f}"] = blend_keys_toward_raw(
            baseline, raw, raw_weight=weight
        )

    # Keep the production timing/shape as the prior, move each anchor only
    # halfway back to its observation, and add keys solely where dense
    # reconstructed recall still violates the requested floor.
    conservative_anchor = blend_keys_toward_raw(baseline, raw, raw_weight=0.50)
    variants[
        f"adaptive_anchor_0.50_scale_{args.repair_max_scale:.2f}"
    ] = adaptive_add_recall_keys(
        conservative_anchor,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        repair_margin=0.01,
        max_scale=args.repair_max_scale,
    )
    adaptive_split = adaptive_split_recall_keys(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        anchor_margin=0.08,
    )
    variants["adaptive_split_no_interval_target"] = adaptive_split
    baseline_key_count = sum(
        args.start_frame <= keyframe.frame <= args.end_frame
        for values in baseline.values()
        for segment in values
        for keyframe in segment.keyframes
    )
    variants["adaptive_split_same_production_budget"] = refine_to_key_budget(
        adaptive_split,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_key_count=baseline_key_count,
        anchor_recall_floor=min(0.999, args.recall_floor + 0.07),
    )
    variants["adaptive_split_projected_smooth"] = projected_temporal_smooth(
        adaptive_split,
        raw,
        key_recall_floor=min(0.999, args.recall_floor + 0.05),
        strength=0.50,
        iterations=2,
    )
    variants[
        "lexicographic_recall_stability"
    ] = lexicographic_recall_stability_optimize(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )

    minimax = minimax_recall_positions(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    variants["minimax_raw_fixed_budget"] = minimax
    floor_area = floor_area_positions(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=min(0.999, args.recall_floor + 0.01),
    )
    variants["floor_area_raw_fixed_budget"] = floor_area
    for vote_weight in (0.15, 0.30, 0.50):
        variants[f"minimax_pair_vote_{vote_weight:.2f}"] = pair_vote_refine(
            minimax, raw, vote_weight=vote_weight
        )
    for scale in (1.02, 1.04, 1.06, 1.08, 1.10):
        variants[f"minimax_global_scale_{scale:.2f}"] = scale_all_keys(minimax, scale)
    variants[
        f"minimax_interval_floor_{args.recall_floor:.2f}"
    ] = repair_interval_recall_with_scale(
        minimax,
        raw,
        recall_floor=args.recall_floor,
    )
    # Exact polygon operations and independently scaled shared endpoints can
    # leave a small numerical undershoot.  A one-point internal guard margin
    # makes the requested external floor an actual pass/fail guarantee.
    guarded_floor = min(0.999, args.recall_floor + 0.01)
    variants[
        f"minimax_interval_guard_{args.recall_floor:.2f}"
    ] = repair_interval_recall_with_scale(
        minimax,
        raw,
        recall_floor=guarded_floor,
        max_scale=1.50,
        binary_steps=10,
    )
    variants[
        f"floor_area_interval_guard_{args.recall_floor:.2f}"
    ] = repair_interval_recall_with_scale(
        floor_area,
        raw,
        recall_floor=guarded_floor,
        max_scale=1.50,
        binary_steps=10,
    )
    for raw_weight in (0.50, 0.75, 1.00):
        anchored = blend_keys_toward_raw(baseline, raw, raw_weight=raw_weight)
        variants[
            f"anchored_blend_{raw_weight:.2f}_guard_{args.recall_floor:.2f}"
        ] = repair_interval_recall_with_scale(
            anchored,
            raw,
            recall_floor=guarded_floor,
            max_scale=1.50,
            binary_steps=10,
        )
    variants[
        f"production_interval_floor_{args.recall_floor:.2f}"
    ] = repair_interval_recall_with_scale(
        baseline,
        raw,
        recall_floor=args.recall_floor,
    )

    summaries = {}
    frame_rows = []
    for name, segments in variants.items():
        started = time.perf_counter()
        evaluations = evaluate_segments(raw, segments)
        if not evaluations:
            raise RuntimeError(
                f"variant {name!r} produced no comparable frame geometry"
            )
        summary = summarize(
            evaluations,
            segments,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        summary["evaluation_seconds"] = time.perf_counter() - started
        summaries[name] = summary
        for item in evaluations:
            frame_rows.append(
                {
                    "variant": name,
                    "frame": item.frame,
                    "track_id": item.track_id,
                    "segment_id": item.segment_id,
                    "is_keyframe": int(item.is_keyframe),
                    "recall": item.recall,
                    "precision": item.precision,
                    "iou": item.iou,
                    "area_ratio": item.area_ratio,
                    "excess_area_ratio": item.excess_area_ratio,
                    "centroid_error_px": item.centroid_error_px,
                }
            )
        print(
            f"{name}: keys={summary['keyframe_count']} "
            f"min_recall={summary['recall_min']:.4f} "
            f"below90={summary['recall_below_090']} "
            f"iou={summary['iou_mean']:.4f} "
            f"precision={summary['precision_mean']:.4f}",
            flush=True,
        )

    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "recall_floor": args.recall_floor,
        "repair_max_scale": args.repair_max_scale,
        "variants": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    keyframe_root = args.output_dir / "keyframes"
    keyframe_root.mkdir(parents=True, exist_ok=True)
    for name, segments in variants.items():
        rows = []
        for track_id, track_segments in segments.items():
            for segment in track_segments:
                for keyframe in segment.keyframes:
                    if not (args.start_frame <= keyframe.frame <= args.end_frame):
                        continue
                    components = []
                    for slot, component in keyframe.components:
                        components.append(
                            {
                                "slot": slot,
                                "kind": component.kind,
                                "values": component.values,
                            }
                        )
                    rows.append(
                        {
                            "track_id": track_id,
                            "segment_id": segment.segment_id,
                            "frame": keyframe.frame,
                            "components": components,
                        }
                    )
        (keyframe_root / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
