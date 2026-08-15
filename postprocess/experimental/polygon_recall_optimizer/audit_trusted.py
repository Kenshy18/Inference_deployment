#!/usr/bin/env python3
"""Compare Production, prior Pareto, and trusted-mask keyframe results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fixed_budget import evaluate_segments, load_raw_masks, load_segments
from .temporal_consensus import build_segment_bounded_temporal_consensus
from .trusted_optimizer import tail_quality_summary


def _summary(raw, segments) -> dict[str, float | int]:
    evaluations = evaluate_segments(raw, segments)
    output = tail_quality_summary(evaluations)
    distances = np.asarray(
        [
            item.raw_geometry.hausdorff_distance(item.predicted_geometry)
            for item in evaluations
        ],
        dtype=np.float64,
    )
    keyframes = [
        keyframe.frame
        for values in segments.values()
        for segment in values
        for keyframe in segment.keyframes
    ]
    gaps = [
        right.frame - left.frame
        for values in segments.values()
        for segment in values
        for left, right in zip(segment.keyframes, segment.keyframes[1:])
    ]
    output.update(
        {
            "keyframe_count": len(keyframes),
            "mean_key_interval": float(np.mean(gaps)) if gaps else 0.0,
            "key_interval_q95": float(np.quantile(gaps, 0.95)) if gaps else 0.0,
            "max_key_interval": int(max(gaps)) if gaps else 0,
            "hausdorff_q95_px": float(np.quantile(distances, 0.95)),
            "hausdorff_q99_px": float(np.quantile(distances, 0.99)),
            "hausdorff_max_px": float(np.max(distances)),
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--mode", action="append", nargs=2, metavar=("NAME", "SQLITE"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mode:
        raise SystemExit("at least one --mode NAME SQLITE is required")
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    modes = {
        name: load_segments(
            Path(path),
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        for name, path in args.mode
    }
    reference_segments = next(iter(modes.values()))
    trusted = build_segment_bounded_temporal_consensus(
        raw,
        reference_segments,
        radius=2,
    )
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "frame_range": [args.start_frame, args.end_frame],
        "modes": {
            name: {
                "trusted_reference": _summary(trusted.trusted_masks, segments),
                "raw_observation": _summary(raw, segments),
            }
            for name, segments in modes.items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["modes"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
