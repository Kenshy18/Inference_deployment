#!/usr/bin/env python3
"""Benchmark quality-gated persistent correspondence on real SQLite tracks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

EXPERIMENTAL = Path(__file__).resolve().parents[1]
for value in (EXPERIMENTAL, EXPERIMENTAL / "temporal_vertex_decimation_20260812"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.native_dp import native_temporal_dp_sequence  # noqa: E402
from humanlike_vertex_placement_20260812.persistent_correspondence import (  # noqa: E402
    quality_gated_persistent_correspondence,
)
from temporal_vertex_decimation_20260812.optimizer import (  # noqa: E402
    RasterSequenceEvaluator,
    evaluate_sequence,
)
from temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--windows", default="3,5,9")
    parser.add_argument("--budgets", default="0,0.001,0.002,0.004")
    parser.add_argument("--target-modes", default="fractions,arc_gaps")
    args = parser.parse_args()
    frames, _track, _label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    baseline = native_temporal_dp_sequence(
        polygons,
        args.vertices,
        frame_indices=frames,
        cut_frames=cuts,
        temporal_weight=0.003,
        distance_weight=2.0,
        missing_area_weight=1.0,
    )
    rows = []
    for mode in args.target_modes.split(","):
      for window in [int(value) for value in args.windows.split(",")]:
        for budget in [float(value) for value in args.budgets.split(",")]:
            started = time.perf_counter()
            sequence = quality_gated_persistent_correspondence(
                polygons,
                baseline,
                evaluator,
                temporal_window=window,
                iou_loss_budget=budget,
                target_mode=mode,
            )
            seconds = time.perf_counter() - started
            metrics = evaluate_sequence(
                evaluator,
                sequence,
                initial_vertices=args.vertices,
                temporal_weight=0.05,
                tail_weight=0.20,
                vertex_weight=0.0,
                check_self_intersections=True,
            )
            row = {
                "target_mode": mode,
                "window": window,
                "iou_loss_budget": budget,
                "seconds": seconds,
                "fps": len(frames) / max(seconds, 1e-12),
                "metrics": asdict(metrics),
            }
            rows.append(row)
            print(
                f"track={args.track_id} K={args.vertices} mode={mode} window={window} budget={budget} "
                f"fps={row['fps']:.1f} mean={metrics.mean_iou:.6f} "
                f"min_iou={metrics.minimum_iou:.6f} min_recall={metrics.minimum_recall:.6f} "
                f"temporal={metrics.temporal_residual:.6f}"
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({
            "privacy": "SQLite polygon geometry only; no video pixels opened",
            "track": args.track_id,
            "vertices": args.vertices,
            "frames": len(frames),
            "baseline": asdict(evaluate_sequence(
                evaluator,
                baseline,
                initial_vertices=args.vertices,
                temporal_weight=0.05,
                tail_weight=0.20,
                vertex_weight=0.0,
                check_self_intersections=True,
            )),
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
