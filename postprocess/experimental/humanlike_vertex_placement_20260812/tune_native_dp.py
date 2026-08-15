#!/usr/bin/env python3
"""Tune native DP against both per-frame fit and keyframe interpolation."""

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

from humanlike_vertex_placement_20260812.interpolation_audit import _audit  # noqa: E402
from humanlike_vertex_placement_20260812.native_dp import native_temporal_dp_sequence  # noqa: E402
from temporal_vertex_decimation_20260812.optimizer import RasterSequenceEvaluator, evaluate_sequence  # noqa: E402
from temporal_vertex_decimation_20260812.run_experiment import load_single_component_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--temporal-weights", default="0.001,0.003,0.01,0.03")
    parser.add_argument("--distance-weights", default="2,4")
    parser.add_argument("--missing-area-weights", default="0.5,1")
    args = parser.parse_args()
    frames, _track_id, _label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    rows = []
    for temporal in [float(value) for value in args.temporal_weights.split(",")]:
        for distance in [float(value) for value in args.distance_weights.split(",")]:
            for missing in [float(value) for value in args.missing_area_weights.split(",")]:
                started = time.perf_counter()
                sequence = native_temporal_dp_sequence(
                    polygons,
                    args.vertices,
                    frame_indices=frames,
                    cut_frames=cuts,
                    temporal_weight=temporal,
                    distance_weight=distance,
                    missing_area_weight=missing,
                )
                seconds = time.perf_counter() - started
                fit = evaluate_sequence(
                    evaluator, sequence, initial_vertices=48, temporal_weight=0.05,
                    tail_weight=0.20, vertex_weight=0.02, check_self_intersections=True,
                )
                interpolation3 = _audit(evaluator, frames, cuts, sequence, 3)
                interpolation5 = _audit(evaluator, frames, cuts, sequence, 5)
                row = {
                    "temporal_weight": temporal,
                    "distance_weight": distance,
                    "missing_area_weight": missing,
                    "seconds": seconds,
                    "fps": len(frames) / max(seconds, 1e-9),
                    "fit": asdict(fit),
                    "interpolation_3": interpolation3,
                    "interpolation_5": interpolation5,
                }
                rows.append(row)
                print(
                    temporal, distance, missing,
                    f"fit_mean={fit.mean_iou:.6f}", f"fit_minrec={fit.minimum_recall:.6f}",
                    f"fit_temp={fit.temporal_residual:.6f}",
                    f"i3_mean={interpolation3.get('mean_iou', 0):.6f}",
                    f"i3_q01={interpolation3.get('q01_iou', 0):.6f}",
                    f"fps={row['fps']:.1f}",
                )
    feasible = [
        row for row in rows
        if row["fit"]["minimum_recall"] >= 0.97
        and row["fit"]["self_intersections"] == 0
    ]
    best = max(
        feasible or rows,
        key=lambda row: (
            row["interpolation_3"].get("q01_iou", 0.0),
            row["fit"]["q01_iou"],
            row["fit"]["mean_iou"],
        ),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "frames": len(frames), "vertices": args.vertices,
        "best": best, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST", best["temporal_weight"], best["distance_weight"], best["missing_area_weight"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
