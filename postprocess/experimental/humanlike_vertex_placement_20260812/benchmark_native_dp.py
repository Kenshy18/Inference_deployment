#!/usr/bin/env python3
"""Sweep native global spatial/temporal fixed-count simplification."""

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
from temporal_vertex_decimation_20260812.optimizer import RasterSequenceEvaluator, evaluate_sequence  # noqa: E402
from temporal_vertex_decimation_20260812.run_experiment import load_single_component_track, write_masks_sqlite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--vertices", type=int, default=17)
    parser.add_argument("--temporal-weights", default="0,0.01,0.03,0.1,0.3")
    parser.add_argument("--distance-weights", default="0.25,1")
    parser.add_argument("--missing-area-weights", default="2,4,8")
    args = parser.parse_args()
    frames, track_id, label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    rows = []
    best = None
    for temporal in [float(x) for x in args.temporal_weights.split(",")]:
        for distance in [float(x) for x in args.distance_weights.split(",")]:
            for missing in [float(x) for x in args.missing_area_weights.split(",")]:
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
                metrics = evaluate_sequence(
                    evaluator, sequence, initial_vertices=48, temporal_weight=0.05,
                    tail_weight=0.20, vertex_weight=0.02, check_self_intersections=True,
                )
                row = {
                    "temporal_weight": temporal,
                    "distance_weight": distance,
                    "missing_area_weight": missing,
                    "seconds": seconds,
                    "fps": len(frames) / max(seconds, 1e-9),
                    "metrics": asdict(metrics),
                }
                rows.append(row)
                print(temporal, distance, missing,
                      f"mean={metrics.mean_iou:.6f}", f"min={metrics.minimum_iou:.6f}",
                      f"q01={metrics.q01_iou:.6f}", f"recall={metrics.minimum_recall:.6f}",
                      f"temporal={metrics.temporal_residual:.6f}",
                      f"cross={metrics.self_intersections}", f"fps={row['fps']:.1f}")
                feasible = metrics.minimum_recall >= 0.97 and metrics.self_intersections == 0
                key = (0 if feasible else 1, -metrics.q01_iou, -metrics.mean_iou, metrics.temporal_residual)
                if best is None or key < best[0]:
                    best = (key, row, sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best is not None:
        write_masks_sqlite(args.output_dir / "best.sqlite", frames, track_id, label, best[2])
    (args.output_dir / "report.json").write_text(json.dumps({
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "frames": len(frames), "vertices": args.vertices,
        "best": best[1] if best else None, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
