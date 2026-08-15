#!/usr/bin/env python3
"""Sweep quality-gated temporal smoothing."""

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

from humanlike_vertex_placement_20260812.adaptive_smoothing import (  # noqa: E402
    adaptive_smoothed_rdp_sequence,
)
from temporal_vertex_decimation_20260812.optimizer import (  # noqa: E402
    RasterSequenceEvaluator,
    evaluate_sequence,
)
from temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
    write_masks_sqlite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--vertices", type=int, default=17)
    parser.add_argument("--windows", default="3,5,9")
    parser.add_argument("--budgets", default="0.001,0.003,0.006")
    parser.add_argument("--weights", default="0.05,0.12,0.25")
    args = parser.parse_args()
    frames, track_id, label, polygons, _cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    rows = []
    best = None
    for window in [int(value) for value in args.windows.split(",")]:
        for budget in [float(value) for value in args.budgets.split(",")]:
            for weight in [float(value) for value in args.weights.split(",")]:
                started = time.perf_counter()
                sequence = adaptive_smoothed_rdp_sequence(
                    polygons,
                    args.vertices,
                    evaluator,
                    temporal_window=window,
                    iou_loss_budget=budget,
                    temporal_weight=weight,
                )
                seconds = time.perf_counter() - started
                metrics = evaluate_sequence(
                    evaluator,
                    sequence,
                    initial_vertices=48,
                    temporal_weight=0.05,
                    tail_weight=0.20,
                    vertex_weight=0.02,
                    check_self_intersections=True,
                )
                row = {
                    "window": window,
                    "iou_loss_budget": budget,
                    "temporal_weight": weight,
                    "seconds": seconds,
                    "fps": len(frames) / max(seconds, 1e-9),
                    "metrics": asdict(metrics),
                }
                rows.append(row)
                print(
                    window,
                    budget,
                    weight,
                    f"mean={metrics.mean_iou:.6f}",
                    f"min={metrics.minimum_iou:.6f}",
                    f"q01={metrics.q01_iou:.6f}",
                    f"recall={metrics.minimum_recall:.6f}",
                    f"temporal={metrics.temporal_residual:.6f}",
                    f"fps={row['fps']:.1f}",
                )
                feasible = metrics.minimum_recall >= 0.97
                key = (
                    0 if feasible else 1,
                    -metrics.q01_iou,
                    metrics.temporal_residual,
                    -metrics.mean_iou,
                )
                if best is None or key < best[0]:
                    best = (key, row, sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best is not None:
        write_masks_sqlite(args.output_dir / "best.sqlite", frames, track_id, label, best[2])
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "privacy": "SQLite polygon geometry only; no video pixels opened",
                "frames": len(frames),
                "vertices": args.vertices,
                "best": best[1] if best else None,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
