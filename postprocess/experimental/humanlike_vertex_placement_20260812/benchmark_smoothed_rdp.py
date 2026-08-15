#!/usr/bin/env python3
"""Sweep robust temporal smoothing of spatial RDP vertices."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

EXPERIMENTAL = Path(__file__).resolve().parents[1]
TEMPORAL_EXPERIMENT = EXPERIMENTAL / "temporal_vertex_decimation_20260812"
for value in (EXPERIMENTAL, TEMPORAL_EXPERIMENT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.smoothed_rdp import (  # noqa: E402
    smoothed_rdp_sequence,
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
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--vertices", type=int, default=17)
    parser.add_argument("--windows", default="3,5,9")
    parser.add_argument("--blends", default="0.1,0.25,0.5,0.75")
    parser.add_argument("--minimum-gap-ratios", default="0.02,0.08")
    args = parser.parse_args()
    frames, track_id, label, polygons, _cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    rows = []
    best = None
    for window in [int(value) for value in args.windows.split(",")]:
        for blend in [float(value) for value in args.blends.split(",")]:
            for gap in [float(value) for value in args.minimum_gap_ratios.split(",")]:
                started = time.perf_counter()
                sequence = smoothed_rdp_sequence(
                    polygons,
                    args.vertices,
                    temporal_window=window,
                    smooth_blend=blend,
                    minimum_gap_ratio=gap,
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
                    "blend": blend,
                    "minimum_gap_ratio": gap,
                    "seconds": seconds,
                    "fps": len(frames) / max(seconds, 1e-9),
                    "metrics": asdict(metrics),
                }
                rows.append(row)
                print(
                    window,
                    blend,
                    gap,
                    f"mean={metrics.mean_iou:.6f}",
                    f"min={metrics.minimum_iou:.6f}",
                    f"q01={metrics.q01_iou:.6f}",
                    f"recall={metrics.minimum_recall:.6f}",
                    f"temporal={metrics.temporal_residual:.6f}",
                    f"cross={metrics.self_intersections}",
                    f"fps={row['fps']:.1f}",
                )
                feasible = metrics.minimum_recall >= 0.97
                key = (
                    0 if feasible else 1,
                    -metrics.q01_iou,
                    -metrics.mean_iou,
                    metrics.temporal_residual,
                )
                if best is None or key < best[0]:
                    best = (key, row, sequence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best is not None:
        write_masks_sqlite(
            args.output_dir / "best.sqlite",
            frames,
            track_id,
            label,
            best[2],
        )
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
