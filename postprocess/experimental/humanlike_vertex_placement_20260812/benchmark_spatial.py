#!/usr/bin/env python3
"""Benchmark corner-aware spatial placement against equal-arc contours."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

EXPERIMENTAL = Path(__file__).resolve().parents[1]
if str(EXPERIMENTAL) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL))
TEMPORAL_EXPERIMENT = EXPERIMENTAL / "temporal_vertex_decimation_20260812"
if str(TEMPORAL_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_EXPERIMENT))

from humanlike_vertex_placement_20260812.spatial import (  # noqa: E402
    approximate_sequence,
    rdp_fixed_count,
    visvalingam_fixed_count,
)
from temporal_vertex_decimation_20260812.optimizer import (  # noqa: E402
    RasterSequenceEvaluator,
    align_temporal_dense,
    evaluate_sequence,
)
from temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
    write_masks_sqlite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--counts", default="8,10,12,14,17,20,24")
    return parser


def main() -> int:
    args = _parser().parse_args()
    frames, track_id, label, polygons, _cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    counts = sorted({int(value) for value in args.counts.split(",")})
    evaluator = RasterSequenceEvaluator(polygons)
    methods = {
        "equal_arc": None,
        "rdp": rdp_fixed_count,
        "visvalingam": visvalingam_fixed_count,
    }
    rows = []
    sequences: dict[tuple[str, int], np.ndarray] = {}
    for count in counts:
        for name, method in methods.items():
            started = time.perf_counter()
            if method is None:
                sequence = align_temporal_dense(polygons, count)
            else:
                sequence = approximate_sequence(polygons, count, method)
            build_seconds = time.perf_counter() - started
            metrics = evaluate_sequence(
                evaluator,
                sequence,
                initial_vertices=max(counts),
                temporal_weight=0.05,
                tail_weight=0.20,
                vertex_weight=0.02,
                check_self_intersections=True,
            )
            sequences[(name, count)] = sequence
            rows.append(
                {
                    "method": name,
                    "vertices": count,
                    "build_seconds": build_seconds,
                    "build_fps": len(frames) / max(build_seconds, 1e-9),
                    "metrics": asdict(metrics),
                }
            )
            print(
                name,
                count,
                f"mean={metrics.mean_iou:.6f}",
                f"min={metrics.minimum_iou:.6f}",
                f"q01={metrics.q01_iou:.6f}",
                f"recall={metrics.minimum_recall:.6f}",
                f"temporal={metrics.temporal_residual:.6f}",
                f"fps={len(frames) / max(build_seconds, 1e-9):.1f}",
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Persist each method's lowest-count Recall-feasible point for inspection.
    for name in methods:
        feasible = [
            row
            for row in rows
            if row["method"] == name
            and float(row["metrics"]["minimum_recall"]) >= 0.97
        ]
        if not feasible:
            continue
        selected = min(feasible, key=lambda value: int(value["vertices"]))
        count = int(selected["vertices"])
        write_masks_sqlite(
            args.output_dir / f"{name}_minimum_recall_feasible_{count}.sqlite",
            frames,
            track_id,
            label,
            sequences[(name, count)],
        )
    report = {
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "input_sqlite": str(args.input_sqlite.resolve()),
        "track_id": track_id,
        "frames": len(frames),
        "rows": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
