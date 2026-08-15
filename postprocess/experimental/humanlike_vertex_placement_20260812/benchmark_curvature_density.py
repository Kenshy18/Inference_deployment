#!/usr/bin/env python3
"""Sweep fast curvature-density polygon placement."""

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

from humanlike_vertex_placement_20260812.curvature_density import (  # noqa: E402
    curvature_saliency,
    sample_density,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--counts", default="12,14,17,20,24")
    parser.add_argument("--dense-vertices", type=int, default=256)
    parser.add_argument("--weights", default="0.5,1,2,4,8")
    parser.add_argument("--powers", default="1,2")
    parser.add_argument("--spatial-radii", default="1,2,4")
    parser.add_argument("--temporal-windows", default="1,5")
    args = parser.parse_args()
    frames, track_id, label, polygons, _cuts = load_single_component_track(
        args.input_sqlite, args.track_id, args.max_frames
    )
    evaluator = RasterSequenceEvaluator(polygons)
    setup_started = time.perf_counter()
    dense = align_temporal_dense(polygons, args.dense_vertices)
    dense_seconds = time.perf_counter() - setup_started
    rows = []
    best = None
    saliency_cache = {}
    for radius in [int(value) for value in args.spatial_radii.split(",")]:
        for window in [int(value) for value in args.temporal_windows.split(",")]:
            saliency_cache[(radius, window)] = curvature_saliency(
                dense, spatial_radius=radius, temporal_window=window
            )
    for count in [int(value) for value in args.counts.split(",")]:
        for weight in [float(value) for value in args.weights.split(",")]:
            for power in [float(value) for value in args.powers.split(",")]:
                for (radius, window), saliency in saliency_cache.items():
                    started = time.perf_counter()
                    sequence = sample_density(
                        dense,
                        saliency,
                        count,
                        curvature_weight=weight,
                        curvature_power=power,
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
                        "vertices": count,
                        "curvature_weight": weight,
                        "curvature_power": power,
                        "spatial_radius": radius,
                        "temporal_window": window,
                        "sample_seconds": seconds,
                        "sample_fps": len(frames) / max(seconds, 1e-9),
                        "metrics": asdict(metrics),
                    }
                    rows.append(row)
                    feasible = metrics.minimum_recall >= 0.97
                    key = (
                        0 if feasible else 1,
                        count,
                        -metrics.q01_iou,
                        -metrics.mean_iou,
                        metrics.temporal_residual,
                    )
                    if best is None or key < best[0]:
                        best = (key, row, sequence)
                    print(
                        count,
                        weight,
                        power,
                        radius,
                        window,
                        f"mean={metrics.mean_iou:.6f}",
                        f"q01={metrics.q01_iou:.6f}",
                        f"recall={metrics.minimum_recall:.6f}",
                        f"temporal={metrics.temporal_residual:.6f}",
                        f"sample_fps={row['sample_fps']:.1f}",
                    )
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
                "dense_setup_seconds": dense_seconds,
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
