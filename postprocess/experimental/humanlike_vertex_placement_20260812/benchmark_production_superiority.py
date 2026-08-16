#!/usr/bin/env python3
"""Compare the new 14--18 vertex method with the real old Production basis.

Only SQLite polygon geometry is read.  Source video frames are never decoded.
Run one track per process so the exact raster evaluator cannot accumulate
native buffers during the full benchmark sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

EXPERIMENTAL = Path(__file__).resolve().parents[1]
for value in (EXPERIMENTAL, EXPERIMENTAL / "temporal_vertex_decimation_20260812"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.compare_production_vertex_budget import (  # noqa: E402
    _interpolated_metrics,
    _keyframe_metrics,
    _movement,
    _production_keyframes,
    _sequence_metrics,
)
from humanlike_vertex_placement_20260812.geometry_audit import (  # noqa: E402
    boundary_reconstruction_metrics,
    tangential_correspondence_metrics,
)
from humanlike_vertex_placement_20260812.quality_repair import (  # noqa: E402
    persistent_line_fit_quality_guarded,
)
from temporal_vertex_decimation_20260812.optimizer import (  # noqa: E402
    RasterSequenceEvaluator,
    align_current_equal_arc,
)
from temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
)


def _method_metrics(evaluator, frames, references, sequence, production_frames):
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    keys = {frame: sequence[frame_to_index[frame]] for frame in production_frames}
    return {
        "per_frame": _sequence_metrics(evaluator, sequence),
        "production_keyframes": _keyframe_metrics(evaluator, frames, keys),
        "production_schedule_interpolated": _interpolated_metrics(evaluator, frames, keys),
        "movement": _movement(keys),
        "tangential": tangential_correspondence_metrics(
            dict(zip(frames, references, strict=True)), keys
        ),
        "boundary": boundary_reconstruction_metrics(references, sequence),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--production-sqlite", type=Path, default=Path("data/12月KPI動画.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertices", default="14,15,16,17,18")
    args = parser.parse_args()
    counts = [int(value) for value in args.vertices.split(",")]
    frames, _track, label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track, -1
    )
    evaluator = RasterSequenceEvaluator(polygons)
    with sqlite3.connect(args.production_sqlite) as connection:
        production_all = _production_keyframes(connection, args.track)
    frame_set = set(frames)
    production = {
        frame: value for frame, value in production_all.items() if frame in frame_set
    }
    production_counts = sorted({len(value) for value in production.values()})
    report = {
        "privacy": "SQLite polygon geometry only; no source video opened",
        "track": str(args.track),
        "label": label,
        "frames": len(frames),
        "old_production_actual": {
            "vertices": production_counts,
            "keyframes": _keyframe_metrics(evaluator, frames, production),
            "interpolated": _interpolated_metrics(evaluator, frames, production),
            "movement": _movement(production),
            "tangential": tangential_correspondence_metrics(
                dict(zip(frames, polygons, strict=True)), production
            ),
        },
        "rows": [],
    }
    for count in counts:
        old_fixed = align_current_equal_arc(polygons, count)
        started = time.perf_counter()
        candidate, repair_stats = persistent_line_fit_quality_guarded(
            polygons,
            count,
            dense_vertices=64,
            coverage_quantile=0.65,
            maximum_intersection_radius=0.2,
            intersection_regularization=0.01,
        )
        seconds = time.perf_counter() - started
        old_metrics = _method_metrics(
            evaluator, frames, polygons, old_fixed, production
        )
        candidate_metrics = _method_metrics(
            evaluator, frames, polygons, candidate, production
        )
        row = {
            "vertices": count,
            "old_production_fixed_count": old_metrics,
            "new_persistent_line_fit": candidate_metrics,
            "new_seconds": seconds,
            "new_fps": len(frames) / max(seconds, 1e-12),
            "quality_repair": {
                "repaired_frames": repair_stats.repaired_frames,
                "fallback_frames": repair_stats.fallback_frames,
                "tested_blends": repair_stats.tested_blends,
            },
        }
        report["rows"].append(row)
        old = old_metrics["per_frame"]
        new = candidate_metrics["per_frame"]
        print(
            f"track={args.track} K={count} fps={row['new_fps']:.1f} "
            f"old(meanIoU={old['mean_iou']:.6f},minR={old['minimum_recall']:.6f}) "
            f"new(meanIoU={new['mean_iou']:.6f},minR={new['minimum_recall']:.6f})",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
