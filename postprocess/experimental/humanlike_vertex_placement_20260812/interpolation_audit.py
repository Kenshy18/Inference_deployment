#!/usr/bin/env python3
"""Measure whether fixed vertex IDs interpolate cleanly between keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

EXPERIMENTAL = Path(__file__).resolve().parents[1]
for value in (EXPERIMENTAL, EXPERIMENTAL / "temporal_vertex_decimation_20260812"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.native_dp import native_temporal_dp_sequence  # noqa: E402
from humanlike_vertex_placement_20260812.spatial import approximate_sequence, rdp_fixed_count  # noqa: E402
from temporal_vertex_decimation_20260812.optimizer import RasterSequenceEvaluator, align_temporal_dense  # noqa: E402
from temporal_vertex_decimation_20260812.run_experiment import load_single_component_track  # noqa: E402


def _audit(evaluator, frames, cuts, sequence, interval):
    ious = []
    recalls = []
    cuts = {int(value) for value in cuts}
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    for first, frame in enumerate(frames):
        end_frame = int(frame) + int(interval)
        last = frame_to_index.get(end_frame)
        if last is None or any(value in cuts for value in range(int(frame) + 1, end_frame + 1)):
            continue
        for current in range(first + 1, last):
            alpha = (int(frames[current]) - int(frame)) / float(interval)
            polygon = (1.0 - alpha) * sequence[first] + alpha * sequence[last]
            iou, recall = evaluator.frame_metrics(current, polygon)
            ious.append(iou)
            recalls.append(recall)
    if not ious:
        return {"samples": 0}
    return {
        "samples": len(ious),
        "mean_iou": float(np.mean(ious)),
        "minimum_iou": float(np.min(ious)),
        "q01_iou": float(np.quantile(ious, 0.01)),
        "q05_iou": float(np.quantile(ious, 0.05)),
        "mean_recall": float(np.mean(recalls)),
        "minimum_recall": float(np.min(recalls)),
        "q01_recall": float(np.quantile(recalls, 0.01)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--vertices", type=int, default=20)
    parser.add_argument("--intervals", default="3,5,8,10")
    args = parser.parse_args()
    frames, _track_id, _label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track_id, -1
    )
    evaluator = RasterSequenceEvaluator(polygons)
    sequences = {
        "equal_arc": align_temporal_dense(polygons, args.vertices),
        "independent_rdp": approximate_sequence(polygons, args.vertices, rdp_fixed_count),
        "native_global_temporal_dp": native_temporal_dp_sequence(
            polygons,
            args.vertices,
            frame_indices=frames,
            cut_frames=cuts,
            temporal_weight=0.003,
            distance_weight=2.0,
            missing_area_weight=1.0,
        ),
    }
    report = {
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "frames": len(frames),
        "vertices": args.vertices,
        "methods": {},
    }
    for name, sequence in sequences.items():
        report["methods"][name] = {
            str(interval): _audit(evaluator, frames, cuts, sequence, interval)
            for interval in [int(value) for value in args.intervals.split(",")]
        }
        print(name, json.dumps(report["methods"][name], ensure_ascii=False))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
