#!/usr/bin/env python3
"""Audit temporal area jumps without opening video pixels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..polygon_recall_optimizer.fixed_budget import load_raw_masks, load_segments
from ..polygon_recall_optimizer.superior import direct_geometry_at


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def audit(
    source_sqlite: Path,
    result_sqlite: Path,
    *,
    label: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, object]:
    raw = load_raw_masks(
        source_sqlite,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    segments = load_segments(
        result_sqlite,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    positive_growth: list[float] = []
    symmetric_change: list[float] = []
    absolute_log_delta: list[float] = []
    local_bulge: list[float] = []
    raw_relative: list[float] = []
    jumps: list[dict[str, object]] = []
    bulges: list[dict[str, object]] = []
    raw_ratios: list[dict[str, object]] = []
    evaluated = 0
    for track_id, track_segments in segments.items():
        for segment in track_segments:
            values: list[tuple[int, float, float]] = []
            for frame in range(segment.first_frame, segment.last_frame + 1):
                observation = raw.get((frame, track_id))
                if observation is None:
                    continue
                predicted_area = float(direct_geometry_at(segment, frame).area)
                raw_area = float(observation.geometry.area)
                if predicted_area <= 0.0 or raw_area <= 0.0:
                    continue
                values.append((frame, predicted_area, raw_area))
                ratio = predicted_area / raw_area
                raw_relative.append(ratio)
                raw_ratios.append(
                    {
                        "frame": int(frame),
                        "track_id": str(track_id),
                        "predicted_area": predicted_area,
                        "raw_area": raw_area,
                        "predicted_to_raw_ratio": ratio,
                    }
                )
                evaluated += 1
            for index in range(1, len(values)):
                frame, area, _raw_area = values[index]
                previous_frame, previous_area, _previous_raw = values[index - 1]
                if frame != previous_frame + 1:
                    continue
                ratio = area / previous_area
                growth = max(1.0, ratio)
                symmetric = max(ratio, 1.0 / ratio)
                log_delta = abs(math.log(ratio))
                positive_growth.append(growth)
                symmetric_change.append(symmetric)
                absolute_log_delta.append(log_delta)
                jumps.append(
                    {
                        "from_frame": int(previous_frame),
                        "to_frame": int(frame),
                        "track_id": str(track_id),
                        "previous_area": previous_area,
                        "current_area": area,
                        "growth_ratio": ratio,
                        "symmetric_change_ratio": symmetric,
                    }
                )
            for index, (frame, area, _raw_area) in enumerate(values):
                neighbors = [
                    values[candidate][1]
                    for candidate in range(
                        max(0, index - 2), min(len(values), index + 3)
                    )
                    if candidate != index
                    and abs(values[candidate][0] - frame) <= 2
                ]
                if not neighbors:
                    continue
                ratio = area / max(float(np.median(neighbors)), 1e-9)
                local_bulge.append(ratio)
                bulges.append(
                    {
                        "frame": int(frame),
                        "track_id": str(track_id),
                        "predicted_area": area,
                        "neighbor_median_area": float(np.median(neighbors)),
                        "local_bulge_ratio": ratio,
                    }
                )
    jumps.sort(key=lambda item: item["symmetric_change_ratio"], reverse=True)
    bulges.sort(key=lambda item: item["local_bulge_ratio"], reverse=True)
    raw_ratios.sort(
        key=lambda item: item["predicted_to_raw_ratio"], reverse=True
    )
    return {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "result_sqlite": str(result_sqlite.resolve()),
        "evaluated_frames": evaluated,
        "adjacent_positive_growth_ratio": _quantiles(positive_growth),
        "adjacent_symmetric_area_change_ratio": _quantiles(symmetric_change),
        "adjacent_absolute_log_area_delta": _quantiles(absolute_log_delta),
        "local_predicted_bulge_ratio": _quantiles(local_bulge),
        "predicted_to_raw_area_ratio": _quantiles(raw_relative),
        "predicted_to_raw_over_2": int(np.sum(np.asarray(raw_relative) > 2.0)),
        "predicted_to_raw_over_3": int(np.sum(np.asarray(raw_relative) > 3.0)),
        "worst_adjacent_changes": jumps[:30],
        "worst_local_bulges": bulges[:30],
        "worst_raw_relative_frames": raw_ratios[:30],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--result-sqlite", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()
    reports = {
        str(path.resolve()): audit(
            args.source_sqlite,
            path,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        for path in args.result_sqlite
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
