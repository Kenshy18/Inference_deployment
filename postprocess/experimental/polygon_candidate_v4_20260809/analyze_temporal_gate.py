#!/usr/bin/env python3
"""Screen cheap geometry features for temporal-candidate gating.

The analysis is intentionally conservative: it reports how many currently
selected temporal keys a threshold would retain, but does not alter the DP.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experimental.polygon_recall_optimizer.fixed_budget import (
    _raw_keyframe,
    load_raw_masks,
)
from experimental.polygon_recall_optimizer.pareto_dp import _keyframe_geometry


def _iou(left, right) -> float:
    intersection = float(left.intersection(right).area)
    union = float(left.area + right.area - intersection)
    return intersection / union if union else 1.0


def _screen(scores: np.ndarray, positive: np.ndarray, target_recall: float):
    positive_scores = scores[positive]
    if not len(positive_scores):
        return None
    # Keep values at or above the lowest threshold that retains the requested
    # fraction of known temporal selections.
    threshold = float(np.quantile(positive_scores, 1.0 - target_recall))
    kept = scores >= threshold - 1e-12
    return {
        "target_positive_recall": target_recall,
        "threshold": threshold,
        "actual_positive_recall": float(np.mean(kept[positive])),
        "candidate_frame_keep_ratio": float(np.mean(kept)),
        "candidate_frame_reduction_ratio": float(1.0 - np.mean(kept)),
        "negative_keep_ratio": float(np.mean(kept[~positive])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--usage-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    usage = json.loads(args.usage_json.read_text(encoding="utf-8"))
    by_track = {}
    for (frame, track_id), value in raw.items():
        by_track.setdefault(str(track_id), {})[int(frame)] = value
    rows = []
    for record in usage["records"]:
        frame = int(record["frame"])
        track_id = str(record["track_id"])
        value = raw.get((frame, track_id))
        if value is None:
            continue
        bounds = value.geometry.bounds
        at_border = (
            bounds[0] <= 10.0
            or bounds[1] <= 10.0
            or bounds[2] >= args.width - 10.0
            or bounds[3] >= args.height - 10.0
        )
        # Border frames already skip temporal generation by construction and
        # therefore do not belong in an interior gate analysis.
        if at_border:
            continue
        track = by_track[track_id]
        prior = track.get(frame - 1)
        following = track.get(frame + 1)
        neighbors = [neighbor for neighbor in (prior, following) if neighbor]
        if not neighbors:
            continue
        adjacent_iou = [_iou(value.geometry, neighbor.geometry) for neighbor in neighbors]
        area = max(float(value.geometry.area), 1e-9)
        area_jump = max(
            abs(math.log(max(float(neighbor.geometry.area), 1e-9) / area))
            for neighbor in neighbors
        )
        center = np.asarray(value.geometry.centroid.coords[0], dtype=np.float64)
        speeds = [
            float(
                np.linalg.norm(
                    np.asarray(neighbor.geometry.centroid.coords[0]) - center
                )
                / math.sqrt(area)
            )
            for neighbor in neighbors
        ]
        acceleration = 0.0
        if prior is not None and following is not None:
            acceleration = float(
                np.linalg.norm(
                    np.asarray(prior.geometry.centroid.coords[0])
                    - 2.0 * center
                    + np.asarray(following.geometry.centroid.coords[0])
                )
                / math.sqrt(area)
            )
        uniform = _keyframe_geometry(
            _raw_keyframe(value, point_count=23, point_strategy="uniform")
        )
        rows.append(
            {
                "frame": frame,
                "track_id": track_id,
                "positive": not record["classification"].startswith("legacy"),
                "adjacent_iou_loss": 1.0 - min(adjacent_iou),
                "area_log_jump": area_jump,
                "centroid_speed_normalized": max(speeds),
                "centroid_acceleration_normalized": acceleration,
                "uniform23_iou_loss": 1.0 - _iou(value.geometry, uniform),
                "source_vertex_count": len(value.geometry.exterior.coords) - 1,
            }
        )
    positive = np.asarray([row["positive"] for row in rows], dtype=bool)
    feature_names = [
        "adjacent_iou_loss",
        "area_log_jump",
        "centroid_speed_normalized",
        "centroid_acceleration_normalized",
        "uniform23_iou_loss",
        "source_vertex_count",
    ]
    screens = {}
    for name in feature_names:
        scores = np.asarray([row[name] for row in rows], dtype=np.float64)
        screens[name] = {
            "positive_mean": float(np.mean(scores[positive])),
            "negative_mean": float(np.mean(scores[~positive])),
            "retain_99pct": _screen(scores, positive, 0.99),
            "retain_95pct": _screen(scores, positive, 0.95),
        }
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "configuration": {
            **vars(args),
            "source_sqlite": str(args.source_sqlite),
            "usage_json": str(args.usage_json),
            "output_json": str(args.output_json),
        },
        "sample_count": len(rows),
        "positive_temporal_keys": int(np.count_nonzero(positive)),
        "negative_legacy_keys": int(np.count_nonzero(~positive)),
        "screens": screens,
        "records": rows,
        "warning": (
            "Selected-key screening is not a safe production gate by itself; "
            "a gated DP must be rerun and compared before adoption."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
