#!/usr/bin/env python3
"""Compare uniform boundary sampling with corner-preserving 23-point RDP.

This is a geometry-only analysis.  It never opens the source video.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from experimental.polygon_recall_optimizer.fixed_budget import (
    _raw_keyframe,
    load_raw_masks,
)
from experimental.polygon_recall_optimizer.pareto_dp import _keyframe_geometry
from overlay_renderer.keyframe_cache import Component, Keyframe


def _metrics(reference, candidate) -> tuple[float, float]:
    intersection = float(reference.intersection(candidate).area)
    union = float(reference.area + candidate.area - intersection)
    return intersection / float(reference.area), intersection / union


def _pad_edges(points: np.ndarray, target: int) -> np.ndarray:
    """Add collinear midpoints without changing the simplified polygon."""

    values = [np.asarray(point, dtype=np.float64) for point in points]
    while len(values) < target:
        lengths = [
            float(np.linalg.norm(values[(index + 1) % len(values)] - value))
            for index, value in enumerate(values)
        ]
        index = int(np.argmax(lengths))
        midpoint = 0.5 * (values[index] + values[(index + 1) % len(values)])
        values.insert(index + 1, midpoint)
    return np.asarray(values, dtype=np.float64)


def _rdp_keyframe(raw, point_count: int) -> Keyframe | None:
    geometry = raw.geometry
    if geometry.geom_type != "Polygon" or geometry.is_empty:
        return None
    original_count = len(geometry.exterior.coords) - 1
    if original_count <= point_count:
        points = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
    else:
        min_x, min_y, max_x, max_y = geometry.bounds
        high = max(math.hypot(max_x - min_x, max_y - min_y), 1.0)
        low = 0.0
        best = None
        # Find the least simplification that reaches the point budget.  RDP
        # retains high-curvature corners that uniform arc-length sampling can
        # skip, which is the behavior under test.
        for _step in range(24):
            tolerance = 0.5 * (low + high)
            simplified = geometry.simplify(tolerance, preserve_topology=True)
            if simplified.geom_type != "Polygon" or simplified.is_empty:
                low = tolerance
                continue
            count = len(simplified.exterior.coords) - 1
            if count <= point_count:
                best = simplified
                high = tolerance
            else:
                low = tolerance
        if best is None:
            return None
        points = np.asarray(best.exterior.coords[:-1], dtype=np.float64)
    if len(points) < 3 or len(points) > point_count:
        return None
    points = _pad_edges(points, point_count)
    return Keyframe(
        int(raw.frame), ((0, Component("polygon", points.tolist())),)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--include-frame", type=int, action="append", default=[])
    args = parser.parse_args()

    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    included = set(args.include_frame)
    selected = [
        value
        for (frame, _track_id), value in sorted(raw.items())
        if frame in included or (frame - args.start_frame) % max(args.stride, 1) == 0
    ]
    started = time.perf_counter()
    records = []
    for value in selected:
        uniform = _raw_keyframe(value, point_count=args.point_count, point_strategy="uniform")
        rdp = _rdp_keyframe(value, args.point_count)
        if rdp is None:
            continue
        uniform_recall, uniform_iou = _metrics(
            value.geometry, _keyframe_geometry(uniform)
        )
        rdp_recall, rdp_iou = _metrics(value.geometry, _keyframe_geometry(rdp))
        records.append(
            {
                "frame": int(value.frame),
                "track_id": str(value.track_id),
                "source_vertices": len(value.geometry.exterior.coords) - 1,
                "uniform_recall": uniform_recall,
                "uniform_iou": uniform_iou,
                "rdp_recall": rdp_recall,
                "rdp_iou": rdp_iou,
            }
        )
    elapsed = time.perf_counter() - started
    uniform_iou = np.asarray([row["uniform_iou"] for row in records])
    rdp_iou = np.asarray([row["rdp_iou"] for row in records])
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "configuration": {
            **vars(args),
            "source_sqlite": str(args.source_sqlite),
            "output_json": str(args.output_json),
        },
        "sample_count": len(records),
        "elapsed_seconds": elapsed,
        "uniform_recall_feasible": sum(row["uniform_recall"] >= 0.97 for row in records),
        "rdp_recall_feasible": sum(row["rdp_recall"] >= 0.97 for row in records),
        "uniform_failed_rdp_passed": sum(
            row["uniform_recall"] < 0.97 <= row["rdp_recall"] for row in records
        ),
        "uniform_passed_rdp_failed": sum(
            row["rdp_recall"] < 0.97 <= row["uniform_recall"] for row in records
        ),
        "mean_iou_uniform": float(np.mean(uniform_iou)),
        "mean_iou_rdp": float(np.mean(rdp_iou)),
        "q01_iou_uniform": float(np.quantile(uniform_iou, 0.01)),
        "q01_iou_rdp": float(np.quantile(rdp_iou, 0.01)),
        "rdp_iou_better": int(np.count_nonzero(rdp_iou > uniform_iou + 1e-12)),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
