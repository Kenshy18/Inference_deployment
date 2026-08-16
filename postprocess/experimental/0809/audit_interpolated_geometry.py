#!/usr/bin/env python3
"""Audit keyframe/interpolated polygon topology without opening video pixels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _best_alignment(reference: np.ndarray, candidate: np.ndarray):
    best = (math.inf, False, 0)
    for reverse, variant in ((False, candidate), (True, candidate[::-1])):
        for shift in range(len(variant)):
            shifted = np.roll(variant, shift, axis=0)
            error = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if error < best[0]:
                best = (error, reverse, shift)
    return best


def _audit_label(root: Path) -> dict[str, object]:
    key_path = root / "runtime/opt/final_keyframes.json"
    dense_path = root / "runtime/opt/interpolated_union.json"
    keys = json.loads(key_path.read_text(encoding="utf-8"))
    dense = json.loads(dense_path.read_text(encoding="utf-8"))

    key_invalid = 0
    dense_invalid = 0
    winding_flips = 0
    alignment_reversals = 0
    alignment_shifts = 0
    point_counts: Counter[int] = Counter()

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for key in keys:
        grouped.setdefault((str(key["track_id"]), int(key["run_id"])), []).append(key)
        for values in key["polygons"]:
            points = np.asarray(values, dtype=np.float64)
            point_counts[len(points)] += 1
            polygon = Polygon(points)
            key_invalid += int(not polygon.is_valid or polygon.is_empty)

    for values in grouped.values():
        values.sort(key=lambda item: int(item["frame"]))
        for left, right in zip(values, values[1:]):
            for left_values, right_values in zip(left["polygons"], right["polygons"]):
                left_points = np.asarray(left_values, dtype=np.float64)
                right_points = np.asarray(right_values, dtype=np.float64)
                if len(left_points) != len(right_points):
                    continue
                winding_flips += int(
                    _signed_area(left_points) * _signed_area(right_points) < 0.0
                )
                _error, reverse, shift = _best_alignment(left_points, right_points)
                alignment_reversals += int(reverse)
                alignment_shifts += int(shift != 0)

    dense_polygons = 0
    for row in dense:
        for values in row["polygons"]:
            dense_polygons += 1
            polygon = Polygon(np.asarray(values, dtype=np.float64))
            dense_invalid += int(not polygon.is_valid or polygon.is_empty)

    return {
        "keyframe_rows": len(keys),
        "keyframe_polygons": int(sum(point_counts.values())),
        "keyframe_invalid_polygons": key_invalid,
        "interpolated_rows": len(dense),
        "interpolated_polygons": dense_polygons,
        "interpolated_invalid_polygons": dense_invalid,
        "adjacent_winding_flips": winding_flips,
        "best_alignment_reversals": alignment_reversals,
        "best_alignment_nonzero_shifts": alignment_shifts,
        "point_counts": dict(sorted(point_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base = args.output_root.expanduser().resolve() / args.profile
    labels = {}
    for label_root in sorted(path for path in base.iterdir() if path.is_dir()):
        if (label_root / "runtime/opt/final_keyframes.json").is_file():
            labels[label_root.name] = _audit_label(label_root)
    totals = {
        key: sum(int(values[key]) for values in labels.values())
        for key in (
            "keyframe_rows",
            "keyframe_polygons",
            "keyframe_invalid_polygons",
            "interpolated_rows",
            "interpolated_polygons",
            "interpolated_invalid_polygons",
            "adjacent_winding_flips",
            "best_alignment_reversals",
            "best_alignment_nonzero_shifts",
        )
    }
    result = {
        "schema_version": 1,
        "privacy": "SQLite-derived polygon coordinates only; no video frames decoded.",
        "output_root": str(args.output_root.expanduser().resolve()),
        "profile": args.profile,
        "labels": labels,
        "totals": totals,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
