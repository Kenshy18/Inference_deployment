"""Ellipse gap interpolation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from contracts.ellipses import canonicalize_ellipse, ellipse_pair_cost


def kffill_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing frames between union rows without crossing track or ellipse-count changes."
    )
    parser.add_argument("--input-union-json", required=True)
    parser.add_argument("--input-metrics-csv", required=True)
    parser.add_argument("--output-union-json", required=True)
    parser.add_argument("--output-metrics-csv", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--max-gap", type=int, default=30)
    return parser.parse_args(argv)


def kffill_load_union_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.sort(key=lambda row: (int(str(row["track_id"])), int(row["frame"])))
    return rows


def kffill_load_metrics(
    path: Path,
) -> tuple[list[str], dict[tuple[int, str], dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        lookup: dict[tuple[int, str], dict[str, str]] = {}
        for row in reader:
            lookup[int(row["frame"]), str(row["track_id"])] = dict(row)
    return (fieldnames, lookup)


def kffill_linear_interp(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * float(a) + alpha * float(b)


def kffill_interpolate_angle_deg(theta0: float, theta1: float, alpha: float) -> float:
    left = float(theta0)
    right = float(theta1)
    candidates = [right - 180.0, right, right + 180.0]
    best = min(candidates, key=lambda value: abs(value - left))
    out = kffill_linear_interp(left, best, alpha)
    return canonicalize_ellipse([0.0, 0.0, 1.0, 1.0, out])[4]


def kffill_interpolate_ellipse(
    left: list[float], right: list[float], alpha: float
) -> list[float]:
    cx = kffill_linear_interp(left[0], right[0], alpha)
    cy = kffill_linear_interp(left[1], right[1], alpha)
    log_a = kffill_linear_interp(
        math.log(max(float(left[2]), 1e-06)),
        math.log(max(float(right[2]), 1e-06)),
        alpha,
    )
    log_b = kffill_linear_interp(
        math.log(max(float(left[3]), 1e-06)),
        math.log(max(float(right[3]), 1e-06)),
        alpha,
    )
    theta = kffill_interpolate_angle_deg(float(left[4]), float(right[4]), alpha)
    return canonicalize_ellipse([cx, cy, math.exp(log_a), math.exp(log_b), theta])


def kffill_stabilize_right_slots(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    if len(left) != 2 or len(right) != 2:
        return right
    keep_cost = ellipse_pair_cost(
        left[0], right[0], 1.0, 0.65, 0.2
    ) + ellipse_pair_cost(left[1], right[1], 1.0, 0.65, 0.2)
    swap_cost = ellipse_pair_cost(
        left[0], right[1], 1.0, 0.65, 0.2
    ) + ellipse_pair_cost(left[1], right[0], 1.0, 0.65, 0.2)
    if swap_cost < keep_cost:
        return [right[1], right[0]]
    return right


def kffill_choose_fill_mode(left_mode: str, right_mode: str, alpha: float) -> str:
    if left_mode == right_mode:
        return left_mode
    return left_mode if alpha < 0.5 else right_mode


def kffill_interpolate_metric_row(
    fieldnames: list[str],
    left_row: dict[str, str],
    right_row: dict[str, str],
    frame: int,
    track_id: str,
    mode: str,
    run_id: int,
    ellipse_params: list[list[float]],
    alpha: float,
) -> dict[str, str]:
    out: dict[str, str] = {}
    numeric_fields = {
        "gt_area",
        "pred_area",
        "intersection",
        "union",
        "recall",
        "precision",
        "iou",
        "weighted_error",
    }
    for field in fieldnames:
        if field == "frame":
            out[field] = str(int(frame))
        elif field == "track_id":
            out[field] = str(track_id)
        elif field == "mode":
            out[field] = str(mode)
        elif field == "run_id":
            out[field] = str(int(run_id))
        elif field == "has_keyframe":
            out[field] = "0"
        elif field == "is_gap_filled":
            out[field] = "1"
        elif field == "ellipse_params":
            out[field] = json.dumps(ellipse_params, ensure_ascii=False)
        elif field in numeric_fields and field in left_row and (field in right_row):
            out[field] = str(
                kffill_linear_interp(
                    float(left_row[field]), float(right_row[field]), alpha
                )
            )
        else:
            out[field] = left_row.get(field, "")
    return out


def kffill_main(argv: list[str] | None = None) -> None:
    args = kffill_parse_args(argv)
    input_union = Path(args.input_union_json)
    input_metrics = Path(args.input_metrics_csv)
    output_union = Path(args.output_union_json)
    output_metrics = Path(args.output_metrics_csv)
    output_summary = Path(args.output_summary_json)
    output_union.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    union_rows = kffill_load_union_rows(input_union)
    fieldnames, metric_lookup = kffill_load_metrics(input_metrics)
    if "is_gap_filled" not in fieldnames:
        fieldnames.append("is_gap_filled")
    by_track: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in union_rows:
        by_track[str(row["track_id"])].append(row)
    for rows in by_track.values():
        rows.sort(key=lambda row: int(row["frame"]))
    filled_union_rows: list[dict[str, object]] = []
    filled_metric_rows: list[dict[str, str]] = []
    gap_count = 0
    filled_frame_count = 0
    for track_id, track_rows in sorted(by_track.items(), key=lambda item: int(item[0])):
        for idx, left in enumerate(track_rows):
            left_frame = int(left["frame"])
            left_mode = str(left.get("mode", ""))
            left_run_id = int(left.get("run_id", -1))
            left_ellipses = [
                canonicalize_ellipse(ellipse) for ellipse in left["ellipse_params"]
            ]
            filled_union_rows.append(left)
            metric_row = metric_lookup.get((left_frame, track_id))
            if metric_row is not None:
                metric_row = dict(metric_row)
                metric_row["is_gap_filled"] = "0"
                filled_metric_rows.append(metric_row)
            if idx + 1 >= len(track_rows):
                continue
            right = track_rows[idx + 1]
            right_frame = int(right["frame"])
            right_mode = str(right.get("mode", ""))
            gap = right_frame - left_frame - 1
            if gap <= 0 or gap > int(args.max_gap):
                continue
            if len(left["ellipse_params"]) != len(right["ellipse_params"]):
                continue
            left_metrics = metric_lookup.get((left_frame, track_id))
            right_metrics = metric_lookup.get((right_frame, track_id))
            if left_metrics is None or right_metrics is None:
                continue
            right_ellipses = [
                canonicalize_ellipse(ellipse) for ellipse in right["ellipse_params"]
            ]
            right_ellipses = kffill_stabilize_right_slots(left_ellipses, right_ellipses)
            gap_count += 1
            for missing_frame in range(left_frame + 1, right_frame):
                alpha = (missing_frame - left_frame) / float(right_frame - left_frame)
                mode = kffill_choose_fill_mode(left_mode, right_mode, alpha)
                ellipses = [
                    kffill_interpolate_ellipse(
                        left_ellipses[slot_id], right_ellipses[slot_id], alpha
                    )
                    for slot_id in range(len(left_ellipses))
                ]
                filled_union_rows.append(
                    {
                        "track_id": track_id,
                        "mode": mode,
                        "run_id": left_run_id,
                        "frame": int(missing_frame),
                        "ellipse_params": ellipses,
                        "has_keyframe": 0,
                    }
                )
                filled_metric_rows.append(
                    kffill_interpolate_metric_row(
                        fieldnames=fieldnames,
                        left_row=left_metrics,
                        right_row=right_metrics,
                        frame=missing_frame,
                        track_id=track_id,
                        mode=mode,
                        run_id=left_run_id,
                        ellipse_params=ellipses,
                        alpha=alpha,
                    )
                )
                filled_frame_count += 1
    filled_union_rows.sort(
        key=lambda row: (int(str(row["track_id"])), int(row["frame"]))
    )
    filled_metric_rows.sort(
        key=lambda row: (int(row["frame"]), int(str(row["track_id"])))
    )
    output_union.write_text(
        json.dumps(filled_union_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with output_metrics.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in filled_metric_rows:
            writer.writerow(row)
    summary = {
        "input_union_json": str(input_union),
        "input_metrics_csv": str(input_metrics),
        "output_union_json": str(output_union),
        "output_metrics_csv": str(output_metrics),
        "max_gap": int(args.max_gap),
        "input_rows": int(len(union_rows)),
        "output_rows": int(len(filled_union_rows)),
        "filled_gaps": int(gap_count),
        "filled_frames": int(filled_frame_count),
    }
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
