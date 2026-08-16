#!/usr/bin/env python3
"""Diagnose temporal risks before adding temporal constraints.

The analysis is geometry-only.  It never opens video pixels.  It compares the
fixed-key per-key pair-vote result with the no-pair-vote best-v4 path and the
exact source-mask metrics emitted by Phase 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = ("女性器", "男性器", "結合部分")


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"min": 0.0, "q01": 0.0, "q05": 0.0, "median": 0.0, "q95": 0.0, "q99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _load_metrics(root: Path, label: str) -> list[dict[str, object]]:
    path = root / label / "runtime/exact/keyframe_exact_metrics.csv"
    output: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            output.append(
                {
                    "label": label,
                    "frame": int(raw["frame"]),
                    "track_id": str(raw["track_id"]),
                    "run_id": int(raw["run_id"]),
                    "has_keyframe": bool(int(raw["has_keyframe"])),
                    "gt_area": float(raw["gt_area"]),
                    "pred_area": float(raw["pred_area"]),
                    "recall": float(raw["recall"]),
                    "precision": float(raw["precision"]),
                    "iou": float(raw["iou"]),
                }
            )
    return output


def _load_vectors(root: Path, label: str) -> dict[tuple[int, str], np.ndarray]:
    path = root / label / "runtime/pred/predictions.sqlite"
    output: dict[tuple[int, str], np.ndarray] = {}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        for frame, track_id, payload in db.execute(
            "SELECT frame, track_id, polygons FROM masks"
        ):
            polygons = json.loads(str(payload))
            points = [point for polygon in polygons for point in polygon]
            output[(int(frame), str(track_id))] = np.asarray(
                points, dtype=np.float64
            ).reshape(-1, 2)
    return output


def _neighbor_median(values: np.ndarray, index: int, radius: int = 3) -> float:
    neighbors = [
        float(values[pos])
        for pos in range(max(0, index - radius), min(len(values), index + radius + 1))
        if pos != index
    ]
    return float(np.median(neighbors)) if neighbors else float(values[index])


def _vector_acceleration(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray, radius: float
) -> tuple[float, float, float] | None:
    if previous.shape != current.shape or following.shape != current.shape:
        return None
    acceleration = following - 2.0 * current + previous
    centroid_acceleration = np.mean(acceleration, axis=0)
    shape_acceleration = acceleration - centroid_acceleration
    normalizer = max(float(radius), 1.0)
    return (
        float(np.mean(np.linalg.norm(acceleration, axis=1)) / normalizer),
        float(np.linalg.norm(centroid_acceleration) / normalizer),
        float(np.mean(np.linalg.norm(shape_acceleration, axis=1)) / normalizer),
    )


def _rank(rows: list[dict[str, object]], field: str, count: int = 30, reverse: bool = True):
    eligible = [row for row in rows if row.get(field) is not None]
    return sorted(eligible, key=lambda row: float(row[field]), reverse=reverse)[:count]


def _compact(row: dict[str, object]) -> dict[str, object]:
    fields = (
        "label", "track_id", "run_id", "frame", "has_keyframe", "iou",
        "recall", "area_ratio", "local_iou_drop", "iou_delta_vs_best_v4",
        "total_acceleration", "shape_acceleration", "centroid_acceleration",
        "shape_acceleration_delta_vs_best_v4", "area_acceleration",
        "area_acceleration_excess_vs_source", "area_acceleration_delta_vs_best_v4",
    )
    return {key: row[key] for key in fields if key in row and row[key] is not None}


def _merge_sections(
    rows: list[dict[str, object]], *, max_gap: int = 4
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["label"]), str(row["track_id"]), int(row["run_id"]))].append(row)
    sections: list[dict[str, object]] = []
    for (label, track_id, run_id), values in grouped.items():
        values.sort(key=lambda row: int(row["frame"]))
        current: list[dict[str, object]] = []
        for row in values:
            if current and int(row["frame"]) - int(current[-1]["frame"]) > max_gap:
                sections.append(_section(label, track_id, run_id, current))
                current = []
            current.append(row)
        if current:
            sections.append(_section(label, track_id, run_id, current))
    return sorted(
        sections,
        key=lambda row: (
            -float(row["severity"]), str(row["label"]), int(row["start_frame"])
        ),
    )


def _section(label: str, track_id: str, run_id: int, rows: list[dict[str, object]]):
    return {
        "label": label,
        "track_id": track_id,
        "run_id": run_id,
        "start_frame": min(int(row["frame"]) for row in rows),
        "end_frame": max(int(row["frame"]) for row in rows),
        "frames": [int(row["frame"]) for row in rows],
        "keyframe_hits": sum(bool(row["has_keyframe"]) for row in rows),
        "minimum_iou": min(float(row["iou"]) for row in rows),
        "minimum_recall": min(float(row["recall"]) for row in rows),
        "maximum_local_iou_drop": max(float(row["local_iou_drop"]) for row in rows),
        "minimum_iou_delta_vs_best_v4": min(
            float(row["iou_delta_vs_best_v4"]) for row in rows
        ),
        "maximum_shape_acceleration": max(
            float(row.get("shape_acceleration") or 0.0) for row in rows
        ),
        "maximum_area_ratio": max(float(row["area_ratio"]) for row in rows),
        "severity": max(float(row["severity"]) for row in rows),
        "reasons": sorted({reason for row in rows for reason in row["reasons"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    class_summary: dict[str, object] = {}
    for label in LABELS:
        current_metrics = _load_metrics(args.current_root.resolve(), label)
        baseline_metrics = {
            (str(row["track_id"]), int(row["run_id"]), int(row["frame"])): row
            for row in _load_metrics(args.baseline_root.resolve(), label)
        }
        current_vectors = _load_vectors(args.current_root.resolve(), label)
        baseline_vectors = _load_vectors(args.baseline_root.resolve(), label)
        grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for row in current_metrics:
            grouped[(str(row["track_id"]), int(row["run_id"]))].append(row)

        label_rows: list[dict[str, object]] = []
        for (track_id, run_id), rows in grouped.items():
            rows.sort(key=lambda row: int(row["frame"]))
            ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
            recalls = np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
            pred_log_area = np.log(
                np.maximum([float(row["pred_area"]) for row in rows], 1.0)
            )
            baseline_pred_log_area = np.log(
                np.maximum(
                    [
                        float(
                            baseline_metrics[
                                (track_id, run_id, int(row["frame"]))
                            ]["pred_area"]
                        )
                        for row in rows
                    ],
                    1.0,
                )
            )
            source_log_area = np.log(
                np.maximum([float(row["gt_area"]) for row in rows], 1.0)
            )
            for index, row in enumerate(rows):
                frame = int(row["frame"])
                base = baseline_metrics[(track_id, run_id, frame)]
                row["area_ratio"] = float(row["pred_area"]) / max(
                    float(row["gt_area"]), 1.0
                )
                row["local_iou_drop"] = max(
                    0.0, _neighbor_median(ious, index) - float(row["iou"])
                )
                row["local_recall_drop"] = max(
                    0.0, _neighbor_median(recalls, index) - float(row["recall"])
                )
                row["iou_delta_vs_best_v4"] = float(row["iou"]) - float(base["iou"])
                row["recall_delta_vs_best_v4"] = float(row["recall"]) - float(base["recall"])
                row["area_ratio_delta_vs_best_v4"] = row["area_ratio"] - (
                    float(base["pred_area"]) / max(float(base["gt_area"]), 1.0)
                )
                if 0 < index < len(rows) - 1 and (
                    int(rows[index - 1]["frame"]) + 1 == frame
                    and frame + 1 == int(rows[index + 1]["frame"])
                ):
                    row["area_acceleration"] = abs(
                        float(pred_log_area[index + 1] - 2 * pred_log_area[index] + pred_log_area[index - 1])
                    )
                    source_acceleration = abs(
                        float(source_log_area[index + 1] - 2 * source_log_area[index] + source_log_area[index - 1])
                    )
                    baseline_area_acceleration = abs(
                        float(
                            baseline_pred_log_area[index + 1]
                            - 2 * baseline_pred_log_area[index]
                            + baseline_pred_log_area[index - 1]
                        )
                    )
                    row["source_area_acceleration"] = source_acceleration
                    row["baseline_area_acceleration"] = baseline_area_acceleration
                    row["area_acceleration_delta_vs_best_v4"] = (
                        float(row["area_acceleration"]) - baseline_area_acceleration
                    )
                    row["area_acceleration_excess_vs_source"] = max(
                        0.0, float(row["area_acceleration"]) - source_acceleration
                    )
                    key_previous = (int(rows[index - 1]["frame"]), track_id)
                    key_current = (frame, track_id)
                    key_following = (int(rows[index + 1]["frame"]), track_id)
                    radius = math.sqrt(max(float(row["gt_area"]), 1.0) / math.pi)
                    current_acceleration = _vector_acceleration(
                        current_vectors[key_previous],
                        current_vectors[key_current],
                        current_vectors[key_following],
                        radius,
                    )
                    baseline_acceleration = _vector_acceleration(
                        baseline_vectors[key_previous],
                        baseline_vectors[key_current],
                        baseline_vectors[key_following],
                        radius,
                    )
                    if current_acceleration is not None:
                        row["total_acceleration"], row["centroid_acceleration"], row["shape_acceleration"] = current_acceleration
                    if baseline_acceleration is not None:
                        row["baseline_total_acceleration"], row["baseline_centroid_acceleration"], row["baseline_shape_acceleration"] = baseline_acceleration
                    if current_acceleration is not None and baseline_acceleration is not None:
                        row["shape_acceleration_delta_vs_best_v4"] = (
                            float(row["shape_acceleration"])
                            - float(row["baseline_shape_acceleration"])
                        )
                        row["centroid_acceleration_delta_vs_best_v4"] = (
                            float(row["centroid_acceleration"])
                            - float(row["baseline_centroid_acceleration"])
                        )
                label_rows.append(row)

        class_summary[label] = {
            "rows": len(label_rows),
            "keyframes": sum(bool(row["has_keyframe"]) for row in label_rows),
            "iou": _quantiles([float(row["iou"]) for row in label_rows]),
            "recall": _quantiles([float(row["recall"]) for row in label_rows]),
            "area_ratio": _quantiles([float(row["area_ratio"]) for row in label_rows]),
            "local_iou_drop": _quantiles([float(row["local_iou_drop"]) for row in label_rows]),
            "shape_acceleration": _quantiles([float(row["shape_acceleration"]) for row in label_rows if row.get("shape_acceleration") is not None]),
            "shape_acceleration_delta_vs_best_v4": _quantiles([float(row["shape_acceleration_delta_vs_best_v4"]) for row in label_rows if row.get("shape_acceleration_delta_vs_best_v4") is not None]),
            "area_acceleration_delta_vs_best_v4": _quantiles([float(row["area_acceleration_delta_vs_best_v4"]) for row in label_rows if row.get("area_acceleration_delta_vs_best_v4") is not None]),
            "pair_vote_iou_regression_001": sum(float(row["iou_delta_vs_best_v4"]) < -0.01 for row in label_rows),
            "pair_vote_iou_regression_005": sum(float(row["iou_delta_vs_best_v4"]) < -0.05 for row in label_rows),
            "recall_violations": sum(float(row["recall"]) + 1e-12 < args.recall_floor for row in label_rows),
        }
        all_rows.extend(label_rows)

    # Distribution-derived temporal thresholds avoid pretending that a fixed
    # pixel threshold is comparable across object sizes and classes.
    shape_values = [float(row["shape_acceleration"]) for row in all_rows if row.get("shape_acceleration") is not None]
    shape_delta_values = [float(row["shape_acceleration_delta_vs_best_v4"]) for row in all_rows if row.get("shape_acceleration_delta_vs_best_v4") is not None]
    area_accel_values = [float(row["area_acceleration_excess_vs_source"]) for row in all_rows if row.get("area_acceleration_excess_vs_source") is not None]
    area_accel_delta_values = [float(row["area_acceleration_delta_vs_best_v4"]) for row in all_rows if row.get("area_acceleration_delta_vs_best_v4") is not None]
    shape_threshold = float(np.quantile(shape_values, 0.995))
    shape_delta_threshold = max(0.0, float(np.quantile(shape_delta_values, 0.995)))
    area_accel_threshold = float(np.quantile(area_accel_values, 0.995))

    flagged: list[dict[str, object]] = []
    for row in all_rows:
        reasons: list[str] = []
        severity = 0.0
        if float(row["iou"]) < 0.50:
            reasons.append("iou_below_050")
            severity = max(severity, 0.50 - float(row["iou"]))
        if float(row["local_iou_drop"]) > 0.10:
            reasons.append("local_iou_drop_over_010")
            severity = max(severity, float(row["local_iou_drop"]))
        if float(row["iou_delta_vs_best_v4"]) < -0.01:
            reasons.append("pair_vote_regression_over_001")
            severity = max(severity, -float(row["iou_delta_vs_best_v4"]))
        if float(row["recall"]) + 1e-12 < args.recall_floor:
            reasons.append("recall_violation")
            severity = max(severity, args.recall_floor - float(row["recall"]))
        if row.get("shape_acceleration") is not None and float(row["shape_acceleration"]) >= shape_threshold:
            reasons.append("shape_acceleration_top_005")
            severity = max(severity, float(row["shape_acceleration"]))
        if row.get("shape_acceleration_delta_vs_best_v4") is not None and float(row["shape_acceleration_delta_vs_best_v4"]) >= shape_delta_threshold and shape_delta_threshold > 0:
            reasons.append("new_shape_acceleration_top_005")
            severity = max(severity, float(row["shape_acceleration_delta_vs_best_v4"]))
        if row.get("area_acceleration_excess_vs_source") is not None and float(row["area_acceleration_excess_vs_source"]) >= area_accel_threshold:
            reasons.append("area_acceleration_excess_top_005")
            severity = max(severity, float(row["area_acceleration_excess_vs_source"]))
        if reasons:
            row["reasons"] = reasons
            row["severity"] = severity
            flagged.append(row)

    fields = sorted({key for row in all_rows for key in row})
    with (output_dir / "temporal_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "privacy": "SQLite polygon geometry only; video pixels were not opened.",
        "current_root": str(args.current_root.resolve()),
        "comparison_root": str(args.baseline_root.resolve()),
        "rows": len(all_rows),
        "thresholds": {
            "shape_acceleration_top_0_5_percent": shape_threshold,
            "new_shape_acceleration_top_0_5_percent": shape_delta_threshold,
            "area_acceleration_excess_top_0_5_percent": area_accel_threshold,
            "local_iou_drop": 0.10,
            "pair_vote_iou_regression": -0.01,
            "low_iou": 0.50,
            "recall_floor": float(args.recall_floor),
        },
        "class_summary": class_summary,
        "overall": {
            "iou": _quantiles([float(row["iou"]) for row in all_rows]),
            "recall": _quantiles([float(row["recall"]) for row in all_rows]),
            "area_ratio": _quantiles([float(row["area_ratio"]) for row in all_rows]),
            "local_iou_drop": _quantiles([float(row["local_iou_drop"]) for row in all_rows]),
            "shape_acceleration": _quantiles(shape_values),
            "shape_acceleration_delta_vs_best_v4": _quantiles(shape_delta_values),
            "area_acceleration_excess_vs_source": _quantiles(area_accel_values),
            "area_acceleration_delta_vs_best_v4": _quantiles(area_accel_delta_values),
            "pair_vote_iou_regression_001": sum(float(row["iou_delta_vs_best_v4"]) < -0.01 for row in all_rows),
            "pair_vote_iou_regression_005": sum(float(row["iou_delta_vs_best_v4"]) < -0.05 for row in all_rows),
            "recall_violations": sum(float(row["recall"]) + 1e-12 < args.recall_floor for row in all_rows),
            "flagged_rows": len(flagged),
        },
        "top": {
            "lowest_iou": [_compact(row) for row in _rank(all_rows, "iou", reverse=False)],
            "largest_local_iou_drop": [_compact(row) for row in _rank(all_rows, "local_iou_drop")],
            "largest_pair_vote_regression": [_compact(row) for row in _rank(all_rows, "iou_delta_vs_best_v4", reverse=False)],
            "largest_shape_acceleration": [_compact(row) for row in _rank(all_rows, "shape_acceleration")],
            "largest_new_shape_acceleration": [_compact(row) for row in _rank(all_rows, "shape_acceleration_delta_vs_best_v4")],
            "largest_area_ratio": [_compact(row) for row in _rank(all_rows, "area_ratio")],
            "largest_new_area_acceleration": [_compact(row) for row in _rank(all_rows, "area_acceleration_delta_vs_best_v4")],
        },
        "sections": _merge_sections(flagged),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
