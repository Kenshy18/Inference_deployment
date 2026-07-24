from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np

from contracts.ellipses import ellipses_to_polygons


def _parse_polygons(value: str) -> list[np.ndarray]:
    return [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in json.loads(value)
    ]


def _exact_metrics(
    gt_polygons: list[np.ndarray], predicted_polygons: list[np.ndarray]
) -> dict[str, float]:
    rounded = [
        np.round(polygon).astype(np.int32)
        for polygon in gt_polygons + predicted_polygons
    ]
    points = np.concatenate(rounded, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    shift = minimum.astype(np.int32)
    shape = (
        int(maximum[1] - minimum[1] + 1),
        int(maximum[0] - minimum[0] + 1),
    )
    gt_mask = np.zeros(shape, dtype=np.uint8)
    predicted_mask = np.zeros(shape, dtype=np.uint8)
    for polygon in rounded[: len(gt_polygons)]:
        cv2.fillPoly(gt_mask, [polygon - shift], 1)
    for polygon in rounded[len(gt_polygons) :]:
        cv2.fillPoly(predicted_mask, [polygon - shift], 1)
    gt_area = int(gt_mask.sum())
    predicted_area = int(predicted_mask.sum())
    intersection = int((gt_mask & predicted_mask).sum())
    union = int((gt_mask | predicted_mask).sum())
    return {
        "gt_area": float(gt_area),
        "pred_area": float(predicted_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": intersection / gt_area if gt_area else 1.0,
        "precision": (intersection / predicted_area if predicted_area else 1.0),
        "iou": intersection / union if union else 1.0,
    }


def kfeval_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact evaluator for keyframe-optimized ellipse sequences."
    )
    parser.add_argument("--input-union-json", required=True)
    parser.add_argument("--input-tracked-sqlite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics-csv", default="")
    return parser.parse_args(argv)


def kfeval_load_union_rows(path: Path) -> dict[tuple[int, str], dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["frame"]), str(row["track_id"]))
        lookup[key] = row
    return lookup


def kfeval_aggregate_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    gt_area = sum((float(row["gt_area"]) for row in rows))
    pred_area = sum((float(row["pred_area"]) for row in rows))
    intersection = sum((float(row["intersection"]) for row in rows))
    union = sum((float(row["union"]) for row in rows))
    weighted_error = sum((float(row["weighted_error"]) for row in rows))
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "row_count": float(len(rows)),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "global_recall": float(recall),
        "global_precision": float(precision),
        "global_iou": float(iou),
        "weighted_error_total": float(weighted_error),
        "weighted_error_mean": float(weighted_error / max(len(rows), 1)),
    }


def kfeval_load_baseline_summary(path: Path) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "gt_area": float(row["gt_area"]),
                "pred_area": float(row["pred_area"]),
                "intersection": float(row["intersection"]),
                "union": float(row["union"]),
                "weighted_error": float(row["weighted_error"]),
            }
            rows.append(entry)
    return kfeval_aggregate_summary(rows)


def kfeval_main(argv: list[str] | None = None) -> None:
    args = kfeval_parse_args(argv)
    union_path = Path(args.input_union_json)
    tracked_sqlite = Path(args.input_tracked_sqlite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_lookup = kfeval_load_union_rows(union_path)
    result_rows: list[dict[str, object]] = []
    conn = sqlite3.connect(str(tracked_sqlite))
    cur = conn.cursor()
    for frame, track_id, polygons_json in cur.execute(
        "SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)"
    ):
        key = (int(frame), str(track_id))
        pred_entry = pred_lookup.get(key)
        if pred_entry is None:
            continue
        gt_polys = _parse_polygons(str(polygons_json))
        pred_polys = [
            np.asarray(polygon, dtype=np.float32)
            for polygon in ellipses_to_polygons(pred_entry["ellipse_params"])
        ]
        metrics = _exact_metrics(gt_polys, pred_polys)
        metrics["weighted_error"] = float(
            2 * (metrics["gt_area"] - metrics["intersection"])
            + metrics["pred_area"]
            - metrics["intersection"]
        )
        result_rows.append(
            {
                "frame": int(frame),
                "track_id": str(track_id),
                "mode": str(pred_entry.get("mode", "")),
                "run_id": int(pred_entry.get("run_id", -1)),
                "has_keyframe": int(pred_entry.get("has_keyframe", 0)),
                "gt_area": float(metrics["gt_area"]),
                "pred_area": float(metrics["pred_area"]),
                "intersection": float(metrics["intersection"]),
                "union": float(metrics["union"]),
                "recall": float(metrics["recall"]),
                "precision": float(metrics["precision"]),
                "iou": float(metrics["iou"]),
                "weighted_error": float(metrics["weighted_error"]),
                "ellipse_params": json.dumps(
                    pred_entry["ellipse_params"], ensure_ascii=False
                ),
            }
        )
    conn.close()
    result_rows.sort(key=lambda row: (int(row["frame"]), int(str(row["track_id"]))))
    metrics_csv = output_dir / "keyframe_exact_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "track_id",
                "mode",
                "run_id",
                "has_keyframe",
                "gt_area",
                "pred_area",
                "intersection",
                "union",
                "recall",
                "precision",
                "iou",
                "weighted_error",
                "ellipse_params",
            ],
        )
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)
    summary = {
        "input_union_json": str(union_path),
        "input_tracked_sqlite": str(tracked_sqlite),
        "optimized": kfeval_aggregate_summary(result_rows),
    }
    if args.baseline_metrics_csv:
        baseline_path = Path(args.baseline_metrics_csv)
        summary["baseline"] = kfeval_load_baseline_summary(baseline_path)
        summary["delta_vs_baseline"] = {
            key: float(summary["optimized"][key] - summary["baseline"][key])
            for key in [
                "global_recall",
                "global_precision",
                "global_iou",
                "weighted_error_total",
                "weighted_error_mean",
            ]
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
