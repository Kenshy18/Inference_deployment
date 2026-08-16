#!/usr/bin/env python3
"""Evaluate Phase-2 outputs on a prediction-independent global rounding grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

import native_interval_metrics


MODES = ("exact", "validated_cuda", "cuda_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-matrix", type=Path, required=True)
    parser.add_argument("--validated-cuda-matrix", type=Path, required=True)
    parser.add_argument("--cuda-only-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    return parser.parse_args()


def load_masks(path: str | Path) -> dict[tuple[int, str], list[list[list[float]]]]:
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT frame, CAST(track_id AS TEXT), polygons FROM masks").fetchall()
    return {
        (int(frame), str(track_id)): json.loads(polygons)
        for frame, track_id, polygons in rows
        if polygons
    }


def matrix_rows(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["label"]): row for row in payload["rows"]}, payload["completed_profiles"][0]


def evaluate_label(task: dict[str, object]) -> dict[str, object]:
    label = str(task["label"])
    recall_floor = float(task["recall_floor"])
    raw = load_masks(str(task["source_sqlite"]))
    predictions = {
        mode: load_masks(str(task[f"{mode}_sqlite"])) for mode in MODES
    }
    identities = sorted(raw)
    missing = {
        mode: [identity for identity in identities if identity not in predictions[mode]]
        for mode in MODES
    }
    if any(missing.values()):
        raise RuntimeError(
            f"missing predictions for {label}: "
            + ", ".join(f"{mode}={len(values)}" for mode, values in missing.items())
        )
    rows: list[dict[str, object]] = []
    gt_invariance_failures = 0
    for frame, track_id in identities:
        gt_areas = []
        row: dict[str, object] = {
            "label": label,
            "frame": int(frame),
            "track_id": str(track_id),
        }
        for mode in MODES:
            metrics = dict(
                native_interval_metrics.canonical_metrics(
                    raw[(frame, track_id)], predictions[mode][(frame, track_id)]
                )
            )
            gt_areas.append(int(round(float(metrics["gt_area"]))))
            for key in (
                "gt_area",
                "pred_area",
                "intersection",
                "union",
                "recall",
                "precision",
                "iou",
            ):
                row[f"{mode}_{key}"] = float(metrics[key])
            gt_area = float(metrics["gt_area"])
            row[f"{mode}_area_ratio"] = (
                float(metrics["pred_area"]) / gt_area if gt_area > 0.0 else 1.0
            )
            row[f"{mode}_recall_violation"] = int(
                float(metrics["recall"]) + 1e-12 < recall_floor
            )
        if len(set(gt_areas)) != 1:
            gt_invariance_failures += 1
        rows.append(row)
    return {
        "label": label,
        "rows": rows,
        "gt_area_invariance_failures": int(gt_invariance_failures),
    }


def summarize(rows: list[dict[str, object]], mode: str, recall_floor: float) -> dict[str, object]:
    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[f"{mode}_{name}"]) for row in rows], dtype=np.float64)

    recall = values("recall")
    iou = values("iou")
    precision = values("precision")
    gt_area = values("gt_area")
    pred_area = values("pred_area")
    area_ratio = values("area_ratio")
    violation = recall + 1e-12 < float(recall_floor)
    return {
        "rows": int(len(rows)),
        "recall_mean": float(np.mean(recall)),
        "recall_min": float(np.min(recall)),
        "recall_q001": float(np.quantile(recall, 0.001)),
        "recall_q01": float(np.quantile(recall, 0.01)),
        "recall_q05": float(np.quantile(recall, 0.05)),
        "recall_violations": int(np.count_nonzero(violation)),
        "recall_violation_rate": float(np.mean(violation)),
        "iou_mean": float(np.mean(iou)),
        "iou_min": float(np.min(iou)),
        "iou_q001": float(np.quantile(iou, 0.001)),
        "iou_q01": float(np.quantile(iou, 0.01)),
        "iou_q05": float(np.quantile(iou, 0.05)),
        "precision_mean": float(np.mean(precision)),
        "area_ratio_mean": float(np.mean(area_ratio)),
        "area_ratio_q99": float(np.quantile(area_ratio, 0.99)),
        "area_ratio_max": float(np.max(area_ratio)),
        "global_recall": float(np.sum([float(row[f"{mode}_intersection"]) for row in rows]) / max(np.sum(gt_area), 1.0)),
        "global_precision": float(np.sum([float(row[f"{mode}_intersection"]) for row in rows]) / max(np.sum(pred_area), 1.0)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices = {}
    aggregates = {}
    for mode, path in (
        ("exact", args.exact_matrix),
        ("validated_cuda", args.validated_cuda_matrix),
        ("cuda_only", args.cuda_only_matrix),
    ):
        matrices[mode], aggregates[mode] = matrix_rows(path)
    labels = sorted(set(matrices["exact"]) & set(matrices["validated_cuda"]) & set(matrices["cuda_only"]))
    tasks = []
    for label in labels:
        source = str(matrices["exact"][label]["source_sqlite"])
        if any(str(matrices[mode][label]["source_sqlite"]) != source for mode in MODES):
            raise RuntimeError(f"source SQLite mismatch for {label}")
        tasks.append(
            {
                "label": label,
                "recall_floor": args.recall_floor,
                "source_sqlite": source,
                **{
                    f"{mode}_sqlite": str(matrices[mode][label]["prediction_sqlite"])
                    for mode in MODES
                },
            }
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(evaluate_label, tasks))
    all_rows = [row for result in results for row in result["rows"]]
    columns = list(all_rows[0])
    csv_path = args.output_dir / "canonical_frame_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    by_label = {
        str(result["label"]): {
            mode: summarize(result["rows"], mode, args.recall_floor)
            for mode in MODES
        }
        for result in results
    }
    overall = {mode: summarize(all_rows, mode, args.recall_floor) for mode in MODES}
    worst_cuda = sorted(
        all_rows,
        key=lambda row: (
            float(row["cuda_only_recall"]),
            float(row["cuda_only_iou"]),
            int(row["frame"]),
        ),
    )[:50]
    report = {
        "schema_version": 1,
        "experimental": True,
        "privacy": "SQLite polygon geometry only; no video frame was decoded.",
        "evaluation_contract": {
            "rounding": "global float32 coordinates rounded once with IEEE-754 nearest/ties-to-even",
            "roi": "compact union ROI selected only after global integer rounding",
            "contours": "each contour filled independently and OR-composited",
            "gt_prediction_invariant": True,
            "recall_floor": float(args.recall_floor),
        },
        "evaluated_rows": int(len(all_rows)),
        "gt_area_invariance_failures": int(
            sum(int(result["gt_area_invariance_failures"]) for result in results)
        ),
        "overall": overall,
        "by_label": by_label,
        "runtime_reference": {
            mode: {
                "profile_wall_seconds": aggregates[mode].get("profile_wall_seconds"),
                "video_fps": aggregates[mode].get("video_fps"),
                "keyframes": aggregates[mode].get("keyframes"),
                "actual_mean_interval": aggregates[mode].get("actual_mean_interval"),
            }
            for mode in MODES
        },
        "worst_cuda_only": [
            {
                key: row[key]
                for key in (
                    "label",
                    "frame",
                    "track_id",
                    "cuda_only_gt_area",
                    "cuda_only_pred_area",
                    "cuda_only_recall",
                    "cuda_only_precision",
                    "cuda_only_iou",
                    "validated_cuda_recall",
                    "validated_cuda_iou",
                    "exact_recall",
                    "exact_iou",
                )
            }
            for row in worst_cuda
        ],
        "frame_metrics_csv": str(csv_path.resolve()),
    }
    report_path = args.output_dir / "canonical_evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
