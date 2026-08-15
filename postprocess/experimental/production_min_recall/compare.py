#!/usr/bin/env python3
"""Compare unmodified Production-v22 with the minimum-Recall experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
POSTPROCESS_ROOT = HERE.parents[1]
PRODUCTION_RUNTIME = (
    POSTPROCESS_ROOT / "vendor" / "original_polygon" / "original_run_standalone.py"
)
EXPERIMENT_RUNTIME = HERE / "runtime.py"
DEFAULT_PREDICTOR = POSTPROCESS_ROOT / "models" / "polygon_point_predictor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-interval", type=float, default=10.0)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--predictor-device", default="cuda")
    parser.add_argument("--predictor-model-dir", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def _command(runtime: Path, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(runtime),
        "__onefile_polygon_optimize",
        "--input-sqlite",
        str(args.input_sqlite.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--target-ratio",
        str(1.0 / float(args.target_interval)),
        "--anchors-per-contour",
        "48",
        "--point-predictor-model-dir",
        str(args.predictor_model_dir.resolve()),
        "--predictor-device",
        str(args.predictor_device),
        "--predictor-batch-size",
        "256",
        "--adaptive-point-quantile",
        "0.95",
        "--adaptive-point-offset",
        "10",
        "--min-anchors-per-contour",
        "8",
        "--gapfill-max-gap",
        "15",
        "--max-run-frames",
        "30000",
        "--run-overlap-frames",
        "900",
        "--recall-min",
        str(args.recall_floor),
        "--max-gap",
        "30",
        "--num-workers",
        str(args.num_workers),
        "--stream-sqlite-rows",
        "--evaluate-exact",
        "--write-pred-sqlite",
        "--adaptive-anchor-counts",
        "--gapfill-enabled",
    ]


def _run(name: str, runtime: Path, output: Path, args: argparse.Namespace) -> dict[str, object]:
    summary_path = output / "summary.json"
    log_path = args.output_dir / f"{name}.log"
    if args.reuse and summary_path.is_file():
        wall = None
    else:
        output.mkdir(parents=True, exist_ok=True)
        command = _command(runtime, output, args)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=str(POSTPROCESS_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"{name} failed with code {process.returncode}; see {log_path}"
            )
    return {"wall_seconds": wall, "log": str(log_path), "output": str(output)}


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _metrics(root: Path, floor: float) -> dict[str, object]:
    metrics_path = root / "exact" / "keyframe_exact_metrics.csv"
    rows: list[dict[str, str]] = []
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    recalls = [float(row["recall"]) for row in rows]
    ious = [float(row["iou"]) for row in rows]
    precisions = [float(row["precision"]) for row in rows]
    area_ratios = [float(row["pred_area"]) / max(float(row["gt_area"]), 1.0) for row in rows]

    keyframes = json.loads((root / "opt" / "final_keyframes.json").read_text())
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in keyframes:
        grouped[(str(row["track_id"]), int(row["run_id"]))].append(int(row["frame"]))
    total_span = sum(max(values) - min(values) for values in grouped.values() if values)
    interval_count = sum(max(len(values) - 1, 0) for values in grouped.values())
    summary = json.loads((root / "summary.json").read_text())
    pred_sqlite = root / "pred" / "predictions.sqlite"
    with sqlite3.connect(f"file:{pred_sqlite.resolve()}?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        pred_rows = int(connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
    return {
        "rows": len(rows),
        "keyframes": len(keyframes),
        "segment_count": len(grouped),
        "actual_mean_interval": total_span / max(interval_count, 1),
        "recall_mean": float(np.mean(recalls)),
        "recall_min": min(recalls),
        "recall_q01": _quantile(recalls, 0.01),
        "recall_q05": _quantile(recalls, 0.05),
        "recall_below_floor": sum(value + 1e-12 < floor for value in recalls),
        "iou_mean": float(np.mean(ious)),
        "iou_min": min(ious),
        "iou_q01": _quantile(ious, 0.01),
        "iou_q05": _quantile(ious, 0.05),
        "precision_mean": float(np.mean(precisions)),
        "area_ratio_mean": float(np.mean(area_ratios)),
        "area_ratio_q95": _quantile(area_ratios, 0.95),
        "optimizer_seconds": float(summary["optimizer_summary"]["optimizer_seconds"]),
        "prediction_rows": pred_rows,
        "sqlite_integrity": integrity,
        "prediction_sqlite": str(pred_sqlite.resolve()),
        "keyframes_json": str((root / "opt" / "final_keyframes.json").resolve()),
    }


def main() -> int:
    args = parse_args()
    if args.target_interval <= 0:
        raise SystemExit("--target-interval must be positive")
    if not 0 < args.recall_floor <= 1:
        raise SystemExit("--recall-floor must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = args.output_dir / "production"
    minimum_root = args.output_dir / "production_minimum_recall"
    timing = {
        "production": _run("production", PRODUCTION_RUNTIME, baseline_root, args),
        "production_minimum_recall": _run(
            "production_minimum_recall", EXPERIMENT_RUNTIME, minimum_root, args
        ),
    }
    production = _metrics(baseline_root, args.recall_floor)
    minimum = _metrics(minimum_root, args.recall_floor)
    report = {
        "schema_version": 1,
        "privacy": "SQLite geometry only; no video frame was opened.",
        "input_sqlite": str(args.input_sqlite.resolve()),
        "target_interval": args.target_interval,
        "recall_floor": args.recall_floor,
        "controlled_variables": {
            "production_source": str(PRODUCTION_RUNTIME),
            "candidate_generation": "identical",
            "pair_vote": "identical",
            "border_and_endpoint_preparation": "already materialized in the common input SQLite",
            "difference": "average Recall budget versus minimum per-frame Recall deficit",
        },
        "timing": timing,
        "production": production,
        "production_minimum_recall": minimum,
        "delta_minimum_minus_production": {
            key: float(minimum[key]) - float(production[key])
            for key in (
                "keyframes",
                "actual_mean_interval",
                "recall_mean",
                "recall_min",
                "recall_q01",
                "iou_mean",
                "iou_min",
                "iou_q01",
                "precision_mean",
                "area_ratio_mean",
                "optimizer_seconds",
            )
        },
        "validation": {
            "production_integrity": production["sqlite_integrity"],
            "minimum_integrity": minimum["sqlite_integrity"],
            "minimum_recall_constraint_satisfied": minimum["recall_below_floor"] == 0,
            "row_counts_equal": production["rows"] == minimum["rows"],
            "finite_metrics": all(
                math.isfinite(float(value))
                for section in (production, minimum)
                for key, value in section.items()
                if key in {"recall_mean", "recall_min", "iou_mean", "iou_min"}
            ),
        },
    }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
