#!/usr/bin/env python3
"""Production support runner for the hard-Recall penalty matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from production.polygon.runtime.algorithm_ids import PHASE1_RAW_ALGORITHM_ID
from production.polygon.runtime.diagnostics import classify_streams


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
POSTPROCESS = ROOT / "postprocess"
RUNTIME = HERE / "phase1_runtime.py"
DEFAULT_SOURCE_ROOT = ROOT / "output/production_polygon_source"
DEFAULT_OUTPUT_ROOT = ROOT / "output/production_polygon_phase1"
DEFAULT_PREDICTOR = POSTPROCESS / "models/polygon_point_predictor"
LABELS = ("女性器", "男性器", "結合部分")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--intervals", default="1,3,5,8,10,15")
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--label-workers",
        type=int,
        default=3,
        help=(
            "number of independent class jobs to run concurrently; each class "
            "still uses --num-workers DP workers"
        ),
    )
    parser.add_argument("--predictor-device", default="cuda")
    parser.add_argument("--predictor-model-dir", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="rebuild metrics/reports from completed runtime artifacts without rerunning DP",
    )
    return parser.parse_args()


def _discover_inputs(source_root: Path) -> dict[str, Path]:
    manifest_path = (
        source_root
        / "interval_10/production_raw/work/04_classwise_postprocess/classwise_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output: dict[str, Path] = {}
    for group in manifest["groups"]:
        pipeline_manifest = Path(group["pipeline_manifest"])
        details = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
        polygon_stage = next(
            stage
            for stage in details["stages"]
            if stage["id"] == "polygon_optimization"
        )
        optimizer = polygon_stage["metadata"]["optimizer"]
        source = Path(optimizer["input_sqlite"]).resolve()
        for label in group["labels"]:
            output[str(label)] = source
    missing = [label for label in LABELS if label not in output]
    if missing:
        raise RuntimeError(f"missing prepared Production inputs: {missing}")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(
    source: Path, output: Path, interval: int, args: argparse.Namespace
) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME),
        "__onefile_polygon_optimize",
        "--input-sqlite",
        str(source),
        "--output-dir",
        str(output),
        "--target-ratio",
        str(1.0 / float(interval)),
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


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _metrics(
    output: Path,
    source: Path,
    label: str,
    interval: int,
    wall: float,
    *,
    audit_name: str = "phase1_audit.json",
) -> dict[str, object]:
    exact_path = output / "exact/keyframe_exact_metrics.csv"
    rows: list[dict[str, str]] = []
    with exact_path.open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    recalls = [float(row["recall"]) for row in rows]
    ious = [float(row["iou"]) for row in rows]
    precisions = [float(row["precision"]) for row in rows]
    area_ratios = [
        float(row["pred_area"]) / max(float(row["gt_area"]), 1e-12) for row in rows
    ]

    stream_rows: list[dict[str, str]] = []
    with (output / "opt/stream_segments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        stream_rows.extend(csv.DictReader(handle))
    audit = json.loads((output / audit_name).read_text(encoding="utf-8"))
    recall_floor = float(audit["recall_floor"])
    diagnostics = classify_streams(rows, stream_rows, recall_floor=recall_floor)
    optimizer_fallback_keys = diagnostics.optimizer_fallback
    legacy_budget_diagnostic_keys = diagnostics.legacy_budget_diagnostic
    infeasible_keys = diagnostics.final_exact_infeasible
    feasible_rows = [
        row
        for row in rows
        if (str(row["track_id"]), int(row["run_id"])) not in infeasible_keys
    ]
    feasible_recalls = [float(row["recall"]) for row in feasible_rows]
    feasible_ious = [float(row["iou"]) for row in feasible_rows]
    feasible_area_ratios = [
        float(row["pred_area"]) / max(float(row["gt_area"]), 1e-12)
        for row in feasible_rows
    ]

    keyframes = json.loads((output / "opt/final_keyframes.json").read_text())
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    feasible_grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in keyframes:
        key = (str(row["track_id"]), int(row["run_id"]))
        grouped[key].append(int(row["frame"]))
        if key not in infeasible_keys:
            feasible_grouped[key].append(int(row["frame"]))
    total_span = sum(max(values) - min(values) for values in grouped.values() if values)
    interval_count = sum(max(len(values) - 1, 0) for values in grouped.values())
    feasible_span = sum(
        max(values) - min(values) for values in feasible_grouped.values() if values
    )
    feasible_interval_count = sum(
        max(len(values) - 1, 0) for values in feasible_grouped.values()
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    pred_sqlite = output / "pred/predictions.sqlite"
    with sqlite3.connect(
        f"file:{pred_sqlite.resolve()}?mode=ro", uri=True
    ) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        prediction_rows = int(
            connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
        )

    target_keys = sum(int(row["target_keyframes"]) for row in stream_rows)
    chosen_keys = sum(int(row["chosen_keyframes"]) for row in stream_rows)
    infeasible_streams = len(infeasible_keys)
    feasible_stream_rows = [
        row
        for row in stream_rows
        if (str(row["track_id"]), int(row["run_id"])) not in infeasible_keys
    ]
    optimizer = summary["optimizer_summary"]
    return {
        "label": label,
        "requested_interval": int(interval),
        "target_keyframes": target_keys,
        "keyframes": chosen_keys,
        "target_keyframe_error": chosen_keys - target_keys,
        "actual_mean_interval": total_span / max(interval_count, 1),
        "observation_rows": len(rows),
        "segment_count": len(grouped),
        "stream_count": len(stream_rows),
        "infeasible_streams": int(infeasible_streams),
        "optimizer_fallback_streams": len(optimizer_fallback_keys),
        "legacy_budget_diagnostic_streams": len(legacy_budget_diagnostic_keys),
        "hard_recall_feasible": int(infeasible_streams) == 0,
        "feasible_observation_rows": len(feasible_rows),
        "feasible_target_keyframes": sum(
            int(row["target_keyframes"]) for row in feasible_stream_rows
        ),
        "feasible_keyframes": sum(
            int(row["chosen_keyframes"]) for row in feasible_stream_rows
        ),
        "feasible_actual_mean_interval": feasible_span
        / max(feasible_interval_count, 1),
        "feasible_recall_min": min(feasible_recalls, default=1.0),
        "feasible_recall_violations": sum(
            value + 1e-12 < float(audit["recall_floor"]) for value in feasible_recalls
        ),
        "feasible_iou_mean": float(np.mean(feasible_ious)) if feasible_ious else 1.0,
        "feasible_iou_min": min(feasible_ious, default=1.0),
        "feasible_iou_q01": _quantile(feasible_ious, 0.01) if feasible_ious else 1.0,
        "feasible_area_ratio_max": max(feasible_area_ratios, default=1.0),
        "recall_mean": float(np.mean(recalls)),
        "recall_min": min(recalls),
        "recall_q01": _quantile(recalls, 0.01),
        "recall_q05": _quantile(recalls, 0.05),
        "recall_violations": sum(
            value + 1e-12 < float(audit["recall_floor"]) for value in recalls
        ),
        "iou_mean": float(np.mean(ious)),
        "iou_min": min(ious),
        "iou_q01": _quantile(ious, 0.01),
        "iou_q05": _quantile(ious, 0.05),
        "precision_mean": float(np.mean(precisions)),
        "area_ratio_mean": float(np.mean(area_ratios)),
        "area_ratio_q95": _quantile(area_ratios, 0.95),
        "area_ratio_q99": _quantile(area_ratios, 0.99),
        "area_ratio_max": max(area_ratios),
        "optimizer_seconds": float(optimizer["optimizer_seconds"]),
        "wall_seconds": float(wall),
        "interval_evaluations": int(optimizer["interval_eval_count"]),
        "interval_evaluation_frames": int(optimizer["interval_eval_frames"]),
        "candidate_frames": int(optimizer["candidate_frame_count_total"]),
        "mean_state_count": float(optimizer["mean_state_count"]),
        "pair_vote_seconds": float(
            optimizer["stage_seconds_total"]["pair_vote_refine_seconds"]
        ),
        "repair_seconds": float(
            optimizer["stage_seconds_total"]["exact_recall_repair_seconds"]
        ),
        "source_sqlite": str(source),
        "source_sha256": _sha256(source),
        "prediction_sqlite": str(pred_sqlite.resolve()),
        "keyframes_json": str((output / "opt/final_keyframes.json").resolve()),
        "sqlite_integrity": integrity,
        "prediction_rows": prediction_rows,
        "contract": audit,
    }


def _run_cell(
    source: Path, label: str, interval: int, args: argparse.Namespace
) -> dict[str, object]:
    root = args.output_root / f"interval_{interval}" / label
    report_path = root / "metrics.json"
    if args.reanalyze_existing:
        output = root / "runtime"
        if not (output / "summary.json").is_file():
            raise RuntimeError(f"completed runtime is missing: {output}")
        previous = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        wall = float(previous.get("wall_seconds", 0.0))
        metrics = _metrics(output, source, label, interval, wall)
        report_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metrics
    if report_path.is_file() and not args.force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    output = root / "runtime"
    command = _command(source, output, interval, args)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(POSTPROCESS), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.perf_counter()
    with (root / "run.log").open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=POSTPROCESS,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Phase 1 failed for {label} interval {interval}; see {root / 'run.log'}"
        )
    metrics = _metrics(output, source, label, interval, wall)
    report_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _aggregate(rows: list[dict[str, object]], interval: int) -> dict[str, object]:
    selected = [row for row in rows if int(row["requested_interval"]) == interval]
    count = sum(int(row["observation_rows"]) for row in selected)
    feasible_count = sum(int(row["feasible_observation_rows"]) for row in selected)
    keyframes = sum(int(row["keyframes"]) for row in selected)
    target_keys = sum(int(row["target_keyframes"]) for row in selected)
    # The exact aggregate interval is computed by summing each class's span,
    # reconstructed from interval_count = span / actual interval.
    interval_counts = [
        max(int(row["keyframes"]) - int(row["segment_count"]), 0) for row in selected
    ]
    spans = [
        float(row["actual_mean_interval"]) * value
        for row, value in zip(selected, interval_counts, strict=True)
    ]
    return {
        "label": "ALL",
        "requested_interval": interval,
        "target_keyframes": target_keys,
        "keyframes": keyframes,
        "target_keyframe_error": keyframes - target_keys,
        "actual_mean_interval": sum(spans) / max(sum(interval_counts), 1),
        "observation_rows": count,
        "recall_min": min(float(row["recall_min"]) for row in selected),
        "recall_violations": sum(int(row["recall_violations"]) for row in selected),
        "infeasible_streams": sum(int(row["infeasible_streams"]) for row in selected),
        "optimizer_fallback_streams": sum(
            int(row["optimizer_fallback_streams"]) for row in selected
        ),
        "legacy_budget_diagnostic_streams": sum(
            int(row["legacy_budget_diagnostic_streams"]) for row in selected
        ),
        "hard_recall_feasible": all(
            bool(row["hard_recall_feasible"]) for row in selected
        ),
        "feasible_observation_rows": feasible_count,
        "feasible_target_keyframes": sum(
            int(row["feasible_target_keyframes"]) for row in selected
        ),
        "feasible_keyframes": sum(int(row["feasible_keyframes"]) for row in selected),
        "feasible_actual_mean_interval": sum(
            float(row["feasible_actual_mean_interval"])
            * max(
                int(row["feasible_keyframes"])
                - (int(row["stream_count"]) - int(row["infeasible_streams"])),
                0,
            )
            for row in selected
        )
        / max(
            sum(
                max(
                    int(row["feasible_keyframes"])
                    - (int(row["stream_count"]) - int(row["infeasible_streams"])),
                    0,
                )
                for row in selected
            ),
            1,
        ),
        "feasible_recall_min": min(
            float(row["feasible_recall_min"]) for row in selected
        ),
        "feasible_recall_violations": sum(
            int(row["feasible_recall_violations"]) for row in selected
        ),
        "feasible_iou_mean": sum(
            float(row["feasible_iou_mean"]) * int(row["feasible_observation_rows"])
            for row in selected
        )
        / max(feasible_count, 1),
        "feasible_iou_q01_by_class_min": min(
            float(row["feasible_iou_q01"]) for row in selected
        ),
        "feasible_area_ratio_max": max(
            float(row["feasible_area_ratio_max"]) for row in selected
        ),
        "iou_mean": sum(
            float(row["iou_mean"]) * int(row["observation_rows"]) for row in selected
        )
        / max(count, 1),
        "iou_q01_by_class_min": min(float(row["iou_q01"]) for row in selected),
        "area_ratio_max": max(float(row["area_ratio_max"]) for row in selected),
        "optimizer_seconds": sum(float(row["optimizer_seconds"]) for row in selected),
        "wall_seconds": sum(float(row["wall_seconds"]) for row in selected),
    }


def _write_report(
    root: Path, rows: list[dict[str, object]], aggregates: list[dict[str, object]]
) -> None:
    columns = [
        "label",
        "requested_interval",
        "actual_mean_interval",
        "target_keyframes",
        "keyframes",
        "target_keyframe_error",
        "iou_mean",
        "iou_q01",
        "iou_min",
        "recall_min",
        "recall_violations",
        "infeasible_streams",
        "hard_recall_feasible",
        "feasible_observation_rows",
        "feasible_target_keyframes",
        "feasible_keyframes",
        "feasible_actual_mean_interval",
        "feasible_recall_min",
        "feasible_recall_violations",
        "feasible_iou_mean",
        "feasible_iou_q01",
        "feasible_iou_min",
        "feasible_area_ratio_max",
        "area_ratio_max",
        "optimizer_seconds",
        "wall_seconds",
    ]
    with (root / "phase1_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Phase 1 — Production raw-only hard Recall, pair-vote off",
        "",
        "All metrics use SQLite polygon geometry only. No video frame was opened.",
        "",
        "## Aggregate movable range",
        "",
        "The quality columns below describe only streams for which a hard-Recall raw-only path exists. Infeasible streams are reported separately and were never silently repaired.",
        "",
        "| Requested interval | Feasible actual interval | Feasible target keys | Feasible actual keys | Feasible mean IoU | Feasible min Recall | Feasible violations | Infeasible streams | Optimizer s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {requested_interval} | {feasible_actual_mean_interval:.3f} | "
            "{feasible_target_keyframes} | {feasible_keyframes} | "
            "{feasible_iou_mean:.6f} | {feasible_recall_min:.6f} | "
            "{feasible_recall_violations} | {infeasible_streams} | "
            "{optimizer_seconds:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Classwise results",
            "",
            "| Class | Requested | Feasible actual | Feasible keys | Mean IoU | q01 IoU | Min IoU | Min Recall | Violations | Infeasible streams | Max area ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {requested_interval} | "
            "{feasible_actual_mean_interval:.3f} | {feasible_keyframes} | "
            "{feasible_iou_mean:.6f} | {feasible_iou_q01:.6f} | "
            "{feasible_iou_min:.6f} | {feasible_recall_min:.6f} | "
            "{feasible_recall_violations} | {infeasible_streams} | "
            "{feasible_area_ratio_max:.3f} |".format(**row)
        )
    (root / "PHASE1_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.predictor_model_dir = args.predictor_model_dir.expanduser().resolve()
    intervals = [
        int(value.strip()) for value in args.intervals.split(",") if value.strip()
    ]
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    if any(value < 1 for value in intervals):
        raise ValueError("intervals must be >= 1")
    if any(value not in LABELS for value in labels):
        raise ValueError(f"labels must be selected from {LABELS}")
    if not 0.0 < args.recall_floor <= 1.0:
        raise ValueError("recall-floor must be in (0, 1]")
    if args.num_workers < 1:
        raise ValueError("num-workers must be >= 1")
    if args.label_workers < 1:
        raise ValueError("label-workers must be >= 1")
    sources = _discover_inputs(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for interval in intervals:

        def run_label(label: str) -> dict[str, object]:
            print(f"[phase1] label={label} requested_interval={interval}", flush=True)
            row = _run_cell(sources[label], label, interval, args)
            print(
                "[phase1-result] "
                f"label={label} requested={interval} "
                f"actual={row['actual_mean_interval']:.3f} keys={row['keyframes']} "
                f"iou={row['iou_mean']:.6f} recall_min={row['recall_min']:.6f}",
                flush=True,
            )
            return row

        effective_label_workers = min(int(args.label_workers), len(labels))
        if effective_label_workers == 1:
            interval_rows = [run_label(label) for label in labels]
        else:
            # Each class has disjoint source/output artifacts.  Running these
            # subprocesses concurrently changes scheduling only; the per-run
            # candidate construction, DP and ordered serialization are kept
            # byte-deterministic.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=effective_label_workers
            ) as executor:
                interval_rows = list(executor.map(run_label, labels))
        rows.extend(interval_rows)
    aggregates = [_aggregate(rows, interval) for interval in intervals]
    payload = {
        "schema_version": 1,
        "production_support": True,
        "privacy": "SQLite geometry only; no video frame was opened.",
        "algorithm": PHASE1_RAW_ALGORITHM_ID,
        "recall_floor": args.recall_floor,
        "intervals": intervals,
        "labels": labels,
        "execution": {
            "label_workers": min(int(args.label_workers), len(labels)),
            "dp_workers_per_label": int(args.num_workers),
            "maximum_concurrent_dp_workers": min(int(args.label_workers), len(labels))
            * int(args.num_workers),
        },
        "source_inputs": {label: str(sources[label]) for label in labels},
        "rows": rows,
        "aggregates": aggregates,
    }
    (args.output_root / "phase1_matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(args.output_root, rows, aggregates)
    if any(int(row["infeasible_streams"]) for row in rows):
        print(
            "[phase1-conclusion] one or more raw-only streams have no feasible "
            "hard-Recall path; diagnostic all-raw paths are included and "
            "reported, not silently repaired.",
            flush=True,
        )
    print(json.dumps({"aggregates": aggregates}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
