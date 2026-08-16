#!/usr/bin/env python3
"""Audit and compare the V3 CPU-exact and default-CUDA benchmark outputs.

This analyzer is deliberately geometry-only.  It reads SQLite polygons and
optimizer CSV files, and never opens the source videos or raster frames.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import native_interval_metrics
import numpy as np

from .benchmark_v3_exact_vs_default import DEFAULT_OUTPUT_ROOT
from .validation import audit_sqlite


LABELS = ("女性器", "男性器", "結合部分")
MODES = ("default_cuda", "cpu_exact")
ERROR_THRESHOLDS = (0.999, 0.99, 0.95, 0.90)


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: None for key in ("min", "q001", "q01", "q05", "median", "mean", "max")}
    return {
        "min": float(np.min(array)),
        "q001": float(np.quantile(array, 0.001)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _prediction_db(phase2_root: Path, label: str) -> Path:
    matches = list(
        phase2_root.glob(f"polygon14_keyframe_v1/{label}/runtime/pred/predictions.sqlite")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction DB for {label}: {matches}")
    return matches[0]


def _exact_csv(phase2_root: Path, label: str) -> Path:
    matches = list(
        phase2_root.glob(
            f"polygon14_keyframe_v1/{label}/runtime/exact/keyframe_exact_metrics.csv"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact CSV for {label}: {matches}")
    return matches[0]


def _iter_predictions(path: Path):
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT frame,track_id,polygons FROM masks "
            "ORDER BY frame,CAST(track_id AS INTEGER),track_id"
        )
        for frame, track_id, polygons in cursor:
            yield int(frame), str(track_id), str(polygons)
    finally:
        connection.close()


def _read_exact(path: Path) -> dict[tuple[int, str, str], dict[str, float]]:
    rows: dict[tuple[int, str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame"]), str(row["track_id"]), str(row["run_id"]))
            rows[key] = {
                name: float(row[name])
                for name in ("recall", "precision", "iou", "gt_area", "pred_area")
            }
    return rows


def _compare_label(
    default_db: Path,
    exact_db: Path,
    default_csv: Path,
    exact_csv: Path,
    *,
    run_id: str,
    interval: int,
    label: str,
    worst_heap: list[tuple[float, str, dict[str, Any]]],
) -> dict[str, Any]:
    mutual_ious: list[float] = []
    area_ratios: list[float] = []
    exact_json = 0
    row_count = 0
    left = _iter_predictions(default_db)
    right = _iter_predictions(exact_db)
    while True:
        try:
            a = next(left)
        except StopIteration:
            a = None
        try:
            b = next(right)
        except StopIteration:
            b = None
        if a is None or b is None:
            if a != b:
                raise RuntimeError(
                    f"prediction row count mismatch: {run_id}/{interval}/{label}"
                )
            break
        if a[:2] != b[:2]:
            raise RuntimeError(
                f"prediction identity mismatch: {run_id}/{interval}/{label}: "
                f"{a[:2]} != {b[:2]}"
            )
        frame, track_id = a[:2]
        if a[2] == b[2]:
            exact_json += 1
            iou = 1.0
            area_ratio = 1.0
        else:
            metrics = dict(
                native_interval_metrics.exact_metrics(
                    json.loads(a[2]), json.loads(b[2])
                )
            )
            iou = float(metrics["iou"])
            default_area = float(metrics["gt_area"])
            area_ratio = float(metrics["pred_area"]) / max(default_area, 1.0)
        mutual_ious.append(iou)
        area_ratios.append(area_ratio)
        row_count += 1
        item = {
            "run_id": run_id,
            "target_interval": interval,
            "label": label,
            "frame": frame,
            "track_id": track_id,
            "mutual_iou": iou,
            "cpu_to_default_area_ratio": area_ratio,
            "polygon_json_exact": a[2] == b[2],
        }
        identity = f"{run_id}:{interval}:{label}:{frame}:{track_id}"
        heap_item = (-iou, identity, item)
        if len(worst_heap) < 200:
            heapq.heappush(worst_heap, heap_item)
        elif iou < -worst_heap[0][0]:
            heapq.heapreplace(worst_heap, heap_item)

    default_metrics = _read_exact(default_csv)
    exact_metrics = _read_exact(exact_csv)
    if default_metrics.keys() != exact_metrics.keys():
        raise RuntimeError(f"exact metric identities differ: {run_id}/{interval}/{label}")
    iou_deltas = []
    recall_deltas = []
    for key in default_metrics:
        iou_deltas.append(default_metrics[key]["iou"] - exact_metrics[key]["iou"])
        recall_deltas.append(
            default_metrics[key]["recall"] - exact_metrics[key]["recall"]
        )
    return {
        "run_id": run_id,
        "target_interval": interval,
        "label": label,
        "rows": row_count,
        "polygon_json_exact_rows": exact_json,
        "polygon_json_exact_rate": exact_json / max(row_count, 1),
        "mutual_iou": _quantiles(mutual_ious),
        "mutual_iou_below": {
            str(threshold): int(sum(value < threshold for value in mutual_ious))
            for threshold in ERROR_THRESHOLDS
        },
        "cpu_to_default_area_ratio": _quantiles(area_ratios),
        "quality_iou_delta_default_minus_exact": _quantiles(iou_deltas),
        "quality_recall_delta_default_minus_exact": _quantiles(recall_deltas),
        "_mutual_iou_values": mutual_ious,
        "_area_ratio_values": area_ratios,
        "_quality_iou_delta_values": iou_deltas,
        "_quality_recall_delta_values": recall_deltas,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_run(result: dict[str, Any]) -> dict[str, Any]:
    timing = result["timing"]
    quality = result["quality"]
    audit = result["final_audit"]
    phase2_root = Path(str(result["artifacts"]["phase2_root"]))
    matrix = json.loads(
        (phase2_root / "phase2_matrix.json").read_text(encoding="utf-8")
    )
    optimizer_seconds = max(
        (float(item["wall_seconds"]) for item in matrix.get("rows", [])),
        default=float(timing["optimizer_seconds"]),
    )
    raw_frames = int(result["raw_frames"])
    shared_seconds = float(timing["shared_upstream_seconds"])
    export_seconds = float(timing["export_seconds"])
    return {
        "run_id": result["run_id"],
        "mode": result["mode"],
        "interval_evaluation": result["interval_evaluation"],
        "target_interval": result["target_interval"],
        "raw_frames": result["raw_frames"],
        "raw_detections": result["raw_detections"],
        "optimizer_seconds": optimizer_seconds,
        "optimizer_wrapper_seconds": timing.get(
            "optimizer_wrapper_seconds", timing["optimizer_seconds"]
        ),
        "export_seconds": export_seconds,
        "shared_upstream_seconds": shared_seconds,
        "optimizer_video_fps": raw_frames / max(optimizer_seconds, 1e-9),
        "equivalent_end_to_end_fps": raw_frames
        / max(shared_seconds + optimizer_seconds + export_seconds, 1e-9),
        "observation_rows": quality["observation_rows"],
        "keyframes": quality["keyframes"],
        "actual_mean_interval": quality["actual_mean_interval"],
        "recall_min": quality["recall_min"],
        "recall_violations": quality["recall_violations"],
        "infeasible_streams": quality["infeasible_streams"],
        "iou_mean": quality["iou_mean"],
        "iou_q01_by_class_min": quality["iou_q01_by_class_min"],
        "area_ratio_max": quality["area_ratio_max"],
        "sqlite_integrity_ok": audit["integrity_ok"],
        "sqlite_foreign_key_errors": audit["foreign_key_error_count"],
        "sqlite_schema_sha256": result["final_schema_fingerprint"],
        "sqlite_path": result["artifacts"]["result_sqlite"],
        "sqlite_sha256": result["artifacts"]["result_sqlite_sha256"],
    }


def _aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for interval in sorted({int(row["target_interval"]) for row in rows}):
        for mode in MODES:
            selected = [
                row
                for row in rows
                if int(row["target_interval"]) == interval and row["mode"] == mode
            ]
            frames = sum(int(row["raw_frames"]) for row in selected)
            observations = sum(int(row["observation_rows"]) for row in selected)
            optimizer_seconds = sum(float(row["optimizer_seconds"]) for row in selected)
            per_video_fps = [
                float(row["optimizer_video_fps"]) for row in selected
            ]
            e2e_seconds = sum(
                float(row["optimizer_seconds"])
                + float(row["export_seconds"])
                + float(row["shared_upstream_seconds"])
                for row in selected
            )
            output.append(
                {
                    "target_interval": interval,
                    "mode": mode,
                    "videos": len(selected),
                    "frames": frames,
                    "optimizer_seconds": optimizer_seconds,
                    "optimizer_fps": frames / optimizer_seconds,
                    "optimizer_video_fps_min": min(per_video_fps),
                    "optimizer_video_fps_median": float(np.median(per_video_fps)),
                    "optimizer_video_fps_max": max(per_video_fps),
                    "equivalent_end_to_end_seconds": e2e_seconds,
                    "equivalent_end_to_end_fps": frames / e2e_seconds,
                    "weighted_iou_mean": sum(
                        float(row["iou_mean"]) * int(row["observation_rows"])
                        for row in selected
                    )
                    / max(observations, 1),
                    "recall_min": min(float(row["recall_min"]) for row in selected),
                    "recall_violations": sum(
                        int(row["recall_violations"]) for row in selected
                    ),
                    "keyframes": sum(int(row["keyframes"]) for row in selected),
                    "actual_interval_weighted": sum(
                        float(row["actual_mean_interval"])
                        * int(row["observation_rows"])
                        for row in selected
                    )
                    / max(observations, 1),
                }
            )
    return output


def _aggregate_errors(
    rows: list[dict[str, Any]],
    values: dict[int, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    output = []
    for interval in sorted({int(row["target_interval"]) for row in rows}):
        selected = [
            row
            for row in rows
            if int(row["target_interval"]) == interval and int(row["rows"]) > 0
        ]
        if not selected:
            continue
        total = sum(int(row["rows"]) for row in selected)
        interval_values = values[interval]
        mutual = interval_values["mutual_iou"]
        quality_iou_delta = interval_values["quality_iou_delta"]
        quality_recall_delta = interval_values["quality_recall_delta"]
        area_ratio = interval_values["area_ratio"]
        output.append(
            {
                "target_interval": interval,
                "rows": total,
                "polygon_json_exact_rate": sum(
                    int(row["polygon_json_exact_rows"]) for row in selected
                )
                / max(total, 1),
                **{f"mutual_iou_{key}": value for key, value in _quantiles(mutual).items()},
                **{
                    f"mutual_iou_below_{threshold}": sum(
                        int(row["mutual_iou_below"][str(threshold)])
                        for row in selected
                    )
                    for threshold in ERROR_THRESHOLDS
                },
                **{
                    f"quality_iou_delta_{key}": value
                    for key, value in _quantiles(quality_iou_delta).items()
                },
                **{
                    f"quality_recall_delta_{key}": value
                    for key, value in _quantiles(quality_recall_delta).items()
                },
                **{
                    f"cpu_to_default_area_ratio_{key}": value
                    for key, value in _quantiles(area_ratio).items()
                },
            }
        )
    return output


def _collect_recall_violations(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    identities: dict[tuple[str, int, str], set[tuple[str, int, str]]] = {}
    for result in results:
        run_id = str(result["run_id"])
        interval = int(result["target_interval"])
        mode = str(result["mode"])
        phase2_root = Path(str(result["artifacts"]["phase2_root"]))
        current: set[tuple[str, int, str]] = set()
        for label in LABELS:
            with _exact_csv(phase2_root, label).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    recall = float(row["recall"])
                    if recall >= 0.97:
                        continue
                    identity = (label, int(row["frame"]), str(row["track_id"]))
                    current.add(identity)
                    details.append(
                        {
                            "run_id": run_id,
                            "target_interval": interval,
                            "mode": mode,
                            "label": label,
                            "frame": identity[1],
                            "track_id": identity[2],
                            "run_segment_id": row["run_id"],
                            "has_keyframe": int(row["has_keyframe"]),
                            "recall": recall,
                            "iou": float(row["iou"]),
                        }
                    )
        identities[(run_id, interval, mode)] = current

    for row in details:
        identity = (str(row["label"]), int(row["frame"]), str(row["track_id"]))
        other_mode = "cpu_exact" if row["mode"] == "default_cuda" else "default_cuda"
        other = identities[(str(row["run_id"]), int(row["target_interval"]), other_mode)]
        row["also_violates_other_mode"] = identity in other

    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[
            (
                str(row["run_id"]),
                int(row["target_interval"]),
                str(row["mode"]),
                str(row["label"]),
            )
        ].append(row)
    summary = []
    for (run_id, interval, mode, label), rows in sorted(groups.items()):
        summary.append(
            {
                "run_id": run_id,
                "target_interval": interval,
                "mode": mode,
                "label": label,
                "violations": len(rows),
                "other_mode_shared": sum(
                    bool(row["also_violates_other_mode"]) for row in rows
                ),
                "mode_only": sum(
                    not bool(row["also_violates_other_mode"]) for row in rows
                ),
                "tracks": len({str(row["track_id"]) for row in rows}),
                "first_frame": min(int(row["frame"]) for row in rows),
                "last_frame": max(int(row["frame"]) for row in rows),
                "recall_min": min(float(row["recall"]) for row in rows),
            }
        )
    return details, summary


def _write_analysis_sqlite(
    path: Path,
    runs: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    error_aggregates: list[dict[str, Any]],
    worst: list[dict[str, Any]],
    recall_violations: list[dict[str, Any]],
    recall_violation_summary: list[dict[str, Any]],
) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        def write_table(name: str, values: list[dict[str, Any]]) -> None:
            if not values:
                return
            columns = list(values[0])
            connection.execute(
                f"CREATE TABLE {name} ("
                + ",".join(f'"{column}"' for column in columns)
                + ")"
            )
            connection.executemany(
                f"INSERT INTO {name} VALUES ("
                + ",".join("?" for _ in columns)
                + ")",
                [
                    tuple(
                        json.dumps(row[column], ensure_ascii=False, sort_keys=True)
                        if isinstance(row[column], (dict, list))
                        else row[column]
                        for column in columns
                    )
                    for row in values
                ],
            )

        write_table("runs", runs)
        write_table("aggregate_runtime_quality", aggregates)
        write_table("paired_errors", errors)
        write_table("aggregate_errors", error_aggregates)
        write_table("worst_output_differences", worst)
        write_table("recall_violations", recall_violations)
        write_table("recall_violation_summary", recall_violation_summary)
        connection.commit()
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    summary_path = root / "batch_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"completed batch summary is missing: {summary_path}")
    batch = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = int(batch["expected_final_sqlites"])
    if batch.get("status") != "complete" or int(batch["completed"]) != expected:
        raise RuntimeError(f"batch is incomplete: {batch.get('completed')}/{expected}")
    run_rows = [_flatten_run(result) for result in batch["results"]]
    if len(run_rows) != expected:
        raise RuntimeError(f"result count mismatch: {len(run_rows)} != {expected}")

    schema_hashes = {str(row["sqlite_schema_sha256"]) for row in run_rows}
    if len(schema_hashes) != 1:
        raise RuntimeError(f"final SQLite schema drift: {sorted(schema_hashes)}")
    for row in run_rows:
        audit = audit_sqlite(Path(str(row["sqlite_path"])))
        if not audit.ok:
            raise RuntimeError(f"final SQLite audit failed: {audit.to_dict()}")

    by_key = {
        (str(result["run_id"]), int(result["target_interval"]), str(result["mode"])): result
        for result in batch["results"]
    }
    errors: list[dict[str, Any]] = []
    worst_heap: list[tuple[float, str, dict[str, Any]]] = []
    error_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "mutual_iou": [],
            "area_ratio": [],
            "quality_iou_delta": [],
            "quality_recall_delta": [],
        }
    )
    run_ids = sorted({str(result["run_id"]) for result in batch["results"]})
    intervals = sorted({int(result["target_interval"]) for result in batch["results"]})
    for run_id in run_ids:
        for interval in intervals:
            default = by_key[(run_id, interval, "default_cuda")]
            exact = by_key[(run_id, interval, "cpu_exact")]
            default_root = Path(str(default["artifacts"]["phase2_root"]))
            exact_root = Path(str(exact["artifacts"]["phase2_root"]))
            for label in LABELS:
                detail = _compare_label(
                        _prediction_db(default_root, label),
                        _prediction_db(exact_root, label),
                        _exact_csv(default_root, label),
                        _exact_csv(exact_root, label),
                        run_id=run_id,
                        interval=interval,
                        label=label,
                        worst_heap=worst_heap,
                    )
                error_values[interval]["mutual_iou"].extend(
                    detail.pop("_mutual_iou_values")
                )
                error_values[interval]["area_ratio"].extend(
                    detail.pop("_area_ratio_values")
                )
                error_values[interval]["quality_iou_delta"].extend(
                    detail.pop("_quality_iou_delta_values")
                )
                error_values[interval]["quality_recall_delta"].extend(
                    detail.pop("_quality_recall_delta_values")
                )
                errors.append(detail)
            print(f"[compared] {run_id} interval={interval}", flush=True)

    worst = [item for _, _, item in sorted(worst_heap, key=lambda pair: -pair[0])]
    aggregates = _aggregate_runs(run_rows)
    error_aggregates = _aggregate_errors(errors, error_values)
    recall_violations, recall_violation_summary = _collect_recall_violations(
        batch["results"]
    )
    total_recall_violations = sum(
        int(row["recall_violations"]) for row in run_rows
    )
    validation = {
        "status": (
            "pass" if total_recall_violations == 0 else "pass_with_quality_blocker"
        ),
        "expected_results": expected,
        "actual_results": len(run_rows),
        "final_sqlites": len(list((root / "final_sqlite").glob("*/*/*.sqlite"))),
        "all_sqlite_integrity_ok": all(bool(row["sqlite_integrity_ok"]) for row in run_rows),
        "total_foreign_key_errors": sum(int(row["sqlite_foreign_key_errors"]) for row in run_rows),
        "production_recall_gate": (
            "pass" if total_recall_violations == 0 else "fail"
        ),
        "total_recall_violations_across_configurations": total_recall_violations,
        "schema_fingerprints": sorted(schema_hashes),
        "paired_prediction_rows": sum(int(row["rows"]) for row in errors),
        "native_metric_implementation": native_interval_metrics.implementation,
        "privacy": "No video or frame image was opened; comparison used polygon geometry only.",
    }
    analysis = {
        "validation": validation,
        "aggregate_runtime_quality": aggregates,
        "aggregate_paired_errors": error_aggregates,
        "paired_errors_by_video_label": errors,
        "worst_output_differences": worst,
        "recall_violation_summary": recall_violation_summary,
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(root / "runs.csv", run_rows)
    _write_csv(root / "aggregate_runtime_quality.csv", aggregates)
    _write_csv(root / "paired_errors_by_video_label.csv", errors)
    _write_csv(root / "aggregate_paired_errors.csv", error_aggregates)
    _write_csv(root / "worst_output_differences.csv", worst)
    _write_csv(root / "recall_violations.csv", recall_violations)
    _write_csv(root / "recall_violation_summary.csv", recall_violation_summary)
    _write_analysis_sqlite(
        root / "analysis.sqlite",
        run_rows,
        aggregates,
        errors,
        error_aggregates,
        worst,
        recall_violations,
        recall_violation_summary,
    )
    (root / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis["aggregate_runtime_quality"], ensure_ascii=False, indent=2))
    print(json.dumps(analysis["aggregate_paired_errors"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
