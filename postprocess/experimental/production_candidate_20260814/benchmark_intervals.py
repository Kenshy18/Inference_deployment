#!/usr/bin/env python3
"""Collect an interval 1..6 comparison against legacy Production.

The evaluator only opens SQLite/JSON geometry artifacts.  It never decodes a
video frame.  Arm-local exact metrics measure approximation quality against
each arm's tracked source; common-raw metrics measure both arms against the
same canonical score-filtered AI masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from nms.component_aware import _raster_mask

from .config import LABELS, POLYGON_PROFILE_ID
from .polygon.topology import polygon_is_simple
from .validation import audit_sqlite, schema_fingerprint


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(paths: Iterable[Path]) -> dict[str, float | int]:
    values: dict[str, list[float]] = {
        "recall": [],
        "precision": [],
        "iou": [],
        "area_ratio": [],
    }
    source_keyframes = 0
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                recall = float(row["recall"])
                precision = float(row["precision"])
                iou = float(row["iou"])
                gt_area = float(row["gt_area"])
                pred_area = float(row["pred_area"])
                values["recall"].append(recall)
                values["precision"].append(precision)
                values["iou"].append(iou)
                values["area_ratio"].append(
                    1.0 if gt_area <= 0.0 else pred_area / gt_area
                )
                source_keyframes += int(row.get("has_keyframe", 0))
    if not values["iou"]:
        raise RuntimeError("exact metrics contain no evaluated rows")
    result: dict[str, float | int] = {
        "evaluated_rows": len(values["iou"]),
        "source_keyframes": source_keyframes,
    }
    for metric in ("recall", "precision", "iou"):
        current = values[metric]
        result[f"{metric}_mean"] = sum(current) / len(current)
        result[f"{metric}_min"] = min(current)
        result[f"{metric}_q01"] = _quantile(current, 0.01)
        result[f"{metric}_q05"] = _quantile(current, 0.05)
    result["recall_violations"] = sum(
        value < 0.97 - 1e-9 for value in values["recall"]
    )
    area = values["area_ratio"]
    result.update(
        {
            "area_ratio_mean": sum(area) / len(area),
            "area_ratio_q95": _quantile(area, 0.95),
            "area_ratio_q99": _quantile(area, 0.99),
            "area_ratio_max": max(area),
        }
    )
    return result


def _prediction_paths(root: Path, arm: str, interval: int) -> list[tuple[str | None, Path]]:
    if arm == "legacy_production":
        stage = next((root / arm / f"interval_{interval}").glob("*_polygon_optimization"))
        return [(None, stage / "predictions.sqlite")]
    interval_root = root / arm / f"interval_{interval}" / POLYGON_PROFILE_ID
    return [
        (label, interval_root / label / "runtime/pred/predictions.sqlite")
        for label in LABELS
    ]


def _keyframe_paths(root: Path, arm: str, interval: int) -> list[Path]:
    if arm == "legacy_production":
        stage = next((root / arm / f"interval_{interval}").glob("*_polygon_optimization"))
        return [stage / "vendor_output/opt/final_keyframes.json"]
    interval_root = root / arm / f"interval_{interval}" / POLYGON_PROFILE_ID
    return [
        interval_root / label / "runtime/opt/final_keyframes.json"
        for label in LABELS
    ]


def _exact_paths(root: Path, arm: str, interval: int) -> list[Path]:
    if arm == "legacy_production":
        stage = next((root / arm / f"interval_{interval}").glob("*_polygon_optimization"))
        return [stage / "vendor_output/exact/keyframe_exact_metrics.csv"]
    interval_root = root / arm / f"interval_{interval}" / POLYGON_PROFILE_ID
    return [
        interval_root / label / "runtime/exact/keyframe_exact_metrics.csv"
        for label in LABELS
    ]


def _optimizer_runtime(root: Path, arm: str, interval: int) -> tuple[float, float | None]:
    if arm == "legacy_production":
        manifest = json.loads(
            (root / arm / f"interval_{interval}" / "pipeline_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        stage = next(
            row
            for row in manifest["stages"]
            if row["implementation"] == "approximation.polygon.production_v22"
        )
        reported = json.loads(
            Path(stage["artifacts"]["polygon_v22_summary"]).read_text(encoding="utf-8")
        )["optimizer_summary"].get("optimizer_seconds")
        return float(stage["elapsed_seconds"]), (
            None if reported is None else float(reported)
        )
    matrix = json.loads(
        (root / arm / f"interval_{interval}" / "phase2_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = matrix["completed_profiles"][-1]
    return float(aggregate["profile_wall_seconds"]), float(
        aggregate["optimizer_seconds"]
    )


def _reported_interval(root: Path, arm: str, interval: int) -> float | None:
    if arm == "legacy_production":
        stage = next((root / arm / f"interval_{interval}").glob("*_polygon_optimization"))
        summary = json.loads(
            (stage / "vendor_output/summary.json").read_text(encoding="utf-8")
        )
        rate = float(summary["optimizer_summary"]["mean_keyframe_rate"])
        return None if rate <= 0.0 else 1.0 / rate
    matrix = json.loads(
        (root / arm / f"interval_{interval}" / "phase2_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    return float(matrix["completed_profiles"][-1]["actual_mean_interval"])


def _software_sqlite(root: Path, arm: str, interval: int) -> Path:
    if arm == "legacy_production":
        standardized = (
            root
            / arm
            / "software_sqlite"
            / f"interval_{interval}"
            / "12月KPI動画.sqlite"
        )
        if standardized.is_file():
            return standardized
        candidate = list(
            (root / arm / f"interval_{interval}").glob(
                "*_integrated_result_sqlite/result.sqlite"
            )
        )
        if len(candidate) != 1:
            raise RuntimeError(f"legacy result SQLite not unique: {candidate}")
        return candidate[0]
    return root / arm / "software_sqlite" / f"interval_{interval}" / "12月KPI動画.sqlite"


def _load_predictions(
    root: Path, arm: str, interval: int
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, int], int]:
    result = {label: defaultdict(list) for label in LABELS}
    invalid = {label: 0 for label in LABELS}
    row_count = 0
    for fixed_label, path in _prediction_paths(root, arm, interval):
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(masks)")}
            query = "SELECT frame,polygons"
            if fixed_label is None and "label" in columns:
                query += ",label"
            query += " FROM masks ORDER BY frame,track_id"
            for row in db.execute(query):
                row_count += 1
                frame = int(row[0])
                label = fixed_label if fixed_label is not None else str(row[2])
                if label not in result:
                    continue
                polygons = []
                for value in json.loads(str(row[1])):
                    polygon = np.asarray(value, dtype=np.float32)
                    if len(polygon) >= 3:
                        polygons.append(polygon)
                        if not polygon_is_simple(polygon):
                            invalid[label] += 1
                result[label][frame].extend(polygons)
    return (
        {label: dict(frames) for label, frames in result.items()},
        invalid,
        row_count,
    )


def _raster_pair(
    raw_detections: list[dict[str, Any]], final_polygons: list[np.ndarray]
) -> tuple[int, int, int]:
    raw = [value for det in raw_detections if (value := _raster_mask(det)) is not None]
    final_detection = {"polygons": [polygon.tolist() for polygon in final_polygons]}
    final = _raster_mask(final_detection) if final_polygons else None
    rasters = raw + ([] if final is None else [final])
    if not rasters:
        return 0, 0, 0
    left = min(value.left for value in rasters)
    top = min(value.top for value in rasters)
    right = max(value.right for value in rasters)
    bottom = max(value.bottom for value in rasters)
    canvas_shape = (bottom - top + 1, right - left + 1)

    def union(values: list[Any]) -> np.ndarray:
        canvas = np.zeros(canvas_shape, dtype=np.uint8)
        for value in values:
            y1, y2 = value.top - top, value.bottom - top + 1
            x1, x2 = value.left - left, value.right - left + 1
            np.maximum(canvas[y1:y2, x1:x2], value.mask, out=canvas[y1:y2, x1:x2])
        return canvas

    raw_mask = union(raw)
    final_mask = union([] if final is None else [final])
    return (
        int(raw_mask.sum()),
        int(final_mask.sum()),
        int(np.logical_and(raw_mask, final_mask).sum()),
    )


def _common_raw_metrics(
    scored_jsonl: Path,
    predictions: dict[str, dict[int, list[np.ndarray]]],
) -> dict[str, float | int]:
    cells = {
        label: {
            "raw": 0,
            "final": 0,
            "intersection": 0,
            "raw_nonempty_frames": 0,
            "final_only_frames": 0,
            "recall": [],
            "precision": [],
            "iou": [],
        }
        for label in LABELS
    }
    with scored_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["frame_index"])
            raw_by_label = {label: [] for label in LABELS}
            for detection in record.get("detections", []):
                label = str(detection.get("label") or detection.get("class_name") or "")
                if label in raw_by_label:
                    raw_by_label[label].append(detection)
            for label in LABELS:
                raw_area, final_area, intersection = _raster_pair(
                    raw_by_label[label], predictions[label].get(frame, [])
                )
                if raw_area == 0 and final_area == 0:
                    continue
                cell = cells[label]
                cell["raw"] += raw_area
                cell["final"] += final_area
                cell["intersection"] += intersection
                union = raw_area + final_area - intersection
                if raw_area > 0:
                    cell["raw_nonempty_frames"] += 1
                    cell["recall"].append(intersection / raw_area)
                    cell["precision"].append(
                        1.0 if final_area == 0 else intersection / final_area
                    )
                    cell["iou"].append(1.0 if union == 0 else intersection / union)
                elif final_area > 0:
                    cell["final_only_frames"] += 1
    result: dict[str, float | int] = {
        "raw_pixels": sum(int(cell["raw"]) for cell in cells.values()),
        "final_pixels": sum(int(cell["final"]) for cell in cells.values()),
        "intersection_pixels": sum(
            int(cell["intersection"]) for cell in cells.values()
        ),
        "raw_nonempty_frames": sum(
            int(cell["raw_nonempty_frames"]) for cell in cells.values()
        ),
        "final_only_frames": sum(
            int(cell["final_only_frames"]) for cell in cells.values()
        ),
    }
    raw = int(result["raw_pixels"])
    final = int(result["final_pixels"])
    intersection = int(result["intersection_pixels"])
    union = raw + final - intersection
    result.update(
        {
            "pixel_weighted_recall": 1.0 if raw == 0 else intersection / raw,
            "pixel_weighted_precision": (
                1.0 if final == 0 else intersection / final
            ),
            "pixel_weighted_iou": 1.0 if union == 0 else intersection / union,
        }
    )
    all_recalls = list(
        itertools.chain.from_iterable(cell["recall"] for cell in cells.values())
    )
    all_ious = list(
        itertools.chain.from_iterable(cell["iou"] for cell in cells.values())
    )
    result.update(
        {
            "frame_recall_mean": sum(all_recalls) / len(all_recalls),
            "frame_recall_q01": _quantile(all_recalls, 0.01),
            "frame_recall_min": min(all_recalls),
            "frame_iou_mean": sum(all_ious) / len(all_ious),
            "frame_iou_q01": _quantile(all_ious, 0.01),
            "frame_iou_min": min(all_ious),
        }
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path, scored_jsonl: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for interval in range(1, 7):
        for arm in ("legacy_production", "production_candidate_20260814"):
            exact = _read_exact(_exact_paths(root, arm, interval))
            keyframes = sum(
                len(json.loads(path.read_text(encoding="utf-8")))
                for path in _keyframe_paths(root, arm, interval)
            )
            predictions, invalid, prediction_rows = _load_predictions(
                root, arm, interval
            )
            runtime, reported_runtime = _optimizer_runtime(root, arm, interval)
            sqlite_path = _software_sqlite(root, arm, interval)
            sqlite_audit = audit_sqlite(sqlite_path)
            if not sqlite_audit.ok:
                raise RuntimeError(f"SQLite validation failed: {sqlite_path}")
            common = _common_raw_metrics(scored_jsonl, predictions)
            actual = prediction_rows / keyframes
            row: dict[str, Any] = {
                "arm": arm,
                "target_interval": interval,
                "prediction_rows": prediction_rows,
                "keyframes": keyframes,
                "actual_interval": actual,
                "optimizer_reported_interval": _reported_interval(
                    root, arm, interval
                ),
                "target_error": actual - interval,
                "target_achievement": min(actual, interval) / max(actual, interval),
                "optimizer_wall_seconds": runtime,
                "optimizer_reported_seconds": reported_runtime,
                "video_fps": 23510.0 / runtime,
                "invalid_polygon_rings": sum(invalid.values()),
                "result_sqlite": str(sqlite_path.resolve()),
                "result_sqlite_size_bytes": sqlite_path.stat().st_size,
                "result_sqlite_sha256": _sha256(sqlite_path),
                "result_schema_sha256": schema_fingerprint(sqlite_path),
                **exact,
                **{f"common_raw_{key}": value for key, value in common.items()},
            }
            rows.append(row)
    by_interval: list[dict[str, Any]] = []
    for interval in range(1, 7):
        legacy = next(
            row
            for row in rows
            if row["target_interval"] == interval
            and row["arm"] == "legacy_production"
        )
        candidate = next(
            row
            for row in rows
            if row["target_interval"] == interval
            and row["arm"] == "production_candidate_20260814"
        )
        by_interval.append(
            {
                "target_interval": interval,
                "candidate_minus_legacy_iou_mean": (
                    candidate["iou_mean"] - legacy["iou_mean"]
                ),
                "candidate_minus_legacy_iou_q01": (
                    candidate["iou_q01"] - legacy["iou_q01"]
                ),
                "candidate_minus_legacy_recall_min": (
                    candidate["recall_min"] - legacy["recall_min"]
                ),
                "recall_violations_removed": (
                    legacy["recall_violations"] - candidate["recall_violations"]
                ),
                "candidate_minus_legacy_actual_interval": (
                    candidate["actual_interval"] - legacy["actual_interval"]
                ),
                "candidate_vs_legacy_runtime_ratio": (
                    candidate["optimizer_wall_seconds"]
                    / legacy["optimizer_wall_seconds"]
                ),
                "candidate_minus_legacy_common_raw_iou": (
                    candidate["common_raw_pixel_weighted_iou"]
                    - legacy["common_raw_pixel_weighted_iou"]
                ),
            }
        )
    schema_hashes = {row["result_schema_sha256"] for row in rows}
    payload = {
        "schema_version": 1,
        "scope": {
            "input": "v3__kpi_2025_12",
            "target_intervals": list(range(1, 7)),
            "legacy": "legacy NMS + legacy polygon approximation + legacy v22 DP/pair-vote",
            "candidate": "virtual-component Mask NMS + polygon14 + minimum-Recall DP + constrained pair-vote",
            "recall_floor": 0.97,
            "semantic_ground_truth_available": False,
            "privacy": "SQLite/JSON mask geometry only; no video frame decoded.",
        },
        "rows": rows,
        "by_interval": by_interval,
        "validation": {
            "all_sqlite_integrity_ok": True,
            "all_sqlite_foreign_keys_ok": True,
            "schema_hash_count": len(schema_hashes),
            "schema_hashes": sorted(schema_hashes),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = collect(args.root.expanduser().resolve(), args.scored_jsonl.resolve())
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "interval_metrics.csv", payload["rows"])
    _write_csv(output / "interval_deltas.csv", payload["by_interval"])
    print(json.dumps({"comparison": str(output / "comparison.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
