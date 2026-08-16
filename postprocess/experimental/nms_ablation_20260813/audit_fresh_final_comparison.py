#!/usr/bin/env python3
"""Audit a fresh legacy-vs-component-NMS fixed-downstream run.

The script is deliberately read-only with respect to pipeline artifacts.  It
validates the two software-facing SQLite files, aggregates the exact Phase-2
quality CSVs, and compares the two dense polygon outputs on a common raster
grid.  No video frames are decoded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POSTPROCESS = Path(__file__).resolve().parents[2]
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from nms.component_aware import _raster_mask  # noqa: E402


LABELS = ("女性器", "男性器", "結合部分")
PROFILE = Path("polygon14/interval_6/polygon14_keyframe_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    return parser.parse_args()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_ro(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _software_sqlite_audit(path: Path) -> dict[str, Any]:
    with _open_ro(path) as db:
        objects = [
            {"type": row[0], "name": row[1], "sql": row[2] or ""}
            for row in db.execute(
                """
                SELECT type,name,sql FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name
                """
            )
        ]
        canonical = json.dumps(
            objects, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        result_info = dict(db.execute("SELECT key,value FROM result_schema_info"))
        counts = {}
        for table in (
            "frames",
            "detections",
            "segmentations",
            "tracking_assignments",
            "tracks",
            "mask_track_segments",
            "mask_keyframes",
            "keyframe_components",
            "keyframe_polygon_rings",
            "keyframe_polygon_points",
        ):
            counts[table] = int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "file_sha256": _sha256(path),
            "schema_sha256": hashlib.sha256(canonical).hexdigest(),
            "integrity_check": str(db.execute("PRAGMA integrity_check").fetchone()[0]),
            "foreign_key_errors": len(db.execute("PRAGMA foreign_key_check").fetchall()),
            "result_schema_info": result_info,
            "counts": counts,
        }


def _exact_quality(arm: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_recalls: list[float] = []
    all_ious: list[float] = []
    class_rows: list[dict[str, Any]] = []
    for label in LABELS:
        path = arm / PROFILE / label / "runtime/exact/keyframe_exact_metrics.csv"
        recalls: list[float] = []
        ious: list[float] = []
        keyframes = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                recalls.append(float(row["recall"]))
                ious.append(float(row["iou"]))
                keyframes += int(row["has_keyframe"])
        all_recalls.extend(recalls)
        all_ious.extend(ious)
        class_rows.append(
            {
                "label": label,
                "rows": len(ious),
                "keyframes": keyframes,
                "recall_min": min(recalls),
                "recall_violations": sum(value < 0.97 - 1e-9 for value in recalls),
                "iou_mean": sum(ious) / len(ious),
                "iou_q01": _quantile(ious, 0.01),
                "iou_q05": _quantile(ious, 0.05),
                "iou_min": min(ious),
            }
        )
    return (
        {
            "rows": len(all_ious),
            "keyframes": sum(row["keyframes"] for row in class_rows),
            "recall_min": min(all_recalls),
            "recall_violations": sum(value < 0.97 - 1e-9 for value in all_recalls),
            "iou_mean": sum(all_ious) / len(all_ious),
            "iou_q01": _quantile(all_ious, 0.01),
            "iou_q05": _quantile(all_ious, 0.05),
            "iou_min": min(all_ious),
        },
        class_rows,
    )


def _prediction_groups(path: Path) -> Iterator[tuple[int, list[np.ndarray]]]:
    with _open_ro(path) as db:
        rows = db.execute("SELECT frame,polygons FROM masks ORDER BY frame,track_id")
        for frame, group in itertools.groupby(rows, key=lambda row: int(row["frame"])):
            polygons: list[np.ndarray] = []
            for row in group:
                for polygon in json.loads(str(row["polygons"])):
                    array = np.asarray(polygon, dtype=np.float32)
                    if len(array) >= 3:
                        polygons.append(array)
            yield frame, polygons


def _raster_pair(
    left: list[np.ndarray], right: list[np.ndarray]
) -> tuple[int, int, int, int, float]:
    points = left + right
    if not points:
        return 0, 0, 0, 0, 1.0
    combined = np.concatenate(points, axis=0)
    origin = np.floor(combined.min(axis=0)).astype(np.int32) - 2
    maximum = np.ceil(combined.max(axis=0)).astype(np.int32) + 2
    width, height = (maximum - origin + 1).tolist()
    if width <= 0 or height <= 0:
        raise ValueError("invalid joint polygon bounds")
    masks = []
    for polygons in (left, right):
        mask = np.zeros((height, width), dtype=np.uint8)
        contours = [np.round(polygon - origin).astype(np.int32) for polygon in polygons]
        if contours:
            cv2.fillPoly(mask, contours, 1)
        masks.append(mask)
    left_area = int(masks[0].sum())
    right_area = int(masks[1].sum())
    intersection = int(np.logical_and(masks[0], masks[1]).sum())
    union = left_area + right_area - intersection
    iou = 1.0 if union == 0 else intersection / union
    return left_area, right_area, intersection, union, iou


def _dense_comparison(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for label in LABELS:
        left_path = root / "legacy_production" / PROFILE / label / "runtime/pred/predictions.sqlite"
        right_path = root / "component_mask_v2" / PROFILE / label / "runtime/pred/predictions.sqlite"
        left = iter(_prediction_groups(left_path))
        right = iter(_prediction_groups(right_path))
        left_item = next(left, None)
        right_item = next(right, None)
        ious: list[float] = []
        area_ratios: list[float] = []
        left_only = right_only = equal = 0
        while left_item is not None or right_item is not None:
            if right_item is None or (left_item is not None and left_item[0] < right_item[0]):
                frame, left_polygons = left_item
                right_polygons: list[np.ndarray] = []
                left_item = next(left, None)
                left_only += 1
            elif left_item is None or right_item[0] < left_item[0]:
                frame, right_polygons = right_item
                left_polygons = []
                right_item = next(right, None)
                right_only += 1
            else:
                frame = left_item[0]
                left_polygons = left_item[1]
                right_polygons = right_item[1]
                left_item = next(left, None)
                right_item = next(right, None)
            left_area, right_area, intersection, union, iou = _raster_pair(
                left_polygons, right_polygons
            )
            ratio = math.nan if left_area == 0 else right_area / left_area
            if iou == 1.0 and left_area == right_area:
                equal += 1
            ious.append(iou)
            if math.isfinite(ratio):
                area_ratios.append(ratio)
            frame_rows.append(
                {
                    "label": label,
                    "frame": frame,
                    "legacy_area": left_area,
                    "candidate_area": right_area,
                    "intersection": intersection,
                    "union": union,
                    "cross_arm_iou": iou,
                    "candidate_to_legacy_area_ratio": ratio,
                }
            )
        summaries.append(
            {
                "label": label,
                "union_frame_count": len(ious),
                "pixel_identical_frames": equal,
                "legacy_only_frames": left_only,
                "candidate_only_frames": right_only,
                "cross_arm_iou_mean": sum(ious) / len(ious),
                "cross_arm_iou_q01": _quantile(ious, 0.01),
                "cross_arm_iou_q05": _quantile(ious, 0.05),
                "cross_arm_iou_min": min(ious),
                "frames_iou_below_0p5": sum(value < 0.5 for value in ious),
                "area_ratio_median": _quantile(area_ratios, 0.5),
                "area_ratio_q05": _quantile(area_ratios, 0.05),
                "area_ratio_q95": _quantile(area_ratios, 0.95),
            }
        )
    return summaries, frame_rows


def _prediction_map(root: Path, arm: str) -> dict[str, dict[int, list[np.ndarray]]]:
    result: dict[str, dict[int, list[np.ndarray]]] = {}
    for label in LABELS:
        path = root / arm / PROFILE / label / "runtime/pred/predictions.sqlite"
        result[label] = {frame: polygons for frame, polygons in _prediction_groups(path)}
    return result


def _raster_union_pair(
    raw_detections: list[dict[str, Any]], final_polygons: list[np.ndarray]
) -> tuple[int, int, int, int, float, float, float]:
    raw_rasters = [value for det in raw_detections if (value := _raster_mask(det)) is not None]
    final_detection = {"polygons": [polygon.tolist() for polygon in final_polygons]}
    final_raster = _raster_mask(final_detection) if final_polygons else None
    rasters = raw_rasters + ([final_raster] if final_raster is not None else [])
    if not rasters:
        return 0, 0, 0, 0, 1.0, 1.0, 1.0
    left = min(value.left for value in rasters)
    top = min(value.top for value in rasters)
    right = max(value.right for value in rasters)
    bottom = max(value.bottom for value in rasters)
    height = bottom - top + 1
    width = right - left + 1

    def combine(values: list[Any]) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for value in values:
            y1, y2 = value.top - top, value.bottom - top + 1
            x1, x2 = value.left - left, value.right - left + 1
            np.maximum(canvas[y1:y2, x1:x2], value.mask, out=canvas[y1:y2, x1:x2])
        return canvas

    raw_mask = combine(raw_rasters)
    final_mask = combine([] if final_raster is None else [final_raster])
    raw_area = int(raw_mask.sum())
    final_area = int(final_mask.sum())
    intersection = int(np.logical_and(raw_mask, final_mask).sum())
    union = raw_area + final_area - intersection
    recall = 1.0 if raw_area == 0 else intersection / raw_area
    precision = 1.0 if final_area == 0 else intersection / final_area
    iou = 1.0 if union == 0 else intersection / union
    return raw_area, final_area, intersection, union, recall, precision, iou


def _common_raw_reference(scored: Path, root: Path) -> list[dict[str, Any]]:
    predictions = {
        arm: _prediction_map(root, arm)
        for arm in ("legacy_production", "component_mask_v2")
    }
    values: dict[tuple[str, str], dict[str, list[float] | int]] = {}
    for arm in predictions:
        for label in LABELS:
            values[(arm, label)] = {
                "recall": [],
                "precision": [],
                "iou": [],
                "raw_pixels": 0,
                "covered_pixels": 0,
                "final_pixels": 0,
            }
    with scored.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["frame_index"])
            by_label = {label: [] for label in LABELS}
            for detection in record.get("detections", []):
                label = str(detection.get("label") or detection.get("class_name") or "")
                if label in by_label:
                    by_label[label].append(detection)
            for arm, arm_predictions in predictions.items():
                for label in LABELS:
                    raw_area, final_area, intersection, _, recall, precision, iou = _raster_union_pair(
                        by_label[label], arm_predictions[label].get(frame, [])
                    )
                    if raw_area == 0:
                        continue
                    cell = values[(arm, label)]
                    cell["recall"].append(recall)
                    cell["precision"].append(precision)
                    cell["iou"].append(iou)
                    cell["raw_pixels"] += raw_area
                    cell["covered_pixels"] += intersection
                    cell["final_pixels"] += final_area
    rows: list[dict[str, Any]] = []
    for (arm, label), cell in values.items():
        recalls = cell["recall"]
        precisions = cell["precision"]
        ious = cell["iou"]
        rows.append(
            {
                "arm": arm,
                "label": label,
                "raw_nonempty_frames": len(recalls),
                "raw_pixel_weighted_recall": cell["covered_pixels"] / cell["raw_pixels"],
                "frame_recall_mean": sum(recalls) / len(recalls),
                "frame_recall_q01": _quantile(recalls, 0.01),
                "frame_recall_min": min(recalls),
                "frame_precision_mean": sum(precisions) / len(precisions),
                "frame_iou_mean": sum(ious) / len(ious),
                "frame_iou_q01": _quantile(ious, 0.01),
                "frame_iou_min": min(ious),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    class_quality_rows: list[dict[str, Any]] = []
    for name in ("legacy_production", "component_mask_v2"):
        arm = root / name
        manifest = json.loads((arm / "arm_manifest.json").read_text(encoding="utf-8"))
        exact, classes = _exact_quality(arm)
        software = _software_sqlite_audit(arm / "software_sqlite/12月KPI動画.sqlite")
        arms[name] = {
            "nms": manifest["nms"],
            "tracking": manifest["tracking"],
            "polygon14": {
                "elapsed_seconds": manifest["polygon14"]["elapsed_seconds"],
                "actual_mean_interval": manifest["polygon14"]["manifest_payload"]["runs"][0]["actual_mean_interval"],
            },
            "exact": exact,
            "software_sqlite": software,
        }
        for row in classes:
            class_quality_rows.append({"arm": name, **row})
    dense_summary, dense_frames = _dense_comparison(root)
    common_raw = _common_raw_reference(args.scored_jsonl.resolve(), root)
    summary = {
        "scope": {
            "source": "v3__kpi_2025_12",
            "controlled_variable": "NMS, hole fill, and island handling only",
            "fixed_downstream": {
                "polygon_profile": "polygon14_keyframe_v1",
                "vertices_per_component": 14,
                "minimum_recall": 0.97,
                "target_interval": 6,
                "pair_vote_sweeps": 2,
            },
            "semantic_ground_truth_available": False,
        },
        "arms": arms,
        "dense_cross_arm": dense_summary,
        "common_raw_reference": common_raw,
        "schema_identical": (
            arms["legacy_production"]["software_sqlite"]["schema_sha256"]
            == arms["component_mask_v2"]["software_sqlite"]["schema_sha256"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "class_quality.csv", class_quality_rows)
    _write_csv(output / "dense_cross_arm_summary.csv", dense_summary)
    _write_csv(output / "dense_cross_arm_frames.csv", dense_frames)
    _write_csv(output / "common_raw_reference.csv", common_raw)
    sqlite_rows = [
        {"arm": name, **value["software_sqlite"]}
        for name, value in arms.items()
    ]
    for row in sqlite_rows:
        row["result_schema_info"] = json.dumps(row["result_schema_info"], ensure_ascii=False)
        row["counts"] = json.dumps(row["counts"], ensure_ascii=False)
    _write_csv(output / "software_sqlite_validation.csv", sqlite_rows)
    print(json.dumps({"summary": str(output / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
