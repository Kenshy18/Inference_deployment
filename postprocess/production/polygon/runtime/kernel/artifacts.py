"""Optimizer SQLite/JSON persistence and exact-evaluation artifacts."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from .geometry import (
    compute_exact_metrics_from_polygons,
    compute_weighted_error,
    parse_polygons,
)
from .types import TrackRow


def load_rows(sqlite_path: Path) -> list[TrackRow]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        raw_rows = conn.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
        ).fetchall()
    finally:
        conn.close()
    rows: list[TrackRow] = []
    for frame, track_id, polygons_json in raw_rows:
        rows.append(
            TrackRow(
                frame=int(frame),
                track_id=str(track_id),
                polygons=parse_polygons(str(polygons_json)),
            )
        )
    return rows


def write_csv(
    rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_rows_to_pred_sqlite(
    union_rows: list[dict[str, object]], output_sqlite: Path
) -> None:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        output_sqlite.unlink()
    conn = sqlite3.connect(str(output_sqlite))
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
        for row in union_rows:
            cur.execute(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    int(row["frame"]),
                    str(row["track_id"]),
                    json.dumps(row["polygons"], ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def aggregate_exact_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    gt_area = sum(float(row["gt_area"]) for row in rows)
    pred_area = sum(float(row["pred_area"]) for row in rows)
    intersection = sum(float(row["intersection"]) for row in rows)
    union = sum(float(row["union"]) for row in rows)
    weighted_error = sum(float(row["weighted_error"]) for row in rows)
    mean_recall = (
        float(
            np.mean(
                np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
            )
        )
        if rows
        else 1.0
    )
    mean_precision = (
        float(
            np.mean(
                np.asarray([float(row["precision"]) for row in rows], dtype=np.float64)
            )
        )
        if rows
        else 1.0
    )
    mean_iou = (
        float(
            np.mean(np.asarray([float(row["iou"]) for row in rows], dtype=np.float64))
        )
        if rows
        else 1.0
    )
    return {
        "row_count": float(len(rows)),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
        "global_precision": float(intersection / pred_area) if pred_area > 0 else 1.0,
        "global_iou": float(intersection / union) if union > 0 else 1.0,
        "mean_recall": float(mean_recall),
        "mean_precision": float(mean_precision),
        "mean_iou": float(mean_iou),
        "weighted_error_total": float(weighted_error),
        "weighted_error_mean": float(weighted_error / max(len(rows), 1)),
    }


def evaluate_union_exact(
    union_rows: list[dict[str, object]], tracked_sqlite: Path, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_lookup = {(int(row["frame"]), str(row["track_id"])): row for row in union_rows}
    result_rows: list[dict[str, object]] = []
    conn = sqlite3.connect(str(tracked_sqlite))
    try:
        cur = conn.cursor()
        for frame, track_id, polygons_json in cur.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)"
        ):
            key = (int(frame), str(track_id))
            pred = pred_lookup.get(key)
            if pred is None:
                continue
            gt_polys = parse_polygons(str(polygons_json))
            pred_polys = [
                np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                for poly in pred["polygons"]
            ]
            metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
            weighted_error = float(compute_weighted_error(metrics))
            result_rows.append(
                {
                    "frame": int(frame),
                    "track_id": str(track_id),
                    "run_id": int(pred.get("run_id", -1)),
                    "has_keyframe": int(pred.get("has_keyframe", 0)),
                    "gt_area": float(metrics["gt_area"]),
                    "pred_area": float(metrics["pred_area"]),
                    "intersection": float(metrics["intersection"]),
                    "union": float(metrics["union"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "iou": float(metrics["iou"]),
                    "weighted_error": weighted_error,
                }
            )
    finally:
        conn.close()
    metrics_csv = output_dir / "keyframe_exact_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "track_id",
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
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                result_rows,
                key=lambda row: (int(row["frame"]), int(str(row["track_id"]))),
            )
        )
    summary = {
        "input_tracked_sqlite": str(tracked_sqlite),
        "optimized": aggregate_exact_rows(result_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_compact_json_array(output_path: Path, rows) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for row in rows:
            if first:
                first = False
            else:
                f.write(",")
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("]")


class SqliteUnionRowStore:
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        if self.store_path.exists():
            self.store_path.unlink()
        self.conn = sqlite3.connect(str(self.store_path))
        self.conn.execute(
            "CREATE TABLE union_rows (frame INTEGER NOT NULL, track_id TEXT NOT NULL, track_sort INTEGER NOT NULL, row_json TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX idx_union_rows_order ON union_rows(frame, track_sort)"
        )
        self.row_count = 0

    def add_rows(self, rows) -> int:
        inserted = 0

        def iter_records():
            nonlocal inserted
            for row in rows:
                inserted += 1
                track_id = str(row["track_id"])
                yield (
                    int(row["frame"]),
                    track_id,
                    int(track_id),
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                )

        self.conn.executemany(
            "INSERT INTO union_rows(frame, track_id, track_sort, row_json) VALUES (?, ?, ?, ?)",
            iter_records(),
        )
        self.row_count += int(inserted)
        return int(inserted)

    def commit(self) -> None:
        self.conn.commit()

    def iter_rows_sorted(self):
        self.commit()
        for (row_json,) in self.conn.execute(
            "SELECT row_json FROM union_rows ORDER BY frame, track_sort"
        ):
            yield json.loads(str(row_json))

    def write_union_json(self, output_path: Path) -> None:
        write_compact_json_array(output_path, self.iter_rows_sorted())

    def write_pred_sqlite(self, output_sqlite: Path) -> None:
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        self.commit()
        out_conn = sqlite3.connect(str(output_sqlite))
        try:
            cur = out_conn.cursor()
            cur.execute(
                "CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)"
            )
            cur.executemany(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    (
                        int(row["frame"]),
                        str(row["track_id"]),
                        json.dumps(row["polygons"], ensure_ascii=False),
                    )
                    for row in self.iter_rows_sorted()
                ),
            )
            out_conn.commit()
        finally:
            out_conn.close()

    def evaluate_exact(
        self, tracked_sqlite: Path, output_dir: Path
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.commit()
        metrics_csv = output_dir / "keyframe_exact_metrics.csv"
        attached = False
        totals = {
            "row_count": 0.0,
            "gt_area": 0.0,
            "pred_area": 0.0,
            "intersection": 0.0,
            "union": 0.0,
            "weighted_error_total": 0.0,
            "recall_sum": 0.0,
            "precision_sum": 0.0,
            "iou_sum": 0.0,
        }
        try:
            self.conn.execute(
                "ATTACH DATABASE ? AS tracked_eval", (str(tracked_sqlite),)
            )
            attached = True
            rows_iter = self.conn.execute(
                """
                SELECT m.frame, m.track_id, m.polygons, u.row_json
                FROM tracked_eval.masks AS m
                JOIN union_rows AS u
                  ON u.frame = m.frame AND u.track_id = CAST(m.track_id AS TEXT)
                ORDER BY m.frame, CAST(m.track_id AS INTEGER)
                """
            )
            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame",
                        "track_id",
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
                    ],
                )
                writer.writeheader()
                for frame, track_id, polygons_json, row_json in rows_iter:
                    pred = json.loads(str(row_json))
                    gt_polys = parse_polygons(str(polygons_json))
                    pred_polys = [
                        np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                        for poly in pred["polygons"]
                    ]
                    metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
                    weighted_error = float(compute_weighted_error(metrics))
                    result_row = {
                        "frame": int(frame),
                        "track_id": str(track_id),
                        "run_id": int(pred.get("run_id", -1)),
                        "has_keyframe": int(pred.get("has_keyframe", 0)),
                        "gt_area": float(metrics["gt_area"]),
                        "pred_area": float(metrics["pred_area"]),
                        "intersection": float(metrics["intersection"]),
                        "union": float(metrics["union"]),
                        "recall": float(metrics["recall"]),
                        "precision": float(metrics["precision"]),
                        "iou": float(metrics["iou"]),
                        "weighted_error": weighted_error,
                    }
                    writer.writerow(result_row)
                    totals["row_count"] += 1.0
                    totals["gt_area"] += float(result_row["gt_area"])
                    totals["pred_area"] += float(result_row["pred_area"])
                    totals["intersection"] += float(result_row["intersection"])
                    totals["union"] += float(result_row["union"])
                    totals["weighted_error_total"] += weighted_error
                    totals["recall_sum"] += float(result_row["recall"])
                    totals["precision_sum"] += float(result_row["precision"])
                    totals["iou_sum"] += float(result_row["iou"])
        finally:
            if attached:
                self.conn.execute("DETACH DATABASE tracked_eval")
        row_count = float(totals["row_count"])
        gt_area = float(totals["gt_area"])
        pred_area = float(totals["pred_area"])
        intersection = float(totals["intersection"])
        union = float(totals["union"])
        weighted_error = float(totals["weighted_error_total"])
        optimized = {
            "row_count": row_count,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "intersection": intersection,
            "union": union,
            "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
            "global_precision": float(intersection / pred_area)
            if pred_area > 0
            else 1.0,
            "global_iou": float(intersection / union) if union > 0 else 1.0,
            "mean_recall": float(totals["recall_sum"] / max(row_count, 1.0)),
            "mean_precision": float(totals["precision_sum"] / max(row_count, 1.0)),
            "mean_iou": float(totals["iou_sum"] / max(row_count, 1.0)),
            "weighted_error_total": weighted_error,
            "weighted_error_mean": float(weighted_error / max(row_count, 1.0)),
        }
        summary = {
            "input_tracked_sqlite": str(tracked_sqlite),
            "optimized": optimized,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def close(self, unlink: bool = False) -> None:
        self.conn.close()
        if bool(unlink):
            try:
                self.store_path.unlink()
            except FileNotFoundError:
                pass


__all__ = (
    "SqliteUnionRowStore",
    "aggregate_exact_rows",
    "evaluate_union_exact",
    "load_rows",
    "union_rows_to_pred_sqlite",
    "write_compact_json_array",
    "write_csv",
)
