"""Exact SQLite-geometry metrics for the 0809 raw-only baseline matrix."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from experimental.polygon_recall_optimizer.fixed_budget import (
    FrameEvaluation,
    evaluate_segments,
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.sqlite_export import schema_fingerprint


LABELS = ("女性器", "男性器", "結合部分")


def _distribution(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            f"{prefix}_mean": 1.0,
            f"{prefix}_min": 1.0,
            f"{prefix}_q001": 1.0,
            f"{prefix}_q01": 1.0,
            f"{prefix}_q05": 1.0,
            f"{prefix}_q95": 1.0,
            f"{prefix}_q99": 1.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_q001": float(np.quantile(array, 0.001)),
        f"{prefix}_q01": float(np.quantile(array, 0.01)),
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
        f"{prefix}_q99": float(np.quantile(array, 0.99)),
    }


def _key_stats(segments: dict, start_frame: int, end_frame: int) -> dict[str, float | int]:
    count = 0
    span = 0
    intervals = 0
    for values in segments.values():
        for segment in values:
            frames = [
                key.frame
                for key in segment.keyframes
                if start_frame <= key.frame <= end_frame
            ]
            count += len(frames)
            if len(frames) >= 2:
                span += frames[-1] - frames[0]
                intervals += len(frames) - 1
    return {
        "keyframe_count": int(count),
        "mean_temporal_key_interval": float(span / intervals) if intervals else 0.0,
    }


def _quality(rows: list[FrameEvaluation], recall_floor: float) -> dict[str, object]:
    recalls = np.asarray([row.recall for row in rows], dtype=np.float64)
    output: dict[str, object] = {
        "evaluated_observations": len(rows),
        "recall_violations": int(np.sum(recalls + 1e-12 < recall_floor)),
        "recall_below_090": int(np.sum(recalls + 1e-12 < 0.90)),
        "recall_below_095": int(np.sum(recalls + 1e-12 < 0.95)),
    }
    output.update(_distribution((row.recall for row in rows), "recall"))
    output.update(_distribution((row.iou for row in rows), "iou"))
    output.update(_distribution((row.precision for row in rows), "precision"))
    output.update(_distribution((row.area_ratio for row in rows), "area_ratio"))
    output.update(
        _distribution((row.excess_area_ratio for row in rows), "excess_area_ratio")
    )
    output.update(
        _distribution((row.centroid_error_px for row in rows), "centroid_error_px")
    )
    return output


def _sqlite_contract(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        fingerprint = schema_fingerprint(connection)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        rows = connection.execute(
            """
            SELECT t.label, s.shape_type, COUNT(*)
            FROM mask_track_segments s
            JOIN tracks t ON t.track_id=s.track_id
            WHERE t.domain='genital'
            GROUP BY t.label, s.shape_type
            ORDER BY t.label, s.shape_type
            """
        ).fetchall()
        policies = connection.execute(
            """
            SELECT label, shape_mode, keyframe_interval, max_gap
            FROM class_postprocess_policies
            ORDER BY label
            """
        ).fetchall()
        frame_count, first_frame, last_frame = connection.execute(
            "SELECT COUNT(*), MIN(frame_index), MAX(frame_index) FROM frames"
        ).fetchone()
        key_count, multi_component_keys, max_components = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(component_count > 1), 0),
                   COALESCE(MAX(component_count), 0)
            FROM (
                SELECT k.id, COUNT(DISTINCT c.slot_index) AS component_count
                FROM mask_keyframes k
                JOIN mask_track_segments s ON s.id=k.segment_id
                JOIN tracks t ON t.track_id=s.track_id
                JOIN keyframe_components c ON c.keyframe_id=k.id
                WHERE t.domain='genital'
                GROUP BY k.id
            )
            """
        ).fetchone()
    return {
        "schema_fingerprint": fingerprint,
        "integrity_check": integrity,
        "foreign_key_errors": foreign_keys,
        "video_frames": int(frame_count),
        "video_frame_range": [
            int(first_frame) if first_frame is not None else None,
            int(last_frame) if last_frame is not None else None,
        ],
        "genital_keyframes_with_components": int(key_count),
        "genital_multi_component_keyframes": int(multi_component_keys),
        "genital_max_components_per_keyframe": int(max_components),
        "segment_shape_counts": [
            {"label": str(label), "shape_type": str(shape), "segments": int(count)}
            for label, shape, count in rows
        ],
        "policies": [
            {
                "label": str(label),
                "shape_mode": str(shape),
                "keyframe_interval": int(interval),
                "max_gap": int(gap),
            }
            for label, shape, interval, gap in policies
        ],
    }


def evaluate_sqlite(
    raw_reference_sqlite: Path,
    result_sqlite: Path,
    *,
    recall_floor: float = 0.97,
) -> dict[str, object]:
    by_class: dict[str, object] = {}
    all_rows: list[FrameEvaluation] = []
    total_raw = 0
    total_keys = 0
    total_span = 0.0
    total_intervals = 0
    for label in LABELS:
        raw = load_raw_masks(
            raw_reference_sqlite,
            label=label,
            start_frame=0,
            end_frame=2**31 - 1,
        )
        frames = [frame for frame, _track in raw]
        if not frames:
            by_class[label] = {"raw_observations": 0, "evaluated_observations": 0}
            continue
        first, last = min(frames), max(frames)
        segments = load_segments(
            result_sqlite, label=label, start_frame=first, end_frame=last
        )
        rows = evaluate_segments(raw, segments)
        keys = _key_stats(segments, first, last)
        interval_count = sum(
            max(
                0,
                len(
                    [
                        key
                        for key in segment.keyframes
                        if first <= key.frame <= last
                    ]
                )
                - 1,
            )
            for values in segments.values()
            for segment in values
        )
        by_class[label] = {
            "frame_range": [first, last],
            "raw_observations": len(raw),
            "coverage_ratio": len(rows) / max(len(raw), 1),
            **keys,
            **_quality(rows, recall_floor),
        }
        all_rows.extend(rows)
        total_raw += len(raw)
        total_keys += int(keys["keyframe_count"])
        total_span += float(keys["mean_temporal_key_interval"]) * interval_count
        total_intervals += interval_count

    aggregate = {
        "raw_observations": total_raw,
        "coverage_ratio": len(all_rows) / max(total_raw, 1),
        "keyframe_count": total_keys,
        "mean_temporal_key_interval": (
            total_span / total_intervals if total_intervals else 0.0
        ),
        **_quality(all_rows, recall_floor),
    }
    return {
        "privacy": "SQLite geometry only; video pixels were not opened.",
        "raw_reference_sqlite": str(raw_reference_sqlite.resolve()),
        "result_sqlite": str(result_sqlite.resolve()),
        "recall_floor": recall_floor,
        "aggregate": aggregate,
        "classes": by_class,
        "sqlite": _sqlite_contract(result_sqlite),
    }


def write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
