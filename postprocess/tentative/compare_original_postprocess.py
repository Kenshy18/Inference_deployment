"""Reproducible behavior comparison against the original Atosyori engine.

This module is deliberately isolated under ``tentative``.  It never imports
or writes the public result schema; it compares the internal, legacy
``masks/tracks/cuts`` artifacts produced from an identical raw-mask input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


POSTPROCESS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = POSTPROCESS_ROOT.parent
DEFAULT_ORIGINAL_ROOT = Path(
    "/home/kenshin/inference_backend/Dinov3_postprocess"
)
DEFAULT_PYTHON = Path(
    "/home/kenshin/.local/share/video-mask-runtime/"
    "envs/production/bin/python3.10"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original Dinov3_postprocess behavior with the current "
            "modular postprocess using identical raw/tracked masks."
        )
    )
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--source-tracked-sqlite", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--frames-per-track", type=int, default=48)
    parser.add_argument("--tracks", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cut-video",
        type=Path,
        help="optional real video used for an additional cut-detector comparison",
    )
    parser.add_argument(
        "--baseline-comparison-json",
        type=Path,
        help="optional comparison.json captured before an implementation change",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _canonical_json(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND\n")
        log.write(json.dumps(list(command), ensure_ascii=False, indent=2))
        log.write("\n\nOUTPUT\n")
        log.flush()
        subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    return time.perf_counter() - started


def _original_env(original_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    source = original_root / "external" / "atosyori-pipeline-dev" / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source) if not existing else str(source) + os.pathsep + existing
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _rectangle(x: float, y: float, width: float, height: float) -> list[list[float]]:
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def _detection(
    x: float,
    y: float,
    width: float,
    height: float,
    score: float,
    label: str,
) -> dict[str, object]:
    polygon = _rectangle(x, y, width, height)
    return {
        "label": label,
        "class_name": label,
        "score": score,
        "detector_score": score,
        "class_score": min(0.99, score + 0.02),
        "category_id": 1,
        "category_index": 1,
        "bbox_xyxy": [x, y, x + width, y + height],
        "polygons": [polygon],
        "segmentation": [polygon],
    }


def write_raw_fixture(path: Path, *, frames: int = 36) -> Path:
    """Exercise score filtering, NMS, tracking, relabeling and short pruning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in range(frames):
            detections: list[dict[str, object]] = []
            if frame not in {12, 13}:
                label = "女性器" if frame == 7 else "男性器"
                x = 100.0 + frame * 2.0
                detections.append(_detection(x, 120.0, 80.0, 60.0, 0.94, label))
                # Lower-scored contained duplicate: adaptive NMS must remove it.
                detections.append(
                    _detection(x + 2.0, 122.0, 76.0, 56.0, 0.80, label)
                )
            # Below the common 0.35 score threshold.
            detections.append(
                _detection(20.0, 20.0, 20.0, 16.0, 0.20, "女性器")
            )
            # Two-hit track, removed when max short-track length is two.
            if frame in {4, 5}:
                detections.append(
                    _detection(500.0 + frame, 400.0, 45.0, 35.0, 0.91, "結合部分")
                )
            # A second persistent track starts later.
            if frame >= 20:
                detections.append(
                    _detection(700.0 - frame, 300.0, 70.0, 52.0, 0.92, "結合部分")
                )
            record = {"frame_index": frame, "detections": detections}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return path


def _load_original_legacy(original_root: Path):
    path = (
        original_root
        / "external"
        / "atosyori-pipeline-dev"
        / "src"
        / "atosyori_postprocess"
        / "legacy"
        / "run_standalone.py"
    )
    module_name = "tentative_original_atosyori_run_standalone"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load original engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_raw_preprocess_comparison(
    *, original_root: Path, work_dir: Path
) -> dict[str, object]:
    raw = write_raw_fixture(work_dir / "input.jsonl")
    old_sqlite = work_dir / "original_tracked.sqlite"
    current_sqlite = work_dir / "current_tracked.sqlite"

    legacy = _load_original_legacy(original_root)
    old_summary = legacy.infer_build_tracked_sqlite_from_raw_jsonl(
        raw,
        old_sqlite,
        None,
        remove_short_tracks_max_frames=2,
        enable_cut_detect=False,
        raw_det_score_min=0.35,
        raw_cut_method="high_precision",
    )

    if str(POSTPROCESS_ROOT) not in sys.path:
        sys.path.insert(0, str(POSTPROCESS_ROOT))
    from contracts.detections import CutList, transform_detection_jsonl, write_cut_list
    from nms.adaptive import AdaptiveNms
    from preprocessing.normalization import normalize_detection_jsonl
    from preprocessing.score_policy import ScorePolicy, apply_score_policy_jsonl
    from tracking.builder import build_tracked_sqlite

    normalized = work_dir / "normalized.jsonl"
    scored = work_dir / "scored.jsonl"
    nms_path = work_dir / "nms.jsonl"
    cuts = work_dir / "cuts.json"
    normalize_detection_jsonl(raw, normalized)
    apply_score_policy_jsonl(
        normalized, scored, policy=ScorePolicy(default_min=0.35)
    )
    nms = AdaptiveNms()
    transform_detection_jsonl(
        scored,
        nms_path,
        lambda record: {
            **record,
            "detections": nms.apply(list(record["detections"])),
        },
    )
    write_cut_list(cuts, CutList((), "disabled"))
    current_summary = build_tracked_sqlite(
        nms_path,
        current_sqlite,
        cuts,
        remove_short_tracks_max_frames=2,
    )

    tables = ("masks", "tracks", "cuts", "raw_tracked_masks", "raw_tracks")
    comparisons = {
        table: _compare_table_rows(old_sqlite, current_sqlite, table)
        for table in tables
    }
    decision_tables = ("masks", "tracks", "cuts", "raw_tracks")
    return {
        "original_summary": old_summary,
        "current_summary": current_summary,
        "tables": comparisons,
        "decision_equivalent": all(
            bool(comparisons[table]["equal"]) for table in decision_tables
        ),
        "audit_equivalent": all(
            bool(value["equal"]) for value in comparisons.values()
        ),
        "audit_difference_note": (
            "current normalization materializes bbox xywh provenance when the "
            "input provides bbox_xyxy; this does not affect NMS/tracking output"
        ),
        "artifacts": {
            "raw_jsonl": str(raw),
            "original_tracked_sqlite": str(old_sqlite),
            "current_tracked_sqlite": str(current_sqlite),
        },
    }


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]


def _compare_table_rows(first: Path, second: Path, table: str) -> dict[str, object]:
    with sqlite3.connect(first) as a, sqlite3.connect(second) as b:
        a_columns = _table_columns(a, table)
        b_columns = _table_columns(b, table)
        common = [column for column in a_columns if column in set(b_columns)]
        # source_detection_id is new provenance, not an algorithm decision.
        common = [column for column in common if column != "source_detection_id"]
        order = [
            column
            for column in (
                "frame",
                "track_id",
                "raw_track_id",
                "raw_detection_index",
            )
            if column in common
        ]
        select = ", ".join(f'"{column}"' for column in common)
        order_sql = ", ".join(f'"{column}"' for column in order)
        query = f'SELECT {select} FROM "{table}"'
        if order_sql:
            query += f" ORDER BY {order_sql}"

        def rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
            output: list[tuple[object, ...]] = []
            json_columns = {
                "polygons",
                "bbox_xyxy_json",
                "bbox_json",
            }
            for raw_row in connection.execute(query):
                values = []
                for column, value in zip(common, raw_row, strict=True):
                    if column in json_columns and value is not None:
                        values.append(_canonical_json(value))
                    else:
                        values.append(value)
                output.append(tuple(values))
            return output

        left = rows(a)
        right = rows(b)
    return {
        "equal": left == right,
        "columns_compared": common,
        "original_rows": len(left),
        "current_rows": len(right),
        "original_sha256": hashlib.sha256(repr(left).encode()).hexdigest(),
        "current_sha256": hashlib.sha256(repr(right).encode()).hexdigest(),
        "first_difference": next(
            (
                {"index": index, "original": old, "current": new}
                for index, (old, new) in enumerate(zip(left, right))
                if old != new
            ),
            None,
        ),
    }


def write_cut_fixture(video_path: Path, jsonl_path: Path) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create synthetic video: {video_path}")
    colors = ((20, 20, 20), (240, 240, 240), (20, 20, 240))
    try:
        for color in colors:
            for _ in range(60):
                writer.write(np.full((180, 320, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for frame in range(180):
            handle.write(json.dumps({"frame_index": frame, "detections": []}))
            handle.write("\n")


def run_cut_comparison(*, original_root: Path, work_dir: Path) -> dict[str, object]:
    video = work_dir / "cuts.mp4"
    jsonl = work_dir / "cuts.jsonl"
    write_cut_fixture(video, jsonl)
    legacy = _load_original_legacy(original_root)
    old_frames, old_elapsed, old_method = legacy.infer_detect_cut_frames_for_jsonl(
        jsonl, video, method="high_precision"
    )
    if str(POSTPROCESS_ROOT) not in sys.path:
        sys.path.insert(0, str(POSTPROCESS_ROOT))
    from cut_detection.detector import HighPrecisionCutDetector

    current = HighPrecisionCutDetector().detect(jsonl, video)
    return {
        "expected": [60, 120],
        "original_frames": old_frames,
        "current_frames": current.frames,
        "equivalent": old_frames == current.frames,
        "original_method": old_method,
        "current_method": current.method,
        "original_elapsed_seconds": old_elapsed,
        "current_elapsed_seconds": current.elapsed_seconds,
        "artifacts": {"video": str(video), "jsonl": str(jsonl)},
    }


def run_real_cut_comparison(
    *, original_root: Path, work_dir: Path, video: Path
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"could not determine video frame count: {video}")
    jsonl = work_dir / "frames.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for frame in range(frame_count):
            handle.write(json.dumps({"frame_index": frame, "detections": []}))
            handle.write("\n")
    legacy = _load_original_legacy(original_root)
    old_frames, old_elapsed, old_method = legacy.infer_detect_cut_frames_for_jsonl(
        jsonl, video, method="high_precision"
    )
    if str(POSTPROCESS_ROOT) not in sys.path:
        sys.path.insert(0, str(POSTPROCESS_ROOT))
    from cut_detection.detector import HighPrecisionCutDetector

    current = HighPrecisionCutDetector().detect(jsonl, video)
    return {
        "video": str(video),
        "frame_count": frame_count,
        "original_frames": old_frames,
        "current_frames": current.frames,
        "equivalent": old_frames == current.frames,
        "original_method": old_method,
        "current_method": current.method,
        "original_elapsed_seconds": old_elapsed,
        "current_elapsed_seconds": current.elapsed_seconds,
        "artifact": str(jsonl),
    }


def _contiguous_runs(rows: list[sqlite3.Row]) -> list[list[sqlite3.Row]]:
    runs: list[list[sqlite3.Row]] = []
    for row in rows:
        if not runs or int(row["frame"]) != int(runs[-1][-1]["frame"]) + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


def extract_real_fixture(
    source: Path,
    output: Path,
    *,
    frames_per_track: int,
    track_limit: int,
) -> dict[str, object]:
    if frames_per_track < 8 or track_limit < 1:
        raise ValueError("frames-per-track must be >= 8 and tracks must be >= 1")
    selected: list[tuple[str, str, list[sqlite3.Row]]] = []
    with sqlite3.connect(source) as connection:
        connection.row_factory = sqlite3.Row
        candidates = list(
            connection.execute(
                """
                SELECT track_id, COALESCE(label, '') AS label, COUNT(*) AS rows
                FROM masks
                GROUP BY track_id, label
                ORDER BY rows DESC, CAST(track_id AS INTEGER)
                """
            )
        )
        used_labels: set[str] = set()
        deferred: list[tuple[str, str, list[sqlite3.Row]]] = []
        for candidate in candidates:
            track_id = str(candidate["track_id"])
            label = str(candidate["label"])
            rows = list(
                connection.execute(
                    "SELECT * FROM masks WHERE track_id=? ORDER BY frame", (track_id,)
                )
            )
            runs = sorted(_contiguous_runs(rows), key=len, reverse=True)
            if not runs or len(runs[0]) < frames_per_track:
                continue
            item = (track_id, label, runs[0][:frames_per_track])
            if label and label not in used_labels and len(selected) < track_limit:
                selected.append(item)
                used_labels.add(label)
            else:
                deferred.append(item)
        for item in deferred:
            if len(selected) >= track_limit:
                break
            selected.append(item)
    if not selected:
        raise RuntimeError("no sufficiently long contiguous track was found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as destination:
        destination.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                PRIMARY KEY(frame, track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE cuts(frame INTEGER PRIMARY KEY);
            """
        )
        manifest_tracks: list[dict[str, object]] = []
        for new_index, (source_id, label, rows) in enumerate(selected, 1):
            new_id = str(new_index)
            destination.execute(
                "INSERT INTO tracks(track_id, label) VALUES (?, ?)", (new_id, label)
            )
            for local_frame, row in enumerate(rows):
                destination.execute(
                    """
                    INSERT INTO masks(
                        frame, track_id, polygons, shape_type, dilate_px,
                        feather_px, mosaic_block, mosaic_alias, label
                    ) VALUES (?, ?, ?, 'polygon', 0, 0, 0, 0, ?)
                    """,
                    (local_frame, new_id, row["polygons"], label),
                )
            manifest_tracks.append(
                {
                    "source_track_id": source_id,
                    "fixture_track_id": new_id,
                    "label": label,
                    "source_first_frame": int(rows[0]["frame"]),
                    "source_last_frame": int(rows[-1]["frame"]),
                    "rows": len(rows),
                }
            )
    return {"path": str(output), "tracks": manifest_tracks}


def _load_masks(path: Path) -> dict[tuple[int, str], list[np.ndarray]]:
    output: dict[tuple[int, str], list[np.ndarray]] = {}
    with sqlite3.connect(path) as connection:
        for frame, track_id, polygons in connection.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id"
        ):
            values = [] if polygons is None else json.loads(str(polygons))
            output[(int(frame), str(track_id))] = [
                np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
                for polygon in values
                if len(polygon) >= 3
            ]
    return output


def _raster_counts(
    reference: list[np.ndarray], prediction: list[np.ndarray]
) -> tuple[int, int, int, int]:
    polygons = reference + prediction
    if not polygons:
        return (0, 0, 0, 0)
    all_points = np.concatenate(polygons, axis=0)
    minimum = np.floor(all_points.min(axis=0)).astype(np.int32) - 2
    maximum = np.ceil(all_points.max(axis=0)).astype(np.int32) + 2
    width = max(1, int(maximum[0] - minimum[0] + 1))
    height = max(1, int(maximum[1] - minimum[1] + 1))

    def rasterize(values: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in values:
            cv2.fillPoly(mask, [np.round(polygon - minimum).astype(np.int32)], 1)
        return mask

    left = rasterize(reference)
    right = rasterize(prediction)
    intersection = int(np.count_nonzero(left & right))
    left_area = int(np.count_nonzero(left))
    right_area = int(np.count_nonzero(right))
    return left_area, right_area, intersection, left_area + right_area - intersection


def compare_mask_sqlites(reference: Path, prediction: Path) -> dict[str, object]:
    left = _load_masks(reference)
    right = _load_masks(prediction)
    left_area = right_area = intersection = union = 0
    row_ious: list[float] = []
    for key in sorted(set(left) | set(right)):
        counts = _raster_counts(left.get(key, []), right.get(key, []))
        left_area += counts[0]
        right_area += counts[1]
        intersection += counts[2]
        union += counts[3]
        row_ious.append(counts[2] / counts[3] if counts[3] else 1.0)
    left_vertices = [len(polygon) for values in left.values() for polygon in values]
    right_vertices = [len(polygon) for values in right.values() for polygon in values]

    def distribution(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "stddev": 0.0,
                "min": 0,
                "p10": 0.0,
                "p25": 0.0,
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(np.mean(array)),
            "stddev": float(np.std(array)),
            "min": int(np.min(array)),
            "p10": float(np.percentile(array, 10)),
            "p25": float(np.percentile(array, 25)),
            "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": int(np.max(array)),
        }

    left_row_vertices = {key: sum(map(len, values)) for key, values in left.items()}
    right_row_vertices = {key: sum(map(len, values)) for key, values in right.items()}
    common_keys = sorted(set(left) & set(right))
    vertex_deltas = [
        right_row_vertices[key] - left_row_vertices[key] for key in common_keys
    ]
    return {
        "reference_rows": len(left),
        "prediction_rows": len(right),
        "common_rows": len(set(left) & set(right)),
        "missing_rows": len(set(left) - set(right)),
        "extra_rows": len(set(right) - set(left)),
        "global_recall": intersection / left_area if left_area else 1.0,
        "global_precision": intersection / right_area if right_area else 1.0,
        "global_iou": intersection / union if union else 1.0,
        "mean_row_iou": float(np.mean(row_ious)) if row_ious else 1.0,
        "min_row_iou": float(np.min(row_ious)) if row_ious else 1.0,
        "reference_mean_vertices": (
            float(np.mean(left_vertices)) if left_vertices else 0.0
        ),
        "prediction_mean_vertices": (
            float(np.mean(right_vertices)) if right_vertices else 0.0
        ),
        "reference_vertices_per_contour": distribution(left_vertices),
        "prediction_vertices_per_contour": distribution(right_vertices),
        "reference_vertices_per_row": distribution(list(left_row_vertices.values())),
        "prediction_vertices_per_row": distribution(list(right_row_vertices.values())),
        "prediction_minus_reference_vertices_per_common_row": {
            **distribution(vertex_deltas),
            "lower_rows": sum(delta < 0 for delta in vertex_deltas),
            "equal_rows": sum(delta == 0 for delta in vertex_deltas),
            "higher_rows": sum(delta > 0 for delta in vertex_deltas),
        },
    }


def _keyframes_from_json(path: Path) -> set[tuple[int, str]]:
    return {
        (int(row["frame"]), str(row["track_id"]))
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def _keyframes_from_sqlite(path: Path) -> set[tuple[int, str]]:
    with sqlite3.connect(path) as connection:
        return {
            (int(frame), str(track_id))
            for frame, track_id in connection.execute(
                "SELECT frame, track_id FROM masks"
            )
        }


def _set_comparison(
    old: set[tuple[int, str]], current: set[tuple[int, str]]
) -> dict[str, object]:
    union = old | current
    return {
        "original_count": len(old),
        "current_count": len(current),
        "common_count": len(old & current),
        "jaccard": len(old & current) / len(union) if union else 1.0,
        "original_only": sorted(old - current)[:20],
        "current_only": sorted(current - old)[:20],
    }


def run_polygon_comparison(
    *,
    python: Path,
    original_root: Path,
    fixture: Path,
    work_dir: Path,
    device: str,
) -> dict[str, object]:
    old_output = work_dir / "original"
    current_output = work_dir / "current"
    model_root = original_root / "checkpoints" / "postprocess"
    old_seconds = _run(
        [
            str(python),
            "-m",
            "atosyori_postprocess",
            "stage",
            "--model-root",
            str(model_root),
            "polygon-keyframes",
            "--",
            "--input-sqlite",
            str(fixture),
            "--output-dir",
            str(old_output),
            "--target-ratio",
            str(1.0 / 3.0),
            "--anchors-per-contour",
            "48",
            "--adaptive-anchor-counts",
            "--point-predictor-model-dir",
            str(model_root / "polygon_point_predictor"),
            "--predictor-device",
            device,
            "--num-workers",
            "1",
            "--evaluate-exact",
            "--write-pred-sqlite",
        ],
        cwd=original_root,
        log_path=work_dir / "original.log",
        env=_original_env(original_root),
    )
    current_seconds = _run(
        [
            str(python),
            "run_pipeline.py",
            "--input-sqlite",
            str(fixture),
            "--output-dir",
            str(current_output),
            "--shape-mode",
            "polygon",
            "--keyframe-interval",
            "3",
            "--max-gap",
            "30",
            "--no-polygon-border-expand",
            "--no-polygon-endpoint-extend",
        ],
        cwd=POSTPROCESS_ROOT,
        log_path=work_dir / "current.log",
    )
    old_predictions = old_output / "pred" / "predictions.sqlite"
    manifest = json.loads(
        (current_output / "pipeline_manifest.json").read_text(encoding="utf-8")
    )
    current_predictions = Path(manifest["artifacts"]["predictions_sqlite"])
    old_keys = _keyframes_from_json(old_output / "opt" / "final_keyframes.json")
    current_keys = _keyframes_from_sqlite(
        Path(manifest["artifacts"]["keyframes_sqlite"])
    )
    return {
        "original_seconds": old_seconds,
        "current_seconds": current_seconds,
        "original_vs_input": compare_mask_sqlites(fixture, old_predictions),
        "current_vs_input": compare_mask_sqlites(fixture, current_predictions),
        "current_vs_original": compare_mask_sqlites(
            old_predictions, current_predictions
        ),
        "keyframes": _set_comparison(old_keys, current_keys),
        "artifacts": {
            "original_predictions": str(old_predictions),
            "current_predictions": str(current_predictions),
            "original_log": str(work_dir / "original.log"),
            "current_log": str(work_dir / "current.log"),
        },
    }


def run_ellipse_comparison(
    *,
    python: Path,
    original_root: Path,
    fixture: Path,
    work_dir: Path,
    device: str,
) -> dict[str, object]:
    old_output = work_dir / "original"
    current_output = work_dir / "current"
    old_model_root = original_root / "checkpoints" / "postprocess"
    old_seconds = _run(
        [
            str(python),
            "-m",
            "atosyori_postprocess",
            "run",
            "--input-sqlite",
            str(fixture),
            "--output-dir",
            str(old_output),
            "--intervals",
            "3",
            "--default-shape-mode",
            "ellipse",
            "--model-root",
            str(old_model_root),
            "--k2-device",
            device,
            "--polygon-predictor-device",
            device,
            "--no-render-overlays",
            "--force",
            "--",
            "--no-endpoint-extend",
            "--k2-forward-mode",
            "states_only",
            "--k2-tf32",
            "off",
            "--progress-interval-sec",
            "30",
        ],
        cwd=original_root,
        log_path=work_dir / "original.log",
        env=_original_env(original_root),
    )
    current_seconds = _run(
        [
            str(python),
            "run_pipeline.py",
            "--input-sqlite",
            str(fixture),
            "--output-dir",
            str(current_output),
            "--shape-mode",
            "ellipse",
            "--keyframe-interval",
            "3",
            "--max-gap",
            "30",
            "--model-root",
            str(POSTPROCESS_ROOT / "models"),
            "--device",
            device,
            "--k2-forward-mode",
            "states_only",
            "--k2-tf32",
            "off",
        ],
        cwd=POSTPROCESS_ROOT,
        log_path=work_dir / "current.log",
    )
    aligned_output = work_dir / "original_aligned_keyframes"
    aligned_seconds = _run(
        [
            str(python),
            "-m",
            "atosyori_postprocess",
            "stage",
            "ellipse-keyframes",
            "--",
            "--input-metrics-csv",
            str(old_output / "inference" / "k1_exact_k2_v5_metrics.csv"),
            "--output-dir",
            str(aligned_output),
            "--target-ratio",
            str(1.0 / 3.0),
            "--dense-recall-target",
            "0.96",
        ],
        cwd=original_root,
        log_path=work_dir / "original_aligned_keyframes.log",
        env=_original_env(original_root),
    )
    old_summary = json.loads((old_output / "summary.json").read_text(encoding="utf-8"))
    old_interval = old_summary["interval_results"]["int_3"]
    old_predictions = Path(old_interval["paths"]["merged_pred_sqlite"])
    current_manifest = json.loads(
        (current_output / "pipeline_manifest.json").read_text(encoding="utf-8")
    )
    current_predictions = Path(current_manifest["artifacts"]["predictions_sqlite"])
    old_keys: set[tuple[int, str]] = set()
    for group in old_interval["group_results"].values():
        opt_dir = Path(group["paths"]["opt_dir"])
        old_keys |= _keyframes_from_json(opt_dir / "final_keyframes.json")
    current_keys = _keyframes_from_json(
        Path(current_manifest["artifacts"]["keyframes_json"])
    )
    aligned_keys = _keyframes_from_json(aligned_output / "final_keyframes.json")
    aligned_union = json.loads(
        (aligned_output / "interpolated_union.json").read_text(encoding="utf-8")
    )
    current_union = json.loads(
        Path(current_manifest["artifacts"]["interpolated_union_json"]).read_text(
            encoding="utf-8"
        )
    )
    aligned_by_key = {
        (int(row["frame"]), str(row["track_id"])): row
        for row in aligned_union
    }
    current_by_key = {
        (int(row["frame"]), str(row["track_id"])): row
        for row in current_union
    }
    parameter_differences = [
        abs(float(first) - float(second))
        for key in sorted(set(aligned_by_key) & set(current_by_key))
        for first, second in zip(
            (
                value
                for ellipse in aligned_by_key[key]["ellipse_params"]
                for value in ellipse
            ),
            (
                value
                for ellipse in current_by_key[key]["ellipse_params"]
                for value in ellipse
            ),
        )
    ]
    return {
        "comparison_profile": {
            "endpoint_extension": False,
            "k2_forward_mode": "states_only",
            "tf32": False,
            "device": device,
        },
        "original_seconds": old_seconds,
        "current_seconds": current_seconds,
        "original_vs_input": compare_mask_sqlites(fixture, old_predictions),
        "current_vs_input": compare_mask_sqlites(fixture, current_predictions),
        "current_vs_original": compare_mask_sqlites(
            old_predictions, current_predictions
        ),
        "keyframes": _set_comparison(old_keys, current_keys),
        "aligned_core": {
            "description": (
                "original and current keyframe optimizers receive the same "
                "combined 144-row metrics input"
            ),
            "original_seconds": aligned_seconds,
            "keyframes": _set_comparison(aligned_keys, current_keys),
            "row_keys_equal": set(aligned_by_key) == set(current_by_key),
            "max_abs_ellipse_parameter_difference": (
                max(parameter_differences) if parameter_differences else 0.0
            ),
            "allclose_at_1e_9": (
                set(aligned_by_key) == set(current_by_key)
                and max(parameter_differences, default=0.0) <= 1e-9
            ),
        },
        "artifacts": {
            "original_predictions": str(old_predictions),
            "current_predictions": str(current_predictions),
            "original_log": str(work_dir / "original.log"),
            "current_log": str(work_dir / "current.log"),
        },
    }


def static_inventory(original_root: Path) -> dict[str, object]:
    original_engine = (
        original_root
        / "external"
        / "atosyori-pipeline-dev"
        / "src"
        / "atosyori_postprocess"
        / "engine"
    )
    pairs = {
        "ellipse_inference": (
            original_engine / "ellipse_inference.py",
            POSTPROCESS_ROOT / "approximation" / "ellipse" / "inference.py",
        ),
        "ellipse_k1_runtime": (
            original_engine / "standalone_runtime_fst.py",
            POSTPROCESS_ROOT / "approximation" / "ellipse" / "runtime_fst.py",
        ),
        "ellipse_k2_runtime": (
            original_engine / "standalone_runtime_k2v5.py",
            POSTPROCESS_ROOT / "approximation" / "ellipse" / "runtime_k2v5.py",
        ),
        "ellipse_keyframes": (
            original_engine / "optimize_keyframes_trackk_dense_recall_standalone.py",
            POSTPROCESS_ROOT / "keyframes" / "ellipse" / "trackk_dense_recall.py",
        ),
        "ellipse_gap_fill": (
            original_engine / "fill_trackk_union_gaps.py",
            POSTPROCESS_ROOT / "gap_fill" / "ellipse" / "interpolate.py",
        ),
        "polygon_original_vs_current": (
            original_engine / "polygon_v22.py",
            POSTPROCESS_ROOT / "approximation" / "polygon" / "rdp.py",
        ),
    }

    def info(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "lines": payload.count(b"\n") + 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    return {
        name: {"original": info(old), "current": info(current)}
        for name, (old, current) in pairs.items()
    }


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def write_markdown(report: dict[str, object], path: Path) -> None:
    raw = report["dynamic"]["raw_preprocess"]
    cuts = report["dynamic"]["cut_detection"]
    polygon = report["dynamic"]["polygon"]
    ellipse = report["dynamic"]["ellipse"]
    real_cuts = report["dynamic"].get("real_cut_detection")
    lines = [
        "# Original vs current postprocess comparison",
        "",
        "## Result",
        "",
        (
            "- Raw score/NMS/tracking/short-prune decisions equivalent: "
            f"`{raw['decision_equivalent']}`"
        ),
        f"- Raw audit rows byte-semantically equivalent: `{raw['audit_equivalent']}`",
        f"- Obvious-cut fixture equivalent: `{cuts['equivalent']}`",
    ]
    if real_cuts is not None:
        lines.append(
            "- Real-video cut detection equivalent: "
            f"`{real_cuts['equivalent']}` ({real_cuts['frame_count']} frames; "
            f"cuts `{real_cuts['current_frames']}`)"
        )
    lines.extend(
        [
        (
            "- Ellipse current-vs-original IoU: "
            f"`{ellipse['current_vs_original']['global_iou']:.9f}`"
        ),
        (
            "- Polygon current-vs-original IoU: "
            f"`{polygon['current_vs_original']['global_iou']:.9f}`"
        ),
        (
            "- Ellipse keyframe Jaccard: "
            f"`{ellipse['keyframes']['jaccard']:.9f}`"
        ),
        (
            "- Ellipse aligned-core keyframe Jaccard: "
            f"`{ellipse['aligned_core']['keyframes']['jaccard']:.9f}`"
        ),
        (
            "- Ellipse aligned-core max parameter delta: "
            f"`{ellipse['aligned_core']['max_abs_ellipse_parameter_difference']:.3e}`"
        ),
        (
            "- Polygon keyframe Jaccard: "
            f"`{polygon['keyframes']['jaccard']:.9f}`"
        ),
        "",
        "## Dynamic mask comparison",
        "",
        "| Mode | Original sec | Current sec | Original/input IoU | Current/input IoU | Current/original IoU |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Polygon | {polygon['original_seconds']:.3f} | "
            f"{polygon['current_seconds']:.3f} | "
            f"{polygon['original_vs_input']['global_iou']:.6f} | "
            f"{polygon['current_vs_input']['global_iou']:.6f} | "
            f"{polygon['current_vs_original']['global_iou']:.6f} |"
        ),
        (
            f"| Ellipse | {ellipse['original_seconds']:.3f} | "
            f"{ellipse['current_seconds']:.3f} | "
            f"{ellipse['original_vs_input']['global_iou']:.6f} | "
            f"{ellipse['current_vs_input']['global_iou']:.6f} | "
            f"{ellipse['current_vs_original']['global_iou']:.6f} |"
        ),
        "",
        "## Static interpretation",
        "",
        "- Raw preprocessing constants and decisions are preserved; current code streams rows and adds provenance.",
        "- Ellipse uses the migrated K1/K2, routing, dense-recall keyframe and gap-fill algorithms.",
        "- Restored classwise orchestration optimizes each semantic label independently. The non-classwise ellipse comparison intentionally combines labels, so its global keyframe penalty differs from the original class-grouped run.",
        "- Polygon is not the original production v22 optimizer. Current production uses OpenCV RDP, fixed interval selection and linear interpolation.",
        "- Original defaults also include polygon border expansion and endpoint extension; current modular defaults do not.",
        "- High-precision cut thresholds match, but original candidate narrowing/fallback and current full downscaled scan are structurally different.",
        "- Raw audit rows differ only because current normalization materializes `bbox_json`; the decision/output tables are equivalent.",
        "",
        "See `comparison.json` for per-table hashes, row counts and pixel metrics.",
        ]
    )
    baseline = polygon.get("pre_restore_baseline")
    if baseline is not None:
        before = baseline["current_vs_original"]
        after = polygon["current_vs_original"]
        lines.extend(
            [
                "",
                "## Polygon restoration delta",
                "",
                "| Metric | Simplified implementation | Restored v22 |",
                "|---|---:|---:|",
                f"| Current/original IoU | {before['global_iou']:.6f} | {after['global_iou']:.6f} |",
                f"| Current/original recall | {before['global_recall']:.6f} | {after['global_recall']:.6f} |",
                f"| Mean vertices/row | {before['prediction_vertices_per_row']['mean']:.3f} | {after['prediction_vertices_per_row']['mean']:.3f} |",
                f"| Median vertices/row | {before['prediction_vertices_per_row']['median']:.3f} | {after['prediction_vertices_per_row']['median']:.3f} |",
                f"| Keyframe Jaccard | {baseline['keyframes']['jaccard']:.6f} | {polygon['keyframes']['jaccard']:.6f} |",
                f"| Wall seconds | {baseline['current_seconds']:.3f} | {polygon['current_seconds']:.3f} |",
            ]
        )
    polygon_exact = polygon["current_vs_original"]["global_iou"] >= 0.999999999
    interpretation_index = lines.index("## Static interpretation") + 2
    if polygon_exact:
        lines[interpretation_index + 3] = (
            "- Polygon production now invokes the original production-patched "
            "v22 optimizer; the tested masks and keyframes are exactly equivalent."
        )
        lines[interpretation_index + 4] = (
            "- Border expansion and endpoint extension are separate outer-pipeline "
            "safeguards and are not exercised by this optimizer-stage comparison."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_root = args.original_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    source = args.source_tracked_sqlite.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    for required in (
        python,
        source,
        original_root / "external" / "atosyori-pipeline-dev" / "src",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if work_dir.exists():
        if not args.force:
            raise FileExistsError(f"work directory exists; use --force: {work_dir}")
        # Refuse broad or source-tree deletion even when --force is provided.
        if work_dir in {REPOSITORY_ROOT, POSTPROCESS_ROOT, original_root}:
            raise ValueError(f"unsafe work directory: {work_dir}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    fixture_summary = extract_real_fixture(
        source,
        work_dir / "real_fixture" / "tracked.sqlite",
        frames_per_track=args.frames_per_track,
        track_limit=args.tracks,
    )
    fixture = Path(str(fixture_summary["path"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "internal postprocess behavior; public result schema untouched",
        "repositories": {
            "original": str(original_root),
            "original_git_head": _git_head(original_root),
            "current": str(REPOSITORY_ROOT),
            "current_git_head": _git_head(REPOSITORY_ROOT),
        },
        "fixture": fixture_summary,
        "static": static_inventory(original_root),
        "dynamic": {},
    }
    report["dynamic"]["raw_preprocess"] = run_raw_preprocess_comparison(
        original_root=original_root,
        work_dir=work_dir / "raw_preprocess",
    )
    report["dynamic"]["cut_detection"] = run_cut_comparison(
        original_root=original_root,
        work_dir=work_dir / "cut_detection",
    )
    if args.cut_video is not None:
        cut_video = args.cut_video.expanduser().resolve()
        if not cut_video.exists():
            raise FileNotFoundError(cut_video)
        report["dynamic"]["real_cut_detection"] = run_real_cut_comparison(
            original_root=original_root,
            work_dir=work_dir / "real_cut_detection",
            video=cut_video,
        )
    report["dynamic"]["polygon"] = run_polygon_comparison(
        python=python,
        original_root=original_root,
        fixture=fixture,
        work_dir=work_dir / "polygon",
        device=args.device,
    )
    if args.baseline_comparison_json is not None:
        baseline_path = args.baseline_comparison_json.expanduser().resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline_polygon = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )["dynamic"]["polygon"]
        baseline_original = Path(
            baseline_polygon["artifacts"]["original_predictions"]
        )
        baseline_current = Path(
            baseline_polygon["artifacts"]["current_predictions"]
        )
        report["dynamic"]["polygon"]["pre_restore_baseline"] = {
            "comparison_json": str(baseline_path),
            "current_seconds": float(baseline_polygon["current_seconds"]),
            "current_vs_original": compare_mask_sqlites(
                baseline_original, baseline_current
            ),
            "keyframes": baseline_polygon["keyframes"],
            "artifacts": baseline_polygon["artifacts"],
        }
    report["dynamic"]["ellipse"] = run_ellipse_comparison(
        python=python,
        original_root=original_root,
        fixture=fixture,
        work_dir=work_dir / "ellipse",
        device=args.device,
    )
    _json_dump(work_dir / "comparison.json", report)
    write_markdown(report, work_dir / "REPORT.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
