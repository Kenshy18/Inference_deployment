"""Associate canonical detections and persist tracked SQLite.

Score filtering, NMS, and cut detection are upstream artifacts.  This module
contains tracking only.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from contracts.detections import CutList

from .association import (
    AssociationConfig,
    TrackState,
    associate,
    detection_features,
)
from .schema import create_schema
from .records import iter_tracking_records


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _json_or_none(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _majority_label(
    track_id: str,
    counts: dict[str, dict[str, int]],
    first_seen: dict[tuple[str, str], int],
) -> str:
    candidates = counts.get(track_id, {})
    if not candidates:
        return ""
    return max(
        candidates.items(),
        key=lambda item: (item[1], -first_seen.get((track_id, item[0]), 0)),
    )[0]


def build_tracked_sqlite(
    jsonl_path: Path,
    sqlite_path: Path,
    cuts_path: Path,
    *,
    remove_short_tracks_max_frames: int = 2,
    association_config: AssociationConfig | None = None,
) -> dict[str, object]:
    """Track an NMS-filtered JSONL artifact using an explicit cut-list."""

    started = time.perf_counter()
    jsonl_path = Path(jsonl_path)
    sqlite_path = Path(sqlite_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")
    if remove_short_tracks_max_frames < 0:
        raise ValueError("remove_short_tracks_max_frames must be >= 0")

    cut_result = CutList.read(cuts_path)
    cut_frames = list(cut_result.frames)
    cut_frame_set = set(cut_frames)
    association_settings = association_config or AssociationConfig()

    tracks: dict[int, TrackState] = {}
    active_track_ids: list[int] = []
    next_track_id = 1
    current_scene_id = 0
    total_rows = 0
    label_seen_order = 0
    track_label_counts: dict[str, dict[str, int]] = {}
    track_label_first_seen: dict[tuple[str, str], int] = {}
    mask_rows: list[tuple[int, str, str, str]] = []
    raw_rows: list[dict[str, Any]] = []

    for frame_index, frame_detections in iter_tracking_records(jsonl_path):
        if frame_index in cut_frame_set:
            current_scene_id += 1
            active_track_ids.clear()

        detections = frame_detections
        features = [detection_features(detection) for detection in detections]
        active_track_ids = [
            track_id
            for track_id in active_track_ids
            if frame_index - tracks[track_id].last_frame
            <= association_settings.max_gap_frames
        ]
        assignments = associate(
            [tracks[track_id] for track_id in active_track_ids],
            features,
            frame_index,
            association_settings,
        )

        for detection_index, feature in enumerate(features):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = TrackState(
                    track_id=track_id,
                    scene_id=current_scene_id,
                    last_frame=frame_index,
                    features=feature,
                )
                active_track_ids.append(track_id)
                assignments[detection_index] = track_id
            else:
                tracks[track_id].update(frame_index, feature)

        for detection_index, detection in enumerate(detections):
            polygons = detection.get("polygons") or []
            if not polygons:
                continue
            raw_track_id = str(assignments[detection_index])
            final_candidate_label = str(detection.get("class_name", ""))
            raw_label = str(detection.get("label", final_candidate_label))
            polygons_json = json.dumps(
                polygons, ensure_ascii=False, separators=(",", ":")
            )
            mask_rows.append(
                (frame_index, raw_track_id, polygons_json, final_candidate_label)
            )
            track = tracks[int(raw_track_id)]
            raw_rows.append(
                {
                    "frame": frame_index,
                    "raw_track_id": raw_track_id,
                    "raw_detection_index": detection_index,
                    "raw_label": raw_label,
                    "polygons": polygons_json,
                    "score": _optional_float(detection.get("score")),
                    "detector_score": _optional_float(detection.get("detector_score")),
                    "class_score": _optional_float(detection.get("class_score")),
                    "category_id": _optional_int(detection.get("category_id")),
                    "category_index": _optional_int(detection.get("category_index")),
                    "bbox_xyxy_json": _json_or_none(detection.get("bbox_xyxy")),
                    "bbox_json": _json_or_none(detection.get("bbox")),
                    "scene_id": track.scene_id,
                }
            )
            label_counts = track_label_counts.setdefault(raw_track_id, {})
            label_counts[final_candidate_label] = (
                label_counts.get(final_candidate_label, 0) + 1
            )
            label_key = (raw_track_id, final_candidate_label)
            if label_key not in track_label_first_seen:
                track_label_first_seen[label_key] = label_seen_order
                label_seen_order += 1
            total_rows += 1

    track_lengths: dict[str, int] = {}
    for _frame, track_id, _polygons, _label in mask_rows:
        track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
    removed_track_ids = {
        track_id
        for track_id, length in track_lengths.items()
        if length <= remove_short_tracks_max_frames
    }
    kept_track_ids = sorted(
        (track_id for track_id in track_lengths if track_id not in removed_track_ids),
        key=int,
    )
    final_id_by_raw = {
        raw_track_id: str(index)
        for index, raw_track_id in enumerate(kept_track_ids, start=1)
    }
    majority_by_raw = {
        track_id: _majority_label(track_id, track_label_counts, track_label_first_seen)
        for track_id in track_lengths
    }
    final_mask_rows = [
        (
            frame,
            final_id_by_raw[raw_track_id],
            polygons,
            majority_by_raw[raw_track_id],
        )
        for frame, raw_track_id, polygons, _label in mask_rows
        if raw_track_id in final_id_by_raw
    ]
    final_track_rows = [
        (final_id_by_raw[raw_track_id], majority_by_raw[raw_track_id])
        for raw_track_id in kept_track_ids
    ]
    raw_mask_rows = [
        (
            row["frame"],
            row["raw_track_id"],
            row["raw_detection_index"],
            final_id_by_raw.get(str(row["raw_track_id"])),
            int(str(row["raw_track_id"]) not in final_id_by_raw),
            track_lengths.get(str(row["raw_track_id"]), 0),
            row["raw_label"],
            (
                majority_by_raw.get(str(row["raw_track_id"]))
                if str(row["raw_track_id"]) in final_id_by_raw
                else None
            ),
            row["polygons"],
            row["score"],
            row["detector_score"],
            row["class_score"],
            row["category_id"],
            row["category_index"],
            row["bbox_xyxy_json"],
            row["bbox_json"],
            row["scene_id"],
        )
        for row in raw_rows
    ]
    raw_track_rows = [
        (
            raw_track_id,
            final_id_by_raw.get(raw_track_id),
            int(raw_track_id not in final_id_by_raw),
            track_lengths[raw_track_id],
            majority_by_raw[raw_track_id],
            (
                majority_by_raw[raw_track_id]
                if raw_track_id in final_id_by_raw
                else None
            ),
            tracks[int(raw_track_id)].scene_id,
        )
        for raw_track_id in sorted(track_lengths, key=int)
    ]

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    with sqlite3.connect(str(sqlite_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        create_schema(connection)
        if final_mask_rows:
            connection.executemany(
                """
                INSERT OR REPLACE INTO masks(
                    frame, track_id, polygons, shape_type, dilate_px,
                    feather_px, mosaic_block, mosaic_alias, label
                )
                VALUES (?, ?, ?, 'polygon', 0, 0, 0, 0, ?)
                """,
                final_mask_rows,
            )
        if final_track_rows:
            connection.executemany(
                "INSERT OR REPLACE INTO tracks(track_id, label) VALUES (?, ?)",
                final_track_rows,
            )
        if raw_mask_rows:
            connection.executemany(
                """
                INSERT OR REPLACE INTO raw_tracked_masks(
                    frame, raw_track_id, raw_detection_index, final_track_id,
                    removed_by_short_track, raw_track_length, raw_label,
                    final_label, polygons, score, detector_score, class_score,
                    category_id, category_index, bbox_xyxy_json, bbox_json,
                    scene_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                raw_mask_rows,
            )
        if raw_track_rows:
            connection.executemany(
                """
                INSERT OR REPLACE INTO raw_tracks(
                    raw_track_id, final_track_id, removed_by_short_track,
                    raw_track_length, raw_label, final_label, scene_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                raw_track_rows,
            )
        if cut_frames:
            connection.executemany(
                "INSERT OR IGNORE INTO cuts(frame) VALUES (?)",
                [(frame,) for frame in cut_frames],
            )
        connection.execute(
            """
            INSERT INTO cut_detection_metadata(
                id, schema_version, method, elapsed_seconds, cut_count,
                frame_semantics
            )
            VALUES (1, 1, ?, ?, ?, 'first_frame_of_new_scene')
            """,
            (
                cut_result.method,
                cut_result.elapsed_seconds,
                len(cut_frames),
            ),
        )

    removed_rows = sum(track_lengths[track_id] for track_id in removed_track_ids)
    mixed_label_tracks = sum(
        len(track_label_counts.get(track_id, {})) > 1 for track_id in kept_track_ids
    )
    relabeled_rows = sum(
        raw_track_id in final_id_by_raw and label != majority_by_raw[raw_track_id]
        for _frame, raw_track_id, _polygons, label in mask_rows
    )
    return {
        "input_jsonl": str(jsonl_path),
        "tracked_sqlite": str(sqlite_path),
        "rows_before_prune": total_rows,
        "rows_after_prune": len(final_mask_rows),
        "removed_short_tracks": len(removed_track_ids),
        "removed_rows": removed_rows,
        "tracks_after_prune": len(final_id_by_raw),
        "raw_tracked_rows": len(raw_mask_rows),
        "raw_tracks": len(raw_track_rows),
        "raw_removed_rows": removed_rows,
        "mixed_label_tracks": mixed_label_tracks,
        "relabeled_mask_rows": relabeled_rows,
        "track_label_policy": "majority_vote_per_track",
        "cuts_detected": len(cut_frames),
        "scenes": current_scene_id + 1,
        "cut_detection_method": cut_result.method,
        "cut_detection_elapsed_sec": cut_result.elapsed_seconds,
        "remove_short_tracks_max_frames": remove_short_tracks_max_frames,
        "elapsed_sec": time.perf_counter() - started,
    }
