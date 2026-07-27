"""Associate canonical detections and persist tracked SQLite.

Score filtering, NMS, and cut detection are upstream artifacts.  This module
contains tracking only.  Per-mask rows are staged in SQLite in bounded batches
so Python memory does not grow with video duration.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from contracts.detections import CutList

from .association import (
    AssociationConfig,
    TrackState,
    associate,
    detection_features,
)
from .records import iter_tracking_records
from .schema import create_schema


_STAGING_BATCH_SIZE = 1024


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


def _create_staging_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE tracking_row_staging(
            frame INTEGER NOT NULL,
            raw_track_id TEXT NOT NULL,
            raw_detection_index INTEGER NOT NULL,
            candidate_label TEXT NOT NULL,
            raw_label TEXT NOT NULL,
            polygons TEXT NOT NULL,
            score REAL,
            detector_score REAL,
            class_score REAL,
            category_id INTEGER,
            category_index INTEGER,
            bbox_xyxy_json TEXT,
            bbox_json TEXT,
            scene_id INTEGER NOT NULL,
            PRIMARY KEY(frame, raw_track_id, raw_detection_index)
        );
        CREATE TABLE track_resolution_staging(
            raw_track_id TEXT PRIMARY KEY,
            final_track_id TEXT,
            removed_by_short_track INTEGER NOT NULL,
            raw_track_length INTEGER NOT NULL,
            majority_label TEXT NOT NULL,
            scene_id INTEGER NOT NULL
        );
        """
    )


def _flush_staging_rows(
    connection: sqlite3.Connection,
    rows: list[tuple[object, ...]],
) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO tracking_row_staging(
            frame, raw_track_id, raw_detection_index,
            candidate_label, raw_label, polygons,
            score, detector_score, class_score,
            category_id, category_index,
            bbox_xyxy_json, bbox_json, scene_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    rows.clear()


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


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

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{sqlite_path.name}.",
        suffix=".building",
        dir=sqlite_path.parent,
        delete=False,
    )
    temporary.close()
    staging_path = Path(temporary.name)

    tracks: dict[int, TrackState] = {}
    track_scene_ids: dict[str, int] = {}
    active_track_ids: list[int] = []
    next_track_id = 1
    current_scene_id = 0
    total_rows = 0
    label_seen_order = 0
    track_lengths: dict[str, int] = {}
    track_label_counts: dict[str, dict[str, int]] = {}
    track_label_first_seen: dict[tuple[str, str], int] = {}
    staging_rows: list[tuple[object, ...]] = []

    try:
        with sqlite3.connect(str(staging_path)) as connection:
            # This file is private until the final atomic replace.  Journaling
            # every staging and projection write only duplicates I/O; a crash
            # leaves the previously published output untouched.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            create_schema(connection)
            _create_staging_schema(connection)

            for frame_index, frame_detections in iter_tracking_records(jsonl_path):
                if frame_index in cut_frame_set:
                    current_scene_id += 1
                    for track_id in active_track_ids:
                        tracks.pop(track_id, None)
                    active_track_ids.clear()

                detections = frame_detections
                features = [detection_features(detection) for detection in detections]
                retained_track_ids: list[int] = []
                for track_id in active_track_ids:
                    if (
                        frame_index - tracks[track_id].last_frame
                        <= association_settings.max_gap_frames
                    ):
                        retained_track_ids.append(track_id)
                    else:
                        tracks.pop(track_id, None)
                active_track_ids = retained_track_ids
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
                        track_scene_ids[str(track_id)] = current_scene_id
                        active_track_ids.append(track_id)
                        assignments[detection_index] = track_id
                    else:
                        tracks[track_id].update(frame_index, feature)

                for detection_index, detection in enumerate(detections):
                    polygons = detection.get("polygons") or []
                    if not polygons:
                        continue
                    raw_track_id = str(assignments[detection_index])
                    candidate_label = str(detection.get("class_name", ""))
                    raw_label = str(detection.get("label", candidate_label))
                    polygons_json = json.dumps(
                        polygons, ensure_ascii=False, separators=(",", ":")
                    )
                    staging_rows.append(
                        (
                            frame_index,
                            raw_track_id,
                            detection_index,
                            candidate_label,
                            raw_label,
                            polygons_json,
                            _optional_float(detection.get("score")),
                            _optional_float(detection.get("detector_score")),
                            _optional_float(detection.get("class_score")),
                            _optional_int(detection.get("category_id")),
                            _optional_int(detection.get("category_index")),
                            _json_or_none(detection.get("bbox_xyxy")),
                            _json_or_none(detection.get("bbox")),
                            track_scene_ids[raw_track_id],
                        )
                    )
                    if len(staging_rows) >= _STAGING_BATCH_SIZE:
                        _flush_staging_rows(connection, staging_rows)

                    track_lengths[raw_track_id] = track_lengths.get(raw_track_id, 0) + 1
                    label_counts = track_label_counts.setdefault(raw_track_id, {})
                    label_counts[candidate_label] = (
                        label_counts.get(candidate_label, 0) + 1
                    )
                    label_key = (raw_track_id, candidate_label)
                    if label_key not in track_label_first_seen:
                        track_label_first_seen[label_key] = label_seen_order
                        label_seen_order += 1
                    total_rows += 1

            _flush_staging_rows(connection, staging_rows)

            removed_track_ids = {
                track_id
                for track_id, length in track_lengths.items()
                if length <= remove_short_tracks_max_frames
            }
            kept_track_ids = sorted(
                (
                    track_id
                    for track_id in track_lengths
                    if track_id not in removed_track_ids
                ),
                key=int,
            )
            final_id_by_raw = {
                raw_track_id: str(index)
                for index, raw_track_id in enumerate(kept_track_ids, start=1)
            }
            majority_by_raw = {
                track_id: _majority_label(
                    track_id,
                    track_label_counts,
                    track_label_first_seen,
                )
                for track_id in track_lengths
            }
            connection.executemany(
                """
                INSERT INTO track_resolution_staging(
                    raw_track_id, final_track_id, removed_by_short_track,
                    raw_track_length, majority_label, scene_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        raw_track_id,
                        final_id_by_raw.get(raw_track_id),
                        int(raw_track_id not in final_id_by_raw),
                        track_lengths[raw_track_id],
                        majority_by_raw[raw_track_id],
                        track_scene_ids[raw_track_id],
                    )
                    for raw_track_id in sorted(track_lengths, key=int)
                ),
            )

            connection.execute(
                """
                INSERT OR REPLACE INTO masks(
                    frame, track_id, polygons, shape_type, dilate_px,
                    feather_px, mosaic_block, mosaic_alias, label
                )
                SELECT s.frame, r.final_track_id, s.polygons,
                       'polygon', 0, 0, 0, 0, r.majority_label
                FROM tracking_row_staging s
                JOIN track_resolution_staging r
                  ON r.raw_track_id=s.raw_track_id
                WHERE r.removed_by_short_track=0
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO tracks(track_id, label)
                SELECT final_track_id, majority_label
                FROM track_resolution_staging
                WHERE removed_by_short_track=0
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO raw_tracked_masks(
                    frame, raw_track_id, raw_detection_index, final_track_id,
                    removed_by_short_track, raw_track_length, raw_label,
                    final_label, polygons, score, detector_score, class_score,
                    category_id, category_index, bbox_xyxy_json, bbox_json,
                    scene_id
                )
                SELECT s.frame, s.raw_track_id, s.raw_detection_index,
                       r.final_track_id, r.removed_by_short_track,
                       r.raw_track_length, s.raw_label,
                       CASE WHEN r.removed_by_short_track=0
                            THEN r.majority_label ELSE NULL END,
                       s.polygons, s.score, s.detector_score, s.class_score,
                       s.category_id, s.category_index,
                       s.bbox_xyxy_json, s.bbox_json, s.scene_id
                FROM tracking_row_staging s
                JOIN track_resolution_staging r
                  ON r.raw_track_id=s.raw_track_id
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO raw_tracks(
                    raw_track_id, final_track_id, removed_by_short_track,
                    raw_track_length, raw_label, final_label, scene_id
                )
                SELECT raw_track_id, final_track_id, removed_by_short_track,
                       raw_track_length, majority_label,
                       CASE WHEN removed_by_short_track=0
                            THEN majority_label ELSE NULL END,
                       scene_id
                FROM track_resolution_staging
                """
            )

            if cut_frames:
                connection.executemany(
                    "INSERT OR IGNORE INTO cuts(frame) VALUES (?)",
                    ((frame,) for frame in cut_frames),
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

            relabeled_rows = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tracking_row_staging s
                    JOIN track_resolution_staging r
                      ON r.raw_track_id=s.raw_track_id
                    WHERE r.removed_by_short_track=0
                      AND s.candidate_label != r.majority_label
                    """
                ).fetchone()[0]
            )
            connection.execute("DROP TABLE tracking_row_staging")
            connection.execute("DROP TABLE track_resolution_staging")
            connection.commit()
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            )
            if journal_mode.lower() != "wal":
                raise RuntimeError(
                    f"failed to finalize tracked SQLite journal: {journal_mode}"
                )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(
                    f"tracked SQLite integrity check failed: {integrity}"
                )
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise RuntimeError("tracked SQLite WAL checkpoint remained busy")

        _remove_sidecars(staging_path)
        _remove_sidecars(sqlite_path)
        os.replace(staging_path, sqlite_path)
    except BaseException:
        if staging_path.exists():
            staging_path.unlink()
        _remove_sidecars(staging_path)
        raise

    removed_rows = sum(track_lengths[track_id] for track_id in removed_track_ids)
    mixed_label_tracks = sum(
        len(track_label_counts.get(track_id, {})) > 1 for track_id in kept_track_ids
    )
    rows_after_prune = total_rows - removed_rows
    return {
        "input_jsonl": str(jsonl_path),
        "tracked_sqlite": str(sqlite_path),
        "rows_before_prune": total_rows,
        "rows_after_prune": rows_after_prune,
        "removed_short_tracks": len(removed_track_ids),
        "removed_rows": removed_rows,
        "tracks_after_prune": len(final_id_by_raw),
        "raw_tracked_rows": total_rows,
        "raw_tracks": len(track_lengths),
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
