"""SQLite schema emitted by the tracking stage."""

from __future__ import annotations

import sqlite3


def create_schema(connection: sqlite3.Connection) -> None:
    """Create a fresh tracked-mask schema on an open connection."""

    cursor = connection.cursor()
    for table in (
        "masks",
        "tracks",
        "cuts",
        "cut_detection_metadata",
        "raw_tracked_masks",
        "raw_tracks",
    ):
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    cursor.execute(
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
        )
        """
    )
    cursor.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)")
    cursor.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
    cursor.execute(
        """
        CREATE TABLE cut_detection_metadata(
            id INTEGER PRIMARY KEY CHECK(id = 1),
            schema_version INTEGER NOT NULL,
            method TEXT NOT NULL CHECK(length(method) > 0),
            elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0),
            cut_count INTEGER NOT NULL CHECK(cut_count >= 0),
            frame_semantics TEXT NOT NULL
                CHECK(frame_semantics = 'first_frame_of_new_scene')
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE raw_tracked_masks(
            frame INTEGER NOT NULL,
            raw_track_id TEXT NOT NULL,
            raw_detection_index INTEGER NOT NULL,
            source_detection_id INTEGER,
            final_track_id TEXT,
            removed_by_short_track INTEGER NOT NULL DEFAULT 0,
            raw_track_length INTEGER NOT NULL DEFAULT 0,
            raw_label TEXT,
            final_label TEXT,
            polygons TEXT,
            score REAL,
            detector_score REAL,
            class_score REAL,
            category_id INTEGER,
            category_index INTEGER,
            bbox_xyxy_json TEXT,
            bbox_json TEXT,
            scene_id INTEGER,
            PRIMARY KEY(frame, raw_track_id, raw_detection_index)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE raw_tracks(
            raw_track_id TEXT PRIMARY KEY,
            final_track_id TEXT,
            removed_by_short_track INTEGER NOT NULL DEFAULT 0,
            raw_track_length INTEGER NOT NULL DEFAULT 0,
            raw_label TEXT,
            final_label TEXT,
            scene_id INTEGER
        )
        """
    )
    cursor.execute(
        "CREATE INDEX idx_raw_tracked_masks_track_frame "
        "ON raw_tracked_masks(raw_track_id, frame)"
    )
    cursor.execute(
        "CREATE INDEX idx_raw_tracked_masks_final_track_frame "
        "ON raw_tracked_masks(final_track_id, frame)"
    )
