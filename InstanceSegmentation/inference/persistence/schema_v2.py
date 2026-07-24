"""Schema for one video run containing one or more inference models."""

from __future__ import annotations

import sqlite3


SCHEMA_NAME = "instance-segmentation-unified-inference"
SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    reported_frame_count INTEGER,
    fps REAL,
    width INTEGER,
    height INTEGER
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);
CREATE TABLE run_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL
);
CREATE TABLE model_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    runtime_model_id TEXT NOT NULL,
    task TEXT NOT NULL,
    backend TEXT NOT NULL,
    UNIQUE(run_id, role),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE model_metadata (
    model_execution_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL,
    PRIMARY KEY(model_execution_id, key),
    FOREIGN KEY(model_execution_id) REFERENCES model_executions(id)
);
CREATE TABLE frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    UNIQUE(run_id, frame_index),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    model_execution_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    score REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    track_id INTEGER,
    source TEXT NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id),
    FOREIGN KEY(model_execution_id) REFERENCES model_executions(id)
);
CREATE TABLE classifications (
    detection_id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    score REAL NOT NULL,
    FOREIGN KEY(detection_id) REFERENCES detections(id)
);
CREATE TABLE classification_probabilities (
    detection_id INTEGER NOT NULL,
    class_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(detection_id, class_index),
    FOREIGN KEY(detection_id) REFERENCES classifications(detection_id)
);
CREATE TABLE segmentations (
    detection_id INTEGER PRIMARY KEY,
    encoding TEXT NOT NULL,
    FOREIGN KEY(detection_id) REFERENCES detections(id)
);
CREATE TABLE segmentation_polygons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    polygon_index INTEGER NOT NULL,
    UNIQUE(detection_id, polygon_index),
    FOREIGN KEY(detection_id) REFERENCES segmentations(detection_id)
);
CREATE TABLE segmentation_points (
    polygon_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    PRIMARY KEY(polygon_id, point_index),
    FOREIGN KEY(polygon_id) REFERENCES segmentation_polygons(id)
);
"""

INDEX_SQL = """
CREATE INDEX idx_frames_run_index
    ON frames(run_id, frame_index);
CREATE INDEX idx_detections_frame
    ON detections(frame_id);
CREATE INDEX idx_detections_execution
    ON detections(model_execution_id);
CREATE INDEX idx_detections_class
    ON detections(class_id);
CREATE INDEX idx_segmentation_polygons_detection
    ON segmentation_polygons(detection_id);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.executemany(
        "INSERT INTO schema_info(key, value) VALUES (?, ?)",
        (
            ("schema_name", SCHEMA_NAME),
            ("schema_version", str(SCHEMA_VERSION)),
        ),
    )


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(INDEX_SQL)


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "create_indexes",
    "initialize_schema",
]
