"""Schema for rich multi-model inference, including face geometry."""

from __future__ import annotations

import sqlite3


SCHEMA_NAME = "instance-segmentation-unified-inference"
SCHEMA_VERSION = 3

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
    group_id INTEGER,
    FOREIGN KEY(frame_id) REFERENCES frames(id),
    FOREIGN KEY(model_execution_id) REFERENCES model_executions(id),
    CHECK(group_id IS NULL OR group_id >= 0)
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
CREATE TABLE face_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_detection_id INTEGER NOT NULL UNIQUE,
    head_detection_id INTEGER UNIQUE,
    face_detection_id INTEGER UNIQUE,
    face_score REAL NOT NULL,
    face_present INTEGER NOT NULL,
    geometry_type TEXT,
    ellipse_cx REAL,
    ellipse_cy REAL,
    ellipse_major_radius REAL,
    ellipse_minor_radius REAL,
    ellipse_theta_radians REAL,
    FOREIGN KEY(anchor_detection_id) REFERENCES detections(id),
    FOREIGN KEY(head_detection_id) REFERENCES detections(id),
    FOREIGN KEY(face_detection_id) REFERENCES detections(id),
    CHECK(face_present IN (0, 1)),
    CHECK(face_score >= 0 AND face_score <= 1),
    CHECK(
        (
            face_present = 0
            AND geometry_type IS NULL
            AND ellipse_cx IS NULL
            AND ellipse_cy IS NULL
            AND ellipse_major_radius IS NULL
            AND ellipse_minor_radius IS NULL
            AND ellipse_theta_radians IS NULL
        )
        OR
        (
            face_present = 1
            AND geometry_type = 'ellipse'
            AND ellipse_cx IS NOT NULL
            AND ellipse_cy IS NOT NULL
            AND ellipse_major_radius >= 0
            AND ellipse_minor_radius >= 0
            AND ellipse_theta_radians IS NOT NULL
        )
    )
);
CREATE TABLE face_keypoints (
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    state INTEGER NOT NULL,
    state_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid INTEGER NOT NULL,
    PRIMARY KEY(observation_id, point_index),
    FOREIGN KEY(observation_id) REFERENCES face_observations(id),
    CHECK(point_index >= 0 AND point_index < 5),
    CHECK(class_id >= 0 AND class_id <= 3),
    CHECK(state >= 0 AND state <= 2),
    CHECK(confidence >= 0 AND confidence <= 1),
    CHECK(valid IN (0, 1))
);
CREATE TABLE face_masks (
    observation_id INTEGER PRIMARY KEY,
    encoding TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    box_x1 REAL NOT NULL,
    box_y1 REAL NOT NULL,
    box_x2 REAL NOT NULL,
    box_y2 REAL NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY(observation_id) REFERENCES face_observations(id),
    CHECK(encoding = 'zlib-u8-probability-v1'),
    CHECK(width > 0 AND height > 0),
    CHECK(box_x2 >= box_x1 AND box_y2 >= box_y1)
);
CREATE TABLE face_keypoint_class_probabilities (
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    class_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(observation_id, point_index, class_index),
    FOREIGN KEY(observation_id, point_index)
        REFERENCES face_keypoints(observation_id, point_index),
    CHECK(class_index >= 0 AND class_index <= 3),
    CHECK(probability >= 0 AND probability <= 1)
);
CREATE TABLE face_keypoint_state_probabilities (
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    state_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(observation_id, point_index, state_index),
    FOREIGN KEY(observation_id, point_index)
        REFERENCES face_keypoints(observation_id, point_index),
    CHECK(state_index >= 1 AND state_index <= 2),
    CHECK(probability >= 0 AND probability <= 1)
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
CREATE INDEX idx_detections_frame_group
    ON detections(frame_id, model_execution_id, group_id);
CREATE INDEX idx_segmentation_polygons_detection
    ON segmentation_polygons(detection_id);
CREATE INDEX idx_face_observations_head
    ON face_observations(head_detection_id);
CREATE INDEX idx_face_observations_face
    ON face_observations(face_detection_id);
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
