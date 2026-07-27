from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path

import cv2
import numpy as np


def create_video(path: Path, *, frames: int = 4) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer is unavailable")
    for index in range(frames):
        image = np.full((48, 64, 3), 30 + index * 10, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


def create_unified_sqlite(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE videos(
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                reported_frame_count INTEGER,
                fps REAL,
                width INTEGER,
                height INTEGER
            );
            CREATE TABLE model_executions(
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL
            );
            CREATE TABLE frames(
                id INTEGER PRIMARY KEY,
                frame_index INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            CREATE TABLE detections(
                id INTEGER PRIMARY KEY,
                frame_id INTEGER NOT NULL,
                model_execution_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL
            );
            CREATE TABLE classifications(
                detection_id INTEGER PRIMARY KEY,
                class_name TEXT NOT NULL,
                score REAL NOT NULL
            );
            CREATE TABLE segmentations(
                detection_id INTEGER PRIMARY KEY,
                encoding TEXT NOT NULL
            );
            CREATE TABLE segmentation_polygons(
                id INTEGER PRIMARY KEY,
                detection_id INTEGER NOT NULL,
                polygon_index INTEGER NOT NULL
            );
            CREATE TABLE segmentation_points(
                polygon_id INTEGER NOT NULL,
                point_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO schema_info(key, value) VALUES (?, ?)",
            (
                ("schema_name", "instance-segmentation-unified-inference"),
                ("schema_version", "2"),
            ),
        )
        connection.execute("INSERT INTO videos VALUES (1, 'test.avi', 4, 10.0, 64, 48)")
        connection.executemany(
            "INSERT INTO model_executions(id, role) VALUES (?, ?)",
            ((1, "instance_segmentation"), (2, "face_detection")),
        )
        connection.executemany(
            "INSERT INTO frames(id, frame_index, width, height) VALUES (?, ?, 64, 48)",
            ((1, 0), (2, 1), (3, 2), (4, 3)),
        )
        connection.execute(
            """
            INSERT INTO detections
            VALUES (10, 1, 1, 'foreground', 0.91, 8, 8, 32, 32)
            """
        )
        connection.execute("INSERT INTO classifications VALUES (10, 'sample', 0.82)")
        connection.execute("INSERT INTO segmentations VALUES (10, 'polygon')")
        connection.execute("INSERT INTO segmentation_polygons VALUES (100, 10, 0)")
        connection.executemany(
            "INSERT INTO segmentation_points VALUES (100, ?, ?, ?)",
            (
                (0, 8.0, 8.0),
                (1, 32.0, 8.0),
                (2, 32.0, 32.0),
                (3, 8.0, 32.0),
            ),
        )
        connection.execute(
            """
            INSERT INTO detections
            VALUES (20, 2, 2, 'Face', 0.95, 18, 10, 42, 34)
            """
        )
    return path


def create_rich_face_sqlite(path: Path) -> Path:
    create_unified_sqlite(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_info SET value='3' WHERE key='schema_version'"
        )
        connection.execute("ALTER TABLE detections ADD COLUMN group_id INTEGER")
        connection.executescript(
            """
            CREATE TABLE face_observations(
                id INTEGER PRIMARY KEY,
                anchor_detection_id INTEGER NOT NULL,
                head_detection_id INTEGER,
                face_detection_id INTEGER,
                face_score REAL NOT NULL,
                face_present INTEGER NOT NULL,
                geometry_type TEXT,
                ellipse_cx REAL,
                ellipse_cy REAL,
                ellipse_major_radius REAL,
                ellipse_minor_radius REAL,
                ellipse_theta_radians REAL
            );
            CREATE TABLE face_keypoints(
                observation_id INTEGER NOT NULL,
                point_index INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                state INTEGER NOT NULL,
                state_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                valid INTEGER NOT NULL
            );
            CREATE TABLE face_masks(
                observation_id INTEGER PRIMARY KEY,
                encoding TEXT,
                width INTEGER,
                height INTEGER,
                box_x1 REAL,
                box_y1 REAL,
                box_x2 REAL,
                box_y2 REAL,
                data BLOB
            );
            CREATE TABLE face_keypoint_class_probabilities(
                observation_id INTEGER,
                point_index INTEGER,
                class_index INTEGER,
                probability REAL
            );
            CREATE TABLE face_keypoint_state_probabilities(
                observation_id INTEGER,
                point_index INTEGER,
                state_index INTEGER,
                probability REAL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO detections(
                id, frame_id, model_execution_id, class_name, score,
                x1, y1, x2, y2, group_id
            ) VALUES (21, 2, 2, 'Head', 0.97, 12, 5, 48, 42, 0)
            """
        )
        connection.execute(
            """
            INSERT INTO face_masks VALUES(
                1, 'zlib-u8-probability-v1', 4, 4,
                18.0, 10.0, 42.0, 34.0, ?
            )
            """,
            (zlib.compress(bytes(range(16))),),
        )
        connection.execute("UPDATE detections SET group_id=0 WHERE id=20")
        connection.execute(
            """
            INSERT INTO face_observations VALUES(
                1, 21, 21, 20, 0.95, 1, 'ellipse',
                30.0, 22.0, 12.0, 8.0, 0.25
            )
            """
        )
        for index, (name, state, valid) in enumerate(
            (
                ("Eye", 2, 1),
                ("Eye", 1, 1),
                ("Nose", 2, 1),
                ("Mouth", 2, 1),
                ("Mouth", 0, 0),
            )
        ):
            connection.execute(
                """
                INSERT INTO face_keypoints VALUES(
                    1, ?, ?, ?, ?, ?, ?, ?, 0.9, ?
                )
                """,
                (
                    index,
                    (1, 1, 2, 3, 3)[index],
                    name,
                    24.0 + index * 3,
                    18.0 + index,
                    state,
                    ("visible", "occluded", "visible", "visible", "absent")[index],
                    valid,
                ),
            )
    return path


def create_mask_sqlite(path: Path) -> Path:
    polygon = json.dumps([[[10.0, 10.0], [36.0, 10.0], [36.0, 36.0], [10.0, 36.0]]])
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                label TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO masks VALUES (0, '7', ?, 'target')",
            (polygon,),
        )
    return path
