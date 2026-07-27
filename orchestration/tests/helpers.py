from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np


def create_video(path: Path, frames: int = 8) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer is unavailable")
    for frame_index in range(frames):
        frame = np.full((48, 64, 3), 30 + frame_index * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def create_unified_sqlite(path: Path, frames: int = 8) -> Path:
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
                run_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                model_id TEXT NOT NULL,
                runtime_model_id TEXT NOT NULL,
                task TEXT NOT NULL,
                backend TEXT NOT NULL
            );
            CREATE TABLE frames(
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );
            CREATE TABLE detections(
                id INTEGER PRIMARY KEY,
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
                source TEXT NOT NULL
            );
            CREATE TABLE classifications(
                detection_id INTEGER PRIMARY KEY,
                class_id INTEGER NOT NULL,
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
        connection.execute(
            "INSERT INTO videos VALUES (1, 'test.avi', ?, 10.0, 64, 48)",
            (frames,),
        )
        connection.executemany(
            "INSERT INTO model_executions VALUES (?, 1, ?, ?, ?, ?, ?)",
            (
                (
                    1,
                    "instance_segmentation",
                    "test-segmentation",
                    "test-segmentation",
                    "instance_segmentation",
                    "test",
                ),
                (
                    2,
                    "face_detection",
                    "test-face",
                    "test-face",
                    "object_detection",
                    "test",
                ),
            ),
        )
        for frame_index in range(frames):
            frame_id = frame_index + 1
            connection.execute(
                "INSERT INTO frames VALUES (?, 1, ?, ?, 64, 48)",
                (frame_id, frame_index, frame_index / 10.0),
            )
            detection_id = 100 + frame_index
            connection.execute(
                """
                INSERT INTO detections VALUES(
                    ?, ?, 1, 0, 'foreground', 0.95,
                    8, 8, 34, 34, NULL, 'test'
                )
                """,
                (detection_id, frame_id),
            )
            connection.execute(
                "INSERT INTO classifications VALUES (?, 1, 'target', 0.90)",
                (detection_id,),
            )
            connection.execute(
                "INSERT INTO segmentations VALUES (?, 'polygon')",
                (detection_id,),
            )
            polygon_id = 1000 + frame_index
            connection.execute(
                "INSERT INTO segmentation_polygons VALUES (?, ?, 0)",
                (polygon_id, detection_id),
            )
            connection.executemany(
                "INSERT INTO segmentation_points VALUES (?, ?, ?, ?)",
                (
                    (polygon_id, 0, 8.0, 8.0),
                    (polygon_id, 1, 34.0, 8.0),
                    (polygon_id, 2, 34.0, 34.0),
                    (polygon_id, 3, 8.0, 34.0),
                ),
            )
            face_id = 200 + frame_index
            connection.execute(
                """
                INSERT INTO detections VALUES(
                    ?, ?, 2, 0, 'Face', 0.93,
                    18, 10, 42, 35, NULL, 'test'
                )
                """,
                (face_id, frame_id),
            )
    return path

