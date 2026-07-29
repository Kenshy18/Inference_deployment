from __future__ import annotations

import sqlite3
import json
import zlib
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


def create_mask_sqlite(path: Path, *, track_id: str = "1") -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
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
        connection.execute(
            """
            INSERT INTO masks(
                frame, track_id, polygons, shape_type, label
            ) VALUES (?, ?, ?, 'polygon', 'target')
            """,
            (
                0,
                track_id,
                json.dumps([[[8, 8], [34, 8], [34, 34], [8, 34]]]),
            ),
        )
        connection.execute(
            "INSERT INTO tracks(track_id, label) VALUES (?, 'target')",
            (track_id,),
        )
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
                    "rtdetr_head_face",
                    "rtdetr_head_face",
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


def create_rich_face_unified_sqlite(path: Path) -> Path:
    create_unified_sqlite(path, frames=1)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_info SET value='3' WHERE key='schema_version'"
        )
        connection.execute(
            """
            UPDATE model_executions
            SET model_id='face_dino_v2',
                runtime_model_id='face_dino_v2',
                backend='tensorrt-fast'
            WHERE role='face_detection'
            """
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
            INSERT INTO detections VALUES(
                300, 1, 2, 1, 'Head', 0.97,
                8, 4, 50, 44, NULL, 'head_detection', 0
            )
            """
        )
        connection.execute(
            """
            INSERT INTO face_masks VALUES(
                1, 'zlib-u8-probability-v1', 64, 64,
                18.0, 10.0, 42.0, 35.0, ?
            )
            """,
            (zlib.compress(bytes([192]) * (64 * 64)),),
        )
        connection.execute("UPDATE detections SET group_id=0 WHERE id=200")
        connection.execute(
            """
            INSERT INTO face_observations VALUES(
                1, 300, 300, 200, 0.93, 1, 'ellipse',
                30.0, 22.0, 12.0, 8.0, 0.2
            )
            """
        )
        for point_index in range(5):
            class_id = (1, 1, 2, 3, 3)[point_index]
            class_name = ("Eye", "Eye", "Nose", "Mouth", "Mouth")[point_index]
            state = (2, 1, 2, 2, 0)[point_index]
            state_name = (
                "visible",
                "occluded",
                "visible",
                "visible",
                "absent",
            )[point_index]
            connection.execute(
                """
                INSERT INTO face_keypoints VALUES(
                    1, ?, ?, ?, ?, ?, ?, ?, 0.9, ?
                )
                """,
                (
                    point_index,
                    class_id,
                    class_name,
                    20.0 + point_index,
                    18.0 + point_index,
                    state,
                    state_name,
                    int(point_index < 4),
                ),
            )
            connection.executemany(
                """
                INSERT INTO face_keypoint_class_probabilities
                VALUES(1, ?, ?, ?)
                """,
                ((point_index, class_index, 0.25) for class_index in range(4)),
            )
            connection.executemany(
                """
                INSERT INTO face_keypoint_state_probabilities
                VALUES(1, ?, ?, ?)
                """,
                ((point_index, state_index, 0.5) for state_index in (1, 2)),
            )
    return path


def keep_only_inference_role(path: Path, role: str) -> Path:
    """Reduce a test unified SQLite to one model role without changing schema."""

    if role not in {"instance_segmentation", "face_detection"}:
        raise ValueError(role)
    with sqlite3.connect(path) as connection:
        removed_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM model_executions WHERE role<>?",
                (role,),
            )
        ]
        if removed_ids:
            placeholders = ",".join("?" for _value in removed_ids)
            detection_ids = [
                int(row[0])
                for row in connection.execute(
                    f"""
                    SELECT id FROM detections
                    WHERE model_execution_id IN ({placeholders})
                    """,
                    removed_ids,
                )
            ]
            if detection_ids:
                detection_placeholders = ",".join("?" for _value in detection_ids)
                polygon_ids = [
                    int(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT id FROM segmentation_polygons
                        WHERE detection_id IN ({detection_placeholders})
                        """,
                        detection_ids,
                    )
                ]
                if polygon_ids:
                    polygon_placeholders = ",".join("?" for _value in polygon_ids)
                    connection.execute(
                        f"""
                        DELETE FROM segmentation_points
                        WHERE polygon_id IN ({polygon_placeholders})
                        """,
                        polygon_ids,
                    )
                for table in (
                    "segmentation_polygons",
                    "segmentations",
                    "classifications",
                ):
                    connection.execute(
                        f"""
                        DELETE FROM {table}
                        WHERE detection_id IN ({detection_placeholders})
                        """,
                        detection_ids,
                    )
                connection.execute(
                    f"""
                    DELETE FROM detections
                    WHERE id IN ({detection_placeholders})
                    """,
                    detection_ids,
                )
            connection.execute(
                f"""
                DELETE FROM model_executions
                WHERE id IN ({placeholders})
                """,
                removed_ids,
            )
    return path
