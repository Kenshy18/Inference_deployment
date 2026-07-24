"""Small synthetic datasets used by tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def rectangle(x: float, y: float, width: float, height: float) -> list[list[float]]:
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def moving_rectangle_polygons(frame: int) -> list[list[list[float]]]:
    x = 100.0 + frame * 5.0
    y = 120.0 + frame * 2.0
    return [rectangle(x, y, 80.0, 60.0)]


def write_sample_sqlite(path: Path, *, frames: int = 8, label: str = "sample") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
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
        conn.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)")
        conn.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO tracks(track_id, label) VALUES ('1', ?)", (label,))
        rows = [
            (
                frame,
                "1",
                json.dumps(moving_rectangle_polygons(frame)),
                "polygon",
                0,
                0,
                0,
                0.0,
                label,
            )
            for frame in range(int(frames))
        ]
        conn.executemany(
            """
            INSERT INTO masks(
                frame, track_id, polygons, shape_type,
                dilate_px, feather_px, mosaic_block, mosaic_alias, label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return path


def write_raw_detector_sqlite(
    path: Path,
    *,
    frames: int = 8,
    label: str = "sample",
    width: int = 64,
    height: int = 48,
    flat_polygons: bool = True,
) -> Path:
    """Write the exact raw_mask_sqlite_v1 shape used by DINOv3."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE frames(
                frame INTEGER PRIMARY KEY,
                time_sec REAL,
                width INTEGER,
                height INTEGER
            );
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                mask_id TEXT NOT NULL,
                detection_index INTEGER NOT NULL,
                label TEXT,
                class_name TEXT,
                category_id INTEGER,
                score REAL,
                detector_score REAL,
                class_score REAL,
                bbox_xyxy TEXT,
                polygons TEXT,
                source_json TEXT,
                PRIMARY KEY(frame, mask_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema', 'raw_mask_sqlite_v1')"
        )
        for frame in range(frames):
            connection.execute(
                "INSERT INTO frames VALUES (?, ?, ?, ?)",
                (frame, frame / 30.0, width, height),
            )
            x = 4.0 + frame
            flat = [x, 8.0, x + 12.0, 8.0, x + 12.0, 20.0, x, 20.0]
            polygons = [flat] if flat_polygons else [
                [[flat[index], flat[index + 1]] for index in range(0, len(flat), 2)]
            ]
            source = {
                "label": label,
                "class_name": label,
                "score": 0.95,
                "bbox_xyxy": [x, 8.0, x + 12.0, 20.0],
                "polygons": polygons,
            }
            connection.execute(
                """
                INSERT INTO masks(
                    frame, mask_id, detection_index, label, class_name,
                    category_id, score, detector_score, class_score,
                    bbox_xyxy, polygons, source_json
                ) VALUES (?, ?, 0, ?, ?, 1, 0.95, 0.95, 0.95, ?, ?, ?)
                """,
                (
                    frame,
                    f"{frame}:0",
                    label,
                    label,
                    json.dumps(source["bbox_xyxy"]),
                    json.dumps(polygons),
                    json.dumps(source, ensure_ascii=False),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def write_unified_inference_sqlite(
    path: Path,
    *,
    frames: int = 8,
    label: str = "sample",
    width: int = 64,
    height: int = 48,
) -> Path:
    """Write a small InstanceSegmentation unified inference schema-v2 DB."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE schema_info(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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
                y REAL NOT NULL,
                PRIMARY KEY(polygon_id, point_index)
            );
            """
        )
        connection.executemany(
            "INSERT INTO schema_info(key, value) VALUES (?, ?)",
            [
                ("schema_name", "instance-segmentation-unified-inference"),
                ("schema_version", "2"),
            ],
        )
        for frame in range(frames):
            frame_id = frame + 1
            connection.execute(
                "INSERT INTO frames VALUES (?, 1, ?, ?, ?, ?)",
                (frame_id, frame, frame / 30.0, width, height),
            )
            if frame == frames - 1:
                continue
            detection_id = frame_id
            x = 4.0 + frame
            connection.execute(
                """
                INSERT INTO detections VALUES(
                    ?, ?, 1, 0, 'detector', 0.75,
                    ?, 8.0, ?, 20.0, NULL, 'detection'
                )
                """,
                (detection_id, frame_id, x, x + 12.0),
            )
            connection.execute(
                "INSERT INTO classifications VALUES (?, 1, ?, 0.95)",
                (detection_id, label),
            )
            connection.execute(
                "INSERT INTO segmentations VALUES (?, 'polygons')",
                (detection_id,),
            )
            connection.execute(
                "INSERT INTO segmentation_polygons VALUES (?, ?, 0)",
                (detection_id, detection_id),
            )
            connection.executemany(
                "INSERT INTO segmentation_points VALUES (?, ?, ?, ?)",
                [
                    (detection_id, 0, x, 8.0),
                    (detection_id, 1, x + 12.0, 8.0),
                    (detection_id, 2, x + 12.0, 20.0),
                    (detection_id, 3, x, 20.0),
                ],
            )
        connection.commit()
    finally:
        connection.close()
    return path
