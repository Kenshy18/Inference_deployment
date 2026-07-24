"""Normalized SQLite persistence for detection and segmentation contracts."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

from contracts import DetectionFrame, InferenceFrame, SegmentationFrame

from .metadata import flatten_metadata


class SqliteWriter:
    """Write one video inference run to a new SQLite database."""

    def __init__(
        self,
        path: Path,
        *,
        overwrite: bool = False,
        safe: bool = True,
        commit_interval: int = 256,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(f"output already exists: {self.path}")
            self.path.unlink()
        self.connection = sqlite3.connect(str(self.path))
        if safe:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        else:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.commit_interval = max(1, int(commit_interval))
        self.pending = 0
        self.closed = False
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                value_type TEXT NOT NULL
            );
            CREATE TABLE frames (
                frame_index INTEGER PRIMARY KEY,
                timestamp_sec REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                task TEXT NOT NULL
            );
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_index INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                classifier_class_id INTEGER,
                classifier_class_name TEXT,
                classifier_score REAL,
                track_id INTEGER,
                source TEXT NOT NULL,
                FOREIGN KEY(frame_index) REFERENCES frames(frame_index)
            );
            CREATE TABLE classification_probabilities (
                detection_id INTEGER NOT NULL,
                class_index INTEGER NOT NULL,
                probability REAL NOT NULL,
                PRIMARY KEY(detection_id, class_index),
                FOREIGN KEY(detection_id) REFERENCES detections(id)
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
        )

    def set_metadata(self, values: Mapping[str, object]) -> None:
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO metadata(key, value, value_type)
            VALUES (?, ?, ?)
            """,
            flatten_metadata(values),
        )

    def _insert_detection(self, frame_index: int, detection) -> int:
        classification = detection.classification
        cursor = self.connection.execute(
            """
            INSERT INTO detections(
                frame_index, class_id, class_name, score,
                x1, y1, x2, y2,
                classifier_class_id, classifier_class_name, classifier_score,
                track_id, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_index,
                detection.class_id,
                detection.class_name,
                detection.score,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2,
                None if classification is None else classification.class_id,
                None if classification is None else classification.class_name,
                None if classification is None else classification.score,
                detection.track_id,
                detection.source,
            ),
        )
        detection_id = int(cursor.lastrowid)
        if classification is not None and classification.probabilities is not None:
            self.connection.executemany(
                """
                INSERT INTO classification_probabilities(
                    detection_id, class_index, probability
                ) VALUES (?, ?, ?)
                """,
                [
                    (detection_id, index, probability)
                    for index, probability in enumerate(
                        classification.probabilities
                    )
                ],
            )
        return detection_id

    def _insert_segmentation(self, detection_id: int, segmentation) -> None:
        self.connection.execute(
            """
            INSERT INTO segmentations(detection_id, encoding)
            VALUES (?, ?)
            """,
            (detection_id, "polygons"),
        )
        for polygon_index, polygon in enumerate(segmentation.polygons):
            cursor = self.connection.execute(
                """
                INSERT INTO segmentation_polygons(
                    detection_id, polygon_index
                ) VALUES (?, ?)
                """,
                (detection_id, polygon_index),
            )
            polygon_id = int(cursor.lastrowid)
            self.connection.executemany(
                """
                INSERT INTO segmentation_points(
                    polygon_id, point_index, x, y
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (polygon_id, point_index, polygon[offset], polygon[offset + 1])
                    for point_index, offset in enumerate(
                        range(0, len(polygon), 2)
                    )
                ],
            )

    def write(self, result: InferenceFrame) -> None:
        if self.closed:
            raise RuntimeError("writer is closed")
        self.connection.execute(
            """
            INSERT INTO frames(
                frame_index, timestamp_sec, width, height, model_id, task
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.frame.index,
                result.frame.timestamp_sec,
                result.frame.width,
                result.frame.height,
                result.model.model_id,
                result.model.task.value,
            ),
        )
        if isinstance(result, DetectionFrame):
            for detection in result.detections:
                self._insert_detection(result.frame.index, detection)
        elif isinstance(result, SegmentationFrame):
            for instance in result.instances:
                detection_id = self._insert_detection(
                    result.frame.index, instance.detection
                )
                self._insert_segmentation(detection_id, instance.segmentation)
        else:
            raise TypeError(f"unsupported inference result: {type(result)!r}")
        self.pending += 1
        if self.pending >= self.commit_interval:
            self.connection.commit()
            self.pending = 0

    def close(self) -> None:
        if self.closed:
            return
        self.connection.executescript(
            """
            CREATE INDEX idx_detections_frame
                ON detections(frame_index);
            CREATE INDEX idx_detections_class
                ON detections(class_id);
            CREATE INDEX idx_segmentation_polygons_detection
                ON segmentation_polygons(detection_id);
            """
        )
        self.connection.commit()
        self.connection.close()
        self.closed = True


__all__ = ["SqliteWriter"]
