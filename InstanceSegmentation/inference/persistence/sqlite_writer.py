"""Normalized SQLite persistence for detection and segmentation contracts."""

from __future__ import annotations

import sqlite3
import zlib
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
                group_id INTEGER,
                FOREIGN KEY(frame_index) REFERENCES frames(frame_index),
                CHECK(group_id IS NULL OR group_id >= 0)
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
                track_id, source, group_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                detection.group_id,
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
                    for index, probability in enumerate(classification.probabilities)
                ],
            )
        return detection_id

    def _insert_face_observation(
        self,
        *,
        anchor_detection_id: int,
        head_detection_id: int | None,
        face_detection_id: int | None,
        observation,
    ) -> None:
        ellipse = observation.ellipse
        cursor = self.connection.execute(
            """
            INSERT INTO face_observations(
                anchor_detection_id, head_detection_id, face_detection_id,
                face_score, face_present, geometry_type,
                ellipse_cx, ellipse_cy,
                ellipse_major_radius, ellipse_minor_radius,
                ellipse_theta_radians
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anchor_detection_id,
                head_detection_id,
                face_detection_id,
                observation.score,
                int(observation.present),
                None if ellipse is None else "ellipse",
                None if ellipse is None else ellipse.cx,
                None if ellipse is None else ellipse.cy,
                None if ellipse is None else ellipse.major_radius,
                None if ellipse is None else ellipse.minor_radius,
                None if ellipse is None else ellipse.theta_radians,
            ),
        )
        observation_id = int(cursor.lastrowid)
        if observation.mask is not None:
            mask = observation.mask
            self.connection.execute(
                """
                INSERT INTO face_masks(
                    observation_id, encoding, width, height,
                    box_x1, box_y1, box_x2, box_y2, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    "zlib-u8-probability-v1",
                    mask.width,
                    mask.height,
                    mask.box_x1,
                    mask.box_y1,
                    mask.box_x2,
                    mask.box_y2,
                    zlib.compress(mask.data, level=1),
                ),
            )
        for point in observation.keypoints:
            self.connection.execute(
                """
                INSERT INTO face_keypoints(
                    observation_id, point_index,
                    class_id, class_name, x, y,
                    state, state_name, confidence, valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    point.point_index,
                    point.class_id,
                    point.class_name,
                    point.x,
                    point.y,
                    point.state,
                    point.state_name,
                    point.confidence,
                    int(point.valid),
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO face_keypoint_class_probabilities(
                    observation_id, point_index, class_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        observation_id,
                        point.point_index,
                        index,
                        probability,
                    )
                    for index, probability in enumerate(point.class_probabilities)
                ],
            )
            self.connection.executemany(
                """
                INSERT INTO face_keypoint_state_probabilities(
                    observation_id, point_index, state_index, probability
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        observation_id,
                        point.point_index,
                        index + 1,
                        probability,
                    )
                    for index, probability in enumerate(point.state_probabilities)
                ],
            )

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
                    for point_index, offset in enumerate(range(0, len(polygon), 2))
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
            inserted: list[tuple[object, int]] = []
            grouped: dict[int, dict[str, int]] = {}
            for detection in result.detections:
                detection_id = self._insert_detection(result.frame.index, detection)
                inserted.append((detection, detection_id))
                if detection.group_id is not None:
                    identities = grouped.setdefault(detection.group_id, {})
                    key = detection.class_name.lower()
                    if key in identities:
                        raise ValueError(
                            "duplicate detection class within face group "
                            f"{detection.group_id}: {detection.class_name}"
                        )
                    identities[key] = detection_id
            for detection, detection_id in inserted:
                if detection.face_observation is None:
                    continue
                assert detection.group_id is not None
                identities = grouped[detection.group_id]
                self._insert_face_observation(
                    anchor_detection_id=detection_id,
                    head_detection_id=identities.get("head"),
                    face_detection_id=identities.get("face"),
                    observation=detection.face_observation,
                )
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
