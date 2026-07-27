from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from contracts import (
    BoundingBox,
    Classification,
    Detection,
    DetectionFrame,
    FaceEllipse,
    FaceKeypoint,
    FaceMask,
    FaceObservation,
    FrameReference,
    ModelDescriptor,
    Segmentation,
    SegmentationFrame,
    SegmentationInstance,
    TaskType,
)
from persistence import AsyncSqliteWriter, SqliteWriter, UnifiedSqliteWriter


def segmentation_result() -> SegmentationFrame:
    model = ModelDescriptor(
        model_id="test_segmenter",
        task=TaskType.INSTANCE_SEGMENTATION,
        implementation="tests.FakeSegmenter",
    )
    detection = Detection(
        class_id=2,
        class_name="object",
        score=0.9,
        bbox=BoundingBox(1, 2, 9, 10),
        classification=Classification(
            class_id=2,
            class_name="object",
            score=0.8,
            probabilities=(0.1, 0.2, 0.7),
        ),
    )
    return SegmentationFrame(
        model=model,
        frame=FrameReference(index=0, timestamp_sec=0.0, width=16, height=12),
        instances=(
            SegmentationInstance(
                detection=detection,
                segmentation=Segmentation(polygons=((1, 2, 9, 2, 9, 10),)),
            ),
        ),
    )


def face_result() -> DetectionFrame:
    model = ModelDescriptor(
        model_id="face_dino_v2",
        task=TaskType.OBJECT_DETECTION,
        implementation="tests.FakeFaceDinoV2",
    )
    keypoints = tuple(
        FaceKeypoint(
            point_index=index,
            class_id=(1, 1, 2, 3, 3)[index],
            x=4.0 + index,
            y=5.0 + index,
            state=(2, 1, 2, 2, 0)[index],
            confidence=0.8,
            valid=index < 4,
            class_probabilities=(0.05, 0.8, 0.1, 0.05),
            state_probabilities=(0.2, 0.8),
        )
        for index in range(5)
    )
    observation = FaceObservation(
        score=0.85,
        present=True,
        ellipse=FaceEllipse(8.0, 6.0, 4.0, 2.0, 0.25),
        keypoints=keypoints,
        mask=FaceMask(
            width=4,
            height=4,
            box_x1=4.0,
            box_y1=2.0,
            box_x2=12.0,
            box_y2=10.0,
            data=bytes(range(16)),
        ),
    )
    return DetectionFrame(
        model=model,
        frame=FrameReference(index=0, timestamp_sec=0, width=16, height=12),
        detections=(
            Detection(
                class_id=1,
                class_name="Head",
                score=0.92,
                bbox=BoundingBox(2, 1, 14, 11),
                source="head_detection",
                group_id=0,
                face_observation=observation,
            ),
            Detection(
                class_id=2,
                class_name="Face",
                score=0.85,
                bbox=BoundingBox(4, 4, 12, 8),
                source="ellipse_detection",
                group_id=0,
            ),
        ),
    )


class PersistenceTest(unittest.TestCase):
    def test_async_sqlite_preserves_contract_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "async.sqlite"
            writer = AsyncSqliteWriter(output, queue_size=1)
            writer.set_metadata({"backend": "async-test"})
            writer.write(segmentation_result())
            writer.close()

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM segmentations").fetchone()[
                        0
                    ],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='backend'"
                    ).fetchone()[0],
                    "async-test",
                )

    def test_sqlite_normalizes_detection_and_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.sqlite"
            writer = SqliteWriter(output)
            writer.set_metadata(
                {
                    "source": "unit-test",
                    "video": {"fps": 30.0, "shape": [12, 16]},
                }
            )
            writer.write(segmentation_result())
            writer.close()

            connection = sqlite3.connect(output)
            try:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM segmentations").fetchone()[
                        0
                    ],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentation_polygons"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentation_points"
                    ).fetchone()[0],
                    3,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM classification_probabilities"
                    ).fetchone()[0],
                    3,
                )
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                self.assertEqual(metadata["video.fps"], "30.0")
                self.assertEqual(metadata["video.shape.0"], "12")
                task = connection.execute("SELECT task FROM frames").fetchone()[0]
                self.assertEqual(task, "instance_segmentation")
            finally:
                connection.close()

    def test_sqlite_accepts_object_detection(self) -> None:
        model = ModelDescriptor(
            model_id="test_detector",
            task=TaskType.OBJECT_DETECTION,
            implementation="tests.FakeDetector",
        )
        result = DetectionFrame(
            model=model,
            frame=FrameReference(index=0, timestamp_sec=0, width=16, height=12),
            detections=(
                Detection(
                    class_id=0,
                    class_name="face",
                    score=0.75,
                    bbox=BoundingBox(2, 3, 8, 9),
                    track_id=4,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "detection.sqlite"
            writer = SqliteWriter(output)
            writer.write(result)
            writer.close()
            connection = sqlite3.connect(output)
            try:
                row = connection.execute(
                    "SELECT class_name, track_id FROM detections"
                ).fetchone()
                self.assertEqual(row, ("face", 4))
            finally:
                connection.close()

    def test_existing_database_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.sqlite"
            writer = SqliteWriter(output)
            writer.close()
            with self.assertRaises(FileExistsError):
                SqliteWriter(output)
            replacement = SqliteWriter(output, overwrite=True)
            replacement.close()

    def test_rich_face_geometry_survives_unified_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "face.sqlite"
            writer = SqliteWriter(candidate)
            writer.set_metadata(
                {
                    "model_id": "face_dino_v2",
                    "task": "object_detection",
                    "video": {
                        "frames": 1,
                        "fps": 30.0,
                        "width": 16,
                        "height": 12,
                    },
                }
            )
            writer.write(face_result())
            writer.close()

            unified = root / "unified.sqlite"
            merged = UnifiedSqliteWriter(
                unified,
                input_path=root / "input.mp4",
                mode="face",
            )
            summary = merged.import_model_output(
                candidate,
                role="face_detection",
                model_id="face_dino_v2",
                backend="tensorrt-fast",
            )
            merged.close()

            self.assertEqual(1, summary.face_observations)
            self.assertEqual(5, summary.face_keypoints)
            with sqlite3.connect(unified) as connection:
                self.assertEqual(
                    "3",
                    connection.execute(
                        "SELECT value FROM schema_info " "WHERE key='schema_version'"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    ("Head", "Face"),
                    tuple(
                        row[0]
                        for row in connection.execute(
                            "SELECT class_name FROM detections ORDER BY id"
                        )
                    ),
                )
                self.assertEqual(
                    (1, 2, "ellipse", 8.0, 6.0, 4.0, 2.0, 0.25),
                    connection.execute(
                        """
                        SELECT head_detection_id, face_detection_id,
                               geometry_type, ellipse_cx, ellipse_cy,
                               ellipse_major_radius, ellipse_minor_radius,
                               ellipse_theta_radians
                        FROM face_observations
                        """
                    ).fetchone(),
                )
                self.assertEqual(
                    5,
                    connection.execute(
                        "SELECT COUNT(*) FROM face_keypoints"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    20,
                    connection.execute(
                        "SELECT COUNT(*) " "FROM face_keypoint_class_probabilities"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    10,
                    connection.execute(
                        "SELECT COUNT(*) " "FROM face_keypoint_state_probabilities"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    ("zlib-u8-probability-v1", 4, 4, 16),
                    connection.execute(
                        """
                        SELECT encoding, width, height, LENGTH(data)
                        FROM face_masks
                        """
                    ).fetchone()[:3]
                    + (
                        len(
                            zlib.decompress(
                                connection.execute(
                                    "SELECT data FROM face_masks"
                                ).fetchone()[0]
                            )
                        ),
                    ),
                )
                self.assertIsNone(
                    connection.execute(
                        """
                        SELECT 1 FROM sqlite_master
                        WHERE type='table'
                          AND name='imported_detection_ids_staging'
                        """
                    ).fetchone()
                )


if __name__ == "__main__":
    unittest.main()
