from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from contracts import (
    BoundingBox,
    Classification,
    Detection,
    DetectionFrame,
    FrameReference,
    ModelDescriptor,
    Segmentation,
    SegmentationFrame,
    SegmentationInstance,
    TaskType,
)
from persistence import AsyncSqliteWriter, SqliteWriter


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
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentations"
                    ).fetchone()[0],
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
                    connection.execute(
                        "SELECT COUNT(*) FROM detections"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM segmentations"
                    ).fetchone()[0],
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
                metadata = dict(
                    connection.execute("SELECT key, value FROM metadata")
                )
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


if __name__ == "__main__":
    unittest.main()
