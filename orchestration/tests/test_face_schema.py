from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from orchestration.contracts import ArtifactError, validate_inference_sqlite

from helpers import create_rich_face_unified_sqlite


class FaceSchemaTests(unittest.TestCase):
    def test_face_dino_v2_requires_and_validates_rich_schema_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sqlite = create_rich_face_unified_sqlite(Path(temporary) / "rich.sqlite")

            stats = validate_inference_sqlite(
                sqlite,
                require_segmentation=True,
                require_faces=True,
                expected_face_model="face_dino_v2",
            )

            self.assertEqual(3, stats["schema_version"])
            self.assertEqual(1, stats["face_observations"])
            self.assertEqual(5, stats["face_keypoints"])

    def test_face_dino_v2_detection_cannot_silently_lose_rich_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sqlite = create_rich_face_unified_sqlite(Path(temporary) / "rich.sqlite")

            with sqlite3.connect(sqlite) as connection:
                connection.execute("DELETE FROM face_observations")

            with self.assertRaisesRegex(ArtifactError, "no rich face observation"):
                validate_inference_sqlite(
                    sqlite,
                    require_segmentation=True,
                    require_faces=True,
                    expected_face_model="face_dino_v2",
                )


if __name__ == "__main__":
    unittest.main()
