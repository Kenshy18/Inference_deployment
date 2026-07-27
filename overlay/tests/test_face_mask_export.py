from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from overlay_renderer.face_mask_export import export_face_masks

from helpers import create_rich_face_sqlite, create_unified_sqlite


class FaceMaskExportTests(unittest.TestCase):
    def test_eye_masks_are_written_to_a_separate_compatible_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "inference.sqlite")
            output = root / "eyes.sqlite"
            before = source.read_bytes()

            summary = export_face_masks(
                source,
                output,
                target="eyes",
                eye_shape="rectangle",
            )

            self.assertEqual(before, source.read_bytes())
            self.assertEqual(1, summary["rows"])
            with sqlite3.connect(output) as connection:
                info = dict(connection.execute("SELECT key, value FROM schema_info"))
                row = connection.execute(
                    """
                    SELECT frame, track_id, polygons, shape_type, label,
                           source_observation_id, derivation, confidence
                    FROM masks
                    """
                ).fetchone()
                self.assertEqual("face-privacy-mask-sqlite", info["schema_name"])
                self.assertEqual("1", info["schema_version"])
                assert row is not None
                self.assertEqual(1, row[0])
                self.assertEqual("face-observation:1", row[1])
                self.assertEqual("rectangle", row[3])
                self.assertEqual("Eyes", row[4])
                self.assertEqual(1, row[5])
                self.assertIn(
                    row[6],
                    {"eye-keypoints", "ellipse-fallback"},
                )
                if row[6] == "eye-keypoints":
                    self.assertGreater(row[7], 0.0)
                else:
                    self.assertEqual(0.0, row[7])
                polygon = json.loads(row[2])
                self.assertEqual(1, len(polygon))
                self.assertEqual(4, len(polygon[0]))
                self.assertEqual(
                    "ok",
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                )

    def test_schema_v2_is_rejected_for_privacy_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_unified_sqlite(root / "legacy.sqlite")
            with self.assertRaisesRegex(ValueError, "schema-v3"):
                export_face_masks(
                    source,
                    root / "eyes.sqlite",
                    target="eyes",
                )

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "inference.sqlite")
            output = root / "face.sqlite"
            export_face_masks(source, output, target="face")
            with self.assertRaises(FileExistsError):
                export_face_masks(source, output, target="face")
            export_face_masks(
                source,
                output,
                target="face",
                overwrite=True,
            )


if __name__ == "__main__":
    unittest.main()
