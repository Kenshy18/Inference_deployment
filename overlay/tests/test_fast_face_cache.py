from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from helpers import create_rich_face_sqlite
from overlay_renderer.fast_face_cache import materialize_fast_face_cache


class FastFaceCacheTests(unittest.TestCase):
    def test_detailed_cache_contains_sparse_rich_face_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "rich.sqlite")
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "UPDATE face_masks SET data=?",
                    (zlib.compress(bytes([255]) * 16),),
                )
            output = root / "cache.sqlite"

            summary = materialize_fast_face_cache(
                source,
                output,
                display_style="detailed",
                start_frame=0,
                end_frame=10,
            )

            self.assertEqual(1, summary["items"])
            self.assertEqual(5, summary["keypoints"])
            self.assertGreater(summary["probability_mask_dots"], 0)
            with sqlite3.connect(output) as connection:
                row = connection.execute(
                    """
                    SELECT frame, detailed, ellipse_cx, ellipse_cy
                    FROM fast_face_items
                    """
                ).fetchone()
                self.assertEqual((1, 1, 30.0, 22.0), row)
                self.assertEqual(
                    "overlay-fast-face-cache",
                    connection.execute(
                        """
                        SELECT value FROM fast_face_cache_info
                        WHERE key='schema_name'
                        """
                    ).fetchone()[0],
                )

    def test_simple_cache_omits_probability_mask_dots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "rich.sqlite")
            output = root / "cache.sqlite"

            summary = materialize_fast_face_cache(
                source,
                output,
                display_style="simple",
                start_frame=0,
                end_frame=None,
            )

            self.assertEqual(0, summary["probability_mask_dots"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    0,
                    connection.execute(
                        "SELECT detailed FROM fast_face_items"
                    ).fetchone()[0],
                )

    def test_fixed_v3_schema_caches_legacy_face_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "legacy-v3.sqlite")
            with sqlite3.connect(source) as connection:
                connection.executescript(
                    """
                    DELETE FROM face_keypoint_state_probabilities;
                    DELETE FROM face_keypoint_class_probabilities;
                    DELETE FROM face_keypoints;
                    DELETE FROM face_masks;
                    DELETE FROM face_observations;
                    CREATE TABLE result_capabilities(
                        name TEXT PRIMARY KEY,
                        available INTEGER NOT NULL
                    );
                    INSERT INTO result_capabilities
                    VALUES ('rich_face_geometry', 0);
                    """
                )
            output = root / "cache.sqlite"

            summary = materialize_fast_face_cache(
                source,
                output,
                display_style="detailed",
                start_frame=0,
                end_frame=10,
            )

            self.assertEqual(2, summary["items"])
            self.assertEqual(0, summary["keypoints"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    2,
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM fast_face_items
                        WHERE box_x1 IS NOT NULL AND box_y1 IS NOT NULL
                          AND box_x2 IS NOT NULL AND box_y2 IS NOT NULL
                        """
                    ).fetchone()[0],
                )

    def test_cache_filters_before_materializing_face_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_rich_face_sqlite(root / "rich.sqlite")
            with sqlite3.connect(source) as connection:
                connection.execute("UPDATE detections SET score=0.40 WHERE id=21")
            output = root / "cache.sqlite"

            summary = materialize_fast_face_cache(
                source,
                output,
                display_style="detailed",
                start_frame=0,
                end_frame=10,
                head_detection_score_threshold=0.55,
            )

            self.assertEqual(0, summary["items"])
            self.assertEqual(0.55, summary["head_detection_score_threshold"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    "0.55",
                    connection.execute(
                        """
                        SELECT value FROM fast_face_cache_info
                        WHERE key='head_detection_score_threshold'
                        """
                    ).fetchone()[0],
                )


if __name__ == "__main__":
    unittest.main()
