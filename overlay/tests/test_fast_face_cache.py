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


if __name__ == "__main__":
    unittest.main()
