from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from artifacts.unified_sqlite import build_integrated_result
from tests.helpers import write_sample_sqlite, write_unified_inference_sqlite


class EditableGeometryTests(unittest.TestCase):
    def test_native_ellipse_keyframes_are_not_recovered_from_polygons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            with sqlite3.connect(final) as connection:
                connection.execute("UPDATE masks SET shape_type='ellipse'")
            keyframes = root / "final_keyframes.json"
            keyframes.write_text(
                json.dumps(
                    [
                        {
                            "track_id": "1",
                            "mode": "K1",
                            "run_id": 0,
                            "slot_id": 0,
                            "frame": 0,
                            "ellipse": [120.0, 130.0, 40.0, 20.0, -30.0],
                        },
                        {
                            "track_id": "1",
                            "mode": "K1",
                            "run_id": 0,
                            "slot_id": 0,
                            "frame": 4,
                            "ellipse": [140.0, 138.0, 44.0, 22.0, 20.0],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "result.sqlite"

            summary = build_integrated_result(
                inference,
                tracked,
                final,
                output,
                ellipse_keyframes_json=keyframes,
            )

            self.assertEqual(2, summary["mask_keyframes"])
            with sqlite3.connect(output) as connection:
                rows = connection.execute(
                    """
                    SELECT k.frame, c.slot_index, e.cx, e.cy,
                           e.radius_x, e.radius_y, e.theta_radians
                    FROM mask_keyframes k
                    JOIN keyframe_components c ON c.keyframe_id=k.id
                    JOIN keyframe_ellipses e ON e.component_id=c.id
                    ORDER BY k.frame
                    """
                ).fetchall()
                self.assertEqual(2, len(rows))
                self.assertAlmostEqual(math.radians(-30.0), rows[0][6])
                self.assertAlmostEqual(math.radians(20.0), rows[1][6])
                self.assertEqual(
                    ("ellipse", "ellipse_log_axes_short_angle_v1"),
                    connection.execute(
                        """
                        SELECT shape_type, interpolation_method
                        FROM mask_track_segments
                        """
                    ).fetchone(),
                )

    def test_polygon_keyframes_keep_only_selected_native_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            keys = write_sample_sqlite(root / "keys.sqlite", frames=5)
            with sqlite3.connect(keys) as connection:
                connection.execute("DELETE FROM masks WHERE frame NOT IN (0, 4)")
            output = root / "result.sqlite"

            summary = build_integrated_result(
                inference,
                tracked,
                final,
                output,
                polygon_keyframes_sqlite=keys,
            )

            self.assertEqual(2, summary["mask_keyframes"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [(0,), (4,)],
                    connection.execute(
                        "SELECT frame FROM mask_keyframes ORDER BY frame"
                    ).fetchall(),
                )
                self.assertEqual(
                    8,
                    connection.execute(
                        "SELECT COUNT(*) FROM keyframe_polygon_points"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    "keyframe-primary-v3",
                    connection.execute(
                        """
                        SELECT value FROM result_schema_info
                        WHERE key='compatibility_profile'
                        """
                    ).fetchone()[0],
                )


if __name__ == "__main__":
    unittest.main()
