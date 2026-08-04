from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from artifacts.unified_sqlite import build_integrated_result
from tests.helpers import write_sample_sqlite, write_unified_inference_sqlite


def _write_split_ellipse_keyframes(
    path: Path,
    *,
    right_components: int = 1,
) -> Path:
    rows = [
        {
            "track_id": "1",
            "mode": "K1",
            "run_id": 0,
            "slot_id": 0,
            "frame": 0,
            "ellipse": [120.0, 130.0, 40.0, 20.0, -30.0],
        }
    ]
    for slot_id in range(right_components):
        rows.append(
            {
                "track_id": "1",
                "mode": "K1",
                "run_id": 1,
                "slot_id": slot_id,
                "frame": 4,
                "ellipse": [
                    140.0 + slot_id * 10.0,
                    138.0,
                    44.0,
                    22.0,
                    20.0,
                ],
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _add_postprocess_provenance(
    path: Path,
    *,
    included_frames: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE mask_postprocess_provenance(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                label TEXT NOT NULL,
                policy_source TEXT NOT NULL,
                shape_mode TEXT NOT NULL,
                keyframe_interval INTEGER NOT NULL,
                max_gap INTEGER NOT NULL,
                is_gap_filled INTEGER NOT NULL,
                PRIMARY KEY(frame, track_id)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO mask_postprocess_provenance(
                frame, track_id, label, policy_source, shape_mode,
                keyframe_interval, max_gap, is_gap_filled
            ) VALUES (?, '1', 'sample', 'test', 'ellipse', 3, 3, ?)
            """,
            ((frame, int(frame in (1, 2, 3))) for frame in included_frames),
        )


class EditableGeometryTests(unittest.TestCase):
    def test_explicit_ellipse_gapfill_connects_editable_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            _add_postprocess_provenance(final)
            keys = _write_split_ellipse_keyframes(root / "keyframes.json")
            output = root / "result.sqlite"

            summary = build_integrated_result(
                inference,
                tracked,
                final,
                output,
                ellipse_keyframes_json=keys,
            )

            self.assertEqual(1, summary["mask_segments"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [(0, 4, "ellipse_gapfill_connected_runs")],
                    connection.execute(
                        """
                        SELECT start_frame, end_frame, segment_reason
                        FROM mask_track_segments
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    [(0, 0), (4, 1)],
                    connection.execute(
                        """
                        SELECT frame, keyframe_index
                        FROM mask_keyframes ORDER BY frame
                        """
                    ).fetchall(),
                )

    def test_partial_gapfill_provenance_keeps_runs_disconnected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            _add_postprocess_provenance(final, included_frames=(0, 4))
            keys = _write_split_ellipse_keyframes(root / "keyframes.json")
            output = root / "result.sqlite"

            summary = build_integrated_result(
                inference,
                tracked,
                final,
                output,
                ellipse_keyframes_json=keys,
            )

            self.assertEqual(2, summary["mask_segments"])
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [(0, 0), (4, 4)],
                    connection.execute(
                        """
                        SELECT start_frame, end_frame
                        FROM mask_track_segments ORDER BY start_frame
                        """
                    ).fetchall(),
                )

    def test_explicit_gapfill_cannot_cross_a_cut(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            _add_postprocess_provenance(final)
            with sqlite3.connect(final) as connection:
                connection.execute("INSERT INTO cuts(frame) VALUES (2)")
            keys = _write_split_ellipse_keyframes(root / "keyframes.json")

            with self.assertRaisesRegex(
                ValueError,
                "cannot be reconstructed from editable keyframes",
            ):
                build_integrated_result(
                    inference,
                    tracked,
                    final,
                    root / "result.sqlite",
                    ellipse_keyframes_json=keys,
                )

    def test_explicit_gapfill_rejects_component_structure_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_unified_inference_sqlite(
                root / "inference.sqlite", frames=5
            )
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=5)
            final = write_sample_sqlite(root / "final.sqlite", frames=5)
            _add_postprocess_provenance(final)
            keys = _write_split_ellipse_keyframes(
                root / "keyframes.json",
                right_components=2,
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot be reconstructed from editable keyframes",
            ):
                build_integrated_result(
                    inference,
                    tracked,
                    final,
                    root / "result.sqlite",
                    ellipse_keyframes_json=keys,
                )

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
