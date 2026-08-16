from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from overlay_renderer.keyframe_cache import (
    is_keyframe_primary,
    materialize_overlay_cache,
    materialize_overlay_cache_shards,
)


class KeyframeCacheTests(unittest.TestCase):
    def _write_source(self, path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE result_schema_info(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT INTO result_schema_info VALUES
                    ('compatibility_profile', 'keyframe-primary-v3');
                CREATE TABLE tracks(
                    track_id TEXT PRIMARY KEY,
                    label TEXT,
                    domain TEXT NOT NULL
                );
                INSERT INTO tracks VALUES
                    ('1', 'target', 'genital'),
                    ('face:1', 'Eyes', 'face_privacy');
                CREATE TABLE mask_track_segments(
                    id INTEGER PRIMARY KEY,
                    track_id TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    interpolation_method TEXT NOT NULL
                );
                INSERT INTO mask_track_segments
                    VALUES
                        (1, '1', 0, 2, 'none'),
                        (2, 'face:1', 1, 1, 'none');
                CREATE TABLE mask_keyframes(
                    id INTEGER PRIMARY KEY,
                    segment_id INTEGER NOT NULL,
                    frame INTEGER NOT NULL
                );
                INSERT INTO mask_keyframes VALUES
                    (1, 1, 0), (2, 1, 2), (3, 2, 1);
                CREATE TABLE keyframe_components(
                    id INTEGER PRIMARY KEY,
                    keyframe_id INTEGER NOT NULL,
                    slot_index INTEGER NOT NULL,
                    geometry_type TEXT NOT NULL
                );
                INSERT INTO keyframe_components
                    VALUES
                        (1, 1, 0, 'polygon'),
                        (2, 2, 0, 'polygon'),
                        (3, 3, 0, 'ellipse');
                CREATE TABLE keyframe_ellipses(
                    component_id INTEGER PRIMARY KEY,
                    cx REAL, cy REAL, radius_x REAL, radius_y REAL,
                    theta_radians REAL
                );
                INSERT INTO keyframe_ellipses
                    VALUES (3, 5, 4, 2, 1, 0.25);
                CREATE TABLE keyframe_rectangles(
                    component_id INTEGER PRIMARY KEY,
                    cx REAL, cy REAL, half_width REAL, half_height REAL,
                    theta_radians REAL
                );
                CREATE TABLE keyframe_polygon_rings(
                    id INTEGER PRIMARY KEY,
                    component_id INTEGER NOT NULL,
                    ring_index INTEGER NOT NULL,
                    ring_role TEXT NOT NULL
                );
                INSERT INTO keyframe_polygon_rings
                    VALUES (1, 1, 0, 'exterior'), (2, 2, 0, 'exterior');
                CREATE TABLE keyframe_polygon_points(
                    ring_id INTEGER NOT NULL,
                    point_index INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL
                );
                INSERT INTO keyframe_polygon_points VALUES
                    (1, 0, 0, 0), (1, 1, 2, 0),
                    (1, 2, 2, 2), (1, 3, 0, 2),
                    (2, 0, 2, 0), (2, 1, 4, 0),
                    (2, 2, 4, 2), (2, 3, 2, 2);
                CREATE TABLE tracking_assignments(
                    source_detection_id INTEGER PRIMARY KEY,
                    frame INTEGER NOT NULL,
                    final_track_id TEXT,
                    final_label TEXT,
                    removed_by_short_track INTEGER NOT NULL
                );
                INSERT INTO tracking_assignments
                    VALUES (6, 0, '1', 'target', 0),
                           (7, 1, '1', 'target', 0),
                           (8, 2, '1', 'target', 0);
                CREATE TABLE segmentation_polygons(
                    id INTEGER PRIMARY KEY,
                    detection_id INTEGER NOT NULL,
                    polygon_index INTEGER NOT NULL
                );
                INSERT INTO segmentation_polygons
                    VALUES (2, 6, 0), (3, 7, 0), (4, 8, 0);
                CREATE TABLE segmentation_points(
                    polygon_id INTEGER NOT NULL,
                    point_index INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL
                );
                INSERT INTO segmentation_points VALUES
                    (2, 0, 0, 0), (2, 1, 2, 0),
                    (2, 2, 2, 2), (2, 3, 0, 2),
                    (3, 0, 1, 0), (3, 1, 3, 0),
                    (3, 2, 3, 2), (3, 3, 1, 2),
                    (4, 0, 2, 0), (4, 1, 4, 0),
                    (4, 2, 4, 2), (4, 3, 2, 2);
                """
            )

    def test_materializes_final_interpolation_and_tracked_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.sqlite"
            self._write_source(source)
            self.assertTrue(is_keyframe_primary(source))

            final = root / "final-cache.sqlite"
            final_summary = materialize_overlay_cache(source, final, mode="final")
            self.assertEqual(4, final_summary["rows"])
            with sqlite3.connect(final) as connection:
                middle = json.loads(
                    connection.execute(
                        """
                        SELECT polygons FROM masks
                        WHERE frame=1 AND track_id='1'
                        """
                    ).fetchone()[0]
                )
                self.assertEqual(
                    64,
                    len(
                        json.loads(
                            connection.execute(
                                """
                                SELECT polygons FROM masks
                                WHERE frame=1 AND track_id='face:1'
                                """
                            ).fetchone()[0]
                        )[0]
                    ),
                )
                self.assertEqual(
                    0,
                    connection.execute("SELECT COUNT(*) FROM mask_ellipses").fetchone()[
                        0
                    ],
                )
                self.assertEqual(
                    [(0, 1), (1, 0), (2, 1)],
                    connection.execute(
                        """
                        SELECT frame, is_keyframe FROM masks
                        WHERE track_id='1' ORDER BY frame
                        """
                    ).fetchall(),
                )
            self.assertEqual(8, len(middle[0]))
            self.assertEqual([1.0, 0.0], middle[0][0])
            self.assertEqual([3.0, 0.0], middle[0][2])
            self.assertEqual([3.0, 2.0], middle[0][4])
            self.assertEqual([1.0, 2.0], middle[0][6])

            genital = root / "genital-cache.sqlite"
            genital_summary = materialize_overlay_cache(
                source,
                genital,
                mode="final",
                mask_domain="genital",
            )
            self.assertEqual(3, genital_summary["rows"])
            self.assertEqual("genital", genital_summary["mask_domain"])
            with sqlite3.connect(genital) as connection:
                self.assertEqual(
                    [("1",)],
                    connection.execute(
                        "SELECT DISTINCT track_id FROM masks"
                    ).fetchall(),
                )
                self.assertEqual(
                    0,
                    connection.execute("SELECT COUNT(*) FROM mask_ellipses").fetchone()[
                        0
                    ],
                )

            tracked = root / "tracked-cache.sqlite"
            tracked_summary = materialize_overlay_cache(source, tracked, mode="tracked")
            self.assertEqual(3, tracked_summary["rows"])
            with sqlite3.connect(tracked) as connection:
                self.assertEqual(
                    (0, "1", "target"),
                    connection.execute(
                        """
                        SELECT frame, track_id, label FROM masks
                        ORDER BY frame LIMIT 1
                        """
                    ).fetchone(),
                )

            shards = materialize_overlay_cache_shards(
                source,
                root / "shards",
                mode="final",
                frame_ranges=[(0, 0), (1, 2)],
                workers=2,
            )
            self.assertEqual("parallel-frame-range-shards", shards["strategy"])
            self.assertEqual(4, shards["rows"])
            self.assertEqual(2, len(shards["shards"]))
            self.assertEqual(
                [1, 3],
                [int(shard["rows"]) for shard in shards["shards"]],
            )
            with sqlite3.connect(
                Path(str(shards["shards"][1]["cache_sqlite"]))
            ) as connection:
                self.assertEqual(
                    (1, "face:1", 64, "Eyes", 1),
                    connection.execute(
                        """
                        SELECT frame, track_id, point_count, label, is_keyframe
                        FROM mask_ellipses
                        """
                    ).fetchone(),
                )

    def test_final_interpolation_never_crosses_a_cut(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.sqlite"
            self._write_source(source)
            with sqlite3.connect(source) as connection:
                connection.executescript(
                    "CREATE TABLE cuts(frame INTEGER PRIMARY KEY);"
                    "INSERT INTO cuts VALUES (1);"
                )

            cache = root / "final-cache.sqlite"
            summary = materialize_overlay_cache(
                source,
                cache,
                mode="final",
                mask_domain="genital",
            )
            self.assertEqual(2, summary["rows"])
            with sqlite3.connect(cache) as connection:
                self.assertEqual(
                    [(0, "1", 1), (2, "1", 1)],
                    connection.execute(
                        "SELECT frame, track_id, is_keyframe "
                        "FROM masks ORDER BY frame"
                    ).fetchall(),
                )
                self.assertEqual(
                    [(1,)],
                    connection.execute("SELECT frame FROM cuts").fetchall(),
                )

    def test_index_interpolation_keeps_keyframe_without_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.sqlite"
            self._write_source(source)
            with sqlite3.connect(source) as connection:
                connection.executescript(
                    """
                    UPDATE mask_track_segments
                    SET interpolation_method='linear_polygon_index_v1'
                    WHERE id=1;
                    UPDATE tracking_assignments
                    SET removed_by_short_track=1
                    WHERE frame=1 AND final_track_id='1';
                    INSERT INTO mask_keyframes VALUES (4, 1, 1);
                    INSERT INTO keyframe_components
                        VALUES (4, 4, 0, 'polygon');
                    INSERT INTO keyframe_polygon_rings
                        VALUES (3, 4, 0, 'exterior');
                    INSERT INTO keyframe_polygon_points VALUES
                        (3, 0, 100, 0), (3, 1, 102, 0),
                        (3, 2, 102, 2), (3, 3, 100, 2);
                    """
                )

            cache = root / "final-cache.sqlite"
            materialize_overlay_cache(
                source,
                cache,
                mode="final",
                mask_domain="genital",
            )
            with sqlite3.connect(cache) as connection:
                middle = json.loads(
                    connection.execute(
                        "SELECT polygons FROM masks " "WHERE frame=1 AND track_id='1'"
                    ).fetchone()[0]
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        "SELECT is_keyframe FROM masks "
                        "WHERE frame=1 AND track_id='1'"
                    ).fetchone()[0],
                )
            self.assertEqual(
                [[100.0, 0.0], [102.0, 0.0], [102.0, 2.0], [100.0, 2.0]],
                middle[0],
            )


if __name__ == "__main__":
    unittest.main()
