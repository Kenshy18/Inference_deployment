from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from orchestration.contracts import public_result_schema_signature
from orchestration.rescale_result_sqlite import (
    VideoGeometry,
    rescale_inference_sqlite_for_postprocess,
    rescale_result_sqlite,
)


COORDINATE_TABLES = {
    "detections": "x1 REAL, y1 REAL, x2 REAL, y2 REAL",
    "segmentation_points": "x REAL, y REAL",
    "face_keypoints": "x REAL, y REAL",
    "face_masks": "box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL",
    "face_observations": (
        "ellipse_cx REAL, ellipse_cy REAL, "
        "ellipse_major_radius REAL, ellipse_minor_radius REAL"
    ),
    "face_track_interpolations": (
        "head_x1 REAL, head_y1 REAL, head_x2 REAL, head_y2 REAL"
    ),
    "face_tracking_assignments": (
        "head_x1 REAL, head_y1 REAL, head_x2 REAL, head_y2 REAL"
    ),
    "keyframe_ellipses": "cx REAL, cy REAL, radius_x REAL, radius_y REAL",
    "keyframe_polygon_points": "x REAL, y REAL",
    "keyframe_rectangles": (
        "cx REAL, cy REAL, half_width REAL, half_height REAL"
    ),
}


def create_proxy_result(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE frames(width INTEGER, height INTEGER)")
        connection.execute("INSERT INTO frames VALUES(1920, 1080)")
        connection.execute(
            """
            CREATE TABLE videos(
                path TEXT,
                reported_frame_count INTEGER,
                fps REAL,
                width INTEGER,
                height INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO videos VALUES('proxy.mp4', 10, 30, 1920, 1080)"
        )
        connection.execute(
            "CREATE TABLE video_streams(width INTEGER, height INTEGER, frame_count INTEGER)"
        )
        connection.execute("INSERT INTO video_streams VALUES(1920, 1080, 10)")
        connection.execute(
            "CREATE TABLE model_metadata(key TEXT, value TEXT, value_type TEXT)"
        )
        connection.execute(
            "INSERT INTO model_metadata VALUES('input', 'proxy.mp4', 'str')"
        )
        connection.execute(
            "CREATE TABLE run_metadata(key TEXT PRIMARY KEY, value TEXT, value_type TEXT)"
        )
        for table, declaration in COORDINATE_TABLES.items():
            columns = declaration.split(",")
            connection.execute(f"CREATE TABLE {table}({declaration})")
            placeholders = ", ".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO {table} VALUES({placeholders})",
                tuple(float(index + 1) for index in range(len(columns))),
            )
    return path


class RescaleResultSqliteTests(unittest.TestCase):
    def test_rescale_preserves_schema_and_restores_source_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_proxy_result(root / "proxy.sqlite")
            output = root / "result.sqlite"
            with sqlite3.connect(source) as connection:
                source_signature = public_result_schema_signature(connection)

            summary = rescale_result_sqlite(
                source,
                output,
                proxy=VideoGeometry(1920, 1080, 30.0, 10),
                original=VideoGeometry(3840, 2160, 30.0, 10),
                original_video=Path("/video/original.mp4"),
            )

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    source_signature,
                    public_result_schema_signature(connection),
                )
                self.assertEqual(
                    (2.0, 4.0, 6.0, 8.0),
                    connection.execute("SELECT * FROM detections").fetchone(),
                )
                self.assertEqual(
                    (3840, 2160),
                    connection.execute("SELECT * FROM frames").fetchone(),
                )
                self.assertEqual(
                    ("/video/original.mp4", 10, 30.0, 3840, 2160),
                    connection.execute("SELECT * FROM videos").fetchone(),
                )
                self.assertEqual(
                    ("2.0",),
                    connection.execute(
                        "SELECT value FROM run_metadata "
                        "WHERE key='analysis_proxy.scale_x_to_source'"
                    ).fetchone(),
                )
            self.assertEqual(2.0, summary["scale_x"])
            self.assertEqual(2.0, summary["scale_y"])

    def test_rescale_rejects_nonuniform_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_proxy_result(root / "proxy.sqlite")
            with self.assertRaisesRegex(ValueError, "uniform scale"):
                rescale_result_sqlite(
                    source,
                    root / "result.sqlite",
                    proxy=VideoGeometry(1920, 1080, 30.0, 10),
                    original=VideoGeometry(4096, 2160, 30.0, 10),
                    original_video=Path("/video/original.mp4"),
                )

    def test_rescale_restores_720p_source_from_canonical_1080p(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_proxy_result(root / "canonical.sqlite")
            output = root / "result.sqlite"

            summary = rescale_result_sqlite(
                source,
                output,
                proxy=VideoGeometry(1920, 1080, 30.0, 10),
                original=VideoGeometry(1280, 720, 30.0, 10),
                original_video=Path("/video/original-720p.mp4"),
            )

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    (2.0 / 3.0, 4.0 / 3.0, 2.0, 8.0 / 3.0),
                    connection.execute("SELECT * FROM detections").fetchone(),
                )
                self.assertEqual(
                    (1280, 720),
                    connection.execute("SELECT * FROM frames").fetchone(),
                )
                self.assertEqual(
                    (1280, 720, 10),
                    connection.execute("SELECT * FROM video_streams").fetchone(),
                )
            self.assertEqual(2.0 / 3.0, summary["scale_x"])
            self.assertEqual(2.0 / 3.0, summary["scale_y"])

    def test_inference_coordinates_expand_to_1080p_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = create_proxy_result(root / "inference.sqlite")
            output = root / "canonical.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute("UPDATE frames SET width=1280, height=720")
                connection.execute(
                    "UPDATE videos SET width=1280, height=720, path='source.mp4'"
                )
                connection.execute(
                    "UPDATE video_streams SET width=1280, height=720"
                )

            summary = rescale_inference_sqlite_for_postprocess(
                source,
                output,
                inference=VideoGeometry(1280, 720, 30.0, 10),
                workspace=VideoGeometry(1920, 1080, 30.0, 10),
                workspace_video=Path("/video/workspace-1080p.mp4"),
            )

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    (1.5, 3.0, 4.5, 6.0),
                    connection.execute("SELECT * FROM detections").fetchone(),
                )
                self.assertEqual(
                    (1920, 1080),
                    connection.execute("SELECT * FROM frames").fetchone(),
                )
                self.assertEqual(
                    ("1.5",),
                    connection.execute(
                        "SELECT value FROM run_metadata WHERE "
                        "key='postprocess_workspace.scale_x'"
                    ).fetchone(),
                )
                self.assertIsNone(
                    connection.execute(
                        "SELECT value FROM run_metadata WHERE "
                        "key='analysis_proxy.scale_x_to_source'"
                    ).fetchone()
                )
            self.assertEqual(1.5, summary["scale_x"])

    def test_inference_rescale_accepts_schema_without_result_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "legacy-inference.sqlite"
            output = root / "canonical.sqlite"
            with sqlite3.connect(source) as connection:
                connection.executescript(
                    """
                    CREATE TABLE frames(width INTEGER, height INTEGER);
                    INSERT INTO frames VALUES(1280, 720);
                    CREATE TABLE videos(
                        path TEXT, reported_frame_count INTEGER, fps REAL,
                        width INTEGER, height INTEGER
                    );
                    INSERT INTO videos VALUES('source.mp4', 10, 30, 1280, 720);
                    CREATE TABLE detections(
                        x1 REAL, y1 REAL, x2 REAL, y2 REAL
                    );
                    INSERT INTO detections VALUES(10, 20, 30, 40);
                    CREATE TABLE segmentation_points(x REAL, y REAL);
                    INSERT INTO segmentation_points VALUES(12, 24);
                    """
                )

            rescale_inference_sqlite_for_postprocess(
                source,
                output,
                inference=VideoGeometry(1280, 720, 30.0, 10),
                workspace=VideoGeometry(1920, 1080, 30.0, 10),
                workspace_video=Path("/video/workspace.mp4"),
            )

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    (15.0, 30.0, 45.0, 60.0),
                    connection.execute("SELECT * FROM detections").fetchone(),
                )
                self.assertEqual(
                    (18.0, 36.0),
                    connection.execute(
                        "SELECT * FROM segmentation_points"
                    ).fetchone(),
                )


if __name__ == "__main__":
    unittest.main()
