from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from artifacts.legacy_sqlite import (
    LEGACY_MASK_COLUMNS,
    LEGACY_SCHEMA_NAME,
    export_legacy_sqlite,
)
from run_pipeline import build_parser, run_pipeline
from tentative.export_legacy_sqlite import main as export_main
from tests.helpers import write_raw_detector_sqlite, write_sample_sqlite


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


class LegacySqliteTests(unittest.TestCase):
    def test_export_projects_only_the_former_three_table_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "current.sqlite", frames=3)
            with sqlite3.connect(source) as connection:
                connection.executescript(
                    """
                    INSERT INTO cuts(frame) VALUES (2);
                    CREATE TABLE raw_tracked_masks(
                        frame INTEGER,
                        track_id TEXT,
                        polygons TEXT
                    );
                    CREATE TABLE raw_tracks(track_id TEXT PRIMARY KEY);
                    CREATE TABLE cut_detection_metadata(
                        id INTEGER PRIMARY KEY,
                        schema_version INTEGER,
                        method TEXT,
                        elapsed_seconds REAL,
                        cut_count INTEGER,
                        frame_semantics TEXT
                    );
                    INSERT INTO cut_detection_metadata VALUES(
                        1, 1, 'frame_diff', 0.25, 1,
                        'first_frame_of_new_scene'
                    );
                    """
                )
            output = root / "legacy.sqlite"

            summary = export_legacy_sqlite(source, output)

            self.assertEqual(LEGACY_SCHEMA_NAME, summary["schema"])
            self.assertEqual(3, summary["masks"])
            self.assertEqual(1, summary["tracks"])
            self.assertEqual(1, summary["cuts"])
            with sqlite3.connect(output) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                self.assertEqual({"masks", "tracks", "cuts"}, tables)
                self.assertEqual(
                    list(LEGACY_MASK_COLUMNS),
                    _columns(connection, "masks"),
                )
                self.assertEqual(
                    [
                        ("frame", "INTEGER", 1, None, 1),
                        ("track_id", "TEXT", 1, None, 2),
                        ("polygons", "TEXT", 0, None, 0),
                        ("shape_type", "TEXT", 0, None, 0),
                        ("dilate_px", "INTEGER", 1, "0", 0),
                        ("feather_px", "INTEGER", 1, "0", 0),
                        ("mosaic_block", "INTEGER", 1, "0", 0),
                        ("mosaic_alias", "REAL", 1, "0", 0),
                        ("label", "TEXT", 0, None, 0),
                    ],
                    [
                        (row[1], row[2], row[3], row[4], row[5])
                        for row in connection.execute("PRAGMA table_info(masks)")
                    ],
                )
                self.assertEqual(["track_id", "label"], _columns(connection, "tracks"))
                self.assertEqual(["frame"], _columns(connection, "cuts"))
                self.assertEqual(
                    [(2,)], connection.execute("SELECT * FROM cuts").fetchall()
                )

    def test_pipeline_option_publishes_additional_legacy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_raw_detector_sqlite(root / "raw.sqlite", frames=4)
            args = build_parser().parse_args(
                [
                    "--input-sqlite",
                    str(source),
                    "--output-dir",
                    str(root / "output"),
                    "--shape-mode",
                    "polygon",
                    "--no-cut-detect",
                    "--remove-short-tracks-max-frames",
                    "0",
                    "--no-polygon-endpoint-extend",
                    "--export-legacy-sqlite",
                ]
            )

            manifest = run_pipeline(args)

            current = Path(manifest["artifacts"]["predictions_sqlite"])
            legacy = Path(manifest["artifacts"]["legacy_predictions_sqlite"])
            self.assertNotEqual(current, legacy)
            self.assertTrue(legacy.is_file())
            with sqlite3.connect(current) as connection:
                current_tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            with sqlite3.connect(legacy) as connection:
                legacy_tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                legacy_masks = int(
                    connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
                )
            self.assertIn("raw_tracked_masks", current_tables)
            self.assertIn("cut_detection_metadata", current_tables)
            self.assertEqual({"masks", "tracks", "cuts"}, legacy_tables)
            self.assertEqual(4, legacy_masks)

    def test_tentative_cli_exports_and_reports_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_sample_sqlite(root / "current.sqlite", frames=2)
            output = root / "legacy.sqlite"
            stdout = StringIO()

            with redirect_stdout(stdout):
                status = export_main(
                    [
                        "--input-sqlite",
                        str(source),
                        "--output-sqlite",
                        str(output),
                    ]
                )

            self.assertEqual(0, status)
            self.assertEqual(
                LEGACY_SCHEMA_NAME, json.loads(stdout.getvalue())["schema"]
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
