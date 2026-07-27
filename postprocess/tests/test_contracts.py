from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from artifacts.contract import (
    OutputContractError,
    validate_mask_sqlite,
)
from preprocessing.normalization import (
    DetectionJsonlContractError,
    normalize_frame_record,
    summarize_detection_jsonl,
)
from preprocessing.score_policy import ScorePolicy
from contracts.mask_sqlite import MaskRow, write_mask_sqlite


class InputContractTests(unittest.TestCase):
    def test_historic_aliases_normalize_to_canonical_fields(self) -> None:
        record = normalize_frame_record(
            {
                "frame_idx": 3,
                "instances": [
                    {
                        "label": "sample",
                        "score": 0.9,
                        "bbox": [10, 20, 30, 40],
                        "segmentation": [[[10, 20], [40, 20], [40, 60]]],
                    }
                ],
            }
        )
        self.assertEqual(record["frame_index"], 3)
        self.assertEqual(record["detections"][0]["bbox_xyxy"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(record["detections"][0]["class_name"], "sample")

    def test_streaming_validator_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            path.write_text(
                json.dumps({"frame_index": 0, "detections": []}) + "\n{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DetectionJsonlContractError, r"input\.jsonl:2"):
                summarize_detection_jsonl(path)

    def test_detection_without_mask_is_rejected(self) -> None:
        with self.assertRaises(DetectionJsonlContractError):
            normalize_frame_record(
                {
                    "frame_index": 0,
                    "detections": [{"class_name": "sample", "score": 0.9}],
                }
            )


class ScorePolicyTests(unittest.TestCase):
    def test_class_threshold_overrides_default(self) -> None:
        policy = ScorePolicy(default_min=0.2, by_label={"strict": 0.8})
        self.assertTrue(policy.accepts({"class_name": "other", "score": 0.3}))
        self.assertFalse(policy.accepts({"class_name": "strict", "score": 0.7}))


class OutputContractTests(unittest.TestCase):
    def _write_sqlite(self, path: Path, polygons: str) -> None:
        connection = sqlite3.connect(str(path))
        try:
            connection.execute(
                "CREATE TABLE masks(frame INTEGER, track_id TEXT, polygons TEXT)"
            )
            connection.execute("CREATE TABLE tracks(track_id TEXT, label TEXT)")
            connection.execute(
                "INSERT INTO masks VALUES (?, ?, ?)", (2, "track-1", polygons)
            )
            connection.execute(
                "INSERT INTO tracks VALUES (?, ?)", ("track-1", "sample")
            )
            connection.commit()
        finally:
            connection.close()

    def test_valid_output_is_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.sqlite"
            self._write_sqlite(path, json.dumps([[[0, 0], [1, 0], [1, 1]]]))
            stats = validate_mask_sqlite(path)
            self.assertEqual(stats.masks, 1)
            self.assertEqual(stats.tracks, 1)
            self.assertEqual(stats.first_frame, 2)

    def test_invalid_polygon_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.sqlite"
            self._write_sqlite(path, "{bad json")
            with self.assertRaises(OutputContractError):
                validate_mask_sqlite(path)

    def test_predictions_without_tracks_table_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.sqlite"
            connection = sqlite3.connect(str(path))
            try:
                connection.execute(
                    "CREATE TABLE masks(frame INTEGER, track_id TEXT, polygons TEXT)"
                )
                connection.execute(
                    "INSERT INTO masks VALUES (?, ?, ?)",
                    (0, "1", json.dumps([[[0, 0], [1, 0], [1, 1]]])),
                )
                connection.commit()
            finally:
                connection.close()

            stats = validate_mask_sqlite(path)
            self.assertEqual(stats.masks, 1)
            self.assertEqual(stats.tracks, 0)

    def test_reference_sqlite_is_backed_up_safely_while_wal_is_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.sqlite"
            output = root / "output.sqlite"
            connection = sqlite3.connect(str(reference))
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE masks(
                        frame INTEGER,
                        track_id TEXT,
                        polygons TEXT,
                        shape_type TEXT,
                        dilate_px INTEGER DEFAULT 0,
                        feather_px INTEGER DEFAULT 0,
                        mosaic_block INTEGER DEFAULT 0,
                        mosaic_alias REAL DEFAULT 0,
                        label TEXT,
                        PRIMARY KEY(frame, track_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)"
                )
                connection.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
                connection.execute(
                    """
                    CREATE TABLE cut_detection_metadata(
                        id INTEGER PRIMARY KEY,
                        schema_version INTEGER,
                        method TEXT,
                        elapsed_seconds REAL,
                        cut_count INTEGER,
                        frame_semantics TEXT
                    )
                    """
                )
                connection.execute("INSERT INTO cuts(frame) VALUES (3)")
                connection.execute(
                    """
                    INSERT INTO cut_detection_metadata
                    VALUES (1, 1, 'fixed', 0.25, 1,
                            'first_frame_of_new_scene')
                    """
                )
                connection.commit()

                write_mask_sqlite(
                    output,
                    [
                        MaskRow(
                            0,
                            "1",
                            json.dumps([[[0, 0], [1, 0], [1, 1]]]),
                            "sample",
                        )
                    ],
                    reference_sqlite=reference,
                )
            finally:
                connection.close()

            with sqlite3.connect(str(output)) as copied:
                self.assertEqual(
                    1,
                    copied.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                )
                self.assertEqual(
                    [(3,)],
                    copied.execute("SELECT frame FROM cuts").fetchall(),
                )
                self.assertEqual(
                    [("fixed", 1)],
                    copied.execute(
                        "SELECT method, cut_count FROM cut_detection_metadata"
                    ).fetchall(),
                )
            stats = validate_mask_sqlite(output)
            self.assertEqual(stats.cuts, 1)
            self.assertEqual(stats.cut_detection_method, "fixed")

    def test_cut_metadata_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.sqlite"
            self._write_sqlite(
                path,
                json.dumps([[[0, 0], [1, 0], [1, 1]]]),
            )
            with sqlite3.connect(str(path)) as connection:
                connection.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
                connection.execute(
                    """
                    CREATE TABLE cut_detection_metadata(
                        id INTEGER PRIMARY KEY,
                        schema_version INTEGER,
                        method TEXT,
                        elapsed_seconds REAL,
                        cut_count INTEGER,
                        frame_semantics TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO cut_detection_metadata
                    VALUES (1, 1, 'fixed', 0.1, 1,
                            'first_frame_of_new_scene')
                    """
                )
            with self.assertRaisesRegex(
                OutputContractError,
                "cut_count does not match",
            ):
                validate_mask_sqlite(path)


if __name__ == "__main__":
    unittest.main()
