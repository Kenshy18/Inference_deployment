from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from contracts.detector_sqlite import (
    detect_mask_sqlite_kind,
    validate_raw_detection_sqlite,
    validate_unified_inference_sqlite,
)
from preprocessing.raw_sqlite import normalize_raw_detection_sqlite
from run_pipeline import build_parser, run_pipeline
from tests.helpers import (
    write_raw_detector_sqlite,
    write_sample_sqlite,
    write_unified_inference_sqlite,
)


class RawSqliteInputTests(unittest.TestCase):
    def test_detects_dinov3_raw_and_tracked_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = write_raw_detector_sqlite(root / "raw.sqlite", frames=2)
            tracked = write_sample_sqlite(root / "tracked.sqlite", frames=2)

            self.assertEqual("raw_detection", detect_mask_sqlite_kind(raw))
            self.assertEqual("tracked", detect_mask_sqlite_kind(tracked))
            validate_raw_detection_sqlite(raw)

    def test_detects_and_validates_unified_inference_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(
                root / "unified.sqlite", frames=3
            )

            self.assertEqual(
                "unified_inference", detect_mask_sqlite_kind(source)
            )
            validate_unified_inference_sqlite(source)

    def test_raw_sqlite_normalizes_flat_dinov3_polygons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = write_raw_detector_sqlite(root / "raw.sqlite", frames=2)
            output = root / "normalized.jsonl"

            stats = normalize_raw_detection_sqlite(raw, output)

            self.assertEqual(2, stats["frames"])
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            detection = records[0]["detections"][0]
            self.assertEqual("sample", detection["class_name"])
            self.assertEqual(4, len(detection["polygons"][0]))
            self.assertEqual([4.0, 8.0], detection["polygons"][0][0])
            self.assertEqual([4.0, 8.0, 16.0, 20.0], detection["bbox_xyxy"])

    def test_default_polygon_pipeline_accepts_raw_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = write_raw_detector_sqlite(root / "raw.sqlite", frames=6)
            args = build_parser().parse_args(
                [
                    "--input-sqlite",
                    str(raw),
                    "--output-dir",
                    str(root / "output"),
                    "--shape-mode",
                    "polygon",
                    "--no-cut-detect",
                    "--remove-short-tracks-max-frames",
                    "0",
                ]
            )

            manifest = run_pipeline(args)

            self.assertTrue(manifest["complete"])
            self.assertEqual(
                "polygon_raw_sqlite_modular", manifest["pipeline"]
            )
            self.assertEqual(
                "preprocessing.raw_sqlite",
                manifest["stages"][0]["implementation"],
            )
            with sqlite3.connect(
                manifest["artifacts"]["predictions_sqlite"]
            ) as connection:
                self.assertEqual(
                    6,
                    connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                )

    def test_unified_inference_normalizes_scores_and_empty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(
                root / "unified.sqlite", frames=3
            )
            output = root / "normalized.jsonl"

            stats = normalize_raw_detection_sqlite(source, output)

            self.assertEqual(3, stats["frames"])
            self.assertEqual(2, stats["detections"])
            self.assertEqual(1, stats["empty_frames"])
            self.assertEqual(8, stats["points"])
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            detection = records[0]["detections"][0]
            self.assertEqual("sample", detection["class_name"])
            self.assertEqual(0.75, detection["detector_score"])
            self.assertEqual(0.95, detection["class_score"])
            self.assertEqual(4, len(detection["polygons"][0]))
            self.assertEqual([], records[-1]["detections"])

    def test_default_polygon_pipeline_accepts_unified_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(
                root / "unified.sqlite", frames=6
            )
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
                ]
            )

            manifest = run_pipeline(args)

            self.assertTrue(manifest["complete"])
            self.assertEqual(
                "unified_inference",
                detect_mask_sqlite_kind(source),
            )
            self.assertEqual(
                "preprocessing.raw_sqlite",
                manifest["stages"][0]["implementation"],
            )
            with sqlite3.connect(
                manifest["artifacts"]["predictions_sqlite"]
            ) as connection:
                self.assertEqual(
                    5,
                    connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                )


if __name__ == "__main__":
    unittest.main()
