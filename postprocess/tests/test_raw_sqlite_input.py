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
            source = write_unified_inference_sqlite(root / "unified.sqlite", frames=3)

            self.assertEqual("unified_inference", detect_mask_sqlite_kind(source))
            validate_unified_inference_sqlite(source)

    def test_accepts_unified_inference_schema_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(
                root / "unified-v3.sqlite", frames=3
            )
            with sqlite3.connect(source) as connection:
                connection.execute(
                    """
                    UPDATE schema_info SET value='3'
                    WHERE key='schema_version'
                    """
                )
            output = root / "normalized.jsonl"

            validate_unified_inference_sqlite(source)
            stats = normalize_raw_detection_sqlite(source, output)

            self.assertEqual(
                "instance-segmentation-unified-inference-v3",
                stats["input_schema"],
            )

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
            raw = write_raw_detector_sqlite(root / "raw.sqlite", frames=6, label="男性器")
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
            self.assertEqual("polygon_raw_sqlite_modular", manifest["pipeline"])
            self.assertEqual(
                "preprocessing.raw_sqlite",
                manifest["stages"][0]["implementation"],
            )
            with sqlite3.connect(
                manifest["artifacts"]["predictions_sqlite"]
            ) as connection:
                self.assertEqual(
                    11,
                    connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                )

    def test_unified_inference_normalizes_scores_and_empty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(root / "unified.sqlite", frames=3)
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
                root / "unified.sqlite", frames=6, label="男性器"
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
                    10,
                    connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                )
            result = Path(manifest["artifacts"]["result_sqlite"])
            with sqlite3.connect(result) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("detections", tables)
                self.assertIn("tracking_assignments", tables)
                self.assertNotIn("tracked_masks", tables)
                self.assertNotIn("masks", tables)
                self.assertEqual(
                    5,
                    connection.execute(
                        "SELECT COUNT(*) FROM tracking_assignments"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    "video-mask-integrated-result",
                    connection.execute(
                        """
                        SELECT value FROM result_schema_info
                        WHERE key='schema_name'
                        """
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
                self.assertEqual(
                    0,
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM tracking_assignments AS a
                        LEFT JOIN detections AS d
                          ON d.id=a.source_detection_id
                        WHERE d.id IS NULL
                        """
                    ).fetchone()[0],
                )
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM mask_keyframes"
                    ).fetchone()[0],
                    0,
                )
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM editable_polygon_vertices"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM processing_runs
                        WHERE kind='postprocess'
                        """
                    ).fetchone()[0],
                )

    def test_classwise_polygon_keyframes_are_promoted_to_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_unified_inference_sqlite(
                root / "unified.sqlite", frames=6, label="男性器"
            )
            policy = root / "classwise.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default": {
                            "shape_mode": "polygon",
                            "keyframe_interval": 2,
                            "max_gap": 15,
                        },
                        "classes": {},
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--input-sqlite",
                    str(source),
                    "--output-dir",
                    str(root / "output"),
                    "--shape-mode",
                    "polygon",
                    "--class-postprocess-policy-json",
                    str(policy),
                    "--no-cut-detect",
                    "--remove-short-tracks-max-frames",
                    "0",
                ]
            )

            manifest = run_pipeline(args)

            with sqlite3.connect(manifest["artifacts"]["result_sqlite"]) as connection:
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM mask_keyframes"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        """
                        SELECT available FROM result_capabilities
                        WHERE name='classwise_postprocess'
                        """
                    ).fetchone()[0],
                )


if __name__ == "__main__":
    unittest.main()
