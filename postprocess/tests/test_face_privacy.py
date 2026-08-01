from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from common.config import PipelineConfig, StageSpec
from common.runner import PipelineRunner
from contracts.artifacts import validate_artifact
from face_privacy.geometry import FaceKeypoint, derive_privacy_mask
from face_privacy.sqlite import export_face_masks, merge_face_masks


def write_rich_inference(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE frames(
                id INTEGER PRIMARY KEY, run_id INTEGER, frame_index INTEGER,
                timestamp_sec REAL, width INTEGER, height INTEGER
            );
            CREATE TABLE model_executions(
                id INTEGER PRIMARY KEY, role TEXT
            );
            CREATE TABLE detections(
                id INTEGER PRIMARY KEY, frame_id INTEGER,
                model_execution_id INTEGER, class_id INTEGER,
                class_name TEXT, score REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL
            );
            CREATE TABLE classifications(
                detection_id INTEGER, class_id INTEGER,
                class_name TEXT, score REAL
            );
            CREATE TABLE segmentations(detection_id INTEGER, encoding TEXT);
            CREATE TABLE segmentation_polygons(
                id INTEGER, detection_id INTEGER, polygon_index INTEGER
            );
            CREATE TABLE segmentation_points(
                polygon_id INTEGER, point_index INTEGER, x REAL, y REAL
            );
            CREATE TABLE face_observations(
                id INTEGER PRIMARY KEY, anchor_detection_id INTEGER,
                face_score REAL, face_present INTEGER, geometry_type TEXT,
                ellipse_cx REAL, ellipse_cy REAL,
                ellipse_major_radius REAL, ellipse_minor_radius REAL,
                ellipse_theta_radians REAL
            );
            CREATE TABLE face_keypoints(
                observation_id INTEGER, point_index INTEGER,
                class_name TEXT, x REAL, y REAL,
                confidence REAL, valid INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO schema_info VALUES (?, ?)",
            (
                ("schema_name", "instance-segmentation-unified-inference"),
                ("schema_version", "3"),
            ),
        )
        connection.execute("INSERT INTO frames VALUES (1, 1, 7, 0.25, 200, 160)")
        connection.execute("INSERT INTO model_executions VALUES (1, 'face_detection')")
        connection.execute(
            """
            INSERT INTO detections VALUES(
                1, 1, 1, 0, 'Head', 0.98, 30, 20, 170, 150
            )
            """
        )
        connection.execute(
            """
            INSERT INTO face_observations VALUES(
                1, 1, 0.96, 1, 'ellipse',
                100, 85, 60, 50, ?
            )
            """,
            (math.pi / 2.0,),
        )
        points = (
            (1, 0, "Eye", 70.0, 65.0, 0.98, 1),
            (1, 1, "Eye", 130.0, 65.0, 0.97, 1),
            (1, 2, "Nose", 100.0, 95.0, 0.96, 1),
            (1, 3, "Mouth", 90.0, 120.0, 0.95, 1),
            (1, 4, "Mouth", 110.0, 120.0, 0.95, 1),
        )
        connection.executemany(
            "INSERT INTO face_keypoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            points,
        )
    return path


def write_predictions(path: Path, *, wal: bool = False) -> Path:
    polygon = json.dumps([[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]])
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL, track_id TEXT NOT NULL,
                polygons TEXT, shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT, PRIMARY KEY(frame, track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE cuts(frame INTEGER PRIMARY KEY);
            """
        )
        if wal:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            )
            if journal_mode.lower() != "wal":
                raise RuntimeError("failed to create WAL-mode test fixture")
        connection.execute(
            """
            INSERT INTO masks VALUES(
                7, '1', ?, 'polygon', 0, 0, 0, 0, 'genital'
            )
            """,
            (polygon,),
        )
        connection.execute("INSERT INTO tracks VALUES ('1', 'genital')")
    return path


class FacePrivacyTests(unittest.TestCase):
    def test_adopted_eye_dimensions_are_stable_for_both_shapes(self) -> None:
        keypoints = (
            FaceKeypoint(70.0, 65.0, "Eye", 0.98, True),
            FaceKeypoint(130.0, 65.0, "Eye", 0.97, True),
            FaceKeypoint(100.0, 95.0, "Nose", 0.96, True),
            FaceKeypoint(100.0, 120.0, "Mouth", 0.95, True),
        )
        for shape, points in (("ellipse", 64), ("rectangle", 4)):
            with self.subTest(shape=shape):
                mask = derive_privacy_mask(
                    "eyes",
                    (100.0, 90.0, 60.0, 50.0, math.pi / 2.0),
                    keypoints,
                    eye_shape=shape,
                )
                self.assertIsNotNone(mask)
                assert mask is not None
                self.assertEqual(points, len(mask.polygon))
                xs = [point[0] for point in mask.polygon]
                ys = [point[1] for point in mask.polygon]
                self.assertAlmostEqual(41.8, min(xs), places=5)
                self.assertAlmostEqual(158.2, max(xs), places=5)
                self.assertAlmostEqual(40.7, min(ys), places=5)
                self.assertAlmostEqual(89.3, max(ys), places=5)

    def test_export_is_non_destructive_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_rich_inference(root / "inference.sqlite")
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / "eyes.sqlite"

            summary = export_face_masks(
                source,
                output,
                target="eyes",
                eye_shape="rectangle",
            )

            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(1, summary["rows"])
            with sqlite3.connect(output) as connection:
                row = connection.execute(
                    """
                    SELECT m.frame, m.track_id, m.shape_type, m.label,
                           p.mask_kind, p.derivation, p.confidence
                    FROM masks m
                    JOIN mask_provenance p USING(frame, track_id)
                    """
                ).fetchone()
                self.assertEqual(
                    (
                        7,
                        "face:eyes:0:1",
                        "rectangle",
                        "Eyes",
                        "eyes",
                        "eye-keypoints",
                        0.97,
                    ),
                    row,
                )

    def test_face_and_head_thresholds_filter_only_postprocessed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_rich_inference(root / "inference.sqlite")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            face_filtered = root / "face-filtered.sqlite"
            face_summary = export_face_masks(
                source,
                face_filtered,
                target="eyes",
                face_detection_score_threshold=0.97,
            )
            with sqlite3.connect(face_filtered) as connection:
                self.assertEqual(
                    0, connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        "SELECT COUNT(*) FROM face_tracking_assignments"
                    ).fetchone()[0],
                )
            self.assertEqual(1, face_summary["derivations"]["face_below_threshold"])

            head_filtered = root / "head-filtered.sqlite"
            head_summary = export_face_masks(
                source,
                head_filtered,
                target="eyes",
                head_detection_score_threshold=0.99,
            )
            with sqlite3.connect(head_filtered) as connection:
                self.assertEqual(
                    0, connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
                )
                self.assertEqual(
                    0,
                    connection.execute(
                        "SELECT COUNT(*) FROM face_tracking_assignments"
                    ).fetchone()[0],
                )
            self.assertEqual(1, head_summary["derivations"]["head_below_threshold"])
            self.assertEqual(
                source_hash, hashlib.sha256(source.read_bytes()).hexdigest()
            )

    def test_tracking_keeps_one_id_and_interpolates_only_the_internal_gap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_rich_inference(root / "inference.sqlite")
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "UPDATE frames SET frame_index=0, timestamp_sec=0 WHERE id=1"
                )
                connection.execute(
                    "INSERT INTO frames VALUES (2, 1, 2, 0.08, 200, 160)"
                )
                connection.execute(
                    """
                    INSERT INTO detections VALUES(
                        2, 2, 1, 0, 'Head', 0.97, 32, 20, 172, 150
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO face_observations VALUES(
                        2, 2, 0.95, 1, 'ellipse',
                        102, 85, 60, 50, ?
                    )
                    """,
                    (math.pi / 2.0,),
                )
                connection.executemany(
                    "INSERT INTO face_keypoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        (2, 0, "Eye", 72.0, 65.0, 0.98, 1),
                        (2, 1, "Eye", 132.0, 65.0, 0.97, 1),
                        (2, 2, "Nose", 102.0, 95.0, 0.96, 1),
                        (2, 3, "Mouth", 92.0, 120.0, 0.95, 1),
                        (2, 4, "Mouth", 112.0, 120.0, 0.95, 1),
                    ),
                )
            output = root / "eyes.sqlite"

            summary = export_face_masks(
                source,
                output,
                target="eyes",
                interpolation_max_gap=1,
            )

            self.assertEqual(3, summary["rows"])
            self.assertEqual(1, summary["face_tracks"])
            self.assertEqual(1, summary["interpolated_rows"])
            self.assertEqual(1, summary["interpolated_boxes"])
            validate_artifact("face_masks_sqlite", output)
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [
                        (0, "face:eyes:0:1", 0),
                        (1, "face:eyes:0:1", 1),
                        (2, "face:eyes:0:1", 0),
                    ],
                    connection.execute(
                        """
                        SELECT frame, track_id, is_interpolated
                        FROM mask_provenance ORDER BY frame
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    [
                        (
                            1,
                            "face:0:1",
                            31.0,
                            20.0,
                            171.0,
                            150.0,
                            "linear-two-sided",
                        )
                    ],
                    connection.execute(
                        """
                        SELECT frame, final_track_id,
                               head_x1, head_y1, head_x2, head_y2,
                               interpolation_method
                        FROM face_track_interpolations
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    [("face:0:1", 2)],
                    connection.execute(
                        """
                        SELECT final_track_id, COUNT(*)
                        FROM face_tracking_assignments
                        GROUP BY final_track_id
                        """
                    ).fetchall(),
                )

    def test_merge_preserves_genital_masks_and_adds_face_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_rich_inference(root / "inference.sqlite")
            face_masks = root / "eyes.sqlite"
            export_face_masks(
                source,
                face_masks,
                target="eyes",
                eye_shape="ellipse",
            )
            predictions = write_predictions(root / "predictions.sqlite")
            output = root / "combined.sqlite"

            summary = merge_face_masks(predictions, face_masks, output)

            self.assertEqual(
                {
                    "genital_masks": 1,
                    "face_masks": 1,
                    "total_masks": 2,
                },
                summary,
            )
            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [("1", "genital"), ("face:eyes:0:1", "Eyes")],
                    connection.execute(
                        "SELECT track_id, label FROM masks ORDER BY track_id"
                    ).fetchall(),
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        "SELECT COUNT(*) FROM mask_provenance"
                    ).fetchone()[0],
                )
                self.assertEqual(
                    "ok",
                    connection.execute("PRAGMA integrity_check").fetchone()[0],
                )

    def test_merge_checkpoints_a_wal_mode_source_before_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_rich_inference(root / "inference.sqlite")
            face_masks = root / "eyes.sqlite"
            export_face_masks(source, face_masks, target="eyes")
            predictions = write_predictions(
                root / "predictions.sqlite",
                wal=True,
            )
            output = root / "combined.sqlite"

            merge_face_masks(predictions, face_masks, output)

            with sqlite3.connect(output) as connection:
                self.assertEqual(
                    [("Eyes", 1), ("genital", 1)],
                    connection.execute(
                        """
                        SELECT label, COUNT(*) FROM masks
                        GROUP BY label ORDER BY label
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    "delete",
                    connection.execute("PRAGMA journal_mode").fetchone()[0],
                )
            self.assertFalse(any(root.glob(".combined.sqlite.*.tmp-*")))

    def test_registered_stages_run_as_a_contract_connected_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inference = write_rich_inference(root / "inference.sqlite")
            predictions = write_predictions(root / "predictions.sqlite")
            config = PipelineConfig(
                "face_privacy_test",
                (
                    StageSpec(
                        "face_masks",
                        "face_privacy.masks",
                        {"target": "eyes", "eye_shape": "rectangle"},
                    ),
                    StageSpec("merge", "face_privacy.merge"),
                    StageSpec(
                        "validate",
                        "artifacts.validate",
                        {
                            "source_artifact": "combined_predictions_sqlite",
                            "output_artifact": "combined_validation_report",
                        },
                    ),
                ),
            )

            manifest = PipelineRunner(config, root / "output").run(
                {
                    "input_raw_sqlite": inference,
                    "predictions_sqlite": predictions,
                }
            )

            self.assertTrue(manifest["complete"])
            self.assertIn("face_masks_sqlite", manifest["artifacts"])
            self.assertIn("combined_predictions_sqlite", manifest["artifacts"])
            self.assertIn("combined_validation_report", manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
