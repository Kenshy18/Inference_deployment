from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from classwise.policy import (
    ClassPostprocessSettings,
    load_class_postprocess_policy,
)
from run_pipeline import build_parser, run_pipeline
from tracking.schema import create_schema


def _polygon(x: float) -> str:
    return json.dumps(
        [
            [
                [x, 0.0],
                [x + 10.0, 0.0],
                [x + 10.0, 10.0],
                [x, 10.0],
            ]
        ]
    )


def _tracked_sqlite(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        create_schema(connection)
        journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        if journal_mode.lower() != "wal":
            raise RuntimeError(f"test fixture did not enter WAL mode: {journal_mode}")
        connection.executemany(
            "INSERT INTO tracks(track_id, label) VALUES (?, ?)",
            (("1", "男性器"), ("2", "女性器")),
        )
        connection.executemany(
            """
            INSERT INTO masks(
                frame, track_id, polygons, shape_type, label
            ) VALUES (?, ?, ?, 'polygon', ?)
            """,
            (
                (0, "1", _polygon(0.0), "男性器"),
                (2, "1", _polygon(2.0), "男性器"),
                (0, "2", _polygon(20.0), "女性器"),
                (2, "2", _polygon(22.0), "女性器"),
            ),
        )
        connection.execute(
            """
            INSERT INTO cut_detection_metadata(
                id, schema_version, method, elapsed_seconds, cut_count,
                frame_semantics
            ) VALUES (1, 1, 'test', 0.0, 0, 'first_frame_of_new_scene')
            """
        )
    return path


class ClassPostprocessTests(unittest.TestCase):
    def test_old_polygon_max_gap_is_migrated_to_production_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old-policy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default": {
                            "shape_mode": "polygon",
                            "keyframe_interval": 6,
                            "max_gap": 0,
                        },
                        "classes": {"男性器": {"max_gap": 30}},
                    }
                ),
                encoding="utf-8",
            )
            policy = load_class_postprocess_policy(
                path,
                fallback=ClassPostprocessSettings("polygon", 6, 15),
            )
            self.assertEqual(15, policy.default.max_gap)
            self.assertEqual(15, policy.resolve("男性器").max_gap)

    def test_policy_rejects_retired_ellipse_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default": {
                            "shape_mode": "polygon",
                            "keyframe_interval": 4,
                            "max_gap": 15,
                        },
                        "classes": {
                            "ellipse-class": {
                                "shape_mode": "ellipse",
                                "keyframe_interval": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "polygon only"):
                load_class_postprocess_policy(
                    path,
                    fallback=ClassPostprocessSettings("polygon", 3, 15),
                )

    def test_polygon_classes_use_independent_keyframe_settings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _tracked_sqlite(root / "tracked.sqlite")
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default": {
                            "shape_mode": "polygon",
                            "keyframe_interval": 3,
                            "max_gap": 15,
                        },
                        "classes": {
                            "男性器": {
                                "keyframe_interval": 1,
                                "max_gap": 15,
                            },
                            "女性器": {
                                "keyframe_interval": 2,
                                "max_gap": 15,
                            },
                        },
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
                    "--class-postprocess-policy-json",
                    str(policy),
                ]
            )
            manifest = run_pipeline(args)
            final = Path(manifest["artifacts"]["predictions_sqlite"])
            with sqlite3.connect(final) as connection:
                self.assertEqual(
                    "delete",
                    str(
                        connection.execute("PRAGMA journal_mode").fetchone()[0]
                    ).lower(),
                )
                # Production endpoint protection extends the final observed
                # geometry by five frames before temporal optimization.
                self.assertEqual(
                    list(range(8)),
                    [
                        row[0]
                        for row in connection.execute(
                            "SELECT frame FROM masks "
                            "WHERE track_id='1' ORDER BY frame"
                        )
                    ],
                )
                self.assertEqual(
                    list(range(8)),
                    [
                        row[0]
                        for row in connection.execute(
                            "SELECT frame FROM masks "
                            "WHERE track_id='2' ORDER BY frame"
                        )
                    ],
                )
                self.assertEqual(
                    [
                        ("女性器", "polygon", 2, 15),
                        ("男性器", "polygon", 1, 15),
                    ],
                    connection.execute(
                        """
                        SELECT label, shape_mode, keyframe_interval, max_gap
                        FROM class_postprocess_policies
                        ORDER BY label
                        """
                    ).fetchall(),
                )
                self.assertEqual(
                    1,
                    connection.execute(
                        """
                        SELECT is_gap_filled
                        FROM mask_postprocess_provenance
                        WHERE frame=1 AND track_id='1'
                        """
                    ).fetchone()[0],
                )
            classwise_manifest = Path(manifest["artifacts"]["classwise_manifest"])
            classwise = json.loads(classwise_manifest.read_text(encoding="utf-8"))
            self.assertEqual(2, len(classwise["groups"]))
            self.assertEqual(12, classwise["merge"]["gap_filled_masks"])

    def test_pipeline_config_and_class_policy_are_mutually_exclusive(self) -> None:
        args = build_parser().parse_args(
            [
                "--input-sqlite",
                "tracked.sqlite",
                "--output-dir",
                "output",
                "--pipeline-config",
                "pipeline.json",
                "--class-postprocess-policy-json",
                "policy.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
