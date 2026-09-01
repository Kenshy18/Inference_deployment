from __future__ import annotations

import json
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

from classwise.policy import (
    ClassPostprocessSettings,
    load_class_postprocess_policy,
)
from classwise.curve_parallel import (
    available_cpu_count,
    curve_track_costs,
    partition_curve_tracks,
)
from classwise.curve_scheduling import allocate_curve_group_shards
from classwise.pipeline_factory import build_nested_pipeline
from common.config import PipelineConfig, StageSpec
from production.curve.parallel import _shard_preparations
from common.runner import PipelineRunner
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
    def test_curve_shard_budget_follows_semantic_workload(self) -> None:
        groups = (("1", "2", "3", "4", "5"), ("6", "7"))
        costs = {
            "1": 100,
            "2": 100,
            "3": 100,
            "4": 100,
            "5": 100,
            "6": 10,
            "7": 10,
        }
        self.assertEqual(
            (5, 1),
            allocate_curve_group_shards(
                groups,
                costs,
                process_budget=6,
            ),
        )
        self.assertEqual(
            (1, 1),
            allocate_curve_group_shards(
                groups,
                costs,
                process_budget=1,
            ),
        )

    def test_curve_worker_shards_are_complete_and_point_cost_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            endpoint = Path(temporary) / "endpoint.sqlite"
            with sqlite3.connect(endpoint) as connection:
                connection.execute("CREATE TABLE masks(track_id TEXT NOT NULL)")
                connection.executemany(
                    "INSERT INTO masks(track_id) VALUES (?)",
                    [
                        (track_id,)
                        for track_id in ("1", "2", "3", "4")
                        for _index in range(100)
                    ],
                )
            preparation = {
                "active_labels": ["test"],
                "classes": {
                    "test": {
                        "endpoint_sqlite": str(endpoint),
                        "input_rows": 400,
                    }
                },
                "vertex_policy": {
                    "tracks": {
                        "1": {"vertices_per_component": 20},
                        "2": {"vertices_per_component": 14},
                        "3": {"vertices_per_component": 14},
                        "4": {"vertices_per_component": 14},
                    }
                },
            }
            shards = _shard_preparations(preparation, 2, 0)
            selected = [
                track_id
                for worker, _rows, _tracks, _cost in shards
                for track_id in worker["classes"]["test"]["allowed_track_ids"]
            ]
            self.assertEqual(["1", "2", "3", "4"], sorted(selected))
            self.assertEqual([100, 300], sorted(value[1] for value in shards))
            costs = sorted(value[3] for value in shards)
            self.assertLessEqual(costs[1] / costs[0], 1.30)

    def test_classwise_curve_route_disables_nested_process_pool(self) -> None:
        pipeline = build_nested_pipeline(
            ClassPostprocessSettings("polygon", 3, 15),
            geometry_options={"optimizer_workers": 6},
            geometry_mode="catmull_rom",
            curve_cpu_threads=4,
            selected_track_ids=("1", "2"),
        )
        curve = pipeline.stages[0]
        self.assertEqual(1, curve.options["optimizer_workers"])

    def test_curve_track_partition_is_stable_and_balanced(self) -> None:
        tracks = ("1", "2", "3", "4", "5")
        counts = {"1": 100, "2": 80, "3": 40, "4": 30, "5": 20}
        first = partition_curve_tracks(tracks, counts, 2)
        second = partition_curve_tracks(tuple(reversed(tracks)), counts, 2)
        self.assertEqual(first, second)
        self.assertEqual(set(tracks), {value for shard in first for value in shard})
        self.assertEqual(len(tracks), sum(len(shard) for shard in first))
        loads = [sum(counts[value] for value in shard) for shard in first]
        self.assertLessEqual(max(loads) - min(loads), max(counts.values()))

    def test_curve_work_cost_falls_back_for_empty_tracking_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _tracked_sqlite(Path(temporary) / "tracked.sqlite")
            with sqlite3.connect(source) as connection:
                costs, metric = curve_track_costs(
                    connection,
                    {"1": 2, "2": 2},
                )
            self.assertEqual("source_bbox_area_sum", metric)
            # The canonical schema has the lineage table, but this minimal
            # fixture has no lineage rows.  Both tracks receive the same safe
            # non-zero fallback cost and remain deterministically balanced.
            self.assertEqual({"1": 2, "2": 2}, costs)

    def test_curve_cpu_budget_respects_process_affinity(self) -> None:
        self.assertGreaterEqual(available_cpu_count(), 1)
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            self.assertEqual(len(affinity(0)), available_cpu_count())

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
            progress_output = io.StringIO()
            with redirect_stdout(progress_output):
                manifest = run_pipeline(args)
            progress_events = [
                json.loads(line.split(" ", 1)[1])
                for line in progress_output.getvalue().splitlines()
                if line.startswith("[phase-progress] ")
            ]
            self.assertTrue(
                any(
                    str(event["detail"]).startswith("classwise:男性器:polygon:")
                    and event["stage_progress"] is not None
                    and not event["estimated"]
                    for event in progress_events
                )
            )
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
            self.assertEqual(
                {1, 2},
                {
                    int(group["settings"]["keyframe_interval"])
                    for group in classwise["groups"]
                },
            )
            self.assertEqual(2, classwise["execution"]["classwise_workers"])
            self.assertTrue(classwise["execution"]["parallel"])
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

    def test_curve_classes_share_source_but_emit_disjoint_routes(self) -> None:
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
                    "--class-postprocess-policy-json",
                    str(policy),
                    "--mask-geometry",
                    "catmull_rom",
                ]
            )
            with redirect_stdout(io.StringIO()):
                manifest = run_pipeline(args)
            classwise = json.loads(
                Path(manifest["artifacts"]["classwise_manifest"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("catmull_rom", classwise["execution"]["geometry_mode"])
            self.assertEqual("process_spawn", classwise["execution"]["worker_mode"])
            self.assertTrue(
                all(group["shared_read_only_source"] for group in classwise["groups"])
            )
            self.assertEqual(
                {str(source.resolve())},
                {group["tracked_input"] for group in classwise["groups"]},
            )
            self.assertFalse(
                any(
                    (
                        Path(manifest["artifacts"]["classwise_manifest"]).parent
                        / "groups"
                        / group["id"]
                        / "tracked.sqlite"
                    ).exists()
                    for group in classwise["groups"]
                )
            )
            with sqlite3.connect(manifest["artifacts"]["predictions_sqlite"]) as db:
                self.assertEqual(
                    [("1",), ("2",)],
                    db.execute(
                        "SELECT DISTINCT track_id FROM masks ORDER BY track_id"
                    ).fetchall(),
                )
            for group in classwise["groups"]:
                worker_progress = json.loads(
                    Path(group["worker_progress_json"]).read_text(encoding="utf-8")
                )
                self.assertEqual("complete", worker_progress["detail"])
                self.assertEqual(1.0, worker_progress["fraction"])
                self.assertNotEqual(os.getpid(), group["worker_pid"])
                nested = json.loads(
                    Path(group["pipeline_manifest"]).read_text(encoding="utf-8")
                )
                evaluation = next(
                    stage
                    for stage in nested["stages"]
                    if stage["id"] == "exact_evaluation"
                )["metadata"]
                self.assertEqual(
                    group["input_masks"], evaluation["row_count_reference"]
                )
                self.assertEqual(
                    group["output_masks"], evaluation["row_count_prediction"]
                )

            serial_manifest = PipelineRunner(
                PipelineConfig(
                    "curve_serial_parity",
                    (
                        StageSpec(
                            "classwise_postprocess",
                            "classwise.production",
                            {
                                "geometry_mode": "catmull_rom",
                                "geometry_options": {
                                    "parallel_workers": 1,
                                    "parallel_shards_per_class": 1,
                                    "native_cpu_threads": 1,
                                },
                            },
                        ),
                        StageSpec("output_validation", "artifacts.validate"),
                    ),
                ),
                root / "serial-output",
            ).run(
                {
                    "tracked_sqlite": source,
                    "class_postprocess_policy_json": policy,
                }
            )
            with sqlite3.connect(manifest["artifacts"]["predictions_sqlite"]) as left:
                process_rows = left.execute(
                    "SELECT frame,track_id,polygons,shape_type,label "
                    "FROM masks ORDER BY frame,track_id"
                ).fetchall()
                process_provenance = left.execute(
                    "SELECT * FROM mask_postprocess_provenance "
                    "ORDER BY frame,track_id"
                ).fetchall()
            with sqlite3.connect(
                serial_manifest["artifacts"]["predictions_sqlite"]
            ) as right:
                serial_rows = right.execute(
                    "SELECT frame,track_id,polygons,shape_type,label "
                    "FROM masks ORDER BY frame,track_id"
                ).fetchall()
                serial_provenance = right.execute(
                    "SELECT * FROM mask_postprocess_provenance "
                    "ORDER BY frame,track_id"
                ).fetchall()
            self.assertEqual(process_rows, serial_rows)
            self.assertEqual(process_provenance, serial_provenance)


if __name__ == "__main__":
    unittest.main()
