from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from common.registry import create_stage
from experimental.production_candidate_20260814 import CANDIDATE
from experimental.production_candidate_20260814.config import (
    with_interval_evaluation,
    with_target_interval,
)
from experimental.production_candidate_20260814.export import (
    ensure_empty_label_artifacts,
)
from experimental.production_candidate_20260814.nms.policy import build_policy
from experimental.production_candidate_20260814.polygon.candidate_palette import (
    role_ids,
)
from experimental.production_candidate_20260814.polygon.dp import audit_exact_recall
from experimental.production_candidate_20260814.polygon.engine import (
    assert_runtime_bridge_contract,
    run_polygon_optimizer,
)
from experimental.production_candidate_20260814.polygon.preparation import (
    _finalize_class_projection,
)
from experimental.production_candidate_20260814.validation import (
    audit_sqlite,
    compare_canonical_jsonl,
    compare_sqlite_tables,
    schema_fingerprint,
)


class ProductionCandidate20260814Tests(unittest.TestCase):
    def test_frozen_contract_and_policy_are_complete(self) -> None:
        CANDIDATE.validate()
        self.assertEqual("production_candidate_20260814_v1", CANDIDATE.profile_id)
        self.assertEqual(14, CANDIDATE.spatial.vertices_per_component)
        self.assertFalse(hasattr(CANDIDATE.spatial, "vertex_fallbacks"))
        self.assertEqual(1.05, CANDIDATE.spatial.recall_repair_max_scale)
        self.assertEqual(0.97, CANDIDATE.temporal.recall_floor)
        self.assertEqual(6, CANDIDATE.temporal.target_interval)
        self.assertEqual(2, CANDIDATE.temporal.pair_vote_sweeps)
        self.assertEqual(8, CANDIDATE.runtime.native_batch_threads)
        self.assertEqual(8, CANDIDATE.runtime.pair_vote_threads)
        self.assertEqual(0.5, CANDIDATE.runtime.lazy_fallback_min_seconds)
        self.assertEqual(0.10, CANDIDATE.runtime.cuda_prefilter_deficit_budget)
        self.assertEqual(0.0, CANDIDATE.runtime.cuda_prefilter_small_area)
        self.assertEqual(
            0.10, CANDIDATE.runtime.cuda_prefilter_small_deficit_budget
        )
        self.assertEqual(1024, CANDIDATE.runtime.lazy_fallback_min_exact_edges)
        self.assertEqual(0.875, CANDIDATE.runtime.lazy_fallback_infeasible_ratio)
        self.assertEqual(1, CANDIDATE.runtime.candidate_frame_workers)
        self.assertEqual("cuda_lazy_exact", CANDIDATE.runtime.interval_evaluation)
        policy = build_policy()
        self.assertEqual("adaptive_mask", policy.comparison_policy)
        self.assertEqual(0.20, policy.mask_iou_threshold)
        self.assertEqual(0.10, policy.mask_small_iou_threshold)
        self.assertEqual(0.05, policy.mask_tiny_iou_threshold)
        self.assertEqual(5000.0, policy.mask_small_area)
        self.assertEqual(2000.0, policy.mask_tiny_area)
        self.assertEqual(0.80, policy.mask_containment_coverage_min)
        self.assertEqual(8.0, policy.mask_contain_ratio_max)
        self.assertEqual(5.0, policy.mask_small_contain_ratio_max)
        self.assertEqual(5.0, policy.mask_tiny_contain_ratio_max)

    def test_role_palette_and_runtime_bridge_match(self) -> None:
        self.assertEqual(
            (
                "C02_125",
                "G02",
                "G04",
                "A06",
                "F3_P1",
                "D6_P1",
                "F3_Q75_P1",
            ),
            role_ids("女性器"),
        )
        self.assertEqual("D6_R5_P1", role_ids("男性器")[-1])
        self.assertEqual("VF8_P1", role_ids("結合部分")[-1])
        assert_runtime_bridge_contract()

    def test_interval_specific_role_palettes_match_runtime(self) -> None:
        for interval in range(1, 7):
            config = with_target_interval(interval)
            assert_runtime_bridge_contract(config)
            self.assertEqual(interval, config.temporal.target_interval)
        self.assertEqual(6, len(role_ids("女性器", 1, with_target_interval(1))))
        self.assertEqual(7, len(role_ids("女性器", 2, with_target_interval(2))))
        self.assertEqual(6, len(role_ids("結合部分", 3, with_target_interval(3))))
        self.assertEqual(7, len(role_ids("結合部分", 4, with_target_interval(4))))

    def test_interval_evaluator_can_switch_without_semantic_drift(self) -> None:
        exact = with_interval_evaluation(
            "native_exact",
            with_target_interval(2),
        )
        self.assertEqual("native_exact", exact.runtime.interval_evaluation)
        self.assertEqual(2, exact.temporal.target_interval)
        self.assertEqual(CANDIDATE.nms, exact.nms)
        self.assertEqual(CANDIDATE.spatial, exact.spatial)
        self.assertEqual(CANDIDATE.temporal.recall_floor, exact.temporal.recall_floor)
        assert_runtime_bridge_contract(exact)
        with self.assertRaises(ValueError):
            with_interval_evaluation("not_an_evaluator")

    def test_external_stages_are_constructible_without_registry_changes(self) -> None:
        nms = create_stage(
            "experimental.production_candidate_20260814.nms.stage:CandidateNmsStage"
        )
        self.assertEqual(frozenset({"scored_jsonl"}), nms.requires)
        polygon = create_stage(
            "experimental.production_candidate_20260814.polygon.stage:CandidatePolygonStage",
            {"width": 1920, "height": 1080},
        )
        self.assertEqual(frozenset({"tracked_sqlite", "input_video"}), polygon.requires)

    def test_exact_recall_csv_is_the_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exact.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("recall",))
                writer.writeheader()
                writer.writerow({"recall": "0.9700001"})
                writer.writerow({"recall": "1.0"})
            audit = audit_exact_recall(path)
            self.assertEqual(2, audit["evaluated_rows"])
            self.assertEqual(0, audit["recall_violations"])
            self.assertAlmostEqual(0.9700001, audit["minimum_recall"])

    def test_class_projection_discards_other_raw_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "class.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE masks(frame INTEGER, track_id TEXT)")
                connection.execute("CREATE TABLE tracks(track_id TEXT)")
                connection.execute(
                    "CREATE TABLE raw_tracked_masks(final_label TEXT, value TEXT)"
                )
                connection.execute(
                    "CREATE TABLE raw_tracks(final_label TEXT, value TEXT)"
                )
                connection.execute("INSERT INTO masks VALUES (0, '1')")
                connection.execute("INSERT INTO tracks VALUES ('1')")
                for table in ("raw_tracked_masks", "raw_tracks"):
                    connection.executemany(
                        f"INSERT INTO {table} VALUES (?, ?)",
                        (("女性器", "keep"), ("男性器", "drop"), (None, "drop")),
                    )
            result = _finalize_class_projection(path, "女性器")
            self.assertEqual(1, result["counts"]["raw_tracked_masks"])
            self.assertEqual(1, result["counts"]["raw_tracks"])
            with sqlite3.connect(path) as connection:
                self.assertEqual(
                    [("女性器", "keep")],
                    connection.execute("SELECT * FROM raw_tracked_masks").fetchall(),
                )

    def test_export_contract_supports_absent_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracked = root / "tracked.sqlite"
            with sqlite3.connect(tracked) as connection:
                connection.execute("CREATE TABLE tracks(track_id TEXT, label TEXT)")
                connection.execute("INSERT INTO tracks VALUES ('1', '女性器')")
            created = ensure_empty_label_artifacts(root / "phase2", tracked)
            self.assertEqual(["男性器", "結合部分"], [row["label"] for row in created])
            for label in ("男性器", "結合部分"):
                label_root = root / "phase2" / CANDIDATE.polygon_profile_id / label
                with sqlite3.connect(
                    label_root / "runtime/pred/predictions.sqlite"
                ) as connection:
                    self.assertEqual(
                        0,
                        connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                    )
                self.assertEqual(
                    [],
                    json.loads(
                        (label_root / "runtime/opt/final_keyframes.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

    def test_optimizer_handles_no_active_genital_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_polygon_optimizer(
                root / "empty_source",
                root / "optimizer",
                labels=(),
            )
            self.assertEqual([], result["command"])
            self.assertEqual([], result["active_labels"])
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_parity_helpers_compare_semantics_not_file_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_jsonl = root / "reference.jsonl"
            candidate_jsonl = root / "candidate.jsonl"
            reference_jsonl.write_text(
                '{"frame_index":0,"detections":[],"meta":{"a":1,"b":2}}\n',
                encoding="utf-8",
            )
            candidate_jsonl.write_text(
                '{ "meta": {"b":2,"a":1}, "detections":[], "frame_index":0 }\n',
                encoding="utf-8",
            )
            self.assertTrue(
                compare_canonical_jsonl(reference_jsonl, candidate_jsonl).equal
            )

            reference_sqlite = root / "reference.sqlite"
            candidate_sqlite = root / "candidate.sqlite"
            for path, rows in (
                (reference_sqlite, ((2, 2.0), (1, 1.0))),
                (candidate_sqlite, ((1, 1.0000005), (2, 2.0))),
            ):
                with sqlite3.connect(path) as connection:
                    connection.execute(
                        "CREATE TABLE values_table(id INTEGER PRIMARY KEY, value REAL)"
                    )
                    connection.executemany(
                        "INSERT INTO values_table(id,value) VALUES (?,?)", rows
                    )
            self.assertEqual(
                schema_fingerprint(reference_sqlite),
                schema_fingerprint(candidate_sqlite),
            )
            self.assertTrue(audit_sqlite(candidate_sqlite).ok)
            self.assertTrue(
                compare_sqlite_tables(
                    reference_sqlite,
                    candidate_sqlite,
                    tables=("values_table",),
                ).equal
            )


if __name__ == "__main__":
    unittest.main()
