from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from production.polygon.runtime.candidate_config import CANDIDATE
from production.polygon.runtime.candidate_config import (
    legacy_fixed14_candidate,
    with_interval_evaluation,
    with_target_interval,
)
from nms.component_virtual import ProductionVirtualComponentNms
from nms.production import PRODUCTION_OPTIONS
from production.polygon.runtime.candidate_palette import (
    role_ids,
)
from production.polygon.runtime.dp import audit_exact_recall
from production.polygon.runtime.engine import (
    assert_runtime_bridge_contract,
    run_polygon_optimizer,
)
from production.polygon.preparation import (
    _finalize_class_projection,
)
from production.polygon.vertex_policy import (
    build_vertex_policy,
    select_vertex_count,
)
from contracts.mask_sqlite import MaskRow, write_mask_sqlite


class ProductionCandidate20260814Tests(unittest.TestCase):
    def test_frozen_contract_and_policy_are_complete(self) -> None:
        CANDIDATE.validate()
        self.assertEqual(
            "production_candidate_adaptive_vertices_v2", CANDIDATE.profile_id
        )
        self.assertEqual(
            (14, 16, 18, 20),
            CANDIDATE.spatial.allowed_vertices_per_component,
        )
        self.assertEqual(0.999, CANDIDATE.spatial.track_area_quantile)
        self.assertEqual(
            (0.03, 0.10, 0.25),
            CANDIDATE.spatial.screen_occupancy_thresholds,
        )
        self.assertEqual(16.0, CANDIDATE.preparation.border_max_expand_px)
        self.assertEqual(16.0, CANDIDATE.preparation.border_influence_px)
        self.assertTrue(CANDIDATE.preparation.border_corner_support)
        self.assertEqual(1.05, CANDIDATE.spatial.recall_repair_max_scale)
        self.assertEqual(0.97, CANDIDATE.temporal.recall_floor)
        self.assertEqual(6, CANDIDATE.temporal.target_interval)
        self.assertEqual(2, CANDIDATE.temporal.pair_vote_sweeps)
        self.assertEqual(8, CANDIDATE.runtime.native_batch_threads)
        self.assertEqual(8, CANDIDATE.runtime.pair_vote_threads)
        self.assertEqual(0.5, CANDIDATE.runtime.lazy_fallback_min_seconds)
        self.assertEqual(0.10, CANDIDATE.runtime.cuda_prefilter_deficit_budget)
        self.assertEqual(0.0, CANDIDATE.runtime.cuda_prefilter_small_area)
        self.assertEqual(0.10, CANDIDATE.runtime.cuda_prefilter_small_deficit_budget)
        self.assertEqual(1024, CANDIDATE.runtime.lazy_fallback_min_exact_edges)
        self.assertEqual(0.875, CANDIDATE.runtime.lazy_fallback_infeasible_ratio)
        self.assertEqual(1, CANDIDATE.runtime.candidate_frame_workers)
        self.assertEqual("native_exact", CANDIDATE.runtime.interval_evaluation)
        policy = ProductionVirtualComponentNms(**PRODUCTION_OPTIONS)
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

    def test_vertex_policy_uses_strict_screen_occupancy_thresholds(self) -> None:
        self.assertEqual(14, select_vertex_count(0.03))
        self.assertEqual(16, select_vertex_count(0.030000001))
        self.assertEqual(16, select_vertex_count(0.10))
        self.assertEqual(18, select_vertex_count(0.100000001))
        self.assertEqual(18, select_vertex_count(0.25))
        self.assertEqual(20, select_vertex_count(0.250000001))

    def test_vertex_policy_is_track_fixed_and_pre_border(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mask_sqlite(
                root / "tracked.sqlite",
                [
                    MaskRow(
                        0,
                        "1",
                        json.dumps([[[0, 0], [30, 0], [30, 10], [0, 10]]]),
                        "女性器",
                        "polygon",
                    ),
                    MaskRow(
                        1,
                        "1",
                        json.dumps([[[0, 0], [40, 0], [40, 10], [0, 10]]]),
                        "女性器",
                        "polygon",
                    ),
                    MaskRow(
                        0,
                        "2",
                        json.dumps([[[0, 0], [40, 0], [40, 40], [0, 40]]]),
                        "男性器",
                        "polygon",
                    ),
                ],
            )
            output = root / "vertex_policy.json"
            payload = build_vertex_policy(
                source,
                output,
                width=100,
                height=100,
                track_labels={"1": "女性器", "2": "男性器"},
            )
            self.assertEqual("tracked_pre_border", payload["source_stage"])
            self.assertEqual(16, payload["tracks"]["1"]["vertices_per_component"])
            self.assertEqual(18, payload["tracks"]["2"]["vertices_per_component"])
            self.assertTrue(output.is_file())

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

    def test_legacy_fixed14_profile_remains_reproducible(self) -> None:
        legacy = legacy_fixed14_candidate()
        self.assertEqual("production_candidate_20260814_v1", legacy.profile_id)
        self.assertEqual("polygon14_keyframe_v1", legacy.polygon_profile_id)
        self.assertFalse(legacy.spatial.adaptive_vertex_policy)
        self.assertEqual(14, legacy.spatial.vertices_per_component)
        self.assertEqual(40.0, legacy.preparation.border_max_expand_px)
        self.assertEqual(24.0, legacy.preparation.border_influence_px)
        self.assertFalse(legacy.preparation.border_corner_support)
        assert_runtime_bridge_contract(legacy)

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

if __name__ == "__main__":
    unittest.main()
