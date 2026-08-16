from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from contracts.detections import CutList, dumps_json_line, write_cut_list
from tracking.association import detection_features
from tracking.builder import build_tracked_sqlite
from tracking.records import prepare_detection

from nms.adaptive import AdaptiveNms
from nms.component_aware import ComponentAwareMaskNms
from nms.component_virtual import VirtualComponentMaskNms, VirtualComponentNms
from nms.mask_adaptive import AdaptiveMaskNms
from nms.components import (
    fill_holes_and_remove_tiny_islands,
    remove_redundant_islands_candidate_v1,
    remove_redundant_surviving_islands,
    remove_small_foreground_components,
)


def polygon(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def detection(
    polygons: list[list[list[float]]], score: float, detection_id: int
) -> dict[str, object]:
    xs = [point[0] for item in polygons for point in item]
    ys = [point[1] for item in polygons for point in item]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "source_detection_id": detection_id,
        "score": score,
        "bbox_xyxy": box,
        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
        "polygons": polygons,
        "segmentation": polygons,
    }


class SmallIslandCleanupTests(unittest.TestCase):
    def test_removes_small_foreground_but_preserves_hole(self) -> None:
        outer = polygon(40, 0, 80, 40)
        hole = polygon(50, 10, 60, 20)
        island = polygon(19, 10, 21, 12)
        source = detection([outer, hole, island], 0.9, 1)

        cleaned = remove_small_foreground_components(source, ratio_max=0.10)

        self.assertEqual([outer, hole], cleaned["polygons"])
        self.assertEqual([40.0, 0.0, 80.0, 40.0], cleaned["bbox_xyxy"])
        self.assertEqual([outer, hole, island], source["polygons"])

    def test_keeps_second_foreground_larger_than_ten_percent(self) -> None:
        main = polygon(0, 0, 10, 10)
        secondary = polygon(20, 0, 24, 4)  # 16% of the main component.
        source = detection([main, secondary], 0.9, 1)
        self.assertIs(
            source,
            remove_small_foreground_components(source, ratio_max=0.10),
        )

    def test_cleanup_prevents_tiny_island_from_suppressing_distinct_mask(self) -> None:
        main = polygon(40, 0, 80, 40)
        tiny_island = polygon(19, 10, 21, 12)
        local = polygon(0, 0, 25, 25)
        merged = detection([main, tiny_island], 0.9, 1)
        distinct = detection([local], 0.8, 2)

        without_cleanup = AdaptiveNms(remove_small_islands=False).apply(
            [merged, distinct]
        )
        with_cleanup = AdaptiveNms(
            remove_small_islands=True, small_island_ratio_max=0.10
        ).apply([merged, distinct])

        self.assertEqual([1], [item["source_detection_id"] for item in without_cleanup])
        self.assertEqual([1, 2], [item["source_detection_id"] for item in with_cleanup])
        self.assertEqual(1, len(with_cleanup[0]["polygons"]))

    def test_candidate_removes_one_percent_island_unconditionally(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 125, 10)  # 0.5% of the main component.
        source = detection([main, island], 0.9, 1)

        cleaned = remove_redundant_islands_candidate_v1([source])

        self.assertEqual([main], cleaned[0]["polygons"])
        self.assertEqual([main, island], source["polygons"])

    def test_candidate_fills_all_holes_unconditionally(self) -> None:
        outer = polygon(0, 0, 100, 100)
        hole = polygon(25, 25, 75, 75)
        source = detection([outer, hole], 0.9, 1)

        cleaned = remove_redundant_islands_candidate_v1([source])

        self.assertEqual([outer], cleaned[0]["polygons"])
        self.assertEqual([outer, hole], source["polygons"])

    def test_candidate_removes_component_redundant_with_other_instance(self) -> None:
        main = polygon(0, 0, 100, 100)
        # 6.25% of the owner main, but only 25% of the covering instance.
        island = polygon(120, 0, 145, 25)
        covering = polygon(115, -5, 165, 45)
        owner = detection([main, island], 0.9, 1)
        other = detection([covering], 0.8, 2)

        cleaned = remove_redundant_islands_candidate_v1([owner, other])

        self.assertEqual([main], cleaned[0]["polygons"])
        self.assertEqual([covering], cleaned[1]["polygons"])

    def test_candidate_keeps_component_larger_than_thirty_percent_of_other(
        self,
    ) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 145, 25)
        covering = polygon(118, -2, 148, 28)  # Island is 69.4% of other.
        owner = detection([main, island], 0.9, 1)
        other = detection([covering], 0.8, 2)

        cleaned = remove_redundant_islands_candidate_v1([owner, other])

        self.assertEqual([main, island], cleaned[0]["polygons"])

    def test_candidate_keeps_component_with_less_than_ninety_percent_coverage(
        self,
    ) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 145, 25)
        # Covers 80% of the island, while being sufficiently large.
        covering = polygon(120, 0, 140, 40)
        owner = detection([main, island], 0.9, 1)
        other = detection([covering], 0.8, 2)

        cleaned = remove_redundant_islands_candidate_v1([owner, other])

        self.assertEqual([main, island], cleaned[0]["polygons"])

    def test_candidate_is_available_but_not_enabled_by_default(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 145, 25)
        covering = polygon(115, -5, 165, 45)
        owner = detection([main, island], 0.9, 1)
        other = detection([covering], 0.8, 2)

        default_result = AdaptiveNms().apply([owner, other])
        candidate_result = AdaptiveNms(
            island_cleanup_policy="production_candidate_v1"
        ).apply([owner, other])

        self.assertEqual(2, len(default_result[0]["polygons"]))
        self.assertEqual(1, len(candidate_result[0]["polygons"]))


class ComponentAwareMaskNmsTests(unittest.TestCase):
    def test_preprocess_fills_holes_and_removes_exactly_one_percent(self) -> None:
        main = polygon(0, 0, 100, 100)
        hole = polygon(20, 20, 30, 30)
        one_percent = polygon(120, 0, 130, 10)
        above_one_percent = polygon(140, 0, 151, 10)
        source = detection([main, hole, one_percent, above_one_percent], 0.9, 1)
        source["custom_metadata"] = "preserved"

        cleaned, stats = fill_holes_and_remove_tiny_islands([source])

        self.assertEqual([main, above_one_percent], cleaned[0]["polygons"])
        self.assertEqual(cleaned[0]["polygons"], cleaned[0]["segmentation"])
        self.assertEqual([0.0, 0.0, 151.0, 100.0], cleaned[0]["bbox_xyxy"])
        self.assertEqual("preserved", cleaned[0]["custom_metadata"])
        self.assertEqual(1, stats.holes_filled)
        self.assertEqual(1, stats.tiny_islands_removed)
        self.assertEqual(
            [main, hole, one_percent, above_one_percent], source["polygons"]
        )

    def test_full_mask_iou_suppresses_lower_score_instance(self) -> None:
        main = polygon(0, 0, 100, 100)
        small_difference = polygon(120, 0, 135, 20)  # 3% of main.
        winner = detection([main], 0.9, 1)
        loser = detection([main, small_difference], 0.8, 2)

        result, diagnostics = ComponentAwareMaskNms().apply_with_diagnostics(
            [winner, loser]
        )

        self.assertEqual([1], [item["source_detection_id"] for item in result])
        self.assertEqual(1, diagnostics.nms_suppressed)

    def test_same_score_is_stable_by_input_order(self) -> None:
        shape = polygon(0, 0, 100, 100)
        first = detection([shape], 0.9, 7)
        second = detection([shape], 0.9, 8)
        result = ComponentAwareMaskNms().apply([first, second])
        self.assertEqual([7], [item["source_detection_id"] for item in result])

    def test_bbox_overlap_without_mask_overlap_does_not_suppress(self) -> None:
        first = detection([polygon(0, 0, 20, 20), polygon(80, 80, 100, 100)], 0.9, 1)
        second = detection([polygon(80, 0, 100, 20), polygon(0, 80, 20, 100)], 0.8, 2)
        result = ComponentAwareMaskNms().apply([first, second])
        self.assertEqual([1, 2], [item["source_detection_id"] for item in result])

    def test_redundant_island_is_removed_only_after_both_instances_survive(
        self,
    ) -> None:
        owner_main = polygon(0, 0, 100, 100)
        owner_island = polygon(120, 0, 140, 20)
        covering_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, owner_island], 0.9, 1)
        covering = detection([covering_main], 0.8, 2)

        result, diagnostics = ComponentAwareMaskNms().apply_with_diagnostics(
            [owner, covering]
        )

        self.assertEqual([1, 2], [item["source_detection_id"] for item in result])
        self.assertEqual([owner_main], result[0]["polygons"])
        self.assertEqual([covering_main], result[1]["polygons"])
        self.assertEqual([0.0, 0.0, 140.0, 100.0], result[0]["_association_bbox_xyxy"])
        self.assertEqual(10400.0, result[0]["_association_mask_area"])
        self.assertEqual(1, diagnostics.redundant_islands_removed)

    def test_tracking_uses_pre_survivor_cleanup_association_geometry(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        owner_island = polygon(120, 0, 140, 20)
        covering_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, owner_island], 0.9, 1)
        covering = detection([covering_main], 0.8, 2)

        cleaned = ComponentAwareMaskNms().apply([owner, covering])[0]
        prepared = prepare_detection(cleaned)
        features = detection_features(prepared)

        self.assertEqual([0.0, 0.0, 100.0, 100.0], prepared["bbox_xyxy"])
        self.assertEqual((0.0, 0.0, 140.0, 100.0), features.bbox)
        self.assertEqual(10400.0, features.polygon_area)
        self.assertAlmostEqual(10400.0 / 14000.0, features.fill_ratio)

    def test_malformed_association_geometry_is_rejected(self) -> None:
        source = detection([polygon(0, 0, 10, 10)], 0.9, 1)
        source["_association_bbox_xyxy"] = [0, 0, float("nan"), 10]
        source["_association_mask_area"] = 100
        with self.assertRaisesRegex(ValueError, "must be finite"):
            prepare_detection(source)

    def test_association_geometry_is_not_persisted_to_tracking_sqlite(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        owner_island = polygon(120, 0, 140, 20)
        covering_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, owner_island], 0.9, 1)
        covering = detection([covering_main], 0.8, 2)
        retained = ComponentAwareMaskNms().apply([owner, covering])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "candidate.jsonl"
            jsonl.write_bytes(
                dumps_json_line({"frame_index": 0, "detections": retained})
            )
            cuts = write_cut_list(
                root / "cuts.json",
                CutList(frames=(), method="unit-test"),
            )
            output = root / "tracked.sqlite"
            build_tracked_sqlite(
                jsonl,
                output,
                cuts,
                remove_short_tracks_max_frames=0,
            )

            with sqlite3.connect(output) as connection:
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(raw_tracked_masks)"
                    )
                }
                row = connection.execute(
                    """
                    SELECT polygons, bbox_xyxy_json
                    FROM raw_tracked_masks
                    WHERE source_detection_id=1
                    """
                ).fetchone()

        self.assertNotIn("_association_bbox_xyxy", columns)
        self.assertNotIn("_association_mask_area", columns)
        self.assertEqual([owner_main], json.loads(row[0]))
        self.assertEqual([0.0, 0.0, 100.0, 100.0], json.loads(row[1]))

    def test_component_coverage_below_threshold_keeps_island(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        # Clearly below 80% coverage but large enough for the area-ratio gate.
        partial = polygon(120, 0, 133, 40)
        owner = detection([main, island], 0.9, 1)
        other = detection([partial], 0.8, 2)
        result, stats = remove_redundant_surviving_islands([owner, other])
        self.assertEqual([main, island], result[0]["polygons"])
        self.assertEqual(0, stats.redundant_islands_removed)

    def test_component_larger_than_half_other_main_is_kept(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        smaller_cover = polygon(118, -2, 142, 22)
        owner = detection([main, island], 0.9, 1)
        other = detection([smaller_cover], 0.8, 2)
        result, stats = remove_redundant_surviving_islands([owner, other])
        self.assertEqual([main, island], result[0]["polygons"])
        self.assertEqual(0, stats.redundant_islands_removed)

    def test_island_is_not_compared_to_another_island(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        owner_island = polygon(120, 0, 140, 20)
        other_main = polygon(200, 0, 240, 40)
        other_island = polygon(115, -5, 155, 35)
        owner = detection([owner_main, owner_island], 0.9, 1)
        other = detection([other_main, other_island], 0.8, 2)
        result, stats = remove_redundant_surviving_islands([owner, other])
        self.assertEqual([owner_main, owner_island], result[0]["polygons"])
        self.assertEqual(0, stats.redundant_islands_removed)

    def test_component_cleanup_is_order_independent(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        cover = polygon(115, -5, 155, 35)
        owner = detection([main, island], 0.9, 1)
        other = detection([cover], 0.8, 2)
        forward = ComponentAwareMaskNms().apply([owner, other])
        reverse = ComponentAwareMaskNms().apply([other, owner])
        by_id_forward = {
            item["source_detection_id"]: item["polygons"] for item in forward
        }
        by_id_reverse = {
            item["source_detection_id"]: item["polygons"] for item in reverse
        }
        self.assertEqual(by_id_forward, by_id_reverse)

    def test_crossing_contour_is_not_misclassified_as_hole(self) -> None:
        outer = polygon(0, 0, 100, 100)
        # Its first point lies inside outer, but the contour crosses outside.
        crossing = [[90, 40], [120, 40], [120, 60], [90, 60]]
        source = detection([outer, crossing], 0.9, 1)
        cleaned, stats = fill_holes_and_remove_tiny_islands(
            [source], unconditional_owner_ratio_max=0.0
        )
        self.assertEqual([outer, crossing], cleaned[0]["polygons"])
        self.assertEqual(0, stats.holes_filled)

    def test_contour_crossing_concave_parent_is_not_a_hole(self) -> None:
        concave_parent = [
            [0, 0],
            [100, 0],
            [100, 100],
            [70, 100],
            [70, 30],
            [30, 30],
            [30, 100],
            [0, 100],
        ]
        # Every vertex is inside one of the U-shape's arms, but the horizontal
        # edges cross the open center and therefore are not fully contained.
        crossing_child = polygon(20, 80, 80, 90)
        source = detection([concave_parent, crossing_child], 0.9, 1)
        cleaned, stats = fill_holes_and_remove_tiny_islands(
            [source], unconditional_owner_ratio_max=0.0
        )
        self.assertEqual([concave_parent, crossing_child], cleaned[0]["polygons"])
        self.assertEqual(0, stats.holes_filled)


class VirtualComponentNmsTests(unittest.TestCase):
    def test_single_component_main_preserves_legacy_bbox_at_threshold(self) -> None:
        # Regression for V3 HEYZO-3545 frame 14379. The canonical detection
        # bbox includes the raster convention and is slightly wider than the
        # contour vertex bounds. Recomputing it moved bbox IoU from 0.1043 to
        # 0.0991 and incorrectly crossed the legacy 0.10 threshold.
        outer = detection([polygon(819.5, 304.5, 1016.5, 528.5)], 0.792, 7809)
        outer["bbox_xyxy"] = [817.0, 303.0, 1018.0, 529.5]
        outer["bbox"] = [817.0, 303.0, 201.0, 226.5]
        inner = detection([polygon(818.5, 468.5, 892.5, 528.5)], 0.350, 7810)
        inner["bbox_xyxy"] = [817.0, 467.0, 893.0, 529.5]
        inner["bbox"] = [817.0, 467.0, 76.0, 62.5]

        legacy_ids = [
            item["source_detection_id"] for item in AdaptiveNms().apply([outer, inner])
        ]
        virtual_ids = [
            item["source_detection_id"]
            for item in VirtualComponentNms().apply([outer, inner])
        ]

        self.assertEqual([7809], legacy_ids)
        self.assertEqual(legacy_ids, virtual_ids)

    def test_main_vs_main_reuses_legacy_nms_and_drops_losing_owner(self) -> None:
        high = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        low = detection([polygon(0, 0, 100, 100), polygon(150, 0, 170, 20)], 0.8, 2)

        retained, stats, trace = VirtualComponentNms().apply_with_trace([high, low])

        self.assertEqual([1], [item["source_detection_id"] for item in retained])
        self.assertEqual(1, stats.main_owners_suppressed)
        self.assertEqual("main_main_legacy_nms", trace[0]["reason"])

    def test_island_vs_island_removes_only_lower_score_island(self) -> None:
        first_main = polygon(0, 0, 100, 100)
        shared_island = polygon(120, 0, 140, 20)
        second_main = polygon(200, 0, 300, 100)
        first = detection([first_main, shared_island], 0.9, 1)
        second = detection([second_main, shared_island], 0.8, 2)

        retained, stats, trace = VirtualComponentNms().apply_with_trace([first, second])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual([first_main, shared_island], by_id[1]["polygons"])
        self.assertEqual([second_main], by_id[2]["polygons"])
        self.assertEqual(1, stats.island_island_suppressed)
        self.assertTrue(
            any(item["reason"] == "island_island_legacy_nms" for item in trace)
        )

    def test_island_vs_island_score_tie_is_stable_by_input_order(self) -> None:
        shared_island = polygon(120, 0, 140, 20)
        first = detection([polygon(0, 0, 100, 100), shared_island], 0.8, 11)
        second = detection([polygon(200, 0, 300, 100), shared_island], 0.8, 12)

        retained = VirtualComponentNms().apply([first, second])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual(2, len(by_id[11]["polygons"]))
        self.assertEqual(1, len(by_id[12]["polygons"]))

    def test_high_score_island_cannot_delete_lower_score_main(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, island], 0.95, 1)
        other = detection([other_main], 0.40, 2)

        retained, stats, trace = VirtualComponentNms().apply_with_trace([owner, other])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual({1, 2}, set(by_id))
        self.assertEqual([owner_main], by_id[1]["polygons"])
        self.assertEqual([other_main], by_id[2]["polygons"])
        self.assertEqual(1, stats.island_main_suppressed)
        self.assertTrue(
            any(item["reason"] == "island_subordinate_to_main" for item in trace)
        )

    def test_island_main_rule_is_asymmetric_in_score(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, island], 0.20, 1)
        other = detection([other_main], 0.95, 2)

        by_id = {
            item["source_detection_id"]: item
            for item in VirtualComponentNms().apply([owner, other])
        }

        self.assertEqual([owner_main], by_id[1]["polygons"])
        self.assertEqual([other_main], by_id[2]["polygons"])

    def test_island_main_keeps_component_outside_eighty_fifty_gate(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        partial = polygon(120, 0, 133, 40)
        owner = detection([owner_main, island], 0.9, 1)
        other = detection([partial], 0.8, 2)

        retained, stats = VirtualComponentNms().apply_with_diagnostics([owner, other])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual([owner_main, island], by_id[1]["polygons"])
        self.assertEqual(0, stats.island_main_suppressed)

    def test_island_main_includes_exact_fifty_percent_area_boundary(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        # Island area=400, other main area=800: the 50% gate is inclusive.
        other_main = polygon(120, 0, 140, 40)
        owner = detection([owner_main, island], 0.9, 1)
        other = detection([other_main], 0.8, 2)

        retained, stats = VirtualComponentNms().apply_with_diagnostics([owner, other])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual([owner_main], by_id[1]["polygons"])
        self.assertEqual(1, stats.island_main_suppressed)

    def test_island_main_keeps_component_just_over_fifty_percent_area(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        # Island area=400, other main area=760: ratio is about 52.6%.
        other_main = polygon(120, 0, 139, 40)
        owner = detection([owner_main, island], 0.9, 1)
        other = detection([other_main], 0.8, 2)

        retained, stats = VirtualComponentNms().apply_with_diagnostics([owner, other])
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual([owner_main, island], by_id[1]["polygons"])
        self.assertEqual(0, stats.island_main_suppressed)

    def test_same_owner_components_are_never_compared(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(90, 40, 110, 60)
        source = detection([main, island], 0.9, 1)

        retained, stats = VirtualComponentNms(
            unconditional_owner_ratio_max=0.0
        ).apply_with_diagnostics([source])

        self.assertEqual([main, island], retained[0]["polygons"])
        self.assertEqual(0, stats.main_main_pairs)
        self.assertEqual(0, stats.island_island_pairs)
        self.assertEqual(0, stats.island_main_pairs)

    def test_island_cleanup_keeps_tracking_geometry_private(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, island], 0.9, 1)
        other = detection([other_main], 0.8, 2)

        by_id = {
            item["source_detection_id"]: item
            for item in VirtualComponentNms().apply([owner, other])
        }

        self.assertEqual([0.0, 0.0, 140.0, 100.0], by_id[1]["_association_bbox_xyxy"])
        self.assertEqual([0.0, 0.0, 100.0, 100.0], by_id[1]["bbox_xyxy"])
        self.assertEqual([owner_main, island], owner["polygons"])

    def test_hole_fill_preserves_raw_tracking_geometry(self) -> None:
        outer = polygon(0, 0, 100, 100)
        hole = polygon(20, 20, 80, 80)
        source = detection([outer, hole], 0.9, 1)

        result = VirtualComponentNms().apply([source])[0]

        self.assertEqual([outer], result["polygons"])
        self.assertEqual([0.0, 0.0, 100.0, 100.0], result["_association_bbox_xyxy"])
        # Tracking historically adds absolute contour areas, including holes.
        self.assertEqual(13600.0, result["_association_mask_area"])

    def test_tiny_island_cleanup_preserves_raw_tracking_geometry(self) -> None:
        main = polygon(0, 0, 100, 100)
        tiny = polygon(120, 0, 125, 5)
        source = detection([main, tiny], 0.9, 1)

        result = VirtualComponentNms().apply([source])[0]

        self.assertEqual([main], result["polygons"])
        self.assertEqual([0.0, 0.0, 125.0, 100.0], result["_association_bbox_xyxy"])
        self.assertEqual(10025.0, result["_association_mask_area"])


class AdaptiveMaskNmsTests(unittest.TestCase):
    def test_disjoint_masks_with_overlapping_bboxes_are_retained(self) -> None:
        l_shape = [
            [0, 0],
            [100, 0],
            [100, 20],
            [20, 20],
            [20, 100],
            [0, 100],
        ]
        cavity = polygon(30, 30, 60, 60)
        high = detection([l_shape], 0.9, 1)
        low = detection([cavity], 0.8, 2)

        retained = AdaptiveMaskNms().apply([high, low])

        self.assertEqual([1, 2], [item["source_detection_id"] for item in retained])

    def test_subordinate_coverage_catches_low_symmetric_iou(self) -> None:
        large = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        small = detection([polygon(10, 10, 60, 60)], 0.8, 2)

        retained = AdaptiveMaskNms().apply([large, small])

        self.assertEqual([1], [item["source_detection_id"] for item in retained])

    def test_similar_size_overlap_uses_mask_iou(self) -> None:
        first = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        second = detection([polygon(20, 0, 120, 100)], 0.8, 2)

        retained = AdaptiveMaskNms().apply([first, second])

        self.assertEqual([1], [item["source_detection_id"] for item in retained])

    def test_adaptive_band_uses_production_contour_area_not_raster_area(self) -> None:
        # The 44x44 contour has continuous area 1,936 (tiny band), while
        # native inclusive rasterization occupies 45x45=2,025 pixels (small
        # band).  Production's historical band is the former.  The overlap is
        # between the tiny 0.05 and small 0.10 IoU thresholds and the size
        # ratio is too large for the containment shortcut.
        large = detection([polygon(0, 0, 140, 140)], 0.9, 1)
        small = detection([polygon(-4, 100, 40, 144)], 0.8, 2)
        policy = AdaptiveMaskNms()
        metrics = policy.pair_metrics(large, small)

        self.assertLess(policy.pair_threshold_area(large, small), 2000.0)
        self.assertGreater(min(metrics.first_area, metrics.second_area), 2000)
        self.assertGreater(metrics.iou, 0.05)
        self.assertLess(metrics.iou, 0.10)
        self.assertEqual(
            [1],
            [item["source_detection_id"] for item in policy.apply([large, small])],
        )


class VirtualComponentMaskNmsTests(unittest.TestCase):
    def test_tiny_island_is_removed_without_suppressing_disjoint_main(self) -> None:
        # The high-score owner has a concave main whose bbox covers the other
        # main, plus a tiny island inside that other main.  Legacy bbox NMS
        # suppresses the other owner.  Mask v4 must remove only the <=1% island
        # and retain both non-overlapping mains (HEYZO-3560 f82039 pattern).
        l_shape = [
            [0, 0],
            [100, 0],
            [100, 20],
            [20, 20],
            [20, 100],
            [0, 100],
        ]
        tiny = polygon(35, 35, 38, 38)
        other_main = polygon(30, 30, 60, 60)
        owner = detection([l_shape, tiny], 0.9, 1)
        other = detection([other_main], 0.8, 2)

        retained, stats, trace = VirtualComponentMaskNms().apply_with_trace(
            [owner, other]
        )
        by_id = {item["source_detection_id"]: item for item in retained}

        self.assertEqual({1, 2}, set(by_id))
        self.assertEqual([l_shape], by_id[1]["polygons"])
        self.assertEqual(1, stats.tiny_islands_removed)
        self.assertEqual(0, stats.main_owners_suppressed)
        self.assertFalse(any(row["reason"] == "main_main_mask_nms" for row in trace))

    def test_nested_main_is_suppressed_by_directed_mask_coverage(self) -> None:
        large = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        small = detection([polygon(10, 10, 60, 60)], 0.8, 2)

        retained, stats, trace = VirtualComponentMaskNms().apply_with_trace(
            [large, small]
        )

        self.assertEqual([1], [item["source_detection_id"] for item in retained])
        self.assertEqual(1, stats.main_owners_suppressed)
        self.assertEqual("main_main_mask_nms", trace[0]["reason"])
        self.assertEqual("mask_contained", trace[0]["suppression_reason"])
        self.assertGreaterEqual(trace[0]["smaller_coverage"], 0.80)


if __name__ == "__main__":
    unittest.main()
