from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from contracts.detections import CutList, dumps_json_line, write_cut_list
from nms.component_virtual import ProductionVirtualComponentNms
from nms.components import fill_holes_and_remove_tiny_islands
from nms.config import PRODUCTION_NMS_CONFIG, ProductionNmsConfig
from nms.mask_adaptive import AdaptiveMaskNms
from nms.production import PRODUCTION_OPTIONS
from tracking.association import detection_features
from tracking.builder import build_tracked_sqlite
from tracking.records import prepare_detection


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


class ProductionNmsConfigTests(unittest.TestCase):
    def test_stage_options_come_from_the_canonical_config(self) -> None:
        self.assertEqual(
            PRODUCTION_NMS_CONFIG.implementation_options(), PRODUCTION_OPTIONS
        )
        self.assertNotIn("comparison_policy", PRODUCTION_OPTIONS)

    def test_invalid_threshold_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            ProductionNmsConfig(
                mask_iou_threshold=0.05,
                mask_small_iou_threshold=0.10,
                mask_tiny_iou_threshold=0.20,
            ).validate()


class TopologyCleanupTests(unittest.TestCase):
    def test_fills_holes_and_removes_at_most_one_percent_islands(self) -> None:
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

    def test_crossing_contour_is_not_misclassified_as_hole(self) -> None:
        outer = polygon(0, 0, 100, 100)
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
        crossing_child = polygon(20, 80, 80, 90)
        source = detection([concave_parent, crossing_child], 0.9, 1)
        cleaned, stats = fill_holes_and_remove_tiny_islands(
            [source], unconditional_owner_ratio_max=0.0
        )
        self.assertEqual([concave_parent, crossing_child], cleaned[0]["polygons"])
        self.assertEqual(0, stats.holes_filled)


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
        retained = AdaptiveMaskNms().apply(
            [detection([l_shape], 0.9, 1), detection([cavity], 0.8, 2)]
        )
        self.assertEqual([1, 2], [item["source_detection_id"] for item in retained])

    def test_directed_coverage_catches_low_symmetric_iou(self) -> None:
        large = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        small = detection([polygon(10, 10, 60, 60)], 0.8, 2)
        retained = AdaptiveMaskNms().apply([large, small])
        self.assertEqual([1], [item["source_detection_id"] for item in retained])

    def test_similar_size_overlap_uses_mask_iou(self) -> None:
        first = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        second = detection([polygon(20, 0, 120, 100)], 0.8, 2)
        retained = AdaptiveMaskNms().apply([first, second])
        self.assertEqual([1], [item["source_detection_id"] for item in retained])

    def test_area_band_uses_continuous_contour_area(self) -> None:
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


class ProductionVirtualComponentNmsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ProductionVirtualComponentNms(**PRODUCTION_OPTIONS)

    def test_main_mask_nms_suppresses_lower_score_owner(self) -> None:
        high = detection([polygon(0, 0, 100, 100)], 0.9, 1)
        low = detection([polygon(0, 0, 100, 100), polygon(150, 0, 170, 20)], 0.8, 2)
        retained, stats, trace = self.policy.apply_with_trace([high, low])
        self.assertEqual([1], [item["source_detection_id"] for item in retained])
        self.assertEqual(1, stats.main_owners_suppressed)
        self.assertEqual("main_main_mask_nms", trace[0]["reason"])

    def test_single_component_preserves_canonical_bbox(self) -> None:
        outer = detection([polygon(819.5, 304.5, 1016.5, 528.5)], 0.792, 7809)
        outer["bbox_xyxy"] = [817.0, 303.0, 1018.0, 529.5]
        outer["bbox"] = [817.0, 303.0, 201.0, 226.5]
        inner = detection([polygon(818.5, 468.5, 892.5, 528.5)], 0.350, 7810)
        inner["bbox_xyxy"] = [817.0, 467.0, 893.0, 529.5]
        inner["bbox"] = [817.0, 467.0, 76.0, 62.5]
        retained = self.policy.apply([outer, inner])
        self.assertEqual([7809], [item["source_detection_id"] for item in retained])

    def test_island_vs_island_removes_only_lower_score_island(self) -> None:
        shared = polygon(120, 0, 140, 20)
        first = detection([polygon(0, 0, 100, 100), shared], 0.9, 1)
        second = detection([polygon(200, 0, 300, 100), shared], 0.8, 2)
        retained, stats, trace = self.policy.apply_with_trace([first, second])
        by_id = {item["source_detection_id"]: item for item in retained}
        self.assertEqual(2, len(by_id[1]["polygons"]))
        self.assertEqual(1, len(by_id[2]["polygons"]))
        self.assertEqual(1, stats.island_island_suppressed)
        self.assertTrue(any(row["reason"] == "island_island_mask_nms" for row in trace))

    def test_island_score_cannot_delete_another_main(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, island], 0.95, 1)
        other = detection([other_main], 0.40, 2)
        retained, stats, trace = self.policy.apply_with_trace([owner, other])
        by_id = {item["source_detection_id"]: item for item in retained}
        self.assertEqual({1, 2}, set(by_id))
        self.assertEqual([owner_main], by_id[1]["polygons"])
        self.assertEqual([other_main], by_id[2]["polygons"])
        self.assertEqual(1, stats.island_main_suppressed)
        self.assertTrue(any(row["reason"] == "island_subordinate_to_main" for row in trace))

    def test_island_main_gate_keeps_low_coverage_component(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        partial = polygon(120, 0, 133, 40)
        retained, stats = self.policy.apply_with_diagnostics(
            [detection([main, island], 0.9, 1), detection([partial], 0.8, 2)]
        )
        by_id = {item["source_detection_id"]: item for item in retained}
        self.assertEqual([main, island], by_id[1]["polygons"])
        self.assertEqual(0, stats.island_main_suppressed)

    def test_island_main_area_boundary_is_inclusive(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other = polygon(120, 0, 140, 40)
        retained, stats = self.policy.apply_with_diagnostics(
            [detection([main, island], 0.9, 1), detection([other], 0.8, 2)]
        )
        by_id = {item["source_detection_id"]: item for item in retained}
        self.assertEqual([main], by_id[1]["polygons"])
        self.assertEqual(1, stats.island_main_suppressed)

    def test_same_owner_components_are_never_compared(self) -> None:
        source = detection(
            [polygon(0, 0, 100, 100), polygon(90, 40, 110, 60)], 0.9, 1
        )
        policy = ProductionVirtualComponentNms(
            **{**PRODUCTION_OPTIONS, "unconditional_owner_ratio_max": 0.0}
        )
        retained, stats = policy.apply_with_diagnostics([source])
        self.assertEqual(2, len(retained[0]["polygons"]))
        self.assertEqual(0, stats.main_main_pairs)
        self.assertEqual(0, stats.island_island_pairs)
        self.assertEqual(0, stats.island_main_pairs)

    def test_cleanup_preserves_raw_tracking_geometry(self) -> None:
        owner_main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        other_main = polygon(115, -5, 155, 35)
        owner = detection([owner_main, island], 0.9, 1)
        other = detection([other_main], 0.8, 2)
        by_id = {
            item["source_detection_id"]: item
            for item in self.policy.apply([owner, other])
        }
        self.assertEqual([0.0, 0.0, 140.0, 100.0], by_id[1]["_association_bbox_xyxy"])
        self.assertEqual([0.0, 0.0, 100.0, 100.0], by_id[1]["bbox_xyxy"])
        self.assertEqual([owner_main, island], owner["polygons"])
        prepared = prepare_detection(by_id[1])
        features = detection_features(prepared)
        self.assertEqual((0.0, 0.0, 140.0, 100.0), features.bbox)

    def test_private_association_geometry_is_not_persisted(self) -> None:
        main = polygon(0, 0, 100, 100)
        island = polygon(120, 0, 140, 20)
        cover = polygon(115, -5, 155, 35)
        retained = self.policy.apply(
            [detection([main, island], 0.9, 1), detection([cover], 0.8, 2)]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.jsonl"
            source.write_bytes(
                dumps_json_line({"frame_index": 0, "detections": retained})
            )
            cuts = write_cut_list(
                root / "cuts.json", CutList(frames=(), method="test")
            )
            output = root / "tracked.sqlite"
            build_tracked_sqlite(
                source, output, cuts, remove_short_tracks_max_frames=0
            )
            with sqlite3.connect(output) as connection:
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(raw_tracked_masks)"
                    )
                }
                row = connection.execute(
                    "SELECT polygons, bbox_xyxy_json FROM raw_tracked_masks "
                    "WHERE source_detection_id=1"
                ).fetchone()
        self.assertNotIn("_association_bbox_xyxy", columns)
        self.assertNotIn("_association_mask_area", columns)
        self.assertEqual([main], json.loads(row[0]))
        self.assertEqual([0.0, 0.0, 100.0, 100.0], json.loads(row[1]))


if __name__ == "__main__":
    unittest.main()
