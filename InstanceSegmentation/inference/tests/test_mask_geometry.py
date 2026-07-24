from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mask_geometry import DEFAULT_MAX_MASK_POINTS, mask_to_polygons


def _point_count(polygons: list[list[float]]) -> int:
    return sum(len(polygon) // 2 for polygon in polygons)


def _rasterize(polygons: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points - 0.5).astype(np.int32)
        cv2.fillPoly(result, [points], 1)
    return result


class MaskGeometryTest(unittest.TestCase):
    def test_contours_always_use_chain_approx_simple(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 4:28] = 1

        with patch(
            "mask_geometry.polygonize.cv2.findContours",
            wraps=cv2.findContours,
        ) as find_contours:
            mask_to_polygons(mask)

        self.assertEqual(find_contours.call_args.args[2], cv2.CHAIN_APPROX_SIMPLE)

    def test_complex_contour_is_bounded_with_high_iou(self) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        center = np.asarray([256.0, 256.0])
        angles = np.linspace(0.0, 2.0 * np.pi, num=720, endpoint=False)
        radii = 180.0 + 24.0 * np.sin(17.0 * angles)
        points = center + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)

        polygons = mask_to_polygons(mask)

        self.assertLessEqual(_point_count(polygons), DEFAULT_MAX_MASK_POINTS)
        restored = _rasterize(polygons, mask.shape)
        intersection = np.count_nonzero(mask & restored)
        union = np.count_nonzero(mask | restored)
        self.assertGreater(intersection / union, 0.97)

    def test_total_budget_applies_across_many_contours(self) -> None:
        mask = np.zeros((256, 256), dtype=np.uint8)
        for index in range(60):
            row, column = divmod(index, 10)
            x = 4 + column * 24
            y = 4 + row * 40
            mask[y : y + 8, x : x + 8] = 1

        polygons = mask_to_polygons(mask)

        self.assertLessEqual(len(polygons), DEFAULT_MAX_MASK_POINTS // 3)
        self.assertLessEqual(_point_count(polygons), DEFAULT_MAX_MASK_POINTS)
        self.assertTrue(all(len(polygon) >= 6 for polygon in polygons))

    def test_simple_contour_and_source_offset_are_preserved(self) -> None:
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[3:15, 5:25] = 1

        polygons = mask_to_polygons(
            mask,
            x_offset=100.0,
            y_offset=200.0,
        )

        self.assertEqual(_point_count(polygons), 4)
        points = np.asarray(polygons[0]).reshape(-1, 2)
        self.assertGreaterEqual(float(points[:, 0].min()), 105.5)
        self.assertGreaterEqual(float(points[:, 1].min()), 203.5)


if __name__ == "__main__":
    unittest.main()
