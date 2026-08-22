"""OpenCV pixel-parity tests for the shared CUDA raster evaluator."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from production.raster.cuda_opencv import (
    CudaOpenCvRasterizer,
    PackedContours,
    cuda_available,
)
from production.raster.cuda_interval_exact import CudaExactIntervalBatch
from production.raster.geometry import catmull_rom_boundaries, ellipse_boundaries
from production.raster.metrics import CudaGeometryMetrics
from production.curve.runtime.native_cpu import ExactDoubleRasterBatch, native_module


def _opencv_mask(boundary: np.ndarray, padding: int = 2) -> np.ndarray:
    value = np.asarray(boundary, dtype=np.float64)
    minimum = np.floor(np.min(value, axis=0)).astype(np.int32) - int(padding)
    maximum = np.ceil(np.max(value, axis=0)).astype(np.int32) + int(padding)
    mask = np.zeros(
        (maximum[1] - minimum[1] + 1, maximum[0] - minimum[0] + 1),
        dtype=np.uint8,
    )
    cv2.fillPoly(mask, [np.rint(value - minimum).astype(np.int32)], 1)
    return mask


def _opencv_metrics(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    minimum = (
        np.floor(
            np.minimum(np.min(reference, axis=0), np.min(prediction, axis=0))
        ).astype(np.int32)
        - 2
    )
    maximum = (
        np.ceil(
            np.maximum(np.max(reference, axis=0), np.max(prediction, axis=0))
        ).astype(np.int32)
        + 2
    )
    shape = (maximum[1] - minimum[1] + 1, maximum[0] - minimum[0] + 1)
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(left, [np.rint(reference - minimum).astype(np.int32)], 1)
    cv2.fillPoly(right, [np.rint(prediction - minimum).astype(np.int32)], 1)
    left_area = int(cv2.countNonZero(left))
    right_area = int(cv2.countNonZero(right))
    intersection = int(cv2.countNonZero(cv2.bitwise_and(left, right)))
    union = left_area + right_area - intersection
    return np.asarray(
        (
            left_area,
            right_area,
            intersection,
            union,
            intersection / left_area if left_area else 1.0,
            intersection / right_area if right_area else 1.0,
            intersection / union if union else 1.0,
        ),
        dtype=np.float64,
    )


@unittest.skipUnless(cuda_available(), "CUDA/CuPy is unavailable")
class CudaOpenCvRasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raster = CudaOpenCvRasterizer()

    def assert_masks_equal(self, boundaries: np.ndarray) -> None:
        actual = self.raster.rasterize(boundaries)
        for boundary, mask in zip(boundaries, actual):
            np.testing.assert_array_equal(mask, _opencv_mask(boundary))

    def test_adversarial_integer_and_half_pixel_polygons(self) -> None:
        values = np.asarray(
            (
                ((0, 0), (10, 0), (10, 10), (0, 10)),
                ((0.5, 0.5), (10.5, 0.5), (7.5, 8.5), (0.5, 10.5)),
                ((-7.5, -3.5), (5.5, -3.5), (1.5, 2.5), (-6.5, 8.5)),
                ((0, 0), (8, 0), (8, 3), (3, 3), (3, 8), (0, 8)),
            ),
            dtype=object,
        )
        for boundary in values:
            self.assert_masks_equal(np.asarray(boundary, dtype=np.float64)[None])

    def test_random_concave_polygons(self) -> None:
        random = np.random.default_rng(20260822)
        for point_count in range(3, 21):
            cases = []
            for _ in range(6):
                angles = np.sort(random.random(point_count) * 2.0 * np.pi)
                radii = random.uniform(4.0, 48.0, point_count)
                center = random.uniform(-25.0, 100.0, 2)
                cases.append(
                    center
                    + np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
                )
            self.assert_masks_equal(np.stack(cases))

    def test_ellipse_96_point_contract(self) -> None:
        random = np.random.default_rng(31)
        ellipses = np.column_stack(
            (
                random.uniform(-20.0, 200.0, (64, 2)),
                random.uniform(2.0, 60.0, (64, 2)),
                random.uniform(-180.0, 180.0, 64),
            )
        )
        self.assert_masks_equal(ellipse_boundaries(ellipses, points=96))

    def test_catmull_rom_224_point_contract(self) -> None:
        random = np.random.default_rng(47)
        angle = np.arange(14, dtype=np.float64) * (2.0 * np.pi / 14.0)
        controls = []
        for _ in range(32):
            radius = random.uniform(8.0, 55.0, 14)
            center = random.uniform(-20.0, 200.0, 2)
            controls.append(
                center
                + np.column_stack((np.cos(angle) * radius, np.sin(angle) * radius))
            )
        boundaries = catmull_rom_boundaries(np.stack(controls))
        self.assertEqual(224, boundaries.shape[1])
        self.assert_masks_equal(boundaries)

    def test_multiple_components_are_independently_unioned(self) -> None:
        contours = np.asarray(
            (
                ((0, 0), (12, 0), (12, 12), (0, 12)),
                ((5, 5), (17, 5), (17, 17), (5, 17)),
            ),
            dtype=np.float64,
        )
        origin = np.asarray(((-2, -2),), dtype=np.int32)
        shape = np.asarray(((22, 22),), dtype=np.int32)
        packed = PackedContours(
            points=contours,
            point_counts=np.asarray((4, 4), dtype=np.int32),
            contour_cases=np.asarray((0, 0), dtype=np.int32),
            origins=origin,
            shapes=shape,
        )
        actual = self.raster.rasterize_packed(packed)[0]
        expected = np.zeros((22, 22), dtype=np.uint8)
        for contour in contours:
            cv2.fillPoly(
                expected,
                [np.rint(contour - origin[0]).astype(np.int32)],
                1,
            )
        np.testing.assert_array_equal(actual, expected)

    def test_gpu_metric_reduction_is_exact(self) -> None:
        random = np.random.default_rng(53)
        ellipses = np.column_stack(
            (
                random.uniform(-20.0, 200.0, (96, 2)),
                random.uniform(3.0, 50.0, (96, 2)),
                random.uniform(-180.0, 180.0, 96),
            )
        )
        shifted = ellipses.copy()
        shifted[:, :2] += random.normal(0.0, 3.0, (96, 2))
        shifted[:, 2:4] *= random.uniform(0.8, 1.2, (96, 2))
        references = ellipse_boundaries(ellipses)
        predictions = ellipse_boundaries(shifted)
        actual = self.raster.metrics(references, predictions)
        expected = np.stack(
            [
                _opencv_metrics(reference, prediction)
                for reference, prediction in zip(references, predictions)
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_geometry_level_ellipse_and_catmull_rom_api_is_exact(self) -> None:
        metrics = CudaGeometryMetrics()
        ellipse_left = np.asarray(((100.0, 80.0, 35.0, 18.0, -22.0),))
        ellipse_right = np.asarray(((102.0, 79.0, 36.0, 17.0, -18.0),))
        actual_ellipse = metrics.ellipses(ellipse_left, ellipse_right)
        expected_ellipse = np.stack(
            [
                _opencv_metrics(left, right)
                for left, right in zip(
                    ellipse_boundaries(ellipse_left),
                    ellipse_boundaries(ellipse_right),
                )
            ]
        )
        np.testing.assert_array_equal(actual_ellipse, expected_ellipse)

        angle = np.arange(14) * (2.0 * np.pi / 14.0)
        left = np.column_stack((np.cos(angle) * 30.0, np.sin(angle) * 18.0))[None]
        right = left + np.asarray((1.5, -0.5))
        actual_curve = metrics.catmull_rom(left, right)
        expected_curve = np.stack(
            [
                _opencv_metrics(a, b)
                for a, b in zip(
                    catmull_rom_boundaries(left),
                    catmull_rom_boundaries(right),
                )
            ]
        )
        np.testing.assert_array_equal(actual_curve, expected_curve)

    @unittest.skipUnless(
        native_module() is not None, "native OpenCV oracle unavailable"
    )
    def test_fused_interval_graph_is_bit_exact(self) -> None:
        frame_count = 15
        state_count = 3
        point_count = 14
        angle = np.arange(point_count) * (2.0 * np.pi / point_count)
        controls = np.empty((frame_count, state_count, point_count, 2))
        references = []
        for frame in range(frame_count):
            center = np.asarray((100.0 + frame * 1.2, 80.0 + 3.0 * np.sin(frame / 3.0)))
            base = center + np.column_stack(
                (
                    np.cos(angle) * (35.0 + 2.0 * np.sin(angle * 3 + frame * 0.2)),
                    np.sin(angle) * (20.0 + 2.0 * np.cos(angle * 2 - frame * 0.1)),
                )
            )
            references.append(catmull_rom_boundaries(base)[0])
            for state, scale in enumerate((1.0, 1.01, 1.03)):
                controls[frame, state] = center + (base - center) * scale
        candidates = np.stack(
            [
                catmull_rom_boundaries(controls[:, state])
                for state in range(state_count)
            ],
            axis=1,
        )
        edges = np.asarray(
            [
                (end - gap, start_state, end, end_state)
                for end in range(1, frame_count)
                for gap in range(1, min(6, end) + 1)
                for start_state in range(state_count)
                for end_state in range(state_count)
            ],
            dtype=np.int32,
        )
        expected = ExactDoubleRasterBatch(references).edge_metrics(
            candidates,
            edges,
            recall_floor=0.97,
            low_iou_quadratic_weight=4.0,
            threads=4,
            check_topology=False,
        )
        actual = CudaExactIntervalBatch(references).edge_metrics(
            candidates,
            edges,
            recall_floor=0.97,
            low_iou_quadratic_weight=4.0,
            check_topology=False,
        )
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        native_module() is not None, "native OpenCV oracle unavailable"
    )
    def test_fused_polygon_and_ellipse_graphs_are_bit_exact(self) -> None:
        frame_count = 12
        state_count = 3
        edges = np.asarray(
            [
                (end - gap, left, end, right)
                for end in range(1, frame_count)
                for gap in range(1, min(5, end) + 1)
                for left in range(state_count)
                for right in range(state_count)
            ],
            dtype=np.int32,
        )
        ellipse_parameters = np.empty((frame_count, state_count, 5), np.float64)
        for frame in range(frame_count):
            for state, scale in enumerate((1.0, 1.03, 1.07)):
                ellipse_parameters[frame, state] = (
                    100.0 + 2.0 * frame,
                    80.0 + 5.0 * np.sin(frame / 4.0),
                    35.0 * scale,
                    20.0 * scale,
                    -30.0 + 2.0 * frame,
                )
        ellipse_candidates = np.stack(
            [
                ellipse_boundaries(ellipse_parameters[:, state])
                for state in range(state_count)
            ],
            axis=1,
        )
        ellipse_references = [
            ellipse_boundaries(ellipse_parameters[frame, 0])[0]
            for frame in range(frame_count)
        ]

        point_count = 14
        angle = np.arange(point_count) * (2.0 * np.pi / point_count)
        polygon_candidates = np.empty(
            (frame_count, state_count, point_count, 2), np.float64
        )
        polygon_references = []
        for frame in range(frame_count):
            center = np.asarray((120.0 + frame, 90.0 + 4.0 * np.sin(frame / 3.0)))
            base = center + np.column_stack(
                (
                    np.cos(angle) * (30.0 + 3.0 * np.sin(angle * 3 + frame * 0.2)),
                    np.sin(angle) * (18.0 + 2.0 * np.cos(angle * 2)),
                )
            )
            for state, scale in enumerate((1.0, 1.02, 1.05)):
                polygon_candidates[frame, state] = center + (base - center) * scale
            dense = np.arange(28) * (2.0 * np.pi / 28.0)
            polygon_references.append(
                center
                + np.column_stack(
                    (
                        np.cos(dense) * (30.0 + 3.0 * np.sin(dense * 3 + frame * 0.2)),
                        np.sin(dense) * (18.0 + 2.0 * np.cos(dense * 2)),
                    )
                )
            )

        for references, candidates, recall_floor in (
            (ellipse_references, ellipse_candidates, 0.97),
            (polygon_references, polygon_candidates, 0.90),
        ):
            expected = ExactDoubleRasterBatch(references).edge_metrics(
                candidates,
                edges,
                recall_floor=recall_floor,
                low_iou_quadratic_weight=4.0,
                threads=4,
                check_topology=False,
            )
            actual = CudaExactIntervalBatch(references).edge_metrics(
                candidates,
                edges,
                recall_floor=recall_floor,
                low_iou_quadratic_weight=4.0,
                check_topology=False,
            )
            np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        native_module() is not None, "native OpenCV oracle unavailable"
    )
    def test_fused_large_roi_uses_exact_shared_memory_tiles(self) -> None:
        frame_count = 6
        state_count = 2
        point_count = 16
        angle = np.arange(point_count) * (2.0 * np.pi / point_count)
        references = []
        candidates = np.empty(
            (frame_count, state_count, point_count, 2), dtype=np.float64
        )
        for frame in range(frame_count):
            center = np.asarray((600.0 + 3.0 * frame, 500.0 - 2.0 * frame))
            base = center + np.column_stack(
                (
                    np.cos(angle) * (310.0 + 15.0 * np.sin(angle * 3)),
                    np.sin(angle) * (220.0 + 10.0 * np.cos(angle * 2)),
                )
            )
            references.append(base)
            candidates[frame, 0] = base
            candidates[frame, 1] = center + (base - center) * 1.025
        edges = np.asarray(
            [
                (end - gap, left, end, right)
                for end in range(1, frame_count)
                for gap in range(1, min(3, end) + 1)
                for left in range(state_count)
                for right in range(state_count)
            ],
            dtype=np.int32,
        )
        expected = ExactDoubleRasterBatch(references).edge_metrics(
            candidates,
            edges,
            recall_floor=0.97,
            low_iou_quadratic_weight=4.0,
            threads=4,
            check_topology=False,
        )
        actual = CudaExactIntervalBatch(references).edge_metrics(
            candidates,
            edges,
            recall_floor=0.97,
            low_iou_quadratic_weight=4.0,
            check_topology=False,
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
