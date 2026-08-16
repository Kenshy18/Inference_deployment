from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experimental.humanlike_vertex_placement_20260812.candidate import (
    quality_guarded_vertex_placement,
)
from experimental.humanlike_vertex_placement_20260812.native_dp import (
    native_temporal_dp_sequence,
)
from experimental.humanlike_vertex_placement_20260812.persistent_line_fit import (
    _fit_line,
    _fit_lines,
    persistent_line_fit_sequence,
)
from experimental.humanlike_vertex_placement_20260812.quality_repair import (
    persistent_line_fit_quality_guarded,
)
from experimental.humanlike_vertex_placement_20260812.spatial import (
    align_polygon_sequence,
    rdp_fixed_count,
)
from experimental.temporal_vertex_decimation_20260812.optimizer import (
    RasterSequenceEvaluator,
    _best_phase,
    evaluate_sequence,
    has_self_intersection,
    resample_closed,
)


NATIVE_BUILD = (
    Path(__file__).resolve().parents[1]
    / "experimental/humanlike_vertex_placement_20260812/native_temporal/build"
)


def _native_available() -> bool:
    return any(NATIVE_BUILD.glob("native_temporal_polygon*.so"))


def _concave_contour(shift_x: float = 0.0) -> np.ndarray:
    polygon = np.asarray(
        [
            [0, 0], [60, 0], [60, 18], [38, 18], [38, 38],
            [60, 38], [60, 60], [0, 60], [0, 38], [22, 38],
            [22, 18], [0, 18],
        ],
        dtype=np.float64,
    )
    polygon[:, 0] += shift_x
    return resample_closed(polygon, 144)


def _scalar_best_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    best = np.asarray(candidate, dtype=np.float64)
    best_cost = float("inf")
    for variant in (candidate, candidate[::-1]):
        for shift in range(len(candidate)):
            rolled = np.roll(variant, -shift, axis=0)
            delta = rolled - reference
            cost = float(np.mean(np.sum(delta * delta, axis=1)))
            if cost < best_cost:
                best_cost = cost
                best = rolled
    return np.asarray(best, dtype=np.float64).copy()


def test_vectorized_phase_is_bit_exact_to_scalar_reference() -> None:
    rng = np.random.default_rng(20260814)
    for count in (3, 7, 14, 64):
        for _case in range(25):
            reference = rng.normal(size=(count, 2))
            candidate = rng.normal(size=(count, 2))
            expected = _scalar_best_phase(reference, candidate)
            actual = _best_phase(
                reference,
                candidate,
                allow_reverse=True,
                procrustes=False,
            )
            assert np.array_equal(actual, expected)


def test_batched_line_fit_is_bit_exact_to_scalar_reference() -> None:
    rng = np.random.default_rng(2026081401)
    values = rng.normal(size=(127, 19, 2)) * 40.0 + 100.5
    expected = [_fit_line(frame, 0.65) for frame in values]
    normals, offsets = _fit_lines(values, 0.65)
    assert np.array_equal(normals, np.asarray([row[0] for row in expected]))
    assert np.array_equal(offsets, np.asarray([row[1] for row in expected]))


def test_rdp_uses_exact_count_and_preserves_concave_corners() -> None:
    source = _concave_contour()
    result = rdp_fixed_count(source, 12)
    evaluator = RasterSequenceEvaluator([source])
    iou, recall = evaluator.frame_metrics(0, result)
    assert result.shape == (12, 2)
    assert iou >= 0.97
    assert recall >= 0.97
    assert not has_self_intersection(result)


def test_translation_phase_removes_cyclic_start_offset() -> None:
    first = rdp_fixed_count(_concave_contour(), 12)
    second = np.roll(first + np.asarray([3.0, 1.0]), 6, axis=0)
    aligned = align_polygon_sequence([first, second])
    midpoint = 0.5 * (aligned[0] + aligned[1])
    assert not has_self_intersection(midpoint)
    assert np.mean(np.linalg.norm(aligned[1] - aligned[0], axis=1)) < 5.0


@pytest.mark.skipif(not _native_available(), reason="native experiment has not been built")
def test_native_densifies_a_contour_with_fewer_points_than_target() -> None:
    square = np.asarray([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.float64)
    source = [square + np.asarray([float(frame), 0.0]) for frame in range(5)]
    result = native_temporal_dp_sequence(source, 20, frame_indices=range(5))
    evaluator = RasterSequenceEvaluator(source)
    metrics = evaluate_sequence(
        evaluator,
        result,
        initial_vertices=48,
        temporal_weight=0.05,
        vertex_weight=0.02,
    )
    assert result.shape == (5, 20, 2)
    assert metrics.minimum_recall >= 0.99
    assert metrics.self_intersections == 0


@pytest.mark.skipif(not _native_available(), reason="native experiment has not been built")
def test_native_temporal_dp_is_fixed_count_and_recall_guarded() -> None:
    source = [_concave_contour(float(frame)) for frame in range(9)]
    result = native_temporal_dp_sequence(
        source,
        12,
        frame_indices=range(9),
        temporal_weight=0.003,
        distance_weight=2.0,
        missing_area_weight=1.0,
    )
    evaluator = RasterSequenceEvaluator(source)
    metrics = evaluate_sequence(
        evaluator,
        result,
        initial_vertices=48,
        temporal_weight=0.05,
        vertex_weight=0.02,
    )
    assert result.shape == (9, 12, 2)
    assert metrics.minimum_recall >= 0.98
    assert metrics.self_intersections == 0


@pytest.mark.skipif(not _native_available(), reason="native experiment has not been built")
def test_quality_guard_keeps_one_count_for_entire_track() -> None:
    source = [_concave_contour(float(frame)) for frame in range(7)]
    result = quality_guarded_vertex_placement(
        source,
        frame_indices=range(7),
        candidate_counts=(12, 16, 20),
        recall_floor=0.97,
        minimum_iou_floor=0.95,
    )
    assert result.polygons.shape == (7, result.vertices, 2)
    assert result.vertices in {12, 16, 20}
    assert result.attempts[-1].metrics.minimum_recall >= 0.97


def test_persistent_line_fit_is_deterministic_simple_and_fixed_count() -> None:
    source = [_concave_contour(float(frame) * 1.5) for frame in range(9)]
    first = persistent_line_fit_sequence(source, 12, dense_vertices=64)
    second = persistent_line_fit_sequence(source, 12, dense_vertices=64)
    evaluator = RasterSequenceEvaluator(source)
    metrics = evaluate_sequence(
        evaluator,
        first,
        initial_vertices=12,
        temporal_weight=0.05,
        vertex_weight=0.0,
    )
    assert first.shape == (9, 12, 2)
    assert np.array_equal(first, second)
    assert metrics.minimum_recall >= 0.97
    assert metrics.self_intersections == 0


def test_persistent_line_fit_keeps_interpolable_phase_during_rotation() -> None:
    base = _concave_contour() - np.asarray([30.0, 30.0])
    source = []
    for frame in range(7):
        angle = np.deg2rad(4.0 * frame)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        source.append(base @ rotation.T + np.asarray([50.0 + frame, 50.0]))
    result = persistent_line_fit_sequence(source, 12, dense_vertices=64)
    evaluator = RasterSequenceEvaluator(source)
    for frame in range(1, len(source) - 1):
        interpolated = 0.5 * (result[frame - 1] + result[frame + 1])
        iou, recall = evaluator.frame_metrics(frame, interpolated)
        assert iou >= 0.90
        assert recall >= 0.90
        assert not has_self_intersection(interpolated)


def test_quality_repair_only_changes_frames_below_the_exact_floor() -> None:
    source = [_concave_contour(float(frame)) for frame in range(5)]
    raw = persistent_line_fit_sequence(
        source, 12, dense_vertices=64, allocation_distance_weight=0.0
    )
    repaired, stats = persistent_line_fit_quality_guarded(
        source,
        12,
        dense_vertices=64,
        allocation_distance_weight=0.0,
    )
    evaluator = RasterSequenceEvaluator(source)
    raw_metrics = evaluate_sequence(
        evaluator, raw, initial_vertices=12, temporal_weight=0.05, vertex_weight=0.0
    )
    repaired_metrics = evaluate_sequence(
        evaluator,
        repaired,
        initial_vertices=12,
        temporal_weight=0.05,
        vertex_weight=0.0,
    )
    assert raw_metrics.minimum_recall < 0.97
    assert stats.repaired_frames > 0
    assert repaired_metrics.minimum_recall >= 0.97
    assert repaired_metrics.minimum_iou >= 0.95
    assert repaired_metrics.self_intersections == 0
