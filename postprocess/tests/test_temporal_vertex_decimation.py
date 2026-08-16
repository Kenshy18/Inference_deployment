from __future__ import annotations

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    DecimationConfig,
    align_temporal_dense,
    current_equal_arc_baseline,
    has_self_intersection,
    optimize_temporal_vertices,
    temporal_residuals,
    _similarity_residual,
)


def _rectangle(x: float, y: float, width: float, height: float) -> np.ndarray:
    return np.asarray(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float64,
    )


def test_temporal_decimation_keeps_one_fixed_vertex_count_and_recall() -> None:
    polygons = [
        _rectangle(float(frame * 3), 10.0, 80.0, 30.0)
        for frame in range(12)
    ]
    result = optimize_temporal_vertices(
        polygons,
        DecimationConfig(
            initial_vertices=24,
            minimum_vertices=4,
            recall_floor=0.97,
            shortlist=6,
            local_refine_radius=1,
        ),
    )
    assert result.polygons.shape[0] == len(polygons)
    assert result.polygons.shape[1] == result.metrics.vertices
    assert result.metrics.minimum_recall >= 0.97
    assert result.metrics.self_intersections == 0
    assert result.metrics.vertices < 24


def test_temporal_phase_forbids_direction_flip() -> None:
    base = np.asarray(
        [[0, 0], [5, 0], [8, 3], [3, 7], [-1, 3]], dtype=np.float64
    )
    polygons = [np.roll(base + [frame, 0], frame % len(base), axis=0) for frame in range(7)]
    aligned = align_temporal_dense(polygons, 20)
    areas = [
        0.5
        * np.sum(
            frame[:, 0] * np.roll(frame[:, 1], -1)
            - np.roll(frame[:, 0], -1) * frame[:, 1]
        )
        for frame in aligned
    ]
    assert all(value > 0 for value in areas)


def test_new_correspondence_is_no_worse_on_rigid_motion() -> None:
    base = np.asarray(
        [[0, 0], [30, 0], [38, 8], [20, 15], [0, 10]], dtype=np.float64
    )
    polygons = [np.roll(base + [4 * frame, 2 * frame], frame % 5, axis=0) for frame in range(10)]
    _baseline, baseline_metrics = current_equal_arc_baseline(polygons, 16)
    dense = align_temporal_dense(polygons, 16)
    assert dense.shape == (10, 16, 2)
    assert baseline_metrics.self_intersections == 0
    assert not any(has_self_intersection(frame) for frame in dense)


def test_batched_temporal_residual_matches_reference_svd() -> None:
    rng = np.random.default_rng(20260812)
    sequence = rng.normal(size=(8, 12, 2)) * 20.0
    expected = np.asarray(
        [
            _similarity_residual(sequence[index - 1], sequence[index])
            for index in range(1, len(sequence))
        ]
    )
    actual = temporal_residuals(sequence)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
