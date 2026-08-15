from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experimental.production_candidate_polygon14 import CANDIDATE
from experimental.production_candidate_polygon14.integration import (
    apply_spatial_candidate,
)
from experimental.production_candidate_polygon14.run import build_command
from experimental.production_candidate_polygon14.spatial import build_spatial_track
from experimental.production_candidate_polygon14.topology_guard import (
    first_invalid_edge_frame,
    local_key_update_is_simple,
    polygon_is_simple,
    repair_decoded_path,
)
from experimental.temporal_vertex_decimation_20260812.optimizer import (
    has_self_intersection,
    resample_closed,
)


def _contour(offset_x: float, offset_y: float, scale: float = 1.0) -> np.ndarray:
    points = np.asarray(
        [
            [0, 0], [40, 0], [40, 12], [28, 12], [28, 30], [40, 30],
            [40, 42], [0, 42], [0, 30], [12, 30], [12, 12], [0, 12],
        ],
        dtype=np.float64,
    )
    return resample_closed(
        points * float(scale) + np.asarray([offset_x, offset_y]), 96
    )


def _two_component_track(frames: int = 6) -> list[list[np.ndarray]]:
    return [
        [
            _contour(10.0 + frame, 20.0),
            _contour(100.0 - 0.5 * frame, 80.0, 0.65),
        ]
        for frame in range(frames)
    ]


class _ExactEvaluator:
    def __init__(self, minimum_vertices: int = 14) -> None:
        self.minimum_vertices = int(minimum_vertices)
        self.calls: list[int] = []

    def exact_frame_metrics(
        self, _frame: int, _vector, _components: int, vertices: int
    ) -> tuple[float, ...]:
        self.calls.append(int(vertices))
        recall = 0.98 if int(vertices) >= self.minimum_vertices else 0.96
        iou = 0.96
        return (100.0, 100.0, 98.0, 102.0, recall, 0.98, iou)


def test_candidate_contract_is_explicit_and_schema_preserving() -> None:
    assert CANDIDATE.profile_id == "polygon14_keyframe_v1"
    assert CANDIDATE.vertices_per_component == 14
    assert CANDIDATE.vertex_fallbacks == (14, 16, 18, 20)
    assert CANDIDATE.spatial_recall_floor == 0.97
    assert CANDIDATE.temporal_recall_floor == 0.97
    assert CANDIDATE.temporal_target_kind == "soft_keyframe_interval"
    assert CANDIDATE.topology_constraint == "simple_polygon_hard_constraint"
    assert CANDIDATE.topology_dp_mode == "lazy_selected_edge_split"
    assert CANDIDATE.output_schema == "unchanged"


def test_spatial_builder_is_deterministic_and_supports_component_slots() -> None:
    source = _two_component_track()
    first, stats = build_spatial_track(source)
    second, repeated_stats = build_spatial_track(source)
    assert first.shape == (6, 2, 14, 2)
    assert np.array_equal(first, second)
    assert stats == repeated_stats
    assert stats.frames == 6
    assert stats.components == 2
    assert all(
        not has_self_intersection(first[frame, component])
        for frame in range(6)
        for component in range(2)
    )


def test_integration_replaces_only_anchors_and_preserves_recall_reference() -> None:
    gt_polygons = _two_component_track()
    reference_copy = [
        [component.copy() for component in frame] for frame in gt_polygons
    ]
    run = SimpleNamespace(
        anchors_per_contour=14,
        gt_polygons=gt_polygons,
        anchors=np.zeros((6, 2, 14, 2), dtype=np.float32),
        run_target_total_points=0,
    )
    profile: dict[str, float | int] = {}
    apply_spatial_candidate(run, profile, endpoint_evaluator=_ExactEvaluator())
    assert run.gt_polygons is gt_polygons
    for before, after in zip(reference_copy, run.gt_polygons):
        for expected, actual in zip(before, after):
            assert np.array_equal(expected, actual)
    assert run.anchors.shape == (6, 2, 14, 2)
    assert run.run_target_total_points == 28
    assert profile["polygon14_frames"] == 6
    assert profile["polygon14_components"] == 2
    assert profile["polygon_vertex_selected_14_runs"] == 1


def test_integration_falls_back_track_wide_to_smallest_exact_feasible_count() -> None:
    gt_polygons = _two_component_track()
    run = SimpleNamespace(
        stream_id="synthetic",
        anchors_per_contour=20,
        gt_polygons=gt_polygons,
        anchors=np.zeros((6, 2, 20, 2), dtype=np.float32),
        run_target_total_points=0,
    )
    evaluator = _ExactEvaluator(minimum_vertices=18)
    profile: dict[str, float | int] = {}
    apply_spatial_candidate(run, profile, endpoint_evaluator=evaluator)
    assert run.anchors.shape == (6, 2, 18, 2)
    assert run.anchors_per_contour == 18
    assert run.run_target_total_points == 36
    assert set(evaluator.calls) == {14, 16, 18}
    assert profile["polygon_vertex_selected_18_runs"] == 1
    assert profile["polygon_vertex_fallback_runs"] == 1


def test_integration_fails_closed_when_twenty_points_cannot_meet_recall() -> None:
    gt_polygons = _two_component_track(frames=2)
    run = SimpleNamespace(
        stream_id="synthetic-infeasible",
        anchors_per_contour=20,
        gt_polygons=gt_polygons,
        anchors=np.zeros((2, 2, 20, 2), dtype=np.float32),
        run_target_total_points=0,
    )
    with pytest.raises(RuntimeError, match="through 20 vertices"):
        apply_spatial_candidate(
            run,
            {},
            endpoint_evaluator=_ExactEvaluator(minimum_vertices=22),
        )


def test_candidate_temporal_palette_is_exactly_the_frozen_baseline() -> None:
    experiment = Path(__file__).resolve().parents[1] / "experimental/0809"
    sys.path.insert(0, str(experiment))
    try:
        runtime = importlib.import_module("phase2_runtime")
        assert CANDIDATE.profile_id in runtime.VALID_PROFILES
        for interval in (1.0, 3.0, 6.0, 8.0):
            for label in ("女性器", "男性器", "結合部分"):
                assert runtime._class_role_state_profile(
                    CANDIDATE.profile_id, label, interval
                ) == runtime._class_role_state_profile(
                    "new_production_v1", label, interval
                )
    finally:
        sys.path.remove(str(experiment))


def test_runner_fixes_polygon_count_and_uses_selected_edge_exact_validation(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        source_root=tmp_path / "source",
        labels="男性器",
        num_workers=1,
        label_workers=1,
        max_tracks=1,
        force=False,
    )
    command = build_command(args, 6, tmp_path / "result")
    joined = " ".join(command)
    assert "--profiles polygon14_keyframe_v1" in joined
    assert "--anchors-per-contour 20" in joined
    assert "--min-anchors-per-contour 14" in joined
    assert "--no-adaptive-anchor-counts" in command
    assert "--cuda-lazy-exact" in command
    assert "--pair-vote-per-key" in command


def test_topology_guard_rejects_crossing_but_keeps_valid_polygon() -> None:
    valid = np.asarray([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)
    crossing = np.asarray([[0, 0], [4, 4], [0, 4], [4, 0]], dtype=np.float32)
    assert polygon_is_simple(valid)
    assert not polygon_is_simple(crossing)


def test_topology_guard_checks_interpolated_frames_not_only_endpoints() -> None:
    start = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    end = np.asarray(
        [
            [-3.0752938, -1.8057977],
            [1.0097507, -0.8455412],
            [2.4680433, -0.9661310],
            [-0.2129090, 0.9398172],
        ],
        dtype=np.float32,
    )
    assert polygon_is_simple(start)
    assert polygon_is_simple(end)
    module = SimpleNamespace(
        interpolate_vectors=lambda left, right, alpha: (
            (1.0 - alpha) * left + alpha * right
        ).astype(np.float32),
        split_vector_to_polygons=lambda vector, _components, _vertices: [
            np.asarray(vector, dtype=np.float32).reshape(4, 2)
        ],
    )
    run = SimpleNamespace(contour_count=1, anchors_per_contour=4)
    assert first_invalid_edge_frame(module, run, 0, start, 2, end) == 1


def test_topology_guard_splits_only_the_invalid_selected_edge() -> None:
    start = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    end = np.asarray(
        [
            [-3.0752938, -1.8057977],
            [1.0097507, -0.8455412],
            [2.4680433, -0.9661310],
            [-0.2129090, 0.9398172],
        ],
        dtype=np.float32,
    )
    middle = 0.25 * start + 0.75 * end

    def split(vector, _components, _vertices):
        return [np.asarray(vector, dtype=np.float32).reshape(4, 2)]

    module = SimpleNamespace(
        interpolate_vectors=lambda left, right, alpha: (
            (1.0 - alpha) * left + alpha * right
        ).astype(np.float32),
        split_vector_to_polygons=split,
        interval_cost_from_vectors=lambda *_args, **_kwargs: SimpleNamespace(
            cost=0.0
        ),
    )
    run = SimpleNamespace(
        contour_count=1,
        anchors_per_contour=4,
        stream_id="synthetic",
    )
    candidate = lambda vector: SimpleNamespace(
        vector=np.asarray(vector, dtype=np.float32),
        polygons=split(vector, 1, 4),
    )
    candidates = [
        [candidate(start)],
        [candidate(middle)],
        [candidate(end)],
    ]
    stats: dict[str, float | int] = {}
    frames, states = repair_decoded_path(
        module,
        run,
        [0, 2],
        [0, 0],
        candidates,
        SimpleNamespace(),
        None,
        stats,
    )
    assert frames == [0, 1, 2]
    assert states == [0, 0, 0]
    assert stats["dp_invalid_edges"] == 1
    assert stats["dp_inserted_keys"] == 1


def test_pair_vote_gate_rejects_only_the_crossing_trial() -> None:
    valid = np.asarray([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)
    crossing = np.asarray([[0, 0], [4, 4], [0, 4], [4, 0]], dtype=np.float32)
    module = SimpleNamespace(
        interpolate_vectors=lambda left, right, alpha: (
            (1.0 - alpha) * left + alpha * right
        ).astype(np.float32),
        split_vector_to_polygons=lambda vector, _components, _vertices: [
            np.asarray(vector, dtype=np.float32).reshape(4, 2)
        ],
    )
    run = SimpleNamespace(contour_count=1, anchors_per_contour=4)
    current = np.stack([valid, valid, valid], axis=0)
    assert local_key_update_is_simple(
        module, run, [0, 1, 2], current, 1, valid
    )
    assert not local_key_update_is_simple(
        module, run, [0, 1, 2], current, 1, crossing
    )
