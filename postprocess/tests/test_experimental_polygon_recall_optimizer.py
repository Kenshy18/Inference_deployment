from __future__ import annotations

import sqlite3

import numpy as np
from experimental.polygon_recall_optimizer.audit_superior import _vertex_safety_audit
from experimental.polygon_recall_optimizer.fixed_budget import (
    Component,
    Keyframe,
    RawMask,
    Segment,
    _raw_keyframe,
    adaptive_add_recall_keys,
    adaptive_split_recall_keys,
    evaluate_segments,
    lexicographic_recall_stability_optimize,
    minimax_recall_positions,
    projected_temporal_smooth,
)
from experimental.polygon_recall_optimizer.pareto_dp import (
    GlobalParetoPoint,
    _anchor_metrics,
    _build_pair_vote_sources,
    _evaluate_anchor_edge,
    _evaluate_edge,
    _fast_numpy_align,
    _keyframe_geometry,
    _make_feasible_anchors,
    _prepare_anchor,
    _select_frontier_index,
    optimize_pareto_frontier,
    canonicalize_selected_path,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    audit_border_safety,
    border_geometry_feasible,
    build_border_safety_constraints,
    compare_geometry_paths,
    evaluate_direct,
    expand_border_constraints,
)
from experimental.polygon_recall_optimizer.temporal_consensus import (
    build_asymmetric_temporal_consensus,
)
from experimental.polygon_recall_optimizer.production_recall_guard import (
    guard_production_recall,
)
from experimental.polygon_recall_optimizer.run_production_guard import (
    _restore_requested_class_policy,
)
from experimental.polygon_recall_optimizer.trusted_optimizer import (
    _select_local_point,
)
from experimental.polygon_recall_optimizer.pareto_dp import LocalParetoPoint
from overlay_renderer.keyframe_cache import (
    _numpy_align,
    _numpy_resample,
    _polygon_components_for_final_frame,
)
from shapely.geometry import MultiPolygon, Polygon


def _square(frame: int, track_id: str, center_x: float) -> RawMask:
    points = np.asarray(
        [
            [center_x - 5.0, -5.0],
            [center_x + 5.0, -5.0],
            [center_x + 5.0, 5.0],
            [center_x - 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    return RawMask(
        frame=frame,
        track_id=track_id,
        geometry=Polygon(points),
        primary_points=points,
        score=1.0,
    )


def _key(raw: RawMask) -> Keyframe:
    return Keyframe(
        raw.frame,
        ((0, Component("polygon", raw.primary_points.tolist())),),
    )


def _rectangle(
    frame: int,
    track_id: str,
    *,
    center_x: float = 0.0,
    half_width: float = 5.0,
    half_height: float = 5.0,
) -> RawMask:
    points = np.asarray(
        [
            [center_x - half_width, -half_height],
            [center_x + half_width, -half_height],
            [center_x + half_width, half_height],
            [center_x - half_width, half_height],
        ],
        dtype=np.float64,
    )
    return RawMask(
        frame=frame,
        track_id=track_id,
        geometry=Polygon(points),
        primary_points=points,
        score=1.0,
    )


def test_superior_border_constraint_is_monotone_and_resolution_aware() -> None:
    points = np.asarray([[0.0, 20.0], [8.0, 20.0], [8.0, 30.0], [0.0, 30.0]])
    raw = RawMask(
        0,
        "1",
        Polygon(points),
        points,
        1.0,
    )
    constraints, summary = expand_border_constraints(
        {(0, "1"): raw},
        width=100,
        height=80,
        config=BorderExpansionConfig(),
    )
    expanded = constraints[(0, "1")]

    assert expanded.geometry.covers(raw.geometry)
    assert float(np.min(expanded.primary_points[:, 0])) < 0.0
    assert summary["changed_masks"] == 1
    assert summary["width"] == 100
    assert summary["height"] == 80


def test_border_constraint_cannot_hide_original_mask_recall_failure() -> None:
    original_points = np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    constraint_points = np.asarray(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 10.0], [0.0, 10.0]]
    )
    # This covers 97% of the larger constraint but only 70% of the original
    # left-most mask.  A union-only Recall check would incorrectly accept it.
    predicted_points = np.asarray(
        [[3.0, 0.0], [100.0, 0.0], [100.0, 10.0], [3.0, 10.0]]
    )
    constraint = RawMask(0, "1", Polygon(constraint_points), constraint_points, 1.0)
    original = RawMask(0, "1", Polygon(original_points), original_points, 1.0)
    keyframe = Keyframe(
        0,
        ((0, Component("polygon", predicted_points.tolist())),),
    )

    recall, _iou = _anchor_metrics(constraint, keyframe, original)

    assert recall == 0.7


def test_superior_stored_vertices_match_aligned_reader_path() -> None:
    left = np.asarray(
        [[0.0, 0.0], [12.0, 0.0], [10.0, 7.0], [1.0, 9.0]],
        dtype=np.float64,
    )
    right = np.asarray(
        [[3.0, 1.0], [14.0, 2.0], [12.0, 10.0], [2.0, 11.0]],
        dtype=np.float64,
    )
    # Deliberately reverse and roll the second key as a real exported SQLite
    # may do before correspondence canonicalization.
    left = _numpy_resample(left, 8)
    right = np.roll(_numpy_resample(right, 8)[::-1], 2, axis=0)
    path = canonicalize_selected_path(
        (
            Keyframe(0, ((0, Component("polygon", left.tolist())),)),
            Keyframe(4, ((0, Component("polygon", right.tolist())),)),
        )
    )
    segment = Segment(1, "1", 0, 4, "linear_polygon_index_v1", path)
    raw = {
        (frame, "1"): RawMask(
            frame,
            "1",
            Polygon(left),
            left,
            1.0,
        )
        for frame in range(5)
    }
    aligned = evaluate_segments(raw, {"1": [segment]})
    direct = evaluate_direct(raw, {"1": [segment]})
    agreement = compare_geometry_paths(aligned, direct)

    assert agreement["symmetric_difference_max_area"] < 1e-8
    assert agreement["nonzero_difference_count"] == 0


def test_target_interval_quality_floor_never_trades_away_baseline_iou() -> None:
    frontier = [
        GlobalParetoPoint(10, 0.1, 10.0, 0.70, 0.97, (0,)),
        GlobalParetoPoint(12, 0.12, 8.0, 0.81, 0.97, (1,)),
        GlobalParetoPoint(15, 0.15, 6.0, 0.86, 0.97, (2,)),
    ]
    selected = _select_frontier_index(
        frontier,
        selection="target_interval_quality_floor",
        preference=0.5,
        key_budget=None,
        target_key_frequency=None,
        target_mean_key_interval=10.0,
        minimum_mean_iou=0.80,
    )

    assert selected == 1


def test_indexed_interpolation_is_preserved_across_missing_observation_gap() -> None:
    left = Keyframe(
        0,
        (
            (
                0,
                Component(
                    "polygon",
                    _numpy_resample(
                        np.asarray([[0.0, 0.0], [8.0, 0.0], [7.0, 7.0], [0.0, 6.0]]),
                        8,
                    ).tolist(),
                ),
            ),
        ),
    )
    right_points = np.roll(
        _numpy_resample(
            np.asarray([[3.0, 1.0], [11.0, 2.0], [10.0, 9.0], [2.0, 8.0]]),
            8,
        )[::-1],
        3,
        axis=0,
    )
    right = canonicalize_selected_path(
        (left, Keyframe(4, ((0, Component("polygon", right_points.tolist())),)))
    )[1]
    expected = 0.5 * (
        np.asarray(left.components[0][1].values)
        + np.asarray(right.components[0][1].values)
    )
    actual = _polygon_components_for_final_frame(
        [left, right],
        {0: (left.components[0][1],), 4: (right.components[0][1],)},
        [0, 4],
        2,
        "linear_polygon_index_v1",
    )

    assert np.allclose(np.asarray(actual[0].values), expected)


def test_temporal_consensus_keeps_supported_area_on_one_frame_contraction() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(
            frame,
            track_id,
            center_x=float(frame),
            half_width=2.0 if frame == 3 else 5.0,
        )
        for frame in range(7)
    }

    result = build_asymmetric_temporal_consensus(
        raw,
        radius=2,
        boundary_tolerance_ratio=0.0,
        minimum_boundary_tolerance_px=0.0,
    )
    trusted = result.trusted_masks[(3, track_id)]
    observed = raw[(3, track_id)]

    assert trusted.geometry.area > observed.geometry.area * 2.0
    assert result.diagnostics[(3, track_id)].classification == "sudden_contraction"


def test_temporal_consensus_rejects_unsupported_one_frame_expansion() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(
            frame,
            track_id,
            center_x=float(frame),
            half_width=12.0 if frame == 3 else 5.0,
        )
        for frame in range(7)
    }

    result = build_asymmetric_temporal_consensus(
        raw,
        radius=2,
        boundary_tolerance_ratio=0.0,
        minimum_boundary_tolerance_px=0.0,
    )
    trusted = result.trusted_masks[(3, track_id)]
    observed = raw[(3, track_id)]

    assert trusted.geometry.area < observed.geometry.area * 0.60
    assert result.diagnostics[(3, track_id)].classification == "sudden_expansion"


def test_temporal_consensus_tracks_constant_velocity_without_smearing() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(
            frame,
            track_id,
            center_x=float(frame * 10),
        )
        for frame in range(7)
    }

    result = build_asymmetric_temporal_consensus(
        raw,
        radius=2,
        boundary_tolerance_ratio=0.0,
        minimum_boundary_tolerance_px=0.0,
    )
    trusted = result.trusted_masks[(3, track_id)]
    observed = raw[(3, track_id)]

    assert trusted.geometry.symmetric_difference(observed.geometry).area < 1e-8
    assert result.diagnostics[(3, track_id)].classification == "temporally_supported"


def test_temporal_consensus_accepts_persistent_expansion() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(
            frame,
            track_id,
            center_x=float(frame),
            half_width=10.0 if 2 <= frame <= 4 else 5.0,
        )
        for frame in range(7)
    }

    result = build_asymmetric_temporal_consensus(
        raw,
        radius=2,
        boundary_tolerance_ratio=0.0,
        minimum_boundary_tolerance_px=0.0,
    )
    trusted = result.trusted_masks[(3, track_id)]

    assert trusted.geometry.area > 180.0
    assert result.diagnostics[(3, track_id)].added_ratio < 0.05


def test_temporal_consensus_uses_one_sided_support_at_track_start() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(
            frame,
            track_id,
            center_x=float(frame),
            half_width=12.0 if frame == 0 else 5.0,
        )
        for frame in range(5)
    }

    result = build_asymmetric_temporal_consensus(
        raw,
        radius=2,
        boundary_tolerance_ratio=0.0,
        minimum_boundary_tolerance_px=0.0,
    )

    assert result.trusted_masks[(0, track_id)].geometry.area < 120.0
    assert result.diagnostics[(0, track_id)].classification == "sudden_expansion"


def test_trusted_selector_takes_large_quality_gain_after_target_budget() -> None:
    endpoints = (
        Keyframe(0, ((0, Component("polygon", [[0, 0], [1, 0], [0, 1]])),)),
        Keyframe(10, ((0, Component("polygon", [[0, 0], [1, 0], [0, 1]])),)),
    )
    frontier = [
        LocalParetoPoint(2, 5.0, 11, 0.97, endpoints, -100.0, 0.50),
        LocalParetoPoint(3, 8.0, 11, 0.97, endpoints, -70.0, 0.85),
        LocalParetoPoint(4, 9.0, 11, 0.97, endpoints, -69.0, 0.90),
    ]

    selected = _select_local_point(frontier, target_mean_key_interval=10.0)

    assert selected == 2


def test_minimax_uses_same_budget_and_captures_periodic_excursion() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        # Center -> right-hand excursion -> center.  Endpoints alone alias the
        # entire motion to a static mask.
        center_x = 20.0 * (1.0 - abs(frame - 5) / 5.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=(
                    _key(raw[(0, track_id)]),
                    _key(raw[(1, track_id)]),
                    _key(raw[(10, track_id)]),
                ),
            )
        ]
    }

    optimized = minimax_recall_positions(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        point_count=8,
    )
    optimized_frames = [item.frame for item in optimized[track_id][0].keyframes]
    assert len(optimized_frames) == 3
    assert 5 in optimized_frames

    baseline_min = min(item.recall for item in evaluate_segments(raw, baseline))
    optimized_min = min(item.recall for item in evaluate_segments(raw, optimized))
    assert optimized_min > baseline_min


def test_adaptive_adds_only_needed_key_and_enforces_dense_recall_floor() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        center_x = 20.0 * (1.0 - abs(frame - 5) / 5.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=(_key(raw[(0, track_id)]), _key(raw[(10, track_id)])),
            )
        ]
    }

    optimized = adaptive_add_recall_keys(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        repair_margin=0.01,
        max_scale=1.05,
    )
    frames = [item.frame for item in optimized[track_id][0].keyframes]
    evaluations = evaluate_segments(raw, optimized)

    assert frames[0] == 0
    assert frames[-1] == 10
    assert 5 in frames
    assert len(frames) < len(raw)
    assert min(item.recall for item in evaluations) >= 0.90


def test_production_recall_guard_retains_production_positions_and_adds_one_key() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        center_x = 20.0 * (1.0 - abs(frame - 5) / 5.0)
        raw[(frame, track_id)] = _square(frame, track_id, center_x)
    left = _key(raw[(0, track_id)])
    right = _key(raw[(10, track_id)])
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=(left, right),
            )
        ]
    }

    result = guard_production_recall(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        guard_margin=0.001,
        point_count=8,
    )
    guarded = result.segments[track_id][0]
    frames = [keyframe.frame for keyframe in guarded.keyframes]
    evaluations = evaluate_segments(raw, result.segments)

    assert frames == [0, 5, 10]
    assert guarded.keyframes[0] == left
    assert guarded.keyframes[-1] == right
    assert result.adjusted_production_keys == 0
    assert result.added_recall_keys == 1
    assert min(item.recall for item in evaluations) >= 0.90


def test_production_recall_guard_repairs_only_violating_production_anchor() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _square(frame, track_id, float(frame)) for frame in range(11)
    }
    left = _key(raw[(0, track_id)])
    wrong_middle = _key(_square(5, track_id, 9.0))
    right = _key(raw[(10, track_id)])
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=(left, wrong_middle, right),
            )
        ]
    }

    result = guard_production_recall(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        guard_margin=0.001,
        point_count=8,
    )
    guarded = result.segments[track_id][0]
    evaluations = evaluate_segments(raw, result.segments)

    assert [keyframe.frame for keyframe in guarded.keyframes] == [0, 5, 10]
    assert guarded.keyframes[0] == left
    assert guarded.keyframes[-1] == right
    assert guarded.keyframes[1] != wrong_middle
    assert result.adjusted_production_keys == 1
    assert result.added_recall_keys == 0
    assert min(item.recall for item in evaluations) >= 0.90


def test_production_recall_guard_preserves_multiple_raw_components() -> None:
    track_id = "1"
    left_points = np.asarray(
        [[-8.0, -2.0], [-4.0, -2.0], [-4.0, 2.0], [-8.0, 2.0]],
        dtype=np.float64,
    )
    right_points = np.asarray(
        [[4.0, -2.0], [8.0, -2.0], [8.0, 2.0], [4.0, 2.0]],
        dtype=np.float64,
    )
    raw = {}
    for frame in range(3):
        raw[(frame, track_id)] = RawMask(
            frame=frame,
            track_id=track_id,
            geometry=MultiPolygon([Polygon(left_points), Polygon(right_points)]),
            primary_points=left_points,
            score=1.0,
            component_points=(left_points, right_points),
        )

    def multi_key(frame: int) -> Keyframe:
        return Keyframe(
            frame,
            (
                (0, Component("polygon", left_points.tolist())),
                (1, Component("polygon", right_points.tolist())),
            ),
        )

    wrong_middle = Keyframe(
        1, ((0, Component("polygon", left_points.tolist())),)
    )
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=2,
                interpolation_method="polygon_linear",
                keyframes=(multi_key(0), wrong_middle, multi_key(2)),
            )
        ]
    }
    result = guard_production_recall(
        baseline,
        raw,
        start_frame=0,
        end_frame=2,
        recall_floor=0.90,
        guard_margin=0.001,
        point_count=8,
    )
    guarded = result.segments[track_id][0]
    middle = next(key for key in guarded.keyframes if key.frame == 1)

    assert len(middle.components) == 2
    assert result.adjusted_production_keys == 1
    assert min(item.recall for item in evaluate_segments(raw, result.segments)) >= 0.90


def test_guard_export_restores_requested_class_policy(tmp_path) -> None:
    baseline = tmp_path / "baseline.sqlite"
    output = tmp_path / "output.sqlite"
    schema = """
        CREATE TABLE class_postprocess_policies(
            label TEXT PRIMARY KEY,
            policy_source TEXT NOT NULL,
            shape_mode TEXT NOT NULL,
            keyframe_interval INTEGER NOT NULL,
            max_gap INTEGER NOT NULL
        )
    """
    for path, interval in ((baseline, 3), (output, 7)):
        with sqlite3.connect(path) as connection:
            connection.execute(schema)
            connection.execute(
                "INSERT INTO class_postprocess_policies VALUES (?, ?, ?, ?, ?)",
                ("男性器", "class", "polygon", interval, 15),
            )
    _restore_requested_class_policy(baseline, output, label="男性器")
    with sqlite3.connect(output) as connection:
        row = connection.execute(
            "SELECT policy_source, shape_mode, keyframe_interval, max_gap "
            "FROM class_postprocess_policies WHERE label='男性器'"
        ).fetchone()
    assert row == ("class", "polygon", 3, 15)


def test_adaptive_split_has_no_interval_target_and_prunes_redundant_keys() -> None:
    track_id = "1"
    raw = {}
    for frame in range(21):
        center_x = 20.0 * (1.0 - abs(frame - 10) / 10.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    production_keys = tuple(_key(raw[(frame, track_id)]) for frame in range(21))
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=20,
                interpolation_method="polygon_linear",
                keyframes=production_keys,
            )
        ]
    }

    optimized = adaptive_split_recall_keys(
        baseline,
        raw,
        start_frame=0,
        end_frame=20,
        recall_floor=0.90,
        anchor_margin=0.05,
    )
    frames = [item.frame for item in optimized[track_id][0].keyframes]
    evaluations = evaluate_segments(raw, optimized)

    assert frames == [0, 10, 20]
    assert min(item.recall for item in evaluations) >= 0.90


def test_projected_smoothing_preserves_key_recall_floor() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _square(frame, track_id, float(frame)) for frame in range(5)
    }
    noisy_centers = [0.0, 1.5, 1.5, 3.5, 4.0]
    keys = tuple(
        _key(_square(frame, track_id, center))
        for frame, center in enumerate(noisy_centers)
    )
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=4,
                interpolation_method="polygon_linear",
                keyframes=keys,
            )
        ]
    }

    smoothed = projected_temporal_smooth(
        baseline,
        raw,
        key_recall_floor=0.90,
        strength=0.75,
        iterations=2,
    )
    evaluations = evaluate_segments(raw, smoothed)

    assert min(item.recall for item in evaluations if item.is_keyframe) >= 0.90


def test_lexicographic_optimizer_uses_no_more_keys_and_enforces_floor() -> None:
    track_id = "1"
    raw = {}
    for frame in range(21):
        center_x = 20.0 * (1.0 - abs(frame - 10) / 10.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    production_keys = tuple(_key(raw[(frame, track_id)]) for frame in range(21))
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=20,
                interpolation_method="polygon_linear",
                keyframes=production_keys,
            )
        ]
    }

    optimized = lexicographic_recall_stability_optimize(
        baseline,
        raw,
        start_frame=0,
        end_frame=20,
        recall_floor=0.90,
    )
    evaluations = evaluate_segments(raw, optimized)

    assert len(optimized[track_id][0].keyframes) <= len(production_keys)
    assert min(item.recall for item in evaluations) >= 0.90


def test_pareto_dp_keeps_key_frequency_iou_front_under_fixed_recall() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        center_x = 4.0 * (1.0 - abs(frame - 5) / 5.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    production_keys = tuple(_key(raw[(frame, track_id)]) for frame in range(11))
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=production_keys,
            )
        ]
    }

    result = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.50,
        max_edge_span_frames=20,
        selection="max_iou",
        point_count=8,
    )

    assert [point.keyframe_count for point in result.frontier] == [2, 3]
    assert result.frontier[0].mean_iou < result.frontier[1].mean_iou
    assert result.selected.keyframe_count == 3
    assert result.selected.min_recall >= 0.50
    evaluations = evaluate_segments(raw, result.segments)
    assert min(item.recall for item in evaluations) >= 0.50


def test_pareto_dp_removes_infeasible_low_key_solution() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        center_x = 20.0 * (1.0 - abs(frame - 5) / 5.0)
        mask = _square(frame, track_id, center_x)
        raw[(frame, track_id)] = mask
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=tuple(_key(raw[(frame, track_id)]) for frame in range(11)),
            )
        ]
    }

    result = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        selection="min_keys",
        point_count=8,
    )

    assert result.selected.keyframe_count == 3
    assert result.selected.min_recall >= 0.90


def test_pareto_edge_evaluates_exact_right_key_without_resampling_it() -> None:
    track_id = "1"
    left = _square(0, track_id, 0.0)
    irregular_points = np.asarray(
        [
            [-7.0, -3.0],
            [2.0, -7.0],
            [9.0, -1.0],
            [5.0, 8.0],
            [-4.0, 6.0],
        ],
        dtype=np.float64,
    )
    right = RawMask(
        frame=1,
        track_id=track_id,
        geometry=Polygon(irregular_points),
        primary_points=irregular_points,
        score=1.0,
    )
    edge = _evaluate_edge(
        0,
        1,
        [0, 1],
        [_key(left), _key(right)],
        {0: left, 1: right},
        recall_floor=0.999999,
    )

    assert edge is not None
    assert edge.min_recall >= 0.999999
    assert edge.iou_sum >= 0.999999


def test_cached_right_anchor_metrics_are_edge_exact() -> None:
    raw = {(frame, "1"): _square(frame, "1", float(frame)) for frame in range(3)}
    left = _key(raw[(0, "1")])
    right = _key(raw[(2, "1")])
    frames = [0, 1, 2]
    raw_by_frame = {frame: raw[(frame, "1")] for frame in frames}
    raw_areas = {
        frame: float(item.geometry.area) for frame, item in raw_by_frame.items()
    }
    raw_bounds = {frame: item.geometry.bounds for frame, item in raw_by_frame.items()}
    arguments = (
        0,
        0,
        2,
        0,
        frames,
        _prepare_anchor(left),
        _prepare_anchor(right),
        raw_by_frame,
        raw_areas,
        raw_bounds,
        raw_by_frame,
        raw_areas,
    )

    uncached = _evaluate_anchor_edge(*arguments, recall_floor=0.90)
    cached = _evaluate_anchor_edge(
        *arguments,
        right_anchor_metrics=_anchor_metrics(raw_by_frame[2], right, raw_by_frame[2]),
        recall_floor=0.90,
    )

    assert cached == uncached


def test_fast_alignment_matches_production_alignment() -> None:
    random = np.random.default_rng(20260806)
    for count in (8, 17, 23):
        reference = random.normal(size=(count, 2))
        candidate = random.normal(size=(count, 2))
        expected = _numpy_align(reference, candidate)
        actual = _fast_numpy_align(reference, candidate)
        assert np.array_equal(actual, expected)


def test_multiple_anchor_shapes_can_trade_iou_for_fewer_keys() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        half_size = 5.0 + 2.0 * (1.0 - abs(frame - 5) / 5.0)
        points = np.asarray(
            [
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size],
            ],
            dtype=np.float64,
        )
        raw[(frame, track_id)] = RawMask(
            frame=frame,
            track_id=track_id,
            geometry=Polygon(points),
            primary_points=points,
            score=1.0,
        )
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=10,
                interpolation_method="polygon_linear",
                keyframes=tuple(_key(raw[(frame, track_id)]) for frame in range(11)),
            )
        ]
    }
    single = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=1,
        selection="min_keys",
    )
    multiple = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=3,
        anchor_expansion=0.50,
        selection="min_keys",
    )
    scalar_multiple = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=3,
        anchor_expansion=0.50,
        edge_batch_size=1,
        selection="min_keys",
    )
    target_only = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=3,
        anchor_expansion=0.50,
        target_mean_key_interval=10.0,
        solver_mode="target_only",
    )
    parallel = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=3,
        anchor_expansion=0.50,
        edge_processes=2,
        selection="min_keys",
    )

    assert single.selected.keyframe_count == 3
    assert multiple.selected.keyframe_count == 2
    assert multiple.selected.min_recall >= 0.90
    assert multiple.selected.mean_iou > 0.75
    assert multiple.frontier == scalar_multiple.frontier
    assert multiple.segments == scalar_multiple.segments
    assert len(target_only.frontier) == 1
    assert target_only.selected == next(
        point
        for point in multiple.frontier
        if point.keyframe_count == target_only.selected.keyframe_count
    )
    assert target_only.segments == multiple.segments
    assert parallel.frontier == multiple.frontier
    assert parallel.segments == multiple.segments


def test_pair_vote_is_additive_and_cannot_displace_core_anchor_states() -> None:
    raw = _rectangle(0, "1", half_width=5.0, half_height=4.0)
    segment = Segment(
        segment_id=1,
        track_id="1",
        first_frame=0,
        last_frame=0,
        interpolation_method="polygon_linear",
        keyframes=(_key(raw),),
    )
    proposal_points = np.asarray(
        [[-6.0, -4.0], [5.5, -4.5], [6.0, 4.0], [-5.5, 4.5]],
        dtype=np.float64,
    )
    proposal = Keyframe(
        0,
        ((0, Component("polygon", proposal_points.tolist())),),
    )
    core = _make_feasible_anchors(
        segment,
        raw,
        quality_raw=raw,
        recall_floor=0.97,
        point_count=8,
        max_anchor_scale=1.25,
        anchor_state_count=4,
        anchor_expansion=0.30,
        stored_vertex_contract=True,
    )
    augmented = _make_feasible_anchors(
        segment,
        raw,
        quality_raw=raw,
        recall_floor=0.97,
        point_count=8,
        max_anchor_scale=1.25,
        anchor_state_count=4,
        anchor_expansion=0.30,
        stored_vertex_contract=True,
        extra_sources=[proposal],
    )

    assert len(augmented) >= len(core)
    for expected, actual in zip(core, augmented[: len(core)], strict=True):
        assert np.array_equal(
            np.asarray(expected.components[0][1].values),
            np.asarray(actual.components[0][1].values),
        )


def test_pair_vote_sources_are_fixed_count_and_frame_local() -> None:
    raw = {
        frame: _rectangle(
            frame,
            "1",
            center_x=float(frame),
            half_width=5.0 + 0.2 * frame,
        )
        for frame in range(7)
    }
    proposals = _build_pair_vote_sources(
        list(range(7)), raw, point_count=8, max_edge_span_frames=6
    )

    assert proposals
    for frame, values in proposals.items():
        assert 0 <= frame <= 6
        assert all(value.frame == frame for value in values)
        assert all(len(value.components[0][1].values) == 8 for value in values)


def test_only_selected_stored_vertex_path_is_canonicalized() -> None:
    track_id = "1"
    raw = {
        (frame, track_id): _rectangle(frame, track_id, center_x=float(frame))
        for frame in range(5)
    }
    baseline = {
        track_id: [
            Segment(
                segment_id=1,
                track_id=track_id,
                first_frame=0,
                last_frame=4,
                interpolation_method="linear_polygon_aligned_v1",
                keyframes=tuple(_key(raw[(frame, track_id)]) for frame in range(5)),
            )
        ]
    }
    result = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=4,
        recall_floor=0.90,
        max_edge_span_frames=4,
        point_count=8,
        anchor_state_count=2,
        selection="target_interval",
        target_mean_key_interval=2.0,
        stored_vertex_contract=True,
    )
    selected = result.segments[track_id][0]

    assert selected.interpolation_method == "linear_polygon_index_v1"
    previous = None
    for keyframe in selected.keyframes:
        points = np.asarray(keyframe.components[0][1].values)
        if previous is not None:
            assert np.array_equal(points, _fast_numpy_align(previous, points))
        previous = points


def test_target_interval_selects_nearest_pareto_point_without_relaxing_recall() -> None:
    frontier = [
        GlobalParetoPoint(10, 0.10, 11.0, 0.80, 0.97, (0,)),
        GlobalParetoPoint(12, 0.12, 9.0, 0.85, 0.97, (1,)),
        GlobalParetoPoint(20, 0.20, 5.0, 0.92, 0.97, (2,)),
    ]

    index = _select_frontier_index(
        frontier,
        selection="target_interval",
        preference=0.5,
        key_budget=None,
        target_key_frequency=None,
        target_mean_key_interval=10.0,
    )

    # Both first points are one frame away; the higher-IoU point wins.
    assert index == 1


def test_unreachable_target_interval_selects_nearest_feasible_boundary() -> None:
    frontier = [
        GlobalParetoPoint(10, 0.10, 9.0, 0.80, 0.97, (0,)),
        GlobalParetoPoint(20, 0.20, 5.0, 0.92, 0.97, (1,)),
    ]

    index = _select_frontier_index(
        frontier,
        selection="target_interval",
        preference=0.5,
        key_budget=None,
        target_key_frequency=None,
        target_mean_key_interval=10.0,
    )

    assert index == 0


def test_curvature_preserving_raw_key_beats_uniform_resampling() -> None:
    track_id = "1"
    points = np.asarray(
        [
            [0.0, 0.0],
            [40.0, 0.0],
            [40.0, 10.0],
            [25.0, 10.0],
            [24.0, 35.0],
            [23.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )
    raw = RawMask(0, track_id, Polygon(points), points, 1.0)
    uniform = _keyframe_geometry(_raw_keyframe(raw, point_count=8))
    simplified = _keyframe_geometry(
        _raw_keyframe(raw, point_count=8, point_strategy="simplify_budget")
    )

    assert raw.geometry.hausdorff_distance(simplified) == 0.0
    assert raw.geometry.hausdorff_distance(uniform) > 0.0


def test_hard_iou_and_boundary_floors_reject_bad_interpolation_edges() -> None:
    track_id = "1"
    raw = {}
    for frame in range(11):
        half_size = 5.0 + 2.0 * (1.0 - abs(frame - 5) / 5.0)
        points = np.asarray(
            [
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size],
            ],
            dtype=np.float64,
        )
        raw[(frame, track_id)] = RawMask(frame, track_id, Polygon(points), points, 1.0)
    baseline = {
        track_id: [
            Segment(
                1,
                track_id,
                0,
                10,
                "polygon_linear",
                tuple(_key(raw[(frame, track_id)]) for frame in range(11)),
            )
        ]
    }

    result = optimize_pareto_frontier(
        baseline,
        raw,
        start_frame=0,
        end_frame=10,
        recall_floor=0.90,
        anchor_iou_floor=0.90,
        frame_iou_floor=0.90,
        max_frame_hausdorff_px=2.0,
        max_edge_span_frames=20,
        point_count=8,
        anchor_state_count=3,
        anchor_expansion=0.50,
        selection="min_keys",
    )
    evaluations = evaluate_segments(raw, result.segments)

    assert min(item.recall for item in evaluations) >= 0.90
    assert min(item.iou for item in evaluations) >= 0.90
    assert (
        max(
            item.raw_geometry.hausdorff_distance(item.predicted_geometry)
            for item in evaluations
        )
        <= 2.0
    )


def test_vertex_safety_audit_accepts_canonical_point_index_motion() -> None:
    left = _square(0, "1", 0.0)
    right = _square(10, "1", 4.0)
    segment = Segment(
        1,
        "1",
        0,
        10,
        "linear_polygon_index_v1",
        (_key(left), _key(right)),
    )

    audit = _vertex_safety_audit({"1": [segment]})

    assert audit["adjacent_best_alignment_reversal_count"] == 0
    assert audit["adjacent_best_alignment_nonzero_shift_count"] == 0
    assert audit["invalid_integer_frame_count"] == 0
    assert audit["invalid_fractional_sample_count"] == 0


def test_vertex_safety_audit_rejects_shifted_vertex_correspondence() -> None:
    left = _square(0, "1", 0.0)
    right_points = np.roll(left.primary_points + np.asarray([4.0, 0.0]), 1, axis=0)
    right = Keyframe(
        10,
        ((0, Component("polygon", right_points.tolist())),),
    )
    segment = Segment(
        1,
        "1",
        0,
        10,
        "linear_polygon_index_v1",
        (_key(left), right),
    )

    audit = _vertex_safety_audit({"1": [segment]})

    assert audit["adjacent_best_alignment_nonzero_shift_count"] == 1
    assert audit["non_rigid_motion"]["maximum"] > 0.0


def test_border_safety_requires_visible_strip_recall_and_offcanvas_extent() -> None:
    points = np.asarray(
        [[80.0, 20.0], [99.0, 20.0], [99.0, 40.0], [80.0, 40.0]],
        dtype=np.float64,
    )
    raw = {(0, "1"): RawMask(0, "1", Polygon(points), points, 1.0)}
    config = BorderExpansionConfig()
    expanded, _summary = expand_border_constraints(
        raw, width=100, height=60, config=config
    )
    constraints, summary = build_border_safety_constraints(
        raw, expanded, width=100, height=60, config=config
    )
    constraint = constraints[(0, "1")]

    assert summary["side_counts"]["right"] == 1
    assert not border_geometry_feasible(raw[(0, "1")].geometry, constraint)
    expanded_geometry = Polygon(expanded[(0, "1")].primary_points)
    assert border_geometry_feasible(expanded_geometry, constraint)
    assert expanded_geometry.bounds[2] >= 105.0


def test_pareto_enforces_border_safety_on_every_interpolated_frame() -> None:
    raw: dict[tuple[int, str], RawMask] = {}
    for frame in range(5):
        points = np.asarray(
            [[80.0, 20.0], [99.0, 20.0], [99.0, 40.0], [80.0, 40.0]],
            dtype=np.float64,
        )
        raw[(frame, "1")] = RawMask(frame, "1", Polygon(points), points, 1.0)
    baseline = {
        "1": [
            Segment(
                1,
                "1",
                0,
                4,
                "linear_polygon_index_v1",
                tuple(_key(raw[(frame, "1")]) for frame in range(5)),
            )
        ]
    }
    config = BorderExpansionConfig()
    expanded, _summary = expand_border_constraints(
        raw, width=100, height=60, config=config
    )
    constraints, _summary = build_border_safety_constraints(
        raw, expanded, width=100, height=60, config=config
    )

    result = optimize_pareto_frontier(
        baseline,
        expanded,
        quality_masks=raw,
        border_constraints=constraints,
        visible_bounds=(0.0, 0.0, 100.0, 60.0),
        start_frame=0,
        end_frame=4,
        recall_floor=0.97,
        max_edge_span_frames=10,
        point_count=8,
        anchor_state_count=2,
        anchor_expansion=0.20,
        selection="min_keys",
        stored_vertex_contract=True,
    )
    audit = audit_border_safety(constraints, result.segments)

    assert result.selected.keyframe_count == 2
    assert audit["passed"]
    assert audit["minimum_local_recall"] >= 0.995
    rows = evaluate_segments(
        raw,
        result.segments,
        visible_rectangle=Polygon(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]]
        ),
        border_constraints=constraints,
    )
    assert len(rows) == 5
