from __future__ import annotations

import numpy as np
import experimental.polygon_recall_optimizer.pareto_dp as pareto_dp
from shapely.geometry import Polygon
from types import SimpleNamespace

from experimental.polygon_candidate_v4_20260809.low_dim_refinement import (
    transform_low_dim,
)
from experimental.polygon_candidate_v4_20260809.run_full_v4 import (
    _distributed_targets,
)
from experimental.polygon_recall_optimizer.fixed_budget import RawMask, Segment
from experimental.polygon_recall_optimizer.pareto_dp import (
    _corner_preserving_keyframe,
    _keyframe_geometry,
)
from experimental.polygon_recall_optimizer.geometric_candidates import (
    build_geometric_candidates,
)
from overlay_renderer.keyframe_cache import Component, Keyframe


def _square() -> Keyframe:
    return Keyframe(
        7,
        (
            (
                0,
                Component(
                    "polygon",
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                ),
            ),
        ),
    )


def test_affine_parameters_translate_in_normalized_object_units():
    transformed = transform_low_dim(
        _square(), np.asarray([0.1, -0.1, 0.0, 0.0, 0.0, 0.0]), normal_control_count=0
    )
    points = np.asarray(transformed.components[0][1].values)
    assert np.allclose(points, [[1.0, -1.0], [11.0, -1.0], [11.0, 9.0], [1.0, 9.0]])


def test_positive_normal_controls_expand_counter_clockwise_polygon():
    original = Polygon(_square().components[0][1].values)
    transformed = transform_low_dim(
        _square(), np.asarray([0.0] * 6 + [0.1] * 4), normal_control_count=4
    )
    expanded = Polygon(transformed.components[0][1].values)
    assert expanded.area > original.area
    assert expanded.contains(original)


def test_geometric_candidate_family_contains_axis_and_envelope_hypotheses():
    points = np.asarray(_square().components[0][1].values, dtype=np.float64)
    raw = {
        frame: RawMask(
            frame=frame,
            track_id="1",
            geometry=Polygon(points * (1.0 + 0.01 * frame)),
            primary_points=points * (1.0 + 0.01 * frame),
            score=0.9,
            component_points=(points * (1.0 + 0.01 * frame),),
        )
        for frame in range(5)
    }
    candidates = build_geometric_candidates(2, raw, point_count=23, radius=2)
    assert {value.name for value in candidates} == {
        "axis_major",
        "axis_minor",
        "axis_balanced",
        "axis_envelope",
    }


def test_capped_refinement_budget_covers_each_segment_before_global_ranking():
    segments = {
        "1": [
            SimpleNamespace(
                segment_id=10,
                track_id="1",
                keyframes=(Keyframe(0, ()), Keyframe(2, ())),
            )
        ],
        "2": [
            SimpleNamespace(
                segment_id=20,
                track_id="2",
                keyframes=(Keyframe(10, ()), Keyframe(12, ())),
            )
        ],
    }
    rows = [
        SimpleNamespace(segment_id=10, frame=frame, iou=0.9, area_ratio=1.0)
        for frame in range(3)
    ] + [
        SimpleNamespace(segment_id=20, frame=frame, iou=0.8, area_ratio=1.0)
        for frame in range(10, 13)
    ]
    targets = _distributed_targets(segments, rows, maximum=2)
    assert {track_id for _frame, track_id in targets} == {"1", "2"}


def test_corner_preserving_keyframe_keeps_fixed_vertex_contract():
    angles = np.linspace(0.0, 2.0 * np.pi, 151, endpoint=False)
    radii = np.full_like(angles, 20.0)
    radii[0] = 45.0
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    raw = RawMask(
        frame=9,
        track_id="7",
        geometry=Polygon(points),
        primary_points=points,
        score=0.9,
        component_points=(points,),
    )

    keyframe = _corner_preserving_keyframe(raw, 23)

    assert keyframe is not None
    assert len(keyframe.components[0][1].values) == 23
    assert _keyframe_geometry(keyframe).is_valid


def test_optional_temporal_candidate_failure_falls_back_to_legacy(monkeypatch):
    raw_values = {}
    for frame in range(3):
        points = np.asarray(
            [
                [frame, 0.0],
                [frame + 10.0, 0.0],
                [frame + 10.0, 10.0],
                [frame, 10.0],
            ],
            dtype=np.float64,
        )
        raw_values[(frame, "1")] = RawMask(
            frame=frame,
            track_id="1",
            geometry=Polygon(points),
            primary_points=points,
            score=0.9,
            component_points=(points,),
        )
    segment = Segment(
        segment_id=1,
        track_id="1",
        first_frame=0,
        last_frame=2,
        interpolation_method="linear_polygon_index_v1",
        keyframes=tuple(
            Keyframe(
                frame,
                (
                    (
                        0,
                        Component(
                            "polygon",
                            raw_values[(frame, "1")].primary_points.tolist(),
                        ),
                    ),
                ),
            )
            for frame in (0, 2)
        ),
    )

    def no_temporal_candidate(*_args, **_kwargs):
        raise RuntimeError("synthetic optional candidate failure")

    monkeypatch.setattr(
        pareto_dp, "_make_temporal7_anchors", no_temporal_candidate
    )
    frontier, _edges, _feasible, _states = pareto_dp.optimize_segment_pareto(
        segment,
        raw_values,
        quality_masks=raw_values,
        start_frame=0,
        end_frame=2,
        recall_floor=0.90,
        max_edge_span_frames=2,
        point_count=8,
        anchor_state_count=1,
        stored_vertex_contract=True,
        candidate_mode="legacy_temporal_recall_interior",
    )

    assert frontier
