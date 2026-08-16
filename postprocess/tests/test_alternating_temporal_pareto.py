from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box

from overlay_renderer.keyframe_cache import Component, Keyframe

from experimental.alternating_temporal_pareto.refinement import _blend_keyframe
from experimental.alternating_temporal_pareto.independent_border import (
    build_independent_border_constraints,
)
from experimental.polygon_recall_optimizer.fixed_budget import RawMask, Segment
from experimental.polygon_recall_optimizer.pareto_dp import (
    _keyframe_geometry,
    _minimum_directional_border_anchor,
    optimize_segment_pareto,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    border_geometry_feasible,
)
from experimental.polygon_recall_optimizer.temporal_candidates import (
    build_temporal_candidates,
)


def _raw_masks(count: int = 12):
    output = {}
    for frame in range(count):
        left = 1.5 * frame
        points = np.asarray(
            [[left, 0.0], [left + 20.0, 0.0], [left + 20.0, 10.0], [left, 10.0]],
            dtype=np.float64,
        )
        output[(frame, "1")] = RawMask(
            frame=frame,
            track_id="1",
            geometry=box(left, 0.0, left + 20.0, 10.0),
            primary_points=points,
            score=0.9,
            component_points=(points,),
        )
    return output


def _key(frame: int, left: float) -> Keyframe:
    return Keyframe(
        frame,
        (
            (
                0,
                Component(
                    "polygon",
                    [
                        [left, 0.0],
                        [left + 20.0, 0.0],
                        [left + 20.0, 10.0],
                        [left, 10.0],
                    ],
                ),
            ),
        ),
    )


def test_temporal_candidate_contract_has_seven_production_independent_states():
    raw = {frame: value for (frame, _track), value in _raw_masks().items()}
    candidates = build_temporal_candidates(6, raw, point_count=23)
    assert [candidate.name for candidate in candidates] == [
        "initial_raw",
        "short_iou",
        "short_recall",
        "medium_iou",
        "medium_recall",
        "long_iou",
        "long_recall",
    ]
    reference_center = raw[6].geometry.centroid
    for candidate in candidates:
        polygon = Polygon(candidate.keyframe.components[0][1].values)
        # Uniform perimeter resampling need not preserve the exact analytic
        # centroid, but rigid temporal alignment must remove the frame motion.
        assert abs(polygon.centroid.x - reference_center.x) < 0.1
        assert abs(polygon.centroid.y - reference_center.y) < 0.1


def test_temporal7_dp_does_not_depend_on_input_keyframe_shapes():
    raw = _raw_masks()
    first = Segment(
        1,
        "1",
        0,
        11,
        "linear_polygon_index_v1",
        (_key(0, 0.0), _key(11, 16.5)),
    )
    unrelated = Segment(
        1,
        "1",
        0,
        11,
        "linear_polygon_index_v1",
        (_key(0, 500.0), _key(11, -500.0)),
    )
    common = dict(
        quality_masks=raw,
        start_frame=0,
        end_frame=11,
        recall_floor=0.97,
        max_edge_span_frames=12,
        point_count=23,
        stored_vertex_contract=True,
        candidate_mode="temporal7",
    )
    left = optimize_segment_pareto(first, raw, **common)[0]
    right = optimize_segment_pareto(unrelated, raw, **common)[0]
    assert [(point.keyframe_count, point.iou_sum) for point in left] == [
        (point.keyframe_count, point.iou_sum) for point in right
    ]
    assert [
        [key.components for key in point.keyframes] for point in left
    ] == [[key.components for key in point.keyframes] for point in right]


def test_legacy_temporal_union_cannot_displace_legacy_pareto_paths():
    raw = _raw_masks()
    segment = Segment(
        1,
        "1",
        0,
        11,
        "linear_polygon_index_v1",
        (_key(0, 0.0), _key(11, 16.5)),
    )
    common = dict(
        quality_masks=raw,
        start_frame=0,
        end_frame=11,
        recall_floor=0.97,
        max_edge_span_frames=12,
        point_count=23,
        anchor_state_count=4,
        anchor_expansion=0.30,
        pair_vote_states=True,
        stored_vertex_contract=True,
    )
    legacy = optimize_segment_pareto(
        segment, raw, candidate_mode="legacy", **common
    )[0]
    union = optimize_segment_pareto(
        segment, raw, candidate_mode="legacy_temporal_union", **common
    )[0]
    union_by_keys = {point.keyframe_count: point for point in union}
    for point in legacy:
        assert point.keyframe_count in union_by_keys
        assert union_by_keys[point.keyframe_count].iou_sum + 1e-9 >= point.iou_sum
        assert union_by_keys[point.keyframe_count].min_recall + 1e-9 >= 0.97


def test_shape_refinement_blend_is_continuous_vertex_interpolation():
    left = _key(4, 0.0)
    right = _key(4, 10.0)
    blended = _blend_keyframe(4, left, right, 0.25)
    points = np.asarray(blended.components[0][1].values)
    assert np.allclose(points[:, 0], np.asarray([2.5, 22.5, 22.5, 2.5]))


def test_independent_border_repair_extends_only_the_touched_side():
    points = np.asarray(
        [[20.0, 70.0], [80.0, 70.0], [80.0, 99.0], [20.0, 99.0]],
        dtype=np.float64,
    )
    raw = RawMask(
        frame=0,
        track_id="1",
        geometry=Polygon(points),
        primary_points=points,
        score=0.9,
        component_points=(points,),
    )
    constraints, borders, preparation = build_independent_border_constraints(
        {(0, "1"): raw},
        width=100,
        height=100,
        config=BorderExpansionConfig(
            trigger_px=2.0,
            expand_ratio=0.2,
            min_expand_px=4.0,
            max_expand_px=12.0,
            influence_px=16.0,
        ),
        local_recall_floor=0.97,
    )
    assert constraints[(0, "1")] is raw
    assert preparation["production_transform_used"] is False
    source = Keyframe(0, ((0, Component("polygon", points.tolist())),))
    repaired = _minimum_directional_border_anchor(
        source,
        raw,
        raw,
        0.97,
        borders[(0, "1")],
        box(0.0, 0.0, 100.0, 100.0),
        max_anchor_scale=1.25,
    )
    assert repaired is not None
    geometry = _keyframe_geometry(repaired)
    assert border_geometry_feasible(geometry, borders[(0, "1")])
    assert geometry.bounds[1] >= 69.0
