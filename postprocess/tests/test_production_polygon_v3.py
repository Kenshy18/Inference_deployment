from __future__ import annotations

import numpy as np
from shapely.geometry import box

from experimental.polygon_recall_optimizer.fixed_budget import (
    Component,
    Keyframe,
    RawMask,
    Segment,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderFrameConstraint,
    BorderSideConstraint,
)
from experimental.production_polygon_v3.optimizer import _edge_metrics
from experimental.production_polygon_v3.penalty import _Edge, _Graph, _decode_graph


def _key(frame: int, x0: float, y0: float, x1: float, y1: float) -> Keyframe:
    return Keyframe(
        frame,
        (
            (
                0,
                Component(
                    "polygon",
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                ),
            ),
        ),
    )


def _raw(frame: int, x0: float, y0: float, x1: float, y1: float) -> RawMask:
    geometry = box(x0, y0, x1, y1)
    points = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
    return RawMask(frame, "1", geometry, points, 1.0, (points,))


def _segment(left: Keyframe, right: Keyframe) -> Segment:
    return Segment(
        segment_id=1,
        track_id="1",
        first_frame=left.frame,
        last_frame=right.frame,
        interpolation_method="linear_polygon_index_v1",
        keyframes=(left, right),
    )


def test_normal_recall_is_a_per_frame_minimum() -> None:
    left = _key(0, 0, 0, 10, 10)
    right = _key(2, 0, 0, 10, 10)
    raw = {
        0: _raw(0, 0, 0, 10, 10),
        1: _raw(1, 0, 0, 20, 10),
        2: _raw(2, 0, 0, 10, 10),
    }
    feasible, _iou, minimum, _border, _count = _edge_metrics(
        _segment(left, right),
        left,
        right,
        raw,
        {},
        normal_recall_floor=0.97,
    )
    assert not feasible
    assert minimum == 0.5


def test_border_recall_and_extent_are_independent_hard_constraints() -> None:
    left = _key(0, 0, 0, 10, 10)
    right = _key(2, 0, 0, 10, 10)
    raw = {frame: _raw(frame, 0, 0, 10, 10) for frame in range(3)}
    border = BorderFrameConstraint(
        sides=(
            BorderSideConstraint(
                side="left",
                visible_reference=box(0, 0, 10, 10),
                visible_area=100.0,
                required_coordinate=-6.0,
            ),
        ),
        local_recall_floor=0.97,
        max_repair_px=40.0,
        quality_domain=box(10, 0, 100, 100),
    )
    feasible, _iou, normal, local, _count = _edge_metrics(
        _segment(left, right),
        left,
        right,
        raw,
        {1: border},
        normal_recall_floor=0.97,
    )
    assert not feasible
    assert normal == 1.0
    assert local == 1.0


def test_border_safe_edge_passes_both_minimum_constraints() -> None:
    left = _key(0, -6, 0, 10, 10)
    right = _key(2, -6, 0, 10, 10)
    raw = {frame: _raw(frame, 0, 0, 10, 10) for frame in range(3)}
    border = BorderFrameConstraint(
        sides=(
            BorderSideConstraint(
                side="left",
                visible_reference=box(0, 0, 10, 10),
                visible_area=100.0,
                required_coordinate=-6.0,
            ),
        ),
        local_recall_floor=0.97,
        max_repair_px=40.0,
        quality_domain=box(10, 0, 100, 100),
    )
    feasible, _iou, normal, local, _count = _edge_metrics(
        _segment(left, right),
        left,
        right,
        raw,
        {1: border},
        normal_recall_floor=0.97,
    )
    assert feasible
    assert normal == 1.0
    assert local == 1.0


def test_soft_key_penalty_can_add_or_remove_a_key() -> None:
    first = _key(0, 0, 0, 10, 10)
    middle = _key(1, 0, 0, 10, 10)
    last = _key(2, 0, 0, 10, 10)
    segment = Segment(
        segment_id=1,
        track_id="1",
        first_frame=0,
        last_frame=2,
        interpolation_method="linear_polygon_index_v1",
        keyframes=(first, middle, last),
    )
    graph = _Graph(
        segment=segment,
        keys=(first, middle, last),
        incoming=(
            (),
            (_Edge(0, 1, 0.1),),
            (_Edge(0, 2, 1.0), _Edge(1, 2, 0.1)),
        ),
    )
    _fine, fine_count, fine_loss = _decode_graph(graph, penalty=0.1)
    _sparse, sparse_count, sparse_loss = _decode_graph(graph, penalty=2.0)
    assert fine_count == 3
    assert fine_loss == 0.2
    assert sparse_count == 2
    assert sparse_loss == 1.0
