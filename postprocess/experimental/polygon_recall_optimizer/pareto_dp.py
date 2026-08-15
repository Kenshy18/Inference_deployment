"""Recall-constrained Pareto dynamic programming for polygon keyframes.

The production optimizer targets a requested key count through scalar penalties
and can alter key shapes after decoding.  This experimental solver instead:

* treats the requested minimum per-frame recall as a hard edge constraint;
* retains the best cumulative IoU for every reachable key count;
* removes dominated states exactly at every DAG node;
* combines independent track-segment frontiers with a second Pareto DP; and
* selects a knee (or an explicit preference) only after preserving the front.

Only SQLite polygon geometry is used.  Video pixels are not opened.
"""

from __future__ import annotations

import bisect
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import shapely
from overlay_renderer.keyframe_cache import (
    Component,
    Keyframe,
    _numpy_resample,
)

from .fixed_budget import (
    RawMask,
    Segment,
    _primary_component,
    _raw_keyframe,
    _safe_anchor_keyframe,
    geometry_from_arrays,
)
from .geometric_candidates import build_geometric_candidates
from .superior import BorderFrameConstraint, border_geometry_feasible
from .temporal_candidates import build_temporal_candidates


@dataclass(frozen=True)
class EdgeMetrics:
    start_index: int
    start_anchor_index: int
    end_index: int
    end_anchor_index: int
    iou_sum: float
    frame_count: int
    min_recall: float
    quality_sum: float = 0.0
    min_iou: float = 1.0


@dataclass(frozen=True)
class LocalParetoPoint:
    keyframe_count: int
    iou_sum: float
    frame_count: int
    min_recall: float
    keyframes: tuple[Keyframe, ...]
    quality_sum: float = 0.0
    min_iou: float = 1.0

    @property
    def mean_iou(self) -> float:
        return self.iou_sum / max(self.frame_count, 1)


@dataclass(frozen=True)
class GlobalParetoPoint:
    keyframe_count: int
    key_frequency: float
    mean_key_interval: float
    mean_iou: float
    min_recall: float
    local_point_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParetoOptimizationResult:
    recall_floor: float
    frontier: tuple[GlobalParetoPoint, ...]
    selected_index: int
    segments: dict[str, list[Segment]]
    raw_frame_count: int
    anchor_state_total: int
    edge_evaluations: int
    feasible_edges: int
    worker_count: int
    elapsed_seconds: float

    @property
    def selected(self) -> GlobalParetoPoint:
        return self.frontier[self.selected_index]


@dataclass(frozen=True)
class _NodeState:
    iou_sum: float
    quality_sum: float
    min_iou: float
    min_recall: float
    current_anchor_index: int
    previous_index: int
    previous_anchor_index: int
    previous_key_count: int


@dataclass(frozen=True)
class _GlobalState:
    iou_sum: float
    min_recall: float
    previous_key_count: int
    local_point_index: int


@dataclass(frozen=True)
class _PreparedAnchor:
    keyframe: Keyframe
    values: np.ndarray
    sampled: np.ndarray
    geometry: object
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class _SegmentTask:
    segment: Segment
    raw_masks: dict[tuple[int, str], RawMask]
    quality_masks: dict[tuple[int, str], RawMask] | None
    border_constraints: dict[tuple[int, str], BorderFrameConstraint] | None
    visible_bounds: tuple[float, float, float, float] | None
    start_frame: int
    end_frame: int
    recall_floor: float
    anchor_iou_floor: float
    anchor_relative_iou_margin: float | None
    frame_iou_floor: float
    anchor_point_strategy: str
    max_frame_hausdorff_px: float | None
    max_edge_span_frames: int
    point_count: int
    max_anchor_scale: float
    anchor_state_count: int
    anchor_expansion: float
    edge_threads: int
    edge_processes: int
    dominance_epsilon: float
    quality_mode: str
    stored_vertex_contract: bool
    pair_vote_states: bool
    edge_batch_size: int
    candidate_mode: str
    temporal_window_radii: tuple[int, int, int]
    temporal_recall_quantile: float
    extra_anchor_states: dict[tuple[int, str], tuple[Keyframe, ...]] | None


@dataclass(frozen=True)
class _EdgeContext:
    frames: list[int]
    prepared_by_frame: list[list[_PreparedAnchor]]
    anchor_metrics_by_frame: list[list[tuple[float, float]]]
    raw_by_frame: dict[int, RawMask]
    raw_areas: dict[int, float]
    raw_bounds: dict[int, tuple[float, float, float, float]]
    quality_by_frame: dict[int, RawMask]
    quality_areas: dict[int, float]
    metric_quality_by_frame: dict[int, object]
    metric_quality_areas: dict[int, float]
    border_by_frame: dict[int, BorderFrameConstraint]
    visible_rectangle: object | None
    recall_floor: float
    frame_iou_floor: float
    max_frame_hausdorff_px: float | None
    max_edge_span_frames: int
    quality_mode: str
    edge_batch_size: int


_EDGE_PROCESS_CONTEXT: _EdgeContext | None = None


def _iou_utility(iou: float, mode: str) -> float:
    """Additive overlap utility used by the exact DAG optimizer.

    ``tail_harmonic`` has no IoU threshold, but gives a much larger marginal
    reward to repairing a bad frame than polishing an already-good frame.
    This prevents thousands of small gains from cheaply compensating one
    severe local failure.
    """

    # ``np.clip`` is disproportionately expensive for the millions of scalar
    # edge-frame evaluations performed here.  These branches preserve its
    # scalar semantics, including passing NaN through unchanged.
    value = float(iou)
    if value < 1e-6:
        value = 1e-6
    elif value > 1.0:
        value = 1.0
    if mode == "mean_iou":
        return value
    if mode == "log_iou":
        return math.log(value)
    if mode in {"tail_harmonic", "tail_boundary"}:
        return -1.0 / (value * value)
    raise ValueError(f"unsupported quality mode: {mode}")


def _fast_numpy_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the production roll/reversal alignment."""

    count = len(candidate)
    positions = np.arange(count)
    shifts = np.arange(count)[:, None]
    indices = (positions[None, :] - shifts) % count
    forward = candidate[indices]
    reverse = candidate[::-1][indices]
    variants = np.concatenate((forward, reverse), axis=0)
    errors = np.mean(
        np.sum(np.square(variants - reference[None, :, :]), axis=2), axis=1
    )
    return variants[int(np.argmin(errors))]


def _prepare_anchor(
    keyframe: Keyframe, *, stored_vertex_contract: bool = False
) -> _PreparedAnchor:
    component = _primary_component(keyframe)
    if component is None:
        values = np.empty((0, 2), dtype=np.float64)
        geometry = geometry_from_arrays([])
        return _PreparedAnchor(keyframe, values, values, geometry, geometry.bounds)
    values = np.asarray(component.values, dtype=np.float64)
    count = max(8, len(values))
    # The V3 editor contract interpolates point_index.  In the superior mode
    # candidate anchors already contain the exact fixed-count points that will
    # be exported, so resampling them a second time would make the DP optimize
    # a different curve from the saved SQLite.
    sampled = (
        values.copy() if stored_vertex_contract else _numpy_resample(values, count)
    )
    geometry = geometry_from_arrays([values])
    return _PreparedAnchor(
        keyframe=keyframe,
        values=values,
        sampled=sampled,
        geometry=geometry,
        bounds=geometry.bounds,
    )


def _anchor_metrics(
    raw: RawMask,
    keyframe: Keyframe,
    quality_raw: RawMask | None = None,
    visible_rectangle=None,
    border_constraint: BorderFrameConstraint | None = None,
) -> tuple[float, float]:
    component = _primary_component(keyframe)
    if component is None:
        return 0.0, 0.0
    geometry = geometry_from_arrays([np.asarray(component.values, dtype=np.float64)])
    return _anchor_geometry_metrics(
        raw,
        geometry,
        quality_raw,
        visible_rectangle=visible_rectangle,
        border_constraint=border_constraint,
    )


def _anchor_geometry_metrics(
    raw: RawMask,
    geometry,
    quality_raw: RawMask | None = None,
    *,
    visible_rectangle=None,
    border_constraint: BorderFrameConstraint | None = None,
) -> tuple[float, float]:
    """Evaluate a prebuilt anchor geometry without duplicate GEOS work."""

    intersection = float(raw.geometry.intersection(geometry).area)
    raw_area = float(raw.geometry.area)
    quality = raw if quality_raw is None else quality_raw
    if quality is raw:
        quality_intersection = intersection
        quality_area = raw_area
    else:
        quality_intersection = float(quality.geometry.intersection(geometry).area)
        quality_area = float(quality.geometry.area)
    metric_geometry = geometry
    if visible_rectangle is not None:
        bounds = geometry.bounds
        visible_bounds = visible_rectangle.bounds
        if (
            bounds[0] < visible_bounds[0]
            or bounds[1] < visible_bounds[1]
            or bounds[2] > visible_bounds[2]
            or bounds[3] > visible_bounds[3]
        ):
            metric_geometry = geometry.intersection(visible_rectangle)
    predicted_area = float(metric_geometry.area)
    constraint_recall = intersection / raw_area if raw_area else 1.0
    quality_recall = quality_intersection / quality_area if quality_area else 1.0
    if border_constraint is not None:
        metric_quality = quality.geometry.intersection(
            border_constraint.quality_domain
        )
        metric_geometry = metric_geometry.intersection(
            border_constraint.quality_domain
        )
        quality_area = float(metric_quality.area)
        predicted_area = float(metric_geometry.area)
        quality_intersection = float(metric_quality.intersection(metric_geometry).area)
    union = quality_area + predicted_area - quality_intersection
    # Border expansion adds a second Production safeguard.  It must never
    # replace the original AI-mask constraint, otherwise missing original
    # pixels can be hidden by correctly covering the added border area.
    recall = min(constraint_recall, quality_recall)
    iou = quality_intersection / union if union else 1.0
    return recall, iou


def _visible_metric_geometry(geometry, visible_rectangle):
    """Clip only the invisible margin used by border-safe interpolation."""

    if visible_rectangle is None:
        return geometry
    bounds = geometry.bounds
    visible_bounds = visible_rectangle.bounds
    if (
        bounds[0] < visible_bounds[0]
        or bounds[1] < visible_bounds[1]
        or bounds[2] > visible_bounds[2]
        or bounds[3] > visible_bounds[3]
    ):
        return geometry.intersection(visible_rectangle)
    return geometry


def _scale_anchor(keyframe: Keyframe, scale: float) -> Keyframe:
    component = _primary_component(keyframe)
    if component is None:
        return keyframe
    points = np.asarray(component.values, dtype=np.float64)
    center = np.mean(points, axis=0)
    scaled = center + float(scale) * (points - center)
    return Keyframe(
        keyframe.frame,
        ((0, Component("polygon", scaled.tolist())),),
    )


def _buffer_anchor(keyframe: Keyframe, distance: float) -> Keyframe:
    """Repair fixed-vertex border coverage with minimal local dilation."""

    component = _primary_component(keyframe)
    if component is None:
        return keyframe
    point_count = len(component.values)
    geometry = _keyframe_geometry(keyframe).buffer(float(distance), join_style=2)
    if geometry.is_empty:
        return keyframe
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    polygon = max(polygons, key=lambda item: float(item.area))
    points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    buffered = RawMask(
        frame=keyframe.frame,
        track_id="",
        geometry=polygon,
        primary_points=points,
        score=1.0,
    )
    return _raw_keyframe(buffered, point_count=point_count, point_strategy="uniform")


def _minimum_border_buffer_anchor(
    source: Keyframe,
    raw: RawMask,
    quality_raw: RawMask | None,
    recall_floor: float,
    border_constraint: BorderFrameConstraint,
    visible_rectangle,
) -> Keyframe | None:
    low_distance = 0.0
    high_distance: float | None = None
    maximum = float(border_constraint.max_repair_px)
    distances = [
        value
        for value in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, maximum)
        if value <= maximum
    ]
    distances = sorted(set(distances))
    for distance in distances:
        repaired = _buffer_anchor(source, float(distance))
        repaired_recall, _repaired_iou = _anchor_metrics(
            raw, repaired, quality_raw, visible_rectangle, border_constraint
        )
        if repaired_recall >= recall_floor and border_geometry_feasible(
            _keyframe_geometry(repaired), border_constraint
        ):
            high_distance = float(distance)
            break
        low_distance = float(distance)
    if high_distance is None:
        return None
    for _step in range(7):
        middle = 0.5 * (low_distance + high_distance)
        repaired = _buffer_anchor(source, middle)
        repaired_recall, _repaired_iou = _anchor_metrics(
            raw, repaired, quality_raw, visible_rectangle, border_constraint
        )
        if repaired_recall >= recall_floor and border_geometry_feasible(
            _keyframe_geometry(repaired), border_constraint
        ):
            high_distance = middle
        else:
            low_distance = middle
    return _buffer_anchor(source, high_distance)


def _directional_border_anchor(
    source: Keyframe,
    border_constraint: BorderFrameConstraint,
    influence_px: float,
) -> Keyframe:
    """Extend only vertices near a touched side to its required coordinate."""

    component = _primary_component(source)
    if component is None:
        return source
    points = np.asarray(component.values, dtype=np.float64).copy()
    influence = max(float(influence_px), 1.0)
    for side in border_constraint.sides:
        axis = 0 if side.side in {"left", "right"} else 1
        outward_positive = side.side in {"right", "bottom"}
        extreme = (
            float(np.max(points[:, axis]))
            if outward_positive
            else float(np.min(points[:, axis]))
        )
        delta = float(side.required_coordinate) - extreme
        if (outward_positive and delta <= 0.0) or (
            not outward_positive and delta >= 0.0
        ):
            continue
        if outward_positive:
            normalized = (points[:, axis] - (extreme - influence)) / influence
        else:
            normalized = ((extreme + influence) - points[:, axis]) / influence
        clipped = np.clip(normalized, 0.0, 1.0)
        weights = clipped * clipped * (3.0 - 2.0 * clipped)
        points[:, axis] += delta * weights
        # Floating-point roundoff must not create an extent violation.
        extreme_index = (
            int(np.argmax(points[:, axis]))
            if outward_positive
            else int(np.argmin(points[:, axis]))
        )
        points[extreme_index, axis] = float(side.required_coordinate)
    return Keyframe(
        source.frame,
        ((0, Component("polygon", points.tolist())),),
    )


def _minimum_directional_border_anchor(
    source: Keyframe,
    raw: RawMask,
    quality_raw: RawMask,
    recall_floor: float,
    border_constraint: BorderFrameConstraint,
    visible_rectangle,
    *,
    max_anchor_scale: float,
) -> Keyframe | None:
    """Find the highest-IoU feasible local border extension."""

    component = _primary_component(source)
    if component is None:
        return None
    points = np.asarray(component.values, dtype=np.float64)
    spans = np.ptp(points, axis=0)
    maximum_influence = max(float(np.max(spans)), 24.0)
    influences = {
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
        48.0,
        64.0,
        96.0,
        128.0,
        maximum_influence * 0.25,
        maximum_influence * 0.50,
        maximum_influence,
    }
    feasible: list[tuple[float, float, Keyframe]] = []
    for influence in sorted(value for value in influences if value > 0.0):
        directional = _directional_border_anchor(
            source, border_constraint, float(influence)
        )
        candidate = _minimum_feasible_anchor(
            directional,
            raw,
            quality_raw=quality_raw,
            recall_floor=recall_floor,
            max_anchor_scale=max_anchor_scale,
            border_constraint=border_constraint,
            visible_rectangle=visible_rectangle,
        )
        if candidate is None:
            continue
        recall, iou = _anchor_metrics(
            raw,
            candidate,
            quality_raw,
            visible_rectangle,
            border_constraint,
        )
        geometry = _keyframe_geometry(candidate)
        if recall + 1e-12 >= recall_floor and border_geometry_feasible(
            geometry, border_constraint
        ):
            feasible.append((iou, -float(influence), candidate))
    if not feasible:
        return None
    return max(feasible, key=lambda item: (item[0], item[1]))[2]


def _normalize_polygon_anchor(keyframe: Keyframe, point_count: int) -> Keyframe:
    """Materialize the exact fixed-count vertex representation for SQLite."""

    component = _primary_component(keyframe)
    if component is None:
        return keyframe
    points = _numpy_resample(
        np.asarray(component.values, dtype=np.float64), max(8, int(point_count))
    )
    return Keyframe(
        keyframe.frame,
        ((0, Component("polygon", points.tolist())),),
    )


def _pad_polygon_edges(points: np.ndarray, point_count: int) -> np.ndarray:
    """Reach a fixed vertex count by adding collinear edge midpoints."""

    values = [np.asarray(point, dtype=np.float64) for point in points]
    while len(values) < point_count:
        lengths = [
            float(np.linalg.norm(values[(index + 1) % len(values)] - point))
            for index, point in enumerate(values)
        ]
        index = int(np.argmax(lengths))
        midpoint = 0.5 * (values[index] + values[(index + 1) % len(values)])
        values.insert(index + 1, midpoint)
    return np.asarray(values, dtype=np.float64)


def _corner_preserving_keyframe(raw: RawMask, point_count: int) -> Keyframe | None:
    """Fit a fixed-count polygon while retaining high-curvature vertices.

    Uniform arc-length sampling can skip a narrow but important corner and may
    fail Recall even after radial scaling.  Topology-preserving RDP selects the
    least simplification that reaches the budget, then collinear padding keeps
    the exact geometry while satisfying the editor's fixed-count contract.
    """

    geometry = raw.geometry
    if geometry.is_empty or geometry.geom_type != "Polygon":
        return None
    source = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
    if len(source) <= point_count:
        points = source
    else:
        min_x, min_y, max_x, max_y = geometry.bounds
        low = 0.0
        high = max(math.hypot(max_x - min_x, max_y - min_y), 1.0)
        best = None
        for _step in range(20):
            tolerance = 0.5 * (low + high)
            simplified = geometry.simplify(tolerance, preserve_topology=True)
            if simplified.is_empty or simplified.geom_type != "Polygon":
                low = tolerance
                continue
            count = len(simplified.exterior.coords) - 1
            if count <= point_count:
                best = simplified
                high = tolerance
            else:
                low = tolerance
        if best is None:
            return None
        points = np.asarray(best.exterior.coords[:-1], dtype=np.float64)
    if len(points) < 3 or len(points) > point_count:
        return None
    points = _pad_polygon_edges(points, point_count)
    return Keyframe(
        int(raw.frame), ((0, Component("polygon", points.tolist())),)
    )


def _build_pair_vote_sources(
    frames: list[int],
    quality_by_frame: dict[int, RawMask],
    *,
    point_count: int,
    max_edge_span_frames: int,
) -> dict[int, list[Keyframe]]:
    """Generate Recall-projectable Production pair-vote endpoint proposals.

    Production fits two endpoint vectors by least squares over each interval,
    then averages proposals after key selection.  Applying that mutation after
    DP caused the historical Recall regressions.  Here proposals for several
    useful spans become ordinary node states *before* DP, so every selected
    edge is revalidated densely and no post-decode geometry mutation exists.
    """

    if len(frames) < 3:
        return {}
    aligned: list[np.ndarray] = []
    reference: np.ndarray | None = None
    for frame in frames:
        keyframe = _raw_keyframe(
            quality_by_frame[frame],
            point_count=point_count,
            point_strategy="uniform",
        )
        component = _primary_component(keyframe)
        if component is None:
            aligned.append(np.empty((0, 2), dtype=np.float64))
            continue
        points = np.asarray(component.values, dtype=np.float64)
        if reference is not None and len(points) == len(reference):
            points = _fast_numpy_align(reference, points)
        aligned.append(points)
        reference = points

    maximum = max(2, int(max_edge_span_frames))
    requested_spans = sorted(
        {
            min(maximum, value)
            for value in (4, 8, 10, 15, 20, 24, maximum)
            if min(maximum, value) >= 2
        }
    )
    proposals: dict[int, list[Keyframe]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for start_index, start_frame in enumerate(frames[:-1]):
        for requested_span in requested_spans:
            target = start_frame + requested_span
            end_index = bisect.bisect_left(frames, target, lo=start_index + 1)
            candidates = [
                index
                for index in (end_index - 1, end_index)
                if start_index < index < len(frames)
                and frames[index] - start_frame <= maximum
            ]
            if not candidates:
                continue
            end_index = min(
                candidates,
                key=lambda index: abs((frames[index] - start_frame) - requested_span),
            )
            identity = (start_index, end_index)
            if identity in seen_pairs:
                continue
            seen_pairs.add(identity)
            values = aligned[start_index : end_index + 1]
            if not values or any(value.shape != values[0].shape for value in values):
                continue
            times = np.asarray(frames[start_index : end_index + 1], dtype=np.float64)
            span = max(float(times[-1] - times[0]), 1.0)
            alpha = (times - times[0]) / span
            design = np.column_stack((1.0 - alpha, alpha))
            targets = np.stack(values, axis=0).reshape(len(values), -1)
            gram = design.T @ design
            endpoints = np.linalg.solve(
                gram + 1e-8 * np.eye(2, dtype=np.float64),
                design.T @ targets,
            ).reshape(2, len(values[0]), 2)
            left_frame = frames[start_index]
            right_frame = frames[end_index]
            proposals.setdefault(left_frame, []).append(
                Keyframe(
                    left_frame,
                    ((0, Component("polygon", endpoints[0].tolist())),),
                )
            )
            proposals.setdefault(right_frame, []).append(
                Keyframe(
                    right_frame,
                    ((0, Component("polygon", endpoints[1].tolist())),),
                )
            )
    return proposals


def _minimum_feasible_anchor(
    source: Keyframe,
    raw: RawMask,
    *,
    quality_raw: RawMask | None = None,
    recall_floor: float,
    max_anchor_scale: float,
    border_constraint: BorderFrameConstraint | None = None,
    visible_rectangle=None,
) -> Keyframe | None:
    source_recall, _source_iou = _anchor_metrics(
        raw, source, quality_raw, visible_rectangle, border_constraint
    )
    if source_recall >= recall_floor and border_geometry_feasible(
        _keyframe_geometry(source), border_constraint
    ):
        return source
    maximum = _scale_anchor(source, max_anchor_scale)
    maximum_recall, _maximum_iou = _anchor_metrics(
        raw, maximum, quality_raw, visible_rectangle, border_constraint
    )
    if maximum_recall < recall_floor or not border_geometry_feasible(
        _keyframe_geometry(maximum), border_constraint
    ):
        return None
    low = 1.0
    high = float(max_anchor_scale)
    # Bracket the first feasible scale; recall is usually monotone here, while
    # the grid protects against small non-monotonic polygon effects.
    previous = 1.0
    for scale in np.linspace(1.0, float(max_anchor_scale), 33)[1:]:
        candidate = _scale_anchor(source, float(scale))
        candidate_recall, _candidate_iou = _anchor_metrics(
            raw, candidate, quality_raw, visible_rectangle, border_constraint
        )
        if candidate_recall >= recall_floor and border_geometry_feasible(
            _keyframe_geometry(candidate), border_constraint
        ):
            low = previous
            high = float(scale)
            break
        previous = float(scale)
    for _step in range(12):
        middle = 0.5 * (low + high)
        candidate = _scale_anchor(source, middle)
        candidate_recall, _candidate_iou = _anchor_metrics(
            raw, candidate, quality_raw, visible_rectangle, border_constraint
        )
        if candidate_recall >= recall_floor and border_geometry_feasible(
            _keyframe_geometry(candidate), border_constraint
        ):
            high = middle
        else:
            low = middle
    return _scale_anchor(source, high)


def _feasible_base_anchors(
    segment: Segment,
    raw: RawMask,
    *,
    quality_raw: RawMask | None = None,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    anchor_point_strategy: str = "uniform",
    stored_vertex_contract: bool = False,
    corner_preserving_source: bool = False,
    border_constraint: BorderFrameConstraint | None = None,
    visible_rectangle=None,
) -> list[tuple[float, float, Keyframe]]:
    """Build each base source's minimum feasible anchor exactly once."""
    sources = [
        _safe_anchor_keyframe(
            segment,
            raw,
            anchor_recall=recall_floor,
            point_count=point_count,
        ),
        _raw_keyframe(
            raw,
            point_count=point_count,
            point_strategy=anchor_point_strategy,
        ),
    ]
    if quality_raw is not None:
        # ``raw`` may carry Production's off-canvas border-expanded anchor.
        # Retain the unmodified observation as a separate candidate so the
        # optimizer can avoid needless expansion when the visible Recall floor
        # is already satisfied.  Border behavior remains available rather
        # than being forced on nearly half of all frames.
        sources.append(
            _raw_keyframe(
                quality_raw,
                point_count=point_count,
                point_strategy=anchor_point_strategy,
            )
        )
    if stored_vertex_contract:
        sources = [_normalize_polygon_anchor(source, point_count) for source in sources]
    feasible: list[tuple[float, float, Keyframe]] = []
    for source in sources:
        candidate = _minimum_feasible_anchor(
            source,
            raw,
            quality_raw=quality_raw,
            recall_floor=recall_floor,
            max_anchor_scale=max_anchor_scale,
            border_constraint=border_constraint,
            visible_rectangle=visible_rectangle,
        )
        if candidate is None:
            continue
        candidate_recall, candidate_iou = _anchor_metrics(
            raw, candidate, quality_raw, visible_rectangle, border_constraint
        )
        feasible.append((candidate_iou, candidate_recall, candidate))
    if corner_preserving_source and not feasible:
        # RDP is substantially better at fitting an isolated complex contour,
        # but using it routinely changes vertex semantics between frames and
        # regresses linear interpolation.  Restrict it to the rare case where
        # every temporally aligned legacy source is infeasible.
        corner = _corner_preserving_keyframe(quality_raw or raw, point_count)
        if corner is not None:
            candidate = _minimum_feasible_anchor(
                corner,
                raw,
                quality_raw=quality_raw,
                recall_floor=recall_floor,
                max_anchor_scale=max_anchor_scale,
                border_constraint=border_constraint,
                visible_rectangle=visible_rectangle,
            )
            if candidate is not None:
                candidate_recall, candidate_iou = _anchor_metrics(
                    raw,
                    candidate,
                    quality_raw,
                    visible_rectangle,
                    border_constraint,
                )
                feasible.append((candidate_iou, candidate_recall, candidate))
    if border_constraint is not None and not feasible:
        # The expanded-constraint source already contains both the original
        # contour and Production's off-canvas extension.  One minimal buffered
        # repair is enough to recover fixed-vertex approximation loss when no
        # ordinary source can satisfy the hard constraints. Doing
        # the same expensive search for every alternate source tripled anchor
        # construction time without adding a better state in the incident set.
        repaired = _minimum_border_buffer_anchor(
            sources[1],
            raw,
            quality_raw,
            recall_floor,
            border_constraint,
            visible_rectangle,
        )
        if repaired is not None:
            repaired_recall, repaired_iou = _anchor_metrics(
                raw, repaired, quality_raw, visible_rectangle, border_constraint
            )
            feasible.append((repaired_iou, repaired_recall, repaired))
    return feasible


def _make_feasible_anchor(
    segment: Segment,
    raw: RawMask,
    *,
    quality_raw: RawMask | None = None,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    anchor_point_strategy: str = "uniform",
    stored_vertex_contract: bool = False,
    extra_sources: list[Keyframe] | None = None,
    border_constraint: BorderFrameConstraint | None = None,
    visible_rectangle=None,
) -> Keyframe:
    """Choose the highest-IoU anchor that satisfies the fixed recall floor."""

    feasible = _feasible_base_anchors(
        segment,
        raw,
        quality_raw=quality_raw,
        recall_floor=recall_floor,
        point_count=point_count,
        max_anchor_scale=max_anchor_scale,
        anchor_point_strategy=anchor_point_strategy,
        stored_vertex_contract=stored_vertex_contract,
        border_constraint=border_constraint,
        visible_rectangle=visible_rectangle,
    )
    if not feasible:
        raise RuntimeError(
            f"frame {raw.frame} track {raw.track_id} cannot satisfy recall "
            f"floor {recall_floor:.6f} with max anchor scale "
            f"{max_anchor_scale:.3f}"
        )
    return max(feasible, key=lambda item: (item[0], item[1]))[2]


def _keyframe_geometry(keyframe: Keyframe):
    component = _primary_component(keyframe)
    if component is None:
        return geometry_from_arrays([])
    return geometry_from_arrays([np.asarray(component.values, dtype=np.float64)])


def _geometry_iou(left, right) -> float:
    intersection = float(left.intersection(right).area)
    union = float(left.area + right.area - intersection)
    return intersection / union if union else 1.0


def _make_feasible_anchors(
    segment: Segment,
    raw: RawMask,
    *,
    quality_raw: RawMask | None = None,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    anchor_state_count: int,
    anchor_expansion: float,
    anchor_iou_floor: float = 0.0,
    anchor_relative_iou_margin: float | None = None,
    anchor_point_strategy: str = "uniform",
    max_anchor_hausdorff_px: float | None = None,
    stored_vertex_contract: bool = False,
    corner_preserving_source: bool = False,
    extra_sources: list[Keyframe] | None = None,
    border_constraint: BorderFrameConstraint | None = None,
    visible_rectangle=None,
) -> list[Keyframe]:
    """Return diverse recall-feasible shape states for one candidate frame.

    State zero deliberately matches the single-state implementation. Further
    states retain alternative production/raw sources and mildly expanded
    variants. Locally lower-IoU states are not discarded because they can form
    longer, safer interpolation edges and reduce the global key count.
    """

    requested = max(1, int(anchor_state_count))
    feasible_sources = _feasible_base_anchors(
        segment,
        raw,
        quality_raw=quality_raw,
        recall_floor=recall_floor,
        point_count=point_count,
        max_anchor_scale=max_anchor_scale,
        anchor_point_strategy=anchor_point_strategy,
        stored_vertex_contract=stored_vertex_contract,
        corner_preserving_source=corner_preserving_source,
        border_constraint=border_constraint,
        visible_rectangle=visible_rectangle,
    )
    if not feasible_sources:
        raise RuntimeError(
            f"frame {raw.frame} track {raw.track_id} cannot satisfy recall "
            f"floor {recall_floor:.6f} with max anchor scale "
            f"{max_anchor_scale:.3f}"
        )
    primary_iou, primary_recall, primary = max(
        feasible_sources, key=lambda item: (item[0], item[1])
    )
    primary_hausdorff = raw.geometry.hausdorff_distance(_keyframe_geometry(primary))
    if (
        primary_recall < recall_floor
        or primary_iou < anchor_iou_floor
        or (
            max_anchor_hausdorff_px is not None
            and primary_hausdorff > max_anchor_hausdorff_px
        )
    ):
        raise RuntimeError(
            f"frame {raw.frame} track {raw.track_id} cannot satisfy anchor "
            f"IoU floor {anchor_iou_floor:.6f}"
        )
    if requested == 1:
        return [primary]

    pool = [primary]
    expansion = max(0.0, float(anchor_expansion))
    if expansion <= 0.04 or requested <= 1:
        expansion_scales = np.linspace(1.0, 1.0 + expansion, requested + 1)
    else:
        # Retain the already validated mild 4% state while also exposing a
        # genuinely enlarged state for low-keyframe Pareto points.  A plain
        # linspace at expansion=0.30 omitted the useful 1.04 candidate and
        # regressed the high-keyframe side of the frontier.
        expansion_scales = np.asarray(
            [
                1.0,
                1.04,
                *np.linspace(1.04, 1.0 + expansion, requested + 1)[1:],
            ],
            dtype=np.float64,
        )
    for _iou, _recall, feasible in feasible_sources:
        for scale in expansion_scales:
            candidate = _scale_anchor(feasible, float(scale))
            recall, iou = _anchor_metrics(
                raw, candidate, quality_raw, visible_rectangle, border_constraint
            )
            hausdorff = raw.geometry.hausdorff_distance(_keyframe_geometry(candidate))
            if (
                recall >= recall_floor
                and iou >= anchor_iou_floor
                and border_geometry_feasible(
                    _keyframe_geometry(candidate), border_constraint
                )
                and (
                    max_anchor_hausdorff_px is None
                    or hausdorff <= max_anchor_hausdorff_px
                )
            ):
                pool.append(candidate)

    unique: list[tuple[Keyframe, object]] = []
    for candidate in pool:
        geometry = _keyframe_geometry(candidate)
        if geometry.is_empty:
            continue
        if any(_geometry_iou(geometry, prior) >= 0.9999 for _item, prior in unique):
            continue
        unique.append((candidate, geometry))
    if not unique:
        return [primary]

    if anchor_relative_iou_margin is not None:
        margin = max(0.0, float(anchor_relative_iou_margin))
        scored = [
            (
                _anchor_metrics(
                    raw,
                    candidate,
                    quality_raw,
                    visible_rectangle,
                    border_constraint,
                )[1],
                candidate,
                geometry,
            )
            for candidate, geometry in unique
        ]
        best_iou = max(item[0] for item in scored)
        unique = [
            (candidate, geometry)
            for iou, candidate, geometry in scored
            if iou + 1e-12 >= best_iou - margin
        ]

    primary_geometry = _keyframe_geometry(primary)
    primary_index = max(
        range(len(unique)),
        key=lambda index: _geometry_iou(unique[index][1], primary_geometry),
    )
    selected = [unique.pop(primary_index)]
    # Reserve half of the state budget for high-IoU exploitation before
    # farthest-point diversity.  Pure farthest-point selection over-favored
    # very inflated anchors and could make the high-keyframe frontier worse
    # even though the original primary state was still present.
    exploitation_slots = min(
        len(unique),
        0 if requested <= 3 else max(1, (requested - 2) // 2),
    )
    for _slot in range(exploitation_slots):
        candidate_index = max(
            range(len(unique)),
            key=lambda index: _anchor_metrics(
                raw,
                unique[index][0],
                quality_raw,
                visible_rectangle,
                border_constraint,
            )[1],
        )
        selected.append(unique.pop(candidate_index))
    while unique and len(selected) < requested:
        # Farthest-point selection preserves genuinely different shapes rather
        # than several numerically equivalent expansion factors. Local IoU is
        # the tie-breaker, not a pruning objective.
        candidate_index = max(
            range(len(unique)),
            key=lambda index: (
                min(
                    1.0 - _geometry_iou(unique[index][1], chosen[1])
                    for chosen in selected
                ),
                _anchor_metrics(
                    raw,
                    unique[index][0],
                    quality_raw,
                    visible_rectangle,
                    border_constraint,
                )[1],
            ),
        )
        selected.append(unique.pop(candidate_index))
    if extra_sources:
        # Pair-vote is inherited as one additive proposal state.  It must not
        # displace any of the four proven raw/Production/expansion states;
        # doing so improved a few very long edges but regressed the rest of
        # the Pareto front.  The DP may use this fifth state only where it
        # genuinely improves a densely revalidated path.
        pair_candidates: list[tuple[float, Keyframe, object]] = []
        for source in extra_sources:
            normalized = (
                _normalize_polygon_anchor(source, point_count)
                if stored_vertex_contract
                else source
            )
            feasible = _minimum_feasible_anchor(
                normalized,
                raw,
                quality_raw=quality_raw,
                recall_floor=recall_floor,
                max_anchor_scale=max_anchor_scale,
                border_constraint=border_constraint,
                visible_rectangle=visible_rectangle,
            )
            if feasible is None:
                continue
            recall, iou = _anchor_metrics(
                raw, feasible, quality_raw, visible_rectangle, border_constraint
            )
            geometry = _keyframe_geometry(feasible)
            if recall < recall_floor or iou < anchor_iou_floor:
                continue
            if any(
                _geometry_iou(geometry, candidate_geometry) >= 0.9999
                for _candidate, candidate_geometry in selected
            ):
                continue
            pair_candidates.append((iou, feasible, geometry))
        if pair_candidates:
            _iou, candidate, geometry = max(pair_candidates, key=lambda item: item[0])
            selected.append((candidate, geometry))
    return [candidate for candidate, _geometry in selected]


def _make_temporal7_anchors(
    frame: int,
    raw_by_frame: dict[int, RawMask],
    constraint_raw: RawMask,
    quality_raw: RawMask,
    *,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    window_radii: tuple[int, int, int],
    recall_quantile: float,
    border_constraint: BorderFrameConstraint | None,
    visible_rectangle,
    extra_sources: tuple[Keyframe, ...] = (),
    candidate_names: frozenset[str] | None = None,
) -> list[Keyframe]:
    """Return Production-independent temporal candidates plus refinements."""

    generated = build_temporal_candidates(
        frame,
        raw_by_frame,
        point_count=point_count,
        window_radii=window_radii,
        recall_quantile=recall_quantile,
    )
    sources = [
        value.keyframe
        for value in generated
        if candidate_names is None or value.name in candidate_names
    ]
    sources.extend(extra_sources)
    feasible: list[tuple[float, float, Keyframe, object]] = []
    for source in sources:
        candidate = _minimum_feasible_anchor(
            source,
            constraint_raw,
            quality_raw=quality_raw,
            recall_floor=recall_floor,
            max_anchor_scale=max_anchor_scale,
            border_constraint=border_constraint,
            visible_rectangle=visible_rectangle,
        )
        if candidate is None and border_constraint is not None:
            candidate = _minimum_directional_border_anchor(
                source,
                constraint_raw,
                quality_raw,
                recall_floor,
                border_constraint,
                visible_rectangle,
                max_anchor_scale=max_anchor_scale,
            )
        if candidate is None and border_constraint is not None:
            candidate = _minimum_border_buffer_anchor(
                source,
                constraint_raw,
                quality_raw,
                recall_floor,
                border_constraint,
                visible_rectangle,
            )
        if candidate is None:
            continue
        geometry = _keyframe_geometry(candidate)
        if geometry.is_empty or not border_geometry_feasible(
            geometry, border_constraint
        ):
            continue
        recall, iou = _anchor_metrics(
            constraint_raw,
            candidate,
            quality_raw,
            visible_rectangle,
            border_constraint,
        )
        if recall + 1e-12 < recall_floor:
            continue
        if any(_geometry_iou(geometry, prior[3]) >= 0.9999 for prior in feasible):
            continue
        feasible.append((iou, recall, candidate, geometry))
    if not feasible:
        raise RuntimeError(
            f"frame {frame} track {constraint_raw.track_id} has no feasible "
            "Production-independent temporal anchor"
        )
    # Keep the seven declared initial states plus at most two refined states.
    # Deduplication above usually leaves fewer and avoids redundant edge pairs.
    feasible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in feasible[:9]]


def _merge_anchor_sets(*groups: list[Keyframe]) -> list[Keyframe]:
    """Union candidate families without displacing legacy states."""

    output: list[tuple[Keyframe, object]] = []
    for group in groups:
        for candidate in group:
            geometry = _keyframe_geometry(candidate)
            if geometry.is_empty:
                continue
            if any(_geometry_iou(geometry, prior) >= 0.9999 for _item, prior in output):
                continue
            output.append((candidate, geometry))
    return [candidate for candidate, _geometry in output]


_TEMPORAL_UNION_NAMES = {
    "legacy_temporal_short": frozenset(("short_iou", "short_recall")),
    "legacy_temporal_medium": frozenset(("medium_iou", "medium_recall")),
    "legacy_temporal_long": frozenset(("long_iou", "long_recall")),
    "legacy_temporal_iou": frozenset(
        ("short_iou", "medium_iou", "long_iou")
    ),
    "legacy_temporal_recall": frozenset(
        ("short_recall", "medium_recall", "long_recall")
    ),
    # Screen-edge handling is already a strong legacy candidate family.  Keep
    # temporal Recall candidates for interior frames only so their robust
    # radial expansion cannot grow back into the visible image while trying
    # to satisfy an off-canvas directional extent.
    "legacy_temporal_recall_interior": frozenset(
        ("short_recall", "medium_recall", "long_recall")
    ),
    "legacy_temporal_recall_interior_rdp_fallback": frozenset(
        ("short_recall", "medium_recall", "long_recall")
    ),
    "legacy_temporal_union": frozenset(
        (
            "short_iou",
            "short_recall",
            "medium_iou",
            "medium_recall",
            "long_iou",
            "long_recall",
        )
    ),
}

_GEOMETRIC_UNION_NAMES = {
    "legacy_geometric_directional": frozenset(("axis_major", "axis_minor")),
    "legacy_geometric_axis": frozenset(
        ("axis_major", "axis_minor", "axis_balanced")
    ),
    "legacy_geometric_envelope": frozenset(("axis_envelope",)),
    "legacy_geometric_union": frozenset(
        ("axis_major", "axis_minor", "axis_balanced", "axis_envelope")
    ),
}

_COMBINED_UNION_NAMES = {
    "legacy_recall_geometric": (
        _TEMPORAL_UNION_NAMES["legacy_temporal_recall"],
        _GEOMETRIC_UNION_NAMES["legacy_geometric_union"],
    ),
    "legacy_recall_axis": (
        _TEMPORAL_UNION_NAMES["legacy_temporal_recall"],
        _GEOMETRIC_UNION_NAMES["legacy_geometric_axis"],
    ),
    "legacy_longrecall_axis": (
        frozenset(("long_recall",)),
        _GEOMETRIC_UNION_NAMES["legacy_geometric_axis"],
    ),
    "legacy_longrecall_directional": (
        frozenset(("long_recall",)),
        _GEOMETRIC_UNION_NAMES["legacy_geometric_directional"],
    ),
}


def _make_geometric_anchors(
    frame: int,
    raw_by_frame: dict[int, RawMask],
    constraint_raw: RawMask,
    quality_raw: RawMask,
    *,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    border_constraint: BorderFrameConstraint | None,
    visible_rectangle,
    candidate_names: frozenset[str],
) -> list[Keyframe]:
    generated = build_geometric_candidates(
        frame, raw_by_frame, point_count=point_count
    )
    output: list[tuple[Keyframe, object]] = []
    for value in generated:
        if value.name not in candidate_names:
            continue
        candidate = _minimum_feasible_anchor(
            value.keyframe,
            constraint_raw,
            quality_raw=quality_raw,
            recall_floor=recall_floor,
            max_anchor_scale=max_anchor_scale,
            border_constraint=border_constraint,
            visible_rectangle=visible_rectangle,
        )
        if candidate is None and border_constraint is not None:
            candidate = _minimum_directional_border_anchor(
                value.keyframe,
                constraint_raw,
                quality_raw,
                recall_floor,
                border_constraint,
                visible_rectangle,
                max_anchor_scale=max_anchor_scale,
            )
        if candidate is None:
            continue
        geometry = _keyframe_geometry(candidate)
        if geometry.is_empty or not border_geometry_feasible(
            geometry, border_constraint
        ):
            continue
        if any(_geometry_iou(geometry, prior) >= 0.9999 for _item, prior in output):
            continue
        output.append((candidate, geometry))
    return [candidate for candidate, _geometry in output]


def _evaluate_anchor_edge(
    start_index: int,
    start_anchor_index: int,
    end_index: int,
    end_anchor_index: int,
    frames: list[int],
    left: _PreparedAnchor,
    right: _PreparedAnchor,
    raw_by_frame: dict[int, RawMask],
    raw_areas: dict[int, float],
    raw_bounds: dict[int, tuple[float, float, float, float]],
    quality_by_frame: dict[int, RawMask] | None = None,
    quality_areas: dict[int, float] | None = None,
    border_by_frame: dict[int, BorderFrameConstraint] | None = None,
    visible_rectangle=None,
    right_anchor_metrics: tuple[float, float] | None = None,
    *,
    recall_floor: float,
    frame_iou_floor: float = 0.0,
    max_frame_hausdorff_px: float | None = None,
    quality_mode: str = "mean_iou",
) -> EdgeMetrics | None:
    if len(left.values) == 0 or len(right.values) == 0:
        return None
    left_values = left.values
    right_values = right.values
    # Match the production/overlay interpolation exactly.  It resamples both
    # endpoint polygons, even when they already contain the same point count.
    count = max(8, len(left_values), len(right_values))
    left_points = (
        left.sampled
        if len(left.sampled) == count
        else _numpy_resample(left_values, count)
    )
    right_sampled = (
        right.sampled
        if len(right.sampled) == count
        else _numpy_resample(right_values, count)
    )
    right_points = _fast_numpy_align(left_points, right_sampled)
    span = max(right.keyframe.frame - left.keyframe.frame, 1)
    iou_sum = 0.0
    minimum_recall = 1.0
    minimum_iou = 1.0
    quality_sum = 0.0
    evaluated = 0
    # The left endpoint is counted by the preceding edge (or initialization).
    for frame in frames[start_index + 1 : end_index + 1]:
        if frame == right.keyframe.frame:
            # Exact keyframes are not interpolated by the production reader.
            # In particular, resampling an endpoint can slightly alter its
            # boundary and must not decide hard-recall feasibility.
            predicted = right.geometry
            predicted_bounds = right.bounds
        else:
            alpha = (frame - left.keyframe.frame) / span
            predicted_points = (1.0 - alpha) * left_points + alpha * right_points
            minimum = np.min(predicted_points, axis=0)
            maximum = np.max(predicted_points, axis=0)
            predicted_bounds = (
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            )
            predicted = None
        raw_area = raw_areas[frame]
        raw_box = raw_bounds[frame]
        overlap_width = max(
            0.0,
            min(predicted_bounds[2], raw_box[2]) - max(predicted_bounds[0], raw_box[0]),
        )
        overlap_height = max(
            0.0,
            min(predicted_bounds[3], raw_box[3]) - max(predicted_bounds[1], raw_box[1]),
        )
        # Bounding-box overlap is a conservative upper bound on polygon
        # intersection. Rejecting below this bound cannot remove a feasible
        # edge and avoids an exact GEOS intersection for gross misses.
        if overlap_width * overlap_height + 1e-12 < recall_floor * raw_area:
            return None
        constraint_mask = raw_by_frame[frame]
        raw = constraint_mask.geometry
        quality_mask = (
            constraint_mask if quality_by_frame is None else quality_by_frame[frame]
        )
        current_border = (
            None if border_by_frame is None else border_by_frame.get(frame)
        )
        if predicted is not None and not border_geometry_feasible(
            predicted, current_border
        ):
            return None
        if frame == right.keyframe.frame and right_anchor_metrics is not None:
            # Every edge ending at this anchor previously recomputed the same
            # two polygon intersections.  The anchor metrics have already
            # been evaluated with the identical geometries and are also used
            # by the feasibility guard, so reusing them changes no objective,
            # constraint, tie-breaker, or stored keyframe.
            recall, iou = right_anchor_metrics
            if recall < recall_floor or iou < frame_iou_floor:
                return None
            if (
                max_frame_hausdorff_px is not None
                and raw.hausdorff_distance(right.geometry) > max_frame_hausdorff_px
            ):
                return None
            utility = _iou_utility(iou, quality_mode)
            if quality_mode == "tail_boundary":
                boundary_scale = math.sqrt(max(raw_area, 1.0))
                boundary_error = (
                    quality_mask.geometry.hausdorff_distance(right.geometry)
                    / boundary_scale
                )
                utility -= 20.0 * boundary_error * boundary_error
            iou_sum += iou
            quality_sum += utility
            minimum_iou = min(minimum_iou, iou)
            minimum_recall = min(minimum_recall, recall)
            evaluated += 1
            continue
        if predicted is None:
            predicted = geometry_from_arrays([predicted_points])
        if not border_geometry_feasible(
            predicted, current_border
        ):
            return None
        intersection = float(raw.intersection(predicted).area)
        constraint_recall = intersection / raw_area if raw_area else 1.0
        if constraint_recall < recall_floor:
            return None
        if quality_mask is constraint_mask:
            # Border expansion intentionally leaves unaffected observations as
            # the exact same RawMask object.  Their quality and constraint
            # intersections are therefore identical and need only one GEOS
            # call.  About half of the current benchmark frames use this path.
            quality_area = raw_area
            quality_intersection = intersection
            quality_recall = constraint_recall
        else:
            quality_area = (
                float(quality_mask.geometry.area)
                if quality_areas is None
                else quality_areas[frame]
            )
            quality_intersection = float(
                quality_mask.geometry.intersection(predicted).area
            )
            quality_recall = (
                quality_intersection / quality_area if quality_area else 1.0
            )
        recall = min(constraint_recall, quality_recall)
        if recall < recall_floor:
            return None
        metric_predicted = _visible_metric_geometry(
            predicted, visible_rectangle
        )
        if current_border is not None:
            metric_predicted = metric_predicted.intersection(
                current_border.quality_domain
            )
            metric_quality = quality_mask.geometry.intersection(
                current_border.quality_domain
            )
            quality_area = float(metric_quality.area)
            quality_intersection = float(
                metric_quality.intersection(metric_predicted).area
            )
        predicted_area = float(metric_predicted.area)
        union = quality_area + predicted_area - quality_intersection
        iou = quality_intersection / union if union else 1.0
        if iou < frame_iou_floor:
            return None
        if (
            max_frame_hausdorff_px is not None
            and raw.hausdorff_distance(predicted) > max_frame_hausdorff_px
        ):
            return None
        iou_sum += iou
        utility = _iou_utility(iou, quality_mode)
        if quality_mode == "tail_boundary":
            # Trusted consensus contours are temporally denoised, so a soft
            # normalized Hausdorff term can expose narrow/tip failures that
            # area IoU misses without turning one raw contour pixel into a
            # feasibility constraint.
            boundary_scale = math.sqrt(max(raw_area, 1.0))
            boundary_error = (
                quality_mask.geometry.hausdorff_distance(predicted) / boundary_scale
            )
            utility -= 20.0 * boundary_error * boundary_error
        quality_sum += utility
        minimum_iou = min(minimum_iou, iou)
        minimum_recall = min(minimum_recall, recall)
        evaluated += 1
    return EdgeMetrics(
        start_index=start_index,
        start_anchor_index=start_anchor_index,
        end_index=end_index,
        end_anchor_index=end_anchor_index,
        iou_sum=iou_sum,
        frame_count=evaluated,
        min_recall=minimum_recall,
        quality_sum=quality_sum,
        min_iou=minimum_iou,
    )


def _evaluate_anchor_edge_frame_batch_reference(
    start_index: int,
    start_anchor_index: int,
    end_index: int,
    end_anchor_index: int,
    frames: list[int],
    left: _PreparedAnchor,
    right: _PreparedAnchor,
    raw_by_frame: dict[int, RawMask],
    raw_areas: dict[int, float],
    raw_bounds: dict[int, tuple[float, float, float, float]],
    quality_by_frame: dict[int, RawMask] | None = None,
    quality_areas: dict[int, float] | None = None,
    right_anchor_metrics: tuple[float, float] | None = None,
    *,
    recall_floor: float,
    frame_iou_floor: float = 0.0,
    quality_mode: str = "mean_iou",
    edge_batch_size: int = 4,
) -> EdgeMetrics | None:
    """Rejected frame-direction batching retained as a benchmark reference.

    Polygon construction, validity, intersection, and area are executed by
    Shapely's C-level array loops.  Metric accumulation remains in frame order
    so the DP receives the same floating-point values and tie breakers as the
    scalar implementation.  It is not used by the optimizer: early-rejection
    overhead made it slower than scalar evaluation on real tracks.  Rare
    invalid interpolations retain the robust scalar repair path.
    """

    if len(left.values) == 0 or len(right.values) == 0:
        return None
    count = max(8, len(left.values), len(right.values))
    left_points = (
        left.sampled
        if len(left.sampled) == count
        else _numpy_resample(left.values, count)
    )
    right_sampled = (
        right.sampled
        if len(right.sampled) == count
        else _numpy_resample(right.values, count)
    )
    right_points = _fast_numpy_align(left_points, right_sampled)
    span = max(right.keyframe.frame - left.keyframe.frame, 1)
    iou_sum = 0.0
    minimum_recall = 1.0
    minimum_iou = 1.0
    quality_sum = 0.0
    evaluated = 0
    intermediate_frames = frames[start_index + 1 : end_index]
    batch_size = max(2, int(edge_batch_size))

    for offset in range(0, len(intermediate_frames), batch_size):
        batch_frames = intermediate_frames[offset : offset + batch_size]
        alpha = np.asarray(
            [
                (frame - left.keyframe.frame) / span
                for frame in batch_frames
            ],
            dtype=np.float64,
        )[:, None, None]
        predicted_points = (
            (1.0 - alpha) * left_points[None, :, :]
            + alpha * right_points[None, :, :]
        )

        minima = np.min(predicted_points, axis=1)
        maxima = np.max(predicted_points, axis=1)
        raw_area_values = np.asarray(
            [raw_areas[frame] for frame in batch_frames], dtype=np.float64
        )
        raw_box_values = np.asarray(
            [raw_bounds[frame] for frame in batch_frames], dtype=np.float64
        )
        overlap_widths = np.maximum(
            0.0,
            np.minimum(maxima[:, 0], raw_box_values[:, 2])
            - np.maximum(minima[:, 0], raw_box_values[:, 0]),
        )
        overlap_heights = np.maximum(
            0.0,
            np.minimum(maxima[:, 1], raw_box_values[:, 3])
            - np.maximum(minima[:, 1], raw_box_values[:, 1]),
        )
        if np.any(
            overlap_widths * overlap_heights + 1e-12
            < recall_floor * raw_area_values
        ):
            return None

        predicted = np.asarray(shapely.polygons(predicted_points), dtype=object)
        valid = np.asarray(shapely.is_valid(predicted), dtype=bool)
        nonempty = ~np.asarray(shapely.is_empty(predicted), dtype=bool)
        positive = np.asarray(shapely.area(predicted), dtype=np.float64) > 0.0
        repair_indices = np.flatnonzero(~(valid & nonempty & positive))
        for index in repair_indices:
            predicted[index] = geometry_from_arrays([predicted_points[index]])

        predicted_areas = np.asarray(shapely.area(predicted), dtype=np.float64)
        constraint_masks = [raw_by_frame[frame] for frame in batch_frames]
        constraint_geometries = np.asarray(
            [mask.geometry for mask in constraint_masks], dtype=object
        )
        intersections = np.asarray(
            shapely.area(shapely.intersection(constraint_geometries, predicted)),
            dtype=np.float64,
        )
        constraint_recalls = np.divide(
            intersections,
            raw_area_values,
            out=np.ones_like(intersections),
            where=raw_area_values != 0.0,
        )
        if np.any(constraint_recalls < recall_floor):
            return None

        if quality_by_frame is None:
            quality_masks = constraint_masks
        else:
            quality_masks = [quality_by_frame[frame] for frame in batch_frames]
        quality_area_values = np.asarray(
            [
                (
                    raw_area_values[index]
                    if quality_masks[index] is constraint_masks[index]
                    else (
                        float(quality_masks[index].geometry.area)
                        if quality_areas is None
                        else quality_areas[frame]
                    )
                )
                for index, frame in enumerate(batch_frames)
            ],
            dtype=np.float64,
        )
        quality_intersections = intersections.copy()
        changed = np.asarray(
            [
                quality_masks[index] is not constraint_masks[index]
                for index in range(len(batch_frames))
            ],
            dtype=bool,
        )
        if np.any(changed):
            quality_geometries = np.asarray(
                [
                    quality_masks[index].geometry
                    for index in np.flatnonzero(changed)
                ],
                dtype=object,
            )
            quality_intersections[changed] = np.asarray(
                shapely.area(
                    shapely.intersection(quality_geometries, predicted[changed])
                ),
                dtype=np.float64,
            )
        quality_recalls = np.divide(
            quality_intersections,
            quality_area_values,
            out=np.ones_like(quality_intersections),
            where=quality_area_values != 0.0,
        )
        recalls = np.minimum(constraint_recalls, quality_recalls)
        if np.any(recalls < recall_floor):
            return None
        unions = quality_area_values + predicted_areas - quality_intersections
        ious = np.divide(
            quality_intersections,
            unions,
            out=np.ones_like(quality_intersections),
            where=unions != 0.0,
        )
        if np.any(ious < frame_iou_floor):
            return None

        # Preserve the scalar summation order and exact tie-breaking inputs.
        for recall, iou in zip(recalls, ious):
            recall_value = float(recall)
            iou_value = float(iou)
            iou_sum += iou_value
            quality_sum += _iou_utility(iou_value, quality_mode)
            minimum_iou = min(minimum_iou, iou_value)
            minimum_recall = min(minimum_recall, recall_value)
            evaluated += 1

    # The right endpoint is an exact stored keyframe and is intentionally not
    # reconstructed by the vectorized interpolation path.
    frame = frames[end_index]
    if right_anchor_metrics is None:
        constraint_mask = raw_by_frame[frame]
        quality_mask = (
            constraint_mask if quality_by_frame is None else quality_by_frame[frame]
        )
        predicted_area = float(right.geometry.area)
        intersection = float(constraint_mask.geometry.intersection(right.geometry).area)
        constraint_area = raw_areas[frame]
        constraint_recall = intersection / constraint_area if constraint_area else 1.0
        if quality_mask is constraint_mask:
            quality_area = constraint_area
            quality_intersection = intersection
            quality_recall = constraint_recall
        else:
            quality_area = (
                float(quality_mask.geometry.area)
                if quality_areas is None
                else quality_areas[frame]
            )
            quality_intersection = float(
                quality_mask.geometry.intersection(right.geometry).area
            )
            quality_recall = quality_intersection / quality_area if quality_area else 1.0
        recall = min(constraint_recall, quality_recall)
        union = quality_area + predicted_area - quality_intersection
        iou = quality_intersection / union if union else 1.0
    else:
        recall, iou = right_anchor_metrics
    if recall < recall_floor or iou < frame_iou_floor:
        return None
    iou_sum += iou
    quality_sum += _iou_utility(iou, quality_mode)
    minimum_iou = min(minimum_iou, iou)
    minimum_recall = min(minimum_recall, recall)
    evaluated += 1
    return EdgeMetrics(
        start_index=start_index,
        start_anchor_index=start_anchor_index,
        end_index=end_index,
        end_anchor_index=end_anchor_index,
        iou_sum=iou_sum,
        frame_count=evaluated,
        min_recall=minimum_recall,
        quality_sum=quality_sum,
        min_iou=minimum_iou,
    )


def _evaluate_anchor_group(
    context: _EdgeContext,
    start_index: int,
    end_index: int,
    pairs: list[tuple[int, int, _PreparedAnchor, _PreparedAnchor]],
) -> list[EdgeMetrics]:
    """Evaluate anchor pairs together while retaining frame-wise rejection.

    Unlike batching frames within one edge, this layout preserves immediate
    rejection at every frame.  It only moves the independent anchor-pair GEOS
    calls into Shapely's C array loop.
    """

    if not pairs:
        return []
    prepared: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    point_count: int | None = None
    for start_anchor_index, end_anchor_index, left, right in pairs:
        if len(left.values) == 0 or len(right.values) == 0:
            continue
        count = max(8, len(left.values), len(right.values))
        if point_count is None:
            point_count = count
        elif count != point_count:
            # The Superior stored-vertex contract always has one point count.
            # Generic callers with mixed polygon sizes retain the scalar path.
            return []
        left_points = (
            left.sampled
            if len(left.sampled) == count
            else _numpy_resample(left.values, count)
        )
        right_sampled = (
            right.sampled
            if len(right.sampled) == count
            else _numpy_resample(right.values, count)
        )
        prepared.append(
            (
                start_anchor_index,
                end_anchor_index,
                left_points,
                _fast_numpy_align(left_points, right_sampled),
            )
        )
    if len(prepared) != len(pairs):
        return []

    size = len(prepared)
    alive = np.ones(size, dtype=bool)
    iou_sums = [0.0] * size
    quality_sums = [0.0] * size
    minimum_ious = [1.0] * size
    minimum_recalls = [1.0] * size
    evaluated_counts = [0] * size
    left_stack = np.stack([item[2] for item in prepared], axis=0)
    right_stack = np.stack([item[3] for item in prepared], axis=0)
    left_frame = pairs[0][2].keyframe.frame
    right_frame = pairs[0][3].keyframe.frame
    span = max(right_frame - left_frame, 1)

    for frame in context.frames[start_index + 1 : end_index]:
        active = np.flatnonzero(alive)
        if len(active) == 0:
            break
        alpha = (frame - left_frame) / span
        points = (
            (1.0 - alpha) * left_stack[active]
            + alpha * right_stack[active]
        )
        minima = np.min(points, axis=1)
        maxima = np.max(points, axis=1)
        raw_area = context.raw_areas[frame]
        raw_box = context.raw_bounds[frame]
        overlap_widths = np.maximum(
            0.0,
            np.minimum(maxima[:, 0], raw_box[2])
            - np.maximum(minima[:, 0], raw_box[0]),
        )
        overlap_heights = np.maximum(
            0.0,
            np.minimum(maxima[:, 1], raw_box[3])
            - np.maximum(minima[:, 1], raw_box[1]),
        )
        bbox_valid = (
            overlap_widths * overlap_heights + 1e-12
            >= context.recall_floor * raw_area
        )
        alive[active[~bbox_valid]] = False
        active = active[bbox_valid]
        points = points[bbox_valid]
        if len(active) == 0:
            continue

        predicted = np.asarray(shapely.polygons(points), dtype=object)
        valid = np.asarray(shapely.is_valid(predicted), dtype=bool)
        nonempty = ~np.asarray(shapely.is_empty(predicted), dtype=bool)
        positive = np.asarray(shapely.area(predicted), dtype=np.float64) > 0.0
        for local_index in np.flatnonzero(~(valid & nonempty & positive)):
            predicted[local_index] = geometry_from_arrays([points[local_index]])

        border_constraint = context.border_by_frame.get(frame)
        single_side_intersections = None
        if border_constraint is not None:
            predicted_bounds = np.asarray(shapely.bounds(predicted), dtype=np.float64)
            border_valid = np.ones(len(predicted), dtype=bool)
            bound_columns = {"left": 0, "top": 1, "right": 2, "bottom": 3}
            for side in border_constraint.sides:
                coordinates = predicted_bounds[:, bound_columns[side.side]]
                if side.side in {"left", "top"}:
                    border_valid &= coordinates <= side.required_coordinate + 1e-9
                else:
                    border_valid &= coordinates >= side.required_coordinate - 1e-9
                local_intersections = np.asarray(
                    shapely.area(
                        shapely.intersection(side.visible_reference, predicted)
                    ),
                    dtype=np.float64,
                )
                if len(border_constraint.sides) == 1:
                    single_side_intersections = local_intersections
                if side.visible_area:
                    border_valid &= (
                        local_intersections / side.visible_area + 1e-12
                        >= border_constraint.local_recall_floor
                    )
            alive[active[~border_valid]] = False
            active = active[border_valid]
            predicted = predicted[border_valid]
            if single_side_intersections is not None:
                single_side_intersections = single_side_intersections[border_valid]
            if len(active) == 0:
                continue

        metric_predicted = predicted
        if border_constraint is not None:
            metric_predicted = np.asarray(
                shapely.intersection(
                    predicted, border_constraint.quality_domain
                ),
                dtype=object,
            )
        elif context.visible_rectangle is not None:
            geometry_bounds = np.asarray(shapely.bounds(predicted), dtype=np.float64)
            visible_bounds = context.visible_rectangle.bounds
            outside = (
                (geometry_bounds[:, 0] < visible_bounds[0])
                | (geometry_bounds[:, 1] < visible_bounds[1])
                | (geometry_bounds[:, 2] > visible_bounds[2])
                | (geometry_bounds[:, 3] > visible_bounds[3])
            )
            if np.any(outside):
                metric_predicted = predicted.copy()
                metric_predicted[outside] = shapely.intersection(
                    predicted[outside], context.visible_rectangle
                )
        predicted_areas = np.asarray(
            shapely.area(metric_predicted), dtype=np.float64
        )

        constraint_mask = context.raw_by_frame[frame]
        intersections = np.asarray(
            shapely.area(shapely.intersection(constraint_mask.geometry, predicted)),
            dtype=np.float64,
        )
        constraint_recalls = (
            intersections / raw_area
            if raw_area
            else np.ones_like(intersections)
        )
        constraint_valid = constraint_recalls >= context.recall_floor
        alive[active[~constraint_valid]] = False
        active = active[constraint_valid]
        predicted = predicted[constraint_valid]
        predicted_areas = predicted_areas[constraint_valid]
        intersections = intersections[constraint_valid]
        constraint_recalls = constraint_recalls[constraint_valid]
        if single_side_intersections is not None:
            single_side_intersections = single_side_intersections[constraint_valid]
        if len(active) == 0:
            continue

        quality_mask = context.quality_by_frame[frame]
        metric_quality_intersections = None
        if quality_mask is constraint_mask:
            quality_area = raw_area
            quality_intersections = intersections
            quality_recalls = constraint_recalls
        else:
            quality_area = context.quality_areas[frame]
            if single_side_intersections is not None:
                metric_quality = context.metric_quality_by_frame[frame]
                metric_quality_intersections = np.asarray(
                    shapely.area(shapely.intersection(metric_quality, predicted)),
                    dtype=np.float64,
                )
                quality_intersections = (
                    metric_quality_intersections + single_side_intersections
                )
            else:
                quality_intersections = np.asarray(
                    shapely.area(
                        shapely.intersection(quality_mask.geometry, predicted)
                    ),
                    dtype=np.float64,
                )
            quality_recalls = (
                quality_intersections / quality_area
                if quality_area
                else np.ones_like(quality_intersections)
            )
        recalls = np.minimum(constraint_recalls, quality_recalls)
        if border_constraint is not None:
            metric_quality = context.metric_quality_by_frame[frame]
            quality_area = context.metric_quality_areas[frame]
            quality_intersections = (
                metric_quality_intersections
                if metric_quality_intersections is not None
                else np.asarray(
                    shapely.area(
                        shapely.intersection(metric_quality, predicted)
                    ),
                    dtype=np.float64,
                )
            )
        unions = quality_area + predicted_areas - quality_intersections
        ious = np.divide(
            quality_intersections,
            unions,
            out=np.ones_like(quality_intersections),
            where=unions != 0.0,
        )
        metric_valid = (recalls >= context.recall_floor) & (
            ious >= context.frame_iou_floor
        )
        alive[active[~metric_valid]] = False
        for global_index, recall, iou in zip(
            active[metric_valid], recalls[metric_valid], ious[metric_valid]
        ):
            index = int(global_index)
            recall_value = float(recall)
            iou_value = float(iou)
            iou_sums[index] += iou_value
            quality_sums[index] += _iou_utility(iou_value, context.quality_mode)
            minimum_ious[index] = min(minimum_ious[index], iou_value)
            minimum_recalls[index] = min(minimum_recalls[index], recall_value)
            evaluated_counts[index] += 1

    output: list[EdgeMetrics] = []
    for index in np.flatnonzero(alive):
        pair_index = int(index)
        start_anchor_index, end_anchor_index, _left, right = pairs[pair_index]
        if not border_geometry_feasible(
            right.geometry, context.border_by_frame.get(right_frame)
        ):
            continue
        recall, iou = context.anchor_metrics_by_frame[end_index][end_anchor_index]
        if recall < context.recall_floor or iou < context.frame_iou_floor:
            continue
        iou_sums[pair_index] += iou
        quality_sums[pair_index] += _iou_utility(iou, context.quality_mode)
        minimum_ious[pair_index] = min(minimum_ious[pair_index], iou)
        minimum_recalls[pair_index] = min(minimum_recalls[pair_index], recall)
        evaluated_counts[pair_index] += 1
        output.append(
            EdgeMetrics(
                start_index=start_index,
                start_anchor_index=start_anchor_index,
                end_index=end_index,
                end_anchor_index=end_anchor_index,
                iou_sum=iou_sums[pair_index],
                frame_count=evaluated_counts[pair_index],
                min_recall=minimum_recalls[pair_index],
                quality_sum=quality_sums[pair_index],
                min_iou=minimum_ious[pair_index],
            )
        )
    return output


def _evaluate_edge(
    start_index: int,
    end_index: int,
    frames: list[int],
    anchors: list[Keyframe],
    raw_by_frame: dict[int, RawMask],
    *,
    recall_floor: float,
    frame_iou_floor: float = 0.0,
    max_frame_hausdorff_px: float | None = None,
    quality_mode: str = "mean_iou",
) -> EdgeMetrics | None:
    """Compatibility wrapper for single-anchor tests and callers."""

    raw_areas = {frame: float(raw.geometry.area) for frame, raw in raw_by_frame.items()}
    raw_bounds = {frame: raw.geometry.bounds for frame, raw in raw_by_frame.items()}
    return _evaluate_anchor_edge(
        start_index,
        0,
        end_index,
        0,
        frames,
        _prepare_anchor(anchors[start_index]),
        _prepare_anchor(anchors[end_index]),
        raw_by_frame,
        raw_areas,
        raw_bounds,
        raw_by_frame,
        raw_areas,
        recall_floor=recall_floor,
        frame_iou_floor=frame_iou_floor,
        max_frame_hausdorff_px=max_frame_hausdorff_px,
        quality_mode=quality_mode,
    )


def _evaluate_context_start_edges(
    context: _EdgeContext, start_index: int
) -> tuple[int, list[EdgeMetrics]]:
    frames = context.frames
    start = frames[start_index]
    final_frame = (
        frames[-1]
        if context.max_edge_span_frames <= 0
        else start + int(context.max_edge_span_frames)
    )
    stop = bisect.bisect_right(frames, final_frame, lo=start_index + 1)
    evaluated = 0
    output: list[EdgeMetrics] = []
    for end_index in range(start_index + 1, stop):
        pairs = [
            (start_anchor_index, end_anchor_index, left, right)
            for start_anchor_index, left in enumerate(
                context.prepared_by_frame[start_index]
            )
            for end_anchor_index, right in enumerate(
                context.prepared_by_frame[end_index]
            )
        ]
        evaluated += len(pairs)
        point_counts = {
            max(8, len(left.values), len(right.values))
            for _start_anchor, _end_anchor, left, right in pairs
        }
        vectorized = (
            context.edge_batch_size > 1
            and context.max_frame_hausdorff_px is None
            and context.quality_mode != "tail_boundary"
            and len(point_counts) == 1
        )
        if vectorized:
            grouped_edges: list[EdgeMetrics] = []
            for offset in range(0, len(pairs), context.edge_batch_size):
                batch = pairs[offset : offset + context.edge_batch_size]
                grouped_edges.extend(
                    _evaluate_anchor_group(context, start_index, end_index, batch)
                )
            output.extend(grouped_edges)
            continue
        # Scalar reference/fallback path.
        for start_anchor_index, end_anchor_index, left, right in pairs:
            edge = _evaluate_anchor_edge(
                start_index,
                start_anchor_index,
                end_index,
                end_anchor_index,
                frames,
                left,
                right,
                context.raw_by_frame,
                context.raw_areas,
                context.raw_bounds,
                context.quality_by_frame,
                context.quality_areas,
                border_by_frame=context.border_by_frame,
                visible_rectangle=context.visible_rectangle,
                right_anchor_metrics=(
                    context.anchor_metrics_by_frame[end_index][end_anchor_index]
                ),
                recall_floor=context.recall_floor,
                frame_iou_floor=context.frame_iou_floor,
                max_frame_hausdorff_px=context.max_frame_hausdorff_px,
                quality_mode=context.quality_mode,
            )
            if edge is not None:
                output.append(edge)
    return evaluated, output


def _evaluate_process_start_edges(
    start_index: int,
) -> tuple[int, list[EdgeMetrics]]:
    if _EDGE_PROCESS_CONTEXT is None:
        raise RuntimeError("edge process context is not initialized")
    return _evaluate_context_start_edges(_EDGE_PROCESS_CONTEXT, start_index)


def _prune_node_states(
    states: dict[int, _NodeState], *, improvement_epsilon: float
) -> dict[int, _NodeState]:
    """Drop states dominated by a solution with no more keys and no less utility."""

    output: dict[int, _NodeState] = {}
    best_quality = -math.inf
    for key_count in sorted(states):
        state = states[key_count]
        if state.quality_sum > best_quality + improvement_epsilon:
            output[key_count] = state
            best_quality = state.quality_sum
    return output


def _reconstruct_local_path(
    back_by_node: list[list[dict[int, _NodeState]]],
    anchors_by_frame: list[list[Keyframe]],
    key_count: int,
    final_anchor_index: int,
) -> tuple[Keyframe, ...]:
    node_index = len(anchors_by_frame) - 1
    anchor_index = int(final_anchor_index)
    remaining = int(key_count)
    path: list[Keyframe] = []
    while node_index >= 0 and remaining >= 1:
        path.append(anchors_by_frame[node_index][anchor_index])
        if node_index == 0:
            break
        state = back_by_node[node_index][anchor_index][remaining]
        node_index = state.previous_index
        anchor_index = state.previous_anchor_index
        remaining = state.previous_key_count
    path.reverse()
    if not path or path[0].frame != anchors_by_frame[0][0].frame:
        raise RuntimeError("failed to reconstruct a Pareto keyframe path")
    return tuple(path)


def canonicalize_selected_path(
    keyframes: tuple[Keyframe, ...],
) -> tuple[Keyframe, ...]:
    """Store the exact pairwise alignment used by the polygon reader.

    Geometry is unchanged at every keyframe.  Only winding and cyclic origin
    are changed, so point_index interpolation in an editor produces the exact
    path represented by ``linear_polygon_index_v1``.
    """

    output: list[Keyframe] = []
    reference: np.ndarray | None = None
    for keyframe in keyframes:
        component = _primary_component(keyframe)
        if component is None:
            output.append(keyframe)
            continue
        points = np.asarray(component.values, dtype=np.float64)
        if reference is not None:
            if len(points) != len(reference):
                raise ValueError("stored vertex contract requires fixed point count")
            points = _fast_numpy_align(reference, points)
        output.append(
            Keyframe(
                keyframe.frame,
                ((0, Component("polygon", points.tolist())),),
            )
        )
        reference = points
    return tuple(output)


def optimize_segment_pareto(
    segment: Segment,
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    quality_masks: dict[tuple[int, str], RawMask] | None = None,
    border_constraints: dict[tuple[int, str], BorderFrameConstraint] | None = None,
    visible_bounds: tuple[float, float, float, float] | None = None,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    anchor_iou_floor: float = 0.0,
    anchor_relative_iou_margin: float | None = None,
    frame_iou_floor: float = 0.0,
    anchor_point_strategy: str = "uniform",
    max_frame_hausdorff_px: float | None = None,
    max_edge_span_frames: int = 60,
    point_count: int = 23,
    max_anchor_scale: float = 1.25,
    anchor_state_count: int = 1,
    anchor_expansion: float = 0.04,
    edge_threads: int = 1,
    edge_processes: int = 1,
    dominance_epsilon: float = 1e-10,
    quality_mode: str = "mean_iou",
    stored_vertex_contract: bool = False,
    pair_vote_states: bool = False,
    edge_batch_size: int = 32,
    candidate_mode: str = "legacy",
    temporal_window_radii: tuple[int, int, int] = (2, 5, 10),
    temporal_recall_quantile: float = 0.90,
    extra_anchor_states: dict[tuple[int, str], tuple[Keyframe, ...]] | None = None,
) -> tuple[list[LocalParetoPoint], int, int, int]:
    """Return every non-dominated (key count, IoU) path for one segment."""

    raw_by_frame = {
        frame: raw
        for (frame, track_id), raw in raw_masks.items()
        if track_id == segment.track_id
        and segment.first_frame <= frame <= segment.last_frame
        and start_frame <= frame <= end_frame
    }
    frames = sorted(raw_by_frame)
    if not frames:
        return [], 0, 0, 0
    quality_by_frame = {
        frame: (
            raw_by_frame[frame]
            if quality_masks is None
            else quality_masks.get((frame, segment.track_id), raw_by_frame[frame])
        )
        for frame in frames
    }
    border_by_frame = {
        frame: constraint
        for (frame, track_id), constraint in (border_constraints or {}).items()
        if track_id == segment.track_id and frame in raw_by_frame
    }
    visible_rectangle = (
        None if visible_bounds is None else shapely.box(*visible_bounds)
    )
    pair_vote_sources = (
        _build_pair_vote_sources(
            frames,
            quality_by_frame,
            point_count=point_count,
            max_edge_span_frames=max_edge_span_frames,
        )
        if pair_vote_states
        else {}
    )
    if quality_mode not in {
        "mean_iou",
        "log_iou",
        "tail_harmonic",
        "tail_boundary",
    }:
        raise ValueError(f"unsupported quality mode: {quality_mode}")
    if candidate_mode not in {
        "legacy",
        "legacy_rdp_fallback",
        "temporal7",
        *_COMBINED_UNION_NAMES,
        *_TEMPORAL_UNION_NAMES,
        *_GEOMETRIC_UNION_NAMES,
    }:
        raise ValueError(f"unsupported candidate mode: {candidate_mode}")
    if candidate_mode == "temporal7":
        anchors_by_frame = [
            _make_temporal7_anchors(
                frame,
                quality_by_frame,
                raw_by_frame[frame],
                quality_by_frame[frame],
                recall_floor=recall_floor,
                point_count=point_count,
                max_anchor_scale=max_anchor_scale,
                window_radii=temporal_window_radii,
                recall_quantile=temporal_recall_quantile,
                border_constraint=border_by_frame.get(frame),
                visible_rectangle=visible_rectangle,
                extra_sources=(extra_anchor_states or {}).get(
                    (frame, segment.track_id), ()
                ),
            )
            for frame in frames
        ]
    elif candidate_mode in {"legacy", "legacy_rdp_fallback"}:
        anchors_by_frame = [
            _make_feasible_anchors(
                segment,
                raw_by_frame[frame],
                quality_raw=quality_by_frame[frame],
                recall_floor=recall_floor,
                point_count=point_count,
                max_anchor_scale=max_anchor_scale,
                anchor_state_count=anchor_state_count,
                anchor_expansion=anchor_expansion,
                anchor_iou_floor=anchor_iou_floor,
                anchor_relative_iou_margin=anchor_relative_iou_margin,
                anchor_point_strategy=anchor_point_strategy,
                max_anchor_hausdorff_px=max_frame_hausdorff_px,
                stored_vertex_contract=stored_vertex_contract,
                corner_preserving_source=(
                    candidate_mode == "legacy_rdp_fallback"
                ),
                extra_sources=pair_vote_sources.get(frame),
                border_constraint=border_by_frame.get(frame),
                visible_rectangle=visible_rectangle,
            )
            for frame in frames
        ]
    else:
        combined_names = _COMBINED_UNION_NAMES.get(candidate_mode)
        temporal_names = (
            combined_names[0]
            if combined_names is not None
            else _TEMPORAL_UNION_NAMES.get(candidate_mode)
        )
        geometric_names = (
            combined_names[1]
            if combined_names is not None
            else _GEOMETRIC_UNION_NAMES.get(candidate_mode)
        )
        anchors_by_frame = []
        for frame in frames:
            legacy = _make_feasible_anchors(
                segment,
                raw_by_frame[frame],
                quality_raw=quality_by_frame[frame],
                recall_floor=recall_floor,
                point_count=point_count,
                max_anchor_scale=max_anchor_scale,
                anchor_state_count=anchor_state_count,
                anchor_expansion=anchor_expansion,
                anchor_iou_floor=anchor_iou_floor,
                anchor_relative_iou_margin=anchor_relative_iou_margin,
                anchor_point_strategy=anchor_point_strategy,
                max_anchor_hausdorff_px=max_frame_hausdorff_px,
                stored_vertex_contract=stored_vertex_contract,
                corner_preserving_source=(
                    candidate_mode
                    == "legacy_temporal_recall_interior_rdp_fallback"
                ),
                extra_sources=pair_vote_sources.get(frame),
                border_constraint=border_by_frame.get(frame),
                visible_rectangle=visible_rectangle,
            )
            supplemental: list[list[Keyframe]] = []
            if temporal_names is not None and not (
                candidate_mode
                in {
                    "legacy_temporal_recall_interior",
                    "legacy_temporal_recall_interior_rdp_fallback",
                }
                and border_by_frame.get(frame) is not None
            ):
                try:
                    temporal = _make_temporal7_anchors(
                        frame,
                        quality_by_frame,
                        raw_by_frame[frame],
                        quality_by_frame[frame],
                        recall_floor=recall_floor,
                        point_count=point_count,
                        max_anchor_scale=max_anchor_scale,
                        window_radii=temporal_window_radii,
                        recall_quantile=temporal_recall_quantile,
                        border_constraint=border_by_frame.get(frame),
                        visible_rectangle=visible_rectangle,
                        extra_sources=(extra_anchor_states or {}).get(
                            (frame, segment.track_id), ()
                        ),
                        candidate_names=temporal_names,
                    )
                except RuntimeError:
                    # Supplemental temporal states are optional.  A complex
                    # contour may be feasible through the legacy/RDP base yet
                    # have no feasible temporal consensus at this frame.  The
                    # DP must retain the base family rather than aborting the
                    # whole video.
                    temporal = []
                supplemental.append(temporal)
            if geometric_names is not None:
                supplemental.append(
                    _make_geometric_anchors(
                        frame,
                        quality_by_frame,
                        raw_by_frame[frame],
                        quality_by_frame[frame],
                        recall_floor=recall_floor,
                        point_count=point_count,
                        max_anchor_scale=max_anchor_scale,
                        border_constraint=border_by_frame.get(frame),
                        visible_rectangle=visible_rectangle,
                        candidate_names=geometric_names,
                    )
                )
            anchors_by_frame.append(_merge_anchor_sets(legacy, *supplemental))
    prepared_by_frame = [
        [
            _prepare_anchor(anchor, stored_vertex_contract=stored_vertex_contract)
            for anchor in anchors
        ]
        for anchors in anchors_by_frame
    ]
    raw_areas = {frame: float(raw.geometry.area) for frame, raw in raw_by_frame.items()}
    raw_bounds = {frame: raw.geometry.bounds for frame, raw in raw_by_frame.items()}
    quality_areas = {
        frame: float(raw.geometry.area) for frame, raw in quality_by_frame.items()
    }
    metric_quality_by_frame = {
        frame: (
            quality_by_frame[frame].geometry
            if border_by_frame.get(frame) is None
            else quality_by_frame[frame].geometry.intersection(
                border_by_frame[frame].quality_domain
            )
        )
        for frame in frames
    }
    metric_quality_areas = {
        frame: float(geometry.area)
        for frame, geometry in metric_quality_by_frame.items()
    }
    anchor_metrics_by_frame = [
        [
            _anchor_metrics(
                raw_by_frame[frame],
                anchor,
                quality_by_frame[frame],
                visible_rectangle,
                border_by_frame.get(frame),
            )
            for anchor in anchors
        ]
        for frame, anchors in zip(frames, anchors_by_frame)
    ]
    if any(
        recall < recall_floor
        for metrics in anchor_metrics_by_frame
        for recall, _iou in metrics
    ):
        raise RuntimeError(
            f"segment {segment.segment_id} contains an anchor below recall "
            f"floor {recall_floor:.6f}"
        )
    if any(
        iou < anchor_iou_floor
        for metrics in anchor_metrics_by_frame
        for _recall, iou in metrics
    ):
        raise RuntimeError(
            f"segment {segment.segment_id} contains an anchor below IoU "
            f"floor {anchor_iou_floor:.6f}"
        )

    incoming: list[list[list[EdgeMetrics]]] = [
        [[] for _anchor in anchors] for anchors in anchors_by_frame
    ]
    edge_context = _EdgeContext(
        frames=frames,
        prepared_by_frame=prepared_by_frame,
        anchor_metrics_by_frame=anchor_metrics_by_frame,
        raw_by_frame=raw_by_frame,
        raw_areas=raw_areas,
        raw_bounds=raw_bounds,
        quality_by_frame=quality_by_frame,
        quality_areas=quality_areas,
        metric_quality_by_frame=metric_quality_by_frame,
        metric_quality_areas=metric_quality_areas,
        border_by_frame=border_by_frame,
        visible_rectangle=visible_rectangle,
        recall_floor=recall_floor,
        frame_iou_floor=frame_iou_floor,
        max_frame_hausdorff_px=max_frame_hausdorff_px,
        max_edge_span_frames=max_edge_span_frames,
        quality_mode=quality_mode,
        edge_batch_size=edge_batch_size,
    )
    requested_threads = min(max(1, int(edge_threads)), max(len(frames) - 1, 1))
    requested_processes = min(max(1, int(edge_processes)), max(len(frames) - 1, 1))
    start_indices = range(max(len(frames) - 1, 0))
    if requested_processes > 1:
        global _EDGE_PROCESS_CONTEXT
        _EDGE_PROCESS_CONTEXT = edge_context
        chunksize = max(1, len(frames) // (requested_processes * 8))
        try:
            with ProcessPoolExecutor(
                max_workers=requested_processes,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                edge_batches = list(
                    executor.map(
                        _evaluate_process_start_edges,
                        start_indices,
                        chunksize=chunksize,
                    )
                )
        finally:
            _EDGE_PROCESS_CONTEXT = None
    elif requested_threads > 1:
        with ThreadPoolExecutor(max_workers=requested_threads) as executor:
            edge_batches = list(
                executor.map(
                    lambda index: _evaluate_context_start_edges(edge_context, index),
                    start_indices,
                )
            )
    else:
        edge_batches = [
            _evaluate_context_start_edges(edge_context, index)
            for index in start_indices
        ]
    edge_evaluations = 0
    feasible_edges = 0
    for evaluated, edges in edge_batches:
        edge_evaluations += evaluated
        feasible_edges += len(edges)
        for edge in edges:
            incoming[edge.end_index][edge.end_anchor_index].append(edge)

    states_by_node: list[list[dict[int, _NodeState]]] = [
        [{} for _anchor in anchors] for anchors in anchors_by_frame
    ]
    for anchor_index, (first_recall, first_iou) in enumerate(
        anchor_metrics_by_frame[0]
    ):
        states_by_node[0][anchor_index][1] = _NodeState(
            iou_sum=first_iou,
            quality_sum=_iou_utility(first_iou, quality_mode),
            min_iou=first_iou,
            min_recall=first_recall,
            current_anchor_index=anchor_index,
            previous_index=-1,
            previous_anchor_index=-1,
            previous_key_count=0,
        )
    for end_index in range(1, len(frames)):
        for end_anchor_index in range(len(anchors_by_frame[end_index])):
            candidates: dict[int, _NodeState] = {}
            for edge in incoming[end_index][end_anchor_index]:
                previous_states = states_by_node[edge.start_index][
                    edge.start_anchor_index
                ]
                for previous_key_count, previous in previous_states.items():
                    key_count = previous_key_count + 1
                    candidate = _NodeState(
                        iou_sum=previous.iou_sum + edge.iou_sum,
                        quality_sum=previous.quality_sum + edge.quality_sum,
                        min_iou=min(previous.min_iou, edge.min_iou),
                        min_recall=min(previous.min_recall, edge.min_recall),
                        current_anchor_index=end_anchor_index,
                        previous_index=edge.start_index,
                        previous_anchor_index=edge.start_anchor_index,
                        previous_key_count=previous_key_count,
                    )
                    current = candidates.get(key_count)
                    if (
                        current is None
                        or candidate.quality_sum > current.quality_sum + 1e-12
                        or (
                            abs(candidate.quality_sum - current.quality_sum) <= 1e-12
                            and (
                                candidate.min_iou > current.min_iou + 1e-12
                                or (
                                    abs(candidate.min_iou - current.min_iou) <= 1e-12
                                    and candidate.iou_sum > current.iou_sum + 1e-12
                                )
                            )
                        )
                    ):
                        candidates[key_count] = candidate
            states_by_node[end_index][end_anchor_index] = _prune_node_states(
                candidates, improvement_epsilon=dominance_epsilon
            )

    final_candidates: dict[int, _NodeState] = {}
    for states in states_by_node[-1]:
        for key_count, candidate in states.items():
            current = final_candidates.get(key_count)
            if (
                current is None
                or candidate.quality_sum > current.quality_sum + 1e-12
                or (
                    abs(candidate.quality_sum - current.quality_sum) <= 1e-12
                    and (
                        candidate.min_iou > current.min_iou + 1e-12
                        or (
                            abs(candidate.min_iou - current.min_iou) <= 1e-12
                            and candidate.iou_sum > current.iou_sum + 1e-12
                        )
                    )
                )
            ):
                final_candidates[key_count] = candidate
    final_states = _prune_node_states(
        final_candidates, improvement_epsilon=dominance_epsilon
    )
    if not final_states:
        raise RuntimeError(
            f"segment {segment.segment_id} has no feasible path at recall "
            f"floor {recall_floor:.6f}; increase max_edge_span_frames only if "
            "the segment is static enough to skip farther"
        )
    points = [
        LocalParetoPoint(
            keyframe_count=key_count,
            iou_sum=state.iou_sum,
            frame_count=len(frames),
            min_recall=state.min_recall,
            # Keep references to the shared candidate anchors while the full
            # local front is combined.  Canonicalizing every Pareto path here
            # copied the same polygon arrays thousands of times and dominated
            # both memory and runtime.  Only the globally selected path needs
            # materialization for SQLite.
            keyframes=_reconstruct_local_path(
                states_by_node,
                anchors_by_frame,
                key_count,
                state.current_anchor_index,
            ),
            quality_sum=state.quality_sum,
            min_iou=state.min_iou,
        )
        for key_count, state in sorted(final_states.items())
    ]
    anchor_state_total = sum(len(anchors) for anchors in anchors_by_frame)
    return points, edge_evaluations, feasible_edges, anchor_state_total


def _optimize_segment_task(
    task: _SegmentTask,
) -> tuple[list[LocalParetoPoint], int, int, int]:
    """Process-safe adapter for independent segment optimization."""

    return optimize_segment_pareto(
        task.segment,
        task.raw_masks,
        quality_masks=task.quality_masks,
        border_constraints=task.border_constraints,
        visible_bounds=task.visible_bounds,
        start_frame=task.start_frame,
        end_frame=task.end_frame,
        recall_floor=task.recall_floor,
        anchor_iou_floor=task.anchor_iou_floor,
        anchor_relative_iou_margin=task.anchor_relative_iou_margin,
        frame_iou_floor=task.frame_iou_floor,
        anchor_point_strategy=task.anchor_point_strategy,
        max_frame_hausdorff_px=task.max_frame_hausdorff_px,
        max_edge_span_frames=task.max_edge_span_frames,
        point_count=task.point_count,
        max_anchor_scale=task.max_anchor_scale,
        anchor_state_count=task.anchor_state_count,
        anchor_expansion=task.anchor_expansion,
        edge_threads=task.edge_threads,
        edge_processes=task.edge_processes,
        dominance_epsilon=task.dominance_epsilon,
        quality_mode=task.quality_mode,
        stored_vertex_contract=task.stored_vertex_contract,
        pair_vote_states=task.pair_vote_states,
        edge_batch_size=task.edge_batch_size,
        candidate_mode=task.candidate_mode,
        temporal_window_radii=task.temporal_window_radii,
        temporal_recall_quantile=task.temporal_recall_quantile,
        extra_anchor_states=task.extra_anchor_states,
    )


def _prune_global_states(
    states: dict[int, _GlobalState], *, improvement_epsilon: float
) -> dict[int, _GlobalState]:
    output: dict[int, _GlobalState] = {}
    best_iou = -math.inf
    for key_count in sorted(states):
        state = states[key_count]
        if state.iou_sum > best_iou + improvement_epsilon:
            output[key_count] = state
            best_iou = state.iou_sum
    return output


def _select_frontier_index(
    frontier: list[GlobalParetoPoint],
    *,
    selection: str,
    preference: float,
    key_budget: int | None,
    target_key_frequency: float | None,
    target_mean_key_interval: float | None,
    minimum_mean_iou: float | None = None,
) -> int:
    if not frontier:
        raise ValueError("empty Pareto frontier")
    if selection == "min_keys":
        return 0
    if selection == "max_iou":
        return len(frontier) - 1
    if selection == "key_budget":
        if key_budget is None:
            raise ValueError("key_budget selection requires key_budget")
        eligible = [
            index
            for index, point in enumerate(frontier)
            if point.keyframe_count <= int(key_budget)
        ]
        return eligible[-1] if eligible else 0
    if selection == "target_frequency":
        if target_key_frequency is None or target_key_frequency <= 0.0:
            raise ValueError(
                "target_frequency selection requires a positive target_key_frequency"
            )
        return min(
            range(len(frontier)),
            key=lambda index: (
                abs(frontier[index].key_frequency - target_key_frequency),
                -frontier[index].mean_iou,
            ),
        )
    if selection == "target_interval":
        if target_mean_key_interval is None or target_mean_key_interval <= 0.0:
            raise ValueError(
                "target_interval selection requires a positive target_mean_key_interval"
            )
        return min(
            range(len(frontier)),
            key=lambda index: (
                abs(frontier[index].mean_key_interval - target_mean_key_interval),
                -frontier[index].mean_iou,
            ),
        )
    if selection == "target_interval_quality_floor":
        if target_mean_key_interval is None or target_mean_key_interval <= 0.0:
            raise ValueError(
                "target_interval_quality_floor selection requires a positive "
                "target_mean_key_interval"
            )
        if minimum_mean_iou is None:
            raise ValueError(
                "target_interval_quality_floor selection requires " "minimum_mean_iou"
            )
        eligible = [
            index
            for index, point in enumerate(frontier)
            if point.mean_iou + 1e-12 >= float(minimum_mean_iou)
        ]
        if not eligible:
            # The caller performs the explicit non-regression assertion.  The
            # best-quality point gives that assertion the strongest possible
            # diagnostic instead of silently preferring the key target.
            return max(
                range(len(frontier)),
                key=lambda index: frontier[index].mean_iou,
            )
        return min(
            eligible,
            key=lambda index: (
                abs(frontier[index].mean_key_interval - target_mean_key_interval),
                -frontier[index].mean_iou,
                frontier[index].keyframe_count,
            ),
        )

    key_values = np.asarray(
        [point.keyframe_count for point in frontier], dtype=np.float64
    )
    iou_values = np.asarray([point.mean_iou for point in frontier], dtype=np.float64)
    key_span = max(float(key_values[-1] - key_values[0]), 1.0)
    iou_span = max(float(iou_values[-1] - iou_values[0]), 1e-12)
    key_cost = (key_values - key_values[0]) / key_span
    iou_loss = (iou_values[-1] - iou_values) / iou_span
    if selection == "preference":
        quality_weight = float(np.clip(preference, 0.0, 1.0))
        distances = (1.0 - quality_weight) * np.square(
            key_cost
        ) + quality_weight * np.square(iou_loss)
    elif selection == "knee":
        distances = np.square(key_cost) + np.square(iou_loss)
    else:
        raise ValueError(f"unsupported Pareto selection: {selection}")
    return int(np.argmin(distances))


def optimize_pareto_frontier(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    quality_masks: dict[tuple[int, str], RawMask] | None = None,
    border_constraints: dict[tuple[int, str], BorderFrameConstraint] | None = None,
    visible_bounds: tuple[float, float, float, float] | None = None,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    anchor_iou_floor: float = 0.0,
    anchor_relative_iou_margin: float | None = None,
    frame_iou_floor: float = 0.0,
    anchor_point_strategy: str = "uniform",
    max_frame_hausdorff_px: float | None = None,
    max_edge_span_frames: int = 60,
    point_count: int = 23,
    max_anchor_scale: float = 1.25,
    anchor_state_count: int = 1,
    anchor_expansion: float = 0.04,
    selection: str = "knee",
    preference: float = 0.50,
    key_budget: int | None = None,
    target_key_frequency: float | None = None,
    target_mean_key_interval: float | None = None,
    minimum_mean_iou: float | None = None,
    workers: int = 1,
    edge_threads: int = 1,
    edge_processes: int = 1,
    dominance_epsilon: float = 1e-10,
    quality_mode: str = "mean_iou",
    stored_vertex_contract: bool = False,
    pair_vote_states: bool = False,
    edge_batch_size: int = 32,
    solver_mode: str = "full",
    candidate_mode: str = "legacy",
    temporal_window_radii: tuple[int, int, int] = (2, 5, 10),
    temporal_recall_quantile: float = 0.90,
    extra_anchor_states: dict[tuple[int, str], tuple[Keyframe, ...]] | None = None,
) -> ParetoOptimizationResult:
    """Optimize and combine exact segment Pareto fronts under a recall floor."""

    if not (0.0 < recall_floor <= 1.0):
        raise ValueError("recall_floor must be in (0, 1]")
    if not (0.0 <= anchor_iou_floor <= 1.0):
        raise ValueError("anchor_iou_floor must be in [0, 1]")
    if not (0.0 <= frame_iou_floor <= 1.0):
        raise ValueError("frame_iou_floor must be in [0, 1]")
    if anchor_point_strategy not in {"uniform", "simplify_budget"}:
        raise ValueError(f"unsupported anchor_point_strategy: {anchor_point_strategy}")
    if max_frame_hausdorff_px is not None and max_frame_hausdorff_px <= 0.0:
        raise ValueError("max_frame_hausdorff_px must be positive")
    if quality_mode not in {
        "mean_iou",
        "log_iou",
        "tail_harmonic",
        "tail_boundary",
    }:
        raise ValueError(f"unsupported quality mode: {quality_mode}")
    if solver_mode not in {"full", "target_only"}:
        raise ValueError(f"unsupported solver mode: {solver_mode}")
    if candidate_mode not in {
        "legacy",
        "legacy_rdp_fallback",
        "temporal7",
        *_COMBINED_UNION_NAMES,
        *_TEMPORAL_UNION_NAMES,
        *_GEOMETRIC_UNION_NAMES,
    }:
        raise ValueError(f"unsupported candidate mode: {candidate_mode}")
    if solver_mode == "target_only" and (
        target_mean_key_interval is None or target_mean_key_interval <= 0.0
    ):
        raise ValueError("target_only requires a positive target_mean_key_interval")
    started = time.perf_counter()
    identities: list[tuple[str, int]] = []
    source_segments: list[Segment] = []
    local_frontiers: list[list[LocalParetoPoint]] = []
    edge_evaluations = 0
    feasible_edges = 0
    anchor_state_total = 0
    tasks: list[_SegmentTask] = []
    task_identities: list[tuple[str, int]] = []
    task_segments: list[Segment] = []
    for track_id, values in segments.items():
        for segment in values:
            segment_raw = {
                identity: raw
                for identity, raw in raw_masks.items()
                if identity[1] == track_id
                and segment.first_frame <= identity[0] <= segment.last_frame
                and start_frame <= identity[0] <= end_frame
            }
            if not segment_raw:
                continue
            tasks.append(
                _SegmentTask(
                    segment=segment,
                    raw_masks=segment_raw,
                    quality_masks=(
                        None
                        if quality_masks is None
                        else {
                            identity: raw
                            for identity, raw in quality_masks.items()
                            if identity[1] == track_id
                            and segment.first_frame <= identity[0] <= segment.last_frame
                            and start_frame <= identity[0] <= end_frame
                        }
                    ),
                    border_constraints=(
                        None
                        if border_constraints is None
                        else {
                            identity: constraint
                            for identity, constraint in border_constraints.items()
                            if identity[1] == track_id
                            and segment.first_frame <= identity[0] <= segment.last_frame
                            and start_frame <= identity[0] <= end_frame
                        }
                    ),
                    visible_bounds=visible_bounds,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    recall_floor=recall_floor,
                    anchor_iou_floor=anchor_iou_floor,
                    anchor_relative_iou_margin=anchor_relative_iou_margin,
                    frame_iou_floor=frame_iou_floor,
                    anchor_point_strategy=anchor_point_strategy,
                    max_frame_hausdorff_px=max_frame_hausdorff_px,
                    max_edge_span_frames=max_edge_span_frames,
                    point_count=point_count,
                    max_anchor_scale=max_anchor_scale,
                    anchor_state_count=anchor_state_count,
                    anchor_expansion=anchor_expansion,
                    edge_threads=edge_threads,
                    edge_processes=edge_processes,
                    dominance_epsilon=dominance_epsilon,
                    quality_mode=quality_mode,
                    stored_vertex_contract=stored_vertex_contract,
                    pair_vote_states=pair_vote_states,
                    edge_batch_size=edge_batch_size,
                    candidate_mode=candidate_mode,
                    temporal_window_radii=tuple(
                        int(value) for value in temporal_window_radii
                    ),
                    temporal_recall_quantile=float(temporal_recall_quantile),
                    extra_anchor_states=(
                        None
                        if extra_anchor_states is None
                        else {
                            identity: tuple(states)
                            for identity, states in extra_anchor_states.items()
                            if identity[1] == track_id
                            and segment.first_frame <= identity[0] <= segment.last_frame
                            and start_frame <= identity[0] <= end_frame
                        }
                    ),
                )
            )
            task_identities.append((track_id, segment.segment_id))
            task_segments.append(segment)
    effective_workers = min(max(1, int(workers)), max(len(tasks), 1))
    if effective_workers > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Submit the longest segments first.  Input order can otherwise
            # leave one large track running after every other worker becomes
            # idle.  Results are restored to the deterministic source order
            # before the global Pareto combination.
            scheduled = sorted(
                enumerate(tasks),
                key=lambda item: len(item[1].raw_masks),
                reverse=True,
            )
            futures = {
                index: executor.submit(_optimize_segment_task, task)
                for index, task in scheduled
            }
            task_results = [futures[index].result() for index in range(len(tasks))]
    else:
        task_results = [_optimize_segment_task(task) for task in tasks]

    for identity, segment, result in zip(task_identities, task_segments, task_results):
        local, evaluated, feasible, candidate_states = result
        if not local:
            continue
        identities.append(identity)
        source_segments.append(segment)
        local_frontiers.append(local)
        edge_evaluations += evaluated
        feasible_edges += feasible
        anchor_state_total += candidate_states
    if not local_frontiers:
        raise RuntimeError("no segment has raw masks in the requested range")

    raw_frame_count = sum(front[0].frame_count for front in local_frontiers)
    # Consecutive intervals are counted independently inside each segment.
    total_span = 0
    for segment, local in zip(source_segments, local_frontiers):
        frames = [keyframe.frame for keyframe in local[-1].keyframes]
        if frames:
            total_span += frames[-1] - frames[0]
    target_key_count = None
    if solver_mode == "target_only":
        target_key_count = int(
            round(total_span / float(target_mean_key_interval))
            + len(local_frontiers)
        )
        minimum_total = sum(point.keyframe_count for point in (front[0] for front in local_frontiers))
        maximum_total = sum(point.keyframe_count for point in (front[-1] for front in local_frontiers))
        if not minimum_total <= target_key_count <= maximum_total:
            raise RuntimeError(
                "target-only key count is outside the feasible range: "
                f"target={target_key_count}, feasible={minimum_total}..{maximum_total}"
            )

    stages: list[dict[int, _GlobalState]] = []
    current: dict[int, _GlobalState] = {
        0: _GlobalState(
            iou_sum=0.0,
            min_recall=1.0,
            previous_key_count=-1,
            local_point_index=-1,
        )
    }
    local_minimums = [front[0].keyframe_count for front in local_frontiers]
    local_maximums = [front[-1].keyframe_count for front in local_frontiers]
    for stage_position, local in enumerate(local_frontiers):
        combined: dict[int, _GlobalState] = {}
        remaining_minimum = sum(local_minimums[stage_position + 1 :])
        remaining_maximum = sum(local_maximums[stage_position + 1 :])
        for previous_key_count, previous in current.items():
            for local_index, point in enumerate(local):
                key_count = previous_key_count + point.keyframe_count
                if target_key_count is not None and (
                    key_count + remaining_minimum > target_key_count
                    or key_count + remaining_maximum < target_key_count
                ):
                    continue
                candidate = _GlobalState(
                    iou_sum=previous.iou_sum + point.iou_sum,
                    min_recall=min(previous.min_recall, point.min_recall),
                    previous_key_count=previous_key_count,
                    local_point_index=local_index,
                )
                existing = combined.get(key_count)
                if (
                    existing is None
                    or candidate.iou_sum > existing.iou_sum + 1e-12
                    or (
                        existing is not None
                        and abs(candidate.iou_sum - existing.iou_sum) <= 1e-12
                        and candidate.min_recall > existing.min_recall
                    )
                ):
                    combined[key_count] = candidate
        current = _prune_global_states(combined, improvement_epsilon=dominance_epsilon)
        stages.append(current)

    frontier: list[GlobalParetoPoint] = []
    global_choices: list[tuple[int, ...]] = []
    final_items = sorted(current.items())
    if target_key_count is not None:
        target_state = current.get(target_key_count)
        if target_state is None:
            raise RuntimeError(
                f"target-only could not construct exact key count {target_key_count}"
            )
        final_items = [(target_key_count, target_state)]
    for key_count, state in final_items:
        choices = [-1] * len(local_frontiers)
        cursor_key_count = key_count
        for stage_index in range(len(stages) - 1, -1, -1):
            stage_state = stages[stage_index][cursor_key_count]
            choices[stage_index] = stage_state.local_point_index
            cursor_key_count = stage_state.previous_key_count
        mean_interval = total_span / max(key_count - len(local_frontiers), 1)
        global_choices.append(tuple(choices))
        frontier.append(
            GlobalParetoPoint(
                keyframe_count=key_count,
                key_frequency=key_count / max(raw_frame_count, 1),
                mean_key_interval=mean_interval,
                mean_iou=state.iou_sum / max(raw_frame_count, 1),
                min_recall=state.min_recall,
                local_point_indices=tuple(choices),
            )
        )

    selected_index = (
        0
        if target_key_count is not None
        else _select_frontier_index(
            frontier,
            selection=selection,
            preference=preference,
            key_budget=key_budget,
            target_key_frequency=target_key_frequency,
            target_mean_key_interval=target_mean_key_interval,
            minimum_mean_iou=minimum_mean_iou,
        )
    )
    selected_choices = global_choices[selected_index]
    selected_by_identity = {}
    for index, (identity, choice) in enumerate(zip(identities, selected_choices)):
        path = local_frontiers[index][choice].keyframes
        selected_by_identity[identity] = (
            canonicalize_selected_path(path) if stored_vertex_contract else path
        )
    output = {
        track_id: [
            replace(
                segment,
                interpolation_method=(
                    "linear_polygon_index_v1"
                    if stored_vertex_contract
                    else segment.interpolation_method
                ),
                keyframes=selected_by_identity.get(
                    (track_id, segment.segment_id), segment.keyframes
                ),
            )
            for segment in values
        ]
        for track_id, values in segments.items()
    }
    return ParetoOptimizationResult(
        recall_floor=recall_floor,
        frontier=tuple(frontier),
        selected_index=selected_index,
        segments=output,
        raw_frame_count=raw_frame_count,
        anchor_state_total=anchor_state_total,
        edge_evaluations=edge_evaluations,
        feasible_edges=feasible_edges,
        worker_count=effective_workers,
        elapsed_seconds=time.perf_counter() - started,
    )
