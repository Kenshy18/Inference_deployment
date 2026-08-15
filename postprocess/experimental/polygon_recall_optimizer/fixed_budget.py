"""Experimental fixed-budget polygon optimizer and exact geometry evaluator.

The module deliberately does not participate in the production stage registry.  It
reads the public V3 SQLite contract, reconstructs masks with the same interpolation
code as the overlay, and returns in-memory keyframe alternatives for comparison.
Video pixels are neither opened nor required.
"""

from __future__ import annotations

import bisect
import heapq
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "overlay" / "src"))

from overlay_renderer.keyframe_cache import (  # noqa: E402
    Component,
    Keyframe,
    _component_polygons,
    _components_at,
    _load_keyframes,
    _numpy_align,
    _numpy_resample,
)


@dataclass(frozen=True)
class RawMask:
    frame: int
    track_id: str
    geometry: Polygon | MultiPolygon
    primary_points: np.ndarray
    score: float
    component_points: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class Segment:
    segment_id: int
    track_id: str
    first_frame: int
    last_frame: int
    interpolation_method: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True)
class FrameEvaluation:
    frame: int
    track_id: str
    segment_id: int
    is_keyframe: bool
    raw_area: float
    predicted_area: float
    intersection_area: float
    recall: float
    precision: float
    iou: float
    area_ratio: float
    excess_area_ratio: float
    centroid_error_px: float
    raw_geometry: Polygon | MultiPolygon
    predicted_geometry: Polygon | MultiPolygon


def _polygonal(value):
    if value.is_empty:
        return GeometryCollection()
    valid = make_valid(value)
    if isinstance(valid, (Polygon, MultiPolygon)):
        return valid
    polygons = [
        part
        for part in getattr(valid, "geoms", ())
        if isinstance(part, (Polygon, MultiPolygon))
    ]
    return unary_union(polygons) if polygons else GeometryCollection()


def geometry_from_arrays(arrays: Iterable[np.ndarray]):
    arrays = list(arrays)
    if len(arrays) == 1:
        points = np.asarray(arrays[0], dtype=np.float64).reshape(-1, 2)
        if len(points) >= 3:
            candidate = Polygon(points)
            # The former path called make_valid and unary_union even for one
            # already-valid polygon.  GEOS returns byte-identical geometry in
            # that case, so bypass the redundant repairs.  Invalid/degenerate
            # interpolation still uses the original robust path below.
            if candidate.is_valid and not candidate.is_empty and candidate.area > 0.0:
                return candidate
    geometries = []
    for raw_points in arrays:
        points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3:
            continue
        candidate = _polygonal(Polygon(points))
        if not candidate.is_empty and candidate.area > 0.0:
            geometries.append(candidate)
    return _polygonal(unary_union(geometries)) if geometries else GeometryCollection()


def load_raw_masks(
    path: Path,
    *,
    label: str,
    start_frame: int,
    end_frame: int,
) -> dict[tuple[int, str], RawMask]:
    grouped: dict[tuple[int, str], dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scores: dict[tuple[int, str], float] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT a.frame, a.final_track_id, a.selected_score,
                   sp.polygon_index, pt.point_index, pt.x, pt.y
            FROM tracking_assignments a
            JOIN segmentation_polygons sp ON sp.detection_id=a.source_detection_id
            JOIN segmentation_points pt ON pt.polygon_id=sp.id
            WHERE a.removed_by_short_track=0
              AND a.final_label=?
              AND a.frame BETWEEN ? AND ?
            ORDER BY a.frame, a.final_track_id, sp.polygon_index, pt.point_index
            """,
            (label, int(start_frame), int(end_frame)),
        )
        for frame, track_id, score, polygon_index, _point_index, x, y in rows:
            key = (int(frame), str(track_id))
            grouped[key][int(polygon_index)].append((float(x), float(y)))
            scores[key] = float(score)

    output: dict[tuple[int, str], RawMask] = {}
    for (frame, track_id), polygons in grouped.items():
        arrays = [
            np.asarray(points, dtype=np.float64)
            for _index, points in sorted(polygons.items())
        ]
        primary = max(arrays, key=lambda item: float(abs(Polygon(item).area)))
        output[(frame, track_id)] = RawMask(
            frame=frame,
            track_id=track_id,
            geometry=geometry_from_arrays(arrays),
            primary_points=primary,
            score=scores[(frame, track_id)],
            component_points=tuple(arrays),
        )
    return output


def load_segments(
    path: Path,
    *,
    label: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, list[Segment]]:
    output: dict[str, list[Segment]] = defaultdict(list)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.track_id, s.start_frame, s.end_frame,
                   s.interpolation_method
            FROM mask_track_segments s
            JOIN tracks t ON t.track_id=s.track_id
            WHERE t.domain='genital' AND t.label=?
              AND s.end_frame>=? AND s.start_frame<=?
            ORDER BY s.track_id, s.start_frame
            """,
            (label, int(start_frame), int(end_frame)),
        )
        for segment_id, track_id, first, last, method in rows:
            output[str(track_id)].append(
                Segment(
                    segment_id=int(segment_id),
                    track_id=str(track_id),
                    first_frame=int(first),
                    last_frame=int(last),
                    interpolation_method=str(method),
                    keyframes=tuple(_load_keyframes(connection, int(segment_id))),
                )
            )
    return dict(output)


def _segment_at(
    segments: dict[str, list[Segment]], frame: int, track_id: str
) -> Segment | None:
    for segment in segments.get(track_id, ()):
        if segment.first_frame <= frame <= segment.last_frame:
            return segment
    return None


def geometry_at(segment: Segment, frame: int):
    keyframes = list(segment.keyframes)
    if not keyframes or frame < keyframes[0].frame or frame > keyframes[-1].frame:
        return GeometryCollection()
    components = _components_at(keyframes, int(frame), segment.interpolation_method)
    arrays = [
        np.asarray(points, dtype=np.float64)
        for points in _component_polygons(components)
    ]
    return geometry_from_arrays(arrays)


def evaluate_segments(
    raw_masks: dict[tuple[int, str], RawMask],
    segments: dict[str, list[Segment]],
    *,
    visible_rectangle=None,
    border_constraints=None,
) -> list[FrameEvaluation]:
    output: list[FrameEvaluation] = []
    key_sets = {
        segment.segment_id: {keyframe.frame for keyframe in segment.keyframes}
        for values in segments.values()
        for segment in values
    }
    for (frame, track_id), raw in sorted(raw_masks.items()):
        segment = _segment_at(segments, frame, track_id)
        if segment is None:
            continue
        predicted = geometry_at(segment, frame)
        if predicted.is_empty:
            continue
        raw_area = float(raw.geometry.area)
        full_intersection = float(raw.geometry.intersection(predicted).area)
        metric_predicted = predicted
        if visible_rectangle is not None:
            bounds = predicted.bounds
            visible_bounds = visible_rectangle.bounds
            if (
                bounds[0] < visible_bounds[0]
                or bounds[1] < visible_bounds[1]
                or bounds[2] > visible_bounds[2]
                or bounds[3] > visible_bounds[3]
            ):
                metric_predicted = predicted.intersection(visible_rectangle)
        metric_raw = raw.geometry
        border_constraint = (
            None
            if border_constraints is None
            else border_constraints.get((frame, track_id))
        )
        if border_constraint is not None:
            metric_predicted = metric_predicted.intersection(
                border_constraint.quality_domain
            )
            metric_raw = raw.geometry.intersection(border_constraint.quality_domain)
        predicted_area = float(metric_predicted.area)
        metric_raw_area = float(metric_raw.area)
        metric_intersection = float(metric_raw.intersection(metric_predicted).area)
        union = metric_raw_area + predicted_area - metric_intersection
        raw_center = raw.geometry.centroid
        predicted_center = (
            predicted.centroid
            if metric_predicted.is_empty
            else metric_predicted.centroid
        )
        output.append(
            FrameEvaluation(
                frame=frame,
                track_id=track_id,
                segment_id=segment.segment_id,
                is_keyframe=frame in key_sets[segment.segment_id],
                raw_area=raw_area,
                predicted_area=predicted_area,
                intersection_area=full_intersection,
                recall=full_intersection / raw_area if raw_area else 1.0,
                precision=(
                    metric_intersection / predicted_area if predicted_area else 1.0
                ),
                iou=metric_intersection / union if union else 1.0,
                area_ratio=predicted_area / raw_area if raw_area else 1.0,
                excess_area_ratio=(predicted_area - metric_intersection) / raw_area
                if raw_area
                else 0.0,
                centroid_error_px=math.hypot(
                    predicted_center.x - raw_center.x,
                    predicted_center.y - raw_center.y,
                ),
                raw_geometry=raw.geometry,
                predicted_geometry=predicted,
            )
        )
    return output


def _geometry_iou(left, right) -> float:
    intersection = float(left.intersection(right).area)
    union = float(left.area + right.area - intersection)
    return intersection / union if union else 1.0


def summarize(
    evaluations: list[FrameEvaluation],
    segments: dict[str, list[Segment]],
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, float | int]:
    recalls = np.asarray([item.recall for item in evaluations], dtype=np.float64)
    precisions = np.asarray([item.precision for item in evaluations], dtype=np.float64)
    ious = np.asarray([item.iou for item in evaluations], dtype=np.float64)
    excess = np.asarray(
        [item.excess_area_ratio for item in evaluations], dtype=np.float64
    )
    centroid_errors = np.asarray(
        [item.centroid_error_px for item in evaluations], dtype=np.float64
    )
    key_recalls = np.asarray(
        [item.recall for item in evaluations if item.is_keyframe], dtype=np.float64
    )
    key_count = sum(
        start_frame <= keyframe.frame <= end_frame
        for values in segments.values()
        for segment in values
        for keyframe in segment.keyframes
    )

    by_track = {(item.frame, item.track_id): item for item in evaluations}
    raw_adjacent: list[float] = []
    predicted_adjacent: list[float] = []
    area_log_delta: list[float] = []
    relative_area_log_delta: list[float] = []
    centroid_by_track: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    centroid_residual_by_track: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(
        list
    )
    for item in evaluations:
        previous = by_track.get((item.frame - 1, item.track_id))
        if previous is not None:
            raw_adjacent.append(_geometry_iou(previous.raw_geometry, item.raw_geometry))
            predicted_adjacent.append(
                _geometry_iou(previous.predicted_geometry, item.predicted_geometry)
            )
            area_log_delta.append(
                abs(
                    math.log(
                        max(item.predicted_area, 1e-6)
                        / max(previous.predicted_area, 1e-6)
                    )
                )
            )
            relative_area_log_delta.append(
                abs(
                    math.log(
                        max(item.area_ratio, 1e-6) / max(previous.area_ratio, 1e-6)
                    )
                )
            )
        center = item.predicted_geometry.centroid
        raw_center = item.raw_geometry.centroid
        centroid_by_track[item.track_id].append(
            (item.frame, np.asarray([center.x, center.y], dtype=np.float64))
        )
        centroid_residual_by_track[item.track_id].append(
            (
                item.frame,
                np.asarray(
                    [center.x - raw_center.x, center.y - raw_center.y],
                    dtype=np.float64,
                ),
            )
        )

    accelerations: list[float] = []
    for values in centroid_by_track.values():
        values.sort(key=lambda item: item[0])
        for index in range(2, len(values)):
            if values[index][0] != values[index - 1][0] + 1:
                continue
            if values[index - 1][0] != values[index - 2][0] + 1:
                continue
            current_velocity = values[index][1] - values[index - 1][1]
            previous_velocity = values[index - 1][1] - values[index - 2][1]
            accelerations.append(
                float(np.linalg.norm(current_velocity - previous_velocity))
            )

    residual_accelerations: list[float] = []
    for values in centroid_residual_by_track.values():
        values.sort(key=lambda item: item[0])
        for index in range(2, len(values)):
            if values[index][0] != values[index - 1][0] + 1:
                continue
            if values[index - 1][0] != values[index - 2][0] + 1:
                continue
            current_velocity = values[index][1] - values[index - 1][1]
            previous_velocity = values[index - 1][1] - values[index - 2][1]
            residual_accelerations.append(
                float(np.linalg.norm(current_velocity - previous_velocity))
            )

    def quantile(values: np.ndarray | list[float], probability: float) -> float:
        return float(np.quantile(values, probability)) if len(values) else 0.0

    def mean(values: np.ndarray | list[float]) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    return {
        "frame_count": len(evaluations),
        "keyframe_count": int(key_count),
        "recall_mean": mean(recalls),
        "recall_min": float(np.min(recalls)),
        "recall_q01": quantile(recalls, 0.01),
        "recall_q05": quantile(recalls, 0.05),
        "recall_below_090": int(np.sum(recalls < 0.90)),
        "recall_below_095": int(np.sum(recalls < 0.95)),
        "recall_below_097": int(np.sum(recalls < 0.97)),
        "keyframe_recall_min": float(np.min(key_recalls)) if len(key_recalls) else 1.0,
        "keyframe_recall_below_090": int(np.sum(key_recalls < 0.90)),
        "precision_mean": mean(precisions),
        "precision_q05": quantile(precisions, 0.05),
        "iou_mean": mean(ious),
        "iou_q05": quantile(ious, 0.05),
        "excess_area_ratio_mean": mean(excess),
        "excess_area_ratio_q95": quantile(excess, 0.95),
        "centroid_error_mean_px": mean(centroid_errors),
        "centroid_error_q95_px": quantile(centroid_errors, 0.95),
        "raw_adjacent_iou_mean": mean(raw_adjacent),
        "predicted_adjacent_iou_mean": mean(predicted_adjacent),
        "predicted_area_log_delta_mean": mean(area_log_delta),
        "predicted_area_log_delta_q95": quantile(area_log_delta, 0.95),
        "relative_area_log_delta_mean": mean(relative_area_log_delta),
        "relative_area_log_delta_q95": quantile(relative_area_log_delta, 0.95),
        "predicted_centroid_acceleration_mean_px": mean(accelerations),
        "predicted_centroid_acceleration_q95_px": quantile(accelerations, 0.95),
        "centroid_residual_acceleration_mean_px": mean(residual_accelerations),
        "centroid_residual_acceleration_q95_px": quantile(residual_accelerations, 0.95),
    }


def _polygon_components(keyframe: Keyframe) -> list[tuple[int, Component]]:
    return [
        (slot, component)
        for slot, component in keyframe.components
        if component.kind == "polygon"
    ]


def _primary_component(keyframe: Keyframe) -> Component | None:
    polygons = _polygon_components(keyframe)
    if not polygons:
        return None
    return max(
        (component for _slot, component in polygons),
        key=lambda component: abs(Polygon(component.values).area),
    )


def _simplify_to_vertex_budget(points: np.ndarray, budget: int) -> np.ndarray:
    """Retain high-curvature vertices instead of uniformly sampling perimeter."""

    polygon = Polygon(np.asarray(points, dtype=np.float64))
    requested = max(8, int(budget))
    if len(polygon.exterior.coords) - 1 <= requested:
        return np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    min_x, min_y, max_x, max_y = polygon.bounds
    lower = 0.0
    upper = math.hypot(max_x - min_x, max_y - min_y)
    selected = polygon
    for _ in range(24):
        tolerance = 0.5 * (lower + upper)
        candidate = polygon.simplify(tolerance, preserve_topology=True)
        if not isinstance(candidate, Polygon) or candidate.is_empty:
            upper = tolerance
            continue
        if len(candidate.exterior.coords) - 1 <= requested:
            selected = candidate
            upper = tolerance
        else:
            lower = tolerance
    return np.asarray(selected.exterior.coords[:-1], dtype=np.float64)


def _raw_keyframe(
    raw: RawMask,
    *,
    point_count: int,
    point_strategy: str = "uniform",
) -> Keyframe:
    if point_strategy == "uniform":
        points = _numpy_resample(
            np.asarray(raw.primary_points, dtype=np.float64), max(8, int(point_count))
        )
    elif point_strategy == "simplify_budget":
        points = _simplify_to_vertex_budget(raw.primary_points, point_count)
    else:
        raise ValueError(f"unsupported raw-key point strategy: {point_strategy}")
    return Keyframe(
        frame=raw.frame,
        components=((0, Component("polygon", points.tolist())),),
    )


def _resample_keyframe(keyframe: Keyframe, *, point_count: int) -> Keyframe:
    component = _primary_component(keyframe)
    if component is None:
        return keyframe
    points = _numpy_resample(
        np.asarray(component.values, dtype=np.float64), max(8, int(point_count))
    )
    return Keyframe(
        frame=keyframe.frame,
        components=((0, Component("polygon", points.tolist())),),
    )


def _replace_segment_keys(segment: Segment, keyframes: Iterable[Keyframe]) -> Segment:
    return replace(
        segment,
        keyframes=tuple(sorted(keyframes, key=lambda keyframe: keyframe.frame)),
    )


def transform_segments(
    segments: dict[str, list[Segment]],
    transform,
) -> dict[str, list[Segment]]:
    return {
        track_id: [transform(segment) for segment in values]
        for track_id, values in segments.items()
    }


def raw_anchor_at_existing_positions(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    def transform(segment: Segment) -> Segment:
        keys = []
        for keyframe in segment.keyframes:
            raw = raw_masks.get((keyframe.frame, segment.track_id))
            keys.append(
                _raw_keyframe(raw, point_count=point_count)
                if raw is not None
                else _resample_keyframe(keyframe, point_count=point_count)
            )
        return _replace_segment_keys(segment, keys)

    return transform_segments(segments, transform)


def blend_keys_toward_raw(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    raw_weight: float,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    weight = float(np.clip(raw_weight, 0.0, 1.0))

    def transform(segment: Segment) -> Segment:
        output = []
        for original in segment.keyframes:
            current = _resample_keyframe(original, point_count=point_count)
            raw = raw_masks.get((original.frame, segment.track_id))
            component = _primary_component(current)
            if raw is None or component is None:
                output.append(current)
                continue
            current_points = np.asarray(component.values, dtype=np.float64)
            raw_points = _numpy_resample(raw.primary_points, len(current_points))
            raw_points = _numpy_align(current_points, raw_points)
            blended = (1.0 - weight) * current_points + weight * raw_points
            output.append(
                Keyframe(
                    original.frame,
                    ((0, Component("polygon", blended.tolist())),),
                )
            )
        return _replace_segment_keys(segment, output)

    return transform_segments(segments, transform)


def _safe_anchor_keyframe(
    segment: Segment,
    raw: RawMask,
    *,
    anchor_recall: float,
    point_count: int,
) -> Keyframe:
    """Project the smooth production shape minimally toward one observation."""

    components = _components_at(
        list(segment.keyframes), raw.frame, segment.interpolation_method
    )
    polygons = [
        np.asarray(component.values, dtype=np.float64)
        for component in components
        if component.kind == "polygon"
    ]
    if not polygons:
        return _raw_keyframe(raw, point_count=point_count)
    base = max(polygons, key=lambda points: abs(Polygon(points).area))
    base = _numpy_resample(base, max(8, int(point_count)))
    observed = _numpy_resample(raw.primary_points, len(base))
    observed = _numpy_align(base, observed)

    def recall_at(weight: float) -> float:
        points = (1.0 - weight) * base + weight * observed
        geometry = geometry_from_arrays([points])
        intersection = float(raw.geometry.intersection(geometry).area)
        return intersection / float(raw.geometry.area) if raw.geometry.area else 1.0

    if recall_at(0.0) >= anchor_recall:
        weight = 0.0
    elif recall_at(1.0) < anchor_recall:
        weight = 1.0
    else:
        # Recall is normally monotone along this short projection, but use a
        # grid to bracket the first feasible region before binary refinement.
        low = 0.0
        high = 1.0
        for candidate in np.linspace(0.05, 1.0, 20):
            if recall_at(float(candidate)) >= anchor_recall:
                high = float(candidate)
                low = max(0.0, high - 0.05)
                break
        for _step in range(10):
            middle = 0.5 * (low + high)
            if recall_at(middle) >= anchor_recall:
                high = middle
            else:
                low = middle
        weight = high
    points = (1.0 - weight) * base + weight * observed
    return Keyframe(raw.frame, ((0, Component("polygon", points.tolist())),))


def adaptive_split_recall_keys(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    anchor_margin: float = 0.05,
    repair_margin: float = 0.005,
    point_count: int = 23,
    prune: bool = True,
) -> dict[str, list[Segment]]:
    """Build keys from endpoints by splitting only recall-violating spans.

    Unlike the production interval target, this top-down construction has no
    desired key spacing.  The optional bottom-up pass removes any key whose
    neighbours can be joined without breaking the dense recall floor.
    """

    anchor_floor = min(0.999, float(recall_floor) + float(anchor_margin))
    internal_floor = min(0.999, float(recall_floor) + float(repair_margin))

    def transform(segment: Segment) -> Segment:
        raw_by_frame = {
            frame: raw
            for (frame, track_id), raw in raw_masks.items()
            if track_id == segment.track_id
            and segment.first_frame <= frame <= segment.last_frame
            and start_frame <= frame <= end_frame
        }
        if not raw_by_frame:
            return segment
        anchor_cache: dict[int, Keyframe] = {}

        def anchor(frame: int) -> Keyframe:
            cached = anchor_cache.get(frame)
            if cached is None:
                cached = _safe_anchor_keyframe(
                    segment,
                    raw_by_frame[frame],
                    anchor_recall=anchor_floor,
                    point_count=point_count,
                )
                anchor_cache[frame] = cached
            return cached

        inside_first = min(raw_by_frame)
        inside_last = max(raw_by_frame)
        selected: dict[int, Keyframe] = {
            keyframe.frame: _resample_keyframe(keyframe, point_count=point_count)
            for keyframe in segment.keyframes
            if keyframe.frame < start_frame or keyframe.frame > end_frame
        }
        selected[inside_first] = anchor(inside_first)
        selected[inside_last] = anchor(inside_last)

        heap: list[tuple[float, float, float, int, int, int]] = []

        def push(left_frame: int, right_frame: int) -> None:
            worst = _interval_worst_raw_frame(
                selected[left_frame], selected[right_frame], raw_by_frame
            )
            if worst is None:
                return
            recall, iou, negative_center_error, frame = worst
            heapq.heappush(
                heap,
                (
                    recall,
                    iou,
                    negative_center_error,
                    frame,
                    left_frame,
                    right_frame,
                ),
            )

        frames = sorted(selected)
        for left_frame, right_frame in zip(frames, frames[1:]):
            push(left_frame, right_frame)

        while heap:
            recall, _iou, _center, frame, left_frame, right_frame = heapq.heappop(heap)
            current = sorted(selected)
            left_index = bisect.bisect_left(current, left_frame)
            if (
                left_index >= len(current) - 1
                or current[left_index] != left_frame
                or current[left_index + 1] != right_frame
            ):
                continue
            if recall >= internal_floor:
                continue
            if frame in selected or frame not in raw_by_frame:
                continue
            selected[frame] = anchor(frame)
            push(left_frame, frame)
            push(frame, right_frame)

        if prune:
            while True:
                frames = sorted(selected)
                removable: list[tuple[float, float, int]] = []
                for index in range(1, len(frames) - 1):
                    frame = frames[index]
                    if not (start_frame <= frame <= end_frame):
                        continue
                    left_frame = frames[index - 1]
                    right_frame = frames[index + 1]
                    worst = _interval_worst_raw_frame(
                        selected[left_frame], selected[right_frame], raw_by_frame
                    )
                    if worst is None or worst[0] >= internal_floor:
                        minimum_recall = 1.0 if worst is None else worst[0]
                        minimum_iou = 1.0 if worst is None else worst[1]
                        removable.append((minimum_recall, minimum_iou, frame))
                if not removable:
                    break
                # Remove the most redundant key first, then reconsider its
                # newly joined neighbourhood.
                _recall, _iou, frame = max(removable)
                del selected[frame]

        return _replace_segment_keys(segment, selected.values())

    return transform_segments(segments, transform)


def projected_temporal_smooth(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    key_recall_floor: float,
    strength: float = 0.50,
    iterations: int = 2,
) -> dict[str, list[Segment]]:
    """Reduce key velocity changes while projecting back to a recall floor."""

    requested_strength = float(np.clip(strength, 0.0, 1.0))

    def transform(segment: Segment) -> Segment:
        keys = list(segment.keyframes)
        for _iteration in range(max(0, int(iterations))):
            source = list(keys)
            output = list(keys)
            for index in range(1, len(source) - 1):
                previous = _primary_component(source[index - 1])
                current = _primary_component(source[index])
                following = _primary_component(source[index + 1])
                raw = raw_masks.get((source[index].frame, segment.track_id))
                if (
                    previous is None
                    or current is None
                    or following is None
                    or raw is None
                ):
                    continue
                current_points = np.asarray(current.values, dtype=np.float64)
                previous_points = _numpy_resample(
                    np.asarray(previous.values, dtype=np.float64), len(current_points)
                )
                following_points = _numpy_resample(
                    np.asarray(following.values, dtype=np.float64), len(current_points)
                )
                previous_points = _numpy_align(current_points, previous_points)
                following_points = _numpy_align(current_points, following_points)
                alpha = (source[index].frame - source[index - 1].frame) / max(
                    source[index + 1].frame - source[index - 1].frame, 1
                )
                temporal_target = (
                    1.0 - alpha
                ) * previous_points + alpha * following_points

                def recall_at(weight: float) -> float:
                    points = (1.0 - weight) * current_points + weight * temporal_target
                    geometry = geometry_from_arrays([points])
                    intersection = float(raw.geometry.intersection(geometry).area)
                    return (
                        intersection / float(raw.geometry.area)
                        if raw.geometry.area
                        else 1.0
                    )

                if recall_at(0.0) < key_recall_floor:
                    continue
                if recall_at(requested_strength) >= key_recall_floor:
                    accepted = requested_strength
                else:
                    low = 0.0
                    high = requested_strength
                    for _step in range(10):
                        middle = 0.5 * (low + high)
                        if recall_at(middle) >= key_recall_floor:
                            low = middle
                        else:
                            high = middle
                    accepted = low
                points = (1.0 - accepted) * current_points + accepted * temporal_target
                output[index] = Keyframe(
                    source[index].frame,
                    ((0, Component("polygon", points.tolist())),),
                )
            keys = output
        return _replace_segment_keys(segment, keys)

    return transform_segments(segments, transform)


def refine_to_key_budget(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    target_key_count: int,
    anchor_recall_floor: float = 0.97,
    dense_recall_floor: float = 0.902,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    """Allocate a remaining key budget to the worst current interpolation spans."""

    output = {
        track_id: list(track_segments) for track_id, track_segments in segments.items()
    }
    current_count = sum(
        start_frame <= keyframe.frame <= end_frame
        for values in output.values()
        for segment in values
        for keyframe in segment.keyframes
    )
    remaining = max(0, int(target_key_count) - current_count)
    if remaining <= 0:
        return output

    evaluations = evaluate_segments(raw_masks, output)
    segment_lookup = {
        (track_id, segment.segment_id): segment
        for track_id, values in output.items()
        for segment in values
    }
    # Only one insertion is selected per current span in a pass. This prevents
    # spending the entire budget on a cluster of adjacent bad frames.
    by_span: dict[
        tuple[str, int, int, int], list[tuple[float, FrameEvaluation]]
    ] = defaultdict(list)
    for item in evaluations:
        segment = segment_lookup[(item.track_id, item.segment_id)]
        frames = [keyframe.frame for keyframe in segment.keyframes]
        position = bisect.bisect_left(frames, item.frame)
        if position < len(frames) and frames[position] == item.frame:
            continue
        if position <= 0 or position >= len(frames):
            continue
        left = frames[position - 1]
        right = frames[position]
        # IoU is primary after recall feasibility. Excess area and normalized
        # centroid error break ties toward visibly poor interpolations.
        score = (
            (1.0 - item.iou)
            + 0.25 * max(item.excess_area_ratio, 0.0)
            + 0.10 * min(item.centroid_error_px / 50.0, 1.0)
        )
        span = (item.track_id, item.segment_id, left, right)
        by_span[span].append((score, item))

    feasible: list[tuple[float, FrameEvaluation, Keyframe]] = []
    for (track_id, segment_id, left, right), candidates in by_span.items():
        segment = segment_lookup[(track_id, segment_id)]
        keys_by_frame = {keyframe.frame: keyframe for keyframe in segment.keyframes}
        segment_raw = {
            frame: raw
            for (frame, candidate_track), raw in raw_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= frame <= segment.last_frame
        }
        for score, item in sorted(candidates, key=lambda value: value[0], reverse=True):
            candidate = _safe_anchor_keyframe(
                segment,
                raw_masks[(item.frame, track_id)],
                anchor_recall=anchor_recall_floor,
                point_count=point_count,
            )
            left_worst = _interval_worst_raw_frame(
                keys_by_frame[left], candidate, segment_raw
            )
            right_worst = _interval_worst_raw_frame(
                candidate, keys_by_frame[right], segment_raw
            )
            left_recall = 1.0 if left_worst is None else left_worst[0]
            right_recall = 1.0 if right_worst is None else right_worst[0]
            if min(left_recall, right_recall) >= dense_recall_floor:
                feasible.append((score, item, candidate))
                break

    chosen = sorted(feasible, key=lambda value: value[0], reverse=True)[:remaining]
    additions: dict[tuple[str, int], list[Keyframe]] = defaultdict(list)
    for _score, item, candidate in chosen:
        additions[(item.track_id, item.segment_id)].append(candidate)

    return {
        track_id: [
            _replace_segment_keys(
                segment,
                list(segment.keyframes)
                + additions.get((track_id, segment.segment_id), []),
            )
            for segment in values
        ]
        for track_id, values in output.items()
    }


def lexicographic_recall_stability_optimize(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float = 0.90,
    anchor_margin: float = 0.08,
    smoothing_key_margin: float = 0.07,
    smoothing_strength: float = 0.10,
) -> dict[str, list[Segment]]:
    """Lexicographic optimizer: safety, compactness, then stability.

    1. Build an interval-free, recall-feasible key set from track endpoints.
    2. Smooth key motion while projecting every key to its recall constraint.
    3. Re-check every reconstructed frame and add only the remaining failures.

    Expansion is disabled in the final repair so that key insertion, rather
    than hidden dilation, pays for any recall lost during smoothing.
    """

    split = adaptive_split_recall_keys(
        segments,
        raw_masks,
        start_frame=start_frame,
        end_frame=end_frame,
        recall_floor=recall_floor,
        anchor_margin=anchor_margin,
        repair_margin=0.005,
    )
    smoothed = projected_temporal_smooth(
        split,
        raw_masks,
        key_recall_floor=min(0.999, float(recall_floor) + float(smoothing_key_margin)),
        strength=smoothing_strength,
        iterations=1,
    )
    return adaptive_add_recall_keys(
        smoothed,
        raw_masks,
        start_frame=start_frame,
        end_frame=end_frame,
        recall_floor=recall_floor,
        repair_margin=0.005,
        max_scale=1.0,
        max_rounds=12,
        safe_anchor_floor=min(0.999, float(recall_floor) + float(smoothing_key_margin)),
    )


def _interval_worst_raw_frame(
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
) -> tuple[float, float, float, int] | None:
    worst: tuple[float, float, float, int] | None = None
    left_component = _primary_component(left)
    right_component = _primary_component(right)
    if left_component is None or right_component is None:
        return None
    left_points = np.asarray(left_component.values, dtype=np.float64)
    right_points = _numpy_resample(
        np.asarray(right_component.values, dtype=np.float64), len(left_points)
    )
    right_points = _numpy_align(left_points, right_points)
    for frame in sorted(raw_by_frame):
        if frame <= left.frame or frame >= right.frame:
            continue
        alpha = (frame - left.frame) / max(right.frame - left.frame, 1)
        predicted = geometry_from_arrays(
            [(1.0 - alpha) * left_points + alpha * right_points]
        )
        raw = raw_by_frame[frame].geometry
        intersection = float(raw.intersection(predicted).area)
        recall = intersection / float(raw.area) if raw.area else 1.0
        union = float(raw.area + predicted.area - intersection)
        iou = intersection / union if union else 1.0
        raw_center = raw.centroid
        predicted_center = predicted.centroid
        centroid_error = math.hypot(
            predicted_center.x - raw_center.x,
            predicted_center.y - raw_center.y,
        )
        # Recall/IoU plateaus at zero for large excursions.  Prefer the
        # largest positional excursion within that plateau, which places a
        # key at a periodic-motion apex rather than at its first bad frame.
        candidate = (recall, iou, -centroid_error, frame)
        if worst is None or candidate < worst:
            worst = candidate
    return worst


def minimax_recall_positions(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    point_count: int = 23,
    max_gap: int | None = None,
) -> dict[str, list[Segment]]:
    """Choose the same number of keys by greedily splitting worst-recall spans."""

    def transform(segment: Segment) -> Segment:
        baseline_inside = [
            keyframe
            for keyframe in segment.keyframes
            if start_frame <= keyframe.frame <= end_frame
        ]
        target_inside = len(baseline_inside)
        if target_inside <= 0:
            return segment
        raw_by_frame = {
            frame: raw
            for (frame, track_id), raw in raw_masks.items()
            if track_id == segment.track_id
            and segment.first_frame <= frame <= segment.last_frame
            and start_frame <= frame <= end_frame
        }
        if not raw_by_frame:
            return segment

        selected: dict[int, Keyframe] = {
            keyframe.frame: _resample_keyframe(keyframe, point_count=point_count)
            for keyframe in segment.keyframes
            if keyframe.frame < start_frame or keyframe.frame > end_frame
        }
        mandatory_inside: dict[int, Keyframe] = {}
        for keyframe in baseline_inside:
            raw = raw_by_frame.get(keyframe.frame)
            if raw is None:
                mandatory_inside[keyframe.frame] = _resample_keyframe(
                    keyframe, point_count=point_count
                )
        if segment.first_frame >= start_frame and segment.first_frame <= end_frame:
            raw = raw_by_frame.get(segment.first_frame)
            if raw is not None:
                mandatory_inside[segment.first_frame] = _raw_keyframe(
                    raw, point_count=point_count
                )
        if segment.last_frame >= start_frame and segment.last_frame <= end_frame:
            raw = raw_by_frame.get(segment.last_frame)
            if raw is not None:
                mandatory_inside[segment.last_frame] = _raw_keyframe(
                    raw, point_count=point_count
                )
        selected.update(mandatory_inside)

        while (
            sum(start_frame <= frame <= end_frame for frame in selected) < target_inside
        ):
            frames = sorted(selected)
            overlong = [
                (right_frame - left_frame, left_frame, right_frame)
                for left_frame, right_frame in zip(frames, frames[1:])
                if max_gap is not None
                and right_frame - left_frame > max(1, int(max_gap))
                and any(
                    left_frame < frame < right_frame
                    for frame in raw_by_frame
                    if frame not in selected
                )
            ]
            if overlong:
                _gap, left_frame, right_frame = max(overlong)
                midpoint = 0.5 * (left_frame + right_frame)
                eligible = [
                    frame
                    for frame in raw_by_frame
                    if left_frame < frame < right_frame and frame not in selected
                ]
                chosen_frame = min(eligible, key=lambda frame: abs(frame - midpoint))
                selected[chosen_frame] = _raw_keyframe(
                    raw_by_frame[chosen_frame], point_count=point_count
                )
                continue
            candidates: list[tuple[float, float, float, int]] = []
            for left_frame, right_frame in zip(frames, frames[1:]):
                worst = _interval_worst_raw_frame(
                    selected[left_frame], selected[right_frame], raw_by_frame
                )
                if worst is not None:
                    candidates.append(worst)
            if not candidates:
                remaining = [
                    frame for frame in sorted(raw_by_frame) if frame not in selected
                ]
                if not remaining:
                    break
                chosen_frame = remaining[len(remaining) // 2]
            else:
                _recall, _iou, _negative_center_error, chosen_frame = min(candidates)
            selected[chosen_frame] = _raw_keyframe(
                raw_by_frame[chosen_frame], point_count=point_count
            )

        # A pathological segment can have more mandatory gap-fill keys than its
        # original in-range budget.  Preserve those keys rather than deleting
        # contract-required coverage.
        return _replace_segment_keys(segment, selected.values())

    return transform_segments(segments, transform)


def _interval_floor_area_priority(
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
    *,
    recall_floor: float,
    max_scale: float,
) -> tuple[float, int] | None:
    """Return interval repair cost and the frame that best splits that cost."""

    left_component = _primary_component(left)
    right_component = _primary_component(right)
    if left_component is None or right_component is None:
        return None
    base_left = np.asarray(left_component.values, dtype=np.float64)
    base_right = _numpy_resample(
        np.asarray(right_component.values, dtype=np.float64), len(base_left)
    )
    base_right = _numpy_align(base_left, base_right)
    frames = [
        frame for frame in sorted(raw_by_frame) if left.frame < frame < right.frame
    ]
    if not frames:
        return None

    best_rows = None
    for scale in np.linspace(1.0, float(max_scale), 17):
        left_center = np.mean(base_left, axis=0)
        right_center = np.mean(base_right, axis=0)
        left_points = left_center + scale * (base_left - left_center)
        right_points = right_center + scale * (base_right - right_center)
        rows = []
        for frame in frames:
            alpha = (frame - left.frame) / max(right.frame - left.frame, 1)
            predicted = geometry_from_arrays(
                [(1.0 - alpha) * left_points + alpha * right_points]
            )
            raw = raw_by_frame[frame].geometry
            intersection = float(raw.intersection(predicted).area)
            raw_area = float(raw.area)
            recall = intersection / raw_area if raw_area else 1.0
            excess = (
                (float(predicted.area) - intersection) / raw_area if raw_area else 0.0
            )
            raw_center = raw.centroid
            predicted_center = predicted.centroid
            center_error = math.hypot(
                predicted_center.x - raw_center.x,
                predicted_center.y - raw_center.y,
            ) / max(math.sqrt(raw_area), 1e-6)
            rows.append((frame, recall, excess, center_error))
        best_rows = rows
        if min(row[1] for row in rows) >= recall_floor:
            break
    assert best_rows is not None
    deficits = [max(recall_floor - row[1], 0.0) for row in best_rows]
    total_excess = sum(row[2] for row in best_rows)
    max_excess = max(row[2] for row in best_rows)
    # A violation is always more expensive than over-mask.  Among feasible
    # intervals, allocate keys where the integrated and tail expansion costs
    # are largest.
    priority = 1_000.0 * sum(deficits) + total_excess + 2.0 * max_excess
    split = max(
        best_rows,
        key=lambda row: (
            1_000.0 * max(recall_floor - row[1], 0.0) + row[2] + 0.25 * row[3],
            -abs(row[0] - 0.5 * (left.frame + right.frame)),
        ),
    )[0]
    return float(priority), int(split)


def floor_area_positions(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    max_scale: float = 1.30,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    """Use the fixed key budget to minimize recall repair expansion cost."""

    def transform(segment: Segment) -> Segment:
        baseline_inside = [
            keyframe
            for keyframe in segment.keyframes
            if start_frame <= keyframe.frame <= end_frame
        ]
        target_inside = len(baseline_inside)
        if target_inside <= 0:
            return segment
        raw_by_frame = {
            frame: raw
            for (frame, track_id), raw in raw_masks.items()
            if track_id == segment.track_id
            and segment.first_frame <= frame <= segment.last_frame
            and start_frame <= frame <= end_frame
        }
        if not raw_by_frame:
            return segment
        selected: dict[int, Keyframe] = {
            keyframe.frame: _resample_keyframe(keyframe, point_count=point_count)
            for keyframe in segment.keyframes
            if keyframe.frame < start_frame or keyframe.frame > end_frame
        }
        for keyframe in baseline_inside:
            if keyframe.frame not in raw_by_frame:
                selected[keyframe.frame] = _resample_keyframe(
                    keyframe, point_count=point_count
                )
        for endpoint in (segment.first_frame, segment.last_frame):
            if start_frame <= endpoint <= end_frame and endpoint in raw_by_frame:
                selected[endpoint] = _raw_keyframe(
                    raw_by_frame[endpoint], point_count=point_count
                )
        if len(selected) < 2:
            for frame in (min(raw_by_frame), max(raw_by_frame)):
                selected[frame] = _raw_keyframe(
                    raw_by_frame[frame], point_count=point_count
                )

        heap: list[tuple[float, int, int, int]] = []

        def push(left_frame: int, right_frame: int) -> None:
            result = _interval_floor_area_priority(
                selected[left_frame],
                selected[right_frame],
                raw_by_frame,
                recall_floor=recall_floor,
                max_scale=max_scale,
            )
            if result is None:
                return
            priority, split = result
            heapq.heappush(heap, (-priority, left_frame, right_frame, split))

        initial_frames = sorted(selected)
        for left_frame, right_frame in zip(initial_frames, initial_frames[1:]):
            push(left_frame, right_frame)
        while (
            sum(start_frame <= frame <= end_frame for frame in selected) < target_inside
        ):
            chosen = None
            while heap:
                _negative_priority, left_frame, right_frame, split = heapq.heappop(heap)
                frames = sorted(selected)
                position = bisect.bisect_left(frames, left_frame)
                if (
                    position + 1 < len(frames)
                    and frames[position] == left_frame
                    and frames[position + 1] == right_frame
                    and split not in selected
                ):
                    chosen = (left_frame, right_frame, split)
                    break
            if chosen is None:
                remaining = [frame for frame in raw_by_frame if frame not in selected]
                if not remaining:
                    break
                frames = sorted(selected)
                split = max(
                    remaining,
                    key=lambda frame: min(abs(frame - keyframe) for keyframe in frames),
                )
                position = bisect.bisect_left(frames, split)
                left_frame = frames[max(0, position - 1)]
                right_frame = frames[min(len(frames) - 1, position)]
            else:
                left_frame, right_frame, split = chosen
            selected[split] = _raw_keyframe(
                raw_by_frame[split], point_count=point_count
            )
            push(left_frame, split)
            push(split, right_frame)
        return _replace_segment_keys(segment, selected.values())

    return transform_segments(segments, transform)


def pair_vote_refine(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    vote_weight: float,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    """Apply local least-squares endpoint votes with an explicit anchor blend."""

    weight = float(np.clip(vote_weight, 0.0, 1.0))

    def transform(segment: Segment) -> Segment:
        keys = [
            _resample_keyframe(keyframe, point_count=point_count)
            for keyframe in segment.keyframes
        ]
        if len(keys) <= 1 or weight <= 0.0:
            return _replace_segment_keys(segment, keys)
        raw_by_frame = {
            frame: raw
            for (frame, track_id), raw in raw_masks.items()
            if track_id == segment.track_id
            and segment.first_frame <= frame <= segment.last_frame
        }
        proposals: list[list[tuple[np.ndarray, float]]] = [[] for _keyframe in keys]
        base_points = [
            np.asarray(_primary_component(keyframe).values, dtype=np.float64)
            for keyframe in keys
        ]
        for left_index in range(len(keys) - 1):
            right_index = left_index + 1
            left = keys[left_index]
            right = keys[right_index]
            rows = []
            targets = []
            reference = base_points[left_index]
            for frame in sorted(raw_by_frame):
                if frame < left.frame or frame > right.frame:
                    continue
                alpha = (frame - left.frame) / max(right.frame - left.frame, 1)
                points = _numpy_resample(
                    raw_by_frame[frame].primary_points, point_count
                )
                points = _numpy_align(reference, points)
                rows.append([1.0 - alpha, alpha])
                targets.append(points.reshape(-1))
            if len(rows) < 2:
                continue
            design = np.asarray(rows, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            solution = np.linalg.solve(
                design.T @ design + 1e-8 * np.eye(2),
                design.T @ target,
            )
            span_weight = float(len(rows))
            proposals[left_index].append(
                (solution[0].reshape(point_count, 2), span_weight)
            )
            proposals[right_index].append(
                (solution[1].reshape(point_count, 2), span_weight)
            )

        output = []
        for index, keyframe in enumerate(keys):
            anchor = base_points[index]
            if not proposals[index]:
                output.append(keyframe)
                continue
            total = sum(item_weight for _points, item_weight in proposals[index])
            voted = sum(
                points * item_weight for points, item_weight in proposals[index]
            ) / max(total, 1e-8)
            voted = _numpy_align(anchor, voted)
            constrained = (1.0 - weight) * anchor + weight * voted
            output.append(
                Keyframe(
                    keyframe.frame,
                    ((0, Component("polygon", constrained.tolist())),),
                )
            )
        return _replace_segment_keys(segment, output)

    return transform_segments(segments, transform)


def _scale_keyframe(keyframe: Keyframe, scale: float) -> Keyframe:
    output: list[tuple[int, Component]] = []
    for slot, component in keyframe.components:
        if component.kind != "polygon":
            output.append((slot, component))
            continue
        points = np.asarray(component.values, dtype=np.float64)
        center = np.mean(points, axis=0)
        scaled = center + float(scale) * (points - center)
        output.append((slot, Component("polygon", scaled.tolist())))
    return Keyframe(keyframe.frame, tuple(output))


def scale_all_keys(
    segments: dict[str, list[Segment]], scale: float
) -> dict[str, list[Segment]]:
    return transform_segments(
        segments,
        lambda segment: _replace_segment_keys(
            segment,
            (_scale_keyframe(keyframe, scale) for keyframe in segment.keyframes),
        ),
    )


def repair_interval_recall_with_scale(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    recall_floor: float,
    max_scale: float = 1.30,
    binary_steps: int = 8,
) -> dict[str, list[Segment]]:
    """Find a conservative per-key scale from adjacent interval requirements."""

    def transform(segment: Segment) -> Segment:
        keys = list(segment.keyframes)
        if not keys:
            return segment
        raw_by_frame = {
            frame: raw
            for (frame, track_id), raw in raw_masks.items()
            if track_id == segment.track_id
            and segment.first_frame <= frame <= segment.last_frame
        }
        raw_frames = sorted(raw_by_frame)
        scales = np.ones((len(keys),), dtype=np.float64)

        def interval_min_recall(
            left_index: int, right_index: int, scale: float
        ) -> float:
            left = _scale_keyframe(keys[left_index], scale)
            right = _scale_keyframe(keys[right_index], scale)
            left_component = _primary_component(left)
            right_component = _primary_component(right)
            if left_component is None or right_component is None:
                return 1.0
            left_points = np.asarray(left_component.values, dtype=np.float64)
            right_points = _numpy_resample(
                np.asarray(right_component.values, dtype=np.float64),
                len(left_points),
            )
            right_points = _numpy_align(left_points, right_points)
            minimum = 1.0
            observed = False
            begin = bisect.bisect_left(raw_frames, left.frame)
            finish = bisect.bisect_right(raw_frames, right.frame)
            for frame in raw_frames[begin:finish]:
                raw = raw_by_frame[frame]
                observed = True
                alpha = (frame - left.frame) / max(right.frame - left.frame, 1)
                predicted = geometry_from_arrays(
                    [(1.0 - alpha) * left_points + alpha * right_points]
                )
                intersection = float(raw.geometry.intersection(predicted).area)
                minimum = min(
                    minimum,
                    intersection / float(raw.geometry.area)
                    if raw.geometry.area
                    else 1.0,
                )
            return minimum if observed else 1.0

        for left_index in range(len(keys) - 1):
            right_index = left_index + 1
            if interval_min_recall(left_index, right_index, 1.0) >= recall_floor:
                continue
            if interval_min_recall(left_index, right_index, max_scale) < recall_floor:
                required = max_scale
            else:
                low = 1.0
                high = float(max_scale)
                for _step in range(max(1, int(binary_steps))):
                    middle = 0.5 * (low + high)
                    if (
                        interval_min_recall(left_index, right_index, middle)
                        >= recall_floor
                    ):
                        high = middle
                    else:
                        low = middle
                required = high
            scales[left_index] = max(scales[left_index], required)
            scales[right_index] = max(scales[right_index], required)
        return _replace_segment_keys(
            segment,
            (
                _scale_keyframe(keyframe, scales[index])
                for index, keyframe in enumerate(keys)
            ),
        )

    return transform_segments(segments, transform)


def _replacement_score(
    evaluations: list[FrameEvaluation], recall_floor: float
) -> tuple[int, float, float, float, float]:
    recalls = np.asarray([item.recall for item in evaluations], dtype=np.float64)
    deficits = np.maximum(float(recall_floor) - recalls, 0.0)
    excess = np.asarray(
        [item.excess_area_ratio for item in evaluations], dtype=np.float64
    )
    ious = np.asarray([item.iou for item in evaluations], dtype=np.float64)
    return (
        int(np.sum(deficits > 0.0)),
        float(np.max(deficits)) if len(deficits) else 0.0,
        float(np.sum(deficits)),
        float(np.mean(excess)) if len(excess) else 0.0,
        -float(np.mean(ious)) if len(ious) else -1.0,
    )


def swap_refine_fixed_budget(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    repair_margin: float = 0.01,
    max_scale: float = 1.50,
    donor_candidates: int = 10,
    max_iterations: int = 12,
    point_count: int = 23,
) -> dict[str, list[Segment]]:
    """Swap redundant keys into worst violations without changing key counts."""

    output = {
        track_id: list(track_segments) for track_id, track_segments in segments.items()
    }
    internal_floor = min(0.999, float(recall_floor) + float(repair_margin))

    for track_id, track_segments in list(output.items()):
        refined_track: list[Segment] = []
        for source_segment in track_segments:
            segment_raw = {
                key: value
                for key, value in raw_masks.items()
                if key[1] == track_id
                and source_segment.first_frame <= key[0] <= source_segment.last_frame
                and start_frame <= key[0] <= end_frame
            }
            if not segment_raw:
                refined_track.append(source_segment)
                continue
            current = source_segment

            def repaired(candidate: Segment) -> tuple[Segment, list[FrameEvaluation]]:
                repaired_map = repair_interval_recall_with_scale(
                    {track_id: [candidate]},
                    segment_raw,
                    recall_floor=internal_floor,
                    max_scale=max_scale,
                    binary_steps=9,
                )
                repaired_segment = repaired_map[track_id][0]
                evaluations = evaluate_segments(
                    segment_raw, {track_id: [repaired_segment]}
                )
                return repaired_segment, evaluations

            current_repaired, current_evaluations = repaired(current)
            current_score = _replacement_score(current_evaluations, recall_floor)
            for _iteration in range(max(1, int(max_iterations))):
                worst = min(current_evaluations, key=lambda item: item.recall)
                if worst.recall >= recall_floor:
                    break
                if worst.frame in {keyframe.frame for keyframe in current.keyframes}:
                    break
                raw_worst = segment_raw.get((worst.frame, track_id))
                if raw_worst is None:
                    break
                keys = list(current.keyframes)
                removable: list[tuple[float, int]] = []
                for index in range(1, len(keys) - 1):
                    keyframe = keys[index]
                    if not (start_frame <= keyframe.frame <= end_frame):
                        continue
                    raw_key = segment_raw.get((keyframe.frame, track_id))
                    if raw_key is None or keyframe.frame == worst.frame:
                        continue
                    predicted = geometry_from_arrays(
                        np.asarray(points, dtype=np.float64)
                        for points in _component_polygons(
                            _components_at(
                                [keys[index - 1], keys[index + 1]],
                                keyframe.frame,
                                "polygon_linear",
                            )
                        )
                    )
                    intersection = float(raw_key.geometry.intersection(predicted).area)
                    removal_recall = intersection / float(raw_key.geometry.area)
                    removable.append((removal_recall, index))
                if not removable:
                    break
                # A high reconstruction recall means the key is redundant if
                # removed.  Evaluate only the most promising donors exactly.
                removable.sort(reverse=True)
                best = None
                for _removal_recall, donor_index in removable[:donor_candidates]:
                    trial_keys = [
                        keyframe
                        for index, keyframe in enumerate(keys)
                        if index != donor_index
                    ]
                    trial_keys.append(_raw_keyframe(raw_worst, point_count=point_count))
                    trial = _replace_segment_keys(current, trial_keys)
                    trial_repaired, trial_evaluations = repaired(trial)
                    trial_score = _replacement_score(trial_evaluations, recall_floor)
                    if best is None or trial_score < best[0]:
                        best = (
                            trial_score,
                            trial,
                            trial_repaired,
                            trial_evaluations,
                        )
                if best is None or best[0] >= current_score:
                    break
                current_score, current, current_repaired, current_evaluations = best
            refined_track.append(current_repaired)
        output[track_id] = refined_track
    return output


def adaptive_add_recall_keys(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    repair_margin: float = 0.01,
    max_scale: float = 1.25,
    max_rounds: int = 12,
    point_count: int = 23,
    safe_anchor_floor: float | None = None,
) -> dict[str, list[Segment]]:
    """Add one worst violating frame per interpolation span until safe."""

    positions = {
        track_id: list(track_segments) for track_id, track_segments in segments.items()
    }
    internal_floor = min(0.999, float(recall_floor) + float(repair_margin))
    final = positions
    for _round in range(max(1, int(max_rounds))):
        final = repair_interval_recall_with_scale(
            positions,
            raw_masks,
            recall_floor=internal_floor,
            max_scale=max_scale,
            binary_steps=10,
        )
        evaluations = evaluate_segments(raw_masks, final)
        violations = [item for item in evaluations if item.recall < recall_floor]
        if not violations:
            return final
        grouped: dict[tuple[str, int, int, int], FrameEvaluation] = {}
        segment_lookup = {
            (track_id, segment.segment_id): segment
            for track_id, values in positions.items()
            for segment in values
        }
        for item in violations:
            segment = segment_lookup[(item.track_id, item.segment_id)]
            frames = [keyframe.frame for keyframe in segment.keyframes]
            position = bisect.bisect_left(frames, item.frame)
            if position < len(frames) and frames[position] == item.frame:
                left = right = item.frame
            else:
                left = frames[position - 1]
                right = frames[position]
            key = (item.track_id, item.segment_id, left, right)
            if key not in grouped or item.recall < grouped[key].recall:
                grouped[key] = item
        changed = False
        next_positions: dict[str, list[Segment]] = {}
        for track_id, values in positions.items():
            next_values = []
            for segment in values:
                additions = [
                    item
                    for (
                        candidate_track,
                        segment_id,
                        _left,
                        _right,
                    ), item in grouped.items()
                    if candidate_track == track_id and segment_id == segment.segment_id
                ]
                keys = list(segment.keyframes)
                existing = {keyframe.frame for keyframe in keys}
                for item in additions:
                    raw = raw_masks.get((item.frame, track_id))
                    if raw is None:
                        continue
                    replacement = (
                        _safe_anchor_keyframe(
                            segment,
                            raw,
                            anchor_recall=float(safe_anchor_floor),
                            point_count=point_count,
                        )
                        if safe_anchor_floor is not None
                        else _raw_keyframe(raw, point_count=point_count)
                    )
                    if item.frame in existing:
                        keys = [
                            replacement if keyframe.frame == item.frame else keyframe
                            for keyframe in keys
                        ]
                    else:
                        keys.append(replacement)
                        existing.add(item.frame)
                    changed = True
                next_values.append(_replace_segment_keys(segment, keys))
            next_positions[track_id] = next_values
        positions = next_positions
        if not changed:
            return final
    return final
