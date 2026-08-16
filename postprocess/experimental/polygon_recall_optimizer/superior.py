"""Production-preserving helpers for the superior polygon Pareto solver.

This module keeps the V3 schema and Production segment topology intact.  It
adds Production's border expansion to the hard Recall reference, evaluates IoU
against the unmodified AI observation, and independently reconstructs the
stored point_index interpolation used by editing software.
"""

from __future__ import annotations

import bisect
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from shapely.geometry import box

from production.polygon.input_geometry import _expand_polygon

from .fixed_budget import (
    FrameEvaluation,
    RawMask,
    Segment,
    _polygon_components,
    _polygonal,
    geometry_from_arrays,
)


@dataclass(frozen=True)
class BorderExpansionConfig:
    enabled: bool = True
    trigger_px: float = 10.0
    expand_ratio: float = 0.10
    min_expand_px: float = 6.0
    max_expand_px: float = 40.0
    influence_px: float = 24.0


@dataclass(frozen=True)
class BorderSideConstraint:
    """Hard visible-strip coverage and off-canvas extent for one frame side."""

    side: str
    visible_reference: object
    visible_area: float
    required_coordinate: float


@dataclass(frozen=True)
class BorderFrameConstraint:
    sides: tuple[BorderSideConstraint, ...]
    local_recall_floor: float
    max_repair_px: float
    quality_domain: object


def video_dimensions(path: Path) -> tuple[int, int]:
    """Read authoritative dimensions without decoding video pixels."""

    source = Path(path).expanduser().resolve()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT width, height FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
    if row is None or int(row[0]) <= 0 or int(row[1]) <= 0:
        raise ValueError(f"SQLite has no valid video dimensions: {source}")
    return int(row[0]), int(row[1])


def _raw_arrays(raw: RawMask) -> tuple[np.ndarray, ...]:
    if raw.component_points:
        return tuple(
            np.asarray(points, dtype=np.float64) for points in raw.component_points
        )
    geometry = raw.geometry
    polygons = []
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    if polygons:
        return tuple(
            np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
            for polygon in polygons
        )
    return (np.asarray(raw.primary_points, dtype=np.float64),)


def expand_border_constraints(
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    width: int,
    height: int,
    config: BorderExpansionConfig = BorderExpansionConfig(),
) -> tuple[dict[tuple[int, str], RawMask], dict[str, object]]:
    """Apply the exact Production border transform to hard-Recall masks.

    The original observation is unioned back into the constraint geometry,
    then the hard-Recall geometry is clipped to the visible video rectangle.
    Production deliberately pushes border vertices outside that rectangle;
    invisible off-canvas area must guide anchor construction but must not
    consume Recall budget or lower the visible-mask IoU.
    """

    if not config.enabled:
        return dict(raw_masks), {
            "enabled": False,
            "total_masks": len(raw_masks),
            "changed_masks": 0,
            "width": int(width),
            "height": int(height),
        }
    output: dict[tuple[int, str], RawMask] = {}
    visible_rectangle = box(0.0, 0.0, float(width), float(height))
    changed = 0
    touched = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    non_superset_before_union = 0
    for identity, raw in raw_masks.items():
        arrays = _raw_arrays(raw)
        expanded_arrays: list[np.ndarray] = []
        item_changed = False
        for points in arrays:
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            touched["left"] += int(float(minimum[0]) <= config.trigger_px)
            touched["right"] += int(
                float(maximum[0]) >= float(width - 1) - config.trigger_px
            )
            touched["top"] += int(float(minimum[1]) <= config.trigger_px)
            touched["bottom"] += int(
                float(maximum[1]) >= float(height - 1) - config.trigger_px
            )
            expanded, current_changed = _expand_polygon(
                np.asarray(points, dtype=np.float32),
                width=int(width),
                height=int(height),
                trigger_px=float(config.trigger_px),
                expand_ratio=float(config.expand_ratio),
                min_expand_px=float(config.min_expand_px),
                max_expand_px=float(config.max_expand_px),
                influence_px=float(config.influence_px),
            )
            expanded_arrays.append(np.asarray(expanded, dtype=np.float64))
            item_changed = item_changed or current_changed
        if not item_changed:
            output[identity] = raw
            continue
        changed += 1
        expanded_geometry = geometry_from_arrays(expanded_arrays)
        missing_original = raw.geometry.difference(expanded_geometry)
        non_superset_before_union += int(
            not missing_original.is_empty and float(missing_original.area) > 1e-8
        )
        guidance_geometry = _polygonal(expanded_geometry.union(raw.geometry))
        constraint_geometry = _polygonal(
            guidance_geometry.intersection(visible_rectangle)
        )
        guidance_polygons = (
            [guidance_geometry]
            if guidance_geometry.geom_type == "Polygon"
            else list(guidance_geometry.geoms)
        )
        guidance_arrays = [
            np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
            for polygon in guidance_polygons
            if not polygon.is_empty and float(polygon.area) > 0.0
        ]
        primary = max(
            guidance_arrays,
            key=lambda points: abs(
                float(
                    np.sum(
                        points[:, 0] * np.roll(points[:, 1], -1)
                        - np.roll(points[:, 0], -1) * points[:, 1]
                    )
                )
            ),
        )
        output[identity] = RawMask(
            frame=raw.frame,
            track_id=raw.track_id,
            geometry=constraint_geometry,
            primary_points=primary,
            score=raw.score,
            component_points=tuple(guidance_arrays),
        )
    return output, {
        "enabled": True,
        "total_masks": len(raw_masks),
        "changed_masks": changed,
        "changed_ratio": changed / max(len(raw_masks), 1),
        "side_counts": touched,
        "non_superset_before_union": non_superset_before_union,
        "constraint_domain": "visible_video_rectangle",
        "width": int(width),
        "height": int(height),
        "parameters": {
            "trigger_px": config.trigger_px,
            "expand_ratio": config.expand_ratio,
            "min_expand_px": config.min_expand_px,
            "max_expand_px": config.max_expand_px,
            "influence_px": config.influence_px,
        },
    }


def build_border_safety_constraints(
    raw_masks: dict[tuple[int, str], RawMask],
    expanded_masks: dict[tuple[int, str], RawMask],
    *,
    width: int,
    height: int,
    config: BorderExpansionConfig = BorderExpansionConfig(),
    local_recall_floor: float = 0.995,
) -> tuple[dict[tuple[int, str], BorderFrameConstraint], dict[str, object]]:
    """Build side-local constraints without charging off-canvas area to IoU."""

    if not 0.0 < local_recall_floor <= 1.0:
        raise ValueError("local_recall_floor must be in (0, 1]")
    if not config.enabled:
        return {}, {
            "enabled": False,
            "frame_count": 0,
            "side_counts": {side: 0 for side in ("left", "right", "top", "bottom")},
            "local_recall_floor": float(local_recall_floor),
        }
    strips = {
        "left": box(0.0, 0.0, float(config.influence_px), float(height)),
        "right": box(
            float(width) - float(config.influence_px),
            0.0,
            float(width),
            float(height),
        ),
        "top": box(0.0, 0.0, float(width), float(config.influence_px)),
        "bottom": box(
            0.0,
            float(height) - float(config.influence_px),
            float(width),
            float(height),
        ),
    }
    output: dict[tuple[int, str], BorderFrameConstraint] = {}
    side_counts = {side: 0 for side in strips}
    for identity, raw in raw_masks.items():
        arrays = _raw_arrays(raw)
        minimum = np.min(np.concatenate(arrays, axis=0), axis=0)
        maximum = np.max(np.concatenate(arrays, axis=0), axis=0)
        touched: list[str] = []
        if float(minimum[0]) <= config.trigger_px:
            touched.append("left")
        if float(maximum[0]) >= float(width - 1) - config.trigger_px:
            touched.append("right")
        if float(minimum[1]) <= config.trigger_px:
            touched.append("top")
        if float(maximum[1]) >= float(height - 1) - config.trigger_px:
            touched.append("bottom")
        if not touched:
            continue
        expanded = expanded_masks[identity]
        expanded_arrays = _raw_arrays(expanded)
        expanded_points = np.concatenate(expanded_arrays, axis=0)
        expanded_minimum = np.min(expanded_points, axis=0)
        expanded_maximum = np.max(expanded_points, axis=0)
        constraints: list[BorderSideConstraint] = []
        for side in touched:
            reference = _polygonal(raw.geometry.intersection(strips[side]))
            area = float(reference.area)
            if area <= 1e-8:
                continue
            required = {
                "left": min(float(expanded_minimum[0]), -float(config.min_expand_px)),
                "right": max(
                    float(expanded_maximum[0]),
                    float(width - 1) + float(config.min_expand_px),
                ),
                "top": min(float(expanded_minimum[1]), -float(config.min_expand_px)),
                "bottom": max(
                    float(expanded_maximum[1]),
                    float(height - 1) + float(config.min_expand_px),
                ),
            }[side]
            constraints.append(
                BorderSideConstraint(
                    side=side,
                    visible_reference=reference,
                    visible_area=area,
                    required_coordinate=required,
                )
            )
            side_counts[side] += 1
        if constraints:
            excluded = strips[constraints[0].side]
            for side in constraints[1:]:
                excluded = excluded.union(strips[side.side])
            quality_domain = box(0.0, 0.0, float(width), float(height)).difference(
                excluded
            )
            output[identity] = BorderFrameConstraint(
                sides=tuple(constraints),
                local_recall_floor=float(local_recall_floor),
                max_repair_px=float(config.max_expand_px),
                quality_domain=quality_domain,
            )
    return output, {
        "enabled": True,
        "frame_count": len(output),
        "side_counts": side_counts,
        "local_recall_floor": float(local_recall_floor),
        "strip_width_px": float(config.influence_px),
        "minimum_offcanvas_px": float(config.min_expand_px),
        "extent_source": "production_expand_polygon",
        "quality_objective_domain": "visible_frame_minus_constrained_border_strips",
    }


def border_geometry_metrics(
    geometry, constraint: BorderFrameConstraint | None
) -> tuple[float, int]:
    """Return minimum local Recall and count of side-extent violations."""

    if constraint is None:
        return 1.0, 0
    bounds = geometry.bounds
    minimum_recall = 1.0
    extent_violations = 0
    for side in constraint.sides:
        intersection = float(side.visible_reference.intersection(geometry).area)
        recall = intersection / side.visible_area if side.visible_area else 1.0
        minimum_recall = min(minimum_recall, recall)
        coordinate = {
            "left": bounds[0],
            "right": bounds[2],
            "top": bounds[1],
            "bottom": bounds[3],
        }[side.side]
        if side.side in {"left", "top"}:
            extent_violations += int(coordinate > side.required_coordinate + 1e-9)
        else:
            extent_violations += int(coordinate < side.required_coordinate - 1e-9)
    return minimum_recall, extent_violations


def border_geometry_feasible(
    geometry, constraint: BorderFrameConstraint | None
) -> bool:
    recall, extent_violations = border_geometry_metrics(geometry, constraint)
    return bool(
        extent_violations == 0
        and (constraint is None or recall + 1e-12 >= constraint.local_recall_floor)
    )


def audit_border_safety(
    constraints: dict[tuple[int, str], BorderFrameConstraint],
    segments: dict[str, list[Segment]],
) -> dict[str, object]:
    """Independently audit stored point-index interpolation at every edge frame."""

    minimum_local_recall = 1.0
    recall_violations = 0
    extent_violations = 0
    missing_frames = 0
    side_count = 0
    worst: list[dict[str, object]] = []
    for (frame, track_id), constraint in sorted(constraints.items()):
        segment = next(
            (
                value
                for value in segments.get(track_id, ())
                if value.first_frame <= frame <= value.last_frame
            ),
            None,
        )
        if segment is None:
            missing_frames += 1
            continue
        geometry = direct_geometry_at(segment, frame)
        if geometry.is_empty:
            missing_frames += 1
            continue
        local_recall, current_extent_violations = border_geometry_metrics(
            geometry, constraint
        )
        minimum_local_recall = min(minimum_local_recall, local_recall)
        recall_violations += int(local_recall + 1e-12 < constraint.local_recall_floor)
        extent_violations += current_extent_violations
        side_count += len(constraint.sides)
        worst.append(
            {
                "frame": int(frame),
                "track_id": track_id,
                "local_recall": float(local_recall),
                "extent_violations": int(current_extent_violations),
                "sides": [side.side for side in constraint.sides],
            }
        )
    worst.sort(
        key=lambda item: (
            float(item["local_recall"]),
            -int(item["extent_violations"]),
        )
    )
    return {
        "frame_count": len(constraints),
        "side_count": side_count,
        "minimum_local_recall": minimum_local_recall,
        "recall_violations": recall_violations,
        "extent_violations": extent_violations,
        "missing_frames": missing_frames,
        "passed": (
            recall_violations == 0 and extent_violations == 0 and missing_frames == 0
        ),
        "worst_frames": worst[:20],
    }


def supported_single_component_segments(
    segments: dict[str, list[Segment]],
) -> tuple[dict[str, list[Segment]], list[dict[str, object]]]:
    """Keep unsupported Production topology unchanged instead of degrading it."""

    supported: dict[str, list[Segment]] = {}
    fallbacks: list[dict[str, object]] = []
    for track_id, values in segments.items():
        accepted: list[Segment] = []
        for segment in values:
            counts = [len(_polygon_components(key)) for key in segment.keyframes]
            point_counts = [
                len(component.values)
                for key in segment.keyframes
                for _slot, component in _polygon_components(key)
            ]
            if counts and all(count == 1 for count in counts) and point_counts:
                accepted.append(segment)
            else:
                fallbacks.append(
                    {
                        "track_id": track_id,
                        "segment_id": segment.segment_id,
                        "reason": "non_single_polygon_topology",
                        "component_counts": counts,
                    }
                )
        if accepted:
            supported[track_id] = accepted
    return supported, fallbacks


def direct_geometry_at(segment: Segment, frame: int):
    """Reconstruct point_index interpolation without roll/reversal alignment."""

    keys = list(segment.keyframes)
    if not keys or frame < keys[0].frame or frame > keys[-1].frame:
        return geometry_from_arrays([])
    frames = [key.frame for key in keys]
    position = bisect.bisect_left(frames, int(frame))
    if position < len(keys) and frames[position] == frame:
        arrays = [
            np.asarray(component.values, dtype=np.float64)
            for _slot, component in _polygon_components(keys[position])
        ]
        return geometry_from_arrays(arrays)
    if position == 0 or position == len(keys):
        return geometry_from_arrays([])
    left = keys[position - 1]
    right = keys[position]
    left_components = _polygon_components(left)
    right_components = _polygon_components(right)
    if len(left_components) != len(right_components):
        return geometry_from_arrays([])
    alpha = (int(frame) - left.frame) / max(right.frame - left.frame, 1)
    arrays: list[np.ndarray] = []
    for (_left_slot, left_component), (
        _right_slot,
        right_component,
    ) in zip(left_components, right_components, strict=True):
        left_points = np.asarray(left_component.values, dtype=np.float64)
        right_points = np.asarray(right_component.values, dtype=np.float64)
        if len(left_points) != len(right_points):
            return geometry_from_arrays([])
        arrays.append((1.0 - alpha) * left_points + alpha * right_points)
    return geometry_from_arrays(arrays)


def evaluate_direct(
    references: dict[tuple[int, str], RawMask],
    segments: dict[str, list[Segment]],
    *,
    visible_rectangle=None,
    border_constraints: dict[tuple[int, str], BorderFrameConstraint] | None = None,
) -> list[FrameEvaluation]:
    output: list[FrameEvaluation] = []
    key_sets = {
        segment.segment_id: {key.frame for key in segment.keyframes}
        for values in segments.values()
        for segment in values
    }
    for (frame, track_id), raw in sorted(references.items()):
        segment = next(
            (
                value
                for value in segments.get(track_id, ())
                if value.first_frame <= frame <= value.last_frame
            ),
            None,
        )
        if segment is None:
            continue
        predicted = direct_geometry_at(segment, frame)
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
                excess_area_ratio=(
                    float(metric_predicted.difference(metric_raw).area) / raw_area
                    if raw_area
                    else 0.0
                ),
                centroid_error_px=math.hypot(
                    predicted_center.x - raw.geometry.centroid.x,
                    predicted_center.y - raw.geometry.centroid.y,
                ),
                raw_geometry=raw.geometry,
                predicted_geometry=predicted,
            )
        )
    return output


def summarize_minimal(rows: list[FrameEvaluation]) -> dict[str, float | int]:
    if not rows:
        return {
            "frame_count": 0,
            "recall_min": 1.0,
            "recall_mean": 1.0,
            "iou_min": 1.0,
            "iou_mean": 1.0,
        }
    return {
        "frame_count": len(rows),
        "recall_min": min(row.recall for row in rows),
        "recall_mean": mean(row.recall for row in rows),
        "iou_min": min(row.iou for row in rows),
        "iou_mean": mean(row.iou for row in rows),
        "precision_mean": mean(row.precision for row in rows),
        "area_ratio_mean": mean(row.area_ratio for row in rows),
    }


def compare_geometry_paths(
    left: list[FrameEvaluation], right: list[FrameEvaluation]
) -> dict[str, float | int]:
    left_by_identity = {(row.frame, row.track_id): row for row in left}
    right_by_identity = {(row.frame, row.track_id): row for row in right}
    identities = sorted(set(left_by_identity).intersection(right_by_identity))
    differences = [
        float(
            left_by_identity[identity]
            .predicted_geometry.symmetric_difference(
                right_by_identity[identity].predicted_geometry
            )
            .area
        )
        for identity in identities
    ]
    return {
        "sample_count": len(identities),
        "symmetric_difference_max_area": max(differences, default=0.0),
        "symmetric_difference_mean_area": mean(differences) if differences else 0.0,
        "nonzero_difference_count": sum(value > 1e-7 for value in differences),
    }


__all__ = [
    "BorderExpansionConfig",
    "BorderFrameConstraint",
    "BorderSideConstraint",
    "audit_border_safety",
    "border_geometry_feasible",
    "border_geometry_metrics",
    "build_border_safety_constraints",
    "compare_geometry_paths",
    "direct_geometry_at",
    "evaluate_direct",
    "expand_border_constraints",
    "summarize_minimal",
    "supported_single_component_segments",
    "video_dimensions",
]
