"""Production-preserving polygon V3 with two per-frame Recall constraints.

The optimizer deliberately starts from the validated Production keyframe path.
It keeps every Production key position and shape that is already feasible.  It
only repairs an existing anchor when that anchor violates a hard constraint,
and only inserts a key when a Production interpolation edge cannot satisfy:

1. minimum full-mask Recall against the original AI observation; and
2. minimum side-local Recall plus the Production off-canvas extent contract
   for observations touching a video boundary.

The two constraints are evaluated independently at every observed frame.  No
mean Recall budget, post-decode pair-vote mutation, fixed global key count, or
free 30 percent expansion state is used here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from ..polygon_recall_optimizer.fixed_budget import (
    Keyframe,
    RawMask,
    Segment,
    _primary_component,
    _raw_keyframe,
    _safe_anchor_keyframe,
    _scale_keyframe,
    geometry_from_arrays,
)
from ..polygon_recall_optimizer.pareto_dp import canonicalize_selected_path
from ..polygon_recall_optimizer.superior import (
    BorderFrameConstraint,
    border_geometry_metrics,
    direct_geometry_at,
)


@dataclass(frozen=True)
class ProductionPolygonV3Result:
    segments: dict[str, list[Segment]]
    adjusted_production_keys: int
    added_constraint_keys: int
    evaluated_edges: int
    feasible_edges: int
    elapsed_seconds: float
    segment_diagnostics: tuple[dict[str, object], ...]


def _keyframe_geometry(keyframe: Keyframe):
    component = _primary_component(keyframe)
    if component is None:
        return geometry_from_arrays([])
    return geometry_from_arrays(
        [np.asarray(component.values, dtype=np.float64)]
    )


def _normal_metrics(raw: RawMask, geometry) -> tuple[float, float]:
    intersection = float(raw.geometry.intersection(geometry).area)
    raw_area = float(raw.geometry.area)
    predicted_area = float(geometry.area)
    union = raw_area + predicted_area - intersection
    return (
        intersection / raw_area if raw_area else 1.0,
        intersection / union if union else 1.0,
    )


def _dual_metrics(
    raw: RawMask,
    geometry,
    border_constraint: BorderFrameConstraint | None,
) -> tuple[float, float, float, int]:
    normal_recall, iou = _normal_metrics(raw, geometry)
    border_recall, extent_violations = border_geometry_metrics(
        geometry, border_constraint
    )
    return normal_recall, border_recall, iou, extent_violations


def _dual_feasible(
    raw: RawMask,
    geometry,
    border_constraint: BorderFrameConstraint | None,
    *,
    normal_recall_floor: float,
) -> bool:
    normal, border, _iou, extent = _dual_metrics(
        raw, geometry, border_constraint
    )
    border_floor = (
        0.0
        if border_constraint is None
        else float(border_constraint.local_recall_floor)
    )
    return bool(
        normal + 1e-12 >= normal_recall_floor
        and border + 1e-12 >= border_floor
        and extent == 0
    )


def _minimum_dual_scale(
    source: Keyframe,
    raw: RawMask,
    border_constraint: BorderFrameConstraint | None,
    *,
    normal_recall_floor: float,
    max_scale: float,
) -> Keyframe | None:
    """Return the first centroid scale satisfying both hard constraints."""

    geometry = _keyframe_geometry(source)
    if _dual_feasible(
        raw,
        geometry,
        border_constraint,
        normal_recall_floor=normal_recall_floor,
    ):
        return source
    low = 1.0
    high: float | None = None
    previous = 1.0
    for scale in np.linspace(1.0, float(max_scale), 65)[1:]:
        candidate = _scale_keyframe(source, float(scale))
        if _dual_feasible(
            raw,
            _keyframe_geometry(candidate),
            border_constraint,
            normal_recall_floor=normal_recall_floor,
        ):
            low = previous
            high = float(scale)
            break
        previous = float(scale)
    if high is None:
        return None
    for _step in range(14):
        middle = 0.5 * (low + high)
        candidate = _scale_keyframe(source, middle)
        if _dual_feasible(
            raw,
            _keyframe_geometry(candidate),
            border_constraint,
            normal_recall_floor=normal_recall_floor,
        ):
            high = middle
        else:
            low = middle
    return _scale_keyframe(source, high)


def _dual_safe_anchor(
    segment: Segment,
    raw: RawMask,
    expanded_raw: RawMask,
    border_constraint: BorderFrameConstraint | None,
    *,
    normal_recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    production_key: Keyframe | None,
) -> Keyframe:
    """Keep a feasible Production anchor, otherwise make a minimal repair."""

    if production_key is not None:
        if _dual_feasible(
            raw,
            _keyframe_geometry(production_key),
            border_constraint,
            normal_recall_floor=normal_recall_floor,
        ):
            return production_key

    # The candidates are deliberately conservative.  Unlike Superior V1/V2,
    # V3 does not add optional 4..30 percent expansion states.  Scaling is used
    # only to reach a hard constraint and stops at its first feasible value.
    sources = [
        _safe_anchor_keyframe(
            segment,
            raw,
            anchor_recall=normal_recall_floor,
            point_count=point_count,
        ),
        _raw_keyframe(raw, point_count=point_count),
    ]
    if border_constraint is not None:
        sources.append(_raw_keyframe(expanded_raw, point_count=point_count))

    feasible: list[tuple[float, float, float, Keyframe]] = []
    for source in sources:
        candidate = _minimum_dual_scale(
            source,
            raw,
            border_constraint,
            normal_recall_floor=normal_recall_floor,
            max_scale=max_anchor_scale,
        )
        if candidate is None:
            continue
        normal, border, iou, extent = _dual_metrics(
            raw, _keyframe_geometry(candidate), border_constraint
        )
        if extent == 0:
            feasible.append((iou, normal, border, candidate))
    if not feasible:
        raise RuntimeError(
            f"frame {raw.frame} track {raw.track_id} cannot form an anchor "
            f"satisfying normal Recall {normal_recall_floor:.6f} and the "
            "screen-edge constraint"
        )
    return max(feasible, key=lambda item: (item[0], item[1], item[2]))[3]


def _edge_metrics(
    segment: Segment,
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
    border_by_frame: dict[int, BorderFrameConstraint],
    *,
    normal_recall_floor: float,
) -> tuple[bool, float, float, float, int]:
    """Densely evaluate one edge with the editor's stored-point semantics."""

    canonical = canonicalize_selected_path((left, right))
    temporary = Segment(
        segment_id=-1,
        track_id=segment.track_id,
        first_frame=canonical[0].frame,
        last_frame=canonical[1].frame,
        interpolation_method="linear_polygon_index_v1",
        keyframes=canonical,
    )
    minimum_normal = 1.0
    minimum_border = 1.0
    iou_sum = 0.0
    quality_count = 0
    for frame in sorted(raw_by_frame):
        if frame < left.frame or frame > right.frame:
            continue
        raw = raw_by_frame[frame]
        predicted = direct_geometry_at(temporary, frame)
        if predicted.is_empty:
            return False, iou_sum, 0.0, 0.0, quality_count
        normal, border, iou, extent = _dual_metrics(
            raw, predicted, border_by_frame.get(frame)
        )
        minimum_normal = min(minimum_normal, normal)
        minimum_border = min(minimum_border, border)
        if frame > left.frame:
            iou_sum += iou
            quality_count += 1
        border_floor = (
            0.0
            if frame not in border_by_frame
            else float(border_by_frame[frame].local_recall_floor)
        )
        if (
            normal + 1e-12 < normal_recall_floor
            or border + 1e-12 < border_floor
            or extent != 0
        ):
            return (
                False,
                iou_sum,
                minimum_normal,
                minimum_border,
                quality_count,
            )
    return True, iou_sum, minimum_normal, minimum_border, quality_count


def _guard_interval(
    segment: Segment,
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
    border_by_frame: dict[int, BorderFrameConstraint],
    anchors: dict[int, Keyframe],
    *,
    normal_recall_floor: float,
) -> tuple[list[Keyframe], int, int]:
    """Insert the fewest keys needed inside one Production interval."""

    interior = sorted(frame for frame in raw_by_frame if left.frame < frame < right.frame)
    positions = [left.frame, *interior, right.frame]
    candidates = {left.frame: left, right.frame: right}
    candidates.update({frame: anchors[frame] for frame in interior})
    # State: added keys, negative accumulated IoU, predecessor.
    best: list[tuple[int, float, int] | None] = [None] * len(positions)
    best[0] = (0, 0.0, -1)
    evaluated = 0
    feasible_count = 0
    for right_index in range(1, len(positions)):
        for left_index in range(right_index):
            previous = best[left_index]
            if previous is None:
                continue
            evaluated += 1
            feasible, iou_sum, _normal, _border, _count = _edge_metrics(
                segment,
                candidates[positions[left_index]],
                candidates[positions[right_index]],
                raw_by_frame,
                border_by_frame,
                normal_recall_floor=normal_recall_floor,
            )
            if not feasible:
                continue
            feasible_count += 1
            added = previous[0] + (
                0 if right_index == len(positions) - 1 else 1
            )
            proposal = (added, previous[1] - iou_sum, left_index)
            current = best[right_index]
            if current is None or proposal[:2] < current[:2]:
                best[right_index] = proposal
    if best[-1] is None:
        raise RuntimeError(
            f"no dual-Recall-safe path in Production interval "
            f"{left.frame}..{right.frame} for segment {segment.segment_id}"
        )
    selected_indices: list[int] = []
    cursor = len(positions) - 1
    while cursor >= 0:
        selected_indices.append(cursor)
        predecessor = best[cursor][2] if best[cursor] is not None else -1
        if predecessor < 0:
            break
        cursor = predecessor
    selected_indices.reverse()
    return (
        [candidates[positions[index]] for index in selected_indices],
        evaluated,
        feasible_count,
    )


def guard_production_v3(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    expanded_masks: dict[tuple[int, str], RawMask],
    border_constraints: dict[tuple[int, str], BorderFrameConstraint],
    *,
    start_frame: int,
    end_frame: int,
    normal_recall_floor: float,
    point_count: int = 0,
    max_anchor_scale: float = 1.25,
) -> ProductionPolygonV3Result:
    """Apply dual hard Recall guards while preserving Production structure."""

    if not 0.0 < normal_recall_floor <= 1.0:
        raise ValueError("normal_recall_floor must be in (0, 1]")
    started = time.perf_counter()
    output: dict[str, list[Segment]] = {}
    adjusted_total = 0
    added_total = 0
    evaluated_total = 0
    feasible_total = 0
    diagnostics: list[dict[str, object]] = []

    for track_id, track_segments in segments.items():
        guarded_track: list[Segment] = []
        for segment in track_segments:
            first_component = (
                _primary_component(segment.keyframes[0])
                if segment.keyframes
                else None
            )
            segment_point_count = (
                int(point_count)
                if int(point_count) > 0
                else (
                    len(first_component.values)
                    if first_component is not None
                    else 23
                )
            )
            segment_raw = {
                frame: raw
                for (frame, candidate_track), raw in raw_masks.items()
                if candidate_track == track_id
                and segment.first_frame <= frame <= segment.last_frame
                and start_frame <= frame <= end_frame
            }
            if not segment_raw or len(segment.keyframes) < 2:
                guarded_track.append(segment)
                continue
            production_by_frame = {
                keyframe.frame: keyframe for keyframe in segment.keyframes
            }
            anchor_cache: dict[int, Keyframe] = {}
            adjusted_frames: list[int] = []
            for frame, raw in segment_raw.items():
                production_key = production_by_frame.get(frame)
                anchor = _dual_safe_anchor(
                    segment,
                    raw,
                    expanded_masks[(frame, track_id)],
                    border_constraints.get((frame, track_id)),
                    normal_recall_floor=normal_recall_floor,
                    point_count=segment_point_count,
                    max_anchor_scale=max_anchor_scale,
                    production_key=production_key,
                )
                anchor_cache[frame] = anchor
                if production_key is not None and anchor != production_key:
                    adjusted_frames.append(frame)

            guarded_production = [
                anchor_cache.get(
                    keyframe.frame,
                    keyframe,
                )
                for keyframe in segment.keyframes
            ]
            selected: list[Keyframe] = [guarded_production[0]]
            segment_evaluated = 0
            segment_feasible = 0
            for left, right in zip(guarded_production, guarded_production[1:]):
                interval_raw = {
                    frame: raw
                    for frame, raw in segment_raw.items()
                    if left.frame <= frame <= right.frame
                }
                if not interval_raw:
                    selected.append(right)
                    continue
                interval_border = {
                    frame: border_constraints[(frame, track_id)]
                    for frame in interval_raw
                    if (frame, track_id) in border_constraints
                }
                interval_keys, evaluated, feasible = _guard_interval(
                    segment,
                    left,
                    right,
                    interval_raw,
                    interval_border,
                    anchor_cache,
                    normal_recall_floor=normal_recall_floor,
                )
                selected.extend(interval_keys[1:])
                segment_evaluated += evaluated
                segment_feasible += feasible

            canonical = canonicalize_selected_path(tuple(selected))
            guarded = replace(
                segment,
                interpolation_method="linear_polygon_index_v1",
                keyframes=canonical,
            )
            original_frames = set(production_by_frame)
            added_frames = sorted(
                key.frame for key in guarded.keyframes if key.frame not in original_frames
            )
            adjusted_total += len(adjusted_frames)
            added_total += len(added_frames)
            evaluated_total += segment_evaluated
            feasible_total += segment_feasible
            diagnostics.append(
                {
                    "track_id": track_id,
                    "segment_id": int(segment.segment_id),
                    "production_key_count": len(segment.keyframes),
                    "v3_key_count": len(guarded.keyframes),
                    "adjusted_production_key_count": len(adjusted_frames),
                    "adjusted_production_frames": adjusted_frames,
                    "added_constraint_key_count": len(added_frames),
                    "added_constraint_frames": added_frames,
                    "border_observation_count": sum(
                        (frame, track_id) in border_constraints for frame in segment_raw
                    ),
                    "evaluated_edges": segment_evaluated,
                    "feasible_edges": segment_feasible,
                }
            )
            guarded_track.append(guarded)
        output[track_id] = guarded_track

    return ProductionPolygonV3Result(
        segments=output,
        adjusted_production_keys=adjusted_total,
        added_constraint_keys=added_total,
        evaluated_edges=evaluated_total,
        feasible_edges=feasible_total,
        elapsed_seconds=time.perf_counter() - started,
        segment_diagnostics=tuple(diagnostics),
    )


__all__ = ["ProductionPolygonV3Result", "guard_production_v3"]
