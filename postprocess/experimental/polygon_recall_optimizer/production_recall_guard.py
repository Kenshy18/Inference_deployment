"""Minimal Recall safety guard for Production polygon keyframes.

This experiment deliberately keeps every Production keyframe position.  It
only changes a Production anchor when that anchor itself violates the requested
raw-observation Recall floor, and only inserts an additional key when the
Production interpolation interval cannot satisfy the floor.  No temporal
consensus mask is constructed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .fixed_budget import (
    Component,
    Keyframe,
    RawMask,
    Segment,
    _numpy_resample,
    _raw_keyframe,
    _replace_segment_keys,
    _safe_anchor_keyframe,
    _scale_keyframe,
    geometry_at,
    geometry_from_arrays,
)


@dataclass(frozen=True)
class ProductionRecallGuardResult:
    segments: dict[str, list[Segment]]
    adjusted_production_keys: int
    added_recall_keys: int
    evaluated_edges: int
    feasible_edges: int
    elapsed_seconds: float
    segment_diagnostics: tuple[dict[str, object], ...]


def _anchor_metrics(raw: RawMask, keyframe: Keyframe) -> tuple[float, float]:
    polygons = [
        np.asarray(component.values, dtype=np.float64)
        for _slot, component in keyframe.components
        if component.kind == "polygon"
    ]
    if not polygons:
        return 0.0, 0.0
    predicted = geometry_from_arrays(polygons)
    intersection = float(raw.geometry.intersection(predicted).area)
    raw_area = float(raw.geometry.area)
    predicted_area = float(predicted.area)
    union = raw_area + predicted_area - intersection
    return (
        intersection / raw_area if raw_area else 1.0,
        intersection / union if union else 1.0,
    )


def _raw_components_keyframe(raw: RawMask, *, point_count: int) -> Keyframe:
    """Represent every raw connected component without changing the schema."""

    components = tuple(
        (
            slot,
            Component(
                "polygon",
                _numpy_resample(
                    np.asarray(points, dtype=np.float64), max(8, int(point_count))
                ).tolist(),
            ),
        )
        for slot, points in enumerate(raw.component_points)
    )
    if not components:
        return _raw_keyframe(raw, point_count=point_count)
    return Keyframe(frame=raw.frame, components=components)


def _minimally_scaled(
    keyframe: Keyframe,
    raw: RawMask,
    *,
    recall_floor: float,
    max_scale: float,
) -> Keyframe | None:
    recall, _iou = _anchor_metrics(raw, keyframe)
    if recall >= recall_floor:
        return keyframe
    low = 1.0
    high: float | None = None
    previous = 1.0
    # Polygon Recall is normally monotonic under this small centroid scale,
    # but explicitly bracket the first feasible region before bisection.
    for scale in np.linspace(1.0, float(max_scale), 41)[1:]:
        candidate = _scale_keyframe(keyframe, float(scale))
        candidate_recall, _candidate_iou = _anchor_metrics(raw, candidate)
        if candidate_recall >= recall_floor:
            low = previous
            high = float(scale)
            break
        previous = float(scale)
    if high is None:
        return None
    for _step in range(14):
        middle = 0.5 * (low + high)
        candidate = _scale_keyframe(keyframe, middle)
        candidate_recall, _candidate_iou = _anchor_metrics(raw, candidate)
        if candidate_recall >= recall_floor:
            high = middle
        else:
            low = middle
    return _scale_keyframe(keyframe, high)


def _guard_anchor(
    segment: Segment,
    raw: RawMask,
    *,
    recall_floor: float,
    point_count: int,
    max_anchor_scale: float,
    production_key: Keyframe | None,
) -> Keyframe:
    """Return the least disruptive Recall-safe anchor for one observation."""

    if production_key is not None:
        production_recall, _production_iou = _anchor_metrics(raw, production_key)
        if production_recall >= recall_floor:
            return production_key

    if len(raw.component_points) > 1:
        # A primary-component projection would silently discard raw islands.
        # The forceful guard is allowed to add a key, not to alter topology.
        candidates = [_raw_components_keyframe(raw, point_count=point_count)]
    else:
        candidates = [
            _safe_anchor_keyframe(
                segment,
                raw,
                anchor_recall=recall_floor,
                point_count=point_count,
            ),
            _raw_keyframe(raw, point_count=point_count),
        ]
    feasible: list[tuple[float, float, Keyframe]] = []
    for source in candidates:
        candidate = _minimally_scaled(
            source,
            raw,
            recall_floor=recall_floor,
            max_scale=max_anchor_scale,
        )
        if candidate is None:
            continue
        recall, iou = _anchor_metrics(raw, candidate)
        feasible.append((iou, recall, candidate))
    if not feasible:
        raise RuntimeError(
            f"frame {raw.frame} track {raw.track_id} cannot form a polygon "
            f"anchor with Recall >= {recall_floor:.6f}"
        )
    # Among Recall-safe alternatives, use the closest match to the raw mask.
    return max(feasible, key=lambda item: (item[0], item[1]))[2]


def _edge_metrics(
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
    *,
    interpolation_method: str,
    recall_floor: float,
) -> tuple[bool, float, float, int]:
    """Evaluate one edge with exactly the overlay reconstruction semantics."""

    temporary = Segment(
        segment_id=-1,
        track_id=next(iter(raw_by_frame.values())).track_id,
        first_frame=left.frame,
        last_frame=right.frame,
        interpolation_method=interpolation_method,
        keyframes=(left, right),
    )
    minimum_recall = 1.0
    iou_sum = 0.0
    quality_count = 0
    for frame in sorted(raw_by_frame):
        if frame < left.frame or frame > right.frame:
            continue
        raw = raw_by_frame[frame]
        predicted = geometry_at(temporary, frame)
        if predicted.is_empty:
            return False, 0.0, 0.0, quality_count
        intersection = float(raw.geometry.intersection(predicted).area)
        raw_area = float(raw.geometry.area)
        predicted_area = float(predicted.area)
        recall = intersection / raw_area if raw_area else 1.0
        minimum_recall = min(minimum_recall, recall)
        # Count (left, right] so a decoded path counts every observation once.
        if frame > left.frame:
            union = raw_area + predicted_area - intersection
            iou_sum += intersection / union if union else 1.0
            quality_count += 1
        if recall + 1e-12 < recall_floor:
            return False, iou_sum, minimum_recall, quality_count
    return True, iou_sum, minimum_recall, quality_count


def _guard_production_interval(
    segment: Segment,
    left: Keyframe,
    right: Keyframe,
    raw_by_frame: dict[int, RawMask],
    anchor_cache: dict[int, Keyframe],
    *,
    recall_floor: float,
) -> tuple[list[Keyframe], int, int]:
    """Find the fewest added keys required inside one Production interval."""

    interior = sorted(
        frame for frame in raw_by_frame if left.frame < frame < right.frame
    )
    positions = [left.frame, *interior, right.frame]
    anchors = {left.frame: left, right.frame: right}
    anchors.update({frame: anchor_cache[frame] for frame in interior})

    # State value: (additional key count, negative IoU sum, predecessor).
    best: list[tuple[int, float, int] | None] = [None] * len(positions)
    best[0] = (0, 0.0, -1)
    evaluated_edges = 0
    feasible_edges = 0
    for right_index in range(1, len(positions)):
        for left_index in range(right_index):
            prior = best[left_index]
            if prior is None:
                continue
            evaluated_edges += 1
            feasible, iou_sum, _minimum_recall, _count = _edge_metrics(
                anchors[positions[left_index]],
                anchors[positions[right_index]],
                raw_by_frame,
                interpolation_method=segment.interpolation_method,
                recall_floor=recall_floor,
            )
            if not feasible:
                continue
            feasible_edges += 1
            added = prior[0] + (0 if right_index == len(positions) - 1 else 1)
            candidate = (added, prior[1] - iou_sum, left_index)
            current = best[right_index]
            if current is None or candidate[:2] < current[:2]:
                best[right_index] = candidate

    if best[-1] is None:
        raise RuntimeError(
            f"no Recall-safe path in Production interval {left.frame}..{right.frame} "
            f"for segment {segment.segment_id}"
        )
    selected_indices = []
    cursor = len(positions) - 1
    while cursor >= 0:
        selected_indices.append(cursor)
        predecessor = best[cursor][2] if best[cursor] is not None else -1
        if predecessor < 0:
            break
        cursor = predecessor
    selected_indices.reverse()
    return (
        [anchors[positions[index]] for index in selected_indices],
        evaluated_edges,
        feasible_edges,
    )


def guard_production_recall(
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
    guard_margin: float = 0.002,
    point_count: int = 23,
    max_anchor_scale: float = 1.50,
) -> ProductionRecallGuardResult:
    """Add a hard dense Recall guard while retaining Production's structure."""

    if not 0.0 < recall_floor <= 1.0:
        raise ValueError("recall_floor must be in (0, 1]")
    internal_floor = min(0.9999, float(recall_floor) + max(0.0, guard_margin))
    started = time.perf_counter()
    output: dict[str, list[Segment]] = {}
    adjusted_total = 0
    added_total = 0
    edge_total = 0
    feasible_total = 0
    diagnostics: list[dict[str, object]] = []

    for track_id, values in segments.items():
        guarded_values: list[Segment] = []
        for segment in values:
            segment_raw = {
                frame: raw
                for (frame, candidate_track), raw in raw_masks.items()
                if candidate_track == track_id
                and segment.first_frame <= frame <= segment.last_frame
                and start_frame <= frame <= end_frame
            }
            if not segment_raw or not segment.keyframes:
                guarded_values.append(segment)
                continue

            production_by_frame = {
                keyframe.frame: keyframe for keyframe in segment.keyframes
            }
            anchor_cache: dict[int, Keyframe] = {}
            adjusted_frames: list[int] = []
            for frame, raw in segment_raw.items():
                production_key = production_by_frame.get(frame)
                anchor = _guard_anchor(
                    segment,
                    raw,
                    recall_floor=internal_floor,
                    point_count=point_count,
                    max_anchor_scale=max_anchor_scale,
                    production_key=production_key,
                )
                anchor_cache[frame] = anchor
                if production_key is not None and anchor != production_key:
                    adjusted_frames.append(frame)

            guarded_production = [
                anchor_cache.get(keyframe.frame, keyframe)
                for keyframe in segment.keyframes
            ]
            selected: list[Keyframe] = [guarded_production[0]]
            segment_edges = 0
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
                interval_keys, evaluated, feasible = _guard_production_interval(
                    segment,
                    left,
                    right,
                    interval_raw,
                    anchor_cache,
                    recall_floor=internal_floor,
                )
                selected.extend(interval_keys[1:])
                segment_edges += evaluated
                segment_feasible += feasible

            guarded = _replace_segment_keys(segment, selected)
            original_frames = set(production_by_frame)
            added_frames = sorted(
                keyframe.frame
                for keyframe in guarded.keyframes
                if keyframe.frame not in original_frames
            )
            adjusted_total += len(adjusted_frames)
            added_total += len(added_frames)
            edge_total += segment_edges
            feasible_total += segment_feasible
            diagnostics.append(
                {
                    "track_id": track_id,
                    "segment_id": segment.segment_id,
                    "production_key_count": len(segment.keyframes),
                    "guarded_key_count": len(guarded.keyframes),
                    "adjusted_production_key_count": len(adjusted_frames),
                    "adjusted_production_frames": adjusted_frames,
                    "added_recall_key_count": len(added_frames),
                    "added_recall_frames": added_frames,
                    "evaluated_edges": segment_edges,
                    "feasible_edges": segment_feasible,
                }
            )
            guarded_values.append(guarded)
        output[track_id] = guarded_values

    return ProductionRecallGuardResult(
        segments=output,
        adjusted_production_keys=adjusted_total,
        added_recall_keys=added_total,
        evaluated_edges=edge_total,
        feasible_edges=feasible_total,
        elapsed_seconds=time.perf_counter() - started,
        segment_diagnostics=tuple(diagnostics),
    )
