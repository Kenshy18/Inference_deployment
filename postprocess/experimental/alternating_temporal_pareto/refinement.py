"""Path-conditioned continuous shape refinement for the temporal DP."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar
from shapely.geometry import box

from ..polygon_recall_optimizer.fixed_budget import (
    FrameEvaluation,
    Keyframe,
    RawMask,
    Segment,
    _primary_component,
)
from ..polygon_recall_optimizer.pareto_dp import (
    _fast_numpy_align,
    _keyframe_geometry,
    _minimum_border_buffer_anchor,
    _minimum_directional_border_anchor,
    _minimum_feasible_anchor,
    canonicalize_selected_path,
)
from ..polygon_recall_optimizer.superior import (
    BorderFrameConstraint,
    border_geometry_feasible,
    direct_geometry_at,
)
from ..polygon_recall_optimizer.temporal_candidates import (
    build_temporal_candidates,
)
from overlay_renderer.keyframe_cache import Component


@dataclass(frozen=True)
class RefinementResult:
    extra_states: dict[tuple[int, str], tuple[Keyframe, ...]]
    elapsed_seconds: float
    selected_key_targets: int
    problem_frame_targets: int
    optimized_targets: int
    objective_evaluations: int
    accepted_states: int
    baseline_loss_sum: float
    refined_loss_sum: float
    records: tuple[dict[str, object], ...]


def _segment_for_frame(
    segments: dict[str, list[Segment]], track_id: str, frame: int
) -> Segment | None:
    return next(
        (
            segment
            for segment in segments.get(track_id, ())
            if segment.first_frame <= frame <= segment.last_frame
        ),
        None,
    )


def _keyframe_points(keyframe: Keyframe) -> np.ndarray:
    component = _primary_component(keyframe)
    if component is None:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(component.values, dtype=np.float64)


def _blend_keyframe(
    frame: int, left: Keyframe, right: Keyframe, alpha: float
) -> Keyframe:
    left_points = _keyframe_points(left)
    right_points = _fast_numpy_align(left_points, _keyframe_points(right))
    points = (1.0 - float(alpha)) * left_points + float(alpha) * right_points
    return Keyframe(
        int(frame), ((0, Component("polygon", points.tolist())),)
    )


def _candidate_segment(
    segment: Segment, target_frame: int, candidate: Keyframe
) -> Segment:
    keys = [key for key in segment.keyframes if key.frame != target_frame]
    keys.append(candidate)
    keys.sort(key=lambda item: item.frame)
    return replace(
        segment,
        interpolation_method="linear_polygon_index_v1",
        keyframes=canonicalize_selected_path(tuple(keys)),
    )


def _affected_bounds(segment: Segment, target_frame: int) -> tuple[int, int]:
    frames = [key.frame for key in segment.keyframes]
    left = max((frame for frame in frames if frame < target_frame), default=frames[0])
    right = min((frame for frame in frames if frame > target_frame), default=frames[-1])
    return int(left), int(right)


def _local_loss(
    segment: Segment,
    candidate: Keyframe,
    quality_by_frame: dict[int, RawMask],
    constraint_by_frame: dict[int, RawMask],
    border_by_frame: dict[int, BorderFrameConstraint],
    *,
    recall_floor: float,
    width: int,
    height: int,
    regularization_source: Keyframe,
) -> tuple[float, bool, int]:
    temporary = _candidate_segment(segment, candidate.frame, candidate)
    lower, upper = _affected_bounds(segment, candidate.frame)
    visible = box(0.0, 0.0, float(width), float(height))
    losses: list[float] = []
    excess: list[float] = []
    area_ratios: list[float] = []
    evaluated = 0
    for frame in sorted(quality_by_frame):
        if frame < lower or frame > upper:
            continue
        predicted = direct_geometry_at(temporary, frame)
        if predicted.is_empty:
            return 1e12, False, evaluated
        quality = quality_by_frame[frame]
        constraint = constraint_by_frame[frame]
        quality_area = float(quality.geometry.area)
        constraint_area = float(constraint.geometry.area)
        quality_intersection = float(quality.geometry.intersection(predicted).area)
        constraint_intersection = float(
            constraint.geometry.intersection(predicted).area
        )
        recall = min(
            quality_intersection / quality_area if quality_area else 1.0,
            constraint_intersection / constraint_area if constraint_area else 1.0,
        )
        current_border = border_by_frame.get(frame)
        if recall + 1e-12 < recall_floor or not border_geometry_feasible(
            predicted, current_border
        ):
            return 1e9 + (recall_floor - recall) * 1e6, False, evaluated
        metric_predicted = predicted.intersection(visible)
        metric_quality = quality.geometry
        if current_border is not None:
            metric_predicted = metric_predicted.intersection(
                current_border.quality_domain
            )
            metric_quality = metric_quality.intersection(
                current_border.quality_domain
            )
        intersection = float(metric_quality.intersection(metric_predicted).area)
        predicted_area = float(metric_predicted.area)
        metric_quality_area = float(metric_quality.area)
        union = metric_quality_area + predicted_area - intersection
        iou = intersection / union if union else 1.0
        losses.append(1.0 - iou)
        excess.append(
            float(metric_predicted.difference(metric_quality).area)
            / max(metric_quality_area, 1e-9)
        )
        area_ratios.append(predicted_area / max(metric_quality_area, 1e-9))
        evaluated += 1
    if not losses:
        return 1e12, False, evaluated
    loss_values = np.asarray(losses, dtype=np.float64)
    tail_count = max(1, int(math.ceil(0.20 * len(loss_values))))
    tail = float(np.mean(np.sort(loss_values)[-tail_count:]))
    excess_values = np.asarray(excess, dtype=np.float64)
    area_values = np.asarray(area_ratios, dtype=np.float64)
    loss = (
        float(np.mean(loss_values))
        + 0.75 * tail
        + 0.20 * float(np.quantile(excess_values, 0.95))
        + 0.30 * max(0.0, float(np.quantile(area_values, 0.95)) - 1.50) ** 2
        + 0.15 * max(0.0, float(np.max(area_values)) - 2.00) ** 2
    )
    source_points = _keyframe_points(regularization_source)
    candidate_points = _fast_numpy_align(source_points, _keyframe_points(candidate))
    scale = math.sqrt(
        max(float(_keyframe_geometry(regularization_source).area), 1.0)
    )
    displacement = float(
        np.mean(np.linalg.norm(candidate_points - source_points, axis=1))
        / scale
    )
    loss += 0.02 * displacement * displacement
    return loss, True, evaluated


def _repair_source(
    source: Keyframe,
    constraint: RawMask,
    quality: RawMask,
    border: BorderFrameConstraint | None,
    *,
    recall_floor: float,
    max_anchor_scale: float,
    visible,
) -> Keyframe | None:
    candidate = _minimum_feasible_anchor(
        source,
        constraint,
        quality_raw=quality,
        recall_floor=recall_floor,
        max_anchor_scale=max_anchor_scale,
        border_constraint=border,
        visible_rectangle=visible,
    )
    if candidate is None and border is not None:
        candidate = _minimum_directional_border_anchor(
            source,
            constraint,
            quality,
            recall_floor,
            border,
            visible,
            max_anchor_scale=max_anchor_scale,
        )
    if candidate is None and border is not None:
        candidate = _minimum_border_buffer_anchor(
            source,
            constraint,
            quality,
            recall_floor,
            border,
            visible,
        )
    return candidate


def _local_metrics(
    segment: Segment,
    candidate: Keyframe,
    quality_by_frame: dict[int, RawMask],
    border_by_frame: dict[int, BorderFrameConstraint],
    *,
    width: int,
    height: int,
) -> dict[str, object]:
    """Exact local diagnostics saved only for accepted refinements."""

    temporary = _candidate_segment(segment, candidate.frame, candidate)
    lower, upper = _affected_bounds(segment, candidate.frame)
    visible = box(0.0, 0.0, float(width), float(height))
    recalls: list[float] = []
    ious: list[float] = []
    area_ratios: list[float] = []
    frames: list[int] = []
    for frame in sorted(quality_by_frame):
        if frame < lower or frame > upper:
            continue
        predicted = direct_geometry_at(temporary, frame)
        quality = quality_by_frame[frame]
        quality_area = float(quality.geometry.area)
        intersection = float(quality.geometry.intersection(predicted).area)
        recalls.append(intersection / quality_area if quality_area else 1.0)
        current_border = border_by_frame.get(frame)
        metric_predicted = predicted.intersection(visible)
        metric_quality = quality.geometry
        if current_border is not None:
            metric_predicted = metric_predicted.intersection(
                current_border.quality_domain
            )
            metric_quality = metric_quality.intersection(
                current_border.quality_domain
            )
        metric_intersection = float(
            metric_quality.intersection(metric_predicted).area
        )
        predicted_area = float(metric_predicted.area)
        metric_quality_area = float(metric_quality.area)
        union = metric_quality_area + predicted_area - metric_intersection
        ious.append(metric_intersection / union if union else 1.0)
        area_ratios.append(predicted_area / max(metric_quality_area, 1e-9))
        frames.append(int(frame))
    return {
        "affected_first_frame": int(lower),
        "affected_last_frame": int(upper),
        "evaluated_frames": len(frames),
        "recall_min": float(np.min(recalls)),
        "recall_mean": float(np.mean(recalls)),
        "iou_mean": float(np.mean(ious)),
        "iou_q05": float(np.quantile(ious, 0.05)),
        "iou_min": float(np.min(ious)),
        "area_ratio_mean": float(np.mean(area_ratios)),
        "area_ratio_q95": float(np.quantile(area_ratios, 0.95)),
        "area_ratio_max": float(np.max(area_ratios)),
    }


def _problem_frames(
    rows: list[FrameEvaluation],
    segments: dict[str, list[Segment]],
    *,
    limit: int,
) -> list[tuple[int, str]]:
    if not rows or limit <= 0:
        return []
    ious = np.asarray([row.iou for row in rows], dtype=np.float64)
    areas = np.asarray([row.area_ratio for row in rows], dtype=np.float64)
    iou_cutoff = float(np.quantile(ious, 0.10))
    area_cutoff = max(1.50, float(np.quantile(areas, 0.95)))
    candidates = [
        row
        for row in rows
        if row.iou <= iou_cutoff or row.area_ratio >= area_cutoff
    ]
    candidates.sort(
        key=lambda row: ((1.0 - row.iou) + 0.50 * max(0.0, row.area_ratio - 1.0)),
        reverse=True,
    )
    selected: list[tuple[int, str]] = []
    used_intervals: set[tuple[str, int, int]] = set()
    for row in candidates:
        segment = _segment_for_frame(segments, row.track_id, row.frame)
        if segment is None:
            continue
        keys = [key.frame for key in segment.keyframes]
        left = max((frame for frame in keys if frame <= row.frame), default=keys[0])
        right = min((frame for frame in keys if frame >= row.frame), default=keys[-1])
        identity = (row.track_id, int(left), int(right))
        if identity in used_intervals or row.frame in {left, right}:
            continue
        used_intervals.add(identity)
        selected.append((row.frame, row.track_id))
        if len(selected) >= limit:
            break
    return selected


def refine_selected_path(
    segments: dict[str, list[Segment]],
    quality_masks: dict[tuple[int, str], RawMask],
    constraint_masks: dict[tuple[int, str], RawMask],
    border_constraints: dict[tuple[int, str], BorderFrameConstraint],
    evaluations: list[FrameEvaluation],
    *,
    recall_floor: float,
    point_count: int,
    window_radii: tuple[int, int, int],
    recall_quantile: float,
    max_anchor_scale: float,
    width: int,
    height: int,
    max_problem_frames: int = 96,
    max_selected_keys: int = 0,
) -> RefinementResult:
    """Refine selected keys and comparable low-quality insertion positions."""

    started = time.perf_counter()
    selected_targets = [
        (key.frame, track_id)
        for track_id, values in sorted(segments.items())
        for segment in values
        for key in segment.keyframes
    ]
    if max_selected_keys > 0 and len(selected_targets) > max_selected_keys:
        stride = max(1, len(selected_targets) // max_selected_keys)
        selected_targets = selected_targets[::stride][:max_selected_keys]
    problem_targets = _problem_frames(
        evaluations, segments, limit=max_problem_frames
    )
    targets = list(dict.fromkeys(selected_targets + problem_targets))
    visible = box(0.0, 0.0, float(width), float(height))
    extras: dict[tuple[int, str], tuple[Keyframe, ...]] = {}
    objective_evaluations = 0
    accepted = 0
    baseline_loss_sum = 0.0
    refined_loss_sum = 0.0
    optimized_targets = 0
    records: list[dict[str, object]] = []
    selected_target_set = set(selected_targets)
    for frame, track_id in targets:
        segment = _segment_for_frame(segments, track_id, frame)
        quality = quality_masks.get((frame, track_id))
        constraint = constraint_masks.get((frame, track_id))
        if segment is None or quality is None or constraint is None:
            continue
        segment_quality = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in quality_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        segment_constraints = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in constraint_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        segment_borders = {
            candidate_frame: value
            for (candidate_frame, candidate_track), value in border_constraints.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        current = next(
            (key for key in segment.keyframes if key.frame == frame), None
        )
        temporal = build_temporal_candidates(
            frame,
            segment_quality,
            point_count=point_count,
            window_radii=window_radii,
            recall_quantile=recall_quantile,
        )
        repaired = [
            candidate
            for value in temporal
            if (
                candidate := _repair_source(
                    value.keyframe,
                    constraint,
                    quality,
                    segment_borders.get(frame),
                    recall_floor=recall_floor,
                    max_anchor_scale=max_anchor_scale,
                    visible=visible,
                )
            )
            is not None
        ]
        if not repaired:
            continue
        seed = current or repaired[0]
        seed_loss, seed_feasible, evaluated = _local_loss(
            segment,
            seed,
            segment_quality,
            segment_constraints,
            segment_borders,
            recall_floor=recall_floor,
            width=width,
            height=height,
            regularization_source=seed,
        )
        objective_evaluations += evaluated
        if not seed_feasible:
            seed = min(
                repaired,
                key=lambda candidate: _local_loss(
                    segment,
                    candidate,
                    segment_quality,
                    segment_constraints,
                    segment_borders,
                    recall_floor=recall_floor,
                    width=width,
                    height=height,
                    regularization_source=candidate,
                )[0],
            )
            seed_loss, seed_feasible, evaluated = _local_loss(
                segment,
                seed,
                segment_quality,
                segment_constraints,
                segment_borders,
                recall_floor=recall_floor,
                width=width,
                height=height,
                regularization_source=seed,
            )
            objective_evaluations += evaluated
        if not seed_feasible:
            continue
        candidates: list[tuple[float, Keyframe]] = []
        for alternate in repaired:
            if _keyframe_geometry(seed).intersection(
                _keyframe_geometry(alternate)
            ).area / max(
                _keyframe_geometry(seed).union(_keyframe_geometry(alternate)).area,
                1e-9,
            ) >= 0.9999:
                continue

            def objective(alpha: float) -> float:
                nonlocal objective_evaluations
                blended = _blend_keyframe(frame, seed, alternate, alpha)
                loss, _feasible, evaluated_count = _local_loss(
                    segment,
                    blended,
                    segment_quality,
                    segment_constraints,
                    segment_borders,
                    recall_floor=recall_floor,
                    width=width,
                    height=height,
                    regularization_source=seed,
                )
                objective_evaluations += evaluated_count
                return loss

            result = minimize_scalar(
                objective,
                bounds=(0.0, 1.0),
                method="bounded",
                options={"maxiter": 18, "xatol": 2e-3},
            )
            blended = _blend_keyframe(frame, seed, alternate, float(result.x))
            loss, feasible, evaluated = _local_loss(
                segment,
                blended,
                segment_quality,
                segment_constraints,
                segment_borders,
                recall_floor=recall_floor,
                width=width,
                height=height,
                regularization_source=seed,
            )
            objective_evaluations += evaluated
            if feasible and loss + 1e-8 < seed_loss:
                candidates.append((loss, blended))
        candidates.sort(key=lambda item: item[0])
        unique: list[tuple[float, Keyframe]] = []
        for item in candidates:
            geometry = _keyframe_geometry(item[1])
            if any(
                geometry.intersection(_keyframe_geometry(prior[1])).area
                / max(geometry.union(_keyframe_geometry(prior[1])).area, 1e-9)
                >= 0.9999
                for prior in unique
            ):
                continue
            unique.append(item)
            if len(unique) >= 2:
                break
        if unique:
            extras[(frame, track_id)] = tuple(item[1] for item in unique)
            optimized_targets += 1
            accepted += len(unique)
            baseline_loss_sum += seed_loss
            refined_loss_sum += unique[0][0]
            records.append(
                {
                    "frame": int(frame),
                    "track_id": str(track_id),
                    "target_kind": (
                        "selected_key"
                        if (frame, track_id) in selected_target_set
                        else "problem_frame_insertion"
                    ),
                    "accepted_states": len(unique),
                    "baseline_loss": float(seed_loss),
                    "refined_loss": float(unique[0][0]),
                    "baseline": _local_metrics(
                        segment,
                        seed,
                        segment_quality,
                        segment_borders,
                        width=width,
                        height=height,
                    ),
                    "refined": _local_metrics(
                        segment,
                        unique[0][1],
                        segment_quality,
                        segment_borders,
                        width=width,
                        height=height,
                    ),
                }
            )
    return RefinementResult(
        extra_states=extras,
        elapsed_seconds=time.perf_counter() - started,
        selected_key_targets=len(selected_targets),
        problem_frame_targets=len(problem_targets),
        optimized_targets=optimized_targets,
        objective_evaluations=objective_evaluations,
        accepted_states=accepted,
        baseline_loss_sum=baseline_loss_sum,
        refined_loss_sum=refined_loss_sum,
        records=tuple(records),
    )


__all__ = ["RefinementResult", "refine_selected_path"]
