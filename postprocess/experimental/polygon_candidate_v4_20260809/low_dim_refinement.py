"""Bounded affine and low-frequency contour refinement for DP anchor states."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from experimental.alternating_temporal_pareto.refinement import (
    _candidate_segment,
    _keyframe_points,
    _local_loss,
    _segment_for_frame,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    RawMask,
    Segment,
    _raw_keyframe,
)
from experimental.polygon_recall_optimizer.pareto_dp import _keyframe_geometry
from experimental.polygon_recall_optimizer.superior import BorderFrameConstraint
from overlay_renderer.keyframe_cache import Component, Keyframe


@dataclass(frozen=True)
class LowDimCandidate:
    model: str
    keyframe: Keyframe
    loss: float
    baseline_loss: float
    evaluations: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class LowDimRefinementResult:
    extra_states: dict[tuple[int, str], tuple[Keyframe, ...]]
    states_by_model: dict[str, dict[tuple[int, str], tuple[Keyframe, ...]]]
    records: tuple[dict[str, object], ...]
    elapsed_seconds: float
    objective_evaluations: int
    accepted_states: int


@dataclass(frozen=True)
class SequentialRefinementResult:
    segments: dict[str, list[Segment]]
    records: tuple[dict[str, object], ...]
    elapsed_seconds: float
    accepted_keys: int


def _outward_normals(points: np.ndarray) -> np.ndarray:
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    length = np.linalg.norm(tangent, axis=1)
    tangent = np.divide(
        tangent,
        np.maximum(length[:, None], 1e-9),
        out=np.zeros_like(tangent),
        where=length[:, None] > 1e-9,
    )
    signed_twice_area = float(
        np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    )
    if signed_twice_area >= 0.0:  # CCW: exterior is the right-hand normal.
        return np.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    return np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)


def _periodic_offsets(control: np.ndarray, point_count: int) -> np.ndarray:
    positions = np.arange(point_count, dtype=np.float64)
    control_positions = np.linspace(
        0.0, float(point_count), len(control) + 1, dtype=np.float64
    )
    values = np.concatenate((control, control[:1]))
    return np.interp(positions, control_positions, values)


def transform_low_dim(
    seed: Keyframe,
    parameters: np.ndarray,
    *,
    normal_control_count: int,
) -> Keyframe:
    """Apply 6-DoF affine plus smooth normal offsets to one polygon."""

    points = _keyframe_points(seed)
    if len(points) < 3:
        return seed
    values = np.asarray(parameters, dtype=np.float64)
    expected = 6 + int(normal_control_count)
    if len(values) != expected:
        raise ValueError(f"expected {expected} parameters, got {len(values)}")
    center = np.mean(points, axis=0)
    scale = math.sqrt(max(float(_keyframe_geometry(seed).area), 1.0))
    dx, dy, angle, log_sx, log_sy, shear = values[:6]
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    affine = rotation @ np.asarray(
        (
            (math.exp(float(log_sx)), float(shear)),
            (0.0, math.exp(float(log_sy))),
        ),
        dtype=np.float64,
    )
    transformed = (points - center) @ affine.T + center
    transformed += scale * np.asarray((dx, dy), dtype=np.float64)
    if normal_control_count:
        normals = _outward_normals(transformed)
        offsets = _periodic_offsets(values[6:], len(transformed))
        transformed = transformed + scale * offsets[:, None] * normals
    return Keyframe(
        int(seed.frame),
        ((0, Component("polygon", transformed.tolist())),),
    )


def _coordinate_optimize(
    segment: Segment,
    seed: Keyframe,
    quality: dict[int, RawMask],
    constraints: dict[int, RawMask],
    borders: dict[int, BorderFrameConstraint],
    *,
    recall_floor: float,
    width: int,
    height: int,
    normal_control_count: int,
    initial_parameters: np.ndarray | None = None,
    rounds: int = 3,
) -> LowDimCandidate | None:
    parameter_count = 6 + int(normal_control_count)
    current = (
        np.zeros(parameter_count, dtype=np.float64)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64).copy()
    )
    lower = np.asarray(
        [-0.08, -0.08, -0.25, -0.16, -0.16, -0.12]
        + [-0.12] * int(normal_control_count),
        dtype=np.float64,
    )
    upper = np.asarray(
        [0.08, 0.08, 0.25, 0.22, 0.22, 0.12]
        + [0.20] * int(normal_control_count),
        dtype=np.float64,
    )
    steps = np.asarray(
        [0.025, 0.025, 0.06, 0.045, 0.045, 0.035]
        + [0.035] * int(normal_control_count),
        dtype=np.float64,
    )

    baseline_loss, baseline_feasible, evaluated = _local_loss(
        segment,
        seed,
        quality,
        constraints,
        borders,
        recall_floor=recall_floor,
        width=width,
        height=height,
        regularization_source=seed,
    )
    total_evaluations = evaluated
    if not baseline_feasible:
        return None
    best_loss = baseline_loss
    best_keyframe = seed
    for _round in range(max(1, int(rounds))):
        for position in range(parameter_count):
            alternatives: list[tuple[float, np.ndarray, Keyframe]] = []
            for direction in (-1.0, 1.0):
                proposed = current.copy()
                proposed[position] = np.clip(
                    proposed[position] + direction * steps[position],
                    lower[position],
                    upper[position],
                )
                if abs(proposed[position] - current[position]) < 1e-12:
                    continue
                candidate = transform_low_dim(
                    seed,
                    proposed,
                    normal_control_count=normal_control_count,
                )
                loss, feasible, count = _local_loss(
                    segment,
                    candidate,
                    quality,
                    constraints,
                    borders,
                    recall_floor=recall_floor,
                    width=width,
                    height=height,
                    regularization_source=seed,
                )
                total_evaluations += count
                if feasible:
                    alternatives.append((loss, proposed, candidate))
            if alternatives:
                loss, proposed, candidate = min(alternatives, key=lambda item: item[0])
                if loss + 1e-8 < best_loss:
                    best_loss = loss
                    current = proposed
                    best_keyframe = candidate
        steps *= 0.5
    if best_loss + 1e-7 >= baseline_loss:
        return None
    return LowDimCandidate(
        model=("affine" if normal_control_count == 0 else f"normal{normal_control_count}"),
        keyframe=best_keyframe,
        loss=float(best_loss),
        baseline_loss=float(baseline_loss),
        evaluations=int(total_evaluations),
        parameters=tuple(float(value) for value in current),
    )


def refine_targets_low_dim(
    segments: dict[str, list[Segment]],
    quality_masks: dict[tuple[int, str], RawMask],
    constraint_masks: dict[tuple[int, str], RawMask],
    border_constraints: dict[tuple[int, str], BorderFrameConstraint],
    targets: list[tuple[int, str]],
    *,
    recall_floor: float,
    width: int,
    height: int,
    normal_control_count: int = 6,
    rounds: int = 3,
) -> LowDimRefinementResult:
    """Optimize selected existing keys with C1 and C2 candidate states."""

    started = time.perf_counter()
    extras: dict[tuple[int, str], tuple[Keyframe, ...]] = {}
    by_model: dict[str, dict[tuple[int, str], tuple[Keyframe, ...]]] = {
        "affine": {},
        f"normal{int(normal_control_count)}": {},
    }
    records: list[dict[str, object]] = []
    total_evaluations = 0
    accepted = 0
    for frame, track_id in dict.fromkeys(targets):
        segment = _segment_for_frame(segments, track_id, frame)
        if segment is None:
            continue
        seed = next((key for key in segment.keyframes if key.frame == frame), None)
        target_kind = "selected_key"
        if seed is None:
            raw_seed = quality_masks.get((frame, track_id))
            if raw_seed is None:
                continue
            seed = _raw_keyframe(raw_seed, point_count=23)
            target_kind = "problem_frame_insertion"
        quality = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in quality_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        constraints = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in constraint_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        borders = {
            candidate_frame: value
            for (candidate_frame, candidate_track), value in border_constraints.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        affine = _coordinate_optimize(
            segment,
            seed,
            quality,
            constraints,
            borders,
            recall_floor=recall_floor,
            width=width,
            height=height,
            normal_control_count=0,
            rounds=rounds,
        )
        initial = np.zeros(6 + int(normal_control_count), dtype=np.float64)
        if affine is not None:
            initial[:6] = np.asarray(affine.parameters, dtype=np.float64)
        normal = _coordinate_optimize(
            segment,
            seed,
            quality,
            constraints,
            borders,
            recall_floor=recall_floor,
            width=width,
            height=height,
            normal_control_count=normal_control_count,
            initial_parameters=initial,
            rounds=rounds,
        )
        candidates = [value for value in (affine, normal) if value is not None]
        total_evaluations += sum(value.evaluations for value in candidates)
        if not candidates:
            continue
        candidates.sort(key=lambda value: value.loss)
        extras[(frame, track_id)] = tuple(value.keyframe for value in candidates)
        for value in candidates:
            by_model[value.model][(frame, track_id)] = (value.keyframe,)
        accepted += len(candidates)
        records.append(
            {
                "frame": int(frame),
                "track_id": track_id,
                "target_kind": target_kind,
                "states": [
                    {
                        "model": value.model,
                        "baseline_loss": value.baseline_loss,
                        "loss": value.loss,
                        "gain": value.baseline_loss - value.loss,
                        "evaluations": value.evaluations,
                        "parameters": list(value.parameters),
                    }
                    for value in candidates
                ],
            }
        )
    return LowDimRefinementResult(
        extra_states=extras,
        states_by_model=by_model,
        records=tuple(records),
        elapsed_seconds=time.perf_counter() - started,
        objective_evaluations=total_evaluations,
        accepted_states=accepted,
    )


def refine_path_sequential(
    segments: dict[str, list[Segment]],
    quality_masks: dict[tuple[int, str], RawMask],
    constraint_masks: dict[tuple[int, str], RawMask],
    border_constraints: dict[tuple[int, str], BorderFrameConstraint],
    targets: list[tuple[int, str]],
    *,
    recall_floor: float,
    width: int,
    height: int,
    normal_control_count: int = 6,
    rounds: int = 3,
) -> SequentialRefinementResult:
    """Greedily refine fixed key positions, preserving exact local constraints."""

    started = time.perf_counter()
    current = {track_id: list(values) for track_id, values in segments.items()}
    records: list[dict[str, object]] = []
    accepted = 0
    for frame, track_id in dict.fromkeys(targets):
        segment = _segment_for_frame(current, track_id, frame)
        if segment is None:
            continue
        seed = next((key for key in segment.keyframes if key.frame == frame), None)
        if seed is None:
            continue
        quality = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in quality_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        constraints = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in constraint_masks.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        borders = {
            candidate_frame: value
            for (candidate_frame, candidate_track), value in border_constraints.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        affine = _coordinate_optimize(
            segment,
            seed,
            quality,
            constraints,
            borders,
            recall_floor=recall_floor,
            width=width,
            height=height,
            normal_control_count=0,
            rounds=rounds,
        )
        initial = np.zeros(6 + int(normal_control_count), dtype=np.float64)
        if affine is not None:
            initial[:6] = np.asarray(affine.parameters, dtype=np.float64)
        normal = _coordinate_optimize(
            segment,
            seed,
            quality,
            constraints,
            borders,
            recall_floor=recall_floor,
            width=width,
            height=height,
            normal_control_count=normal_control_count,
            initial_parameters=initial,
            rounds=rounds,
        )
        options = [value for value in (affine, normal) if value is not None]
        if not options:
            continue
        winner = min(options, key=lambda value: value.loss)
        updated = _candidate_segment(segment, frame, winner.keyframe)
        current[track_id] = [
            updated if value.segment_id == segment.segment_id else value
            for value in current[track_id]
        ]
        accepted += 1
        records.append(
            {
                "frame": int(frame),
                "track_id": track_id,
                "model": winner.model,
                "baseline_loss": winner.baseline_loss,
                "loss": winner.loss,
                "gain": winner.baseline_loss - winner.loss,
                "parameters": list(winner.parameters),
            }
        )
    return SequentialRefinementResult(
        segments=current,
        records=tuple(records),
        elapsed_seconds=time.perf_counter() - started,
        accepted_keys=accepted,
    )


__all__ = [
    "LowDimCandidate",
    "LowDimRefinementResult",
    "SequentialRefinementResult",
    "refine_path_sequential",
    "refine_targets_low_dim",
    "transform_low_dim",
]
