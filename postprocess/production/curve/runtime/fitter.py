"""Track-wise fitting of persistent Catmull--Rom interpolation points.

The fitted variables are persistent contour locations and per-frame isotropic
repair scales.  Cubic handles are never optimization variables.  One contour
location therefore keeps the same semantic identity throughout a track.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

import numpy as np

from production.polygon.runtime.spatial_support.optimizer import (
    align_temporal_dense,
    orient_ccw,
    temporal_residuals,
)

from .curve_fit import WholeCurveFitStats, fit_whole_curve_controls
from .metrics import CurveMetrics, frame_raster_metrics, sequence_raster_metrics
from .model import bezier_segments, sample_closed_curve, sample_curve_sequence
from .native_cpu import ExactDoubleRasterBatch, create_exact_raster_batch
from .topology import has_strict_self_intersection, strict_self_intersection_batch


@dataclass(frozen=True, slots=True)
class FitConfig:
    control_point_count: int = 8
    dense_contour_samples: int = 192
    samples_per_segment: int = 16
    recall_floor: float = 0.97
    index_refine_passes: int = 2
    index_refine_radius: int = 8
    proxy_max_frames: int = 32
    tail_weight: float = 0.35
    recall_penalty: float = 30.0
    temporal_weight: float = 0.05
    scale_maximum: float = 1.08
    scale_step: float = 0.002
    scale_size_penalty: float = 0.02
    scale_transition_weight: float = 8.0
    whole_curve_fit_enabled: bool = True
    whole_curve_anchor_weight: float = 1.0
    whole_curve_temporal_weight: float = 2.0
    whole_curve_maximum_correction_fraction: float = 0.40
    whole_curve_blends: tuple[float, ...] = (0.50, 0.75, 1.0)
    temporal_phase_stabilization_enabled: bool = True
    temporal_phase_jump_ratio: float = 8.0
    temporal_phase_jump_error: float = 0.50
    native_cpu_batches: bool = True
    native_cpu_threads: int = 8
    native_batch_cases: int = 4096
    native_reference_cache_bytes: int = 256 * 1024 * 1024

    def validate(self) -> None:
        if int(self.control_point_count) < 3:
            raise ValueError("control_point_count must be at least three")
        if int(self.dense_contour_samples) < int(self.control_point_count) * 3:
            raise ValueError("dense_contour_samples must be at least 3x control points")
        if int(self.samples_per_segment) < 2:
            raise ValueError("samples_per_segment must be at least two")
        if not 0.0 < float(self.recall_floor) <= 1.0:
            raise ValueError("recall_floor must be in (0, 1]")
        if float(self.scale_maximum) < 1.0 or float(self.scale_step) <= 0.0:
            raise ValueError("invalid scale search interval")
        if float(self.whole_curve_anchor_weight) < 0.0:
            raise ValueError("whole_curve_anchor_weight must be nonnegative")
        if float(self.whole_curve_temporal_weight) < 0.0:
            raise ValueError("whole_curve_temporal_weight must be nonnegative")
        if float(self.whole_curve_maximum_correction_fraction) < 0.0:
            raise ValueError(
                "whole_curve_maximum_correction_fraction must be nonnegative"
            )
        if any(not 0.0 < float(value) <= 1.0 for value in self.whole_curve_blends):
            raise ValueError("whole_curve_blends must contain values in (0, 1]")
        if float(self.temporal_phase_jump_ratio) <= 1.0:
            raise ValueError("temporal_phase_jump_ratio must be greater than one")
        if float(self.temporal_phase_jump_error) <= 0.0:
            raise ValueError("temporal_phase_jump_error must be positive")
        if int(self.native_cpu_threads) < 1:
            raise ValueError("native_cpu_threads must be positive")
        if int(self.native_reference_cache_bytes) < 0:
            raise ValueError("native_reference_cache_bytes must be non-negative")
        if int(self.native_batch_cases) < 1:
            raise ValueError("native_batch_cases must be positive")


@dataclass(frozen=True, slots=True)
class SequenceFitResult:
    controls: np.ndarray
    segments: np.ndarray
    sampled_curves: np.ndarray
    persistent_dense_indices: tuple[int, ...]
    repair_scales: np.ndarray
    metrics: CurveMetrics
    initial_metrics: CurveMetrics
    unresolved_recall_frames: tuple[int, ...]
    objective_before: float
    objective_after: float
    elapsed_seconds: float
    exact_frame_evaluations: int
    whole_curve_fit_applied: bool
    whole_curve_selected_blend: float
    whole_curve_stats: WholeCurveFitStats | None
    temporal_phase_shifts: tuple[int, ...]
    native_reference_cache: dict[str, int]
    config: FitConfig

    def summary(self) -> dict[str, object]:
        return {
            "algorithm": "closed_uniform_catmull_rom_tension_1_bezier_factor_1_over_6",
            "editable_variables": "interpolation_points_P_only",
            "derived_handles": True,
            "config": asdict(self.config),
            "frames": int(len(self.controls)),
            "persistent_dense_indices": list(self.persistent_dense_indices),
            "metrics": asdict(self.metrics),
            "initial_metrics": asdict(self.initial_metrics),
            "unresolved_recall_frames": list(self.unresolved_recall_frames),
            "objective_before": float(self.objective_before),
            "objective_after": float(self.objective_after),
            "elapsed_seconds": float(self.elapsed_seconds),
            "exact_frame_evaluations": int(self.exact_frame_evaluations),
            "whole_curve_fit_applied": bool(self.whole_curve_fit_applied),
            "whole_curve_selected_blend": float(self.whole_curve_selected_blend),
            "whole_curve_stats": (
                asdict(self.whole_curve_stats)
                if self.whole_curve_stats is not None
                else None
            ),
            "temporal_phase_shifts": list(self.temporal_phase_shifts),
            "native_reference_cache": dict(self.native_reference_cache),
        }


def _representative_frames(frame_count: int, maximum: int) -> np.ndarray:
    count = min(int(frame_count), max(1, int(maximum)))
    return np.unique(np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int32))


def _stabilize_temporal_phase(
    sampled: np.ndarray,
    *,
    jump_ratio: float = 8.0,
    jump_error: float = 0.50,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Remove cyclic point-number jumps without changing any frame geometry.

    Procrustes phase matching intentionally removes rotation before comparing
    shapes.  That is useful for shape statistics, but a near-symmetric object
    can then choose a phase half a contour away in one frame.  Curve geometry
    is invariant to a cyclic roll of P; temporal interpolation is not.  This
    second pass therefore keeps orientation in the comparison, removes only
    translation, and selects the closest cyclic roll from a central gauge.
    """
    source = np.asarray(sampled, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 2:
        raise ValueError("sampled contours must have shape (frames, points, 2)")
    if len(source) <= 1:
        return np.ascontiguousarray(source), tuple(0 for _frame in source)
    count = int(source.shape[1])
    phase_indices = (
        np.arange(count, dtype=np.intp)[:, None]
        + np.arange(count, dtype=np.intp)[None, :]
    ) % count
    center = len(source) // 2
    output = np.empty_like(source)
    shifts = np.zeros((len(source),), dtype=np.int32)
    output[center] = source[center]

    def align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, int]:
        reference_centered = reference - np.mean(reference, axis=0, keepdims=True)
        candidate_centered = candidate - np.mean(candidate, axis=0, keepdims=True)
        scale = max(
            float(np.mean(np.sum(reference_centered * reference_centered, axis=1))),
            1e-12,
        )
        direct_delta = candidate_centered - reference_centered
        direct_cost = float(
            np.mean(np.sum(direct_delta * direct_delta, axis=1)) / scale
        )
        # The full N-way cyclic search cannot change the result below the
        # absolute jump gate.  Normal video motion takes this exact fast path;
        # ambiguous flips still run the original exhaustive comparison.
        if direct_cost < float(jump_error):
            return np.asarray(candidate, dtype=np.float64), 0
        variants = candidate[phase_indices]
        variants_centered = variants - np.mean(variants, axis=1, keepdims=True)
        delta = variants_centered - reference_centered[None, :, :]
        costs = np.mean(np.sum(delta * delta, axis=2), axis=1) / scale
        best = int(np.argmin(costs))
        # Preserve the shape-only phase in normal motion.  Re-phase only a
        # discontinuity whose direct correspondence is both absolutely bad
        # and many times worse than another cyclic roll.  This avoids tuning
        # ordinary rotations while catching the half-contour flips that make
        # linear P interpolation collapse.
        shift = (
            best
            if best != 0
            and float(costs[0]) >= float(jump_error)
            and float(costs[0]) >= float(jump_ratio) * max(float(costs[best]), 1e-12)
            else 0
        )
        return np.asarray(variants[shift], dtype=np.float64), shift

    for frame in range(center + 1, len(source)):
        output[frame], shifts[frame] = align(output[frame - 1], source[frame])
    for frame in range(center - 1, -1, -1):
        output[frame], shifts[frame] = align(output[frame + 1], source[frame])
    return np.ascontiguousarray(output), tuple(int(value) for value in shifts)


def _normalized_curvature_saliency(dense: np.ndarray, stride: int) -> np.ndarray:
    value = np.asarray(dense, dtype=np.float64)
    previous = np.roll(value, int(stride), axis=1)
    following = np.roll(value, -int(stride), axis=1)
    left = value - previous
    right = following - value
    left /= np.maximum(np.linalg.norm(left, axis=2, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=2, keepdims=True), 1e-12)
    turn = np.arccos(np.clip(np.sum(left * right, axis=2), -1.0, 1.0))
    chord = following - previous
    denominator = np.maximum(np.sum(chord * chord, axis=2), 1e-12)
    alpha = np.clip(np.sum((value - previous) * chord, axis=2) / denominator, 0.0, 1.0)
    projection = previous + alpha[:, :, None] * chord
    deviation = np.linalg.norm(value - projection, axis=2)
    scale = np.sqrt(
        np.maximum(
            np.mean(
                np.sum(
                    (value - np.mean(value, axis=1, keepdims=True)) ** 2,
                    axis=2,
                ),
                axis=1,
            ),
            1.0,
        )
    )
    normalized_deviation = deviation / scale[:, None]
    combined = turn + 2.0 * normalized_deviation
    return np.median(combined, axis=0) + 0.5 * np.quantile(combined, 0.90, axis=0)


def _circular_distance(
    indices: np.ndarray, selected: list[int], count: int
) -> np.ndarray:
    if not selected:
        return np.full_like(indices, float(count), dtype=np.float64)
    output = np.full((len(indices),), float(count), dtype=np.float64)
    for value in selected:
        direct = np.abs(indices - int(value))
        output = np.minimum(output, np.minimum(direct, count - direct))
    return output


def _saliency_indices(dense: np.ndarray, control_points: int) -> np.ndarray:
    count = int(dense.shape[1])
    stride = max(1, count // (2 * int(control_points)))
    saliency = _normalized_curvature_saliency(dense, stride)
    candidates = np.arange(count, dtype=np.int32)
    selected = [int(np.argmax(saliency))]
    minimum_spacing = max(2, count // (int(control_points) * 3))
    while len(selected) < int(control_points):
        distance = _circular_distance(candidates, selected, count)
        eligible = distance >= float(minimum_spacing)
        if not np.any(eligible):
            eligible = distance > 0.0
        score = (saliency + 0.05) * np.maximum(distance, 1.0)
        score[~eligible] = -np.inf
        selected.append(int(np.argmax(score)))
    return np.asarray(sorted(selected), dtype=np.int32)


def _uniform_indices(count: int, control_points: int, phase: int = 0) -> np.ndarray:
    values = np.floor(
        np.arange(control_points, dtype=np.float64)
        * float(count)
        / float(control_points)
    ).astype(np.int32)
    return np.asarray((values + int(phase)) % int(count), dtype=np.int32)


def _roll_gauge(
    dense: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(sorted(set(int(value) for value in indices)), dtype=np.int32)
    start = int(ordered[0])
    rolled = np.roll(dense, -start, axis=1)
    shifted = np.asarray(sorted((ordered - start) % dense.shape[1]), dtype=np.int32)
    return rolled, shifted


def _proxy_objective(
    references: list[np.ndarray],
    controls: np.ndarray,
    config: FitConfig,
    frame_indices: np.ndarray,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> float:
    curves = sample_curve_sequence(controls, config.samples_per_segment)
    selected = np.ascontiguousarray(curves[frame_indices], dtype=np.float64)
    if exact_raster is not None:
        invalid = strict_self_intersection_batch(
            selected,
            threads=int(config.native_cpu_threads),
        )
        if np.any(invalid):
            return float("inf")
        metrics = exact_raster.metrics(
            np.asarray(frame_indices, dtype=np.int32),
            selected,
            threads=int(config.native_cpu_threads),
        )
        iou_values = np.ascontiguousarray(metrics[:, 6], dtype=np.float64)
        recall_values = np.ascontiguousarray(metrics[:, 4], dtype=np.float64)
        counter[0] += len(selected)
    else:
        ious = []
        recalls = []
        for frame in frame_indices:
            curve = curves[int(frame)]
            if has_strict_self_intersection(curve):
                return float("inf")
            iou, recall = frame_raster_metrics(references[int(frame)], curve)
            counter[0] += 1
            ious.append(iou)
            recalls.append(recall)
        iou_values = np.asarray(ious, dtype=np.float64)
        recall_values = np.asarray(recalls, dtype=np.float64)
    deficits = np.maximum(float(config.recall_floor) - recall_values, 0.0)
    residual = temporal_residuals(controls)
    temporal = float(np.mean(residual)) if residual.size else 0.0
    return float(
        1.0
        - np.mean(iou_values)
        + float(config.tail_weight) * (1.0 - np.quantile(iou_values, 0.05))
        + float(config.recall_penalty) * np.mean(deficits * deficits)
        + float(config.temporal_weight) * temporal
    )


def _proxy_objectives_batch(
    references: list[np.ndarray],
    controls: list[np.ndarray],
    config: FitConfig,
    frame_indices: np.ndarray,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> list[float]:
    """Evaluate independent persistent-index trials in one exact CPU batch."""

    if not controls:
        return []
    if exact_raster is None:
        return [
            _proxy_objective(
                references,
                value,
                config,
                frame_indices,
                counter,
                None,
            )
            for value in controls
        ]
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 4 or values.shape[3] != 2:
        raise ValueError("proxy controls must have shape (trials,frames,points,2)")
    trial_count, frame_count, point_count = values.shape[:3]
    curves = sample_curve_sequence(
        values.reshape(trial_count * frame_count, point_count, 2),
        config.samples_per_segment,
    ).reshape(trial_count, frame_count, -1, 2)
    selected = np.ascontiguousarray(curves[:, frame_indices], dtype=np.float64)
    invalid = strict_self_intersection_batch(
        selected.reshape(-1, selected.shape[-2], 2),
        threads=int(config.native_cpu_threads),
    ).reshape(trial_count, len(frame_indices))
    valid = np.flatnonzero(~np.any(invalid, axis=1))
    output = [float("inf")] * trial_count
    if not len(valid):
        return output
    boundaries = selected[valid].reshape(-1, selected.shape[-2], 2)
    frames = np.tile(np.asarray(frame_indices, dtype=np.int32), len(valid))
    metrics = exact_raster.metrics(
        frames,
        boundaries,
        threads=int(config.native_cpu_threads),
    ).reshape(len(valid), len(frame_indices), 7)
    counter[0] += int(len(valid) * len(frame_indices))
    for packed_index, trial_index in enumerate(valid):
        iou_values = np.asarray(metrics[packed_index, :, 6], dtype=np.float64)
        recall_values = np.asarray(metrics[packed_index, :, 4], dtype=np.float64)
        deficits = np.maximum(float(config.recall_floor) - recall_values, 0.0)
        residual = temporal_residuals(values[int(trial_index)])
        temporal = float(np.mean(residual)) if residual.size else 0.0
        output[int(trial_index)] = float(
            1.0
            - np.mean(iou_values)
            + float(config.tail_weight) * (1.0 - np.quantile(iou_values, 0.05))
            + float(config.recall_penalty) * np.mean(deficits * deficits)
            + float(config.temporal_weight) * temporal
        )
    return output


def _initial_persistent_locations(
    references: list[np.ndarray],
    dense: np.ndarray,
    config: FitConfig,
    frame_indices: np.ndarray,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = int(dense.shape[1])
    point_count = int(config.control_point_count)
    phase_step = max(1, count // (point_count * 4))
    candidates = [_saliency_indices(dense, point_count)]
    for phase in range(0, max(1, count // point_count), phase_step):
        candidates.append(_uniform_indices(count, point_count, phase))
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for indices in candidates:
        rolled, shifted = _roll_gauge(dense, indices)
        if len(shifted) != point_count:
            continue
        prepared.append((rolled, shifted, rolled[:, shifted]))
    scores = _proxy_objectives_batch(
        references,
        [value[2] for value in prepared],
        config,
        frame_indices,
        counter,
        exact_raster,
    )
    best = None
    for (rolled, shifted, _controls), score in zip(prepared, scores, strict=True):
        if best is None or score < best[0]:
            best = (score, rolled, shifted)
    if best is None:
        raise RuntimeError("no valid Catmull-Rom initialization")
    return best[1], best[2], float(best[0])


def _refine_persistent_locations(
    references: list[np.ndarray],
    dense: np.ndarray,
    indices: np.ndarray,
    config: FitConfig,
    frame_indices: np.ndarray,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, float]:
    active = np.asarray(indices, dtype=np.int32).copy()
    controls = dense[:, active]
    best_score = _proxy_objective(
        references,
        controls,
        config,
        frame_indices,
        counter,
        exact_raster,
    )
    radius = max(1, int(config.index_refine_radius))
    offsets = sorted(
        set((-radius, -max(1, radius // 2), -1, 0, 1, max(1, radius // 2), radius))
    )
    for _pass in range(max(0, int(config.index_refine_passes))):
        changed = False
        # Index zero is the persistent cyclic gauge.  Other indices may move
        # only inside their neighbours, so identities can never swap.
        for position in range(1, len(active)):
            lower = int(active[position - 1]) + 2
            upper = (
                int(active[position + 1]) - 2
                if position + 1 < len(active)
                else int(dense.shape[1]) - 2
            )
            current = int(active[position])
            trial_indices: list[np.ndarray] = []
            trial_values: list[int] = []
            for offset in offsets:
                candidate = current + int(offset)
                if candidate < lower or candidate > upper or candidate == current:
                    continue
                trial = active.copy()
                trial[position] = int(candidate)
                trial_indices.append(trial)
                trial_values.append(int(candidate))
            scores = _proxy_objectives_batch(
                references,
                [dense[:, trial] for trial in trial_indices],
                config,
                frame_indices,
                counter,
                exact_raster,
            )
            local_best = current
            local_score = best_score
            for candidate, score in zip(trial_values, scores, strict=True):
                if score + 1e-12 < local_score:
                    local_score = float(score)
                    local_best = int(candidate)
            if local_best != current:
                active[position] = local_best
                best_score = local_score
                changed = True
        if not changed:
            break
    return active, float(best_score)


def _scaled_controls(controls: np.ndarray, scales: np.ndarray) -> np.ndarray:
    centers = np.mean(controls, axis=1, keepdims=True)
    return centers + scales[:, None, None] * (controls - centers)


def _repair_scale_path(
    references: list[np.ndarray],
    controls: np.ndarray,
    config: FitConfig,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    scales = np.arange(
        1.0,
        float(config.scale_maximum) + float(config.scale_step) * 0.5,
        float(config.scale_step),
        dtype=np.float64,
    )
    frame_count = len(references)
    state_count = len(scales)
    local = np.full((frame_count, state_count), np.inf, dtype=np.float64)
    recall_table = np.zeros_like(local)
    native_scale_metrics = (
        exact_raster is not None
        and os.environ.get("MASK_CURVE_NATIVE_CONTROL_METRICS", "1") != "0"
        and bool(getattr(exact_raster, "supports_native_catmull", False))
        and callable(getattr(exact_raster, "catmull_scale_metrics", None))
    )
    if native_scale_metrics:
        metrics = exact_raster.catmull_scale_metrics(
            controls,
            scales,
            samples_per_segment=int(config.samples_per_segment),
            threads=int(config.native_cpu_threads),
            check_topology=True,
        )
        valid = metrics[:, :, 7] > 0.5
        counter[0] += int(np.count_nonzero(valid))
        recall_table[valid] = metrics[:, :, 4][valid]
        feasible = valid & (
            metrics[:, :, 4] + 1e-12 >= float(config.recall_floor)
        )
        scale_penalties = float(config.scale_size_penalty) * (scales - 1.0) ** 2
        objective = 1.0 - metrics[:, :, 6] + scale_penalties[None, :]
        local[feasible] = objective[feasible]
    elif exact_raster is not None:
        centers = np.mean(controls, axis=1, keepdims=True)
        candidates = centers[:, None, :, :] + scales[None, :, None, None] * (
            controls[:, None, :, :] - centers[:, None, :, :]
        )
        curves = sample_curve_sequence(
            candidates.reshape(-1, controls.shape[1], 2),
            config.samples_per_segment,
        )
        invalid = strict_self_intersection_batch(
            curves,
            threads=int(config.native_cpu_threads),
        )
        valid_indices = np.flatnonzero(~invalid)
        metrics = exact_raster.metrics(
            (valid_indices // state_count).astype(np.int32),
            curves[valid_indices],
            threads=int(config.native_cpu_threads),
        )
        counter[0] += len(valid_indices)
        for case_index, values in zip(valid_indices, metrics, strict=True):
            frame = int(case_index // state_count)
            state = int(case_index % state_count)
            iou = float(values[6])
            recall = float(values[4])
            recall_table[frame, state] = recall
            if recall + 1e-12 >= float(config.recall_floor):
                local[frame, state] = (
                    1.0
                    - iou
                    + float(config.scale_size_penalty)
                    * (float(scales[state]) - 1.0) ** 2
                )
    else:
        for frame in range(frame_count):
            center = np.mean(controls[frame], axis=0, keepdims=True)
            for state, scale in enumerate(scales):
                candidate = center + float(scale) * (controls[frame] - center)
                curve = sample_closed_curve(candidate, config.samples_per_segment)
                if has_strict_self_intersection(curve):
                    continue
                iou, recall = frame_raster_metrics(references[frame], curve)
                counter[0] += 1
                recall_table[frame, state] = recall
                if recall + 1e-12 >= float(config.recall_floor):
                    local[frame, state] = (
                        1.0
                        - iou
                        + float(config.scale_size_penalty) * (float(scale) - 1.0) ** 2
                    )
        if not np.any(np.isfinite(local[frame])):
            # Keep processing rather than aborting.  The highest-Recall state
            # is selected and explicitly reported as unresolved.
            state = int(np.argmax(recall_table[frame]))
            local[frame, state] = (
                1.0 + float(config.recall_floor) - recall_table[frame, state]
            )
    cost = np.full_like(local, np.inf)
    parent = np.full((frame_count, state_count), -1, dtype=np.int32)
    cost[0] = local[0]
    for frame in range(1, frame_count):
        transition = (
            float(config.scale_transition_weight)
            * (scales[:, None] - scales[None, :]) ** 2
        )
        combined = cost[frame - 1][:, None] + transition
        parent[frame] = np.argmin(combined, axis=0)
        cost[frame] = local[frame] + np.min(combined, axis=0)
    states = np.empty((frame_count,), dtype=np.int32)
    states[-1] = int(np.argmin(cost[-1]))
    for frame in range(frame_count - 1, 0, -1):
        states[frame - 1] = parent[frame, states[frame]]
    selected = scales[states]
    unresolved = tuple(
        int(frame)
        for frame, state in enumerate(states)
        if recall_table[frame, state] + 1e-12 < float(config.recall_floor)
    )
    return selected, unresolved


def _metric_objective(metrics: CurveMetrics, config: FitConfig) -> float:
    return float(
        1.0
        - metrics.mean_iou
        + float(config.tail_weight) * (1.0 - metrics.q05_iou)
        + float(config.temporal_weight) * metrics.temporal_residual
        + float(config.recall_penalty)
        * max(0.0, float(config.recall_floor) - metrics.minimum_recall) ** 2
    )


def _sequence_raster_metrics_exact(
    references: list[np.ndarray],
    curves: np.ndarray,
    controls: np.ndarray,
    config: FitConfig,
    exact_raster: ExactDoubleRasterBatch | None,
) -> CurveMetrics:
    if exact_raster is None:
        return sequence_raster_metrics(
            references,
            curves,
            controls,
            recall_floor=float(config.recall_floor),
        )
    native = exact_raster.metrics(
        np.arange(len(curves), dtype=np.int32),
        curves,
        threads=int(config.native_cpu_threads),
    )
    ious = np.asarray(native[:, 6], dtype=np.float64)
    recalls = np.asarray(native[:, 4], dtype=np.float64)
    residuals = temporal_residuals(np.asarray(controls, dtype=np.float64))
    flat = residuals.reshape(-1) if residuals.size else np.zeros((1,), np.float64)
    intersections = strict_self_intersection_batch(
        curves,
        threads=int(config.native_cpu_threads),
    )
    return CurveMetrics(
        frames=int(len(references)),
        mean_iou=float(np.mean(ious)),
        minimum_iou=float(np.min(ious)),
        q01_iou=float(np.quantile(ious, 0.01)),
        q05_iou=float(np.quantile(ious, 0.05)),
        mean_recall=float(np.mean(recalls)),
        minimum_recall=float(np.min(recalls)),
        recall_violations=int(
            np.count_nonzero(recalls + 1e-12 < float(config.recall_floor))
        ),
        temporal_residual=float(np.mean(flat)),
        temporal_q95=float(np.quantile(flat, 0.95)),
        self_intersections=int(np.count_nonzero(intersections)),
    )


def _repaired_candidate(
    references: list[np.ndarray],
    controls: np.ndarray,
    config: FitConfig,
    counter: list[int],
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], CurveMetrics]:
    scales, unresolved = _repair_scale_path(
        references,
        controls,
        config,
        counter,
        exact_raster,
    )
    repaired = np.ascontiguousarray(
        _scaled_controls(controls, scales), dtype=np.float64
    )
    curves = sample_curve_sequence(repaired, int(config.samples_per_segment))
    metrics = _sequence_raster_metrics_exact(
        references,
        curves,
        repaired,
        config,
        exact_raster,
    )
    counter[0] += len(references)
    return repaired, np.asarray(scales), unresolved, metrics


def fit_sequence(
    references: list[np.ndarray],
    config: FitConfig = FitConfig(),
    *,
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> SequenceFitResult:
    """Fit one persistent closed Catmull--Rom curve to a contour sequence."""
    config.validate()
    source = [orient_ccw(value) for value in references]
    if not source:
        raise ValueError("at least one reference contour is required")
    started = time.perf_counter()
    counter = [0]
    raster = exact_raster
    if raster is None and bool(config.native_cpu_batches):
        try:
            raster = create_exact_raster_batch(
                source,
                maximum_cache_bytes=int(config.native_reference_cache_bytes),
                maximum_batch_cases=int(config.native_batch_cases),
            )
        except RuntimeError:
            raster = None
    dense = align_temporal_dense(source, int(config.dense_contour_samples))
    phase_shifts = tuple(0 for _frame in source)
    if bool(config.temporal_phase_stabilization_enabled):
        dense, phase_shifts = _stabilize_temporal_phase(
            dense,
            jump_ratio=float(config.temporal_phase_jump_ratio),
            jump_error=float(config.temporal_phase_jump_error),
        )
    representatives = _representative_frames(len(source), config.proxy_max_frames)
    dense, indices, _initial_proxy = _initial_persistent_locations(
        source,
        dense,
        config,
        representatives,
        counter,
        raster,
    )
    initial_controls = np.ascontiguousarray(dense[:, indices], dtype=np.float64)
    initial_curves = sample_curve_sequence(
        initial_controls, int(config.samples_per_segment)
    )
    initial_metrics = _sequence_raster_metrics_exact(
        source,
        initial_curves,
        initial_controls,
        config,
        raster,
    )
    counter[0] += len(source)
    refined_indices, _refined_proxy = _refine_persistent_locations(
        source,
        dense,
        indices,
        config,
        representatives,
        counter,
        raster,
    )
    base_controls = np.ascontiguousarray(dense[:, refined_indices], dtype=np.float64)
    fitted_controls = base_controls
    fit_stats = None
    if bool(config.whole_curve_fit_enabled):
        fitted_controls, fit_stats = fit_whole_curve_controls(
            dense,
            refined_indices,
            base_controls,
            samples_per_segment=int(config.samples_per_segment),
            anchor_weight=float(config.whole_curve_anchor_weight),
            temporal_correction_weight=float(config.whole_curve_temporal_weight),
            maximum_correction_chord_fraction=float(
                config.whole_curve_maximum_correction_fraction
            ),
        )
    candidates = [(0.0, base_controls)]
    if fit_stats is not None:
        for blend in sorted(set(float(value) for value in config.whole_curve_blends)):
            candidates.append(
                (
                    blend,
                    np.ascontiguousarray(
                        base_controls + blend * (fitted_controls - base_controls),
                        dtype=np.float64,
                    ),
                )
            )
    selected = None
    for blend, candidate in candidates:
        repaired = _repaired_candidate(
            source,
            candidate,
            config,
            counter,
            raster,
        )
        objective = _metric_objective(repaired[3], config)
        key = (
            objective,
            -repaired[3].minimum_iou,
            -repaired[3].q05_iou,
            blend,
        )
        if selected is None or key < selected[0]:
            selected = (key, blend, repaired)
    if selected is None:  # pragma: no cover - base is always present
        raise RuntimeError("no spatial curve candidate was evaluated")
    _key, selected_blend, (controls, scales, unresolved, metrics) = selected
    curves = sample_curve_sequence(controls, int(config.samples_per_segment))
    return SequenceFitResult(
        controls=controls,
        segments=np.asarray([bezier_segments(frame) for frame in controls]),
        sampled_curves=curves,
        persistent_dense_indices=tuple(int(value) for value in refined_indices),
        repair_scales=np.asarray(scales, dtype=np.float64),
        metrics=metrics,
        initial_metrics=initial_metrics,
        unresolved_recall_frames=unresolved,
        objective_before=_metric_objective(initial_metrics, config),
        objective_after=_metric_objective(metrics, config),
        elapsed_seconds=float(time.perf_counter() - started),
        exact_frame_evaluations=int(counter[0]),
        whole_curve_fit_applied=bool(selected_blend > 0.0),
        whole_curve_selected_blend=float(selected_blend),
        whole_curve_stats=fit_stats,
        temporal_phase_shifts=phase_shifts,
        native_reference_cache=(
            {} if raster is None else raster.cache_stats()
        ),
        config=config,
    )


__all__ = ("FitConfig", "SequenceFitResult", "fit_sequence")
