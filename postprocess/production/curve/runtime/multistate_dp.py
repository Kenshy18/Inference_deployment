"""Exact CPU multistate DP for the Production curve-shape palette."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from production.polygon.runtime.spatial_support.optimizer import temporal_residuals

from .keyframe_dp import (
    BoundaryRenderer,
    EdgeEvaluation,
    KeyframeDpConfig,
    KeyframeDpResult,
    _EPSILON,
    _frame_metrics,
    _quality_rescue_keys,
    _refine_pair_vote,
    _valid_audit_losses,
    audit_dense_path,
    interpolate_controls,
    materialize_controls,
    render_control_sequence,
)
from .native_cpu import (
    ExactDoubleRasterBatch,
    create_exact_raster_batch,
    native_module,
)
from .topology import has_strict_self_intersection, strict_self_intersection_batch


@dataclass(frozen=True, slots=True)
class _NodePath:
    frames: tuple[int, ...]
    states: tuple[int, ...]
    raw_cost: float


@dataclass(frozen=True, slots=True)
class CurvePointRefineConfig:
    enabled: bool = True
    sweeps: int = 2
    step_chord_fraction: float = 0.04
    minimum_step: float = 0.35
    maximum_step: float = 2.5
    scheduler: str = "sequential"


def _local_materialized_controls(
    chosen: tuple[int, ...],
    key_controls: np.ndarray,
    key_position: int,
) -> tuple[int, np.ndarray]:
    left_position = max(0, int(key_position) - 1)
    right_position = min(len(chosen) - 1, int(key_position) + 1)
    left_frame = int(chosen[left_position])
    right_frame = int(chosen[right_position])
    local_indices = tuple(
        int(frame - left_frame) for frame in chosen[left_position : right_position + 1]
    )
    local_keys = key_controls[left_position : right_position + 1]
    return left_frame, materialize_controls(
        right_frame - left_frame + 1,
        local_indices,
        local_keys,
    )


def _local_score(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    key_controls: np.ndarray,
    key_position: int,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> float | None:
    start, controls = _local_materialized_controls(chosen, key_controls, key_position)
    _boundaries, audit = audit_dense_path(
        references[start : start + len(controls)],
        controls,
        renderer,
        exact_raster=exact_raster,
        frame_offset=start,
        threads=int(config.native_cpu_threads),
    )
    if np.any(~audit.topology_valid) or np.any(
        audit.recall + _EPSILON < float(config.recall_floor)
    ):
        return None
    losses = 1.0 - audit.iou
    return float(
        -np.mean(losses + float(config.low_iou_quadratic_weight) * losses * losses)
    )


def _local_scores_batch(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    trial_key_controls: np.ndarray,
    key_position: int,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    exact_raster: ExactDoubleRasterBatch | None,
    minimum_score: float | None = None,
) -> list[float | None]:
    """Evaluate independent local P trials in one exact CPU raster batch."""
    trials = np.asarray(trial_key_controls, dtype=np.float64)
    if exact_raster is None:
        return [
            _local_score(
                references,
                chosen,
                trial,
                key_position,
                renderer,
                config,
                None,
            )
            for trial in trials
        ]
    left_position = max(0, int(key_position) - 1)
    right_position = min(len(chosen) - 1, int(key_position) + 1)
    start = int(chosen[left_position])
    end = int(chosen[right_position])
    local_indices = tuple(
        int(frame - start) for frame in chosen[left_position : right_position + 1]
    )
    span = end - start + 1
    dense_trials = np.asarray(
        [
            materialize_controls(
                span,
                local_indices,
                trial[left_position : right_position + 1],
            )
            for trial in trials
        ],
        dtype=np.float64,
    )
    rendered = render_control_sequence(renderer, dense_trials)
    boundaries = rendered.reshape(-1, rendered.shape[-2], 2)
    batch_threads = int(config.native_cpu_threads)
    frame_indices = np.tile(
        np.arange(start, end + 1, dtype=np.int32),
        len(trials),
    )
    metrics = exact_raster.metrics(
        frame_indices,
        boundaries,
        threads=batch_threads,
    ).reshape(len(trials), span, 7)
    output: list[float | None] = [None] * len(trials)
    needs_topology: list[int] = []
    for trial_index in range(len(trials)):
        recall = metrics[trial_index, :, 4]
        if np.any(recall + _EPSILON < float(config.recall_floor)):
            continue
        losses = 1.0 - metrics[trial_index, :, 6]
        score = float(
            -np.mean(losses + float(config.low_iou_quadratic_weight) * losses * losses)
        )
        output[trial_index] = score
        if minimum_score is None or score > float(minimum_score) + _EPSILON:
            needs_topology.append(trial_index)
    if needs_topology:
        shaped = boundaries.reshape(len(trials), span, boundaries.shape[1], 2)
        selected = shaped[np.asarray(needs_topology, dtype=np.int32)].reshape(
            -1,
            boundaries.shape[1],
            2,
        )
        topology = ~strict_self_intersection_batch(
            selected,
            threads=batch_threads,
        ).reshape(len(needs_topology), span)
        for selected_index, trial_index in enumerate(needs_topology):
            if np.any(~topology[selected_index]):
                output[trial_index] = None
    return output


def _local_scores_group(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    key_controls: np.ndarray,
    trials: list[tuple[int, int, np.ndarray]],
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    exact_raster: ExactDoubleRasterBatch,
    minimum_scores: list[float | None] | None = None,
) -> list[float | None]:
    """Evaluate trials from disjoint key neighbourhoods in one native call.

    Key positions separated by at least three do not share an interpolated
    edge.  Their local objective evaluations are therefore independent and
    can be flattened into one exact raster batch.  Spans may have different
    lengths; ``frame_indices`` keeps every rendered boundary paired with its
    original reference frame.
    """

    if not trials:
        return []
    if minimum_scores is not None and len(minimum_scores) != len(trials):
        raise ValueError("minimum_scores must match grouped curve trials")
    current = np.asarray(key_controls, dtype=np.float64)
    dense_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    spans: list[int] = []
    for key_position, point_position, trial_point in trials:
        left_position = max(0, int(key_position) - 1)
        right_position = min(len(chosen) - 1, int(key_position) + 1)
        start = int(chosen[left_position])
        end = int(chosen[right_position])
        local_indices = tuple(
            int(frame - start) for frame in chosen[left_position : right_position + 1]
        )
        local_keys = current[left_position : right_position + 1].copy()
        if int(point_position) >= 0:
            local_keys[
                int(key_position) - left_position, int(point_position)
            ] = np.asarray(trial_point, dtype=np.float64)
        span = int(end - start + 1)
        dense_parts.append(materialize_controls(span, local_indices, local_keys))
        frame_parts.append(np.arange(start, end + 1, dtype=np.int32))
        spans.append(span)
    dense = np.concatenate(dense_parts, axis=0)
    frame_indices = np.concatenate(frame_parts, axis=0)
    threads = int(config.native_cpu_threads)
    catmull_samples = getattr(renderer, "_catmull_samples_per_segment", None)
    native_controls = (
        catmull_samples is not None
        and os.environ.get("MASK_CURVE_NATIVE_CONTROL_METRICS", "1") != "0"
        and bool(getattr(exact_raster, "supports_native_catmull", False))
        and callable(getattr(exact_raster, "catmull_metrics", None))
    )
    if native_controls:
        metrics = exact_raster.catmull_metrics(
            frame_indices,
            dense,
            samples_per_segment=int(catmull_samples),
            threads=threads,
            check_topology=False,
        )
    else:
        boundaries = render_control_sequence(renderer, dense)
        metrics = exact_raster.metrics(
            frame_indices,
            boundaries,
            threads=threads,
        )
    output: list[float | None] = [None] * len(trials)
    promising: list[int] = []
    offsets = np.cumsum(np.asarray((0, *spans), dtype=np.int64))
    for trial_index in range(len(trials)):
        first = int(offsets[trial_index])
        last = int(offsets[trial_index + 1])
        recall = metrics[first:last, 4]
        if np.any(recall + _EPSILON < float(config.recall_floor)):
            continue
        losses = 1.0 - metrics[first:last, 6]
        score = float(
            -np.mean(losses + float(config.low_iou_quadratic_weight) * losses * losses)
        )
        output[trial_index] = score
        minimum = None if minimum_scores is None else minimum_scores[trial_index]
        if minimum is None or score > float(minimum) + _EPSILON:
            promising.append(trial_index)
    if promising:
        if native_controls and callable(
            getattr(exact_raster, "catmull_topology", None)
        ):
            selected_controls = np.concatenate(
                [
                    dense[int(offsets[trial_index]) : int(offsets[trial_index + 1])]
                    for trial_index in promising
                ],
                axis=0,
            )
            topology = (
                exact_raster.catmull_topology(
                    selected_controls,
                    samples_per_segment=int(catmull_samples),
                    threads=threads,
                )
                > 0
            )
            topology_offsets = np.cumsum(
                np.asarray(
                    (0, *(spans[trial_index] for trial_index in promising)),
                    dtype=np.int64,
                )
            )
            for position, trial_index in enumerate(promising):
                first = int(topology_offsets[position])
                last = int(topology_offsets[position + 1])
                if np.any(~topology[first:last]):
                    output[trial_index] = None
        else:
            selected_parts = [
                boundaries[int(offsets[trial_index]) : int(offsets[trial_index + 1])]
                for trial_index in promising
            ]
            topology = ~strict_self_intersection_batch(
                np.concatenate(selected_parts, axis=0),
                threads=threads,
            )
            topology_offsets = np.cumsum(
                np.asarray(
                    (0, *(spans[trial_index] for trial_index in promising)),
                    dtype=np.int64,
                )
            )
            for position, trial_index in enumerate(promising):
                first = int(topology_offsets[position])
                last = int(topology_offsets[position + 1])
                if np.any(~topology[first:last]):
                    output[trial_index] = None
    return output


def _refine_curve_points_color_batched(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    key_controls: np.ndarray,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    refine: CurvePointRefineConfig,
    exact_raster: ExactDoubleRasterBatch,
) -> tuple[np.ndarray, int, int]:
    """Coordinate ascent with disjoint key neighbourhoods batched together."""

    output = np.asarray(key_controls, dtype=np.float64).copy()
    trials = 0
    accepted = 0
    multipliers = (-1.0, -0.5, 0.5, 1.0)
    for sweep in range(int(refine.sweeps)):
        changed = False
        colors = list(range(3))
        if sweep % 2:
            colors.reverse()
        for color in colors:
            # Point traversal direction in the sequential oracle depends on
            # key parity.  Splitting each colour by parity preserves that
            # direction while retaining non-overlapping local spans.
            parities = (0, 1) if sweep % 2 == 0 else (1, 0)
            for parity in parities:
                key_positions = [
                    position
                    for position in range(len(chosen))
                    if position % 3 == color and position % 2 == parity
                ]
                if sweep % 2:
                    key_positions.reverse()
                if not key_positions:
                    continue
                baseline_trials = [
                    (position, -1, np.zeros((2,), dtype=np.float64))
                    for position in key_positions
                ]
                baseline_values = _local_scores_group(
                    references,
                    chosen,
                    output,
                    baseline_trials,
                    renderer,
                    config,
                    exact_raster,
                )
                baselines: dict[int, float] = {}
                for position, score in zip(key_positions, baseline_values, strict=True):
                    if score is None:  # pragma: no cover - exact-audited input
                        raise RuntimeError(
                            "batched point refinement received an invalid path"
                        )
                    baselines[int(position)] = float(score)
                point_positions = list(range(output.shape[1]))
                if (sweep + parity) % 2:
                    point_positions.reverse()
                for point_position in point_positions:
                    grouped_trials: list[tuple[int, int, np.ndarray]] = []
                    grouped_minimums: list[float] = []
                    owners: list[tuple[int, np.ndarray]] = []
                    for key_position in key_positions:
                        current = output[key_position, point_position].copy()
                        previous = output[
                            key_position,
                            (point_position - 1) % output.shape[1],
                        ]
                        following = output[
                            key_position,
                            (point_position + 1) % output.shape[1],
                        ]
                        tangent = following - previous
                        length = float(np.linalg.norm(tangent))
                        if length <= 1e-9:
                            continue
                        normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
                        normal /= max(float(np.linalg.norm(normal)), 1e-12)
                        chord = 0.5 * (
                            float(np.linalg.norm(current - previous))
                            + float(np.linalg.norm(following - current))
                        )
                        step = float(
                            np.clip(
                                float(refine.step_chord_fraction) * chord,
                                float(refine.minimum_step),
                                float(refine.maximum_step),
                            )
                        )
                        for multiplier in multipliers:
                            point = current + float(multiplier) * step * normal
                            grouped_trials.append((key_position, point_position, point))
                            grouped_minimums.append(baselines[key_position])
                            owners.append((key_position, point))
                    scores = _local_scores_group(
                        references,
                        chosen,
                        output,
                        grouped_trials,
                        renderer,
                        config,
                        exact_raster,
                        grouped_minimums,
                    )
                    best: dict[int, tuple[float, np.ndarray]] = {}
                    for score, (key_position, point) in zip(
                        scores, owners, strict=True
                    ):
                        trials += 1
                        if score is None or score <= baselines[key_position] + _EPSILON:
                            continue
                        previous_best = best.get(key_position)
                        if previous_best is None or score > previous_best[0] + _EPSILON:
                            best[key_position] = (float(score), point)
                    for key_position, (score, point) in best.items():
                        output[key_position, point_position] = point
                        baselines[key_position] = float(score)
                        accepted += 1
                        changed = True
        if not changed:
            break
    return output, int(trials), int(accepted)


def _refine_curve_points(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    key_controls: np.ndarray,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    refine: CurvePointRefineConfig,
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, int, int, float]:
    """Coordinate-ascent P along local contour normals with exact gates."""
    if not bool(refine.enabled) or int(refine.sweeps) <= 0:
        return np.asarray(key_controls, dtype=np.float64), 0, 0, 0.0
    output = np.asarray(key_controls, dtype=np.float64).copy()
    _before_boundaries, before_audit = audit_dense_path(
        references,
        materialize_controls(len(references), chosen, output),
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    before_losses = 1.0 - before_audit.iou
    before_score = float(
        -np.mean(
            before_losses
            + float(config.low_iou_quadratic_weight) * before_losses * before_losses
        )
    )
    scheduler = str(refine.scheduler)
    if scheduler not in {"sequential", "color_batched"}:
        raise ValueError(f"unsupported curve point refinement scheduler: {scheduler}")
    if scheduler == "color_batched":
        if exact_raster is None:
            raise RuntimeError(
                "color-batched curve point refinement requires native exact raster"
            )
        output, trials, accepted = _refine_curve_points_color_batched(
            references,
            chosen,
            output,
            renderer,
            config,
            refine,
            exact_raster,
        )
        _after_boundaries, after_audit = audit_dense_path(
            references,
            materialize_controls(len(references), chosen, output),
            renderer,
            exact_raster=exact_raster,
            threads=int(config.native_cpu_threads),
        )
        after_losses = 1.0 - after_audit.iou
        after_score = float(
            -np.mean(
                after_losses
                + float(config.low_iou_quadratic_weight) * after_losses * after_losses
            )
        )
        return output, int(trials), int(accepted), float(after_score - before_score)
    trials = 0
    accepted = 0
    multipliers = (-1.0, -0.5, 0.5, 1.0)
    for sweep in range(int(refine.sweeps)):
        changed = False
        key_positions = list(range(len(chosen)))
        if sweep % 2:
            key_positions.reverse()
        for key_position in key_positions:
            point_positions = list(range(output.shape[1]))
            if (sweep + key_position) % 2:
                point_positions.reverse()
            baseline = _local_score(
                references,
                chosen,
                output,
                key_position,
                renderer,
                config,
                exact_raster,
            )
            if baseline is None:  # pragma: no cover - caller exact-audited
                raise RuntimeError("point refinement received an invalid path")
            for point_position in point_positions:
                current = output[key_position, point_position].copy()
                previous = output[key_position, (point_position - 1) % output.shape[1]]
                following = output[key_position, (point_position + 1) % output.shape[1]]
                tangent = following - previous
                length = float(np.linalg.norm(tangent))
                if length <= 1e-9:
                    continue
                # P is CCW, hence (dy, -dx) is the outward normal.
                normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                chord = 0.5 * (
                    float(np.linalg.norm(current - previous))
                    + float(np.linalg.norm(following - current))
                )
                step = float(
                    np.clip(
                        float(refine.step_chord_fraction) * chord,
                        float(refine.minimum_step),
                        float(refine.maximum_step),
                    )
                )
                best_score = float(baseline)
                best_point = current
                trial_controls = []
                trial_points = []
                for multiplier in multipliers:
                    trial = output.copy()
                    trial[key_position, point_position] = (
                        current + float(multiplier) * step * normal
                    )
                    trial_controls.append(trial)
                    trial_points.append(trial[key_position, point_position].copy())
                scores = _local_scores_batch(
                    references,
                    chosen,
                    np.asarray(trial_controls, dtype=np.float64),
                    key_position,
                    renderer,
                    config,
                    exact_raster,
                    float(baseline),
                )
                for score, trial_point in zip(scores, trial_points, strict=True):
                    trials += 1
                    if score is not None and score > best_score + 1e-12:
                        best_score = float(score)
                        best_point = trial_point
                if not np.array_equal(best_point, current):
                    output[key_position, point_position] = best_point
                    baseline = float(best_score)
                    accepted += 1
                    changed = True
        if not changed:
            break
    _after_boundaries, after_audit = audit_dense_path(
        references,
        materialize_controls(len(references), chosen, output),
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    after_losses = 1.0 - after_audit.iou
    after_score = float(
        -np.mean(
            after_losses
            + float(config.low_iou_quadratic_weight) * after_losses * after_losses
        )
    )
    return output, int(trials), int(accepted), float(after_score - before_score)


def _refinement_guard_score(audit, config: KeyframeDpConfig) -> float:
    losses = 1.0 - np.asarray(audit.iou, dtype=np.float64)
    return float(
        -np.mean(losses + float(config.low_iou_quadratic_weight) * losses * losses)
    )


def _guard_refined_controls(
    references: list[np.ndarray],
    chosen: tuple[int, ...],
    baseline_key_controls: np.ndarray,
    refined_key_controls: np.ndarray,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    exact_raster: ExactDoubleRasterBatch | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, bool, float]:
    """Back off refinement if it moves a local frame-quality debt.

    Pair-vote and P-coordinate ascent optimize overlapping local means. A move
    can therefore improve their global objective while making one unrelated
    frame slightly worse. The DP/quality-rescue result is the immutable
    baseline here. We retain the best deterministic blend toward the refined
    controls that preserves exact Recall/topology and bounds minimum/low-tail
    IoU plus worst-case area regression. This still permits harmless quality
    redistribution between already-good frames.
    """

    baseline_keys = np.asarray(baseline_key_controls, dtype=np.float64)
    refined_keys = np.asarray(refined_key_controls, dtype=np.float64)
    baseline_dense = materialize_controls(
        len(references),
        chosen,
        baseline_keys,
    )
    _baseline_boundaries, baseline_audit = audit_dense_path(
        references,
        baseline_dense,
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    if np.any(~baseline_audit.topology_valid) or np.any(
        baseline_audit.recall + _EPSILON < float(config.recall_floor)
    ):
        raise RuntimeError("refinement guard received an invalid DP baseline")
    if np.array_equal(baseline_keys, refined_keys):
        return (
            refined_keys,
            baseline_dense,
            _baseline_boundaries,
            baseline_audit,
            False,
            1.0,
        )

    allowed_iou_drop = float(config.quality_rescue_maximum_iou_regression)
    allowed_area_growth = float(config.quality_rescue_maximum_area_ratio_regression)
    baseline_minimum_iou = float(np.min(baseline_audit.iou))
    baseline_q01_iou = float(np.quantile(baseline_audit.iou, 0.01))
    baseline_q05_iou = float(np.quantile(baseline_audit.iou, 0.05))
    baseline_maximum_area = float(np.max(baseline_audit.area_ratio))
    # A global minimum/quantile guard alone can still move the worst-frame
    # debt from one timestamp to another.  Keep a frame-local envelope around
    # the exact DP/rescue baseline as well.  Frames already below the soft
    # quality target (for example a genuinely corrupted AI observation) are
    # not forced toward raw; they merely cannot regress by more than the same
    # small allowance.  Frames at or above the target may never be refined
    # across it.  This preserves temporal repair while preventing pair-vote or
    # P-coordinate ascent from sacrificing one previously-good frame.
    baseline_iou = np.asarray(baseline_audit.iou, dtype=np.float64)
    per_frame_iou_floor = np.where(
        baseline_iou < float(config.quality_rescue_iou_floor),
        baseline_iou - allowed_iou_drop,
        float(config.quality_rescue_iou_floor),
    )
    baseline_area = np.asarray(baseline_audit.area_ratio, dtype=np.float64)
    per_frame_area_cap = np.where(
        baseline_area > float(config.quality_rescue_area_ratio_cap),
        baseline_area + allowed_area_growth,
        float(config.quality_rescue_area_ratio_cap),
    )

    def evaluate(keys: np.ndarray):
        dense = materialize_controls(len(references), chosen, keys)
        boundaries, audit = audit_dense_path(
            references,
            dense,
            renderer,
            exact_raster=exact_raster,
            threads=int(config.native_cpu_threads),
        )
        hard_valid = bool(
            np.all(audit.topology_valid)
            and np.all(audit.recall + _EPSILON >= float(config.recall_floor))
        )
        quality_valid = bool(
            np.all(audit.iou + _EPSILON >= per_frame_iou_floor)
            and np.all(audit.area_ratio <= per_frame_area_cap + _EPSILON)
            and float(np.min(audit.iou)) + allowed_iou_drop + _EPSILON
            >= baseline_minimum_iou
            and float(np.quantile(audit.iou, 0.01)) + allowed_iou_drop + _EPSILON
            >= baseline_q01_iou
            and float(np.quantile(audit.iou, 0.05)) + allowed_iou_drop + _EPSILON
            >= baseline_q05_iou
            and float(np.max(audit.area_ratio))
            <= baseline_maximum_area + allowed_area_growth + _EPSILON
        )
        return dense, boundaries, audit, hard_valid and quality_valid

    dense, boundaries, audit, valid = evaluate(refined_keys)
    if valid:
        return refined_keys, dense, boundaries, audit, False, 1.0

    hard_bad = (~audit.topology_valid) | (
        audit.recall + _EPSILON < float(config.recall_floor)
    )
    local_quality_bad = (audit.iou + _EPSILON < per_frame_iou_floor) | (
        audit.area_ratio > per_frame_area_cap + _EPSILON
    )
    global_minimum_bad = bool(
        float(np.min(audit.iou)) + allowed_iou_drop + _EPSILON < baseline_minimum_iou
    )
    global_q01_bad = bool(
        float(np.quantile(audit.iou, 0.01)) + allowed_iou_drop + _EPSILON
        < baseline_q01_iou
    )
    global_q05_bad = bool(
        float(np.quantile(audit.iou, 0.05)) + allowed_iou_drop + _EPSILON
        < baseline_q05_iou
    )
    global_area_bad = bool(
        float(np.max(audit.area_ratio))
        > baseline_maximum_area + allowed_area_growth + _EPSILON
    )
    global_bad = np.zeros(len(audit.iou), dtype=bool)
    if global_minimum_bad:
        global_bad |= audit.iou <= baseline_minimum_iou - allowed_iou_drop + _EPSILON
    if global_q01_bad:
        global_bad |= audit.iou <= float(np.quantile(audit.iou, 0.01)) + _EPSILON
    if global_q05_bad:
        global_bad |= audit.iou <= float(np.quantile(audit.iou, 0.05)) + _EPSILON
    if global_area_bad:
        global_bad |= (
            audit.area_ratio >= baseline_maximum_area + allowed_area_growth - _EPSILON
        )
    bad_frames = np.flatnonzero(hard_bad | local_quality_bad | global_bad)
    affected: set[int] = set()
    chosen_array = np.asarray(chosen, dtype=np.int64)
    for frame in bad_frames:
        right = int(np.searchsorted(chosen_array, int(frame), side="left"))
        right = min(max(right, 0), len(chosen) - 1)
        left = max(0, right - 1)
        affected.update((left, right))

    candidates: list[
        tuple[tuple[float, float], float, np.ndarray, np.ndarray, np.ndarray, object]
    ] = []
    delta = refined_keys - baseline_keys
    # Revert only the key neighbourhoods responsible for a bad frame. This
    # preserves refinements elsewhere instead of globally diluting every P.
    # Expand by one key only if a mixed baseline/refined boundary cannot pass.
    for _expansion in range(3):
        selected = np.asarray(sorted(affected), dtype=np.int32)
        for alpha in np.linspace(15.0 / 16.0, 0.0, 16):
            trial_keys = refined_keys.copy()
            trial_keys[selected] = (
                baseline_keys[selected] + float(alpha) * delta[selected]
            )
            trial_dense, trial_boundaries, trial_audit, trial_valid = evaluate(
                trial_keys
            )
            if not trial_valid:
                continue
            candidates.append(
                (
                    (_refinement_guard_score(trial_audit, config), float(alpha)),
                    float(alpha),
                    np.ascontiguousarray(trial_keys),
                    trial_dense,
                    trial_boundaries,
                    trial_audit,
                )
            )
        if candidates:
            break
        expanded = set(affected)
        for position in affected:
            expanded.add(max(0, position - 1))
            expanded.add(min(len(chosen) - 1, position + 1))
        affected = expanded
    if not candidates:
        # The exact DP baseline is always valid. This final branch handles an
        # unexpectedly nonlocal interaction without terminating the batch.
        candidates.append(
            (
                (_refinement_guard_score(baseline_audit, config), 0.0),
                0.0,
                np.ascontiguousarray(baseline_keys),
                baseline_dense,
                _baseline_boundaries,
                baseline_audit,
            )
        )
    if not candidates:  # pragma: no cover - alpha=0 is an audited invariant
        raise RuntimeError("refinement guard could not recover its valid baseline")
    _score, alpha, keys, dense, boundaries, audit = max(
        candidates,
        key=lambda value: value[0],
    )
    return keys, dense, boundaries, audit, True, float(alpha)


class _MultistateIntervalEvaluator:
    def __init__(
        self,
        references: list[np.ndarray],
        state_controls: np.ndarray,
        renderer: BoundaryRenderer,
        config: KeyframeDpConfig,
        exact_raster: ExactDoubleRasterBatch | None = None,
        seed_evaluator: "_MultistateIntervalEvaluator | None" = None,
    ) -> None:
        self.references = references
        self.controls = np.asarray(state_controls, dtype=np.float64)
        self.renderer = renderer
        self.config = config
        self.exact_raster = exact_raster
        self.seed_evaluator = seed_evaluator
        self.cache: dict[tuple[int, int, int, int], EdgeEvaluation] = {}
        self.frame_evaluations = 0
        self.native_batch_precomputed = False
        self._native_decode_edges: np.ndarray | None = None
        self._native_decode_costs: np.ndarray | None = None
        self._native_edge_count = 0
        self._native_reused_edge_count = 0
        self._shape_distance_unique_pairs = 0
        self._maximum_gap = 0
        self._lazy_topology_decodes = 0
        self._lazy_topology_checked_edges = 0
        self._lazy_topology_checked_frames = 0
        self._lazy_topology_rejected_edges = 0
        self._lazy_topology_edge_cache: dict[tuple[int, int, int, int], bool] = {}
        self._states_share_similarity_shape = self._detect_similarity_states()

    def _detect_similarity_states(self) -> bool:
        """Return whether states differ only by positive isotropic scale.

        Shape distance removes translation, rotation and isotropic scale.  The
        Production curve palette is therefore shape-identical across states,
        so evaluating the same frame pair once is mathematically sufficient.
        Keep a strict geometric check here because this evaluator is also used
        by tests and may receive arbitrary future state palettes.
        """

        values = np.asarray(self.controls, dtype=np.float64)
        if values.shape[1] <= 1:
            return True
        base = values[:, 0]
        base_zero = base - np.mean(base, axis=1, keepdims=True)
        denominator = np.sum(base_zero * base_zero, axis=(1, 2))
        tolerance = 2e-11 * max(1.0, float(np.max(np.abs(values))))
        for state in range(1, values.shape[1]):
            candidate = values[:, state]
            candidate_zero = candidate - np.mean(
                candidate,
                axis=1,
                keepdims=True,
            )
            scale = np.ones((len(values),), dtype=np.float64)
            regular = denominator > 1e-18
            scale[regular] = (
                np.sum(
                    base_zero[regular] * candidate_zero[regular],
                    axis=(1, 2),
                )
                / denominator[regular]
            )
            if np.any(scale <= 0.0):
                return False
            residual = candidate_zero - scale[:, None, None] * base_zero
            if float(np.max(np.abs(residual))) > tolerance:
                return False
        return True

    def _shape_distances_for_edges(
        self,
        graph_edges: np.ndarray,
        *,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Compute the exact legacy shape term without materializing all pairs.

        The previous implementation built two endpoint tensors and one
        interleaved tensor for every graph edge at once.  That is harmless for
        a 60-frame experiment but makes resident memory scale poorly on long
        tracks.  Chunking preserves the same ``temporal_residuals`` arithmetic
        and edge order while bounding temporary memory.
        """
        edges = np.asarray(graph_edges, dtype=np.int32)
        inverse: np.ndarray | None = None
        if self._states_share_similarity_shape:
            # All current curve states are positive isotropic scales of the
            # same P geometry.  ``temporal_residuals`` removes translation,
            # rotation and scale, hence every state pair of the same two
            # frames has the same mathematical distance.  Deduplicate frame
            # pairs, while retaining the scalar oracle below.
            pairs, inverse = np.unique(
                edges[:, (0, 2)],
                axis=0,
                return_inverse=True,
            )
            compact = np.column_stack(
                (
                    pairs[:, 0],
                    np.zeros((len(pairs),), dtype=np.int32),
                    pairs[:, 1],
                    np.zeros((len(pairs),), dtype=np.int32),
                )
            )
        else:
            compact = edges
        self._shape_distance_unique_pairs += int(len(compact))
        compact_output = np.empty((len(compact),), dtype=np.float64)
        for offset in range(0, len(compact), max(1, int(batch_size))):
            chunk = compact[offset : offset + max(1, int(batch_size))]
            left = self.controls[chunk[:, 0], chunk[:, 1]]
            right = self.controls[chunk[:, 2], chunk[:, 3]]
            paired = np.empty(
                (len(chunk) * 2, self.controls.shape[2], 2),
                dtype=np.float64,
            )
            paired[0::2] = left
            paired[1::2] = right
            compact_output[offset : offset + len(chunk)] = np.mean(
                temporal_residuals(paired)[0::2],
                axis=1,
            )
        if inverse is None:
            return compact_output
        return np.ascontiguousarray(compact_output[inverse], dtype=np.float64)

    def precompute_all(self, maximum_gap: int) -> bool:
        """Populate every edge with exact CPU batches, preserving DP semantics."""
        if not bool(self.config.native_cpu_batches):
            return False
        raster = self.exact_raster
        if raster is None:
            try:
                raster = create_exact_raster_batch(
                    self.references,
                    maximum_cache_bytes=int(self.config.native_reference_cache_bytes),
                    maximum_batch_cases=int(self.config.native_batch_cases),
                )
            except RuntimeError:
                return False
        frame_count = int(self.controls.shape[0])
        state_count = int(self.controls.shape[1])
        self._maximum_gap = int(maximum_gap)
        edge_count = int(
            sum(min(int(maximum_gap), end) for end in range(1, frame_count))
            * state_count
            * state_count
        )
        graph_edges = np.empty((edge_count, 4), dtype=np.int32)
        cursor = 0
        # This is already the stable decode order used by the legacy Python
        # loops: end frame, end state, start frame, start state.
        for end in range(1, frame_count):
            first_start = max(0, end - int(maximum_gap))
            for end_state in range(state_count):
                for start in range(first_start, end):
                    size = state_count
                    rows = graph_edges[cursor : cursor + size]
                    rows[:, 0] = int(start)
                    rows[:, 1] = np.arange(state_count, dtype=np.int32)
                    rows[:, 2] = int(end)
                    rows[:, 3] = int(end_state)
                    cursor += size
        if cursor != edge_count:  # pragma: no cover - defensive invariant
            raise RuntimeError("curve graph edge allocation mismatch")
        spans = graph_edges[:, 2] - graph_edges[:, 0]
        threads = int(self.config.native_cpu_threads)
        endpoint_boundaries = render_control_sequence(
            self.renderer,
            self.controls,
        )
        costs = np.full((edge_count,), np.inf, dtype=np.float64)
        pending = np.ones((edge_count,), dtype=bool)
        seed = self.seed_evaluator
        if (
            seed is not None
            and seed._native_decode_edges is not None
            and seed._native_decode_costs is not None
            and seed.controls.shape[0] == self.controls.shape[0]
            and seed.controls.shape[2:] == self.controls.shape[2:]
            and seed.controls.shape[1] <= self.controls.shape[1]
            and np.array_equal(
                seed.controls,
                self.controls[:, : seed.controls.shape[1]],
            )
        ):
            seed_costs = {
                tuple(int(value) for value in edge): float(cost)
                for edge, cost in zip(
                    seed._native_decode_edges,
                    seed._native_decode_costs,
                    strict=True,
                )
            }
            reusable = (graph_edges[:, 1] < seed.controls.shape[1]) & (
                graph_edges[:, 3] < seed.controls.shape[1]
            )
            reusable_indices = np.flatnonzero(reusable)
            for index in reusable_indices:
                key = tuple(int(value) for value in graph_edges[index])
                cached = seed_costs.get(key)
                if cached is None:  # pragma: no cover - construction invariant
                    continue
                costs[index] = cached
                pending[index] = False
            self._native_reused_edge_count = int(np.count_nonzero(~pending))
            self.frame_evaluations = int(seed.frame_evaluations)
            # These counters describe total work for the two-stage fast/full
            # state search.  The full evaluator reuses the seed graph and its
            # exact topology cache, so carry the already-paid work forward.
            self._lazy_topology_decodes = int(seed._lazy_topology_decodes)
            self._lazy_topology_checked_edges = int(seed._lazy_topology_checked_edges)
            self._lazy_topology_checked_frames = int(seed._lazy_topology_checked_frames)
            self._lazy_topology_rejected_edges = int(seed._lazy_topology_rejected_edges)
            self._lazy_topology_edge_cache.update(
                {
                    edge: valid
                    for edge, valid in seed._lazy_topology_edge_cache.items()
                    if edge[1] < self.controls.shape[1]
                    and edge[3] < self.controls.shape[1]
                }
            )

        pending_edges = graph_edges[pending]
        pending_spans = spans[pending]
        if len(pending_edges):
            exact = raster.edge_metrics(
                endpoint_boundaries,
                pending_edges,
                recall_floor=float(self.config.recall_floor),
                low_iou_quadratic_weight=float(self.config.low_iou_quadratic_weight),
                threads=threads,
                check_topology=False,
            )
            loss_totals = exact[:, 0]
            minimum_recalls = exact[:, 2]
            frames_covered = np.rint(exact[:, 3]).astype(np.int32)
            topology_valid = exact[:, 4] > 0.5
            self.frame_evaluations += int(np.sum(frames_covered, dtype=np.int64))
            if float(self.config.shape_distance_weight) > _EPSILON:
                shape_distances = self._shape_distances_for_edges(pending_edges)
            else:
                shape_distances = np.zeros((len(pending_edges),), dtype=np.float64)
            floor = float(self.config.recall_floor)
            feasible = (
                topology_valid
                & (frames_covered == pending_spans)
                & (minimum_recalls + _EPSILON >= floor)
            )
            costs[pending] = np.where(
                feasible,
                loss_totals
                + float(self.config.shape_distance_weight) * shape_distances,
                np.inf,
            )
        self.native_batch_precomputed = True
        self._native_decode_edges = np.ascontiguousarray(graph_edges)
        self._native_decode_costs = np.ascontiguousarray(
            costs,
            dtype=np.float64,
        )
        self._native_edge_count = int(edge_count)
        return True

    def decode_native(
        self,
        frame_count: int,
        state_count: int,
        first_losses: np.ndarray,
        penalty: float,
    ) -> _NodePath | None:
        module = native_module()
        if (
            module is None
            or not hasattr(module, "decode_penalty_path")
            or self._native_decode_edges is None
            or self._native_decode_costs is None
        ):
            return None
        initial = np.ascontiguousarray(first_losses, dtype=np.float64)
        while True:
            frames, states, raw_cost = module.decode_penalty_path(
                self._native_decode_edges,
                self._native_decode_costs,
                initial,
                int(frame_count),
                int(state_count),
                float(penalty),
            )
            if not frames:
                raise RuntimeError("no hard-Recall-feasible multistate curve path")
            path = _NodePath(
                tuple(int(value) for value in frames),
                tuple(int(value) for value in states),
                float(raw_cost),
            )
            self._lazy_topology_decodes += 1
            invalid = self._invalid_path_edges(path)
            if not invalid:
                return path
            for edge in invalid:
                index = self._native_edge_offset(*edge)
                self._native_decode_costs[index] = np.inf
            self._lazy_topology_rejected_edges += int(len(invalid))

    def decode_cardinality_native(
        self,
        frame_count: int,
        state_count: int,
        first_losses: np.ndarray,
        target_count: int,
        maximum_count: int,
    ) -> _NodePath | None:
        """Return the lowest-loss hard-Recall path at (or just above) K.

        Unlike a Lagrangian penalty sweep, this retains non-convex points of
        the key-count/IoU frontier.  Lazy topology rejection has the same
        semantics as ``decode_native``: a selected invalid edge is removed
        and the fixed-cardinality graph is decoded again.
        """

        module = native_module()
        if (
            module is None
            or not hasattr(module, "decode_cardinality_path")
            or self._native_decode_edges is None
            or self._native_decode_costs is None
        ):
            return None
        initial = np.ascontiguousarray(first_losses, dtype=np.float64)
        while True:
            frames, states, raw_cost, _actual_count = module.decode_cardinality_path(
                self._native_decode_edges,
                self._native_decode_costs,
                initial,
                int(frame_count),
                int(state_count),
                int(target_count),
                int(maximum_count),
            )
            if not frames:
                return None
            path = _NodePath(
                tuple(int(value) for value in frames),
                tuple(int(value) for value in states),
                float(raw_cost),
            )
            self._lazy_topology_decodes += 1
            invalid = self._invalid_path_edges(path)
            if not invalid:
                return path
            for edge in invalid:
                index = self._native_edge_offset(*edge)
                self._native_decode_costs[index] = np.inf
            self._lazy_topology_rejected_edges += int(len(invalid))

    def _native_edge_offset(
        self,
        start_frame: int,
        start_state: int,
        end_frame: int,
        end_state: int,
    ) -> int:
        """Map one edge to the stable native construction order exactly."""

        states = int(self.controls.shape[1])
        gap = int(self._maximum_gap)
        prefix = sum(min(gap, end) for end in range(1, int(end_frame)))
        first_start = max(0, int(end_frame) - gap)
        starts = int(end_frame) - first_start
        index = (
            prefix * states * states
            + int(end_state) * starts * states
            + (int(start_frame) - first_start) * states
            + int(start_state)
        )
        expected = np.asarray(
            (start_frame, start_state, end_frame, end_state),
            dtype=np.int32,
        )
        if not np.array_equal(self._native_decode_edges[index], expected):
            raise RuntimeError("native curve edge offset contract drifted")
        return int(index)

    def _invalid_path_edges(
        self,
        path: _NodePath,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return selected DP edges whose dense curve is not simple.

        A penalty sweep repeatedly selects many of the same edges.  Topology
        depends only on the fixed endpoint states, not on lambda, so cache the
        exact result per edge.  The cache is bounded by the small union of
        decoded paths rather than by the complete O(T * gap * states^2)
        graph.
        """

        if len(path.frames) <= 1:
            return ()
        edges = tuple(
            (
                int(path.frames[position]),
                int(path.states[position]),
                int(path.frames[position + 1]),
                int(path.states[position + 1]),
            )
            for position in range(len(path.frames) - 1)
        )
        unknown = tuple(
            edge for edge in edges if edge not in self._lazy_topology_edge_cache
        )
        if unknown:
            dense_blocks: list[np.ndarray] = []
            spans: list[int] = []
            for start, start_state, end, end_state in unknown:
                span = int(end - start)
                left = self.controls[start, start_state]
                right = self.controls[end, end_state]
                dense_blocks.append(
                    np.asarray(
                        [
                            interpolate_controls(
                                left,
                                right,
                                float(step / span),
                            )
                            for step in range(1, span + 1)
                        ],
                        dtype=np.float64,
                    )
                )
                spans.append(span)
            dense = np.concatenate(dense_blocks, axis=0)
            boundaries = render_control_sequence(self.renderer, dense)
            intersects = strict_self_intersection_batch(
                boundaries,
                threads=int(self.config.native_cpu_threads),
            )
            offset = 0
            for edge, span in zip(unknown, spans, strict=True):
                self._lazy_topology_edge_cache[edge] = not bool(
                    np.any(intersects[offset : offset + span])
                )
                offset += span
            self._lazy_topology_checked_edges += int(len(unknown))
            self._lazy_topology_checked_frames += int(len(intersects))
        return tuple(edge for edge in edges if not self._lazy_topology_edge_cache[edge])

    def edge(
        self,
        start: int,
        start_state: int,
        end: int,
        end_state: int,
    ) -> EdgeEvaluation:
        key = (int(start), int(start_state), int(end), int(end_state))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if end <= start:
            raise ValueError("edge end must be after start")
        losses: list[float] = []
        recalls: list[float] = []
        span = int(end - start)
        left = self.controls[start, start_state]
        right = self.controls[end, end_state]
        frames = tuple(range(start + 1, end + 1))
        interpolated = np.asarray(
            [
                interpolate_controls(
                    left,
                    right,
                    float((frame - start) / span),
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        boundaries = render_control_sequence(self.renderer, interpolated)
        self.frame_evaluations += len(frames)
        topology_valid = not bool(np.any(strict_self_intersection_batch(boundaries)))
        for frame, boundary in zip(frames, boundaries, strict=True):
            if not topology_valid:
                break
            iou, recall, _precision, _area = _frame_metrics(
                self.references[frame], boundary
            )
            losses.append(1.0 - float(iou))
            recalls.append(float(recall))
            if recall + _EPSILON < float(self.config.recall_floor):
                break
        feasible = bool(
            topology_valid
            and len(losses) == span
            and min(recalls, default=0.0) + _EPSILON >= float(self.config.recall_floor)
        )
        shape_residual = temporal_residuals(np.stack((left, right), axis=0))
        shape_distance = float(np.mean(shape_residual)) if shape_residual.size else 0.0
        values = np.asarray(losses, dtype=np.float64)
        cost = (
            float(
                np.sum(
                    values
                    + float(self.config.low_iou_quadratic_weight) * values * values
                )
            )
            + float(self.config.shape_distance_weight) * shape_distance
            if feasible
            else float("inf")
        )
        result = EdgeEvaluation(
            cost=cost,
            frames_covered=len(losses),
            minimum_recall=float(min(recalls, default=0.0)),
            mean_iou=float(1.0 - np.mean(values)) if len(values) else 0.0,
            topology_valid=topology_valid,
        )
        self.cache[key] = result
        return result


def isotropic_curve_states(
    controls: np.ndarray,
    scales: tuple[float, ...] = (1.0, 1.005, 1.015, 1.025, 1.035),
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create a compact bounded scale palette without giant masks.

    A tiny inward state is useful when the spatial Recall repair has surplus
    coverage.  It receives no special exemption: the same exact per-frame
    Recall gate rejects it whenever contraction would lose source pixels.
    """
    source = np.asarray(controls, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 2:
        raise ValueError("controls must have shape (frames, points, 2)")
    values = tuple(float(value) for value in scales)
    if not values or any(value < 0.95 for value in values):
        raise ValueError("curve state scales must be nonempty and >= 0.95")
    center = np.mean(source, axis=1, keepdims=True)
    states = np.stack([center + value * (source - center) for value in values], axis=1)
    return np.ascontiguousarray(states), tuple(f"scale_{value:.3f}" for value in values)


def _first_node_losses(
    references: list[np.ndarray],
    controls: np.ndarray,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
) -> np.ndarray:
    output = np.full((controls.shape[1],), np.inf, dtype=np.float64)
    for state in range(controls.shape[1]):
        boundary = renderer(controls[0, state])
        if has_strict_self_intersection(boundary):
            continue
        iou, recall, _precision, _area = _frame_metrics(references[0], boundary)
        if recall + _EPSILON >= float(config.recall_floor):
            loss = 1.0 - float(iou)
            output[state] = loss + float(config.low_iou_quadratic_weight) * loss * loss
    if not np.any(np.isfinite(output)):
        raise RuntimeError("no feasible first multistate curve candidate")
    return output


def _decode(
    frame_count: int,
    state_count: int,
    evaluator: _MultistateIntervalEvaluator,
    key_penalty: float,
    maximum_gap: int,
    first_losses: np.ndarray,
) -> _NodePath:
    native = evaluator.decode_native(
        frame_count,
        state_count,
        first_losses,
        float(key_penalty),
    )
    if native is not None:
        return native
    cost = np.full((frame_count, state_count), np.inf, dtype=np.float64)
    raw = np.full_like(cost, np.inf)
    parent_frame = np.full((frame_count, state_count), -1, dtype=np.int32)
    parent_state = np.full_like(parent_frame, -1)
    cost[0] = first_losses + float(key_penalty)
    raw[0] = first_losses
    for end in range(1, frame_count):
        for end_state in range(state_count):
            for start in range(max(0, end - int(maximum_gap)), end):
                for start_state in range(state_count):
                    if not np.isfinite(cost[start, start_state]):
                        continue
                    edge = evaluator.edge(start, start_state, end, end_state)
                    if not edge.feasible:
                        continue
                    candidate_raw = float(raw[start, start_state] + edge.cost)
                    candidate = float(
                        cost[start, start_state] + edge.cost + key_penalty
                    )
                    if candidate < cost[end, end_state] - 1e-12 or (
                        abs(candidate - cost[end, end_state]) <= 1e-12
                        and candidate_raw < raw[end, end_state]
                    ):
                        cost[end, end_state] = candidate
                        raw[end, end_state] = candidate_raw
                        parent_frame[end, end_state] = int(start)
                        parent_state[end, end_state] = int(start_state)
    end_state = int(np.argmin(cost[-1]))
    if not np.isfinite(cost[-1, end_state]):
        raise RuntimeError("no hard-Recall-feasible multistate curve path")
    frames: list[int] = []
    states: list[int] = []
    frame = frame_count - 1
    state = end_state
    while frame >= 0:
        frames.append(int(frame))
        states.append(int(state))
        next_frame = int(parent_frame[frame, state])
        next_state = int(parent_state[frame, state])
        frame, state = next_frame, next_state
    frames.reverse()
    states.reverse()
    return _NodePath(tuple(frames), tuple(states), float(raw[-1, end_state]))


def _decode_fixed_cardinality(
    frame_count: int,
    state_count: int,
    evaluator: _MultistateIntervalEvaluator,
    target_count: int,
    maximum_count: int,
    maximum_gap: int,
    first_losses: np.ndarray,
) -> _NodePath:
    """Scalar oracle for the native fixed-cardinality decoder."""

    bounded_maximum = min(int(frame_count), max(int(target_count), int(maximum_count)))
    cost = np.full(
        (bounded_maximum + 1, frame_count, state_count),
        np.inf,
        dtype=np.float64,
    )
    parent_frame = np.full(cost.shape, -1, dtype=np.int32)
    parent_state = np.full(cost.shape, -1, dtype=np.int16)
    cost[1, 0] = np.asarray(first_losses, dtype=np.float64)
    for end in range(1, frame_count):
        for end_state in range(state_count):
            for start in range(max(0, end - int(maximum_gap)), end):
                last_count = min(bounded_maximum - 1, start + 1)
                for start_state in range(state_count):
                    edge = evaluator.edge(start, start_state, end, end_state)
                    if not edge.feasible:
                        continue
                    for count in range(1, last_count + 1):
                        previous = float(cost[count, start, start_state])
                        if not np.isfinite(previous):
                            continue
                        candidate = previous + float(edge.cost)
                        current = float(cost[count + 1, end, end_state])
                        old_parent = (
                            int(parent_frame[count + 1, end, end_state]),
                            int(parent_state[count + 1, end, end_state]),
                        )
                        if candidate < current - _EPSILON or (
                            abs(candidate - current) <= _EPSILON
                            and (int(start), int(start_state)) < old_parent
                        ):
                            cost[count + 1, end, end_state] = candidate
                            parent_frame[count + 1, end, end_state] = int(start)
                            parent_state[count + 1, end, end_state] = int(start_state)

    selected_count = -1
    selected_state = -1
    count_order = (
        *range(int(target_count), bounded_maximum + 1),
        *range(int(target_count) - 1, 0, -1),
    )
    for count in count_order:
        state = int(np.argmin(cost[count, -1]))
        if np.isfinite(cost[count, -1, state]):
            selected_count = int(count)
            selected_state = int(state)
            break
    if selected_state < 0:
        raise RuntimeError("no hard-Recall-feasible fixed-cardinality curve path")
    frames: list[int] = []
    states: list[int] = []
    frame = frame_count - 1
    state = selected_state
    count = selected_count
    while count > 0:
        frames.append(int(frame))
        states.append(int(state))
        if count == 1:
            break
        next_frame = int(parent_frame[count, frame, state])
        next_state = int(parent_state[count, frame, state])
        if next_frame < 0 or next_state < 0:
            raise RuntimeError("broken fixed-cardinality curve predecessor chain")
        frame, state = next_frame, next_state
        count -= 1
    frames.reverse()
    states.reverse()
    return _NodePath(
        tuple(frames),
        tuple(states),
        float(cost[selected_count, -1, selected_state]),
    )


def _select_path(
    frame_count: int,
    state_count: int,
    evaluator: _MultistateIntervalEvaluator,
    target_count: int,
    config: KeyframeDpConfig,
    first_losses: np.ndarray,
) -> tuple[_NodePath, float]:
    if str(config.path_selection_mode) == "fixed_cardinality":
        maximum_count = min(
            int(frame_count),
            max(
                int(target_count),
                int(
                    np.ceil(
                        float(target_count) * float(config.cardinality_maximum_factor)
                    )
                ),
                int(target_count) + 8,
            ),
        )
        path = evaluator.decode_cardinality_native(
            int(frame_count),
            int(state_count),
            first_losses,
            int(target_count),
            int(maximum_count),
        )
        if path is None:
            # The compact probe palette can require more than the bounded K
            # range. Decode its minimum-cardinality path so the caller can
            # switch to the wider state palette instead of failing before the
            # fallback has a chance to run.
            path = evaluator.decode_native(
                int(frame_count),
                int(state_count),
                first_losses,
                max(float(config.penalty_maximum), 1.0e12),
            )
            if path is None:
                path = _decode_fixed_cardinality(
                    int(frame_count),
                    int(state_count),
                    evaluator,
                    int(target_count),
                    int(frame_count),
                    int(config.maximum_gap),
                    first_losses,
                )
        return path, 0.0

    candidates: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[float, _NodePath]
    ] = {}

    def evaluate(penalty: float) -> _NodePath:
        path = _decode(
            frame_count,
            state_count,
            evaluator,
            float(penalty),
            int(config.maximum_gap),
            first_losses,
        )
        key = (path.frames, path.states)
        existing = candidates.get(key)
        if existing is None or path.raw_cost < existing[1].raw_cost:
            candidates[key] = (float(penalty), path)
        return path

    low = 0.0
    high = float(config.penalty_maximum)
    evaluate(low)
    evaluate(high)
    for _step in range(int(config.penalty_binary_steps)):
        middle = 0.5 * (low + high)
        path = evaluate(middle)
        if len(path.frames) > int(target_count):
            low = middle
        else:
            high = middle
    penalty, path = min(
        candidates.values(),
        key=lambda value: (
            abs(len(value[1].frames) - int(target_count)),
            value[1].raw_cost,
            len(value[1].frames),
            value[1].frames,
            value[1].states,
        ),
    )
    return path, float(penalty)


def optimize_multistate_keyframes(
    references: list[np.ndarray],
    state_controls: np.ndarray,
    *,
    state_labels: tuple[str, ...],
    base_controls: np.ndarray,
    representation: str,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    point_refine: CurvePointRefineConfig = CurvePointRefineConfig(),
    interval_renderer: BoundaryRenderer | None = None,
    fallback_state_controls: np.ndarray | None = None,
    fallback_state_labels: tuple[str, ...] | None = None,
    fast_state_target_ratio: float = 0.0,
    fast_state_quality_probe: bool = True,
) -> KeyframeDpResult:
    """Optimize frame, state and P jointly under exact per-frame Recall."""
    config.validate()
    controls = np.asarray(state_controls, dtype=np.float64)
    if controls.ndim != 4 or controls.shape[3] != 2:
        raise ValueError("state_controls must have shape (frames, states, points, 2)")
    if len(references) != len(controls) or len(state_labels) != controls.shape[1]:
        raise ValueError("multistate controls/labels/reference dimensions disagree")
    base = np.asarray(base_controls, dtype=np.float64)
    if base.shape != controls[:, 0].shape:
        raise ValueError("base_controls shape does not match one state")
    fallback_controls = None
    fallback_labels = None
    if fallback_state_controls is not None:
        fallback_controls = np.asarray(fallback_state_controls, dtype=np.float64)
        fallback_labels = tuple(fallback_state_labels or ())
        if (
            fallback_controls.ndim != 4
            or fallback_controls.shape[0] != controls.shape[0]
            or fallback_controls.shape[2:] != controls.shape[2:]
            or len(fallback_labels) != fallback_controls.shape[1]
        ):
            raise ValueError("fallback curve states/labels do not match fast states")
        if not 0.0 < float(fast_state_target_ratio) <= 1.0:
            raise ValueError("fast_state_target_ratio must be in (0, 1]")
    started = time.perf_counter()
    exact_raster = None
    if bool(config.native_cpu_batches):
        try:
            exact_raster = create_exact_raster_batch(
                references,
                maximum_cache_bytes=int(config.native_reference_cache_bytes),
                maximum_batch_cases=int(config.native_batch_cases),
            )
        except RuntimeError:
            exact_raster = None
    graph_renderer = renderer if interval_renderer is None else interval_renderer
    target_count = max(
        2, min(len(controls), int(round(len(controls) / config.target_interval)))
    )
    initial_state_count = int(controls.shape[1])
    state_search_fallback = False
    probe_started = time.perf_counter()

    def solve_graph(
        candidates: np.ndarray,
        seed_evaluator: _MultistateIntervalEvaluator | None = None,
    ) -> tuple[_MultistateIntervalEvaluator, _NodePath, float]:
        candidate_evaluator = _MultistateIntervalEvaluator(
            references,
            candidates,
            graph_renderer,
            config,
            exact_raster,
            seed_evaluator,
        )
        candidate_evaluator.precompute_all(int(config.maximum_gap))
        candidate_first_losses = _first_node_losses(
            references,
            candidates,
            renderer,
            config,
        )
        candidate_path, candidate_penalty = _select_path(
            len(candidates),
            candidates.shape[1],
            candidate_evaluator,
            target_count,
            config,
            candidate_first_losses,
        )
        return candidate_evaluator, candidate_path, candidate_penalty

    evaluator, path, penalty = solve_graph(controls)
    probe_seconds = 0.0
    if fallback_controls is not None:
        maximum_fast_keys = int(
            np.floor(
                len(controls)
                / (float(config.target_interval) * float(fast_state_target_ratio))
                + _EPSILON
            )
        )
        use_full_palette = len(path.frames) > max(2, maximum_fast_keys)
        if not use_full_palette and bool(fast_state_quality_probe):
            provisional_controls = np.asarray(
                [
                    controls[frame, state]
                    for frame, state in zip(path.frames, path.states, strict=True)
                ],
                dtype=np.float64,
            )
            provisional_dense = materialize_controls(
                len(controls),
                path.frames,
                provisional_controls,
            )
            _boundaries, provisional_audit = audit_dense_path(
                references,
                provisional_dense,
                renderer,
                exact_raster=exact_raster,
                threads=int(config.native_cpu_threads),
            )
            use_full_palette = bool(
                np.min(provisional_audit.iou)
                < float(config.quality_rescue_iou_floor) - _EPSILON
                or np.max(provisional_audit.area_ratio)
                > float(config.quality_rescue_area_ratio_cap) + _EPSILON
            )
        if use_full_palette:
            probe_seconds = float(time.perf_counter() - probe_started)
            controls = fallback_controls
            state_labels = fallback_labels or ()
            evaluator, path, penalty = solve_graph(
                controls,
                evaluator,
            )
            state_search_fallback = True
    if graph_renderer is not renderer:
        selected_controls = np.asarray(
            [
                controls[frame, state]
                for frame, state in zip(path.frames, path.states, strict=True)
            ],
            dtype=np.float64,
        )
        selected_dense = materialize_controls(
            len(controls),
            path.frames,
            selected_controls,
        )
        _selected_boundaries, selected_audit = audit_dense_path(
            references,
            selected_dense,
            renderer,
            exact_raster=exact_raster,
            threads=int(config.native_cpu_threads),
        )
        if _valid_audit_losses(selected_audit, config) is None:
            # The coarse curve is only a CPU screen.  It may never weaken the
            # hard Recall/topology contract: rerun the complete exact graph if
            # its selected path does not pass the authoritative renderer.
            evaluator = _MultistateIntervalEvaluator(
                references,
                controls,
                renderer,
                config,
                exact_raster,
            )
            evaluator.precompute_all(int(config.maximum_gap))
            exact_first_losses = _first_node_losses(
                references,
                controls,
                renderer,
                config,
            )
            path, penalty = _select_path(
                len(controls),
                controls.shape[1],
                evaluator,
                target_count,
                config,
                exact_first_losses,
            )
    chosen_frames = path.frames
    chosen_states = path.states
    key_controls = np.asarray(
        [controls[frame, state] for frame, state in zip(path.frames, path.states)],
        dtype=np.float64,
    )
    rescue_inserted: tuple[int, ...] = ()
    if bool(config.quality_rescue_enabled):
        chosen_frames, key_controls, rescue_inserted = _quality_rescue_keys(
            references,
            base,
            chosen_frames,
            renderer,
            config,
            initial_key_controls=key_controls,
            exact_raster=exact_raster,
        )
        state_by_frame = dict(zip(path.frames, path.states))
        chosen_states = tuple(
            int(state_by_frame.get(frame, -1)) for frame in chosen_frames
        )
        if (
            fallback_controls is not None
            and not state_search_fallback
            and rescue_inserted
        ):
            # Rescue keys indicate that the compact palette reached the target
            # only by leaving a material local-quality debt.  Re-solve with the
            # full state palette before the expensive coordinate refinements.
            probe_seconds = float(time.perf_counter() - probe_started)
            controls = fallback_controls
            state_labels = fallback_labels or ()
            evaluator, path, penalty = solve_graph(
                controls,
                evaluator,
            )
            state_search_fallback = True
            chosen_frames = path.frames
            chosen_states = path.states
            key_controls = np.asarray(
                [
                    controls[frame, state]
                    for frame, state in zip(path.frames, path.states, strict=True)
                ],
                dtype=np.float64,
            )
            rescue_inserted = ()
            chosen_frames, key_controls, rescue_inserted = _quality_rescue_keys(
                references,
                base,
                chosen_frames,
                renderer,
                config,
                initial_key_controls=key_controls,
                exact_raster=exact_raster,
            )
            state_by_frame = dict(zip(path.frames, path.states))
            chosen_states = tuple(
                int(state_by_frame.get(frame, -1)) for frame in chosen_frames
            )
    refinement_baseline_controls = np.asarray(key_controls, dtype=np.float64).copy()
    dp_finished = time.perf_counter()
    pair_trials = 0
    pair_accepted = 0
    pair_gain = 0.0
    if bool(config.pair_vote_enabled) and len(chosen_frames) > 1:
        key_controls, pair_trials, pair_accepted, pair_gain = _refine_pair_vote(
            references,
            base,
            chosen_frames,
            renderer,
            config,
            initial_key_controls=key_controls,
            exact_raster=exact_raster,
        )
    pair_finished = time.perf_counter()
    point_controls, point_trials, point_accepted, point_gain = _refine_curve_points(
        references,
        chosen_frames,
        key_controls,
        renderer,
        config,
        point_refine,
        exact_raster,
    )
    key_controls = point_controls
    point_finished = time.perf_counter()
    (
        key_controls,
        dense,
        boundaries,
        audit,
        refinement_guard_triggered,
        refinement_guard_alpha,
    ) = _guard_refined_controls(
        references,
        chosen_frames,
        refinement_baseline_controls,
        key_controls,
        renderer,
        config,
        exact_raster,
    )
    violations = int(
        np.count_nonzero(audit.recall + _EPSILON < float(config.recall_floor))
    )
    if violations or np.any(~audit.topology_valid):
        raise RuntimeError(
            f"final multistate audit failed: recall={violations}, "
            f"topology={np.count_nonzero(~audit.topology_valid)}"
        )
    finished = time.perf_counter()
    return KeyframeDpResult(
        representation=str(representation),
        chosen_indices=chosen_frames,
        keyframe_controls=key_controls,
        dense_controls=dense,
        dense_boundaries=boundaries,
        audit=audit,
        target_keyframes=int(target_count),
        target_interval=int(config.target_interval),
        effective_interval=float(len(controls) / max(len(chosen_frames), 1)),
        selected_lambda=float(penalty),
        edge_evaluations=int(
            evaluator._native_edge_count
            if evaluator.native_batch_precomputed
            else len(evaluator.cache)
        ),
        edge_frame_evaluations=int(evaluator.frame_evaluations),
        pair_vote_trials=int(pair_trials),
        pair_vote_accepted=int(pair_accepted),
        pair_vote_iou_gain=float(pair_gain),
        point_refine_trials=int(point_trials),
        point_refine_accepted=int(point_accepted),
        point_refine_iou_gain=float(point_gain),
        quality_rescue_inserted_indices=rescue_inserted,
        dp_seconds=float(dp_finished - started),
        pair_vote_seconds=float(pair_finished - dp_finished),
        point_refine_seconds=float(point_finished - pair_finished),
        final_audit_seconds=float(finished - point_finished),
        elapsed_seconds=float(finished - started),
        native_reference_cache=(
            {} if exact_raster is None else exact_raster.cache_stats()
        ),
        config=config,
        chosen_state_indices=chosen_states,
        chosen_state_labels=tuple(
            state_labels[state] if state >= 0 else "quality_rescue"
            for state in chosen_states
        ),
        state_search_initial_count=int(initial_state_count),
        state_search_final_count=int(controls.shape[1]),
        state_search_fallback=bool(state_search_fallback),
        state_search_probe_seconds=float(probe_seconds),
        native_graph_reused_edges=int(evaluator._native_reused_edge_count),
        native_shape_distance_unique_pairs=int(evaluator._shape_distance_unique_pairs),
        native_lazy_topology_decodes=int(evaluator._lazy_topology_decodes),
        native_lazy_topology_checked_edges=int(evaluator._lazy_topology_checked_edges),
        native_lazy_topology_checked_frames=int(
            evaluator._lazy_topology_checked_frames
        ),
        native_lazy_topology_rejected_edges=int(
            evaluator._lazy_topology_rejected_edges
        ),
        refinement_guard_triggered=bool(refinement_guard_triggered),
        refinement_guard_alpha=float(refinement_guard_alpha),
    )


__all__ = (
    "CurvePointRefineConfig",
    "isotropic_curve_states",
    "optimize_multistate_keyframes",
)
