"""Hard-minimum-Recall keyframe DP shared by polygon and curve reviews.

The graph and key penalty follow the Production optimizer's semantics.  Only
the endpoint representation is injected: a polygon renderer connects P with
lines, while the curve renderer derives Catmull--Rom Bezier handles from P at
every interpolated frame.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from production.polygon.runtime.spatial_support.optimizer import (
    temporal_residuals,
)

from .model import sample_closed_curve, sample_curve_sequence
from .native_cpu import ExactDoubleRasterBatch, create_exact_raster_batch
from .topology import has_strict_self_intersection, strict_self_intersection_batch


BoundaryRenderer = Callable[[np.ndarray], np.ndarray]
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class KeyframeDpConfig:
    target_interval: int = 6
    recall_floor: float = 0.97
    maximum_gap: int = 30
    penalty_binary_steps: int = 18
    penalty_maximum: float = 1024.0
    shape_distance_weight: float = 0.4
    pair_vote_enabled: bool = True
    pair_vote_sweeps: int = 2
    pair_vote_global_steps: int = 32
    pair_vote_local_steps: int = 8
    low_iou_quadratic_weight: float = 0.0
    quality_rescue_enabled: bool = False
    quality_rescue_iou_floor: float = 0.70
    quality_rescue_regret_floor: float = 0.12
    quality_rescue_area_ratio_cap: float = 1.35
    quality_rescue_maximum_extra_keys: int = 4
    quality_rescue_density_budget: bool = True
    quality_rescue_relaxation_sweeps: int = 2
    quality_rescue_relaxation_steps: int = 16
    quality_rescue_maximum_iou_regression: float = 0.005
    quality_rescue_maximum_area_ratio_regression: float = 0.01
    native_cpu_batches: bool = True
    native_cpu_threads: int = 8
    native_batch_cases: int = 4096
    native_reference_cache_bytes: int = 256 * 1024 * 1024

    def validate(self) -> None:
        if int(self.target_interval) < 1:
            raise ValueError("target_interval must be positive")
        if not 0.0 < float(self.recall_floor) <= 1.0:
            raise ValueError("recall_floor must be in (0, 1]")
        if int(self.maximum_gap) < 1:
            raise ValueError("maximum_gap must be positive")
        if int(self.penalty_binary_steps) < 1:
            raise ValueError("penalty_binary_steps must be positive")
        if int(self.pair_vote_sweeps) < 0:
            raise ValueError("pair_vote_sweeps must be nonnegative")
        if float(self.low_iou_quadratic_weight) < 0.0:
            raise ValueError("low_iou_quadratic_weight must be nonnegative")
        if not 0.0 <= float(self.quality_rescue_iou_floor) <= 1.0:
            raise ValueError("quality_rescue_iou_floor must be in [0, 1]")
        if float(self.quality_rescue_regret_floor) < 0.0:
            raise ValueError("quality_rescue_regret_floor must be nonnegative")
        if float(self.quality_rescue_area_ratio_cap) < 1.0:
            raise ValueError("quality_rescue_area_ratio_cap must be at least one")
        if int(self.quality_rescue_maximum_extra_keys) < 0:
            raise ValueError("quality_rescue_maximum_extra_keys must be nonnegative")
        if int(self.quality_rescue_relaxation_sweeps) < 0:
            raise ValueError("quality_rescue_relaxation_sweeps must be nonnegative")
        if int(self.quality_rescue_relaxation_steps) < 1:
            raise ValueError("quality_rescue_relaxation_steps must be positive")
        if float(self.quality_rescue_maximum_iou_regression) < 0.0:
            raise ValueError("quality rescue IoU regression must be nonnegative")
        if float(self.quality_rescue_maximum_area_ratio_regression) < 0.0:
            raise ValueError("quality rescue area regression must be nonnegative")
        if int(self.native_cpu_threads) < 1:
            raise ValueError("native_cpu_threads must be positive")
        if int(self.native_batch_cases) < 1:
            raise ValueError("native_batch_cases must be positive")
        if int(self.native_reference_cache_bytes) < 0:
            raise ValueError("native_reference_cache_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class RasterAudit:
    iou: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    area_ratio: np.ndarray
    topology_valid: np.ndarray


@dataclass(frozen=True, slots=True)
class EdgeEvaluation:
    cost: float
    frames_covered: int
    minimum_recall: float
    mean_iou: float
    topology_valid: bool

    @property
    def feasible(self) -> bool:
        return bool(np.isfinite(self.cost))


@dataclass(frozen=True, slots=True)
class KeyframeDpResult:
    representation: str
    chosen_indices: tuple[int, ...]
    keyframe_controls: np.ndarray
    dense_controls: np.ndarray
    dense_boundaries: np.ndarray
    audit: RasterAudit
    target_keyframes: int
    target_interval: int
    effective_interval: float
    selected_lambda: float
    edge_evaluations: int
    edge_frame_evaluations: int
    pair_vote_trials: int
    pair_vote_accepted: int
    pair_vote_iou_gain: float
    point_refine_trials: int
    point_refine_accepted: int
    point_refine_iou_gain: float
    quality_rescue_inserted_indices: tuple[int, ...]
    dp_seconds: float
    pair_vote_seconds: float
    point_refine_seconds: float
    final_audit_seconds: float
    elapsed_seconds: float
    native_reference_cache: dict[str, int]
    config: KeyframeDpConfig
    chosen_state_indices: tuple[int, ...] = ()
    chosen_state_labels: tuple[str, ...] = ()
    state_search_initial_count: int = 0
    state_search_final_count: int = 0
    state_search_fallback: bool = False
    state_search_probe_seconds: float = 0.0
    native_graph_reused_edges: int = 0
    native_shape_distance_unique_pairs: int = 0
    native_lazy_topology_decodes: int = 0
    native_lazy_topology_checked_edges: int = 0
    native_lazy_topology_checked_frames: int = 0
    native_lazy_topology_rejected_edges: int = 0
    refinement_guard_triggered: bool = False
    refinement_guard_alpha: float = 1.0

    def summary(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "config": asdict(self.config),
            "frames": int(len(self.dense_controls)),
            "control_points": int(self.dense_controls.shape[1]),
            "target_keyframes": int(self.target_keyframes),
            "chosen_keyframes": int(len(self.chosen_indices)),
            "chosen_indices": list(self.chosen_indices),
            "chosen_state_indices": list(self.chosen_state_indices),
            "chosen_state_labels": list(self.chosen_state_labels),
            "state_search_initial_count": int(self.state_search_initial_count),
            "state_search_final_count": int(self.state_search_final_count),
            "state_search_fallback": bool(self.state_search_fallback),
            "state_search_probe_seconds": float(self.state_search_probe_seconds),
            "native_graph_reused_edges": int(self.native_graph_reused_edges),
            "native_shape_distance_unique_pairs": int(
                self.native_shape_distance_unique_pairs
            ),
            "native_lazy_topology_decodes": int(self.native_lazy_topology_decodes),
            "native_lazy_topology_checked_edges": int(
                self.native_lazy_topology_checked_edges
            ),
            "native_lazy_topology_checked_frames": int(
                self.native_lazy_topology_checked_frames
            ),
            "native_lazy_topology_rejected_edges": int(
                self.native_lazy_topology_rejected_edges
            ),
            "refinement_guard_triggered": bool(self.refinement_guard_triggered),
            "refinement_guard_alpha": float(self.refinement_guard_alpha),
            "target_interval": int(self.target_interval),
            "effective_interval": float(self.effective_interval),
            "selected_lambda": float(self.selected_lambda),
            "mean_iou": float(np.mean(self.audit.iou)),
            "minimum_iou": float(np.min(self.audit.iou)),
            "q01_iou": float(np.quantile(self.audit.iou, 0.01)),
            "q05_iou": float(np.quantile(self.audit.iou, 0.05)),
            "mean_recall": float(np.mean(self.audit.recall)),
            "minimum_recall": float(np.min(self.audit.recall)),
            "recall_violations": int(
                np.count_nonzero(
                    self.audit.recall + _EPSILON < float(self.config.recall_floor)
                )
            ),
            "mean_precision": float(np.mean(self.audit.precision)),
            "area_ratio_mean": float(np.mean(self.audit.area_ratio)),
            "area_ratio_q95": float(np.quantile(self.audit.area_ratio, 0.95)),
            "area_ratio_maximum": float(np.max(self.audit.area_ratio)),
            "topology_invalid_frames": int(
                np.count_nonzero(~self.audit.topology_valid)
            ),
            "temporal_control_residual": float(
                np.mean(temporal_residuals(self.dense_controls))
                if len(self.dense_controls) > 1
                else 0.0
            ),
            "edge_evaluations": int(self.edge_evaluations),
            "edge_frame_evaluations": int(self.edge_frame_evaluations),
            "pair_vote_trials": int(self.pair_vote_trials),
            "pair_vote_accepted": int(self.pair_vote_accepted),
            "pair_vote_iou_gain": float(self.pair_vote_iou_gain),
            "point_refine_trials": int(self.point_refine_trials),
            "point_refine_accepted": int(self.point_refine_accepted),
            "point_refine_iou_gain": float(self.point_refine_iou_gain),
            "quality_rescue_inserted": int(len(self.quality_rescue_inserted_indices)),
            "quality_rescue_inserted_indices": list(
                self.quality_rescue_inserted_indices
            ),
            "dp_seconds": float(self.dp_seconds),
            "pair_vote_seconds": float(self.pair_vote_seconds),
            "point_refine_seconds": float(self.point_refine_seconds),
            "final_audit_seconds": float(self.final_audit_seconds),
            "elapsed_seconds": float(self.elapsed_seconds),
            "native_reference_cache": dict(self.native_reference_cache),
        }


def polygon_renderer(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points, dtype=np.float64)


def catmull_rom_renderer(samples_per_segment: int) -> BoundaryRenderer:
    samples = int(samples_per_segment)
    if samples < 2:
        raise ValueError("samples_per_segment must be at least two")

    def render(points: np.ndarray) -> np.ndarray:
        return sample_closed_curve(points, samples)

    setattr(render, "_catmull_samples_per_segment", samples)
    return render


def render_control_sequence(
    renderer: BoundaryRenderer,
    controls: np.ndarray,
) -> np.ndarray:
    """Render control arrays while preserving scalar Catmull arithmetic."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim < 3 or values.shape[-1] != 2:
        raise ValueError("controls must end in (points, 2)")
    leading = values.shape[:-2]
    flattened = values.reshape(-1, values.shape[-2], 2)
    samples = getattr(renderer, "_catmull_samples_per_segment", None)
    if samples is None:
        rendered = np.asarray([renderer(points) for points in flattened])
    else:
        rendered = sample_curve_sequence(flattened, int(samples))
    return np.ascontiguousarray(
        rendered.reshape(*leading, rendered.shape[-2], 2),
        dtype=np.float64,
    )


def interpolate_controls(
    start: np.ndarray, end: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate P first; renderers derive their geometry afterwards."""
    return np.ascontiguousarray(
        (1.0 - float(alpha)) * np.asarray(start, dtype=np.float64)
        + float(alpha) * np.asarray(end, dtype=np.float64),
        dtype=np.float64,
    )


def _raster_counts(
    reference: np.ndarray, candidate: np.ndarray, padding: int = 2
) -> tuple[int, int, int, int]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
    minimum = np.floor(np.minimum(np.min(left, axis=0), np.min(right, axis=0))).astype(
        np.int64
    ) - int(padding)
    maximum = np.ceil(np.maximum(np.max(left, axis=0), np.max(right, axis=0))).astype(
        np.int64
    ) + int(padding)
    width = max(1, int(maximum[0] - minimum[0] + 1))
    height = max(1, int(maximum[1] - minimum[1] + 1))
    reference_mask = np.zeros((height, width), dtype=np.uint8)
    candidate_mask = np.zeros_like(reference_mask)
    cv2.fillPoly(
        reference_mask,
        [np.rint(left - minimum[None, :]).astype(np.int32)],
        1,
    )
    cv2.fillPoly(
        candidate_mask,
        [np.rint(right - minimum[None, :]).astype(np.int32)],
        1,
    )
    reference_area = int(cv2.countNonZero(reference_mask))
    candidate_area = int(cv2.countNonZero(candidate_mask))
    intersection = int(
        cv2.countNonZero(cv2.bitwise_and(reference_mask, candidate_mask))
    )
    union = int(reference_area + candidate_area - intersection)
    return reference_area, candidate_area, intersection, union


def _frame_metrics(
    reference: np.ndarray, boundary: np.ndarray
) -> tuple[float, float, float, float]:
    reference_area, candidate_area, intersection, union = _raster_counts(
        reference, boundary
    )
    iou = float(intersection / union) if union else 1.0
    recall = float(intersection / reference_area) if reference_area else 1.0
    precision = float(intersection / candidate_area) if candidate_area else 1.0
    area_ratio = float(candidate_area / reference_area) if reference_area else 1.0
    return iou, recall, precision, area_ratio


def materialize_controls(
    frame_count: int,
    chosen_indices: tuple[int, ...] | list[int],
    keyframe_controls: np.ndarray,
) -> np.ndarray:
    chosen = tuple(int(value) for value in chosen_indices)
    keys = np.asarray(keyframe_controls, dtype=np.float64)
    if len(chosen) != len(keys) or not chosen:
        raise ValueError(
            "chosen indices and keyframe controls must be nonempty and equal"
        )
    output = np.empty((int(frame_count), *keys.shape[1:]), dtype=np.float64)
    interval = 0
    for frame in range(int(frame_count)):
        if frame <= chosen[0]:
            output[frame] = keys[0]
            continue
        if frame >= chosen[-1]:
            output[frame] = keys[-1]
            continue
        while interval + 1 < len(chosen) and frame > chosen[interval + 1]:
            interval += 1
        right = interval + 1
        left_frame = chosen[interval]
        right_frame = chosen[right]
        alpha = float((frame - left_frame) / max(right_frame - left_frame, 1))
        output[frame] = interpolate_controls(keys[interval], keys[right], alpha)
    return output


def audit_dense_path(
    references: list[np.ndarray],
    dense_controls: np.ndarray,
    renderer: BoundaryRenderer,
    *,
    exact_raster: ExactDoubleRasterBatch | None = None,
    frame_offset: int = 0,
    threads: int = 1,
) -> tuple[np.ndarray, RasterAudit]:
    boundaries = render_control_sequence(renderer, dense_controls)
    if exact_raster is None:
        values = np.asarray(
            [
                _frame_metrics(reference, boundary)
                for reference, boundary in zip(references, boundaries, strict=True)
            ],
            dtype=np.float64,
        )
    else:
        native = exact_raster.metrics(
            np.arange(
                int(frame_offset),
                int(frame_offset) + len(boundaries),
                dtype=np.int32,
            ),
            boundaries,
            threads=max(1, int(threads)),
        )
        area_ratio = np.ones((len(native),), dtype=np.float64)
        nonempty = native[:, 0] > 0.0
        area_ratio[nonempty] = native[nonempty, 1] / native[nonempty, 0]
        values = np.column_stack((native[:, 6], native[:, 4], native[:, 5], area_ratio))
    topology = ~strict_self_intersection_batch(
        boundaries,
        threads=max(1, int(threads)),
    )
    return boundaries, RasterAudit(
        iou=np.ascontiguousarray(values[:, 0], dtype=np.float64),
        recall=np.ascontiguousarray(values[:, 1], dtype=np.float64),
        precision=np.ascontiguousarray(values[:, 2], dtype=np.float64),
        area_ratio=np.ascontiguousarray(values[:, 3], dtype=np.float64),
        topology_valid=topology,
    )


class _IntervalEvaluator:
    def __init__(
        self,
        references: list[np.ndarray],
        frame_controls: np.ndarray,
        renderer: BoundaryRenderer,
        config: KeyframeDpConfig,
    ) -> None:
        self.references = references
        self.controls = np.asarray(frame_controls, dtype=np.float64)
        self.renderer = renderer
        self.config = config
        self.cache: dict[tuple[int, int], EdgeEvaluation] = {}
        self.frame_evaluations = 0

    def edge(self, start: int, end: int) -> EdgeEvaluation:
        key = (int(start), int(end))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if end <= start:
            raise ValueError("edge end must be after start")
        ious: list[float] = []
        recalls: list[float] = []
        span = int(end - start)
        frames = tuple(range(start + 1, end + 1))
        boundaries = np.asarray(
            [
                self.renderer(
                    interpolate_controls(
                        self.controls[start],
                        self.controls[end],
                        float((frame - start) / span),
                    )
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        self.frame_evaluations += len(frames)
        topology_valid = not bool(np.any(strict_self_intersection_batch(boundaries)))
        for frame, boundary in zip(frames, boundaries, strict=True):
            if not topology_valid:
                break
            iou, recall, _precision, _area_ratio = _frame_metrics(
                self.references[frame], boundary
            )
            ious.append(iou)
            recalls.append(recall)
            if recall + _EPSILON < float(self.config.recall_floor):
                break
        feasible = bool(
            topology_valid
            and len(ious) == span
            and min(recalls, default=0.0) + _EPSILON >= float(self.config.recall_floor)
        )
        shape_residual = temporal_residuals(
            np.stack((self.controls[start], self.controls[end]), axis=0)
        )
        shape_distance = float(np.mean(shape_residual)) if shape_residual.size else 0.0
        losses = 1.0 - np.asarray(ious, dtype=np.float64)
        cost = (
            float(
                np.sum(
                    losses
                    + float(self.config.low_iou_quadratic_weight) * losses * losses
                )
            )
            + float(self.config.shape_distance_weight) * shape_distance
            if feasible
            else float("inf")
        )
        result = EdgeEvaluation(
            cost=cost,
            frames_covered=len(ious),
            minimum_recall=float(min(recalls, default=0.0)),
            mean_iou=float(np.mean(ious)) if ious else 0.0,
            topology_valid=topology_valid,
        )
        self.cache[key] = result
        return result


def _decode(
    frame_count: int,
    evaluator: _IntervalEvaluator,
    key_penalty: float,
    maximum_gap: int,
    first_loss: float,
) -> tuple[tuple[int, ...], float]:
    cost = np.full((frame_count,), np.inf, dtype=np.float64)
    raw_cost = np.full_like(cost, np.inf)
    parent = np.full((frame_count,), -1, dtype=np.int32)
    cost[0] = float(first_loss) + float(key_penalty)
    raw_cost[0] = float(first_loss)
    for end in range(1, frame_count):
        for start in range(max(0, end - int(maximum_gap)), end):
            if not np.isfinite(cost[start]):
                continue
            edge = evaluator.edge(start, end)
            if not edge.feasible:
                continue
            candidate = float(cost[start] + edge.cost + key_penalty)
            candidate_raw = float(raw_cost[start] + edge.cost)
            if candidate < cost[end] - 1e-12 or (
                abs(candidate - cost[end]) <= 1e-12 and candidate_raw < raw_cost[end]
            ):
                cost[end] = candidate
                raw_cost[end] = candidate_raw
                parent[end] = int(start)
    if not np.isfinite(cost[-1]):
        raise RuntimeError("no hard-Recall-feasible keyframe path")
    selected = []
    cursor = frame_count - 1
    while cursor >= 0:
        selected.append(int(cursor))
        cursor = int(parent[cursor])
    selected.reverse()
    return tuple(selected), float(raw_cost[-1])


def _select_penalty_path(
    frame_count: int,
    evaluator: _IntervalEvaluator,
    target_count: int,
    config: KeyframeDpConfig,
    first_loss: float,
) -> tuple[tuple[int, ...], float]:
    candidates: dict[tuple[int, ...], tuple[float, float]] = {}

    def evaluate(penalty: float) -> tuple[int, ...]:
        path, raw_cost = _decode(
            frame_count,
            evaluator,
            penalty,
            int(config.maximum_gap),
            first_loss,
        )
        existing = candidates.get(path)
        if existing is None or raw_cost < existing[1]:
            candidates[path] = (float(penalty), float(raw_cost))
        return path

    low = 0.0
    high = float(config.penalty_maximum)
    evaluate(low)
    evaluate(high)
    for _ in range(int(config.penalty_binary_steps)):
        middle = 0.5 * (low + high)
        path = evaluate(middle)
        if len(path) > int(target_count):
            low = middle
        else:
            high = middle
    path, (penalty, _raw) = min(
        candidates.items(),
        key=lambda item: (
            abs(len(item[0]) - int(target_count)),
            item[1][1],
            len(item[0]),
            item[0],
        ),
    )
    return path, float(penalty)


def _pair_vote_targets(
    full_controls: np.ndarray, chosen: tuple[int, ...]
) -> np.ndarray:
    base = np.asarray(full_controls[list(chosen)], dtype=np.float64)
    proposals: list[list[tuple[np.ndarray, float]]] = [[] for _ in chosen]
    for position in range(len(chosen) - 1):
        left = int(chosen[position])
        right = int(chosen[position + 1])
        span = max(right - left, 1)
        x = np.asarray(
            [
                [(right - frame) / span, (frame - left) / span]
                for frame in range(left, right + 1)
            ],
            dtype=np.float64,
        )
        y = np.asarray(full_controls[left : right + 1], dtype=np.float64).reshape(
            right - left + 1, -1
        )
        gram = x.T @ x
        endpoints = np.linalg.solve(gram + 1e-8 * np.eye(2), x.T @ y)
        weight = float(right - left + 1)
        proposals[position].append((endpoints[0].reshape(base.shape[1:]), weight))
        proposals[position + 1].append((endpoints[1].reshape(base.shape[1:]), weight))
    output = base.copy()
    for position, values in enumerate(proposals):
        if values:
            total = sum(weight for _value, weight in values)
            output[position] = sum(value * weight for value, weight in values) / max(
                total, 1e-12
            )
    return output


def _valid_path_score(
    audit: RasterAudit,
    config: KeyframeDpConfig,
) -> float | None:
    if np.any(~audit.topology_valid) or np.any(
        audit.recall + _EPSILON < float(config.recall_floor)
    ):
        return None
    losses = 1.0 - audit.iou
    return float(
        -np.mean(losses + float(config.low_iou_quadratic_weight) * losses * losses)
    )


def _audit_losses(audit: RasterAudit, config: KeyframeDpConfig) -> np.ndarray:
    values = 1.0 - np.asarray(audit.iou, dtype=np.float64)
    return np.asarray(
        values + float(config.low_iou_quadratic_weight) * values * values,
        dtype=np.float64,
    )


def _valid_audit_losses(
    audit: RasterAudit,
    config: KeyframeDpConfig,
) -> np.ndarray | None:
    if np.any(~audit.topology_valid) or np.any(
        audit.recall + _EPSILON < float(config.recall_floor)
    ):
        return None
    return _audit_losses(audit, config)


def _valid_dense_trial_losses_batch(
    references: list[np.ndarray],
    dense_trials: np.ndarray,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    exact_raster: ExactDoubleRasterBatch | None,
    *,
    frame_offset: int = 0,
) -> list[np.ndarray | None]:
    """Score independent dense paths in one exact native CPU batch.

    Pair-vote trials do not depend on one another until a winner is selected.
    Batching only changes scheduling: trial order, float64 geometry, OpenCV
    raster semantics, hard Recall, and topology gates remain identical.
    """
    trials = np.asarray(dense_trials, dtype=np.float64)
    if trials.ndim != 4 or trials.shape[3] != 2:
        raise ValueError("dense_trials must have shape (trials,frames,points,2)")
    if exact_raster is None:
        output: list[np.ndarray | None] = []
        for trial in trials:
            _boundaries, audit = audit_dense_path(
                references,
                trial,
                renderer,
                threads=int(config.native_cpu_threads),
            )
            output.append(_valid_audit_losses(audit, config))
        return output

    trial_count, span = int(trials.shape[0]), int(trials.shape[1])
    rendered = render_control_sequence(renderer, trials)
    boundaries = rendered.reshape(-1, rendered.shape[-2], 2)
    threads = max(1, int(config.native_cpu_threads))
    topology = ~strict_self_intersection_batch(
        boundaries,
        threads=threads,
    ).reshape(trial_count, span)
    frames = np.tile(
        np.arange(
            int(frame_offset),
            int(frame_offset) + span,
            dtype=np.int32,
        ),
        trial_count,
    )
    metrics = exact_raster.metrics(
        frames,
        boundaries,
        threads=threads,
    ).reshape(trial_count, span, 7)
    output = []
    for index in range(trial_count):
        recall = metrics[index, :, 4]
        if np.any(~topology[index]) or np.any(
            recall + _EPSILON < float(config.recall_floor)
        ):
            output.append(None)
            continue
        loss = 1.0 - metrics[index, :, 6]
        output.append(
            np.asarray(
                loss + float(config.low_iou_quadratic_weight) * loss * loss,
                dtype=np.float64,
            )
        )
    return output


def _quality_rescue_keys(
    references: list[np.ndarray],
    full_controls: np.ndarray,
    chosen: tuple[int, ...],
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    initial_key_controls: np.ndarray | None = None,
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[tuple[int, ...], np.ndarray, tuple[int, ...]]:
    """Insert evidence-backed keys when soft target pressure breaks shape.

    Recall remains a hard constraint.  A frame is eligible only when its own
    spatial candidate is substantially better than the interpolated result
    and the latter has either low IoU or excessive area.  Thus this does not
    turn the target interval into a hard key count.  Production uses the
    current run length as the only insertion bound; experiments may opt into
    an explicit positive key cap.
    """
    active = tuple(int(value) for value in chosen)
    active_controls = np.asarray(
        full_controls[list(active)]
        if initial_key_controls is None
        else initial_key_controls,
        dtype=np.float64,
    )
    if len(active_controls) != len(active):
        raise ValueError("initial rescue controls do not match chosen keys")
    inserted: list[int] = []
    configured_maximum = int(config.quality_rescue_maximum_extra_keys)
    # Zero is the Production-safe spelling of "no artificial quota".  The
    # finite bound below is simply the number of frames that are not already
    # keys, so termination and memory remain bounded by the current run/chunk.
    maximum_insertions = (
        max(0, len(full_controls) - len(active))
        if configured_maximum == 0
        else configured_maximum
    )
    if bool(config.quality_rescue_density_budget):
        target_count = max(
            2,
            min(
                len(full_controls),
                int(round(len(full_controls) / config.target_interval)),
            ),
        )
        effective_interval = float(len(full_controls) / max(len(active), 1))
        # A nearly per-frame DP path has almost no interpolation left to
        # rescue.  Spending the full fixed budget there merely turns a soft
        # target into every-frame keys.  Tie the rescue allowance to the
        # remaining interval slack while retaining the configured hard cap.
        density_allowance = max(
            0,
            int(
                np.floor(
                    float(target_count) * max(effective_interval - 1.0, 0.0) + _EPSILON
                )
            ),
        )
        maximum_insertions = min(maximum_insertions, density_allowance)
    _spatial_boundaries, spatial = audit_dense_path(
        references,
        full_controls,
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    spatial_feasible = np.logical_and(
        spatial.topology_valid,
        spatial.recall + _EPSILON >= float(config.recall_floor),
    )
    rejected: set[int] = set()
    refined_active: set[int] = set()
    minimum_iou_guard: np.ndarray | None = None
    maximum_area_guard: np.ndarray | None = None
    while True:
        dense = materialize_controls(len(full_controls), active, active_controls)
        _boundaries, audit = audit_dense_path(
            references,
            dense,
            renderer,
            exact_raster=exact_raster,
            threads=int(config.native_cpu_threads),
        )
        baseline_score = _valid_path_score(audit, config)
        if baseline_score is None:
            raise RuntimeError("quality rescue received an invalid DP path")
        if minimum_iou_guard is None:
            allowed_drop = float(config.quality_rescue_maximum_iou_regression)
            minimum_iou_guard = np.where(
                audit.iou < float(config.quality_rescue_iou_floor),
                audit.iou,
                np.maximum(
                    float(config.quality_rescue_iou_floor),
                    audit.iou - allowed_drop,
                ),
            )
            allowed_area_rise = float(
                config.quality_rescue_maximum_area_ratio_regression
            )
            maximum_area_guard = np.where(
                audit.area_ratio > float(config.quality_rescue_area_ratio_cap),
                audit.area_ratio,
                np.minimum(
                    float(config.quality_rescue_area_ratio_cap),
                    audit.area_ratio + allowed_area_rise,
                ),
            )
        baseline_losses = _audit_losses(audit, config)
        regrets = spatial.iou - audit.iou
        eligible = np.logical_and(
            spatial_feasible,
            regrets + _EPSILON >= float(config.quality_rescue_regret_floor),
        )
        eligible &= np.logical_or(
            audit.iou < float(config.quality_rescue_iou_floor),
            audit.area_ratio > float(config.quality_rescue_area_ratio_cap),
        )
        active_set = set(active)
        if len(inserted) >= maximum_insertions:
            # The key budget is exhausted, but a support/DP key already in
            # the path may itself be the bad frame.  Moving that existing key
            # costs no additional key and must remain possible.
            eligible &= np.asarray(
                [frame in active_set for frame in range(len(full_controls))],
                dtype=bool,
            )
        if refined_active:
            eligible[list(refined_active)] = False
        if rejected:
            eligible[list(rejected)] = False
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            break
        candidate = int(
            max(
                candidates,
                key=lambda frame: (
                    # Rescue is a lower-tail guard, not another global-mean
                    # optimizer.  Repair the largest explicit quality debt
                    # first; otherwise a frame with a large spatial regret but
                    # acceptable absolute IoU can consume the small key budget
                    # before the true worst frame is reached.
                    float(
                        max(
                            float(config.quality_rescue_iou_floor)
                            - float(audit.iou[frame]),
                            0.0,
                        )
                    ),
                    float(
                        max(
                            float(audit.area_ratio[frame])
                            - float(config.quality_rescue_area_ratio_cap),
                            0.0,
                        )
                    ),
                    float(regrets[frame]),
                    -int(frame),
                ),
            )
        )
        interpolated = dense[candidate]
        desired = full_controls[candidate]
        best_controls = None
        best_indices = None
        best_inserted = None
        best_score = float(baseline_score)
        best_gain_per_key = 0.0
        best_rank: tuple[float, ...] | None = None
        remaining_budget = maximum_insertions - len(inserted)
        candidate_is_active = candidate in active_set
        support_options = [
            (),
            (candidate - 1,),
            (candidate + 1,),
            (candidate - 1, candidate + 1),
        ]
        unique_insertions: list[tuple[int, ...]] = []
        for supports in support_options:
            proposed = tuple(
                sorted(
                    {
                        frame
                        for frame in (
                            (() if candidate_is_active else (candidate,)) + supports
                        )
                        if 0 < int(frame) < len(full_controls) - 1
                        and int(frame) not in active_set
                    }
                )
            )
            if (
                (not candidate_is_active and candidate not in proposed)
                or len(proposed) > remaining_budget
                or proposed in unique_insertions
            ):
                continue
            unique_insertions.append(proposed)
        # Alpha zero is the existing feasible interpolation.  Moving only a
        # fraction towards the spatial target can improve the bad frame while
        # preserving Recall in both adjacent intervals.  If one key cannot
        # move without invalidating a long neighbouring edge, an unchanged
        # support key at candidate-1 and/or candidate+1 first localizes that
        # movement.  Supports start on the exact old interpolation, so their
        # insertion alone cannot alter the output.
        control_by_frame = {
            int(frame): active_controls[position]
            for position, frame in enumerate(active)
        }
        if candidate_is_active:
            candidate_position = int(np.searchsorted(active, candidate, side="left"))
            left_active_position = max(0, candidate_position - 1)
            right_active_position = min(
                len(active) - 1,
                candidate_position + 1,
            )
        else:
            right_active_position = int(
                np.searchsorted(active, candidate, side="right")
            )
            left_active_position = right_active_position - 1
        affected_start = int(active[left_active_position])
        affected_end = int(active[right_active_position])
        for new_frames in unique_insertions:
            trial_indices = tuple(sorted((*active, *new_frames)))
            for alpha in np.linspace(0.0, 1.0, 33)[1:]:
                inserted_control = interpolated + float(alpha) * (
                    desired - interpolated
                )
                trial_controls = np.asarray(
                    [
                        (
                            inserted_control
                            if frame == candidate
                            else control_by_frame.get(frame, dense[frame])
                        )
                        for frame in trial_indices
                    ],
                    dtype=np.float64,
                )
                local_positions = tuple(
                    position
                    for position, frame in enumerate(trial_indices)
                    if affected_start <= int(frame) <= affected_end
                )
                local_indices = tuple(
                    int(trial_indices[position] - affected_start)
                    for position in local_positions
                )
                local_controls = trial_controls[list(local_positions)]
                trial_dense = materialize_controls(
                    affected_end - affected_start + 1,
                    local_indices,
                    local_controls,
                )
                _trial_boundaries, trial_audit = audit_dense_path(
                    references[affected_start : affected_end + 1],
                    trial_dense,
                    renderer,
                    exact_raster=exact_raster,
                    frame_offset=affected_start,
                    threads=int(config.native_cpu_threads),
                )
                local_losses = _valid_audit_losses(trial_audit, config)
                if local_losses is None:
                    continue
                if minimum_iou_guard is None or maximum_area_guard is None:
                    raise RuntimeError("quality rescue guard initialization failed")
                local_guard = slice(affected_start, affected_end + 1)
                if np.any(
                    trial_audit.iou + _EPSILON < minimum_iou_guard[local_guard]
                ) or np.any(
                    trial_audit.area_ratio > maximum_area_guard[local_guard] + _EPSILON
                ):
                    continue
                trial_losses = baseline_losses.copy()
                trial_losses[affected_start : affected_end + 1] = local_losses
                trial_score = float(-np.mean(trial_losses))
                if trial_score <= baseline_score + 1e-12:
                    continue
                gain_per_key = float(
                    (trial_score - baseline_score) / max(len(new_frames), 1)
                )
                local_candidate = int(candidate - affected_start)
                candidate_iou = float(trial_audit.iou[local_candidate])
                candidate_area = float(trial_audit.area_ratio[local_candidate])
                repaired = bool(
                    candidate_iou + _EPSILON >= float(config.quality_rescue_iou_floor)
                    and candidate_area
                    <= float(config.quality_rescue_area_ratio_cap) + _EPSILON
                )
                # The rescue exists to prevent a locally bad frame from being
                # sacrificed for global average cost.  If an exact-valid
                # support bundle reaches both requested quality guards, it
                # outranks a cheaper partial repair.  Among complete repairs,
                # use the fewest added keys; among partial repairs, maximize
                # the target-frame IoU before global gain-per-key.
                rank = (
                    float(int(repaired)),
                    float(-len(new_frames) if repaired else candidate_iou),
                    float(
                        candidate_iou
                        if repaired
                        else -max(
                            candidate_area
                            - float(config.quality_rescue_area_ratio_cap),
                            0.0,
                        )
                    ),
                    float(gain_per_key),
                    float(trial_score),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_gain_per_key = gain_per_key
                    best_score = float(trial_score)
                    best_controls = trial_controls
                    best_indices = trial_indices
                    best_inserted = new_frames
        if best_controls is not None:
            if best_indices is None or best_inserted is None:  # pragma: no cover
                raise RuntimeError("rescue support bookkeeping failed")
            active = best_indices
            active_controls = best_controls
            inserted.extend(best_inserted)
            refined_active.add(candidate)
            rejected.clear()
        else:
            rejected.add(candidate)
    inserted_set = set(inserted)
    relaxation = np.linspace(0.0, 1.0, int(config.quality_rescue_relaxation_steps) + 1)[
        1:
    ]
    for sweep in range(int(config.quality_rescue_relaxation_sweeps)):
        changed = False
        dense = materialize_controls(len(full_controls), active, active_controls)
        _boundaries, audit = audit_dense_path(
            references,
            dense,
            renderer,
            exact_raster=exact_raster,
            threads=int(config.native_cpu_threads),
        )
        active_losses = _valid_audit_losses(audit, config)
        if active_losses is None:  # pragma: no cover - insertion audit is exact
            raise RuntimeError("invalid path during quality-rescue relaxation")
        positions = [
            position
            for position, frame in enumerate(active)
            if int(frame) in inserted_set
        ]
        if sweep % 2:
            positions.reverse()
        for position in positions:
            current = active_controls[position].copy()
            desired = full_controls[int(active[position])]
            left_position = max(0, int(position) - 1)
            right_position = min(len(active) - 1, int(position) + 1)
            affected_start = int(active[left_position])
            affected_end = int(active[right_position])
            local_indices = tuple(
                int(frame - affected_start)
                for frame in active[left_position : right_position + 1]
            )
            best_score = float(-np.mean(active_losses))
            best_control = current
            best_local_losses = active_losses[affected_start : affected_end + 1].copy()
            for alpha in relaxation:
                trial_controls = active_controls.copy()
                trial_controls[position] = current + float(alpha) * (desired - current)
                trial_dense = materialize_controls(
                    affected_end - affected_start + 1,
                    local_indices,
                    trial_controls[left_position : right_position + 1],
                )
                _trial_boundaries, trial_audit = audit_dense_path(
                    references[affected_start : affected_end + 1],
                    trial_dense,
                    renderer,
                    exact_raster=exact_raster,
                    frame_offset=affected_start,
                    threads=int(config.native_cpu_threads),
                )
                local_losses = _valid_audit_losses(trial_audit, config)
                if local_losses is None:
                    continue
                if minimum_iou_guard is None or maximum_area_guard is None:
                    raise RuntimeError("quality rescue guard initialization failed")
                local_guard = slice(affected_start, affected_end + 1)
                if np.any(
                    trial_audit.iou + _EPSILON < minimum_iou_guard[local_guard]
                ) or np.any(
                    trial_audit.area_ratio > maximum_area_guard[local_guard] + _EPSILON
                ):
                    continue
                trial_losses = active_losses.copy()
                trial_losses[affected_start : affected_end + 1] = local_losses
                trial_score = float(-np.mean(trial_losses))
                if trial_score > best_score + 1e-12:
                    best_score = float(trial_score)
                    best_control = trial_controls[position].copy()
                    best_local_losses = local_losses
            if not np.array_equal(best_control, current):
                active_controls[position] = best_control
                active_losses[affected_start : affected_end + 1] = best_local_losses
                changed = True
        if not changed:
            break
    return active, active_controls, tuple(inserted)


def _refine_pair_vote(
    references: list[np.ndarray],
    full_controls: np.ndarray,
    chosen: tuple[int, ...],
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig,
    initial_key_controls: np.ndarray | None = None,
    exact_raster: ExactDoubleRasterBatch | None = None,
) -> tuple[np.ndarray, int, int, float]:
    baseline = np.asarray(
        full_controls[list(chosen)]
        if initial_key_controls is None
        else initial_key_controls,
        dtype=np.float64,
    )
    dense = materialize_controls(len(full_controls), chosen, baseline)
    _boundaries, audit = audit_dense_path(
        references,
        dense,
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    best_score = _valid_path_score(audit, config)
    if best_score is None:
        raise RuntimeError("DP path failed its own exact final audit")
    initial_score = float(best_score)
    best_losses = _audit_losses(audit, config)
    best = baseline.copy()
    voted = _pair_vote_targets(full_controls, chosen)
    trials = 0
    accepted = 0
    global_alphas = np.linspace(
        0.0,
        1.0,
        int(config.pair_vote_global_steps) + 1,
    )[1:]
    global_controls = [
        baseline + float(alpha) * (voted - baseline) for alpha in global_alphas
    ]
    global_dense = np.asarray(
        [
            materialize_controls(len(full_controls), chosen, trial)
            for trial in global_controls
        ],
        dtype=np.float64,
    )
    global_losses = _valid_dense_trial_losses_batch(
        references,
        global_dense,
        renderer,
        config,
        exact_raster,
    )
    for trial, losses in zip(global_controls, global_losses, strict=True):
        trials += 1
        if losses is None:
            continue
        score = float(-np.mean(losses))
        if score > best_score + 1e-12:
            best_score = score
            best_losses = losses
            best = trial
            accepted += 1
    local_grid = np.linspace(0.0, 1.0, int(config.pair_vote_local_steps) + 1)
    for _sweep in range(int(config.pair_vote_sweeps)):
        changed = False
        for position in range(len(chosen)):
            local_best = best[position].copy()
            local_score = float(best_score)
            left_position = max(0, int(position) - 1)
            right_position = min(len(chosen) - 1, int(position) + 1)
            start_frame = int(chosen[left_position])
            end_frame = int(chosen[right_position])
            local_indices = tuple(
                int(frame - start_frame)
                for frame in chosen[left_position : right_position + 1]
            )
            local_best_losses = best_losses[start_frame : end_frame + 1].copy()
            local_trials = []
            local_values = []
            for alpha in local_grid:
                local = best[left_position : right_position + 1].copy()
                local[position - left_position] = baseline[position] + float(alpha) * (
                    voted[position] - baseline[position]
                )
                local_trials.append(
                    materialize_controls(
                        end_frame - start_frame + 1,
                        local_indices,
                        local,
                    )
                )
                local_values.append(local[position - left_position].copy())
            losses_batch = _valid_dense_trial_losses_batch(
                references[start_frame : end_frame + 1],
                np.asarray(local_trials, dtype=np.float64),
                renderer,
                config,
                exact_raster,
                frame_offset=start_frame,
            )
            for trial_value, local_losses in zip(
                local_values,
                losses_batch,
                strict=True,
            ):
                trials += 1
                if local_losses is None:
                    continue
                trial_losses = best_losses.copy()
                trial_losses[start_frame : end_frame + 1] = local_losses
                score = float(-np.mean(trial_losses))
                if score is not None and score > local_score + 1e-12:
                    local_score = float(score)
                    local_best = trial_value
                    local_best_losses = local_losses
            if local_score > best_score + 1e-12:
                best[position] = local_best
                best_score = local_score
                best_losses[start_frame : end_frame + 1] = local_best_losses
                accepted += 1
                changed = True
        if not changed:
            break
    return best, int(trials), int(accepted), float(best_score - initial_score)


def optimize_keyframes(
    references: list[np.ndarray],
    frame_controls: np.ndarray,
    *,
    representation: str,
    renderer: BoundaryRenderer,
    config: KeyframeDpConfig = KeyframeDpConfig(),
) -> KeyframeDpResult:
    config.validate()
    controls = np.asarray(frame_controls, dtype=np.float64)
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("frame_controls must have shape (frames, points, 2)")
    if len(references) != len(controls) or not len(controls):
        raise ValueError("references and frame controls must have equal nonzero length")
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
    started = time.perf_counter()
    evaluator = _IntervalEvaluator(references, controls, renderer, config)
    first_boundary = renderer(controls[0])
    first_iou, first_recall, _precision, _ratio = _frame_metrics(
        references[0], first_boundary
    )
    if has_strict_self_intersection(first_boundary) or first_recall + _EPSILON < float(
        config.recall_floor
    ):
        raise RuntimeError("first spatial candidate violates hard quality constraints")
    target_count = max(
        2, min(len(controls), int(round(len(controls) / config.target_interval)))
    )
    chosen, penalty = _select_penalty_path(
        len(controls), evaluator, target_count, config, 1.0 - first_iou
    )
    rescue_inserted: tuple[int, ...] = ()
    key_controls = np.asarray(controls[list(chosen)], dtype=np.float64)
    if bool(config.quality_rescue_enabled):
        chosen, key_controls, rescue_inserted = _quality_rescue_keys(
            references,
            controls,
            chosen,
            renderer,
            config,
            exact_raster=exact_raster,
        )
    dp_finished = time.perf_counter()
    pair_trials = 0
    pair_accepted = 0
    pair_gain = 0.0
    if bool(config.pair_vote_enabled) and len(chosen) > 1:
        key_controls, pair_trials, pair_accepted, pair_gain = _refine_pair_vote(
            references,
            controls,
            chosen,
            renderer,
            config,
            initial_key_controls=key_controls,
            exact_raster=exact_raster,
        )
    pair_vote_finished = time.perf_counter()
    dense_controls = materialize_controls(len(controls), chosen, key_controls)
    boundaries, audit = audit_dense_path(
        references,
        dense_controls,
        renderer,
        exact_raster=exact_raster,
        threads=int(config.native_cpu_threads),
    )
    violations = int(
        np.count_nonzero(audit.recall + _EPSILON < float(config.recall_floor))
    )
    if violations or np.any(~audit.topology_valid):
        raise RuntimeError(
            f"final hard audit failed: recall={violations}, "
            f"topology={np.count_nonzero(~audit.topology_valid)}"
        )
    final_audit_finished = time.perf_counter()
    return KeyframeDpResult(
        representation=str(representation),
        chosen_indices=chosen,
        keyframe_controls=key_controls,
        dense_controls=dense_controls,
        dense_boundaries=boundaries,
        audit=audit,
        target_keyframes=int(target_count),
        target_interval=int(config.target_interval),
        effective_interval=float(len(controls) / max(len(chosen), 1)),
        selected_lambda=float(penalty),
        edge_evaluations=int(len(evaluator.cache)),
        edge_frame_evaluations=int(evaluator.frame_evaluations),
        pair_vote_trials=int(pair_trials),
        pair_vote_accepted=int(pair_accepted),
        pair_vote_iou_gain=float(pair_gain),
        point_refine_trials=0,
        point_refine_accepted=0,
        point_refine_iou_gain=0.0,
        quality_rescue_inserted_indices=rescue_inserted,
        dp_seconds=float(dp_finished - started),
        pair_vote_seconds=float(pair_vote_finished - dp_finished),
        point_refine_seconds=0.0,
        final_audit_seconds=float(final_audit_finished - pair_vote_finished),
        elapsed_seconds=float(final_audit_finished - started),
        native_reference_cache=(
            {} if exact_raster is None else exact_raster.cache_stats()
        ),
        config=config,
    )


__all__ = (
    "KeyframeDpConfig",
    "KeyframeDpResult",
    "RasterAudit",
    "audit_dense_path",
    "catmull_rom_renderer",
    "interpolate_controls",
    "materialize_controls",
    "optimize_keyframes",
    "polygon_renderer",
    "render_control_sequence",
)
