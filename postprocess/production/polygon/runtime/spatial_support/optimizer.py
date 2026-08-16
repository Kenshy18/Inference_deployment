"""Fast, deterministic track-wise polygon vertex decimation.

The experiment treats one polygon vertex index as a trajectory through a
track/cut segment.  A removal therefore removes the same index from every
frame, preserving the fixed vertex count and correspondence contract required
by the editor and by linear keyframe interpolation.

Only polygon geometry is consumed.  Video pixels are never opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import cv2
import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class DecimationConfig:
    initial_vertices: int = 48
    minimum_vertices: int = 6
    recall_floor: float = 0.97
    shortlist: int = 10
    temporal_weight: float = 0.05
    tail_weight: float = 0.20
    vertex_weight: float = 0.02
    local_refine_radius: int = 0
    local_refine_passes: int = 1
    native_threads: int = 8


@dataclass(frozen=True)
class SequenceMetrics:
    vertices: int
    mean_iou: float
    minimum_iou: float
    q01_iou: float
    q05_iou: float
    mean_recall: float
    minimum_recall: float
    temporal_residual: float
    temporal_q95: float
    self_intersections: int
    objective: float


@dataclass
class TemporalDecimationResult:
    dense_aligned: np.ndarray
    active_indices: tuple[int, ...]
    polygons: np.ndarray
    metrics: SequenceMetrics
    greedy_terminal_active_indices: tuple[int, ...]
    greedy_terminal_polygons: np.ndarray
    greedy_terminal_metrics: SequenceMetrics
    curve: list[SequenceMetrics]
    snapshots: dict[int, np.ndarray]
    elapsed_seconds: float
    exact_candidate_evaluations: int
    stopped_reason: str


def normalize_closed(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(value) > 1 and np.allclose(value[0], value[-1]):
        value = value[:-1]
    if len(value) < 3:
        raise ValueError("a polygon requires at least three distinct points")
    return value


def signed_area(points: np.ndarray) -> float:
    value = normalize_closed(points)
    return 0.5 * float(
        np.sum(
            value[:, 0] * np.roll(value[:, 1], -1)
            - np.roll(value[:, 0], -1) * value[:, 1]
        )
    )


def orient_ccw(points: np.ndarray) -> np.ndarray:
    value = normalize_closed(points)
    return value[::-1].copy() if signed_area(value) < 0.0 else value.copy()


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    value = orient_ccw(points)
    count = max(3, int(count))
    following = np.roll(value, -1, axis=0)
    lengths = np.linalg.norm(following - value, axis=1)
    total = float(np.sum(lengths))
    if total <= _EPS:
        return np.repeat(value[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.linspace(0.0, total, count, endpoint=False)
    segment_ids = np.searchsorted(cumulative, positions, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(value) - 1)
    alpha = (positions - cumulative[segment_ids]) / np.maximum(
        lengths[segment_ids], _EPS
    )
    return (1.0 - alpha[:, None]) * value[segment_ids] + alpha[:, None] * following[
        segment_ids
    ]


def _normalized_shape(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    centered = value - np.mean(value, axis=0, keepdims=True)
    scale = math.sqrt(max(float(np.mean(np.sum(centered * centered, axis=1))), _EPS))
    return centered / scale


def _procrustes_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = _normalized_shape(candidate)
    right = _normalized_shape(reference)
    u, _singular, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    residual = left @ rotation - right
    return float(np.mean(np.sum(residual * residual, axis=1)))


def _best_phase(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    allow_reverse: bool,
    procrustes: bool,
) -> np.ndarray:
    count = int(len(candidate))
    variants = [np.asarray(candidate, dtype=np.float64)]
    if allow_reverse:
        variants.append(np.asarray(candidate[::-1], dtype=np.float64))
    if not procrustes:
        # Evaluate all cyclic phases in one NumPy kernel.  This preserves the
        # reference implementation's candidate order and strict tie rule:
        # the lowest forward shift wins, and a reversed phase replaces it
        # only when its cost is strictly smaller.
        phase_indices = (
            np.arange(count, dtype=np.intp)[:, None]
            + np.arange(count, dtype=np.intp)[None, :]
        ) % count
        best = variants[0]
        best_cost = float("inf")
        for variant in variants:
            rolled = variant[phase_indices]
            delta = rolled - reference[None, :, :]
            costs = np.mean(np.sum(delta * delta, axis=2), axis=1)
            shift = int(np.argmin(costs))
            cost = float(costs[shift])
            if cost < best_cost:
                best_cost = cost
                best = rolled[shift]
        return np.asarray(best, dtype=np.float64).copy()
    best = variants[0]
    best_cost = float("inf")
    for variant in variants:
        for shift in range(count):
            rolled = np.roll(variant, -shift, axis=0)
            if procrustes:
                cost = _procrustes_error(reference, rolled)
            else:
                delta = rolled - reference
                cost = float(np.mean(np.sum(delta * delta, axis=1)))
            if cost < best_cost:
                best_cost = cost
                best = rolled
    return np.asarray(best, dtype=np.float64).copy()


def align_current_equal_arc(polygons: Iterable[np.ndarray], count: int) -> np.ndarray:
    """Reproduce the current forward, XY-MSE, reversible phase alignment."""
    output: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for polygon in polygons:
        current = resample_closed(polygon, count)
        if previous is not None:
            current = _best_phase(
                previous,
                current,
                allow_reverse=True,
                procrustes=False,
            )
        output.append(current)
        previous = current
    return np.asarray(output, dtype=np.float64)


def align_temporal_dense(polygons: Iterable[np.ndarray], count: int) -> np.ndarray:
    """Align from a central gauge in both directions using shape-only phase.

    Direction reversal is forbidden.  Translation, rotation, and scale are
    removed only while scoring the cyclic phase, so ordinary object motion does
    not cause a vertex-number shift.
    """
    sampled = np.asarray(
        [resample_closed(polygon, count) for polygon in polygons],
        dtype=np.float64,
    )
    if len(sampled) <= 1:
        return sampled
    center = len(sampled) // 2
    aligned = np.empty_like(sampled)
    aligned[center] = sampled[center]
    for frame in range(center + 1, len(sampled)):
        aligned[frame] = _best_phase(
            aligned[frame - 1],
            sampled[frame],
            allow_reverse=False,
            procrustes=True,
        )
    for frame in range(center - 1, -1, -1):
        aligned[frame] = _best_phase(
            aligned[frame + 1],
            sampled[frame],
            allow_reverse=False,
            procrustes=True,
        )
    return aligned


class RasterSequenceEvaluator:
    """Exact OpenCV raster metrics in one small ROI per source polygon."""

    def __init__(self, references: Iterable[np.ndarray], padding: int = 3) -> None:
        self.references = [orient_ccw(value) for value in references]
        self.origins: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.areas: list[int] = []
        self.native_full_batch = None
        native_build = Path(__file__).resolve().parents[1] / "native_interval" / "build"
        if native_build.is_dir() and str(native_build) not in sys.path:
            sys.path.insert(0, str(native_build))
        try:
            native = importlib.import_module("native_interval_metrics")
        except ImportError:
            native = None
        if native is not None:
            self.native_full_batch = getattr(
                native, "pair_vote_full_metrics_batch", None
            )
        for polygon in self.references:
            minimum = np.floor(np.min(polygon, axis=0)).astype(np.int32) - int(padding)
            maximum = np.ceil(np.max(polygon, axis=0)).astype(np.int32) + int(padding)
            shape = (
                max(1, int(maximum[1] - minimum[1] + 1)),
                max(1, int(maximum[0] - minimum[0] + 1)),
            )
            mask = np.zeros(shape, dtype=np.uint8)
            points = np.rint(polygon - minimum[None, :]).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
            self.origins.append(minimum.astype(np.float64))
            self.masks.append(mask)
            self.areas.append(int(cv2.countNonZero(mask)))

    def frame_metrics(self, frame: int, polygon: np.ndarray) -> tuple[float, float]:
        gt = self.masks[int(frame)]
        pred = np.zeros_like(gt)
        points = np.rint(
            np.asarray(polygon, dtype=np.float64) - self.origins[int(frame)][None, :]
        ).astype(np.int32)
        cv2.fillPoly(pred, [points], 1)
        intersection = int(cv2.countNonZero(cv2.bitwise_and(gt, pred)))
        pred_area = int(cv2.countNonZero(pred))
        gt_area = int(self.areas[int(frame)])
        union = int(gt_area + pred_area - intersection)
        recall = float(intersection / gt_area) if gt_area else 1.0
        iou = float(intersection / union) if union else 1.0
        return iou, recall

    def metrics(self, polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ious = np.empty((len(polygons),), dtype=np.float64)
        recalls = np.empty((len(polygons),), dtype=np.float64)
        for frame, polygon in enumerate(polygons):
            ious[frame], recalls[frame] = self.frame_metrics(frame, polygon)
        return ious, recalls

    def batch_mean_iou_min_recall(
        self,
        sequences: list[np.ndarray],
        *,
        threads: int,
    ) -> np.ndarray | None:
        """Evaluate same-sized dense sequences in one exact native batch."""
        if self.native_full_batch is None or not sequences:
            return None
        frame_count = len(self.references)
        values = self.native_full_batch(
            [[np.asarray(polygon, dtype=np.float32)] for polygon in self.references],
            np.arange(frame_count, dtype=np.int32),
            np.asarray(sequences, dtype=np.float32)[:, :, None, :, :],
            1,
            int(sequences[0].shape[1]),
            max(1, int(threads)),
        )
        return np.asarray(values, dtype=np.float64)


def _similarity_residual(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    lc = np.mean(left, axis=0)
    rc = np.mean(right, axis=0)
    left0 = left - lc
    right0 = right - rc
    covariance = left0.T @ right0
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left0 * left0), _EPS))
    prediction = scale * (left0 @ rotation) + rc
    normalizer = math.sqrt(max(float(abs(signed_area(right))), 1.0))
    return np.linalg.norm(prediction - right, axis=1) / normalizer


def temporal_residuals(polygons: np.ndarray) -> np.ndarray:
    if len(polygons) <= 1:
        return np.zeros((0, polygons.shape[1] if polygons.ndim >= 2 else 0))
    # Batched 2-D similarity least squares.  One NumPy call evaluates all
    # adjacent frame pairs and preserves the scalar reference implementation's
    # exact rotation/reflection branch and scale convention.
    value = np.asarray(polygons, dtype=np.float64)
    left = value[:-1]
    right = value[1:]
    left_center = np.mean(left, axis=1, keepdims=True)
    right_center = np.mean(right, axis=1, keepdims=True)
    left_zero = left - left_center
    right_zero = right - right_center
    covariance = np.einsum("tni,tnj->tij", left_zero, right_zero)
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    reflected = np.linalg.det(rotation) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vt
    scale = np.sum(singular, axis=1) / np.maximum(
        np.sum(left_zero * left_zero, axis=(1, 2)), _EPS
    )
    prediction = scale[:, None, None] * (left_zero @ rotation) + right_center
    residual = np.linalg.norm(prediction - right, axis=2)
    right_x = right[:, :, 0]
    right_y = right[:, :, 1]
    twice_area = np.abs(
        np.sum(
            right_x * np.roll(right_y, -1, axis=1)
            - np.roll(right_x, -1, axis=1) * right_y,
            axis=1,
        )
    )
    normalizer = np.sqrt(np.maximum(0.5 * twice_area, 1.0))
    return np.asarray(residual / normalizer[:, None], dtype=np.float64)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # This scalar primitive sits inside an O(V^2) polygon audit.  np.cross
    # constructs and normalizes tiny arrays on every call and dominated the
    # full-track comparison.  Direct 2-D arithmetic is exactly the same test.
    return float(
        (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    )


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return bool(ab_c * ab_d < -1e-8 and cd_a * cd_b < -1e-8)


def has_self_intersection(points: np.ndarray) -> bool:
    value = np.asarray(points, dtype=np.float64)
    count = len(value)
    for first in range(count):
        a = value[first]
        b = value[(first + 1) % count]
        for second in range(first + 2, count):
            if second == first or (second + 1) % count == first:
                continue
            if first == 0 and second == count - 1:
                continue
            c = value[second]
            d = value[(second + 1) % count]
            if _segments_cross(a, b, c, d):
                return True
    return False


def evaluate_sequence(
    evaluator: RasterSequenceEvaluator,
    polygons: np.ndarray,
    *,
    initial_vertices: int,
    temporal_weight: float,
    tail_weight: float = 0.0,
    vertex_weight: float,
    check_self_intersections: bool = True,
) -> SequenceMetrics:
    ious, recalls = evaluator.metrics(polygons)
    residuals = temporal_residuals(polygons)
    flat_residuals = residuals.reshape(-1) if residuals.size else np.zeros((1,))
    mean_temporal = float(np.mean(flat_residuals))
    vertex_ratio = float(polygons.shape[1] / max(int(initial_vertices), 1))
    objective = (
        1.0
        - float(np.mean(ious))
        + float(temporal_weight) * mean_temporal
        + float(tail_weight) * (1.0 - float(np.quantile(ious, 0.01)))
        + float(vertex_weight) * vertex_ratio
    )
    intersections = (
        sum(has_self_intersection(frame) for frame in polygons)
        if bool(check_self_intersections)
        else 0
    )
    return SequenceMetrics(
        vertices=int(polygons.shape[1]),
        mean_iou=float(np.mean(ious)),
        minimum_iou=float(np.min(ious)),
        q01_iou=float(np.quantile(ious, 0.01)),
        q05_iou=float(np.quantile(ious, 0.05)),
        mean_recall=float(np.mean(recalls)),
        minimum_recall=float(np.min(recalls)),
        temporal_residual=mean_temporal,
        temporal_q95=float(np.quantile(flat_residuals, 0.95)),
        self_intersections=int(intersections),
        objective=float(objective),
    )


def current_equal_arc_baseline(
    references: Iterable[np.ndarray],
    vertices: int,
    *,
    evaluator: RasterSequenceEvaluator | None = None,
    initial_vertices: int | None = None,
    temporal_weight: float = 0.05,
    tail_weight: float = 0.0,
    vertex_weight: float = 0.02,
    check_self_intersections: bool = True,
) -> tuple[np.ndarray, SequenceMetrics]:
    source = [np.asarray(value, dtype=np.float64) for value in references]
    implementation = evaluator or RasterSequenceEvaluator(source)
    polygons = align_current_equal_arc(source, int(vertices))
    metrics = evaluate_sequence(
        implementation,
        polygons,
        initial_vertices=int(initial_vertices or vertices),
        temporal_weight=float(temporal_weight),
        tail_weight=float(tail_weight),
        vertex_weight=float(vertex_weight),
        check_self_intersections=bool(check_self_intersections),
    )
    return polygons, metrics


def _surrogate_removal_scores(sequence: np.ndarray) -> np.ndarray:
    previous = np.roll(sequence, 1, axis=1)
    following = np.roll(sequence, -1, axis=1)
    twice_triangle = np.abs(
        (sequence[:, :, 0] - previous[:, :, 0])
        * (following[:, :, 1] - previous[:, :, 1])
        - (sequence[:, :, 1] - previous[:, :, 1])
        * (following[:, :, 0] - previous[:, :, 0])
    )
    areas = np.asarray(
        [max(abs(signed_area(frame)), 1.0) for frame in sequence], dtype=np.float64
    )
    shape_cost = np.mean(twice_triangle / (2.0 * areas[:, None]), axis=0)
    residuals = temporal_residuals(sequence)
    instability = (
        np.mean(residuals, axis=0)
        if residuals.size
        else np.zeros((sequence.shape[1],), dtype=np.float64)
    )
    # Removing an unstable but geometrically redundant trajectory is useful.
    return shape_cost - 0.02 * instability


def _candidate_objective(
    metrics: SequenceMetrics,
    temporal_weight: float,
    tail_weight: float = 0.0,
) -> float:
    return float(
        1.0
        - metrics.mean_iou
        + float(temporal_weight) * metrics.temporal_residual
        + float(tail_weight) * (1.0 - metrics.q01_iou)
    )


def _new_edge_crosses_polygon(
    polygon: np.ndarray,
    start_position: int,
    end_position: int,
    ignored_edges: set[int],
) -> bool:
    """Check one changed edge against unchanged polygon edges only."""
    value = np.asarray(polygon, dtype=np.float64)
    count = len(value)
    a = value[int(start_position) % count]
    b = value[int(end_position) % count]
    for edge in range(count):
        if edge in ignored_edges:
            continue
        edge_end = (edge + 1) % count
        if edge in {start_position, end_position} or edge_end in {
            start_position,
            end_position,
        }:
            continue
        if _segments_cross(a, b, value[edge], value[edge_end]):
            return True
    return False


def _removal_preserves_simplicity(sequence: np.ndarray, position: int) -> bool:
    """O(T*N) exact crossing gate for deleting one polygon vertex."""
    count = int(sequence.shape[1])
    previous = (int(position) - 1) % count
    following = (int(position) + 1) % count
    ignored = {previous, int(position)}
    return not any(
        _new_edge_crosses_polygon(frame, previous, following, ignored)
        for frame in sequence
    )


def _refine_active_indices(
    dense: np.ndarray,
    active: list[int],
    evaluator: RasterSequenceEvaluator,
    config: DecimationConfig,
    exact_counter: list[int],
) -> list[int]:
    if config.local_refine_radius <= 0 or len(active) <= 4:
        return active
    active = sorted(int(value) for value in active)
    for _pass in range(max(0, int(config.local_refine_passes))):
        changed = False
        # Keep the first/last index as a stable cyclic gauge.  Interior
        # trajectories can move to nearby dense samples without crossing.
        for position in range(1, len(active) - 1):
            current_index = int(active[position])
            candidates = [current_index]
            for delta in range(
                -config.local_refine_radius, config.local_refine_radius + 1
            ):
                candidate = current_index + int(delta)
                if active[position - 1] < candidate < active[position + 1]:
                    candidates.append(candidate)
            best_index = current_index
            best_metrics = evaluate_sequence(
                evaluator,
                dense[:, active],
                initial_vertices=config.initial_vertices,
                temporal_weight=config.temporal_weight,
                tail_weight=config.tail_weight,
                vertex_weight=config.vertex_weight,
                check_self_intersections=False,
            )
            best_score = _candidate_objective(
                best_metrics,
                config.temporal_weight,
                config.tail_weight,
            )
            for candidate in sorted(set(candidates)):
                if candidate == current_index:
                    continue
                trial_indices = list(active)
                trial_indices[position] = int(candidate)
                trial = dense[:, trial_indices]
                metrics = evaluate_sequence(
                    evaluator,
                    trial,
                    initial_vertices=config.initial_vertices,
                    temporal_weight=config.temporal_weight,
                    tail_weight=config.tail_weight,
                    vertex_weight=config.vertex_weight,
                    check_self_intersections=True,
                )
                exact_counter[0] += 1
                if (
                    metrics.minimum_recall + 1e-12 >= config.recall_floor
                    and metrics.self_intersections == 0
                    and metrics.minimum_iou + 1e-12 >= best_metrics.minimum_iou
                    and metrics.q01_iou + 1e-12 >= best_metrics.q01_iou
                ):
                    score = _candidate_objective(
                        metrics,
                        config.temporal_weight,
                        config.tail_weight,
                    )
                    if score + 1e-12 < best_score:
                        best_score = score
                        best_index = int(candidate)
            if best_index != current_index:
                active[position] = best_index
                changed = True
        if not changed:
            break
    return active


def optimize_temporal_vertices(
    references: Iterable[np.ndarray],
    config: DecimationConfig | None = None,
    *,
    snapshot_counts: Iterable[int] = (),
) -> TemporalDecimationResult:
    settings = config or DecimationConfig()
    if settings.minimum_vertices < 3:
        raise ValueError("minimum_vertices must be >= 3")
    if settings.initial_vertices < settings.minimum_vertices:
        raise ValueError("initial_vertices must be >= minimum_vertices")
    source = [np.asarray(value, dtype=np.float64) for value in references]
    if not source:
        raise ValueError("at least one source polygon is required")
    started = time.perf_counter()
    evaluator = RasterSequenceEvaluator(source)
    dense = align_temporal_dense(source, settings.initial_vertices)
    active = list(range(settings.initial_vertices))
    exact_counter = [0]
    curve: list[SequenceMetrics] = []
    active_by_count: dict[int, tuple[int, ...]] = {}
    snapshots: dict[int, np.ndarray] = {}
    requested = {int(value) for value in snapshot_counts}
    stop_reason = "minimum_vertices_reached"

    while True:
        current = dense[:, active]
        metrics = evaluate_sequence(
            evaluator,
            current,
            initial_vertices=settings.initial_vertices,
            temporal_weight=settings.temporal_weight,
            tail_weight=settings.tail_weight,
            vertex_weight=settings.vertex_weight,
            check_self_intersections=(len(curve) == 0),
        )
        curve.append(metrics)
        active_by_count[len(active)] = tuple(int(value) for value in active)
        if len(active) in requested:
            snapshots[len(active)] = current.copy()
        if len(active) <= settings.minimum_vertices:
            break

        surrogate = _surrogate_removal_scores(current)
        order = np.argsort(surrogate, kind="stable").tolist()
        shortlist = order[: min(len(order), max(1, int(settings.shortlist)))]
        feasible: list[tuple[float, int, float]] = []

        def evaluate_positions(positions: Iterable[int]) -> None:
            candidate_positions: list[int] = []
            candidate_sequences: list[np.ndarray] = []
            for position in positions:
                if not _removal_preserves_simplicity(current, int(position)):
                    continue
                trial_active = active[: int(position)] + active[int(position) + 1 :]
                candidate_positions.append(int(position))
                candidate_sequences.append(dense[:, trial_active])
            if not candidate_sequences:
                return
            batch = evaluator.batch_mean_iou_min_recall(
                candidate_sequences,
                threads=settings.native_threads,
            )
            exact_counter[0] += len(candidate_sequences)
            if batch is None:
                batch_rows = []
                for trial in candidate_sequences:
                    metrics = evaluate_sequence(
                        evaluator,
                        trial,
                        initial_vertices=settings.initial_vertices,
                        temporal_weight=settings.temporal_weight,
                        tail_weight=settings.tail_weight,
                        vertex_weight=settings.vertex_weight,
                        check_self_intersections=False,
                    )
                    batch_rows.append((metrics.mean_iou, metrics.minimum_recall))
            else:
                batch_rows = [(float(row[0]), float(row[1])) for row in batch]
            for position, trial, (mean_iou, minimum_recall) in zip(
                candidate_positions,
                candidate_sequences,
                batch_rows,
                strict=True,
            ):
                if minimum_recall + 1e-12 >= settings.recall_floor:
                    residuals = temporal_residuals(trial)
                    temporal_mean = float(np.mean(residuals)) if residuals.size else 0.0
                    feasible.append(
                        (
                            1.0
                            - float(mean_iou)
                            + settings.temporal_weight * temporal_mean,
                            int(position),
                            float(minimum_recall),
                        )
                    )

        evaluate_positions(shortlist)
        if not feasible and len(shortlist) < len(order):
            evaluate_positions(order[len(shortlist) :])
        if not feasible:
            stop_reason = "no_single_trajectory_removal_satisfies_recall"
            break
        # The native evaluator intentionally follows the Production float32
        # raster path.  A one-pixel rounding difference can nevertheless move
        # minimum Recall by a few 1e-3 on very small masks.  Keep the native
        # batch as the fast screen, then independently audit the best few
        # candidates with the experiment's float64/Python raster path.  This
        # makes the hard constraint conservative without returning to an
        # O(candidates * frames) Python hot loop.
        selected: tuple[float, int, float] | None = None
        for native_candidate in sorted(
            feasible,
            key=lambda value: (value[0], -value[2], value[1]),
        ):
            _native_score, candidate_position, _native_recall = native_candidate
            trial_active = (
                active[: int(candidate_position)]
                + active[int(candidate_position) + 1 :]
            )
            audited = evaluate_sequence(
                evaluator,
                dense[:, trial_active],
                initial_vertices=settings.initial_vertices,
                temporal_weight=settings.temporal_weight,
                tail_weight=settings.tail_weight,
                vertex_weight=settings.vertex_weight,
                check_self_intersections=False,
            )
            exact_counter[0] += 1
            if audited.minimum_recall + 1e-12 < settings.recall_floor:
                continue
            audited_candidate = (
                _candidate_objective(
                    audited,
                    settings.temporal_weight,
                    settings.tail_weight,
                ),
                int(candidate_position),
                float(audited.minimum_recall),
            )
            if selected is None or audited_candidate < selected:
                selected = audited_candidate
        if selected is None:
            stop_reason = "native_candidates_failed_conservative_recall_audit"
            break
        _score, removed_position, _minimum_recall = selected
        del active[int(removed_position)]

    greedy_terminal_active = list(active)
    greedy_terminal_polygons = dense[:, greedy_terminal_active]
    greedy_terminal_metrics = evaluate_sequence(
        evaluator,
        greedy_terminal_polygons,
        initial_vertices=settings.initial_vertices,
        temporal_weight=settings.temporal_weight,
        tail_weight=settings.tail_weight,
        vertex_weight=settings.vertex_weight,
        check_self_intersections=True,
    )
    recommended_curve_metrics = min(
        curve,
        key=lambda value: (value.objective, -value.vertices),
    )
    active = list(active_by_count[int(recommended_curve_metrics.vertices)])
    active = _refine_active_indices(
        dense,
        active,
        evaluator,
        settings,
        exact_counter,
    )
    final_polygons = dense[:, active]
    final_metrics = evaluate_sequence(
        evaluator,
        final_polygons,
        initial_vertices=settings.initial_vertices,
        temporal_weight=settings.temporal_weight,
        tail_weight=settings.tail_weight,
        vertex_weight=settings.vertex_weight,
        check_self_intersections=True,
    )
    if final_metrics.vertices in requested:
        snapshots[final_metrics.vertices] = final_polygons.copy()
    return TemporalDecimationResult(
        dense_aligned=dense,
        active_indices=tuple(int(value) for value in active),
        polygons=final_polygons,
        metrics=final_metrics,
        greedy_terminal_active_indices=tuple(
            int(value) for value in greedy_terminal_active
        ),
        greedy_terminal_polygons=greedy_terminal_polygons,
        greedy_terminal_metrics=greedy_terminal_metrics,
        curve=curve,
        snapshots=snapshots,
        elapsed_seconds=float(time.perf_counter() - started),
        exact_candidate_evaluations=int(exact_counter[0]),
        stopped_reason=stop_reason,
    )
