"""Quality-guarded candidate for replacing fixed 48-point equal-arc sampling."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    RasterSequenceEvaluator,
    SequenceMetrics,
    evaluate_sequence,
)

from .native_dp import native_temporal_dp_sequence


@dataclass(frozen=True)
class PlacementAttempt:
    vertices: int
    seconds: float
    metrics: SequenceMetrics


@dataclass(frozen=True)
class PlacementResult:
    polygons: np.ndarray
    vertices: int
    attempts: tuple[PlacementAttempt, ...]
    elapsed_seconds: float


def quality_guarded_vertex_placement(
    polygons: Iterable[np.ndarray],
    *,
    frame_indices: Iterable[int] | None = None,
    cut_frames: Iterable[int] = (),
    candidate_counts: tuple[int, ...] = (16, 17, 18, 20, 24, 32, 48),
    recall_floor: float = 0.97,
    minimum_iou_floor: float = 0.95,
    temporal_weight: float = 0.003,
    distance_weight: float = 2.0,
    missing_area_weight: float = 1.0,
) -> PlacementResult:
    """Use 20 vertices normally and increase only when exact QA requires it.

    A count is fixed for the complete contiguous track/cut segment.  The exact
    raster audit prevents a difficult frame from silently receiving the same
    kind of invalid shortcut that motivated this experiment.
    """
    source = [np.asarray(value, dtype=np.float64) for value in polygons]
    frames = None if frame_indices is None else list(frame_indices)
    cuts = list(cut_frames)
    counts = tuple(sorted({max(3, int(value)) for value in candidate_counts}))
    if not counts:
        raise ValueError("candidate_counts must not be empty")
    evaluator = RasterSequenceEvaluator(source)
    attempts: list[PlacementAttempt] = []
    started_all = time.perf_counter()
    last_sequence = None
    for count in counts:
        started = time.perf_counter()
        sequence = native_temporal_dp_sequence(
            source,
            count,
            frame_indices=frames,
            cut_frames=cuts,
            temporal_weight=temporal_weight,
            distance_weight=distance_weight,
            missing_area_weight=missing_area_weight,
        )
        seconds = time.perf_counter() - started
        metrics = evaluate_sequence(
            evaluator,
            sequence,
            initial_vertices=max(counts),
            temporal_weight=0.05,
            tail_weight=0.20,
            vertex_weight=0.02,
            check_self_intersections=True,
        )
        attempts.append(PlacementAttempt(count, seconds, metrics))
        last_sequence = sequence
        if (
            metrics.minimum_recall + 1e-12 >= float(recall_floor)
            and metrics.minimum_iou + 1e-12 >= float(minimum_iou_floor)
            and metrics.self_intersections == 0
        ):
            return PlacementResult(
                polygons=sequence,
                vertices=count,
                attempts=tuple(attempts),
                elapsed_seconds=time.perf_counter() - started_all,
            )
    if last_sequence is None:
        raise AssertionError("normalized candidate counts unexpectedly empty")
    # The final attempt remains useful for diagnostics, but silently violating
    # a declared quality contract would be unsafe for Production.
    final = attempts[-1].metrics
    raise RuntimeError(
        "no vertex count satisfied the quality guard: "
        f"last_count={attempts[-1].vertices} "
        f"min_recall={final.minimum_recall:.6f} "
        f"min_iou={final.minimum_iou:.6f} "
        f"self_intersections={final.self_intersections}"
    )
