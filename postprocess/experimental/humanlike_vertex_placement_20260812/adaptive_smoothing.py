"""Quality-gated temporal smoothing of independently fitted RDP vertices.

The spatial RDP polygon is the quality anchor.  Temporal smoothing is accepted
frame by frame only while an exact raster recall floor and a bounded IoU loss
remain satisfied.  This prevents the failure mode of applying one global
smoothing strength to both easy and geometrically critical frames.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    has_self_intersection,
    orient_ccw,
    temporal_residuals,
)

from .smoothed_rdp import (
    _median_smooth,
    _point_fractions,
    _project_order,
    _sample_fraction,
    _temporal_unwrap,
)
from .spatial import align_polygon_sequence, rdp_fixed_count


class _FrameEvaluator(Protocol):
    def frame_metrics(self, frame: int, polygon: np.ndarray) -> tuple[float, float]: ...


def _local_temporal(sequence: np.ndarray, frame: int, candidate: np.ndarray) -> float:
    values = []
    if frame > 0:
        values.append(temporal_residuals(np.asarray([sequence[frame - 1], candidate]))[0])
    if frame + 1 < len(sequence):
        values.append(temporal_residuals(np.asarray([candidate, sequence[frame + 1]]))[0])
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def adaptive_smoothed_rdp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    evaluator: _FrameEvaluator,
    *,
    temporal_window: int = 5,
    recall_floor: float = 0.97,
    iou_loss_budget: float = 0.003,
    temporal_weight: float = 0.12,
    minimum_gap_ratio: float = 0.02,
    passes: int = 2,
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> np.ndarray:
    """Return fixed-count polygons with an exact quality gate per frame."""
    contours = [orient_ccw(value) for value in polygons]
    if not contours:
        return np.empty((0, 0, 2), dtype=np.float64)
    spatial = align_polygon_sequence(
        [rdp_fixed_count(value, int(target)) for value in contours]
    )
    fractions = np.asarray(
        [
            _point_fractions(contour, points)
            for contour, points in zip(contours, spatial, strict=True)
        ],
        dtype=np.float64,
    )
    fractions = _temporal_unwrap(fractions)
    smooth_target = _median_smooth(fractions, int(temporal_window))
    baseline_iou = np.asarray(
        [evaluator.frame_metrics(frame, polygon)[0] for frame, polygon in enumerate(spatial)],
        dtype=np.float64,
    )
    output = spatial.copy()
    selected = fractions.copy()
    gap = max(float(minimum_gap_ratio) / max(int(target), 1), 1e-6)
    order = list(range(len(output)))
    for pass_index in range(max(1, int(passes))):
        if pass_index % 2:
            order.reverse()
        for frame in order:
            candidates: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
            for alpha in alpha_grid:
                trial_fraction = (
                    (1.0 - float(alpha)) * fractions[frame]
                    + float(alpha) * smooth_target[frame]
                )
                trial_fraction = _project_order(trial_fraction[None, :], gap)[0]
                trial = _sample_fraction(contours[frame], trial_fraction)
                if has_self_intersection(trial):
                    continue
                iou, recall = evaluator.frame_metrics(frame, trial)
                if recall + 1e-12 < float(recall_floor):
                    continue
                if iou + 1e-12 < baseline_iou[frame] - float(iou_loss_budget):
                    continue
                temporal = _local_temporal(output, frame, trial)
                quality_loss = max(0.0, float(baseline_iou[frame] - iou))
                score = quality_loss + float(temporal_weight) * temporal
                candidates.append((score, -iou, temporal, trial_fraction, trial))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            selected[frame] = candidates[0][3]
            output[frame] = candidates[0][4]
    return output
