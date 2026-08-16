"""Quality-gated temporal correspondence for corner-aware polygon vertices.

The spatial/native DP remains the quality anchor.  This stage only slides each
vertex along its source contour towards a robust temporal trajectory.  Every
accepted move is checked against exact per-frame Recall/IoU, so correspondence
cannot silently buy stability by breaking the mask contract.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    has_self_intersection,
    orient_ccw,
)

from .smoothed_rdp import (
    _median_smooth,
    _point_fractions,
    _project_order,
    _sample_fraction,
    _temporal_unwrap,
)


class _FrameEvaluator(Protocol):
    def frame_metrics(self, frame: int, polygon: np.ndarray) -> tuple[float, float]: ...


def _similarity_residual(left: np.ndarray, right: np.ndarray) -> float:
    left_center = np.mean(left, axis=0)
    right_center = np.mean(right, axis=0)
    left_zero = left - left_center
    right_zero = right - right_center
    covariance = left_zero.T @ right_zero
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left_zero * left_zero), 1e-12))
    prediction = scale * (left_zero @ rotation) + right_center
    residual = np.linalg.norm(right - prediction, axis=1)
    area = 0.5 * abs(float(np.sum(
        right[:, 0] * np.roll(right[:, 1], -1)
        - np.roll(right[:, 0], -1) * right[:, 1]
    )))
    radius = math.sqrt(max(area, 1.0) / math.pi)
    return float(np.mean(residual) / max(radius, 1.0))


def _local_correspondence_cost(
    sequence: np.ndarray,
    frame: int,
    candidate: np.ndarray,
) -> float:
    values = []
    if frame > 0:
        values.append(_similarity_residual(sequence[frame - 1], candidate))
    if frame + 1 < len(sequence):
        values.append(_similarity_residual(candidate, sequence[frame + 1]))
    return float(np.mean(values)) if values else 0.0


def quality_gated_persistent_correspondence(
    polygons: Iterable[np.ndarray],
    baseline: np.ndarray,
    evaluator: _FrameEvaluator,
    *,
    temporal_window: int = 5,
    recall_floor: float = 0.97,
    iou_floor: float = 0.95,
    iou_loss_budget: float = 0.002,
    quality_loss_weight: float = 0.20,
    minimum_gap_ratio: float = 0.02,
    passes: int = 2,
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    target_mode: str = "fractions",
) -> np.ndarray:
    """Reduce tangential ID drift while retaining the baseline shape contract."""
    contours = [orient_ccw(value) for value in polygons]
    output = np.asarray(baseline, dtype=np.float64).copy()
    if len(output) == 0:
        return output
    target = int(output.shape[1])
    fractions = np.asarray(
        [
            _point_fractions(contour, points)
            for contour, points in zip(contours, output, strict=True)
        ],
        dtype=np.float64,
    )
    fractions = _temporal_unwrap(fractions)
    if target_mode == "fractions":
        smooth_target = _median_smooth(fractions, int(temporal_window))
        base_gaps = target_gaps = None
    elif target_mode in {"arc_gaps", "uniform_gaps"}:
        base_gaps = np.diff(
            np.concatenate([fractions, fractions[:, :1] + 1.0], axis=1),
            axis=1,
        )
        if target_mode == "arc_gaps":
            target_gaps = _median_smooth(base_gaps, int(temporal_window))
            target_gaps = np.maximum(target_gaps, 1e-6)
            target_gaps /= np.sum(target_gaps, axis=1, keepdims=True)
        else:
            target_gaps = np.full_like(base_gaps, 1.0 / float(target))
        smooth_target = None
    else:
        raise ValueError(f"unknown target_mode: {target_mode}")
    selected = fractions.copy()
    baseline_iou = np.asarray(
        [evaluator.frame_metrics(frame, polygon)[0] for frame, polygon in enumerate(output)],
        dtype=np.float64,
    )
    gap = max(float(minimum_gap_ratio) / max(target, 1), 1e-6)
    forward = list(range(len(output)))
    for pass_index in range(max(1, int(passes))):
        order = forward if pass_index % 2 == 0 else list(reversed(forward))
        for frame in order:
            choices = []
            for alpha in alpha_grid:
                if target_mode == "fractions":
                    trial_fractions = (
                        (1.0 - float(alpha)) * fractions[frame]
                        + float(alpha) * smooth_target[frame]
                    )
                else:
                    trial_gaps = (
                        (1.0 - float(alpha)) * base_gaps[frame]
                        + float(alpha) * target_gaps[frame]
                    )
                    trial_gaps = np.maximum(trial_gaps, 1e-6)
                    trial_gaps /= float(np.sum(trial_gaps))
                    trial_fractions = fractions[frame, 0] + np.concatenate(
                        [[0.0], np.cumsum(trial_gaps[:-1])]
                    )
                trial_fractions = _project_order(trial_fractions[None, :], gap)[0]
                trial = _sample_fraction(contours[frame], trial_fractions)
                if has_self_intersection(trial):
                    continue
                iou, recall = evaluator.frame_metrics(frame, trial)
                required_iou = max(float(iou_floor), baseline_iou[frame] - float(iou_loss_budget))
                if recall + 1e-12 < float(recall_floor) or iou + 1e-12 < required_iou:
                    continue
                correspondence = _local_correspondence_cost(output, frame, trial)
                quality_loss = max(0.0, baseline_iou[frame] - float(iou))
                score = correspondence + float(quality_loss_weight) * quality_loss
                choices.append((score, -float(iou), -float(recall), trial_fractions, trial))
            if choices:
                choices.sort(key=lambda value: (value[0], value[1], value[2]))
                selected[frame] = choices[0][3]
                output[frame] = choices[0][4]
    return output
