"""Exact, sparse quality repair for persistent line-fit polygons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import (
    RasterSequenceEvaluator,
    _best_phase,
    has_self_intersection,
)

from .persistent_line_fit import persistent_line_fit_sequence
from .spatial import rdp_fixed_count


@dataclass(frozen=True)
class RepairStats:
    frames: int
    repaired_frames: int
    fallback_frames: int
    tested_blends: int


def persistent_line_fit_quality_guarded(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    recall_floor: float = 0.97,
    iou_floor: float = 0.95,
    blend_steps: int = 32,
    **line_fit_options,
) -> tuple[np.ndarray, RepairStats]:
    """Keep persistent identities, repairing only exact quality violations."""
    references = [np.asarray(value, dtype=np.float64) for value in polygons]
    output = persistent_line_fit_sequence(references, target, **line_fit_options)
    evaluator = RasterSequenceEvaluator(references)
    repaired = 0
    fallbacks = 0
    tested = 0
    for frame in range(len(output)):
        iou, recall = evaluator.frame_metrics(frame, output[frame])
        if iou >= float(iou_floor) and recall >= float(recall_floor):
            continue
        fallback = rdp_fixed_count(references[frame], int(target))
        fallback = _best_phase(
            output[frame], fallback, allow_reverse=False, procrustes=False
        )
        original = output[frame].copy()
        accepted = None
        for alpha in np.linspace(1.0 / blend_steps, 1.0, blend_steps):
            tested += 1
            candidate = (1.0 - float(alpha)) * original + float(alpha) * fallback
            if has_self_intersection(candidate):
                continue
            candidate_iou, candidate_recall = evaluator.frame_metrics(frame, candidate)
            if candidate_iou >= float(iou_floor) and candidate_recall >= float(recall_floor):
                accepted = candidate
                break
        if accepted is None:
            fallback_iou, fallback_recall = evaluator.frame_metrics(frame, fallback)
            if (
                not has_self_intersection(fallback)
                and fallback_iou >= float(iou_floor)
                and fallback_recall >= float(recall_floor)
            ):
                accepted = fallback
                fallbacks += 1
        if accepted is not None:
            output[frame] = accepted
            repaired += 1
    return output, RepairStats(
        frames=len(output),
        repaired_frames=repaired,
        fallback_frames=fallbacks,
        tested_blends=tested,
    )
