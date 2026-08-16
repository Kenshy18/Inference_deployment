"""Production-independent temporal polygon candidate generation.

Only tracked raw-mask polygons are consumed.  Neighbouring contours are
aligned to the candidate frame with a rigid Procrustes transform before a
point-wise temporal statistic is taken, so ordinary object translation and
rotation do not turn a temporal window into a swept-area union.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from overlay_renderer.keyframe_cache import Component, Keyframe, _numpy_resample

from .fixed_budget import RawMask


@dataclass(frozen=True)
class TemporalCandidate:
    name: str
    keyframe: Keyframe


def _align_order(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    count = len(candidate)
    positions = np.arange(count)
    shifts = np.arange(count)[:, None]
    indices = (positions[None, :] - shifts) % count
    forward = candidate[indices]
    reverse = candidate[::-1][indices]
    variants = np.concatenate((forward, reverse), axis=0)
    errors = np.mean(
        np.sum(np.square(variants - reference[None, :, :]), axis=2), axis=1
    )
    return variants[int(np.argmin(errors))]


def _rigid_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align translation and rotation while retaining size/non-rigid change."""

    ordered = _align_order(reference, candidate)
    reference_center = np.mean(reference, axis=0)
    candidate_center = np.mean(ordered, axis=0)
    left = ordered - candidate_center
    right = reference - reference_center
    covariance = left.T @ right
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return left @ rotation + reference_center


def _polygon_keyframe(frame: int, points: np.ndarray) -> Keyframe:
    return Keyframe(
        int(frame),
        ((0, Component("polygon", np.asarray(points, dtype=np.float64).tolist())),),
    )


def _temporal_shapes(
    reference: np.ndarray,
    aligned: np.ndarray,
    *,
    recall_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust central and outward-coverage contours."""

    central = np.median(aligned, axis=0)
    center = np.mean(central, axis=0)
    radial = central - center
    lengths = np.linalg.norm(radial, axis=1)
    fallback = reference - np.mean(reference, axis=0)
    fallback_lengths = np.linalg.norm(fallback, axis=1)
    directions = np.divide(
        radial,
        lengths[:, None],
        out=np.divide(
            fallback,
            np.maximum(fallback_lengths[:, None], 1e-9),
            out=np.zeros_like(fallback),
            where=fallback_lengths[:, None] > 1e-9,
        ),
        where=lengths[:, None] > 1e-9,
    )
    tangents = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
    offsets = aligned - center[None, None, :]
    radial_samples = np.sum(offsets * directions[None, :, :], axis=2)
    tangent_samples = np.sum(offsets * tangents[None, :, :], axis=2)
    coverage_radius = np.quantile(
        radial_samples, float(recall_quantile), axis=0
    )
    coverage_tangent = np.median(tangent_samples, axis=0)
    coverage = (
        center
        + coverage_radius[:, None] * directions
        + coverage_tangent[:, None] * tangents
    )
    return central, coverage


def build_temporal_candidates(
    frame: int,
    raw_by_frame: dict[int, RawMask],
    *,
    point_count: int,
    window_radii: tuple[int, int, int] = (2, 5, 10),
    recall_quantile: float = 0.90,
) -> tuple[TemporalCandidate, ...]:
    """Create raw + short/medium/long IoU/Recall temporal states.

    The temporal windows are clipped to the current track/cut segment by the
    caller-provided ``raw_by_frame`` mapping.  No Production keyframe shape is
    read or used.
    """

    raw = raw_by_frame[int(frame)]
    reference = _numpy_resample(
        np.asarray(raw.primary_points, dtype=np.float64), int(point_count)
    )
    output = [TemporalCandidate("initial_raw", _polygon_keyframe(frame, reference))]
    labels = ("short", "medium", "long")
    for label, radius in zip(labels, window_radii, strict=True):
        neighbours = [
            candidate
            for candidate_frame, candidate in sorted(raw_by_frame.items())
            if abs(int(candidate_frame) - int(frame)) <= int(radius)
        ]
        aligned = np.stack(
            [
                _rigid_align(
                    reference,
                    _numpy_resample(
                        np.asarray(candidate.primary_points, dtype=np.float64),
                        int(point_count),
                    ),
                )
                for candidate in neighbours
            ],
            axis=0,
        )
        central, coverage = _temporal_shapes(
            reference, aligned, recall_quantile=recall_quantile
        )
        output.extend(
            (
                TemporalCandidate(
                    f"{label}_iou", _polygon_keyframe(frame, central)
                ),
                TemporalCandidate(
                    f"{label}_recall", _polygon_keyframe(frame, coverage)
                ),
            )
        )
    return tuple(output)


__all__ = ["TemporalCandidate", "build_temporal_candidates"]
