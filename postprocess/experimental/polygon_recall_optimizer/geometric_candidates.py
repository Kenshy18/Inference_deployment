"""Analytic Production-independent polygon anchor hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experimental.polygon_recall_optimizer.fixed_budget import RawMask
from experimental.polygon_recall_optimizer.temporal_candidates import _rigid_align
from overlay_renderer.keyframe_cache import Component, Keyframe, _numpy_resample


@dataclass(frozen=True)
class GeometricCandidate:
    name: str
    keyframe: Keyframe


def _key(frame: int, points: np.ndarray) -> Keyframe:
    return Keyframe(
        int(frame),
        ((0, Component("polygon", np.asarray(points, dtype=np.float64).tolist())),),
    )


def _principal_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return center, basis


def _axis_scale(
    points: np.ndarray, center: np.ndarray, basis: np.ndarray, sx: float, sy: float
) -> np.ndarray:
    local = (points - center) @ basis
    local *= np.asarray((float(sx), float(sy)), dtype=np.float64)
    return local @ basis.T + center


def build_geometric_candidates(
    frame: int,
    raw_by_frame: dict[int, RawMask],
    *,
    point_count: int,
    radius: int = 5,
    envelope_quantile: float = 0.95,
) -> tuple[GeometricCandidate, ...]:
    """Return anisotropic and asymmetric local-envelope anchor shapes."""

    raw = raw_by_frame[int(frame)]
    reference = _numpy_resample(
        np.asarray(raw.primary_points, dtype=np.float64), int(point_count)
    )
    center, basis = _principal_basis(reference)
    output = [
        GeometricCandidate(
            "axis_major", _key(frame, _axis_scale(reference, center, basis, 1.18, 1.04))
        ),
        GeometricCandidate(
            "axis_minor", _key(frame, _axis_scale(reference, center, basis, 1.04, 1.18))
        ),
        GeometricCandidate(
            "axis_balanced",
            _key(frame, _axis_scale(reference, center, basis, 1.12, 1.12)),
        ),
    ]
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
    reference_local = (reference - center) @ basis
    aligned_local = (aligned - center[None, None, :]) @ basis
    current_low = np.min(reference_local, axis=0)
    current_high = np.max(reference_local, axis=0)
    neighbour_low = np.quantile(
        np.min(aligned_local, axis=1), 1.0 - float(envelope_quantile), axis=0
    )
    neighbour_high = np.quantile(
        np.max(aligned_local, axis=1), float(envelope_quantile), axis=0
    )
    target_low = np.minimum(current_low, neighbour_low)
    target_high = np.maximum(current_high, neighbour_high)
    normalized = (reference_local - current_low) / np.maximum(
        current_high - current_low, 1e-9
    )
    envelope_local = target_low + normalized * (target_high - target_low)
    envelope = envelope_local @ basis.T + center
    output.append(GeometricCandidate("axis_envelope", _key(frame, envelope)))
    return tuple(output)


__all__ = ["GeometricCandidate", "build_geometric_candidates"]
