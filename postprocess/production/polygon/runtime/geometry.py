"""Small deterministic geometry helpers shared by Production candidates."""

from __future__ import annotations

import numpy as np


def align_order(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Choose the cyclic direction/offset closest to ``reference``."""
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


def rigid_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align translation and rotation while preserving size and deformation."""
    ordered = align_order(reference, candidate)
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


def temporal_shapes(
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
    coverage_radius = np.quantile(radial_samples, float(recall_quantile), axis=0)
    coverage_tangent = np.median(tangent_samples, axis=0)
    coverage = (
        center
        + coverage_radius[:, None] * directions
        + coverage_tangent[:, None] * tangents
    )
    return central, coverage


def principal_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return center, basis


def axis_scale(
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    local = (points - center) @ basis
    local *= np.asarray((float(scale_x), float(scale_y)), dtype=np.float64)
    return local @ basis.T + center


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """Equal-arclength resampling with the parity-frozen NumPy arithmetic."""
    value = np.asarray(points, dtype=np.float64)
    following = np.roll(value, -1, axis=0)
    lengths = np.linalg.norm(following - value, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-6:
        return np.repeat(value[:1], int(count), axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, perimeter, int(count), endpoint=False)
    output = np.empty((int(count), 2), dtype=np.float64)
    for index, distance in enumerate(samples):
        segment = min(
            max(
                int(np.searchsorted(cumulative, distance, side="right") - 1),
                0,
            ),
            len(value) - 1,
        )
        ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-6)
        output[index] = (1.0 - ratio) * value[segment] + ratio * following[segment]
    return output


__all__ = (
    "align_order",
    "axis_scale",
    "principal_basis",
    "resample_closed",
    "rigid_align",
    "temporal_shapes",
)
