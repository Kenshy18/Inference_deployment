"""Linear whole-curve fitting for the fixed Catmull--Rom model.

The editor curve is linear in its interpolation points ``P`` because every
Bezier handle is a fixed linear combination of neighbouring points.  This
module uses that property to fit the *whole* sampled curve instead of merely
placing ``P`` on the source contour.  Free Bezier handles are never created.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import sampling_matrix


@dataclass(frozen=True, slots=True)
class WholeCurveFitStats:
    boundary_rms_before: float
    boundary_rms_after: float
    mean_control_shift: float
    maximum_control_shift: float
    clipped_controls: int


def catmull_rom_basis_matrix(
    control_point_count: int,
    samples_per_segment: int,
) -> np.ndarray:
    """Return the exact matrix mapping ``P`` to sampled curve positions."""
    return sampling_matrix(int(control_point_count), int(samples_per_segment))


def contour_arc_targets(
    aligned_dense: np.ndarray,
    persistent_dense_indices: np.ndarray,
    samples_per_segment: int,
) -> np.ndarray:
    """Sample the source arc assigned to every persistent curve segment."""
    dense = np.asarray(aligned_dense, dtype=np.float64)
    indices = np.asarray(persistent_dense_indices, dtype=np.int64).reshape(-1)
    samples = int(samples_per_segment)
    if dense.ndim != 3 or dense.shape[2] != 2:
        raise ValueError("aligned_dense must have shape (frames, samples, 2)")
    if len(indices) < 3 or samples < 1:
        raise ValueError("invalid control-point or curve-sample count")
    dense_count = int(dense.shape[1])
    if np.any(indices < 0) or np.any(indices >= dense_count):
        raise ValueError("persistent dense index is out of range")
    if len(set(int(value) for value in indices)) != len(indices):
        raise ValueError("persistent dense indices must be unique")
    output = np.empty((len(dense), len(indices) * samples, 2), dtype=np.float64)
    sample_offsets = np.arange(samples, dtype=np.float64) / float(samples)
    for segment, start in enumerate(indices):
        end = int(indices[(segment + 1) % len(indices)])
        distance = (end - int(start)) % dense_count
        if distance <= 0:
            distance += dense_count
        positions = float(start) + sample_offsets * float(distance)
        floors = np.floor(positions)
        left = floors.astype(np.int64) % dense_count
        right = (left + 1) % dense_count
        alpha = positions - floors
        destination = slice(segment * samples, (segment + 1) * samples)
        output[:, destination] = (1.0 - alpha)[None, :, None] * dense[:, left] + alpha[
            None, :, None
        ] * dense[:, right]
    return np.ascontiguousarray(output, dtype=np.float64)


def _smooth_temporal_correction(
    correction: np.ndarray,
    weight: float,
) -> np.ndarray:
    value = np.asarray(correction, dtype=np.float64)
    if len(value) <= 1 or float(weight) <= 0.0:
        return value.copy()
    frame_count = len(value)
    amount = float(weight)
    # The normal matrix is symmetric tridiagonal.  Solving it as a dense
    # T-by-T matrix made this one regularizer cubic in video length and was
    # the largest spatial-fit cost on long Production chunks. Thomas elimination
    # is the same float64 linear system in O(T * P), with bounded memory.
    diagonal = np.full((frame_count,), 1.0 + 2.0 * amount, dtype=np.float64)
    diagonal[0] = 1.0 + amount
    diagonal[-1] = 1.0 + amount
    off_diagonal = -amount
    right = value.reshape(frame_count, -1).copy()
    for frame in range(1, frame_count):
        factor = off_diagonal / diagonal[frame - 1]
        diagonal[frame] -= factor * off_diagonal
        right[frame] -= factor * right[frame - 1]
    output = np.empty_like(right)
    output[-1] = right[-1] / diagonal[-1]
    for frame in range(frame_count - 2, -1, -1):
        output[frame] = (right[frame] - off_diagonal * output[frame + 1]) / diagonal[
            frame
        ]
    return output.reshape(value.shape)


def _clip_control_corrections(
    base: np.ndarray,
    correction: np.ndarray,
    maximum_chord_fraction: float,
) -> tuple[np.ndarray, int]:
    if float(maximum_chord_fraction) <= 0.0:
        return np.asarray(correction, dtype=np.float64).copy(), 0
    source = np.asarray(base, dtype=np.float64)
    value = np.asarray(correction, dtype=np.float64).copy()
    previous = np.linalg.norm(source - np.roll(source, 1, axis=1), axis=2)
    following = np.linalg.norm(np.roll(source, -1, axis=1) - source, axis=2)
    radius = float(maximum_chord_fraction) * np.minimum(previous, following)
    magnitude = np.linalg.norm(value, axis=2)
    clipped = magnitude > radius + 1e-12
    scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-12))
    value *= scale[:, :, None]
    return value, int(np.count_nonzero(clipped))


def fit_whole_curve_controls(
    aligned_dense: np.ndarray,
    persistent_dense_indices: np.ndarray,
    initial_controls: np.ndarray,
    *,
    samples_per_segment: int,
    anchor_weight: float = 1.0,
    temporal_correction_weight: float = 2.0,
    maximum_correction_chord_fraction: float = 0.40,
) -> tuple[np.ndarray, WholeCurveFitStats]:
    """Fit P under the exact fixed-handle model and preserve point identity.

    The per-frame objective is ``||A P - Y||^2 + w ||P - P_initial||^2``.
    Only the fitted correction is smoothed in time, so ordinary translation,
    rotation and scale changes are not delayed by the regularizer.
    """
    initial = np.asarray(initial_controls, dtype=np.float64)
    if initial.ndim != 3 or initial.shape[2] != 2:
        raise ValueError("initial_controls must have shape (frames, points, 2)")
    if float(anchor_weight) < 0.0:
        raise ValueError("anchor_weight must be nonnegative")
    basis = catmull_rom_basis_matrix(int(initial.shape[1]), int(samples_per_segment))
    targets = contour_arc_targets(
        aligned_dense,
        persistent_dense_indices,
        int(samples_per_segment),
    )
    if len(targets) != len(initial):
        raise ValueError("dense contours and controls must have equal frame count")
    gram = basis.T @ basis + float(anchor_weight) * np.eye(
        initial.shape[1], dtype=np.float64
    )
    right = np.einsum("nm,tmc->tnc", basis.T, targets)
    right += float(anchor_weight) * initial
    packed = right.transpose(1, 0, 2).reshape(initial.shape[1], -1)
    fitted = (
        np.linalg.solve(gram, packed)
        .reshape(initial.shape[1], len(initial), 2)
        .transpose(1, 0, 2)
    )
    correction = _smooth_temporal_correction(
        fitted - initial, float(temporal_correction_weight)
    )
    correction, clipped = _clip_control_corrections(
        initial, correction, float(maximum_correction_chord_fraction)
    )
    output = np.ascontiguousarray(initial + correction, dtype=np.float64)
    before = np.einsum("mn,tnc->tmc", basis, initial) - targets
    after = np.einsum("mn,tnc->tmc", basis, output) - targets
    shift = np.linalg.norm(correction, axis=2)
    return output, WholeCurveFitStats(
        boundary_rms_before=float(np.sqrt(np.mean(before * before))),
        boundary_rms_after=float(np.sqrt(np.mean(after * after))),
        mean_control_shift=float(np.mean(shift)),
        maximum_control_shift=float(np.max(shift)),
        clipped_controls=int(clipped),
    )


__all__ = (
    "WholeCurveFitStats",
    "catmull_rom_basis_matrix",
    "contour_arc_targets",
    "fit_whole_curve_controls",
)
