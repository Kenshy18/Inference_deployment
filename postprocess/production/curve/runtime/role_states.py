"""Adapt Production polygon role candidates to Catmull--Rom control points."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from production.polygon.runtime.role_candidate_pool import build_role_candidate

from .curve_fit import catmull_rom_basis_matrix
from .keyframe_dp import render_control_sequence


_CORE_ROLE_IDS = {
    "男性器": ("C02_125", "A06_K3", "D6_R5_P1"),
    "結合部分": ("C02_125", "A06", "VF8_P1"),
}


def curve_role_ids(label: str) -> tuple[str, ...]:
    """Return the minimal non-raw role set used for interval flexibility."""

    return tuple(_CORE_ROLE_IDS.get(str(label), ()))


def polygon_role_curve_states(
    controls: np.ndarray,
    frame_numbers: np.ndarray,
    role_ids: tuple[str, ...],
    *,
    renderer,
    samples_per_segment: int,
    anchor_weight: float = 0.10,
    use_sampled_boundaries: bool = False,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return raw plus role-derived Catmull states with persistent P phase.

    By default the temporal role rules operate directly on the interpolating
    Catmull points.  That preserves their count and cyclic correspondence and
    avoids a dense boundary render/least-squares projection.  The optional
    sampled-boundary path is retained for controlled experiments only; it
    still projects back to P and never introduces free Bezier handles.
    """

    source = np.asarray(controls, dtype=np.float64)
    frames = np.asarray(frame_numbers, dtype=np.int64)
    if source.ndim != 3 or source.shape[2] != 2:
        raise ValueError("controls must have shape (frames,points,2)")
    if frames.shape != (len(source),):
        raise ValueError("frame_numbers must match controls")
    labels = ("raw", *tuple(str(value) for value in role_ids))
    states = np.empty(
        (len(source), len(labels), source.shape[1], 2),
        dtype=np.float64,
    )
    states[:, 0] = source
    sampled = None
    role_anchors = source
    basis = None
    gram = None
    if bool(use_sampled_boundaries):
        sampled = render_control_sequence(renderer, source)
        role_anchors = sampled
        basis = catmull_rom_basis_matrix(
            int(source.shape[1]),
            int(samples_per_segment),
        )
        weight = float(anchor_weight)
        gram = basis.T @ basis + weight * np.eye(
            source.shape[1],
            dtype=np.float64,
        )
    run = SimpleNamespace(
        anchors=np.ascontiguousarray(role_anchors[:, None, :, :], dtype=np.float64),
        contour_count=1,
        frame_numbers=frames,
    )
    for frame in range(len(source)):
        for state, role_id in enumerate(role_ids, start=1):
            target = np.asarray(
                build_role_candidate(run, frame, str(role_id))[0],
                dtype=np.float64,
            )
            if bool(use_sampled_boundaries):
                assert sampled is not None and basis is not None and gram is not None
                if target.shape != sampled[frame].shape:
                    raise RuntimeError("role candidate changed sampled boundary size")
                right = basis.T @ target + float(anchor_weight) * source[frame]
                states[frame, state] = np.linalg.solve(gram, right)
            else:
                if target.shape != source[frame].shape:
                    raise RuntimeError("role candidate changed control-point count")
                states[frame, state] = target
    return np.ascontiguousarray(states), labels


__all__ = ("curve_role_ids", "polygon_role_curve_states")
