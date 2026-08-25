"""Deterministic state palettes for the Production Catmull--Rom DP.

The editable values remain the interpolation points ``P``. Bézier handles are
always derived by the fixed uniform Catmull--Rom contract at render time.
"""

from __future__ import annotations

import numpy as np


def _isotropic_state(controls: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(controls, dtype=np.float64)
    center = np.mean(source, axis=1, keepdims=True)
    return np.ascontiguousarray(center + float(scale) * (source - center))


def _clamped_endpoint_fit(
    controls: np.ndarray,
    start: int,
    end: int,
    endpoint: int,
    maximum_correction_fraction: float,
) -> np.ndarray:
    """Fit a linear temporal endpoint and bound movement from the raw key."""

    values = np.asarray(controls[start : end + 1], dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    design = np.column_stack((1.0 - alpha, alpha))
    gram = design.T @ design + 1e-8 * np.eye(2, dtype=np.float64)
    fitted = np.linalg.solve(gram, design.T @ values.reshape(len(values), -1))
    candidate = fitted[int(endpoint)].reshape(values.shape[1:])
    reference = np.asarray(controls[start if endpoint == 0 else end])
    centered = reference - np.mean(reference, axis=0, keepdims=True)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    maximum = max(1.0, float(maximum_correction_fraction) * radius)
    delta = candidate - reference
    rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    if rms > maximum:
        candidate = reference + (maximum / max(rms, 1e-12)) * delta
    return np.ascontiguousarray(candidate)


def temporal_endpoint_states(
    controls: np.ndarray,
    *,
    horizon: int,
    scale: float,
    maximum_correction_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward/backward interval-aware endpoint candidates.

    Linear motion remains represented by interpolation between keys. These
    candidates alter only the residual endpoint shape needed to explain the
    intervening frames.
    """

    source = np.asarray(controls, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 2:
        raise ValueError("controls must have shape (frames, points, 2)")
    bounded_horizon = max(2, int(horizon))
    forward = source.copy()
    backward = source.copy()
    for frame in range(len(source)):
        right = min(len(source) - 1, frame + bounded_horizon)
        if right - frame >= 2:
            forward[frame] = _clamped_endpoint_fit(
                source,
                frame,
                right,
                0,
                float(maximum_correction_fraction),
            )
        left = max(0, frame - bounded_horizon)
        if frame - left >= 2:
            backward[frame] = _clamped_endpoint_fit(
                source,
                left,
                frame,
                1,
                float(maximum_correction_fraction),
            )
    if abs(float(scale) - 1.0) > 1e-12:
        forward = _isotropic_state(forward, float(scale))
        backward = _isotropic_state(backward, float(scale))
    return np.ascontiguousarray(forward), np.ascontiguousarray(backward)


def interval_aware_fallback_states(
    controls: np.ndarray,
    *,
    target_interval: int,
    coverage_scale_maximum: float = 1.06,
    endpoint_scale_maximum: float = 1.08,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build four complementary states for difficult sparse streams."""

    source = np.asarray(controls, dtype=np.float64)
    target = max(1, int(target_interval))
    coverage_scale = min(
        float(coverage_scale_maximum),
        1.0 + 0.02 * max(0, target - 2),
    )
    endpoint_scale = min(
        float(endpoint_scale_maximum),
        1.0 + 0.02 * max(0, target - 2),
    )
    values = [source, _isotropic_state(source, coverage_scale)]
    labels = ["scale_1.000", f"scale_{coverage_scale:.3f}"]
    forward, backward = temporal_endpoint_states(
        source,
        horizon=max(2, target),
        scale=endpoint_scale,
        maximum_correction_fraction=0.10,
    )
    values.extend((forward, backward))
    labels.extend(
        (
            f"forward_ls_h{target}_s{endpoint_scale:.3f}",
            f"backward_ls_h{target}_s{endpoint_scale:.3f}",
        )
    )
    return np.ascontiguousarray(np.stack(values, axis=1)), tuple(labels)


__all__ = ("interval_aware_fallback_states", "temporal_endpoint_states")
