"""CPU-only closed Catmull--Rom fitting and exact temporal optimization."""

from .fitter import FitConfig, SequenceFitResult, fit_sequence
from .keyframe_dp import KeyframeDpConfig, KeyframeDpResult, catmull_rom_renderer
from .model import bezier_segments, sample_closed_curve
from .multistate_dp import (
    CurvePointRefineConfig,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)

__all__ = (
    "CurvePointRefineConfig",
    "FitConfig",
    "KeyframeDpConfig",
    "KeyframeDpResult",
    "SequenceFitResult",
    "bezier_segments",
    "catmull_rom_renderer",
    "fit_sequence",
    "isotropic_curve_states",
    "optimize_multistate_keyframes",
    "sample_closed_curve",
)
