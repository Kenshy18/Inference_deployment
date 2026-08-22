"""Closed uniform Catmull--Rom curve fitting experiment.

Only the interpolation points are editable.  Cubic Bezier handles are always
derived from adjacent interpolation points with the fixed factor ``1 / 6``.
"""

from production.curve.runtime.curve_fit import (
    WholeCurveFitStats,
    catmull_rom_basis_matrix,
    contour_arc_targets,
    fit_whole_curve_controls,
)
from production.curve.runtime.fitter import FitConfig, SequenceFitResult, fit_sequence
from production.curve.runtime.keyframe_dp import (
    KeyframeDpConfig,
    KeyframeDpResult,
    catmull_rom_renderer,
    optimize_keyframes,
    polygon_renderer,
)
from production.curve.runtime.model import (
    bezier_segments,
    evaluate_cubic,
    normalize_control_points,
    sample_closed_curve,
)
from production.curve.runtime.multistate_dp import (
    CurvePointRefineConfig,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)

__all__ = (
    "FitConfig",
    "KeyframeDpConfig",
    "KeyframeDpResult",
    "CurvePointRefineConfig",
    "SequenceFitResult",
    "WholeCurveFitStats",
    "bezier_segments",
    "catmull_rom_basis_matrix",
    "catmull_rom_renderer",
    "contour_arc_targets",
    "evaluate_cubic",
    "fit_sequence",
    "fit_whole_curve_controls",
    "isotropic_curve_states",
    "normalize_control_points",
    "optimize_keyframes",
    "optimize_multistate_keyframes",
    "polygon_renderer",
    "sample_closed_curve",
)
