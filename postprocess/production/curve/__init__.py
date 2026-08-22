"""Production closed Catmull--Rom mask postprocessing."""

from .runtime import (
    CurvePointRefineConfig,
    FitConfig,
    KeyframeDpConfig,
    KeyframeDpResult,
    fit_sequence,
    optimize_multistate_keyframes,
    sample_closed_curve,
)
from .stage import ProductionCurveStage

__all__ = (
    "CurvePointRefineConfig",
    "FitConfig",
    "KeyframeDpConfig",
    "KeyframeDpResult",
    "fit_sequence",
    "optimize_multistate_keyframes",
    "sample_closed_curve",
    "ProductionCurveStage",
)
