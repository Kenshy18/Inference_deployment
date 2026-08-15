"""Track-wise temporal polygon vertex decimation experiment."""

from .optimizer import (
    DecimationConfig,
    SequenceMetrics,
    TemporalDecimationResult,
    current_equal_arc_baseline,
    optimize_temporal_vertices,
)

__all__ = [
    "DecimationConfig",
    "SequenceMetrics",
    "TemporalDecimationResult",
    "current_equal_arc_baseline",
    "optimize_temporal_vertices",
]
