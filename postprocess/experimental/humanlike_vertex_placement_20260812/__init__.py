"""Fast corner-aware polygon placement experiments."""

from .spatial import (
    align_polygon_sequence,
    rdp_fixed_count,
    visvalingam_fixed_count,
)
from .candidate import quality_guarded_vertex_placement

__all__ = [
    "align_polygon_sequence",
    "rdp_fixed_count",
    "visvalingam_fixed_count",
    "quality_guarded_vertex_placement",
]
