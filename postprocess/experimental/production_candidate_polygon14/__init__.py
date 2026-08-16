"""Production candidate: adaptive 14/16/18/20 points plus frozen temporal DP."""

from .config import CANDIDATE, Polygon14CandidateConfig
from .spatial import SpatialBuildStats, build_spatial_track

__all__ = (
    "CANDIDATE",
    "Polygon14CandidateConfig",
    "SpatialBuildStats",
    "build_spatial_track",
)
