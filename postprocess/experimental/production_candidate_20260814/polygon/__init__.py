"""Approved spatial and temporal polygon pipeline."""

from .candidate_palette import role_ids
from .engine import run_polygon_optimizer
from .spatial import build_spatial_track
from .stage import CandidatePolygonStage

__all__ = (
    "CandidatePolygonStage",
    "build_spatial_track",
    "role_ids",
    "run_polygon_optimizer",
)
