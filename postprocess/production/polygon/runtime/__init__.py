"""Parity-frozen polygon optimizer owned by the Production package.

This package contains the complete runtime needed by
``ProductionPolygonStage``.  It deliberately does not import from
``postprocess.experimental`` so a deployed backend remains self-contained.
"""

from .candidate_config import CANDIDATE, CandidateConfig
from .engine import run_polygon_optimizer

__all__ = ("CANDIDATE", "CandidateConfig", "run_polygon_optimizer")
