"""Adaptive-vertex end-to-end postprocess candidate.

This package is deliberately not registered as the default Production
pipeline.  It exposes one explicit contract and one explicit runner so the
complete topology/NMS/polygon/keyframe stack can be validated before the
eventual promotion.
"""

from .config import (
    CANDIDATE,
    CandidateConfig,
    legacy_fixed14_candidate,
    with_interval_evaluation,
    with_target_interval,
)

__all__ = (
    "CANDIDATE",
    "CandidateConfig",
    "legacy_fixed14_candidate",
    "with_interval_evaluation",
    "with_target_interval",
)
