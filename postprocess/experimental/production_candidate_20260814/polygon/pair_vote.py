"""Pair-vote contract and exact evaluator boundary."""

from __future__ import annotations

import os
from collections.abc import MutableMapping

from experimental.new_production.fast_pair_vote import ExactPairVoteEvaluator

from ..config import CANDIDATE, CandidateConfig


def pair_vote_environment(
    config: CandidateConfig = CANDIDATE,
) -> dict[str, str]:
    config.validate()
    return {
        "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
        "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": str(
            config.runtime.pair_vote_threads
        ),
    }


def apply_pair_vote_environment(
    environment: MutableMapping[str, str] | None = None,
    config: CandidateConfig = CANDIDATE,
) -> MutableMapping[str, str]:
    target = os.environ if environment is None else environment
    target.update(pair_vote_environment(config))
    return target


__all__ = (
    "ExactPairVoteEvaluator",
    "apply_pair_vote_environment",
    "pair_vote_environment",
)
