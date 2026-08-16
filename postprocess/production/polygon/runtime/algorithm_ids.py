"""Stable identifiers emitted by the deployed polygon optimizer.

These names describe the current Production contract. Keeping them in one
place prevents historical experiment names from leaking into release
manifests while leaving the parity-frozen geometry implementation untouched.
"""

PHASE1_RAW_ALGORITHM_ID = "production_polygon_v3_raw_only_hard_min_recall_no_pair_vote"
PHASE2_PER_KEY_PAIR_VOTE_ALGORITHM_ID = (
    "production_polygon_v3_multishape_hard_min_recall_per_key_pair_vote"
)
PHASE2_CONSTRAINED_PAIR_VOTE_ALGORITHM_ID = (
    "production_polygon_v3_multishape_hard_min_recall_constrained_pair_vote"
)
PHASE2_POST_DP_PAIR_VOTE_ALGORITHM_ID = (
    "production_polygon_v3_multishape_hard_min_recall_post_dp_pair_vote"
)
PHASE2_NO_PAIR_VOTE_ALGORITHM_ID = (
    "production_polygon_v3_multishape_hard_min_recall_no_pair_vote"
)


__all__ = (
    "PHASE1_RAW_ALGORITHM_ID",
    "PHASE2_CONSTRAINED_PAIR_VOTE_ALGORITHM_ID",
    "PHASE2_NO_PAIR_VOTE_ALGORITHM_ID",
    "PHASE2_PER_KEY_PAIR_VOTE_ALGORITHM_ID",
    "PHASE2_POST_DP_PAIR_VOTE_ALGORITHM_ID",
)
