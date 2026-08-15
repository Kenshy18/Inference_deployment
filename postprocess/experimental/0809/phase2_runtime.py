#!/usr/bin/env python3
"""Phase-2 initial-shape search on the Phase-1 hard-Recall penalty DP.

The Production source and Phase-1 constraint implementation stay unchanged.
This isolated runtime adds a small, screened set of initial polygon states per
frame.  Pair-vote and post-decode repair remain disabled.  No video pixels are
opened; candidates are derived only from tracked SQLite polygon geometry.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from experimental.polygon_recall_optimizer.geometric_candidates import (
    _axis_scale,
    _principal_basis,
)
from experimental.polygon_recall_optimizer.temporal_candidates import (
    _rigid_align,
    _temporal_shapes,
)

from phase1_runtime import (
    _EPSILON,
    _load_production_runtime,
    _patch_embedded_optimizer,
)
from role_candidate_pool import ROLE_IDS, build_role_candidate


HERE = Path(__file__).resolve().parent
PROFILE_ENV = "MASK_PIPELINE_PHASE2_CANDIDATES"
NATIVE_BATCH_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH"
NATIVE_BATCH_THREADS_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS"
NATIVE_BATCH_EXACT_VERIFY_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH_EXACT_VERIFY"
NATIVE_DP_ENV = "MASK_PIPELINE_PHASE2_NATIVE_DP"
CUDA_SHAPE_ENV = "MASK_PIPELINE_PHASE2_CUDA_SHAPE"
CUDA_PREFILTER_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER"
CUDA_PREFILTER_BUDGET_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET"
CUDA_PREFILTER_SMALL_AREA_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_AREA"
)
CUDA_PREFILTER_SMALL_BUDGET_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_BUDGET"
)
CUDA_PREFILTER_VERIFY_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_VERIFY"
CUDA_EXACT_HINT_ENV = "MASK_PIPELINE_PHASE2_CUDA_EXACT_HINT"
CUDA_EXACT_HINT_COUNT_ENV = "MASK_PIPELINE_PHASE2_CUDA_EXACT_HINT_COUNT"
CUDA_LAZY_EXACT_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_EXACT"
CUDA_APPROX_ONLY_ENV = "MASK_PIPELINE_PHASE2_CUDA_APPROX_ONLY"
CUDA_LAZY_DEFICIT_PENALTY_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_DEFICIT_PENALTY"
CUDA_LAZY_MAX_SECONDS_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_MAX_SECONDS"
CUDA_LAZY_MIN_RETAINED_RATIO_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_MIN_RETAINED_RATIO"
CUDA_LAZY_FALLBACK_MIN_SECONDS_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_SECONDS"
)
CUDA_LAZY_FALLBACK_MIN_EDGES_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_EDGES"
)
CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO"
)
CUDA_LAZY_STATE_PAIR_BATCH_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_STATE_PAIR_BATCH"
)
CANDIDATE_FRAME_WORKERS_ENV = "MASK_PIPELINE_PHASE2_CANDIDATE_FRAME_WORKERS"
GC_INTERVAL_ENV = "MASK_PIPELINE_PHASE2_GC_INTERVAL"
OPENCV_THREADS_ENV = "MASK_PIPELINE_PHASE2_OPENCV_THREADS"
PAIR_VOTE_ENV = "MASK_PIPELINE_PHASE2_PAIR_VOTE"
PAIR_VOTE_CONSTRAINED_ENV = "MASK_PIPELINE_PHASE2_PAIR_VOTE_CONSTRAINED"
PAIR_VOTE_PER_KEY_ENV = "MASK_PIPELINE_PHASE2_PAIR_VOTE_PER_KEY"
PAIR_VOTE_SWEEPS_ENV = "MASK_PIPELINE_PHASE2_PAIR_VOTE_SWEEPS"
NEW_PRODUCTION_FAST_PAIR_VOTE_ENV = "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE"
PERSISTENT_LINE_FIT_BASE_ENV = "MASK_PIPELINE_PERSISTENT_LINE_FIT_BASE"
PERSISTENT_LINE_FIT_VERTICES_ENV = "MASK_PIPELINE_PERSISTENT_LINE_FIT_VERTICES"
SCALE_STATE_PROFILES = {
    "scale_quad_104_108_112_116": (1.04, 1.08, 1.12, 1.16),
    "scale_sextet_102_104_106_108_112_116": (
        1.02, 1.04, 1.06, 1.08, 1.12, 1.16,
    ),
    # Nested state-count benchmark profiles. Each higher profile contains all
    # factors from the preceding profile so scaling measurements do not change
    # the already available state family.
    "scale_states_08": (1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.16),
    "scale_states_09": (1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16),
    "scale_states_11": (
        1.01, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16,
    ),
    "scale_states_13": (
        1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08,
        1.10, 1.12, 1.14, 1.16,
    ),
    "scale_states_15": (
        1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08,
        1.09, 1.10, 1.11, 1.12, 1.14, 1.16,
    ),
}
PRODUCTION_CANDIDATE_BASELINE_V1 = (
    "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
)


ROLE_STATE_PROFILES = {
    # Two-state profiles are used for candidate-specific rescue/quality audits.
    **{f"role_{role.lower()}": (role,) for role in ROLE_IDS},
    # The fixed six roles proposed for the first joint experiment.
    "role_initial_six": ("A2", "D6", "B3", "C1", "E2", "F3"),
    "role_initial_six_border": ("A2", "D6", "B3", "C1", "E2", "G3"),
    # Two screening batches cover all ten priority candidates while retaining
    # the already profiled raw + six-state CUDA topology.
    "role_priority_batch1": ("A2", "A4", "D6", "B3", "C1", "C6"),
    "role_priority_batch2": ("E2", "F3", "G3", "Z1", "A2", "C1"),
    "orthogonal_initial_six": ("A07", "A06", "G02", "G04", "C02", "E02"),
    "orthogonal_c02_consensus": (
        "C02", "A2_P1", "A4_P1", "D6_P1", "B3_P1", "E2_P1",
    ),
    "orthogonal_c02_local": (
        "C02", "C1_P1", "C6_P1", "E2_P1", "F3_P1", "G3_P1",
    ),
    "orthogonal_c02_endpoints": (
        "C02", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "orthogonal_c02_115_endpoints": (
        "C02_115", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    # Frozen candidate-shape baseline.  Keep the historical profile alias so
    # old experiment commands and artifacts remain reproducible.
    "orthogonal_c02_125_endpoints": PRODUCTION_CANDIDATE_BASELINE_V1,
    "production_candidate_baseline_v1": PRODUCTION_CANDIDATE_BASELINE_V1,
    "search_c02_120": (
        "C02_120", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_c02_130": (
        "C02_130", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_a06_k3": (
        "C02_125", "G02", "G04", "A06_K3", "F3_P1", "D6_P1",
    ),
    "search_a06_k4": (
        "C02_125", "G02", "G04", "A06_K4", "F3_P1", "D6_P1",
    ),
    "search_g_h3": (
        "C02_125", "G02_H3", "G04_H3", "A06", "F3_P1", "D6_P1",
    ),
    "search_g_h8": (
        "C02_125", "G02_H8", "G04_H8", "A06", "F3_P1", "D6_P1",
    ),
    "search_f3_q65": (
        "C02_125", "G02", "G04", "A06", "F3_Q65_P1", "D6_P1",
    ),
    "search_f3_q75": (
        "C02_125", "G02", "G04", "A06", "F3_Q75_P1", "D6_P1",
    ),
    "search_d6_r5": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_R5_P1",
    ),
    "search_female_combo1": (
        "C02_125", "G02_H3", "G04_H3", "A06", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_male_combo1": (
        "C02_120", "G02", "G04", "A06_K4", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_tail_combo1": (
        "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_m_tail_d6base": (
        "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_Q75_P1", "D6_P1",
    ),
    "search_m_tail_a06base": (
        "C02_125", "G02_H3", "G04_H3", "A06", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_m_tail_gbase": (
        "C02_125", "G02", "G04", "A06_K3", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_m_tail_fbase": (
        "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1", "D6_R5_P1",
    ),
    "search_f_f75_g3": (
        "C02_125", "G02_H3", "G04_H3", "A06", "F3_Q75_P1", "D6_P1",
    ),
    "search_f_f75_d6r5": (
        "C02_125", "G02", "G04", "A06", "F3_Q75_P1", "D6_R5_P1",
    ),
    "search_j_dual130_no_d6": (
        "C02_125", "C02_130", "G02", "G04", "A06", "F3_P1",
    ),
    "search_j_dual130_no_f3": (
        "C02_125", "C02_130", "G02", "G04", "A06", "D6_P1",
    ),
    "search_j_dual135_no_d6": (
        "C02_125", "C02", "G02", "G04", "A06", "F3_P1",
    ),
    "search_j_triple_caps": (
        "C02_120", "C02_125", "C02_130", "G02", "G04", "A06",
    ),
    "search_j_cap_ladder": (
        "C02_115", "C02_120", "C02_125", "C02_130", "C02", "A06",
    ),
    "search_j_support8_k2_135_greplace": (
        "C02_125", "GF8_K2_135", "GB8_K2_135", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_k2_150_greplace": (
        "C02_125", "GF8_K2_150", "GB8_K2_150", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_k3_150_greplace": (
        "C02_125", "GF8_K3_150", "GB8_K3_150", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support12_k2_150_greplace": (
        "C02_125", "GF12_K2_150", "GB12_K2_150", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_k2_135_localreplace": (
        "C02_125", "G02", "G04", "A06", "GF8_K2_135", "GB8_K2_135",
    ),
    "search_j_support8_k2_150_localreplace": (
        "C02_125", "G02", "G04", "A06", "GF8_K2_150", "GB8_K2_150",
    ),
    "search_j_baseline_plus130": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1", "C02_130",
    ),
    "search_j_baseline_plus130_135": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
        "C02_130", "C02",
    ),
    "search_j_baseline_plus_support8": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
        "GF8_K2_135", "GB8_K2_135",
    ),
    "search_j_c02_150": (
        "C02_150", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_c02_175": (
        "C02_175", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_s200_greplace": (
        "C02_125", "GF8_K2_200", "GB8_K2_200", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_t200_greplace": (
        "C02_125", "GFT8_K2_200", "GBT8_K2_200", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_s200_localreplace": (
        "C02_125", "G02", "G04", "A06", "GF8_K2_200", "GB8_K2_200",
    ),
    "search_j_support8_t200_localreplace": (
        "C02_125", "G02", "G04", "A06", "GFT8_K2_200", "GBT8_K2_200",
    ),
    "search_j_support8_k1_200_greplace": (
        "C02_125", "GF8_K1_200", "GB8_K1_200", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_k1_300_greplace": (
        "C02_125", "GF8_K1_300", "GB8_K1_300", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_support8_k1_200_localreplace": (
        "C02_125", "G02", "G04", "A06", "GF8_K1_200", "GB8_K1_200",
    ),
    "search_j_baseline_plus_support1": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
        "GF8_K1_200", "GB8_K1_200",
    ),
    "search_j_ctr4_125_creplace": (
        "CTR4_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_ctr4_150_creplace": (
        "CTR4_150", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_ctr8_150_creplace": (
        "CTR8_150", "G02", "G04", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_c125_ctr4_no_d6": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "CTR4_150",
    ),
    "search_j_c125_ctr4_no_f3": (
        "C02_125", "G02", "G04", "A06", "CTR4_150", "D6_P1",
    ),
    "search_j_vfit8_greplace": (
        "C02_125", "VF8_P1", "VB8_P1", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_vfit12_greplace": (
        "C02_125", "VF12_P1", "VB12_P1", "A06", "F3_P1", "D6_P1",
    ),
    "search_j_vfit8_localreplace": (
        "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
    ),
    "search_j_vfit12_localreplace": (
        "C02_125", "G02", "G04", "A06", "VF12_P1", "VB12_P1",
    ),
    "search_j_vfit6_localreplace": (
        "C02_125", "G02", "G04", "A06", "VF6_P1", "VB6_P1",
    ),
    "search_j_vfit10_localreplace": (
        "C02_125", "G02", "G04", "A06", "VF10_P1", "VB10_P1",
    ),
    "search_j_vfit8_robust_localreplace": (
        "C02_125", "G02", "G04", "A06", "VFR8_P1", "VBR8_P1",
    ),
    "search_f_baseline_plus_f75": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1", "F3_Q75_P1",
    ),
    "search_j_baseline_plus_vfit8": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
        "VF8_P1", "VB8_P1",
    ),
    "search_j_baseline_plus_vf8": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1", "VF8_P1",
    ),
    "search_j_baseline_plus_vb8": (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1", "VB8_P1",
    ),
    "support30_three_windows_two_areas": (
        "S30_R2_A105", "S30_R2_A125",
        "S30_R5_A105", "S30_R5_A125",
        "S30_R10_A105", "S30_R10_A125",
    ),
}
CLASS_ROLE_STATE_PROFILES = {
    "production_candidate_superior_v2": {
        "女性器": (
            "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
            "F3_Q75_P1",
        ),
        "男性器": (
            "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1", "D6_R5_P1",
        ),
        "結合部分": (
            "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
            "VF8_P1",
        ),
    },
    # Target-aware successor to v2.  Very short target intervals do not need
    # the additional temporal states, while the joint-region forward-only
    # superset caused one extra CUDA/exact recall disagreement.  For intervals
    # 5+ the paired VF8 replacement gives the useful long-edge coverage without
    # increasing the known infeasible-stream count on the screening set.
    "production_candidate_superior_v3": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    # Selected research handoff.  It follows the quality-oriented forward VF8
    # superset around interval 5 and changes to the reach-oriented paired VF8
    # replacement around interval 8.  The other two classes keep their
    # independently screened palettes.
    "production_candidate_best_v4": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    # Frozen pre-temporal baseline.  Keep this profile stable even if later
    # candidate-search aliases are revised.
    "new_production_v1": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    # Explicit Production candidate.  Its temporal state palette is frozen to
    # new_production_v1; the semantic change is the preceding track-level
    # 14/16/18/20 spatial polygon selection.  Keeping a separate profile ID is
    # important for auditability and prevents an experimental representation
    # change from being mistaken for the established temporal baseline.
    "polygon14_keyframe_v1": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_ls110": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_ls115": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_inverse115": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_inverse120": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_vf_inverse115": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_v4_ivf115": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_v4_ivb115": {
        "女性器": (),
        "男性器": (),
        "結合部分": (),
    },
    "search_interval8_ivb_drop_g02": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "search_interval8_ivb_drop_g04": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "search_interval8_ivb_drop_vb": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    # Interval-8 bounded candidate selected after the 2026-08-11 ablation.
    # Female keeps v4 (already near the requested interval), male replaces the
    # weaker tail roles with bounded inverse endpoints, and joint adds only
    # the backward inverse endpoint whose contribution remained unique.
    "production_candidate_bounded_v5": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "production_candidate_bounded_v5_110": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "production_candidate_bounded_safe_v6": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "search_male_ivb_drop_f3": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "search_male_ivb_drop_d6": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "search_male_ivb_drop_g04": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
    "production_candidate_interval8_safe_v7": {
        "女性器": (), "男性器": (), "結合部分": (),
    },
}


def _class_role_state_profile(
    profile: str,
    label: str,
    target_interval: float,
) -> tuple[str, ...]:
    if profile not in {
        "production_candidate_superior_v3",
        "production_candidate_best_v4",
        "new_production_v1",
        "polygon14_keyframe_v1",
        "search_interval8_ls110",
        "search_interval8_ls115",
        "search_interval8_inverse115",
        "search_interval8_inverse120",
        "search_interval8_vf_inverse115",
        "search_interval8_v4_ivf115",
        "search_interval8_v4_ivb115",
        "search_interval8_ivb_drop_g02",
        "search_interval8_ivb_drop_g04",
        "search_interval8_ivb_drop_vb",
        "production_candidate_bounded_v5",
        "production_candidate_bounded_v5_110",
        "production_candidate_bounded_safe_v6",
        "search_male_ivb_drop_f3",
        "search_male_ivb_drop_d6",
        "search_male_ivb_drop_g04",
        "production_candidate_interval8_safe_v7",
    }:
        return tuple(CLASS_ROLE_STATE_PROFILES[profile][label])

    baseline = (
        "C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1",
    )
    if profile == "polygon14_keyframe_v1":
        return _class_role_state_profile(
            "new_production_v1", label, target_interval
        )
    if profile == "new_production_v1":
        if label == "女性器":
            return baseline if target_interval < 2.0 else baseline + ("F3_Q75_P1",)
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1",
                "D6_R5_P1",
            )
        if label == "結合部分":
            if target_interval < 4.0:
                return baseline
            if target_interval < 7.0:
                return baseline + ("VF8_P1",)
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
            )
        raise ValueError(f"unsupported Phase-2 class label: {label!r}")
    if profile in {"search_interval8_ls110", "search_interval8_ls115"}:
        suffix = "110" if profile.endswith("110") else "115"
        # Keep the strongest four baseline roles and dedicate the final two
        # states to bounded forward/backward interval-8 endpoint fits.
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                f"LSF8_{suffix}", f"LSB8_{suffix}",
            )
        return (
            "C02_125", "G02", "G04", "A06",
            f"LSF8_{suffix}", f"LSB8_{suffix}",
        )
    if profile in {"search_interval8_inverse115", "search_interval8_inverse120"}:
        suffix = "115" if profile.endswith("115") else "120"
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                f"IVF8_{suffix}", f"IVB8_{suffix}",
            )
        return (
            "C02_125", "G02", "G04", "A06",
            f"IVF8_{suffix}", f"IVB8_{suffix}",
        )
    if profile == "search_interval8_vf_inverse115":
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "F3_P1", "D6_R5_P1", "IVF8_115", "IVB8_115",
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
                "IVF8_115", "IVB8_115",
            )
        return baseline + ("F3_Q75_P1", "IVF8_115", "IVB8_115")
    if profile in {"search_interval8_v4_ivf115", "search_interval8_v4_ivb115"}:
        inverse = "IVF8_115" if profile.endswith("ivf115") else "IVB8_115"
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1",
                "D6_R5_P1", inverse,
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
                inverse,
            )
        return baseline + ("F3_Q75_P1", inverse)
    if profile in {
        "search_interval8_ivb_drop_g02",
        "search_interval8_ivb_drop_g04",
        "search_interval8_ivb_drop_vb",
    }:
        if label != "結合部分":
            return _class_role_state_profile(
                "production_candidate_best_v4", label, target_interval
            )
        roles = ["C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1"]
        removed = {
            "search_interval8_ivb_drop_g02": "G02",
            "search_interval8_ivb_drop_g04": "G04",
            "search_interval8_ivb_drop_vb": "VB8_P1",
        }[profile]
        roles.remove(removed)
        roles.append("IVB8_115")
        return tuple(roles)
    if profile == "production_candidate_bounded_v5":
        if label == "女性器":
            return baseline + ("F3_Q75_P1",)
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "IVF8_115", "IVB8_115",
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
                "IVB8_115",
            )
    if profile == "production_candidate_bounded_v5_110":
        if label == "女性器":
            return baseline + ("F3_Q75_P1",)
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "IVF8_110", "IVB8_110",
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
                "IVB8_110",
            )
    if profile == "production_candidate_bounded_safe_v6":
        if label == "女性器":
            return baseline + ("F3_Q75_P1",)
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "IVF8_115", "IVB8_115",
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
            )
    if profile in {"search_male_ivb_drop_f3", "search_male_ivb_drop_d6"}:
        if label != "男性器":
            return _class_role_state_profile(
                "production_candidate_best_v4", label, target_interval
            )
        if profile.endswith("drop_f3"):
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "D6_R5_P1", "IVB8_115",
            )
        return (
            "C02_125", "G02_H3", "G04_H3", "A06_K3",
            "F3_P1", "IVB8_115",
        )
    if profile == "search_male_ivb_drop_g04":
        if label != "男性器":
            return _class_role_state_profile(
                "production_candidate_best_v4", label, target_interval
            )
        return (
            "C02_125", "G02_H3", "A06_K3", "F3_P1", "D6_R5_P1",
            "IVB8_115",
        )
    if profile == "production_candidate_interval8_safe_v7":
        if label == "女性器":
            return baseline + ("F3_Q75_P1",)
        if label == "男性器":
            return (
                "C02_125", "G02_H3", "G04_H3", "A06_K3",
                "D6_R5_P1", "IVB8_115",
            )
        if label == "結合部分":
            return (
                "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
            )
    if label == "女性器":
        if target_interval < 2.0:
            return baseline
        return baseline + ("F3_Q75_P1",)
    if label == "男性器":
        return (
            "C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1",
            "D6_R5_P1",
        )
    if label == "結合部分":
        if target_interval < 4.0:
            return baseline
        if (
            profile == "production_candidate_best_v4"
            and target_interval < 7.0
        ):
            return baseline + ("VF8_P1",)
        return (
            "C02_125", "G02", "G04", "A06", "VF8_P1", "VB8_P1",
        )
    raise ValueError(f"unsupported Phase-2 class label: {label!r}")
MIXED_STATE_PROFILES = {
    # Small isotropic coverage controls are retained in these ablations so the
    # role candidates are asked to improve quality, not to mask CUDA/raster
    # endpoint deficits by themselves.
    "role_hybrid_six_102": ("S102", "F3", "E2", "B3", "C1", "D6"),
    "role_hybrid_six_102_104": ("S102", "S104", "F3", "E2", "B3", "C1"),
}
VALID_PROFILES = {
    "raw_baseline",
    "scale_best",
    "temporal_central_best",
    "temporal_recall_best",
    "axis_best",
    "broad_top2",
    "scale_104",
    "scale_108",
    "scale_112",
    "scale_116",
    "scale_pair_104_112",
    "scale_pair_108_112",
} | set(SCALE_STATE_PROFILES) | set(ROLE_STATE_PROFILES) | set(MIXED_STATE_PROFILES) | set(CLASS_ROLE_STATE_PROFILES)


def _build_dense_edge_array(
    predecessor_starts: list[int], state_count: int
) -> np.ndarray:
    """Construct the constant-state DP graph without Python tuple objects."""
    states = int(state_count)
    if states < 1:
        raise ValueError("state_count must be positive")
    state_pairs = states * states
    edge_count = sum(
        (end_pos - int(predecessor_starts[end_pos])) * state_pairs
        for end_pos in range(1, len(predecessor_starts))
    )
    edges = np.empty((edge_count, 4), dtype=np.int32)
    pair_start_states = np.repeat(np.arange(states, dtype=np.int32), states)
    pair_end_states = np.tile(np.arange(states, dtype=np.int32), states)
    offset = 0
    for end_pos in range(1, len(predecessor_starts)):
        first = int(predecessor_starts[end_pos])
        predecessor_count = int(end_pos - first)
        if predecessor_count <= 0:
            continue
        count = predecessor_count * state_pairs
        target = edges[offset : offset + count]
        target[:, 0] = np.repeat(
            np.arange(first, end_pos, dtype=np.int32), state_pairs
        )
        target[:, 1] = np.tile(pair_start_states, predecessor_count)
        target[:, 2] = int(end_pos)
        target[:, 3] = np.tile(pair_end_states, predecessor_count)
        offset += count
    if offset != edge_count:
        raise RuntimeError(f"dense edge construction mismatch: {offset} != {edge_count}")
    return edges


def _componentwise_scale(anchors: np.ndarray, factor: float) -> np.ndarray:
    output = np.asarray(anchors, dtype=np.float64).copy()
    for slot in range(len(output)):
        center = np.mean(output[slot], axis=0)
        output[slot] = center + float(factor) * (output[slot] - center)
    return output.astype(np.float32)


def _temporal_vectors(
    run,
    frame_index: int,
    *,
    radii: tuple[int, ...],
    quantiles: tuple[float, ...],
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    central_output: list[tuple[str, np.ndarray]] = []
    recall_output: list[tuple[str, np.ndarray]] = []
    reference_slots = np.asarray(run.anchors[int(frame_index)], dtype=np.float64)
    current_frame = int(run.frame_numbers[int(frame_index)])
    for radius in radii:
        neighbour_indices = [
            index
            for index, frame in enumerate(run.frame_numbers.tolist())
            if abs(int(frame) - current_frame) <= int(radius)
        ]
        if len(neighbour_indices) < 2:
            continue
        aligned_by_slot: list[np.ndarray] = []
        for slot, reference in enumerate(reference_slots):
            aligned_by_slot.append(
                np.stack(
                    [
                        _rigid_align(
                            np.asarray(reference, dtype=np.float64),
                            np.asarray(run.anchors[index][slot], dtype=np.float64),
                        )
                        for index in neighbour_indices
                    ],
                    axis=0,
                )
            )
        central_slots = []
        for slot, aligned in enumerate(aligned_by_slot):
            central, _coverage = _temporal_shapes(
                reference_slots[slot], aligned, recall_quantile=0.90
            )
            central_slots.append(central)
        central_output.append(
            (f"temporal_central_r{radius}", np.asarray(central_slots, dtype=np.float32))
        )
        for quantile in quantiles:
            coverage_slots = []
            for slot, aligned in enumerate(aligned_by_slot):
                _central, coverage = _temporal_shapes(
                    reference_slots[slot],
                    aligned,
                    recall_quantile=float(quantile),
                )
                coverage_slots.append(coverage)
            recall_output.append(
                (
                    f"temporal_recall_r{radius}_q{int(round(100 * quantile))}",
                    np.asarray(coverage_slots, dtype=np.float32),
                )
            )
    return central_output, recall_output


def _axis_vectors(run, frame_index: int) -> list[tuple[str, np.ndarray]]:
    anchors = np.asarray(run.anchors[int(frame_index)], dtype=np.float64)
    variants = {
        "axis_major": (1.18, 1.04),
        "axis_minor": (1.04, 1.18),
        "axis_balanced": (1.12, 1.12),
    }
    output: list[tuple[str, np.ndarray]] = []
    for name, (scale_x, scale_y) in variants.items():
        slots = []
        for points in anchors:
            center, basis = _principal_basis(points)
            slots.append(
                _axis_scale(points, center, basis, float(scale_x), float(scale_y))
            )
        output.append((name, np.asarray(slots, dtype=np.float32)))
    return output


def _patch_phase2_candidates(module: ModuleType, profile: str) -> ModuleType:
    if profile not in VALID_PROFILES:
        raise ValueError(f"unsupported Phase-2 candidate profile: {profile}")
    original_builder = module.build_frame_candidates
    role_generation_stats: dict[str, dict[str, int]] = {}
    module._phase2_role_generation_stats = role_generation_stats
    if profile in CLASS_ROLE_STATE_PROFILES:
        label = os.environ.get("MASK_PIPELINE_PHASE2_LABEL", "").strip()
        try:
            target_interval = float(
                os.environ.get("MASK_PIPELINE_PHASE2_TARGET_INTERVAL", "5")
            )
            active_role_ids = _class_role_state_profile(
                profile, label, target_interval
            )
        except KeyError as exc:
            raise ValueError(
                f"{profile} requires MASK_PIPELINE_PHASE2_LABEL in "
                f"{sorted(CLASS_ROLE_STATE_PROFILES[profile])}; got {label!r}"
            ) from exc
    else:
        active_role_ids = ROLE_STATE_PROFILES.get(profile)
    module._phase2_active_role_ids = active_role_ids
    pipeline_profile: dict[str, float | int] = {}
    module._phase2_pipeline_profile = pipeline_profile

    def add_profile_time(name: str, elapsed: float) -> None:
        pipeline_profile[name] = float(pipeline_profile.get(name, 0.0)) + float(elapsed)

    # Production's optimizer timer includes input preparation, streaming output,
    # exact QA, and artifact emission, while its per-run stage table does not.
    # Keep those phases visible so performance work targets measured costs.  The
    # wrappers are deliberately transparent and do not alter iteration order.
    original_iter_streams = module.iter_track_streams_from_sqlite

    def profiled_iter_track_streams_from_sqlite(*args, **kwargs):
        iterator = iter(original_iter_streams(*args, **kwargs))
        while True:
            started = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                add_profile_time("prepare_track_streams_seconds", time.perf_counter() - started)
                return
            add_profile_time("prepare_track_streams_seconds", time.perf_counter() - started)
            pipeline_profile["prepared_track_streams"] = int(
                pipeline_profile.get("prepared_track_streams", 0)
            ) + 1
            yield value

    module.iter_track_streams_from_sqlite = profiled_iter_track_streams_from_sqlite

    predictor_method = module.LearnedPointPredictor.predict_total_points_batch

    def profiled_predict_total_points_batch(self, *args, **kwargs):
        started = time.perf_counter()
        result = predictor_method(self, *args, **kwargs)
        add_profile_time("point_predictor_seconds", time.perf_counter() - started)
        pipeline_profile["point_predictor_calls"] = int(
            pipeline_profile.get("point_predictor_calls", 0)
        ) + 1
        return result

    module.LearnedPointPredictor.predict_total_points_batch = (
        profiled_predict_total_points_batch
    )

    store_class = module.SqliteUnionRowStore
    for method_name, profile_name in (
        ("add_rows", "union_store_add_seconds"),
        ("write_union_json", "write_union_json_seconds"),
        ("write_pred_sqlite", "write_pred_sqlite_seconds"),
        ("evaluate_exact", "evaluate_exact_seconds"),
    ):
        original_method = getattr(store_class, method_name)

        def make_profiled_method(method, timer_name):
            def profiled_method(self, *args, **kwargs):
                started = time.perf_counter()
                result = method(self, *args, **kwargs)
                add_profile_time(timer_name, time.perf_counter() - started)
                return result

            return profiled_method

        setattr(
            store_class,
            method_name,
            make_profiled_method(original_method, profile_name),
        )

    original_compact_json = module.write_compact_json_array

    def profiled_write_compact_json_array(*args, **kwargs):
        started = time.perf_counter()
        result = original_compact_json(*args, **kwargs)
        add_profile_time("write_compact_json_seconds", time.perf_counter() - started)
        return result

    module.write_compact_json_array = profiled_write_compact_json_array

    def fast_compute_mask_descriptors(mask: np.ndarray) -> dict[str, float | int]:
        """Production-equivalent descriptors without materializing all pixels.

        ``np.cov(nonzero(mask))`` computes the same covariance eigensystem as
        normalized binary image moments; the sample/population denominator is a
        common scalar and therefore cancels from the eccentricity ratio.  The
        previous path allocated every foreground coordinate for every frame.
        """
        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        area = float(binary.sum())
        h, w = binary.shape[:2]
        contours, hierarchy = module.cv2.findContours(
            binary, module.cv2.RETR_CCOMP, module.cv2.CHAIN_APPROX_NONE
        )
        if not contours or area <= 0.0:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "bbox_w": 0.0,
                "bbox_h": 0.0,
                "area_ratio": 0.0,
                "compactness": 0.0,
                "aspect_ratio": 1.0,
                "extent": 0.0,
                "solidity": 0.0,
                "components": 0,
                "holes": 0,
                "eccentricity": 0.0,
            }
        outer = max(contours, key=module.cv2.contourArea)
        perimeter = float(module.cv2.arcLength(outer, True))
        _x, _y, bw, bh = module.cv2.boundingRect(outer)
        bbox_area = float(max(bw * bh, 1))
        hull = module.cv2.convexHull(outer)
        hull_area = float(max(module.cv2.contourArea(hull), 1.0))
        compactness = float(
            (perimeter * perimeter) / max(4.0 * math.pi * area, 1e-6)
        )
        if area >= 2.0:
            moments = module.cv2.moments(binary, binaryImage=True)
            inv_area = 1.0 / max(float(moments["m00"]), 1e-12)
            covariance = np.asarray(
                [
                    [
                        float(moments["mu20"]) * inv_area,
                        float(moments["mu11"]) * inv_area,
                    ],
                    [
                        float(moments["mu11"]) * inv_area,
                        float(moments["mu02"]) * inv_area,
                    ],
                ],
                dtype=np.float64,
            )
            eigvals = np.sort(
                np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
            )[::-1]
            eccentricity = float(
                np.sqrt(max(0.0, 1.0 - float(eigvals[1] / eigvals[0])))
            )
        else:
            eccentricity = 0.0
        component_count = 0
        hole_count = 0
        if hierarchy is not None:
            parents = np.asarray(hierarchy[0], dtype=np.int32)[:, 3]
            component_count = int(np.count_nonzero(parents < 0))
            hole_count = int(len(parents) - component_count)
        return {
            "area": area,
            "perimeter": perimeter,
            "bbox_w": float(bw),
            "bbox_h": float(bh),
            "area_ratio": float(area / max(h * w, 1)),
            "compactness": compactness,
            "aspect_ratio": float(max(bw, 1) / max(bh, 1)),
            "extent": float(area / bbox_area),
            "solidity": float(area / hull_area),
            "components": int(component_count),
            "holes": int(hole_count),
            "eccentricity": eccentricity,
        }

    module.compute_mask_descriptors = fast_compute_mask_descriptors

    if os.environ.get("MASK_PIPELINE_PHASE2_DEEP_PROFILE", "").strip() == "1":
        for function_name in (
            "align_contour_slots",
            "align_polygon_phase",
            "resample_closed_contour",
            "build_local_mask_from_polygons",
            "compute_mask_descriptors",
            "build_track_segments_with_gapfill",
            "split_long_track_segments",
        ):
            original_function = getattr(module, function_name)

            def make_profiled_function(function, timer_name):
                def profiled_function(*args, **kwargs):
                    started = time.perf_counter()
                    result = function(*args, **kwargs)
                    add_profile_time(timer_name, time.perf_counter() - started)
                    pipeline_profile[f"{timer_name}_calls"] = int(
                        pipeline_profile.get(f"{timer_name}_calls", 0)
                    ) + 1
                    return result

                return profiled_function

            setattr(
                module,
                function_name,
                make_profiled_function(
                    original_function, f"deep_{function_name}_seconds"
                ),
            )

    def make_candidate(
        run,
        frame_index,
        label,
        anchors,
        runtime_args,
        endpoint_values=None,
    ):
        vector = module.flatten_contours(np.asarray(anchors, dtype=np.float32))
        if not np.all(np.isfinite(vector)):
            return None
        polygons = module.split_vector_to_polygons(
            vector, run.contour_count, run.anchors_per_contour
        )
        if any(len(polygon) < 3 for polygon in polygons):
            return None
        endpoint_evaluator = getattr(run, "_phase2_endpoint_evaluator", None)
        if endpoint_values is not None:
            metrics = {
                "recall": float(endpoint_values[4]),
                "iou": float(endpoint_values[6]),
            }
        elif endpoint_evaluator is None:
            metrics = module.compute_exact_metrics_from_polygons(
                run.gt_polygons[int(frame_index)], polygons
            )
        else:
            endpoint_values = endpoint_evaluator.exact_frame_metrics(
                int(frame_index),
                np.asarray(vector, dtype=np.float32),
                int(run.contour_count),
                int(run.anchors_per_contour),
            )
            # Candidate construction consumes only these two exact fields.
            # The native evaluator reuses its parity-aware GT raster cache;
            # all metric arithmetic remains identical to the scalar path.
            metrics = {
                "recall": float(endpoint_values[4]),
                "iou": float(endpoint_values[6]),
            }
        budget = float(module.recall_budget_from_metrics(metrics))
        # An added endpoint that already violates the hard floor cannot occur
        # in any feasible edge.  Do not spend quadratic DP work on it.
        if budget > _EPSILON:
            return None
        frame_loss = float(module.frame_accuracy_loss(metrics, runtime_args))
        area, center, radii, mean_radius = module.vector_proxy_stats(
            vector, run.contour_count, run.anchors_per_contour
        )
        return module.ShapeCandidate(
            label=str(label),
            vector=np.asarray(vector, dtype=np.float32),
            polygons=polygons,
            frame_loss=frame_loss,
            objective=frame_loss,
            recall_budget=budget,
            area=float(area),
            center=np.asarray(center, dtype=np.float32),
            radii=np.asarray(radii, dtype=np.float32),
            mean_radius=float(mean_radius),
        )

    def deduplicate(raw, candidates):
        output = []
        known = [np.asarray(raw.vector, dtype=np.float32)]
        scale = max(float(getattr(raw, "mean_radius", 0.0)), 1.0)
        for candidate in sorted(
            candidates,
            key=lambda value: (
                float(value.frame_loss),
                float(value.area),
                str(value.label),
            ),
        ):
            vector = np.asarray(candidate.vector, dtype=np.float32)
            distance = min(
                float(np.sqrt(np.mean(np.square(vector - previous)))) / scale
                for previous in known
            )
            if distance <= 1e-4:
                continue
            output.append(candidate)
            known.append(vector)
        return output

    def best_family(raw, values):
        valid = [value for value in values if value is not None]
        deduped = deduplicate(raw, valid)
        return deduped[:1]

    def raw_fallback(raw, label):
        """Keep the dense five-state topology when a scale is redundant/invalid."""
        return module.ShapeCandidate(
            label=str(label),
            vector=np.asarray(raw.vector, dtype=np.float32).copy(),
            polygons=[np.asarray(value, dtype=np.float32).copy() for value in raw.polygons],
            frame_loss=float(raw.frame_loss),
            objective=float(raw.objective),
            recall_budget=float(raw.recall_budget),
            area=float(raw.area),
            center=np.asarray(raw.center, dtype=np.float32).copy(),
            radii=np.asarray(raw.radii, dtype=np.float32).copy(),
            mean_radius=float(raw.mean_radius),
        )

    def build_frame_candidates(run, contexts, eval_contexts, runtime_args):
        if profile == "polygon14_keyframe_v1":
            from experimental.production_candidate_polygon14.integration import (
                apply_spatial_candidate,
            )

            if not bool(getattr(module, "_phase1_native_interval_enabled", False)):
                raise RuntimeError(
                    "adaptive Production vertices require native exact interval evaluation"
                )
            run._phase2_endpoint_evaluator = (
                module._phase1_get_native_interval_evaluator(
                    eval_contexts, run.gt_polygons
                )
            )
            apply_spatial_candidate(
                run,
                pipeline_profile,
                endpoint_evaluator=run._phase2_endpoint_evaluator,
            )
        if (
            profile != "polygon14_keyframe_v1"
            and
            os.environ.get(PERSISTENT_LINE_FIT_BASE_ENV, "").strip() == "1"
            and not bool(getattr(run, "_persistent_line_fit_base_applied", False))
        ):
            # Experimental replay only: change the per-frame polygon
            # representation while retaining run.gt_polygons as the exact
            # source-mask reference used by DP and pair-vote.  This lets us
            # rerun the already-frozen new_production optimizer without
            # silently changing its Recall denominator.
            if int(run.contour_count) != 1:
                raise RuntimeError(
                    "persistent-line-fit replay currently requires exactly "
                    f"one contour slot; stream={run.stream_id!r} "
                    f"contours={run.contour_count}"
                )
            target_vertices = int(
                os.environ.get(
                    PERSISTENT_LINE_FIT_VERTICES_ENV,
                    str(run.anchors_per_contour),
                )
            )
            if target_vertices != int(run.anchors_per_contour):
                raise RuntimeError(
                    "persistent-line-fit target must match the prepared "
                    "run anchor count: "
                    f"target={target_vertices} prepared={run.anchors_per_contour}"
                )
            from experimental.humanlike_vertex_placement_20260812.quality_repair import (
                persistent_line_fit_quality_guarded,
            )

            base_started = time.perf_counter()
            references = [
                np.asarray(frame_polygons[0], dtype=np.float64)
                for frame_polygons in run.gt_polygons
            ]
            sequence, repair_stats = persistent_line_fit_quality_guarded(
                references,
                target_vertices,
                dense_vertices=64,
                coverage_quantile=0.65,
                maximum_intersection_radius=0.2,
                intersection_regularization=0.01,
            )
            run.anchors = np.ascontiguousarray(
                sequence[:, None, :, :], dtype=np.float32
            )
            run.run_target_total_points = int(target_vertices)
            run._persistent_line_fit_base_applied = True
            pipeline_profile["persistent_line_fit_base_seconds"] = float(
                pipeline_profile.get("persistent_line_fit_base_seconds", 0.0)
            ) + float(time.perf_counter() - base_started)
            pipeline_profile["persistent_line_fit_base_frames"] = int(
                pipeline_profile.get("persistent_line_fit_base_frames", 0)
            ) + int(repair_stats.frames)
            pipeline_profile["persistent_line_fit_repaired_frames"] = int(
                pipeline_profile.get("persistent_line_fit_repaired_frames", 0)
            ) + int(repair_stats.repaired_frames)
            pipeline_profile["persistent_line_fit_fallback_frames"] = int(
                pipeline_profile.get("persistent_line_fit_fallback_frames", 0)
            ) + int(repair_stats.fallback_frames)
        raw_by_frame = original_builder(run, contexts, eval_contexts, runtime_args)
        if profile == "raw_baseline":
            return raw_by_frame

        # Endpoint feasibility used to cross the Python/C++ boundary once for
        # every frame/state pair.  Generate the role geometry in the original
        # deterministic order, then evaluate the independent endpoint masks in
        # one native OpenMP batch.  Candidate order, exact raster arithmetic,
        # and all downstream DP inputs remain unchanged.
        batched_role_anchors = None
        batched_role_metrics = None
        endpoint_evaluator = getattr(run, "_phase2_endpoint_evaluator", None)
        if (
            active_role_ids is not None
            and endpoint_evaluator is not None
            and hasattr(endpoint_evaluator, "exact_frame_metrics_batch")
        ):
            batch_started = time.perf_counter()
            batched_role_anchors = []
            valid_frames = []
            valid_vectors = []
            valid_positions = []
            expected_shape = tuple(np.asarray(run.anchors[0]).shape)

            def generate_role_frame(frame_index):
                frame_values = []
                generation_times = []
                for role_index, role_id in enumerate(active_role_ids):
                    generation_started = time.perf_counter()
                    generated = np.asarray(
                        build_role_candidate(run, frame_index, role_id),
                        dtype=np.float32,
                    )
                    generation_times.append(
                        float(time.perf_counter() - generation_started)
                    )
                    frame_values.append(generated)
                return frame_values, generation_times

            frame_workers = max(
                1, int(os.environ.get(CANDIDATE_FRAME_WORKERS_ENV, "1"))
            )
            if frame_workers == 1 or len(raw_by_frame) <= 1:
                generated_frames = map(
                    generate_role_frame, range(len(raw_by_frame))
                )
            else:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(frame_workers, len(raw_by_frame))
                )
                generated_frames = executor.map(
                    generate_role_frame, range(len(raw_by_frame))
                )
            try:
                for frame_index, (frame_values, generation_times) in enumerate(
                    generated_frames
                ):
                    for role_index, (role_id, generated) in enumerate(
                        zip(active_role_ids, frame_values)
                    ):
                        stats = role_generation_stats.setdefault(
                            role_id,
                            {
                                "generated": 0,
                                "endpoint_feasible": 0,
                                "active": 0,
                                "fallback": 0,
                                "generation_seconds": 0.0,
                            },
                        )
                        stats["generated"] += 1
                        stats["generation_seconds"] += generation_times[role_index]
                        if generated.shape == expected_shape and np.all(
                            np.isfinite(generated)
                        ):
                            valid_frames.append(int(frame_index))
                            valid_vectors.append(generated.reshape(-1, 2))
                            valid_positions.append(
                                (int(frame_index), int(role_index))
                            )
                    batched_role_anchors.append(frame_values)
            finally:
                if frame_workers > 1 and len(raw_by_frame) > 1:
                    executor.shutdown(wait=True)
            pipeline_profile["candidate_frame_workers"] = int(frame_workers)
            batched_role_metrics = [
                [None for _role_id in active_role_ids]
                for _frame_index in range(len(raw_by_frame))
            ]
            if valid_vectors:
                endpoint_started = time.perf_counter()
                native_values = endpoint_evaluator.exact_frame_metrics_batch(
                    np.ascontiguousarray(valid_frames, dtype=np.int32),
                    np.ascontiguousarray(np.stack(valid_vectors), dtype=np.float32),
                    int(run.contour_count),
                    int(run.anchors_per_contour),
                    max(
                        1,
                        int(
                            os.environ.get(
                                "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS", "1"
                            )
                        ),
                    ),
                )
                add_profile_time(
                    "candidate_endpoint_batch_seconds",
                    time.perf_counter() - endpoint_started,
                )
                pipeline_profile["candidate_endpoint_batch_cases"] = int(
                    pipeline_profile.get("candidate_endpoint_batch_cases", 0)
                ) + len(valid_positions)
                for position, values in zip(valid_positions, native_values):
                    frame_index, role_index = position
                    batched_role_metrics[frame_index][role_index] = values
            add_profile_time(
                "candidate_role_batch_seconds",
                time.perf_counter() - batch_started,
            )

        output = []
        for frame_index, raw_values in enumerate(raw_by_frame):
            raw = raw_values[0]
            needed = (
                {
                    "scale_best",
                    "temporal_central_best",
                    "temporal_recall_best",
                    "axis_best",
                }
                if profile == "broad_top2"
                else {profile}
            )
            families: dict[str, list[object]] = {}
            fixed_scales = {
                "scale_104": 1.04,
                "scale_108": 1.08,
                "scale_112": 1.12,
                "scale_116": 1.16,
            }
            if profile in fixed_scales:
                candidate = make_candidate(
                    run,
                    frame_index,
                    profile,
                    _componentwise_scale(
                        run.anchors[frame_index], fixed_scales[profile]
                    ),
                    runtime_args,
                )
                families[profile] = best_family(raw, [candidate])
            scale_pairs = {
                "scale_pair_104_112": (1.04, 1.12),
                "scale_pair_108_112": (1.08, 1.12),
                **SCALE_STATE_PROFILES,
            }
            if profile in scale_pairs:
                candidates = []
                for factor in scale_pairs[profile]:
                    label = f"scale_{factor:.2f}"
                    candidate = make_candidate(
                        run,
                        frame_index,
                        label,
                        _componentwise_scale(run.anchors[frame_index], factor),
                        runtime_args,
                    )
                    if profile in SCALE_STATE_PROFILES:
                        candidates.append(
                            candidate
                            if candidate is not None
                            else raw_fallback(raw, f"{label}_raw_fallback")
                        )
                    elif candidate is not None:
                        candidates.append(candidate)
                families[profile] = (
                    candidates
                    if profile in SCALE_STATE_PROFILES
                    else deduplicate(raw, candidates)
                )
            if active_role_ids is not None:
                candidates = []
                for role_index, role_id in enumerate(active_role_ids):
                    stats = role_generation_stats.setdefault(
                        role_id,
                        {
                            "generated": 0,
                            "endpoint_feasible": 0,
                            "active": 0,
                            "fallback": 0,
                            "generation_seconds": 0.0,
                        },
                    )
                    endpoint_values = None
                    if batched_role_anchors is None:
                        stats["generated"] += 1
                        generation_started = time.perf_counter()
                        generated_anchors = build_role_candidate(
                            run, frame_index, role_id
                        )
                        stats["generation_seconds"] += float(
                            time.perf_counter() - generation_started
                        )
                    else:
                        generated_anchors = batched_role_anchors[frame_index][role_index]
                        endpoint_values = batched_role_metrics[frame_index][role_index]
                    candidate = make_candidate(
                        run,
                        frame_index,
                        role_id,
                        generated_anchors,
                        runtime_args,
                        endpoint_values,
                    )
                    if candidate is not None:
                        stats["endpoint_feasible"] += 1
                        scale = max(float(getattr(raw, "mean_radius", 0.0)), 1.0)
                        distance = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        np.asarray(candidate.vector, dtype=np.float32)
                                        - np.asarray(raw.vector, dtype=np.float32)
                                    )
                                )
                            )
                            / scale
                        )
                        if distance > 1e-4:
                            stats["active"] += 1
                    else:
                        stats["fallback"] += 1
                    candidates.append(
                        candidate
                        if candidate is not None
                        else raw_fallback(raw, f"{role_id}_raw_fallback")
                    )
                families[profile] = candidates
            if profile in MIXED_STATE_PROFILES:
                candidates = []
                for candidate_id in MIXED_STATE_PROFILES[profile]:
                    if candidate_id.startswith("S"):
                        factor = float(candidate_id[1:]) / 100.0
                        candidate = make_candidate(
                            run,
                            frame_index,
                            candidate_id,
                            _componentwise_scale(run.anchors[frame_index], factor),
                            runtime_args,
                        )
                    else:
                        stats = role_generation_stats.setdefault(
                            candidate_id,
                            {"generated": 0, "endpoint_feasible": 0, "active": 0, "fallback": 0},
                        )
                        stats["generated"] += 1
                        candidate = make_candidate(
                            run,
                            frame_index,
                            candidate_id,
                            build_role_candidate(run, frame_index, candidate_id),
                            runtime_args,
                        )
                        if candidate is not None:
                            stats["endpoint_feasible"] += 1
                            stats["active"] += 1
                        else:
                            stats["fallback"] += 1
                    candidates.append(
                        candidate
                        if candidate is not None
                        else raw_fallback(raw, f"{candidate_id}_raw_fallback")
                    )
                families[profile] = candidates
            if "scale_best" in needed:
                scale_candidates = [
                    make_candidate(
                        run,
                        frame_index,
                        f"scale_{factor:.2f}",
                        _componentwise_scale(run.anchors[frame_index], factor),
                        runtime_args,
                    )
                    for factor in (1.02, 1.04, 1.06, 1.10, 1.14)
                ]
                families["scale_best"] = best_family(raw, scale_candidates)
            if needed & {"temporal_central_best", "temporal_recall_best"}:
                temporal_central, temporal_recall = _temporal_vectors(
                    run,
                    frame_index,
                    radii=(2, 5, 10),
                    quantiles=(0.90, 0.95, 0.97),
                )
                if "temporal_central_best" in needed:
                    central_candidates = [
                        make_candidate(
                            run, frame_index, label, anchors, runtime_args
                        )
                        for label, anchors in temporal_central
                    ]
                    families["temporal_central_best"] = best_family(
                        raw, central_candidates
                    )
                if "temporal_recall_best" in needed:
                    recall_candidates = [
                        make_candidate(
                            run, frame_index, label, anchors, runtime_args
                        )
                        for label, anchors in temporal_recall
                    ]
                    families["temporal_recall_best"] = best_family(
                        raw, recall_candidates
                    )
            if "axis_best" in needed:
                axis_candidates = [
                    make_candidate(run, frame_index, label, anchors, runtime_args)
                    for label, anchors in _axis_vectors(run, frame_index)
                ]
                families["axis_best"] = best_family(raw, axis_candidates)
            if profile == "broad_top2":
                champions = [value[0] for value in families.values() if value]
                additions = deduplicate(raw, champions)[:2]
            else:
                additions = families[profile]
            output.append([raw, *additions])
        return output

    def run_hard_multistate_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        runtime_args,
        eval_contexts=None,
    ):
        """Penalty DP on the exact hard-Recall feasible multistate graph.

        Production's generic multistate solver performs an outer binary search
        over a soft Recall multiplier.  Phase 2 has no soft Recall trade-off:
        ``phase1_runtime.interval_cost_from_vectors`` already maps every
        violating edge to +inf.  Removing that redundant multiplier search is
        both semantically exact and substantially faster.
        """

        if all(len(values) == 1 for values in candidates_by_frame):
            return module.run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                runtime_args,
                eval_contexts=eval_contexts,
            )
        frames = [int(value) for value in candidate_frames]
        if frames != list(range(len(run.frame_numbers))):
            raise RuntimeError("Phase 2 candidate pool must remain dense")
        node_count = len(frames)
        if node_count == 1:
            feasible = [
                (float(candidate.frame_loss), int(state))
                for state, candidate in enumerate(candidates_by_frame[frames[0]])
                if float(candidate.recall_budget) <= _EPSILON
            ]
            state = min(feasible)[1] if feasible else 0
            return [frames[0]], [state], {"interval_evals": 0, "interval_frames": 0}, {}, 0.0
        target_interval = max(
            1, int(round(1.0 / max(float(runtime_args.target_ratio), 1e-6)))
        )
        dynamic_max_gap = max(
            int(runtime_args.max_gap),
            int(
                math.ceil(
                    float(runtime_args.dynamic_max_gap_factor)
                    * float(target_interval)
                )
            ),
        )
        predecessor_starts = [0] * node_count
        for end in range(1, node_count):
            predecessor_starts[end] = int(
                bisect.bisect_left(
                    frames, frames[end] - dynamic_max_gap, 0, end
                )
            )
        edge_cache: dict[tuple[int, int, int, int], object] = {}
        counters = {"interval_evals": 0, "interval_frames": 0}
        native_batch_cache: dict[
            tuple[int, int, int, int], tuple[object, float]
        ] = {}
        native_batch_profile: dict[str, object] = {
            "enabled": False,
            "threads": 0,
            "precomputed_edges": 0,
            "precompute_seconds": 0.0,
            "used_exact_failures": 0,
        }
        native_batch_requested = os.environ.get(NATIVE_BATCH_ENV, "").strip() == "1"
        native_dp_requested = os.environ.get(NATIVE_DP_ENV, "").strip() == "1"
        native_metrics = None
        native_edge_array = None
        native_edge_costs = None
        native_decode_edge_array = None
        native_decode_edge_costs = None
        native_decode_indices = None
        native_initial_losses = None
        native_incremental_decoder = None
        lazy_exact_enabled = False
        lazy_exact_requested = False
        cuda_approx_only_requested = False
        lazy_exact_verified = None
        lazy_exact_edge_offsets = None
        lazy_exact_candidate_vectors = None
        lazy_exact_evaluator = None
        lazy_exact_threads = 1
        lazy_exact_parameters = None
        lazy_exact_started = None
        lazy_dense_costs_loaded = False
        lazy_state_pair_batch = (
            os.environ.get(CUDA_LAZY_STATE_PAIR_BATCH_ENV, "0").strip() == "1"
        )
        lazy_exact_max_seconds = max(
            0.0, float(os.environ.get(CUDA_LAZY_MAX_SECONDS_ENV, "0"))
        )
        lazy_fallback_min_seconds = max(
            0.0,
            float(os.environ.get(CUDA_LAZY_FALLBACK_MIN_SECONDS_ENV, "5.0")),
        )
        lazy_fallback_min_edges = max(
            1,
            int(os.environ.get(CUDA_LAZY_FALLBACK_MIN_EDGES_ENV, "1000")),
        )
        lazy_fallback_infeasible_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    os.environ.get(
                        CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO_ENV, "0.90"
                    )
                ),
            ),
        )
        native_batch_exact_verify = (
            os.environ.get(NATIVE_BATCH_EXACT_VERIFY_ENV, "").strip() == "1"
        )
        if native_batch_requested:
            if not bool(getattr(module, "_phase1_native_interval_enabled", False)):
                raise RuntimeError(
                    f"{NATIVE_BATCH_ENV}=1 requires MASK_PIPELINE_PHASE1_NATIVE_INTERVAL=1"
                )
            state_counts = [len(values) for values in candidates_by_frame]
            if len(set(state_counts)) != 1:
                raise RuntimeError(
                    "native batch currently requires a constant candidate-state count"
                )
            edge_build_started = time.perf_counter()
            edge_array = _build_dense_edge_array(
                predecessor_starts, state_counts[0]
            )
            edge_build_seconds = time.perf_counter() - edge_build_started
            candidate_stack_started = time.perf_counter()
            candidate_vectors = np.stack(
                [
                    np.stack(
                        [np.asarray(candidate.vector, dtype=np.float32)
                         for candidate in values],
                        axis=0,
                    )
                    for values in candidates_by_frame
                ],
                axis=0,
            )
            candidate_stack_seconds = time.perf_counter() - candidate_stack_started
            threads = max(
                1,
                int(
                    os.environ.get(
                        NATIVE_BATCH_THREADS_ENV,
                        str(min(8, os.cpu_count() or 1)),
                    )
                ),
            )
            evaluator = module._phase1_get_native_interval_evaluator(
                eval_contexts, run.gt_polygons
            )
            native_metrics = sys.modules.get("native_interval_metrics")
            if native_metrics is None:
                raise RuntimeError("native_interval_metrics disappeared after initialization")
            batch_started = time.perf_counter()
            cuda_prefilter_profile: dict[str, object] = {"enabled": False}
            batch_edge_array = edge_array
            retained_indices = None
            cuda_recall_deficit = None
            cuda_recall_hint_frames = None
            cuda_exact_hint_requested = (
                os.environ.get(CUDA_EXACT_HINT_ENV, "").strip() == "1"
            )
            # Frame hints are intentionally opt-in for the all-edge exact path.
            # Adding the same CUDA pass to the already-pruned lazy path preserved
            # every output but slowed the full KPI workload, so the established
            # lazy mode must not pay that transfer/launch overhead by default.
            cuda_return_frame_hints = cuda_exact_hint_requested
            cuda_prefilter_verify = (
                os.environ.get(CUDA_PREFILTER_VERIFY_ENV, "").strip() == "1"
            )
            if os.environ.get(CUDA_PREFILTER_ENV, "").strip() == "1":
                from cuda_interval_raster import evaluate_cached_intervals

                requested_prefilter_budget = max(
                    0.0,
                    float(os.environ.get(CUDA_PREFILTER_BUDGET_ENV, "0.10")),
                )
                small_area_threshold = max(
                    0.0,
                    float(os.environ.get(CUDA_PREFILTER_SMALL_AREA_ENV, "0")),
                )
                small_area_budget = max(
                    requested_prefilter_budget,
                    float(
                        os.environ.get(
                            CUDA_PREFILTER_SMALL_BUDGET_ENV,
                            str(requested_prefilter_budget),
                        )
                    ),
                )
                reference_areas = [
                    sum(
                        abs(
                            float(
                                module.cv2.contourArea(
                                    np.asarray(polygon, dtype=np.float32)
                                )
                            )
                        )
                        for polygon in frame_polygons
                        if len(polygon) >= 3
                    )
                    for frame_polygons in run.gt_polygons
                ]
                median_reference_area = float(
                    np.median(np.asarray(reference_areas, dtype=np.float64))
                ) if reference_areas else 0.0
                prefilter_budget = (
                    small_area_budget
                    if (
                        small_area_threshold > 0.0
                        and median_reference_area < small_area_threshold
                    )
                    else requested_prefilter_budget
                )
                cuda_result = evaluate_cached_intervals(
                    candidate_vectors,
                    edge_array,
                    eval_contexts,
                    iou_weight=float(runtime_args.interval_iou_weight),
                    recall_floor=float(runtime_args.recall_min),
                    return_frame_hints=bool(cuda_return_frame_hints),
                    recall_hint_count=max(
                        1,
                        min(
                            8,
                            int(os.environ.get(CUDA_EXACT_HINT_COUNT_ENV, "8")),
                        ),
                    ),
                )
                if cuda_return_frame_hints:
                    (
                        _cuda_loss,
                        cuda_recall_deficit,
                        cuda_recall_hint_frames,
                        _cuda_covered,
                        cuda_prefilter_details,
                    ) = cuda_result
                else:
                    (
                        _cuda_loss,
                        cuda_recall_deficit,
                        _cuda_covered,
                        cuda_prefilter_details,
                    ) = cuda_result
                screened_indices = np.flatnonzero(
                    np.asarray(cuda_recall_deficit) <= prefilter_budget
                )
                if not cuda_exact_hint_requested:
                    retained_indices = screened_indices
                if not cuda_prefilter_verify and not cuda_exact_hint_requested:
                    batch_edge_array = np.ascontiguousarray(
                        edge_array[retained_indices], dtype=np.int32
                    )
                cuda_prefilter_profile = {
                    **cuda_prefilter_details,
                    "enabled": True,
                    "deficit_budget": float(prefilter_budget),
                    "requested_deficit_budget": float(
                        requested_prefilter_budget
                    ),
                    "small_area_threshold": float(small_area_threshold),
                    "small_area_budget": float(small_area_budget),
                    "median_reference_area": float(median_reference_area),
                    "retained_edges": int(len(screened_indices)),
                    "rejected_edges": int(len(edge_array) - len(screened_indices)),
                    "retained_ratio": float(
                        len(screened_indices) / max(len(edge_array), 1)
                    ),
                    "verification_mode": bool(cuda_prefilter_verify),
                    "hint_only_mode": bool(cuda_exact_hint_requested),
                }
            lazy_exact_requested = (
                os.environ.get(CUDA_LAZY_EXACT_ENV, "").strip() == "1"
            )
            cuda_approx_only_requested = (
                os.environ.get(CUDA_APPROX_ONLY_ENV, "").strip() == "1"
            )
            if lazy_exact_requested and cuda_approx_only_requested:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 and {CUDA_APPROX_ONLY_ENV}=1 "
                    "are mutually exclusive"
                )
            if cuda_exact_hint_requested and (
                lazy_exact_requested or cuda_approx_only_requested
            ):
                raise RuntimeError(
                    f"{CUDA_EXACT_HINT_ENV}=1 is mutually exclusive with "
                    f"{CUDA_LAZY_EXACT_ENV}=1 and {CUDA_APPROX_ONLY_ENV}=1"
                )
            lazy_exact_enabled = bool(lazy_exact_requested)
            lazy_min_retained_ratio = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(CUDA_LAZY_MIN_RETAINED_RATIO_ENV, "0.60")
                    ),
                ),
            )
            retained_ratio = float(
                len(retained_indices) / max(len(edge_array), 1)
            ) if retained_indices is not None else 1.0
            lazy_auto_disabled_reason = None
            if lazy_exact_enabled and retained_ratio < lazy_min_retained_ratio:
                # A low retained ratio predicts that the approximate graph is
                # dominated by Recall failures. In that regime lazy DP tends
                # to churn through rejected paths, while the dense exact batch
                # is both faster and byte-stable.
                lazy_exact_enabled = False
                lazy_auto_disabled_reason = "low_cuda_retained_ratio"
            if lazy_exact_enabled and cuda_recall_deficit is None:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 requires {CUDA_PREFILTER_ENV}=1"
                )
            if cuda_approx_only_requested and cuda_recall_deficit is None:
                raise RuntimeError(
                    f"{CUDA_APPROX_ONLY_ENV}=1 requires {CUDA_PREFILTER_ENV}=1"
                )
            if lazy_exact_enabled and not native_dp_requested:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 requires {NATIVE_DP_ENV}=1"
                )
            if cuda_approx_only_requested and not native_dp_requested:
                raise RuntimeError(
                    f"{CUDA_APPROX_ONLY_ENV}=1 requires {NATIVE_DP_ENV}=1"
                )
            cuda_shape_profile: dict[str, object] = {"enabled": False}
            precomputed_shape_distances = None
            use_cuda_shape = (
                os.environ.get(CUDA_SHAPE_ENV, "").strip() == "1"
                and not (
                    cuda_exact_hint_requested
                    and not lazy_exact_requested
                    and not cuda_approx_only_requested
                )
            )
            if use_cuda_shape:
                from cuda_shape_distance import compute_shape_distances

                # Approximate/lazy CUDA constructs costs for the complete
                # graph. A prefilter may make batch_edge_array smaller, so
                # using it here would produce an incompatible shape vector.
                shape_edge_array = (
                    edge_array
                    if lazy_exact_requested or cuda_approx_only_requested
                    else batch_edge_array
                )
                precomputed_shape_distances, cuda_shape_details = (
                    compute_shape_distances(
                        candidate_vectors,
                        shape_edge_array,
                        float(run.scale),
                    )
                )
                cuda_shape_profile = {
                    "enabled": True,
                    **cuda_shape_details,
                }
            evaluation_parameters = (
                int(run.contour_count),
                int(run.anchors_per_contour),
                float(runtime_args.interval_iou_weight),
                float(runtime_args.recall_min),
                float(run.scale),
                float(runtime_args.shape_update_threshold_ratio),
                float(runtime_args.shape_switch_weight),
                float(runtime_args.shape_distance_weight),
                float(runtime_args.shape_penalty_adapt_gain),
                float(runtime_args.shape_distance_relief),
                float(runtime_args.shape_switch_relief),
                float(runtime_args.shape_distance_min_scale),
                float(runtime_args.shape_switch_min_scale),
            )
            if lazy_exact_enabled or cuda_approx_only_requested:
                # CUDA supplies dense approximate raster losses.  Only edges
                # that enter a candidate DP path are subsequently evaluated by
                # the exact OpenCV engine.  This keeps the hard Recall contract
                # on every accepted path without rasterizing the entire dense
                # graph on the CPU.
                if precomputed_shape_distances is None:
                    from cuda_shape_distance import compute_shape_distances

                    precomputed_shape_distances, cuda_shape_details = (
                        compute_shape_distances(
                            candidate_vectors,
                            edge_array,
                            float(run.scale),
                        )
                    )
                    cuda_shape_profile = {
                        "enabled": True,
                        "implicit_for_lazy_exact": True,
                        **cuda_shape_details,
                    }
                distance = np.asarray(precomputed_shape_distances, dtype=np.float64)
                cuda_frame_loss = np.asarray(_cuda_loss, dtype=np.float64)
                covered = np.asarray(_cuda_covered, dtype=np.float64)
                frame_loss_mean = cuda_frame_loss / np.maximum(covered, 1.0)
                base = 1.0 + max(float(runtime_args.shape_penalty_adapt_gain), 0.0) * np.maximum(
                    frame_loss_mean, 0.0
                )
                distance_scale = np.maximum(
                    float(runtime_args.shape_distance_min_scale),
                    1.0 / np.maximum(
                        np.power(base, max(float(runtime_args.shape_distance_relief), 0.0)),
                        1e-6,
                    ),
                )
                switch_scale = np.maximum(
                    float(runtime_args.shape_switch_min_scale),
                    1.0 / np.maximum(
                        np.power(base, max(float(runtime_args.shape_switch_relief), 0.0)),
                        1e-6,
                    ),
                )
                update = (
                    distance > float(runtime_args.shape_update_threshold_ratio)
                ).astype(np.float64)
                approximate_cost = (
                    cuda_frame_loss
                    + float(runtime_args.shape_switch_weight) * switch_scale * update
                    + float(runtime_args.shape_distance_weight) * distance_scale * distance
                )
                lazy_deficit_penalty = max(
                    0.0,
                    float(os.environ.get(CUDA_LAZY_DEFICIT_PENALTY_ENV, "0")),
                )
                approximate_cost += lazy_deficit_penalty * np.asarray(
                    cuda_recall_deficit, dtype=np.float64
                )
                batch_array = np.zeros((len(edge_array), 9), dtype=np.float64)
                batch_array[:, 0] = approximate_cost
                batch_array[:, 1] = distance
                batch_array[:, 2] = update
                batch_array[:, 3] = covered
                batch_array[:, 4] = frame_loss_mean
                batch_array[:, 5] = distance_scale
                batch_array[:, 6] = switch_scale
                cuda_rejected = np.asarray(cuda_recall_deficit) > float(
                    cuda_prefilter_profile["deficit_budget"]
                )
                batch_array[cuda_rejected, 0] = np.inf
                batch_array[cuda_rejected, 7] = 1.0
                batch_values = batch_array
                if lazy_exact_enabled:
                    lazy_exact_verified = np.asarray(cuda_rejected, dtype=bool).copy()
                    lazy_exact_edge_offsets = np.zeros((node_count,), dtype=np.int64)
                    running_edge_offset = 0
                    for end_pos in range(1, node_count):
                        lazy_exact_edge_offsets[end_pos] = running_edge_offset
                        running_edge_offset += (
                            end_pos - predecessor_starts[end_pos]
                        ) * state_counts[end_pos] * state_counts[end_pos]
                    lazy_exact_candidate_vectors = candidate_vectors
                    lazy_exact_evaluator = evaluator
                    lazy_exact_threads = int(threads)
                    lazy_exact_parameters = evaluation_parameters
            else:
                batch_values = evaluator.evaluate_edge_batch(
                    candidate_vectors,
                    batch_edge_array,
                    *evaluation_parameters,
                    int(threads),
                    precomputed_shape_distances,
                    True,
                    False,
                    bool(cuda_exact_hint_requested),
                    cuda_recall_hint_frames,
                )
            evaluated_batch_array = np.asarray(batch_values)
            if cuda_approx_only_requested:
                # Approximation-only benchmarking deliberately accepts the
                # CUDA graph as final.  The normal post-run exact audit remains
                # enabled so quality and hard-Recall drift are measured rather
                # than hidden.
                batch_array = evaluated_batch_array
            elif not lazy_exact_enabled:
                if retained_indices is None or cuda_prefilter_verify:
                    batch_array = evaluated_batch_array
                else:
                    batch_array = np.zeros((len(edge_array), 9), dtype=np.float64)
                    # Rejected edges are hard-infeasible. Only cost and the two
                    # Recall columns affect graph membership for these rows.
                    batch_array[:, 0] = np.inf
                    batch_array[:, 7] = 1.0
                    batch_array[retained_indices] = evaluated_batch_array
            if cuda_prefilter_verify and cuda_recall_deficit is not None:
                cuda_rejected = np.asarray(cuda_recall_deficit) > float(
                    cuda_prefilter_profile["deficit_budget"]
                )
                cpu_feasible = (
                    (batch_array[:, 7] <= _EPSILON)
                    & (batch_array[:, 8] <= _EPSILON)
                )
                false_rejected = np.flatnonzero(cuda_rejected & cpu_feasible)
                cuda_prefilter_profile["false_rejected_feasible_edges"] = int(
                    len(false_rejected)
                )
                if len(false_rejected):
                    false_deficits = np.asarray(cuda_recall_deficit)[false_rejected]
                    cuda_prefilter_profile["false_rejected_deficit_min"] = float(
                        np.min(false_deficits)
                    )
                    cuda_prefilter_profile["false_rejected_deficit_max"] = float(
                        np.max(false_deficits)
                    )
                    cuda_prefilter_profile["false_rejected_deficit_quantiles"] = [
                        float(value)
                        for value in np.quantile(false_deficits, [0.25, 0.5, 0.75])
                    ]
                cuda_prefilter_profile["false_rejected_examples"] = [
                    [int(value) for value in edge_array[index].tolist()]
                    for index in false_rejected[:12]
                ]
                cuda_loss_array = np.asarray(_cuda_loss, dtype=np.float64)
                cuda_deficit_array = np.asarray(
                    cuda_recall_deficit, dtype=np.float64
                )
                cpu_cached_loss = (
                    np.asarray(batch_array[:, 4], dtype=np.float64)
                    * np.asarray(batch_array[:, 3], dtype=np.float64)
                )
                cpu_cached_deficit = np.asarray(
                    batch_array[:, 7], dtype=np.float64
                )
                frames_covered_array = np.maximum(
                    np.asarray(batch_array[:, 3], dtype=np.float64), 1.0
                )

                def error_summary(values: np.ndarray) -> dict[str, object]:
                    finite = np.asarray(values, dtype=np.float64)
                    finite = finite[np.isfinite(finite)]
                    if not len(finite):
                        return {"count": 0}
                    return {
                        "count": int(len(finite)),
                        "mean": float(np.mean(finite)),
                        "q50": float(np.quantile(finite, 0.50)),
                        "q90": float(np.quantile(finite, 0.90)),
                        "q95": float(np.quantile(finite, 0.95)),
                        "q99": float(np.quantile(finite, 0.99)),
                        "q999": float(np.quantile(finite, 0.999)),
                        "max": float(np.max(finite)),
                    }

                loss_abs_error = np.abs(cuda_loss_array - cpu_cached_loss)
                deficit_abs_error = np.abs(
                    cuda_deficit_array - cpu_cached_deficit
                )
                cuda_prefilter_profile["numeric_error"] = {
                    "interval_iou_loss_absolute": error_summary(loss_abs_error),
                    "mean_frame_iou_loss_absolute": error_summary(
                        loss_abs_error / frames_covered_array
                    ),
                    "recall_deficit_sum_absolute": error_summary(
                        deficit_abs_error
                    ),
                    "mean_frame_recall_deficit_absolute": error_summary(
                        deficit_abs_error / frames_covered_array
                    ),
                }
                exact_feasible = (
                    (batch_array[:, 7] <= _EPSILON)
                    & (batch_array[:, 8] <= _EPSILON)
                )
                budget_audit = {}
                for audit_budget in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25):
                    cuda_keep = cuda_deficit_array <= float(audit_budget)
                    budget_audit[f"{audit_budget:.2f}"] = {
                        "retained_edges": int(np.count_nonzero(cuda_keep)),
                        "retained_ratio": float(np.mean(cuda_keep)),
                        "false_rejected_exact_feasible": int(
                            np.count_nonzero((~cuda_keep) & exact_feasible)
                        ),
                        "retained_exact_infeasible": int(
                            np.count_nonzero(cuda_keep & (~exact_feasible))
                        ),
                    }
                cuda_prefilter_profile["budget_audit"] = budget_audit
            if native_dp_requested:
                native_edge_array = edge_array
                native_edge_costs = np.asarray(batch_array[:, 0], dtype=np.float64).copy()
                native_edge_costs[
                    (batch_array[:, 7] > _EPSILON)
                    | (batch_array[:, 8] > _EPSILON)
                ] = np.inf
                native_initial_losses = np.asarray(
                    [
                        float(candidate.frame_loss)
                        if float(candidate.recall_budget) <= _EPSILON
                        else np.inf
                        for candidate in candidates_by_frame[frames[0]]
                    ],
                    dtype=np.float64,
                )
                if retained_indices is not None and not cuda_prefilter_verify:
                    native_decode_indices = np.asarray(
                        retained_indices, dtype=np.int64
                    ).copy()
                    native_decode_edge_array = np.ascontiguousarray(
                        native_edge_array[native_decode_indices], dtype=np.int32
                    )
                    native_decode_edge_costs = np.ascontiguousarray(
                        native_edge_costs[native_decode_indices], dtype=np.float64
                    )
                else:
                    native_decode_edge_array = native_edge_array
                    native_decode_edge_costs = native_edge_costs
                if hasattr(native_metrics, "IncrementalPenaltyPathDecoder"):
                    native_incremental_decoder = (
                        native_metrics.IncrementalPenaltyPathDecoder(
                            native_decode_edge_array,
                            native_initial_losses,
                            int(node_count),
                            int(len(candidates_by_frame[frames[0]])),
                        )
                    )
            else:
                for edge, values in zip(edge_array, batch_array):
                    key = tuple(int(value) for value in edge)
                    native_batch_cache[key] = (
                        module.IntervalCost(
                            cost=float(values[0]),
                            shape_distance=float(values[1]),
                            shape_update=float(values[2]),
                            frames_covered=int(values[3]),
                            frame_loss_mean=float(values[4]),
                            shape_distance_scale=float(values[5]),
                            shape_switch_scale=float(values[6]),
                            recall_budget=float(values[7]),
                        ),
                        float(values[8]),
                    )
            native_batch_profile = {
                "enabled": True,
                "threads": int(threads),
                "native_dp": bool(native_dp_requested),
                "incremental_native_dp": bool(
                    native_incremental_decoder is not None
                ),
                "cuda_lazy_exact": {
                    "requested": bool(lazy_exact_requested),
                    "enabled": bool(lazy_exact_enabled),
                    "auto_disabled_reason": lazy_auto_disabled_reason,
                    "minimum_retained_ratio": float(lazy_min_retained_ratio),
                    "max_seconds_before_dense_fallback": float(
                        lazy_exact_max_seconds
                    ),
                    "fallback_min_seconds": float(lazy_fallback_min_seconds),
                    "fallback_min_exact_edges": int(lazy_fallback_min_edges),
                    "fallback_infeasible_ratio": float(
                        lazy_fallback_infeasible_ratio
                    ),
                    "approximate_deficit_penalty": (
                        float(lazy_deficit_penalty) if lazy_exact_enabled else 0.0
                    ),
                    "exact_edges": 0,
                    "exact_batches": 0,
                    "decode_retries": 0,
                    "frame_hints_enabled": bool(
                        lazy_exact_enabled
                        and cuda_recall_hint_frames is not None
                    ),
                    "frame_hint_count": int(
                        cuda_recall_hint_frames.shape[1]
                        if (
                            cuda_recall_hint_frames is not None
                            and np.asarray(cuda_recall_hint_frames).ndim == 2
                        )
                        else (1 if cuda_recall_hint_frames is not None else 0)
                    ),
                },
                "cuda_approx_only": {
                    "requested": bool(cuda_approx_only_requested),
                    "enabled": bool(cuda_approx_only_requested),
                    "exact_edge_validation": False
                    if cuda_approx_only_requested
                    else None,
                },
                "cuda_exact_hint": {
                    "requested": bool(cuda_exact_hint_requested),
                    "enabled": bool(cuda_exact_hint_requested),
                    "filtered_edges": 0,
                    "hinted_edges": int(len(cuda_recall_hint_frames))
                    if cuda_recall_hint_frames is not None
                    else 0,
                },
                "cuda_shape": cuda_shape_profile,
                "cuda_prefilter": cuda_prefilter_profile,
                "precomputed_edges": int(len(edge_array)),
                "decode_edges": int(
                    len(native_decode_edge_array)
                    if native_decode_edge_array is not None
                    else len(edge_array)
                ),
                "decode_edge_ratio": float(
                    len(native_decode_edge_array) / max(len(edge_array), 1)
                    if native_decode_edge_array is not None
                    else 1.0
                ),
                "precompute_seconds": float(time.perf_counter() - batch_started),
                "edge_build_seconds": float(edge_build_seconds),
                "candidate_stack_seconds": float(candidate_stack_seconds),
                "context_statistics": dict(evaluator.context_statistics()),
                "cached_failures_precomputed": int(
                    np.count_nonzero(np.asarray(batch_values)[:, 7] > _EPSILON)
                ),
                "exact_failures_precomputed": int(
                    np.count_nonzero(np.asarray(batch_values)[:, 8] > _EPSILON)
                ),
                "used_exact_failures": 0,
                "exact_verify_edges": 0,
                "exact_verify_classification_mismatches": 0,
                "exact_verify_examples": [],
            }
            if lazy_exact_enabled:
                lazy_exact_started = time.perf_counter()

        def edge(start_pos: int, start_state: int, end_pos: int, end_state: int):
            key = (
                int(start_pos),
                int(start_state),
                int(end_pos),
                int(end_state),
            )
            value = edge_cache.get(key)
            if value is not None:
                return value
            start_frame = frames[start_pos]
            end_frame = frames[end_pos]
            left = candidates_by_frame[start_frame][start_state]
            right = candidates_by_frame[end_frame][end_state]
            precomputed_entry = native_batch_cache.get(key)
            precomputed = (
                precomputed_entry[0] if precomputed_entry is not None else None
            )
            precomputed_exact_deficit = (
                precomputed_entry[1] if precomputed_entry is not None else None
            )
            if (
                precomputed_exact_deficit is not None
                and float(precomputed_exact_deficit) > _EPSILON
            ):
                native_batch_profile["used_exact_failures"] = int(
                    native_batch_profile["used_exact_failures"]
                ) + 1
            if (
                native_batch_exact_verify
                and precomputed_entry is not None
                and float(precomputed.recall_budget) <= _EPSILON
            ):
                reference_deficit = 0.0
                for frame_idx in range(start_frame + 1, end_frame + 1):
                    if frame_idx == end_frame:
                        polygons = right.polygons
                    else:
                        alpha = float(
                            (frame_idx - start_frame)
                            / max(end_frame - start_frame, 1)
                        )
                        polygons = module.split_vector_to_polygons(
                            module.interpolate_vectors(
                                left.vector, right.vector, alpha
                            ),
                            run.contour_count,
                            run.anchors_per_contour,
                        )
                    metrics = module.compute_exact_metrics_from_polygons(
                        run.gt_polygons[frame_idx], polygons
                    )
                    reference_deficit += max(
                        float(runtime_args.recall_min) - float(metrics["recall"]),
                        0.0,
                    )
                    if reference_deficit > _EPSILON:
                        break
                native_batch_profile["exact_verify_edges"] = int(
                    native_batch_profile["exact_verify_edges"]
                ) + 1
                reference_fails = reference_deficit > _EPSILON
                batch_fails = float(precomputed_exact_deficit) > _EPSILON
                if reference_fails != batch_fails:
                    native_batch_profile[
                        "exact_verify_classification_mismatches"
                    ] = int(
                        native_batch_profile[
                            "exact_verify_classification_mismatches"
                        ]
                    ) + 1
                    examples = native_batch_profile["exact_verify_examples"]
                    if len(examples) < 12:
                        examples.append(
                            {
                                "key": list(key),
                                "reference_deficit": float(reference_deficit),
                                "batch_deficit": float(precomputed_exact_deficit),
                            }
                        )
            value = module.interval_cost_from_vectors(
                run,
                start_frame,
                left.vector,
                end_frame,
                right.vector,
                runtime_args,
                include_start=False,
                eval_contexts=eval_contexts,
                start_candidate=left,
                end_candidate=right,
                precomputed_interval_info=precomputed,
                precomputed_exact_deficit=precomputed_exact_deficit,
            )
            edge_cache[key] = value
            counters["interval_evals"] += 1
            counters["interval_frames"] += int(value.frames_covered)
            return value

        decoded: dict[
            tuple[int, float, tuple[int, ...]],
            tuple[list[int], list[int], float, float],
        ] = {}
        decode_profiles: list[dict[str, float | int]] = []

        def record_decode(started: float, cache_before: int) -> None:
            decode_profiles.append(
                {
                    "seconds": float(time.perf_counter() - started),
                    "cache_before": int(cache_before),
                    "cache_after": int(len(edge_cache)),
                }
            )

        def emit_solver_profile() -> None:
            first = decode_profiles[0] if decode_profiles else None
            cached = [
                row
                for row in decode_profiles
                if int(row["cache_before"]) == int(row["cache_after"])
            ]
            payload = {
                "stream_id": str(run.stream_id),
                "frames": int(node_count),
                "mean_state_count": float(
                    np.mean([len(values) for values in candidates_by_frame])
                ),
                "decode_calls": int(len(decode_profiles)),
                "decode_seconds": float(
                    sum(float(row["seconds"]) for row in decode_profiles)
                ),
                "first_decode_seconds": (
                    float(first["seconds"]) if first is not None else 0.0
                ),
                "cached_decode_calls": int(len(cached)),
                "cached_decode_seconds": float(
                    sum(float(row["seconds"]) for row in cached)
                ),
                "cached_decode_mean_seconds": (
                    float(sum(float(row["seconds"]) for row in cached) / len(cached))
                    if cached
                    else 0.0
                ),
                "edge_cache_entries": int(len(edge_cache)),
                "interval_evaluations": int(counters["interval_evals"]),
                "interval_evaluation_frames": int(counters["interval_frames"]),
                "interval_kernel": dict(
                    getattr(module, "_phase1_interval_profile", {})
                ),
                "native_batch": dict(native_batch_profile),
            }
            print(
                "[phase2-dp-profile] "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

        def lazy_edge_index(
            start_pos: int,
            start_state: int,
            end_pos: int,
            end_state: int,
        ) -> int:
            if lazy_exact_edge_offsets is None:
                raise RuntimeError("lazy exact edge offsets are unavailable")
            state_count = len(candidates_by_frame[frames[0]])
            start_offset = int(start_pos) - int(predecessor_starts[end_pos])
            if start_offset < 0:
                raise RuntimeError("selected edge is outside the dense predecessor graph")
            return int(
                lazy_exact_edge_offsets[end_pos]
                + start_offset * state_count * state_count
                + int(start_state) * state_count
                + int(end_state)
            )

        def fallback_lazy_to_dense_exact(reason: str) -> None:
            nonlocal lazy_dense_costs_loaded
            if not lazy_exact_enabled:
                return
            if (
                lazy_exact_candidate_vectors is None
                or lazy_exact_evaluator is None
                or lazy_exact_parameters is None
                or native_edge_array is None
                or native_edge_costs is None
                or native_decode_edge_costs is None
                or lazy_exact_verified is None
            ):
                raise RuntimeError("lazy exact dense fallback was not initialized")
            fallback_started = time.perf_counter()
            if retained_indices is None:
                dense_indices = np.arange(len(native_edge_array), dtype=np.int64)
            else:
                dense_indices = np.asarray(retained_indices, dtype=np.int64)
            # Some retained edges may already have been evaluated with the
            # pair-dependent exact Recall rasterizer before the dense-cost
            # fallback is triggered.  Preserve those decisions.  In
            # particular, an exact-infeasible edge must never be resurrected
            # by the cheaper cached-Recall pass below.
            previously_exact = np.asarray(
                lazy_exact_verified[dense_indices], dtype=bool
            ).copy()
            previous_exact_costs = np.asarray(
                native_edge_costs[dense_indices], dtype=np.float64
            ).copy()
            dense_edges = np.ascontiguousarray(
                native_edge_array[dense_indices], dtype=np.int32
            )
            dense_values = np.asarray(
                lazy_exact_evaluator.evaluate_edge_batch(
                    lazy_exact_candidate_vectors,
                    dense_edges,
                    *lazy_exact_parameters,
                    int(lazy_exact_threads),
                    None,
                    True,
                    True,
                )
            )
            dense_costs = np.asarray(dense_values[:, 0], dtype=np.float64)
            dense_costs[
                dense_values[:, 7] > _EPSILON
            ] = np.inf
            dense_costs[previously_exact] = previous_exact_costs[
                previously_exact
            ]
            native_edge_costs[:] = np.inf
            native_edge_costs[dense_indices] = dense_costs
            if native_decode_indices is not None:
                native_decode_edge_costs[:] = native_edge_costs[
                    native_decode_indices
                ]
            # The complete retained graph now has native objective costs.
            # Cached-Recall failures are final; cached-feasible edges still
            # require the pair-dependent exact Recall check if DP selects
            # them.  Repeated shortest-path validation therefore reaches the
            # same optimum as eager dense exact Recall without rasterizing
            # every feasible edge twice.
            cached_infeasible = np.asarray(dense_values[:, 7]) > _EPSILON
            lazy_exact_verified[dense_indices] = (
                cached_infeasible | previously_exact
            )
            lazy_dense_costs_loaded = True
            lazy_profile = native_batch_profile["cuda_lazy_exact"]
            lazy_profile["fallback_dense_exact"] = False
            lazy_profile["fallback_dense_native_costs"] = True
            lazy_profile["fallback_exact_recall_mode"] = "selected_paths"
            lazy_profile["fallback_reason"] = str(reason)
            lazy_profile["fallback_edges"] = int(len(dense_indices))
            lazy_profile["fallback_seconds"] = float(
                time.perf_counter() - fallback_started
            )
            lazy_profile["fallback_infeasible_edges"] = int(
                np.count_nonzero(~np.isfinite(dense_costs))
            )
            lazy_profile["fallback_cached_infeasible_edges"] = int(
                np.count_nonzero(cached_infeasible)
            )
            lazy_profile["fallback_preserved_exact_edges"] = int(
                np.count_nonzero(previously_exact)
            )
            lazy_profile["fallback_preserved_exact_infeasible_edges"] = int(
                np.count_nonzero(
                    previously_exact & ~np.isfinite(previous_exact_costs)
                )
            )
            lazy_profile["fallback_exact_pending_edges"] = int(
                np.count_nonzero(~(cached_infeasible | previously_exact))
            )

        def exactify_lazy_path(
            positions: list[int], states: list[int]
        ) -> tuple[int, int]:
            if not lazy_exact_enabled:
                return 0, int(node_count)
            if (
                lazy_exact_verified is None
                or lazy_exact_candidate_vectors is None
                or lazy_exact_evaluator is None
                or lazy_exact_parameters is None
                or native_edge_array is None
                or native_edge_costs is None
                or native_decode_edge_costs is None
            ):
                raise RuntimeError("lazy exact runtime was not initialized")
            selected_path_indices = np.asarray(
                [
                    lazy_edge_index(
                        positions[index - 1],
                        states[index - 1],
                        positions[index],
                        states[index],
                    )
                    for index in range(1, len(positions))
                ],
                dtype=np.int64,
            )
            if lazy_state_pair_batch and len(selected_path_indices):
                state_count = len(candidates_by_frame[frames[0]])
                state_pairs = state_count * state_count
                # Every state pair over a selected frame interval shares the
                # same reference-frame span.  Validate the complete small
                # block in one native batch so near-identical alternative
                # paths do not force another full-graph DP scan one edge at a
                # time.  This changes evaluation order only; costs, Recall,
                # candidates, and tie-breaking remain untouched.
                selected_indices = np.concatenate(
                    [
                        np.arange(
                            int(edge_index) - (
                                int(states[path_index - 1]) * state_count
                                + int(states[path_index])
                            ),
                            int(edge_index) - (
                                int(states[path_index - 1]) * state_count
                                + int(states[path_index])
                            ) + state_pairs,
                            dtype=np.int64,
                        )
                        for path_index, edge_index in enumerate(
                            selected_path_indices, start=1
                        )
                    ]
                )
            else:
                selected_indices = selected_path_indices
            if not len(selected_indices):
                return 0, int(node_count)
            selected_indices = np.unique(selected_indices)
            pending = selected_indices[~lazy_exact_verified[selected_indices]]
            if not len(pending):
                return 0, int(node_count)
            if lazy_exact_started is not None:
                lazy_elapsed = time.perf_counter() - lazy_exact_started
                lazy_profile = native_batch_profile["cuda_lazy_exact"]
                exact_edges_so_far = int(lazy_profile["exact_edges"])
                infeasible_ratio = float(
                    int(lazy_profile.get("exact_infeasible_edges", 0))
                    / max(exact_edges_so_far, 1)
                )
                if (
                    not lazy_dense_costs_loaded
                    and
                    lazy_elapsed >= lazy_fallback_min_seconds
                    and exact_edges_so_far >= lazy_fallback_min_edges
                    and infeasible_ratio >= lazy_fallback_infeasible_ratio
                ):
                    fallback_lazy_to_dense_exact("high_exact_infeasible_ratio")
                    return 1, 0
                if (
                    not lazy_dense_costs_loaded
                    and
                    lazy_exact_max_seconds > 0.0
                    and lazy_elapsed >= lazy_exact_max_seconds
                ):
                    fallback_lazy_to_dense_exact("time_budget")
                    return 1, 0
            exact_edges = np.ascontiguousarray(
                native_edge_array[pending], dtype=np.int32
            )
            exact_started = time.perf_counter()
            exact_values = np.asarray(
                lazy_exact_evaluator.evaluate_edge_batch(
                    lazy_exact_candidate_vectors,
                    exact_edges,
                    *lazy_exact_parameters,
                    int(lazy_exact_threads),
                    None,
                    True,
                    False,
                    True,
                    (
                        np.ascontiguousarray(
                            np.asarray(cuda_recall_hint_frames)[pending],
                            dtype=np.int32,
                        )
                        if cuda_recall_hint_frames is not None
                        else None
                    ),
                )
            )
            lazy_profile = native_batch_profile["cuda_lazy_exact"]
            lazy_profile["exact_evaluation_seconds"] = float(
                lazy_profile.get("exact_evaluation_seconds", 0.0)
            ) + float(time.perf_counter() - exact_started)
            exact_costs = np.asarray(exact_values[:, 0], dtype=np.float64)
            exact_costs[
                (exact_values[:, 7] > _EPSILON)
                | (exact_values[:, 8] > _EPSILON)
            ] = np.inf
            native_edge_costs[pending] = exact_costs
            if native_decode_indices is not None:
                compact_positions = np.searchsorted(
                    native_decode_indices, pending
                )
                if (
                    np.any(compact_positions >= len(native_decode_indices))
                    or np.any(
                        native_decode_indices[compact_positions] != pending
                    )
                ):
                    raise RuntimeError(
                        "selected lazy edge is outside the compact decode graph"
                    )
                native_decode_edge_costs[compact_positions] = exact_costs
            else:
                compact_positions = np.asarray(pending, dtype=np.int64)
            lazy_exact_verified[pending] = True
            lazy_profile["exact_edges"] = int(lazy_profile["exact_edges"]) + int(
                len(pending)
            )
            lazy_profile["exact_batches"] = int(lazy_profile["exact_batches"]) + 1
            lazy_profile["exact_infeasible_edges"] = int(
                lazy_profile.get("exact_infeasible_edges", 0)
            ) + int(np.count_nonzero(~np.isfinite(exact_costs)))
            return int(len(pending)), int(np.min(exact_edges[:, 2]))

        def decode(penalty: float):
            decode_started = time.perf_counter()
            cache_before = len(edge_cache)
            if native_dp_requested:
                if (
                    native_metrics is None
                    or native_edge_array is None
                    or native_edge_costs is None
                    or native_decode_edge_array is None
                    or native_decode_edge_costs is None
                    or native_initial_losses is None
                ):
                    raise RuntimeError(
                        f"{NATIVE_DP_ENV}=1 requires {NATIVE_BATCH_ENV}=1"
                    )
                recompute_from = 0
                while True:
                    if native_incremental_decoder is None:
                        positions, states, raw_cost = (
                            native_metrics.decode_penalty_path(
                                native_decode_edge_array,
                                native_decode_edge_costs,
                                native_initial_losses,
                                int(node_count),
                                int(len(candidates_by_frame[frames[0]])),
                                float(penalty),
                            )
                        )
                    else:
                        positions, states, raw_cost = (
                            native_incremental_decoder.decode(
                                native_decode_edge_costs,
                                float(penalty),
                                int(recompute_from),
                            )
                        )
                    positions = [int(value) for value in positions]
                    states = [int(value) for value in states]
                    if not positions or not lazy_exact_enabled:
                        break
                    exactified, recompute_from = exactify_lazy_path(
                        positions, states
                    )
                    if exactified <= 0:
                        break
                    lazy_profile = native_batch_profile["cuda_lazy_exact"]
                    lazy_profile["decode_retries"] = int(
                        lazy_profile["decode_retries"]
                    ) + 1
                if not positions:
                    fallback = (
                        list(frames),
                        [0] * len(frames),
                        float("inf"),
                        float(penalty),
                    )
                    decoded[(len(frames), float("inf"), tuple(fallback[1]))] = fallback
                    record_decode(decode_started, cache_before)
                    return fallback
                selected_frames = [frames[position] for position in positions]
                value = (
                    selected_frames,
                    states,
                    float(raw_cost),
                    float(penalty),
                )
                decoded[(len(selected_frames), round(value[2], 12), tuple(states))] = value
                record_decode(decode_started, cache_before)
                return value
            costs = [
                np.full(len(candidates_by_frame[frame]), np.inf, dtype=np.float64)
                for frame in frames
            ]
            raw_costs = [np.full_like(value, np.inf) for value in costs]
            counts = [
                np.full(len(value), 2**30, dtype=np.int32)
                for value in costs
            ]
            back_pos = [np.full(len(value), -1, dtype=np.int32) for value in costs]
            back_state = [
                np.full(len(value), -1, dtype=np.int16) for value in costs
            ]
            for state, candidate in enumerate(candidates_by_frame[frames[0]]):
                if float(candidate.recall_budget) <= _EPSILON:
                    raw = float(candidate.frame_loss)
                    costs[0][state] = raw + float(penalty)
                    raw_costs[0][state] = raw
                    counts[0][state] = 1
            for end_pos in range(1, node_count):
                end_state_count = len(candidates_by_frame[frames[end_pos]])
                for start_pos in range(predecessor_starts[end_pos], end_pos):
                    finite_start = np.flatnonzero(np.isfinite(costs[start_pos]))
                    if not len(finite_start):
                        continue
                    for start_state in finite_start.tolist():
                        for end_state in range(end_state_count):
                            info = edge(
                                start_pos, start_state, end_pos, end_state
                            )
                            if not math.isfinite(float(info.cost)):
                                continue
                            candidate_raw = float(raw_costs[start_pos][start_state]) + float(
                                info.cost
                            )
                            candidate_count = int(counts[start_pos][start_state]) + 1
                            candidate_cost = candidate_raw + float(penalty) * candidate_count
                            current_cost = float(costs[end_pos][end_state])
                            current_raw = float(raw_costs[end_pos][end_state])
                            current_count = int(counts[end_pos][end_state])
                            if (
                                candidate_cost < current_cost - 1e-12
                                or (
                                    abs(candidate_cost - current_cost) <= 1e-12
                                    and (
                                        candidate_raw < current_raw - 1e-12
                                        or (
                                            abs(candidate_raw - current_raw) <= 1e-12
                                            and candidate_count < current_count
                                        )
                                    )
                                )
                            ):
                                costs[end_pos][end_state] = candidate_cost
                                raw_costs[end_pos][end_state] = candidate_raw
                                counts[end_pos][end_state] = candidate_count
                                back_pos[end_pos][end_state] = start_pos
                                back_state[end_pos][end_state] = start_state
            final_states = np.flatnonzero(np.isfinite(costs[-1]))
            if not len(final_states):
                fallback = (
                    list(frames),
                    [0] * len(frames),
                    float("inf"),
                    float(penalty),
                )
                decoded[(len(frames), float("inf"), tuple(fallback[1]))] = fallback
                record_decode(decode_started, cache_before)
                return fallback
            final_state = min(
                final_states.tolist(),
                key=lambda state: (
                    float(costs[-1][state]),
                    float(raw_costs[-1][state]),
                    int(counts[-1][state]),
                    int(state),
                ),
            )
            positions = []
            states = []
            position = node_count - 1
            state = int(final_state)
            while position >= 0:
                positions.append(position)
                states.append(state)
                if position == 0:
                    break
                previous_position = int(back_pos[position][state])
                previous_state = int(back_state[position][state])
                if previous_position < 0 or previous_state < 0:
                    raise RuntimeError("broken Phase 2 DP predecessor chain")
                position, state = previous_position, previous_state
            positions.reverse()
            states.reverse()
            selected_frames = [frames[position] for position in positions]
            value = (
                selected_frames,
                states,
                float(raw_costs[-1][final_state]),
                float(penalty),
            )
            decoded[
                (len(selected_frames), round(value[2], 12), tuple(states))
            ] = value
            record_decode(decode_started, cache_before)
            return value

        low = 0.0
        low_value = decode(low)
        if not math.isfinite(float(low_value[2])):
            emit_solver_profile()
            public_cache = {
                (frames[start], start_state, frames[end], end_state, 0): value
                for (start, start_state, end, end_state), value in edge_cache.items()
            }
            return (
                low_value[0],
                low_value[1],
                counters,
                public_cache,
                float(low_value[3]),
            )
        high = 1.0
        high_value = decode(high)
        maximum = max(float(runtime_args.penalty_max), 1.0)
        while len(high_value[0]) > int(target_count) and high < maximum:
            high = min(high * 2.0, maximum)
            high_value = decode(high)
            if high >= maximum:
                break
        if len(low_value[0]) > int(target_count):
            for _step in range(max(1, int(runtime_args.penalty_binary_steps))):
                middle = 0.5 * (low + high)
                value = decode(middle)
                if len(value[0]) > int(target_count):
                    low = middle
                    low_value = value
                else:
                    high = middle
                    high_value = value
        selected = min(
            decoded.values(),
            key=lambda value: (
                abs(len(value[0]) - int(target_count)),
                value[2],
                len(value[0]),
                tuple(value[1]),
            ),
        )
        public_cache = {
            (frames[start], start_state, frames[end], end_state, 0): value
            for (start, start_state, end, end_state), value in edge_cache.items()
        }
        if native_dp_requested:
            counters["interval_evals"] = int(len(native_edge_array))
            counters["interval_frames"] = int(
                np.sum(native_edge_array[:, 2] - native_edge_array[:, 0])
            )
            public_cache = {}
        emit_solver_profile()
        return (
            selected[0],
            selected[1],
            counters,
            public_cache,
            float(selected[3]),
        )

    module.build_frame_candidates = build_frame_candidates
    if profile == "polygon14_keyframe_v1":
        from experimental.production_candidate_polygon14.topology_guard import (
            repair_decoded_path,
        )

        topology_stats: dict[str, float | int] = {
            "dp_selected_edges_checked": 0,
            "dp_invalid_edges": 0,
            "dp_inserted_keys": 0,
            "dp_guard_seconds": 0.0,
            "pair_vote_paths_checked": 0,
            "pair_vote_paths_rejected": 0,
            "pair_vote_local_trials_checked": 0,
            "pair_vote_local_trials_rejected": 0,
            "pair_vote_guard_seconds": 0.0,
        }
        module._polygon14_topology_guard_stats = topology_stats

        def topology_guarded_penalty_path(
            run,
            candidate_frames,
            candidates_by_frame,
            target_count,
            runtime_args,
            eval_contexts=None,
        ):
            result = run_hard_multistate_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                runtime_args,
                eval_contexts=eval_contexts,
            )
            frames, states = repair_decoded_path(
                module,
                run,
                result[0],
                result[1],
                candidates_by_frame,
                runtime_args,
                eval_contexts,
                topology_stats,
            )
            return frames, states, result[2], result[3], result[4]

        module.run_multistate_penalty_path = topology_guarded_penalty_path
    else:
        module.run_multistate_penalty_path = run_hard_multistate_penalty_path
    module._phase2_candidate_profile = profile
    module._phase2_candidate_patched = True
    return module


def _write_audit(
    output_dir: Path,
    recall_floor: float,
    patched_module: ModuleType | None,
    profile: str,
) -> dict[str, object]:
    candidate_contract = None
    if profile == "polygon14_keyframe_v1":
        from experimental.production_candidate_polygon14 import CANDIDATE

        candidate_contract = CANDIDATE.to_dict()
    metrics_path = output_dir / "exact/keyframe_exact_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    stream_rows = []
    with (output_dir / "opt/stream_segments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        stream_rows.extend(csv.DictReader(handle))
    infeasible = {
        (str(row["track_id"]), int(row["run_id"]))
        for row in stream_rows
        if (
            not math.isfinite(float(row["objective"]))
            or float(row["recall_budget_violation"]) > _EPSILON
        )
    }
    feasible_recalls = [
        float(row["recall"])
        for row in metric_rows
        if (str(row["track_id"]), int(row["run_id"])) not in infeasible
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    optimizer = summary["optimizer_summary"]
    selected_state_counts: dict[str, int] = {}
    selected_state_pair_counts: dict[str, int] = {}
    state_labels = {"0": "raw"}
    active_roles = (
        getattr(patched_module, "_phase2_active_role_ids", None)
        if patched_module is not None
        else None
    )
    if active_roles is not None:
        state_labels.update(
            {
                str(index): role_id
                for index, role_id in enumerate(
                    active_roles, start=1
                )
            }
        )
    elif profile in SCALE_STATE_PROFILES:
        state_labels.update(
            {
                str(index): f"scale_{factor:.3f}"
                for index, factor in enumerate(
                    SCALE_STATE_PROFILES[profile], start=1
                )
            }
        )
    final_keyframes_path = output_dir / "opt/final_keyframes.json"
    if final_keyframes_path.is_file():
        final_keys = json.loads(final_keyframes_path.read_text(encoding="utf-8"))
        for key in final_keys:
            state = str(int(key.get("candidate_id", 0)))
            selected_state_counts[state] = selected_state_counts.get(state, 0) + 1
        grouped_keys: dict[tuple[str, int], list[dict[str, object]]] = {}
        for key in final_keys:
            group = (str(key["track_id"]), int(key["run_id"]))
            grouped_keys.setdefault(group, []).append(key)
        for keys in grouped_keys.values():
            keys.sort(key=lambda value: int(value["frame"]))
            for left, right in zip(keys, keys[1:]):
                left_state = str(int(left.get("candidate_id", 0)))
                right_state = str(int(right.get("candidate_id", 0)))
                pair = (
                    f"{state_labels.get(left_state, left_state)}"
                    f"->{state_labels.get(right_state, right_state)}"
                )
                selected_state_pair_counts[pair] = (
                    selected_state_pair_counts.get(pair, 0) + 1
                )
    pair_vote_requested = os.environ.get(PAIR_VOTE_ENV, "0").strip() == "1"
    constrained_pair_vote = (
        os.environ.get(PAIR_VOTE_CONSTRAINED_ENV, "0").strip() == "1"
    )
    per_key_pair_vote = (
        os.environ.get(PAIR_VOTE_PER_KEY_ENV, "0").strip() == "1"
    )
    audit = {
        "schema_version": 1,
        "algorithm": (
            "production_v22_multishape_hard_min_recall_per_key_pair_vote"
            if per_key_pair_vote
            else (
                "production_v22_multishape_hard_min_recall_constrained_pair_vote"
                if constrained_pair_vote
                else (
                    "production_v22_multishape_hard_min_recall_post_dp_pair_vote"
                    if pair_vote_requested
                    else "production_v22_multishape_hard_min_recall_no_pair_vote"
                )
            )
        ),
        "candidate_profile": profile,
        "production_candidate_contract": candidate_contract,
        "recall_floor": float(recall_floor),
        "evaluated_rows": len(metric_rows),
        "minimum_recall": min(
            (float(row["recall"]) for row in metric_rows), default=1.0
        ),
        "mean_iou": sum(float(row["iou"]) for row in metric_rows)
        / max(len(metric_rows), 1),
        "infeasible_streams": len(infeasible),
        "feasible_exact_minimum_recall": min(feasible_recalls, default=1.0),
        "feasible_exact_violations": sum(
            value + 1e-12 < float(recall_floor) for value in feasible_recalls
        ),
        "mean_state_count": float(optimizer["mean_state_count"]),
        "candidate_state_labels": state_labels,
        "selected_candidate_ids": selected_state_counts,
        "selected_candidate_pairs": dict(
            sorted(
                selected_state_pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "role_generation": (
            dict(getattr(patched_module, "_phase2_role_generation_stats", {}))
            if patched_module is not None
            else {}
        ),
        "pair_vote_acceleration": (
            dict(getattr(patched_module, "_phase2_pair_vote_fast_stats", {}))
            if patched_module is not None
            else {}
        ),
        "topology_guard": (
            dict(
                getattr(
                    patched_module,
                    "_polygon14_topology_guard_stats",
                    {},
                )
            )
            if patched_module is not None
            else {}
        ),
        "pair_vote_disabled": not bool(optimizer["pair_vote_refine_enabled"]),
        "post_decode_shape_repair_disabled": bool(
            patched_module is not None
            and getattr(patched_module, "_phase1_exact_repair_disabled", False)
        ),
        "dense_candidate_pool": int(optimizer["candidate_frame_count_total"])
        == int(optimizer["row_count"]),
        "semantic_changes": {
            "candidate_shapes": profile,
            "spatial_polygon_representation": (
                "track-wise 14/16/18/20-point line-fit fallback with native "
                "exact Recall repair; tracked source masks remain the exact "
                "Recall reference"
                if profile == "polygon14_keyframe_v1"
                else "unchanged"
            ),
            "candidate_positions": "all prepared observations and gap-filled frames",
            "recall": "hard per-frame edge feasibility",
            "selection": "Production quality loss plus lambda per key",
            "pair_vote": (
                "per-key IoU-only coordinate optimization toward Production pair-vote; exact per-frame minimum Recall floor"
                if per_key_pair_vote
                else (
                    "IoU-only constrained blend of best-v4 and Production pair-vote; exact per-frame minimum Recall floor"
                    if constrained_pair_vote
                    else (
                        "Production post-DP least-squares endpoint vote enabled"
                        if pair_vote_requested
                        else "disabled"
                    )
                )
            ),
            "post_decode_repair": "disabled",
            "topology": (
                "lazy hard constraint on selected DP interpolation, "
                "pair-vote keyframes, and final dense interpolation"
                if profile == "polygon14_keyframe_v1"
                else "unchanged"
            ),
        },
    }
    pair_vote_mode_matches = bool(audit["pair_vote_disabled"]) != bool(
        pair_vote_requested
    )
    audit["implementation_contract_satisfied"] = (
        pair_vote_mode_matches
        and bool(audit["post_decode_shape_repair_disabled"])
        and bool(audit["dense_candidate_pool"])
        and (
            pair_vote_requested
            or int(audit["feasible_exact_violations"]) == 0
        )
    )
    if not bool(audit["implementation_contract_satisfied"]):
        raise RuntimeError(f"Phase 2 contract audit failed: {audit}")
    (output_dir / "phase2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    # Each class is normally scheduled as an independent process.  Leaving
    # OpenCV at its machine-wide default in every process oversubscribes the
    # 24-core host (3 x 24 threads).  This changes scheduling only; all raster
    # operations and emitted geometry remain identical.
    import cv2

    cv2.setNumThreads(
        max(1, int(os.environ.get(OPENCV_THREADS_ENV, str(os.cpu_count() or 1))))
    )
    profile = os.environ.get(PROFILE_ENV, "raw_baseline").strip()
    if profile not in VALID_PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {sorted(VALID_PROFILES)}")
    source = _load_production_runtime()
    original_builder = source._build_embedded_polygon_v22_module
    patched_holder: list[ModuleType] = []

    def build_patched_module() -> ModuleType:
        patched = _patch_embedded_optimizer(original_builder())
        patched._phase1_exact_repair_disabled = True
        patched = _patch_phase2_candidates(patched, profile)
        pair_vote_enabled = os.environ.get(PAIR_VOTE_ENV, "0").strip() == "1"
        constrained_pair_vote = (
            os.environ.get(PAIR_VOTE_CONSTRAINED_ENV, "0").strip() == "1"
        )
        per_key_pair_vote = (
            os.environ.get(PAIR_VOTE_PER_KEY_ENV, "0").strip() == "1"
        )
        if constrained_pair_vote and not pair_vote_enabled:
            raise RuntimeError("constrained pair-vote requires pair-vote enabled")
        if per_key_pair_vote and not constrained_pair_vote:
            raise RuntimeError("per-key pair-vote requires constrained pair-vote")
        production_pair_vote = patched.pair_vote_refine_keyframe_vectors
        pair_vote_fast_stats: dict[str, float | int | bool | str] = {
            "requested": bool(
                os.environ.get(NEW_PRODUCTION_FAST_PAIR_VOTE_ENV, "0").strip()
                == "1"
            ),
            "enabled": False,
            "mode": "reference",
        }
        patched._phase2_pair_vote_fast_stats = pair_vote_fast_stats

        if constrained_pair_vote:
            topology_guard_enabled = profile == "polygon14_keyframe_v1"
            if topology_guard_enabled:
                from experimental.production_candidate_polygon14.topology_guard import (
                    local_key_update_is_simple,
                    path_is_simple,
                )

                topology_guard_stats = patched._polygon14_topology_guard_stats

            def constrained_pair_vote_refine(
                run,
                chosen_frames,
                keyframe_vectors,
                args,
            ):
                baseline = patched.np.asarray(
                    keyframe_vectors, dtype=patched.np.float32
                )
                if len(chosen_frames) <= 1:
                    return baseline
                voted = patched.np.asarray(
                    production_pair_vote(
                        run, chosen_frames, baseline, args
                    ),
                    dtype=patched.np.float32,
                )
                delta = voted - baseline
                if bool(patched.np.allclose(delta, 0.0, atol=1e-7)):
                    return baseline

                exact_evaluator = None
                if bool(pair_vote_fast_stats["requested"]):
                    if profile not in {
                        "new_production_v1",
                        "polygon14_keyframe_v1",
                    }:
                        raise RuntimeError(
                            "fast pair-vote is restricted to the frozen "
                            "new-production temporal profiles"
                        )
                    from experimental.new_production.fast_pair_vote import (
                        ExactPairVoteEvaluator,
                    )

                    exact_evaluator = ExactPairVoteEvaluator(
                        patched,
                        run,
                        [int(value) for value in chosen_frames],
                        baseline,
                        voted,
                        pair_vote_fast_stats,
                    )
                    pair_vote_fast_stats["enabled"] = True

                def full_metrics(vectors):
                    if exact_evaluator is not None:
                        return exact_evaluator.full_metrics(vectors)
                    return patched.exact_interpolated_metrics(
                        run, chosen_frames, vectors
                    )

                recall_floor = float(args.recall_min)
                evaluations: dict[float, tuple[float, float, patched.np.ndarray]] = {}

                def evaluate(alpha: float):
                    alpha = float(min(max(alpha, 0.0), 1.0))
                    cache_key = round(alpha, 12)
                    cached = evaluations.get(cache_key)
                    if cached is not None:
                        return cached
                    trial = (
                        baseline.astype(patched.np.float64)
                        + alpha * delta.astype(patched.np.float64)
                    ).astype(patched.np.float32)
                    rows, _loss, mean_iou, _mean_recall, _precision, _global = (
                        full_metrics(trial)
                    )
                    minimum_recall = min(
                        (float(row["recall"]) for row in rows),
                        default=1.0,
                    )
                    value = (float(mean_iou), minimum_recall, trial)
                    evaluations[cache_key] = value
                    return value

                def evaluate_many(alphas):
                    normalized = [
                        float(min(max(float(alpha), 0.0), 1.0))
                        for alpha in alphas
                    ]
                    missing = []
                    missing_trials = []
                    if exact_evaluator is not None:
                        for alpha in normalized:
                            cache_key = round(alpha, 12)
                            if cache_key in evaluations:
                                continue
                            trial = (
                                baseline.astype(patched.np.float64)
                                + alpha * delta.astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            missing.append((cache_key, trial))
                            missing_trials.append(trial)
                        if missing_trials:
                            metrics = exact_evaluator.full_metrics_many(
                                missing_trials
                            )
                            for (cache_key, trial), (
                                mean_iou,
                                minimum_recall,
                            ) in zip(missing, metrics):
                                evaluations[cache_key] = (
                                    float(mean_iou),
                                    float(minimum_recall),
                                    trial,
                                )
                        return [evaluations[round(alpha, 12)] for alpha in normalized]
                    return [evaluate(alpha) for alpha in normalized]

                # Pure objective: maximize exact dense mean IoU.  Recall is a
                # hard per-frame constraint.  There is deliberately no motion,
                # shape, or temporal-smoothness penalty in this experiment.
                coarse = [index / 32.0 for index in range(33)]
                feasible: list[tuple[float, float, patched.np.ndarray]] = []
                for alpha, (mean_iou, minimum_recall, trial) in zip(
                    coarse, evaluate_many(coarse)
                ):
                    if minimum_recall + 1e-12 >= recall_floor:
                        feasible.append((mean_iou, alpha, trial))
                if not feasible:
                    if exact_evaluator is not None:
                        exact_evaluator.close()
                    return baseline
                _best_iou, best_alpha, _best_trial = max(
                    feasible, key=lambda item: (item[0], -item[1])
                )
                # Refine the best coarse neighborhood to 1/256 alpha.  The
                # raster objective is not assumed monotone or differentiable.
                refine_start = max(0.0, best_alpha - 1.0 / 32.0)
                refine_end = min(1.0, best_alpha + 1.0 / 32.0)
                refine_steps = int(round((refine_end - refine_start) * 256.0))
                refine_alphas = [
                    refine_start + index / 256.0
                    for index in range(refine_steps + 1)
                ]
                for alpha, (mean_iou, minimum_recall, trial) in zip(
                    refine_alphas, evaluate_many(refine_alphas)
                ):
                    if minimum_recall + 1e-12 >= recall_floor:
                        feasible.append((mean_iou, alpha, trial))

                def best_topology_valid(
                    values: list[tuple[float, float, patched.np.ndarray]],
                ):
                    if not topology_guard_enabled:
                        return max(values, key=lambda item: (item[0], -item[1]))
                    guard_started = time.perf_counter()
                    for item in sorted(
                        values,
                        key=lambda candidate: (candidate[0], -candidate[1]),
                        reverse=True,
                    ):
                        topology_guard_stats["pair_vote_paths_checked"] = int(
                            topology_guard_stats["pair_vote_paths_checked"]
                        ) + 1
                        if path_is_simple(
                            patched,
                            run,
                            [int(value) for value in chosen_frames],
                            item[2],
                        ):
                            topology_guard_stats["pair_vote_guard_seconds"] = float(
                                topology_guard_stats["pair_vote_guard_seconds"]
                            ) + (time.perf_counter() - guard_started)
                            return item
                        topology_guard_stats["pair_vote_paths_rejected"] = int(
                            topology_guard_stats["pair_vote_paths_rejected"]
                        ) + 1
                    topology_guard_stats["pair_vote_guard_seconds"] = float(
                        topology_guard_stats["pair_vote_guard_seconds"]
                    ) + (time.perf_counter() - guard_started)
                    return None

                selected = best_topology_valid(feasible)
                if selected is None:
                    if exact_evaluator is not None:
                        exact_evaluator.close()
                    return baseline
                if not per_key_pair_vote:
                    result = selected[2]
                else:
                    result = _per_key_refine(
                        run=run,
                        chosen_frames=chosen_frames,
                        baseline=baseline,
                        voted=voted,
                        initial=selected[2],
                        recall_floor=recall_floor,
                        exact_evaluator=exact_evaluator,
                    )
                if exact_evaluator is not None:
                    exact_evaluator.close()
                return result

            def _per_key_refine(
                *,
                run,
                chosen_frames,
                baseline,
                voted,
                initial,
                recall_floor: float,
                exact_evaluator,
            ):
                """Coordinate-ascent alpha per fixed key, with exact local gates."""
                chosen = [int(value) for value in chosen_frames]
                current = patched.np.asarray(initial, dtype=patched.np.float32).copy()
                delta = (
                    patched.np.asarray(voted, dtype=patched.np.float32)
                    - patched.np.asarray(baseline, dtype=patched.np.float32)
                )
                # Recover each initial alpha from its exact line projection.
                alphas = patched.np.zeros((len(chosen),), dtype=patched.np.float64)
                for key_pos in range(len(chosen)):
                    direction = delta[key_pos].astype(patched.np.float64).reshape(-1)
                    displacement = (
                        current[key_pos].astype(patched.np.float64)
                        - baseline[key_pos].astype(patched.np.float64)
                    ).reshape(-1)
                    denominator = float(direction @ direction)
                    if denominator > 1e-12:
                        alphas[key_pos] = float(
                            patched.np.clip(
                                float(displacement @ direction) / denominator,
                                0.0,
                                1.0,
                            )
                        )

                def local_metrics(key_pos: int, trial_vector):
                    if exact_evaluator is not None:
                        return exact_evaluator.local_metrics(
                            current, key_pos, trial_vector
                        )
                    left_key = max(0, key_pos - 1)
                    right_key = min(len(chosen) - 1, key_pos + 1)
                    start_frame = chosen[left_key]
                    end_frame = chosen[right_key]
                    iou_total = 0.0
                    minimum_recall = 1.0
                    for frame_idx in range(start_frame, end_frame + 1):
                        if frame_idx <= chosen[0]:
                            vector = trial_vector if key_pos == 0 else current[0]
                        elif frame_idx >= chosen[-1]:
                            vector = (
                                trial_vector
                                if key_pos == len(chosen) - 1
                                else current[-1]
                            )
                        else:
                            right_pos = int(
                                patched.np.searchsorted(
                                    patched.np.asarray(chosen, dtype=patched.np.int32),
                                    frame_idx,
                                    side="left",
                                )
                            )
                            left_pos = max(0, right_pos - 1)
                            if frame_idx == chosen[right_pos]:
                                vector = (
                                    trial_vector
                                    if right_pos == key_pos
                                    else current[right_pos]
                                )
                            else:
                                alpha_frame = float(
                                    (frame_idx - chosen[left_pos])
                                    / max(chosen[right_pos] - chosen[left_pos], 1)
                                )
                                left_vector = (
                                    trial_vector
                                    if left_pos == key_pos
                                    else current[left_pos]
                                )
                                right_vector = (
                                    trial_vector
                                    if right_pos == key_pos
                                    else current[right_pos]
                                )
                                vector = patched.interpolate_vectors(
                                    left_vector, right_vector, alpha_frame
                                )
                        polygons = patched.split_vector_to_polygons(
                            vector,
                            run.contour_count,
                            run.anchors_per_contour,
                        )
                        metrics = patched.compute_exact_metrics_from_polygons(
                            run.gt_polygons[frame_idx], polygons
                        )
                        iou_total += float(metrics["iou"])
                        minimum_recall = min(
                            minimum_recall, float(metrics["recall"])
                        )
                    return iou_total, minimum_recall

                def local_metrics_many(key_pos: int, trial_vectors):
                    if exact_evaluator is not None:
                        return exact_evaluator.local_metrics_many(
                            current, key_pos, trial_vectors
                        )
                    return [
                        local_metrics(key_pos, trial_vector)
                        for trial_vector in trial_vectors
                    ]

                # Alternate forward/backward coordinate sweeps.  Two sweeps
                # reproduce the original experiment; larger values measure
                # how close that result is to coordinate-wise saturation.
                sweep_count = max(
                    1,
                    int(os.environ.get(PAIR_VOTE_SWEEPS_ENV, "2") or "2"),
                )
                for sweep_index in range(sweep_count):
                    order = (
                        range(len(chosen))
                        if sweep_index % 2 == 0
                        else range(len(chosen) - 1, -1, -1)
                    )
                    for key_pos in order:
                        if bool(
                            patched.np.allclose(
                                delta[key_pos], 0.0, atol=1e-7
                            )
                        ):
                            continue
                        coarse = [index / 16.0 for index in range(17)]
                        coarse.append(float(alphas[key_pos]))
                        candidates: list[tuple[float, float, patched.np.ndarray]] = []
                        seen = set()
                        coarse_trials = []
                        for alpha in coarse:
                            alpha = float(min(max(alpha, 0.0), 1.0))
                            cache_key = round(alpha, 12)
                            if cache_key in seen:
                                continue
                            seen.add(cache_key)
                            trial = (
                                baseline[key_pos].astype(patched.np.float64)
                                + alpha
                                * delta[key_pos].astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            coarse_trials.append((alpha, trial))
                        coarse_metrics = local_metrics_many(
                            key_pos, [trial for _alpha, trial in coarse_trials]
                        )
                        for (alpha, trial), (iou_sum, minimum_recall) in zip(
                            coarse_trials, coarse_metrics
                        ):
                            if minimum_recall + 1e-12 >= recall_floor:
                                candidates.append((iou_sum, alpha, trial))
                        if not candidates:
                            continue
                        _coarse_iou, coarse_alpha, _coarse_trial = max(
                            candidates, key=lambda item: (item[0], -item[1])
                        )
                        refine_start = max(0.0, coarse_alpha - 1.0 / 16.0)
                        refine_end = min(1.0, coarse_alpha + 1.0 / 16.0)
                        refine_steps = int(
                            round((refine_end - refine_start) * 128.0)
                        )
                        refine_trials = []
                        for index in range(refine_steps + 1):
                            alpha = refine_start + index / 128.0
                            cache_key = round(alpha, 12)
                            if cache_key in seen:
                                continue
                            seen.add(cache_key)
                            trial = (
                                baseline[key_pos].astype(patched.np.float64)
                                + alpha
                                * delta[key_pos].astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            refine_trials.append((alpha, trial))
                        refine_metrics = local_metrics_many(
                            key_pos, [trial for _alpha, trial in refine_trials]
                        )
                        for (alpha, trial), (iou_sum, minimum_recall) in zip(
                            refine_trials, refine_metrics
                        ):
                            if minimum_recall + 1e-12 >= recall_floor:
                                candidates.append((iou_sum, alpha, trial))
                        ordered_candidates = sorted(
                            candidates,
                            key=lambda item: (item[0], -item[1]),
                            reverse=True,
                        )
                        selected_candidate = None
                        guard_started = time.perf_counter()
                        for candidate in ordered_candidates:
                            if topology_guard_enabled:
                                topology_guard_stats[
                                    "pair_vote_local_trials_checked"
                                ] = int(
                                    topology_guard_stats[
                                        "pair_vote_local_trials_checked"
                                    ]
                                ) + 1
                                if not local_key_update_is_simple(
                                    patched,
                                    run,
                                    chosen,
                                    current,
                                    key_pos,
                                    candidate[2],
                                ):
                                    topology_guard_stats[
                                        "pair_vote_local_trials_rejected"
                                    ] = int(
                                        topology_guard_stats[
                                            "pair_vote_local_trials_rejected"
                                        ]
                                    ) + 1
                                    continue
                            selected_candidate = candidate
                            break
                        if topology_guard_enabled:
                            topology_guard_stats["pair_vote_guard_seconds"] = float(
                                topology_guard_stats["pair_vote_guard_seconds"]
                            ) + (time.perf_counter() - guard_started)
                        if selected_candidate is None:
                            continue
                        best_iou, best_alpha, best_trial = selected_candidate
                        current_iou, current_recall = local_metrics(
                            key_pos, current[key_pos]
                        )
                        if (
                            current_recall + 1e-12 >= recall_floor
                            and current_iou > best_iou + 1e-12
                        ):
                            continue
                        current[key_pos] = best_trial
                        alphas[key_pos] = float(best_alpha)

                # Defensive whole-track validation.  Coordinate updates are
                # locally sufficient, but never emit an unverified path.
                if exact_evaluator is not None:
                    rows, _loss, _iou, _recall, _precision, _global = (
                        exact_evaluator.full_metrics(current)
                    )
                else:
                    rows, _loss, _iou, _recall, _precision, _global = (
                        patched.exact_interpolated_metrics(
                            run, chosen_frames, current
                        )
                    )
                if min((float(row["recall"]) for row in rows), default=1.0) + 1e-12 < recall_floor:
                    return initial
                if topology_guard_enabled and not path_is_simple(
                    patched,
                    run,
                    chosen,
                    current,
                ):
                    return initial
                return current

            patched.pair_vote_refine_keyframe_vectors = (
                constrained_pair_vote_refine
            )
        previous_defaults = patched.apply_fixed_practical_defaults

        def apply_pair_vote_defaults(args: argparse.Namespace) -> argparse.Namespace:
            args = previous_defaults(args)
            # Isolate pair-vote's contribution.  The DP and its candidate
            # states are unchanged, and the later mean-Recall expansion repair
            # remains disabled so it cannot hide or compensate vote effects.
            args.pair_vote_refine_enabled = bool(pair_vote_enabled)
            args.exact_recall_repair_enabled = False
            return args

        patched.apply_fixed_practical_defaults = apply_pair_vote_defaults
        patched._phase2_pair_vote_enabled = bool(pair_vote_enabled)
        patched._phase2_constrained_pair_vote_enabled = bool(
            constrained_pair_vote
        )
        patched._phase2_per_key_pair_vote_enabled = bool(per_key_pair_vote)
        patched_holder.append(patched)
        return patched

    source._build_embedded_polygon_v22_module = build_patched_module
    os.environ["ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA"] = "1"
    gc_interval = max(1, int(os.environ.get(GC_INTERVAL_ENV, "1") or "1"))
    real_gc_collect = gc.collect
    gc_profile: dict[str, float | int] = {
        "requested_calls": 0,
        "executed_calls": 0,
        "skipped_calls": 0,
        "seconds": 0.0,
        "interval": int(gc_interval),
    }

    def throttled_gc_collect(*args, **kwargs):
        gc_profile["requested_calls"] = int(gc_profile["requested_calls"]) + 1
        if int(gc_profile["requested_calls"]) % int(gc_interval) != 0:
            gc_profile["skipped_calls"] = int(gc_profile["skipped_calls"]) + 1
            return 0
        started = time.perf_counter()
        result = real_gc_collect(*args, **kwargs)
        gc_profile["seconds"] = float(gc_profile["seconds"]) + (
            time.perf_counter() - started
        )
        gc_profile["executed_calls"] = int(gc_profile["executed_calls"]) + 1
        return result

    if gc_interval > 1:
        gc.collect = throttled_gc_collect
    try:
        source.dispatch_main()
    finally:
        if gc_interval > 1:
            gc.collect = real_gc_collect
            started = time.perf_counter()
            real_gc_collect()
            gc_profile["seconds"] = float(gc_profile["seconds"]) + (
                time.perf_counter() - started
            )
            gc_profile["executed_calls"] = int(gc_profile["executed_calls"]) + 1
        print(json.dumps({"phase2_gc_profile": gc_profile}), flush=True)
    if patched_holder:
        print(
            json.dumps(
                {
                    "phase2_pipeline_profile": dict(
                        getattr(
                            patched_holder[-1], "_phase2_pipeline_profile", {}
                        )
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--recall-min", type=float, default=0.97)
        known, _unknown = parser.parse_known_args(sys.argv[2:])
        audit = _write_audit(
            known.output_dir,
            known.recall_min,
            patched_holder[-1] if patched_holder else None,
            profile,
        )
        print(json.dumps({"phase2_audit": audit}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
