"""Frozen Phase-2 profiles, environment keys, and vertex-policy lookup."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_ENV = "MASK_PIPELINE_PHASE2_CANDIDATES"
SPATIAL_VERTEX_POLICY_ENV = "MASK_PIPELINE_SPATIAL_VERTEX_POLICY_JSON"
POLYGON_CONSTRAINED_PROFILES = {
    "polygon14_keyframe_v1",
    "polygon_adaptive_keyframe_v2",
}


@lru_cache(maxsize=4)
def _load_spatial_vertex_policy(path_value: str) -> dict[str, object]:
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("polygon_profile_id") != "polygon_adaptive_keyframe_v2":
        raise RuntimeError(f"unexpected spatial vertex policy profile: {path}")
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        raise RuntimeError(f"spatial vertex policy has no track map: {path}")
    return payload


def _spatial_vertices_for_track(track_id: str) -> int:
    path_value = os.environ.get(SPATIAL_VERTEX_POLICY_ENV, "").strip()
    if not path_value:
        raise RuntimeError(f"{SPATIAL_VERTEX_POLICY_ENV} is required")
    payload = _load_spatial_vertex_policy(path_value)
    tracks = payload["tracks"]
    entry = tracks.get(str(track_id))
    if not isinstance(entry, dict):
        raise RuntimeError(f"spatial vertex policy has no track {track_id!r}")
    vertices = int(entry.get("vertices_per_component", 0))
    if vertices not in (14, 16, 18, 20):
        raise RuntimeError(
            f"invalid spatial vertex count for track {track_id!r}: {vertices}"
        )
    return vertices


NATIVE_BATCH_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH"
NATIVE_BATCH_THREADS_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS"
NATIVE_BATCH_EXACT_VERIFY_ENV = "MASK_PIPELINE_PHASE2_NATIVE_BATCH_EXACT_VERIFY"
NATIVE_DP_ENV = "MASK_PIPELINE_PHASE2_NATIVE_DP"
CUDA_SHAPE_ENV = "MASK_PIPELINE_PHASE2_CUDA_SHAPE"
CUDA_PREFILTER_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER"
CUDA_PREFILTER_BUDGET_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET"
CUDA_PREFILTER_SMALL_AREA_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_AREA"
CUDA_PREFILTER_SMALL_BUDGET_ENV = "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_BUDGET"
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
CUDA_LAZY_FALLBACK_MIN_EDGES_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_EDGES"
CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO_ENV = (
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO"
)
CUDA_LAZY_STATE_PAIR_BATCH_ENV = "MASK_PIPELINE_PHASE2_CUDA_LAZY_STATE_PAIR_BATCH"
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
PRODUCTION_ROLE_PROFILES = frozenset(
    {"polygon14_keyframe_v1", "polygon_adaptive_keyframe_v2"}
)
CLASS_ROLE_STATE_PROFILES = {
    profile: {"女性器": (), "男性器": (), "結合部分": ()} for profile in PRODUCTION_ROLE_PROFILES
}
SCALE_STATE_PROFILES: dict[str, tuple[float, ...]] = {}
ROLE_STATE_PROFILES: dict[str, tuple[str, ...]] = {}
MIXED_STATE_PROFILES: dict[str, tuple[str, ...]] = {}


def _class_role_state_profile(
    profile: str,
    label: str,
    target_interval: float,
) -> tuple[str, ...]:
    """Return the only palettes shipped by the promoted runtime."""
    if profile not in PRODUCTION_ROLE_PROFILES:
        raise ValueError(f"unsupported Production profile: {profile!r}")
    from production.polygon.runtime.candidate_config import (
        CANDIDATE,
        with_target_interval,
    )
    from production.polygon.runtime.candidate_palette import role_ids

    interval = max(1, int(round(float(target_interval))))
    return role_ids(label, interval, with_target_interval(interval, CANDIDATE))


VALID_PROFILES = PRODUCTION_ROLE_PROFILES
