"""Narrow adapter from the prepared polygon run to polygon14_keyframe_v1."""

from __future__ import annotations

import time

import numpy as np

from .config import CANDIDATE, Polygon14CandidateConfig
from .spatial import build_spatial_track


def apply_spatial_candidate(
    run,
    profile: dict[str, float | int],
    config: Polygon14CandidateConfig = CANDIDATE,
) -> None:
    """Replace candidate anchors only; never replace the Recall reference."""
    if bool(getattr(run, "_polygon14_candidate_applied", False)):
        return
    if int(run.anchors_per_contour) != int(config.vertices_per_component):
        raise RuntimeError(
            f"{config.profile_id} requires "
            f"anchors_per_contour={config.vertices_per_component}; "
            f"prepared={run.anchors_per_contour}"
        )
    # run.gt_polygons deliberately remains untouched. Both DP edge feasibility
    # and pair-vote use it as the exact source-mask reference.
    started = time.perf_counter()
    anchors, stats = build_spatial_track(run.gt_polygons, config)
    run.anchors = np.ascontiguousarray(anchors, dtype=np.float32)
    run.run_target_total_points = int(
        stats.components * stats.vertices_per_component
    )
    run._polygon14_candidate_applied = True
    profile["polygon14_spatial_seconds"] = float(
        profile.get("polygon14_spatial_seconds", 0.0)
    ) + float(time.perf_counter() - started)
    profile["polygon14_frames"] = int(profile.get("polygon14_frames", 0)) + int(
        stats.frames
    )
    profile["polygon14_components"] = int(
        profile.get("polygon14_components", 0)
    ) + int(stats.components)
    profile["polygon14_repaired_component_frames"] = int(
        profile.get("polygon14_repaired_component_frames", 0)
    ) + int(stats.repaired_component_frames)
    profile["polygon14_fallback_component_frames"] = int(
        profile.get("polygon14_fallback_component_frames", 0)
    ) + int(stats.fallback_component_frames)
