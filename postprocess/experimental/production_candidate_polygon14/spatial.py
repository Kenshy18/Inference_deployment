"""Track-level spatial polygon construction for polygon14_keyframe_v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..humanlike_vertex_placement_20260812.quality_repair import (
    persistent_line_fit_quality_guarded,
)

from .config import CANDIDATE, Polygon14CandidateConfig


@dataclass(frozen=True)
class SpatialBuildStats:
    frames: int
    components: int
    vertices_per_component: int
    repaired_component_frames: int
    fallback_component_frames: int
    tested_blends: int


def build_spatial_track(
    frame_components: Iterable[Sequence[np.ndarray]],
    config: Polygon14CandidateConfig = CANDIDATE,
) -> tuple[np.ndarray, SpatialBuildStats]:
    """Build fixed-ID 14-point polygons for every persistent component slot.

    Component slots must already be aligned by the upstream tracker.  Each
    slot is optimized independently; the temporal DP later evaluates their
    union against the unchanged per-frame tracked source mask.
    """
    frames = [
        [np.asarray(component, dtype=np.float64) for component in components]
        for components in frame_components
    ]
    if not frames:
        return (
            np.empty((0, 0, int(config.vertices_per_component), 2), dtype=np.float32),
            SpatialBuildStats(0, 0, int(config.vertices_per_component), 0, 0, 0),
        )
    component_count = len(frames[0])
    if component_count < 1:
        raise ValueError("polygon14 candidate requires at least one component slot")
    if any(len(components) != component_count for components in frames):
        raise ValueError("component-slot count changed inside one track segment")
    output = np.empty(
        (
            len(frames),
            component_count,
            int(config.vertices_per_component),
            2,
        ),
        dtype=np.float32,
    )
    repaired = 0
    fallbacks = 0
    tested = 0
    for slot in range(component_count):
        sequence, stats = persistent_line_fit_quality_guarded(
            [components[slot] for components in frames],
            int(config.vertices_per_component),
            recall_floor=float(config.spatial_recall_floor),
            iou_floor=float(config.spatial_iou_floor),
            dense_vertices=int(config.spatial_dense_vertices),
            coverage_quantile=float(config.spatial_coverage_quantile),
            maximum_intersection_radius=float(
                config.spatial_maximum_intersection_radius
            ),
            intersection_regularization=float(
                config.spatial_intersection_regularization
            ),
        )
        output[:, slot, :, :] = np.asarray(sequence, dtype=np.float32)
        repaired += int(stats.repaired_frames)
        fallbacks += int(stats.fallback_frames)
        tested += int(stats.tested_blends)
    return output, SpatialBuildStats(
        frames=len(frames),
        components=component_count,
        vertices_per_component=int(config.vertices_per_component),
        repaired_component_frames=repaired,
        fallback_component_frames=fallbacks,
        tested_blends=tested,
    )
