"""Stable boundary around the self-contained Production optimizer runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .preparation import prepare_classwise_source
from .runtime.engine import run_polygon_optimizer

from ..config import (
    RUNTIME_CANDIDATE_PROFILE_ID,
    RUNTIME_POLYGON_PROFILE_ID,
    ProductionConfig,
)


def build_runtime_config(config: ProductionConfig):
    """Translate the public contract to the parity-frozen internal payload."""
    from .runtime.candidate_config import (
        CANDIDATE,
        with_interval_evaluation,
        with_target_interval,
    )

    config.validate()
    runtime = with_target_interval(
        config.target_interval,
        CANDIDATE,
    )
    runtime = with_interval_evaluation(config.interval_evaluation, runtime)
    if runtime.profile_id != RUNTIME_CANDIDATE_PROFILE_ID:
        raise RuntimeError("polygon runtime profile contract drift")
    if runtime.polygon_profile_id != RUNTIME_POLYGON_PROFILE_ID:
        raise RuntimeError("polygon runtime optimizer profile drift")
    if (
        runtime.spatial.adaptive_vertex_policy != config.adaptive_vertex_policy
        or runtime.spatial.allowed_vertices_per_component
        != config.allowed_vertices_per_component
        or runtime.spatial.track_area_quantile != config.track_area_quantile
        or runtime.spatial.screen_occupancy_thresholds
        != config.screen_occupancy_thresholds
        or runtime.spatial.vertex_selection_source != config.vertex_selection_source
    ):
        raise RuntimeError("polygon runtime adaptive-vertex contract drift")
    if runtime.spatial.recall_floor != config.spatial_recall_floor:
        raise RuntimeError("polygon runtime spatial Recall contract drift")
    if (
        runtime.spatial.recall_repair_max_scale
        != config.spatial_recall_repair_max_scale
    ):
        raise RuntimeError("polygon runtime Recall repair scale contract drift")
    if runtime.spatial.iou_floor != config.spatial_iou_floor:
        raise RuntimeError("polygon runtime spatial IoU contract drift")
    if runtime.temporal.recall_floor != config.temporal_recall_floor:
        raise RuntimeError("polygon runtime temporal Recall contract drift")
    if runtime.temporal.pair_vote_sweeps != config.pair_vote_sweeps:
        raise RuntimeError("polygon runtime pair-vote contract drift")
    if (
        runtime.preparation.border_max_expand_px != config.border_max_expand_px
        or runtime.preparation.border_influence_px != config.border_influence_px
        or runtime.preparation.border_corner_support != config.border_corner_support
    ):
        raise RuntimeError("polygon runtime border contract drift")
    return runtime


def prepare_inputs(
    tracked_sqlite: Path,
    output_root: Path,
    *,
    width: int,
    height: int,
    input_video: Path | None,
    config: ProductionConfig,
) -> tuple[Path, dict[str, object]]:
    return prepare_classwise_source(
        tracked_sqlite,
        output_root,
        width=width,
        height=height,
        input_video=input_video,
        config=build_runtime_config(config),
    )


def optimize(
    source_root: Path,
    output_root: Path,
    *,
    labels: tuple[str, ...],
    max_tracks: int,
    force: bool,
    config: ProductionConfig,
    progress_callback: Callable[[str, float | None, float | None], None] | None = None,
) -> dict[str, object]:
    return run_polygon_optimizer(
        source_root,
        output_root,
        config=build_runtime_config(config),
        labels=labels,
        max_tracks=max_tracks,
        force=force,
        progress_callback=progress_callback,
    )


__all__ = ("build_runtime_config", "optimize", "prepare_inputs")
