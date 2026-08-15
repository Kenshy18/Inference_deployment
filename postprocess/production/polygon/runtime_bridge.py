"""Single quarantine boundary around the parity-frozen optimizer runtime."""

from __future__ import annotations

from pathlib import Path

from experimental.production_candidate_20260814.polygon.engine import (
    run_polygon_optimizer,
)
from experimental.production_candidate_20260814.polygon.preparation import (
    prepare_classwise_source,
)

from ..config import ProductionConfig


def build_runtime_config(config: ProductionConfig):
    """Translate the stable contract at the sole experimental boundary."""
    from experimental.production_candidate_20260814.config import (
        with_interval_evaluation,
        with_target_interval,
    )

    config.validate()
    runtime = with_target_interval(config.target_interval)
    runtime = with_interval_evaluation(config.interval_evaluation, runtime)
    if runtime.spatial.vertices_per_component != config.vertices_per_component:
        raise RuntimeError("polygon runtime vertex contract drift")
    if runtime.spatial.vertex_fallbacks != config.vertex_fallbacks:
        raise RuntimeError("polygon runtime vertex fallback contract drift")
    if runtime.spatial.recall_floor != config.spatial_recall_floor:
        raise RuntimeError("polygon runtime spatial Recall contract drift")
    if runtime.spatial.iou_floor != config.spatial_iou_floor:
        raise RuntimeError("polygon runtime spatial IoU contract drift")
    if runtime.temporal.recall_floor != config.temporal_recall_floor:
        raise RuntimeError("polygon runtime temporal Recall contract drift")
    if runtime.temporal.pair_vote_sweeps != config.pair_vote_sweeps:
        raise RuntimeError("polygon runtime pair-vote contract drift")
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
) -> dict[str, object]:
    return run_polygon_optimizer(
        source_root,
        output_root,
        config=build_runtime_config(config),
        labels=labels,
        max_tracks=max_tracks,
        force=force,
        require_exact_recall=config.require_zero_exact_recall_violations,
    )


__all__ = ("build_runtime_config", "optimize", "prepare_inputs")
