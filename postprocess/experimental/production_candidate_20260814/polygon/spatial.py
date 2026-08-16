"""Stable boundary around the approved track-consistent adaptive fitter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from dataclasses import replace

from experimental.production_candidate_polygon14.config import (
    CANDIDATE as APPROVED_SPATIAL_CONTRACT,
)
from experimental.production_candidate_polygon14.spatial import (
    SpatialBuildStats,
    build_spatial_track as _build_spatial_track,
)

from ..config import CANDIDATE, CandidateConfig


def _assert_same_contract(config: CandidateConfig) -> None:
    expected = APPROVED_SPATIAL_CONTRACT
    actual = config.spatial
    fields = {
        "recall_floor": (actual.recall_floor, expected.spatial_recall_floor),
        "recall_repair_max_scale": (
            actual.recall_repair_max_scale,
            expected.spatial_recall_repair_max_scale,
        ),
        "iou_floor": (actual.iou_floor, expected.spatial_iou_floor),
        "dense_vertices": (actual.dense_vertices, expected.spatial_dense_vertices),
        "coverage_quantile": (
            actual.coverage_quantile,
            expected.spatial_coverage_quantile,
        ),
        "maximum_intersection_radius": (
            actual.maximum_intersection_radius,
            expected.spatial_maximum_intersection_radius,
        ),
        "intersection_regularization": (
            actual.intersection_regularization,
            expected.spatial_intersection_regularization,
        ),
    }
    mismatch = {
        name: {"candidate": first, "approved": second}
        for name, (first, second) in fields.items()
        if first != second
    }
    if mismatch:
        raise RuntimeError(f"spatial candidate contract drift: {mismatch}")


def build_spatial_track(
    frame_components: Iterable[Sequence[np.ndarray]],
    config: CandidateConfig = CANDIDATE,
    *,
    vertices_per_component: int = 14,
) -> tuple[np.ndarray, SpatialBuildStats]:
    config.validate()
    _assert_same_contract(config)
    vertices = int(vertices_per_component)
    if vertices not in config.spatial.allowed_vertices_per_component:
        raise ValueError(
            f"vertices_per_component must be one of "
            f"{config.spatial.allowed_vertices_per_component}"
        )
    spatial_contract = replace(
        APPROVED_SPATIAL_CONTRACT,
        vertices_per_component=vertices,
    )
    return _build_spatial_track(frame_components, spatial_contract)


__all__ = ("SpatialBuildStats", "build_spatial_track")
