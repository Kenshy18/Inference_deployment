"""Immutable public contract for the deployed Production post-processor.

The parity-frozen internal runtime has its own detailed payload, but it lives
under :mod:`production` and is validated against this public contract before
every run.
"""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_ID = "production_polygon_adaptive_recall_cpu_exact_v3"
RUNTIME_CANDIDATE_PROFILE_ID = "production_candidate_adaptive_vertices_v2"
RUNTIME_POLYGON_PROFILE_ID = "polygon_adaptive_keyframe_v2"
LABELS = ("女性器", "男性器", "結合部分")
ALLOWED_VERTICES = (14, 16, 18, 20)
SCREEN_OCCUPANCY_THRESHOLDS = (0.03, 0.10, 0.25)


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    profile_id: str = PROFILE_ID
    labels: tuple[str, ...] = LABELS
    target_interval: int = 6
    adaptive_vertex_policy: bool = True
    allowed_vertices_per_component: tuple[int, ...] = ALLOWED_VERTICES
    track_area_quantile: float = 0.999
    screen_occupancy_thresholds: tuple[float, ...] = SCREEN_OCCUPANCY_THRESHOLDS
    vertex_selection_source: str = "tracked_pre_border"
    spatial_recall_floor: float = 0.97
    spatial_recall_repair_max_scale: float = 1.05
    spatial_iou_floor: float = 0.95
    temporal_recall_floor: float = 0.97
    interval_evaluation: str = "native_exact"
    pair_vote_sweeps: int = 2
    remove_short_tracks_max_frames: int = 10
    gapfill_max_gap: int = 15
    border_max_expand_px: float = 16.0
    border_influence_px: float = 16.0
    border_corner_support: bool = True

    @property
    def vertices_per_component(self) -> int:
        """Compatibility minimum; the track policy chooses up to 20 points."""
        return int(self.allowed_vertices_per_component[0])

    def validate(self) -> None:
        if self.labels != LABELS:
            raise ValueError(f"Production labels are frozen to {LABELS}")
        if self.target_interval < 1:
            raise ValueError("target_interval must be >= 1")
        if not self.adaptive_vertex_policy:
            raise ValueError("Production adaptive vertex policy must remain enabled")
        if self.allowed_vertices_per_component != ALLOWED_VERTICES:
            raise ValueError(
                f"Production vertex counts are frozen to {ALLOWED_VERTICES}"
            )
        if self.screen_occupancy_thresholds != SCREEN_OCCUPANCY_THRESHOLDS:
            raise ValueError(
                "Production screen-occupancy thresholds are frozen to "
                f"{SCREEN_OCCUPANCY_THRESHOLDS}"
            )
        if self.track_area_quantile != 0.999:
            raise ValueError("Production track-area quantile must be 0.999")
        if self.vertex_selection_source != "tracked_pre_border":
            raise ValueError("Production vertex selection must use pre-border masks")
        if self.interval_evaluation != "native_exact":
            raise ValueError("Production interval evaluation is CPU native_exact")
        if not 1.0 <= float(self.spatial_recall_repair_max_scale) <= 1.05:
            raise ValueError("Production Recall repair scale must be in [1, 1.05]")
        if self.border_max_expand_px != 16.0:
            raise ValueError("Production border expansion must be capped at 16 px")
        if self.border_influence_px != 16.0:
            raise ValueError("Production border influence band must be 16 px")
        if not self.border_corner_support:
            raise ValueError("Production two-axis corner support must remain enabled")
        if self.gapfill_max_gap != 15:
            raise ValueError("Production polygon gap-fill limit must remain 15 frames")
        for name, value in (
            ("spatial recall", self.spatial_recall_floor),
            ("spatial IoU", self.spatial_iou_floor),
            ("temporal recall", self.temporal_recall_floor),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


PRODUCTION = ProductionConfig()
PRODUCTION.validate()


__all__ = (
    "ALLOWED_VERTICES",
    "LABELS",
    "PROFILE_ID",
    "RUNTIME_CANDIDATE_PROFILE_ID",
    "RUNTIME_POLYGON_PROFILE_ID",
    "SCREEN_OCCUPANCY_THRESHOLDS",
    "PRODUCTION",
    "ProductionConfig",
)
