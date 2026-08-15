"""Immutable Production contract and translation to the validated runtime.

The public contract lives here.  The older experimental dataclasses are only
used as a compatibility payload for the frozen optimizer bridge; callers do
not configure that implementation directly.
"""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_ID = "production_polygon14_cpu_exact_v1"
RUNTIME_POLYGON_PROFILE_ID = "polygon14_keyframe_v1"
LABELS = ("女性器", "男性器", "結合部分")


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    profile_id: str = PROFILE_ID
    labels: tuple[str, ...] = LABELS
    target_interval: int = 6
    vertices_per_component: int = 14
    spatial_recall_floor: float = 0.97
    spatial_iou_floor: float = 0.95
    temporal_recall_floor: float = 0.97
    interval_evaluation: str = "native_exact"
    pair_vote_sweeps: int = 2
    remove_short_tracks_max_frames: int = 10
    # The selected CPU corpus was published with an exact audit even when the
    # fixed 14-point spatial representation was infeasible.  Preserve that
    # evaluated output and expose every violation in the manifest.
    require_zero_exact_recall_violations: bool = False

    def validate(self) -> None:
        if self.labels != LABELS:
            raise ValueError(f"Production labels are frozen to {LABELS}")
        if self.target_interval < 1:
            raise ValueError("target_interval must be >= 1")
        if self.vertices_per_component != 14:
            raise ValueError("Production requires exactly 14 vertices")
        if self.interval_evaluation != "native_exact":
            raise ValueError("Production interval evaluation is CPU native_exact")
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
    "LABELS",
    "PROFILE_ID",
    "RUNTIME_POLYGON_PROFILE_ID",
    "PRODUCTION",
    "ProductionConfig",
)
