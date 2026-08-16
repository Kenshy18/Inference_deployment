"""Single immutable semantic contract for the 2026-08-14 candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


PROFILE_ID = "production_candidate_adaptive_vertices_v2"
POLYGON_PROFILE_ID = "polygon_adaptive_keyframe_v2"
LEGACY_PROFILE_ID = "production_candidate_20260814_v1"
LEGACY_POLYGON_PROFILE_ID = "polygon14_keyframe_v1"
LABELS = ("女性器", "男性器", "結合部分")
INTERVAL_EVALUATION_MODES = ("cuda_lazy_exact", "native_exact")


@dataclass(frozen=True, slots=True)
class NmsConfig:
    fill_all_holes: bool = True
    unconditional_owner_island_ratio_max: float = 0.01
    island_other_coverage_min: float = 0.80
    island_to_other_area_max: float = 0.50
    mask_iou_threshold: float = 0.20
    mask_small_iou_threshold: float = 0.10
    mask_tiny_iou_threshold: float = 0.05
    small_area: float = 5000.0
    tiny_area: float = 2000.0
    containment_coverage_min: float = 0.80
    contain_ratio_max: float = 8.0
    small_contain_ratio_max: float = 5.0
    tiny_contain_ratio_max: float = 5.0
    adaptive_band_area: str = "production_continuous_contour_or_bbox"
    overlap_geometry: str = "exact_native_pixel_mask"
    bbox_role: str = "broad_phase_only"


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    remove_short_tracks_max_frames: int = 10
    association_geometry: str = "raw_before_topology_cleanup"


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    adaptive_vertex_policy: bool = True
    allowed_vertices_per_component: tuple[int, ...] = (14, 16, 18, 20)
    track_area_quantile: float = 0.999
    screen_occupancy_thresholds: tuple[float, ...] = (0.03, 0.10, 0.25)
    vertex_selection_source: str = "tracked_pre_border"
    vertex_selection_comparison: str = "strictly_greater_than_threshold"
    recall_floor: float = 0.97
    recall_repair_max_scale: float = 1.05
    iou_floor: float = 0.95
    dense_vertices: int = 64
    coverage_quantile: float = 0.65
    maximum_intersection_radius: float = 0.20
    intersection_regularization: float = 0.01

    @property
    def vertices_per_component(self) -> int:
        """Compatibility minimum for the parity-frozen fixed-14 bridge."""
        return int(self.allowed_vertices_per_component[0])


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    border_trigger_px: float = 10.0
    border_expand_ratio: float = 0.10
    border_min_expand_px: float = 6.0
    border_max_expand_px: float = 16.0
    border_influence_px: float = 16.0
    border_corner_support: bool = True
    endpoint_extend_frames: int = 5
    endpoint_motion_frames: int = 10
    endpoint_max_speed_px: float = 1000.0


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    target_interval: int = 6
    recall_floor: float = 0.97
    objective: str = "mean_iou_vs_soft_keyframe_interval"
    optimizer: str = "new_production_v1_multistate_dp"
    selected_edge_validation: str = "cuda_lazy_exact_then_exact_final_audit"
    pair_vote_mode: str = "per_key_iou_under_exact_recall"
    pair_vote_sweeps: int = 2
    topology_constraint: str = "simple_polygon_at_keys_and_every_integer_frame"
    invalid_edge_policy: str = "lazy_split_with_recall_feasible_key"
    invalid_pair_vote_policy: str = "best_valid_iou_trial_or_dp_fallback"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    label_workers: int = 3
    optimizer_workers: int = 1
    candidate_frame_workers: int = 1
    pair_vote_threads: int = 8
    native_batch_threads: int = 8
    # Keep one conservative, data-independent CUDA screening budget.  More
    # aggressive area-adaptive filtering remains experimental until it has
    # been validated on multiple independent SQLite inputs.
    cuda_prefilter_deficit_budget: float = 0.10
    cuda_prefilter_small_area: float = 0.0
    cuda_prefilter_small_deficit_budget: float = 0.10
    lazy_fallback_min_seconds: float = 0.5
    lazy_fallback_min_exact_edges: int = 1024
    lazy_fallback_infeasible_ratio: float = 0.875
    gc_interval: int = 8
    gapfill_max_gap: int = 15
    keyframe_max_gap: int = 30
    max_run_frames: int = 30000
    run_overlap_frames: int = 900
    predictor_device: str = "cpu"
    interval_evaluation: str = "native_exact"


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    profile_id: str = PROFILE_ID
    polygon_profile_id: str = POLYGON_PROFILE_ID
    labels: tuple[str, ...] = LABELS
    nms: NmsConfig = field(default_factory=NmsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_schema: str = "unchanged_unified_v3_revision5"

    def validate(self) -> None:
        if self.labels != LABELS:
            raise ValueError(f"candidate labels are frozen to {LABELS}")
        adaptive_profile = (
            self.profile_id == PROFILE_ID
            and self.polygon_profile_id == POLYGON_PROFILE_ID
            and self.spatial.adaptive_vertex_policy
        )
        legacy_profile = (
            self.profile_id == LEGACY_PROFILE_ID
            and self.polygon_profile_id == LEGACY_POLYGON_PROFILE_ID
            and not self.spatial.adaptive_vertex_policy
        )
        if not (adaptive_profile or legacy_profile):
            raise ValueError("candidate profile and polygon profile do not match")
        if self.spatial.allowed_vertices_per_component != (14, 16, 18, 20):
            raise ValueError("candidate vertex counts must be (14, 16, 18, 20)")
        if self.spatial.screen_occupancy_thresholds != (0.03, 0.10, 0.25):
            raise ValueError(
                "candidate occupancy thresholds must be (0.03, 0.10, 0.25)"
            )
        if not (
            math.isfinite(float(self.spatial.track_area_quantile))
            and 0.0 < float(self.spatial.track_area_quantile) <= 1.0
        ):
            raise ValueError("track area quantile must be in (0, 1]")
        if len(self.spatial.screen_occupancy_thresholds) + 1 != len(
            self.spatial.allowed_vertices_per_component
        ):
            raise ValueError(
                "vertex counts must have exactly one more entry than thresholds"
            )
        if any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0
            for value in self.spatial.screen_occupancy_thresholds
        ) or tuple(sorted(self.spatial.screen_occupancy_thresholds)) != (
            self.spatial.screen_occupancy_thresholds
        ):
            raise ValueError(
                "occupancy thresholds must be finite and strictly increasing"
            )
        if self.spatial.vertex_selection_source != "tracked_pre_border":
            raise ValueError(
                "vertex selection must use tracked masks before border expansion"
            )
        if (
            self.spatial.vertex_selection_comparison
            != "strictly_greater_than_threshold"
        ):
            raise ValueError(
                "vertex threshold comparison is part of the frozen contract"
            )
        for name, value in (
            ("spatial recall", self.spatial.recall_floor),
            ("spatial IoU", self.spatial.iou_floor),
            ("temporal recall", self.temporal.recall_floor),
            ("NMS IoU", self.nms.mask_iou_threshold),
            ("NMS small IoU", self.nms.mask_small_iou_threshold),
            ("NMS tiny IoU", self.nms.mask_tiny_iou_threshold),
            ("NMS containment coverage", self.nms.containment_coverage_min),
        ):
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not math.isfinite(float(self.spatial.recall_repair_max_scale)) or not (
            1.0 <= float(self.spatial.recall_repair_max_scale) <= 1.05
        ):
            raise ValueError("spatial Recall repair scale must be in [1, 1.05]")
        if adaptive_profile:
            if not (
                math.isfinite(float(self.preparation.border_max_expand_px))
                and 0.0 < float(self.preparation.border_max_expand_px) <= 16.0
            ):
                raise ValueError("candidate border expansion must be capped at 16 px")
            if not (
                math.isfinite(float(self.preparation.border_influence_px))
                and float(self.preparation.border_influence_px) == 16.0
            ):
                raise ValueError("candidate border influence band must be 16 px")
            if not self.preparation.border_corner_support:
                raise ValueError("candidate border corner support must remain enabled")
        elif (
            self.preparation.border_max_expand_px != 40.0
            or self.preparation.border_influence_px != 24.0
            or self.preparation.border_corner_support
        ):
            raise ValueError("legacy fixed-14 border contract drift")
        if not (
            self.nms.mask_tiny_iou_threshold
            <= self.nms.mask_small_iou_threshold
            <= self.nms.mask_iou_threshold
        ):
            raise ValueError("adaptive NMS IoU thresholds must be nondecreasing")
        if not 0.0 < self.nms.tiny_area < self.nms.small_area:
            raise ValueError("adaptive NMS areas require 0 < tiny < small")
        for name, value in (
            ("NMS containment ratio", self.nms.contain_ratio_max),
            ("NMS small containment ratio", self.nms.small_contain_ratio_max),
            ("NMS tiny containment ratio", self.nms.tiny_contain_ratio_max),
        ):
            if not math.isfinite(float(value)) or float(value) < 1.0:
                raise ValueError(f"{name} must be finite and at least one")
        if self.temporal.target_interval < 1:
            raise ValueError("target interval must be positive")
        if self.temporal.pair_vote_sweeps < 1:
            raise ValueError("pair-vote sweeps must be positive")
        for name, value in (
            (
                "unconditional island ratio",
                self.nms.unconditional_owner_island_ratio_max,
            ),
            ("island coverage", self.nms.island_other_coverage_min),
            ("island-to-other area ratio", self.nms.island_to_other_area_max),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (
            ("short-track cutoff", self.tracking.remove_short_tracks_max_frames),
            ("label workers", self.runtime.label_workers),
            ("optimizer workers", self.runtime.optimizer_workers),
            ("candidate frame workers", self.runtime.candidate_frame_workers),
            ("pair-vote threads", self.runtime.pair_vote_threads),
            ("native threads", self.runtime.native_batch_threads),
            (
                "lazy fallback minimum exact edges",
                self.runtime.lazy_fallback_min_exact_edges,
            ),
            ("GC interval", self.runtime.gc_interval),
            ("gap-fill maximum", self.runtime.gapfill_max_gap),
            ("keyframe maximum gap", self.runtime.keyframe_max_gap),
            ("maximum run frames", self.runtime.max_run_frames),
            ("run overlap", self.runtime.run_overlap_frames),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            min(
                self.runtime.label_workers,
                self.runtime.optimizer_workers,
                self.runtime.candidate_frame_workers,
                self.runtime.pair_vote_threads,
                self.runtime.native_batch_threads,
                self.runtime.gc_interval,
            )
            < 1
        ):
            raise ValueError("worker, thread, and GC settings must be positive")
        if not math.isfinite(float(self.runtime.lazy_fallback_min_seconds)) or (
            float(self.runtime.lazy_fallback_min_seconds) < 0.0
        ):
            raise ValueError(
                "lazy fallback minimum seconds must be finite and non-negative"
            )
        if not (
            math.isfinite(float(self.runtime.cuda_prefilter_deficit_budget))
            and 0.0 <= float(self.runtime.cuda_prefilter_deficit_budget) <= 1.0
        ):
            raise ValueError("CUDA prefilter deficit budget must be in [0, 1]")
        if not (
            math.isfinite(float(self.runtime.cuda_prefilter_small_area))
            and float(self.runtime.cuda_prefilter_small_area) >= 0.0
        ):
            raise ValueError("CUDA prefilter small area must be non-negative")
        if not (
            math.isfinite(float(self.runtime.cuda_prefilter_small_deficit_budget))
            and 0.0 <= float(self.runtime.cuda_prefilter_small_deficit_budget) <= 1.0
        ):
            raise ValueError("CUDA prefilter small budget must be in [0, 1]")
        if (
            self.runtime.cuda_prefilter_small_deficit_budget
            < self.runtime.cuda_prefilter_deficit_budget
        ):
            raise ValueError("CUDA small-mask budget cannot be less conservative")
        if not (
            math.isfinite(float(self.runtime.lazy_fallback_infeasible_ratio))
            and 0.0 <= float(self.runtime.lazy_fallback_infeasible_ratio) <= 1.0
        ):
            raise ValueError("lazy fallback infeasible ratio must be in [0, 1]")
        if self.runtime.interval_evaluation not in INTERVAL_EVALUATION_MODES:
            raise ValueError(
                "interval evaluation must be one of "
                f"{INTERVAL_EVALUATION_MODES}, got "
                f"{self.runtime.interval_evaluation!r}"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


CANDIDATE = CandidateConfig()
CANDIDATE.validate()


def legacy_fixed14_candidate() -> CandidateConfig:
    """Return the exact pre-v2 contract used by the promoted stable profile."""
    from dataclasses import replace

    value = replace(
        CANDIDATE,
        profile_id=LEGACY_PROFILE_ID,
        polygon_profile_id=LEGACY_POLYGON_PROFILE_ID,
        spatial=replace(CANDIDATE.spatial, adaptive_vertex_policy=False),
        preparation=replace(
            CANDIDATE.preparation,
            border_max_expand_px=40.0,
            border_influence_px=24.0,
            border_corner_support=False,
        ),
    )
    value.validate()
    return value


def with_target_interval(
    target_interval: int,
    config: CandidateConfig = CANDIDATE,
) -> CandidateConfig:
    """Return the same candidate contract with a different soft interval target."""
    from dataclasses import replace

    updated = replace(
        config,
        temporal=replace(config.temporal, target_interval=int(target_interval)),
    )
    updated.validate()
    return updated


def with_interval_evaluation(
    interval_evaluation: str,
    config: CandidateConfig = CANDIDATE,
) -> CandidateConfig:
    """Select only the interval evaluator; all quality semantics stay frozen."""
    from dataclasses import replace

    updated = replace(
        config,
        runtime=replace(
            config.runtime,
            interval_evaluation=str(interval_evaluation),
        ),
    )
    updated.validate()
    return updated
