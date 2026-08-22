"""Frozen semantic and runtime contract for Production Catmull--Rom masks."""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_ID = "production_catmull_rom_cpu_exact_v1"
INTERPOLATION_METHOD = "catmull_rom_uniform_tension_1_v1"
CURVE_CONTRACT = "closed_uniform_catmull_rom_tension_1_factor_1_over_6"


@dataclass(frozen=True, slots=True)
class CurveProductionConfig:
    target_interval: int = 6
    recall_floor: float = 0.97
    samples_per_segment: int = 16
    dense_contour_samples: int = 192
    # Four states retain the validated coverage range while removing the
    # redundant 1.5% rung.  This reduces the exact DP graph from 25 to 16
    # state pairs per frame edge; full-corpus tail-quality gates verify it.
    state_scales: tuple[float, ...] = (1.0, 1.005, 1.025, 1.035)
    fast_state_scales: tuple[float, ...] = (1.0, 1.005)
    fast_state_target_ratio: float = 0.95
    low_iou_quadratic_weight: float = 4.0
    # Four target intervals preserve the selected V3 paths while avoiding
    # exact evaluation of graph edges that the supported 1--6 range never uses.
    maximum_gap: int = 24
    pair_vote_sweeps: int = 2
    point_refine_sweeps: int = 2
    point_refine_scheduler: str = "color_batched"
    # A local interpolation guard, not a similarity constraint on the AI
    # source.  0.85 removes visible one-frame collapses/inflation while keeping
    # the requested interval meaningfully adjustable; stricter 0.90/0.92
    # settings forced highly non-rigid tracks close to every-frame keys.
    quality_rescue_iou_floor: float = 0.85
    quality_rescue_regret_floor: float = 0.04
    quality_rescue_area_ratio_cap: float = 1.20
    # Zero means that no artificial key-count ceiling is imposed.  A rescue key
    # is still accepted only when the independent spatial curve is materially
    # better and every exact Recall/topology/lower-tail guard remains valid.
    # Consequently difficult motion pays with additional keys instead of a
    # silently bad frame.  Positive values remain available to experiments as
    # an explicit insertion cap.
    quality_rescue_maximum_extra_keys: int = 0
    # The former target-density allowance could stop rescue before its explicit
    # IoU/area guards were met.  Disabling it preserves the soft-target rule.
    quality_rescue_density_budget: bool = False
    quality_rescue_maximum_iou_regression: float = 0.005
    quality_rescue_maximum_area_ratio_regression: float = 0.01
    native_cpu_threads: int = 8
    native_batch_cases: int = 4096
    # The exact evaluator is bounded by 600-frame chunks.  A 2 GiB per-group
    # ceiling lets process-sharded routes keep large 1080p masks in OpenCV
    # bitmap form (fast) while still bounding long-video memory.  The scheduler
    # caps the default to six concurrent groups, so the theoretical cache
    # ceiling is 12 GiB and normally remains far below it.
    native_reference_cache_bytes: int = 2 * 1024 * 1024 * 1024
    max_run_frames: int = 600
    run_overlap_frames: int = 30
    gapfill_max_gap: int = 15
    spatial_scale_maximum: float = 1.08
    spatial_scale_step: float = 0.002
    emergency_scale_maximum: float = 1.35

    def validate(self) -> None:
        if int(self.target_interval) < 1:
            raise ValueError("curve target interval must be at least one")
        if not 0.0 < float(self.recall_floor) <= 1.0:
            raise ValueError("curve Recall floor must be in (0, 1]")
        if int(self.samples_per_segment) != 16:
            raise ValueError("Production curve sampling is frozen to 16 per segment")
        if int(self.dense_contour_samples) < 60:
            raise ValueError("dense contour sampling is too small")
        if not self.state_scales or any(
            not 0.95 <= float(value) <= 1.10 for value in self.state_scales
        ):
            raise ValueError("curve state scales must remain in [0.95, 1.10]")
        if not self.fast_state_scales or any(
            float(value) not in self.state_scales for value in self.fast_state_scales
        ):
            raise ValueError("fast curve states must be a subset of all states")
        if not 0.0 < float(self.fast_state_target_ratio) <= 1.0:
            raise ValueError("fast state target ratio must be in (0, 1]")
        if int(self.maximum_gap) < 1 or int(self.gapfill_max_gap) < 0:
            raise ValueError("curve temporal gap settings are invalid")
        if int(self.pair_vote_sweeps) < 0 or int(self.point_refine_sweeps) < 0:
            raise ValueError("curve refinement sweeps must be non-negative")
        if not 0.0 <= float(self.quality_rescue_iou_floor) <= 1.0:
            raise ValueError("curve quality rescue IoU floor must be in [0, 1]")
        if float(self.quality_rescue_regret_floor) < 0.0:
            raise ValueError("curve quality rescue regret floor must be non-negative")
        if float(self.quality_rescue_area_ratio_cap) < 1.0:
            raise ValueError("curve quality rescue area cap must be at least one")
        if int(self.quality_rescue_maximum_extra_keys) < 0:
            raise ValueError("curve quality rescue key budget must be non-negative")
        if float(self.quality_rescue_maximum_iou_regression) < 0.0:
            raise ValueError("curve quality rescue IoU regression must be non-negative")
        if float(self.quality_rescue_maximum_area_ratio_regression) < 0.0:
            raise ValueError(
                "curve quality rescue area regression must be non-negative"
            )
        if self.point_refine_scheduler not in {"sequential", "color_batched"}:
            raise ValueError("unsupported curve point refinement scheduler")
        if min(int(self.native_cpu_threads), int(self.native_batch_cases)) < 1:
            raise ValueError("curve CPU batch settings must be positive")
        if int(self.native_reference_cache_bytes) < 0:
            raise ValueError("curve reference cache budget must be non-negative")
        if int(self.max_run_frames) < 32:
            raise ValueError("curve run chunks must contain at least 32 frames")
        if not 0 <= int(self.run_overlap_frames) < int(self.max_run_frames) // 2:
            raise ValueError("curve overlap must be below half the chunk length")
        if (
            not 1.0
            <= float(self.spatial_scale_maximum)
            <= float(self.emergency_scale_maximum)
        ):
            raise ValueError("curve spatial scale limits are invalid")
        if float(self.spatial_scale_step) <= 0.0:
            raise ValueError("curve spatial scale step must be positive")


CURVE_PRODUCTION = CurveProductionConfig()
CURVE_PRODUCTION.validate()


__all__ = (
    "CURVE_CONTRACT",
    "CURVE_PRODUCTION",
    "INTERPOLATION_METHOD",
    "PROFILE_ID",
    "CurveProductionConfig",
)
