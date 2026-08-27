"""Canonical immutable configuration for Production topology cleanup and NMS."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ProductionNmsConfig:
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

    def validate(self) -> None:
        unit_interval_values = {
            "unconditional owner-island ratio": (
                self.unconditional_owner_island_ratio_max
            ),
            "island coverage": self.island_other_coverage_min,
            "island-to-other area ratio": self.island_to_other_area_max,
            "mask IoU": self.mask_iou_threshold,
            "small-mask IoU": self.mask_small_iou_threshold,
            "tiny-mask IoU": self.mask_tiny_iou_threshold,
            "containment coverage": self.containment_coverage_min,
        }
        for name, value in unit_interval_values.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not (
            self.mask_tiny_iou_threshold
            <= self.mask_small_iou_threshold
            <= self.mask_iou_threshold
        ):
            raise ValueError("mask IoU thresholds must be nondecreasing")
        if not 0.0 < self.tiny_area < self.small_area:
            raise ValueError("mask area bands require 0 < tiny < small")
        for name, value in (
            ("containment ratio", self.contain_ratio_max),
            ("small containment ratio", self.small_contain_ratio_max),
            ("tiny containment ratio", self.tiny_contain_ratio_max),
        ):
            if not math.isfinite(float(value)) or float(value) < 1.0:
                raise ValueError(f"{name} must be finite and at least one")
        if self.overlap_geometry != "exact_native_pixel_mask":
            raise ValueError("Production NMS must use exact native pixel masks")
        if self.bbox_role != "broad_phase_only":
            raise ValueError("bounding boxes may only be used as a broad phase")

    def implementation_options(self) -> dict[str, object]:
        """Translate semantic names to the NMS engine's constructor contract."""
        self.validate()
        return {
            "fill_all_holes": self.fill_all_holes,
            "unconditional_owner_ratio_max": (
                self.unconditional_owner_island_ratio_max
            ),
            "island_other_coverage_min": self.island_other_coverage_min,
            "island_to_other_area_max": self.island_to_other_area_max,
            "mask_iou_threshold": self.mask_iou_threshold,
            "mask_small_iou_threshold": self.mask_small_iou_threshold,
            "mask_tiny_iou_threshold": self.mask_tiny_iou_threshold,
            "mask_small_area": self.small_area,
            "mask_tiny_area": self.tiny_area,
            "mask_containment_coverage_min": self.containment_coverage_min,
            "mask_contain_ratio_max": self.contain_ratio_max,
            "mask_small_contain_ratio_max": self.small_contain_ratio_max,
            "mask_tiny_contain_ratio_max": self.tiny_contain_ratio_max,
        }


PRODUCTION_NMS_CONFIG = ProductionNmsConfig()
PRODUCTION_NMS_CONFIG.validate()


__all__ = ("PRODUCTION_NMS_CONFIG", "ProductionNmsConfig")
