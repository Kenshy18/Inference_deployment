"""Stable Production entrypoint for virtual-component exact-mask NMS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .stages import VirtualComponentNmsStage


PRODUCTION_OPTIONS: dict[str, object] = {
    "comparison_policy": "adaptive_mask",
    "fill_all_holes": True,
    "unconditional_owner_ratio_max": 0.01,
    "island_other_coverage_min": 0.80,
    "island_to_other_area_max": 0.50,
    "mask_iou_threshold": 0.20,
    "mask_small_iou_threshold": 0.10,
    "mask_tiny_iou_threshold": 0.05,
    "mask_small_area": 5000.0,
    "mask_tiny_area": 2000.0,
    "mask_containment_coverage_min": 0.80,
    "mask_contain_ratio_max": 8.0,
    "mask_small_contain_ratio_max": 5.0,
    "mask_tiny_contain_ratio_max": 5.0,
}


@dataclass(frozen=True)
class ProductionMaskNmsStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "production_virtual_component_mask_nms_v1"
    requires: frozenset[str] = frozenset({"scored_jsonl"})
    provides: frozenset[str] = frozenset({"nms_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        unknown = set(self.options) - {"profile_note"}
        if unknown:
            raise ValueError(
                "Production NMS thresholds are frozen; unsupported options: "
                f"{sorted(unknown)}"
            )
        result = VirtualComponentNmsStage(dict(PRODUCTION_OPTIONS)).run(context)
        return StageResult(
            result.artifacts,
            {
                **result.metadata,
                "status": "production",
                "profile": self.name,
                "thresholds_frozen": True,
            },
        )


__all__ = ("PRODUCTION_OPTIONS", "ProductionMaskNmsStage")
