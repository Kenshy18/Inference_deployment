"""Stable Production entrypoint for virtual-component exact-mask NMS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.live_preview import (
    active_postprocess_preview,
    geometry_from_detection_record,
)
from contracts.detections import transform_detection_jsonl
from contracts.stages import StageContext, StageResult

from .component_virtual import (
    ProductionVirtualComponentNms,
    VirtualComponentNmsDiagnostics,
)


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
        implementation = ProductionVirtualComponentNms(**PRODUCTION_OPTIONS)
        output = context.stage_dir / "nms.jsonl"
        diagnostic_totals = {
            key: 0
            for key in VirtualComponentNmsDiagnostics.__dataclass_fields__
            if key not in {"input_detections", "output_detections"}
        }

        def suppress(record: dict[str, Any]) -> dict[str, Any]:
            transformed = dict(record)
            retained, diagnostics = implementation.apply_with_diagnostics(
                list(record["detections"])
            )
            transformed["detections"] = retained
            for key in diagnostic_totals:
                diagnostic_totals[key] += int(getattr(diagnostics, key))
            return transformed

        preview = active_postprocess_preview()

        def show_result(_before: dict[str, Any], after: dict[str, Any]) -> None:
            if preview is not None and preview.should_sample(context.stage_id):
                preview.submit(
                    geometry_from_detection_record(
                        after,
                        stage=context.stage_id,
                        label=self.name,
                        detail=f"Production mask NMS kept {len(after['detections'])}",
                    )
                )

        stats = transform_detection_jsonl(
            context.artifacts["scored_jsonl"],
            output,
            suppress,
            on_record=show_result,
        )
        return StageResult(
            {"nms_jsonl": output},
            {
                **stats,
                **diagnostic_totals,
                "algorithm": implementation.name,
                "fill_all_holes": implementation.fill_all_holes,
                "unconditional_owner_ratio_max": (
                    implementation.unconditional_owner_ratio_max
                ),
                "island_other_coverage_min": (
                    implementation.island_other_coverage_min
                ),
                "island_to_other_area_max": implementation.island_to_other_area_max,
                "mask_iou_threshold": implementation.mask_iou_threshold,
                "mask_small_iou_threshold": implementation.mask_small_iou_threshold,
                "mask_tiny_iou_threshold": implementation.mask_tiny_iou_threshold,
                "mask_small_area": implementation.mask_small_area,
                "mask_tiny_area": implementation.mask_tiny_area,
                "mask_containment_coverage_min": (
                    implementation.mask_containment_coverage_min
                ),
                "mask_contain_ratio_max": implementation.mask_contain_ratio_max,
                "mask_small_contain_ratio_max": (
                    implementation.mask_small_contain_ratio_max
                ),
                "mask_tiny_contain_ratio_max": (
                    implementation.mask_tiny_contain_ratio_max
                ),
                "status": "production",
                "profile": self.name,
                "thresholds_frozen": True,
            },
        )


__all__ = ("PRODUCTION_OPTIONS", "ProductionMaskNmsStage")
