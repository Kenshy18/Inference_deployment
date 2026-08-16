"""Pipeline stage owned by non-maximum suppression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.detections import transform_detection_jsonl
from contracts.stages import StageContext, StageResult
from common.live_preview import (
    active_postprocess_preview,
    geometry_from_detection_record,
)

from .adaptive import AdaptiveNms
from .component_aware import ComponentAwareMaskNms
from .component_virtual import (
    VirtualComponentMaskNms,
    VirtualComponentNms,
    VirtualComponentNmsDiagnostics,
)


@dataclass(frozen=True)
class AdaptiveNmsStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "non_maximum_suppression"
    requires: frozenset[str] = frozenset({"scored_jsonl"})
    provides: frozenset[str] = frozenset({"nms_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        implementation = AdaptiveNms(
            iou_threshold=float(self.options.get("iou_threshold", 0.20)),
            small_iou_threshold=float(self.options.get("small_iou_threshold", 0.10)),
            tiny_iou_threshold=float(self.options.get("tiny_iou_threshold", 0.05)),
            island_cleanup_policy=str(
                self.options.get("island_cleanup_policy", "disabled")
            ),
            remove_small_islands=bool(self.options.get("remove_small_islands", False)),
            small_island_ratio_max=float(
                self.options.get("small_island_ratio_max", 0.10)
            ),
            fill_all_holes=bool(self.options.get("fill_all_holes", True)),
            island_unconditional_owner_ratio_max=float(
                self.options.get("island_unconditional_owner_ratio_max", 0.01)
            ),
            island_other_coverage_min=float(
                self.options.get("island_other_coverage_min", 0.90)
            ),
            island_to_other_area_max=float(
                self.options.get("island_to_other_area_max", 0.30)
            ),
        )
        output = context.stage_dir / "nms.jsonl"

        def suppress(record: dict[str, Any]) -> dict[str, Any]:
            transformed = dict(record)
            transformed["detections"] = implementation.apply(list(record["detections"]))
            return transformed

        preview = active_postprocess_preview()

        def show_result(_before: dict[str, Any], after: dict[str, Any]) -> None:
            if preview is not None and preview.should_sample(context.stage_id):
                preview.submit(
                    geometry_from_detection_record(
                        after,
                        stage=context.stage_id,
                        label=self.name,
                        detail=f"NMS kept {len(after['detections'])}",
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
                "algorithm": implementation.name,
                "island_cleanup_policy": implementation.island_cleanup_policy,
                "remove_small_islands": implementation.remove_small_islands,
                "small_island_ratio_max": implementation.small_island_ratio_max,
                "fill_all_holes": implementation.fill_all_holes,
                "island_unconditional_owner_ratio_max": (
                    implementation.island_unconditional_owner_ratio_max
                ),
                "island_other_coverage_min": (implementation.island_other_coverage_min),
                "island_to_other_area_max": (implementation.island_to_other_area_max),
            },
        )


@dataclass(frozen=True)
class ComponentAwareMaskNmsStage:
    """Opt-in mask-IoU/component-topology Production candidate."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "component_aware_mask_nms_candidate_v2"
    requires: frozenset[str] = frozenset({"scored_jsonl"})
    provides: frozenset[str] = frozenset({"nms_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        implementation = ComponentAwareMaskNms(
            mask_iou_threshold=float(self.options.get("mask_iou_threshold", 0.70)),
            fill_all_holes=bool(self.options.get("fill_all_holes", True)),
            unconditional_owner_ratio_max=float(
                self.options.get("unconditional_owner_ratio_max", 0.01)
            ),
            island_other_coverage_min=float(
                self.options.get("island_other_coverage_min", 0.80)
            ),
            island_to_other_area_max=float(
                self.options.get("island_to_other_area_max", 0.50)
            ),
        )
        output = context.stage_dir / "nms.jsonl"
        diagnostic_totals = {
            key: 0
            for key in (
                "holes_filled",
                "tiny_islands_removed",
                "bbox_overlap_pairs",
                "mask_iou_pairs",
                "nms_suppressed",
                "redundant_islands_removed",
            )
        }

        def suppress(record: dict[str, Any]) -> dict[str, Any]:
            transformed = dict(record)
            retained, diagnostics = implementation.apply_with_diagnostics(
                list(record["detections"])
            )
            transformed["detections"] = retained
            values = diagnostics.as_dict()
            for key in diagnostic_totals:
                diagnostic_totals[key] += int(values[key])
            return transformed

        preview = active_postprocess_preview()

        def show_result(_before: dict[str, Any], after: dict[str, Any]) -> None:
            if preview is not None and preview.should_sample(context.stage_id):
                preview.submit(
                    geometry_from_detection_record(
                        after,
                        stage=context.stage_id,
                        label=self.name,
                        detail=f"Mask NMS kept {len(after['detections'])}",
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
                "mask_iou_threshold": implementation.mask_iou_threshold,
                "fill_all_holes": implementation.fill_all_holes,
                "unconditional_owner_ratio_max": (
                    implementation.unconditional_owner_ratio_max
                ),
                "island_other_coverage_min": (implementation.island_other_coverage_min),
                "island_to_other_area_max": (implementation.island_to_other_area_max),
            },
        )


@dataclass(frozen=True)
class VirtualComponentNmsStage:
    """Opt-in role-aware virtual-component NMS Production candidate v3."""

    options: dict[str, Any] = field(default_factory=dict)
    name: str = "virtual_component_nms_candidate_v3"
    requires: frozenset[str] = frozenset({"scored_jsonl"})
    provides: frozenset[str] = frozenset({"nms_jsonl"})

    def run(self, context: StageContext) -> StageResult:
        comparison_policy = str(self.options.get("comparison_policy", "legacy_bbox"))
        implementation_class = (
            VirtualComponentMaskNms
            if comparison_policy == "adaptive_mask"
            else VirtualComponentNms
        )
        implementation = implementation_class(
            fill_all_holes=bool(self.options.get("fill_all_holes", True)),
            unconditional_owner_ratio_max=float(
                self.options.get("unconditional_owner_ratio_max", 0.01)
            ),
            island_other_coverage_min=float(
                self.options.get("island_other_coverage_min", 0.80)
            ),
            island_to_other_area_max=float(
                self.options.get("island_to_other_area_max", 0.50)
            ),
            legacy_iou_threshold=float(self.options.get("legacy_iou_threshold", 0.20)),
            legacy_small_iou_threshold=float(
                self.options.get("legacy_small_iou_threshold", 0.10)
            ),
            legacy_tiny_iou_threshold=float(
                self.options.get("legacy_tiny_iou_threshold", 0.05)
            ),
            comparison_policy=comparison_policy,
            mask_iou_threshold=float(self.options.get("mask_iou_threshold", 0.20)),
            mask_small_iou_threshold=float(
                self.options.get("mask_small_iou_threshold", 0.10)
            ),
            mask_tiny_iou_threshold=float(
                self.options.get("mask_tiny_iou_threshold", 0.05)
            ),
            mask_small_area=float(self.options.get("mask_small_area", 5000.0)),
            mask_tiny_area=float(self.options.get("mask_tiny_area", 2000.0)),
            mask_containment_coverage_min=float(
                self.options.get("mask_containment_coverage_min", 0.80)
            ),
            mask_contain_ratio_max=float(
                self.options.get("mask_contain_ratio_max", 8.0)
            ),
            mask_small_contain_ratio_max=float(
                self.options.get("mask_small_contain_ratio_max", 5.0)
            ),
            mask_tiny_contain_ratio_max=float(
                self.options.get("mask_tiny_contain_ratio_max", 5.0)
            ),
        )
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
                        detail=f"Virtual component NMS kept {len(after['detections'])}",
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
                "island_other_coverage_min": (implementation.island_other_coverage_min),
                "island_to_other_area_max": (implementation.island_to_other_area_max),
                "legacy_iou_threshold": implementation.legacy_iou_threshold,
                "legacy_small_iou_threshold": (
                    implementation.legacy_small_iou_threshold
                ),
                "legacy_tiny_iou_threshold": (implementation.legacy_tiny_iou_threshold),
                "comparison_policy": implementation.comparison_policy,
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
            },
        )
