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
            {**stats, "algorithm": implementation.name},
        )
