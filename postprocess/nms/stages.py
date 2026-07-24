"""Pipeline stage owned by non-maximum suppression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.detections import transform_detection_jsonl
from contracts.stages import StageContext, StageResult

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

        stats = transform_detection_jsonl(
            context.artifacts["scored_jsonl"], output, suppress
        )
        return StageResult(
            {"nms_jsonl": output},
            {**stats, "algorithm": implementation.name},
        )
