"""Pipeline stages owned by evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .mask_iou import evaluate_mask_sqlites


@dataclass(frozen=True)
class MaskIouEvaluationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "mask_iou_evaluation"
    requires: frozenset[str] = frozenset({"tracked_sqlite", "predictions_sqlite"})
    provides: frozenset[str] = frozenset({"evaluation_summary"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "summary.json"
        summary = evaluate_mask_sqlites(
            context.artifacts["tracked_sqlite"],
            context.artifacts["predictions_sqlite"],
            output,
        )
        return StageResult({"evaluation_summary": output}, summary)
