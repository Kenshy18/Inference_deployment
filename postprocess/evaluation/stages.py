"""Pipeline stages owned by evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .exact import kfeval_main
from .mask_iou import evaluate_mask_sqlites


@dataclass(frozen=True)
class ExactEllipseEvaluationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "ellipse_exact_evaluation"
    requires: frozenset[str] = frozenset({"filled_union_json", "tracked_sqlite"})
    provides: frozenset[str] = frozenset({"evaluation_summary"})

    def run(self, context: StageContext) -> StageResult:
        kfeval_main(
            [
                "--input-union-json",
                str(context.artifacts["filled_union_json"]),
                "--input-tracked-sqlite",
                str(context.artifacts["tracked_sqlite"]),
                "--output-dir",
                str(context.stage_dir),
            ]
        )
        return StageResult({"evaluation_summary": context.stage_dir / "summary.json"})


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
