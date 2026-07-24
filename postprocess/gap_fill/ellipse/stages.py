"""Pipeline stage owned by ellipse gap filling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .interpolate import kffill_main


@dataclass(frozen=True)
class EllipseGapFillStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "ellipse_gap_fill"
    requires: frozenset[str] = frozenset(
        {"interpolated_union_json", "approximation_metrics_csv"}
    )
    provides: frozenset[str] = frozenset({"filled_union_json", "filled_metrics_csv"})

    def run(self, context: StageContext) -> StageResult:
        union = context.stage_dir / "interpolated_union.json"
        metrics = context.stage_dir / "interpolated_metrics.csv"
        kffill_main(
            [
                "--input-union-json",
                str(context.artifacts["interpolated_union_json"]),
                "--input-metrics-csv",
                str(context.artifacts["approximation_metrics_csv"]),
                "--output-union-json",
                str(union),
                "--output-metrics-csv",
                str(metrics),
                "--output-summary-json",
                str(context.stage_dir / "summary.json"),
                "--max-gap",
                str(self.options.get("max_gap", 30)),
            ]
        )
        return StageResult({"filled_union_json": union, "filled_metrics_csv": metrics})
