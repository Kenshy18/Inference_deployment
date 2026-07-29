"""Pipeline stage owned by polygon gap filling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .interpolate import LinearPolygonInterpolator, fill_keyframe_gaps_sqlite


@dataclass(frozen=True)
class PolygonGapFillStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "polygon_gap_fill"
    requires: frozenset[str] = frozenset({"approximated_sqlite", "keyframes_sqlite"})
    provides: frozenset[str] = frozenset({"predictions_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        implementation = LinearPolygonInterpolator(
            minimum_points=int(self.options.get("minimum_points", 8))
        )
        output = context.stage_dir / "predictions.sqlite"
        fill_keyframe_gaps_sqlite(
            context.artifacts["keyframes_sqlite"],
            context.artifacts["approximated_sqlite"],
            output,
            interpolator=implementation,
            max_gap=(
                None
                if self.options.get("max_gap") is None
                else int(self.options["max_gap"])
            ),
        )
        return StageResult(
            {"predictions_sqlite": output},
            {
                "algorithm": implementation.name,
                "max_gap": self.options.get("max_gap"),
            },
        )
