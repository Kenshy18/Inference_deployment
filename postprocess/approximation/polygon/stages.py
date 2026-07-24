"""Pipeline stages owned by polygon approximation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .rdp import OpenCvRdpApproximator, approximate_sqlite


@dataclass(frozen=True)
class RdpApproximationStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "polygon_approximation"
    requires: frozenset[str] = frozenset({"tracked_sqlite"})
    provides: frozenset[str] = frozenset({"approximated_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        implementation = OpenCvRdpApproximator(
            epsilon_ratio=float(self.options.get("epsilon_ratio", 0.01)),
            minimum_epsilon_px=float(self.options.get("minimum_epsilon_px", 0.5)),
        )
        output = context.stage_dir / "approximated.sqlite"
        approximate_sqlite(
            context.artifacts["tracked_sqlite"],
            output,
            approximator=implementation,
        )
        return StageResult(
            {"approximated_sqlite": output},
            {"algorithm": implementation.name},
        )
