"""Pipeline stages owned by polygon keyframe selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .interval import IntervalKeyframeSelector, select_keyframes_sqlite


@dataclass(frozen=True)
class IntervalKeyframesStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "polygon_keyframes"
    requires: frozenset[str] = frozenset({"approximated_sqlite"})
    provides: frozenset[str] = frozenset({"keyframes_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        selector = IntervalKeyframeSelector(
            interval_frames=int(self.options.get("interval_frames", 3))
        )
        output = context.stage_dir / "keyframes.sqlite"
        select_keyframes_sqlite(
            context.artifacts["approximated_sqlite"],
            output,
            selector=selector,
        )
        return StageResult(
            {"keyframes_sqlite": output},
            {
                "algorithm": selector.name,
                "interval_frames": selector.interval_frames,
            },
        )
