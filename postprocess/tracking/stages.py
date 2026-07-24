"""Pipeline stage owned by temporal tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.stages import StageContext, StageResult

from .builder import build_tracked_sqlite


@dataclass(frozen=True)
class TrackingStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "tracking"
    requires: frozenset[str] = frozenset({"nms_jsonl", "cuts_json"})
    provides: frozenset[str] = frozenset({"tracked_sqlite"})

    def run(self, context: StageContext) -> StageResult:
        output = context.stage_dir / "tracked.sqlite"
        summary = build_tracked_sqlite(
            context.artifacts["nms_jsonl"],
            output,
            context.artifacts["cuts_json"],
            remove_short_tracks_max_frames=int(
                self.options.get("remove_short_tracks_max_frames", 10)
            ),
        )
        return StageResult({"tracked_sqlite": output}, summary)
