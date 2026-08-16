"""Contract implemented by every independently replaceable pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


ProgressCallback = Callable[[str, float | None, float | None], None]


@dataclass(frozen=True)
class StageContext:
    pipeline_name: str
    stage_id: str
    output_dir: Path
    stage_dir: Path
    artifacts: Mapping[str, Path]
    progress_callback: ProgressCallback | None = None

    def report_progress(
        self,
        detail: str,
        stage_fraction: float | None = None,
        fps: float | None = None,
    ) -> None:
        """Publish optional in-stage progress without coupling stage logic."""

        if self.progress_callback is not None:
            self.progress_callback(str(detail), stage_fraction, fps)


@dataclass(frozen=True)
class StageResult:
    artifacts: Mapping[str, Path]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PostprocessStage(Protocol):
    """A feature transformation connected only through named artifacts."""

    name: str
    requires: frozenset[str]
    provides: frozenset[str]

    def run(self, context: StageContext) -> StageResult:
        """Create new artifacts without mutating input artifacts."""
