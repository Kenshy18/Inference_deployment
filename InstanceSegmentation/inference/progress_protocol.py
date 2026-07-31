"""Low-overhead structured progress events shared by inference engines."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


PROGRESS_MARKER = "[phase-progress]"
PHASE_ENVIRONMENT = "MASK_PIPELINE_PROGRESS_PHASE"
INTERVAL_ENVIRONMENT = "MASK_PIPELINE_PROGRESS_INTERVAL_SEC"


def emit_phase_progress(
    phase: str,
    *,
    state: str,
    completed: int,
    total: int | None,
    detail: str,
    fps: float | None = None,
) -> None:
    payload = {
        "phase": str(phase),
        "state": str(state),
        "completed": max(0, int(completed)),
        "total": None if total is None else max(0, int(total)),
        "detail": str(detail),
        "fps": None if fps is None else max(0.0, float(fps)),
    }
    print(
        f"{PROGRESS_MARKER} "
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}",
        flush=True,
    )


@dataclass
class InferenceProgressReporter:
    """Emit at most one event per interval plus mandatory boundary events."""

    phase: str
    total: int
    interval_seconds: float = 0.3
    _last_emit: float = 0.0
    _last_completed: int = 0

    @classmethod
    def from_environment(
        cls,
        *,
        available_frames: int,
        max_frames: int | None,
    ) -> InferenceProgressReporter | None:
        phase = os.environ.get(PHASE_ENVIRONMENT, "").strip()
        if not phase:
            return None
        total = max(0, int(available_frames))
        if max_frames is not None:
            total = min(total, max(0, int(max_frames)))
        try:
            interval = float(os.environ.get(INTERVAL_ENVIRONMENT, "0.3"))
        except ValueError:
            interval = 0.3
        return cls(
            phase=phase,
            total=total,
            interval_seconds=max(0.05, interval),
        )

    def start(self) -> None:
        self._emit(0, detail="frames", fps=None, state="running")

    def update(self, completed: int, *, fps: float | None) -> None:
        completed = min(self.total, max(self._last_completed, int(completed)))
        now = time.monotonic()
        if completed < self.total and now - self._last_emit < self.interval_seconds:
            self._last_completed = completed
            return
        self._emit(completed, detail="frames", fps=fps, state="running")

    def complete(self, completed: int, *, fps: float | None) -> None:
        final_total = max(self.total, int(completed))
        self.total = final_total
        self._emit(final_total, detail="complete", fps=fps, state="complete")

    def _emit(
        self,
        completed: int,
        *,
        detail: str,
        fps: float | None,
        state: str,
    ) -> None:
        emit_phase_progress(
            self.phase,
            state=state,
            completed=completed,
            total=self.total,
            detail=detail,
            fps=fps,
        )
        self._last_emit = time.monotonic()
        self._last_completed = completed


__all__ = [
    "INTERVAL_ENVIRONMENT",
    "InferenceProgressReporter",
    "PHASE_ENVIRONMENT",
    "PROGRESS_MARKER",
    "emit_phase_progress",
]
