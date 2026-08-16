"""Structured progress events for the postprocess stage graph."""

from __future__ import annotations

import json
import math
import os
import threading
import time


class StageGraphProgress:
    """Report exact stage completion plus a low-cost in-stage estimate."""

    def __init__(
        self,
        total_units: int,
        *,
        interval_seconds: float | None = None,
    ) -> None:
        self.total_units = max(1, int(total_units))
        if interval_seconds is None:
            try:
                interval_seconds = float(
                    os.environ.get(
                        "MASK_PIPELINE_PROGRESS_INTERVAL_SEC",
                        "0.3",
                    )
                )
            except ValueError:
                interval_seconds = 0.3
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.completed = 0
        self.detail = "preparing"
        self._active_started: float | None = None
        self._active_fraction: float | None = None
        self._active_fps: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._emit(state="running")
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="postprocess-progress",
            daemon=True,
        )
        self._thread.start()

    def begin_stage(self, completed: int, detail: str) -> None:
        with self._lock:
            self.completed = min(
                self.total_units,
                max(self.completed, int(completed)),
            )
            self.detail = str(detail)
            self._active_started = time.monotonic()
            self._active_fraction = None
            self._active_fps = None
        self._emit(state="running")

    def activity(
        self,
        detail: str,
        stage_fraction: float | None = None,
        fps: float | None = None,
    ) -> None:
        with self._lock:
            self.detail = str(detail)
            if stage_fraction is not None:
                fraction = min(1.0, max(0.0, float(stage_fraction)))
                self._active_fraction = max(self._active_fraction or 0.0, fraction)
            if fps is not None:
                self._active_fps = max(0.0, float(fps))
        self._emit(state="running")

    def finish_stage(self, completed: int, detail: str) -> None:
        with self._lock:
            self.completed = min(
                self.total_units,
                max(self.completed, int(completed)),
            )
            self.detail = str(detail)
            self._active_started = None
            self._active_fraction = None
            self._active_fps = None
        self._emit(state="running")

    def update(self, completed: int, detail: str) -> None:
        """Backward-compatible exact update for callers outside PipelineRunner."""

        self.finish_stage(completed, detail)

    def complete(self) -> None:
        with self._lock:
            self.completed = self.total_units
            self.detail = "complete"
            self._active_started = None
            self._active_fraction = None
            self._active_fps = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._emit(state="complete")

    def fail(self, detail: str) -> None:
        with self._lock:
            self.detail = str(detail)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._emit(state="failed")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._emit(state="running")

    def _emit(self, *, state: str) -> None:
        with self._lock:
            exact_progress = self.completed / self.total_units
            display_progress = exact_progress
            estimated = False
            active_elapsed_seconds: float | None = None
            if state == "running" and self._active_started is not None:
                elapsed = max(0.0, time.monotonic() - self._active_started)
                active_elapsed_seconds = elapsed
                # Prefer exact stage-owned progress. Fall back to a bounded
                # elapsed-time estimate for stages that cannot report units.
                stage_fraction = self._active_fraction
                if stage_fraction is None:
                    stage_fraction = 0.94 * (1.0 - math.exp(-elapsed / 8.0))
                else:
                    estimated = False
                display_progress = min(
                    1.0,
                    (self.completed + stage_fraction) / self.total_units,
                )
                if self._active_fraction is None:
                    estimated = elapsed > 0.0
            payload = {
                "phase": "postprocess",
                "state": state,
                "completed": self.completed,
                "total": self.total_units,
                "display_progress": display_progress,
                "stage_progress": self._active_fraction,
                "estimated": estimated,
                "detail": self.detail,
                "fps": self._active_fps,
                "active_elapsed_seconds": active_elapsed_seconds,
            }
        print(
            "[phase-progress] "
            + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )


__all__ = ["StageGraphProgress"]
