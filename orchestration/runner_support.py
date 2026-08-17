"""Shared types, paths, and publication helpers for workflow orchestration."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .rescale_result_sqlite import VideoGeometry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CLI = (
    REPOSITORY_ROOT / "InstanceSegmentation" / "inference" / "run_inference.py"
)
POSTPROCESS_CLI = REPOSITORY_ROOT / "postprocess" / "run_pipeline.py"
PRECOMPUTE_CUTS_CLI = REPOSITORY_ROOT / "postprocess" / "precompute_cuts.py"
PACKAGE_RESULT_CLI = REPOSITORY_ROOT / "postprocess" / "package_result.py"
OVERLAY_ROOT = REPOSITORY_ROOT / "overlay"
BUNDLED_FFMPEG = OVERLAY_ROOT / ".runtime" / "ffmpeg-nvenc-btbn-8.1" / "bin" / "ffmpeg"
INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt"}


class OrchestrationError(RuntimeError):
    """Raised when a workflow stage fails or returns invalid artifacts."""


@dataclass(frozen=True)
class WorkflowArtifacts:
    inference_sqlite: Path
    tracked_sqlite: Path | None = None
    final_sqlite: Path | None = None
    result_sqlite: Path | None = None
    overlay_sqlite: Path | None = None
    legacy_final_sqlite: Path | None = None


@dataclass
class BackgroundStage:
    name: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO
    started_at_utc: str
    started: float
    waiter: threading.Thread
    completion: list[tuple[int, float, str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_output_stem(value: str) -> str:
    """Return a Windows-safe public artifact stem for a source video."""

    cleaned = re.sub(r'[\\/:*?"<>|]', "_", value).strip(" .")
    return cleaned or "video"


def _atomic_copy(source: Path, destination: Path) -> None:
    """Publish a completed file by hard link, with an atomic copy fallback."""

    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination:
        return
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_destination.with_name(
        f".{resolved_destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        try:
            os.link(resolved_source, temporary)
        except OSError:
            # Cross-device and Windows-backed destinations may not support a
            # hard link. Keep the destination atomic in that case as well.
            shutil.copy2(resolved_source, temporary)
        os.replace(temporary, resolved_destination)
    finally:
        temporary.unlink(missing_ok=True)


def _emit_phase_complete(phase: str, completed: int) -> None:
    print(
        "[phase-progress] "
        + json.dumps(
            {
                "phase": phase,
                "state": "complete",
                "completed": max(0, int(completed)),
                "total": max(0, int(completed)),
                "detail": "complete",
                "fps": None,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
