"""Explicit orchestration for the complete experimental candidate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from contracts.detections import CutList
from preprocessing.raw_sqlite import normalize_raw_detection_sqlite
from preprocessing.score_policy import ScorePolicy, apply_score_policy_jsonl
from tracking.builder import build_tracked_sqlite

from .config import CANDIDATE, CandidateConfig
from .export import export_software_sqlite
from .nms.stage import run_nms_jsonl
from .polygon.engine import run_polygon_optimizer
from .polygon.preparation import prepare_classwise_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dimensions(path: Path) -> tuple[int, int]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(frames)")}
        frame_column = "frame_index" if "frame_index" in columns else "frame"
        row = db.execute(
            f"SELECT width,height FROM frames ORDER BY {frame_column} LIMIT 1"
        ).fetchone()
    if row is None or int(row[0] or 0) <= 0 or int(row[1] or 0) <= 0:
        raise RuntimeError(f"source dimensions are unavailable: {path}")
    return int(row[0]), int(row[1])


def _copy_scored(source: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return {
        "mode": "pre_scored_canonical_jsonl",
        "input": str(Path(source).resolve()),
        "output": str(output),
        "sha256": _sha256(output),
    }


def run_candidate(
    *,
    raw_input_sqlite: Path,
    cuts_json: Path,
    output_root: Path,
    input_video: Path,
    scored_jsonl: Path | None = None,
    score_min: float = 0.30,
    config: CandidateConfig = CANDIDATE,
) -> dict[str, object]:
    """Run the candidate from raw inference SQLite through final SQLite."""
    config.validate()
    if not 0.0 <= float(score_min) <= 1.0:
        raise ValueError("score_min must be in [0, 1]")
    raw = Path(raw_input_sqlite).expanduser().resolve()
    cuts = Path(cuts_json).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    video = Path(input_video).expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)
    if not cuts.is_file():
        raise FileNotFoundError(cuts)
    if not video.is_file():
        raise FileNotFoundError(video)
    CutList.read(cuts)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to mix a candidate run with existing artifacts: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    normalized = root / "01_preprocessing/normalized.jsonl"
    scored = root / "02_score_policy/scored.jsonl"
    if scored_jsonl is None:
        normalization = normalize_raw_detection_sqlite(raw, normalized)
        score = apply_score_policy_jsonl(
            normalized,
            scored,
            policy=ScorePolicy(default_min=float(score_min)),
        )
        score_stage: dict[str, object] = {
            "mode": "normalize_then_score",
            "normalization": normalization,
            "score": score,
            "score_min": float(score_min),
            "normalized_sha256": _sha256(normalized),
            "scored_sha256": _sha256(scored),
        }
    else:
        provided = Path(scored_jsonl).expanduser().resolve()
        if not provided.is_file():
            raise FileNotFoundError(provided)
        score_stage = _copy_scored(provided, scored)

    nms_jsonl = root / "03_nms/nms.jsonl"
    nms_trace = root / "03_nms/nms_trace.jsonl.gz"
    nms = run_nms_jsonl(scored, nms_jsonl, trace_output=nms_trace, config=config)

    tracked = root / "04_tracking/tracked.sqlite"
    tracking = build_tracked_sqlite(
        nms_jsonl,
        tracked,
        cuts,
        remove_short_tracks_max_frames=(config.tracking.remove_short_tracks_max_frames),
    )
    width, height = _dimensions(raw)
    source_root, preparation = prepare_classwise_source(
        tracked,
        root / "05_polygon_preparation",
        width=width,
        height=height,
        input_video=video,
        config=config,
    )
    optimizer = run_polygon_optimizer(
        source_root,
        root / "06_polygon_keyframes",
        config=config,
        labels=tuple(preparation["active_labels"]),
        max_tracks=0,
        force=False,
    )
    final_name = f"{raw.stem}.sqlite"
    software = export_software_sqlite(
        raw,
        tracked,
        Path(str(optimizer["phase2_root"])),
        root / "07_software_sqlite" / final_name,
        config=config,
    )
    manifest = {
        "schema_version": 1,
        "status": "experimental_production_candidate",
        "candidate": config.to_dict(),
        "privacy": "SQLite mask geometry only; video pixels were not decoded.",
        "inputs": {
            "raw_input_sqlite": str(raw),
            "raw_input_sha256": _sha256(raw),
            "cuts_json": str(cuts),
            "cuts_sha256": _sha256(cuts),
            "input_video": str(video),
        },
        "stages": {
            "score_policy": score_stage,
            "nms": nms,
            "tracking": tracking,
            "polygon_preparation": preparation,
            "polygon_optimizer": optimizer,
            "software_sqlite": software,
        },
        "artifacts": {
            "scored_jsonl": str(scored),
            "nms_jsonl": str(nms_jsonl),
            "nms_trace_jsonl": str(nms_trace),
            "tracked_sqlite": str(tracked),
            "phase2_root": str(optimizer["phase2_root"]),
            "result_sqlite": software["output_sqlite"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "software_facing_sqlite_schema_changed": False,
    }
    manifest_path = root / "candidate_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


__all__ = ("run_candidate",)
