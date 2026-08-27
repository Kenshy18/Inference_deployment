"""Validate and merge one standalone model SQLite into unified schema v3."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .result_import import import_candidate_results


REQUIRED_SOURCE_TABLES = frozenset(
    {
        "metadata",
        "frames",
        "detections",
        "classification_probabilities",
        "segmentations",
        "segmentation_polygons",
        "segmentation_points",
    }
)
ROLE_TASKS = {
    "instance_segmentation": "instance_segmentation",
    "face_detection": "object_detection",
}


@dataclass(frozen=True, slots=True)
class ImportedModelSummary:
    model_execution_id: int
    role: str
    model_id: str
    runtime_model_id: str
    task: str
    backend: str
    frames: int
    detections: int
    classifications: int
    segmentations: int
    face_observations: int
    face_keypoints: int


def import_candidate_database(
    target: sqlite3.Connection,
    source_path: Path,
    *,
    role: str,
    model_id: str,
    backend: str,
) -> ImportedModelSummary:
    if role not in ROLE_TASKS:
        raise ValueError(f"unsupported inference role: {role}")
    source = source_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"model SQLite output not found: {source}")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as candidate:
        candidate.row_factory = sqlite3.Row
        _validate_candidate(candidate, source)
        metadata_rows = tuple(
            candidate.execute(
                "SELECT key, value, value_type FROM metadata ORDER BY key"
            )
        )
        metadata = {str(row["key"]): str(row["value"]) for row in metadata_rows}
        task = metadata.get("task", "")
        expected_task = ROLE_TASKS[role]
        if task != expected_task:
            raise ValueError(
                f"{role} requires task={expected_task}, got {task or '<missing>'}"
            )
        runtime_model_id = metadata.get("model_id", model_id)
        try:
            target.execute("BEGIN")
            execution_id = _insert_execution(
                target,
                role=role,
                model_id=model_id,
                runtime_model_id=runtime_model_id,
                task=task,
                backend=backend,
                metadata_rows=metadata_rows,
            )
            _merge_video_metadata(target, metadata)
            frame_ids = _merge_frames(target, candidate)
            counts = import_candidate_results(
                target,
                candidate,
                execution_id=execution_id,
                frame_ids=frame_ids,
            )
            target.commit()
        except BaseException:
            target.rollback()
            raise
    return ImportedModelSummary(
        model_execution_id=execution_id,
        role=role,
        model_id=model_id,
        runtime_model_id=runtime_model_id,
        task=task,
        backend=backend,
        frames=len(frame_ids),
        detections=counts.detections,
        classifications=counts.classifications,
        segmentations=counts.segmentations,
        face_observations=counts.face_observations,
        face_keypoints=counts.face_keypoints,
    )


def _validate_candidate(candidate: sqlite3.Connection, source: Path) -> None:
    tables = {
        str(row[0])
        for row in candidate.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = REQUIRED_SOURCE_TABLES - tables
    if missing:
        raise ValueError(
            f"model SQLite schema is incomplete; missing={sorted(missing)}"
        )
    if candidate.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError(f"model SQLite integrity check failed: {source}")


def _insert_execution(
    target: sqlite3.Connection,
    *,
    role: str,
    model_id: str,
    runtime_model_id: str,
    task: str,
    backend: str,
    metadata_rows: tuple[sqlite3.Row, ...],
) -> int:
    cursor = target.execute(
        """
        INSERT INTO model_executions(
            run_id, role, model_id, runtime_model_id, task, backend
        ) VALUES (1, ?, ?, ?, ?, ?)
        """,
        (role, model_id, runtime_model_id, task, backend),
    )
    execution_id = int(cursor.lastrowid)
    target.executemany(
        """
        INSERT INTO model_metadata(
            model_execution_id, key, value, value_type
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (
                execution_id,
                str(row["key"]),
                str(row["value"]),
                str(row["value_type"]),
            )
            for row in metadata_rows
        ],
    )
    return execution_id


def _merge_video_metadata(
    target: sqlite3.Connection, metadata: Mapping[str, str]
) -> None:
    incoming = (
        _optional_int(metadata.get("video.frames")),
        _optional_float(metadata.get("video.fps")),
        _optional_int(metadata.get("video.width")),
        _optional_int(metadata.get("video.height")),
    )
    existing = target.execute(
        """
        SELECT reported_frame_count, fps, width, height
        FROM videos WHERE id=1
        """
    ).fetchone()
    assert existing is not None
    merged = tuple(
        new if observed is None else observed
        for observed, new in zip(tuple(existing), incoming)
    )
    for label, observed, new in zip(
        ("frame_count", "fps", "width", "height"), merged, incoming
    ):
        if new is None or observed is None:
            continue
        equal = (
            math.isclose(float(observed), float(new), rel_tol=1e-6)
            if label == "fps"
            else observed == new
        )
        if not equal:
            raise ValueError(
                f"video metadata mismatch for {label}: {observed} != {new}"
            )
    target.execute(
        """
        UPDATE videos
        SET reported_frame_count=?, fps=?, width=?, height=?
        WHERE id=1
        """,
        merged,
    )


def _merge_frames(
    target: sqlite3.Connection,
    candidate: sqlite3.Connection,
) -> dict[int, int]:
    frame_ids: dict[int, int] = {}
    for row in candidate.execute(
        """
        SELECT frame_index, timestamp_sec, width, height
        FROM frames ORDER BY frame_index
        """
    ):
        frame_index = int(row["frame_index"])
        existing = target.execute(
            """
            SELECT id, timestamp_sec, width, height
            FROM frames WHERE run_id=1 AND frame_index=?
            """,
            (frame_index,),
        ).fetchone()
        if existing is None:
            cursor = target.execute(
                """
                INSERT INTO frames(
                    run_id, frame_index, timestamp_sec, width, height
                ) VALUES (1, ?, ?, ?, ?)
                """,
                (
                    frame_index,
                    float(row["timestamp_sec"]),
                    int(row["width"]),
                    int(row["height"]),
                ),
            )
            frame_ids[frame_index] = int(cursor.lastrowid)
            continue
        if (
            not math.isclose(
                float(existing["timestamp_sec"]),
                float(row["timestamp_sec"]),
                abs_tol=1e-6,
            )
            or int(existing["width"]) != int(row["width"])
            or int(existing["height"]) != int(row["height"])
        ):
            raise ValueError(f"frame metadata mismatch at index {frame_index}")
        frame_ids[frame_index] = int(existing["id"])
    return frame_ids


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


__all__ = ["ImportedModelSummary", "import_candidate_database"]
