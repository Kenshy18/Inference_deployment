"""Lightweight cross-component artifact validation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ArtifactError(ValueError):
    """Raised when a workflow artifact does not satisfy its public contract."""


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def validate_inference_sqlite(
    path: Path,
    *,
    require_segmentation: bool,
    require_faces: bool,
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    required_tables = {
        "schema_info",
        "frames",
        "detections",
        "model_executions",
        "segmentations",
        "segmentation_polygons",
        "segmentation_points",
    }
    required_columns = {
        "schema_info": {"key", "value"},
        "frames": {
            "id",
            "run_id",
            "frame_index",
            "timestamp_sec",
            "width",
            "height",
        },
        "detections": {
            "id",
            "frame_id",
            "model_execution_id",
            "class_id",
            "class_name",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
        },
        "model_executions": {"id", "role"},
        "segmentations": {"detection_id", "encoding"},
        "segmentation_polygons": {
            "id",
            "detection_id",
            "polygon_index",
        },
        "segmentation_points": {
            "polygon_id",
            "point_index",
            "x",
            "y",
        },
    }
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        missing = required_tables - _tables(connection)
        if missing:
            raise ArtifactError(
                f"{source}: missing inference table(s): {sorted(missing)}"
            )
        missing_columns = {
            table: sorted(columns - _columns(connection, table))
            for table, columns in required_columns.items()
            if columns - _columns(connection, table)
        }
        if missing_columns:
            raise ArtifactError(
                f"{source}: inference columns missing: {missing_columns}"
            )
        info = dict(connection.execute("SELECT key, value FROM schema_info"))
        if info.get("schema_name") != "instance-segmentation-unified-inference":
            raise ArtifactError(f"{source}: unexpected inference schema")
        if str(info.get("schema_version")) != "2":
            raise ArtifactError(f"{source}: unsupported inference schema version")
        roles = {
            str(row[0])
            for row in connection.execute("SELECT role FROM model_executions")
        }
        if require_segmentation and "instance_segmentation" not in roles:
            raise ArtifactError(f"{source}: instance_segmentation role is absent")
        if require_faces and "face_detection" not in roles:
            raise ArtifactError(f"{source}: face_detection role is absent")
        frame_row = connection.execute(
            "SELECT COUNT(*), MIN(frame_index), MAX(frame_index) FROM frames"
        ).fetchone()
        if frame_row is None or int(frame_row[0]) == 0:
            raise ArtifactError(f"{source}: frames table is empty")
        segmentation_count = int(
            connection.execute("SELECT COUNT(*) FROM segmentations").fetchone()[0]
        )
        if require_segmentation and segmentation_count == 0:
            raise ArtifactError(f"{source}: no segmentation masks were produced")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ArtifactError(f"{source}: integrity check failed: {integrity}")
        return {
            "path": str(source),
            "roles": sorted(roles),
            "frames": int(frame_row[0]),
            "first_frame": int(frame_row[1]),
            "last_frame": int(frame_row[2]),
            "segmentations": segmentation_count,
        }


def validate_mask_sqlite(path: Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = _tables(connection)
        if "masks" not in tables:
            raise ArtifactError(f"{source}: masks table is absent")
        missing = {"frame", "track_id", "polygons"} - _columns(
            connection, "masks"
        )
        if missing:
            raise ArtifactError(f"{source}: masks columns missing: {sorted(missing)}")
        row = connection.execute(
            "SELECT COUNT(*), MIN(frame), MAX(frame) FROM masks"
        ).fetchone()
        assert row is not None
        for frame, track_id, polygons in connection.execute(
            "SELECT frame, track_id, polygons FROM masks"
        ):
            if int(frame) < 0 or not str(track_id):
                raise ArtifactError(f"{source}: invalid mask identity")
            decoded = json.loads(str(polygons))
            if not isinstance(decoded, list):
                raise ArtifactError(f"{source}: polygons must decode to a list")
        return {
            "path": str(source),
            "masks": int(row[0]),
            "first_frame": None if row[1] is None else int(row[1]),
            "last_frame": None if row[2] is None else int(row[2]),
        }


def read_postprocess_manifest(path: Path) -> tuple[Path, Path]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise ArtifactError(f"{manifest_path}: postprocess run is not complete")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArtifactError(f"{manifest_path}: artifacts object is absent")
    try:
        tracked = Path(str(artifacts["tracked_sqlite"])).expanduser().resolve()
        final = Path(str(artifacts["predictions_sqlite"])).expanduser().resolve()
    except KeyError as exc:
        raise ArtifactError(
            f"{manifest_path}: required postprocess artifact is absent: {exc}"
        ) from exc
    validate_mask_sqlite(tracked)
    validate_mask_sqlite(final)
    return tracked, final


__all__ = [
    "ArtifactError",
    "read_postprocess_manifest",
    "validate_inference_sqlite",
    "validate_mask_sqlite",
]
