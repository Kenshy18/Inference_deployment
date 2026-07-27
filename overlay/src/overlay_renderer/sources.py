"""Read-only adapters for inference and postprocess SQLite contracts."""

from __future__ import annotations

import itertools
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import FrameOverlay, OverlayItem, Polygon, SourceInfo


INFERENCE_SCHEMA_NAME = "instance-segmentation-unified-inference"
INFERENCE_SCHEMA_VERSION = "2"

INFERENCE_COLUMNS = {
    "schema_info": {"key", "value"},
    "frames": {"id", "frame_index", "width", "height"},
    "detections": {
        "id",
        "frame_id",
        "model_execution_id",
        "class_name",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
    },
    "model_executions": {"id", "role"},
    "segmentations": {"detection_id"},
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

MASK_COLUMNS = {
    "masks": {"frame", "track_id", "polygons"},
}


class OverlayContractError(ValueError):
    """Raised when an input SQLite file does not satisfy an overlay contract."""


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


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


def _validate_columns(
    connection: sqlite3.Connection,
    required: dict[str, set[str]],
    *,
    path: Path,
) -> None:
    available_tables = _tables(connection)
    missing_tables = set(required) - available_tables
    if missing_tables:
        raise OverlayContractError(
            f"{path}: missing table(s): {', '.join(sorted(missing_tables))}"
        )
    for table, expected_columns in required.items():
        missing = expected_columns - _columns(connection, table)
        if missing:
            raise OverlayContractError(
                f"{path}: {table} is missing column(s): "
                f"{', '.join(sorted(missing))}"
            )


def _validate_inference(connection: sqlite3.Connection, path: Path) -> None:
    _validate_columns(connection, INFERENCE_COLUMNS, path=path)
    info = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM schema_info")
    }
    if info.get("schema_name") != INFERENCE_SCHEMA_NAME:
        raise OverlayContractError(
            f"{path}: unsupported inference schema: "
            f"{info.get('schema_name', '<missing>')}"
        )
    if info.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise OverlayContractError(
            f"{path}: unsupported inference schema version: "
            f"{info.get('schema_version', '<missing>')}"
        )
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise OverlayContractError(f"{path}: SQLite integrity check failed: {integrity}")


def _require_role(
    connection: sqlite3.Connection,
    path: Path,
    role: str,
) -> None:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM model_executions WHERE role=?",
            (role,),
        ).fetchone()[0]
    )
    if count == 0:
        roles = [
            str(row[0])
            for row in connection.execute(
                "SELECT role FROM model_executions ORDER BY role"
            )
        ]
        raise OverlayContractError(
            f"{path}: required inference role {role!r} is absent; "
            f"available roles={roles}"
        )


def _video_metadata(
    connection: sqlite3.Connection,
) -> tuple[int | None, int | None, float | None]:
    if "videos" in _tables(connection):
        row = connection.execute(
            "SELECT width, height, fps FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
        if row is not None:
            return (
                None if row["width"] is None else int(row["width"]),
                None if row["height"] is None else int(row["height"]),
                None if row["fps"] is None else float(row["fps"]),
            )
    row = connection.execute(
        "SELECT MAX(width) AS width, MAX(height) AS height FROM frames"
    ).fetchone()
    return (
        None if row["width"] is None else int(row["width"]),
        None if row["height"] is None else int(row["height"]),
        None,
    )


def inspect_inference_source(path: Path, role: str) -> SourceInfo:
    """Validate a unified inference SQLite and summarize one model role."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_inference(connection, resolved)
        _require_role(connection, resolved, role)
        segmentation_join = (
            "JOIN segmentations s ON s.detection_id=d.id"
            if role == "instance_segmentation"
            else ""
        )
        row = connection.execute(
            f"""
            SELECT COUNT(DISTINCT d.id) AS item_count,
                   MIN(f.frame_index) AS first_frame,
                   MAX(f.frame_index) AS last_frame
            FROM detections d
            JOIN frames f ON f.id=d.frame_id
            JOIN model_executions me ON me.id=d.model_execution_id
            {segmentation_join}
            WHERE me.role=?
            """,
            (role,),
        ).fetchone()
        width, height, fps = _video_metadata(connection)
        return SourceInfo(
            path=resolved,
            schema=INFERENCE_SCHEMA_NAME,
            role=role,
            item_count=int(row["item_count"]),
            first_frame=(
                None if row["first_frame"] is None else int(row["first_frame"])
            ),
            last_frame=None if row["last_frame"] is None else int(row["last_frame"]),
            width=width,
            height=height,
            fps=fps,
        )
    finally:
        connection.close()


def inspect_mask_source(path: Path) -> SourceInfo:
    """Validate a postprocess mask SQLite without importing postprocess code."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_columns(connection, MASK_COLUMNS, path=resolved)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise OverlayContractError(
                f"{resolved}: SQLite integrity check failed: {integrity}"
            )
        row = connection.execute(
            """
            SELECT COUNT(*) AS item_count,
                   MIN(frame) AS first_frame,
                   MAX(frame) AS last_frame
            FROM masks
            """
        ).fetchone()
        return SourceInfo(
            path=resolved,
            schema="postprocess-mask-sqlite",
            role="mask",
            item_count=int(row["item_count"]),
            first_frame=(
                None if row["first_frame"] is None else int(row["first_frame"])
            ),
            last_frame=None if row["last_frame"] is None else int(row["last_frame"]),
        )
    finally:
        connection.close()


def _classification_columns(connection: sqlite3.Connection) -> bool:
    return (
        "classifications" in _tables(connection)
        and {"detection_id", "class_name", "score"}.issubset(
            _columns(connection, "classifications")
        )
    )


def _polygon_from_rows(rows: list[sqlite3.Row]) -> Polygon:
    return tuple((float(row["x"]), float(row["y"])) for row in rows)


def iter_raw_segmentation_frames(path: Path) -> Iterator[FrameOverlay]:
    """Stream raw inference masks grouped by frame."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_inference(connection, resolved)
        _require_role(connection, resolved, "instance_segmentation")
        has_classifications = _classification_columns(connection)
        classification_name = (
            "c.class_name AS classified_name, c.score AS classified_score"
            if has_classifications
            else "NULL AS classified_name, NULL AS classified_score"
        )
        classification_join = (
            "LEFT JOIN classifications c ON c.detection_id=d.id"
            if has_classifications
            else ""
        )
        rows = connection.execute(
            f"""
            SELECT f.frame_index,
                   d.id AS detection_id,
                   d.class_name AS detector_name,
                   d.score AS detector_score,
                   {classification_name},
                   sp.id AS polygon_id,
                   sp.polygon_index,
                   pt.point_index,
                   pt.x,
                   pt.y
            FROM detections d
            JOIN frames f ON f.id=d.frame_id
            JOIN model_executions me ON me.id=d.model_execution_id
            JOIN segmentations s ON s.detection_id=d.id
            JOIN segmentation_polygons sp ON sp.detection_id=d.id
            JOIN segmentation_points pt ON pt.polygon_id=sp.id
            {classification_join}
            WHERE me.role='instance_segmentation'
            ORDER BY f.frame_index, d.id, sp.polygon_index, pt.point_index
            """
        )

        def iter_detections() -> Iterator[tuple[int, OverlayItem]]:
            for _detection_id, detection_rows_iter in itertools.groupby(
                rows, key=lambda row: int(row["detection_id"])
            ):
                detection_rows = list(detection_rows_iter)
                first = detection_rows[0]
                polygons = tuple(
                    _polygon_from_rows(list(polygon_rows))
                    for _polygon_id, polygon_rows in itertools.groupby(
                        detection_rows, key=lambda row: int(row["polygon_id"])
                    )
                )
                classified_name = first["classified_name"]
                label = (
                    str(classified_name)
                    if classified_name not in (None, "")
                    else str(first["detector_name"])
                )
                classified_score = first["classified_score"]
                score = (
                    float(classified_score)
                    if classified_score is not None
                    else float(first["detector_score"])
                )
                detection_id = int(first["detection_id"])
                yield int(first["frame_index"]), OverlayItem(
                    identity=f"detection:{detection_id}",
                    color_key=f"raw:{label}",
                    kind="mask",
                    label=label,
                    score=score,
                    polygons=polygons,
                )

        for frame_index, grouped in itertools.groupby(
            iter_detections(), key=lambda pair: pair[0]
        ):
            yield FrameOverlay(
                frame_index=frame_index,
                items=tuple(item for _frame, item in grouped),
            )
    finally:
        connection.close()


def _decode_polygons(
    value: Any,
    *,
    path: Path,
    frame: int,
    track_id: str,
) -> tuple[Polygon, ...]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise OverlayContractError(
            f"{path}: invalid polygons JSON at frame={frame}, track_id={track_id}"
        ) from exc
    if not isinstance(decoded, list):
        raise OverlayContractError(
            f"{path}: polygons must be a list at frame={frame}, track_id={track_id}"
        )
    polygons: list[Polygon] = []
    for polygon in decoded:
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points: list[tuple[float, float]] = []
        for point in polygon:
            if not isinstance(point, list) or len(point) < 2:
                raise OverlayContractError(
                    f"{path}: invalid point at frame={frame}, track_id={track_id}"
                )
            points.append((float(point[0]), float(point[1])))
        polygons.append(tuple(points))
    return tuple(polygons)


def iter_mask_frames(path: Path) -> Iterator[FrameOverlay]:
    """Stream tracked or final postprocess masks grouped by frame."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_columns(connection, MASK_COLUMNS, path=resolved)
        columns = _columns(connection, "masks")
        label_expression = (
            "COALESCE(label, '') AS label" if "label" in columns else "'' AS label"
        )
        rows = connection.execute(
            f"""
            SELECT frame, track_id, polygons, {label_expression}
            FROM masks
            ORDER BY frame, track_id
            """
        )

        def iter_items() -> Iterator[tuple[int, OverlayItem]]:
            for row in rows:
                frame = int(row["frame"])
                track_id = str(row["track_id"])
                polygons = _decode_polygons(
                    row["polygons"],
                    path=resolved,
                    frame=frame,
                    track_id=track_id,
                )
                if not polygons:
                    continue
                yield frame, OverlayItem(
                    identity=f"track:{track_id}",
                    color_key=f"track:{track_id}",
                    kind="mask",
                    label=str(row["label"]),
                    track_id=track_id,
                    polygons=polygons,
                )

        for frame_index, grouped in itertools.groupby(
            iter_items(), key=lambda pair: pair[0]
        ):
            yield FrameOverlay(
                frame_index=frame_index,
                items=tuple(item for _frame, item in grouped),
            )
    finally:
        connection.close()


def iter_face_frames(path: Path) -> Iterator[FrameOverlay]:
    """Stream face/head detection boxes grouped by frame."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_inference(connection, resolved)
        _require_role(connection, resolved, "face_detection")
        rows = connection.execute(
            """
            SELECT f.frame_index,
                   d.id AS detection_id,
                   d.class_name,
                   d.score,
                   d.x1, d.y1, d.x2, d.y2
            FROM detections d
            JOIN frames f ON f.id=d.frame_id
            JOIN model_executions me ON me.id=d.model_execution_id
            WHERE me.role='face_detection'
            ORDER BY f.frame_index, d.id
            """
        )

        def iter_items() -> Iterator[tuple[int, OverlayItem]]:
            for row in rows:
                frame = int(row["frame_index"])
                label = str(row["class_name"])
                detection_id = int(row["detection_id"])
                yield frame, OverlayItem(
                    identity=f"face:{detection_id}",
                    color_key=f"face:{label}",
                    kind="face",
                    label=label,
                    score=float(row["score"]),
                    box=(
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ),
                )

        for frame_index, grouped in itertools.groupby(
            iter_items(), key=lambda pair: pair[0]
        ):
            yield FrameOverlay(
                frame_index=frame_index,
                items=tuple(item for _frame, item in grouped),
            )
    finally:
        connection.close()


__all__ = [
    "INFERENCE_SCHEMA_NAME",
    "INFERENCE_SCHEMA_VERSION",
    "OverlayContractError",
    "inspect_inference_source",
    "inspect_mask_source",
    "iter_face_frames",
    "iter_mask_frames",
    "iter_raw_segmentation_frames",
]
