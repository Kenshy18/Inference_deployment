"""Contract helpers for detector and tracked mask SQLite inputs."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Literal


MaskSqliteKind = Literal["raw_detection", "unified_inference", "tracked"]

RAW_MASK_COLUMNS = {
    "frame",
    "mask_id",
    "detection_index",
    "label",
    "class_name",
    "category_id",
    "score",
    "detector_score",
    "class_score",
    "bbox_xyxy",
    "polygons",
    "source_json",
}
RAW_FRAME_COLUMNS = {"frame", "time_sec", "width", "height"}
UNIFIED_INFERENCE_COLUMNS = {
    "frames": {"id", "run_id", "frame_index", "timestamp_sec", "width", "height"},
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
    "classifications": {"detection_id", "class_id", "class_name", "score"},
    "segmentations": {"detection_id", "encoding"},
    "segmentation_polygons": {
        "id",
        "detection_id",
        "polygon_index",
    },
    "segmentation_points": {"polygon_id", "point_index", "x", "y"},
    "schema_info": {"key", "value"},
}


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def detect_mask_sqlite_kind(path: Path) -> MaskSqliteKind:
    """Distinguish supported detector and tracked-mask SQLite schemas."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(str(source)) as connection:
        tables = _tables(connection)
        if "masks" in tables:
            columns = _columns(connection, "masks")
            if {"frame", "track_id", "polygons"}.issubset(columns):
                return "tracked"
            if {"frame", "mask_id", "detection_index", "polygons"}.issubset(columns):
                return "raw_detection"
        if set(UNIFIED_INFERENCE_COLUMNS).issubset(tables):
            return "unified_inference"
    raise ValueError(f"{source}: unsupported mask SQLite schema")


def _decode_json_list(value: object, *, field: str, row: str) -> list[object]:
    if value is None or not str(value).strip():
        raise ValueError(f"{row}: {field} is empty")
    decoded = json.loads(str(value))
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"{row}: {field} must be a non-empty JSON list")
    return decoded


def validate_raw_detection_sqlite(path: Path) -> None:
    """Validate the ``raw_mask_sqlite_v1`` shape produced by DINOv3."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(str(source)) as connection:
        tables = _tables(connection)
        missing_tables = {"frames", "masks"} - tables
        if missing_tables:
            raise ValueError(
                f"{source}: missing raw detector tables: {sorted(missing_tables)}"
            )
        missing_frames = RAW_FRAME_COLUMNS - _columns(connection, "frames")
        missing_masks = RAW_MASK_COLUMNS - _columns(connection, "masks")
        if missing_frames or missing_masks:
            raise ValueError(
                f"{source}: raw detector schema mismatch; "
                f"frames missing={sorted(missing_frames)}, "
                f"masks missing={sorted(missing_masks)}"
            )

        frame_count = 0
        previous_frame: int | None = None
        for frame, _time_sec, width, height in connection.execute(
            "SELECT frame, time_sec, width, height FROM frames ORDER BY frame"
        ):
            value = int(frame)
            if value < 0:
                raise ValueError(f"{source}: negative frame index {value}")
            if previous_frame is not None and value <= previous_frame:
                raise ValueError(f"{source}: frame indices must be unique")
            for name, dimension in (("width", width), ("height", height)):
                if dimension is not None and int(dimension) <= 0:
                    raise ValueError(
                        f"{source}: frame {value} has invalid {name}={dimension!r}"
                    )
            previous_frame = value
            frame_count += 1
        if frame_count == 0:
            raise ValueError(f"{source}: frames table is empty")

        orphan = connection.execute(
            """
            SELECT m.frame
            FROM masks AS m
            LEFT JOIN frames AS f ON f.frame = m.frame
            WHERE f.frame IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan is not None:
            raise ValueError(
                f"{source}: mask references missing frame {int(orphan[0])}"
            )

        for (
            frame,
            mask_id,
            detection_index,
            bbox_xyxy,
            polygons,
            score,
            detector_score,
            class_score,
        ) in connection.execute(
            """
            SELECT frame, mask_id, detection_index, bbox_xyxy, polygons,
                   score, detector_score, class_score
            FROM masks
            ORDER BY frame, detection_index, mask_id
            """
        ):
            row = f"{source}: frame={int(frame)}, mask_id={mask_id!r}"
            if not str(mask_id):
                raise ValueError(f"{row}: mask_id is empty")
            if int(detection_index) < 0:
                raise ValueError(f"{row}: detection_index must be >= 0")
            _decode_json_list(polygons, field="polygons", row=row)
            if bbox_xyxy is not None:
                bbox = _decode_json_list(bbox_xyxy, field="bbox_xyxy", row=row)
                if len(bbox) < 4:
                    raise ValueError(f"{row}: bbox_xyxy needs four values")
                if not all(math.isfinite(float(value)) for value in bbox[:4]):
                    raise ValueError(f"{row}: bbox_xyxy contains non-finite values")
            for name, value in (
                ("score", score),
                ("detector_score", detector_score),
                ("class_score", class_score),
            ):
                if value is not None and not math.isfinite(float(value)):
                    raise ValueError(f"{row}: {name} is not finite")


def validate_unified_inference_sqlite(path: Path) -> None:
    """Validate supported InstanceSegmentation unified inference schemas."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(str(source)) as connection:
        tables = _tables(connection)
        missing_tables = set(UNIFIED_INFERENCE_COLUMNS) - tables
        if missing_tables:
            raise ValueError(
                f"{source}: missing unified inference tables: "
                f"{sorted(missing_tables)}"
            )
        missing_columns = {
            table: sorted(required - _columns(connection, table))
            for table, required in UNIFIED_INFERENCE_COLUMNS.items()
            if required - _columns(connection, table)
        }
        if missing_columns:
            raise ValueError(
                f"{source}: unified inference schema mismatch: {missing_columns}"
            )
        schema = dict(connection.execute("SELECT key, value FROM schema_info"))
        if schema.get("schema_name") != "instance-segmentation-unified-inference":
            raise ValueError(
                f"{source}: unexpected schema_name={schema.get('schema_name')!r}"
            )
        if schema.get("schema_version") not in {"2", "3"}:
            raise ValueError(
                f"{source}: unsupported schema_version="
                f"{schema.get('schema_version')!r}"
            )
        frame_stats = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT run_id),
                   MIN(frame_index), MAX(frame_index)
            FROM frames
            """
        ).fetchone()
        if frame_stats is None or int(frame_stats[0]) == 0:
            raise ValueError(f"{source}: frames table is empty")
        if int(frame_stats[1]) != 1:
            raise ValueError(
                f"{source}: exactly one run is required, found {frame_stats[1]}"
            )
        if int(frame_stats[2]) < 0:
            raise ValueError(f"{source}: frame indices must be non-negative")

        orphan_checks = (
            (
                "segmentation",
                """
                SELECT s.detection_id
                FROM segmentations AS s
                LEFT JOIN detections AS d ON d.id = s.detection_id
                WHERE d.id IS NULL
                LIMIT 1
                """,
            ),
            (
                "polygon",
                """
                SELECT p.id
                FROM segmentation_polygons AS p
                LEFT JOIN segmentations AS s ON s.detection_id = p.detection_id
                WHERE s.detection_id IS NULL
                LIMIT 1
                """,
            ),
            (
                "point",
                """
                SELECT pt.polygon_id
                FROM segmentation_points AS pt
                LEFT JOIN segmentation_polygons AS p ON p.id = pt.polygon_id
                WHERE p.id IS NULL
                LIMIT 1
                """,
            ),
        )
        for name, query in orphan_checks:
            orphan = connection.execute(query).fetchone()
            if orphan is not None:
                raise ValueError(f"{source}: orphan {name} row {orphan[0]}")

        missing_polygon = connection.execute(
            """
            SELECT s.detection_id
            FROM segmentations AS s
            LEFT JOIN segmentation_polygons AS p
              ON p.detection_id = s.detection_id
            WHERE p.id IS NULL
            LIMIT 1
            """
        ).fetchone()
        if missing_polygon is not None:
            raise ValueError(
                f"{source}: segmentation {missing_polygon[0]} has no polygon"
            )
        invalid_polygon = connection.execute(
            """
            SELECT p.id
            FROM segmentation_polygons AS p
            LEFT JOIN segmentation_points AS pt ON pt.polygon_id = p.id
            GROUP BY p.id
            HAVING COUNT(pt.point_index) < 3
                OR MIN(pt.point_index) <> 0
                OR MAX(pt.point_index) <> COUNT(pt.point_index) - 1
            LIMIT 1
            """
        ).fetchone()
        if invalid_polygon is not None:
            raise ValueError(
                f"{source}: polygon {invalid_polygon[0]} has invalid points"
            )
        invalid_detection = connection.execute(
            """
            SELECT id
            FROM detections
            WHERE score IS NULL OR x1 IS NULL OR y1 IS NULL
               OR x2 IS NULL OR y2 IS NULL OR x2 < x1 OR y2 < y1
            LIMIT 1
            """
        ).fetchone()
        if invalid_detection is not None:
            raise ValueError(f"{source}: detection {invalid_detection[0]} is invalid")


def validate_detector_input_sqlite(path: Path) -> None:
    """Validate either supported untracked detector SQLite input."""

    kind = detect_mask_sqlite_kind(path)
    if kind == "raw_detection":
        validate_raw_detection_sqlite(path)
        return
    if kind == "unified_inference":
        validate_unified_inference_sqlite(path)
        return
    raise ValueError(f"{path}: tracked SQLite is not a raw detector input")
