"""Normalize supported detector SQLite schemas to canonical JSONL."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from contracts.detector_sqlite import (
    detect_mask_sqlite_kind,
    validate_detector_input_sqlite,
)
from contracts.detections import dumps_json_line, loads_json

from .normalization import normalize_frame_record


def _optional_json(value: object) -> object | None:
    if value is None or not str(value).strip():
        return None
    return loads_json(str(value))


def _iter_masks(connection: sqlite3.Connection) -> sqlite3.Cursor:
    return connection.execute(
        """
        SELECT frame, mask_id, detection_index, label, class_name, category_id,
               score, detector_score, class_score, bbox_xyxy, polygons,
               source_json
        FROM masks
        ORDER BY frame, detection_index, mask_id
        """
    )


def _detection_from_row(row: sqlite3.Row) -> dict[str, Any]:
    source = _optional_json(row["source_json"])
    detection = dict(source) if isinstance(source, dict) else {}
    for name in (
        "label",
        "class_name",
        "category_id",
        "score",
        "detector_score",
        "class_score",
    ):
        if row[name] is not None:
            detection[name] = row[name]
    bbox = _optional_json(row["bbox_xyxy"])
    if bbox is not None:
        detection["bbox_xyxy"] = bbox
    polygons = _optional_json(row["polygons"])
    detection["polygons"] = polygons
    detection["segmentation"] = polygons
    return detection


def _iter_unified_detections(
    connection: sqlite3.Connection,
):
    point_rows = iter(
        connection.execute(
            """
            SELECT f.frame_index, p.detection_id, p.polygon_index,
                   pt.point_index, pt.x, pt.y
            FROM segmentation_polygons AS p
            JOIN segmentations AS s ON s.detection_id = p.detection_id
            JOIN detections AS d ON d.id = s.detection_id
            JOIN frames AS f ON f.id = d.frame_id
            JOIN segmentation_points AS pt ON pt.polygon_id = p.id
            ORDER BY f.frame_index, p.detection_id,
                     p.polygon_index, pt.point_index
            """
        )
    )
    current_point = next(point_rows, None)
    detections = connection.execute(
        """
        SELECT f.frame_index, d.id AS detection_id,
               d.class_id AS detector_class_id,
               d.class_name AS detector_class_name,
               d.score AS detector_score,
               d.x1, d.y1, d.x2, d.y2,
               c.class_id AS classifier_class_id,
               c.class_name AS classifier_class_name,
               c.score AS classifier_score
        FROM detections AS d
        JOIN frames AS f ON f.id = d.frame_id
        JOIN segmentations AS s ON s.detection_id = d.id
        LEFT JOIN classifications AS c ON c.detection_id = d.id
        ORDER BY f.frame_index, d.id
        """
    )
    for row in detections:
        detection_id = int(row["detection_id"])
        polygons: list[list[float]] = []
        polygon: list[float] = []
        polygon_index: int | None = None
        while (
            current_point is not None
            and int(current_point["detection_id"]) == detection_id
        ):
            next_polygon_index = int(current_point["polygon_index"])
            if polygon_index is not None and next_polygon_index != polygon_index:
                polygons.append(polygon)
                polygon = []
            polygon_index = next_polygon_index
            polygon.extend([float(current_point["x"]), float(current_point["y"])])
            current_point = next(point_rows, None)
        if polygon:
            polygons.append(polygon)
        classifier_name = row["classifier_class_name"]
        classifier_id = row["classifier_class_id"]
        class_name = (
            str(classifier_name)
            if classifier_name is not None
            else str(row["detector_class_name"])
        )
        category_id = (
            int(classifier_id)
            if classifier_id is not None
            else int(row["detector_class_id"])
        )
        x1, y1, x2, y2 = (
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        )
        detection: dict[str, Any] = {
            "source_detection_id": detection_id,
            "label": class_name,
            "class_name": class_name,
            "category_id": category_id,
            "score": float(row["detector_score"]),
            "detector_score": float(row["detector_score"]),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "polygons": polygons,
            "segmentation": polygons,
        }
        if row["classifier_score"] is not None:
            detection["class_score"] = float(row["classifier_score"])
        yield int(row["frame_index"]), detection
    if current_point is not None:
        raise ValueError(
            "segmentation point references a missing or unordered detection"
        )


def _normalize_unified_inference_sqlite(
    connection: sqlite3.Connection,
    handle,
) -> dict[str, int | str]:
    detections_iter = iter(_iter_unified_detections(connection))
    current = next(detections_iter, None)
    frames = detections = empty_frames = polygons = points = 0
    for frame_row in connection.execute(
        """
        SELECT frame_index, timestamp_sec, width, height
        FROM frames
        ORDER BY frame_index
        """
    ):
        frame_index = int(frame_row["frame_index"])
        raw_detections: list[dict[str, Any]] = []
        while current is not None and current[0] == frame_index:
            raw_detections.append(current[1])
            polygons += len(current[1]["polygons"])
            points += sum(len(polygon) // 2 for polygon in current[1]["polygons"])
            current = next(detections_iter, None)
        record = normalize_frame_record(
            {
                "frame_index": frame_index,
                "time_sec": frame_row["timestamp_sec"],
                "width": frame_row["width"],
                "height": frame_row["height"],
                "detections": raw_detections,
            }
        )
        canonical = {
            "frame_index": int(record["frame_index"]),
            "time_sec": record["time_sec"],
            "width": record["width"],
            "height": record["height"],
            "detections": record["detections"],
        }
        handle.write(dumps_json_line(canonical))
        frames += 1
        detections += len(raw_detections)
        empty_frames += int(not raw_detections)
    if current is not None:
        raise ValueError(f"detection references missing frame {int(current[0])}")
    schema = dict(connection.execute("SELECT key, value FROM schema_info"))
    return {
        "frames": frames,
        "detections": detections,
        "empty_frames": empty_frames,
        "polygons": polygons,
        "points": points,
        "input_schema": (
            "instance-segmentation-unified-inference-v"
            f"{schema.get('schema_version', 'unknown')}"
        ),
    }


def normalize_raw_detection_sqlite(
    input_path: Path,
    output_path: Path,
) -> dict[str, int | str]:
    """Stream supported detector SQLite rows into canonical JSONL."""

    source = Path(input_path)
    output = Path(output_path)
    validate_detector_input_sqlite(source)
    kind = detect_mask_sqlite_kind(source)
    if source.resolve() == output.resolve():
        raise ValueError("normalization output must differ from input")
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = detections = empty_frames = 0
    with sqlite3.connect(str(source)) as connection, output.open("wb") as handle:
        connection.row_factory = sqlite3.Row
        if kind == "unified_inference":
            return _normalize_unified_inference_sqlite(connection, handle)
        masks = iter(_iter_masks(connection))
        current = next(masks, None)
        for frame_row in connection.execute(
            "SELECT frame, time_sec, width, height FROM frames ORDER BY frame"
        ):
            frame_index = int(frame_row["frame"])
            raw_detections: list[dict[str, Any]] = []
            while current is not None and int(current["frame"]) == frame_index:
                raw_detections.append(_detection_from_row(current))
                current = next(masks, None)
            record: dict[str, Any] = {
                "frame_index": frame_index,
                "detections": raw_detections,
            }
            for name in ("time_sec", "width", "height"):
                if frame_row[name] is not None:
                    record[name] = frame_row[name]
            normalized = normalize_frame_record(record)
            canonical: dict[str, Any] = {
                "frame_index": int(normalized["frame_index"]),
                "detections": normalized["detections"],
            }
            for name in ("time_sec", "width", "height"):
                if name in normalized:
                    canonical[name] = normalized[name]
            handle.write(dumps_json_line(canonical))
            frames += 1
            detections += len(raw_detections)
            empty_frames += int(not raw_detections)
        if current is not None:
            raise ValueError(
                f"{source}: mask references missing frame {int(current['frame'])}"
            )
    return {
        "frames": frames,
        "detections": detections,
        "empty_frames": empty_frames,
        "input_schema": "raw_mask_sqlite_v1",
    }
