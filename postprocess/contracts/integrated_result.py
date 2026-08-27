"""Validation contract for the integrated result SQLite artifact."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .result_schema import (
    INFERENCE_SCHEMA_NAME,
    RESULT_COMPATIBILITY_PROFILE,
    RESULT_CONTRACT_REVISION,
    RESULT_OWNED_VIEWS,
    RESULT_REQUIRED_TABLES,
    RESULT_SCHEMA_NAME,
    RESULT_SCHEMA_VERSION,
)


def _tables(connection: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT name
            FROM {schema}.sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
        )
    }


def _views(connection: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT name FROM {schema}.sqlite_master WHERE type='view'"
        )
    }


def _row_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _capability_row_count(
    connection: sqlite3.Connection,
    name: str,
    source_table: str,
) -> int:
    if name == "face_detection":
        if "model_executions" not in _tables(connection):
            return 0
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM detections AS d
                JOIN model_executions AS m
                  ON m.id=d.model_execution_id
                WHERE m.role='face_detection'
                """
            ).fetchone()[0]
        )
    return _row_count(connection, source_table)


def validate_integrated_result(path: Path) -> dict[str, Any]:
    """Validate the stable raw/tracked/final result contract."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = _tables(connection)
        missing = RESULT_REQUIRED_TABLES - tables
        if missing:
            raise ValueError(
                f"{source}: integrated result tables missing: {sorted(missing)}"
            )
        missing_views = set(RESULT_OWNED_VIEWS) - _views(connection)
        if missing_views:
            raise ValueError(
                f"{source}: integrated result views missing: "
                f"{sorted(missing_views)}"
            )
        inference_info = dict(connection.execute("SELECT key, value FROM schema_info"))
        if inference_info.get("schema_name") != INFERENCE_SCHEMA_NAME:
            raise ValueError(f"{source}: unexpected inference schema")
        result_info = dict(
            connection.execute("SELECT key, value FROM result_schema_info")
        )
        if (
            result_info.get("schema_name") != RESULT_SCHEMA_NAME
            or result_info.get("schema_version") != RESULT_SCHEMA_VERSION
            or result_info.get("contract_revision") != RESULT_CONTRACT_REVISION
            or result_info.get("compatibility_profile") != RESULT_COMPATIBILITY_PROFILE
        ):
            raise ValueError(f"{source}: unsupported integrated result contract")
        capability_rows = list(
            connection.execute(
                """
                SELECT name, available, row_count, source_table, details_json
                FROM result_capabilities ORDER BY name
                """
            )
        )
        expected_capabilities = {
            "raw_inference",
            "instance_segmentation",
            "face_detection",
            "rich_face_geometry",
            "tracking_assignments",
            "face_tracking",
            "final_annotations",
            "cut_detection",
            "classwise_postprocess",
            "face_privacy_masks",
            "native_polygon_keyframes",
            "native_ellipse_keyframes",
            "native_rectangle_keyframes",
        }
        actual_capabilities = {str(row[0]) for row in capability_rows}
        if actual_capabilities != expected_capabilities:
            raise ValueError(
                f"{source}: result capabilities mismatch: "
                f"{sorted(actual_capabilities)}"
            )
        for name, _available, row_count, source_table, details_json in capability_rows:
            actual_count = _capability_row_count(
                connection,
                str(name),
                str(source_table),
            )
            if int(row_count) != actual_count:
                raise ValueError(f"{source}: capability {name!r} row_count mismatch")
            try:
                details = json.loads(str(details_json))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: capability {name!r} details_json is invalid"
                ) from exc
            if not isinstance(details, dict):
                raise ValueError(
                    f"{source}: capability {name!r} details_json must be an object"
                )
        component_rows = list(
            connection.execute(
                """
                SELECT name, status, row_count, source_table, details_json
                FROM result_components ORDER BY name
                """
            )
        )
        if {str(row[0]) for row in component_rows} != expected_capabilities:
            raise ValueError(f"{source}: result components mismatch")
        for name, status, row_count, source_table, details_json in component_rows:
            if str(status) not in {
                "complete",
                "empty",
                "not_requested",
                "unsupported",
                "failed",
            }:
                raise ValueError(f"{source}: invalid component status for {name!r}")
            actual_count = _capability_row_count(
                connection,
                str(name),
                str(source_table),
            )
            if int(row_count) != actual_count:
                raise ValueError(f"{source}: component {name!r} row_count mismatch")
            if not isinstance(json.loads(str(details_json)), dict):
                raise ValueError(f"{source}: component {name!r} details must be object")
        invalid_keyframe = connection.execute(
            """
            SELECT k.id
            FROM mask_keyframes k
            JOIN mask_track_segments s ON s.id=k.segment_id
            WHERE k.frame < s.start_frame OR k.frame > s.end_frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_keyframe is not None:
            raise ValueError(
                f"{source}: keyframe {invalid_keyframe[0]} is outside its segment"
            )
        invalid_component = connection.execute(
            """
            SELECT c.id
            FROM keyframe_components c
            JOIN mask_keyframes k ON k.id=c.keyframe_id
            JOIN mask_track_segments s ON s.id=k.segment_id
            LEFT JOIN keyframe_ellipses e ON e.component_id=c.id
            LEFT JOIN keyframe_rectangles r ON r.component_id=c.id
            LEFT JOIN keyframe_polygon_rings pr ON pr.component_id=c.id
            LEFT JOIN keyframe_polygon_points pp ON pp.ring_id=pr.id
            GROUP BY c.id
            HAVING c.slot_index >= s.component_count
               OR (c.geometry_type='ellipse' AND COUNT(DISTINCT e.component_id) <> 1)
               OR (c.geometry_type='rectangle' AND COUNT(DISTINCT r.component_id) <> 1)
               OR (c.geometry_type='polygon' AND COUNT(pp.point_index) < 3)
            LIMIT 1
            """
        ).fetchone()
        if invalid_component is not None:
            raise ValueError(
                f"{source}: invalid typed keyframe component " f"{invalid_component[0]}"
            )
        uncovered_postprocess_frame = connection.execute(
            """
            SELECT p.frame, p.track_id
            FROM mask_postprocess_provenance p
            WHERE NOT EXISTS (
                SELECT 1
                FROM mask_track_segments s
                WHERE s.track_id=p.track_id
                  AND p.frame BETWEEN s.start_frame AND s.end_frame
                  AND EXISTS (
                      SELECT 1 FROM mask_keyframes k0
                      WHERE k0.segment_id=s.id AND k0.frame<=p.frame
                  )
                  AND EXISTS (
                      SELECT 1 FROM mask_keyframes k1
                      WHERE k1.segment_id=s.id AND k1.frame>=p.frame
                  )
            )
            LIMIT 1
            """
        ).fetchone()
        if uncovered_postprocess_frame is not None:
            raise ValueError(
                f"{source}: postprocess frame {uncovered_postprocess_frame[0]} "
                f"for track {uncovered_postprocess_frame[1]!r} cannot be "
                "reconstructed from editable keyframes"
            )
        dense_tables = {
            "masks",
            "tracked_masks",
            "raw_tracked_masks",
            "tracked_tracks",
        } & tables
        if dense_tables:
            raise ValueError(
                f"{source}: V3 contains duplicated dense tables: "
                f"{sorted(dense_tables)}"
            )
        annotation_state = connection.execute(
            """
            SELECT revision, authoritative_geometry, dense_cache_policy
            FROM annotation_state WHERE id=1
            """
        ).fetchone()
        if annotation_state is None or tuple(annotation_state[1:]) != (
            "mask_keyframes",
            "not_materialized",
        ):
            raise ValueError(f"{source}: invalid annotation_state")
        invalid_tracking_link = connection.execute(
            """
            SELECT a.source_detection_id
            FROM tracking_assignments a
            LEFT JOIN detections d ON d.id=a.source_detection_id
            LEFT JOIN frames f ON f.id=d.frame_id
            WHERE d.id IS NULL OR f.frame_index<>a.frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_tracking_link is not None:
            raise ValueError(
                f"{source}: invalid tracking source detection "
                f"{invalid_tracking_link[0]}"
            )
        invalid_face_tracking_link = connection.execute(
            """
            SELECT a.observation_id
            FROM face_tracking_assignments a
            LEFT JOIN face_observations fo ON fo.id=a.observation_id
            LEFT JOIN detections d ON d.id=a.anchor_detection_id
            LEFT JOIN frames f ON f.id=d.frame_id
            WHERE fo.id IS NULL OR d.id IS NULL OR f.frame_index<>a.frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_face_tracking_link is not None:
            raise ValueError(
                f"{source}: invalid face tracking observation "
                f"{invalid_face_tracking_link[0]}"
            )
        invalid_face_interpolation = connection.execute(
            """
            SELECT i.frame, i.final_track_id
            FROM face_track_interpolations i
            LEFT JOIN face_observations previous
              ON previous.id=i.previous_observation_id
            LEFT JOIN face_observations following
              ON following.id=i.next_observation_id
            WHERE previous.id IS NULL OR following.id IS NULL
               OR i.head_x2<i.head_x1 OR i.head_y2<i.head_y1
            LIMIT 1
            """
        ).fetchone()
        if invalid_face_interpolation is not None:
            raise ValueError(
                f"{source}: invalid face interpolation "
                f"{tuple(invalid_face_interpolation)}"
            )
        counts = {
            "frames": _row_count(connection, "frames"),
            "detections": _row_count(connection, "detections"),
            "segmentations": _row_count(connection, "segmentations"),
            "face_observations": _row_count(connection, "face_observations"),
            "tracking_assignments": _row_count(connection, "tracking_assignments"),
            "face_tracking_assignments": _row_count(
                connection, "face_tracking_assignments"
            ),
            "face_track_interpolations": _row_count(
                connection, "face_track_interpolations"
            ),
            "final_annotations": _row_count(connection, "mask_keyframes"),
            "cuts": _row_count(connection, "cuts"),
            "mask_segments": _row_count(connection, "mask_track_segments"),
            "mask_keyframes": _row_count(connection, "mask_keyframes"),
            "ellipse_keyframes": _row_count(connection, "keyframe_ellipses"),
            "polygon_keyframe_points": _row_count(
                connection, "keyframe_polygon_points"
            ),
            "rectangle_keyframes": _row_count(connection, "keyframe_rectangles"),
        }
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"{source}: integrity check failed: {integrity}")
        return {
            "schema_name": RESULT_SCHEMA_NAME,
            "schema_version": int(RESULT_SCHEMA_VERSION),
            "contract_revision": int(RESULT_CONTRACT_REVISION),
            "compatibility_profile": RESULT_COMPATIBILITY_PROFILE,
            "annotation_revision": int(annotation_state[0]),
            "inference_schema_version": int(inference_info["schema_version"]),
            "capabilities": {
                str(name): {
                    "available": bool(available),
                    "row_count": int(row_count),
                    "source_table": str(source_table),
                    "details": json.loads(str(details_json)),
                }
                for (
                    name,
                    available,
                    row_count,
                    source_table,
                    details_json,
                ) in capability_rows
            },
            **counts,
            "size_bytes": source.stat().st_size,
        }


__all__ = ("validate_integrated_result",)
