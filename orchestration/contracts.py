"""Lightweight cross-component artifact validation."""

from __future__ import annotations

import json
import math
import sqlite3
import zlib
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


def _views(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def validate_inference_sqlite(
    path: Path,
    *,
    require_segmentation: bool,
    require_faces: bool,
    expected_face_model: str | None = None,
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
        "model_executions": {"id", "role", "model_id"},
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
        schema_version = str(info.get("schema_version"))
        if schema_version not in {"2", "3"}:
            raise ArtifactError(f"{source}: unsupported inference schema version")
        executions = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT role, model_id FROM model_executions")
        }
        roles = set(executions)
        if require_segmentation and "instance_segmentation" not in roles:
            raise ArtifactError(f"{source}: instance_segmentation role is absent")
        if require_faces and "face_detection" not in roles:
            raise ArtifactError(f"{source}: face_detection role is absent")
        if (
            require_faces
            and expected_face_model is not None
            and executions.get("face_detection") != expected_face_model
        ):
            raise ArtifactError(
                f"{source}: expected face model {expected_face_model!r}, got "
                f"{executions.get('face_detection')!r}"
            )
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
        face_observations = 0
        face_keypoints = 0
        if executions.get("face_detection") == "face_dino_v2":
            if schema_version != "3":
                raise ArtifactError(
                    f"{source}: face_dino_v2 requires inference schema version 3"
                )
            face_observations, face_keypoints = _validate_rich_faces(
                connection,
                source,
            )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ArtifactError(f"{source}: integrity check failed: {integrity}")
        return {
            "path": str(source),
            "schema_version": int(schema_version),
            "roles": sorted(roles),
            "frames": int(frame_row[0]),
            "first_frame": int(frame_row[1]),
            "last_frame": int(frame_row[2]),
            "segmentations": segmentation_count,
            "face_observations": face_observations,
            "face_keypoints": face_keypoints,
        }


def _validate_rich_faces(
    connection: sqlite3.Connection,
    source: Path,
) -> tuple[int, int]:
    required = {
        "detections": {"group_id"},
        "face_observations": {
            "id",
            "anchor_detection_id",
            "head_detection_id",
            "face_detection_id",
            "face_score",
            "face_present",
            "geometry_type",
            "ellipse_cx",
            "ellipse_cy",
            "ellipse_major_radius",
            "ellipse_minor_radius",
            "ellipse_theta_radians",
        },
        "face_keypoints": {
            "observation_id",
            "point_index",
            "class_id",
            "class_name",
            "x",
            "y",
            "state",
            "state_name",
            "confidence",
            "valid",
        },
        "face_masks": {
            "observation_id",
            "encoding",
            "width",
            "height",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "data",
        },
        "face_keypoint_class_probabilities": {
            "observation_id",
            "point_index",
            "class_index",
            "probability",
        },
        "face_keypoint_state_probabilities": {
            "observation_id",
            "point_index",
            "state_index",
            "probability",
        },
    }
    tables = _tables(connection)
    missing_tables = set(required) - tables
    if missing_tables:
        raise ArtifactError(
            f"{source}: missing rich face table(s): {sorted(missing_tables)}"
        )
    missing_columns = {
        table: sorted(columns - _columns(connection, table))
        for table, columns in required.items()
        if columns - _columns(connection, table)
    }
    if missing_columns:
        raise ArtifactError(f"{source}: rich face columns missing: {missing_columns}")
    invalid_observation = connection.execute(
        """
        SELECT fo.id
        FROM face_observations fo
        JOIN detections anchor ON anchor.id=fo.anchor_detection_id
        JOIN model_executions me ON me.id=anchor.model_execution_id
        LEFT JOIN detections head ON head.id=fo.head_detection_id
        LEFT JOIN detections face ON face.id=fo.face_detection_id
        WHERE me.role <> 'face_detection'
           OR me.model_id <> 'face_dino_v2'
           OR LOWER(anchor.class_name) NOT IN ('head', 'face')
           OR anchor.group_id IS NULL
           OR fo.face_present NOT IN (0, 1)
           OR fo.face_score < 0 OR fo.face_score > 1
           OR anchor.id NOT IN (
                COALESCE(head.id, -1),
                COALESCE(face.id, -1)
           )
           OR (head.id IS NOT NULL AND (
                head.class_name <> 'Head'
                OR head.frame_id <> anchor.frame_id
                OR head.model_execution_id <> anchor.model_execution_id
                OR head.group_id IS NULL
                OR head.group_id <> anchor.group_id
           ))
           OR (face.id IS NOT NULL AND (
                face.class_name <> 'Face'
                OR face.frame_id <> anchor.frame_id
                OR face.model_execution_id <> anchor.model_execution_id
                OR face.group_id IS NULL
                OR face.group_id <> anchor.group_id
                OR ABS(face.score - fo.face_score) > 0.000001
           ))
           OR (fo.face_present = 1 AND (
                fo.geometry_type <> 'ellipse'
                OR fo.ellipse_cx IS NULL OR fo.ellipse_cy IS NULL
                OR fo.ellipse_major_radius IS NULL
                OR fo.ellipse_minor_radius IS NULL
                OR fo.ellipse_theta_radians IS NULL
                OR fo.ellipse_major_radius < 0
                OR fo.ellipse_minor_radius < 0
           ))
        LIMIT 1
        """
    ).fetchone()
    if invalid_observation is not None:
        raise ArtifactError(
            f"{source}: invalid rich face observation {invalid_observation[0]}"
        )
    unlinked_detection = connection.execute(
        """
        SELECT d.id
        FROM detections d
        JOIN model_executions me ON me.id=d.model_execution_id
        LEFT JOIN face_observations fo
          ON fo.head_detection_id=d.id OR fo.face_detection_id=d.id
        WHERE me.model_id='face_dino_v2'
          AND LOWER(d.class_name) IN ('head', 'face')
          AND fo.id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if unlinked_detection is not None:
        raise ArtifactError(
            f"{source}: face_dino_v2 detection {unlinked_detection[0]} "
            "has no rich face observation"
        )
    invalid_keypoints = connection.execute(
        """
        SELECT observation_id
        FROM face_keypoints
        GROUP BY observation_id
        HAVING COUNT(*) <> 5
           OR MIN(point_index) <> 0
           OR MAX(point_index) <> 4
           OR MIN(class_id) < 0 OR MAX(class_id) > 3
           OR MIN(state) < 0 OR MAX(state) > 2
           OR MIN(confidence) < 0 OR MAX(confidence) > 1
           OR MIN(valid) < 0 OR MAX(valid) > 1
        LIMIT 1
        """
    ).fetchone()
    if invalid_keypoints is not None:
        raise ArtifactError(
            f"{source}: invalid face keypoints for observation "
            f"{invalid_keypoints[0]}"
        )
    invalid_class_probabilities = connection.execute(
        """
        SELECT observation_id, point_index
        FROM face_keypoint_class_probabilities
        GROUP BY observation_id, point_index
        HAVING COUNT(*) <> 4
           OR MIN(class_index) <> 0 OR MAX(class_index) <> 3
           OR MIN(probability) < 0 OR MAX(probability) > 1
        LIMIT 1
        """
    ).fetchone()
    if invalid_class_probabilities is not None:
        raise ArtifactError(f"{source}: invalid face keypoint class probabilities")
    invalid_state_probabilities = connection.execute(
        """
        SELECT observation_id, point_index
        FROM face_keypoint_state_probabilities
        GROUP BY observation_id, point_index
        HAVING COUNT(*) <> 2
           OR MIN(state_index) <> 1 OR MAX(state_index) <> 2
           OR MIN(probability) < 0 OR MAX(probability) > 1
        LIMIT 1
        """
    ).fetchone()
    if invalid_state_probabilities is not None:
        raise ArtifactError(f"{source}: invalid face keypoint state probabilities")
    observation_count = int(
        connection.execute("SELECT COUNT(*) FROM face_observations").fetchone()[0]
    )
    keypoint_count = int(
        connection.execute("SELECT COUNT(*) FROM face_keypoints").fetchone()[0]
    )
    if keypoint_count != observation_count * 5:
        raise ArtifactError(
            f"{source}: each face observation must have exactly five keypoints"
        )
    class_probability_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM face_keypoint_class_probabilities"
        ).fetchone()[0]
    )
    state_probability_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM face_keypoint_state_probabilities"
        ).fetchone()[0]
    )
    if class_probability_count != keypoint_count * 4:
        raise ArtifactError(
            f"{source}: each face keypoint must have four class probabilities"
        )
    if state_probability_count != keypoint_count * 2:
        raise ArtifactError(
            f"{source}: each face keypoint must have two state probabilities"
        )
    face_present_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM face_observations WHERE face_present=1"
        ).fetchone()[0]
    )
    invalid_mask_link = connection.execute(
        """
        SELECT fo.id
        FROM face_observations fo
        LEFT JOIN face_masks fm ON fm.observation_id=fo.id
        WHERE (fo.face_present=1 AND fm.observation_id IS NULL)
           OR (fo.face_present=0 AND fm.observation_id IS NOT NULL)
        LIMIT 1
        """
    ).fetchone()
    if invalid_mask_link is not None:
        raise ArtifactError(
            f"{source}: face mask presence mismatch for observation "
            f"{invalid_mask_link[0]}"
        )
    mask_count = 0
    for mask in connection.execute(
        """
        SELECT observation_id, encoding, width, height,
               box_x1, box_y1, box_x2, box_y2, data
        FROM face_masks
        """
    ):
        (
            observation_id,
            encoding,
            width,
            height,
            box_x1,
            box_y1,
            box_x2,
            box_y2,
            data,
        ) = mask
        if (
            str(encoding) != "zlib-u8-probability-v1"
            or int(width) != 64
            or int(height) != 64
            or float(box_x2) < float(box_x1)
            or float(box_y2) < float(box_y1)
        ):
            raise ArtifactError(
                f"{source}: invalid face mask for observation {observation_id}"
            )
        try:
            decoded = zlib.decompress(bytes(data))
        except zlib.error as exc:
            raise ArtifactError(
                f"{source}: corrupt face mask for observation {observation_id}"
            ) from exc
        if len(decoded) != int(width) * int(height):
            raise ArtifactError(
                f"{source}: face mask size mismatch for observation "
                f"{observation_id}"
            )
        mask_count += 1
    if mask_count != face_present_count:
        raise ArtifactError(
            f"{source}: each present face observation must have one mask"
        )
    return observation_count, keypoint_count


def validate_mask_sqlite(path: Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = _tables(connection)
        if "masks" not in tables:
            raise ArtifactError(f"{source}: masks table is absent")
        missing = {"frame", "track_id", "polygons"} - _columns(connection, "masks")
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
        cut_count = 0
        if "cuts" in tables:
            if "frame" not in _columns(connection, "cuts"):
                raise ArtifactError(f"{source}: cuts.frame is absent")
            for (cut_frame,) in connection.execute(
                "SELECT frame FROM cuts ORDER BY frame"
            ):
                if cut_frame is None or int(cut_frame) < 0:
                    raise ArtifactError(f"{source}: invalid cut frame {cut_frame!r}")
                cut_count += 1

        cut_method: str | None = None
        if "cut_detection_metadata" in tables:
            if "cuts" not in tables:
                raise ArtifactError(f"{source}: cut metadata requires cuts")
            required_metadata_columns = {
                "id",
                "schema_version",
                "method",
                "elapsed_seconds",
                "cut_count",
                "frame_semantics",
            }
            missing_metadata_columns = required_metadata_columns - _columns(
                connection, "cut_detection_metadata"
            )
            if missing_metadata_columns:
                raise ArtifactError(
                    f"{source}: cut metadata columns missing: "
                    f"{sorted(missing_metadata_columns)}"
                )
            metadata = connection.execute(
                """
                SELECT id, schema_version, method, elapsed_seconds, cut_count,
                       frame_semantics
                FROM cut_detection_metadata
                """
            ).fetchall()
            stable_empty_cut_metadata = (
                not metadata
                and "result_capabilities" in tables
                and connection.execute(
                    """
                    SELECT available FROM result_capabilities
                    WHERE name='cut_detection'
                    """
                ).fetchone()
                == (0,)
            )
            if stable_empty_cut_metadata:
                metadata = []
            elif len(metadata) != 1:
                raise ArtifactError(
                    f"{source}: cut metadata must contain exactly one row"
                )
            if metadata:
                (
                    metadata_id,
                    schema_version,
                    method,
                    elapsed_seconds,
                    metadata_cut_count,
                    frame_semantics,
                ) = metadata[0]
                if int(metadata_id) != 1 or int(schema_version) != 1:
                    raise ArtifactError(f"{source}: unsupported cut metadata contract")
                cut_method = str(method)
                elapsed_value = float(elapsed_seconds)
                if (
                    not cut_method
                    or not math.isfinite(elapsed_value)
                    or elapsed_value < 0
                    or int(metadata_cut_count) != cut_count
                    or str(frame_semantics) != "first_frame_of_new_scene"
                ):
                    raise ArtifactError(
                        f"{source}: invalid or inconsistent cut metadata"
                    )
        return {
            "path": str(source),
            "masks": int(row[0]),
            "first_frame": None if row[1] is None else int(row[1]),
            "last_frame": None if row[2] is None else int(row[2]),
            "cuts": cut_count,
            "cut_detection_method": cut_method,
        }


def validate_result_sqlite(
    path: Path,
    *,
    require_segmentation: bool,
    require_faces: bool,
    expected_face_model: str | None = None,
) -> dict[str, object]:
    """Validate the stable public result surface used by downstream software."""

    source = Path(path).expanduser().resolve()
    inference = validate_inference_sqlite(
        source,
        require_segmentation=require_segmentation,
        require_faces=require_faces,
        expected_face_model=expected_face_model,
    )
    required_tables = {
        "result_schema_info",
        "result_capabilities",
        "result_components",
        "video_streams",
        "processing_runs",
        "processing_stage_runs",
        "face_observations",
        "face_keypoints",
        "face_masks",
        "face_keypoint_class_probabilities",
        "face_keypoint_state_probabilities",
        "annotation_state",
        "tracking_assignments",
        "tracks",
        "cuts",
        "cut_detection_metadata",
        "raw_tracks",
        "class_postprocess_policies",
        "mask_postprocess_provenance",
        "mask_provenance",
        "mask_track_segments",
        "mask_keyframes",
        "keyframe_components",
        "keyframe_ellipses",
        "keyframe_rectangles",
        "keyframe_polygon_rings",
        "keyframe_polygon_points",
        "mask_geometry_provenance",
    }
    expected_capabilities = {
        "raw_inference": "frames",
        "instance_segmentation": "segmentations",
        "face_detection": "detections",
        "rich_face_geometry": "face_observations",
        "tracking_assignments": "tracking_assignments",
        "final_annotations": "mask_keyframes",
        "cut_detection": "cuts",
        "classwise_postprocess": "mask_postprocess_provenance",
        "face_privacy_masks": "mask_provenance",
        "native_polygon_keyframes": "keyframe_polygon_points",
        "native_ellipse_keyframes": "keyframe_ellipses",
        "native_rectangle_keyframes": "keyframe_rectangles",
    }
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = _tables(connection)
        missing = required_tables - tables
        if missing:
            raise ArtifactError(
                f"{source}: stable result table(s) missing: {sorted(missing)}"
            )
        required_views = {
            "editable_keyframe_components",
            "editable_polygon_vertices",
        }
        missing_views = required_views - _views(connection)
        if missing_views:
            raise ArtifactError(
                f"{source}: stable result view(s) missing: " f"{sorted(missing_views)}"
            )
        info = dict(connection.execute("SELECT key, value FROM result_schema_info"))
        if (
            info.get("schema_name") != "video-mask-integrated-result"
            or str(info.get("schema_version")) != "3"
            or str(info.get("contract_revision")) != "4"
            or info.get("compatibility_profile") != "keyframe-primary-v3"
            or info.get("missing_components") != "capability_rows"
            or info.get("final_data") != "mask_keyframes"
            or info.get("materialized_dense_masks") != "none"
        ):
            raise ArtifactError(f"{source}: unsupported stable result contract")
        rows = list(
            connection.execute(
                """
                SELECT name, available, row_count, source_table, details_json
                FROM result_capabilities ORDER BY name
                """
            )
        )
        names = {str(row[0]) for row in rows}
        if names != set(expected_capabilities):
            raise ArtifactError(
                f"{source}: result capabilities mismatch: {sorted(names)}"
            )
        capabilities: dict[str, object] = {}
        for name, available, row_count, source_table, details_json in rows:
            capability_name = str(name)
            expected_source = expected_capabilities[capability_name]
            if str(source_table) != expected_source:
                raise ArtifactError(
                    f"{source}: capability {name!r} source_table mismatch"
                )
            if capability_name == "face_detection":
                actual_count = (
                    int(
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
                    if "model_executions" in tables
                    else 0
                )
            else:
                actual_count = int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{expected_source}"'
                    ).fetchone()[0]
                )
            if int(row_count) != actual_count:
                raise ArtifactError(f"{source}: capability {name!r} row_count mismatch")
            try:
                details = json.loads(str(details_json))
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"{source}: capability {name!r} has invalid details_json"
                ) from exc
            if not isinstance(details, dict):
                raise ArtifactError(
                    f"{source}: capability {name!r} details must be an object"
                )
            capabilities[str(name)] = {
                "available": bool(available),
                "row_count": int(row_count),
                "source_table": str(source_table),
                "details": details,
            }
        component_rows = list(
            connection.execute(
                """
                SELECT name, status, row_count, source_table, details_json
                FROM result_components ORDER BY name
                """
            )
        )
        if {str(row[0]) for row in component_rows} != set(expected_capabilities):
            raise ArtifactError(f"{source}: result components mismatch")
        components: dict[str, object] = {}
        for name, status, row_count, source_table, details_json in component_rows:
            capability_name = str(name)
            if str(source_table) != expected_capabilities[capability_name]:
                raise ArtifactError(
                    f"{source}: component {name!r} source_table mismatch"
                )
            try:
                details = json.loads(str(details_json))
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"{source}: component {name!r} has invalid details_json"
                ) from exc
            if str(status) not in {
                "complete",
                "empty",
                "not_requested",
                "unsupported",
                "failed",
            }:
                raise ArtifactError(f"{source}: component {name!r} has invalid status")
            if int(row_count) != int(capabilities[capability_name]["row_count"]):
                raise ArtifactError(f"{source}: component {name!r} row_count mismatch")
            components[capability_name] = {
                "status": str(status),
                "row_count": int(row_count),
                "source_table": str(source_table),
                "details": details,
            }
        forbidden_dense = {
            "masks",
            "tracked_masks",
            "raw_tracked_masks",
            "tracked_tracks",
        } & tables
        if forbidden_dense:
            raise ArtifactError(
                f"{source}: V3 contains dense/duplicated tables: "
                f"{sorted(forbidden_dense)}"
            )
        state = connection.execute(
            """
            SELECT revision, authoritative_geometry, dense_cache_policy
            FROM annotation_state WHERE id=1
            """
        ).fetchone()
        if state is None or tuple(state[1:]) != (
            "mask_keyframes",
            "not_materialized",
        ):
            raise ArtifactError(f"{source}: invalid annotation_state")
        postprocess = {
            "mask_segments": int(
                connection.execute(
                    "SELECT COUNT(*) FROM mask_track_segments"
                ).fetchone()[0]
            ),
            "mask_keyframes": int(
                connection.execute(
                    "SELECT COUNT(*) FROM mask_keyframes"
                ).fetchone()[0]
            ),
            "tracking_assignments": int(
                connection.execute(
                    "SELECT COUNT(*) FROM tracking_assignments"
                ).fetchone()[0]
            ),
            "cuts": int(
                connection.execute("SELECT COUNT(*) FROM cuts").fetchone()[0]
            ),
            "annotation_revision": int(state[0]),
        }
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ArtifactError(f"{source}: integrity check failed: {integrity}")
    return {
        "path": str(source),
        "schema_name": "video-mask-integrated-result",
        "schema_version": 3,
        "contract_revision": 4,
        "compatibility_profile": "keyframe-primary-v3",
        "inference": inference,
        "postprocess": postprocess,
        "capabilities": capabilities,
        "components": components,
    }


def validate_legacy_mask_sqlite(path: Path) -> dict[str, object]:
    """Validate the exact former Dinov3_postprocess final SQLite shape."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_columns = {
        "masks": [
            "frame",
            "track_id",
            "polygons",
            "shape_type",
            "dilate_px",
            "feather_px",
            "mosaic_block",
            "mosaic_alias",
            "label",
        ],
        "tracks": ["track_id", "label"],
        "cuts": ["frame"],
    }
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = {
            table for table in _tables(connection) if not table.startswith("sqlite_")
        }
        if tables != set(expected_columns):
            raise ArtifactError(
                f"{source}: legacy tables must be exactly "
                f"{sorted(expected_columns)}, got {sorted(tables)}"
            )
        for table, expected in expected_columns.items():
            actual = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            if actual != expected:
                raise ArtifactError(
                    f"{source}: legacy {table} columns must be {expected}, "
                    f"got {actual}"
                )
    stats = validate_mask_sqlite(source)
    stats["schema"] = "dinov3_postprocess_final_mask_sqlite_v1"
    return stats


def read_postprocess_artifacts(
    path: Path,
) -> tuple[Path, Path, Path | None]:
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
        final_value = artifacts.get("result_sqlite")
        if final_value in (None, ""):
            final_value = artifacts.get("combined_predictions_sqlite")
        if final_value in (None, ""):
            final_value = artifacts["predictions_sqlite"]
        final = Path(str(final_value)).expanduser().resolve()
    except KeyError as exc:
        raise ArtifactError(
            f"{manifest_path}: required postprocess artifact is absent: {exc}"
        ) from exc
    validate_mask_sqlite(tracked)
    if artifacts.get("result_sqlite") not in (None, ""):
        validate_result_sqlite(
            final,
            require_segmentation=False,
            require_faces=False,
        )
    else:
        validate_mask_sqlite(final)
    legacy_value = artifacts.get(
        "combined_legacy_predictions_sqlite",
        artifacts.get("legacy_predictions_sqlite"),
    )
    legacy = (
        None
        if legacy_value in (None, "")
        else Path(str(legacy_value)).expanduser().resolve()
    )
    if legacy is not None:
        validate_legacy_mask_sqlite(legacy)
    return tracked, final, legacy


def read_postprocess_manifest(path: Path) -> tuple[Path, Path]:
    """Read the original required artifact pair without changing its API."""

    tracked, final, _legacy = read_postprocess_artifacts(path)
    return tracked, final


__all__ = [
    "ArtifactError",
    "read_postprocess_artifacts",
    "read_postprocess_manifest",
    "validate_inference_sqlite",
    "validate_legacy_mask_sqlite",
    "validate_mask_sqlite",
    "validate_result_sqlite",
]
