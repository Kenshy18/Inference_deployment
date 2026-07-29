"""Read-only adapters for inference and postprocess SQLite contracts."""

from __future__ import annotations

import heapq
import itertools
import json
import sqlite3
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import (
    FaceKeypointOverlay,
    FaceMaskOverlay,
    FrameOverlay,
    OverlayItem,
    Polygon,
    SourceInfo,
)
from .keyframe_cache import (
    is_keyframe_primary,
    materialize_overlay_cache,
)


INFERENCE_SCHEMA_NAME = "instance-segmentation-unified-inference"
INFERENCE_SCHEMA_VERSION = "3"
INFERENCE_SCHEMA_VERSIONS = frozenset({"2", "3"})

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

RICH_FACE_COLUMNS = {
    "face_observations": {
        "id",
        "head_detection_id",
        "face_detection_id",
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
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


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
    schema_version = info.get("schema_version")
    if schema_version not in INFERENCE_SCHEMA_VERSIONS:
        raise OverlayContractError(
            f"{path}: unsupported inference schema version: "
            f"{info.get('schema_version', '<missing>')}"
        )
    if schema_version == "3":
        _validate_columns(connection, RICH_FACE_COLUMNS, path=path)
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise OverlayContractError(
            f"{path}: SQLite integrity check failed: {integrity}"
        )


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
        if is_keyframe_primary(resolved):
            _validate_columns(
                connection,
                {
                    "result_schema_info": {"key", "value"},
                    "mask_track_segments": {
                        "start_frame",
                        "end_frame",
                    },
                    "mask_keyframes": {"id"},
                    "annotation_state": {
                        "revision",
                        "authoritative_geometry",
                        "dense_cache_policy",
                    },
                },
                path=resolved,
            )
            row = connection.execute(
                """
                SELECT COUNT(*) AS item_count,
                       MIN(start_frame) AS first_frame,
                       MAX(end_frame) AS last_frame
                FROM mask_track_segments
                """
            ).fetchone()
            width, height, fps = _video_metadata(connection)
            return SourceInfo(
                path=resolved,
                schema="keyframe-primary-v3",
                role="mask",
                item_count=int(row["item_count"]),
                first_frame=(
                    None if row["first_frame"] is None else int(row["first_frame"])
                ),
                last_frame=(
                    None if row["last_frame"] is None else int(row["last_frame"])
                ),
                width=width,
                height=height,
                fps=fps,
            )
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
    return "classifications" in _tables(connection) and {
        "detection_id",
        "class_name",
        "score",
    }.issubset(_columns(connection, "classifications"))


def _polygon_from_rows(rows: list[sqlite3.Row]) -> Polygon:
    return tuple((float(row["x"]), float(row["y"])) for row in rows)


def iter_raw_segmentation_frames(
    path: Path,
    *,
    display_style: str = "legacy",
) -> Iterator[FrameOverlay]:
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
                   d.x1, d.y1, d.x2, d.y2,
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
                    color_key=(
                        "genital:simple"
                        if display_style == "simple"
                        else f"raw:{label}"
                    ),
                    kind="mask",
                    label=label,
                    score=score,
                    polygons=polygons,
                    box=(
                        float(first["x1"]),
                        float(first["y1"]),
                        float(first["x2"]),
                        float(first["y2"]),
                    )
                    if display_style == "detailed"
                    else None,
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


def iter_mask_frames(
    path: Path,
    *,
    display_style: str = "legacy",
    prefer_tracked: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    mask_domain: str | None = None,
) -> Iterator[FrameOverlay]:
    """Stream tracked or final postprocess masks grouped by frame."""

    resolved = Path(path).expanduser().resolve()
    if is_keyframe_primary(resolved):
        with tempfile.TemporaryDirectory(prefix="overlay-keyframe-cache-") as temporary:
            cache = Path(temporary) / "masks.sqlite"
            materialize_overlay_cache(
                resolved,
                cache,
                mode="tracked" if prefer_tracked else "final",
                start_frame=start_frame,
                end_frame=end_frame,
                mask_domain=mask_domain,
            )
            yield from iter_mask_frames(
                cache,
                display_style=display_style,
                prefer_tracked=False,
                start_frame=start_frame,
                end_frame=end_frame,
                mask_domain=None,
            )
        return
    connection = _connect_read_only(resolved)
    try:
        _validate_columns(connection, MASK_COLUMNS, path=resolved)
        tables = _tables(connection)
        mask_table = (
            "tracked_masks" if prefer_tracked and "tracked_masks" in tables else "masks"
        )
        if mask_table != "masks":
            _validate_columns(
                connection,
                {mask_table: MASK_COLUMNS["masks"]},
                path=resolved,
            )
        columns = _columns(connection, mask_table)
        label_expression = (
            "COALESCE(m.label, '') AS label" if "label" in columns else "'' AS label"
        )
        has_raw_audit = "raw_tracked_masks" in tables
        audit_join = (
            """
            LEFT JOIN (
                SELECT frame, final_track_id,
                       MAX(score) AS source_score
                FROM raw_tracked_masks
                WHERE removed_by_short_track=0
                  AND final_track_id IS NOT NULL
                GROUP BY frame, final_track_id
            ) audit
              ON audit.frame=m.frame
             AND CAST(audit.final_track_id AS TEXT)=CAST(m.track_id AS TEXT)
            """
            if has_raw_audit and display_style == "detailed"
            else ""
        )
        score_expression = (
            "audit.source_score AS source_score"
            if audit_join
            else "NULL AS source_score"
        )
        domain_join = ""
        domain_where = ""
        parameters: tuple[object, ...] = ()
        if (
            mask_domain is not None
            and "tracks" in tables
            and {"track_id", "domain"}.issubset(_columns(connection, "tracks"))
        ):
            domain_join = (
                "JOIN tracks domain_track "
                "ON CAST(domain_track.track_id AS TEXT)="
                "CAST(m.track_id AS TEXT)"
            )
            domain_where = "WHERE domain_track.domain=?"
            parameters = (mask_domain,)
        rows = connection.execute(
            f"""
            SELECT m.frame, m.track_id, m.polygons, {label_expression},
                   {score_expression}
            FROM {mask_table} m
            {domain_join}
            {audit_join}
            {domain_where}
            ORDER BY m.frame, m.track_id
            """,
            parameters,
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
                    color_key=(
                        "genital:simple"
                        if display_style == "simple"
                        else f"track:{track_id}"
                    ),
                    kind="mask",
                    label=str(row["label"]),
                    score=(
                        None
                        if row["source_score"] is None
                        else float(row["source_score"])
                    ),
                    track_id=track_id,
                    polygons=polygons,
                    provenance=(
                        "GAP-FILL"
                        if display_style == "detailed" and row["source_score"] is None
                        else "OBSERVED"
                        if display_style == "detailed"
                        else None
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


class _RichFaceDetailReader:
    """Read and decompress only the currently rendered face observation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path,
        *,
        include_keypoints: bool,
        include_probability_masks: bool,
    ) -> None:
        self._path = path
        self._points = iter(
            connection.execute(
                """
                SELECT kp.observation_id, kp.class_name, kp.x, kp.y,
                       kp.state, kp.state_name, kp.confidence, kp.valid,
                       (
                           SELECT sp.probability
                           FROM face_keypoint_state_probabilities sp
                           WHERE sp.observation_id=kp.observation_id
                             AND sp.point_index=kp.point_index
                             AND sp.state_index=kp.state
                       ) AS state_confidence
                FROM face_keypoints kp
                ORDER BY observation_id, point_index
                """
            )
            if include_keypoints
            else ()
        )
        self._masks = iter(
            connection.execute(
                """
                SELECT observation_id, encoding, width, height,
                       box_x1, box_y1, box_x2, box_y2, data
                FROM face_masks
                ORDER BY observation_id
                """
            )
            if include_probability_masks
            else ()
        )
        self._point = next(self._points, None)
        self._mask = next(self._masks, None)
        self._cached_observation_id: int | None = None
        self._cached_details: (
            tuple[tuple[FaceKeypointOverlay, ...], FaceMaskOverlay | None] | None
        ) = None

    def get(
        self,
        observation_id: int,
    ) -> tuple[tuple[FaceKeypointOverlay, ...], FaceMaskOverlay | None]:
        if observation_id == self._cached_observation_id:
            assert self._cached_details is not None
            return self._cached_details
        if (
            self._cached_observation_id is not None
            and observation_id < self._cached_observation_id
        ):
            raise OverlayContractError(
                f"{self._path}: face observations are not in render order"
            )

        points: list[FaceKeypointOverlay] = []
        while (
            self._point is not None
            and int(self._point["observation_id"]) < observation_id
        ):
            self._point = next(self._points, None)
        while (
            self._point is not None
            and int(self._point["observation_id"]) == observation_id
        ):
            point = self._point
            points.append(
                FaceKeypointOverlay(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    class_name=str(point["class_name"]),
                    state=int(point["state"]),
                    state_name=str(point["state_name"]),
                    confidence=float(point["confidence"]),
                    valid=bool(point["valid"]),
                    state_confidence=(
                        None
                        if point["state_confidence"] is None
                        else float(point["state_confidence"])
                    ),
                )
            )
            self._point = next(self._points, None)

        while (
            self._mask is not None
            and int(self._mask["observation_id"]) < observation_id
        ):
            self._mask = next(self._masks, None)
        face_mask = None
        if (
            self._mask is not None
            and int(self._mask["observation_id"]) == observation_id
        ):
            face_mask = self._decode_mask(self._mask)
            self._mask = next(self._masks, None)

        details = (tuple(points), face_mask)
        self._cached_observation_id = observation_id
        self._cached_details = details
        return details

    def _decode_mask(self, mask: sqlite3.Row) -> FaceMaskOverlay:
        if str(mask["encoding"]) != "zlib-u8-probability-v1":
            raise OverlayContractError(f"{self._path}: unsupported face mask encoding")
        width = int(mask["width"])
        height = int(mask["height"])
        try:
            probabilities = zlib.decompress(bytes(mask["data"]))
        except zlib.error as exc:
            raise OverlayContractError(f"{self._path}: corrupt face mask") from exc
        if len(probabilities) != width * height:
            raise OverlayContractError(f"{self._path}: face mask size mismatch")
        return FaceMaskOverlay(
            width=width,
            height=height,
            box=(
                float(mask["box_x1"]),
                float(mask["box_y1"]),
                float(mask["box_x2"]),
                float(mask["box_y2"]),
            ),
            probabilities=probabilities,
        )


def iter_face_frames(
    path: Path,
    *,
    include_ellipses: bool = True,
    include_keypoints: bool = True,
    include_probability_masks: bool = True,
    display_style: str = "legacy",
    require_privacy_geometry: bool = False,
) -> Iterator[FrameOverlay]:
    """Stream boxes plus optional rich face ellipses/keypoints by frame."""

    resolved = Path(path).expanduser().resolve()
    connection = _connect_read_only(resolved)
    try:
        _validate_inference(connection, resolved)
        _require_role(connection, resolved, "face_detection")
        has_rich_faces = "face_observations" in _tables(connection)
        if require_privacy_geometry and not has_rich_faces:
            raise OverlayContractError(
                f"{resolved}: face privacy masks require schema-v3 rich face "
                "ellipses and keypoints"
            )
        detail_reader = (
            _RichFaceDetailReader(
                connection,
                resolved,
                include_keypoints=include_keypoints,
                include_probability_masks=include_probability_masks,
            )
            if has_rich_faces
            else None
        )
        if has_rich_faces and display_style in {"detailed", "simple"}:
            assert detail_reader is not None
            has_face_tracking = "face_tracking_assignments" in _tables(connection)
            tracking_columns = (
                """,
                       fta.raw_track_id,
                       fta.final_track_id,
                       fta.removed_by_short_track
                """
                if has_face_tracking
                else """,
                       NULL AS raw_track_id,
                       NULL AS final_track_id,
                       0 AS removed_by_short_track
                """
            )
            tracking_join = (
                """
                LEFT JOIN face_tracking_assignments fta
                  ON fta.observation_id=fo.id
                """
                if has_face_tracking
                else ""
            )
            rows = connection.execute(
                f"""
                SELECT f.frame_index,
                       fo.id AS observation_id,
                       fo.face_score,
                       fo.face_present,
                       fo.geometry_type,
                       fo.ellipse_cx,
                       fo.ellipse_cy,
                       fo.ellipse_major_radius,
                       fo.ellipse_minor_radius,
                       fo.ellipse_theta_radians,
                       COALESCE(h.score, a.score) AS head_score,
                       COALESCE(h.x1, a.x1) AS head_x1,
                       COALESCE(h.y1, a.y1) AS head_y1,
                       COALESCE(h.x2, a.x2) AS head_x2,
                       COALESCE(h.y2, a.y2) AS head_y2
                       {tracking_columns}
                FROM face_observations fo
                JOIN detections a ON a.id=fo.anchor_detection_id
                JOIN frames f ON f.id=a.frame_id
                LEFT JOIN detections h ON h.id=fo.head_detection_id
                {tracking_join}
                ORDER BY f.frame_index, fo.id
                """
            )

            def iter_observations() -> Iterator[tuple[int, OverlayItem]]:
                for row in rows:
                    observation_id = int(row["observation_id"])
                    removed = bool(row["removed_by_short_track"])
                    if removed and display_style != "detailed":
                        continue
                    track_id = (
                        str(row["raw_track_id"])
                        if removed and row["raw_track_id"] is not None
                        else (
                            None
                            if row["final_track_id"] is None
                            else str(row["final_track_id"])
                        )
                    )
                    has_face = bool(row["face_present"])
                    details = (
                        detail_reader.get(observation_id) if has_face else ((), None)
                    )
                    ellipse = (
                        (
                            float(row["ellipse_cx"]),
                            float(row["ellipse_cy"]),
                            float(row["ellipse_major_radius"]),
                            float(row["ellipse_minor_radius"]),
                            float(row["ellipse_theta_radians"]),
                        )
                        if has_face
                        and row["geometry_type"] == "ellipse"
                        and include_ellipses
                        else None
                    )
                    yield int(row["frame_index"]), OverlayItem(
                        identity=(
                            f"face-observation:{observation_id}"
                            if track_id is None
                            else f"face-track:{track_id}"
                        ),
                        color_key=(
                            "face:removed"
                            if removed
                            else (
                                "face:observation"
                                if track_id is None
                                else f"face-track:{track_id}"
                            )
                        ),
                        kind="face",
                        label="Head",
                        score=float(row["head_score"]),
                        box=(
                            (
                                float(row["head_x1"]),
                                float(row["head_y1"]),
                                float(row["head_x2"]),
                                float(row["head_y2"]),
                            )
                            if display_style == "detailed"
                            else None
                        ),
                        ellipse=ellipse,
                        keypoints=(
                            details[0] if has_face and include_keypoints else ()
                        ),
                        face_mask=(
                            details[1]
                            if has_face
                            and include_probability_masks
                            and display_style == "detailed"
                            else None
                        ),
                        face_score=float(row["face_score"]),
                        face_present=has_face,
                        track_id=track_id,
                        provenance=("REMOVED_SHORT_TRACK" if removed else "OBSERVED"),
                    )

            def iter_interpolations() -> Iterator[tuple[int, OverlayItem]]:
                if (
                    display_style != "detailed"
                    or "face_track_interpolations" not in _tables(connection)
                ):
                    return
                for row in connection.execute(
                    """
                    SELECT frame, final_track_id,
                           head_x1, head_y1, head_x2, head_y2
                    FROM face_track_interpolations
                    ORDER BY frame, final_track_id
                    """
                ):
                    track_id = str(row["final_track_id"])
                    yield int(row["frame"]), OverlayItem(
                        identity=f"face-track:{track_id}:interpolated",
                        color_key=f"face-track:{track_id}",
                        kind="face",
                        label="Head",
                        score=None,
                        box=(
                            float(row["head_x1"]),
                            float(row["head_y1"]),
                            float(row["head_x2"]),
                            float(row["head_y2"]),
                        ),
                        track_id=track_id,
                        provenance="INTERPOLATED",
                    )

            combined = heapq.merge(
                iter_observations(),
                iter_interpolations(),
                key=lambda pair: (pair[0], pair[1].identity),
            )
            for frame_index, grouped in itertools.groupby(
                combined,
                key=lambda pair: pair[0],
            ):
                yield FrameOverlay(
                    frame_index=frame_index,
                    items=tuple(item for _frame, item in grouped),
                )
            return
        rich_columns = (
            """
            , fo.id AS observation_id,
              fo.face_detection_id,
              fo.face_present,
              fo.geometry_type,
              fo.ellipse_cx,
              fo.ellipse_cy,
              fo.ellipse_major_radius,
              fo.ellipse_minor_radius,
              fo.ellipse_theta_radians
            """
            if has_rich_faces
            else """
            , NULL AS observation_id,
              NULL AS face_detection_id,
              NULL AS face_present,
              NULL AS geometry_type,
              NULL AS ellipse_cx,
              NULL AS ellipse_cy,
              NULL AS ellipse_major_radius,
              NULL AS ellipse_minor_radius,
              NULL AS ellipse_theta_radians
            """
        )
        rich_join = (
            """
            LEFT JOIN face_observations fo
              ON fo.head_detection_id=d.id OR fo.face_detection_id=d.id
            """
            if has_rich_faces
            else ""
        )
        rows = connection.execute(
            f"""
            SELECT f.frame_index,
                   d.id AS detection_id,
                   d.class_name,
                   d.score,
                   d.x1, d.y1, d.x2, d.y2
                   {rich_columns}
            FROM detections d
            JOIN frames f ON f.id=d.frame_id
            JOIN model_executions me ON me.id=d.model_execution_id
            {rich_join}
            WHERE me.role='face_detection'
            ORDER BY f.frame_index, d.id
            """
        )

        def iter_items() -> Iterator[tuple[int, OverlayItem]]:
            for row in rows:
                frame = int(row["frame_index"])
                label = str(row["class_name"])
                detection_id = int(row["detection_id"])
                observation_id = (
                    None
                    if row["observation_id"] is None
                    else int(row["observation_id"])
                )
                is_rich_face = (
                    label.lower() == "face"
                    and observation_id is not None
                    and bool(row["face_present"])
                    and row["geometry_type"] == "ellipse"
                )
                ellipse = (
                    (
                        float(row["ellipse_cx"]),
                        float(row["ellipse_cy"]),
                        float(row["ellipse_major_radius"]),
                        float(row["ellipse_minor_radius"]),
                        float(row["ellipse_theta_radians"]),
                    )
                    if is_rich_face and include_ellipses
                    else None
                )
                rich_details = (
                    detail_reader.get(observation_id)
                    if is_rich_face
                    and observation_id is not None
                    and detail_reader is not None
                    else ((), None)
                )
                yield frame, OverlayItem(
                    identity=f"face:{detection_id}",
                    color_key=f"face:{label}",
                    kind="face",
                    label=label,
                    score=float(row["score"]),
                    box=(
                        None
                        if is_rich_face and include_ellipses
                        else (
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        )
                    ),
                    ellipse=ellipse,
                    keypoints=rich_details[0] if include_keypoints else (),
                    face_mask=(rich_details[1] if include_probability_masks else None),
                )
                if (
                    label.lower() == "head"
                    and observation_id is not None
                    and row["face_detection_id"] is None
                    and bool(row["face_present"])
                    and row["geometry_type"] == "ellipse"
                    and include_ellipses
                ):
                    assert detail_reader is not None
                    synthetic_details = detail_reader.get(observation_id)
                    yield frame, OverlayItem(
                        identity=f"face-observation:{observation_id}",
                        color_key="face:Face",
                        kind="face",
                        label="Face",
                        ellipse=(
                            float(row["ellipse_cx"]),
                            float(row["ellipse_cy"]),
                            float(row["ellipse_major_radius"]),
                            float(row["ellipse_minor_radius"]),
                            float(row["ellipse_theta_radians"]),
                        ),
                        keypoints=(synthetic_details[0] if include_keypoints else ()),
                        face_mask=(
                            synthetic_details[1] if include_probability_masks else None
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
    "INFERENCE_SCHEMA_VERSIONS",
    "OverlayContractError",
    "inspect_inference_source",
    "inspect_mask_source",
    "iter_face_frames",
    "iter_mask_frames",
    "iter_raw_segmentation_frames",
]
