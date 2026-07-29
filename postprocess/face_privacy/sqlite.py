"""SQLite I/O for canonical face privacy postprocessing."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .geometry import (
    ALGORITHM_VERSION,
    FaceEllipse,
    FaceKeypoint,
    derive_privacy_mask,
)


SCHEMA_NAME = "face-privacy-mask-sqlite"
SCHEMA_VERSION = "1"
INFERENCE_SCHEMA_NAME = "instance-segmentation-unified-inference"


def _read_only(path: Path) -> sqlite3.Connection:
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


def _validate_rich_face_input(connection: sqlite3.Connection, path: Path) -> None:
    info = dict(connection.execute("SELECT key, value FROM schema_info"))
    if (
        info.get("schema_name") != INFERENCE_SCHEMA_NAME
        or str(info.get("schema_version")) != "3"
    ):
        raise ValueError(f"{path}: face privacy requires unified schema-v3")
    required = {
        "frames": {"id", "frame_index"},
        "detections": {"id", "frame_id", "model_execution_id"},
        "model_executions": {"id", "role"},
        "face_observations": {
            "id",
            "anchor_detection_id",
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
            "class_name",
            "x",
            "y",
            "confidence",
            "valid",
        },
    }
    tables = _tables(connection)
    for table, expected in required.items():
        if table not in tables:
            raise ValueError(f"{path}: required rich-face table is absent: {table}")
        missing = expected - _columns(connection, table)
        if missing:
            raise ValueError(f"{path}: {table} columns missing: {sorted(missing)}")
    role = connection.execute(
        "SELECT 1 FROM model_executions WHERE role='face_detection' LIMIT 1"
    ).fetchone()
    if role is None:
        raise ValueError(f"{path}: face_detection execution is absent")
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise ValueError(f"{path}: SQLite integrity check failed: {integrity}")


def _iter_observations(
    connection: sqlite3.Connection,
) -> Iterator[tuple[int, int, float, FaceEllipse | None, tuple[FaceKeypoint, ...]]]:
    points = iter(
        connection.execute(
            """
            SELECT observation_id, point_index, class_name, x, y,
                   confidence, valid
            FROM face_keypoints
            ORDER BY observation_id, point_index
            """
        )
    )
    current = next(points, None)
    observations = connection.execute(
        """
        SELECT f.frame_index, fo.id, fo.face_score, fo.face_present,
               fo.geometry_type, fo.ellipse_cx, fo.ellipse_cy,
               fo.ellipse_major_radius, fo.ellipse_minor_radius,
               fo.ellipse_theta_radians
        FROM face_observations fo
        JOIN detections anchor ON anchor.id=fo.anchor_detection_id
        JOIN frames f ON f.id=anchor.frame_id
        JOIN model_executions me ON me.id=anchor.model_execution_id
        WHERE me.role='face_detection'
        ORDER BY fo.id
        """
    )
    for observation in observations:
        observation_id = int(observation["id"])
        while current is not None and int(current["observation_id"]) < observation_id:
            current = next(points, None)
        keypoints: list[FaceKeypoint] = []
        while current is not None and int(current["observation_id"]) == observation_id:
            keypoints.append(
                FaceKeypoint(
                    x=float(current["x"]),
                    y=float(current["y"]),
                    class_name=str(current["class_name"]),
                    confidence=float(current["confidence"]),
                    valid=bool(current["valid"]),
                )
            )
            current = next(points, None)
        ellipse = (
            (
                float(observation["ellipse_cx"]),
                float(observation["ellipse_cy"]),
                float(observation["ellipse_major_radius"]),
                float(observation["ellipse_minor_radius"]),
                float(observation["ellipse_theta_radians"]),
            )
            if bool(observation["face_present"])
            and observation["geometry_type"] == "ellipse"
            else None
        )
        yield (
            int(observation["frame_index"]),
            observation_id,
            float(observation["face_score"]),
            ellipse,
            tuple(keypoints),
        )


def export_face_masks(
    source: Path,
    output: Path,
    *,
    target: str,
    eye_shape: str = "ellipse",
    minimum_eye_confidence: float = 0.35,
) -> dict[str, object]:
    """Create a software-compatible face mask sidecar without source mutation."""

    resolved_source = Path(source).expanduser().resolve()
    resolved_output = Path(output).expanduser().resolve()
    if resolved_source == resolved_output:
        raise ValueError("face mask output must differ from inference input")
    if target not in {"face", "eyes"}:
        raise ValueError("target must be face or eyes")
    if eye_shape not in {"ellipse", "rectangle"}:
        raise ValueError("eye_shape must be ellipse or rectangle")
    if not 0.0 <= minimum_eye_confidence <= 1.0:
        raise ValueError("minimum_eye_confidence must be between 0 and 1")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.{uuid.uuid4().hex}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    counts: Counter[str] = Counter()
    first_frame: int | None = None
    last_frame: int | None = None
    try:
        with _read_only(resolved_source) as source_connection:
            _validate_rich_face_input(source_connection, resolved_source)
            with sqlite3.connect(temporary) as output_connection:
                output_connection.executescript(
                    """
                    PRAGMA synchronous=FULL;
                    CREATE TABLE schema_info(
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE masks(
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        polygons TEXT NOT NULL,
                        shape_type TEXT NOT NULL,
                        dilate_px INTEGER NOT NULL DEFAULT 0,
                        feather_px INTEGER NOT NULL DEFAULT 0,
                        mosaic_block INTEGER NOT NULL DEFAULT 0,
                        mosaic_alias REAL NOT NULL DEFAULT 0,
                        label TEXT NOT NULL,
                        PRIMARY KEY(frame, track_id)
                    );
                    CREATE TABLE tracks(
                        track_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL
                    );
                    CREATE TABLE mask_provenance(
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        mask_kind TEXT NOT NULL,
                        source_observation_id INTEGER NOT NULL,
                        derivation TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        algorithm_version TEXT NOT NULL,
                        PRIMARY KEY(frame, track_id)
                    );
                    CREATE TABLE face_mask_geometries(
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        geometry_type TEXT NOT NULL
                            CHECK(geometry_type IN ('ellipse', 'rectangle')),
                        cx REAL NOT NULL,
                        cy REAL NOT NULL,
                        half_width REAL NOT NULL CHECK(half_width > 0),
                        half_height REAL NOT NULL CHECK(half_height > 0),
                        theta_radians REAL NOT NULL,
                        PRIMARY KEY(frame, track_id)
                    );
                    CREATE INDEX idx_face_masks_frame ON masks(frame);
                    """
                )
                output_connection.executemany(
                    "INSERT INTO schema_info(key, value) VALUES (?, ?)",
                    sorted(
                        {
                            "schema_name": SCHEMA_NAME,
                            "schema_version": SCHEMA_VERSION,
                            "algorithm_version": ALGORITHM_VERSION,
                            "source_sqlite": str(resolved_source),
                            "target": target,
                            "eye_shape": eye_shape if target == "eyes" else "ellipse",
                            "minimum_eye_confidence": repr(minimum_eye_confidence),
                        }.items()
                    ),
                )
                mask_rows: list[tuple[object, ...]] = []
                provenance_rows: list[tuple[object, ...]] = []
                geometry_rows: list[tuple[object, ...]] = []
                track_rows: list[tuple[str, str]] = []
                for (
                    frame,
                    observation_id,
                    face_score,
                    ellipse,
                    keypoints,
                ) in _iter_observations(source_connection):
                    mask = derive_privacy_mask(
                        target,
                        ellipse,
                        keypoints,
                        eye_shape=eye_shape,
                        minimum_eye_confidence=minimum_eye_confidence,
                    )
                    if mask is None:
                        counts["not_emitted"] += 1
                        continue
                    track_id = f"face:{target}:{observation_id}"
                    label = "Face" if target == "face" else "Eyes"
                    confidence = (
                        face_score
                        if mask.derivation == "face-ellipse"
                        else mask.confidence
                    )
                    polygons = json.dumps(
                        [[[float(x), float(y)] for x, y in mask.polygon]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    mask_rows.append(
                        (
                            frame,
                            track_id,
                            polygons,
                            mask.shape,
                            label,
                        )
                    )
                    provenance_rows.append(
                        (
                            frame,
                            track_id,
                            target,
                            observation_id,
                            mask.derivation,
                            confidence,
                            ALGORITHM_VERSION,
                        )
                    )
                    geometry_rows.append(
                        (
                            frame,
                            track_id,
                            mask.shape,
                            float(mask.center[0]),
                            float(mask.center[1]),
                            float(mask.half_width),
                            float(mask.half_height),
                            float(mask.theta_radians),
                        )
                    )
                    track_rows.append((track_id, label))
                    counts[mask.derivation] += 1
                    first_frame = (
                        frame if first_frame is None else min(first_frame, frame)
                    )
                    last_frame = frame if last_frame is None else max(last_frame, frame)
                    if len(mask_rows) >= 1000:
                        _insert_rows(
                            output_connection,
                            mask_rows,
                            provenance_rows,
                            geometry_rows,
                            track_rows,
                        )
                _insert_rows(
                    output_connection,
                    mask_rows,
                    provenance_rows,
                    geometry_rows,
                    track_rows,
                )
                integrity = str(
                    output_connection.execute("PRAGMA integrity_check").fetchone()[0]
                )
                if integrity != "ok":
                    raise RuntimeError(
                        f"face privacy output integrity failed: {integrity}"
                    )
                output_connection.commit()
        os.replace(temporary, resolved_output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "target": target,
        "shape": eye_shape if target == "eyes" else "ellipse",
        "rows": sum(count for name, count in counts.items() if name != "not_emitted"),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "derivations": dict(sorted(counts.items())),
    }


def _insert_rows(
    connection: sqlite3.Connection,
    masks: list[tuple[object, ...]],
    provenance: list[tuple[object, ...]],
    geometries: list[tuple[object, ...]],
    tracks: list[tuple[str, str]],
) -> None:
    if not masks:
        return
    connection.executemany(
        """
        INSERT INTO masks(
            frame, track_id, polygons, shape_type, label
        ) VALUES (?, ?, ?, ?, ?)
        """,
        masks,
    )
    connection.executemany(
        """
        INSERT INTO mask_provenance(
            frame, track_id, mask_kind, source_observation_id,
            derivation, confidence, algorithm_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        provenance,
    )
    connection.executemany(
        """
        INSERT INTO face_mask_geometries(
            frame, track_id, geometry_type, cx, cy,
            half_width, half_height, theta_radians
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        geometries,
    )
    connection.executemany(
        "INSERT INTO tracks(track_id, label) VALUES (?, ?)",
        tracks,
    )
    masks.clear()
    provenance.clear()
    geometries.clear()
    tracks.clear()


def merge_face_masks(
    predictions: Path,
    face_masks: Path,
    output: Path,
) -> dict[str, object]:
    """Merge face masks into final genital masks using namespaced track IDs."""

    resolved_predictions = Path(predictions).expanduser().resolve()
    resolved_faces = Path(face_masks).expanduser().resolve()
    resolved_output = Path(output).expanduser().resolve()
    if resolved_output in {resolved_predictions, resolved_faces}:
        raise ValueError("merged output must differ from both inputs")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_sidecars = (
        temporary.with_name(f"{temporary.name}-wal"),
        temporary.with_name(f"{temporary.name}-shm"),
    )
    for path in (temporary, *temporary_sidecars):
        if path.exists():
            path.unlink()
    try:
        with _read_only(resolved_predictions) as predictions_connection:
            required = {
                "frame",
                "track_id",
                "polygons",
                "shape_type",
                "dilate_px",
                "feather_px",
                "mosaic_block",
                "mosaic_alias",
                "label",
            }
            missing = required - _columns(predictions_connection, "masks")
            if missing:
                raise ValueError(
                    f"{resolved_predictions}: final masks columns missing: "
                    f"{sorted(missing)}"
                )
            with sqlite3.connect(temporary) as output_connection:
                predictions_connection.backup(output_connection)
        with sqlite3.connect(temporary) as output_connection:
            # SQLite backup preserves the source database's persistent WAL
            # journal mode.  Keep all merged rows in the main temporary file:
            # only that file is atomically renamed into place below.
            journal_mode = str(
                output_connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            )
            if journal_mode.lower() != "delete":
                raise RuntimeError(
                    "failed to switch merged SQLite journal mode to DELETE: "
                    f"{journal_mode}"
                )
            output_connection.execute(
                "ATTACH DATABASE ? AS face_db", (str(resolved_faces),)
            )
            face_info = dict(
                output_connection.execute("SELECT key, value FROM face_db.schema_info")
            )
            if (
                face_info.get("schema_name") != SCHEMA_NAME
                or str(face_info.get("schema_version")) != SCHEMA_VERSION
            ):
                raise ValueError(f"{resolved_faces}: unsupported face mask schema")
            collision = output_connection.execute(
                """
                SELECT 1
                FROM masks existing
                JOIN face_db.masks face
                  ON face.frame=existing.frame
                 AND face.track_id=existing.track_id
                LIMIT 1
                """
            ).fetchone()
            if collision is not None:
                raise ValueError("face mask track IDs collide with existing masks")
            output_connection.execute(
                """
                INSERT INTO masks(
                    frame, track_id, polygons, shape_type, dilate_px,
                    feather_px, mosaic_block, mosaic_alias, label
                )
                SELECT frame, track_id, polygons, shape_type, dilate_px,
                       feather_px, mosaic_block, mosaic_alias, label
                FROM face_db.masks
                """
            )
            if "tracks" in _tables(output_connection):
                track_columns = _columns(output_connection, "tracks")
                if {"track_id", "label"}.issubset(track_columns):
                    output_connection.execute(
                        """
                        INSERT INTO tracks(track_id, label)
                        SELECT track_id, label FROM face_db.tracks
                        """
                    )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mask_provenance(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    mask_kind TEXT NOT NULL,
                    source_observation_id INTEGER NOT NULL,
                    derivation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    PRIMARY KEY(frame, track_id)
                )
                """
            )
            output_connection.execute(
                """
                INSERT INTO mask_provenance
                SELECT * FROM face_db.mask_provenance
                """
            )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_mask_geometries(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    geometry_type TEXT NOT NULL
                        CHECK(geometry_type IN ('ellipse', 'rectangle')),
                    cx REAL NOT NULL,
                    cy REAL NOT NULL,
                    half_width REAL NOT NULL CHECK(half_width > 0),
                    half_height REAL NOT NULL CHECK(half_height > 0),
                    theta_radians REAL NOT NULL,
                    PRIMARY KEY(frame, track_id)
                )
                """
            )
            output_connection.execute(
                """
                INSERT INTO face_mask_geometries
                SELECT * FROM face_db.face_mask_geometries
                """
            )
            genital_count = int(
                output_connection.execute(
                    """
                    SELECT COUNT(*) FROM masks
                    WHERE track_id NOT LIKE 'face:%'
                    """
                ).fetchone()[0]
            )
            face_count = int(
                output_connection.execute(
                    "SELECT COUNT(*) FROM face_db.masks"
                ).fetchone()[0]
            )
            output_connection.commit()
            output_connection.execute("DETACH DATABASE face_db")
            integrity = str(
                output_connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if integrity != "ok":
                raise RuntimeError(f"merged SQLite integrity failed: {integrity}")
        os.replace(temporary, resolved_output)
    finally:
        for path in (temporary, *temporary_sidecars):
            if path.exists():
                path.unlink()
    return {
        "genital_masks": genital_count,
        "face_masks": face_count,
        "total_masks": genital_count + face_count,
    }
