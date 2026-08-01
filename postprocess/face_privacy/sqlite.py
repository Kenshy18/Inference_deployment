"""SQLite I/O for canonical face privacy postprocessing."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .geometry import (
    ALGORITHM_VERSION,
    FaceEllipse,
    FaceKeypoint,
    PrivacyMask,
    derive_privacy_mask,
)
from .tracking import (
    FaceTrackAssignment,
    FaceTrackObservation,
    FaceTrackSummary,
    FaceTracker,
    FaceTrackingConfig,
)


SCHEMA_NAME = "face-privacy-mask-sqlite"
SCHEMA_VERSION = "2"
INFERENCE_SCHEMA_NAME = "instance-segmentation-unified-inference"


@dataclass(frozen=True)
class _RichFaceObservation:
    frame: int
    observation_id: int
    anchor_detection_id: int
    head_score: float
    face_score: float
    bbox: tuple[float, float, float, float]
    ellipse: FaceEllipse | None
    keypoints: tuple[FaceKeypoint, ...]

    def tracking_observation(self) -> FaceTrackObservation:
        return FaceTrackObservation(
            observation_id=self.observation_id,
            anchor_detection_id=self.anchor_detection_id,
            frame=self.frame,
            bbox=self.bbox,
            head_score=self.head_score,
            face_score=self.face_score,
        )


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
) -> Iterator[_RichFaceObservation]:
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
        SELECT f.frame_index, fo.id, fo.anchor_detection_id,
               anchor.score AS head_score,
               anchor.x1, anchor.y1, anchor.x2, anchor.y2,
               fo.face_score, fo.face_present,
               fo.geometry_type, fo.ellipse_cx, fo.ellipse_cy,
               fo.ellipse_major_radius, fo.ellipse_minor_radius,
               fo.ellipse_theta_radians
        FROM face_observations fo
        JOIN detections anchor ON anchor.id=fo.anchor_detection_id
        JOIN frames f ON f.id=anchor.frame_id
        JOIN model_executions me ON me.id=anchor.model_execution_id
        WHERE me.role='face_detection'
        ORDER BY f.frame_index, fo.id
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
        yield _RichFaceObservation(
            frame=int(observation["frame_index"]),
            observation_id=observation_id,
            anchor_detection_id=int(observation["anchor_detection_id"]),
            head_score=float(observation["head_score"]),
            face_score=float(observation["face_score"]),
            bbox=(
                float(observation["x1"]),
                float(observation["y1"]),
                float(observation["x2"]),
                float(observation["y2"]),
            ),
            ellipse=ellipse,
            keypoints=tuple(keypoints),
        )


def export_face_masks(
    source: Path,
    output: Path,
    *,
    target: str,
    eye_shape: str = "ellipse",
    minimum_eye_confidence: float = 0.35,
    face_detection_score_threshold: float = 0.55,
    head_detection_score_threshold: float = 0.55,
    tracking_config: FaceTrackingConfig = FaceTrackingConfig(),
    interpolation_max_gap: int = 3,
    cuts_json: Path | None = None,
) -> dict[str, object]:
    """Create tracked, software-compatible face masks without source mutation."""

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
    if not 0.0 <= face_detection_score_threshold <= 1.0:
        raise ValueError("face_detection_score_threshold must be between 0 and 1")
    if not 0.0 <= head_detection_score_threshold <= 1.0:
        raise ValueError("head_detection_score_threshold must be between 0 and 1")
    tracking_config.validate()
    if interpolation_max_gap < 0:
        raise ValueError("interpolation_max_gap must be non-negative")
    cuts, cut_metadata = _read_cuts(cuts_json)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.{uuid.uuid4().hex}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    counts: Counter[str] = Counter()
    from common.live_preview import PreviewGeometry, active_postprocess_preview

    preview = active_postprocess_preview()
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
                    CREATE TABLE face_tracks(
                        raw_track_id TEXT PRIMARY KEY,
                        final_track_id TEXT,
                        scene_id INTEGER NOT NULL,
                        start_frame INTEGER NOT NULL,
                        end_frame INTEGER NOT NULL,
                        observed_frames INTEGER NOT NULL,
                        maximum_score REAL NOT NULL,
                        mean_score REAL NOT NULL,
                        removed_by_short_track INTEGER NOT NULL,
                        termination_reason TEXT NOT NULL
                    );
                    CREATE TABLE face_tracking_assignments(
                        observation_id INTEGER PRIMARY KEY,
                        anchor_detection_id INTEGER NOT NULL,
                        frame INTEGER NOT NULL,
                        raw_track_id TEXT NOT NULL,
                        final_track_id TEXT,
                        removed_by_short_track INTEGER NOT NULL DEFAULT 0,
                        association_stage TEXT NOT NULL,
                        association_score REAL,
                        head_score REAL NOT NULL,
                        face_score REAL NOT NULL,
                        head_x1 REAL NOT NULL,
                        head_y1 REAL NOT NULL,
                        head_x2 REAL NOT NULL,
                        head_y2 REAL NOT NULL,
                        scene_id INTEGER NOT NULL
                    );
                    CREATE TABLE face_track_interpolations(
                        frame INTEGER NOT NULL,
                        final_track_id TEXT NOT NULL,
                        scene_id INTEGER NOT NULL,
                        previous_observation_id INTEGER NOT NULL,
                        next_observation_id INTEGER NOT NULL,
                        head_x1 REAL NOT NULL,
                        head_y1 REAL NOT NULL,
                        head_x2 REAL NOT NULL,
                        head_y2 REAL NOT NULL,
                        interpolation_method TEXT NOT NULL,
                        PRIMARY KEY(frame, final_track_id)
                    );
                    CREATE TABLE mask_provenance(
                        frame INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        mask_kind TEXT NOT NULL,
                        source_observation_id INTEGER,
                        source_observation_id_end INTEGER,
                        is_interpolated INTEGER NOT NULL DEFAULT 0,
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
                    CREATE TABLE cuts(frame INTEGER PRIMARY KEY);
                    CREATE TABLE cut_detection_metadata(
                        id INTEGER PRIMARY KEY CHECK(id=1),
                        schema_version INTEGER NOT NULL,
                        method TEXT NOT NULL,
                        elapsed_seconds REAL NOT NULL,
                        cut_count INTEGER NOT NULL,
                        frame_semantics TEXT NOT NULL
                    );
                    CREATE INDEX idx_face_masks_frame ON masks(frame);
                    CREATE INDEX idx_face_tracking_frame
                        ON face_tracking_assignments(frame);
                    CREATE INDEX idx_face_tracking_final
                        ON face_tracking_assignments(final_track_id, frame);
                    CREATE INDEX idx_face_track_interpolations_frame
                        ON face_track_interpolations(frame, final_track_id);
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
                            "face_detection_score_threshold": repr(
                                face_detection_score_threshold
                            ),
                            "head_detection_score_threshold": repr(
                                head_detection_score_threshold
                            ),
                            "face_tracking": "head-box-hungarian-v1",
                            "face_tracking_max_gap": str(
                                tracking_config.max_gap_frames
                            ),
                            "face_interpolation_max_gap": str(interpolation_max_gap),
                        }.items()
                    ),
                )
                output_connection.executemany(
                    "INSERT INTO cuts(frame) VALUES (?)",
                    ((frame,) for frame in sorted(cuts)),
                )
                output_connection.execute(
                    """
                    INSERT INTO cut_detection_metadata(
                        id, schema_version, method, elapsed_seconds,
                        cut_count, frame_semantics
                    ) VALUES (1, ?, ?, ?, ?, 'first_frame_of_new_scene')
                    """,
                    (
                        int(cut_metadata["schema_version"]),
                        str(cut_metadata["method"]),
                        float(cut_metadata["elapsed_seconds"]),
                        len(cuts),
                    ),
                )
                mask_rows: list[tuple[object, ...]] = []
                provenance_rows: list[tuple[object, ...]] = []
                geometry_rows: list[tuple[object, ...]] = []
                track_rows: list[tuple[str, str]] = []
                assignment_rows: list[tuple[object, ...]] = []
                tracker = FaceTracker(tracking_config)
                observations = iter(_iter_observations(source_connection))
                current = next(observations, None)
                pending_cuts = iter(sorted(cuts))
                next_cut = next(pending_cuts, None)
                while current is not None:
                    frame = current.frame
                    frame_observations: list[_RichFaceObservation] = []
                    while current is not None and current.frame == frame:
                        frame_observations.append(current)
                        current = next(observations, None)
                    eligible_observations: list[_RichFaceObservation] = []
                    for observation in frame_observations:
                        if observation.head_score < head_detection_score_threshold:
                            counts["head_below_threshold"] += 1
                        else:
                            eligible_observations.append(observation)
                    while next_cut is not None and next_cut <= frame:
                        _unused, cut_completed = tracker.update(
                            next_cut,
                            (),
                            is_cut=True,
                        )
                        if cut_completed:
                            _insert_rows(
                                output_connection,
                                mask_rows,
                                provenance_rows,
                                geometry_rows,
                                track_rows,
                            )
                            _insert_assignment_rows(
                                output_connection,
                                assignment_rows,
                            )
                            _finalize_tracks(
                                output_connection,
                                cut_completed,
                                target=target,
                            )
                        next_cut = next(pending_cuts, None)
                    assignments, completed = tracker.update(
                        frame,
                        [
                            observation.tracking_observation()
                            for observation in eligible_observations
                        ],
                        is_cut=False,
                    )
                    by_observation = {
                        assignment.observation.observation_id: assignment
                        for assignment in assignments
                    }
                    for observation in eligible_observations:
                        assignment = by_observation[observation.observation_id]
                        assignment_rows.append(_assignment_row(assignment))
                        if observation.face_score < face_detection_score_threshold:
                            counts["face_below_threshold"] += 1
                            continue
                        mask = derive_privacy_mask(
                            target,
                            observation.ellipse,
                            observation.keypoints,
                            eye_shape=eye_shape,
                            minimum_eye_confidence=minimum_eye_confidence,
                        )
                        if mask is None:
                            counts["not_emitted"] += 1
                            continue
                        track_id = _privacy_track_id(
                            assignment.raw_track_id,
                            target,
                        )
                        label = "Face" if target == "face" else "Eyes"
                        confidence = (
                            observation.face_score
                            if mask.derivation == "face-ellipse"
                            else mask.confidence
                        )
                        mask_rows.append(_mask_row(frame, track_id, mask, label))
                        provenance_rows.append(
                            (
                                frame,
                                track_id,
                                target,
                                observation.observation_id,
                                observation.observation_id,
                                0,
                                mask.derivation,
                                confidence,
                                ALGORITHM_VERSION,
                            )
                        )
                        geometry_rows.append(_geometry_row(frame, track_id, mask))
                        track_rows.append((track_id, label))
                        counts[mask.derivation] += 1
                        if preview is not None and preview.should_sample(
                            "face_privacy_masks"
                        ):
                            preview.submit(
                                PreviewGeometry(
                                    frame,
                                    "face_privacy_masks",
                                    "face tracking + privacy mask",
                                    polygons=(
                                        tuple(
                                            (float(x), float(y))
                                            for x, y in mask.polygon
                                        ),
                                    ),
                                    boxes=(
                                        tuple(
                                            float(value) for value in observation.bbox
                                        ),
                                    ),
                                    points=tuple(
                                        (float(point.x), float(point.y))
                                        for point in observation.keypoints
                                        if point.valid
                                    ),
                                    track_id=track_id,
                                    detail=f"{target}/{mask.shape} / {assignment.association_stage}",
                                )
                            )
                        first_frame = (
                            frame if first_frame is None else min(first_frame, frame)
                        )
                        last_frame = (
                            frame if last_frame is None else max(last_frame, frame)
                        )
                    if completed:
                        _insert_rows(
                            output_connection,
                            mask_rows,
                            provenance_rows,
                            geometry_rows,
                            track_rows,
                        )
                        _insert_assignment_rows(output_connection, assignment_rows)
                        _finalize_tracks(
                            output_connection,
                            completed,
                            target=target,
                        )
                    elif len(mask_rows) >= 1000 or len(assignment_rows) >= 1000:
                        _insert_rows(
                            output_connection,
                            mask_rows,
                            provenance_rows,
                            geometry_rows,
                            track_rows,
                        )
                        _insert_assignment_rows(output_connection, assignment_rows)
                _insert_rows(
                    output_connection,
                    mask_rows,
                    provenance_rows,
                    geometry_rows,
                    track_rows,
                )
                _insert_assignment_rows(output_connection, assignment_rows)
                _finalize_tracks(
                    output_connection,
                    tracker.finish(),
                    target=target,
                )
                interpolated_boxes = _interpolate_track_boxes(
                    output_connection,
                    maximum_gap=interpolation_max_gap,
                )
                interpolated = _interpolate_short_gaps(
                    output_connection,
                    target=target,
                    maximum_gap=interpolation_max_gap,
                )
                counts["interpolated-linear"] += interpolated
                if interpolated:
                    minimum, maximum = output_connection.execute(
                        "SELECT MIN(frame), MAX(frame) FROM masks"
                    ).fetchone()
                    first_frame = None if minimum is None else int(minimum)
                    last_frame = None if maximum is None else int(maximum)
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
    mask_count = _sidecar_count(resolved_output, "masks")
    track_count = _sidecar_count(resolved_output, "face_tracks")
    kept_track_count = _sidecar_count(
        resolved_output,
        "face_tracks",
        "removed_by_short_track=0",
    )
    return {
        "target": target,
        "shape": eye_shape if target == "eyes" else "ellipse",
        "rows": mask_count,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "derivations": dict(sorted(counts.items())),
        "face_tracks": track_count,
        "kept_face_tracks": kept_track_count,
        "removed_face_tracks": track_count - kept_track_count,
        "tracking_assignments": _sidecar_count(
            resolved_output,
            "face_tracking_assignments",
        ),
        "interpolated_rows": int(counts["interpolated-linear"]),
        "interpolated_boxes": interpolated_boxes,
        "cuts": len(cuts),
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
            source_observation_id_end, is_interpolated,
            derivation, confidence, algorithm_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        "INSERT OR IGNORE INTO tracks(track_id, label) VALUES (?, ?)",
        tracks,
    )
    masks.clear()
    provenance.clear()
    geometries.clear()
    tracks.clear()


def _read_cuts(path: Path | None) -> tuple[set[int], dict[str, object]]:
    if path is None:
        return set(), {
            "schema_version": 1,
            "method": "not_run",
            "elapsed_seconds": 0.0,
        }
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"{resolved}: unsupported cuts schema")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{resolved}: cuts.frames must be a list")
    cuts = {int(frame) for frame in frames}
    if any(frame < 0 for frame in cuts):
        raise ValueError(f"{resolved}: cut frames must be non-negative")
    return cuts, {
        "schema_version": 1,
        "method": str(payload.get("method", "unknown")),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
    }


def _privacy_track_id(raw_track_id: str, target: str) -> str:
    parts = raw_track_id.split(":")
    if len(parts) != 4 or parts[:2] != ["face", "raw"]:
        raise ValueError(f"invalid raw face track ID: {raw_track_id}")
    return f"face:{target}:{parts[2]}:{parts[3]}"


def _subject_track_id(raw_track_id: str) -> str:
    parts = raw_track_id.split(":")
    if len(parts) != 4 or parts[:2] != ["face", "raw"]:
        raise ValueError(f"invalid raw face track ID: {raw_track_id}")
    return f"face:{parts[2]}:{parts[3]}"


def _assignment_row(assignment: FaceTrackAssignment) -> tuple[object, ...]:
    observation = assignment.observation
    return (
        observation.observation_id,
        observation.anchor_detection_id,
        observation.frame,
        assignment.raw_track_id,
        _subject_track_id(assignment.raw_track_id),
        0,
        assignment.association_stage,
        assignment.association_score,
        observation.head_score,
        observation.face_score,
        *observation.bbox,
        assignment.scene_id,
    )


def _insert_assignment_rows(
    connection: sqlite3.Connection,
    rows: list[tuple[object, ...]],
) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO face_tracking_assignments(
            observation_id, anchor_detection_id, frame, raw_track_id,
            final_track_id, removed_by_short_track, association_stage,
            association_score, head_score, face_score,
            head_x1, head_y1, head_x2, head_y2, scene_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    rows.clear()


def _mask_row(
    frame: int,
    track_id: str,
    mask: PrivacyMask,
    label: str,
) -> tuple[object, ...]:
    polygons = json.dumps(
        [[[float(x), float(y)] for x, y in mask.polygon]],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return frame, track_id, polygons, mask.shape, label


def _geometry_row(
    frame: int,
    track_id: str,
    mask: PrivacyMask,
) -> tuple[object, ...]:
    return (
        frame,
        track_id,
        mask.shape,
        float(mask.center[0]),
        float(mask.center[1]),
        float(mask.half_width),
        float(mask.half_height),
        float(mask.theta_radians),
    )


def _finalize_tracks(
    connection: sqlite3.Connection,
    summaries: Sequence[FaceTrackSummary],
    *,
    target: str,
) -> None:
    for summary in summaries:
        connection.execute(
            """
            INSERT INTO face_tracks(
                raw_track_id, final_track_id, scene_id, start_frame,
                end_frame, observed_frames, maximum_score, mean_score,
                removed_by_short_track, termination_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.raw_track_id,
                summary.final_track_id,
                summary.scene_id,
                summary.start_frame,
                summary.end_frame,
                summary.observed_frames,
                summary.maximum_score,
                summary.mean_score,
                int(summary.removed_by_short_track),
                summary.termination_reason,
            ),
        )
        connection.execute(
            """
            UPDATE face_tracking_assignments
            SET final_track_id=?, removed_by_short_track=?
            WHERE raw_track_id=?
            """,
            (
                summary.final_track_id,
                int(summary.removed_by_short_track),
                summary.raw_track_id,
            ),
        )
        if summary.removed_by_short_track:
            privacy_track_id = _privacy_track_id(summary.raw_track_id, target)
            connection.execute(
                "DELETE FROM masks WHERE track_id=?",
                (privacy_track_id,),
            )
            connection.execute(
                "DELETE FROM mask_provenance WHERE track_id=?",
                (privacy_track_id,),
            )
            connection.execute(
                "DELETE FROM face_mask_geometries WHERE track_id=?",
                (privacy_track_id,),
            )
            connection.execute(
                "DELETE FROM tracks WHERE track_id=?",
                (privacy_track_id,),
            )


def _interpolate_angle(first: float, second: float, ratio: float) -> float:
    delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return first + ratio * delta


def _polygon_from_geometry(
    geometry_type: str,
    cx: float,
    cy: float,
    half_width: float,
    half_height: float,
    theta: float,
    *,
    points: int,
) -> tuple[tuple[float, float], ...]:
    cosine = math.cos(theta)
    sine = math.sin(theta)
    if geometry_type == "rectangle":
        local = (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    else:
        local = tuple(
            (
                half_width * math.cos(2.0 * math.pi * index / points),
                half_height * math.sin(2.0 * math.pi * index / points),
            )
            for index in range(points)
        )
    return tuple(
        (
            cx + local_x * cosine - local_y * sine,
            cy + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in local
    )


def _interpolate_track_boxes(
    connection: sqlite3.Connection,
    *,
    maximum_gap: int,
) -> int:
    """Materialize only short, two-sided gaps in kept Head tracks."""

    if maximum_gap <= 0:
        return 0
    rows = connection.execute(
        """
        SELECT frame, final_track_id, scene_id, observation_id,
               head_x1, head_y1, head_x2, head_y2
        FROM face_tracking_assignments
        WHERE removed_by_short_track=0 AND final_track_id IS NOT NULL
        ORDER BY final_track_id, frame, observation_id
        """
    )
    previous: tuple[object, ...] | None = None
    batch: list[tuple[object, ...]] = []
    inserted = 0
    for current in rows:
        current_values = tuple(current)
        if previous is not None and str(previous[1]) == str(current_values[1]):
            first_frame = int(previous[0])
            second_frame = int(current_values[0])
            gap = second_frame - first_frame - 1
            if 0 < gap <= maximum_gap:
                for offset in range(1, gap + 1):
                    ratio = offset / float(gap + 1)
                    box = tuple(
                        float(previous[index])
                        + ratio
                        * (float(current_values[index]) - float(previous[index]))
                        for index in range(4, 8)
                    )
                    batch.append(
                        (
                            first_frame + offset,
                            str(current_values[1]),
                            int(current_values[2]),
                            int(previous[3]),
                            int(current_values[3]),
                            *box,
                            "linear-two-sided",
                        )
                    )
                    inserted += 1
        previous = current_values
        if len(batch) >= 1000:
            connection.executemany(
                """
                INSERT INTO face_track_interpolations(
                    frame, final_track_id, scene_id,
                    previous_observation_id, next_observation_id,
                    head_x1, head_y1, head_x2, head_y2,
                    interpolation_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            batch.clear()
    if batch:
        connection.executemany(
            """
            INSERT INTO face_track_interpolations(
                frame, final_track_id, scene_id,
                previous_observation_id, next_observation_id,
                head_x1, head_y1, head_x2, head_y2,
                interpolation_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
    return inserted


def _interpolate_short_gaps(
    connection: sqlite3.Connection,
    *,
    target: str,
    maximum_gap: int,
) -> int:
    if maximum_gap <= 0:
        return 0
    observed = list(
        connection.execute(
            """
            SELECT g.frame, g.track_id, g.geometry_type, g.cx, g.cy,
                   g.half_width, g.half_height, g.theta_radians,
                   p.source_observation_id, p.confidence,
                   COALESCE(t.label, '')
            FROM face_mask_geometries g
            JOIN mask_provenance p
              ON p.frame=g.frame AND p.track_id=g.track_id
            LEFT JOIN tracks t ON t.track_id=g.track_id
            WHERE p.is_interpolated=0
            ORDER BY g.track_id, g.frame
            """
        )
    )
    mask_rows: list[tuple[object, ...]] = []
    provenance_rows: list[tuple[object, ...]] = []
    geometry_rows: list[tuple[object, ...]] = []
    track_rows: list[tuple[str, str]] = []
    previous: sqlite3.Row | tuple[object, ...] | None = None
    inserted = 0
    for current in observed:
        if previous is not None and str(previous[1]) == str(current[1]):
            first_frame = int(previous[0])
            second_frame = int(current[0])
            gap = second_frame - first_frame - 1
            if 0 < gap <= maximum_gap and str(previous[2]) == str(current[2]):
                for offset in range(1, gap + 1):
                    ratio = offset / float(gap + 1)
                    values = tuple(
                        float(previous[index])
                        + ratio * (float(current[index]) - float(previous[index]))
                        for index in range(3, 7)
                    )
                    theta = _interpolate_angle(
                        float(previous[7]),
                        float(current[7]),
                        ratio,
                    )
                    polygon = _polygon_from_geometry(
                        str(current[2]),
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        theta,
                        points=96 if target == "face" else 64,
                    )
                    frame = first_frame + offset
                    track_id = str(current[1])
                    label = str(current[10])
                    mask_rows.append(
                        (
                            frame,
                            track_id,
                            json.dumps(
                                [[[float(x), float(y)] for x, y in polygon]],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            str(current[2]),
                            label,
                        )
                    )
                    provenance_rows.append(
                        (
                            frame,
                            track_id,
                            target,
                            int(previous[8]),
                            int(current[8]),
                            1,
                            "interpolated-linear",
                            min(float(previous[9]), float(current[9])),
                            ALGORITHM_VERSION,
                        )
                    )
                    geometry_rows.append(
                        (
                            frame,
                            track_id,
                            str(current[2]),
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            theta,
                        )
                    )
                    track_rows.append((track_id, label))
                    inserted += 1
        previous = current
    _insert_rows(
        connection,
        mask_rows,
        provenance_rows,
        geometry_rows,
        track_rows,
    )
    return inserted


def _sidecar_count(path: Path, table: str, where: str | None = None) -> int:
    with sqlite3.connect(path) as connection:
        suffix = "" if where is None else f" WHERE {where}"
        return int(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"{suffix}').fetchone()[0]
        )


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
                    source_observation_id INTEGER,
                    source_observation_id_end INTEGER,
                    is_interpolated INTEGER NOT NULL DEFAULT 0,
                    derivation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    PRIMARY KEY(frame, track_id)
                )
                """
            )
            provenance_columns = _columns(
                output_connection,
                "mask_provenance",
            )
            if "source_observation_id_end" not in provenance_columns:
                output_connection.execute(
                    """
                    ALTER TABLE mask_provenance
                    ADD COLUMN source_observation_id_end INTEGER
                    """
                )
            if "is_interpolated" not in provenance_columns:
                output_connection.execute(
                    """
                    ALTER TABLE mask_provenance
                    ADD COLUMN is_interpolated INTEGER NOT NULL DEFAULT 0
                    """
                )
            output_connection.execute(
                """
                INSERT INTO mask_provenance(
                    frame, track_id, mask_kind, source_observation_id,
                    source_observation_id_end, is_interpolated,
                    derivation, confidence, algorithm_version
                )
                SELECT frame, track_id, mask_kind, source_observation_id,
                       source_observation_id_end, is_interpolated,
                       derivation, confidence, algorithm_version
                FROM face_db.mask_provenance
                """
            )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_tracks(
                    raw_track_id TEXT PRIMARY KEY,
                    final_track_id TEXT,
                    scene_id INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    observed_frames INTEGER NOT NULL,
                    maximum_score REAL NOT NULL,
                    mean_score REAL NOT NULL,
                    removed_by_short_track INTEGER NOT NULL,
                    termination_reason TEXT NOT NULL
                )
                """
            )
            output_connection.execute(
                "INSERT INTO face_tracks SELECT * FROM face_db.face_tracks"
            )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_tracking_assignments(
                    observation_id INTEGER PRIMARY KEY,
                    anchor_detection_id INTEGER NOT NULL,
                    frame INTEGER NOT NULL,
                    raw_track_id TEXT NOT NULL,
                    final_track_id TEXT,
                    removed_by_short_track INTEGER NOT NULL DEFAULT 0,
                    association_stage TEXT NOT NULL,
                    association_score REAL,
                    head_score REAL NOT NULL,
                    face_score REAL NOT NULL,
                    head_x1 REAL NOT NULL,
                    head_y1 REAL NOT NULL,
                    head_x2 REAL NOT NULL,
                    head_y2 REAL NOT NULL,
                    scene_id INTEGER NOT NULL
                )
                """
            )
            output_connection.execute(
                """
                INSERT INTO face_tracking_assignments
                SELECT * FROM face_db.face_tracking_assignments
                """
            )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_track_interpolations(
                    frame INTEGER NOT NULL,
                    final_track_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    previous_observation_id INTEGER NOT NULL,
                    next_observation_id INTEGER NOT NULL,
                    head_x1 REAL NOT NULL,
                    head_y1 REAL NOT NULL,
                    head_x2 REAL NOT NULL,
                    head_y2 REAL NOT NULL,
                    interpolation_method TEXT NOT NULL,
                    PRIMARY KEY(frame, final_track_id)
                )
                """
            )
            output_connection.execute(
                """
                INSERT INTO face_track_interpolations
                SELECT * FROM face_db.face_track_interpolations
                """
            )
            output_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cut_detection_metadata(
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    schema_version INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    cut_count INTEGER NOT NULL,
                    frame_semantics TEXT NOT NULL
                )
                """
            )
            if "cuts" not in _tables(output_connection):
                output_connection.execute(
                    "CREATE TABLE cuts(frame INTEGER PRIMARY KEY)"
                )
            existing_cut_count = int(
                output_connection.execute("SELECT COUNT(*) FROM cuts").fetchone()[0]
            )
            if existing_cut_count == 0:
                output_connection.execute(
                    "INSERT INTO cuts SELECT frame FROM face_db.cuts"
                )
            existing_cut_metadata = output_connection.execute(
                "SELECT 1 FROM cut_detection_metadata WHERE id=1"
            ).fetchone()
            if existing_cut_metadata is None:
                output_connection.execute(
                    """
                    INSERT INTO cut_detection_metadata
                    SELECT * FROM face_db.cut_detection_metadata
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
