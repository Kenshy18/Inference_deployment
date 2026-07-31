"""Materialize sparse rich-face primitives for the native CUDA renderer."""

from __future__ import annotations

import math
import os
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np

from .face_privacy import derive_privacy_mask
from .models import FaceMaskOverlay
from .sources import iter_face_frames


SCHEMA_NAME = "overlay-fast-face-cache"
SCHEMA_VERSION = 1


def _probability_mask_dots(
    mask: FaceMaskOverlay,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    x1, y1, x2, y2 = mask.box
    left = int(math.floor(x1))
    top = int(math.floor(y1))
    right = int(math.ceil(x2))
    bottom = int(math.ceil(y2))
    if right <= left or bottom <= top:
        return np.empty((0, 2), dtype=np.float32)
    probability = np.frombuffer(mask.probabilities, dtype=np.uint8).reshape(
        mask.height,
        mask.width,
    )
    binary = np.asarray(
        probability >= round(threshold * 255.0),
        dtype=np.uint8,
    )
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    spacing = 5.0
    scale = np.asarray(
        (
            (right - left) / float(mask.width),
            (bottom - top) / float(mask.height),
        ),
        dtype=np.float32,
    )
    minimum_output_area = max(
        4.0,
        (right - left) * (bottom - top) * 0.0002,
    )
    offset = np.asarray((left, top), dtype=np.float32)
    dot_groups: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) * float(scale[0] * scale[1]) < (
            minimum_output_area
        ):
            continue
        points = (contour.reshape(-1, 2).astype(np.float32) + 0.5) * scale + offset
        if len(points) < 2:
            continue
        closed = np.concatenate((points, points[:1]), axis=0)
        segments = closed[1:] - closed[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(lengths)))
        total = float(cumulative[-1])
        if total <= 0.0:
            continue
        distances = np.arange(0.0, total, spacing, dtype=np.float32)
        segment_indices = (
            np.searchsorted(
                cumulative,
                distances,
                side="right",
            )
            - 1
        )
        np.clip(
            segment_indices,
            0,
            len(segments) - 1,
            out=segment_indices,
        )
        fractions = (distances - cumulative[segment_indices]) / np.maximum(
            lengths[segment_indices], 1e-6
        )
        dot_groups.append(
            closed[segment_indices]
            + fractions[:, np.newaxis] * segments[segment_indices]
        )
    if not dot_groups:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(dot_groups).astype(np.float32, copy=False)


def materialize_fast_face_cache(
    source: Path,
    output: Path,
    *,
    display_style: str,
    start_frame: int,
    end_frame: int | None,
    include_ellipses: bool = True,
    include_keypoints: bool = True,
    include_probability_masks: bool = True,
    draw_ellipses: bool | None = None,
    draw_keypoints: bool | None = None,
    face_privacy_target: str = "none",
    eye_mask_shape: str = "ellipse",
    minimum_eye_confidence: float = 0.35,
) -> dict[str, object]:
    """Create an immutable, range-bounded cache without copying video frames."""

    if display_style not in {"detailed", "simple", "legacy"}:
        raise ValueError(f"unsupported face display style: {display_style}")
    if draw_ellipses is None:
        draw_ellipses = include_ellipses
    if draw_keypoints is None:
        draw_keypoints = include_keypoints
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    item_rows: list[tuple[object, ...]] = []
    keypoint_rows: list[tuple[object, ...]] = []
    dot_count = 0
    privacy_point_count = 0
    item_id = 0
    frame_count = 0
    first_frame: int | None = None
    last_frame: int | None = None
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                CREATE TABLE fast_face_cache_info(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE fast_face_items(
                    id INTEGER PRIMARY KEY,
                    frame INTEGER NOT NULL,
                    track_id TEXT,
                    label TEXT NOT NULL,
                    score REAL,
                    face_score REAL,
                    face_present INTEGER,
                    provenance TEXT,
                    detailed INTEGER NOT NULL,
                    box_x1 REAL,
                    box_y1 REAL,
                    box_x2 REAL,
                    box_y2 REAL,
                    ellipse_cx REAL,
                    ellipse_cy REAL,
                    ellipse_radius_x REAL,
                    ellipse_radius_y REAL,
                    ellipse_theta REAL,
                    privacy_target TEXT,
                    privacy_shape TEXT,
                    privacy_derivation TEXT,
                    privacy_confidence REAL,
                    mask_dots_i16 BLOB,
                    privacy_points_f32 BLOB
                );
                CREATE TABLE fast_face_keypoints(
                    item_id INTEGER NOT NULL,
                    point_index INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    class_name TEXT NOT NULL,
                    state INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    state_confidence REAL,
                    valid INTEGER NOT NULL,
                    PRIMARY KEY(item_id, point_index)
                );
                CREATE INDEX idx_fast_face_items_frame
                    ON fast_face_items(frame, id);
                CREATE TABLE cuts(frame INTEGER PRIMARY KEY) WITHOUT ROWID;
                """
            )
            connection.executemany(
                "INSERT INTO fast_face_cache_info(key, value) VALUES (?, ?)",
                (
                    ("schema_name", SCHEMA_NAME),
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("source", str(source)),
                    ("display_style", display_style),
                    ("start_frame", str(start_frame)),
                    ("end_frame", "" if end_frame is None else str(end_frame)),
                ),
            )
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_db:
                has_cuts = source_db.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='cuts'"
                ).fetchone()
                if has_cuts is not None:
                    limit = (2**31 - 1) if end_frame is None else end_frame
                    connection.executemany(
                        "INSERT INTO cuts(frame) VALUES (?)",
                        source_db.execute(
                            "SELECT frame FROM cuts WHERE frame BETWEEN ? AND ?",
                            (start_frame, limit),
                        ),
                    )
            frames = iter_face_frames(
                source,
                include_ellipses=include_ellipses,
                include_keypoints=include_keypoints,
                include_probability_masks=(
                    include_probability_masks and display_style == "detailed"
                ),
                display_style=display_style,
            )
            for frame in frames:
                if frame.frame_index < start_frame:
                    continue
                if end_frame is not None and frame.frame_index > end_frame:
                    break
                frame_count += 1
                first_frame = frame.frame_index if first_frame is None else first_frame
                last_frame = frame.frame_index
                for item in frame.items:
                    item_id += 1
                    box = item.box
                    ellipse = item.ellipse
                    privacy = derive_privacy_mask(
                        face_privacy_target,
                        item.ellipse,
                        item.keypoints,
                        eye_shape=eye_mask_shape,
                        minimum_eye_confidence=minimum_eye_confidence,
                    )
                    dots = (
                        _probability_mask_dots(item.face_mask)
                        if item.face_mask is not None
                        else np.empty((0, 2), dtype=np.float32)
                    )
                    dot_count += len(dots)
                    privacy_points = privacy.polygon if privacy is not None else ()
                    privacy_point_count += len(privacy_points)
                    item_rows.append(
                        (
                            item_id,
                            frame.frame_index,
                            item.track_id,
                            item.label,
                            item.score,
                            item.face_score,
                            (
                                None
                                if item.face_present is None
                                else int(item.face_present)
                            ),
                            item.provenance,
                            int(display_style == "detailed"),
                            *(box if box is not None else (None,) * 4),
                            *(
                                ellipse
                                if ellipse is not None and draw_ellipses
                                else (None,) * 5
                            ),
                            *(
                                (
                                    privacy.target,
                                    privacy.shape,
                                    privacy.derivation,
                                    privacy.confidence,
                                )
                                if privacy is not None
                                else (None,) * 4
                            ),
                            (
                                np.asarray(
                                    np.rint(dots),
                                    dtype="<i2",
                                ).tobytes()
                                if dots.size
                                else None
                            ),
                            (
                                np.asarray(
                                    privacy_points,
                                    dtype="<f4",
                                ).tobytes()
                                if privacy_points
                                else None
                            ),
                        )
                    )
                    if draw_keypoints:
                        keypoint_rows.extend(
                            (
                                item_id,
                                index,
                                point.x,
                                point.y,
                                point.class_name,
                                point.state,
                                point.confidence,
                                point.state_confidence,
                                int(point.valid),
                            )
                            for index, point in enumerate(item.keypoints)
                        )
                    if len(item_rows) >= 1000:
                        _insert_rows(
                            connection,
                            item_rows,
                            keypoint_rows,
                        )
            _insert_rows(
                connection,
                item_rows,
                keypoint_rows,
            )
            connection.commit()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output": str(output),
        "display_style": display_style,
        "frames_with_faces": frame_count,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "items": item_id,
        "keypoints": _count(output, "fast_face_keypoints"),
        "probability_mask_dots": dot_count,
        "privacy_polygon_points": privacy_point_count,
        "size_bytes": output.stat().st_size,
    }


def _insert_rows(
    connection: sqlite3.Connection,
    items: list[tuple[object, ...]],
    keypoints: list[tuple[object, ...]],
) -> None:
    if items:
        connection.executemany(
            """
            INSERT INTO fast_face_items VALUES(
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            items,
        )
        items.clear()
    if keypoints:
        connection.executemany(
            "INSERT INTO fast_face_keypoints VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            keypoints,
        )
        keypoints.clear()


def _count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


__all__ = ["materialize_fast_face_cache"]
