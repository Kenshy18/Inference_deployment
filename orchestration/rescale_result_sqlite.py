"""Publish an integrated result SQLite in the original video's pixel space."""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoGeometry:
    width: int
    height: int
    fps: float
    frame_count: int


_COORDINATE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "detections": (("x1", "x"), ("y1", "y"), ("x2", "x"), ("y2", "y")),
    "segmentation_points": (("x", "x"), ("y", "y")),
    "face_keypoints": (("x", "x"), ("y", "y")),
    "face_masks": (
        ("box_x1", "x"),
        ("box_y1", "y"),
        ("box_x2", "x"),
        ("box_y2", "y"),
    ),
    "face_observations": (
        ("ellipse_cx", "x"),
        ("ellipse_cy", "y"),
        ("ellipse_major_radius", "uniform"),
        ("ellipse_minor_radius", "uniform"),
    ),
    "face_track_interpolations": (
        ("head_x1", "x"),
        ("head_y1", "y"),
        ("head_x2", "x"),
        ("head_y2", "y"),
    ),
    "face_tracking_assignments": (
        ("head_x1", "x"),
        ("head_y1", "y"),
        ("head_x2", "x"),
        ("head_y2", "y"),
    ),
    "keyframe_ellipses": (
        ("cx", "x"),
        ("cy", "y"),
        ("radius_x", "x"),
        ("radius_y", "y"),
    ),
    "keyframe_polygon_points": (("x", "x"), ("y", "y")),
    "keyframe_rectangles": (
        ("cx", "x"),
        ("cy", "y"),
        ("half_width", "x"),
        ("half_height", "y"),
    ),
}


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _require_schema(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    required = set(_COORDINATE_COLUMNS) | {"frames", "videos", "video_streams"}
    missing = sorted(required - tables)
    if missing:
        raise ValueError(f"result SQLite cannot be rescaled; missing tables={missing}")
    for table, columns in _COORDINATE_COLUMNS.items():
        available = _table_columns(connection, table)
        missing_columns = sorted(
            column for column, _axis in columns if column not in available
        )
        if missing_columns:
            raise ValueError(
                f"result SQLite cannot be rescaled; {table} missing {missing_columns}"
            )


def _update_model_metadata(
    connection: sqlite3.Connection,
    *,
    original_video: Path,
    original: VideoGeometry,
) -> None:
    if not _table_columns(connection, "model_metadata"):
        return
    replacements = {
        "input": (str(original_video), "str"),
        "video.width": (str(original.width), "int"),
        "video.height": (str(original.height), "int"),
        "video.frames": (str(original.frame_count), "int"),
        "video.fps": (str(original.fps), "float"),
    }
    for key, (value, value_type) in replacements.items():
        connection.execute(
            "UPDATE model_metadata SET value=?, value_type=? WHERE key=?",
            (value, value_type, key),
        )


def _record_transform(
    connection: sqlite3.Connection,
    *,
    proxy: VideoGeometry,
    original: VideoGeometry,
    scale_x: float,
    scale_y: float,
) -> None:
    if not _table_columns(connection, "run_metadata"):
        return
    values = {
        "analysis_proxy.width": (proxy.width, "int"),
        "analysis_proxy.height": (proxy.height, "int"),
        "analysis_proxy.scale_x_to_source": (scale_x, "float"),
        "analysis_proxy.scale_y_to_source": (scale_y, "float"),
        "analysis_proxy.source_width": (original.width, "int"),
        "analysis_proxy.source_height": (original.height, "int"),
    }
    for key, (value, value_type) in values.items():
        connection.execute(
            "INSERT OR REPLACE INTO run_metadata(key, value, value_type) VALUES (?, ?, ?)",
            (key, str(value), value_type),
        )


def rescale_result_sqlite(
    source: Path,
    destination: Path,
    *,
    proxy: VideoGeometry,
    original: VideoGeometry,
    original_video: Path,
) -> dict[str, object]:
    """Copy *source* and convert every public geometry to source-video pixels."""

    if proxy.frame_count != original.frame_count:
        raise ValueError(
            "analysis proxy frame count differs from source: "
            f"proxy={proxy.frame_count}, source={original.frame_count}"
        )
    if not math.isclose(proxy.fps, original.fps, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(
            f"analysis proxy fps differs from source: proxy={proxy.fps}, source={original.fps}"
        )
    scale_x = original.width / proxy.width
    scale_y = original.height / proxy.height
    if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "analysis proxy must use a uniform scale for ellipse geometry: "
            f"scale_x={scale_x}, scale_y={scale_y}"
        )

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    try:
        with sqlite3.connect(temporary) as connection:
            _require_schema(connection)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            for table, columns in _COORDINATE_COLUMNS.items():
                assignments = []
                parameters: list[float] = []
                for column, axis in columns:
                    factor = scale_x if axis in {"x", "uniform"} else scale_y
                    assignments.append(f'"{column}" = "{column}" * ?')
                    parameters.append(factor)
                connection.execute(
                    f'UPDATE "{table}" SET {", ".join(assignments)}',
                    parameters,
                )
            connection.execute(
                "UPDATE frames SET width=?, height=?",
                (original.width, original.height),
            )
            connection.execute(
                "UPDATE videos SET path=?, reported_frame_count=?, fps=?, width=?, height=?",
                (
                    str(original_video),
                    original.frame_count,
                    original.fps,
                    original.width,
                    original.height,
                ),
            )
            connection.execute(
                "UPDATE video_streams SET width=?, height=?, frame_count=?",
                (original.width, original.height, original.frame_count),
            )
            _update_model_metadata(
                connection,
                original_video=original_video,
                original=original,
            )
            _record_transform(
                connection,
                proxy=proxy,
                original=original,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            bad_frames = connection.execute(
                "SELECT COUNT(*) FROM frames WHERE width != ? OR height != ?",
                (original.width, original.height),
            ).fetchone()[0]
            if bad_frames:
                raise ValueError(
                    f"rescaled SQLite has {bad_frames} invalid frame dimensions"
                )
            connection.commit()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_width": original.width,
        "source_height": original.height,
        "proxy_width": proxy.width,
        "proxy_height": proxy.height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "frame_count": original.frame_count,
        "fps": original.fps,
    }


__all__ = ["VideoGeometry", "rescale_result_sqlite"]
