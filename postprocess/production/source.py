"""Read-only source metadata shared by Production geometry stages."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def source_labels(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
        }
        if "tracks" not in tables:
            return ()
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(tracks)")}
        if "label" not in columns:
            return ()
        return tuple(
            str(row[0])
            for row in db.execute(
                "SELECT DISTINCT COALESCE(label, '') FROM tracks ORDER BY 1"
            )
        )


def source_dimensions(
    path: Path,
    *,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
        }
        if "frames" not in tables:
            if fallback_width <= 0 or fallback_height <= 0:
                raise RuntimeError(f"source dimensions are unavailable: {path}")
            return fallback_width, fallback_height
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(frames)")}
        if not {"width", "height"}.issubset(columns):
            if fallback_width <= 0 or fallback_height <= 0:
                raise RuntimeError(f"source dimensions are unavailable: {path}")
            return fallback_width, fallback_height
        frame_column = "frame_index" if "frame_index" in columns else "frame"
        row = db.execute(
            f"SELECT width,height FROM frames ORDER BY {frame_column} LIMIT 1"
        ).fetchone()
    if row is None or int(row[0] or 0) <= 0 or int(row[1] or 0) <= 0:
        raise RuntimeError(f"source dimensions are unavailable: {path}")
    return int(row[0]), int(row[1])


__all__ = ("source_dimensions", "source_labels")
