"""Public contract I/O for mask SQLite artifacts.

All feature stages use this module instead of another feature's runtime.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MaskRow:
    frame: int
    track_id: str
    polygons: str
    label: str = ""
    shape_type: str = "polygon"


def track_sort_key(track_id: str) -> tuple[int, int | str]:
    text = str(track_id)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def read_mask_rows(path: Path) -> list[MaskRow]:
    with sqlite3.connect(str(path)) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(masks)").fetchall()
        }
        if not {"frame", "track_id", "polygons"}.issubset(columns):
            raise ValueError(f"{path}: masks table does not satisfy the contract")
        label = "COALESCE(label, '')" if "label" in columns else "''"
        shape = (
            "COALESCE(shape_type, 'polygon')"
            if "shape_type" in columns
            else "'polygon'"
        )
        rows = connection.execute(
            f"""
            SELECT frame, track_id, polygons, {label}, {shape}
            FROM masks
            ORDER BY track_id, frame
            """
        ).fetchall()
    output = [
        MaskRow(
            frame=int(frame),
            track_id=str(track_id),
            polygons=str(polygons),
            label=str(row_label),
            shape_type=str(shape_type),
        )
        for frame, track_id, polygons, row_label, shape_type in rows
    ]
    output.sort(key=lambda row: (track_sort_key(row.track_id), row.frame))
    return output


def write_mask_sqlite(
    output_path: Path,
    rows: Iterable[MaskRow],
    *,
    reference_sqlite: Path | None = None,
) -> Path:
    """Write masks while preserving non-mask audit tables from a reference."""

    output_path = Path(output_path)
    reference = None if reference_sqlite is None else Path(reference_sqlite)
    if reference is not None and output_path.resolve() == reference.resolve():
        raise ValueError("output SQLite must differ from reference SQLite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    if reference is not None:
        if not reference.is_file():
            raise FileNotFoundError(reference)
        with sqlite3.connect(str(reference)) as source_connection:
            with sqlite3.connect(str(output_path)) as output_connection:
                source_connection.backup(output_connection)

    with sqlite3.connect(str(output_path)) as connection:
        if reference is None:
            connection.execute(
                """
                CREATE TABLE masks(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    polygons TEXT NOT NULL,
                    shape_type TEXT,
                    dilate_px INTEGER NOT NULL DEFAULT 0,
                    feather_px INTEGER NOT NULL DEFAULT 0,
                    mosaic_block INTEGER NOT NULL DEFAULT 0,
                    mosaic_alias REAL NOT NULL DEFAULT 0,
                    label TEXT,
                    PRIMARY KEY(frame, track_id)
                )
                """
            )
            connection.execute(
                "CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)"
            )
        else:
            connection.execute("DELETE FROM masks")
        materialized = list(rows)
        connection.executemany(
            """
            INSERT OR REPLACE INTO masks(
                frame, track_id, polygons, shape_type, dilate_px, feather_px,
                mosaic_block, mosaic_alias, label
            )
            VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?)
            """,
            [
                (
                    row.frame,
                    row.track_id,
                    row.polygons,
                    row.shape_type,
                    row.label,
                )
                for row in materialized
            ],
        )
        if reference is None:
            labels: dict[str, str] = {}
            for row in materialized:
                labels.setdefault(row.track_id, row.label)
            connection.executemany(
                "INSERT OR REPLACE INTO tracks(track_id, label) VALUES (?, ?)",
                sorted(labels.items(), key=lambda item: track_sort_key(item[0])),
            )
    return output_path
