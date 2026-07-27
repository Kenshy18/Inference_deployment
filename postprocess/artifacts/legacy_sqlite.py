"""Export the legacy DINOv3 postprocess SQLite projection.

This compatibility artifact intentionally contains only the three tables used
by ``Dinov3_postprocess``'s user-facing ``AI後処理最終.sqlite`` contract:
``masks``, ``tracks``, and ``cuts``.  New audit tables remain available only
in the canonical ``predictions.sqlite``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any


LEGACY_SCHEMA_NAME = "dinov3_postprocess_final_mask_sqlite_v1"
LEGACY_TABLES = ("masks", "tracks", "cuts")
LEGACY_MASK_COLUMNS = (
    "frame",
    "track_id",
    "polygons",
    "shape_type",
    "dilate_px",
    "feather_px",
    "mosaic_block",
    "mosaic_alias",
    "label",
)


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _create_legacy_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE masks(
            frame INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            polygons TEXT,
            shape_type TEXT,
            dilate_px INTEGER NOT NULL DEFAULT 0,
            feather_px INTEGER NOT NULL DEFAULT 0,
            mosaic_block INTEGER NOT NULL DEFAULT 0,
            mosaic_alias REAL NOT NULL DEFAULT 0,
            label TEXT,
            PRIMARY KEY(frame, track_id)
        );
        CREATE TABLE tracks(
            track_id TEXT PRIMARY KEY,
            label TEXT
        );
        CREATE TABLE cuts(
            frame INTEGER PRIMARY KEY
        );
        """
    )


def export_legacy_sqlite(
    input_sqlite: Path,
    output_sqlite: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Project a canonical final SQLite into the legacy three-table schema."""

    source = Path(input_sqlite).expanduser().resolve()
    output = Path(output_sqlite).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("legacy output SQLite must differ from its input")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")

    mask_rows = track_rows = cut_rows = 0
    source_connection = sqlite3.connect(
        f"file:{source}?mode=ro",
        uri=True,
    )
    output_connection = sqlite3.connect(str(temporary))
    try:
        source_tables = _tables(source_connection)
        if "masks" not in source_tables:
            raise ValueError(f"{source}: masks table is absent")
        mask_columns = _columns(source_connection, "masks")
        required_mask_columns = {"frame", "track_id", "polygons"}
        missing = required_mask_columns - mask_columns
        if missing:
            raise ValueError(f"{source}: masks columns are missing: {sorted(missing)}")

        _create_legacy_schema(output_connection)
        shape_type = (
            "COALESCE(shape_type, 'polygon')"
            if "shape_type" in mask_columns
            else "'polygon'"
        )
        dilate_px = "COALESCE(dilate_px, 0)" if "dilate_px" in mask_columns else "0"
        feather_px = "COALESCE(feather_px, 0)" if "feather_px" in mask_columns else "0"
        mosaic_block = (
            "COALESCE(mosaic_block, 0)" if "mosaic_block" in mask_columns else "0"
        )
        mosaic_alias = (
            "COALESCE(mosaic_alias, 0.0)" if "mosaic_alias" in mask_columns else "0.0"
        )
        label = "label" if "label" in mask_columns else "NULL"
        masks = source_connection.execute(
            f"""
            SELECT frame, track_id, polygons, {shape_type}, {dilate_px},
                   {feather_px}, {mosaic_block}, {mosaic_alias}, {label}
            FROM masks
            ORDER BY frame, track_id
            """
        )
        for row in masks:
            frame, track_id, polygons = row[:3]
            if frame is None or int(frame) < 0:
                raise ValueError(f"{source}: invalid mask frame {frame!r}")
            if track_id is None or not str(track_id):
                raise ValueError(f"{source}: empty mask track_id")
            decoded = json.loads(str(polygons))
            if not isinstance(decoded, list):
                raise ValueError(
                    f"{source}: polygons must decode to a list at "
                    f"frame={frame}, track_id={track_id}"
                )
            output_connection.execute(
                """
                INSERT INTO masks(
                    frame, track_id, polygons, shape_type, dilate_px,
                    feather_px, mosaic_block, mosaic_alias, label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            mask_rows += 1

        labels_by_track: dict[str, str | None] = {}
        if "tracks" in source_tables:
            track_columns = _columns(source_connection, "tracks")
            if "track_id" in track_columns:
                track_label = "label" if "label" in track_columns else "NULL"
                for track_id, track_label_value in source_connection.execute(
                    f"SELECT track_id, {track_label} FROM tracks " "ORDER BY track_id"
                ):
                    labels_by_track[str(track_id)] = (
                        None if track_label_value is None else str(track_label_value)
                    )
        for track_id, row_label in output_connection.execute(
            "SELECT track_id, label FROM masks ORDER BY track_id, frame"
        ):
            key = str(track_id)
            label_value = None if row_label is None else str(row_label)
            if key not in labels_by_track or (
                labels_by_track[key] is None and label_value is not None
            ):
                labels_by_track[key] = label_value
        if labels_by_track:
            output_connection.executemany(
                "INSERT INTO tracks(track_id, label) VALUES (?, ?)",
                sorted(labels_by_track.items()),
            )
        track_rows = len(labels_by_track)

        if "cuts" in source_tables and "frame" in _columns(source_connection, "cuts"):
            cuts = [
                (int(row[0]),)
                for row in source_connection.execute(
                    "SELECT frame FROM cuts ORDER BY frame"
                )
            ]
            if any(frame < 0 for (frame,) in cuts):
                raise ValueError(f"{source}: cut frames must be non-negative")
            if cuts:
                output_connection.executemany(
                    "INSERT INTO cuts(frame) VALUES (?)",
                    cuts,
                )
            cut_rows = len(cuts)

        output_connection.commit()
        integrity = str(
            output_connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        if integrity != "ok":
            raise RuntimeError(f"legacy SQLite integrity check failed: {integrity}")
    except BaseException:
        try:
            output_connection.rollback()
        finally:
            source_connection.close()
            output_connection.close()
            temporary.unlink(missing_ok=True)
        raise
    else:
        source_connection.close()
        output_connection.close()

    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "schema": LEGACY_SCHEMA_NAME,
        "input_sqlite": str(source),
        "output_sqlite": str(output),
        "tables": list(LEGACY_TABLES),
        "mask_columns": list(LEGACY_MASK_COLUMNS),
        "masks": mask_rows,
        "tracks": track_rows,
        "cuts": cut_rows,
        "size_bytes": output.stat().st_size,
    }
