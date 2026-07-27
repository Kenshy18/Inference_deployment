#!/usr/bin/env python3
"""Clone a postprocess mask SQLite while scaling polygon coordinates."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--scale-x", required=True, type=float)
    result.add_argument("--scale-y", required=True, type=float)
    result.add_argument("--overwrite", action="store_true")
    return result


def scale_polygons(value: Any, scale_x: float, scale_y: float) -> str:
    polygons = json.loads(str(value))
    for polygon in polygons:
        for point in polygon:
            point[0] = float(point[0]) * scale_x
            point[1] = float(point[1]) * scale_y
    return json.dumps(polygons, ensure_ascii=False, separators=(",", ":"))


def clone_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def main() -> None:
    args = parser().parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.scale_x <= 0 or args.scale_y <= 0:
        raise ValueError("scale factors must be positive")
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(destination)
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone_database(source, destination)

    connection = sqlite3.connect(destination)
    try:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(masks)")
        }
        if not {"frame", "track_id", "polygons"}.issubset(columns):
            raise ValueError(f"{source}: unsupported masks table")
        rows = connection.execute(
            "SELECT frame, track_id, polygons FROM masks WHERE polygons IS NOT NULL"
        ).fetchall()
        scaled = [
            (
                scale_polygons(polygons, args.scale_x, args.scale_y),
                frame,
                track_id,
            )
            for frame, track_id, polygons in rows
        ]
        connection.executemany(
            "UPDATE masks SET polygons=? WHERE frame=? AND track_id=?",
            scaled,
        )
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"scaled SQLite integrity check failed: {integrity}")
        first = connection.execute(
            """
            SELECT frame, track_id, polygons
            FROM masks
            WHERE polygons IS NOT NULL
            ORDER BY frame, track_id
            LIMIT 1
            """
        ).fetchone()
        summary = {
            "input": str(source),
            "output": str(destination),
            "scale_x": args.scale_x,
            "scale_y": args.scale_y,
            "masks_scaled": len(scaled),
            "first_scaled_mask": (
                None
                if first is None
                else {
                    "frame": int(first[0]),
                    "track_id": str(first[1]),
                    "first_point": json.loads(str(first[2]))[0][0],
                }
            ),
            "integrity_check": integrity,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
