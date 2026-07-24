"""Validation for tracked and final prediction SQLite artifacts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


class OutputContractError(ValueError):
    """Raised when an SQLite artifact violates the public contract."""


@dataclass(frozen=True)
class OutputStats:
    masks: int
    tracks: int
    first_frame: int | None
    last_frame: int | None

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


def validate_mask_sqlite(path: Path) -> OutputStats:
    if not path.is_file():
        raise OutputContractError(f"SQLite file not found: {path}")
    connection = sqlite3.connect(str(path))
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = {"masks"} - tables
        if missing_tables:
            raise OutputContractError(
                f"missing required table(s): {', '.join(sorted(missing_tables))}"
            )

        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(masks)")
        }
        missing_columns = {"frame", "track_id", "polygons"} - columns
        if missing_columns:
            raise OutputContractError(
                f"masks is missing column(s): {', '.join(sorted(missing_columns))}"
            )

        mask_count = 0
        first_frame: int | None = None
        last_frame: int | None = None
        for frame, track_id, polygons in connection.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id"
        ):
            if frame is None or int(frame) < 0:
                raise OutputContractError(f"invalid frame: {frame!r}")
            if track_id is None or not str(track_id):
                raise OutputContractError(f"empty track_id at frame {frame}")
            try:
                decoded = json.loads(str(polygons))
            except (TypeError, json.JSONDecodeError) as exc:
                raise OutputContractError(
                    f"invalid polygons JSON at frame={frame}, track_id={track_id}"
                ) from exc
            if not isinstance(decoded, list):
                raise OutputContractError(
                    f"polygons must be a list at frame={frame}, track_id={track_id}"
                )
            value = int(frame)
            first_frame = value if first_frame is None else min(first_frame, value)
            last_frame = value if last_frame is None else max(last_frame, value)
            mask_count += 1

        track_count = (
            int(connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
            if "tracks" in tables
            else 0
        )
        return OutputStats(mask_count, track_count, first_frame, last_frame)
    finally:
        connection.close()
