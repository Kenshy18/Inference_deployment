"""Bounded-memory SQLite writers for dense curves and editable P keyframes.

Curve outputs deliberately contain only the public mask contract.  Copying a
tracked SQLite here would retain inference and tracking audit tables that are
not consumed by the evaluator, classwise merger, or editable-geometry import.
On long videos that used to create two full-size, mostly-empty database copies
for every curve job.  The original tracked SQLite remains a separate pipeline
artifact and is passed explicitly to every consumer that needs provenance.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

from .config import CURVE_CONTRACT, INTERPOLATION_METHOD, PROFILE_ID


class MaskSqliteWriter:
    """Stream mask rows into an atomic schema-preserving SQLite artifact."""

    def __init__(self, output: Path, reference: Path, *, commit_rows: int = 2000):
        self.output = Path(output).resolve()
        self.reference = Path(reference).resolve()
        if self.output == self.reference:
            raise ValueError("curve output SQLite must differ from its reference")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(self.output)
        if not self.reference.is_file():
            raise FileNotFoundError(self.reference)
        self.temporary = self.output.with_name(
            f".{self.output.name}.{uuid.uuid4().hex}.tmp"
        )
        self.connection = sqlite3.connect(self.temporary)
        # This is an unpublished temporary file.  Atomic rename happens only
        # after a complete integrity check, so a disk-level rollback journal
        # adds I/O without improving the published artifact's safety.
        self.connection.execute("PRAGMA journal_mode=MEMORY")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.executescript(
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
            );
            CREATE TABLE tracks(
                track_id TEXT PRIMARY KEY,
                label TEXT
            );
            """
        )
        self.connection.commit()
        self.commit_rows = max(1, int(commit_rows))
        self.pending: list[tuple[int, str, str, str]] = []
        self.rows = 0
        self._closed = False

    def _flush(self) -> None:
        if not self.pending:
            return
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO masks(
                frame, track_id, polygons, shape_type, dilate_px, feather_px,
                mosaic_block, mosaic_alias, label
            ) VALUES (?, ?, ?, 'polygon', 0, 0, 0, 0, ?)
            """,
            self.pending,
        )
        labels: dict[str, str] = {}
        for _frame, track_id, _polygons, label in self.pending:
            labels.setdefault(track_id, label)
        self.connection.executemany(
            "INSERT OR REPLACE INTO tracks(track_id,label) VALUES (?,?)",
            tuple(labels.items()),
        )
        self.connection.commit()
        self.pending.clear()

    def append(
        self,
        *,
        frame: int,
        track_id: str,
        polygons: object,
        label: str,
    ) -> None:
        payload = (
            polygons
            if isinstance(polygons, str)
            else json.dumps(polygons, ensure_ascii=False, separators=(",", ":"))
        )
        self.pending.append((int(frame), str(track_id), str(payload), str(label)))
        self.rows += 1
        if len(self.pending) >= self.commit_rows:
            self._flush()

    def add_curve_metadata(self, *, point_policy: dict[str, object]) -> None:
        self._flush()
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS polygon_keyframe_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        values = {
            "interpolation_method": INTERPOLATION_METHOD,
            "profile": PROFILE_ID,
            "curve_contract": CURVE_CONTRACT,
            "editable_variables": "catmull_rom_interpolation_points_P_only",
            "derived_bezier_handles": "true",
            "catmull_rom_tension": "1.0",
            "catmull_rom_bezier_factor": "0.16666666666666666",
            "samples_per_segment": "16",
            "point_policy": json.dumps(
                point_policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        self.connection.executemany(
            "INSERT OR REPLACE INTO polygon_keyframe_metadata(key,value) VALUES (?,?)",
            tuple(sorted(values.items())),
        )
        # ``finalize()`` may have no pending mask rows left, in which case
        # ``_flush()`` intentionally returns without committing.  Persist the
        # interpolation contract here so classwise/unified exporters never
        # silently fall back to linear polygon interpolation.
        self.connection.commit()

    def finalize(self) -> Path:
        if self._closed:
            return self.output
        self._flush()
        integrity = tuple(
            str(row[0]) for row in self.connection.execute("PRAGMA integrity_check")
        )
        foreign_keys = tuple(self.connection.execute("PRAGMA foreign_key_check"))
        self.connection.close()
        self._closed = True
        if integrity != ("ok",) or foreign_keys:
            self.abort()
            raise RuntimeError(
                "invalid curve SQLite: "
                f"integrity={integrity}, foreign_keys={len(foreign_keys)}"
            )
        os.replace(self.temporary, self.output)
        return self.output

    def abort(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True
        for candidate in (
            self.temporary,
            Path(f"{self.temporary}-wal"),
            Path(f"{self.temporary}-shm"),
        ):
            candidate.unlink(missing_ok=True)


__all__ = ("MaskSqliteWriter",)
