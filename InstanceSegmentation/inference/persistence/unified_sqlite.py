"""Lifecycle for one normalized multi-model SQLite inference result."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .candidate_import import (
    ImportedModelSummary,
    import_candidate_database,
)
from .metadata import flatten_metadata
from .schema_v3 import create_indexes, initialize_schema


class UnifiedSqliteWriter:
    """Own a schema-v3 database for exactly one orchestrated video run."""

    def __init__(
        self,
        path: Path,
        *,
        input_path: Path,
        mode: str,
        overwrite: bool = False,
        safe: bool = True,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(f"output already exists: {self.path}")
            self.path.unlink()
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        if safe:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        else:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA foreign_keys=ON")
        initialize_schema(self.connection)
        self.connection.execute(
            "INSERT INTO videos(id, path) VALUES (1, ?)",
            (str(input_path.expanduser().resolve()),),
        )
        self.connection.execute(
            """
            INSERT INTO runs(id, video_id, mode, created_at_utc)
            VALUES (1, 1, ?, ?)
            """,
            (mode, datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()
        self.closed = False

    def set_run_metadata(self, values: Mapping[str, object]) -> None:
        self._require_open()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO run_metadata(key, value, value_type)
            VALUES (?, ?, ?)
            """,
            flatten_metadata(values),
        )
        self.connection.commit()

    def import_model_output(
        self,
        source_path: Path,
        *,
        role: str,
        model_id: str,
        backend: str,
    ) -> ImportedModelSummary:
        self._require_open()
        return import_candidate_database(
            self.connection,
            source_path,
            role=role,
            model_id=model_id,
            backend=backend,
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            create_indexes(self.connection)
            self.connection.commit()
            foreign_key_errors = self.connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if foreign_key_errors:
                raise RuntimeError(
                    "unified SQLite foreign-key errors: " f"{foreign_key_errors[:3]}"
                )
            integrity = str(
                self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if integrity != "ok":
                raise RuntimeError(
                    f"unified SQLite integrity check failed: {integrity}"
                )
        finally:
            self.connection.close()
            self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("unified SQLite writer is closed")


__all__ = ["ImportedModelSummary", "UnifiedSqliteWriter"]
