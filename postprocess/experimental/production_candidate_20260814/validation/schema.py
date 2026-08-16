"""Deterministic file and SQLite structural audits.

The production-candidate pipeline must not make an accidental SQLite schema
migration.  This module deliberately fingerprints only schema declarations
(not SQLite page layout or row insertion order), so independently generated
databases with the same public contract receive the same fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union


DEFAULT_HASH_CHUNK_BYTES = 1024 * 1024
SQLiteSource = Union[str, Path, sqlite3.Connection]


@dataclass(frozen=True)
class SQLiteSchemaObject:
    """One non-internal declaration from ``sqlite_schema``."""

    object_type: str
    name: str
    table_name: str
    sql: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.object_type,
            "name": self.name,
            "table_name": self.table_name,
            "sql": self.sql,
        }


@dataclass(frozen=True)
class ForeignKeyViolation:
    """A normalized row returned by ``PRAGMA foreign_key_check``."""

    table: str
    row_id: int | None
    parent: str
    foreign_key_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "row_id": self.row_id,
            "parent": self.parent,
            "foreign_key_id": self.foreign_key_id,
        }


@dataclass(frozen=True)
class SQLiteAudit:
    """Read-only integrity and schema report for one SQLite artifact."""

    path: Path
    size_bytes: int
    file_sha256: str
    schema_sha256: str
    schema_objects: tuple[SQLiteSchemaObject, ...]
    integrity_messages: tuple[str, ...]
    foreign_key_violations: tuple[ForeignKeyViolation, ...]

    @property
    def integrity_ok(self) -> bool:
        return self.integrity_messages == ("ok",)

    @property
    def foreign_keys_ok(self) -> bool:
        return not self.foreign_key_violations

    @property
    def ok(self) -> bool:
        return self.integrity_ok and self.foreign_keys_ok

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "file_sha256": self.file_sha256,
            "schema_sha256": self.schema_sha256,
            "schema_object_count": len(self.schema_objects),
            "schema_objects": [item.to_dict() for item in self.schema_objects],
            "integrity_check": list(self.integrity_messages),
            "integrity_ok": self.integrity_ok,
            "foreign_key_error_count": len(self.foreign_key_violations),
            "foreign_key_violations": [
                item.to_dict() for item in self.foreign_key_violations
            ],
            "ok": self.ok,
        }


def sha256_file(
    path: str | Path,
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
) -> str:
    """Return a streaming SHA-256 digest without loading the file into RAM."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value using a stable UTF-8 representation."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _read_connection(source: SQLiteSource) -> Iterator[sqlite3.Connection]:
    if isinstance(source, sqlite3.Connection):
        yield source
        return

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # URI read-only mode prevents a typo from silently creating a database and
    # prevents validation code from mutating the artifact being certified.
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def schema_objects(source: SQLiteSource) -> tuple[SQLiteSchemaObject, ...]:
    """Return schema declarations in a deterministic, page-layout-free order."""

    with _read_connection(source) as connection:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, COALESCE(sql, '')
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name, COALESCE(sql, '')
            """
        ).fetchall()
    return tuple(
        SQLiteSchemaObject(
            object_type=str(object_type),
            name=str(name),
            table_name=str(table_name),
            sql=str(sql),
        )
        for object_type, name, table_name, sql in rows
    )


def schema_fingerprint(source: SQLiteSource) -> str:
    """Return a stable SHA-256 fingerprint of all public schema declarations."""

    objects = schema_objects(source)
    return stable_json_sha256([item.to_dict() for item in objects])


def audit_sqlite(path: str | Path) -> SQLiteAudit:
    """Run schema, integrity, and foreign-key checks without modifying SQLite."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    objects: tuple[SQLiteSchemaObject, ...]
    integrity_messages: tuple[str, ...]
    violations: tuple[ForeignKeyViolation, ...]
    with _read_connection(source) as connection:
        objects = schema_objects(connection)
        integrity_messages = tuple(
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        )
        raw_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        violations = tuple(
            ForeignKeyViolation(
                table=str(table),
                row_id=None if row_id is None else int(row_id),
                parent=str(parent),
                foreign_key_id=int(foreign_key_id),
            )
            for table, row_id, parent, foreign_key_id in sorted(
                raw_violations,
                key=lambda row: (
                    str(row[0]),
                    -1 if row[1] is None else int(row[1]),
                    str(row[2]),
                    int(row[3]),
                ),
            )
        )

    return SQLiteAudit(
        path=source,
        size_bytes=source.stat().st_size,
        file_sha256=sha256_file(source),
        schema_sha256=stable_json_sha256([item.to_dict() for item in objects]),
        schema_objects=objects,
        integrity_messages=integrity_messages,
        foreign_key_violations=violations,
    )


def schema_fingerprints_match(a: SQLiteSource, b: SQLiteSource) -> bool:
    """Convenience predicate for enforcing the no-schema-change contract."""

    return schema_fingerprint(a) == schema_fingerprint(b)


def mapping_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash normalized mapping rows; useful for small manifest inventories."""

    return stable_json_sha256([dict(row) for row in rows])
