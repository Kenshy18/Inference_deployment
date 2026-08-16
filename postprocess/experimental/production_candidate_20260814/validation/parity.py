"""Streaming parity checks for candidate pipeline artifacts.

No function in this module reads video pixels.  JSONL is compared one record at
a time after canonical JSON serialization, and SQLite is opened read-only and
compared table-by-table in deterministic declared-column order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import SQLiteAudit, audit_sqlite


DEFAULT_FLOAT_TOLERANCE = 1e-6
DEFAULT_DIAGNOSTIC_CHARACTERS = 2048
_MISSING = object()


class ParityValidationError(ValueError):
    """Raised when an artifact cannot be parsed or safely compared."""


@dataclass(frozen=True)
class JsonlMismatch:
    """Diagnostic for the first canonical JSONL difference."""

    kind: str
    record_number: int
    reference_line_number: int | None
    candidate_line_number: int | None
    reference_preview: str | None
    candidate_preview: str | None
    reference_record_sha256: str | None
    candidate_record_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "record_number": self.record_number,
            "reference_line_number": self.reference_line_number,
            "candidate_line_number": self.candidate_line_number,
            "reference_preview": self.reference_preview,
            "candidate_preview": self.candidate_preview,
            "reference_record_sha256": self.reference_record_sha256,
            "candidate_record_sha256": self.candidate_record_sha256,
        }


@dataclass(frozen=True)
class CanonicalJsonlComparison:
    """Result of an exact canonical-record JSONL comparison."""

    reference_path: Path
    candidate_path: Path
    reference_records: int
    candidate_records: int
    reference_canonical_sha256: str
    candidate_canonical_sha256: str
    first_mismatch: JsonlMismatch | None

    @property
    def equal(self) -> bool:
        return (
            self.first_mismatch is None
            and self.reference_records == self.candidate_records
            and self.reference_canonical_sha256 == self.candidate_canonical_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_path": str(self.reference_path),
            "candidate_path": str(self.candidate_path),
            "equal": self.equal,
            "reference_records": self.reference_records,
            "candidate_records": self.candidate_records,
            "reference_canonical_sha256": self.reference_canonical_sha256,
            "candidate_canonical_sha256": self.candidate_canonical_sha256,
            "first_mismatch": (
                None if self.first_mismatch is None else self.first_mismatch.to_dict()
            ),
        }


@dataclass(frozen=True)
class SQLiteColumn:
    """A column in SQLite declaration order."""

    index: int
    name: str
    declared_type: str
    not_null: bool
    default_sql: str | None
    primary_key_position: int
    hidden: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "declared_type": self.declared_type,
            "not_null": self.not_null,
            "default_sql": self.default_sql,
            "primary_key_position": self.primary_key_position,
            "hidden": self.hidden,
        }


@dataclass(frozen=True)
class SQLiteCellMismatch:
    """First mismatching cell (or missing row) in one table."""

    kind: str
    row_number: int
    column: str | None
    reference_value: object
    candidate_value: object
    absolute_difference: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "row_number": self.row_number,
            "column": self.column,
            "reference_value": _json_safe(self.reference_value),
            "candidate_value": _json_safe(self.candidate_value),
            "absolute_difference": self.absolute_difference,
        }


@dataclass(frozen=True)
class SQLiteTableComparison:
    """Semantic row comparison for a single table."""

    table: str
    reference_columns: tuple[SQLiteColumn, ...]
    candidate_columns: tuple[SQLiteColumn, ...]
    reference_rows: int
    candidate_rows: int
    compared_rows: int
    mismatch_count: int
    first_mismatch: SQLiteCellMismatch | None

    @property
    def columns_equal(self) -> bool:
        return self.reference_columns == self.candidate_columns

    @property
    def equal(self) -> bool:
        return (
            self.columns_equal
            and self.reference_rows == self.candidate_rows
            and self.mismatch_count == 0
            and self.first_mismatch is None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "equal": self.equal,
            "columns_equal": self.columns_equal,
            "reference_columns": [item.to_dict() for item in self.reference_columns],
            "candidate_columns": [item.to_dict() for item in self.candidate_columns],
            "reference_rows": self.reference_rows,
            "candidate_rows": self.candidate_rows,
            "compared_rows": self.compared_rows,
            "mismatch_count": self.mismatch_count,
            "first_mismatch": (
                None if self.first_mismatch is None else self.first_mismatch.to_dict()
            ),
        }


@dataclass(frozen=True)
class SQLiteSemanticComparison:
    """Deterministic comparison across a set of user tables."""

    reference_path: Path
    candidate_path: Path
    float_tolerance: float
    requested_tables: tuple[str, ...]
    missing_from_reference: tuple[str, ...]
    missing_from_candidate: tuple[str, ...]
    tables: tuple[SQLiteTableComparison, ...]

    @property
    def equal(self) -> bool:
        return (
            not self.missing_from_reference
            and not self.missing_from_candidate
            and all(table.equal for table in self.tables)
        )

    @property
    def first_mismatch(self) -> tuple[str, SQLiteCellMismatch] | None:
        for table in self.tables:
            if table.first_mismatch is not None:
                return table.table, table.first_mismatch
        return None

    def to_dict(self) -> dict[str, object]:
        first = self.first_mismatch
        return {
            "reference_path": str(self.reference_path),
            "candidate_path": str(self.candidate_path),
            "equal": self.equal,
            "float_tolerance": self.float_tolerance,
            "requested_tables": list(self.requested_tables),
            "missing_from_reference": list(self.missing_from_reference),
            "missing_from_candidate": list(self.missing_from_candidate),
            "tables": [item.to_dict() for item in self.tables],
            "first_mismatch": (
                None if first is None else {"table": first[0], **first[1].to_dict()}
            ),
        }


@dataclass(frozen=True)
class ParityReport:
    """High-level, serializable validation result for a candidate run."""

    jsonl: CanonicalJsonlComparison | None
    reference_sqlite: SQLiteAudit | None
    candidate_sqlite: SQLiteAudit | None
    sqlite_schema_equal: bool | None
    sqlite_semantic: SQLiteSemanticComparison | None

    @property
    def passed(self) -> bool:
        checks: list[bool] = []
        if self.jsonl is not None:
            checks.append(self.jsonl.equal)
        if self.reference_sqlite is not None:
            checks.append(self.reference_sqlite.ok)
        if self.candidate_sqlite is not None:
            checks.append(self.candidate_sqlite.ok)
        if self.sqlite_schema_equal is not None:
            checks.append(self.sqlite_schema_equal)
        if self.sqlite_semantic is not None:
            checks.append(self.sqlite_semantic.equal)
        return bool(checks) and all(checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "production-candidate-parity-v1",
            "passed": self.passed,
            "jsonl": None if self.jsonl is None else self.jsonl.to_dict(),
            "sqlite": {
                "schema_equal": self.sqlite_schema_equal,
                "reference_audit": (
                    None
                    if self.reference_sqlite is None
                    else self.reference_sqlite.to_dict()
                ),
                "candidate_audit": (
                    None
                    if self.candidate_sqlite is None
                    else self.candidate_sqlite.to_dict()
                ),
                "semantic": (
                    None
                    if self.sqlite_semantic is None
                    else self.sqlite_semantic.to_dict()
                ),
            },
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_records(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
                canonical = _canonical_json(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ParityValidationError(
                    f"{path}:{line_number}: invalid canonical JSON: {exc}"
                ) from exc
            yield line_number, canonical


def _record_sha256(canonical: str | None) -> str | None:
    if canonical is None:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preview(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + f"… <{len(value) - limit} characters omitted>"


def compare_canonical_jsonl(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    diagnostic_characters: int = DEFAULT_DIAGNOSTIC_CHARACTERS,
) -> CanonicalJsonlComparison:
    """Compare canonical JSON records exactly while keeping memory bounded.

    Object key order and insignificant JSON whitespace do not matter.  Every
    value, array order, number representation, and record order does.  Blank
    physical lines are ignored, while diagnostics retain original line numbers.
    """

    if diagnostic_characters <= 0:
        raise ValueError("diagnostic_characters must be positive")
    reference = Path(reference_path).expanduser().resolve()
    candidate = Path(candidate_path).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    reference_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    reference_records = 0
    candidate_records = 0
    first_mismatch: JsonlMismatch | None = None

    pairs = zip_longest(
        _canonical_records(reference),
        _canonical_records(candidate),
        fillvalue=_MISSING,
    )
    for record_number, (reference_item, candidate_item) in enumerate(pairs, 1):
        reference_line: int | None = None
        candidate_line: int | None = None
        reference_value: str | None = None
        candidate_value: str | None = None

        if reference_item is not _MISSING:
            reference_line, reference_value = reference_item  # type: ignore[misc]
            reference_records += 1
            reference_digest.update(reference_value.encode("utf-8"))
            reference_digest.update(b"\n")
        if candidate_item is not _MISSING:
            candidate_line, candidate_value = candidate_item  # type: ignore[misc]
            candidate_records += 1
            candidate_digest.update(candidate_value.encode("utf-8"))
            candidate_digest.update(b"\n")

        if first_mismatch is not None or reference_value == candidate_value:
            continue
        if reference_item is _MISSING:
            kind = "unexpected_candidate_record"
        elif candidate_item is _MISSING:
            kind = "missing_candidate_record"
        else:
            kind = "record_value_mismatch"
        first_mismatch = JsonlMismatch(
            kind=kind,
            record_number=record_number,
            reference_line_number=reference_line,
            candidate_line_number=candidate_line,
            reference_preview=_preview(reference_value, diagnostic_characters),
            candidate_preview=_preview(candidate_value, diagnostic_characters),
            reference_record_sha256=_record_sha256(reference_value),
            candidate_record_sha256=_record_sha256(candidate_value),
        )

    return CanonicalJsonlComparison(
        reference_path=reference,
        candidate_path=candidate,
        reference_records=reference_records,
        candidate_records=candidate_records,
        reference_canonical_sha256=reference_digest.hexdigest(),
        candidate_canonical_sha256=candidate_digest.hexdigest(),
        first_mismatch=first_mismatch,
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@contextmanager
def _readonly(path: Path) -> Iterator[sqlite3.Connection]:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def _user_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    )


def _columns(connection: sqlite3.Connection, table: str) -> tuple[SQLiteColumn, ...]:
    rows = connection.execute(
        f"PRAGMA table_xinfo({_quote_identifier(table)})"
    ).fetchall()
    return tuple(
        SQLiteColumn(
            index=int(index),
            name=str(name),
            declared_type=str(declared_type or ""),
            not_null=bool(not_null),
            default_sql=None if default_sql is None else str(default_sql),
            primary_key_position=int(primary_key_position),
            hidden=int(hidden),
        )
        for (
            index,
            name,
            declared_type,
            not_null,
            default_sql,
            primary_key_position,
            hidden,
        ) in rows
    )


def _ordered_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[SQLiteColumn],
) -> Iterable[tuple[object, ...]]:
    if not columns:
        raise ParityValidationError(f"table has no declared columns: {table}")
    selected = ", ".join(_quote_identifier(column.name) for column in columns)
    ordered = ", ".join(_quote_identifier(column.name) for column in columns)
    query = f"SELECT {selected} FROM {_quote_identifier(table)} " f"ORDER BY {ordered}"
    return connection.execute(query)


def _json_safe(value: object) -> object:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, memoryview):
        return {"bytes_hex": value.tobytes().hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _values_equal(a: object, b: object, tolerance: float) -> tuple[bool, float | None]:
    if isinstance(a, memoryview):
        a = a.tobytes()
    if isinstance(b, memoryview):
        b = b.tobytes()
    if (
        isinstance(a, (int, float))
        and not isinstance(a, bool)
        and isinstance(b, (int, float))
        and not isinstance(b, bool)
    ):
        if isinstance(a, float) or isinstance(b, float):
            left = float(a)
            right = float(b)
            if math.isnan(left) or math.isnan(right):
                return math.isnan(left) and math.isnan(right), None
            if math.isinf(left) or math.isinf(right):
                return left == right, None
            difference = abs(left - right)
            return difference <= tolerance, difference
    return a == b, None


def _compare_table(
    reference: sqlite3.Connection,
    candidate: sqlite3.Connection,
    table: str,
    *,
    float_tolerance: float,
) -> SQLiteTableComparison:
    reference_columns = _columns(reference, table)
    candidate_columns = _columns(candidate, table)
    reference_rows = int(
        reference.execute(
            f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
        ).fetchone()[0]
    )
    candidate_rows = int(
        candidate.execute(
            f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
        ).fetchone()[0]
    )

    if reference_columns != candidate_columns:
        mismatch = SQLiteCellMismatch(
            kind="column_declaration_mismatch",
            row_number=0,
            column=None,
            reference_value=tuple(item.to_dict() for item in reference_columns),
            candidate_value=tuple(item.to_dict() for item in candidate_columns),
        )
        return SQLiteTableComparison(
            table=table,
            reference_columns=reference_columns,
            candidate_columns=candidate_columns,
            reference_rows=reference_rows,
            candidate_rows=candidate_rows,
            compared_rows=0,
            mismatch_count=1,
            first_mismatch=mismatch,
        )

    mismatch_count = 0
    first_mismatch: SQLiteCellMismatch | None = None
    compared_rows = 0
    rows = zip_longest(
        _ordered_rows(reference, table, reference_columns),
        _ordered_rows(candidate, table, candidate_columns),
        fillvalue=_MISSING,
    )
    for row_number, (reference_row, candidate_row) in enumerate(rows, 1):
        compared_rows += 1
        if reference_row is _MISSING or candidate_row is _MISSING:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = SQLiteCellMismatch(
                    kind=(
                        "unexpected_candidate_row"
                        if reference_row is _MISSING
                        else "missing_candidate_row"
                    ),
                    row_number=row_number,
                    column=None,
                    reference_value=reference_row,
                    candidate_value=candidate_row,
                )
            continue

        for column, reference_value, candidate_value in zip(
            reference_columns,
            reference_row,
            candidate_row,
        ):
            equal, difference = _values_equal(
                reference_value,
                candidate_value,
                float_tolerance,
            )
            if equal:
                continue
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = SQLiteCellMismatch(
                    kind="cell_value_mismatch",
                    row_number=row_number,
                    column=column.name,
                    reference_value=reference_value,
                    candidate_value=candidate_value,
                    absolute_difference=difference,
                )

    return SQLiteTableComparison(
        table=table,
        reference_columns=reference_columns,
        candidate_columns=candidate_columns,
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
        compared_rows=compared_rows,
        mismatch_count=mismatch_count,
        first_mismatch=first_mismatch,
    )


def compare_sqlite_tables(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    tables: Sequence[str] | None = None,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> SQLiteSemanticComparison:
    """Compare SQLite rows by public table and declared column order.

    Rows and selected cells use the schema's declared column order.
    Floating-point cells use an absolute coordinate tolerance.  All other
    SQLite values compare exactly.
    """

    if not math.isfinite(float_tolerance) or float_tolerance < 0:
        raise ValueError("float_tolerance must be a finite non-negative value")
    reference_path = Path(reference_path).expanduser().resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()

    with _readonly(reference_path) as reference, _readonly(candidate_path) as candidate:
        reference_tables = set(_user_tables(reference))
        candidate_tables = set(_user_tables(candidate))
        if tables is None:
            requested = tuple(sorted(reference_tables | candidate_tables))
        else:
            requested = tuple(dict.fromkeys(str(table) for table in tables))
            if not requested or any(not table for table in requested):
                raise ValueError("tables must contain at least one non-empty name")

        missing_reference = tuple(
            table for table in requested if table not in reference_tables
        )
        missing_candidate = tuple(
            table for table in requested if table not in candidate_tables
        )
        comparable = (
            table
            for table in requested
            if table in reference_tables and table in candidate_tables
        )
        comparisons = tuple(
            _compare_table(
                reference,
                candidate,
                table,
                float_tolerance=float_tolerance,
            )
            for table in comparable
        )

    return SQLiteSemanticComparison(
        reference_path=reference_path,
        candidate_path=candidate_path,
        float_tolerance=float_tolerance,
        requested_tables=requested,
        missing_from_reference=missing_reference,
        missing_from_candidate=missing_candidate,
        tables=comparisons,
    )


def build_parity_report(
    *,
    reference_jsonl: str | Path | None = None,
    candidate_jsonl: str | Path | None = None,
    reference_sqlite: str | Path | None = None,
    candidate_sqlite: str | Path | None = None,
    sqlite_tables: Sequence[str] | None = None,
    float_tolerance: float = DEFAULT_FLOAT_TOLERANCE,
) -> ParityReport:
    """Run the requested JSONL and/or SQLite parity checks."""

    if (reference_jsonl is None) != (candidate_jsonl is None):
        raise ValueError(
            "reference_jsonl and candidate_jsonl must be supplied together"
        )
    if (reference_sqlite is None) != (candidate_sqlite is None):
        raise ValueError(
            "reference_sqlite and candidate_sqlite must be supplied together"
        )
    if reference_jsonl is None and reference_sqlite is None:
        raise ValueError("at least one artifact pair is required")

    jsonl_result = (
        None
        if reference_jsonl is None or candidate_jsonl is None
        else compare_canonical_jsonl(reference_jsonl, candidate_jsonl)
    )
    if reference_sqlite is None or candidate_sqlite is None:
        return ParityReport(
            jsonl=jsonl_result,
            reference_sqlite=None,
            candidate_sqlite=None,
            sqlite_schema_equal=None,
            sqlite_semantic=None,
        )

    reference_audit = audit_sqlite(reference_sqlite)
    candidate_audit = audit_sqlite(candidate_sqlite)
    semantic = compare_sqlite_tables(
        reference_sqlite,
        candidate_sqlite,
        tables=sqlite_tables,
        float_tolerance=float_tolerance,
    )
    return ParityReport(
        jsonl=jsonl_result,
        reference_sqlite=reference_audit,
        candidate_sqlite=candidate_audit,
        sqlite_schema_equal=(
            reference_audit.schema_sha256 == candidate_audit.schema_sha256
        ),
        sqlite_semantic=semantic,
    )


def write_parity_report(
    path: str | Path,
    report: ParityReport | Mapping[str, object],
) -> Path:
    """Atomically publish a human-readable JSON parity report."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, object]
    if isinstance(report, ParityReport):
        payload = report.to_dict()
    else:
        payload = report
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_and_write_parity_report(
    report_path: str | Path,
    **kwargs: Any,
) -> ParityReport:
    """Build a parity report and atomically publish it in one call."""

    report = build_parity_report(**kwargs)
    write_parity_report(report_path, report)
    return report
