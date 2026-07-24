"""Typed key/value normalization shared by SQLite persistence formats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def _scalar_text(value: object) -> tuple[str, str]:
    if value is None:
        return "", "null"
    if isinstance(value, bool):
        return ("1" if value else "0"), "bool"
    if isinstance(value, int):
        return str(value), "int"
    if isinstance(value, float):
        return repr(value), "float"
    if isinstance(value, Path):
        return str(value), "path"
    if isinstance(value, str):
        return value, "str"
    return str(value), type(value).__name__


def flatten_metadata(
    values: Mapping[str, object],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for key, value in values.items():
        rows.extend(_flatten_value(str(key), value))
    return rows


def _flatten_value(prefix: str, value: object) -> list[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        if not value:
            return [(prefix, "", "empty_mapping")]
        rows: list[tuple[str, str, str]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_value(child, item))
        return rows
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not value:
            return [(prefix, "", "empty_sequence")]
        rows = []
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_flatten_value(child, item))
        return rows
    text, value_type = _scalar_text(value)
    return [(prefix, text, value_type)]


__all__ = ["flatten_metadata"]
