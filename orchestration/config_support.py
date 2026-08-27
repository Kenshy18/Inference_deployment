"""Shared parsing and validation primitives for workflow configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OrchestrationConfigError(ValueError):
    """Raised when a workflow configuration is inconsistent."""


def object_value(value: object, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OrchestrationConfigError(f"{section} must be a JSON object")
    return dict(value)


def reject_unknown(
    values: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise OrchestrationConfigError(
            f"{section} has unknown option(s): {sorted(unknown)}"
        )


def resolve_path(
    value: object | None,
    *,
    base: Path,
    field: str,
    required: bool = False,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise OrchestrationConfigError(f"{field} is required")
        return None
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def optional_int(value: object | None, field: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field} must be an integer") from exc


def optional_float(value: object | None, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field} must be a number") from exc


def string_tuple(value: object | None, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise OrchestrationConfigError(f"{field} must be a list of strings")
    return tuple(str(item) for item in value)


def reject_reserved_args(
    values: tuple[str, ...],
    reserved: set[str],
    field: str,
) -> None:
    conflicts = sorted({value for value in values if value in reserved})
    if conflicts:
        raise OrchestrationConfigError(
            f"{field} must not override managed option(s): {conflicts}"
        )


def validate_class_postprocess_policy(path: Path) -> None:
    field = "postprocess.class_postprocess_policy_json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationConfigError(f"{field} must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise OrchestrationConfigError(f"{field} root must be a JSON object")
    reject_unknown(raw, {"schema_version", "default", "classes"}, field)
    try:
        schema_version = int(raw.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field}.schema_version must be 1") from exc
    if schema_version not in {1, 2}:
        raise OrchestrationConfigError(f"{field}.schema_version must be 1 or 2")
    classes = raw.get("classes", {})
    if not isinstance(classes, dict):
        raise OrchestrationConfigError(f"{field}.classes must be a JSON object")
    sections: list[tuple[str, object]] = [(f"{field}.default", raw.get("default"))]
    sections.extend(
        (f"{field}.classes.{label}", value) for label, value in classes.items()
    )
    for section, value in sections:
        if value is None:
            continue
        if not isinstance(value, dict):
            raise OrchestrationConfigError(f"{section} must be a JSON object")
        allowed = (
            {"shape_mode", "keyframe_interval", "max_gap"}
            if schema_version == 1
            else {"keyframe_interval"}
        )
        reject_unknown(value, allowed, section)
        if "shape_mode" in value and value["shape_mode"] != "polygon":
            raise OrchestrationConfigError(f"{section}.shape_mode must be polygon")
        if "keyframe_interval" in value:
            interval = optional_int(
                value["keyframe_interval"],
                f"{section}.keyframe_interval",
            )
            if interval is None or interval < 1:
                raise OrchestrationConfigError(
                    f"{section}.keyframe_interval must be at least 1"
                )
        if "max_gap" in value:
            max_gap = optional_int(value["max_gap"], f"{section}.max_gap")
            if max_gap != 15:
                raise OrchestrationConfigError(f"{section}.max_gap is fixed at 15")
    if any(not str(label).strip() for label in classes):
        raise OrchestrationConfigError(f"{field}.classes labels must not be empty")


__all__ = (
    "OrchestrationConfigError",
    "object_value",
    "optional_float",
    "optional_int",
    "reject_reserved_args",
    "reject_unknown",
    "resolve_path",
    "string_tuple",
    "validate_class_postprocess_policy",
)
