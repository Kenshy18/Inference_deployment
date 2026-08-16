"""Validated class-specific postprocess policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SETTING_KEYS_V1 = {"shape_mode", "keyframe_interval", "max_gap"}
_SETTING_KEYS_V2 = {"keyframe_interval"}
PRODUCTION_POLYGON_MAX_GAP = 15


@dataclass(frozen=True, slots=True)
class ClassPostprocessSettings:
    shape_mode: str
    keyframe_interval: int
    max_gap: int

    def __post_init__(self) -> None:
        if self.shape_mode != "polygon":
            raise ValueError("Production class policies support polygon only")
        if self.keyframe_interval < 1:
            raise ValueError("keyframe_interval must be at least 1")
        if self.max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        if self.max_gap != PRODUCTION_POLYGON_MAX_GAP:
            raise ValueError(
                "Production polygon max_gap is fixed at "
                f"{PRODUCTION_POLYGON_MAX_GAP}; got {self.max_gap}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "shape_mode": self.shape_mode,
            "keyframe_interval": self.keyframe_interval,
            "max_gap": self.max_gap,
        }


def _settings_object(
    value: object,
    field: str,
    *,
    schema_version: int,
) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    allowed = _SETTING_KEYS_V1 if schema_version == 1 else _SETTING_KEYS_V2
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field} has unknown option(s): {sorted(unknown)}")
    return dict(value)


def _resolve_settings(
    values: Mapping[str, object],
    *,
    fallback: ClassPostprocessSettings,
) -> ClassPostprocessSettings:
    shape_mode = str(values.get("shape_mode", fallback.shape_mode))
    if shape_mode != "polygon":
        raise ValueError("Production class policies support polygon only")
    keyframe_interval = int(values.get("keyframe_interval", fallback.keyframe_interval))
    max_gap = int(values.get("max_gap", fallback.max_gap))
    max_gap = PRODUCTION_POLYGON_MAX_GAP
    return ClassPostprocessSettings(
        shape_mode=shape_mode,
        keyframe_interval=keyframe_interval,
        max_gap=max_gap,
    )


@dataclass(frozen=True, slots=True)
class ClassPostprocessPolicy:
    default: ClassPostprocessSettings
    classes: Mapping[str, ClassPostprocessSettings]

    def resolve(self, label: str) -> ClassPostprocessSettings:
        return self.classes.get(str(label), self.default)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "default": {"keyframe_interval": self.default.keyframe_interval},
            "classes": {
                label: {"keyframe_interval": settings.keyframe_interval}
                for label, settings in sorted(self.classes.items())
            },
        }


def load_class_postprocess_policy(
    path: Path,
    *,
    fallback: ClassPostprocessSettings,
) -> ClassPostprocessPolicy:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("class postprocess policy root must be a JSON object")
    unknown = set(raw) - {"schema_version", "default", "classes"}
    if unknown:
        raise ValueError(
            "class postprocess policy has unknown option(s): " f"{sorted(unknown)}"
        )
    schema_version = int(raw.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise ValueError(
            "unsupported class postprocess policy schema_version="
            f"{schema_version}; expected 1 or 2"
        )
    default_values = _settings_object(
        raw.get("default"),
        "default",
        schema_version=schema_version,
    )
    default = _resolve_settings(default_values, fallback=fallback)
    raw_classes = raw.get("classes", {})
    if not isinstance(raw_classes, dict):
        raise ValueError("classes must be a JSON object")
    classes: dict[str, ClassPostprocessSettings] = {}
    for raw_label, value in raw_classes.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError("class postprocess policy labels must not be empty")
        settings = _settings_object(
            value,
            f"classes.{label}",
            schema_version=schema_version,
        )
        classes[label] = _resolve_settings(settings, fallback=default)
    return ClassPostprocessPolicy(default=default, classes=classes)


__all__ = [
    "ClassPostprocessPolicy",
    "ClassPostprocessSettings",
    "PRODUCTION_POLYGON_MAX_GAP",
    "load_class_postprocess_policy",
]
