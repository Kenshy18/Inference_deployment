"""Built-in and external stage resolution."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from typing import Any, cast

from contracts.stages import PostprocessStage

StageFactory = Callable[[dict[str, Any]], PostprocessStage]
_factories: dict[str, StageFactory] = {}
_discovered = False


def register_stage(name: str, factory: StageFactory, *, replace: bool = False) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("stage implementation name must not be empty")
    if key in _factories and not replace:
        raise ValueError(f"stage implementation already registered: {key}")
    _factories[key] = factory


def _validate_stage(value: object, implementation: str) -> PostprocessStage:
    for attribute in ("name", "requires", "provides", "run"):
        if not hasattr(value, attribute):
            raise TypeError(
                f"{implementation!r} is not a PostprocessStage: missing {attribute}"
            )
    return cast(PostprocessStage, value)


def _load_external(implementation: str, options: dict[str, Any]) -> PostprocessStage:
    module_name, separator, attribute_name = implementation.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("external stage must use 'python.module:attribute' syntax")
    value = getattr(importlib.import_module(module_name), attribute_name)
    if isinstance(value, type):
        value = value(**options)
    elif callable(value) and not hasattr(value, "run"):
        value = value(options)
    elif options:
        raise ValueError(f"{implementation!r} is an instance and cannot accept options")
    return _validate_stage(value, implementation)


def create_stage(
    implementation: str, options: dict[str, Any] | None = None
) -> PostprocessStage:
    discover_stages()
    settings = dict(options or {})
    if ":" in implementation:
        return _load_external(implementation, settings)
    try:
        factory = _factories[implementation]
    except KeyError as exc:
        raise ValueError(
            f"unknown stage {implementation!r}; built-ins: {sorted(_factories)}"
        ) from exc
    return _validate_stage(factory(settings), implementation)


def stage_implementations() -> tuple[str, ...]:
    discover_stages()
    return tuple(sorted(_factories))


def discover_stages() -> None:
    """Register factories published through the stage entry-point group."""

    global _discovered
    if _discovered:
        return
    _discovered = True
    for entry_point in metadata.entry_points().select(group="postprocess.stages"):
        register_stage(entry_point.name, entry_point.load())
