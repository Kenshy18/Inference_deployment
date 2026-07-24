"""TensorRT backbone generation, verification, and runtime integration."""

from __future__ import annotations

from typing import Any

from .bundle import (
    MANIFEST_SCHEMA,
    PROFILE,
    Eva02TrtBundle,
    load_trt_bundle,
)


def __getattr__(name: str) -> Any:
    if name == "Eva02TensorRTBackbone":
        from .runtime import Eva02TensorRTBackbone

        return Eva02TensorRTBackbone
    raise AttributeError(name)

__all__ = [
    "MANIFEST_SCHEMA",
    "PROFILE",
    "Eva02TensorRTBackbone",
    "Eva02TrtBundle",
    "load_trt_bundle",
]
