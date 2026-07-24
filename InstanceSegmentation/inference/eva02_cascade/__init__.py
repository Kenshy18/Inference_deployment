"""Standalone EVA02 Cascade instance-segmentation family.

Heavy PyTorch and Detectron2 modules are imported only by the inference path.
This keeps bundle inspection and environment setup lightweight.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "build_runtime":
        from .assembly import build_runtime

        return build_runtime
    if name == "VideoInferenceSettings":
        from .runtime_contracts import VideoInferenceSettings

        return VideoInferenceSettings
    raise AttributeError(name)


__all__ = ["VideoInferenceSettings", "build_runtime"]
