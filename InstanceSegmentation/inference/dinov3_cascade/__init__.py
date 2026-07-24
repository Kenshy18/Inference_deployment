"""Standalone DINOv3 Cascade instance-segmentation family."""

from .assembly import build_runtime
from .runtime_contracts import VideoInferenceSettings

__all__ = ["VideoInferenceSettings", "build_runtime"]
