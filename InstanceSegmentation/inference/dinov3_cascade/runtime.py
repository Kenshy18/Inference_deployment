"""Loaded DINOv3 Cascade model bundle and autocast policy."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .runtime_contracts import VideoInferenceSettings


@dataclass(slots=True)
class Dinov3Runtime:
    segmenter: torch.nn.Module
    classifier: torch.nn.Module | None
    class_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    target_size: int | tuple[int, int]


def autocast_context(settings: VideoInferenceSettings):
    if not settings.amp or not settings.device.startswith("cuda"):
        return contextlib.nullcontext()
    dtype = torch.float16 if settings.amp_dtype == "fp16" else torch.bfloat16
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


__all__ = ["Dinov3Runtime", "autocast_context"]
