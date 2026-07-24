"""Loaded EVA-02 model bundle and model-specific forward policy."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .runtime_contracts import VideoInferenceSettings


@dataclass(slots=True)
class Eva02Runtime:
    segmenter: torch.nn.Module
    classifier: torch.nn.Module
    class_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    target_size: int
    meta_feature_set: str
    feature_l2norm: bool
    backend: str


def _autocast(settings: VideoInferenceSettings):
    if not settings.amp or not settings.device.startswith("cuda"):
        return contextlib.nullcontext()
    dtype = torch.float16 if settings.amp_dtype == "fp16" else torch.bfloat16
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def run_segmenter(
    segmenter: torch.nn.Module,
    inputs: list[dict[str, object]],
    *,
    settings: VideoInferenceSettings,
) -> list[dict[str, object]]:
    with _autocast(settings):
        if hasattr(segmenter, "inference"):
            return [
                {"instances": instances}
                for instances in segmenter.inference(inputs, do_postprocess=False)
            ]
        return segmenter(inputs)


__all__ = ["Eva02Runtime", "run_segmenter"]
