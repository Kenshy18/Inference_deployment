"""Fixed-batch fused CUDA preprocessing and source-coordinate restoration."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


TARGET_HEIGHT = 736
TARGET_WIDTH = 1280


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    scale_x: float
    scale_y: float
    pad_top: int
    pad_left: int


def letterbox_transform(height: int, width: int) -> LetterboxTransform:
    scale = min(TARGET_HEIGHT / height, TARGET_WIDTH / width)
    resized_height = max(1, int(height * scale))
    resized_width = max(1, int(width * scale))
    return LetterboxTransform(
        source_height=height,
        source_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
        pad_top=(TARGET_HEIGHT - resized_height) // 2,
        pad_left=(TARGET_WIDTH - resized_width) // 2,
    )


def load_fused_preprocessor(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"fused preprocessing plugin not found: {resolved}")
    name = "mh0_preprocess_fused_sm120"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fused preprocessing plugin: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FusedVideoPreprocessor:
    """One bounded pinned buffer and one reusable CUDA output buffer."""

    def __init__(
        self,
        *,
        batch_size: int,
        frame_height: int,
        frame_width: int,
        device: torch.device,
        plugin: Path,
    ) -> None:
        self.batch_size = int(batch_size)
        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.device = device
        self.transform = letterbox_transform(frame_height, frame_width)
        self.extension = load_fused_preprocessor(plugin)
        self.host = torch.empty(
            (batch_size, frame_height, frame_width, 3),
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.host_numpy = self.host.numpy()
        self.device_bgr = torch.empty(
            (batch_size, frame_height, frame_width, 3),
            dtype=torch.uint8,
            device=device,
        )
        self.output = torch.empty(
            (batch_size, 3, TARGET_HEIGHT, TARGET_WIDTH),
            dtype=torch.float32,
            device=device,
        )
        self.copy_complete = torch.cuda.Event()
        self.copy_recorded = False

    def prepare(
        self,
        frames: list[np.ndarray],
    ) -> tuple[torch.Tensor, list[LetterboxTransform]]:
        if len(frames) != self.batch_size:
            raise ValueError(
                f"fixed-B{self.batch_size} preprocessing got {len(frames)} frames"
            )
        if self.copy_recorded:
            self.copy_complete.synchronize()
        for index, frame in enumerate(frames):
            expected = (self.frame_height, self.frame_width, 3)
            if frame.shape != expected or frame.dtype != np.uint8:
                raise ValueError(
                    f"video frame must be uint8 {expected}, got {frame.shape} "
                    f"{frame.dtype}"
                )
            np.copyto(self.host_numpy[index], frame)
        self.device_bgr.copy_(self.host, non_blocking=True)
        stream = torch.cuda.current_stream(self.device)
        self.copy_complete.record(stream)
        self.copy_recorded = True
        self.extension.forward_out(
            self.device_bgr,
            self.output,
            self.transform.resized_height,
            self.transform.resized_width,
            self.transform.pad_top,
            self.transform.pad_left,
            int(stream.cuda_stream),
        )
        return self.output, [self.transform] * self.batch_size


def restore_result(
    result: dict[str, torch.Tensor],
    transform: LetterboxTransform,
) -> dict[str, torch.Tensor]:
    restored = dict(result)

    def restore_xyxy(value: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        value[:, 0::2].sub_(transform.pad_left).div_(transform.scale_x)
        value[:, 1::2].sub_(transform.pad_top).div_(transform.scale_y)
        value[:, 0::2].clamp_(0, transform.source_width)
        value[:, 1::2].clamp_(0, transform.source_height)
        return value

    if "boxes" in restored:
        restored["boxes"] = restore_xyxy(restored["boxes"])
    if "ellipse_mask_boxes" in restored:
        restored["ellipse_mask_boxes"] = restore_xyxy(restored["ellipse_mask_boxes"])
    if "ellipses" in restored:
        ellipses = restored["ellipses"].clone()
        ellipses[:, 0].sub_(transform.pad_left).div_(transform.scale_x)
        ellipses[:, 1].sub_(transform.pad_top).div_(transform.scale_y)
        ellipses[:, 2:4].div_(0.5 * (transform.scale_x + transform.scale_y))
        restored["ellipses"] = ellipses
    if "keypoints" in restored:
        points = restored["keypoints"].clone()
        points[..., 0].sub_(transform.pad_left).div_(transform.scale_x)
        points[..., 1].sub_(transform.pad_top).div_(transform.scale_y)
        restored["keypoints"] = points
    return restored


__all__ = [
    "FusedVideoPreprocessor",
    "LetterboxTransform",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "letterbox_transform",
    "restore_result",
]
