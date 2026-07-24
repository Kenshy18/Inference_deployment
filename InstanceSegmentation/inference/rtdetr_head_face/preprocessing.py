"""RT-DETR source-frame preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch


def letterbox(
    frame: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    target_h, target_w = size
    height, width = frame.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(frame, (resized_w, resized_h))
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, scale, (left, top)


def make_batch(
    frames: Sequence[np.ndarray],
    input_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype | None,
    channels_last: bool,
):
    tensors, sizes, paddings, scales = [], [], [], []
    for frame in frames:
        image, scale, padding = letterbox(frame, input_size)
        tensors.append(
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        sizes.append((frame.shape[0], frame.shape[1]))
        paddings.append(padding)
        scales.append(scale)
    batch = torch.stack(tensors).to(device)
    if dtype is not None:
        batch = batch.to(dtype=dtype)
    if channels_last and device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return (
        batch,
        torch.tensor(sizes, device=device, dtype=torch.float32),
        torch.tensor(paddings, device=device, dtype=torch.float32),
        torch.tensor(scales, device=device, dtype=torch.float32),
        torch.tensor([input_size] * len(frames), device=device, dtype=torch.float32),
    )


__all__ = ["letterbox", "make_batch"]
