"""Direct fixed-shape Co-DINO video preprocessing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def letterbox_params(
    original_height: int,
    original_width: int,
    target_height: int,
    target_width: int,
) -> tuple[float, int, int, int, int]:
    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2
    return scale, new_height, new_width, pad_top, pad_left


def _ceil_to_multiple(value: int, divisor: int) -> int:
    return int((value + divisor - 1) // divisor * divisor)


def prepare_batch_host(
    frames: list[np.ndarray],
    target_size: tuple[int, int],
) -> dict[str, list[Any]]:
    """Run parity-stable OpenCV preprocessing without a CUDA transfer."""

    target_height, target_width = target_size
    padded_height = _ceil_to_multiple(target_height, 32)
    padded_width = _ceil_to_multiple(target_width, 32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    batch = np.zeros((len(frames), padded_height, padded_width, 3), dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        original_height, original_width = frame.shape[:2]
        scale, new_height, new_width, pad_top, pad_left = letterbox_params(
            original_height,
            original_width,
            target_height,
            target_width,
        )
        canvas = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
        if new_height > 0 and new_width > 0:
            resized = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
            canvas[
                pad_top : pad_top + new_height,
                pad_left : pad_left + new_width,
            ] = resized
        rgb = canvas[:, :, [2, 1, 0]].astype(np.float32)
        batch[index, :target_height, :target_width] = (rgb - mean) / std
        scale_factor = np.asarray([scale, scale, scale, scale], dtype=np.float32)
        metadata.append(
            {
                "filename": None,
                "ori_filename": None,
                "ori_shape": (original_height, original_width, 3),
                "img_shape": (target_height, target_width, 3),
                "pad_shape": (padded_height, padded_width, 3),
                "scale_factor": scale_factor,
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": mean.copy(),
                    "std": std.copy(),
                    "to_rgb": True,
                },
            }
        )
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return {"img": [tensor], "img_metas": [metadata]}


def move_prepared_batch(
    model,
    prepared: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Move an already-normalized batch to the model device."""

    tensor = prepared["img"][0]
    parameter = next(model.parameters())
    if parameter.is_cuda:
        tensor = tensor.to(parameter.device, non_blocking=True)
    return {"img": [tensor], "img_metas": prepared["img_metas"]}


def prepare_batch_direct(
    model,
    frames: list[np.ndarray],
    target_size: tuple[int, int],
) -> dict[str, list[Any]]:
    """Run preprocessing and one batch transfer."""

    return move_prepared_batch(
        model,
        prepare_batch_host(frames, target_size),
    )


__all__ = [
    "letterbox_params",
    "move_prepared_batch",
    "prepare_batch_direct",
    "prepare_batch_host",
]
