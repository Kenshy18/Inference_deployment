"""Fixed-shape host preprocessing for MH0 batch inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


INPUT_HEIGHT = 736
INPUT_WIDTH = 1280
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def prepare_batch_host(frames: list[np.ndarray]) -> dict[str, list[Any]]:
    if not frames:
        raise ValueError("frames must not be empty")
    batch = np.empty(
        (len(frames), INPUT_HEIGHT, INPUT_WIDTH, 3),
        dtype=np.float32,
    )
    metadata: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        original_height, original_width = frame.shape[:2]
        scale = min(
            INPUT_HEIGHT / original_height,
            INPUT_WIDTH / original_width,
        )
        resized_height = int(original_height * scale)
        resized_width = int(original_width * scale)
        pad_top = (INPUT_HEIGHT - resized_height) // 2
        pad_left = (INPUT_WIDTH - resized_width) // 2
        canvas = np.full(
            (INPUT_HEIGHT, INPUT_WIDTH, 3),
            128,
            dtype=np.uint8,
        )
        canvas[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = canvas[:, :, [2, 1, 0]].astype(np.float32)
        batch[index] = (rgb - MEAN) / STD
        scale_factor = np.asarray([scale, scale, scale, scale], dtype=np.float32)
        metadata.append(
            {
                "filename": None,
                "ori_filename": None,
                "ori_shape": (original_height, original_width, 3),
                "img_shape": (INPUT_HEIGHT, INPUT_WIDTH, 3),
                "pad_shape": (INPUT_HEIGHT, INPUT_WIDTH, 3),
                "scale_factor": scale_factor,
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": MEAN.copy(),
                    "std": STD.copy(),
                    "to_rgb": True,
                },
            }
        )
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return {"img": [tensor], "img_metas": [metadata]}


def move_batch_to_device(
    prepared: dict[str, list[Any]],
    device: str,
) -> dict[str, list[Any]]:
    tensor = prepared["img"][0].to(device, non_blocking=True)
    return {"img": [tensor], "img_metas": prepared["img_metas"]}


def prepare_fixed_b2(
    frames: list[np.ndarray],
    device: str,
) -> tuple[dict[str, list[Any]], int]:
    if len(frames) not in (1, 2):
        raise ValueError(f"fixed-B2 input requires one or two frames, got {len(frames)}")
    valid_count = len(frames)
    if valid_count == 1:
        frames = [frames[0], frames[0]]
    return move_batch_to_device(prepare_batch_host(frames), device), valid_count


__all__ = [
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "move_batch_to_device",
    "prepare_batch_host",
    "prepare_fixed_b2",
]
