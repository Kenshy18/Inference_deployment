"""Packed and unpacked EVA02 letterbox preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class LetterboxParameters:
    scale: float
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int


class LetterboxTransform:
    def __init__(
        self, src_shape: tuple[int, int], target_size: int, pad_value: int = 128
    ) -> None:
        height, width = src_shape[:2]
        self.scale = min(target_size / height, target_size / width)
        self.new_h = int(height * self.scale)
        self.new_w = int(width * self.scale)
        pad_h = target_size - self.new_h
        pad_w = target_size - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        self.pad_value = int(pad_value)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR
        )
        border_value = self.pad_value if image.ndim == 2 else (self.pad_value,) * 3
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )


def compute_letterbox_parameters(
    height: int, width: int, target_size: int
) -> LetterboxParameters:
    """Preserve the validated restore metadata rounding independently of resize truncation."""

    scale = min(target_size / height, target_size / width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    return LetterboxParameters(
        scale=scale,
        new_h=new_h,
        new_w=new_w,
        pad_top=(target_size - new_h) // 2,
        pad_left=(target_size - new_w) // 2,
    )


def normalize_bgr_image(image_bgr: np.ndarray | None) -> np.ndarray:
    if image_bgr is None:
        raise RuntimeError("Failed to read image")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    return image_bgr


def prepare_batch_inputs(
    frames_bgr: list[np.ndarray],
    target_size: int,
    device: str,
    *,
    gpu_preprocess_float: bool,
    model_half: bool,
    pin_inputs: bool = False,
    pack_inputs: bool = True,
) -> tuple[list[dict[str, object]], list[tuple[int, int, LetterboxParameters]]]:
    """Create model inputs while retaining the production packed-transfer path."""

    inputs: list[dict[str, object]] = []
    metadata: list[tuple[int, int, LetterboxParameters]] = []
    device_is_cuda = device.startswith("cuda")

    if pack_inputs and frames_bgr:
        batch_hwc = np.empty(
            (len(frames_bgr), target_size, target_size, 3), dtype=np.uint8
        )
        for index, frame_bgr in enumerate(frames_bgr):
            image_rgb = cv2.cvtColor(normalize_bgr_image(frame_bgr), cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            transform = LetterboxTransform((height, width), target_size, 128)
            batch_hwc[index] = transform.apply_image(image_rgb)
            metadata.append(
                (
                    height,
                    width,
                    compute_letterbox_parameters(height, width, target_size),
                )
            )
        batch = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).contiguous()
        if device_is_cuda and pin_inputs:
            batch = batch.pin_memory()
        if device_is_cuda and gpu_preprocess_float:
            batch = batch.to(device, non_blocking=True, dtype=torch.uint8).float()
        else:
            batch = batch.float().to(device, non_blocking=True)
        if model_half and device_is_cuda:
            batch = batch.half()
        inputs.extend(
            {"image": batch[index], "height": target_size, "width": target_size}
            for index in range(batch.shape[0])
        )
        return inputs, metadata

    for frame_bgr in frames_bgr:
        image_rgb = cv2.cvtColor(normalize_bgr_image(frame_bgr), cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        transform = LetterboxTransform((height, width), target_size, 128)
        letterboxed = transform.apply_image(image_rgb)
        tensor = torch.from_numpy(letterboxed.transpose(2, 0, 1)).contiguous()
        if device_is_cuda and pin_inputs:
            tensor = tensor.pin_memory()
        if device_is_cuda and gpu_preprocess_float:
            tensor = tensor.to(device, non_blocking=True, dtype=torch.uint8).float()
        else:
            tensor = tensor.float().to(device, non_blocking=True)
        if model_half and device_is_cuda:
            tensor = tensor.half()
        inputs.append({"image": tensor, "height": target_size, "width": target_size})
        metadata.append(
            (height, width, compute_letterbox_parameters(height, width, target_size))
        )
    return inputs, metadata


__all__ = [
    "LetterboxParameters",
    "LetterboxTransform",
    "compute_letterbox_parameters",
    "normalize_bgr_image",
    "prepare_batch_inputs",
]
