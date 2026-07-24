"""DINOv3 Cascade image normalization and letterbox preparation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


TargetSize = int | tuple[int, int]


def unpack_size(size: TargetSize) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


@dataclass(frozen=True, slots=True)
class LetterboxParameters:
    scale: float
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int


class LetterboxTransform:
    """Resize to fit and symmetrically pad to the family model shape."""

    def __init__(
        self,
        src_shape: tuple[int, int],
        target_size: TargetSize,
        pad_value: int = 128,
    ) -> None:
        height, width = src_shape
        target_h, target_w = unpack_size(target_size)
        self.scale = min(target_h / height, target_w / width)
        self.new_h = int(height * self.scale)
        self.new_w = int(width * self.scale)
        pad_h = target_h - self.new_h
        pad_w = target_w - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        self.pad_value = int(pad_value)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image,
            (self.new_w, self.new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        border_value = (
            (self.pad_value,) * image.shape[2] if image.ndim == 3 else self.pad_value
        )
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )

    def parameters(self) -> LetterboxParameters:
        return LetterboxParameters(
            scale=self.scale,
            new_h=self.new_h,
            new_w=self.new_w,
            pad_top=self.pad_top,
            pad_left=self.pad_left,
        )


@dataclass(slots=True)
class PreparedFrame:
    image_bgr: np.ndarray
    tensor_cpu: torch.Tensor
    model_height: int
    model_width: int
    letterbox: LetterboxParameters
    original_height: int
    original_width: int


def normalize_bgr_image(image_bgr: np.ndarray | None) -> np.ndarray:
    if image_bgr is None:
        raise RuntimeError("Failed to read image")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    return image_bgr


def prepare_frame(
    image_bgr: np.ndarray,
    target_size: TargetSize,
    *,
    pin_memory: bool,
) -> PreparedFrame:
    """Prepare one CPU frame; the original BGR array is retained for overlay."""

    image_bgr = normalize_bgr_image(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    target_h, target_w = unpack_size(target_size)
    transform = LetterboxTransform(
        src_shape=(original_height, original_width),
        target_size=target_size,
        pad_value=128,
    )
    letterboxed = transform.apply_image(image_rgb)
    tensor = torch.from_numpy(
        np.ascontiguousarray(letterboxed.transpose(2, 0, 1))
    ).float()
    if pin_memory:
        tensor = tensor.pin_memory()
    return PreparedFrame(
        image_bgr=image_bgr,
        tensor_cpu=tensor,
        model_height=target_h,
        model_width=target_w,
        letterbox=transform.parameters(),
        original_height=original_height,
        original_width=original_width,
    )


def materialize_model_inputs(
    frames: list[PreparedFrame], device: str
) -> list[dict[str, object]]:
    """Perform the single intended host-to-device transfer for a batch."""

    return [
        {
            "image": frame.tensor_cpu.to(device, non_blocking=True),
            "height": frame.model_height,
            "width": frame.model_width,
        }
        for frame in frames
    ]


__all__ = [
    "LetterboxParameters",
    "LetterboxTransform",
    "PreparedFrame",
    "TargetSize",
    "materialize_model_inputs",
    "normalize_bgr_image",
    "prepare_frame",
    "unpack_size",
]
