"""Pinned-memory and CUDA preprocessing for fixed-batch MH0 inference."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .preprocessing import INPUT_HEIGHT, INPUT_WIDTH, MEAN, STD
from .fused_preprocessing import load_fused_preprocessor, preprocess_out


class GPUPreprocessor:
    def __init__(
        self,
        device: str,
        batch_size: int = 2,
        fused_extension: Path | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self._host_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._device_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._output_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._copy_events: dict[tuple[int, int], torch.cuda.Event] = {}
        self._fused_extension = (
            load_fused_preprocessor(fused_extension)
            if fused_extension is not None
            else None
        )
        self._mean = torch.tensor(
            MEAN, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            STD, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def _host_buffer(self, height: int, width: int) -> torch.Tensor:
        key = (height, width)
        value = self._host_buffers.get(key)
        if value is None:
            value = torch.empty(
                (self.batch_size, height, width, 3),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._host_buffers[key] = value
        return value

    def prepare(
        self,
        frames: list[np.ndarray],
        stream: torch.cuda.Stream,
    ) -> tuple[dict[str, list[Any]], int]:
        if not 1 <= len(frames) <= self.batch_size:
            raise ValueError(
                f"fixed-B{self.batch_size} input requires 1..{self.batch_size} "
                f"frames, got {len(frames)}"
            )
        valid_count = len(frames)
        frames = frames + [frames[-1]] * (self.batch_size - valid_count)
        height, width = frames[0].shape[:2]
        if any(frame.shape != (height, width, 3) for frame in frames):
            raise ValueError(
                "GPU fixed-batch preprocessing requires equal BGR frame shapes"
            )
        scale = min(INPUT_HEIGHT / height, INPUT_WIDTH / width)
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        pad_top = (INPUT_HEIGHT - resized_height) // 2
        pad_left = (INPUT_WIDTH - resized_width) // 2

        host = self._host_buffer(height, width)
        copy_complete = self._copy_events.get((height, width))
        if copy_complete is None:
            copy_complete = torch.cuda.Event()
            self._copy_events[(height, width)] = copy_complete
        else:
            copy_complete.synchronize()
        host_view = host.numpy()
        for index, frame in enumerate(frames):
            np.copyto(host_view[index], frame)
        with torch.cuda.stream(stream):
            if self._fused_extension is not None:
                key = (height, width)
                source = self._device_buffers.get(key)
                if source is None:
                    source = torch.empty(
                        (self.batch_size, height, width, 3),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    self._device_buffers[key] = source
                source.copy_(host, non_blocking=True)
                copy_complete.record(stream)
                normalized = self._output_buffers.get(key)
                if normalized is None:
                    normalized = torch.empty(
                        (
                            self.batch_size,
                            3,
                            INPUT_HEIGHT,
                            INPUT_WIDTH,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self._output_buffers[key] = normalized
                preprocess_out(
                    self._fused_extension,
                    source,
                    normalized,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    stream=stream,
                )
            else:
                source = host.to(self.device, non_blocking=True)
                copy_complete.record(stream)
                source = source.permute(0, 3, 1, 2).to(torch.float32)
                resized = functional.interpolate(
                    source,
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                )
                canvas = torch.full(
                    (self.batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH),
                    128.0,
                    dtype=torch.float32,
                    device=self.device,
                )
                canvas[
                    :,
                    :,
                    pad_top : pad_top + resized_height,
                    pad_left : pad_left + resized_width,
                ] = resized
                normalized = (
                    canvas[:, [2, 1, 0]] - self._mean
                ) / self._std
                normalized = normalized.contiguous()

        scale_factor = np.asarray(
            [scale, scale, scale, scale], dtype=np.float32
        )
        metadata = [
            {
                "filename": None,
                "ori_filename": None,
                "ori_shape": (height, width, 3),
                "img_shape": (INPUT_HEIGHT, INPUT_WIDTH, 3),
                "pad_shape": (INPUT_HEIGHT, INPUT_WIDTH, 3),
                "scale_factor": scale_factor.copy(),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": MEAN.copy(),
                    "std": STD.copy(),
                    "to_rgb": True,
                },
            }
            for _ in range(self.batch_size)
        ]
        return {"img": [normalized], "img_metas": [metadata]}, valid_count


__all__ = ["GPUPreprocessor"]
