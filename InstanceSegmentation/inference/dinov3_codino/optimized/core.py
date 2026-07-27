"""Fixed-pointer CUDA Graph for the optimized Co-DINO detector core."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as functional


def _amp_dtype(amp: str) -> torch.dtype:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested, but this GPU does not support bf16")
        return torch.bfloat16
    if amp == "off":
        return torch.float16
    raise ValueError(f"unsupported amp mode: {amp!r}")


@contextmanager
def _capture_safe_spatial_shapes(
    model,
    device: torch.device,
    image_metadata,
    retained_tensors: list[torch.Tensor],
):
    """Reuse fixed query tensors instead of capture-illegal allocations."""

    encoder = model.query_head.transformer.encoder
    transformer = model.query_head.transformer
    shapes = tuple(tuple(value) for value in encoder.feature_shapes)
    fixed = torch.tensor(shapes, device=device, dtype=torch.long)
    batch_size = 2
    input_height, input_width = image_metadata[0]["batch_input_shape"]
    image_masks = torch.ones(
        (batch_size, input_height, input_width),
        device=device,
        dtype=torch.float32,
    )
    for index, metadata in enumerate(image_metadata):
        image_height, image_width = metadata["img_shape"][:2]
        image_masks[index, :image_height, :image_width] = 0
    level_masks = [
        functional.interpolate(image_masks[None], size=shape)
        .to(torch.bool)
        .squeeze(0)
        for shape in shapes
    ]
    valid_ratios = torch.stack(
        [transformer.get_valid_ratio(mask) for mask in level_masks],
        1,
    )
    reference_points = transformer.get_reference_points(
        fixed,
        valid_ratios,
        device,
    )
    token_count = sum(height * width for height, width in shapes)
    empty_memory = torch.zeros(
        (batch_size, token_count, transformer.embed_dims),
        device=device,
        dtype=torch.float32,
    )
    empty_mask = torch.cat([mask.flatten(1) for mask in level_masks], 1)
    _, proposals = transformer.gen_encoder_output_proposals(
        empty_memory,
        empty_mask,
        fixed,
    )
    proposal_valid = torch.isfinite(proposals).all(-1, keepdim=True)
    retained_tensors.extend(
        [
            fixed,
            reference_points,
            empty_mask,
            proposals,
            proposal_valid,
        ]
    )
    original_as_tensor = torch.as_tensor
    original_reference_points = transformer.get_reference_points
    original_proposals = transformer.gen_encoder_output_proposals

    def as_tensor(value, *args, **kwargs):
        requested_device = kwargs.get("device")
        requested_dtype = kwargs.get("dtype")
        if (
            isinstance(value, list)
            and tuple(tuple(item) for item in value) == shapes
            and requested_dtype == torch.long
            and torch.device(requested_device) == device
        ):
            return fixed
        return original_as_tensor(value, *args, **kwargs)

    torch.as_tensor = as_tensor

    def get_reference_points(spatial_shapes, valid_ratios, device):
        del spatial_shapes, valid_ratios, device
        return reference_points

    def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes):
        del memory_padding_mask, spatial_shapes
        output_memory = memory.masked_fill(
            empty_mask.unsqueeze(-1),
            float(0),
        )
        output_memory = output_memory.masked_fill(~proposal_valid, float(0))
        output_memory = transformer.enc_output_norm(
            transformer.enc_output(output_memory)
        )
        return output_memory, proposals

    transformer.get_reference_points = get_reference_points
    transformer.gen_encoder_output_proposals = gen_encoder_output_proposals
    try:
        yield
    finally:
        torch.as_tensor = original_as_tensor
        transformer.get_reference_points = original_reference_points
        transformer.gen_encoder_output_proposals = original_proposals


class FixedB2DetectorGraph:
    """Capture feature extraction and query decoding on one stable B2 input."""

    def __init__(self, model, *, amp: str, warmup_replays: int = 1) -> None:
        self.model = model
        self.amp = amp
        self.warmup_replays = max(1, int(warmup_replays))
        self._graph: torch.cuda.CUDAGraph | None = None
        self._stream: torch.cuda.Stream | None = None
        self._static_input: torch.Tensor | None = None
        self._outputs: tuple[Any, ...] | None = None
        self._features: Any = None
        self._capture_constants: list[torch.Tensor] = []

    def _forward(self, image: torch.Tensor, image_metadata):
        features = self.model.extract_feat(image, image_metadata)
        outputs = self.model.query_head.forward(features, image_metadata)
        return outputs, outputs[-1]

    def _capture(self, image: torch.Tensor, image_metadata) -> None:
        if not image.is_cuda or int(image.shape[0]) != 2:
            raise RuntimeError("optimized Co-DINO core requires fixed CUDA batch 2")
        static_input = torch.empty_like(image)
        static_input.copy_(image)
        stream = torch.cuda.Stream(device=image.device)
        current = torch.cuda.current_stream(device=image.device)
        stream.wait_stream(current)
        dtype = _amp_dtype(self.amp)
        with torch.cuda.stream(stream), torch.inference_mode():
            with torch.amp.autocast(
                "cuda",
                dtype=dtype,
                enabled=self.amp != "off",
            ):
                for _ in range(self.warmup_replays):
                    self._forward(static_input, image_metadata)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with _capture_safe_spatial_shapes(
            self.model,
            image.device,
            image_metadata,
            self._capture_constants,
        ):
            with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
                with torch.amp.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=self.amp != "off",
                ):
                    outputs, features = self._forward(
                        static_input,
                        image_metadata,
                    )
        self._static_input = static_input
        self._stream = stream
        self._graph = graph
        self._outputs = outputs
        self._features = features
        graph.replay()

    def run(self, image: torch.Tensor, image_metadata):
        if self._graph is None:
            self._capture(image, image_metadata)
        else:
            assert self._static_input is not None
            if (
                image.shape != self._static_input.shape
                or image.dtype != self._static_input.dtype
                or image.device != self._static_input.device
            ):
                raise RuntimeError(
                    "optimized Co-DINO received input outside its captured contract"
                )
            self._static_input.copy_(image)
            self._graph.replay()
        assert self._outputs is not None
        return self._outputs, self._features

    def run_host(self, image: torch.Tensor, image_metadata):
        """Copy a pinned host batch directly into the captured input buffer."""

        if image.is_cuda:
            return self.run(image, image_metadata)
        if not image.is_pinned():
            raise RuntimeError(
                "optimized Co-DINO host input must use pinned memory"
            )
        if self._graph is None:
            device = next(self.model.parameters()).device
            staged = image.to(device, non_blocking=True)
            return self.run(staged, image_metadata)
        assert self._static_input is not None
        if (
            image.shape != self._static_input.shape
            or image.dtype != self._static_input.dtype
        ):
            raise RuntimeError(
                "optimized Co-DINO received host input outside its "
                "captured contract"
            )
        self._static_input.copy_(image, non_blocking=True)
        self._graph.replay()
        assert self._outputs is not None
        return self._outputs, self._features


__all__ = ["FixedB2DetectorGraph"]
