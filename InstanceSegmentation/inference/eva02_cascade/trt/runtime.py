"""Direct TensorRT replacement for the pruned EVA-02 ViT backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _torch_dtype(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}") from exc


def _resolved_shape(shape: Any, batch_size: int) -> tuple[int, ...]:
    values: list[int] = []
    for index, dimension in enumerate(shape):
        value = int(dimension)
        if value < 0 and index == 0:
            value = batch_size
        if value < 0:
            raise RuntimeError(f"unresolved TensorRT output shape: {tuple(shape)}")
        values.append(value)
    return tuple(values)


class Eva02TensorRTBackbone(nn.Module):
    """Return the same ``{"last_feat": tensor}`` contract as the PyTorch ViT."""

    def __init__(
        self,
        engine_path: Path,
        *,
        expected_input_name: str = "images",
        expected_output_name: str = "last_feat",
    ) -> None:
        super().__init__()
        import tensorrt as trt

        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"EVA-02 TensorRT engine not found: {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.logger)
        self.engine = self.trt_runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create EVA-02 TensorRT execution context")

        names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        )
        inputs = tuple(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        outputs = tuple(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        if inputs != (expected_input_name,) or outputs != (expected_output_name,):
            raise RuntimeError(
                "unexpected EVA-02 TensorRT IO contract: "
                f"inputs={inputs}, outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_dtype = _torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _torch_dtype(self.engine.get_tensor_dtype(self.output_name))
        minimum, optimum, maximum = self.engine.get_tensor_profile_shape(
            self.input_name, 0
        )
        self.min_shape = tuple(int(value) for value in minimum)
        self.opt_shape = tuple(int(value) for value in optimum)
        self.max_shape = tuple(int(value) for value in maximum)
        self._output: torch.Tensor | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._stream: torch.cuda.Stream | None = None

    def _ensure_output(self, value: torch.Tensor) -> torch.Tensor:
        if not self.context.set_input_shape(self.input_name, tuple(value.shape)):
            raise RuntimeError(f"TensorRT rejected input shape: {tuple(value.shape)}")
        shape = _resolved_shape(
            self.context.get_tensor_shape(self.output_name),
            int(value.shape[0]),
        )
        if (
            self._output is None
            or self._output_shape != shape
            or self._output.device != value.device
        ):
            self._output = torch.empty(
                shape,
                dtype=self.output_dtype,
                device=value.device,
            )
            self._output_shape = shape
        return self._output

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if not value.is_cuda:
            raise RuntimeError("EVA-02 TensorRT backbone requires a CUDA tensor")
        if value.dtype != self.input_dtype:
            value = value.to(dtype=self.input_dtype)
        value = value.contiguous()
        batch = int(value.shape[0])
        if not self.min_shape[0] <= batch <= self.max_shape[0]:
            raise ValueError(
                f"batch size {batch} is outside TensorRT profile "
                f"[{self.min_shape[0]}, {self.max_shape[0]}]"
            )
        output = self._ensure_output(value)
        self.context.set_tensor_address(self.input_name, int(value.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))

        current = torch.cuda.current_stream(device=value.device)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=value.device)
        self._stream.wait_stream(current)
        value.record_stream(self._stream)
        output.record_stream(self._stream)
        succeeded = self.context.execute_async_v3(
            stream_handle=int(self._stream.cuda_stream)
        )
        if not succeeded:
            raise RuntimeError("EVA-02 TensorRT execute_async_v3 failed")
        current.wait_stream(self._stream)
        return {self.output_name: output}


__all__ = ["Eva02TensorRTBackbone"]
