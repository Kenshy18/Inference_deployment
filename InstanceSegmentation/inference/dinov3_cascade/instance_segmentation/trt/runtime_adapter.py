"""Direct TensorRT backbone adapters for the DINOv3 family hot path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .runtime_libraries import _append_runtime_libs_to_env, _preload_vendor_libs


def _trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _torch_dtype_for_input(name: str) -> torch.dtype:
    value = os.environ.get(name, "fp32").lower()
    if value in ("fp32", "float32"):
        return torch.float32
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype in {name}: {value}")


def _shape_tuple(shape: Any, fallback_batch: int | None = None) -> tuple[int, ...]:
    out: list[int] = []
    for idx, dim in enumerate(shape):
        value = int(dim)
        if value < 0:
            if idx == 0 and fallback_batch is not None:
                value = int(fallback_batch)
            else:
                raise RuntimeError(
                    f"Dynamic dimension was not resolved: {tuple(shape)}"
                )
        out.append(value)
    return tuple(out)


class TensorRTBackboneAdapter(nn.Module):
    def __init__(
        self,
        engine_path: str | os.PathLike[str],
        *,
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        import tensorrt as trt

        _append_runtime_libs_to_env()
        _preload_vendor_libs()

        self.engine_path = str(Path(engine_path).expanduser().resolve())
        self.input_dtype = input_dtype or _torch_dtype_for_input(
            "EVA_TRT_BACKBONE_INPUT_DTYPE"
        )
        self.output_dtype = output_dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = Path(self.engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one input and one output, got inputs={inputs}, outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.engine_output_dtype = _trt_dtype_to_torch(
            self.engine.get_tensor_dtype(self.output_name)
        )
        self._output_tensor: torch.Tensor | None = None
        self._output_shape: tuple[int, ...] | None = None
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("EVA_TRT_BACKBONE_DEDICATED_STREAM", "1") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_output(self, x: torch.Tensor) -> torch.Tensor:
        try:
            self.context.set_input_shape(self.input_name, tuple(x.shape))
        except Exception:
            # Static engines on TensorRT 10 can reject set_input_shape even
            # when the incoming shape already matches the engine.
            pass
        out_shape = _shape_tuple(
            self.context.get_tensor_shape(self.output_name),
            fallback_batch=int(x.shape[0]),
        )
        out_dtype = self.output_dtype or self.engine_output_dtype
        if (
            self._output_tensor is None
            or self._output_shape != out_shape
            or self._output_tensor.dtype != out_dtype
        ):
            self._output_tensor = torch.empty(
                out_shape, device=x.device, dtype=out_dtype
            )
            self._output_shape = out_shape
        return self._output_tensor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError("TensorRTBackboneAdapter requires CUDA input")
        if x.dtype != self.input_dtype:
            x = x.to(self.input_dtype)
        x = x.contiguous()
        y = self._ensure_output(x)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(y.data_ptr()))
        current_stream = torch.cuda.current_stream(device=x.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            x.record_stream(self.trt_stream)
            y.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            stream = current_stream.cuda_stream
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return {"last_feat": y}


class TensorRTFeatureDictBackboneAdapter(nn.Module):
    def __init__(
        self,
        engine_path: str | os.PathLike[str],
        *,
        feature_names: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6"),
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
        size_divisibility: int = 0,
        padding_constraints: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        import tensorrt as trt

        _append_runtime_libs_to_env()
        _preload_vendor_libs()

        self.engine_path = str(Path(engine_path).expanduser().resolve())
        self.feature_names = tuple(feature_names)
        self.input_dtype = input_dtype or _torch_dtype_for_input(
            "EVA_TRT_BACKBONE_INPUT_DTYPE"
        )
        self.output_dtype = output_dtype
        self.size_divisibility = size_divisibility
        self.padding_constraints = padding_constraints or {
            "size_divisiblity": 32,
            "square_size": 0,
        }
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            Path(self.engine_path).read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(inputs) != 1:
            raise RuntimeError(f"Expected one input, got inputs={inputs}")
        if tuple(outputs) != self.feature_names:
            if len(outputs) != len(self.feature_names):
                raise RuntimeError(
                    f"Expected outputs {self.feature_names}, got {outputs}"
                )
            self.feature_names = tuple(outputs)
        self.input_name = inputs[0]
        self.output_names = tuple(outputs)
        self.engine_output_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("EVA_TRT_BACKBONE_DEDICATED_STREAM", "1") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        try:
            self.context.set_input_shape(self.input_name, tuple(x.shape))
        except Exception:
            pass
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = _shape_tuple(
                self.context.get_tensor_shape(name), fallback_batch=int(x.shape[0])
            )
            dtype = self.output_dtype or self.engine_output_dtypes[name]
            tensor = self._output_tensors.get(name)
            if (
                tensor is None
                or self._output_shapes.get(name) != shape
                or tensor.dtype != dtype
            ):
                tensor = torch.empty(shape, device=x.device, dtype=dtype)
                self._output_tensors[name] = tensor
                self._output_shapes[name] = shape
            outputs[name] = tensor
        return outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError("TensorRTFeatureDictBackboneAdapter requires CUDA input")
        if x.dtype != self.input_dtype:
            x = x.to(self.input_dtype)
        x = x.contiguous()
        outputs = self._ensure_outputs(x)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        current_stream = torch.cuda.current_stream(device=x.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            x.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            stream = current_stream.cuda_stream
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return outputs


__all__ = [
    "TensorRTBackboneAdapter",
    "TensorRTFeatureDictBackboneAdapter",
]
