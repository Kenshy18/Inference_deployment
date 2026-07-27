"""Fixed-shape TensorRT adapter for the MH0 DINOv3 ViT-S+ backbone."""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from pathlib import Path

import torch


def _site_packages() -> tuple[Path, ...]:
    candidates = [Path(value) for value in sys.path if "site-packages" in value]
    try:
        candidates.extend(Path(value) for value in site.getsitepackages())
    except Exception:
        pass
    return tuple(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))


def prepare_tensorrt_libraries() -> None:
    """Preload pip/conda TensorRT and CUDA libraries before importing TensorRT."""

    exact_names = (
        "tensorrt_libs/libnvinfer.so.10",
        "tensorrt_libs/libnvinfer_plugin.so.10",
        "tensorrt_libs/libnvonnxparser.so.10",
    )
    patterns = (
        "nvidia/cudnn/lib/libcudnn*.so.9",
        "nvidia/cublas/lib/libcublas*.so.12",
        "nvidia/cuda_runtime/lib/libcudart*.so*",
    )
    library_dirs: list[str] = []
    for root in _site_packages():
        trt_libs = root / "tensorrt_libs"
        if trt_libs.is_dir():
            library_dirs.append(str(trt_libs))
        paths = [root / name for name in exact_names]
        paths.extend(
            Path(value)
            for pattern in patterns
            for value in glob.glob(str(root / pattern))
        )
        for path in paths:
            if not path.is_file():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    current = os.environ.get("LD_LIBRARY_PATH")
    if current:
        library_dirs.append(current)
    if library_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)


class TensorRTBackbone(torch.nn.Module):
    """Expose one fixed-shape TensorRT engine through the DINO backbone API."""

    def __init__(self, engine_path: Path) -> None:
        super().__init__()
        prepare_tensorrt_libraries()
        import tensorrt as trt

        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT backbone engine not found: {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("could not create TensorRT execution context")
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
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one TensorRT input/output, got inputs={inputs}, outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_shape = tuple(
            int(value) for value in self.engine.get_tensor_shape(self.input_name)
        )
        self.output_shape = tuple(
            int(value) for value in self.engine.get_tensor_shape(self.output_name)
        )
        self.input_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name), trt
        )
        self.output_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.output_name), trt
        )
        self._output: torch.Tensor | None = None
        self._retained_input: torch.Tensor | None = None
        self._stream = torch.cuda.Stream()

    @staticmethod
    def _torch_dtype(value, trt) -> torch.dtype:
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise TypeError(f"unsupported TensorRT dtype: {value}") from exc

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        if not value.is_cuda:
            raise RuntimeError("TensorRT backbone requires a CUDA tensor")
        if tuple(value.shape) != self.input_shape:
            raise ValueError(
                f"TensorRT backbone requires input {self.input_shape}, got {tuple(value.shape)}"
            )
        prepared = value.to(dtype=self.input_dtype).contiguous()
        if (
            self._output is None
            or self._output.device != prepared.device
            or self._output.dtype != self.output_dtype
        ):
            self._output = torch.empty(
                self.output_shape,
                device=prepared.device,
                dtype=self.output_dtype,
            )
        self._retained_input = prepared
        self.context.set_tensor_address(self.input_name, int(prepared.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(self._output.data_ptr()))
        current_stream = torch.cuda.current_stream(device=prepared.device)
        self._stream.wait_stream(current_stream)
        prepared.record_stream(self._stream)
        self._output.record_stream(self._stream)
        if not self.context.execute_async_v3(
            stream_handle=int(self._stream.cuda_stream)
        ):
            raise RuntimeError("TensorRT backbone execution failed")
        current_stream.wait_stream(self._stream)
        return [self._output]


class TensorRTBackboneNeck(torch.nn.Module):
    """Expose a fixed-shape fused backbone+neck TensorRT engine."""

    OUTPUT_NAMES = ("p2", "p3", "p4", "p5", "p6")

    def __init__(self, engine_path: Path) -> None:
        super().__init__()
        prepare_tensorrt_libraries()
        import tensorrt as trt

        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT backbone+neck engine not found: {self.engine_path}"
            )
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"could not deserialize TensorRT engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "could not create TensorRT backbone+neck context"
            )
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
        if (
            inputs != ("input",)
            or outputs not in (self.OUTPUT_NAMES, self.OUTPUT_NAMES[:-1])
        ):
            raise RuntimeError(
                f"unexpected backbone+neck IO: inputs={inputs}, "
                f"outputs={outputs}"
            )
        self.input_name = inputs[0]
        self.output_names = outputs
        self.input_shape = tuple(
            int(value)
            for value in self.engine.get_tensor_shape(self.input_name)
        )
        self.input_dtype = TensorRTBackbone._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name), trt
        )
        self.output_shapes = {
            name: tuple(
                int(value)
                for value in self.engine.get_tensor_shape(name)
            )
            for name in outputs
        }
        self.output_dtypes = {
            name: TensorRTBackbone._torch_dtype(
                self.engine.get_tensor_dtype(name), trt
            )
            for name in outputs
        }
        self._outputs: dict[str, torch.Tensor] = {}
        self._retained_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not value.is_cuda:
            raise RuntimeError(
                "TensorRT backbone+neck input must be a CUDA tensor"
            )
        if tuple(value.shape) != self.input_shape:
            raise ValueError(
                f"TensorRT backbone+neck requires {self.input_shape}, "
                f"got {tuple(value.shape)}"
            )
        prepared = value.to(dtype=self.input_dtype).contiguous()
        self._retained_input = prepared
        self.context.set_tensor_address(
            self.input_name, int(prepared.data_ptr())
        )
        for name in self.output_names:
            output = self._outputs.get(name)
            if (
                output is None
                or output.device != prepared.device
                or output.dtype != self.output_dtypes[name]
            ):
                output = torch.empty(
                    self.output_shapes[name],
                    dtype=self.output_dtypes[name],
                    device=prepared.device,
                )
                self._outputs[name] = output
            self.context.set_tensor_address(name, int(output.data_ptr()))
        stream = torch.cuda.current_stream(device=prepared.device)
        prepared.record_stream(stream)
        for output in self._outputs.values():
            output.record_stream(stream)
        if not self.context.execute_async_v3(
            stream_handle=int(stream.cuda_stream)
        ):
            raise RuntimeError(
                f"TensorRT execution failed: {self.engine_path}"
            )
        outputs = tuple(self._outputs[name] for name in self.output_names)
        # The MH0 query and mask paths consume P2-P5 only, but the detector's
        # fixed index layout retains a fifth unused slot.
        return outputs if len(outputs) == 5 else outputs + (None,)


__all__ = [
    "TensorRTBackbone",
    "TensorRTBackboneNeck",
    "prepare_tensorrt_libraries",
]
