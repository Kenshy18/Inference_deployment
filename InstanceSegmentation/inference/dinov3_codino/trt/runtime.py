"""TensorRT adapters for the fixed Co-DINO partition group."""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class FixedTrtPartitionSettings:
    backbone_engine: Path
    query_encoder_engine: Path
    decoder_engine: Path
    mask_head_engine: Path
    query_encoder_shapes: tuple[tuple[int, int], ...]
    extra_site_packages: Path | None = None
    query_plugin_extension: Path | None = None

    def __post_init__(self) -> None:
        if not self.query_encoder_shapes:
            raise ValueError("query_encoder_shapes must not be empty")
        if any(
            (height <= 0 or width <= 0 for (height, width) in self.query_encoder_shapes)
        ):
            raise ValueError("query_encoder_shapes dimensions must be positive")


def parse_feature_shapes(text: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid HW shape {item!r}, expected HxW")
        (height_text, width_text) = item.split("x", 1)
        (height, width) = (int(height_text), int(width_text))
        if height < 1 or width < 1:
            raise ValueError(f"Invalid HW shape {item!r}")
        shapes.append((height, width))
    if not shapes:
        raise ValueError("query encoder shapes must define at least one HxW shape")
    return tuple(shapes)


def _collect_site_paths() -> list[str]:
    paths: list[str] = []
    candidates = [path for path in sys.path if path and "site-packages" in path]
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass
    for candidate in candidates:
        resolved = os.path.realpath(candidate)
        if resolved not in paths and os.path.isdir(resolved):
            paths.append(resolved)
    return paths


def _append_runtime_libs_to_env() -> None:
    """Expose venv TensorRT and CUDA vendor libraries to the process."""
    candidates: list[str] = []
    for site_path in _collect_site_paths():
        candidates.extend(
            [
                f"{site_path}/tensorrt_libs",
                f"{site_path}/nvidia/cudnn/lib",
                f"{site_path}/nvidia/cublas/lib",
                f"{site_path}/nvidia/cuda_runtime/lib",
            ]
        )
    configured = os.environ.get("TENSORRT_LIB_DIR")
    if configured:
        candidates.append(configured)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [path for path in candidates if os.path.isdir(path)]
    if current:
        parts.append(current)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _preload_vendor_libs() -> None:
    """Preload TensorRT/CUDA libraries when loader search paths are unreliable."""
    for site_path in _collect_site_paths():
        exact_paths = (
            os.path.join(site_path, "tensorrt_libs", "libnvinfer.so.10"),
            os.path.join(site_path, "tensorrt_libs", "libnvinfer_plugin.so.10"),
            os.path.join(site_path, "tensorrt_libs", "libnvonnxparser.so.10"),
        )
        patterns = (
            os.path.join(site_path, "nvidia", "cudnn", "lib", "libcudnn*.so.9"),
            os.path.join(site_path, "nvidia", "cublas", "lib", "libcublas*.so.12"),
            os.path.join(site_path, "nvidia", "cuda_runtime", "lib", "libcudart*.so*"),
        )
        for path in (
            *exact_paths,
            *(item for pattern in patterns for item in glob.glob(pattern)),
        ):
            if not os.path.isfile(path):
                continue
            try:
                ctypes.CDLL(path)
            except Exception:
                pass


def trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def prepare_tensorrt_runtime(extra_site_packages: Path | None) -> None:
    """Load optional packaged TRT libs without inventing a checkout-local temp path."""
    if extra_site_packages is None or not extra_site_packages.exists():
        return
    extra_site = str(extra_site_packages)
    if extra_site not in sys.path:
        sys.path.append(extra_site)
    trt_libs = extra_site_packages / "tensorrt_libs"
    plugin = trt_libs / "libnvinfer_plugin.so.10"
    if not plugin.exists():
        return
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    vc_plugin = trt_libs / "libnvinfer_vc_plugin.so.10"
    if vc_plugin.exists():
        return
    raw_temp_root = os.environ.get(
        "CODINO_TEMP_ROOT",
        "~/.local/share/dinov3-codino/tmp",
    )
    temp_root = Path(raw_temp_root).expanduser().resolve()
    shim_dir = temp_root / "trt_vc_plugin_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "libnvinfer_vc_plugin.so.10"
    if not shim.exists():
        shim.symlink_to(plugin)
    os.environ[
        "LD_LIBRARY_PATH"
    ] = f"{shim_dir}:{trt_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)


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
    for (idx, dim) in enumerate(shape):
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
            "CODINO_TRT_BACKBONE_INPUT_DTYPE"
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
        self.engine_output_dtype = trt_dtype_to_torch(
            self.engine.get_tensor_dtype(self.output_name)
        )
        self._output_tensor: torch.Tensor | None = None
        self._output_shape: tuple[int, ...] | None = None
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_BACKBONE_DEDICATED_STREAM", "0") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_output(self, x: torch.Tensor) -> torch.Tensor:
        try:
            self.context.set_input_shape(self.input_name, tuple(x.shape))
        except Exception:
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
            "CODINO_TRT_BACKBONE_INPUT_DTYPE"
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
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_BACKBONE_DEDICATED_STREAM", "0") in (
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
        for (name, tensor) in outputs.items():
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


class CoDINOTRTBackboneAdapter(torch.nn.Module):
    """Adapt one DINOv3 backbone tensor to Co-DINO's feature-list API."""

    def __init__(self, engine_path: Path, *, extra_site_packages: Path | None) -> None:
        super().__init__()
        prepare_tensorrt_runtime(extra_site_packages)
        self.trt = TensorRTBackboneAdapter(str(engine_path))

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        return [self.trt(value)["last_feat"]]


class CoDINOTRTFeatureListAdapter(torch.nn.Module):
    """Adapt a full backbone+neck engine dictionary to Co-DINO's feature list."""

    def __init__(
        self,
        engine_path: Path,
        *,
        feature_names: tuple[str, ...],
        extra_site_packages: Path | None,
    ) -> None:
        super().__init__()
        prepare_tensorrt_runtime(extra_site_packages)
        self.feature_names = feature_names
        self.trt = TensorRTFeatureDictBackboneAdapter(
            str(engine_path), feature_names=feature_names
        )

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.trt(value)
        return [outputs[name] for name in self.trt.feature_names]


class CoDINOTRTQueryEncoderAdapter(torch.nn.Module):
    """Replace Co-DINO's transformer encoder with a fixed-shape TensorRT engine."""

    def __init__(
        self,
        engine_path: Path,
        *,
        feature_shapes: tuple[tuple[int, int], ...],
        extra_site_packages: Path | None,
    ) -> None:
        super().__init__()
        prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.feature_shapes = tuple(feature_shapes)
        self.engine_path = str(engine_path.expanduser().resolve())
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
            raise RuntimeError(
                "Failed to create TensorRT query encoder execution context"
            )
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
        )
        outputs = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            )
        )
        self.uses_query_input = len(self.input_names) == 1
        if not self.uses_query_input and len(self.input_names) != len(
            self.feature_shapes
        ):
            raise RuntimeError(
                f"Expected one flattened query input or {len(self.feature_shapes)} feature inputs, got {self.input_names}"
            )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one query encoder output, got {outputs}")
        self.output_name = outputs[0]
        self.input_dtypes = {
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.input_names
        }
        self.output_dtype = trt_dtype_to_torch(
            self.engine.get_tensor_dtype(self.output_name)
        )
        self._output_tensor: torch.Tensor | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_QUERY_ENCODER_DEDICATED_STREAM", "0") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_output(self, device: torch.device) -> torch.Tensor:
        out_shape = tuple(
            (int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        )
        if (
            self._output_tensor is None
            or self._output_shape != out_shape
            or self._output_tensor.dtype != self.output_dtype
        ):
            self._output_tensor = torch.empty(
                out_shape, device=device, dtype=self.output_dtype
            )
            self._output_shape = out_shape
        return self._output_tensor

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not query.is_cuda:
            raise RuntimeError("TensorRT query encoder requires CUDA tensors")
        (total_tokens, batch_size, channels) = query.shape
        expected_tokens = sum((h * w for (h, w) in self.feature_shapes))
        if total_tokens != expected_tokens:
            raise RuntimeError(
                f"Query token count mismatch: expected {expected_tokens}, got {total_tokens}"
            )
        inputs: list[torch.Tensor] = []
        if self.uses_query_input:
            name = self.input_names[0]
            feat = query.contiguous()
            target_dtype = self.input_dtypes[name]
            if feat.dtype != target_dtype:
                feat = feat.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(feat.shape))
            except Exception:
                pass
            inputs.append(feat)
        else:
            by_channel = query.permute(1, 2, 0).contiguous()
            start = 0
            for (name, (height, width)) in zip(self.input_names, self.feature_shapes):
                end = start + height * width
                feat = (
                    by_channel[:, :, start:end]
                    .reshape(batch_size, channels, height, width)
                    .contiguous()
                )
                target_dtype = self.input_dtypes[name]
                if feat.dtype != target_dtype:
                    feat = feat.to(target_dtype)
                try:
                    self.context.set_input_shape(name, tuple(feat.shape))
                except Exception:
                    pass
                inputs.append(feat)
                start = end
        output = self._ensure_output(query.device)
        for (name, feat) in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(feat.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        current_stream = torch.cuda.current_stream(device=query.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for feat in inputs:
                feat.record_stream(self.trt_stream)
            output.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for feat in inputs:
                feat.record_stream(current_stream)
            output.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT query encoder execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        if tuple(output.shape) == tuple(query.shape):
            return output
        if (
            output.ndim == 3
            and output.shape[0] == batch_size
            and (output.shape[1] == total_tokens)
        ):
            return output.permute(1, 0, 2).contiguous()
        raise RuntimeError(
            f"Unexpected TensorRT query encoder output shape: {tuple(output.shape)}"
        )


class CoDINOTRTDecoderAdapter(torch.nn.Module):
    """Replace Co-DINO's fixed-shape transformer decoder with TensorRT."""

    def __init__(self, engine_path: Path, *, extra_site_packages: Path | None) -> None:
        super().__init__()
        prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.engine_path = str(engine_path.expanduser().resolve())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            Path(self.engine_path).read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT decoder engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT decoder execution context")
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
        )
        self.output_names = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            )
        )
        expected_inputs = ("query", "memory", "reference_points")
        expected_outputs = ("inter_states", "inter_references")
        if self.input_names != expected_inputs or self.output_names != expected_outputs:
            raise RuntimeError(
                f"Unexpected decoder engine IO: inputs={self.input_names}, outputs={self.output_names}"
            )
        self.input_dtypes = {
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.input_names
        }
        self.output_dtypes = {
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }
        self.num_layers = int(self.engine.get_tensor_shape("inter_states")[0])
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_DECODER_DEDICATED_STREAM", "0") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_outputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            out_shape = tuple((int(dim) for dim in self.context.get_tensor_shape(name)))
            out_dtype = self.output_dtypes[name]
            tensor = self._output_tensors.get(name)
            if (
                tensor is None
                or self._output_shapes.get(name) != out_shape
                or tensor.dtype != out_dtype
            ):
                tensor = torch.empty(out_shape, device=device, dtype=out_dtype)
                self._output_tensors[name] = tensor
                self._output_shapes[name] = out_shape
            outputs[name] = tensor
        return outputs

    def forward(self, query: torch.Tensor, *args, reference_points=None, **kwargs):
        memory = kwargs.get("value", None)
        if memory is None:
            raise RuntimeError("TensorRT decoder requires value=memory")
        if reference_points is None:
            raise RuntimeError("TensorRT decoder requires reference_points")
        if not query.is_cuda or not memory.is_cuda or (not reference_points.is_cuda):
            raise RuntimeError("TensorRT decoder requires CUDA tensors")
        raw_inputs = {
            "query": query.contiguous(),
            "memory": memory.contiguous(),
            "reference_points": reference_points.contiguous(),
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor)
        outputs = self._ensure_outputs(query.device)
        for (name, tensor) in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for (name, tensor) in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        current_stream = torch.cuda.current_stream(device=query.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT decoder execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return (outputs["inter_states"], outputs["inter_references"])


class CoDINOTRTMaskHeadAdapter(torch.nn.Module):
    """Replace Co-DINO's fixed-shape SimpleRefineMaskHead with TensorRT."""

    def __init__(
        self,
        original_mask_head: torch.nn.Module,
        engine_path: Path,
        *,
        extra_site_packages: Path | None,
    ) -> None:
        super().__init__()
        prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.original_mask_head = original_mask_head
        self.stage_num_classes = original_mask_head.stage_num_classes
        self.pre_upsample_last_stage = original_mask_head.pre_upsample_last_stage
        if any((int(num_classes) != 1 for num_classes in self.stage_num_classes)):
            raise RuntimeError(
                "TensorRT mask head adapter currently supports class-agnostic/1-class heads only."
            )
        self.engine_path = str(engine_path.expanduser().resolve())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            Path(self.engine_path).read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT mask head engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT mask head execution context")
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
        )
        self.output_names = tuple(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            )
        )
        expected_inputs = ("instance_feats", "semantic_feat", "rois")
        expected_core_inputs = ("instance_feats", "sem0", "sem1", "sem2")
        expected_outputs = ("stage0", "stage1", "stage2", "stage3")
        if self.input_names == expected_inputs:
            self.mode = "full"
        elif self.input_names == expected_core_inputs:
            self.mode = "core"
        else:
            raise RuntimeError(
                f"Unexpected mask head engine IO: inputs={self.input_names}, outputs={self.output_names}"
            )
        if self.output_names != expected_outputs:
            raise RuntimeError(
                f"Unexpected mask head engine outputs: {self.output_names}"
            )
        self.input_dtypes = {
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.input_names
        }
        self.output_dtypes = {
            name: trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }
        self.engine_num_rois = int(self.engine.get_tensor_shape("instance_feats")[0])
        self.engine_semantic_batch = (
            int(self.engine.get_tensor_shape("semantic_feat")[0])
            if self.mode == "full"
            else 0
        )
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_MASK_HEAD_DEDICATED_STREAM", "0") in (
            "1",
            "true",
            "True",
        ):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_outputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            out_shape = tuple((int(dim) for dim in self.context.get_tensor_shape(name)))
            out_dtype = self.output_dtypes[name]
            tensor = self._output_tensors.get(name)
            if (
                tensor is None
                or self._output_shapes.get(name) != out_shape
                or tensor.dtype != out_dtype
            ):
                tensor = torch.empty(out_shape, device=device, dtype=out_dtype)
                self._output_tensors[name] = tensor
                self._output_shapes[name] = out_shape
            outputs[name] = tensor
        return outputs

    def _pad_inputs(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_rois = int(rois.shape[0])
        if num_rois > self.engine_num_rois:
            raise RuntimeError(
                "Internal error: _pad_inputs received too many RoIs for the TensorRT mask engine."
            )
        padded_instance_feats = instance_feats.contiguous()
        padded_rois = rois.contiguous()
        if num_rois < self.engine_num_rois:
            feat_pad = torch.zeros(
                (self.engine_num_rois - num_rois, *instance_feats.shape[1:]),
                device=instance_feats.device,
                dtype=instance_feats.dtype,
            )
            roi_pad = torch.zeros(
                (self.engine_num_rois - num_rois, rois.shape[1]),
                device=rois.device,
                dtype=rois.dtype,
            )
            padded_instance_feats = torch.cat(
                (padded_instance_feats, feat_pad), dim=0
            ).contiguous()
            padded_rois = torch.cat((padded_rois, roi_pad), dim=0).contiguous()
        semantic_batch = int(semantic_feat.shape[0])
        if semantic_batch > self.engine_semantic_batch:
            raise RuntimeError(
                f"TensorRT mask head engine expects semantic batch <= {self.engine_semantic_batch}, got {semantic_batch}."
            )
        padded_semantic_feat = semantic_feat.contiguous()
        if semantic_batch < self.engine_semantic_batch:
            sem_pad = torch.zeros(
                (self.engine_semantic_batch - semantic_batch, *semantic_feat.shape[1:]),
                device=semantic_feat.device,
                dtype=semantic_feat.dtype,
            )
            padded_semantic_feat = torch.cat(
                (padded_semantic_feat, sem_pad), dim=0
            ).contiguous()
        return (padded_instance_feats, padded_semantic_feat, padded_rois, num_rois)

    def _execute_fixed(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (instance_feats, semantic_feat, rois, num_rois) = self._pad_inputs(
            instance_feats, semantic_feat, rois
        )
        raw_inputs = {
            "instance_feats": instance_feats,
            "semantic_feat": semantic_feat,
            "rois": rois,
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor.contiguous())
        outputs = self._ensure_outputs(instance_feats.device)
        for (name, tensor) in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for (name, tensor) in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        current_stream = torch.cuda.current_stream(device=instance_feats.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT mask head execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return tuple((outputs[name][:num_rois] for name in self.output_names))

    def _compute_core_semantic_rois(
        self, semantic_feat: torch.Tensor, rois: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        for conv in self.original_mask_head.semantic_convs:
            semantic_feat = conv(semantic_feat)
        roi_feats = []
        rois_fp32 = rois.float() if rois.dtype != torch.float32 else rois
        for stage in self.original_mask_head.stages:
            transformed = stage.relu(stage.semantic_transform_in(semantic_feat))
            transformed_fp32 = (
                transformed.float()
                if transformed.dtype != torch.float32
                else transformed
            )
            roi_feats.append(
                stage.semantic_roi_extractor([transformed_fp32], rois_fp32)
            )
        return tuple(roi_feats)

    def _pad_core_inputs(
        self, instance_feats: torch.Tensor, semantic_rois: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
        num_rois = int(instance_feats.shape[0])
        if num_rois > self.engine_num_rois:
            raise RuntimeError(
                "Internal error: _pad_core_inputs received too many RoIs for the TensorRT mask engine."
            )
        padded_instance_feats = instance_feats.contiguous()
        padded_semantic_rois = [feat.contiguous() for feat in semantic_rois]
        if num_rois < self.engine_num_rois:
            feat_pad = torch.zeros(
                (self.engine_num_rois - num_rois, *instance_feats.shape[1:]),
                device=instance_feats.device,
                dtype=instance_feats.dtype,
            )
            padded_instance_feats = torch.cat(
                (padded_instance_feats, feat_pad), dim=0
            ).contiguous()
            padded = []
            for feat in padded_semantic_rois:
                sem_pad = torch.zeros(
                    (self.engine_num_rois - num_rois, *feat.shape[1:]),
                    device=feat.device,
                    dtype=feat.dtype,
                )
                padded.append(torch.cat((feat, sem_pad), dim=0).contiguous())
            padded_semantic_rois = padded
        return (padded_instance_feats, tuple(padded_semantic_rois), num_rois)

    def _execute_core_fixed(
        self, instance_feats: torch.Tensor, semantic_rois: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (instance_feats, semantic_rois, num_rois) = self._pad_core_inputs(
            instance_feats, semantic_rois
        )
        raw_inputs = {
            "instance_feats": instance_feats,
            "sem0": semantic_rois[0],
            "sem1": semantic_rois[1],
            "sem2": semantic_rois[2],
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor.contiguous())
        outputs = self._ensure_outputs(instance_feats.device)
        for (name, tensor) in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for (name, tensor) in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        current_stream = torch.cuda.current_stream(device=instance_feats.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT mask head execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return tuple((outputs[name][:num_rois] for name in self.output_names))

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        if (
            not instance_feats.is_cuda
            or not semantic_feat.is_cuda
            or (not rois.is_cuda)
        ):
            raise RuntimeError("TensorRT mask head requires CUDA tensors")
        num_rois = int(rois.shape[0])
        if num_rois == 0:
            return self.original_mask_head(
                instance_feats, semantic_feat, rois, roi_labels
            )
        if self.mode == "core":
            semantic_rois = self._compute_core_semantic_rois(semantic_feat, rois)
            if num_rois <= self.engine_num_rois:
                return (
                    list(self._execute_core_fixed(instance_feats, semantic_rois)),
                    [],
                )
            stage_chunks = [[] for _ in self.output_names]
            for start in range(0, num_rois, self.engine_num_rois):
                end = min(start + self.engine_num_rois, num_rois)
                chunk_outputs = self._execute_core_fixed(
                    instance_feats[start:end],
                    tuple((feat[start:end] for feat in semantic_rois)),
                )
                for (idx, tensor) in enumerate(chunk_outputs):
                    stage_chunks[idx].append(tensor.clone())
            return ([torch.cat(chunks, dim=0) for chunks in stage_chunks], [])
        if num_rois <= self.engine_num_rois:
            return (list(self._execute_fixed(instance_feats, semantic_feat, rois)), [])
        stage_chunks = [[] for _ in self.output_names]
        for start in range(0, num_rois, self.engine_num_rois):
            end = min(start + self.engine_num_rois, num_rois)
            chunk_outputs = self._execute_fixed(
                instance_feats[start:end], semantic_feat, rois[start:end]
            )
            for (idx, tensor) in enumerate(chunk_outputs):
                stage_chunks[idx].append(tensor.clone())
        return ([torch.cat(chunks, dim=0) for chunks in stage_chunks], [])

    def get_seg_masks(self, *args, **kwargs):
        return self.original_mask_head.get_seg_masks(*args, **kwargs)


def install_fixed_partitions(model, settings: FixedTrtPartitionSettings):
    """Install all four mutually compatible partitions as one family unit."""
    engine_paths = (
        settings.backbone_engine,
        settings.query_encoder_engine,
        settings.decoder_engine,
        settings.mask_head_engine,
    )
    missing = [path for path in engine_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Co-DINO TensorRT engines not found: {missing}")
    transformer = getattr(getattr(model, "query_head", None), "transformer", None)
    if transformer is None:
        raise RuntimeError("Model has no query_head.transformer to patch")
    if not hasattr(model, "mask_head"):
        raise RuntimeError("Model has no mask_head to patch")
    if settings.query_plugin_extension is not None:
        try:
            from .sm120_msda import register_sm120_msda_plugin
        except ImportError:
            from sm120_msda import register_sm120_msda_plugin

        register_sm120_msda_plugin(settings.query_plugin_extension)
    model.backbone = CoDINOTRTBackboneAdapter(
        settings.backbone_engine, extra_site_packages=settings.extra_site_packages
    ).eval()
    transformer.encoder = CoDINOTRTQueryEncoderAdapter(
        settings.query_encoder_engine,
        feature_shapes=settings.query_encoder_shapes,
        extra_site_packages=settings.extra_site_packages,
    ).eval()
    transformer.decoder = CoDINOTRTDecoderAdapter(
        settings.decoder_engine, extra_site_packages=settings.extra_site_packages
    ).eval()
    model.mask_head = CoDINOTRTMaskHeadAdapter(
        model.mask_head,
        settings.mask_head_engine,
        extra_site_packages=settings.extra_site_packages,
    ).eval()
    return model
