"""Offline TensorRT engine construction for the DINOv3 backbone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from .runtime_libraries import _append_runtime_libs_to_env, _preload_vendor_libs
except ImportError:
    from runtime_libraries import _append_runtime_libs_to_env, _preload_vendor_libs


def _is_shape_tensor(tensor: Any) -> bool:
    attr = getattr(tensor, "is_shape_tensor", False)
    if callable(attr):
        try:
            return bool(attr())
        except Exception:
            return False
    return bool(attr)


def build_engine_from_onnx(
    *,
    onnx_path: str | os.PathLike[str],
    engine_path: str | os.PathLike[str],
    precision: str = "bf16",
    min_shape: tuple[int, int, int, int] | None = None,
    opt_shape: tuple[int, int, int, int] | None = None,
    max_shape: tuple[int, int, int, int] | None = None,
    workspace_bytes: int = 8 << 30,
    force_layer_precision: bool = False,
) -> Path:
    import tensorrt as trt

    _append_runtime_libs_to_env()
    _preload_vendor_libs()

    onnx_path = Path(onnx_path).expanduser().resolve()
    engine_path = Path(engine_path).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(
        trt.Logger.INFO if os.environ.get("EVA_TRT_VERBOSE") else trt.Logger.WARNING
    )
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    # Use the file-path parser so TensorRT can resolve ONNX external-data
    # sidecar files next to the model, e.g. "*.onnx.data".
    if hasattr(parser, "parse_from_file"):
        parsed = parser.parse_from_file(str(onnx_path))
    else:
        parsed = parser.parse(onnx_path.read_bytes())
    if not parsed:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    precision = precision.lower()
    if precision in ("bf16", "bfloat16"):
        config.set_flag(trt.BuilderFlag.BF16)
        target_dtype = trt.DataType.BF16
    elif precision in ("fp16", "float16"):
        config.set_flag(trt.BuilderFlag.FP16)
        target_dtype = trt.DataType.HALF
    elif precision in ("fp32", "float32"):
        target_dtype = trt.DataType.FLOAT
    else:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")

    if force_layer_precision:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        float_dtypes = {trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16}
        for layer_idx in range(network.num_layers):
            layer = network.get_layer(layer_idx)
            if str(layer.type) in {
                "LayerType.CAST",
                "LayerType.SHAPE",
                "LayerType.CONSTANT",
                "LayerType.FILL",
                "LayerType.DECONVOLUTION",
            }:
                continue
            float_outputs = []
            for output_idx in range(layer.num_outputs):
                tensor = layer.get_output(output_idx)
                if (
                    tensor is not None
                    and (not _is_shape_tensor(tensor))
                    and tensor.dtype in float_dtypes
                ):
                    float_outputs.append(output_idx)
            if not float_outputs:
                continue
            try:
                layer.precision = target_dtype
            except Exception:
                pass
            for output_idx in float_outputs:
                try:
                    layer.set_output_type(output_idx, target_dtype)
                except Exception:
                    pass

    if min_shape is not None or opt_shape is not None or max_shape is not None:
        if not (min_shape and opt_shape and max_shape):
            raise ValueError(
                "min_shape, opt_shape, and max_shape must be provided together"
            )
        input_name = network.get_input(0).name
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))
    return engine_path


__all__ = ["build_engine_from_onnx"]
