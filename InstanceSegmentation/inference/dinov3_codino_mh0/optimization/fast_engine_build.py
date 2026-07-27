"""Offline builders for the RTX 5090 fixed-B16 MH0 TensorRT group."""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROTECTED_TYPES = {"NORMALIZATION", "SOFTMAX"}
SKIP_FORCE_TYPES = {"CAST", "SHAPE", "CONSTANT", "FILL", "DECONVOLUTION"}
EXPECTED_PROTECTED = {"query_encoder": 3, "decoder_fp16": 18}
EXPECTED_PLUGIN_NODES = {
    "query_encoder": 1,
    "decoder": 3,
    "decoder_fp16": 3,
}
MAX_AUX_STREAMS = {
    "backbone": 0,
    "query_encoder": 2,
    "decoder": 2,
    "decoder_fp16": 2,
    "mask_head": 0,
    "mask_head_fp16": 0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_shape_tensor(tensor: Any) -> bool:
    value = getattr(tensor, "is_shape_tensor", False)
    return bool(value() if callable(value) else value)


def _short_type(layer: Any) -> str:
    return str(layer.type).rsplit(".", 1)[-1].upper()


def _floating_outputs(layer: Any, trt: Any) -> list[int]:
    float_types = {trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16}
    return [
        index
        for index in range(layer.num_outputs)
        if (tensor := layer.get_output(index)) is not None
        and not _is_shape_tensor(tensor)
        and tensor.dtype in float_types
    ]


def _protect_sensitive_layers(network: Any, trt: Any, component: str) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        kind = _short_type(layer)
        if kind not in PROTECTED_TYPES:
            continue
        outputs = _floating_outputs(layer, trt)
        _require(outputs, f"{component} protected layer has no floating output: {layer.name}")
        layer.precision = trt.DataType.FLOAT
        for output_index in outputs:
            layer.set_output_type(output_index, trt.DataType.FLOAT)
        selected.append(
            {
                "index": index,
                "name": layer.name,
                "type": kind,
                "outputs": outputs,
            }
        )
    expected = EXPECTED_PROTECTED[component]
    _require(
        len(selected) == expected,
        f"{component} protected-layer drift: expected {expected}, got {len(selected)}",
    )
    return {
        "count": len(selected),
        "type_counts": dict(sorted(Counter(row["type"] for row in selected).items())),
        "layers": selected,
    }


def _force_backbone_bf16(network: Any, trt: Any) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_count = 0
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        kind = _short_type(layer)
        if kind in SKIP_FORCE_TYPES:
            continue
        outputs = _floating_outputs(layer, trt)
        if not outputs:
            continue
        layer.precision = trt.DataType.BF16
        for output_index in outputs:
            layer.set_output_type(output_index, trt.DataType.BF16)
        counts[kind] += 1
        output_count += len(outputs)
    _require(
        sum(counts.values()) == 2_099,
        f"backbone BF16 layer drift: expected 2099, got {sum(counts.values())}",
    )
    return {
        "layer_count": sum(counts.values()),
        "output_count": output_count,
        "type_counts": dict(sorted(counts.items())),
    }


def _force_public_outputs_fp32(network: Any, trt: Any) -> list[dict[str, Any]]:
    producers: dict[str, tuple[Any, int]] = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = (layer, output_index)
    records = []
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        _require(tensor.name in producers, f"network output producer missing: {tensor.name}")
        producer, output_index = producers[tensor.name]
        tensor.dtype = trt.DataType.FLOAT
        producer.set_output_type(output_index, trt.DataType.FLOAT)
        records.append({"name": tensor.name, "shape": list(tensor.shape)})
    return records


def _mutate_onnx(component: str, source: Path, output: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(source))
    nodes = [
        node
        for node in model.graph.node
        if node.op_type == "MultiscaleDeformableAttnPlugin_TRT"
    ]
    expected = EXPECTED_PLUGIN_NODES.get(component, 0)
    _require(len(nodes) == expected, f"{component} MSDA-node drift: {len(nodes)}")
    if component == "query_encoder":
        for node in nodes:
            node.op_type = "MSDA_SM120"
            del node.attribute[:]
            node.attribute.append(
                onnx.helper.make_attribute("plugin_namespace", "codino")
            )
    elif component in {"decoder", "decoder_fp16"}:
        for node in nodes:
            del node.attribute[:]
            node.attribute.append(
                onnx.helper.make_attribute("plugin_version", "2")
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return {"plugin_nodes": len(nodes), "onnx": str(output)}


def _compile_plugin(
    runtime_root: Path, cpp: Path, cuda: Path, build_dir: Path, output: Path
) -> dict[str, Any]:
    import torch
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    extension = load(
        name="codino_msda_direct_t140",
        sources=[str(cpp), str(cuda)],
        build_directory=str(build_dir),
        verbose=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
    )
    elapsed = time.perf_counter() - started
    built = Path(extension.__file__).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, output)
    return {
        "component": "plugin",
        "elapsed_seconds": elapsed,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "runtime_root": str(runtime_root),
        "output": str(output),
        "size": output.stat().st_size,
    }


def _register_build_plugin(path: Path) -> None:
    import importlib.util
    import sys
    from typing import Tuple

    import tensorrt_bindings.plugin as trtp
    import torch

    name = "codino_msda_direct_t140"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load SM120 extension")
    extension = importlib.util.module_from_spec(spec)
    sys.modules[name] = extension
    spec.loader.exec_module(extension)

    @trtp.register("codino::MSDA_SM120")
    def description(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
    ) -> Tuple[trtp.TensorDesc]:
        del spatial_shapes, level_start_index, sampling_locations, attention_weights
        return (value.like(),)

    @trtp.impl("codino::MSDA_SM120")
    def implementation(
        value: trtp.Tensor,
        spatial_shapes: trtp.Tensor,
        level_start_index: trtp.Tensor,
        sampling_locations: trtp.Tensor,
        attention_weights: trtp.Tensor,
        outputs: Tuple[trtp.Tensor],
        stream: int,
    ) -> None:
        extension.forward_out(
            torch.as_tensor(value, device="cuda"),
            torch.as_tensor(spatial_shapes, device="cuda"),
            torch.as_tensor(level_start_index, device="cuda"),
            torch.as_tensor(sampling_locations, device="cuda"),
            torch.as_tensor(attention_weights, device="cuda"),
            torch.as_tensor(outputs[0], device="cuda"),
            int(stream),
        )

    @trtp.autotune("codino::MSDA_SM120")
    def autotune(
        value: trtp.TensorDesc,
        spatial_shapes: trtp.TensorDesc,
        level_start_index: trtp.TensorDesc,
        sampling_locations: trtp.TensorDesc,
        attention_weights: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> list[trtp.AutoTuneCombination]:
        del value, spatial_shapes, level_start_index
        del sampling_locations, attention_weights, outputs
        return [
            trtp.AutoTuneCombination(
                "FP16,INT64,INT64,FP16,FP16,FP16", "LINEAR"
            )
        ]


def _require_target_hardware(trt: Any) -> dict[str, Any]:
    import torch

    _require(torch.cuda.is_available(), "CUDA is required for the fast SM120 build")
    capability = tuple(torch.cuda.get_device_capability())
    _require(
        capability == (12, 0),
        f"fast Co-DINO build requires SM120, got SM{capability[0]}{capability[1]}",
    )
    trt_version = tuple(int(part) for part in trt.__version__.split(".")[:2])
    _require(
        trt_version == (10, 13),
        f"fast Co-DINO build requires TensorRT 10.13, got {trt.__version__}",
    )
    return {
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
    }


def _build_engine(
    component: str,
    onnx_path: Path,
    engine_path: Path,
    cache_path: Path,
    plugin: Path | None,
    workspace_gb: int,
) -> dict[str, Any]:
    import tensorrt as trt

    hardware = _require_target_hardware(trt)
    trt.init_libnvinfer_plugins(None, "")
    if component == "query_encoder":
        _require(plugin is not None, "query encoder requires the SM120 plugin")
        _register_build_plugin(plugin)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if component == "query_encoder":
        network_flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS
        )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    _require(
        parser.parse_from_file(str(onnx_path)),
        "TensorRT ONNX parse failed:\n"
        + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)),
    )
    parsed_layers = network.num_layers
    protected = None
    bf16 = None
    if component in EXPECTED_PROTECTED:
        protected = _protect_sensitive_layers(network, trt, component)
    if component == "backbone":
        bf16 = _force_backbone_bf16(network, trt)
    public_outputs = _force_public_outputs_fp32(network, trt)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30
    )
    if component == "backbone":
        config.set_flag(trt.BuilderFlag.BF16)
    elif component in {"query_encoder", "decoder_fp16", "mask_head_fp16"}:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.max_aux_streams = MAX_AUX_STREAMS[component]
    if component == "mask_head":
        config.avg_timing_iterations = 24
    cache = config.create_timing_cache(b"")
    _require(
        cache is not None and config.set_timing_cache(cache, ignore_mismatch=False),
        "cannot attach empty editable timing cache",
    )
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    _require(serialized is not None, f"{component} TensorRT build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    _require(engine is not None, f"{component} engine deserialize failed")
    io = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io.append(
            {
                "name": name,
                "shape": list(engine.get_tensor_shape(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "mode": str(engine.get_tensor_mode(name)),
            }
        )
    inspector = json.loads(
        engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON
        )
    )
    plugin_layers = [
        row
        for row in inspector.get("Layers", [])
        if row.get("LayerType") == "PluginV3"
        and row.get("PluginType") == "MSDA_SM120"
    ]
    if component == "query_encoder":
        _require(len(plugin_layers) == 1, "serialized SM120 plugin-layer drift")
    return {
        "component": component,
        "elapsed_seconds": elapsed,
        "parsed_layers": parsed_layers,
        "execution_layers": engine.num_layers,
        "aux_streams": engine.num_aux_streams,
        "protected_fp32": protected,
        "forced_bf16": bf16,
        "public_outputs": public_outputs,
        "custom_plugin_layers": len(plugin_layers),
        "io": io,
        "engine": str(engine_path),
        "engine_size": engine_path.stat().st_size,
        "timing_cache": str(cache_path),
        "timing_cache_size": cache_path.stat().st_size,
        "hardware": hardware,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        required=True,
        choices=(
            "preflight",
            "plugin",
            "backbone",
            "query_encoder",
            "decoder",
            "decoder_fp16",
            "mask_head",
            "mask_head_fp16",
        ),
    )
    parser.add_argument("--source-onnx", type=Path)
    parser.add_argument("--build-onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--native-cpp", type=Path)
    parser.add_argument("--native-cuda", type=Path)
    parser.add_argument("--native-build-dir", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--workspace-gb", type=int, default=12)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.component == "preflight":
        import tensorrt as trt

        report = {
            "component": "preflight",
            "hardware": _require_target_hardware(trt),
        }
    elif args.component == "plugin":
        required = (
            args.runtime_root,
            args.native_cpp,
            args.native_cuda,
            args.native_build_dir,
            args.plugin,
        )
        _require(all(value is not None for value in required), "plugin arguments missing")
        report = _compile_plugin(
            args.runtime_root,
            args.native_cpp,
            args.native_cuda,
            args.native_build_dir,
            args.plugin,
        )
    else:
        required = (args.source_onnx, args.build_onnx, args.engine, args.timing_cache)
        _require(all(value is not None for value in required), "engine arguments missing")
        mutation = _mutate_onnx(args.component, args.source_onnx, args.build_onnx)
        report = _build_engine(
            args.component,
            args.build_onnx,
            args.engine,
            args.timing_cache,
            args.plugin,
            args.workspace_gb,
        )
        report["graph_mutation"] = mutation
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
