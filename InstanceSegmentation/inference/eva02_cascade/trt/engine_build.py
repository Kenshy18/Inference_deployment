#!/usr/bin/env python3
"""Build one dynamic-batch FP16 EVA-02 backbone engine from ONNX."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

try:
    from .bundle import file_record
except ImportError:
    from bundle import file_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=1280)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=12)
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--optimization-level", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (1 <= args.min_batch <= args.opt_batch <= args.max_batch):
        build_parser().error(
            "batch profile must satisfy 1 <= min <= opt <= max"
        )
    if args.target_size <= 0 or args.workspace_gb <= 0:
        build_parser().error("target size and workspace must be positive")
    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        build_parser().error(f"ONNX file not found: {onnx_path}")

    import tensorrt as trt

    started = time.perf_counter()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    parsed = (
        parser.parse_from_file(str(onnx_path))
        if hasattr(parser, "parse_from_file")
        else parser.parse(onnx_path.read_bytes())
    )
    if not parsed:
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse EVA-02 ONNX:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"expected one input/output, got {network.num_inputs}/{network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    input_tensor.dtype = trt.DataType.HALF
    output_tensor.dtype = trt.DataType.HALF

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * (1 << 30)),
    )
    config.builder_optimization_level = int(args.optimization_level)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (args.min_batch, 3, args.target_size, args.target_size),
        (args.opt_batch, 3, args.target_size, args.target_size),
        (args.max_batch, 3, args.target_size, args.target_size),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the EVA-02 backbone engine")
    engine_path = args.engine.expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("new EVA-02 TensorRT engine cannot be deserialized")
    report = {
        "schema": "eva02-backbone-trt-build-v1",
        "status": "pass",
        "onnx": file_record(onnx_path),
        "engine": file_record(engine_path),
        "tensorrt_version": trt.__version__,
        "precision": "fp16",
        "optimization_level": args.optimization_level,
        "workspace_gb": args.workspace_gb,
        "shape_profile": {
            "target_size": args.target_size,
            "min_batch": args.min_batch,
            "opt_batch": args.opt_batch,
            "max_batch": args.max_batch,
        },
        "io": {
            "input_name": input_tensor.name,
            "input_dtype": str(engine.get_tensor_dtype(input_tensor.name)),
            "output_name": output_tensor.name,
            "output_dtype": str(engine.get_tensor_dtype(output_tensor.name)),
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
