#!/usr/bin/env python3
"""Export and build a fixed-batch MH0 ViT-S+ TensorRT backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DINOV3_USE_XFORMERS", "0")

import torch


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from bootstrap import build_model

try:
    from .trt_backbone import prepare_tensorrt_libraries
except ImportError:
    from trt_backbone import prepare_tensorrt_libraries


DEFAULT_ONNX = BUNDLE_ROOT / "artifacts" / "trt" / "mh0_vitsplus_b2_736x1280.onnx"
DEFAULT_ENGINE = (
    BUNDLE_ROOT / "artifacts" / "trt" / "mh0_vitsplus_b2_736x1280_bf16.engine"
)


class BackboneExportWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.backbone(value)
        if isinstance(output, (tuple, list)):
            return output[-1]
        return output


class BackboneNeckExportWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        neck: torch.nn.Module,
        omit_p6: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.omit_p6 = bool(omit_p6)

    def forward(self, value: torch.Tensor):
        outputs = tuple(self.neck(self.backbone(value)))
        return outputs[:-1] if self.omit_p6 else outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BUNDLE_ROOT / "configs" / "source_resolved_config.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BUNDLE_ROOT / "checkpoints" / "video_pseudo_mh0_epoch6_ema_deploy.pth",
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-gb", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--with-neck", action="store_true")
    parser.add_argument("--omit-p6", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def export_onnx(args: argparse.Namespace) -> dict[str, object]:
    model = build_model(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    if args.with_neck:
        wrapper = BackboneNeckExportWrapper(
            model.backbone, model.neck, omit_p6=args.omit_p6
        ).to(args.device).eval()
        output_names = (
            ["p2", "p3", "p4", "p5"]
            if args.omit_p6
            else ["p2", "p3", "p4", "p5", "p6"]
        )
    else:
        wrapper = BackboneExportWrapper(model.backbone).to(args.device).eval()
        output_names = ["last_feat"]
    dummy = torch.randn(
        (args.batch_size, 3, 736, 1280),
        device=args.device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        sample = wrapper(dummy)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(args.onnx),
            input_names=["input"],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    torch.cuda.synchronize(args.device)
    return {
        "input_shape": list(dummy.shape),
        "output_shape": (
            [list(value.shape) for value in sample]
            if isinstance(sample, (tuple, list))
            else list(sample.shape)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "size": args.onnx.stat().st_size,
        "sha256": sha256(args.onnx),
    }


def _floating_outputs(layer, trt) -> list[int]:
    dtypes = {trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16}
    outputs: list[int] = []
    for index in range(layer.num_outputs):
        tensor = layer.get_output(index)
        if tensor is None or tensor.dtype not in dtypes:
            continue
        is_shape = getattr(tensor, "is_shape_tensor", False)
        is_shape = is_shape() if callable(is_shape) else is_shape
        if not is_shape:
            outputs.append(index)
    return outputs


def build_engine(args: argparse.Namespace) -> dict[str, object]:
    prepare_tensorrt_libraries()
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    skipped = {
        "CAST",
        "SHAPE",
        "CONSTANT",
        "FILL",
        "DECONVOLUTION",
        "RESIZE",
    }
    forced_layers = 0
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        kind = str(layer.type).rsplit(".", 1)[-1].upper()
        if kind in skipped:
            continue
        outputs = _floating_outputs(layer, trt)
        if not outputs:
            continue
        layer.precision = trt.DataType.BF16
        for output_index in outputs:
            layer.set_output_type(output_index, trt.DataType.BF16)
        forced_layers += 1

    producers: dict[str, tuple[object, int]] = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = (layer, output_index)
    for index in range(network.num_outputs):
        output = network.get_output(index)
        output.dtype = trt.DataType.FLOAT
        producer, output_index = producers[output.name]
        producer.set_output_type(output_index, trt.DataType.FLOAT)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb) << 30,
    )
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT backbone build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("new TensorRT backbone engine could not be deserialized")
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
    return {
        "elapsed_seconds": elapsed,
        "parsed_layers": network.num_layers,
        "forced_bf16_layers": forced_layers,
        "engine_layers": engine.num_layers,
        "io": io,
        "size": args.engine.stat().st_size,
        "sha256": sha256(args.engine),
    }


def main() -> int:
    args = parse_args()
    args.onnx = args.onnx.expanduser().resolve()
    args.engine = args.engine.expanduser().resolve()
    metadata = args.engine.with_suffix(".json")
    report: dict[str, object] = {}
    if metadata.is_file():
        try:
            existing = json.loads(metadata.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                report.update(existing)
        except (OSError, json.JSONDecodeError):
            pass
    report.update({
        "profile": f"mh0-vitsplus-trt-backbone-fixed-b{args.batch_size}-v1",
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(args.device),
        "capability": list(torch.cuda.get_device_capability(args.device)),
        "batch_size": args.batch_size,
        "with_neck": bool(args.with_neck),
        "omit_p6": bool(args.omit_p6),
        "onnx": str(args.onnx),
        "engine": str(args.engine),
    })
    if not args.skip_export:
        report["export"] = export_onnx(args)
    if not args.skip_build:
        if not args.onnx.is_file():
            raise FileNotFoundError(f"ONNX not found: {args.onnx}")
        report["build"] = build_engine(args)
    metadata.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {metadata}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
