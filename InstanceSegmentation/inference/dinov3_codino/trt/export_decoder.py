#!/usr/bin/env python3
"""Export/build the fixed-shape Co-DINO transformer decoder TensorRT engine."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DINOV3_USE_XFORMERS", "0")

import torch

TRT_ROOT = Path(__file__).resolve().parent
if str(TRT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRT_ROOT))

try:
    from .build_engines import (
        CACHE_ROOT,
        MODEL_ROOT,
        default_tensorrt_site_packages,
        prepare_codino_imports,
        prepare_tensorrt,
    )
except ImportError:
    from build_engines import (
        CACHE_ROOT,
        MODEL_ROOT,
        default_tensorrt_site_packages,
        prepare_codino_imports,
        prepare_tensorrt,
    )

DEFAULT_RUN_DIR = MODEL_ROOT / "codino" / "detector"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "resolved_config.py"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "epoch_2.pth"
DEFAULT_ONNX = (
    CACHE_ROOT / "exports/onnx/codino_decoder_b2_736x1280_msda_trt_plugin.onnx"
)
DEFAULT_ENGINE = (
    MODEL_ROOT / "codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp32.engine"
)
DEFAULT_TRT_SITE_PACKAGES = default_tensorrt_site_packages()


def _prepare_imports() -> None:
    prepare_codino_imports(patch_mmcv_stream=True)


def parse_hw_shapes(text: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        h_text, w_text = item.split("x", 1)
        shapes.append((int(h_text), int(w_text)))
    if not shapes:
        raise ValueError("No feature shapes were provided.")
    return tuple(shapes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-height", type=int, default=736)
    parser.add_argument("--input-width", type=int, default=1280)
    parser.add_argument("--img-height", type=int, default=720)
    parser.add_argument("--img-width", type=int, default=1280)
    parser.add_argument("--feature-shapes", default="184x320,92x160,46x80,23x40,12x20")
    parser.add_argument("--num-query", type=int, default=100)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--io-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--workspace-gb", type=int, default=12)
    parser.add_argument(
        "--trt-extra-site-packages", type=Path, default=DEFAULT_TRT_SITE_PACKAGES
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def load_detector(args: argparse.Namespace):
    from mmcv import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(str(args.config))
    if "backbone" in cfg.model:
        cfg.model.backbone.pretrained = False
        cfg.model.backbone.weights = None
    if "fp16" in cfg:
        cfg.pop("fp16")
    cfg.load_from = None
    cfg.resume_from = None
    model = init_detector(cfg, str(args.checkpoint), device=args.device)
    model.eval()
    return model


def install_msda_trt_symbolic() -> None:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

    def symbolic(
        g,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        y4 = g.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        shape = g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.long))
        return g.op("Reshape", y4, shape)

    MultiScaleDeformableAttnFunction.symbolic = staticmethod(symbolic)


class DecoderExportWrapper(torch.nn.Module):
    def __init__(
        self,
        decoder: torch.nn.Module,
        reg_branches: torch.nn.ModuleList,
        mask_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.reg_branches = reg_branches
        self.register_buffer("mask_flatten", mask_flatten)
        self.register_buffer("spatial_shapes", spatial_shapes)
        self.register_buffer("level_start_index", level_start_index)
        self.register_buffer("valid_ratios", valid_ratios)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=self.mask_flatten,
            reference_points=reference_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            valid_ratios=self.valid_ratios,
            reg_branches=self.reg_branches,
        )
        return inter_states, inter_references


def build_wrapper(
    model: torch.nn.Module,
    args: argparse.Namespace,
    feature_shapes: tuple[tuple[int, int], ...],
):
    query_head = model.query_head
    transformer = query_head.transformer
    batch_size = int(args.batch_size)
    with torch.inference_mode():
        img_masks = torch.ones(
            (batch_size, args.input_height, args.input_width), device=args.device
        )
        img_masks[:, : args.img_height, : args.img_width] = 0
        mlvl_masks = []
        for height, width in feature_shapes:
            mask = (
                torch.nn.functional.interpolate(img_masks[None], size=(height, width))
                .to(torch.bool)
                .squeeze(0)
            )
            mlvl_masks.append(mask)
        spatial_shapes = torch.as_tensor(
            feature_shapes, dtype=torch.long, device=args.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [transformer.get_valid_ratio(m) for m in mlvl_masks], 1
        )
        mask_flatten = torch.cat([m.flatten(1) for m in mlvl_masks], 1)
    return (
        DecoderExportWrapper(
            transformer.decoder,
            query_head.reg_branches,
            mask_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )
        .to(args.device)
        .eval()
    )


def export_onnx(
    wrapper: torch.nn.Module,
    args: argparse.Namespace,
    feature_shapes: tuple[tuple[int, int], ...],
) -> None:
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    export_dtype = torch.float16 if args.io_dtype == "fp16" else torch.float32
    if export_dtype == torch.float16:
        wrapper = wrapper.half()
    total_tokens = sum(height * width for height, width in feature_shapes)
    dummies = (
        torch.randn(
            (args.num_query, args.batch_size, 256),
            device=args.device,
            dtype=export_dtype,
        ),
        torch.randn(
            (total_tokens, args.batch_size, 256), device=args.device, dtype=export_dtype
        ),
        torch.rand(
            (args.batch_size, args.num_query, 4), device=args.device, dtype=export_dtype
        ),
    )
    with torch.inference_mode():
        samples = wrapper(*dummies)
    print(
        f"[sample] outputs={[tuple(t.shape) for t in samples]} dtypes={[str(t.dtype) for t in samples]}"
    )
    t0 = time.perf_counter()
    with torch.inference_mode():
        try:
            torch.onnx.export(
                wrapper,
                dummies,
                str(args.onnx),
                input_names=["query", "memory", "reference_points"],
                output_names=["inter_states", "inter_references"],
                opset_version=args.opset,
                do_constant_folding=False,
                custom_opsets={"trt": 1},
                dynamo=False,
            )
        except Exception as exc:
            if (
                exc.__class__.__name__ != "CheckerError"
                or not args.onnx.is_file()
                or args.onnx.stat().st_size <= 0
            ):
                raise
            print(
                f"[warn] torch ONNX checker failed after writing {args.onnx}; continuing for TensorRT: {exc}"
            )
    torch.cuda.synchronize()
    print(
        f"[export] wrote {args.onnx} size={args.onnx.stat().st_size} elapsed={time.perf_counter() - t0:.2f}s"
    )


def build_engine(args: argparse.Namespace) -> None:
    prepare_tensorrt(args.trt_extra_site_packages)
    import tensorrt as trt

    trt.init_libnvinfer_plugins(None, "")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb) << 30
    )
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.write_bytes(bytes(serialized))
    print(
        f"[build] wrote {args.engine} size={args.engine.stat().st_size} elapsed={time.perf_counter() - t0:.2f}s"
    )


def main() -> int:
    args = parse_args()
    _prepare_imports()
    install_msda_trt_symbolic()
    feature_shapes = parse_hw_shapes(args.feature_shapes)
    if not args.skip_export:
        model = load_detector(args)
        wrapper = build_wrapper(model, args, feature_shapes)
        export_onnx(wrapper, args, feature_shapes)
    if not args.skip_build:
        build_engine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
