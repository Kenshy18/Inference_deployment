#!/usr/bin/env python3
"""Export/build the fixed-RoI Co-DINO instance mask head TensorRT engine."""

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
DEFAULT_ONNX = CACHE_ROOT / "exports/onnx/codino_mask_head_n1_736x1280.onnx"
DEFAULT_ENGINE = MODEL_ROOT / "codino/trt/codino_mask_head_n1_736x1280_fp32.engine"
DEFAULT_CORE_ONNX = CACHE_ROOT / "exports/onnx/codino_mask_head_core_n1_736x1280.onnx"
DEFAULT_CORE_ENGINE = (
    MODEL_ROOT / "codino/trt/codino_mask_head_core_n1_736x1280_fp32.engine"
)
DEFAULT_TRT_SITE_PACKAGES = default_tensorrt_site_packages()


def _prepare_imports() -> None:
    prepare_codino_imports(patch_mmcv_stream=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mode", choices=("full", "core"), default="full")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-rois", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--semantic-height", type=int, default=184)
    parser.add_argument("--semantic-width", type=int, default=320)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--io-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--workspace-gb", type=int, default=8)
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


class MaskHeadExportWrapper(torch.nn.Module):
    def __init__(self, mask_head: torch.nn.Module, num_rois: int) -> None:
        super().__init__()
        self.mask_head = mask_head
        self.num_rois = int(num_rois)

    def forward(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        roi_labels = torch.zeros((self.num_rois,), dtype=torch.long, device=rois.device)
        stage_instance_preds, _ = self.mask_head(
            instance_feats, semantic_feat, rois, roi_labels
        )
        return tuple(stage_instance_preds)


class MaskHeadCoreExportWrapper(torch.nn.Module):
    def __init__(self, mask_head: torch.nn.Module, num_rois: int) -> None:
        super().__init__()
        self.mask_head = mask_head
        self.num_rois = int(num_rois)

    def forward(
        self,
        instance_feats: torch.Tensor,
        sem0: torch.Tensor,
        sem1: torch.Tensor,
        sem2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_rois = (sem0, sem1, sem2)
        for conv in self.mask_head.instance_convs:
            instance_feats = conv(instance_feats)

        stage_instance_preds = []
        for idx, stage in enumerate(self.mask_head.stages):
            instance_logits = self.mask_head.stage_instance_logits[idx](instance_feats)[
                :, :1
            ]
            upsample_flag = (
                self.mask_head.pre_upsample_last_stage
                or idx < len(self.mask_head.stages) - 1
            )
            fused_feats = torch.cat(
                [instance_feats, semantic_rois[idx], instance_logits.sigmoid()],
                dim=1,
            )
            for conv in stage.fuse_conv:
                fused_feats = stage.relu(conv(fused_feats))
            fused_feats = stage.relu(stage.fuse_transform_out(fused_feats))
            fused_feats = torch.cat([fused_feats, instance_logits.sigmoid()], dim=1)
            instance_feats = (
                stage.upsample(fused_feats) if upsample_flag else fused_feats
            )
            stage_instance_preds.append(instance_logits)

        instance_preds = self.mask_head.stage_instance_logits[-1](instance_feats)[:, :1]
        if not self.mask_head.pre_upsample_last_stage:
            instance_preds = torch.nn.functional.interpolate(
                instance_preds,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        stage_instance_preds.append(instance_preds)
        return tuple(stage_instance_preds)


def export_onnx(wrapper: torch.nn.Module, args: argparse.Namespace) -> None:
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    export_dtype = torch.float16 if args.io_dtype == "fp16" else torch.float32
    if export_dtype == torch.float16:
        wrapper = wrapper.half()

    instance_feats = torch.randn(
        (args.num_rois, 256, 14, 14), device=args.device, dtype=export_dtype
    )
    if args.mode == "core":
        dummies = (
            instance_feats,
            torch.randn(
                (args.num_rois, 256, 14, 14), device=args.device, dtype=export_dtype
            ),
            torch.randn(
                (args.num_rois, 128, 28, 28), device=args.device, dtype=export_dtype
            ),
            torch.randn(
                (args.num_rois, 64, 56, 56), device=args.device, dtype=export_dtype
            ),
        )
        input_names = ["instance_feats", "sem0", "sem1", "sem2"]
    else:
        semantic_feat = torch.randn(
            (args.batch_size, 256, args.semantic_height, args.semantic_width),
            device=args.device,
            dtype=export_dtype,
        )
        rois = torch.zeros((args.num_rois, 5), device=args.device, dtype=torch.float32)
        if args.num_rois > 0:
            rois[:, 0] = (
                torch.arange(args.num_rois, device=args.device) % args.batch_size
            )
            rois[:, 1] = 64.0
            rois[:, 2] = 64.0
            rois[:, 3] = 256.0
            rois[:, 4] = 256.0
        dummies = (instance_feats, semantic_feat, rois)
        input_names = ["instance_feats", "semantic_feat", "rois"]
    with torch.inference_mode():
        samples = wrapper(*dummies)
    print(
        f"[sample] outputs={[tuple(t.shape) for t in samples]} dtypes={[str(t.dtype) for t in samples]}"
    )
    t0 = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummies,
            str(args.onnx),
            input_names=input_names,
            output_names=["stage0", "stage1", "stage2", "stage3"],
            opset_version=args.opset,
            do_constant_folding=False,
            dynamo=False,
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
    elif args.precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
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
    if args.onnx is None:
        args.onnx = DEFAULT_CORE_ONNX if args.mode == "core" else DEFAULT_ONNX
    if args.engine is None:
        args.engine = DEFAULT_CORE_ENGINE if args.mode == "core" else DEFAULT_ENGINE
    if args.num_rois < 1:
        raise ValueError("--num-rois must be >= 1")
    _prepare_imports()
    if not args.skip_export:
        model = load_detector(args)
        wrapper_cls = (
            MaskHeadCoreExportWrapper if args.mode == "core" else MaskHeadExportWrapper
        )
        wrapper = wrapper_cls(model.mask_head, args.num_rois).to(args.device).eval()
        export_onnx(wrapper, args)
    if not args.skip_build:
        build_engine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
