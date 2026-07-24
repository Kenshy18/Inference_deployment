#!/usr/bin/env python3
"""Export the fine-tuned Co-DINO DINOv3 ViT-L backbone to ONNX."""

from __future__ import annotations

import argparse
import json
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
        build_backbone_engine,
        prepare_codino_imports,
    )
except ImportError:
    from build_engines import (
        CACHE_ROOT,
        MODEL_ROOT,
        build_backbone_engine,
        prepare_codino_imports,
    )

DEFAULT_RUN_DIR = MODEL_ROOT / "codino" / "detector"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "resolved_config.py"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "epoch_2.pth"
DEFAULT_OUTPUT = (
    CACHE_ROOT / "exports/onnx/codino_dinov3_vitl_backbone_736x1280_fp32.onnx"
)


def _prepare_imports() -> None:
    prepare_codino_imports(patch_mmcv_stream=False)


class CoDINOBackboneExportWrapper(torch.nn.Module):
    def __init__(
        self, backbone: torch.nn.Module, neck: torch.nn.Module | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        if self.neck is not None:
            return tuple(self.neck(out))
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", type=Path)
    parser.add_argument(
        "--height", type=int, default=736, help="Padded model input height."
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Padded model input width."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument(
        "--with-neck",
        action="store_true",
        help="Export backbone + SFP neck multi-output features.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-gb", type=int, default=12)
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


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    _prepare_imports()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_detector(args)
    wrapper = (
        CoDINOBackboneExportWrapper(
            model.backbone, model.neck if args.with_neck else None
        )
        .to(args.device)
        .eval()
    )
    dummy = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=args.device,
        dtype=torch.float32,
    )

    output_names = ["last_feat"]
    if args.with_neck:
        with torch.inference_mode():
            sample_out = wrapper(dummy)
        output_names = [f"feat{i}" for i in range(len(sample_out))]

    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {"input": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    print(f"[export] config={args.config}")
    print(f"[export] checkpoint={args.checkpoint}")
    print(f"[export] input_shape={tuple(dummy.shape)} dtype={dummy.dtype}")
    print(f"[export] output={output}")

    t0 = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    with torch.inference_mode():
        out = wrapper(dummy)
    output_shapes = (
        [list(t.shape) for t in out]
        if isinstance(out, (tuple, list))
        else [list(out.shape)]
    )
    meta = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "input_shape": list(dummy.shape),
        "output_names": output_names,
        "output_shapes": output_shapes,
        "dtype": str(dummy.dtype),
        "fixed_batch": bool(args.fixed_batch),
        "with_neck": bool(args.with_neck),
        "opset": int(args.opset),
        "export_elapsed_sec": elapsed,
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {output}")
    print(f"[done] wrote {meta_path}")
    if args.engine is not None:
        build_backbone_engine(
            onnx_path=output,
            engine_path=args.engine.expanduser().resolve(),
            workspace_gb=args.workspace_gb,
        )
        print(f"[done] wrote {args.engine.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
