#!/usr/bin/env python3
"""Export the checkpoint-bound, pruned EVA-02 ViT backbone to dynamic ONNX."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

FAMILY_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PARENT = FAMILY_ROOT.parent
DEFAULT_FRAMEWORK_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "detectron2_root"
for _path in (PACKAGE_PARENT, DEFAULT_FRAMEWORK_SOURCE):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eva02_cascade.instance_segmentation.contracts import (
    NATIVE_CONFIG_PATH,
    InstanceSegmentationSettings,
)
from eva02_cascade.instance_segmentation.model import build_segmenter
from eva02_cascade.trt.bundle import file_record


class BackboneExportWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)["last_feat"]


def block_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block indices must be comma-separated integers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=NATIVE_CONFIG_PATH)
    parser.add_argument(
        "--framework-source",
        type=Path,
        default=DEFAULT_FRAMEWORK_SOURCE,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--drop-block-indices",
        type=block_indices,
        default=(19, 21, 22),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    framework = args.framework_source.expanduser().resolve()
    if not (framework / "detectron2").is_dir():
        build_parser().error(
            f"framework source must contain detectron2/: {framework}"
        )
    if str(framework) not in sys.path:
        sys.path.insert(0, str(framework))
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    for path, label in ((checkpoint, "checkpoint"), (config, "config")):
        if not path.is_file():
            build_parser().error(f"{label} not found: {path}")
    if args.target_size <= 0:
        build_parser().error("--target-size must be positive")

    started = time.perf_counter()
    settings = InstanceSegmentationSettings(
        config_path=config,
        checkpoint=checkpoint,
        target_size=args.target_size,
        compile_backbone="none",
        compile_heads="none",
        model_half=False,
        backbone_half=True,
        drop_block_indices=args.drop_block_indices,
    )
    segmenter = build_segmenter(settings, device=args.device)
    backbone = segmenter.backbone.net.eval()
    wrapper = BackboneExportWrapper(backbone).to(args.device).eval()
    parameter = next(backbone.parameters())
    dummy = torch.randn(
        1,
        3,
        args.target_size,
        args.target_size,
        dtype=parameter.dtype,
        device=args.device,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["images"],
            output_names=["last_feat"],
            dynamic_axes={"images": {0: "batch"}, "last_feat": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        sample = wrapper(dummy)
    report = {
        "schema": "eva02-backbone-onnx-export-v1",
        "status": "pass",
        "checkpoint": file_record(checkpoint),
        "config": file_record(config),
        "onnx": file_record(output),
        "target_size": args.target_size,
        "opset": args.opset,
        "drop_block_indices": list(args.drop_block_indices),
        "input_dtype": str(dummy.dtype),
        "sample_output_shape": list(sample.shape),
        "sample_output_dtype": str(sample.dtype),
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
