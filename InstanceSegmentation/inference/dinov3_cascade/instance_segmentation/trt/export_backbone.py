#!/usr/bin/env python3
"""Export the inference DINOv3 ViT-L backbone to ONNX."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("DINOV3_USE_XFORMERS", "0")

import torch


class BackboneExportWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, feature: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature = feature

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.backbone(image)
        if isinstance(output, dict):
            return output[self.feature]
        if isinstance(output, (list, tuple)):
            return output[-1]
        return output


def target_size(value: str) -> tuple[int, int]:
    text = value.lower()
    if "x" in text:
        height, width = text.split("x", 1)
        return int(height), int(width)
    size = int(text)
    return size, size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-size", type=target_size, default=(720, 1280))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights = args.backbone_weights.expanduser().resolve()
    if not weights.is_file():
        build_parser().error(f"backbone weights not found: {weights}")
    if args.batch_size <= 0:
        build_parser().error("--batch-size must be positive")
    height, width = args.target_size

    from detectron2.modeling import DINOv3Backbone

    backbone = (
        DINOv3Backbone(
            img_size=max(height, width),
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            layers_to_use=1,
            out_feature="last_feat",
            weights=str(weights),
            pretrained=True,
        )
        .to(args.device)
        .eval()
    )
    wrapper = BackboneExportWrapper(backbone, "last_feat").to(args.device).eval()
    dummy = torch.randn(
        args.batch_size,
        3,
        height,
        width,
        dtype=torch.float32,
        device=args.device,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        None if args.fixed_batch else {"input": {0: "batch"}, "last_feat": {0: "batch"}}
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["last_feat"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[PASS] ONNX backbone: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
