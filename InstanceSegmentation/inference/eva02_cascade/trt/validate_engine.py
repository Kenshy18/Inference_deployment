#!/usr/bin/env python3
"""Validate a generated EVA-02 TensorRT backbone against its checkpoint."""

from __future__ import annotations

import argparse
import json
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
from eva02_cascade.trt.export_backbone import block_indices
from eva02_cascade.trt.runtime import Eva02TensorRTBackbone


def batches(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batches must be comma-separated integers"
        ) from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("validation batches must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=NATIVE_CONFIG_PATH)
    parser.add_argument(
        "--framework-source",
        type=Path,
        default=DEFAULT_FRAMEWORK_SOURCE,
    )
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=1280)
    parser.add_argument("--batches", type=batches, default=(1, 12, 20))
    parser.add_argument(
        "--drop-block-indices",
        type=block_indices,
        default=(19, 21, 22),
    )
    parser.add_argument("--max-mean-abs", type=float, default=0.08)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
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
    engine = args.engine.expanduser().resolve()
    for path, label in (
        (checkpoint, "checkpoint"),
        (config, "config"),
        (engine, "engine"),
    ):
        if not path.is_file():
            build_parser().error(f"{label} not found: {path}")

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
    pytorch_backbone = segmenter.backbone.net.eval()
    trt_backbone = Eva02TensorRTBackbone(engine).to(args.device).eval()
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)

    measurements: list[dict[str, object]] = []
    passed = True
    with torch.inference_mode():
        for batch_size in args.batches:
            value = torch.randn(
                batch_size,
                3,
                args.target_size,
                args.target_size,
                device=args.device,
                dtype=torch.float16,
            )
            reference = pytorch_backbone(value)["last_feat"]
            candidate = trt_backbone(value)["last_feat"]
            torch.cuda.synchronize()
            difference = (reference.float() - candidate.float()).abs()
            cosine = torch.nn.functional.cosine_similarity(
                reference.float().flatten(),
                candidate.float().flatten(),
                dim=0,
            )
            item = {
                "batch_size": batch_size,
                "shape": list(candidate.shape),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "rmse": float(
                    torch.sqrt(torch.mean(difference.square())).item()
                ),
                "cosine": float(cosine.item()),
            }
            item["pass"] = bool(
                item["mean_abs"] <= args.max_mean_abs
                and item["cosine"] >= args.min_cosine
            )
            passed = passed and bool(item["pass"])
            measurements.append(item)
            del value, reference, candidate, difference, cosine

    report = {
        "schema": "eva02-backbone-trt-validation-v1",
        "status": "pass" if passed else "fail",
        "thresholds": {
            "max_mean_abs": args.max_mean_abs,
            "min_cosine": args.min_cosine,
        },
        "measurements": measurements,
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError("EVA-02 TensorRT validation thresholds were not met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
