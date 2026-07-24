#!/usr/bin/env python3
"""Run standalone DINOv3 Cascade TensorRT-backbone inference on one video."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
DEFAULT_ARTIFACT_ROOT = FAMILY_ROOT / "artifacts"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_ROOT / "detector" / "model_final.pth"
DEFAULT_BACKBONE_WEIGHTS = (
    DEFAULT_ARTIFACT_ROOT
    / "backbone"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_TRT_BACKBONE_ENGINE = (
    DEFAULT_ARTIFACT_ROOT
    / "trt"
    / "dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_CLASSIFIER_CHECKPOINT = DEFAULT_ARTIFACT_ROOT / "classifier" / "best.pt"
DEFAULT_DETECTRON2_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "detectron2_root"
DEFAULT_DINOV3_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "dinov3_root"
DEFAULT_SHARED_ROOT = FAMILY_ROOT / ".runtime" / "shared"
for _source in (DEFAULT_SHARED_ROOT, PACKAGE_PARENT):
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from dinov3_cascade.adapter import Dinov3CascadeAdapter
from dinov3_cascade.assembly import build_runtime
from dinov3_cascade.instance_segmentation.contracts import (
    InstanceSegmentationSettings,
)
from dinov3_cascade.runtime_contracts import VideoInferenceSettings
from mask_geometry import DEFAULT_MAX_MASK_POINTS
from persistence import SqliteWriter
from pipelines import run_video_inference


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def validate_torch_checkpoint(path: Path) -> None:
    """Reject a truncated modern torch archive before allocating the model."""

    with path.open("rb") as stream:
        signature = stream.read(4)
    if not signature.startswith(b"PK"):
        return
    try:
        with zipfile.ZipFile(path):
            pass
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"truncated or corrupt checkpoint archive: {path}") from exc


def target_size(value: str) -> int | tuple[int, int]:
    text = value.lower()
    if "x" not in text:
        return int(text)
    height, width = text.split("x", 1)
    return int(height), int(width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=existing_file)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=existing_file, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backbone-weights", type=existing_file, default=DEFAULT_BACKBONE_WEIGHTS
    )
    parser.add_argument(
        "--trt-backbone-engine",
        type=existing_file,
        default=DEFAULT_TRT_BACKBONE_ENGINE,
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=existing_file,
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    parser.add_argument("--config", type=existing_file)
    parser.add_argument(
        "--detectron2-source",
        type=Path,
        default=(
            Path(os.environ["DINOV3_DETECTRON2_SOURCE"])
            if os.environ.get("DINOV3_DETECTRON2_SOURCE")
            else DEFAULT_DETECTRON2_SOURCE
        ),
    )
    parser.add_argument(
        "--dinov3-source",
        type=Path,
        default=(
            Path(os.environ["DINOV3_FRAMEWORK_SOURCE"])
            if os.environ.get("DINOV3_FRAMEWORK_SOURCE")
            else DEFAULT_DINOV3_SOURCE
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-size", type=target_size, default=(720, 1280))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--rpn-pre-nms-topk", type=int, default=100)
    parser.add_argument("--rpn-post-nms-topk", type=int, default=40)
    parser.add_argument("--rpn-nms-threshold", type=float, default=0.9)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--fast-sqlite",
        action="store_true",
        help="trade crash durability for faster SQLite writes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames < 0:
        build_parser().error("--max-frames must be non-negative")
    for source in (args.detectron2_source, args.dinov3_source):
        resolved = source.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"framework source not found: {resolved}. "
                "Run setup_environment.py first."
            )
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    for name, path in (
        ("checkpoint", args.checkpoint),
        ("backbone weights", args.backbone_weights),
        ("TensorRT backbone engine", args.trt_backbone_engine),
        ("classifier checkpoint", args.classifier_checkpoint),
    ):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} not found: {path}. "
                "Place artifacts in this model folder or override the CLI path."
            )
        if path.suffix != ".engine":
            validate_torch_checkpoint(path)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    runtime, _classifier_payload = build_runtime(
        segmenter_settings=InstanceSegmentationSettings(
            checkpoint=args.checkpoint,
            backbone_weights=str(args.backbone_weights),
            trt_backbone_engine=args.trt_backbone_engine,
            config_path=args.config,
            target_size=args.target_size,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            topk_per_image=args.topk,
            rpn_pre_nms_topk_test=args.rpn_pre_nms_topk,
            rpn_post_nms_topk_test=args.rpn_post_nms_topk,
            rpn_nms_threshold=args.rpn_nms_threshold,
        ),
        classifier_checkpoint=args.classifier_checkpoint,
        device=args.device,
    )
    settings = VideoInferenceSettings(
        device=args.device,
        amp=args.amp != "off",
        amp_dtype="fp16" if args.amp == "off" else args.amp,
        score_threshold=args.score_threshold,
    )
    sink = SqliteWriter(
        output_path,
        overwrite=args.overwrite,
        safe=not args.fast_sqlite,
    )
    result = run_video_inference(
        input_path=args.input,
        adapter=Dinov3CascadeAdapter(runtime, settings),
        writer=sink,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        prefetch_batches=args.prefetch_batches,
        metadata={
            "checkpoint": str(args.checkpoint),
            "backbone_weights": str(args.backbone_weights),
            "trt_backbone_engine": str(args.trt_backbone_engine),
            "classifier_checkpoint": (
                None
                if args.classifier_checkpoint is None
                else str(args.classifier_checkpoint)
            ),
            "max_mask_points": DEFAULT_MAX_MASK_POINTS,
        },
    )
    print(f"saved sqlite to: {output_path}", flush=True)
    print(
        f"processed {result.processed_frames} frames in "
        f"{result.wall_elapsed_sec:.3f}s ({result.wall_fps:.3f} fps)",
        flush=True,
    )
    print(f"wrote {result.result_items} segmentation rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
