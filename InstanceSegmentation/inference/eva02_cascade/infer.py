#!/usr/bin/env python3
"""Run standalone EVA02 Cascade + ROI-classifier inference on one video."""

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
DEFAULT_CLASSIFIER_CHECKPOINT = DEFAULT_ARTIFACT_ROOT / "classifier" / "best.pt"
DEFAULT_FRAMEWORK_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "detectron2_root"
DEFAULT_TRT_BUNDLE = (
    DEFAULT_ARTIFACT_ROOT
    / "trt"
    / "eva02-vit-dynamic-b1-20-fp16-v1"
    / "manifest.json"
)
DEFAULT_SHARED_ROOT = FAMILY_ROOT / ".runtime" / "shared"
for _source in (DEFAULT_SHARED_ROOT, PACKAGE_PARENT):
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))


def _bootstrap_framework_source() -> Path | None:
    raw = os.environ.get("EVA02_FRAMEWORK_SOURCE")
    if "--framework-source" in sys.argv:
        index = sys.argv.index("--framework-source")
        if index + 1 < len(sys.argv):
            raw = sys.argv[index + 1]
    if not raw and DEFAULT_FRAMEWORK_SOURCE.is_dir():
        raw = str(DEFAULT_FRAMEWORK_SOURCE)
    if not raw:
        return None
    source = Path(raw).expanduser().resolve()
    if not (source / "detectron2").is_dir():
        raise FileNotFoundError(
            f"EVA02 framework source must contain detectron2/: {source}"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


BOOTSTRAPPED_FRAMEWORK_SOURCE = _bootstrap_framework_source()

from eva02_cascade.assembly import build_runtime
from eva02_cascade.adapter import Eva02CascadeAdapter
from eva02_cascade.instance_segmentation.contracts import (
    NATIVE_CONFIG_PATH,
    InstanceSegmentationSettings,
)
from eva02_cascade.runtime_contracts import VideoInferenceSettings
from eva02_cascade.trt.bundle import load_trt_bundle
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=existing_file)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=existing_file, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=existing_file,
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    parser.add_argument("--config", type=existing_file, default=NATIVE_CONFIG_PATH)
    parser.add_argument(
        "--framework-source",
        type=Path,
        default=BOOTSTRAPPED_FRAMEWORK_SOURCE,
        help="EVA02-compatible source root containing detectron2/.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("tensorrt-backbone", "pytorch"),
        default="tensorrt-backbone",
    )
    parser.add_argument("--trt-bundle", type=Path, default=DEFAULT_TRT_BUNDLE)
    parser.add_argument(
        "--trt-verify",
        choices=("metadata", "engine", "full"),
        default="full",
    )
    parser.add_argument("--target-size", type=int, default=1280)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="default: TensorRT profile max (20), or 1 for PyTorch",
    )
    parser.add_argument("--classifier-batch-size", type=int, default=1024)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--compile-backbone",
        choices=(
            "none",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
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
    for name, path in (
        ("checkpoint", args.checkpoint),
        ("classifier checkpoint", args.classifier_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} not found: {path}. "
                "Place artifacts in this model folder or override the CLI path."
            )
        validate_torch_checkpoint(path)
    if args.framework_source is None:
        raise FileNotFoundError(
            f"framework source not found: {DEFAULT_FRAMEWORK_SOURCE}. "
            "Run setup_environment.py first."
        )
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    trt_bundle = None
    if args.backend == "tensorrt-backbone":
        trt_bundle = load_trt_bundle(
            args.trt_bundle,
            verify=args.trt_verify,
            checkpoint_path=args.checkpoint,
            classifier_checkpoint=args.classifier_checkpoint,
            config_path=args.config,
        )
        if args.target_size != trt_bundle.target_size:
            build_parser().error(
                f"--target-size must be {trt_bundle.target_size} for this bundle"
            )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (trt_bundle.max_batch if trt_bundle is not None else 1)
    )
    if batch_size <= 0:
        build_parser().error("--batch-size must be positive")
    if trt_bundle is not None and not (
        trt_bundle.min_batch <= batch_size <= trt_bundle.max_batch
    ):
        build_parser().error(
            f"--batch-size must be in [{trt_bundle.min_batch}, "
            f"{trt_bundle.max_batch}] for this bundle"
        )

    segmenter_settings = InstanceSegmentationSettings(
        config_path=args.config,
        checkpoint=args.checkpoint,
        target_size=args.target_size,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        topk_per_image=args.topk,
        compile_backbone=args.compile_backbone,
    )
    runtime, _classifier_payload = build_runtime(
        segmenter_settings=segmenter_settings,
        classifier_checkpoint=args.classifier_checkpoint,
        device=args.device,
        trt_bundle=trt_bundle,
    )
    settings = VideoInferenceSettings(
        device=args.device,
        amp=args.amp != "off",
        amp_dtype="fp16" if args.amp == "off" else args.amp,
        score_threshold=args.score_threshold,
        classifier_batch_size=args.classifier_batch_size,
    )
    sink = SqliteWriter(
        output_path,
        overwrite=args.overwrite,
        safe=not args.fast_sqlite,
    )
    result = run_video_inference(
        input_path=args.input,
        adapter=Eva02CascadeAdapter(runtime, settings),
        writer=sink,
        batch_size=batch_size,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        prefetch_batches=args.prefetch_batches,
        metadata={
            "checkpoint": str(args.checkpoint),
            "classifier_checkpoint": str(args.classifier_checkpoint),
            "config": str(args.config),
            "backend": args.backend,
            "trt_bundle": (
                None if trt_bundle is None else str(trt_bundle.manifest_path)
            ),
            "trt_profile": (
                None if trt_bundle is None else trt_bundle.profile
            ),
            "batch_size": batch_size,
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
