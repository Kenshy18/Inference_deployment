#!/usr/bin/env python3
"""Run RT-DETR Head/Face detection and write normalized SQLite results."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
DEFAULT_FRAMEWORK_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "RT-DETRv4"
DEFAULT_LOCAL_SITE_PACKAGES = FAMILY_ROOT / ".runtime" / "site-packages"
DEFAULT_SHARED_ROOT = FAMILY_ROOT / ".runtime" / "shared"
DEFAULT_CONFIG = "configs/rtv2/rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"
DEFAULT_CHECKPOINT = (
    FAMILY_ROOT / "artifacts" / "detector" / "head_face_best_stg1.pth"
)
for _source in (DEFAULT_SHARED_ROOT, PACKAGE_PARENT):
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--framework-source",
        type=Path,
        default=(
            Path(os.environ["RTDETR_FRAMEWORK_SOURCE"])
            if os.environ.get("RTDETR_FRAMEWORK_SOURCE")
            else DEFAULT_FRAMEWORK_SOURCE
        ),
    )
    parser.add_argument(
        "--extra-site-packages",
        action="append",
        default=(
            [DEFAULT_LOCAL_SITE_PACKAGES]
            if DEFAULT_LOCAL_SITE_PACKAGES.is_dir()
            else []
        ),
        type=Path,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--target-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH")
    )
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--fast-sqlite",
        action="store_true",
        help="trade crash durability for faster SQLite writes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _prepare_runtime(
    framework_source: Path, extra_site_packages: Sequence[Path]
) -> None:
    source = framework_source.expanduser().resolve()
    if not (source / "engine").is_dir() or not (source / "configs").is_dir():
        raise FileNotFoundError(
            f"RT-DETR framework source must contain engine/ and configs/: {source}"
        )
    extras = [str(path.expanduser().resolve()) for path in extra_site_packages]
    for extra in reversed(extras):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    os.environ["RTDETR_FRAMEWORK_SOURCE"] = str(source)
    if extras:
        os.environ["RTDETR_EXTRA_SITE_PACKAGES"] = os.pathsep.join(extras)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1]")
    if not 0.0 <= args.nms_threshold <= 1.0:
        parser.error("--nms-threshold must be in [0, 1]")
    if args.max_detections <= 0:
        parser.error("--max-detections must be positive")
    if not 0.0 < args.max_area_ratio <= 1.0:
        parser.error("--max-area-ratio must be in (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    _prepare_runtime(args.framework_source, args.extra_site_packages)
    from persistence import SqliteWriter
    from pipelines import run_video_inference
    from rtdetr_head_face import model as runtime
    from rtdetr_head_face.adapter import RtDetrHeadFaceAdapter, RtDetrSettings

    input_path = runtime.resolve_existing_path(args.input)
    config_path = runtime.resolve_existing_path(args.config)
    checkpoint_path = runtime.resolve_existing_path(args.checkpoint)
    class_filter = runtime.parse_class_filter(args.classes)
    adapter = RtDetrHeadFaceAdapter(
        RtDetrSettings(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=args.device,
            size=(
                None
                if args.target_size is None
                else tuple(args.target_size)
            ),
            precision=args.precision,
            compile=args.compile,
            channels_last=not args.no_channels_last,
            tf32=not args.no_tf32,
            warmup_iterations=args.warmup_iterations,
            class_filter=(
                None if class_filter is None else frozenset(class_filter)
            ),
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
            max_area_ratio=args.max_area_ratio,
        ),
        batch_size=args.batch_size,
    )
    writer = SqliteWriter(
        output_path,
        overwrite=args.overwrite,
        safe=not args.fast_sqlite,
    )

    def progress(summary) -> None:
        if (
            args.progress_interval > 0
            and summary.processed_frames % args.progress_interval == 0
        ):
            print(
                f"processed={summary.processed_frames} "
                f"rows={summary.result_items} fps={summary.wall_fps:.2f}",
                flush=True,
            )

    result = run_video_inference(
        input_path=input_path,
        adapter=adapter,
        writer=writer,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        warmup_frames=0,
        prefetch_batches=2,
        metadata={
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "classes": args.classes,
            "input_size": list(adapter.input_size),
            "precision": adapter.precision_name,
        },
        progress=progress,
    )
    print(f"saved sqlite to: {output_path}")
    print(
        f"processed {result.processed_frames} frames in "
        f"{result.wall_elapsed_sec:.2f}s ({result.wall_fps:.2f} fps)"
    )
    print(f"wrote {result.result_items} detection rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
