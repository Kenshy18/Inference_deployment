#!/usr/bin/env python3
"""Run DINOv3 ViT-S+ compact Co-DINO MH0 inference into SQLite."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
for source in (
    FAMILY_ROOT / ".runtime" / "shared",
    PACKAGE_PARENT,
    FAMILY_ROOT / ".runtime" / "src" / "codino",
    FAMILY_ROOT / ".runtime" / "src" / "dinov3_root",
    FAMILY_ROOT,
):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

try:
    from .adapter import Mh0Adapter
    from .model import (
        DEFAULT_CHECKPOINT,
        DEFAULT_CONFIG,
        DEFAULT_TRT_BUNDLE,
        build_runtime,
    )
except ImportError:
    from adapter import Mh0Adapter
    from model import (
        DEFAULT_CHECKPOINT,
        DEFAULT_CONFIG,
        DEFAULT_TRT_BUNDLE,
        build_runtime,
    )

from mask_geometry import DEFAULT_MAX_MASK_POINTS
from persistence import AsyncSqliteWriter
from pipelines import run_video_inference


class Progress:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self.last = 0.0

    def __call__(self, summary) -> None:
        now = time.monotonic()
        if self.interval == 0 or now - self.last >= self.interval:
            self.last = now
            print(
                f"[progress] frames={summary.processed_frames} "
                f"detections={summary.result_items} "
                f"compute_fps={summary.compute_fps:.3f}",
                flush=True,
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument(
        "--backend",
        choices=("tensorrt-fast", "pytorch"),
        default="tensorrt-fast",
    )
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--trt-bundle", type=Path, default=DEFAULT_TRT_BUNDLE)
    value.add_argument(
        "--trt-verify",
        choices=("metadata", "engines"),
        default="engines",
    )
    value.add_argument("--score-thresh", type=float, default=0.30)
    value.add_argument("--model-score-thr", type=float)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--warmup-frames", type=int, default=0)
    value.add_argument("--progress-interval-sec", type=float, default=5.0)
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--fast-sqlite", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input video not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    if not 0 <= args.score_thresh <= 1:
        raise ValueError("score threshold must be in [0, 1]")
    model_threshold = (
        args.score_thresh
        if args.model_score_thr is None
        else args.model_score_thr
    )
    runtime = build_runtime(
        config=args.config.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
        backend=args.backend,
        device=args.device,
        model_score_threshold=model_threshold,
        trt_bundle=args.trt_bundle.expanduser().resolve(),
        trt_verify=args.trt_verify,
    )
    batch_size = (
        runtime.fixed_batch_size
        if args.batch_size is None
        else args.batch_size
    )
    if batch_size != runtime.fixed_batch_size:
        raise ValueError(
            f"{args.backend} requires batch size {runtime.fixed_batch_size}, "
            f"got {batch_size}"
        )
    adapter = Mh0Adapter(runtime, score_threshold=args.score_thresh)
    writer = AsyncSqliteWriter(
        output_path,
        overwrite=args.overwrite,
        safe=not args.fast_sqlite,
    )
    summary = run_video_inference(
        input_path=input_path,
        adapter=adapter,
        writer=writer,
        batch_size=batch_size,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        prefetch_batches=2,
        metadata={
            "backend": args.backend,
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "trt_bundle": (
                str(args.trt_bundle.expanduser().resolve())
                if args.backend == "tensorrt-fast"
                else None
            ),
            "score_threshold": args.score_thresh,
            "model_score_threshold": model_threshold,
            "max_mask_points": DEFAULT_MAX_MASK_POINTS,
        },
        progress=Progress(args.progress_interval_sec),
    )
    print(
        f"processed {summary.processed_frames} frames; "
        f"compute={summary.compute_fps:.3f} img/s; "
        f"wall={summary.wall_fps:.3f} fps",
        flush=True,
    )
    print(f"saved SQLite: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
