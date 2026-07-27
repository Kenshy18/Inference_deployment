#!/usr/bin/env python3
"""Run full-TensorRT Face DINO v2 inference into normalized SQLite."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
for source in (FAMILY_ROOT / ".runtime" / "shared", PACKAGE_PARENT):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from face_dino_v2.adapter import FaceDinoV2Adapter, FaceDinoV2Settings
from face_dino_v2.model import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TRT_BUNDLE,
)
from persistence import AsyncSqliteWriter
from pipelines import run_video_inference


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument(
        "--backend",
        choices=("tensorrt-fast",),
        default="tensorrt-fast",
    )
    value.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--trt-bundle", type=Path, default=DEFAULT_TRT_BUNDLE)
    value.add_argument(
        "--trt-verify",
        choices=("metadata", "engines"),
        default="engines",
    )
    value.add_argument("--device", default="cuda:0")
    value.add_argument(
        "--batch-size",
        type=int,
        choices=(8, 16),
        help="must match the selected TensorRT bundle; defaults to bundle batch",
    )
    value.add_argument("--score-threshold", type=float, default=0.30)
    value.add_argument("--classes", nargs="*")
    value.add_argument("--max-frames", type=int)
    value.add_argument("--warmup-iterations", type=int, default=3)
    value.add_argument("--progress-interval", type=int, default=1000)
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
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    classes = FaceDinoV2Adapter.parse_classes(args.classes)
    settings = FaceDinoV2Settings(
        source_root=args.source_root.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
        trt_bundle=args.trt_bundle.expanduser().resolve(),
        device=args.device,
        score_threshold=args.score_threshold,
        warmup_iterations=args.warmup_iterations,
        classes=classes,
        verify=args.trt_verify,
    )
    adapter = FaceDinoV2Adapter(settings)
    batch_size = adapter.runtime.fixed_batch_size
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError(
            f"--batch-size={args.batch_size} does not match B{batch_size} bundle"
        )
    writer = AsyncSqliteWriter(
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
                f"rows={summary.result_items} "
                f"compute_fps={summary.compute_fps:.2f}",
                flush=True,
            )

    summary = run_video_inference(
        input_path=input_path,
        adapter=adapter,
        writer=writer,
        batch_size=batch_size,
        max_frames=args.max_frames,
        warmup_frames=0,
        prefetch_batches=1,
        metadata={
            "backend": args.backend,
            "checkpoint": str(settings.checkpoint),
            "trt_bundle": str(settings.trt_bundle),
            "source_root": str(settings.source_root),
            "score_threshold": args.score_threshold,
            "classes": (
                sorted(classes) if classes is not None else sorted((1, 2))
            ),
            "input_shape": [batch_size, 3, 736, 1280],
            "rich_runtime_outputs": [
                "head_boxes",
                "face_presence",
                "ellipses",
                "keypoints",
                "keypoint_states",
                "point_probabilities",
            ],
        },
        progress=progress,
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
