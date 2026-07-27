#!/usr/bin/env python3
"""Precompute high-precision video cuts for a later postprocess run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from contracts.detections import CutList, write_cut_list
from cut_detection.detector import HighPrecisionCutDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video = args.input_video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    result = HighPrecisionCutDetector().detect_video(
        video,
        max_frames=args.max_frames,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    write_cut_list(
        temporary,
        CutList(tuple(result.frames), result.method, result.elapsed_seconds),
    )
    os.replace(temporary, output)
    print(
        f"cuts={len(result.frames)} method={result.method} "
        f"elapsed={result.elapsed_seconds:.6f}s output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
