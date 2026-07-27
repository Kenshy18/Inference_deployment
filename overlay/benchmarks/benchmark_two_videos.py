#!/usr/bin/env python3
"""Benchmark two independent videos on CPU and NVENC concurrently."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--video-a", required=True, type=Path)
    result.add_argument("--sqlite-a", required=True, type=Path)
    result.add_argument("--video-b", type=Path)
    result.add_argument("--sqlite-b", type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--ffmpeg-bin", required=True, type=Path)
    result.add_argument("--cpu-workers", type=int, default=3)
    result.add_argument("--nvenc-workers", type=int, default=3)
    result.add_argument("--start-frame", type=int, default=0)
    result.add_argument("--end-frame", type=int, default=1797)
    result.add_argument("--h264-crf", type=int, default=18)
    result.add_argument("--h264-preset", default="veryfast")
    result.add_argument("--nvenc-cq", type=int, default=18)
    result.add_argument("--nvenc-preset", default="p5")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be >= start-frame")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    video_a = args.video_a.expanduser().resolve()
    sqlite_a = args.sqlite_a.expanduser().resolve()
    video_b = (args.video_b or args.video_a).expanduser().resolve()
    sqlite_b = (args.sqlite_b or args.sqlite_a).expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    segmented_runner = Path(__file__).with_name("benchmark_segmented.py")
    jobs = (
        (
            "cpu",
            "h264",
            args.cpu_workers,
            video_a,
            sqlite_a,
        ),
        (
            "nvenc",
            "h264_nvenc",
            args.nvenc_workers,
            video_b,
            sqlite_b,
        ),
    )
    processes: list[tuple[str, subprocess.Popen[str], object, Path, list[str]]] = []
    started = time.perf_counter()
    for name, codec, workers, video, sqlite in jobs:
        job_dir = output_dir / name
        log_path = output_dir / f"{name}.log"
        command = [
            sys.executable,
            str(segmented_runner),
            "--video",
            str(video),
            "--sqlite",
            str(sqlite),
            "--output-dir",
            str(job_dir),
            "--ffmpeg-bin",
            str(ffmpeg),
            "--codec",
            codec,
            "--workers",
            str(workers),
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
            "--h264-crf",
            str(args.h264_crf),
            "--h264-preset",
            args.h264_preset,
            "--nvenc-cq",
            str(args.nvenc_cq),
            "--nvenc-preset",
            args.nvenc_preset,
        ]
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ),
        )
        processes.append((name, process, log, job_dir, command))

    failures: list[tuple[str, int]] = []
    for name, process, log, _job_dir, _command in processes:
        return_code = process.wait()
        log.close()
        if return_code != 0:
            failures.append((name, return_code))
    wall_seconds = time.perf_counter() - started
    if failures:
        raise RuntimeError(f"concurrent job failures: {failures}")

    frames = args.end_frame - args.start_frame + 1
    records: list[dict[str, object]] = []
    for name, _process, _log, job_dir, command in processes:
        payload = json.loads(
            (job_dir / "benchmark_summary.json").read_text(encoding="utf-8")
        )
        records.append(
            {
                "job": name,
                "command": command,
                "job_total_seconds": payload["total_seconds"],
                "job_fps": payload["aggregate_fps"],
                "output": payload["final_output"],
                "validation": payload["final_validation"],
            }
        )
    summary = {
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frames_per_video": frames,
        "videos": 2,
        "total_frames": frames * 2,
        "wall_seconds": wall_seconds,
        "aggregate_fps": frames * 2 / wall_seconds,
        "jobs": records,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
