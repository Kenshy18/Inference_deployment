#!/usr/bin/env python3
"""Benchmark one overlay as concurrently encoded H.264 segments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--video", required=True, type=Path)
    result.add_argument("--sqlite", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--ffmpeg-bin", required=True, type=Path)
    result.add_argument(
        "--codec",
        choices=("h264", "h264_nvenc", "hybrid"),
        required=True,
    )
    result.add_argument("--workers", type=int, required=True)
    result.add_argument(
        "--hybrid-cpu-workers",
        type=int,
        help=(
            "number of libx264 workers in hybrid mode; the remaining workers "
            "use NVENC (default: half, rounded down)"
        ),
    )
    result.add_argument("--start-frame", type=int, default=0)
    result.add_argument("--end-frame", type=int, default=1797)
    result.add_argument("--h264-crf", type=int, default=18)
    result.add_argument("--h264-preset", default="veryfast")
    result.add_argument("--nvenc-cq", type=int, default=18)
    result.add_argument("--nvenc-preset", default="p5")
    result.add_argument("--target-bitrate-mbps", type=float)
    result.add_argument("--cpu-weight", type=float, default=1.0)
    result.add_argument("--nvenc-weight", type=float, default=1.0)
    result.add_argument("--no-labels", action="store_true")
    return result


def frame_ranges(start: int, end: int, workers: int) -> list[tuple[int, int]]:
    total = end - start + 1
    base, remainder = divmod(total, workers)
    ranges: list[tuple[int, int]] = []
    current = start
    for index in range(workers):
        length = base + (1 if index < remainder else 0)
        ranges.append((current, current + length - 1))
        current += length
    return ranges


def weighted_frame_ranges(
    start: int,
    end: int,
    weights: list[float],
) -> list[tuple[int, int]]:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("worker weights must be positive")
    total = end - start + 1
    if len(weights) > total:
        raise ValueError("workers cannot exceed the number of frames")
    remaining = total - len(weights)
    weight_total = sum(weights)
    exact = [remaining * weight / weight_total for weight in weights]
    extras = [int(value) for value in exact]
    undistributed = remaining - sum(extras)
    order = sorted(
        range(len(weights)),
        key=lambda index: exact[index] - extras[index],
        reverse=True,
    )
    for index in order[:undistributed]:
        extras[index] += 1
    lengths = [1 + extra for extra in extras]
    ranges: list[tuple[int, int]] = []
    current = start
    for length in lengths:
        ranges.append((current, current + length - 1))
        current += length
    return ranges


def worker_codecs(
    codec: str,
    workers: int,
    hybrid_cpu_workers: int | None,
) -> list[str]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if codec != "hybrid":
        if hybrid_cpu_workers is not None:
            raise ValueError("--hybrid-cpu-workers requires --codec hybrid")
        return [codec] * workers
    if workers < 2:
        raise ValueError("hybrid mode requires at least 2 workers")
    cpu_workers = (
        workers // 2 if hybrid_cpu_workers is None else hybrid_cpu_workers
    )
    if not 1 <= cpu_workers < workers:
        raise ValueError(
            "hybrid CPU workers must be between 1 and workers - 1"
        )
    nvenc_workers = workers - cpu_workers
    # Interleave encoder types so content-complexity changes across the source
    # are less likely to bias one encoder's entire half of the benchmark.
    codecs: list[str] = []
    while cpu_workers or nvenc_workers:
        if cpu_workers:
            codecs.append("h264")
            cpu_workers -= 1
        if nvenc_workers:
            codecs.append("h264_nvenc")
            nvenc_workers -= 1
    return codecs


def inspect_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
    result = {
        "opened": capture.isOpened(),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "codec_fourcc": codec,
        "size_bytes": path.stat().st_size,
    }
    capture.release()
    return result


def main() -> None:
    args = parser().parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be >= start-frame")
    frame_count = args.end_frame - args.start_frame + 1
    if args.workers > frame_count:
        raise ValueError("workers cannot exceed the number of frames")
    codecs = worker_codecs(
        args.codec,
        args.workers,
        args.hybrid_cpu_workers,
    )
    if args.cpu_weight <= 0 or args.nvenc_weight <= 0:
        raise ValueError("worker weights must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    video = args.video.expanduser().resolve()
    sqlite = args.sqlite.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    weights = [
        args.cpu_weight if codec == "h264" else args.nvenc_weight
        for codec in codecs
    ]
    ranges = weighted_frame_ranges(args.start_frame, args.end_frame, weights)
    processes: list[tuple[subprocess.Popen[str], object]] = []
    worker_records: list[dict[str, object]] = []

    parallel_started = time.perf_counter()
    for index, (start, end) in enumerate(ranges):
        worker_codec = codecs[index]
        output = output_dir / f"segment_{index:02d}.mp4"
        manifest = output_dir / f"segment_{index:02d}.json"
        log_path = output_dir / f"segment_{index:02d}.log"
        command = [
            sys.executable,
            "-m",
            "overlay_renderer",
            "--mode",
            "final",
            "--video",
            str(video),
            "--sqlite",
            str(sqlite),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--codec",
            worker_codec,
            "--h264-crf",
            str(args.h264_crf),
            "--h264-preset",
            args.h264_preset,
            "--nvenc-cq",
            str(args.nvenc_cq),
            "--nvenc-preset",
            args.nvenc_preset,
            "--start-frame",
            str(start),
            "--end-frame",
            str(end),
            "--progress-every",
            "0",
        ]
        if args.target_bitrate_mbps is not None:
            command.extend(
                [
                    "--target-bitrate-mbps",
                    str(args.target_bitrate_mbps),
                ]
            )
        if args.no_labels:
            command.append("--no-labels")
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ),
        )
        processes.append((process, log))
        worker_records.append(
            {
                "worker": index,
                "codec": worker_codec,
                "start_frame": start,
                "end_frame": end,
                "output": str(output),
                "manifest": str(manifest),
                "log": str(log_path),
                "command": command,
            }
        )

    failures: list[tuple[int, int]] = []
    for index, (process, log) in enumerate(processes):
        return_code = process.wait()
        log.close()
        worker_records[index]["return_code"] = return_code
        if return_code != 0:
            failures.append((index, return_code))
    parallel_seconds = time.perf_counter() - parallel_started
    if failures:
        raise RuntimeError(f"parallel overlay worker failures: {failures}")

    concat_file = output_dir / "concat.txt"
    concat_file.write_text(
        "".join(
            f"file '{Path(record['output']).as_posix()}'\n"
            for record in worker_records
        ),
        encoding="utf-8",
    )
    final_output = output_dir / f"final_{args.codec}_{args.workers}way.mp4"
    concat_command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(final_output),
    ]
    concat_started = time.perf_counter()
    subprocess.run(concat_command, check=True)
    concat_seconds = time.perf_counter() - concat_started
    for record in worker_records:
        payload = json.loads(Path(record["manifest"]).read_text(encoding="utf-8"))
        record["renderer_seconds"] = payload["summary"]["elapsed_seconds"]
        record["frames_written"] = payload["summary"]["frames_written"]

    summary = {
        "codec": args.codec,
        "workers": args.workers,
        "worker_codecs": codecs,
        "cpu_workers": codecs.count("h264"),
        "nvenc_workers": codecs.count("h264_nvenc"),
        "target_bitrate_mbps": args.target_bitrate_mbps,
        "cpu_weight": args.cpu_weight,
        "nvenc_weight": args.nvenc_weight,
        "worker_weights": weights,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frames": args.end_frame - args.start_frame + 1,
        "parallel_seconds": parallel_seconds,
        "concat_seconds": concat_seconds,
        "total_seconds": parallel_seconds + concat_seconds,
        "aggregate_fps": (
            (args.end_frame - args.start_frame + 1)
            / (parallel_seconds + concat_seconds)
        ),
        "workers_detail": worker_records,
        "final_output": str(final_output),
        "final_validation": inspect_video(final_output),
        "concat_command": concat_command,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
