#!/usr/bin/env python3
"""Sweep H.264 encoders at a shared target capacity and score quality."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--reference", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--ffmpeg-bin", required=True, type=Path)
    result.add_argument("--target-bitrate-mbps", required=True, type=float)
    result.add_argument("--metric-threads", type=int, default=8)
    return result


def inspect_video(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return frames, fps


def run(command: list[str]) -> float:
    started = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - started


def configurations(target: str, buffer_size: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "name": "x264_crf18_veryfast",
            "encoder": "libx264",
            "args": ["-preset", "veryfast", "-crf", "18"],
        },
        {
            "name": "nvenc_cq18_p5",
            "encoder": "h264_nvenc",
            "args": [
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
            ],
        },
    ]
    for preset in ("ultrafast", "superfast", "veryfast", "faster"):
        records.append(
            {
                "name": f"x264_{preset}_cbr",
                "encoder": "libx264",
                "args": [
                    "-preset",
                    preset,
                    "-b:v",
                    target,
                    "-minrate",
                    target,
                    "-maxrate",
                    target,
                    "-bufsize",
                    buffer_size,
                ],
            }
        )
    nvenc_variants = (
        ("p1_single", "p1", "disabled"),
        ("p1_qres", "p1", "qres"),
        ("p3_qres", "p3", "qres"),
        ("p5_qres", "p5", "qres"),
        ("p7_fullres", "p7", "fullres"),
    )
    for name, preset, multipass in nvenc_variants:
        records.append(
            {
                "name": f"nvenc_{name}_cbr",
                "encoder": "h264_nvenc",
                "args": [
                    "-preset",
                    preset,
                    "-tune",
                    "hq",
                    "-rc",
                    "cbr",
                    "-b:v",
                    target,
                    "-minrate",
                    target,
                    "-maxrate",
                    target,
                    "-bufsize",
                    buffer_size,
                    "-cbr_padding",
                    "1",
                    "-multipass",
                    multipass,
                    "-spatial-aq",
                    "1",
                    "-temporal-aq",
                    "1",
                    "-aq-strength",
                    "8",
                ],
            }
        )
    return records


def main() -> None:
    args = parser().parse_args()
    if args.target_bitrate_mbps <= 0:
        raise ValueError("target bitrate must be positive")
    reference = args.reference.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames, fps = inspect_video(reference)
    duration = frames / fps
    target_bps = round(args.target_bitrate_mbps * 1_000_000)
    target = str(target_bps)
    buffer_size = str(target_bps * 2)
    records: list[dict[str, object]] = []

    for configuration in configurations(target, buffer_size):
        name = str(configuration["name"])
        output = output_dir / f"{name}.mp4"
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(reference),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            str(configuration["encoder"]),
            *[str(value) for value in configuration["args"]],
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
        encode_seconds = run(command)
        metrics_path = output_dir / f"{name}_metrics.json"
        metric_filter = (
            "[0:v][1:v]libvmaf="
            "feature='name=psnr|name=float_ssim':"
            f"log_fmt=json:log_path={metrics_path}:"
            f"n_threads={args.metric_threads}"
        )
        metric_command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-i",
            str(reference),
            "-lavfi",
            metric_filter,
            "-f",
            "null",
            "-",
        ]
        metric_seconds = run(metric_command)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pooled = metrics["pooled_metrics"]
        size_bytes = output.stat().st_size
        records.append(
            {
                "name": name,
                "encoder": configuration["encoder"],
                "encode_seconds": encode_seconds,
                "encode_fps": frames / encode_seconds,
                "metric_seconds": metric_seconds,
                "size_bytes": size_bytes,
                "bitrate_mbps": size_bytes * 8 / duration / 1_000_000,
                "capacity_ratio": size_bytes * 8 / duration / target_bps,
                "vmaf": pooled["vmaf"]["mean"],
                "ssim": pooled["float_ssim"]["mean"],
                "psnr_y": pooled["psnr_y"]["mean"],
                "output": str(output),
                "encode_command": command,
                "metric_command": metric_command,
            }
        )
        print(
            f"{name}: fps={frames / encode_seconds:.2f} "
            f"Mbps={size_bytes * 8 / duration / 1_000_000:.2f} "
            f"VMAF={pooled['vmaf']['mean']:.3f}",
            flush=True,
        )

    summary = {
        "reference": str(reference),
        "frames": frames,
        "fps": fps,
        "duration_seconds": duration,
        "target_bitrate_mbps": args.target_bitrate_mbps,
        "records": records,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
