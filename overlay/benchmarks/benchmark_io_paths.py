#!/usr/bin/env python3
"""Compare direct FFmpeg and OpenCV/rawvideo overlay I/O paths."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
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
    result.add_argument("--start-frame", type=int, default=0)
    result.add_argument("--end-frame", type=int, default=1797)
    result.add_argument("--h264-crf", type=int, default=18)
    result.add_argument("--h264-preset", default="veryfast")
    result.add_argument("--nvenc-cq", type=int, default=18)
    result.add_argument("--nvenc-preset", default="p5")
    return result


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps


def empty_mask_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.execute("DELETE FROM masks")
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def run(command: list[str]) -> float:
    started = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ),
    )
    return time.perf_counter() - started


def main() -> None:
    args = parser().parse_args()
    if args.start_frame != 0:
        raise ValueError("direct FFmpeg comparison currently requires start-frame=0")
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be >= start-frame")
    video = args.video.expanduser().resolve()
    sqlite = args.sqlite.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    empty_sqlite = output_dir / "empty_masks.sqlite"
    empty_mask_database(sqlite, empty_sqlite)
    frames = args.end_frame - args.start_frame + 1
    fps = video_fps(video)
    records: list[dict[str, object]] = []

    for codec, encoder in (("h264", "libx264"), ("h264_nvenc", "h264_nvenc")):
        direct_output = output_dir / f"direct_ffmpeg_{codec}.mp4"
        direct_command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            encoder,
        ]
        if codec == "h264":
            direct_command.extend(
                ["-preset", args.h264_preset, "-crf", str(args.h264_crf)]
            )
        else:
            direct_command.extend(
                [
                    "-preset",
                    args.nvenc_preset,
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(args.nvenc_cq),
                    "-b:v",
                    "0",
                ]
            )
        direct_command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(direct_output),
            ]
        )
        direct_seconds = run(direct_command)
        records.append(
            {
                "path": "direct_ffmpeg_no_overlay",
                "codec": codec,
                "seconds": direct_seconds,
                "fps": frames / direct_seconds,
                "output": str(direct_output),
                "size_bytes": direct_output.stat().st_size,
                "command": direct_command,
            }
        )

        for label, database in (
            ("opencv_rawvideo_no_masks", empty_sqlite),
            ("opencv_rawvideo_with_masks", sqlite),
        ):
            output = output_dir / f"{label}_{codec}.mp4"
            manifest = output_dir / f"{label}_{codec}.json"
            command = [
                sys.executable,
                "-m",
                "overlay_renderer",
                "--mode",
                "final",
                "--video",
                str(video),
                "--sqlite",
                str(database),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--codec",
                codec,
                "--h264-crf",
                str(args.h264_crf),
                "--h264-preset",
                args.h264_preset,
                "--nvenc-cq",
                str(args.nvenc_cq),
                "--nvenc-preset",
                args.nvenc_preset,
                "--ffmpeg-bin",
                str(ffmpeg),
                "--start-frame",
                str(args.start_frame),
                "--end-frame",
                str(args.end_frame),
                "--progress-every",
                "0",
            ]
            seconds = run(command)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            records.append(
                {
                    "path": label,
                    "codec": codec,
                    "seconds": seconds,
                    "fps": frames / seconds,
                    "renderer_seconds": payload["summary"]["elapsed_seconds"],
                    "masks_drawn": payload["summary"]["masks_drawn"],
                    "output": str(output),
                    "size_bytes": output.stat().st_size,
                    "command": command,
                }
            )

    summary = {
        "video": str(video),
        "sqlite": str(sqlite),
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "records": records,
    }
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
