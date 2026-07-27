#!/usr/bin/env python3
"""Benchmark concurrent segments rendered by the low-level C++ prototype."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    result = argparse.ArgumentParser()
    result.add_argument("--video", required=True, type=Path)
    result.add_argument("--sqlite", required=True, type=Path)
    result.add_argument(
        "--mode",
        choices=("raw", "tracked", "final", "faces"),
        default="final",
    )
    result.add_argument("--include-faces", action="store_true")
    result.add_argument("--face-sqlite", type=Path)
    result.add_argument("--no-labels", action="store_true")
    result.add_argument("--copy-audio", action="store_true")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--renderer",
        type=Path,
        default=root / "build" / "overlay_lowlevel",
    )
    result.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=root.parent / ".runtime" / "ffmpeg-nvenc" / "bin" / "ffmpeg",
    )
    result.add_argument("--workers", required=True, type=int)
    result.add_argument("--cpu-workers", required=True, type=int)
    result.add_argument("--start-frame", type=int, default=0)
    result.add_argument("--end-frame", type=int)
    result.add_argument("--bitrate-mbps", required=True, type=float)
    result.add_argument("--cpu-preset", default="veryfast")
    result.add_argument("--nvenc-preset", default="p1")
    result.add_argument("--nvenc-gpu", type=int, default=0)
    result.add_argument("--cpu-weight", type=float, default=1.0)
    result.add_argument("--nvenc-weight", type=float, default=1.0)
    result.add_argument("--mask-alpha", type=float, default=0.32)
    result.add_argument("--outline-thickness", type=int, default=2)
    result.add_argument("--box-thickness", type=int, default=2)
    result.add_argument("--decoder-threads", type=int, default=0)
    result.add_argument("--hw-decode", action="store_true")
    result.add_argument("--gpu-pipeline", action="store_true")
    result.add_argument("--faststart", action="store_true")
    result.add_argument("--compact-output", action="store_true")
    return result


def worker_encoders(workers: int, cpu_workers: int) -> list[str]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0 <= cpu_workers <= workers:
        raise ValueError("cpu-workers must be between 0 and workers")
    cpu_remaining = cpu_workers
    nvenc_remaining = workers - cpu_workers
    encoders: list[str] = []
    while cpu_remaining or nvenc_remaining:
        if cpu_remaining:
            encoders.append("libx264")
            cpu_remaining -= 1
        if nvenc_remaining:
            encoders.append("h264_nvenc")
            nvenc_remaining -= 1
    return encoders


def weighted_ranges(
    start: int,
    end: int,
    weights: list[float],
) -> list[tuple[int, int]]:
    if end < start:
        raise ValueError("end-frame must be >= start-frame")
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must be positive")
    total = end - start + 1
    if len(weights) > total:
        raise ValueError("workers exceed frame count")
    remaining = total - len(weights)
    total_weight = sum(weights)
    exact = [remaining * weight / total_weight for weight in weights]
    extras = [int(value) for value in exact]
    missing = remaining - sum(extras)
    order = sorted(
        range(len(weights)),
        key=lambda index: exact[index] - extras[index],
        reverse=True,
    )
    for index in order[:missing]:
        extras[index] += 1
    ranges: list[tuple[int, int]] = []
    current = start
    for extra in extras:
        length = 1 + extra
        ranges.append((current, current + length - 1))
        current += length
    return ranges


def probe_total_frames(ffmpeg: Path, video: Path) -> int:
    """Read the indexed video-frame count without decoding the source."""
    ffprobe = ffmpeg.with_name("ffprobe")
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    value = subprocess.check_output(command, text=True).strip()
    try:
        frames = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "source does not expose an indexed frame count; pass --end-frame"
        ) from exc
    if frames < 1:
        raise RuntimeError(f"invalid source frame count: {frames}")
    return frames


def main() -> None:
    args = parser().parse_args()
    if args.include_faces and args.mode != "final":
        raise ValueError("--include-faces is only valid with --mode final")
    if args.include_faces and args.face_sqlite is None:
        raise ValueError("--include-faces requires --face-sqlite")
    if args.face_sqlite is not None and not args.include_faces:
        raise ValueError(
            "--face-sqlite requires --mode final --include-faces"
        )
    renderer = args.renderer.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    video = args.video.expanduser().resolve()
    sqlite = args.sqlite.expanduser().resolve()
    face_sqlite = (
        None
        if args.face_sqlite is None
        else args.face_sqlite.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    if args.end_frame is None:
        args.end_frame = probe_total_frames(ffmpeg, video) - 1
    output_dir.mkdir(parents=True, exist_ok=False)

    encoders = worker_encoders(args.workers, args.cpu_workers)
    weights = [
        args.cpu_weight if encoder == "libx264" else args.nvenc_weight
        for encoder in encoders
    ]
    ranges = weighted_ranges(args.start_frame, args.end_frame, weights)
    processes: list[
        tuple[subprocess.Popen[str], object, dict[str, object]]
    ] = []

    parallel_started = time.perf_counter()
    for index, ((start, end), encoder) in enumerate(zip(ranges, encoders)):
        output = output_dir / f"segment_{index:02d}.mp4"
        summary_path = output_dir / f"segment_{index:02d}.json"
        error_path = output_dir / f"segment_{index:02d}.stderr.log"
        preset = (
            args.cpu_preset if encoder == "libx264" else args.nvenc_preset
        )
        command = [
            str(renderer),
            "--video",
            str(video),
            "--sqlite",
            str(sqlite),
            "--mode",
            args.mode,
            "--output",
            str(output),
            "--encoder",
            encoder,
            "--preset",
            preset,
            "--bitrate-mbps",
            str(args.bitrate_mbps),
            "--mask-alpha",
            str(args.mask_alpha),
            "--outline-thickness",
            str(args.outline_thickness),
            "--box-thickness",
            str(args.box_thickness),
            "--decoder-threads",
            str(args.decoder_threads),
            "--start-frame",
            str(start),
            "--end-frame",
            str(end),
        ]
        if args.hw_decode:
            command.append("--hw-decode")
        if args.gpu_pipeline and encoder == "h264_nvenc":
            command.append("--gpu-pipeline")
        if encoder == "h264_nvenc":
            command.extend(["--nvenc-gpu", str(args.nvenc_gpu)])
        if args.no_labels:
            command.append("--no-labels")
        if args.include_faces:
            command.extend(
                ["--include-faces", "--face-sqlite", str(face_sqlite)]
            )
        summary_file = summary_path.open("w", encoding="utf-8")
        error_file = error_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=summary_file,
            stderr=error_file,
            text=True,
        )
        record: dict[str, object] = {
            "worker": index,
            "encoder": encoder,
            "preset": preset,
            "start_frame": start,
            "end_frame": end,
            "output": str(output),
            "summary": str(summary_path),
            "stderr": str(error_path),
            "command": command,
        }
        processes.append((process, (summary_file, error_file), record))

    failures: list[tuple[int, int]] = []
    records: list[dict[str, object]] = []
    for index, (process, files, record) in enumerate(processes):
        return_code = process.wait()
        for file in files:
            file.close()
        record["return_code"] = return_code
        if return_code != 0:
            failures.append((index, return_code))
        else:
            record["renderer_summary"] = json.loads(
                Path(str(record["summary"])).read_text(encoding="utf-8")
            )
        records.append(record)
    parallel_seconds = time.perf_counter() - parallel_started
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    concat_path = output_dir / "concat.txt"
    concat_path.write_text(
        "".join(
            f"file '{Path(str(record['output'])).as_posix()}'\n"
            for record in records
        ),
        encoding="utf-8",
    )
    final_output = output_dir / "final.mp4"
    concat_output = (
        output_dir / "video_only.mp4"
        if args.copy_audio
        else final_output
    )
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
        str(concat_path),
        "-c",
        "copy",
    ]
    if args.faststart:
        concat_command.extend(["-movflags", "+faststart"])
    concat_command.extend(["-y", str(concat_output)])
    concat_started = time.perf_counter()
    subprocess.run(concat_command, check=True)
    concat_seconds = time.perf_counter() - concat_started
    frames = args.end_frame - args.start_frame + 1
    audio_mux_seconds = 0.0
    audio_mux_command: list[str] | None = None
    if args.copy_audio:
        source_fps = float(
            records[0]["renderer_summary"]["source_fps"]  # type: ignore[index]
        )
        start_seconds = args.start_frame / source_fps
        duration_seconds = frames / source_fps
        audio_mux_command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(concat_output),
            "-ss",
            f"{start_seconds:.12f}",
            "-t",
            f"{duration_seconds:.12f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c",
            "copy",
        ]
        if args.faststart:
            audio_mux_command.extend(["-movflags", "+faststart"])
        audio_mux_command.extend(["-y", str(final_output)])
        audio_mux_started = time.perf_counter()
        subprocess.run(audio_mux_command, check=True)
        audio_mux_seconds = time.perf_counter() - audio_mux_started
        concat_output.unlink()
    total_seconds = parallel_seconds + concat_seconds + audio_mux_seconds
    summary = {
        "implementation": (
            "cpp-libav-hybrid-cpu-cuda-segmented"
            if args.gpu_pipeline and args.cpu_workers
            else (
                "cpp-libav-nvdec-cuda-nvenc-segmented"
                if args.gpu_pipeline
                else "cpp-libav-yuv420p-segmented"
            )
        ),
        "workers": args.workers,
        "cpu_workers": args.cpu_workers,
        "nvenc_workers": args.workers - args.cpu_workers,
        "nvenc_gpu": args.nvenc_gpu,
        "encoders": encoders,
        "mode": args.mode,
        "include_faces": args.include_faces,
        "face_sqlite": (
            None if face_sqlite is None else str(face_sqlite)
        ),
        "show_labels": not args.no_labels,
        "copy_audio": args.copy_audio,
        "cpu_weight": args.cpu_weight,
        "nvenc_weight": args.nvenc_weight,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frames": frames,
        "parallel_seconds": parallel_seconds,
        "concat_seconds": concat_seconds,
        "audio_mux_seconds": audio_mux_seconds,
        "total_seconds": total_seconds,
        "aggregate_fps": frames / total_seconds,
        "bitrate_mbps": args.bitrate_mbps,
        "decoder_threads": args.decoder_threads,
        "hw_decode": args.hw_decode or args.gpu_pipeline,
        "gpu_pipeline": args.gpu_pipeline,
        "gpu_pipeline_nvenc_only": (
            args.gpu_pipeline and args.cpu_workers == 0
        ),
        "faststart": args.faststart,
        "final_output": str(final_output),
        "size_bytes": final_output.stat().st_size,
        "workers_detail": records,
        "concat_command": concat_command,
        "audio_mux_command": audio_mux_command,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.compact_output:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "frames": frames,
                    "aggregate_fps": summary["aggregate_fps"],
                    "total_seconds": total_seconds,
                    "final_output": str(final_output),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
