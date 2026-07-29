#!/usr/bin/env python3
"""Run concurrent segments with the production native overlay renderer."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


@dataclass(frozen=True)
class VideoFrame:
    """Presentation-order frame information from the container packet index."""

    timestamp: int
    keyframe: bool


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
    result.add_argument(
        "--display-style",
        choices=("legacy", "detailed", "simple"),
        default="legacy",
    )
    result.add_argument(
        "--mask-domain",
        choices=("all", "genital", "face_privacy"),
        default="all",
    )
    result.add_argument("--no-face-probability-masks", action="store_true")
    result.add_argument("--no-face-keypoints", action="store_true")
    result.add_argument("--no-face-ellipses", action="store_true")
    result.add_argument(
        "--face-mask-target",
        choices=("none", "face", "eyes"),
        default="none",
    )
    result.add_argument(
        "--eye-mask-shape",
        choices=("ellipse", "rectangle"),
        default="ellipse",
    )
    result.add_argument("--minimum-eye-confidence", type=float, default=0.35)
    result.add_argument("--no-labels", action="store_true")
    result.add_argument("--copy-audio", action="store_true")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--renderer",
        type=Path,
        default=root / "build" / "overlay_native",
    )
    result.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=(root.parent / ".runtime" / "ffmpeg-nvenc-btbn-8.1" / "bin" / "ffmpeg"),
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


def probe_video_frames(ffmpeg: Path, video: Path) -> list[VideoFrame]:
    """Build a presentation-order packet index without decoding the video."""
    ffprobe = ffmpeg.with_name("ffprobe")
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts,flags",
        "-of",
        "compact=p=0:nk=1",
        str(video),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    packets: list[tuple[int, int, bool]] = []
    for packet_order, line in enumerate(completed.stdout.splitlines()):
        fields = line.split("|")
        if not fields or fields[0] in {"", "N/A"}:
            raise RuntimeError(
                "source video packet has no PTS; fast frame-accurate seek "
                "requires timestamped video packets"
            )
        try:
            timestamp = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(
                f"invalid source video packet PTS: {fields[0]!r}"
            ) from exc
        flags = fields[1] if len(fields) > 1 else ""
        packets.append((timestamp, packet_order, "K" in flags))
    if not packets:
        raise RuntimeError("source has no indexed video packets")

    # Demux order follows DTS and is not presentation order when B-frames are
    # present.  Sorting PTS gives the same ordinal used by decoded frames and
    # by inference SQLite frame_index.
    packets.sort(key=lambda item: (item[0], item[1]))
    frames = [
        VideoFrame(timestamp=timestamp, keyframe=keyframe)
        for timestamp, _, keyframe in packets
    ]
    if any(
        second.timestamp <= first.timestamp for first, second in zip(frames, frames[1:])
    ):
        raise RuntimeError(
            "source video PTS is not strictly increasing in presentation "
            "order; fast frame-accurate seek is unavailable"
        )
    return frames


def probe_reported_frames(ffmpeg: Path, video: Path) -> int | None:
    """Read the container-reported frame count when it is available."""
    ffprobe = ffmpeg.with_name("ffprobe")
    value = subprocess.check_output(
        [
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
        ],
        text=True,
    ).strip()
    if value in {"", "N/A"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid container-reported video frame count: {value!r}"
        ) from exc


def seek_anchor(
    frames: list[VideoFrame],
    frame_index: int,
) -> tuple[int, VideoFrame]:
    """Return the nearest keyframe at or before a requested frame."""
    keyframes = [index for index, frame in enumerate(frames) if frame.keyframe]
    position = bisect.bisect_right(keyframes, frame_index) - 1
    if position < 0:
        raise RuntimeError(
            f"no seekable keyframe exists at or before frame {frame_index}"
        )
    anchor_index = keyframes[position]
    return anchor_index, frames[anchor_index]


def is_keyframe_primary(path: Path) -> bool:
    """Detect V3 without importing the optional Python renderer stack."""

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='result_schema_info'
            """
        ).fetchone()
        if table is None:
            return False
        profile = connection.execute(
            """
            SELECT value FROM result_schema_info
            WHERE key='compatibility_profile'
            """
        ).fetchone()
        return profile is not None and str(profile[0]) == "keyframe-primary-v3"


def materialize_keyframe_shards(
    source: Path,
    output_directory: Path,
    *,
    mode: str,
    ranges: list[tuple[int, int]],
    workers: int,
    mask_domain: str | None,
) -> dict[str, object]:
    # Loading overlay_renderer as a package imports OpenCV, which adds roughly
    # three seconds to a short fast run even though cache generation does not
    # use it.  Load the dependency-free module directly instead.
    started = time.perf_counter()
    module_name = "_overlay_keyframe_cache"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "overlay_renderer"
        / "keyframe_cache.py"
    )
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load keyframe cache module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    module_seconds = time.perf_counter() - started
    summary = module.materialize_overlay_cache_shards(
        source,
        output_directory,
        mode=mode,
        frame_ranges=ranges,
        workers=workers,
        mask_domain=mask_domain,
    )
    summary["module_load_seconds"] = module_seconds
    summary["materialization_seconds"] = float(summary["seconds"])
    summary["seconds"] = time.perf_counter() - started
    return summary


def materialize_fast_face_cache(
    source: Path,
    output: Path,
    *,
    display_style: str,
    start_frame: int,
    end_frame: int,
    include_probability_masks: bool,
    include_keypoints: bool,
    include_ellipses: bool,
    draw_keypoints: bool,
    draw_ellipses: bool,
    face_privacy_target: str,
    eye_mask_shape: str,
    minimum_eye_confidence: float,
) -> dict[str, object]:
    started = time.perf_counter()
    from overlay_renderer.fast_face_cache import (
        materialize_fast_face_cache as materialize,
    )

    summary = materialize(
        source,
        output,
        display_style=display_style,
        start_frame=start_frame,
        end_frame=end_frame,
        include_probability_masks=include_probability_masks,
        include_keypoints=include_keypoints,
        include_ellipses=include_ellipses,
        draw_keypoints=draw_keypoints,
        draw_ellipses=draw_ellipses,
        face_privacy_target=face_privacy_target,
        eye_mask_shape=eye_mask_shape,
        minimum_eye_confidence=minimum_eye_confidence,
    )
    summary["seconds"] = time.perf_counter() - started
    return summary


def main() -> None:
    args = parser().parse_args()
    if args.include_faces and args.face_sqlite is None:
        raise ValueError("--include-faces requires --face-sqlite")
    if args.face_sqlite is not None and not args.include_faces:
        raise ValueError("--face-sqlite requires --include-faces")
    renderer = args.renderer.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    video = args.video.expanduser().resolve()
    sqlite = args.sqlite.expanduser().resolve()
    face_sqlite = (
        None if args.face_sqlite is None else args.face_sqlite.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    index_started = time.perf_counter()
    video_frames = probe_video_frames(ffmpeg, video)
    reported_frames = probe_reported_frames(ffmpeg, video)
    index_seconds = time.perf_counter() - index_started
    if reported_frames is not None and reported_frames != len(video_frames):
        raise RuntimeError(
            "video packet/frame count mismatch: "
            f"packets={len(video_frames)}, reported_frames={reported_frames}"
        )
    if args.end_frame is None:
        args.end_frame = len(video_frames) - 1
    if args.start_frame >= len(video_frames):
        raise ValueError(
            f"start-frame {args.start_frame} exceeds the indexed source "
            f"range 0..{len(video_frames) - 1}"
        )
    if args.end_frame >= len(video_frames):
        raise ValueError(
            f"end-frame {args.end_frame} exceeds the indexed source "
            f"range 0..{len(video_frames) - 1}"
        )
    frame_total = args.end_frame - args.start_frame + 1
    if args.workers > frame_total:
        args.workers = frame_total
        args.cpu_workers = min(args.cpu_workers, args.workers)
    output_dir.mkdir(parents=True, exist_ok=False)

    encoders = worker_encoders(args.workers, args.cpu_workers)
    weights = [
        args.cpu_weight if encoder == "libx264" else args.nvenc_weight
        for encoder in encoders
    ]
    ranges = weighted_ranges(args.start_frame, args.end_frame, weights)
    cache_summary: dict[str, object] | None = None
    face_cache_summary: dict[str, object] | None = None
    worker_sqlites = [sqlite] * len(ranges)
    if args.mode in {"tracked", "final"} and is_keyframe_primary(sqlite):
        cache_summary = materialize_keyframe_shards(
            sqlite,
            output_dir / "keyframe_cache",
            mode=args.mode,
            ranges=ranges,
            workers=args.workers,
            mask_domain=(None if args.mask_domain == "all" else args.mask_domain),
        )
        worker_sqlites = [
            Path(str(shard["cache_sqlite"]))
            for shard in cache_summary["shards"]  # type: ignore[index]
        ]
    if args.display_style in {"detailed", "simple"} and (
        args.mode == "faces" or args.include_faces
    ):
        face_source = sqlite if args.mode == "faces" else face_sqlite
        assert face_source is not None
        face_cache = output_dir / "fast_face_cache.sqlite"
        face_cache_summary = materialize_fast_face_cache(
            face_source,
            face_cache,
            display_style=args.display_style,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            include_probability_masks=not args.no_face_probability_masks,
            include_keypoints=(
                not args.no_face_keypoints or args.face_mask_target == "eyes"
            ),
            include_ellipses=(
                not args.no_face_ellipses or args.face_mask_target != "none"
            ),
            draw_keypoints=not args.no_face_keypoints,
            draw_ellipses=not args.no_face_ellipses,
            face_privacy_target=args.face_mask_target,
            eye_mask_shape=args.eye_mask_shape,
            minimum_eye_confidence=args.minimum_eye_confidence,
        )
        if args.mode == "faces":
            worker_sqlites = [face_cache] * len(ranges)
        else:
            face_sqlite = face_cache
    processes: list[tuple[subprocess.Popen[str], object, dict[str, object]]] = []

    parallel_started = time.perf_counter()
    for index, ((start, end), encoder, worker_sqlite) in enumerate(
        zip(ranges, encoders, worker_sqlites, strict=True)
    ):
        anchor_index, anchor = seek_anchor(video_frames, start)
        output = output_dir / f"segment_{index:02d}.mp4"
        summary_path = output_dir / f"segment_{index:02d}.json"
        error_path = output_dir / f"segment_{index:02d}.stderr.log"
        preset = args.cpu_preset if encoder == "libx264" else args.nvenc_preset
        command = [
            str(renderer),
            "--video",
            str(video),
            "--sqlite",
            str(worker_sqlite),
            "--mode",
            args.mode,
            "--display-style",
            args.display_style,
            "--mask-domain",
            args.mask_domain,
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
            "--seek-frame-index",
            str(anchor_index),
            "--seek-timestamp",
            str(anchor.timestamp),
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
            command.extend(["--include-faces", "--face-sqlite", str(face_sqlite)])
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
            "seek_frame_index": anchor_index,
            "seek_timestamp": anchor.timestamp,
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
            f"file '{Path(str(record['output'])).as_posix()}'\n" for record in records
        ),
        encoding="utf-8",
    )
    final_output = output_dir / "final.mp4"
    concat_output = output_dir / "video_only.mp4" if args.copy_audio else final_output
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
    cache_seconds = 0.0 if cache_summary is None else float(cache_summary["seconds"])
    face_cache_seconds = (
        0.0 if face_cache_summary is None else float(face_cache_summary["seconds"])
    )
    timestamp_deltas = [
        second.timestamp - first.timestamp
        for first, second in zip(video_frames, video_frames[1:])
    ]
    delta_counts = Counter(timestamp_deltas)
    nominal_timestamp_delta = (
        delta_counts.most_common(1)[0][0] if delta_counts else None
    )
    summary = {
        "implementation": (
            "cpp-libav-hybrid-cpu-cuda-segmented"
            if (
                args.gpu_pipeline
                and args.cpu_workers
                and args.cpu_workers < args.workers
            )
            else (
                "cpp-libav-nvdec-cuda-nvenc-segmented"
                if args.gpu_pipeline and args.cpu_workers < args.workers
                else "cpp-libav-yuv420p-segmented"
            )
        ),
        "workers": args.workers,
        "cpu_workers": args.cpu_workers,
        "nvenc_workers": args.workers - args.cpu_workers,
        "nvenc_gpu": args.nvenc_gpu,
        "encoders": encoders,
        "mode": args.mode,
        "display_style": args.display_style,
        "include_faces": args.include_faces,
        "face_sqlite": (None if face_sqlite is None else str(face_sqlite)),
        "show_labels": not args.no_labels,
        "copy_audio": args.copy_audio,
        "cpu_weight": args.cpu_weight,
        "nvenc_weight": args.nvenc_weight,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frames": frames,
        "source_frame_index": {
            "method": "ffprobe-packet-pts",
            "indexed_frames": len(video_frames),
            "container_reported_frames": reported_frames,
            "scan_seconds": index_seconds,
            "nominal_timestamp_delta": nominal_timestamp_delta,
            "non_uniform_timestamp_deltas": (
                sum(
                    count
                    for delta, count in delta_counts.items()
                    if delta != nominal_timestamp_delta
                )
                if nominal_timestamp_delta is not None
                else 0
            ),
        },
        "parallel_seconds": parallel_seconds,
        "concat_seconds": concat_seconds,
        "audio_mux_seconds": audio_mux_seconds,
        "total_seconds": (
            total_seconds + index_seconds + cache_seconds + face_cache_seconds
        ),
        "render_seconds": total_seconds,
        "renderer_total_seconds": total_seconds + index_seconds,
        "aggregate_fps": frames
        / (total_seconds + index_seconds + cache_seconds + face_cache_seconds),
        "bitrate_mbps": args.bitrate_mbps,
        "decoder_threads": args.decoder_threads,
        "hw_decode": args.hw_decode or args.gpu_pipeline,
        "gpu_pipeline": args.gpu_pipeline,
        "gpu_pipeline_nvenc_only": (args.gpu_pipeline and args.cpu_workers == 0),
        "faststart": args.faststart,
        "final_output": str(final_output),
        "size_bytes": final_output.stat().st_size,
        "workers_detail": records,
        "concat_command": concat_command,
        "audio_mux_command": audio_mux_command,
    }
    if cache_summary is not None:
        summary["keyframe_materialization"] = cache_summary
    if face_cache_summary is not None:
        summary["face_primitive_materialization"] = face_cache_summary
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
                    "total_seconds": summary["total_seconds"],
                    "render_seconds": summary["render_seconds"],
                    "frame_index_seconds": index_seconds,
                    "final_output": str(final_output),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
