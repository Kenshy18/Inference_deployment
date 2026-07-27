#!/usr/bin/env python3
"""Validate timing, segmentation, decode integrity, and optional video quality."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    result = argparse.ArgumentParser()
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--summary", type=Path)
    result.add_argument("--reference", type=Path)
    result.add_argument("--report", required=True, type=Path)
    result.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=(
            root.parent
            / ".runtime"
            / "ffmpeg-nvenc-btbn-8.1"
            / "bin"
            / "ffmpeg"
        ),
    )
    result.add_argument("--expected-frames", type=int)
    result.add_argument("--expected-fps", type=float)
    result.add_argument("--minimum-vmaf-mean", type=float)
    result.add_argument("--minimum-boundary-vmaf", type=float)
    result.add_argument(
        "--maximum-boundary-vmaf-regression",
        type=float,
    )
    result.add_argument("--boundary-window", type=int, default=2)
    return result


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def fraction(value: str) -> float:
    return float(Fraction(value))


def probe_streams(ffprobe: Path, output: Path) -> list[dict[str, Any]]:
    value = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_packets",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,width,height,pix_fmt,"
                "color_range,color_space,color_transfer,color_primaries,"
                "avg_frame_rate,r_frame_rate,time_base,start_time,duration,"
                "bit_rate,nb_frames,nb_read_packets,sample_rate"
            ),
            "-of",
            "json",
            str(output),
        ]
    )
    return list(value.get("streams", []))


def probe_timestamps(
    ffprobe: Path,
    output: Path,
    selector: str,
) -> list[float]:
    value = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(output),
        ]
    )
    return [
        float(frame["best_effort_timestamp_time"])
        for frame in value.get("frames", [])
        if frame.get("best_effort_timestamp_time") not in (None, "N/A")
    ]


def probe_audio_packets(
    ffprobe: Path,
    output: Path,
) -> list[dict[str, Any]]:
    value = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,duration_time:packet_side_data",
            "-of",
            "json",
            str(output),
        ]
    )
    result: list[dict[str, Any]] = []
    for packet in value.get("packets", []):
        if packet.get("pts_time") in (None, "N/A"):
            continue
        skip_samples = 0
        for side_data in packet.get("side_data_list", []):
            if side_data.get("side_data_type") == "Skip Samples":
                skip_samples = int(side_data.get("skip_samples", 0))
        result.append(
            {
                "pts": float(packet["pts_time"]),
                "duration": float(packet.get("duration_time", 0.0)),
                "skip_samples": skip_samples,
            }
        )
    return result


def probe_video_packets(
    ffprobe: Path,
    output: Path,
) -> list[dict[str, Any]]:
    value = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,duration_time,flags",
            "-of",
            "json",
            str(output),
        ]
    )
    return [
        {
            "pts": float(packet["pts_time"]),
            "dts": float(packet["dts_time"]),
            "duration": float(packet.get("duration_time", 0.0)),
            "keyframe": "K" in str(packet.get("flags", "")),
        }
        for packet in value.get("packets", [])
        if (
            packet.get("pts_time") not in (None, "N/A")
            and packet.get("dts_time") not in (None, "N/A")
        )
    ]


def probe_audio_frames(
    ffprobe: Path,
    output: Path,
) -> list[dict[str, float | int]]:
    value = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,nb_samples",
            "-of",
            "json",
            str(output),
        ]
    )
    return [
        {
            "pts": float(frame["best_effort_timestamp_time"]),
            "samples": int(frame["nb_samples"]),
        }
        for frame in value.get("frames", [])
        if frame.get("best_effort_timestamp_time") not in (None, "N/A")
    ]


def decode_errors(ffmpeg: Path, output: Path) -> list[str]:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in completed.stderr.splitlines()
        if line.strip()
    ]


def segment_boundaries(
    summary: dict[str, Any] | None,
) -> tuple[list[int], list[dict[str, Any]], list[str]]:
    if summary is None:
        return [], [], []
    records = list(summary.get("workers_detail", []))
    errors: list[str] = []
    boundaries: list[int] = []
    expected_start = int(summary["start_frame"])
    cumulative = 0
    for index, record in enumerate(records):
        start = int(record["start_frame"])
        end = int(record["end_frame"])
        if start != expected_start:
            errors.append(
                f"worker {index} starts at {start}, expected {expected_start}"
            )
        renderer = record.get("renderer_summary", {})
        rendered = int(renderer.get("frames_written", -1))
        expected = end - start + 1
        if rendered != expected:
            errors.append(
                f"worker {index} rendered {rendered}, expected {expected}"
            )
        cumulative += expected
        if index + 1 < len(records):
            boundaries.append(cumulative)
        expected_start = end + 1
    expected_end = int(summary["end_frame"])
    if records and int(records[-1]["end_frame"]) != expected_end:
        errors.append(
            f"last worker ends at {records[-1]['end_frame']}, "
            f"expected {expected_end}"
        )
    if cumulative != int(summary["frames"]):
        errors.append(
            f"worker frame total is {cumulative}, "
            f"summary reports {summary['frames']}"
        )
    return boundaries, records, errors


def quality_metrics(
    ffmpeg: Path,
    output: Path,
    reference: Path,
    boundaries: list[int],
    boundary_window: int,
    metrics_path: Path,
) -> dict[str, Any]:
    metric_filter = (
        "[0:v][1:v]libvmaf="
        "feature='name=psnr|name=float_ssim':"
        f"log_fmt=json:log_path={metrics_path}:n_threads=16"
    )
    subprocess.run(
        [
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
        ],
        check=True,
    )
    value = json.loads(metrics_path.read_text(encoding="utf-8"))
    frames = {
        int(frame["frameNum"]): frame["metrics"]
        for frame in value["frames"]
    }
    boundary_frames = sorted(
        {
            index
            for boundary in boundaries
            for index in range(
                max(0, boundary - boundary_window),
                min(len(frames), boundary + boundary_window + 1),
            )
        }
    )
    boundary_vmaf = [
        float(frames[index]["vmaf"])
        for index in boundary_frames
        if index in frames
    ]
    all_vmaf = [
        float(metrics["vmaf"])
        for metrics in frames.values()
    ]
    boundary_frame_set = set(boundary_frames)
    non_boundary_vmaf = [
        float(metrics["vmaf"])
        for index, metrics in frames.items()
        if index not in boundary_frame_set
    ]
    boundary_mean = (
        statistics.fmean(boundary_vmaf)
        if boundary_vmaf
        else None
    )
    non_boundary_mean = (
        statistics.fmean(non_boundary_vmaf)
        if non_boundary_vmaf
        else None
    )
    return {
        "pooled_metrics": value["pooled_metrics"],
        "frames_compared": len(frames),
        "minimum_frame_vmaf": min(all_vmaf),
        "minimum_frame_vmaf_index": min(
            frames,
            key=lambda index: float(frames[index]["vmaf"]),
        ),
        "frames_below_vmaf_50": sum(
            score < 50.0 for score in all_vmaf
        ),
        "boundary_frames": boundary_frames,
        "boundary_vmaf_mean": boundary_mean,
        "boundary_vmaf_minimum": (
            min(boundary_vmaf)
            if boundary_vmaf
            else None
        ),
        "non_boundary_vmaf_mean": non_boundary_mean,
        "non_boundary_vmaf_minimum": (
            min(non_boundary_vmaf)
            if non_boundary_vmaf
            else None
        ),
        "boundary_vmaf_regression": (
            non_boundary_mean - boundary_mean
            if (
                boundary_mean is not None
                and non_boundary_mean is not None
            )
            else None
        ),
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "value": value})


def main() -> None:
    args = parser().parse_args()
    output = args.output.expanduser().resolve()
    report = args.report.expanduser().resolve()
    ffmpeg = args.ffmpeg_bin.expanduser().resolve()
    ffprobe = ffmpeg.with_name("ffprobe")
    summary = (
        None
        if args.summary is None
        else json.loads(
            args.summary.expanduser().resolve().read_text(encoding="utf-8")
        )
    )
    if not output.is_file():
        raise FileNotFoundError(output)
    streams = probe_streams(ffprobe, output)
    video = next(
        stream for stream in streams if stream["codec_type"] == "video"
    )
    audio = next(
        (
            stream
            for stream in streams
            if stream["codec_type"] == "audio"
        ),
        None,
    )
    fps = (
        args.expected_fps
        if args.expected_fps is not None
        else fraction(video["avg_frame_rate"])
    )
    timestamps = probe_timestamps(ffprobe, output, "v:0")
    expected_frames = args.expected_frames
    if expected_frames is None and summary is not None:
        expected_frames = int(summary["frames"])
    if expected_frames is None:
        expected_frames = int(video.get("nb_read_packets", len(timestamps)))
    expected_step = 1.0 / fps
    deltas = [
        second - first
        for first, second in zip(timestamps, timestamps[1:])
    ]
    timestamp_errors = [
        abs(timestamp - index * expected_step)
        for index, timestamp in enumerate(timestamps)
    ]
    tolerance = max(1e-6, expected_step / 1000.0)
    boundaries, worker_records, range_errors = segment_boundaries(summary)
    boundary_deltas = {
        str(index): timestamps[index] - timestamps[index - 1]
        for index in boundaries
        if 0 < index < len(timestamps)
    }
    video_packets = probe_video_packets(ffprobe, output)
    video_dts = [float(packet["dts"]) for packet in video_packets]
    video_dts_deltas = [
        second - first
        for first, second in zip(video_dts, video_dts[1:])
    ]
    errors = decode_errors(ffmpeg, output)
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "video_frame_count",
        len(timestamps) == expected_frames,
        {"actual": len(timestamps), "expected": expected_frames},
    )
    add_check(
        checks,
        "video_starts_at_zero",
        bool(timestamps) and abs(timestamps[0]) <= tolerance,
        timestamps[0] if timestamps else None,
    )
    add_check(
        checks,
        "video_pts_strictly_monotonic",
        all(delta > 0 for delta in deltas),
        min(deltas) if deltas else None,
    )
    add_check(
        checks,
        "video_pts_uniform",
        all(abs(delta - expected_step) <= tolerance for delta in deltas),
        {
            "expected_step": expected_step,
            "minimum_step": min(deltas) if deltas else None,
            "maximum_step": max(deltas) if deltas else None,
            "maximum_absolute_timestamp_error": (
                max(timestamp_errors) if timestamp_errors else None
            ),
        },
    )
    add_check(
        checks,
        "worker_ranges_contiguous",
        not range_errors,
        range_errors,
    )
    add_check(
        checks,
        "split_boundary_pts_uniform",
        all(
            abs(delta - expected_step) <= tolerance
            for delta in boundary_deltas.values()
        ),
        boundary_deltas,
    )
    add_check(
        checks,
        "video_packet_count",
        len(video_packets) == expected_frames,
        {"actual": len(video_packets), "expected": expected_frames},
    )
    add_check(
        checks,
        "video_dts_strictly_monotonic",
        all(delta > 0 for delta in video_dts_deltas),
        min(video_dts_deltas) if video_dts_deltas else None,
    )
    add_check(
        checks,
        "video_dts_uniform",
        all(
            abs(delta - expected_step) <= tolerance
            for delta in video_dts_deltas
        ),
        {
            "expected_step": expected_step,
            "minimum_step": (
                min(video_dts_deltas)
                if video_dts_deltas
                else None
            ),
            "maximum_step": (
                max(video_dts_deltas)
                if video_dts_deltas
                else None
            ),
        },
    )
    add_check(
        checks,
        "split_boundaries_are_keyframes",
        all(
            index < len(video_packets)
            and bool(video_packets[index]["keyframe"])
            for index in boundaries
        ),
        {
            str(index): (
                bool(video_packets[index]["keyframe"])
                if index < len(video_packets)
                else None
            )
            for index in boundaries
        },
    )
    add_check(
        checks,
        "full_stream_decode",
        not errors,
        errors[:20],
    )
    audio_packets: list[dict[str, Any]] = []
    if audio is not None:
        audio_packets = probe_audio_packets(ffprobe, output)
        audio_frames = probe_audio_frames(ffprobe, output)
        audio_pts = [float(packet["pts"]) for packet in audio_packets]
        audio_gaps = [
            float(current["pts"]) -
            (float(previous["pts"]) + float(previous["duration"]))
            for previous, current in zip(
                audio_packets,
                audio_packets[1:],
            )
        ]
        sample_rate = int(audio.get("sample_rate", 48000))
        decoded_audio_gaps = [
            float(current["pts"]) -
            (
                float(previous["pts"]) +
                int(previous["samples"]) / sample_rate
            )
            for previous, current in zip(
                audio_frames,
                audio_frames[1:],
            )
        ]
        audio_tolerance = 2e-6
        add_check(
            checks,
            "audio_decoded_starts_at_zero",
            bool(audio_frames)
            and abs(float(audio_frames[0]["pts"])) <= audio_tolerance,
            float(audio_frames[0]["pts"]) if audio_frames else None,
        )
        first_packet = audio_packets[0] if audio_packets else None
        preroll_seconds = (
            int(first_packet["skip_samples"]) / sample_rate
            if first_packet is not None
            else 0.0
        )
        add_check(
            checks,
            "audio_packet_preroll_is_signaled",
            first_packet is not None
            and (
                float(first_packet["pts"]) >= -audio_tolerance
                or abs(
                    preroll_seconds + float(first_packet["pts"])
                ) <= audio_tolerance
            ),
            {
                "first_packet_pts": (
                    float(first_packet["pts"])
                    if first_packet is not None
                    else None
                ),
                "skip_samples": (
                    int(first_packet["skip_samples"])
                    if first_packet is not None
                    else None
                ),
                "sample_rate": sample_rate,
            },
        )
        add_check(
            checks,
            "audio_pts_strictly_monotonic",
            all(
                second > first
                for first, second in zip(audio_pts, audio_pts[1:])
            ),
            len(audio_pts),
        )
        add_check(
            checks,
            "audio_packet_continuity",
            all(abs(gap) <= audio_tolerance for gap in audio_gaps),
            {
                "minimum_gap": min(audio_gaps) if audio_gaps else None,
                "maximum_gap": max(audio_gaps) if audio_gaps else None,
            },
        )
        add_check(
            checks,
            "audio_decoded_frame_continuity",
            all(
                abs(gap) <= audio_tolerance
                for gap in decoded_audio_gaps
            ),
            {
                "minimum_gap": (
                    min(decoded_audio_gaps)
                    if decoded_audio_gaps
                    else None
                ),
                "maximum_gap": (
                    max(decoded_audio_gaps)
                    if decoded_audio_gaps
                    else None
                ),
            },
        )
    duration = float(video["duration"])
    bitrate_mbps = output.stat().st_size * 8.0 / duration / 1_000_000.0
    quality: dict[str, Any] | None = None
    if args.reference is not None:
        reference = args.reference.expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="fast-overlay-vmaf-",
            suffix=".json",
            dir=report.parent,
            delete=False,
        ) as temporary:
            metrics_path = Path(temporary.name)
        try:
            quality = quality_metrics(
                ffmpeg,
                output,
                reference,
                boundaries,
                args.boundary_window,
                metrics_path,
            )
        finally:
            metrics_path.unlink(missing_ok=True)
        add_check(
            checks,
            "quality_frame_count",
            quality["frames_compared"] == expected_frames,
            quality["frames_compared"],
        )
        if args.minimum_vmaf_mean is not None:
            mean_vmaf = float(
                quality["pooled_metrics"]["vmaf"]["mean"]
            )
            add_check(
                checks,
                "minimum_vmaf_mean",
                mean_vmaf >= args.minimum_vmaf_mean,
                {
                    "actual": mean_vmaf,
                    "minimum": args.minimum_vmaf_mean,
                },
            )
        if (
            args.minimum_boundary_vmaf is not None
            and quality["boundary_vmaf_minimum"] is not None
        ):
            boundary_minimum = float(
                quality["boundary_vmaf_minimum"]
            )
            add_check(
                checks,
                "minimum_boundary_vmaf",
                boundary_minimum >= args.minimum_boundary_vmaf,
                {
                    "actual": boundary_minimum,
                    "minimum": args.minimum_boundary_vmaf,
                },
            )
        if (
            args.maximum_boundary_vmaf_regression is not None
            and quality["boundary_vmaf_regression"] is not None
        ):
            regression = float(
                quality["boundary_vmaf_regression"]
            )
            add_check(
                checks,
                "maximum_boundary_vmaf_regression",
                regression <= args.maximum_boundary_vmaf_regression,
                {
                    "actual": regression,
                    "maximum": args.maximum_boundary_vmaf_regression,
                },
            )
    value = {
        "schema_version": 1,
        "status": (
            "pass"
            if all(check["passed"] for check in checks)
            else "fail"
        ),
        "output": str(output),
        "summary": (
            None
            if args.summary is None
            else str(args.summary.expanduser().resolve())
        ),
        "reference": (
            None
            if args.reference is None
            else str(args.reference.expanduser().resolve())
        ),
        "media": {
            "size_bytes": output.stat().st_size,
            "bitrate_mbps": bitrate_mbps,
            "fps": fps,
            "video_frames": len(timestamps),
            "video_duration": duration,
            "audio_packets": len(audio_packets),
            "streams": streams,
        },
        "segmentation": {
            "workers": len(worker_records),
            "boundaries": boundaries,
            "boundary_deltas": boundary_deltas,
        },
        "quality": quality,
        "checks": checks,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report.with_suffix(report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if value["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
