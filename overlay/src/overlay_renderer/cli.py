"""Command line entry point for all supported overlay views."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .render import RenderOptions, render_video
from .sources import (
    inspect_inference_source,
    inspect_mask_source,
    iter_face_frames,
    iter_mask_frames,
    iter_raw_segmentation_frames,
)


MODES = ("raw", "tracked", "final", "faces")
EXECUTION_MODES = ("cpu", "nvenc", "fast")
OVERLAY_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = OVERLAY_ROOT / "native"
NATIVE_RUNNER = NATIVE_ROOT / "segmented.py"
NATIVE_RENDERER = NATIVE_ROOT / "build" / "overlay_native"
NATIVE_FFMPEG = (
    OVERLAY_ROOT
    / ".runtime"
    / "ffmpeg-nvenc-btbn-8.1"
    / "bin"
    / "ffmpeg"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overlay-render",
        description=(
            "Render raw inference, tracked, final, or face-only overlays from SQLite."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="cpu",
        help=(
            "cpu: OpenCV+libx264, nvenc: OpenCV+NVENC, "
            "fast: NVDEC+CUDA+parallel NVENC"
        ),
    )
    parser.add_argument(
        "--mode",
        "--overlay-type",
        dest="mode",
        choices=MODES,
        required=True,
        help=(
            "raw: inference masks, tracked: minimal postprocess, "
            "final: final postprocess, faces: face boxes only"
        ),
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--sqlite",
        type=Path,
        required=True,
        help=(
            "unified inference SQLite for raw/faces; postprocess mask SQLite "
            "for tracked/final"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-faces",
        action="store_true",
        help="add face boxes to final mode",
    )
    parser.add_argument(
        "--face-sqlite",
        type=Path,
        help="unified inference SQLite containing role=face_detection",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.32)
    parser.add_argument("--outline-thickness", type=int, default=2)
    parser.add_argument("--box-thickness", type=int, default=2)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument(
        "--codec",
        default=None,
        help=(
            "legacy codec override; execution mode normally selects "
            "h264 or h264_nvenc automatically"
        ),
    )
    parser.add_argument(
        "--h264-crf",
        type=int,
        default=18,
        help="H.264 quality: 0-51, lower is higher quality (default: 18)",
    )
    parser.add_argument(
        "--h264-preset",
        default="veryfast",
        help="libx264 speed/compression preset (default: veryfast)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        help="optional FFmpeg executable; bundled imageio-ffmpeg is used by default",
    )
    parser.add_argument(
        "--nvenc-cq",
        type=int,
        default=18,
        help="NVENC constant-quality target: 0-51 (default: 18)",
    )
    parser.add_argument(
        "--nvenc-preset",
        default=None,
        help=(
            "NVENC preset p1-p7; defaults to p5 for nvenc and p1 for fast"
        ),
    )
    parser.add_argument(
        "--nvenc-gpu",
        type=int,
        default=0,
        help="NVENC GPU index (default: 0)",
    )
    parser.add_argument(
        "--target-bitrate-mbps",
        type=float,
        help=(
            "use a shared constrained bitrate for libx264/NVENC instead of "
            "CRF/CQ rate control"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="segment workers used by fast mode (default: 6)",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="fast-mode workers assigned to libx264 (default: 0)",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="copy source audio in fast mode",
    )
    parser.add_argument(
        "--faststart",
        action="store_true",
        help="move MP4 metadata to the beginning of the file",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.nvenc_preset is None:
        args.nvenc_preset = (
            "p1" if args.execution_mode == "fast" else "p5"
        )
    if args.nvenc_preset not in {
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
    }:
        raise ValueError("--nvenc-preset must be between p1 and p7")
    if args.nvenc_gpu < 0:
        raise ValueError("--nvenc-gpu must be non-negative")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1")
    if args.outline_thickness < 1 or args.box_thickness < 1:
        raise ValueError("line thickness must be at least 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if (
        args.end_frame is not None
        and args.end_frame < args.start_frame
    ):
        raise ValueError("--end-frame must be >= --start-frame")
    if (
        args.target_bitrate_mbps is not None
        and args.target_bitrate_mbps <= 0
    ):
        raise ValueError("--target-bitrate-mbps must be positive")
    if args.include_faces and args.mode != "final":
        raise ValueError("--include-faces is only valid with --mode final")
    if args.mode == "final" and args.include_faces and args.face_sqlite is None:
        raise ValueError("--include-faces requires --face-sqlite")
    if args.face_sqlite is not None and not (
        args.mode == "final" and args.include_faces
    ):
        raise ValueError("--face-sqlite requires --mode final --include-faces")
    if args.execution_mode == "fast":
        if args.target_bitrate_mbps is None:
            raise ValueError(
                "--execution-mode fast requires --target-bitrate-mbps"
            )
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        if not 0 <= args.cpu_workers <= args.workers:
            raise ValueError(
                "--cpu-workers must be between 0 and --workers"
            )
        if args.codec is not None and args.codec.lower() not in {
            "nvenc",
            "h264_nvenc",
        }:
            raise ValueError("fast mode only supports H.264 NVENC output")
    else:
        if args.copy_audio:
            raise ValueError("--copy-audio is currently supported by fast mode")
        if args.faststart:
            raise ValueError("--faststart is currently supported by fast mode")
        if args.cpu_workers:
            raise ValueError("--cpu-workers is only valid with fast mode")
        if args.execution_mode == "nvenc":
            if args.codec is not None and args.codec.lower() not in {
                "nvenc",
                "h264_nvenc",
            }:
                raise ValueError("nvenc mode requires h264_nvenc")
        elif args.codec is not None and args.codec.lower() in {
            "nvenc",
            "h264_nvenc",
        }:
            raise ValueError("cpu mode cannot use an NVENC codec")


def _selected_codec(args: argparse.Namespace) -> str:
    if args.execution_mode in {"nvenc", "fast"}:
        return "h264_nvenc"
    return "h264" if args.codec is None else str(args.codec)


def _atomic_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fast_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> list[str]:
    ffmpeg = (
        NATIVE_FFMPEG
        if args.ffmpeg_bin is None
        else Path(args.ffmpeg_bin).expanduser().resolve()
    )
    command = [
        sys.executable,
        str(NATIVE_RUNNER),
        "--video",
        str(Path(args.video).expanduser().resolve()),
        "--sqlite",
        str(Path(args.sqlite).expanduser().resolve()),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--renderer",
        str(NATIVE_RENDERER),
        "--ffmpeg-bin",
        str(ffmpeg),
        "--workers",
        str(args.workers),
        "--cpu-workers",
        str(args.cpu_workers),
        "--start-frame",
        str(args.start_frame),
        "--bitrate-mbps",
        str(args.target_bitrate_mbps),
        "--nvenc-preset",
        args.nvenc_preset,
        "--nvenc-gpu",
        str(args.nvenc_gpu),
        "--mask-alpha",
        str(args.mask_alpha),
        "--outline-thickness",
        str(args.outline_thickness),
        "--box-thickness",
        str(args.box_thickness),
        "--gpu-pipeline",
        "--compact-output",
    ]
    if args.end_frame is not None:
        command.extend(["--end-frame", str(args.end_frame)])
    if args.no_labels:
        command.append("--no-labels")
    if args.include_faces:
        command.extend(
            [
                "--include-faces",
                "--face-sqlite",
                str(Path(args.face_sqlite).expanduser().resolve()),
            ]
        )
    if args.copy_audio:
        command.append("--copy-audio")
    if args.faststart:
        command.append("--faststart")
    return command


def _run_fast(
    args: argparse.Namespace,
    *,
    output: Path,
    manifest: Path | None,
    sources,
) -> None:
    for required in (NATIVE_RUNNER, NATIVE_RENDERER):
        if not required.is_file():
            raise FileNotFoundError(
                f"fast overlay dependency is missing: {required}; "
                "run overlay/native/build.sh"
            )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / f".{output.stem}.fast-{uuid.uuid4().hex}"
    command = _fast_command(args, output_dir=work_dir)
    completed = False
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = "\n".join(
                value.strip()
                for value in (result.stdout, result.stderr)
                if value.strip()
            )
            raise RuntimeError(
                "fast overlay failed; worker artifacts were retained at "
                f"{work_dir}"
                + (f"\n{detail}" if detail else "")
            )
        generated = work_dir / "final.mp4"
        summary_path = work_dir / "benchmark_summary.json"
        if not generated.is_file() or not summary_path.is_file():
            raise RuntimeError(
                "fast overlay did not create final.mp4 and "
                "benchmark_summary.json"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        os.replace(generated, output)
        summary["final_output"] = str(output)
        summary["temporary_worker_artifacts_retained"] = False
        payload: dict[str, object] = {
            "summary": summary,
            "sources": [source.as_dict() for source in sources],
            "video": str(Path(args.video).expanduser().resolve()),
            "execution_mode": "fast",
            "overlay_type": args.mode,
            "include_faces": bool(args.include_faces),
            "audio_copied": bool(args.copy_audio),
            "encoding": {
                "codec": "h264",
                "segment_encoders": summary.get("encoders", []),
                "nvenc_preset": args.nvenc_preset,
                "nvenc_gpu": args.nvenc_gpu,
                "target_bitrate_mbps": args.target_bitrate_mbps,
                "workers": args.workers,
                "cpu_workers": args.cpu_workers,
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        print(encoded)
        if manifest is not None:
            _atomic_manifest(manifest, payload)
        completed = True
    finally:
        if completed:
            shutil.rmtree(work_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_mode_args(args)
    manifest = (
        None
        if args.manifest is None
        else Path(args.manifest).expanduser().resolve()
    )
    output = Path(args.output).expanduser().resolve()
    if manifest is not None:
        if manifest == output:
            raise ValueError("manifest path must differ from output video")
        if manifest.exists() and not args.overwrite:
            raise FileExistsError(f"manifest already exists: {manifest}")

    mask_frames = None
    face_frames = None
    sources = []
    if args.mode == "raw":
        source = inspect_inference_source(args.sqlite, "instance_segmentation")
        sources.append(source)
        mask_frames = iter_raw_segmentation_frames(args.sqlite)
    elif args.mode in {"tracked", "final"}:
        source = inspect_mask_source(args.sqlite)
        sources.append(source)
        mask_frames = iter_mask_frames(args.sqlite)
    elif args.mode == "faces":
        source = inspect_inference_source(args.sqlite, "face_detection")
        sources.append(source)
        face_frames = iter_face_frames(args.sqlite)

    if args.mode == "final" and args.include_faces:
        face_source = inspect_inference_source(args.face_sqlite, "face_detection")
        sources.append(face_source)
        face_frames = iter_face_frames(args.face_sqlite)

    if args.execution_mode == "fast":
        _run_fast(
            args,
            output=output,
            manifest=manifest,
            sources=sources,
        )
        return

    options = RenderOptions(
        mode=args.mode,
        mask_alpha=args.mask_alpha,
        outline_thickness=args.outline_thickness,
        box_thickness=args.box_thickness,
        show_labels=not args.no_labels,
        codec=_selected_codec(args),
        h264_crf=args.h264_crf,
        h264_preset=args.h264_preset,
        nvenc_cq=args.nvenc_cq,
        nvenc_preset=args.nvenc_preset,
        nvenc_gpu=args.nvenc_gpu,
        target_bitrate_mbps=args.target_bitrate_mbps,
        ffmpeg_bin=args.ffmpeg_bin,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        progress_every=args.progress_every,
    )
    summary = render_video(
        video_path=args.video,
        output_path=output,
        mask_frames=mask_frames,
        face_frames=face_frames,
        sources=tuple(sources),
        options=options,
        overwrite=args.overwrite,
    )
    payload = {
        "summary": summary.as_dict(),
        "sources": [source.as_dict() for source in sources],
        "video": str(Path(args.video).expanduser().resolve()),
        "execution_mode": args.execution_mode,
        "overlay_type": args.mode,
        "include_faces": bool(args.include_faces),
        "audio_copied": False,
        "encoding": {
            "codec": options.normalized_codec,
            "h264_crf": options.h264_crf if options.uses_libx264 else None,
            "h264_preset": (
                options.h264_preset if options.uses_libx264 else None
            ),
            "nvenc_cq": options.nvenc_cq if options.uses_nvenc else None,
            "nvenc_preset": (
                options.nvenc_preset if options.uses_nvenc else None
            ),
            "nvenc_gpu": options.nvenc_gpu if options.uses_nvenc else None,
            "target_bitrate_mbps": options.target_bitrate_mbps,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if manifest is not None:
        _atomic_manifest(manifest, payload)


if __name__ == "__main__":
    main()
