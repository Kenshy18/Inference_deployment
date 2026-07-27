"""Command line entry point for all supported overlay views."""

from __future__ import annotations

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overlay-render",
        description=(
            "Render raw inference, tracked, final, or face-only overlays from SQLite."
        ),
    )
    parser.add_argument("--mode", choices=MODES, required=True)
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
        default="mp4v",
        help=(
            "mp4v (OpenCV), h264/avc1/x264 (FFmpeg libx264), or "
            "h264_nvenc/nvenc (NVIDIA NVENC)"
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
        default="p5",
        help="NVENC preset p1-p7; p1 is fastest, p7 highest quality",
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
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.include_faces and args.mode != "final":
        raise ValueError("--include-faces is only valid with --mode final")
    if args.mode == "final" and args.include_faces and args.face_sqlite is None:
        raise ValueError("--include-faces requires --face-sqlite")
    if args.face_sqlite is not None and not (
        args.mode == "final" and args.include_faces
    ):
        raise ValueError("--face-sqlite requires --mode final --include-faces")


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

    options = RenderOptions(
        mode=args.mode,
        mask_alpha=args.mask_alpha,
        outline_thickness=args.outline_thickness,
        box_thickness=args.box_thickness,
        show_labels=not args.no_labels,
        codec=args.codec,
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
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_name(
            f".{manifest.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(manifest)


if __name__ == "__main__":
    main()
