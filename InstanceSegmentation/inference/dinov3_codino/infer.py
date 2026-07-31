#!/usr/bin/env python3
"""Run optimized TensorRT or stable PyTorch Co-DINO inference."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILY_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = FAMILY_ROOT.parent
DEFAULT_ARTIFACT_ROOT = FAMILY_ROOT / "artifacts"
DEFAULT_RUNTIME_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "codino"
DEFAULT_DINOV3_SOURCE = FAMILY_ROOT / ".runtime" / "src" / "dinov3_root"
DEFAULT_CONFIG = DEFAULT_ARTIFACT_ROOT / "detector" / "resolved_config.py"
DEFAULT_CHECKPOINT = (
    DEFAULT_ARTIFACT_ROOT
    / "detector"
    / "teacher_vitl_codino_epoch6_deploy.pth"
)
DEFAULT_CLASSIFIER_MANIFEST = (
    DEFAULT_ARTIFACT_ROOT / "classifier" / "backbone" / "manifest.json"
)
DEFAULT_TRT_BUNDLE = (
    DEFAULT_ARTIFACT_ROOT
    / "trt"
    / "fast-sm120-fixed-b2-epoch6-v1"
    / "manifest.json"
)
DEFAULT_SHARED_ROOT = FAMILY_ROOT / ".runtime" / "shared"
for _runtime_source in (
    DEFAULT_SHARED_ROOT,
    PACKAGE_PARENT,
    DEFAULT_RUNTIME_SOURCE,
    DEFAULT_DINOV3_SOURCE,
):
    if _runtime_source.is_dir() and str(_runtime_source) not in sys.path:
        sys.path.insert(0, str(_runtime_source))
if str(FAMILY_ROOT) not in sys.path:
    sys.path.insert(0, str(FAMILY_ROOT))

try:
    from .adapter import CoDinoAdapter
    from .model import (
        InstanceSegmentationSettings,
        VideoInferenceSettings,
        build_runtime,
    )
    from .optimized import run_fast_video_inference
    from .trt.bundle import load_engine_bundle
    from .trt.runtime import FixedTrtPartitionSettings, parse_feature_shapes
except ImportError:
    from adapter import CoDinoAdapter
    from model import (
        InstanceSegmentationSettings,
        VideoInferenceSettings,
        build_runtime,
    )
    from optimized import run_fast_video_inference
    from trt.bundle import load_engine_bundle
    from trt.runtime import FixedTrtPartitionSettings, parse_feature_shapes

from mask_geometry import DEFAULT_MAX_MASK_POINTS
from persistence import AsyncSqliteWriter, SqliteWriter
from pipelines import run_video_inference


@dataclass(frozen=True)
class CodinoCandidateIo:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class CodinoCandidateArtifacts:
    config_path: Path
    checkpoint: Path
    runtime_checkpoint: Path
    classifier_manifest: Path | None
    bundle_manifest: Path | None
    bundle_profile: str | None
    backbone_engine: Path | None
    query_engine: Path | None
    decoder_engine: Path | None
    mask_engine: Path | None
    query_shapes: str | None
    query_plugin_extension: Path | None
    fixed_batch_size: int
    extra_site_packages: Path | None


def require_codino_file(value: object, *, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required by the clean Co-DINO candidate")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def optional_codino_directory(value: object, *, label: str) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_codino_io(args: Any) -> CodinoCandidateIo:
    input_path = require_codino_file(args.input, label="input video")
    return CodinoCandidateIo(
        input_path=input_path,
        output_path=Path(args.output).expanduser().resolve(),
    )


def resolve_codino_artifacts(args: Any) -> CodinoCandidateArtifacts:
    config_path = require_codino_file(args.config, label="Co-DINO config")
    checkpoint = require_codino_file(args.checkpoint, label="Co-DINO checkpoint")
    classifier_manifest = (
        require_codino_file(args.classifier_manifest, label="classifier manifest")
        if bool(args.classifier)
        else None
    )
    if str(args.backend) == "pytorch":
        return CodinoCandidateArtifacts(
            config_path=config_path,
            checkpoint=checkpoint,
            runtime_checkpoint=checkpoint,
            classifier_manifest=classifier_manifest,
            bundle_manifest=None,
            bundle_profile=None,
            backbone_engine=None,
            query_engine=None,
            decoder_engine=None,
            mask_engine=None,
            query_shapes=None,
            query_plugin_extension=None,
            fixed_batch_size=int(args.batch_size),
            extra_site_packages=None,
        )
    if not bool(args.classifier):
        raise ValueError("optimized TensorRT Co-DINO requires the classifier")
    bundle = load_engine_bundle(
        Path(args.trt_bundle),
        verify=str(args.trt_verify),
        config_path=config_path,
        checkpoint_path=checkpoint,
        classifier_checkpoint=classifier_manifest,
        runtime_python=Path(sys.executable),
    )
    if bundle.runtime_profile != "fast-b2":
        raise ValueError(
            "the tensorrt-fast backend requires a compatible fixed-B2 bundle"
        )
    if bundle.runtime_checkpoint is None:
        raise ValueError(
            "the tensorrt-fast bundle has no deployment runtime checkpoint"
        )
    return CodinoCandidateArtifacts(
        config_path=config_path,
        checkpoint=checkpoint,
        runtime_checkpoint=bundle.runtime_checkpoint,
        classifier_manifest=classifier_manifest,
        bundle_manifest=bundle.manifest_path,
        bundle_profile=bundle.profile,
        backbone_engine=bundle.backbone_engine,
        query_engine=bundle.query_encoder_engine,
        decoder_engine=bundle.decoder_engine,
        mask_engine=bundle.mask_head_engine,
        query_shapes=bundle.query_shapes,
        query_plugin_extension=bundle.query_plugin_extension,
        fixed_batch_size=bundle.batch_size,
        extra_site_packages=optional_codino_directory(
            args.trt_extra_site_packages, label="TensorRT extra site-packages"
        ),
    )


class CodinoProgressReporter:
    def __init__(self, interval_seconds: float) -> None:
        self._interval = max(0.0, float(interval_seconds))
        self._last_progress = 0.0

    def __call__(self, progress: object) -> None:
        now = time.monotonic()
        processed = int(getattr(progress, "processed_frames"))
        total = getattr(progress, "total_frames", None)
        if now - self._last_progress < self._interval:
            return
        self._last_progress = now
        total_text = "?" if total is None else str(int(total))
        detections = int(
            getattr(progress, "result_items", getattr(progress, "detections", 0))
        )
        print(
            f"[progress] processed={processed}/{total_text} detections={detections} fps={float(getattr(progress, 'wall_fps')):.3f}",
            flush=True,
        )


def configure_codino_torch(torch: Any, *, tf32: bool) -> None:
    """Apply the Co-DINO candidate's CUDA execution policy."""
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def run_native_codino(args: Any) -> int:
    io = resolve_codino_io(args)
    if io.output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output already exists: {io.output_path}")
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    max_frames = args.max_frames
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    artifacts = resolve_codino_artifacts(args)
    if batch_size != artifacts.fixed_batch_size:
        raise ValueError(
            f"{artifacts.bundle_profile} requires batch_size "
            f"{artifacts.fixed_batch_size}, got {batch_size}"
        )
    import torch

    configure_codino_torch(torch, tf32=bool(args.tf32))
    trt_settings = None
    if str(args.backend) == "tensorrt-fast":
        if any(
            value is None
            for value in (
                artifacts.backbone_engine,
                artifacts.query_engine,
                artifacts.decoder_engine,
                artifacts.mask_engine,
                artifacts.query_shapes,
                artifacts.query_plugin_extension,
            )
        ):
            raise RuntimeError("optimized TensorRT artifacts are incomplete")
        trt_settings = FixedTrtPartitionSettings(
            backbone_engine=artifacts.backbone_engine,
            query_encoder_engine=artifacts.query_engine,
            decoder_engine=artifacts.decoder_engine,
            mask_head_engine=artifacts.mask_engine,
            query_encoder_shapes=parse_feature_shapes(artifacts.query_shapes),
            extra_site_packages=artifacts.extra_site_packages,
            query_plugin_extension=artifacts.query_plugin_extension,
        )
    (runtime, _classifier_payload) = build_runtime(
        segmenter_settings=InstanceSegmentationSettings(
            config_path=artifacts.config_path,
            checkpoint=artifacts.runtime_checkpoint,
            target_size=args.target_size,
            score_threshold=float(args.score_thresh),
            model_score_threshold=float(args.model_score_thr),
            trt_deployment_shell=str(args.backend) == "tensorrt-fast",
        ),
        trt_settings=trt_settings,
        classifier_manifest=artifacts.classifier_manifest,
        classifier_mode=str(args.classifier_mode),
        device=str(args.device),
        fixed_batch_size=batch_size,
        disable_mask_iou_head=bool(args.disable_mask_iou_head),
    )
    inference_settings = VideoInferenceSettings(
        amp=str(args.amp),
        batch_size=batch_size,
        score_threshold=float(args.score_thresh),
    )
    writer_type = (
        AsyncSqliteWriter
        if str(args.backend) == "tensorrt-fast"
        else SqliteWriter
    )
    sink = writer_type(
        io.output_path,
        overwrite=bool(args.overwrite),
        safe=not bool(args.fast_sqlite),
    )
    run_metadata = {
        "checkpoint": str(artifacts.checkpoint),
        "classifier_manifest": (
            None
            if artifacts.classifier_manifest is None
            else str(artifacts.classifier_manifest)
        ),
        "classifier_mode": str(args.classifier_mode),
        "config": str(artifacts.config_path),
        "backend": str(args.backend),
        "trt_bundle": (
            None
            if artifacts.bundle_manifest is None
            else str(artifacts.bundle_manifest)
        ),
        "trt_profile": artifacts.bundle_profile,
        "max_mask_points": DEFAULT_MAX_MASK_POINTS,
    }
    if str(args.backend) == "tensorrt-fast":
        contract_result = run_fast_video_inference(
            input_path=io.input_path,
            runtime=runtime,
            writer=sink,
            settings=inference_settings,
            max_frames=max_frames,
            warmup_frames=max(0, int(args.warmup_frames)),
            metadata=run_metadata,
            progress=CodinoProgressReporter(args.progress_interval_sec),
        )
    else:
        contract_result = run_video_inference(
            input_path=io.input_path,
            adapter=CoDinoAdapter(runtime, inference_settings),
            writer=sink,
            batch_size=batch_size,
            max_frames=max_frames,
            warmup_frames=max(0, int(args.warmup_frames)),
            prefetch_batches=2,
            metadata=run_metadata,
            progress=CodinoProgressReporter(args.progress_interval_sec),
        )
    wall_fps = contract_result.wall_fps
    print(
        f"processed {contract_result.processed_frames} frames in "
        f"{contract_result.wall_elapsed_sec:.3f}s ({wall_fps:.3f} fps)",
        flush=True,
    )
    print(
        f"measured compute throughput: {contract_result.compute_fps:.3f} img/s",
        flush=True,
    )
    print(
        f"wrote {contract_result.result_items} segmentation rows",
        flush=True,
    )
    print(f"saved sqlite to: {io.output_path}", flush=True)
    return 0


def parse_codino_target_size(value: str) -> tuple[int, int]:
    parts = str(value).strip().lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target size must be WxH")
    (width, height) = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("target size must be positive")
    return (height, width)


def build_codino_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=("tensorrt-fast", "pytorch"),
        default="tensorrt-fast",
        help="optimized RTX 5090 TensorRT path or stable eager PyTorch path",
    )
    parser.add_argument(
        "--classifier", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--classifier-manifest",
        "--classifier-checkpoint",
        dest="classifier_manifest",
        default=str(DEFAULT_CLASSIFIER_MANIFEST),
        help="backbone ROI classifier manifest (legacy option name is accepted)",
    )
    parser.add_argument(
        "--classifier-mode",
        choices=("fast", "accuracy"),
        default="fast",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--target-size", type=parse_codino_target_size, default=(720, 1280)
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--model-score-thr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fast-sqlite",
        action="store_true",
        help="trade crash durability for faster SQLite writes",
    )
    parser.add_argument("--disable-mask-iou-head", action="store_true")
    parser.add_argument("--trt-bundle", default=str(DEFAULT_TRT_BUNDLE))
    parser.add_argument(
        "--trt-verify",
        choices=("metadata", "engines", "full"),
        default="engines",
    )
    parser.add_argument("--trt-extra-site-packages")
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_codino_parser().parse_args(argv)
    return run_native_codino(args)


if __name__ == "__main__":
    raise SystemExit(main())
