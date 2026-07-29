#!/usr/bin/env python3
"""Run segmentation, face detection, or both and write unified SQLite."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from contracts import TaskType
from orchestration import (
    InferenceMode,
    OrchestrationRequest,
    run_orchestrated_inference,
)
from registry import list_models


def build_parser() -> argparse.ArgumentParser:
    segmentation_models = tuple(
        model.model_id
        for model in list_models(TaskType.INSTANCE_SEGMENTATION)
    )
    face_models = tuple(
        model.model_id
        for model in list_models(TaskType.OBJECT_DETECTION)
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=tuple(mode.value for mode in InferenceMode),
    )
    parser.add_argument(
        "--segmentation-model",
        choices=segmentation_models,
        help="required by segmentation and segmentation-face modes",
    )
    parser.add_argument(
        "--segmentation-backend",
        default="auto",
        choices=("auto", "tensorrt-fast", "tensorrt-backbone", "pytorch"),
    )
    parser.add_argument(
        "--face-model",
        default="rtdetr_head_face",
        choices=face_models,
    )
    parser.add_argument(
        "--face-backend",
        default="auto",
        choices=("auto", "tensorrt-fast", "pytorch"),
        help="face inference backend; auto selects the model default",
    )
    parser.add_argument(
        "--face-classes",
        nargs="*",
        default=("Face", "Head"),
        help="face-detector classes; pass the option with no values for all classes",
    )
    parser.add_argument(
        "--face-trt-bundle",
        type=Path,
        help=(
            "optional TensorRT bundle manifest forwarded to compatible face "
            "models (for example the reviewed Face DINO v2 B16 profile)"
        ),
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for isolated model processes",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--face-warmup-iterations", type=int, default=3)
    parser.add_argument(
        "--parallel-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "run dinov3_codino_mh0 and face_dino_v2 subprocesses concurrently "
            "in segmentation-face mode; each model still writes an isolated "
            "SQLite before atomic merge"
        ),
    )
    parser.add_argument(
        "--parallel-model-stagger-seconds",
        type=float,
        default=0.0,
        help=(
            "when parallel models are enabled, start face inference first and "
            "delay sibling launch to reduce peak GPU power contention"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fast-sqlite",
        action="store_true",
        help="trade crash durability for faster final SQLite creation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = OrchestrationRequest(
            input_path=args.input,
            output_path=args.output,
            mode=InferenceMode(args.mode),
            segmentation_model=args.segmentation_model,
            segmentation_backend=args.segmentation_backend,
            face_model=args.face_model,
            face_backend=args.face_backend,
            face_classes=tuple(args.face_classes),
            face_trt_bundle=args.face_trt_bundle,
            runtime_python=args.runtime_python,
            device=args.device,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            face_warmup_iterations=args.face_warmup_iterations,
            parallel_models=bool(args.parallel_models),
            parallel_model_stagger_seconds=args.parallel_model_stagger_seconds,
            overwrite=args.overwrite,
            fast_sqlite=args.fast_sqlite,
        )
        run_orchestrated_inference(request)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(f"run_inference.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
