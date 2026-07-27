"""Single end-to-end entrypoint for modular mask post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from common.config import (
    PipelineConfig,
    StageSpec,
    default_ellipse_pipeline,
    default_polygon_pipeline,
    load_pipeline_config,
)
from common.runner import PipelineRunner
from common.settings import resolve_models
from contracts.detector_sqlite import detect_mask_sqlite_kind


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run contract-connected postprocess features from raw detector "
            "JSONL, detector SQLite, or an existing tracked SQLite."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-jsonl", type=Path)
    inputs.add_argument("--input-sqlite", type=Path)
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shape-mode", choices=("ellipse", "polygon"), default="ellipse"
    )
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--class-policy-json", type=Path)
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        help="explicitly override the selected pipeline stage",
    )
    parser.add_argument(
        "--score-min",
        type=float,
        help="explicitly override the selected pipeline stage",
    )
    parser.add_argument(
        "--cut-detect",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cut-method")
    parser.add_argument(
        "--precomputed-cuts-json",
        type=Path,
        help=(
            "validated cuts artifact produced independently; replaces the "
            "configured video cut-detection stage"
        ),
    )
    parser.add_argument("--remove-short-tracks-max-frames", type=int)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--k2-run-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--k2-batch-size", type=int)
    parser.add_argument("--k2-prep-workers", type=int)
    parser.add_argument("--k2-precision", choices=("fp32", "fp16"))
    parser.add_argument(
        "--k2-forward-mode",
        choices=("states_only", "full"),
    )
    parser.add_argument(
        "--k2-profile-stages",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--k2-cudnn-benchmark",
        choices=("on", "off"),
    )
    parser.add_argument("--k2-tf32", choices=("default", "on", "off"))
    parser.add_argument(
        "--export-legacy-sqlite",
        "--export-dinov3-legacy-sqlite",
        dest="export_legacy_sqlite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also export a tentative Dinov3_postprocess-compatible "
            "masks/tracks/cuts SQLite"
        ),
    )
    parser.add_argument(
        "--face-mask-target",
        choices=("none", "face", "eyes"),
        default="none",
        help=(
            "append schema-v3 face postprocessing and merge the derived masks "
            "into a combined final SQLite"
        ),
    )
    parser.add_argument(
        "--eye-mask-shape",
        choices=("ellipse", "rectangle"),
        default="ellipse",
        help="shape used when --face-mask-target=eyes",
    )
    parser.add_argument(
        "--minimum-eye-confidence",
        type=float,
        default=0.35,
    )
    return parser


def _configured_pipeline(args: argparse.Namespace) -> PipelineConfig:
    input_sqlite_kind = (
        detect_mask_sqlite_kind(args.input_sqlite)
        if args.input_sqlite is not None
        else None
    )
    include_raw_stages = args.input_jsonl is not None or input_sqlite_kind in {
        "raw_detection",
        "unified_inference",
    }
    if args.pipeline_config is not None:
        source = load_pipeline_config(args.pipeline_config)
    elif args.shape_mode == "polygon":
        source = default_polygon_pipeline(include_preprocess=include_raw_stages)
    else:
        source = default_ellipse_pipeline(include_preprocess=include_raw_stages)

    if args.pipeline_config is None and input_sqlite_kind in {
        "raw_detection",
        "unified_inference",
    }:
        source = PipelineConfig(
            source.name.replace("_modular", "_raw_sqlite_modular"),
            tuple(
                StageSpec(
                    stage.id,
                    (
                        "preprocessing.raw_sqlite"
                        if stage.implementation == "preprocessing.normalize"
                        else stage.implementation
                    ),
                    dict(stage.options),
                    stage.enabled,
                )
                for stage in source.stages
            ),
        )

    if args.keyframe_interval is not None and args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be >= 1")
    stages: list[StageSpec] = []
    for stage in source.stages:
        if (
            args.precomputed_cuts_json is not None
            and stage.implementation == "cut_detection.video"
        ):
            continue
        options = dict(stage.options)
        if (
            stage.implementation == "preprocessing.score_policy"
            and args.score_min is not None
        ):
            options["score_min"] = float(args.score_min)
        elif stage.implementation == "cut_detection.video":
            if args.cut_detect is not None:
                options["enabled"] = bool(args.cut_detect)
            if args.cut_method is not None:
                options["method"] = str(args.cut_method)
        elif (
            stage.implementation == "tracking.greedy"
            and args.remove_short_tracks_max_frames is not None
        ):
            options["remove_short_tracks_max_frames"] = int(
                args.remove_short_tracks_max_frames
            )
        elif (
            stage.implementation == "keyframes.polygon.interval"
            and args.keyframe_interval is not None
        ):
            options["interval_frames"] = int(args.keyframe_interval)
        elif stage.implementation == "approximation.ellipse.production":
            if args.model_root is not None:
                options["model_root"] = str(args.model_root.expanduser().resolve())
                if args.k2_run_dir is None:
                    options["k2_run_dir"] = str(
                        args.model_root.expanduser().resolve() / "k2_v5"
                    )
            else:
                models = resolve_models(
                    Path(options["model_root"]) if options.get("model_root") else None
                )
                options.setdefault("model_root", str(models.root))
                options.setdefault("k2_run_dir", str(models.k2_dir))
            if args.k2_run_dir is not None:
                options["k2_run_dir"] = str(args.k2_run_dir.expanduser().resolve())
            if args.device is not None:
                options["device"] = choose_device(args.device)
            else:
                options.setdefault("device", choose_device("auto"))
            ellipse_extra = list(options.get("extra_args", []))
            for flag, value in (
                ("--k2-batch-size", args.k2_batch_size),
                ("--k2-prep-workers", args.k2_prep_workers),
                ("--k2-precision", args.k2_precision),
                ("--k2-forward-mode", args.k2_forward_mode),
                ("--k2-cudnn-benchmark", args.k2_cudnn_benchmark),
                ("--k2-tf32", args.k2_tf32),
            ):
                if value is not None:
                    ellipse_extra.extend((flag, str(value)))
            if args.k2_profile_stages:
                ellipse_extra.append("--k2-profile-stages")
            if ellipse_extra:
                options["extra_args"] = ellipse_extra
        elif (
            stage.implementation == "keyframes.ellipse.dense"
            and args.keyframe_interval is not None
        ):
            options["target_ratio"] = 1.0 / float(args.keyframe_interval)
        stages.append(
            StageSpec(
                stage.id,
                stage.implementation,
                options,
                stage.enabled,
            )
        )
    if not 0.0 <= args.minimum_eye_confidence <= 1.0:
        raise ValueError("--minimum-eye-confidence must be between 0 and 1")
    if args.face_mask_target != "none":
        stages.extend(
            (
                StageSpec(
                    "face_privacy_masks",
                    "face_privacy.masks",
                    {
                        "target": args.face_mask_target,
                        "eye_shape": args.eye_mask_shape,
                        "minimum_eye_confidence": args.minimum_eye_confidence,
                    },
                ),
                StageSpec(
                    "face_privacy_merge",
                    "face_privacy.merge",
                ),
                StageSpec(
                    "combined_output_validation",
                    "artifacts.validate",
                    {
                        "source_artifact": "combined_predictions_sqlite",
                        "output_artifact": "combined_validation_report",
                        "filename": "combined_validation.json",
                    },
                ),
            )
        )
    if args.export_legacy_sqlite:
        if args.face_mask_target != "none":
            stages.append(
                StageSpec(
                    "combined_legacy_sqlite_export",
                    "artifacts.legacy_sqlite",
                    {
                        "source_artifact": "combined_predictions_sqlite",
                        "output_artifact": (
                            "combined_legacy_predictions_sqlite"
                        ),
                        "filename": "predictions.with_faces.legacy.sqlite",
                    },
                )
            )
        elif not any(
            stage.enabled and stage.implementation == "artifacts.legacy_sqlite"
            for stage in stages
        ):
            stages.append(
                StageSpec(
                    "legacy_sqlite_export",
                    "artifacts.legacy_sqlite",
                )
            )
    return PipelineConfig(source.name, tuple(stages))


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    config = _configured_pipeline(args)
    initial: dict[str, Path] = {}
    if args.input_jsonl is not None:
        if args.face_mask_target != "none":
            raise ValueError(
                "--face-mask-target requires a unified inference SQLite"
            )
        initial["input_jsonl"] = args.input_jsonl
        if args.input_video is not None:
            initial["input_video"] = args.input_video
    else:
        input_sqlite_kind = detect_mask_sqlite_kind(args.input_sqlite)
        if (
            args.face_mask_target != "none"
            and input_sqlite_kind != "unified_inference"
        ):
            raise ValueError(
                "--face-mask-target requires a unified inference SQLite"
            )
        if input_sqlite_kind in {"raw_detection", "unified_inference"}:
            initial["input_raw_sqlite"] = args.input_sqlite
        else:
            initial["tracked_sqlite"] = args.input_sqlite
        if args.input_video is not None:
            initial["input_video"] = args.input_video
    if args.class_policy_json is not None:
        initial["class_policy_json"] = args.class_policy_json
    if args.precomputed_cuts_json is not None:
        cuts = args.precomputed_cuts_json.expanduser().resolve()
        if not cuts.is_file():
            raise FileNotFoundError(cuts)
        initial["cuts_json"] = cuts
    return PipelineRunner(config, args.output_dir).run(initial)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pipeline(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
