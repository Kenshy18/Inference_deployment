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
from common.result_metadata import record_result_processing_run
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
        "--orchestration-config-json",
        type=Path,
        help="resolved orchestrator configuration embedded into result.sqlite",
    )
    parser.add_argument(
        "--shape-mode", choices=("ellipse", "polygon"), default="ellipse"
    )
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--class-policy-json", type=Path)
    parser.add_argument(
        "--class-postprocess-policy-json",
        type=Path,
        help=(
            "route each tracked class through its configured shape, "
            "keyframe interval, and missing-frame gap limit"
        ),
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        help="explicitly override the selected pipeline stage",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        help=(
            "maximum missing-frame run to fill; class policy values override "
            "this fallback"
        ),
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
    parser.add_argument("--face-tracking-max-gap-frames", type=int, default=5)
    parser.add_argument(
        "--face-tracking-high-score-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--face-tracking-low-score-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--face-short-track-max-hits", type=int, default=2)
    parser.add_argument(
        "--face-short-track-keep-score",
        type=float,
        default=0.90,
    )
    parser.add_argument("--face-interpolation-max-gap", type=int, default=3)
    return parser


def _ellipse_stage_options(
    args: argparse.Namespace,
    source_options: dict[str, object] | None = None,
    *,
    resolve_auto_device: bool = True,
) -> dict[str, object]:
    options = dict(source_options or {})
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
        options["device"] = (
            choose_device(args.device) if resolve_auto_device else args.device
        )
    else:
        options.setdefault(
            "device",
            choose_device("auto") if resolve_auto_device else "auto",
        )
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
    return options


def _configured_pipeline(args: argparse.Namespace) -> PipelineConfig:
    if (
        args.pipeline_config is not None
        and args.class_postprocess_policy_json is not None
    ):
        raise ValueError(
            "--pipeline-config and --class-postprocess-policy-json "
            "cannot be combined"
        )
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
    if args.max_gap is not None and args.max_gap < 0:
        raise ValueError("--max-gap must be >= 0")
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
            options = _ellipse_stage_options(args, options)
        elif (
            stage.implementation == "keyframes.ellipse.dense"
            and args.keyframe_interval is not None
        ):
            options["target_ratio"] = 1.0 / float(args.keyframe_interval)
        elif (
            stage.implementation
            in {"gap_fill.polygon.linear", "gap_fill.ellipse.linear"}
            and args.max_gap is not None
        ):
            options["max_gap"] = int(args.max_gap)
        stages.append(
            StageSpec(
                stage.id,
                stage.implementation,
                options,
                stage.enabled,
            )
        )
    if args.class_postprocess_policy_json is not None:
        upstream_implementations = {
            "preprocessing.normalize",
            "preprocessing.raw_sqlite",
            "preprocessing.score_policy",
            "nms.adaptive",
            "cut_detection.video",
            "tracking.greedy",
        }
        upstream = [
            stage
            for stage in stages
            if stage.implementation in upstream_implementations
        ]
        upstream.append(
            StageSpec(
                "classwise_postprocess",
                "classwise.production",
                {
                    "default_shape_mode": args.shape_mode,
                    "default_keyframe_interval": (
                        3
                        if args.keyframe_interval is None
                        else int(args.keyframe_interval)
                    ),
                    "default_max_gap": args.max_gap,
                    "ellipse_options": _ellipse_stage_options(
                        args,
                        resolve_auto_device=False,
                    ),
                    "polygon_options": {},
                },
            )
        )
        upstream.append(
            StageSpec(
                "output_validation",
                "artifacts.validate",
            )
        )
        stages = upstream
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
                        "tracking_max_gap_frames": (args.face_tracking_max_gap_frames),
                        "tracking_high_score_threshold": (
                            args.face_tracking_high_score_threshold
                        ),
                        "tracking_low_score_threshold": (
                            args.face_tracking_low_score_threshold
                        ),
                        "short_track_max_hits": args.face_short_track_max_hits,
                        "short_track_keep_score": (args.face_short_track_keep_score),
                        "interpolation_max_gap": (args.face_interpolation_max_gap),
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
                        "output_artifact": ("combined_legacy_predictions_sqlite"),
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
    if input_sqlite_kind == "unified_inference":
        stages.append(
            StageSpec(
                "integrated_result_sqlite",
                "artifacts.integrated_sqlite",
                {
                    "source_artifact": (
                        "combined_predictions_sqlite"
                        if args.face_mask_target != "none"
                        else "predictions_sqlite"
                    )
                },
            )
        )
    pipeline_name = (
        f"{source.name}_classwise"
        if args.class_postprocess_policy_json is not None
        else source.name
    )
    return PipelineConfig(pipeline_name, tuple(stages))


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    from common.live_preview import (
        activate_postprocess_preview,
        close_postprocess_preview,
    )

    config = _configured_pipeline(args)
    initial: dict[str, Path] = {}
    if args.input_jsonl is not None:
        if args.face_mask_target != "none":
            raise ValueError("--face-mask-target requires a unified inference SQLite")
        initial["input_jsonl"] = args.input_jsonl
        if args.input_video is not None:
            initial["input_video"] = args.input_video
    else:
        input_sqlite_kind = detect_mask_sqlite_kind(args.input_sqlite)
        if args.face_mask_target != "none" and input_sqlite_kind != "unified_inference":
            raise ValueError("--face-mask-target requires a unified inference SQLite")
        if input_sqlite_kind in {"raw_detection", "unified_inference"}:
            initial["input_raw_sqlite"] = args.input_sqlite
        else:
            initial["tracked_sqlite"] = args.input_sqlite
        if args.input_video is not None:
            initial["input_video"] = args.input_video
    if args.class_policy_json is not None:
        initial["class_policy_json"] = args.class_policy_json
    if args.class_postprocess_policy_json is not None:
        initial["class_postprocess_policy_json"] = args.class_postprocess_policy_json
    if args.precomputed_cuts_json is not None:
        cuts = args.precomputed_cuts_json.expanduser().resolve()
        if not cuts.is_file():
            raise FileNotFoundError(cuts)
        initial["cuts_json"] = cuts
    activate_postprocess_preview(args.input_video)
    try:
        manifest = PipelineRunner(
            config,
            args.output_dir,
            emit_progress=True,
        ).run(initial)
    finally:
        close_postprocess_preview()
    result_value = manifest.get("artifacts", {}).get("result_sqlite")
    if result_value:
        specs = {stage.id: stage for stage in config.stages if stage.enabled}
        stage_rows: list[dict[str, object]] = []
        for stage in manifest.get("stages", []):
            stage_id = str(stage["id"])
            spec = specs[stage_id]
            stage_rows.append(
                {
                    "id": stage_id,
                    "implementation": str(stage["implementation"]),
                    "options": dict(spec.options),
                    "device": spec.options.get("device"),
                    "elapsed_seconds": float(stage["elapsed_seconds"]),
                    "status": "complete",
                }
            )
        record_result_processing_run(
            Path(str(result_value)),
            kind="postprocess",
            name=config.name,
            resolved_config={
                "arguments": vars(args),
                "pipeline": {
                    "name": config.name,
                    "stages": [
                        {
                            "id": stage.id,
                            "implementation": stage.implementation,
                            "options": stage.options,
                            "enabled": stage.enabled,
                        }
                        for stage in config.stages
                    ],
                },
            },
            stages=stage_rows,
        )
        if args.orchestration_config_json is not None:
            orchestration_config = json.loads(
                args.orchestration_config_json.read_text(encoding="utf-8")
            )
            record_result_processing_run(
                Path(str(result_value)),
                kind="orchestration",
                name="inference-postprocess-overlay",
                resolved_config=orchestration_config,
                stages=[],
            )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pipeline(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
