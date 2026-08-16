"""Single end-to-end entrypoint for modular mask post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from common.config import (
    PipelineConfig,
    StageSpec,
    default_polygon_pipeline,
    load_pipeline_config,
)
from common.runner import PipelineRunner
from common.result_metadata import record_result_processing_run
from contracts.detector_sqlite import detect_mask_sqlite_kind


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
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--class-policy-json", type=Path)
    parser.add_argument(
        "--class-postprocess-policy-json",
        type=Path,
        help=(
            "route each tracked class through its configured Production "
            "polygon keyframe interval"
        ),
    )
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
    parser.add_argument(
        "--face-detection-score-threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--head-detection-score-threshold",
        type=float,
        default=0.55,
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


def _polygon_stage_options(
    args: argparse.Namespace,
    initial: dict[str, object] | None = None,
) -> dict[str, object]:
    options = {} if initial is None else dict(initial)
    # Production polygon geometry and its exact CPU evaluator are frozen.
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
    else:
        source = default_polygon_pipeline(include_preprocess=include_raw_stages)

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
        elif stage.implementation == "production.polygon_v3_cpu":
            if args.keyframe_interval is not None:
                options["target_interval"] = int(args.keyframe_interval)
            options = _polygon_stage_options(args, options)
            # The promoted profile intentionally ignores the inference device:
            # interval evaluation is exact native CPU by contract.
            options["interval_evaluation"] = "native_exact"
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
            "nms.production_v3",
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
                    "default_keyframe_interval": (
                        6
                        if args.keyframe_interval is None
                        else int(args.keyframe_interval)
                    ),
                    "polygon_options": _polygon_stage_options(args),
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
    if not 0.0 <= args.face_detection_score_threshold <= 1.0:
        raise ValueError("--face-detection-score-threshold must be between 0 and 1")
    if not 0.0 <= args.head_detection_score_threshold <= 1.0:
        raise ValueError("--head-detection-score-threshold must be between 0 and 1")
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
                        "face_detection_score_threshold": (
                            args.face_detection_score_threshold
                        ),
                        "head_detection_score_threshold": (
                            args.head_detection_score_threshold
                        ),
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
