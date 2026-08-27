"""Semantic validation for repository workflow configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_support import (
    OrchestrationConfigError,
    reject_reserved_args as _reject_reserved_args,
    validate_class_postprocess_policy as _validate_class_postprocess_policy,
)

if TYPE_CHECKING:
    from .config import OrchestrationConfig


def validate_config(config: "OrchestrationConfig") -> None:
    if not config.input_video.is_file():
        raise FileNotFoundError(f"input video not found: {config.input_video}")
    if not config.execution.runtime_python.is_file():
        raise FileNotFoundError(
            f"runtime Python not found: {config.execution.runtime_python}"
        )
    if config.inference.mode not in {
        "segmentation",
        "face",
        "segmentation-face",
    }:
        raise OrchestrationConfigError(
            f"unsupported inference.mode={config.inference.mode!r}"
        )
    _reject_reserved_args(
        config.inference.extra_args,
        {
            "--input",
            "--output",
            "--mode",
            "--segmentation-model",
            "--segmentation-backend",
            "--face-model",
            "--face-backend",
            "--face-classes",
            "--face-trt-bundle",
            "--runtime-python",
            "--device",
            "--max-frames",
            "--warmup-frames",
            "--face-warmup-iterations",
            "--parallel-models",
            "--no-parallel-models",
            "--parallel-model-stagger-seconds",
            "--overwrite",
            "--fast-sqlite",
        },
        "inference.extra_args",
    )
    if config.inference.parallel_model_stagger_seconds < 0:
        raise OrchestrationConfigError(
            "inference.parallel_model_stagger_seconds must be >= 0"
        )
    if (
        config.inference.parallel_model_stagger_seconds > 0
        and not config.inference.parallel_models
    ):
        raise OrchestrationConfigError(
            "inference.parallel_model_stagger_seconds requires "
            "inference.parallel_models=true"
        )
    if config.inference.parallel_models:
        raise OrchestrationConfigError(
            "inference.parallel_models=true is retired in Production; "
            "segmentation and face inference must run sequentially"
        )
    if config.inference.enabled:
        if config.inference.input_sqlite is not None:
            raise OrchestrationConfigError(
                "inference.input_sqlite is only valid when inference.enabled=false"
            )
        if (
            config.inference.uses_segmentation
            and not config.inference.segmentation_model
        ):
            raise OrchestrationConfigError(
                "inference.segmentation_model is required"
            )
        if not config.inference.uses_segmentation and (
            config.inference.segmentation_model is not None
        ):
            raise OrchestrationConfigError(
                "face-only inference must not set segmentation_model"
            )
        if (
            config.inference.face_trt_bundle is not None
            and not config.inference.face_trt_bundle.is_file()
        ):
            raise FileNotFoundError(
                "face TensorRT bundle not found: "
                f"{config.inference.face_trt_bundle}"
            )
    elif config.inference.input_sqlite is None:
        raise OrchestrationConfigError(
            "inference.input_sqlite is required when inference.enabled=false"
        )
    if (
        config.inference.face_trt_bundle is not None
        and config.inference.face_model != "face_dino_v2"
    ):
        raise OrchestrationConfigError(
            "inference.face_trt_bundle is currently supported only by "
            "face_dino_v2"
        )
    face_backends = {
        "face_dino_v2": {"auto", "tensorrt-fast"},
        "rtdetr_head_face": {"auto", "pytorch"},
    }
    if (
        config.inference.uses_faces
        and config.inference.face_backend
        not in face_backends.get(config.inference.face_model, set())
    ):
        raise OrchestrationConfigError(
            f"face model {config.inference.face_model!r} does not support "
            f"backend {config.inference.face_backend!r}"
        )
    if config.postprocess.enabled and not config.inference.uses_segmentation:
        raise OrchestrationConfigError(
            "postprocess requires segmentation or segmentation-face inference"
        )
    if config.postprocess.precompute_cuts_during_inference:
        if not config.inference.enabled:
            raise OrchestrationConfigError(
                "postprocess.precompute_cuts_during_inference requires "
                "inference.enabled=true"
            )
        if not config.postprocess.cut_detect:
            raise OrchestrationConfigError(
                "postprocess.precompute_cuts_during_inference requires "
                "postprocess.cut_detect=true"
            )
        if config.postprocess.cut_method not in {None, "high_precision"}:
            raise OrchestrationConfigError(
                "precomputed cut overlap currently supports only "
                "cut_method=high_precision"
            )
    if (
        config.postprocess.pipeline_config is not None
        and config.postprocess.class_postprocess_policy_json is not None
    ):
        raise OrchestrationConfigError(
            "postprocess.pipeline_config and "
            "postprocess.class_postprocess_policy_json cannot be combined"
        )
    for field, path in (
        ("postprocess.pipeline_config", config.postprocess.pipeline_config),
        ("postprocess.class_policy_json", config.postprocess.class_policy_json),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{field} not found: {path}")
    if (
        config.postprocess.class_postprocess_policy_json is not None
        and not config.postprocess.class_postprocess_policy_json.is_file()
    ):
        raise FileNotFoundError(
            "class postprocess policy not found: "
            f"{config.postprocess.class_postprocess_policy_json}"
        )
    if config.postprocess.class_postprocess_policy_json is not None:
        _validate_class_postprocess_policy(
            config.postprocess.class_postprocess_policy_json
        )
    if (
        config.postprocess.keyframe_interval is not None
        and config.postprocess.keyframe_interval < 1
    ):
        raise OrchestrationConfigError(
            "postprocess.keyframe_interval must be at least 1"
        )
    if config.postprocess.mask_geometry not in {"polygon", "catmull_rom"}:
        raise OrchestrationConfigError(
            "postprocess.mask_geometry must be polygon or catmull_rom"
        )
    if (
        config.postprocess.pipeline_config is not None
        and config.postprocess.mask_geometry != "polygon"
    ):
        raise OrchestrationConfigError(
            "postprocess.mask_geometry is selected by a custom pipeline_config; "
            "the global option must remain polygon"
        )
    if config.postprocess.face_mask_target not in {"none", "face", "eyes"}:
        raise OrchestrationConfigError(
            "postprocess.face_mask_target must be none, face, or eyes"
        )
    if config.postprocess.eye_mask_shape not in {"ellipse", "rectangle"}:
        raise OrchestrationConfigError(
            "postprocess.eye_mask_shape must be ellipse or rectangle"
        )
    if not 0.0 <= config.postprocess.minimum_eye_confidence <= 1.0:
        raise OrchestrationConfigError(
            "postprocess.minimum_eye_confidence must be between 0 and 1"
        )
    if not 0.0 <= config.postprocess.face_detection_score_threshold <= 1.0:
        raise OrchestrationConfigError(
            "postprocess.face_detection_score_threshold must be between 0 and 1"
        )
    if not 0.0 <= config.postprocess.head_detection_score_threshold <= 1.0:
        raise OrchestrationConfigError(
            "postprocess.head_detection_score_threshold must be between 0 and 1"
        )
    if config.postprocess.face_tracking_max_gap_frames < 0:
        raise OrchestrationConfigError(
            "postprocess.face_tracking_max_gap_frames must be non-negative"
        )
    if config.postprocess.face_interpolation_max_gap < 0:
        raise OrchestrationConfigError(
            "postprocess.face_interpolation_max_gap must be non-negative"
        )
    if config.postprocess.face_short_track_max_hits < 0:
        raise OrchestrationConfigError(
            "postprocess.face_short_track_max_hits must be non-negative"
        )
    if not (
        0.0
        <= config.postprocess.face_tracking_low_score_threshold
        <= config.postprocess.face_tracking_high_score_threshold
        <= 1.0
    ):
        raise OrchestrationConfigError(
            "face tracking scores must satisfy 0 <= low <= high <= 1"
        )
    if not 0.0 <= config.postprocess.face_short_track_keep_score <= 1.0:
        raise OrchestrationConfigError(
            "postprocess.face_short_track_keep_score must be between 0 and 1"
        )
    if config.postprocess.face_mask_target != "none":
        if not config.inference.uses_faces:
            raise OrchestrationConfigError(
                "face mask postprocess requires face inference"
            )
        if config.inference.face_model != "face_dino_v2":
            raise OrchestrationConfigError(
                "face mask postprocess currently requires face_dino_v2"
            )
    _reject_reserved_args(
        config.postprocess.extra_args,
        {
            "--input-jsonl",
            "--input-sqlite",
            "--input-video",
            "--output-dir",
            "--orchestration-config-json",
            "--shape-mode",
            "--pipeline-config",
            "--class-policy-json",
            "--class-postprocess-policy-json",
            "--score-min",
            "--cut-detect",
            "--no-cut-detect",
            "--cut-method",
            "--precomputed-cuts-json",
            "--remove-short-tracks-max-frames",
            "--keyframe-interval",
            "--mask-geometry",
            "--max-gap",
            "--model-root",
            "--k2-run-dir",
            "--device",
            "--export-legacy-sqlite",
            "--no-export-legacy-sqlite",
            "--export-dinov3-legacy-sqlite",
            "--no-export-dinov3-legacy-sqlite",
            "--face-mask-target",
            "--eye-mask-shape",
            "--minimum-eye-confidence",
            "--face-detection-score-threshold",
            "--head-detection-score-threshold",
            "--face-tracking-max-gap-frames",
            "--face-tracking-high-score-threshold",
            "--face-tracking-low-score-threshold",
            "--face-short-track-max-hits",
            "--face-short-track-keep-score",
            "--face-interpolation-max-gap",
        },
        "postprocess.extra_args",
    )
    if not config.postprocess.enabled:
        if config.postprocess.export_legacy_sqlite:
            raise OrchestrationConfigError(
                "postprocess.export_legacy_sqlite requires "
                "postprocess.enabled=true"
            )
        if config.overlay.enabled and config.overlay.tracked:
            if config.postprocess.tracked_sqlite is None:
                raise OrchestrationConfigError(
                    "tracked overlay requires postprocess.tracked_sqlite "
                    "when postprocess is disabled"
                )
        if config.overlay.enabled and config.overlay.final:
            if (
                config.postprocess.final_sqlite is None
                and config.postprocess.face_mask_target == "none"
            ):
                raise OrchestrationConfigError(
                    "final overlay requires postprocess.final_sqlite "
                    "or face_mask_target when postprocess is disabled"
                )
    faces_requested = config.overlay.enabled and (
        config.overlay.faces or config.overlay.final_include_faces
    )
    if faces_requested and not config.inference.uses_faces:
        raise OrchestrationConfigError(
            "face overlay requires inference.mode=face or segmentation-face"
        )
    if config.overlay.final_include_faces and not config.overlay.final:
        raise OrchestrationConfigError(
            "overlay.final_include_faces requires overlay.final=true"
        )
    if config.overlay.start_frame < 0:
        raise OrchestrationConfigError("overlay.start_frame must be >= 0")
    if (
        config.overlay.end_frame is not None
        and config.overlay.end_frame < config.overlay.start_frame
    ):
        raise OrchestrationConfigError(
            "overlay.end_frame must be >= overlay.start_frame"
        )
    if not 0.0 <= config.overlay.mask_alpha <= 1.0:
        raise OrchestrationConfigError("overlay.mask_alpha must be between 0 and 1")
    if config.overlay.outline_thickness < 1:
        raise OrchestrationConfigError(
            "overlay.outline_thickness must be at least 1"
        )
    if config.overlay.box_thickness < 1:
        raise OrchestrationConfigError("overlay.box_thickness must be at least 1")
    if config.overlay.progress_every < 0:
        raise OrchestrationConfigError(
            "overlay.progress_every must be non-negative"
        )
    if config.overlay.enabled and not (
        config.overlay.presets
        or config.overlay.raw
        or config.overlay.tracked
        or config.overlay.final
        or config.overlay.faces
    ):
        raise OrchestrationConfigError(
            "overlay.enabled=true requires at least one preset or legacy output"
        )
    if config.overlay.backend not in {
        "python_opencv",
        "native",
    }:
        raise OrchestrationConfigError(
            "overlay.backend must be python_opencv or native"
        )
    if config.overlay.execution_mode not in {
        "cpu",
        "nvenc",
        "fast",
    }:
        raise OrchestrationConfigError(
            "overlay.execution_mode must be cpu, nvenc, or fast"
        )
    expected_backend = (
        "native" if config.overlay.execution_mode == "fast" else "python_opencv"
    )
    if config.overlay.backend != expected_backend:
        raise OrchestrationConfigError(
            "overlay.backend does not match overlay.execution_mode"
        )
    if config.overlay.execution_mode == "cpu" and config.overlay.uses_nvenc:
        raise OrchestrationConfigError(
            "overlay.execution_mode=cpu cannot use an NVENC codec"
        )
    if (
        config.overlay.execution_mode in {"nvenc", "fast"}
        and not config.overlay.uses_nvenc
    ):
        raise OrchestrationConfigError(
            f"overlay.execution_mode={config.overlay.execution_mode} "
            "requires codec=h264_nvenc"
        )
    if config.overlay.workers < 1:
        raise OrchestrationConfigError("overlay.workers must be at least 1")
    if not 0 <= config.overlay.cpu_workers <= config.overlay.workers:
        raise OrchestrationConfigError(
            "overlay.cpu_workers must be between 0 and overlay.workers"
        )
    if config.overlay.target_bitrate_mbps is not None and (
        config.overlay.target_bitrate_mbps <= 0
    ):
        raise OrchestrationConfigError(
            "overlay.target_bitrate_mbps must be positive"
        )
    if config.overlay.nvenc_gpu < 0:
        raise OrchestrationConfigError("overlay.nvenc_gpu must be non-negative")
    if not 0 <= config.overlay.h264_crf <= 51:
        raise OrchestrationConfigError("overlay.h264_crf must be between 0 and 51")
    if (
        config.overlay.ffmpeg_bin is not None
        and not config.overlay.ffmpeg_bin.is_file()
    ):
        raise FileNotFoundError(
            f"overlay FFmpeg executable not found: {config.overlay.ffmpeg_bin}"
        )
    if config.overlay.h264_preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise OrchestrationConfigError(
            "overlay.h264_preset is not a supported libx264 preset"
        )
    if not 0 <= config.overlay.nvenc_cq <= 51:
        raise OrchestrationConfigError("overlay.nvenc_cq must be between 0 and 51")
    valid_presets = {
        "genital-detailed",
        "genital-simple",
        "face-detailed",
        "face-simple",
        "combined-detailed",
        "combined-simple",
    }
    invalid_presets = sorted(set(config.overlay.presets) - valid_presets)
    if invalid_presets:
        raise OrchestrationConfigError(
            f"overlay.presets contains unsupported values: {invalid_presets}"
        )
    if len(set(config.overlay.presets)) != len(config.overlay.presets):
        raise OrchestrationConfigError(
            "overlay.presets must not contain duplicates"
        )
    if config.overlay.genital_source not in {"raw", "final"}:
        raise OrchestrationConfigError(
            "overlay.genital_source must be raw or final"
        )
    if config.overlay.face_mask_target not in {"none", "face", "eyes"}:
        raise OrchestrationConfigError(
            "overlay.face_mask_target must be none, face, or eyes"
        )
    if config.overlay.eye_mask_shape not in {"ellipse", "rectangle"}:
        raise OrchestrationConfigError(
            "overlay.eye_mask_shape must be ellipse or rectangle"
        )
    if not 0.0 <= config.overlay.minimum_eye_confidence <= 1.0:
        raise OrchestrationConfigError(
            "overlay.minimum_eye_confidence must be between 0 and 1"
        )
    face_preset_requested = any(
        preset.startswith(("face-", "combined-")) for preset in config.overlay.presets
    )
    if config.overlay.face_mask_target != "none":
        if not config.inference.uses_faces:
            raise OrchestrationConfigError(
                "overlay face privacy mask requires face inference"
            )
        if config.inference.face_model != "face_dino_v2":
            raise OrchestrationConfigError(
                "overlay face privacy mask requires face_dino_v2"
            )
        if not (
            face_preset_requested
            or config.overlay.faces
            or (config.overlay.final and config.overlay.final_include_faces)
        ):
            raise OrchestrationConfigError(
                "overlay face privacy mask requires a face or combined output"
            )
    if face_preset_requested and not config.inference.uses_faces:
        raise OrchestrationConfigError(
            "face/combined overlay presets require face inference"
        )
    if (
        any(
            preset.startswith(("genital-", "combined-"))
            for preset in config.overlay.presets
        )
        and not config.inference.uses_segmentation
    ):
        raise OrchestrationConfigError(
            "genital/combined overlay presets require segmentation inference"
        )
    if config.overlay.nvenc_preset not in {
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
    }:
        raise OrchestrationConfigError(
            "overlay.nvenc_preset must be between p1 and p7"
        )
    if config.overlay.execution_mode == "fast":
        if not config.overlay.uses_nvenc:
            raise OrchestrationConfigError(
                "fast overlay currently requires codec=h264_nvenc"
            )
        if config.overlay.target_bitrate_mbps is None:
            raise OrchestrationConfigError(
                "fast overlay requires overlay.target_bitrate_mbps"
            )
    elif config.overlay.copy_audio:
        raise OrchestrationConfigError(
            "overlay.copy_audio requires " "execution_mode=fast"
        )
    elif config.overlay.cpu_workers != 0:
        raise OrchestrationConfigError(
            "overlay.cpu_workers is only used by " "execution_mode=fast"
        )
    if len(config.overlay.codec) != 4 and not config.overlay.uses_nvenc:
        raise OrchestrationConfigError(
            "overlay.codec must be a four-character FourCC or h264_nvenc"
        )
    _reject_reserved_args(
        config.overlay.extra_args,
        {
            "--execution-mode",
            "--mode",
            "--overlay-type",
            "--video",
            "--sqlite",
            "--output",
            "--manifest",
            "--include-faces",
            "--face-sqlite",
            "--preset",
            "--genital-source",
            "--mask-alpha",
            "--outline-thickness",
            "--box-thickness",
            "--no-labels",
            "--no-face-probability-masks",
            "--no-face-keypoints",
            "--no-face-ellipses",
            "--face-mask-target",
            "--eye-mask-shape",
            "--minimum-eye-confidence",
            "--codec",
            "--h264-crf",
            "--h264-preset",
            "--nvenc-cq",
            "--target-bitrate-mbps",
            "--start-frame",
            "--end-frame",
            "--progress-every",
            "--overwrite",
            "--renderer",
            "--ffmpeg-bin",
            "--output-dir",
            "--workers",
            "--cpu-workers",
            "--bitrate-mbps",
            "--cpu-preset",
            "--nvenc-preset",
            "--nvenc-gpu",
            "--cpu-weight",
            "--nvenc-weight",
            "--decoder-threads",
            "--hw-decode",
            "--gpu-pipeline",
            "--copy-audio",
            "--faststart",
            "--compact-output",
        },
        "overlay.extra_args",
    )


__all__ = ("validate_config",)
