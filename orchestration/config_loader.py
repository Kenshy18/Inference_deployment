"""JSON parser for :class:`OrchestrationConfig`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypeVar

from .config_sections import (
    ExecutionConfig,
    InferenceConfig,
    OverlayConfig,
    PostprocessConfig,
)
from .config_support import (
    OrchestrationConfigError,
    object_value as _object,
    optional_float as _optional_float,
    optional_int as _optional_int,
    reject_unknown as _reject_unknown,
    resolve_path as _resolve_path,
    string_tuple as _string_tuple,
)

ConfigT = TypeVar("ConfigT")


def load_config(config_type: type[ConfigT], path: Path) -> ConfigT:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OrchestrationConfigError(
            f"{config_path}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise OrchestrationConfigError("configuration root must be an object")
    allowed_root = {
        "schema_version",
        "input_video",
        "output_root",
        "execution",
        "inference",
        "postprocess",
        "overlay",
    }
    _reject_unknown(dict(raw), allowed_root, "configuration")
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise OrchestrationConfigError(
            f"unsupported schema_version={schema_version}"
        )
    base = config_path.parent
    input_video = _resolve_path(
        raw.get("input_video"),
        base=base,
        field="input_video",
        required=True,
    )
    output_root = _resolve_path(
        raw.get("output_root"),
        base=base,
        field="output_root",
        required=True,
    )
    assert input_video is not None and output_root is not None

    execution_raw = _object(raw.get("execution"), "execution")
    _reject_unknown(
        execution_raw,
        {"runtime_python", "resume"},
        "execution",
    )
    runtime_python = _resolve_path(
        execution_raw.get("runtime_python", sys.executable),
        base=base,
        field="execution.runtime_python",
        required=True,
    )
    assert runtime_python is not None
    execution = ExecutionConfig(
        runtime_python=runtime_python,
        resume=bool(execution_raw.get("resume", False)),
    )

    inference_raw = _object(raw.get("inference"), "inference")
    inference_allowed = {
        "enabled",
        "input_sqlite",
        "mode",
        "segmentation_model",
        "segmentation_backend",
        "face_model",
        "face_backend",
        "face_classes",
        "face_trt_bundle",
        "device",
        "max_frames",
        "warmup_frames",
        "face_warmup_iterations",
        "parallel_models",
        "parallel_model_stagger_seconds",
        "fast_sqlite",
        "extra_args",
    }
    _reject_unknown(inference_raw, inference_allowed, "inference")
    inference_enabled = bool(inference_raw.get("enabled", True))
    face_classes_value = inference_raw.get("face_classes", ["Face", "Head"])
    inference = InferenceConfig(
        enabled=inference_enabled,
        input_sqlite=_resolve_path(
            inference_raw.get("input_sqlite"),
            base=base,
            field="inference.input_sqlite",
        ),
        mode=str(inference_raw.get("mode", "segmentation")),
        segmentation_model=(
            None
            if inference_raw.get("segmentation_model") in (None, "")
            else str(inference_raw["segmentation_model"])
        ),
        segmentation_backend=str(inference_raw.get("segmentation_backend", "auto")),
        face_model=str(inference_raw.get("face_model", "rtdetr_head_face")),
        face_backend=str(inference_raw.get("face_backend", "auto")),
        face_classes=_string_tuple(
            face_classes_value,
            "inference.face_classes",
        ),
        face_trt_bundle=_resolve_path(
            inference_raw.get("face_trt_bundle"),
            base=base,
            field="inference.face_trt_bundle",
        ),
        device=str(inference_raw.get("device", "cuda:0")),
        max_frames=_optional_int(
            inference_raw.get("max_frames"),
            "inference.max_frames",
        ),
        warmup_frames=int(inference_raw.get("warmup_frames", 0)),
        face_warmup_iterations=int(inference_raw.get("face_warmup_iterations", 3)),
        parallel_models=bool(inference_raw.get("parallel_models", False)),
        parallel_model_stagger_seconds=float(
            inference_raw.get("parallel_model_stagger_seconds", 0.0)
        ),
        fast_sqlite=bool(inference_raw.get("fast_sqlite", False)),
        extra_args=_string_tuple(
            inference_raw.get("extra_args"),
            "inference.extra_args",
        ),
    )

    postprocess_raw = _object(raw.get("postprocess"), "postprocess")
    postprocess_allowed = {
        "enabled",
        "tracked_sqlite",
        "final_sqlite",
        "export_legacy_sqlite",
        "pipeline_config",
        "class_policy_json",
        "class_postprocess_policy_json",
        "score_min",
        "cut_detect",
        "cut_method",
        "precompute_cuts_during_inference",
        "remove_short_tracks_max_frames",
        "keyframe_interval",
        "mask_geometry",
        "extra_args",
        "face_mask_target",
        "eye_mask_shape",
        "minimum_eye_confidence",
        "face_detection_score_threshold",
        "head_detection_score_threshold",
        "face_tracking_max_gap_frames",
        "face_tracking_high_score_threshold",
        "face_tracking_low_score_threshold",
        "face_short_track_max_hits",
        "face_short_track_keep_score",
        "face_interpolation_max_gap",
    }
    _reject_unknown(postprocess_raw, postprocess_allowed, "postprocess")
    postprocess_enabled = bool(
        postprocess_raw.get("enabled", inference.uses_segmentation)
    )
    postprocess = PostprocessConfig(
        enabled=postprocess_enabled,
        tracked_sqlite=_resolve_path(
            postprocess_raw.get("tracked_sqlite"),
            base=base,
            field="postprocess.tracked_sqlite",
        ),
        final_sqlite=_resolve_path(
            postprocess_raw.get("final_sqlite"),
            base=base,
            field="postprocess.final_sqlite",
        ),
        export_legacy_sqlite=bool(
            postprocess_raw.get("export_legacy_sqlite", False)
        ),
        pipeline_config=_resolve_path(
            postprocess_raw.get("pipeline_config"),
            base=base,
            field="postprocess.pipeline_config",
        ),
        class_policy_json=_resolve_path(
            postprocess_raw.get("class_policy_json"),
            base=base,
            field="postprocess.class_policy_json",
        ),
        class_postprocess_policy_json=_resolve_path(
            postprocess_raw.get("class_postprocess_policy_json"),
            base=base,
            field="postprocess.class_postprocess_policy_json",
        ),
        score_min=_optional_float(
            postprocess_raw.get("score_min"),
            "postprocess.score_min",
        ),
        cut_detect=bool(postprocess_raw.get("cut_detect", True)),
        cut_method=(
            None
            if postprocess_raw.get("cut_method") in (None, "")
            else str(postprocess_raw["cut_method"])
        ),
        precompute_cuts_during_inference=bool(
            postprocess_raw.get(
                "precompute_cuts_during_inference",
                False,
            )
        ),
        remove_short_tracks_max_frames=_optional_int(
            postprocess_raw.get("remove_short_tracks_max_frames"),
            "postprocess.remove_short_tracks_max_frames",
        ),
        keyframe_interval=_optional_int(
            postprocess_raw.get("keyframe_interval"),
            "postprocess.keyframe_interval",
        ),
        mask_geometry=str(postprocess_raw.get("mask_geometry", "polygon")),
        extra_args=_string_tuple(
            postprocess_raw.get("extra_args"),
            "postprocess.extra_args",
        ),
        face_mask_target=str(postprocess_raw.get("face_mask_target", "none")),
        eye_mask_shape=str(postprocess_raw.get("eye_mask_shape", "ellipse")),
        minimum_eye_confidence=float(
            postprocess_raw.get("minimum_eye_confidence", 0.35)
        ),
        face_detection_score_threshold=float(
            postprocess_raw.get("face_detection_score_threshold", 0.55)
        ),
        head_detection_score_threshold=float(
            postprocess_raw.get("head_detection_score_threshold", 0.55)
        ),
        face_tracking_max_gap_frames=int(
            postprocess_raw.get("face_tracking_max_gap_frames", 5)
        ),
        face_tracking_high_score_threshold=float(
            postprocess_raw.get(
                "face_tracking_high_score_threshold",
                0.50,
            )
        ),
        face_tracking_low_score_threshold=float(
            postprocess_raw.get(
                "face_tracking_low_score_threshold",
                0.05,
            )
        ),
        face_short_track_max_hits=int(
            postprocess_raw.get("face_short_track_max_hits", 2)
        ),
        face_short_track_keep_score=float(
            postprocess_raw.get("face_short_track_keep_score", 0.90)
        ),
        face_interpolation_max_gap=int(
            postprocess_raw.get("face_interpolation_max_gap", 3)
        ),
    )

    overlay_raw = _object(raw.get("overlay"), "overlay")
    overlay_allowed = {
        "enabled",
        "execution_mode",
        "backend",
        "raw",
        "tracked",
        "final",
        "faces",
        "final_include_faces",
        "presets",
        "genital_source",
        "face_mask_target",
        "eye_mask_shape",
        "minimum_eye_confidence",
        "face_probability_masks",
        "face_keypoints",
        "face_ellipses",
        "mask_alpha",
        "outline_thickness",
        "box_thickness",
        "show_labels",
        "codec",
        "h264_crf",
        "h264_preset",
        "ffmpeg_bin",
        "nvenc_cq",
        "workers",
        "cpu_workers",
        "copy_audio",
        "target_bitrate_mbps",
        "nvenc_preset",
        "nvenc_gpu",
        "faststart",
        "start_frame",
        "end_frame",
        "progress_every",
        "extra_args",
    }
    _reject_unknown(overlay_raw, overlay_allowed, "overlay")
    configured_execution_mode = overlay_raw.get("execution_mode")
    configured_backend = str(overlay_raw.get("backend", "native"))
    configured_codec = str(overlay_raw.get("codec", "h264_nvenc"))
    if (
        configured_execution_mode is None
        and "backend" not in overlay_raw
        and "codec" in overlay_raw
    ):
        configured_backend = "python_opencv"
    if configured_execution_mode is None:
        if configured_backend in {"experimental_cpp", "native"}:
            overlay_execution_mode = "fast"
        elif configured_codec.lower() in {"nvenc", "h264_nvenc"}:
            overlay_execution_mode = "nvenc"
        else:
            overlay_execution_mode = "cpu"
        overlay_backend = (
            "native" if overlay_execution_mode == "fast" else "python_opencv"
        )
        overlay_codec = configured_codec
    else:
        overlay_execution_mode = str(configured_execution_mode)
        if overlay_execution_mode == "fast_parallel":
            overlay_execution_mode = "fast"
        expected_backend = (
            "native" if overlay_execution_mode == "fast" else "python_opencv"
        )
        expected_codec = (
            "h264_nvenc" if overlay_execution_mode in {"nvenc", "fast"} else "h264"
        )
        if "backend" in overlay_raw and configured_backend not in (
            {"native", "experimental_cpp"}
            if overlay_execution_mode == "fast"
            else {expected_backend}
        ):
            raise OrchestrationConfigError(
                "overlay.backend conflicts with overlay.execution_mode"
            )
        if "codec" in overlay_raw and configured_codec.lower() != expected_codec:
            raise OrchestrationConfigError(
                "overlay.codec conflicts with overlay.execution_mode"
            )
        overlay_backend = expected_backend
        overlay_codec = expected_codec
    overlay_presets = _string_tuple(
        overlay_raw.get("presets"),
        "overlay.presets",
    )
    uses_legacy_output_selection = not overlay_presets
    overlay = OverlayConfig(
        enabled=bool(overlay_raw.get("enabled", True)),
        execution_mode=overlay_execution_mode,
        backend=overlay_backend,
        raw=bool(
            overlay_raw.get(
                "raw",
                inference.uses_segmentation and uses_legacy_output_selection,
            )
        ),
        tracked=bool(
            overlay_raw.get(
                "tracked",
                uses_legacy_output_selection
                and (postprocess.enabled or postprocess.tracked_sqlite is not None),
            )
        ),
        final=bool(
            overlay_raw.get(
                "final",
                uses_legacy_output_selection
                and (
                    postprocess.enabled
                    or postprocess.final_sqlite is not None
                    or postprocess.face_mask_target != "none"
                ),
            )
        ),
        faces=bool(
            overlay_raw.get(
                "faces",
                uses_legacy_output_selection
                and inference.uses_faces
                and not inference.uses_segmentation,
            )
        ),
        final_include_faces=bool(overlay_raw.get("final_include_faces", False)),
        presets=overlay_presets,
        genital_source=str(overlay_raw.get("genital_source", "final")),
        face_mask_target=str(overlay_raw.get("face_mask_target", "none")),
        eye_mask_shape=str(overlay_raw.get("eye_mask_shape", "ellipse")),
        minimum_eye_confidence=float(
            overlay_raw.get("minimum_eye_confidence", 0.35)
        ),
        face_probability_masks=bool(
            overlay_raw.get("face_probability_masks", True)
        ),
        face_keypoints=bool(overlay_raw.get("face_keypoints", True)),
        face_ellipses=bool(overlay_raw.get("face_ellipses", True)),
        mask_alpha=float(overlay_raw.get("mask_alpha", 0.32)),
        outline_thickness=int(overlay_raw.get("outline_thickness", 2)),
        box_thickness=int(overlay_raw.get("box_thickness", 2)),
        show_labels=bool(overlay_raw.get("show_labels", True)),
        codec=overlay_codec,
        h264_crf=int(overlay_raw.get("h264_crf", 18)),
        h264_preset=str(overlay_raw.get("h264_preset", "veryfast")),
        ffmpeg_bin=_resolve_path(
            overlay_raw.get("ffmpeg_bin"),
            base=base,
            field="overlay.ffmpeg_bin",
        ),
        nvenc_cq=int(overlay_raw.get("nvenc_cq", 18)),
        workers=int(overlay_raw.get("workers", 6)),
        cpu_workers=int(overlay_raw.get("cpu_workers", 0)),
        copy_audio=bool(overlay_raw.get("copy_audio", False)),
        target_bitrate_mbps=(
            8.0
            if overlay_execution_mode == "fast"
            and overlay_raw.get("target_bitrate_mbps") is None
            else _optional_float(
                overlay_raw.get("target_bitrate_mbps"),
                "overlay.target_bitrate_mbps",
            )
        ),
        nvenc_preset=str(
            overlay_raw.get(
                "nvenc_preset",
                "p1" if overlay_execution_mode == "fast" else "p5",
            )
        ),
        nvenc_gpu=int(overlay_raw.get("nvenc_gpu", 0)),
        faststart=bool(overlay_raw.get("faststart", False)),
        start_frame=int(overlay_raw.get("start_frame", 0)),
        end_frame=_optional_int(
            overlay_raw.get("end_frame"),
            "overlay.end_frame",
        ),
        progress_every=int(overlay_raw.get("progress_every", 300)),
        extra_args=_string_tuple(
            overlay_raw.get("extra_args"),
            "overlay.extra_args",
        ),
    )
    config = config_type(
        schema_version=schema_version,
        config_path=config_path,
        input_video=input_video,
        output_root=output_root,
        execution=execution,
        inference=inference,
        postprocess=postprocess,
        overlay=overlay,
    )
    config.validate()
    return config


__all__ = ("load_config",)
