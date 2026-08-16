"""Typed JSON configuration for the repository-level workflow."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class OrchestrationConfigError(ValueError):
    """Raised when a workflow configuration is inconsistent."""


def _object(value: object, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OrchestrationConfigError(f"{section} must be a JSON object")
    return dict(value)


def _reject_unknown(
    values: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise OrchestrationConfigError(
            f"{section} has unknown option(s): {sorted(unknown)}"
        )


def _resolve_path(
    value: object | None,
    *,
    base: Path,
    field: str,
    required: bool = False,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise OrchestrationConfigError(f"{field} is required")
        return None
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _optional_int(value: object | None, field: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field} must be an integer") from exc


def _optional_float(value: object | None, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field} must be a number") from exc


def _string_tuple(value: object | None, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise OrchestrationConfigError(f"{field} must be a list of strings")
    return tuple(str(item) for item in value)


def _reject_reserved_args(
    values: tuple[str, ...],
    reserved: set[str],
    field: str,
) -> None:
    conflicts = sorted({value for value in values if value in reserved})
    if conflicts:
        raise OrchestrationConfigError(
            f"{field} must not override managed option(s): {conflicts}"
        )


def _validate_class_postprocess_policy(path: Path) -> None:
    field = "postprocess.class_postprocess_policy_json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationConfigError(f"{field} must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise OrchestrationConfigError(f"{field} root must be a JSON object")
    _reject_unknown(
        raw,
        {"schema_version", "default", "classes"},
        field,
    )
    try:
        schema_version = int(raw.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise OrchestrationConfigError(f"{field}.schema_version must be 1") from exc
    if schema_version not in {1, 2}:
        raise OrchestrationConfigError(f"{field}.schema_version must be 1 or 2")
    classes = raw.get("classes", {})
    if not isinstance(classes, dict):
        raise OrchestrationConfigError(f"{field}.classes must be a JSON object")
    sections: list[tuple[str, object]] = [(f"{field}.default", raw.get("default"))]
    sections.extend(
        (f"{field}.classes.{label}", value) for label, value in classes.items()
    )
    for section, value in sections:
        if value is None:
            continue
        if not isinstance(value, dict):
            raise OrchestrationConfigError(f"{section} must be a JSON object")
        allowed = (
            {"shape_mode", "keyframe_interval", "max_gap"}
            if schema_version == 1
            else {"keyframe_interval"}
        )
        _reject_unknown(value, allowed, section)
        if "shape_mode" in value and value["shape_mode"] != "polygon":
            raise OrchestrationConfigError(f"{section}.shape_mode must be polygon")
        if "keyframe_interval" in value:
            interval = _optional_int(
                value["keyframe_interval"],
                f"{section}.keyframe_interval",
            )
            if interval is None or interval < 1:
                raise OrchestrationConfigError(
                    f"{section}.keyframe_interval must be at least 1"
                )
        if "max_gap" in value:
            max_gap = _optional_int(value["max_gap"], f"{section}.max_gap")
            if max_gap != 15:
                raise OrchestrationConfigError(f"{section}.max_gap is fixed at 15")
    if any(not str(label).strip() for label in classes):
        raise OrchestrationConfigError(f"{field}.classes labels must not be empty")


@dataclass(frozen=True)
class ExecutionConfig:
    runtime_python: Path
    resume: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    enabled: bool = True
    input_sqlite: Path | None = None
    mode: str = "segmentation"
    segmentation_model: str | None = None
    segmentation_backend: str = "auto"
    face_model: str = "rtdetr_head_face"
    face_backend: str = "auto"
    face_classes: tuple[str, ...] = ("Face", "Head")
    face_trt_bundle: Path | None = None
    device: str = "cuda:0"
    max_frames: int | None = None
    warmup_frames: int = 0
    face_warmup_iterations: int = 3
    parallel_models: bool = False
    parallel_model_stagger_seconds: float = 0.0
    fast_sqlite: bool = False
    extra_args: tuple[str, ...] = ()

    @property
    def uses_segmentation(self) -> bool:
        return self.mode in {"segmentation", "segmentation-face"}

    @property
    def uses_faces(self) -> bool:
        return self.mode in {"face", "segmentation-face"}


@dataclass(frozen=True)
class PostprocessConfig:
    enabled: bool = True
    tracked_sqlite: Path | None = None
    final_sqlite: Path | None = None
    pipeline_config: Path | None = None
    class_policy_json: Path | None = None
    class_postprocess_policy_json: Path | None = None
    score_min: float | None = None
    cut_detect: bool = True
    cut_method: str | None = None
    precompute_cuts_during_inference: bool = False
    remove_short_tracks_max_frames: int | None = None
    keyframe_interval: int | None = None
    extra_args: tuple[str, ...] = ()
    export_legacy_sqlite: bool = False
    face_mask_target: str = "none"
    eye_mask_shape: str = "ellipse"
    minimum_eye_confidence: float = 0.35
    face_detection_score_threshold: float = 0.55
    head_detection_score_threshold: float = 0.55
    face_tracking_max_gap_frames: int = 5
    face_tracking_high_score_threshold: float = 0.50
    face_tracking_low_score_threshold: float = 0.05
    face_short_track_max_hits: int = 2
    face_short_track_keep_score: float = 0.90
    face_interpolation_max_gap: int = 3

    @property
    def uses_gpu(self) -> bool:
        """The promoted exact optimizer is CPU-only by contract."""
        return False


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool = True
    execution_mode: str = "fast"
    backend: str = "native"
    raw: bool = True
    tracked: bool = True
    final: bool = True
    faces: bool = False
    final_include_faces: bool = False
    presets: tuple[str, ...] = ()
    genital_source: str = "final"
    face_mask_target: str = "none"
    eye_mask_shape: str = "ellipse"
    minimum_eye_confidence: float = 0.35
    face_probability_masks: bool = True
    face_keypoints: bool = True
    face_ellipses: bool = True
    mask_alpha: float = 0.32
    outline_thickness: int = 2
    box_thickness: int = 2
    show_labels: bool = True
    codec: str = "h264_nvenc"
    h264_crf: int = 18
    h264_preset: str = "veryfast"
    ffmpeg_bin: Path | None = None
    nvenc_cq: int = 18
    workers: int = 6
    cpu_workers: int = 0
    copy_audio: bool = False
    target_bitrate_mbps: float | None = 8.0
    nvenc_preset: str = "p1"
    nvenc_gpu: int = 0
    faststart: bool = False
    start_frame: int = 0
    end_frame: int | None = None
    progress_every: int = 300
    extra_args: tuple[str, ...] = ()

    @property
    def uses_nvenc(self) -> bool:
        return self.codec.lower() in {"nvenc", "h264_nvenc"}


@dataclass(frozen=True)
class OrchestrationConfig:
    schema_version: int
    config_path: Path
    input_video: Path
    output_root: Path
    execution: ExecutionConfig
    inference: InferenceConfig
    postprocess: PostprocessConfig
    overlay: OverlayConfig

    @classmethod
    def load(cls, path: Path) -> "OrchestrationConfig":
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
        config = cls(
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

    def validate(self) -> None:
        if not self.input_video.is_file():
            raise FileNotFoundError(f"input video not found: {self.input_video}")
        if not self.execution.runtime_python.is_file():
            raise FileNotFoundError(
                f"runtime Python not found: {self.execution.runtime_python}"
            )
        if self.inference.mode not in {
            "segmentation",
            "face",
            "segmentation-face",
        }:
            raise OrchestrationConfigError(
                f"unsupported inference.mode={self.inference.mode!r}"
            )
        _reject_reserved_args(
            self.inference.extra_args,
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
        if self.inference.parallel_model_stagger_seconds < 0:
            raise OrchestrationConfigError(
                "inference.parallel_model_stagger_seconds must be >= 0"
            )
        if (
            self.inference.parallel_model_stagger_seconds > 0
            and not self.inference.parallel_models
        ):
            raise OrchestrationConfigError(
                "inference.parallel_model_stagger_seconds requires "
                "inference.parallel_models=true"
            )
        if self.inference.parallel_models and (
            not self.inference.enabled
            or self.inference.mode != "segmentation-face"
            or self.inference.segmentation_model != "dinov3_codino_mh0"
            or self.inference.face_model != "face_dino_v2"
        ):
            raise OrchestrationConfigError(
                "inference.parallel_models=true is supported only when "
                "inference.enabled=true and mode=segmentation-face with "
                "segmentation_model=dinov3_codino_mh0 and "
                "face_model=face_dino_v2"
            )
        if self.inference.enabled:
            if self.inference.input_sqlite is not None:
                raise OrchestrationConfigError(
                    "inference.input_sqlite is only valid when inference.enabled=false"
                )
            if (
                self.inference.uses_segmentation
                and not self.inference.segmentation_model
            ):
                raise OrchestrationConfigError(
                    "inference.segmentation_model is required"
                )
            if not self.inference.uses_segmentation and (
                self.inference.segmentation_model is not None
            ):
                raise OrchestrationConfigError(
                    "face-only inference must not set segmentation_model"
                )
            if (
                self.inference.face_trt_bundle is not None
                and not self.inference.face_trt_bundle.is_file()
            ):
                raise FileNotFoundError(
                    "face TensorRT bundle not found: "
                    f"{self.inference.face_trt_bundle}"
                )
        elif self.inference.input_sqlite is None:
            raise OrchestrationConfigError(
                "inference.input_sqlite is required when inference.enabled=false"
            )
        if (
            self.inference.face_trt_bundle is not None
            and self.inference.face_model != "face_dino_v2"
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
            self.inference.uses_faces
            and self.inference.face_backend
            not in face_backends.get(self.inference.face_model, set())
        ):
            raise OrchestrationConfigError(
                f"face model {self.inference.face_model!r} does not support "
                f"backend {self.inference.face_backend!r}"
            )
        if self.postprocess.enabled and not self.inference.uses_segmentation:
            raise OrchestrationConfigError(
                "postprocess requires segmentation or segmentation-face inference"
            )
        if self.postprocess.precompute_cuts_during_inference:
            if not self.inference.enabled:
                raise OrchestrationConfigError(
                    "postprocess.precompute_cuts_during_inference requires "
                    "inference.enabled=true"
                )
            if not self.postprocess.cut_detect:
                raise OrchestrationConfigError(
                    "postprocess.precompute_cuts_during_inference requires "
                    "postprocess.cut_detect=true"
                )
            if self.postprocess.cut_method not in {None, "high_precision"}:
                raise OrchestrationConfigError(
                    "precomputed cut overlap currently supports only "
                    "cut_method=high_precision"
                )
        if (
            self.postprocess.pipeline_config is not None
            and self.postprocess.class_postprocess_policy_json is not None
        ):
            raise OrchestrationConfigError(
                "postprocess.pipeline_config and "
                "postprocess.class_postprocess_policy_json cannot be combined"
            )
        for field, path in (
            ("postprocess.pipeline_config", self.postprocess.pipeline_config),
            ("postprocess.class_policy_json", self.postprocess.class_policy_json),
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{field} not found: {path}")
        if (
            self.postprocess.class_postprocess_policy_json is not None
            and not self.postprocess.class_postprocess_policy_json.is_file()
        ):
            raise FileNotFoundError(
                "class postprocess policy not found: "
                f"{self.postprocess.class_postprocess_policy_json}"
            )
        if self.postprocess.class_postprocess_policy_json is not None:
            _validate_class_postprocess_policy(
                self.postprocess.class_postprocess_policy_json
            )
        if (
            self.postprocess.keyframe_interval is not None
            and self.postprocess.keyframe_interval < 1
        ):
            raise OrchestrationConfigError(
                "postprocess.keyframe_interval must be at least 1"
            )
        if self.postprocess.face_mask_target not in {"none", "face", "eyes"}:
            raise OrchestrationConfigError(
                "postprocess.face_mask_target must be none, face, or eyes"
            )
        if self.postprocess.eye_mask_shape not in {"ellipse", "rectangle"}:
            raise OrchestrationConfigError(
                "postprocess.eye_mask_shape must be ellipse or rectangle"
            )
        if not 0.0 <= self.postprocess.minimum_eye_confidence <= 1.0:
            raise OrchestrationConfigError(
                "postprocess.minimum_eye_confidence must be between 0 and 1"
            )
        if not 0.0 <= self.postprocess.face_detection_score_threshold <= 1.0:
            raise OrchestrationConfigError(
                "postprocess.face_detection_score_threshold must be between 0 and 1"
            )
        if not 0.0 <= self.postprocess.head_detection_score_threshold <= 1.0:
            raise OrchestrationConfigError(
                "postprocess.head_detection_score_threshold must be between 0 and 1"
            )
        if self.postprocess.face_tracking_max_gap_frames < 0:
            raise OrchestrationConfigError(
                "postprocess.face_tracking_max_gap_frames must be non-negative"
            )
        if self.postprocess.face_interpolation_max_gap < 0:
            raise OrchestrationConfigError(
                "postprocess.face_interpolation_max_gap must be non-negative"
            )
        if self.postprocess.face_short_track_max_hits < 0:
            raise OrchestrationConfigError(
                "postprocess.face_short_track_max_hits must be non-negative"
            )
        if not (
            0.0
            <= self.postprocess.face_tracking_low_score_threshold
            <= self.postprocess.face_tracking_high_score_threshold
            <= 1.0
        ):
            raise OrchestrationConfigError(
                "face tracking scores must satisfy 0 <= low <= high <= 1"
            )
        if not 0.0 <= self.postprocess.face_short_track_keep_score <= 1.0:
            raise OrchestrationConfigError(
                "postprocess.face_short_track_keep_score must be between 0 and 1"
            )
        if self.postprocess.face_mask_target != "none":
            if not self.inference.uses_faces:
                raise OrchestrationConfigError(
                    "face mask postprocess requires face inference"
                )
            if self.inference.face_model != "face_dino_v2":
                raise OrchestrationConfigError(
                    "face mask postprocess currently requires face_dino_v2"
                )
        _reject_reserved_args(
            self.postprocess.extra_args,
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
        if not self.postprocess.enabled:
            if self.postprocess.export_legacy_sqlite:
                raise OrchestrationConfigError(
                    "postprocess.export_legacy_sqlite requires "
                    "postprocess.enabled=true"
                )
            if self.overlay.enabled and self.overlay.tracked:
                if self.postprocess.tracked_sqlite is None:
                    raise OrchestrationConfigError(
                        "tracked overlay requires postprocess.tracked_sqlite "
                        "when postprocess is disabled"
                    )
            if self.overlay.enabled and self.overlay.final:
                if (
                    self.postprocess.final_sqlite is None
                    and self.postprocess.face_mask_target == "none"
                ):
                    raise OrchestrationConfigError(
                        "final overlay requires postprocess.final_sqlite "
                        "or face_mask_target when postprocess is disabled"
                    )
        faces_requested = self.overlay.enabled and (
            self.overlay.faces or self.overlay.final_include_faces
        )
        if faces_requested and not self.inference.uses_faces:
            raise OrchestrationConfigError(
                "face overlay requires inference.mode=face or segmentation-face"
            )
        if self.overlay.final_include_faces and not self.overlay.final:
            raise OrchestrationConfigError(
                "overlay.final_include_faces requires overlay.final=true"
            )
        if self.overlay.start_frame < 0:
            raise OrchestrationConfigError("overlay.start_frame must be >= 0")
        if (
            self.overlay.end_frame is not None
            and self.overlay.end_frame < self.overlay.start_frame
        ):
            raise OrchestrationConfigError(
                "overlay.end_frame must be >= overlay.start_frame"
            )
        if not 0.0 <= self.overlay.mask_alpha <= 1.0:
            raise OrchestrationConfigError("overlay.mask_alpha must be between 0 and 1")
        if self.overlay.outline_thickness < 1:
            raise OrchestrationConfigError(
                "overlay.outline_thickness must be at least 1"
            )
        if self.overlay.box_thickness < 1:
            raise OrchestrationConfigError("overlay.box_thickness must be at least 1")
        if self.overlay.progress_every < 0:
            raise OrchestrationConfigError(
                "overlay.progress_every must be non-negative"
            )
        if self.overlay.enabled and not (
            self.overlay.presets
            or self.overlay.raw
            or self.overlay.tracked
            or self.overlay.final
            or self.overlay.faces
        ):
            raise OrchestrationConfigError(
                "overlay.enabled=true requires at least one preset or legacy output"
            )
        if self.overlay.backend not in {
            "python_opencv",
            "native",
        }:
            raise OrchestrationConfigError(
                "overlay.backend must be python_opencv or native"
            )
        if self.overlay.execution_mode not in {
            "cpu",
            "nvenc",
            "fast",
        }:
            raise OrchestrationConfigError(
                "overlay.execution_mode must be cpu, nvenc, or fast"
            )
        expected_backend = (
            "native" if self.overlay.execution_mode == "fast" else "python_opencv"
        )
        if self.overlay.backend != expected_backend:
            raise OrchestrationConfigError(
                "overlay.backend does not match overlay.execution_mode"
            )
        if self.overlay.execution_mode == "cpu" and self.overlay.uses_nvenc:
            raise OrchestrationConfigError(
                "overlay.execution_mode=cpu cannot use an NVENC codec"
            )
        if (
            self.overlay.execution_mode in {"nvenc", "fast"}
            and not self.overlay.uses_nvenc
        ):
            raise OrchestrationConfigError(
                f"overlay.execution_mode={self.overlay.execution_mode} "
                "requires codec=h264_nvenc"
            )
        if self.overlay.workers < 1:
            raise OrchestrationConfigError("overlay.workers must be at least 1")
        if not 0 <= self.overlay.cpu_workers <= self.overlay.workers:
            raise OrchestrationConfigError(
                "overlay.cpu_workers must be between 0 and overlay.workers"
            )
        if self.overlay.target_bitrate_mbps is not None and (
            self.overlay.target_bitrate_mbps <= 0
        ):
            raise OrchestrationConfigError(
                "overlay.target_bitrate_mbps must be positive"
            )
        if self.overlay.nvenc_gpu < 0:
            raise OrchestrationConfigError("overlay.nvenc_gpu must be non-negative")
        if not 0 <= self.overlay.h264_crf <= 51:
            raise OrchestrationConfigError("overlay.h264_crf must be between 0 and 51")
        if (
            self.overlay.ffmpeg_bin is not None
            and not self.overlay.ffmpeg_bin.is_file()
        ):
            raise FileNotFoundError(
                f"overlay FFmpeg executable not found: {self.overlay.ffmpeg_bin}"
            )
        if self.overlay.h264_preset not in {
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
        if not 0 <= self.overlay.nvenc_cq <= 51:
            raise OrchestrationConfigError("overlay.nvenc_cq must be between 0 and 51")
        valid_presets = {
            "genital-detailed",
            "genital-simple",
            "face-detailed",
            "face-simple",
            "combined-detailed",
            "combined-simple",
        }
        invalid_presets = sorted(set(self.overlay.presets) - valid_presets)
        if invalid_presets:
            raise OrchestrationConfigError(
                f"overlay.presets contains unsupported values: {invalid_presets}"
            )
        if len(set(self.overlay.presets)) != len(self.overlay.presets):
            raise OrchestrationConfigError(
                "overlay.presets must not contain duplicates"
            )
        if self.overlay.genital_source not in {"raw", "final"}:
            raise OrchestrationConfigError(
                "overlay.genital_source must be raw or final"
            )
        if self.overlay.face_mask_target not in {"none", "face", "eyes"}:
            raise OrchestrationConfigError(
                "overlay.face_mask_target must be none, face, or eyes"
            )
        if self.overlay.eye_mask_shape not in {"ellipse", "rectangle"}:
            raise OrchestrationConfigError(
                "overlay.eye_mask_shape must be ellipse or rectangle"
            )
        if not 0.0 <= self.overlay.minimum_eye_confidence <= 1.0:
            raise OrchestrationConfigError(
                "overlay.minimum_eye_confidence must be between 0 and 1"
            )
        face_preset_requested = any(
            preset.startswith(("face-", "combined-")) for preset in self.overlay.presets
        )
        if self.overlay.face_mask_target != "none":
            if not self.inference.uses_faces:
                raise OrchestrationConfigError(
                    "overlay face privacy mask requires face inference"
                )
            if self.inference.face_model != "face_dino_v2":
                raise OrchestrationConfigError(
                    "overlay face privacy mask requires face_dino_v2"
                )
            if not (
                face_preset_requested
                or self.overlay.faces
                or (self.overlay.final and self.overlay.final_include_faces)
            ):
                raise OrchestrationConfigError(
                    "overlay face privacy mask requires a face or combined output"
                )
        if face_preset_requested and not self.inference.uses_faces:
            raise OrchestrationConfigError(
                "face/combined overlay presets require face inference"
            )
        if (
            any(
                preset.startswith(("genital-", "combined-"))
                for preset in self.overlay.presets
            )
            and not self.inference.uses_segmentation
        ):
            raise OrchestrationConfigError(
                "genital/combined overlay presets require segmentation inference"
            )
        if self.overlay.nvenc_preset not in {
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
        if self.overlay.execution_mode == "fast":
            if not self.overlay.uses_nvenc:
                raise OrchestrationConfigError(
                    "fast overlay currently requires codec=h264_nvenc"
                )
            if self.overlay.target_bitrate_mbps is None:
                raise OrchestrationConfigError(
                    "fast overlay requires overlay.target_bitrate_mbps"
                )
        elif self.overlay.copy_audio:
            raise OrchestrationConfigError(
                "overlay.copy_audio requires " "execution_mode=fast"
            )
        elif self.overlay.cpu_workers != 0:
            raise OrchestrationConfigError(
                "overlay.cpu_workers is only used by " "execution_mode=fast"
            )
        if len(self.overlay.codec) != 4 and not self.overlay.uses_nvenc:
            raise OrchestrationConfigError(
                "overlay.codec must be a four-character FourCC or h264_nvenc"
            )
        _reject_reserved_args(
            self.overlay.extra_args,
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

    def resolved_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        values = asdict(self)
        values.pop("config_path", None)
        return convert(values)  # type: ignore[return-value]


__all__ = [
    "ExecutionConfig",
    "InferenceConfig",
    "OrchestrationConfig",
    "OrchestrationConfigError",
    "OverlayConfig",
    "PostprocessConfig",
]
