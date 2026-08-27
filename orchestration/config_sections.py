"""Typed sections embedded in an orchestration JSON configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionConfig:
    """Python runtime and workflow-resume policy."""

    runtime_python: Path
    resume: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    """Segmentation and face-model execution settings."""

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
    """Mask post-processing and face-privacy settings."""

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
    mask_geometry: str = "polygon"
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
        """Catmull--Rom is deliberately CPU-only so inference keeps the GPU."""
        return self.enabled and self.mask_geometry == "polygon"


@dataclass(frozen=True)
class OverlayConfig:
    """Overlay selection, drawing, and encoding settings."""

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


__all__ = (
    "ExecutionConfig",
    "InferenceConfig",
    "OverlayConfig",
    "PostprocessConfig",
)
