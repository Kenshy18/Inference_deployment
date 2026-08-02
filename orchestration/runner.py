"""Subprocess-isolated repository-level workflow runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .config import OrchestrationConfig
from .contracts import (
    ArtifactError,
    read_postprocess_artifacts,
    validate_inference_sqlite,
    validate_legacy_mask_sqlite,
    validate_mask_sqlite,
    validate_result_sqlite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CLI = (
    REPOSITORY_ROOT / "InstanceSegmentation" / "inference" / "run_inference.py"
)
POSTPROCESS_CLI = REPOSITORY_ROOT / "postprocess" / "run_pipeline.py"
PRECOMPUTE_CUTS_CLI = REPOSITORY_ROOT / "postprocess" / "precompute_cuts.py"
PACKAGE_RESULT_CLI = REPOSITORY_ROOT / "postprocess" / "package_result.py"
OVERLAY_ROOT = REPOSITORY_ROOT / "overlay"


class OrchestrationError(RuntimeError):
    """Raised when a workflow stage fails or returns invalid artifacts."""


@dataclass(frozen=True)
class WorkflowArtifacts:
    inference_sqlite: Path
    tracked_sqlite: Path | None = None
    final_sqlite: Path | None = None
    result_sqlite: Path | None = None
    legacy_final_sqlite: Path | None = None


@dataclass
class BackgroundStage:
    name: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO
    started_at_utc: str
    started: float
    waiter: threading.Thread
    completion: list[tuple[int, float, str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_output_stem(value: str) -> str:
    """Return a Windows-safe public artifact stem for a source video."""

    cleaned = re.sub(r'[\\/:*?"<>|]', "_", value).strip(" .")
    return cleaned or "video"


def _atomic_copy(source: Path, destination: Path) -> None:
    """Publish a completed file by hard link, with an atomic copy fallback."""

    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination:
        return
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_destination.with_name(
        f".{resolved_destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        try:
            os.link(resolved_source, temporary)
        except OSError:
            # Cross-device and Windows-backed destinations may not support a
            # hard link. Keep the destination atomic in that case as well.
            shutil.copy2(resolved_source, temporary)
        os.replace(temporary, resolved_destination)
    finally:
        temporary.unlink(missing_ok=True)


def _emit_phase_complete(phase: str, completed: int) -> None:
    print(
        "[phase-progress] "
        + json.dumps(
            {
                "phase": phase,
                "state": "complete",
                "completed": max(0, int(completed)),
                "total": max(0, int(completed)),
                "detail": "complete",
                "fps": None,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


class OrchestrationRunner:
    def __init__(
        self,
        config: OrchestrationConfig,
        *,
        resume: bool | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.resume = config.execution.resume if resume is None else bool(resume)
        self.dry_run = dry_run
        self.output_root = config.output_root
        self.logs_dir = self.output_root / "logs"
        self.work_dir = self.logs_dir / "work"
        self.manifest_path = self.logs_dir / "run_manifest.json"
        self.resolved_config_path = self.logs_dir / "resolved_config.json"
        self.inference_dir = self.work_dir / "01_inference"
        self.preflight_dir = self.work_dir / "00_preflight"
        self.postprocess_dir = self.work_dir / "02_postprocess"
        self.overlay_dir = self.output_root / "overlay"
        self.overlay_manifest_dir = self.logs_dir / "overlay"
        self.public_result_path = self.output_root / (
            f"{_safe_output_stem(config.input_video.stem)}.sqlite"
        )
        self.config_hash = hashlib.sha256(
            json.dumps(
                config.resolved_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "planned",
            "config_hash": self.config_hash,
            "config_path": str(config.config_path),
            "input_video": str(config.input_video),
            "output_root": str(config.output_root),
            "started_at_utc": None,
            "completed_at_utc": None,
            "stages": [],
            "artifacts": {},
        }
        self._sqlite_frame_bounds_cache: dict[Path, tuple[int, int]] = {}

    def _sqlite_frame_bounds(self, source: Path) -> tuple[int, int] | None:
        """Return the materialized frame domain without revalidating the SQLite."""

        resolved = Path(source).expanduser().resolve()
        cached = self._sqlite_frame_bounds_cache.get(resolved)
        if cached is not None:
            return cached
        # Command construction is also used by dry-run/unit-test callers before
        # artifacts exist. Runtime overlay stages always receive a published,
        # validated SQLite, so retain the requested-range fallback only for that
        # pre-artifact case.
        if not resolved.is_file():
            return None
        try:
            with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as connection:
                table = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='frames'
                    """
                ).fetchone()
                columns = (
                    {
                        str(row[1])
                        for row in connection.execute('PRAGMA table_info("frames")')
                    }
                    if table is not None
                    else set()
                )
                if "frame_index" not in columns:
                    return None
                row = connection.execute(
                    "SELECT MIN(frame_index), MAX(frame_index) FROM frames"
                ).fetchone()
        except sqlite3.Error as exc:
            raise OrchestrationError(
                f"could not read overlay frame bounds from {resolved}: {exc}"
            ) from exc
        if row is None or row[0] is None or row[1] is None:
            raise OrchestrationError(
                f"overlay source SQLite has no materialized frames: {resolved}"
            )
        bounds = (int(row[0]), int(row[1]))
        self._sqlite_frame_bounds_cache[resolved] = bounds
        return bounds

    def inference_command(self, output: Path) -> list[str]:
        settings = self.config.inference
        command = [
            str(self.config.execution.runtime_python),
            str(INFERENCE_CLI),
            "--input",
            str(self.config.input_video),
            "--output",
            str(output),
            "--mode",
            settings.mode,
            "--runtime-python",
            str(self.config.execution.runtime_python),
            "--device",
            settings.device,
            "--warmup-frames",
            str(settings.warmup_frames),
            "--face-warmup-iterations",
            str(settings.face_warmup_iterations),
        ]
        if settings.uses_segmentation:
            assert settings.segmentation_model is not None
            command.extend(
                [
                    "--segmentation-model",
                    settings.segmentation_model,
                    "--segmentation-backend",
                    settings.segmentation_backend,
                ]
            )
        if settings.uses_faces:
            command.extend(
                [
                    "--face-model",
                    settings.face_model,
                    "--face-backend",
                    settings.face_backend,
                ]
            )
            command.append("--face-classes")
            command.extend(settings.face_classes)
            if settings.face_trt_bundle is not None:
                command.extend(["--face-trt-bundle", str(settings.face_trt_bundle)])
        if settings.max_frames is not None:
            command.extend(["--max-frames", str(settings.max_frames)])
        if settings.parallel_models:
            command.append("--parallel-models")
            if settings.parallel_model_stagger_seconds > 0:
                command.extend(
                    [
                        "--parallel-model-stagger-seconds",
                        str(settings.parallel_model_stagger_seconds),
                    ]
                )
        if settings.fast_sqlite:
            command.append("--fast-sqlite")
        command.extend(settings.extra_args)
        return command

    def postprocess_command(
        self,
        inference_sqlite: Path,
        *,
        precomputed_cuts: Path | None = None,
    ) -> list[str]:
        settings = self.config.postprocess
        output = self.postprocess_dir
        command = [
            str(self.config.execution.runtime_python),
            str(POSTPROCESS_CLI),
            "--input-sqlite",
            str(inference_sqlite),
            "--input-video",
            str(self.config.input_video),
            "--output-dir",
            str(output),
            "--orchestration-config-json",
            str(self.resolved_config_path),
            "--shape-mode",
            settings.shape_mode,
            "--device",
            settings.device,
            "--cut-detect" if settings.cut_detect else "--no-cut-detect",
        ]
        optional = (
            ("--pipeline-config", settings.pipeline_config),
            ("--class-policy-json", settings.class_policy_json),
            (
                "--class-postprocess-policy-json",
                settings.class_postprocess_policy_json,
            ),
            ("--score-min", settings.score_min),
            ("--cut-method", settings.cut_method),
            (
                "--remove-short-tracks-max-frames",
                settings.remove_short_tracks_max_frames,
            ),
            ("--keyframe-interval", settings.keyframe_interval),
            ("--max-gap", settings.max_gap),
            ("--model-root", settings.model_root),
            ("--k2-run-dir", settings.k2_run_dir),
            ("--k2-batch-size", settings.k2_batch_size),
            ("--k2-prep-workers", settings.k2_prep_workers),
            ("--k2-precision", settings.k2_precision),
            ("--k2-forward-mode", settings.k2_forward_mode),
            ("--k2-cudnn-benchmark", settings.k2_cudnn_benchmark),
            ("--k2-tf32", settings.k2_tf32),
        )
        for flag, value in optional:
            if value is not None:
                command.extend([flag, str(value)])
        if settings.export_legacy_sqlite:
            command.append("--export-legacy-sqlite")
        if settings.k2_profile_stages is not None:
            command.append(
                "--k2-profile-stages"
                if settings.k2_profile_stages
                else "--no-k2-profile-stages"
            )
        if precomputed_cuts is not None:
            command.extend(["--precomputed-cuts-json", str(precomputed_cuts)])
        if settings.face_mask_target != "none":
            command.extend(
                [
                    "--face-mask-target",
                    settings.face_mask_target,
                    "--eye-mask-shape",
                    settings.eye_mask_shape,
                    "--minimum-eye-confidence",
                    str(settings.minimum_eye_confidence),
                    "--face-detection-score-threshold",
                    str(settings.face_detection_score_threshold),
                    "--head-detection-score-threshold",
                    str(settings.head_detection_score_threshold),
                    "--face-tracking-max-gap-frames",
                    str(settings.face_tracking_max_gap_frames),
                    "--face-tracking-high-score-threshold",
                    str(settings.face_tracking_high_score_threshold),
                    "--face-tracking-low-score-threshold",
                    str(settings.face_tracking_low_score_threshold),
                    "--face-short-track-max-hits",
                    str(settings.face_short_track_max_hits),
                    "--face-short-track-keep-score",
                    str(settings.face_short_track_keep_score),
                    "--face-interpolation-max-gap",
                    str(settings.face_interpolation_max_gap),
                ]
            )
        command.extend(settings.extra_args)
        return command

    def overlay_command(
        self,
        *,
        mode: str | None,
        source_sqlite: Path,
        output: Path,
        manifest: Path | None = None,
        face_sqlite: Path | None = None,
        preset: str | None = None,
    ) -> list[str]:
        settings = self.config.overlay
        command = [
            str(self.config.execution.runtime_python),
            "-m",
            "overlay_renderer",
            "--execution-mode",
            settings.execution_mode,
            "--video",
            str(self.config.input_video),
            "--sqlite",
            str(source_sqlite),
            "--output",
            str(output),
            "--manifest",
            str(manifest or output.with_suffix(".json")),
            "--mask-alpha",
            str(settings.mask_alpha),
            "--outline-thickness",
            str(settings.outline_thickness),
            "--box-thickness",
            str(settings.box_thickness),
            "--start-frame",
            str(settings.start_frame),
            "--progress-every",
            str(settings.progress_every),
            "--face-mask-target",
            settings.face_mask_target,
            "--eye-mask-shape",
            settings.eye_mask_shape,
            "--minimum-eye-confidence",
            str(settings.minimum_eye_confidence),
            "--face-detection-score-threshold",
            str(self.config.postprocess.face_detection_score_threshold),
            "--head-detection-score-threshold",
            str(self.config.postprocess.head_detection_score_threshold),
        ]
        if settings.ffmpeg_bin is not None:
            command.extend(["--ffmpeg-bin", str(settings.ffmpeg_bin)])
        if preset is None:
            if mode is None:
                raise OrchestrationError("overlay mode or preset is required")
            command.extend(["--mode", mode])
        else:
            command.extend(
                [
                    "--preset",
                    preset,
                    "--genital-source",
                    settings.genital_source,
                ]
            )
        if not settings.face_probability_masks:
            command.append("--no-face-probability-masks")
        if not settings.face_keypoints:
            command.append("--no-face-keypoints")
        if not settings.face_ellipses:
            command.append("--no-face-ellipses")
        effective_end_frame = settings.end_frame
        if (
            effective_end_frame is None
            and self.config.inference.enabled
            and self.config.inference.max_frames is not None
            and self.config.inference.max_frames > 0
        ):
            # A bounded inference SQLite cannot provide overlays beyond its
            # last processed frame. The requested max is only an upper bound:
            # a decoder can legitimately materialize fewer frames near EOF.
            # Keep explicit overlay ranges authoritative, but derive implicit
            # ranges from the actual published SQLite whenever it exists.
            requested_end_frame = self.config.inference.max_frames - 1
            frame_bounds = self._sqlite_frame_bounds(source_sqlite)
            effective_end_frame = (
                requested_end_frame
                if frame_bounds is None
                else min(requested_end_frame, frame_bounds[1])
            )
        if effective_end_frame is not None:
            if effective_end_frame < settings.start_frame:
                raise OrchestrationError(
                    "overlay.start_frame is outside the inferred frame range: "
                    f"start={settings.start_frame}, end={effective_end_frame}"
                )
            command.extend(["--end-frame", str(effective_end_frame)])
        if not settings.show_labels:
            command.append("--no-labels")
        if settings.execution_mode == "cpu" and settings.codec != "h264":
            command.extend(["--codec", settings.codec])
        if settings.execution_mode == "cpu" and settings.codec == "h264":
            command.extend(
                [
                    "--h264-crf",
                    str(settings.h264_crf),
                ]
            )
        if settings.execution_mode in {"cpu", "fast"}:
            command.extend(["--h264-preset", settings.h264_preset])
        if settings.target_bitrate_mbps is not None:
            command.extend(
                [
                    "--target-bitrate-mbps",
                    str(settings.target_bitrate_mbps),
                ]
            )
        if settings.uses_nvenc:
            command.extend(
                [
                    "--nvenc-cq",
                    str(settings.nvenc_cq),
                    "--nvenc-preset",
                    settings.nvenc_preset,
                    "--nvenc-gpu",
                    str(settings.nvenc_gpu),
                ]
            )
        if settings.execution_mode == "fast":
            command.extend(
                [
                    "--workers",
                    str(settings.workers),
                    "--cpu-workers",
                    str(settings.cpu_workers),
                ]
            )
            if settings.copy_audio:
                command.append("--copy-audio")
            if settings.faststart:
                command.append("--faststart")
        if face_sqlite is not None:
            command.extend(["--include-faces", "--face-sqlite", str(face_sqlite)])
        command.extend(settings.extra_args)
        return command

    def package_result_command(
        self,
        *,
        inference_sqlite: Path,
        output: Path,
        tracked_sqlite: Path | None = None,
        final_sqlite: Path | None = None,
        precomputed_cuts: Path | None = None,
    ) -> list[str]:
        command = [
            str(self.config.execution.runtime_python),
            str(PACKAGE_RESULT_CLI),
            "--input-sqlite",
            str(inference_sqlite),
            "--output-sqlite",
            str(output),
            "--orchestration-config-json",
            str(self.resolved_config_path),
        ]
        if tracked_sqlite is not None:
            command.extend(["--tracked-sqlite", str(tracked_sqlite)])
        if final_sqlite is not None:
            command.extend(["--final-sqlite", str(final_sqlite)])
        if self.config.postprocess.face_mask_target != "none":
            command.extend(
                [
                    "--face-mask-target",
                    self.config.postprocess.face_mask_target,
                    "--eye-mask-shape",
                    self.config.postprocess.eye_mask_shape,
                    "--minimum-eye-confidence",
                    str(self.config.postprocess.minimum_eye_confidence),
                    "--face-detection-score-threshold",
                    str(self.config.postprocess.face_detection_score_threshold),
                    "--head-detection-score-threshold",
                    str(self.config.postprocess.head_detection_score_threshold),
                    "--face-tracking-max-gap-frames",
                    str(self.config.postprocess.face_tracking_max_gap_frames),
                    "--face-tracking-high-score-threshold",
                    str(self.config.postprocess.face_tracking_high_score_threshold),
                    "--face-tracking-low-score-threshold",
                    str(self.config.postprocess.face_tracking_low_score_threshold),
                    "--face-short-track-max-hits",
                    str(self.config.postprocess.face_short_track_max_hits),
                    "--face-short-track-keep-score",
                    str(self.config.postprocess.face_short_track_keep_score),
                    "--face-interpolation-max-gap",
                    str(self.config.postprocess.face_interpolation_max_gap),
                ]
            )
        if precomputed_cuts is not None:
            command.extend(["--precomputed-cuts-json", str(precomputed_cuts)])
        return command

    def plan(self) -> dict[str, object]:
        inference_output = self.inference_dir / "inference.sqlite"
        inference_source = (
            inference_output
            if self.config.inference.enabled
            else self.config.inference.input_sqlite
        )
        assert inference_source is not None
        plan: list[dict[str, object]] = []
        if self.config.postprocess.precompute_cuts_during_inference:
            cut_output = self.preflight_dir / "cuts.json"
            cut_command = [
                str(self.config.execution.runtime_python),
                str(PRECOMPUTE_CUTS_CLI),
                "--input-video",
                str(self.config.input_video),
                "--output",
                str(cut_output),
            ]
            if self.config.inference.max_frames is not None:
                cut_command.extend(
                    [
                        "--max-frames",
                        str(self.config.inference.max_frames),
                    ]
                )
            plan.append(
                {
                    "stage": "cut_precompute",
                    "uses_gpu": False,
                    "overlaps_with": "inference",
                    "command": cut_command,
                }
            )
        if self.config.inference.enabled:
            plan.append(
                {
                    "stage": "inference",
                    "uses_gpu": self.config.inference.device.lower().startswith("cuda"),
                    "command": self.inference_command(inference_output),
                }
            )
        else:
            plan.append(
                {
                    "stage": "inference",
                    "action": "reuse",
                    "artifact": str(inference_source),
                }
            )
        if self.config.postprocess.enabled:
            precomputed_cuts = (
                self.preflight_dir / "cuts.json"
                if self.config.postprocess.precompute_cuts_during_inference
                else None
            )
            plan.append(
                {
                    "stage": "postprocess",
                    "uses_gpu": self.config.postprocess.uses_gpu,
                    "command": self.postprocess_command(
                        inference_source,
                        precomputed_cuts=precomputed_cuts,
                    ),
                }
            )
        else:
            result_output = self.public_result_path
            plan.append(
                {
                    "stage": "result_packaging",
                    "uses_gpu": False,
                    "command": self.package_result_command(
                        inference_sqlite=inference_source,
                        tracked_sqlite=self.config.postprocess.tracked_sqlite,
                        final_sqlite=self.config.postprocess.final_sqlite,
                        output=result_output,
                        precomputed_cuts=(
                            self.preflight_dir / "cuts.json"
                            if self.config.postprocess.precompute_cuts_during_inference
                            else None
                        ),
                    ),
                }
            )
        if self.config.overlay.enabled:
            overlay_outputs = [
                preset.replace("-", "_")
                for preset in self.config.overlay.presets
            ]
            overlay_outputs.extend(
                mode
                for mode, enabled in (
                    ("raw", self.config.overlay.raw),
                    ("tracked", self.config.overlay.tracked),
                    ("final", self.config.overlay.final),
                    ("faces", self.config.overlay.faces),
                )
                if enabled
            )
            plan.append(
                {
                    "stage": "overlay",
                    "uses_gpu": self.config.overlay.uses_nvenc,
                    "outputs": overlay_outputs,
                }
            )
        return {
            "config_hash": self.config_hash,
            "input_video": str(self.config.input_video),
            "output_root": str(self.output_root),
            "stages": plan,
        }

    def run(self) -> dict[str, Any]:
        if self.dry_run:
            self._validate_reused_inputs()
            return self.plan()
        self._prepare_output()
        if self.resume and self.manifest.get("status") == "complete":
            self._validate_completed_run()
            return self.manifest
        self.manifest["status"] = "running"
        self.manifest.pop("error", None)
        self.manifest["started_at_utc"] = _utc_now()
        self._save_manifest()
        cut_stage: BackgroundStage | None = None
        try:
            cut_stage = self._start_cut_precompute()
            artifacts = self._run_inference()
            precomputed_cuts = (
                self._finish_background(cut_stage) if cut_stage is not None else None
            )
            cut_stage = None
            artifacts = self._run_postprocess(
                artifacts,
                precomputed_cuts=precomputed_cuts,
            )
            artifacts = self._run_result_packaging(
                artifacts,
                precomputed_cuts=precomputed_cuts,
            )
            self._run_overlays(artifacts)
        except BaseException as exc:
            if cut_stage is not None:
                self._cancel_background(cut_stage)
            self.manifest["status"] = "failed"
            self.manifest["error"] = f"{type(exc).__name__}: {exc}"
            self.manifest["completed_at_utc"] = _utc_now()
            self._save_manifest()
            raise
        self._cleanup_completed_work()
        self.manifest["status"] = "complete"
        self.manifest["completed_at_utc"] = _utc_now()
        self._save_manifest()
        return self.manifest

    def _prepare_output(self) -> None:
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not self.resume:
                raise FileExistsError(
                    f"run manifest already exists: {self.manifest_path}; "
                    "use --resume or a new output_root"
                )
            if previous.get("config_hash") != self.config_hash:
                raise OrchestrationError(
                    "cannot resume: resolved configuration has changed"
                )
            self.manifest = previous
            return
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise FileExistsError(
                f"output_root is not empty: {self.output_root}; "
                "choose a new directory"
            )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.resolved_config_path,
            self.config.resolved_dict(),
        )

    def _validate_reused_inputs(self) -> None:
        if not self.config.inference.enabled:
            assert self.config.inference.input_sqlite is not None
            validate_inference_sqlite(
                self.config.inference.input_sqlite,
                require_segmentation=self.config.inference.uses_segmentation,
                require_faces=self.config.inference.uses_faces,
                expected_face_model=(
                    self.config.inference.face_model
                    if self.config.inference.uses_faces
                    else None
                ),
            )
        if not self.config.postprocess.enabled:
            if self.config.postprocess.tracked_sqlite is not None:
                validate_mask_sqlite(self.config.postprocess.tracked_sqlite)
            if self.config.postprocess.final_sqlite is not None:
                validate_mask_sqlite(self.config.postprocess.final_sqlite)

    def _validate_completed_run(self) -> None:
        """Make completed-run resume a validated no-op after work cleanup."""

        artifacts = self.manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise OrchestrationError("completed run has an invalid artifact manifest")
        missing = [
            name
            for name, value in artifacts.items()
            if not isinstance(value, str)
            or not value
            or not Path(value).expanduser().is_file()
        ]
        if missing:
            raise OrchestrationError(
                "completed run is missing published artifacts: "
                + ", ".join(sorted(missing))
            )
        result = artifacts.get("result_sqlite")
        if not isinstance(result, str) or not result:
            raise OrchestrationError("completed run has no published result_sqlite")
        validate_result_sqlite(
            Path(result),
            require_segmentation=self.config.inference.uses_segmentation,
            require_faces=self.config.inference.uses_faces,
            expected_face_model=(
                self.config.inference.face_model
                if self.config.inference.uses_faces
                else None
            ),
        )

    def _run_inference(self) -> WorkflowArtifacts:
        settings = self.config.inference
        output = self.inference_dir / "inference.sqlite"
        if settings.enabled:
            if self._can_resume_stage("inference", {"inference_sqlite": output}):
                inference_sqlite = output
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                command = self.inference_command(output)
                self._execute("inference", command, cpu_only=False)
                inference_sqlite = output
        else:
            assert settings.input_sqlite is not None
            inference_sqlite = settings.input_sqlite
            self._record_reuse("inference", inference_sqlite)
        stats = validate_inference_sqlite(
            inference_sqlite,
            require_segmentation=settings.uses_segmentation,
            require_faces=settings.uses_faces,
            expected_face_model=(settings.face_model if settings.uses_faces else None),
        )
        if settings.uses_segmentation:
            _emit_phase_complete(
                "segmentation_inference",
                int(stats["frames"]),
            )
        if settings.uses_faces:
            _emit_phase_complete("face_inference", int(stats["frames"]))
        self._publish_artifacts(
            {"inference_sqlite": inference_sqlite},
            validation={"inference_sqlite": stats},
        )
        return WorkflowArtifacts(inference_sqlite=inference_sqlite)

    def _run_postprocess(
        self,
        artifacts: WorkflowArtifacts,
        *,
        precomputed_cuts: Path | None = None,
    ) -> WorkflowArtifacts:
        settings = self.config.postprocess
        if settings.enabled:
            post_root = self.postprocess_dir
            manifest_path = post_root / "pipeline_manifest.json"
            resumed_postprocess = self._can_resume_stage(
                "postprocess",
                {"postprocess_manifest": manifest_path},
            )
            if resumed_postprocess:
                tracked, final, legacy = read_postprocess_artifacts(manifest_path)
            else:
                command = self.postprocess_command(
                    artifacts.inference_sqlite,
                    precomputed_cuts=precomputed_cuts,
                )
                self._execute(
                    "postprocess",
                    command,
                    cpu_only=not settings.uses_gpu,
                )
                tracked, final, legacy = read_postprocess_artifacts(manifest_path)
            if settings.export_legacy_sqlite and legacy is None:
                raise OrchestrationError(
                    "postprocess did not publish legacy_predictions_sqlite"
                )
            validate_mask_sqlite(tracked)
            published = {"postprocess_manifest": manifest_path}
            validation: dict[str, object] = {}
            integrated = False
            try:
                result_validation = validate_result_sqlite(
                    final,
                    require_segmentation=self.config.inference.uses_segmentation,
                    require_faces=self.config.inference.uses_faces,
                    expected_face_model=(
                        self.config.inference.face_model
                        if self.config.inference.uses_faces
                        else None
                    ),
                )
            except ArtifactError:
                # Custom and older pipelines can still return a mask-only final
                # SQLite.  The following result_packaging stage promotes it to
                # the same stable public contract.
                pass
            else:
                integrated = True
                published["result_sqlite"] = final
                validation["result_sqlite"] = result_validation
            if legacy is not None:
                public_legacy = (
                    self.logs_dir
                    / "legacy"
                    / f"{_safe_output_stem(self.config.input_video.stem)}_legacy.sqlite"
                )
                _atomic_copy(legacy, public_legacy)
                published["legacy_final_sqlite"] = public_legacy
                validation["legacy_final_sqlite"] = validate_legacy_mask_sqlite(
                    public_legacy
                )
                legacy = public_legacy
            self._publish_artifacts(
                published,
                validation=validation,
                replace_sqlite_outputs=integrated,
            )
            if resumed_postprocess:
                _emit_phase_complete("postprocess", 1)
            return WorkflowArtifacts(
                inference_sqlite=artifacts.inference_sqlite,
                tracked_sqlite=tracked,
                final_sqlite=final,
                result_sqlite=final if integrated else None,
                legacy_final_sqlite=legacy,
            )
        tracked = settings.tracked_sqlite
        final = settings.final_sqlite
        if tracked is not None:
            self._record_reuse("tracked", tracked)
            validate_mask_sqlite(tracked)
        if final is not None:
            self._record_reuse("final", final)
            validate_mask_sqlite(final)
        return WorkflowArtifacts(
            inference_sqlite=artifacts.inference_sqlite,
            tracked_sqlite=tracked,
            final_sqlite=final,
        )

    def _run_result_packaging(
        self,
        artifacts: WorkflowArtifacts,
        *,
        precomputed_cuts: Path | None = None,
    ) -> WorkflowArtifacts:
        """Guarantee one stable public result SQLite for every mode."""

        if artifacts.result_sqlite is not None:
            output = self.public_result_path
            if not self._can_resume_stage(
                "result_publication",
                {"result_sqlite": output},
            ):
                started = time.perf_counter()
                _atomic_copy(artifacts.result_sqlite, output)
                self._replace_stage_record(
                    {
                        "name": "result_publication",
                        "status": "complete",
                        "source": str(artifacts.result_sqlite),
                        "artifact": str(output),
                        "elapsed_seconds": time.perf_counter() - started,
                        "completed_at_utc": _utc_now(),
                    }
                )
            validation = validate_result_sqlite(
                output,
                require_segmentation=self.config.inference.uses_segmentation,
                require_faces=self.config.inference.uses_faces,
                expected_face_model=(
                    self.config.inference.face_model
                    if self.config.inference.uses_faces
                    else None
                ),
            )
            self._publish_artifacts(
                {"result_sqlite": output},
                validation={"result_sqlite": validation},
                replace_sqlite_outputs=True,
            )
            return WorkflowArtifacts(
                inference_sqlite=artifacts.inference_sqlite,
                tracked_sqlite=artifacts.tracked_sqlite,
                final_sqlite=artifacts.final_sqlite,
                result_sqlite=output,
                legacy_final_sqlite=artifacts.legacy_final_sqlite,
            )

        output = self.public_result_path
        if not self._can_resume_stage(
            "result_packaging",
            {"result_sqlite": output},
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            self._execute(
                "result_packaging",
                self.package_result_command(
                    inference_sqlite=artifacts.inference_sqlite,
                    tracked_sqlite=artifacts.tracked_sqlite,
                    final_sqlite=artifacts.final_sqlite,
                    output=output,
                    precomputed_cuts=precomputed_cuts,
                ),
                cpu_only=True,
            )
        validation = validate_result_sqlite(
            output,
            require_segmentation=self.config.inference.uses_segmentation,
            require_faces=self.config.inference.uses_faces,
            expected_face_model=(
                self.config.inference.face_model
                if self.config.inference.uses_faces
                else None
            ),
        )
        self._publish_artifacts(
            {"result_sqlite": output},
            validation={"result_sqlite": validation},
            replace_sqlite_outputs=True,
        )
        return WorkflowArtifacts(
            inference_sqlite=artifacts.inference_sqlite,
            tracked_sqlite=artifacts.tracked_sqlite,
            final_sqlite=artifacts.final_sqlite,
            result_sqlite=output,
            legacy_final_sqlite=artifacts.legacy_final_sqlite,
        )

    def _start_cut_precompute(self) -> BackgroundStage | None:
        settings = self.config.postprocess
        if not settings.precompute_cuts_during_inference:
            return None
        output = self.preflight_dir / "cuts.json"
        command = [
            str(self.config.execution.runtime_python),
            str(PRECOMPUTE_CUTS_CLI),
            "--input-video",
            str(self.config.input_video),
            "--output",
            str(output),
        ]
        if self.config.inference.max_frames is not None:
            command.extend(["--max-frames", str(self.config.inference.max_frames)])
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / "cut_precompute.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
        started_at_utc = _utc_now()
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except BaseException:
            log_handle.close()
            raise
        completion: list[tuple[int, float, str]] = []

        def wait_for_completion() -> None:
            return_code = process.wait()
            completion.append((return_code, time.perf_counter(), _utc_now()))

        waiter = threading.Thread(
            target=wait_for_completion,
            name="orchestration:cut_precompute",
            daemon=True,
        )
        waiter.start()
        self._replace_stage_record(
            {
                "name": "cut_precompute",
                "status": "running",
                "cpu_only": True,
                "command": command,
                "log": str(log_path),
                "started_at_utc": started_at_utc,
            }
        )
        return BackgroundStage(
            name="cut_precompute",
            command=command,
            process=process,
            log_path=log_path,
            log_handle=log_handle,
            started_at_utc=started_at_utc,
            started=started,
            waiter=waiter,
            completion=completion,
        )

    def _finish_background(self, stage: BackgroundStage) -> Path:
        stage.waiter.join()
        return_code, completed, completed_at_utc = stage.completion[0]
        stage.log_handle.close()
        elapsed = completed - stage.started
        record = {
            "name": stage.name,
            "status": "complete" if return_code == 0 else "failed",
            "cpu_only": True,
            "command": stage.command,
            "log": str(stage.log_path),
            "started_at_utc": stage.started_at_utc,
            "elapsed_seconds": elapsed,
            "completed_at_utc": completed_at_utc,
            "return_code": return_code,
            "overlapped_with": "inference",
            "overlap_window_seconds": time.perf_counter() - stage.started,
        }
        self._replace_stage_record(record)
        if return_code != 0:
            raise OrchestrationError(
                "precomputed cut detection failed with exit code "
                f"{return_code}; see {stage.log_path}"
            )
        output = self.preflight_dir / "cuts.json"
        if not output.is_file() or output.stat().st_size == 0:
            raise OrchestrationError(
                f"precomputed cut detection did not create {output}"
            )
        self._publish_artifacts({"precomputed_cuts": output})
        return output

    def _cancel_background(self, stage: BackgroundStage) -> None:
        if stage.process.poll() is None:
            stage.process.terminate()
            try:
                stage.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                stage.process.kill()
                stage.process.wait()
        stage.waiter.join()
        stage.log_handle.close()

    def _run_overlays(self, artifacts: WorkflowArtifacts) -> None:
        settings = self.config.overlay
        if not settings.enabled:
            return
        output_root = self.overlay_dir
        output_root.mkdir(parents=True, exist_ok=True)
        requested: list[tuple[str, str | None, Path, Path | None, str | None]] = []
        unified = artifacts.result_sqlite or artifacts.inference_sqlite
        if settings.presets:
            requested.extend(
                (
                    preset.replace("-", "_"),
                    None,
                    unified,
                    None,
                    preset,
                )
                for preset in settings.presets
            )
        if settings.raw:
            requested.append(
                (
                    "raw",
                    "raw",
                    artifacts.result_sqlite or artifacts.inference_sqlite,
                    None,
                    None,
                )
            )
        if settings.tracked:
            tracked_source = artifacts.result_sqlite or artifacts.tracked_sqlite
            if tracked_source is None:
                raise OrchestrationError("tracked overlay has no tracked SQLite")
            requested.append(("tracked", "tracked", tracked_source, None, None))
        if settings.final:
            final_source = artifacts.result_sqlite or artifacts.final_sqlite
            if final_source is None:
                raise OrchestrationError("final overlay has no final SQLite")
            requested.append(
                (
                    "final",
                    "final",
                    final_source,
                    (
                        artifacts.result_sqlite or artifacts.inference_sqlite
                        if settings.final_include_faces
                        else None
                    ),
                    None,
                )
            )
        if settings.faces:
            requested.append(
                (
                    "faces",
                    "faces",
                    artifacts.result_sqlite or artifacts.inference_sqlite,
                    None,
                    None,
                )
            )
        for overlay_index, (
            name,
            mode,
            source,
            face_source,
            preset,
        ) in enumerate(requested):
            output = output_root / f"{name}.mp4"
            output_manifest = self.overlay_manifest_dir / f"{name}.json"
            artifact_name = f"overlay_{name}"
            if self._can_resume_stage(
                f"overlay_{name}",
                {
                    artifact_name: output,
                    f"{artifact_name}_manifest": output_manifest,
                },
            ):
                continue
            command = self.overlay_command(
                mode=mode,
                source_sqlite=source,
                output=output,
                manifest=output_manifest,
                face_sqlite=face_source,
                preset=preset,
            )
            self._execute(
                f"overlay_{name}",
                command,
                cpu_only=not settings.uses_nvenc,
                extra_pythonpath=OVERLAY_ROOT / "src",
                extra_environment={
                    "MASK_PIPELINE_PROGRESS_ITEM_INDEX": str(overlay_index),
                    "MASK_PIPELINE_PROGRESS_ITEM_COUNT": str(len(requested)),
                    "MASK_PIPELINE_PROGRESS_ITEM_NAME": name,
                },
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise OrchestrationError(f"overlay did not create output: {output}")
            self._publish_artifacts(
                {
                    artifact_name: output,
                    f"{artifact_name}_manifest": output_manifest,
                }
            )

    def _cleanup_completed_work(self) -> None:
        """Remove reproducible stage data only after every public output exists."""

        resolved_work = self.work_dir.resolve()
        current = dict(self.manifest.get("artifacts", {}))
        removed_artifacts: list[str] = []
        for name, value in list(current.items()):
            if not isinstance(value, str) or not value:
                continue
            try:
                candidate = Path(value).expanduser().resolve()
            except OSError:
                continue
            if candidate == resolved_work or resolved_work in candidate.parents:
                current.pop(name, None)
                removed_artifacts.append(name)
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.manifest["artifacts"] = current
        self.manifest["cleanup"] = {
            "work_directory": str(self.work_dir),
            "work_removed": not self.work_dir.exists(),
            "removed_internal_artifacts": sorted(removed_artifacts),
            "completed_at_utc": _utc_now(),
        }
        self._save_manifest()

    def _execute(
        self,
        stage: str,
        command: list[str],
        *,
        cpu_only: bool,
        extra_pythonpath: Path | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> None:
        log_path = self.logs_dir / f"{stage}.log"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if cpu_only:
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["NVIDIA_VISIBLE_DEVICES"] = "none"
        if extra_pythonpath is not None:
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                str(extra_pythonpath)
                if not existing
                else f"{extra_pythonpath}{os.pathsep}{existing}"
            )
        if extra_environment is not None:
            environment.update(
                {str(key): str(value) for key, value in extra_environment.items()}
            )
        record: dict[str, Any] = {
            "name": stage,
            "status": "running",
            "cpu_only": cpu_only,
            "command": command,
            "log": str(log_path),
            "started_at_utc": _utc_now(),
        }
        self._replace_stage_record(record)
        started = time.perf_counter()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    print(f"[{stage}] {line}", end="", flush=True)
                    if (
                        "[phase-progress]" not in line
                        and "[live-preview]" not in line
                    ):
                        log.write(line)
            return_code = process.wait()
        record["elapsed_seconds"] = time.perf_counter() - started
        record["completed_at_utc"] = _utc_now()
        record["return_code"] = return_code
        record["status"] = "complete" if return_code == 0 else "failed"
        self._replace_stage_record(record)
        if return_code != 0:
            raise OrchestrationError(
                f"stage {stage!r} failed with exit code {return_code}; "
                f"see {log_path}"
            )

    def _record_reuse(self, stage: str, artifact: Path) -> None:
        self._replace_stage_record(
            {
                "name": stage,
                "status": "reused",
                "artifact": str(artifact),
                "completed_at_utc": _utc_now(),
            }
        )

    def _can_resume_stage(
        self,
        stage: str,
        artifacts: dict[str, Path],
    ) -> bool:
        if not self.resume:
            return False
        previous = next(
            (
                item
                for item in self.manifest.get("stages", [])
                if item.get("name") == stage
            ),
            None,
        )
        if previous is None or previous.get("status") not in {"complete", "reused"}:
            return False
        return all(
            path.is_file() and path.stat().st_size > 0 for path in artifacts.values()
        )

    def _replace_stage_record(self, record: dict[str, Any]) -> None:
        stages = [
            item
            for item in self.manifest.get("stages", [])
            if item.get("name") != record["name"]
        ]
        stages.append(record)
        self.manifest["stages"] = stages
        self._save_manifest()

    def _publish_artifacts(
        self,
        artifacts: dict[str, Path],
        *,
        validation: dict[str, object] | None = None,
        replace_sqlite_outputs: bool = False,
    ) -> None:
        current = dict(self.manifest.get("artifacts", {}))
        if replace_sqlite_outputs:
            for name in (
                "inference_sqlite",
                "tracked_sqlite",
                "final_sqlite",
            ):
                current.pop(name, None)
        current.update({name: str(path) for name, path in artifacts.items()})
        self.manifest["artifacts"] = current
        if validation:
            checks = dict(self.manifest.get("validation", {}))
            if replace_sqlite_outputs:
                for name in (
                    "inference_sqlite",
                    "tracked_sqlite",
                    "internal_tracked_sqlite",
                    "final_sqlite",
                ):
                    checks.pop(name, None)
            checks.update(validation)
            self.manifest["validation"] = checks
        self._save_manifest()

    def _save_manifest(self) -> None:
        if self.dry_run:
            return
        _atomic_json(self.manifest_path, self.manifest)


__all__ = [
    "OrchestrationError",
    "OrchestrationRunner",
    "WorkflowArtifacts",
]
