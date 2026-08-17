"""Pure subprocess command assembly for the repository workflow."""

from __future__ import annotations

from pathlib import Path

from .runner_support import (
    INFERENCE_CLI,
    OVERLAY_ROOT,
    PACKAGE_RESULT_CLI,
    POSTPROCESS_CLI,
    PRECOMPUTE_CUTS_CLI,
    REPOSITORY_ROOT,
    OrchestrationError,
    WorkflowArtifacts,
)


class RunnerCommandMixin:
    """Build commands and dry-run plans without executing stages."""

    def inference_command(self, output: Path) -> list[str]:
        settings = self.config.inference
        command = [
            str(self.config.execution.runtime_python),
            str(INFERENCE_CLI),
            "--input",
            str(self.inference_video),
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
            str(self.analysis_video),
            "--output-dir",
            str(output),
            "--orchestration-config-json",
            str(self.resolved_config_path),
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
        )
        for flag, value in optional:
            if value is not None:
                command.extend([flag, str(value)])
        if settings.export_legacy_sqlite:
            command.append("--export-legacy-sqlite")
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
        execution_mode: str | None = None,
    ) -> list[str]:
        settings = self.config.overlay
        selected_execution_mode = execution_mode or settings.execution_mode
        command = [
            str(self.config.execution.runtime_python),
            "-m",
            "overlay_renderer",
            "--execution-mode",
            selected_execution_mode,
            "--video",
            str(self.analysis_video),
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
        if selected_execution_mode == "cpu":
            command.extend(
                [
                    "--h264-crf",
                    str(settings.h264_crf),
                ]
            )
        if selected_execution_mode in {"cpu", "fast"}:
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
        if selected_execution_mode == "fast":
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
                preset.replace("-", "_") for preset in self.config.overlay.presets
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
