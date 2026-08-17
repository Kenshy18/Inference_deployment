"""Subprocess-isolated repository-level workflow runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import OrchestrationConfig
from .contracts import (
    ArtifactError,
    read_postprocess_artifacts,
    validate_inference_sqlite,
    validate_legacy_mask_sqlite,
    validate_mask_sqlite,
    validate_result_sqlite,
)
from .rescale_result_sqlite import VideoGeometry


from .runner_support import (
    OVERLAY_ROOT,
    PRECOMPUTE_CUTS_CLI,
    REPOSITORY_ROOT,
    BackgroundStage,
    OrchestrationError,
    WorkflowArtifacts,
    _atomic_copy,
    _atomic_json,
    _emit_phase_complete,
    _safe_output_stem,
    _utc_now,
)
from .runner_commands import RunnerCommandMixin
from .runner_media import RunnerMediaMixin


class OrchestrationRunner(RunnerMediaMixin, RunnerCommandMixin):
    def __init__(
        self,
        config: OrchestrationConfig,
        *,
        resume: bool | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        # The public configuration always keeps the user-selected source.  An
        # interlaced source is normalized once and all processing stages use
        # this private progressive working copy instead.
        self.processing_video = config.input_video
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
        self.proxy_video_path = self.preflight_dir / "analysis_proxy_1920x1080.mp4"
        self.proxy_result_path = self.preflight_dir / "result_1920x1080.sqlite"
        self.canonical_inference_path = (
            self.preflight_dir / "inference_1920x1080.sqlite"
        )
        self.inference_video = config.input_video
        self.analysis_video = config.input_video
        self.original_geometry: VideoGeometry | None = None
        self.analysis_geometry: VideoGeometry | None = None
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
            self._prepare_processing_video()
            self._prepare_analysis_video()
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
            postprocess_input = self._prepare_postprocess_input(
                artifacts.inference_sqlite
            )
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
                    postprocess_input,
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
                inference_sqlite=postprocess_input,
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

        proxy_run = self.analysis_video == self.proxy_video_path
        if proxy_run and (
            self.original_geometry is None or self.analysis_geometry is None
        ):
            raise OrchestrationError("analysis proxy geometry was not initialized")

        if artifacts.result_sqlite is not None:
            output = self.public_result_path
            expected_publication = {"result_sqlite": output}
            if proxy_run:
                expected_publication["proxy_result_sqlite"] = self.proxy_result_path
            if not self._can_resume_stage(
                "result_publication",
                expected_publication,
            ):
                started = time.perf_counter()
                transform = None
                publication_source = artifacts.result_sqlite
                if proxy_run:
                    _atomic_copy(artifacts.result_sqlite, self.proxy_result_path)
                    assert self.original_geometry is not None
                    assert self.analysis_geometry is not None
                    transform = rescale_result_sqlite(
                        self.proxy_result_path,
                        output,
                        proxy=self.analysis_geometry,
                        original=self.original_geometry,
                        original_video=self.config.input_video,
                    )
                    publication_source = self.proxy_result_path
                else:
                    _atomic_copy(artifacts.result_sqlite, output)
                self._replace_stage_record(
                    {
                        "name": "result_publication",
                        "status": "complete",
                        "source": str(publication_source),
                        "artifact": str(output),
                        "coordinate_transform": transform,
                        "elapsed_seconds": time.perf_counter() - started,
                        "completed_at_utc": _utc_now(),
                    }
                )
            if not proxy_run:
                self._restore_original_video_path(output)
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
            published = {"result_sqlite": output}
            publication_validation: dict[str, object] = {"result_sqlite": validation}
            if proxy_run:
                proxy_validation = validate_result_sqlite(
                    self.proxy_result_path,
                    require_segmentation=self.config.inference.uses_segmentation,
                    require_faces=self.config.inference.uses_faces,
                    expected_face_model=(
                        self.config.inference.face_model
                        if self.config.inference.uses_faces
                        else None
                    ),
                )
                published["proxy_result_sqlite"] = self.proxy_result_path
                publication_validation["proxy_result_sqlite"] = proxy_validation
            self._publish_artifacts(
                published,
                validation=publication_validation,
                replace_sqlite_outputs=True,
            )
            return WorkflowArtifacts(
                inference_sqlite=artifacts.inference_sqlite,
                tracked_sqlite=artifacts.tracked_sqlite,
                final_sqlite=artifacts.final_sqlite,
                result_sqlite=output,
                overlay_sqlite=self.proxy_result_path if proxy_run else output,
                legacy_final_sqlite=artifacts.legacy_final_sqlite,
            )

        output = self.public_result_path
        package_output = self.proxy_result_path if proxy_run else output
        expected_packaging = {"result_sqlite": output}
        if proxy_run:
            expected_packaging["proxy_result_sqlite"] = self.proxy_result_path
        if not self._can_resume_stage(
            "result_packaging",
            expected_packaging,
        ):
            package_output.parent.mkdir(parents=True, exist_ok=True)
            self._execute(
                "result_packaging",
                self.package_result_command(
                    inference_sqlite=artifacts.inference_sqlite,
                    tracked_sqlite=artifacts.tracked_sqlite,
                    final_sqlite=artifacts.final_sqlite,
                    output=package_output,
                    precomputed_cuts=precomputed_cuts,
                ),
                cpu_only=True,
            )
            if proxy_run:
                assert self.original_geometry is not None
                assert self.analysis_geometry is not None
                rescale_result_sqlite(
                    package_output,
                    output,
                    proxy=self.analysis_geometry,
                    original=self.original_geometry,
                    original_video=self.config.input_video,
                )
        if not proxy_run:
            self._restore_original_video_path(output)
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
        published = {"result_sqlite": output}
        validation_payload: dict[str, object] = {"result_sqlite": validation}
        if proxy_run:
            published["proxy_result_sqlite"] = self.proxy_result_path
            validation_payload["proxy_result_sqlite"] = validate_result_sqlite(
                self.proxy_result_path,
                require_segmentation=self.config.inference.uses_segmentation,
                require_faces=self.config.inference.uses_faces,
                expected_face_model=(
                    self.config.inference.face_model
                    if self.config.inference.uses_faces
                    else None
                ),
            )
        self._publish_artifacts(
            published,
            validation=validation_payload,
            replace_sqlite_outputs=True,
        )
        return WorkflowArtifacts(
            inference_sqlite=artifacts.inference_sqlite,
            tracked_sqlite=artifacts.tracked_sqlite,
            final_sqlite=artifacts.final_sqlite,
            result_sqlite=output,
            overlay_sqlite=self.proxy_result_path if proxy_run else output,
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
            str(self.analysis_video),
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
        unified = (
            artifacts.overlay_sqlite
            or artifacts.result_sqlite
            or artifacts.inference_sqlite
        )
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
                    unified,
                    None,
                    None,
                )
            )
        if settings.tracked:
            tracked_source = (
                artifacts.overlay_sqlite
                or artifacts.result_sqlite
                or artifacts.tracked_sqlite
            )
            if tracked_source is None:
                raise OrchestrationError("tracked overlay has no tracked SQLite")
            requested.append(("tracked", "tracked", tracked_source, None, None))
        if settings.final:
            final_source = (
                artifacts.overlay_sqlite
                or artifacts.result_sqlite
                or artifacts.final_sqlite
            )
            if final_source is None:
                raise OrchestrationError("final overlay has no final SQLite")
            requested.append(
                (
                    "final",
                    "final",
                    final_source,
                    (unified if settings.final_include_faces else None),
                    None,
                )
            )
        if settings.faces:
            requested.append(
                (
                    "faces",
                    "faces",
                    unified,
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
            fallback_modes = {
                "fast": ("fast", "nvenc", "cpu"),
                "nvenc": ("nvenc", "cpu"),
                "cpu": ("cpu",),
            }[settings.execution_mode]
            attempts: list[dict[str, object]] = []
            last_error: OrchestrationError | None = None
            for attempt_index, execution_mode in enumerate(fallback_modes):
                output.unlink(missing_ok=True)
                output_manifest.unlink(missing_ok=True)
                command = self.overlay_command(
                    mode=mode,
                    source_sqlite=source,
                    output=output,
                    manifest=output_manifest,
                    face_sqlite=face_source,
                    preset=preset,
                    execution_mode=execution_mode,
                )
                attempt_stage = (
                    f"overlay_{name}"
                    if attempt_index == 0
                    else f"overlay_{name}_{execution_mode}_fallback"
                )
                try:
                    self._execute(
                        attempt_stage,
                        command,
                        cpu_only=execution_mode == "cpu",
                        extra_pythonpath=OVERLAY_ROOT / "src",
                        extra_environment={
                            "MASK_PIPELINE_PROGRESS_ITEM_INDEX": str(overlay_index),
                            "MASK_PIPELINE_PROGRESS_ITEM_COUNT": str(len(requested)),
                            "MASK_PIPELINE_PROGRESS_ITEM_NAME": name,
                        },
                    )
                except OrchestrationError as exc:
                    last_error = exc
                    attempts.append(
                        {
                            "execution_mode": execution_mode,
                            "status": "failed",
                            "error": str(exc),
                            "log": str(self.logs_dir / f"{attempt_stage}.log"),
                        }
                    )
                    if attempt_index + 1 < len(fallback_modes):
                        print(
                            f"[overlay_{name}] {execution_mode} failed; "
                            f"retrying with {fallback_modes[attempt_index + 1]}",
                            flush=True,
                        )
                    continue
                attempts.append(
                    {
                        "execution_mode": execution_mode,
                        "status": "complete",
                        "log": str(self.logs_dir / f"{attempt_stage}.log"),
                    }
                )
                if attempt_index > 0:
                    self._replace_stage_record(
                        {
                            "name": f"overlay_{name}",
                            "status": "complete",
                            "execution_mode": execution_mode,
                            "attempts": attempts,
                            "completed_at_utc": _utc_now(),
                        }
                    )
                break
            else:
                output.unlink(missing_ok=True)
                output_manifest.unlink(missing_ok=True)
                assert last_error is not None
                raise last_error
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
                    if "[phase-progress]" not in line and "[live-preview]" not in line:
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
