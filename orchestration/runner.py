"""Subprocess-isolated repository-level workflow runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .config import OrchestrationConfig
from .contracts import (
    read_postprocess_artifacts,
    validate_inference_sqlite,
    validate_legacy_mask_sqlite,
    validate_mask_sqlite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CLI = (
    REPOSITORY_ROOT / "InstanceSegmentation" / "inference" / "run_inference.py"
)
POSTPROCESS_CLI = REPOSITORY_ROOT / "postprocess" / "run_pipeline.py"
PRECOMPUTE_CUTS_CLI = REPOSITORY_ROOT / "postprocess" / "precompute_cuts.py"
OVERLAY_ROOT = REPOSITORY_ROOT / "overlay"


class OrchestrationError(RuntimeError):
    """Raised when a workflow stage fails or returns invalid artifacts."""


@dataclass(frozen=True)
class WorkflowArtifacts:
    inference_sqlite: Path
    tracked_sqlite: Path | None = None
    final_sqlite: Path | None = None
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
        self.manifest_path = self.output_root / "run_manifest.json"
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
            command.extend(["--face-model", settings.face_model])
            command.append("--face-classes")
            command.extend(settings.face_classes)
            if settings.face_trt_bundle is not None:
                command.extend(
                    ["--face-trt-bundle", str(settings.face_trt_bundle)]
                )
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
        output = self.output_root / "02_postprocess"
        command = [
            str(self.config.execution.runtime_python),
            str(POSTPROCESS_CLI),
            "--input-sqlite",
            str(inference_sqlite),
            "--input-video",
            str(self.config.input_video),
            "--output-dir",
            str(output),
            "--shape-mode",
            settings.shape_mode,
            "--device",
            settings.device,
            "--cut-detect" if settings.cut_detect else "--no-cut-detect",
        ]
        optional = (
            ("--pipeline-config", settings.pipeline_config),
            ("--class-policy-json", settings.class_policy_json),
            ("--score-min", settings.score_min),
            ("--cut-method", settings.cut_method),
            (
                "--remove-short-tracks-max-frames",
                settings.remove_short_tracks_max_frames,
            ),
            ("--keyframe-interval", settings.keyframe_interval),
            ("--model-root", settings.model_root),
            ("--k2-run-dir", settings.k2_run_dir),
        )
        for flag, value in optional:
            if value is not None:
                command.extend([flag, str(value)])
        if settings.export_legacy_sqlite:
            command.append("--export-legacy-sqlite")
        if precomputed_cuts is not None:
            command.extend(
                ["--precomputed-cuts-json", str(precomputed_cuts)]
            )
        if settings.face_mask_target != "none":
            command.extend(
                [
                    "--face-mask-target",
                    settings.face_mask_target,
                    "--eye-mask-shape",
                    settings.eye_mask_shape,
                    "--minimum-eye-confidence",
                    str(settings.minimum_eye_confidence),
                ]
            )
        command.extend(settings.extra_args)
        return command

    def overlay_command(
        self,
        *,
        mode: str,
        source_sqlite: Path,
        output: Path,
        face_sqlite: Path | None = None,
    ) -> list[str]:
        settings = self.config.overlay
        command = [
            str(self.config.execution.runtime_python),
            "-m",
            "overlay_renderer",
            "--execution-mode",
            settings.execution_mode,
            "--mode",
            mode,
            "--video",
            str(self.config.input_video),
            "--sqlite",
            str(source_sqlite),
            "--output",
            str(output),
            "--manifest",
            str(output.with_suffix(".json")),
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
        ]
        if settings.end_frame is not None:
            command.extend(["--end-frame", str(settings.end_frame)])
        if not settings.show_labels:
            command.append("--no-labels")
        if settings.execution_mode == "cpu" and settings.codec != "h264":
            command.extend(["--codec", settings.codec])
        if settings.execution_mode == "cpu" and settings.codec == "h264":
            command.extend(
                [
                    "--h264-crf",
                    str(settings.h264_crf),
                    "--h264-preset",
                    settings.h264_preset,
                ]
            )
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

    def plan(self) -> dict[str, object]:
        inference_output = self.output_root / "01_inference" / "inference.sqlite"
        inference_source = (
            inference_output
            if self.config.inference.enabled
            else self.config.inference.input_sqlite
        )
        assert inference_source is not None
        plan: list[dict[str, object]] = []
        if self.config.postprocess.precompute_cuts_during_inference:
            cut_output = self.output_root / "00_preflight" / "cuts.json"
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
                self.output_root / "00_preflight" / "cuts.json"
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
        if self.config.overlay.enabled:
            plan.append(
                {
                    "stage": "overlay",
                    "uses_gpu": self.config.overlay.uses_nvenc,
                    "outputs": [
                        mode
                        for mode, enabled in (
                            ("raw", self.config.overlay.raw),
                            ("tracked", self.config.overlay.tracked),
                            ("final", self.config.overlay.final),
                            ("faces", self.config.overlay.faces),
                        )
                        if enabled
                    ],
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
        self.manifest["status"] = "running"
        self.manifest.pop("error", None)
        self.manifest["started_at_utc"] = _utc_now()
        self._save_manifest()
        cut_stage: BackgroundStage | None = None
        try:
            cut_stage = self._start_cut_precompute()
            artifacts = self._run_inference()
            precomputed_cuts = (
                self._finish_background(cut_stage)
                if cut_stage is not None
                else None
            )
            cut_stage = None
            artifacts = self._run_postprocess(
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
            self.output_root / "resolved_config.json",
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

    def _run_inference(self) -> WorkflowArtifacts:
        settings = self.config.inference
        output = self.output_root / "01_inference" / "inference.sqlite"
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
            post_root = self.output_root / "02_postprocess"
            manifest_path = post_root / "pipeline_manifest.json"
            if self._can_resume_stage(
                "postprocess",
                {"postprocess_manifest": manifest_path},
            ):
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
            published = {
                "postprocess_manifest": manifest_path,
                "tracked_sqlite": tracked,
                "final_sqlite": final,
            }
            validation = {
                "tracked_sqlite": validate_mask_sqlite(tracked),
                "final_sqlite": validate_mask_sqlite(final),
            }
            if legacy is not None:
                published["legacy_final_sqlite"] = legacy
                validation["legacy_final_sqlite"] = validate_legacy_mask_sqlite(legacy)
            self._publish_artifacts(
                published,
                validation=validation,
            )
            return WorkflowArtifacts(
                inference_sqlite=artifacts.inference_sqlite,
                tracked_sqlite=tracked,
                final_sqlite=final,
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

    def _start_cut_precompute(self) -> BackgroundStage | None:
        settings = self.config.postprocess
        if not settings.precompute_cuts_during_inference:
            return None
        output = self.output_root / "00_preflight" / "cuts.json"
        command = [
            str(self.config.execution.runtime_python),
            str(PRECOMPUTE_CUTS_CLI),
            "--input-video",
            str(self.config.input_video),
            "--output",
            str(output),
        ]
        if self.config.inference.max_frames is not None:
            command.extend(
                ["--max-frames", str(self.config.inference.max_frames)]
            )
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
            completion.append(
                (return_code, time.perf_counter(), _utc_now())
            )

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
        output = self.output_root / "00_preflight" / "cuts.json"
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
        output_root = self.output_root / "03_overlay"
        output_root.mkdir(parents=True, exist_ok=True)
        requested: list[tuple[str, Path, Path | None]] = []
        if settings.raw:
            requested.append(("raw", artifacts.inference_sqlite, None))
        if settings.tracked:
            if artifacts.tracked_sqlite is None:
                raise OrchestrationError("tracked overlay has no tracked SQLite")
            requested.append(("tracked", artifacts.tracked_sqlite, None))
        if settings.final:
            if artifacts.final_sqlite is None:
                raise OrchestrationError("final overlay has no final SQLite")
            requested.append(
                (
                    "final",
                    artifacts.final_sqlite,
                    (
                        artifacts.inference_sqlite
                        if settings.final_include_faces
                        else None
                    ),
                )
            )
        if settings.faces:
            requested.append(("faces", artifacts.inference_sqlite, None))
        for mode, source, face_source in requested:
            output = output_root / f"{mode}.mp4"
            output_manifest = output.with_suffix(".json")
            artifact_name = f"overlay_{mode}"
            if self._can_resume_stage(
                f"overlay_{mode}",
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
                face_sqlite=face_source,
            )
            self._execute(
                f"overlay_{mode}",
                command,
                cpu_only=not settings.uses_nvenc,
                extra_pythonpath=OVERLAY_ROOT / "src",
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise OrchestrationError(f"overlay did not create output: {output}")
            self._publish_artifacts(
                {
                    artifact_name: output,
                    f"{artifact_name}_manifest": output_manifest,
                }
            )

    def _execute(
        self,
        stage: str,
        command: list[str],
        *,
        cpu_only: bool,
        extra_pythonpath: Path | None = None,
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
    ) -> None:
        current = dict(self.manifest.get("artifacts", {}))
        current.update({name: str(path) for name, path in artifacts.items()})
        self.manifest["artifacts"] = current
        if validation:
            checks = dict(self.manifest.get("validation", {}))
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
