"""Validated execution of a configured feature-stage graph."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from .config import PipelineConfig
from .live_preview import active_postprocess_preview
from .progress import StageGraphProgress
from contracts.artifacts import validate_artifact
from contracts.stages import ProgressCallback, StageContext
from .registry import create_stage


_LIVE_PREVIEW_EXCLUDED = {
    # Packaging/integration has no visual geometry of its own.  Keeping it off
    # the Live path also avoids implying that a second mask algorithm ran.
    "artifacts.integrated_sqlite",
    "artifacts.legacy_sqlite",
    "artifacts.union_sqlite",
    "artifacts.validate",
    "face_privacy.merge",
}


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        output_dir: Path,
        *,
        emit_progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.emit_progress = bool(emit_progress)
        self.progress_callback = progress_callback
        self._validated_artifacts: set[tuple[str, Path, int, int]] = set()

    def _validate_once(self, name: str, path: Path) -> None:
        """Validate an immutable stage artifact once for its current file state."""

        resolved = Path(path).resolve()
        stat = resolved.stat()
        identity = (str(name), resolved, stat.st_size, stat.st_mtime_ns)
        if identity in self._validated_artifacts:
            return
        validate_artifact(name, resolved)
        self._validated_artifacts.add(identity)

    def run(self, initial_artifacts: Mapping[str, Path]) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {str(name): Path(path) for name, path in initial_artifacts.items()}
        history: list[dict[str, Any]] = []
        enabled = tuple(spec for spec in self.config.stages if spec.enabled)
        progress = (
            StageGraphProgress(len(enabled) + 1)
            if self.emit_progress and enabled
            else None
        )
        if progress is not None:
            progress.start()
        try:
            if progress is not None:
                progress.begin_stage(0, "preparing:input-validation")
            for name, path in sorted(artifacts.items()):
                self._validate_once(name, path)
            if progress is not None:
                progress.finish_stage(1, "preparing:complete")
            enabled_position = 1
            for position, spec in enumerate(self.config.stages):
                if not spec.enabled:
                    continue
                stage = create_stage(spec.implementation, spec.options)
                missing = sorted(stage.requires - artifacts.keys())
                if missing:
                    raise ValueError(
                        f"stage {spec.id!r} ({stage.name}) is missing artifacts: "
                        f"{missing}; available: {sorted(artifacts)}"
                    )
                if progress is not None:
                    progress.begin_stage(
                        enabled_position,
                        f"{spec.id}:input-validation",
                    )
                for name in sorted(stage.requires):
                    self._validate_once(name, artifacts[name])
                if progress is not None:
                    progress.activity(f"{spec.id}:running")
                preview = (
                    None
                    if spec.implementation in _LIVE_PREVIEW_EXCLUDED
                    else active_postprocess_preview()
                )
                if preview is not None:
                    preview.stage_started(spec.id, stage.name)
                stage_dir = self.output_dir / f"{position:02d}_{spec.id}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                context = StageContext(
                    pipeline_name=self.config.name,
                    stage_id=spec.id,
                    output_dir=self.output_dir,
                    stage_dir=stage_dir,
                    artifacts=dict(artifacts),
                    progress_callback=self._stage_progress_callback(progress),
                )
                started = time.perf_counter()
                result = stage.run(context)
                if progress is not None:
                    progress.activity(f"{spec.id}:output-validation")
                missing_outputs = sorted(stage.provides - result.artifacts.keys())
                if missing_outputs:
                    raise RuntimeError(
                        f"stage {spec.id!r} did not provide declared artifacts: "
                        f"{missing_outputs}"
                    )
                undeclared_outputs = sorted(result.artifacts.keys() - stage.provides)
                if undeclared_outputs:
                    raise RuntimeError(
                        f"stage {spec.id!r} returned undeclared artifacts: "
                        f"{undeclared_outputs}"
                    )
                outputs = {
                    str(name): Path(path) for name, path in result.artifacts.items()
                }
                absent_files = sorted(
                    name for name, path in outputs.items() if not path.exists()
                )
                if absent_files:
                    raise RuntimeError(
                        f"stage {spec.id!r} returned missing files: {absent_files}"
                    )
                overwritten = sorted(outputs.keys() & artifacts.keys())
                if overwritten:
                    raise RuntimeError(
                        f"stage {spec.id!r} attempted to overwrite artifacts: "
                        f"{overwritten}"
                    )
                stage_root = stage_dir.resolve()
                outside_stage_dir = sorted(
                    name
                    for name, path in outputs.items()
                    if not path.resolve().is_relative_to(stage_root)
                )
                if outside_stage_dir:
                    raise RuntimeError(
                        f"stage {spec.id!r} wrote outside its stage directory: "
                        f"{outside_stage_dir}"
                    )
                for name, path in sorted(outputs.items()):
                    self._validate_once(name, path)
                if preview is not None:
                    preview.stage_artifacts(
                        spec.id,
                        stage.name,
                        outputs,
                        result.metadata,
                    )
                artifacts.update(outputs)
                history.append(
                    {
                        "id": spec.id,
                        "implementation": spec.implementation,
                        "name": stage.name,
                        "requires": sorted(stage.requires),
                        "provides": sorted(stage.provides),
                        "artifacts": {
                            name: str(path) for name, path in sorted(outputs.items())
                        },
                        "metadata": dict(result.metadata),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                self._write_manifest(artifacts, history, complete=False)
                enabled_position += 1
                if progress is not None:
                    progress.finish_stage(
                        enabled_position,
                        f"{spec.id}:complete",
                    )
        except BaseException as exc:
            if progress is not None:
                progress.fail(f"{type(exc).__name__}")
            raise
        if progress is not None:
            progress.complete()
        return self._write_manifest(artifacts, history, complete=True)

    def _write_manifest(
        self,
        artifacts: Mapping[str, Path],
        history: list[dict[str, Any]],
        *,
        complete: bool,
    ) -> dict[str, Any]:
        manifest = {
            "schema_version": 1,
            "pipeline": self.config.name,
            "complete": complete,
            "stages": history,
            "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
        }
        (self.output_dir / "pipeline_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _stage_progress_callback(
        self,
        progress: StageGraphProgress | None,
    ) -> ProgressCallback | None:
        callbacks: list[ProgressCallback] = []
        if progress is not None:
            callbacks.append(
                lambda detail, fraction, fps: progress.activity(
                    detail,
                    fraction,
                    fps,
                )
            )
        if self.progress_callback is not None:
            callbacks.append(self.progress_callback)
        if not callbacks:
            return None

        def publish(
            detail: str,
            fraction: float | None,
            fps: float | None,
        ) -> None:
            for callback in callbacks:
                callback(detail, fraction, fps)

        return publish
