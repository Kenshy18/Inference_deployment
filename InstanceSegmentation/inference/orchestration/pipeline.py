"""Execute selected model families and atomically publish one SQLite output."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from contracts import TaskType
from persistence import ImportedModelSummary, UnifiedSqliteWriter
from registry import ModelRegistration, get_model

from .config import OrchestrationRequest
from .model_process import (
    build_invocation,
    execute_invocation,
    execute_invocations_parallel,
)


@dataclass(frozen=True, slots=True)
class OrchestrationSummary:
    output_path: Path
    mode: str
    frames: int
    detections: int
    classifications: int
    segmentations: int
    face_observations: int
    face_keypoints: int
    models: tuple[ImportedModelSummary, ...]


def run_orchestrated_inference(
    request: OrchestrationRequest,
) -> OrchestrationSummary:
    input_path = request.input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input video not found: {input_path}")
    output_path = request.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not request.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    selected: list[tuple[str, ModelRegistration]] = []
    if request.mode.uses_segmentation:
        assert request.segmentation_model is not None
        registration = get_model(request.segmentation_model)
        if registration.task is not TaskType.INSTANCE_SEGMENTATION:
            raise ValueError(
                f"{registration.model_id} is not an instance-segmentation model"
            )
        selected.append(("instance_segmentation", registration))
    if request.mode.uses_face_detection:
        registration = get_model(request.face_model)
        if registration.task is not TaskType.OBJECT_DETECTION:
            raise ValueError(
                f"{registration.model_id} is not an object-detection model"
            )
        selected.append(("face_detection", registration))

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.orchestrating-",
        dir=output_path.parent,
    ) as directory:
        workspace = Path(directory)
        invocations = [
            build_invocation(
                registration,
                role=role,
                output_path=workspace / f"{index:02d}-{role}.sqlite",
                request=request,
            )
            for index, (role, registration) in enumerate(selected)
        ]
        if request.parallel_models and len(invocations) > 1:
            execute_invocations_parallel(
                invocations,
                stagger_seconds=request.parallel_model_stagger_seconds,
            )
        else:
            for invocation in invocations:
                execute_invocation(invocation)

        staging = workspace / "unified.sqlite"
        writer = UnifiedSqliteWriter(
            staging,
            input_path=input_path,
            mode=request.mode.value,
            safe=not request.fast_sqlite,
        )
        imported: list[ImportedModelSummary] = []
        try:
            writer.set_run_metadata(
                {
                    "segmentation_model": request.segmentation_model,
                    "segmentation_backend": request.segmentation_backend,
                    "face_model": (
                        request.face_model if request.mode.uses_face_detection else None
                    ),
                    "face_classes": request.face_classes,
                    "face_trt_bundle": (
                        str(request.face_trt_bundle.expanduser().resolve())
                        if request.face_trt_bundle is not None
                        else None
                    ),
                    "device": request.device,
                    "max_frames": request.max_frames,
                    "parallel_models": request.parallel_models,
                    "parallel_model_stagger_seconds": (
                        request.parallel_model_stagger_seconds
                    ),
                }
            )
            for invocation in invocations:
                imported.append(
                    writer.import_model_output(
                        invocation.output_path,
                        role=invocation.role,
                        model_id=invocation.registration.model_id,
                        backend=invocation.backend,
                    )
                )
        finally:
            writer.close()
        os.replace(staging, output_path)

    frames = max((summary.frames for summary in imported), default=0)
    result = OrchestrationSummary(
        output_path=output_path,
        mode=request.mode.value,
        frames=frames,
        detections=sum(item.detections for item in imported),
        classifications=sum(item.classifications for item in imported),
        segmentations=sum(item.segmentations for item in imported),
        face_observations=sum(item.face_observations for item in imported),
        face_keypoints=sum(item.face_keypoints for item in imported),
        models=tuple(imported),
    )
    print(
        f"[orchestrator] saved {result.mode} SQLite: {output_path}",
        flush=True,
    )
    print(
        f"[orchestrator] frames={result.frames} "
        f"detections={result.detections} "
        f"classifications={result.classifications} "
        f"segmentations={result.segmentations} "
        f"face_observations={result.face_observations} "
        f"face_keypoints={result.face_keypoints}",
        flush=True,
    )
    return result


__all__ = ["OrchestrationSummary", "run_orchestrated_inference"]
