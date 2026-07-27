"""Lightweight model registry; importing it never initializes GPU frameworks."""

from __future__ import annotations

from dataclasses import dataclass

from contracts import TaskType


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    model_id: str
    task: TaskType
    package: str
    adapter_class: str
    backends: tuple[str, ...]
    default_backend: str
    backend_cli_argument: str | None = None


_MODELS = {
    item.model_id: item
    for item in (
        ModelRegistration(
            "dinov3_codino",
            TaskType.INSTANCE_SEGMENTATION,
            "dinov3_codino.adapter",
            "CoDinoAdapter",
            ("tensorrt-fast", "pytorch"),
            "tensorrt-fast",
            "--backend",
        ),
        ModelRegistration(
            "dinov3_codino_mh0",
            TaskType.INSTANCE_SEGMENTATION,
            "dinov3_codino_mh0.adapter",
            "Mh0Adapter",
            ("tensorrt-fast", "pytorch"),
            "tensorrt-fast",
            "--backend",
        ),
        ModelRegistration(
            "dinov3_cascade",
            TaskType.INSTANCE_SEGMENTATION,
            "dinov3_cascade.adapter",
            "Dinov3CascadeAdapter",
            ("tensorrt-backbone",),
            "tensorrt-backbone",
        ),
        ModelRegistration(
            "eva02_cascade",
            TaskType.INSTANCE_SEGMENTATION,
            "eva02_cascade.adapter",
            "Eva02CascadeAdapter",
            ("tensorrt-backbone", "pytorch"),
            "tensorrt-backbone",
            "--backend",
        ),
        ModelRegistration(
            "rtdetr_head_face",
            TaskType.OBJECT_DETECTION,
            "rtdetr_head_face.adapter",
            "RtDetrHeadFaceAdapter",
            ("pytorch",),
            "pytorch",
        ),
        ModelRegistration(
            "face_dino_v2",
            TaskType.OBJECT_DETECTION,
            "face_dino_v2.adapter",
            "FaceDinoV2Adapter",
            ("tensorrt-fast",),
            "tensorrt-fast",
        ),
    )
}


def get_model(model_id: str) -> ModelRegistration:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {model_id!r}; available={sorted(_MODELS)}"
        ) from exc


def list_models(task: TaskType | None = None) -> tuple[ModelRegistration, ...]:
    values = tuple(_MODELS.values())
    if task is None:
        return values
    return tuple(item for item in values if item.task is task)


__all__ = ["ModelRegistration", "get_model", "list_models"]
