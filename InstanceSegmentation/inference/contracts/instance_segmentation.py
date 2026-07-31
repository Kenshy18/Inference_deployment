"""Instance-segmentation task input/output rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .classification import Classification
from .common import Frame, FrameBatch, FrameReference, ModelDescriptor
from .object_detection import BoundingBox, Detection


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Polygon masks in source-image pixel coordinates."""

    polygons: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        for polygon in self.polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError(
                    "each segmentation polygon must contain at least 3 XY points"
                )
            if any(not math.isfinite(value) for value in polygon):
                raise ValueError("segmentation polygon coordinates must be finite")


@dataclass(frozen=True, slots=True)
class SegmentationInstance:
    detection: Detection
    segmentation: Segmentation


@dataclass(frozen=True, slots=True)
class SegmentationFrame:
    model: ModelDescriptor
    frame: FrameReference
    instances: tuple[SegmentationInstance, ...]

    def __post_init__(self) -> None:
        from .common import TaskType

        if self.model.task is not TaskType.INSTANCE_SEGMENTATION:
            raise ValueError(
                "SegmentationFrame requires an instance-segmentation model"
            )


def segmentation_frame_from_rows(
    *,
    model: ModelDescriptor,
    frame: Frame,
    rows: Sequence[Mapping[str, object]],
) -> SegmentationFrame:
    """Normalize validated family rows at the temporary adapter boundary."""

    instances: list[SegmentationInstance] = []
    for row in rows:
        raw_box = row.get("bbox_xyxy")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise ValueError("segmentation row must contain bbox_xyxy[4]")
        class_id = int(row.get("category_id", row.get("class_id", 0)))
        class_name = str(row.get("class_name", row.get("label", class_id)))
        detector_score = float(row.get("detector_score", row.get("score", 0.0)))
        raw_class_score = row.get("class_score", row.get("cls_score"))
        raw_probabilities = row.get("class_probs")
        classification = None
        if raw_class_score is not None:
            classifier_class_id = int(
                row.get("classifier_class_id", class_id)
            )
            classifier_class_name = str(
                row.get("classifier_class_name", class_name)
            )
            probabilities = (
                tuple(float(value) for value in raw_probabilities)
                if isinstance(raw_probabilities, (list, tuple))
                else None
            )
            classification = Classification(
                class_id=classifier_class_id,
                class_name=classifier_class_name,
                score=float(raw_class_score),
                probabilities=probabilities,
            )
        raw_polygons = row.get("polygons", row.get("segmentation", ()))
        polygons = (
            tuple(tuple(float(value) for value in polygon) for polygon in raw_polygons)
            if isinstance(raw_polygons, (list, tuple))
            else ()
        )
        instances.append(
            SegmentationInstance(
                detection=Detection(
                    class_id=class_id,
                    class_name=class_name,
                    score=detector_score,
                    bbox=BoundingBox(*(float(value) for value in raw_box)),
                    classification=classification,
                ),
                segmentation=Segmentation(polygons=polygons),
            )
        )
    return SegmentationFrame(
        model=model,
        frame=FrameReference.from_frame(frame),
        instances=tuple(instances),
    )


@runtime_checkable
class InstanceSegmentationAdapter(Protocol):
    descriptor: ModelDescriptor

    def predict(self, batch: FrameBatch) -> Sequence[SegmentationFrame]:
        ...

    def synchronize(self) -> None:
        ...

    def close(self) -> None:
        ...


__all__ = [
    "InstanceSegmentationAdapter",
    "Segmentation",
    "SegmentationFrame",
    "SegmentationInstance",
    "segmentation_frame_from_rows",
]
