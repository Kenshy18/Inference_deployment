"""Face DINO v2 implementation of the shared object-detection contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from contracts import (
    BoundingBox,
    Detection,
    DetectionFrame,
    FrameBatch,
    FrameReference,
    ModelDescriptor,
    TaskType,
)

from .model import FaceDinoRuntime, build_runtime


LABEL_NAMES = {1: "Head", 2: "Face"}


def ellipse_xyxy(ellipse: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned envelope of cx, cy, major radius, minor radius, theta."""
    cx, cy, major, minor, theta = ellipse
    cosine = math.cos(theta)
    sine = math.sin(theta)
    extent_x = math.sqrt((major * cosine) ** 2 + (minor * sine) ** 2)
    extent_y = math.sqrt((major * sine) ** 2 + (minor * cosine) ** 2)
    return (
        cx - extent_x,
        cy - extent_y,
        cx + extent_x,
        cy + extent_y,
    )


@dataclass(frozen=True, slots=True)
class FaceDinoV2Settings:
    source_root: Path
    checkpoint: Path
    trt_bundle: Path
    device: str = "cuda:0"
    score_threshold: float = 0.30
    warmup_iterations: int = 3
    classes: frozenset[int] | None = None
    verify: str = "engines"


class FaceDinoV2Adapter:
    descriptor = ModelDescriptor(
        model_id="face_dino_v2",
        task=TaskType.OBJECT_DETECTION,
        implementation="tensorrt_vitsplus_compact_codino_face_attributes",
    )

    def __init__(
        self,
        settings: FaceDinoV2Settings,
        runtime: FaceDinoRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime or build_runtime(
            source_root=settings.source_root,
            checkpoint=settings.checkpoint,
            trt_bundle=settings.trt_bundle,
            device=settings.device,
            score_threshold=settings.score_threshold,
            warmup_iterations=settings.warmup_iterations,
            verify=settings.verify,
        )

    @staticmethod
    def parse_classes(values: list[str] | tuple[str, ...] | None):
        if not values:
            return None
        lookup = {name.lower(): class_id for class_id, name in LABEL_NAMES.items()}
        selected = set()
        for value in values:
            selected.add(
                lookup[value.lower()]
                if value.lower() in lookup
                else int(value)
            )
        invalid = selected - set(LABEL_NAMES)
        if invalid:
            raise ValueError(f"unsupported Face DINO classes: {sorted(invalid)}")
        return frozenset(selected)

    def _allowed(self, class_id: int) -> bool:
        return (
            self.settings.classes is None
            or class_id in self.settings.classes
        )

    def predict(self, batch: FrameBatch) -> tuple[DetectionFrame, ...]:
        raw_results = self.runtime.predict(batch.images)
        counts = [len(raw["boxes"]) for raw in raw_results]
        packed = [
            torch.cat(
                (
                    raw["boxes"].detach().float(),
                    raw["scores"].detach().float()[:, None],
                    raw["face_scores"].detach().float()[:, None],
                    raw["face_present"].detach().float()[:, None],
                    raw["ellipses"].detach().float(),
                ),
                dim=1,
            )
            for raw in raw_results
        ]
        packed_rows = (
            torch.cat(packed, dim=0).cpu().tolist()
            if sum(counts)
            else []
        )
        frames: list[DetectionFrame] = []
        offset = 0
        for frame, count in zip(batch.frames, counts):
            detections: list[Detection] = []
            for row in packed_rows[offset : offset + count]:
                box = row[0:4]
                head_score = row[4]
                face_score = row[5]
                present = row[6] >= 0.5
                ellipse = row[7:12]
                if self._allowed(1):
                    detections.append(
                        Detection(
                            class_id=1,
                            class_name="Head",
                            score=float(head_score),
                            bbox=BoundingBox(
                                max(0.0, min(frame.width, float(box[0]))),
                                max(0.0, min(frame.height, float(box[1]))),
                                max(0.0, min(frame.width, float(box[2]))),
                                max(0.0, min(frame.height, float(box[3]))),
                            ),
                            source="head_detection",
                        )
                    )
                if present and self._allowed(2):
                    x1, y1, x2, y2 = ellipse_xyxy(ellipse)
                    detections.append(
                        Detection(
                            class_id=2,
                            class_name="Face",
                            score=float(face_score),
                            bbox=BoundingBox(
                                max(0.0, min(frame.width, x1)),
                                max(0.0, min(frame.height, y1)),
                                max(0.0, min(frame.width, x2)),
                                max(0.0, min(frame.height, y2)),
                            ),
                            source="ellipse_detection",
                        )
                    )
            offset += count
            frames.append(
                DetectionFrame(
                    model=self.descriptor,
                    frame=FrameReference.from_frame(frame),
                    detections=tuple(detections),
                )
            )
        return tuple(frames)

    def synchronize(self) -> None:
        self.runtime.synchronize()

    def close(self) -> None:
        self.runtime.close()


__all__ = [
    "LABEL_NAMES",
    "FaceDinoV2Adapter",
    "FaceDinoV2Settings",
    "ellipse_xyxy",
]
