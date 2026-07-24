"""RT-DETR Head/Face implementation of the object-detection contract."""

from __future__ import annotations

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

from . import model as runtime
from .postprocessing import filter_detections


@dataclass(frozen=True, slots=True)
class RtDetrSettings:
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    size: tuple[int, int] | None = None
    precision: str = "auto"
    compile: bool = False
    channels_last: bool = True
    tf32: bool = True
    warmup_iterations: int = 3
    class_filter: frozenset[int] | None = None
    score_threshold: float = 0.35
    nms_threshold: float = 0.65
    max_detections: int = 300
    max_area_ratio: float = 1.0


class RtDetrHeadFaceAdapter:
    descriptor = ModelDescriptor(
        model_id="rtdetr_head_face",
        task=TaskType.OBJECT_DETECTION,
        implementation="pytorch_rtdetr_head_face",
    )

    def __init__(self, settings: RtDetrSettings, *, batch_size: int) -> None:
        self.settings = settings
        self.batch_size = int(batch_size)
        self.device = torch.device(settings.device)
        self.precision_dtype, self.precision_name = runtime.select_precision(
            self.device, settings.precision
        )
        self.autocast_dtype = (
            self.precision_dtype
            if self.device.type == "cuda" and self.precision_name != "fp32"
            else None
        )
        runtime.configure_torch(self.device, enable_tf32=settings.tf32)
        self.model, self.input_size = runtime.build_model(
            settings.config_path,
            settings.checkpoint_path,
            self.device,
            settings.size,
            use_channels_last=settings.channels_last,
            use_compile=settings.compile,
        )
        runtime.warmup_model(
            self.model,
            self.input_size,
            self.batch_size,
            self.device,
            self.precision_dtype,
            self.autocast_dtype,
            settings.warmup_iterations,
            settings.channels_last,
        )

    @staticmethod
    def _detections_from_tensors(labels, boxes, scores) -> tuple[Detection, ...]:
        output: list[Detection] = []
        for label, box, score in zip(
            labels.tolist(), boxes.tolist(), scores.tolist()
        ):
            class_id = int(label)
            output.append(
                Detection(
                    class_id=class_id,
                    class_name=runtime.LABEL_NAMES.get(class_id, str(class_id)),
                    score=float(score),
                    bbox=BoundingBox(*(float(value) for value in box)),
                )
            )
        return tuple(output)

    def predict(self, batch: FrameBatch) -> tuple[DetectionFrame, ...]:
        valid_count = len(batch)
        model_frames = batch.images
        if self.settings.compile and valid_count < self.batch_size:
            model_frames = [
                *model_frames,
                *[model_frames[-1]] * (self.batch_size - valid_count),
            ]
        labels_batch, boxes_batch, scores_batch = runtime.run_batch(
            self.model,
            model_frames,
            self.input_size,
            self.device,
            self.precision_dtype,
            self.autocast_dtype,
            self.settings.channels_last,
        )
        results: list[DetectionFrame] = []
        for local_index, frame in enumerate(batch.frames):
            labels, boxes, scores = filter_detections(
                labels_batch[local_index],
                boxes_batch[local_index],
                scores_batch[local_index],
                score_threshold=self.settings.score_threshold,
                nms_threshold=self.settings.nms_threshold,
                max_detections=self.settings.max_detections,
                max_area_ratio=self.settings.max_area_ratio,
                class_filter=(
                    None
                    if self.settings.class_filter is None
                    else set(self.settings.class_filter)
                ),
                frame_area=float(frame.width * frame.height),
            )
            results.append(
                DetectionFrame(
                    model=self.descriptor,
                    frame=FrameReference.from_frame(frame),
                    detections=self._detections_from_tensors(
                        labels, boxes, scores
                    ),
                )
            )
        return tuple(results)

    def synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def close(self) -> None:
        return None


__all__ = ["RtDetrHeadFaceAdapter", "RtDetrSettings"]
