"""Co-DINO implementation of the shared segmentation contract."""

from __future__ import annotations

import torch

from contracts import (
    FrameBatch,
    ModelDescriptor,
    SegmentationFrame,
    TaskType,
    segmentation_frame_from_rows,
)

try:
    from .classifier import infer_batch_with_classifier
    from .model import (
        CoDinoRuntime,
        VideoInferenceSettings,
        infer_batch_without_classifier,
    )
    from .postprocessing import (
        detections_to_rows,
        normalize_result,
        restore_boxes,
        restore_segmentations,
    )
except ImportError:
    from classifier import infer_batch_with_classifier
    from model import (
        CoDinoRuntime,
        VideoInferenceSettings,
        infer_batch_without_classifier,
    )
    from postprocessing import (
        detections_to_rows,
        normalize_result,
        restore_boxes,
        restore_segmentations,
    )


class CoDinoAdapter:
    def __init__(
        self,
        runtime: CoDinoRuntime,
        settings: VideoInferenceSettings,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.descriptor = ModelDescriptor(
            model_id=f"dinov3_codino_{runtime.backend}",
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation=(
                "pytorch_codino_roi_classifier"
                if runtime.backend == "pytorch"
                else "tensorrt_partitioned_codino_roi_classifier"
            ),
        )
        if settings.batch_size != runtime.fixed_batch_size:
            raise ValueError(
                f"Co-DINO {runtime.backend} backend requires batch_size "
                f"{runtime.fixed_batch_size}, got {settings.batch_size}"
            )

    def predict(self, batch: FrameBatch) -> tuple[SegmentationFrame, ...]:
        valid_count = len(batch)
        frames = batch.images
        padded = frames
        if valid_count < self.runtime.fixed_batch_size:
            padded = [
                *frames,
                *[frames[-1]] * (self.runtime.fixed_batch_size - valid_count),
            ]
        if self.runtime.classifier is None:
            raw_results = infer_batch_without_classifier(
                self.runtime.model,
                padded,
                amp=self.settings.amp,
                target_size=self.runtime.target_size,
            )
        else:
            raw_results = infer_batch_with_classifier(
                self.runtime.model,
                padded,
                amp=self.settings.amp,
                target_size=self.runtime.target_size,
                classifier=self.runtime.classifier,
                num_classifier_classes=len(self.runtime.class_names),
            )
        results: list[SegmentationFrame] = []
        for frame, raw_result in zip(batch.frames, raw_results[:valid_count]):
            box_results, segmentation_results = normalize_result(raw_result)
            original_shape = frame.image.shape[:2]
            restored_boxes = restore_boxes(
                box_results, original_shape, self.runtime.target_size
            )
            restored_masks = restore_segmentations(
                segmentation_results, original_shape, self.runtime.target_size
            )
            rows = detections_to_rows(
                restored_boxes,
                restored_masks,
                class_names=list(self.runtime.class_names),
                class_ids=list(self.runtime.class_ids),
                score_threshold=self.settings.score_threshold,
            )
            results.append(
                segmentation_frame_from_rows(
                    model=self.descriptor,
                    frame=frame,
                    rows=rows,
                )
            )
        return tuple(results)

    def synchronize(self) -> None:
        if next(self.runtime.model.parameters()).is_cuda:
            torch.cuda.synchronize()

    def close(self) -> None:
        return None


__all__ = ["CoDinoAdapter"]
