"""DINOv3 Cascade implementation of the shared segmentation contract."""

from __future__ import annotations

import torch

from contracts import (
    FrameBatch,
    ModelDescriptor,
    SegmentationFrame,
    TaskType,
    segmentation_frame_from_rows,
)

from .classifier.runtime import classify_outputs
from .instance_segmentation.postprocessing import restore_source_coordinates
from .instance_segmentation.preprocessing import (
    materialize_model_inputs,
    prepare_frame,
)
from .output import drop_auxiliary_instance_fields, instances_to_rows
from .runtime import Dinov3Runtime, autocast_context
from .runtime_contracts import VideoInferenceSettings


class Dinov3CascadeAdapter:
    descriptor = ModelDescriptor(
        model_id="dinov3_cascade",
        task=TaskType.INSTANCE_SEGMENTATION,
        implementation="tensorrt_dinov3_cascade_roi_classifier",
    )

    def __init__(
        self,
        runtime: Dinov3Runtime,
        settings: VideoInferenceSettings,
    ) -> None:
        self.runtime = runtime
        self.settings = settings

    def predict(self, batch: FrameBatch) -> tuple[SegmentationFrame, ...]:
        pin_memory = self.settings.device.startswith("cuda") and torch.cuda.is_available()
        prepared = [
            prepare_frame(
                frame.image,
                self.runtime.target_size,
                pin_memory=pin_memory,
            )
            for frame in batch.frames
        ]
        inputs = materialize_model_inputs(prepared, self.settings.device)
        with torch.inference_mode(), autocast_context(self.settings):
            outputs = self.runtime.segmenter(inputs)
            if self.runtime.classifier is not None:
                classify_outputs(
                    self.runtime.classifier,
                    outputs,
                    self.runtime.target_size,
                    pad_to=max(0, self.settings.classifier_pad_to),
                    pad_multiple=max(0, self.settings.classifier_pad_multiple),
                )
        results: list[SegmentationFrame] = []
        for frame, output, prepared_frame in zip(batch.frames, outputs, prepared):
            instances = output["instances"]
            drop_auxiliary_instance_fields(instances)
            instances = restore_source_coordinates(
                instances,
                prepared_frame.letterbox,
                prepared_frame.original_height,
                prepared_frame.original_width,
                self.runtime.target_size,
            ).to("cpu")
            rows = instances_to_rows(
                instances,
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
        if self.settings.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def close(self) -> None:
        return None


__all__ = ["Dinov3CascadeAdapter"]
