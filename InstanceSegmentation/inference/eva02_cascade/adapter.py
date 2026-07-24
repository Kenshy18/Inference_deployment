"""EVA-02 implementation of the shared instance-segmentation contract."""

from __future__ import annotations

import torch

from contracts import (
    FrameBatch,
    ModelDescriptor,
    SegmentationFrame,
    TaskType,
    segmentation_frame_from_rows,
)

from .classifier.runtime import classify_instance_batch
from .instance_segmentation.postprocessing import restore_source_coordinates
from .instance_segmentation.preprocessing import prepare_batch_inputs
from .output import drop_auxiliary_fields, instances_to_rows
from .runtime import Eva02Runtime, run_segmenter
from .runtime_contracts import VideoInferenceSettings


class Eva02CascadeAdapter:
    def __init__(
        self,
        runtime: Eva02Runtime,
        settings: VideoInferenceSettings,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.descriptor = ModelDescriptor(
            model_id="eva02_cascade",
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation=(
                "tensorrt_backbone_eva02_cascade_roi_classifier"
                if runtime.backend == "tensorrt-backbone"
                else "pytorch_eva02_cascade_roi_classifier"
            ),
        )

    @torch.inference_mode()
    def predict(self, batch: FrameBatch) -> tuple[SegmentationFrame, ...]:
        frames_bgr = batch.images
        inputs, restore_metadata = prepare_batch_inputs(
            frames_bgr,
            self.runtime.target_size,
            self.settings.device,
            gpu_preprocess_float=self.settings.gpu_preprocess_float,
            model_half=self.settings.model_half,
            pin_inputs=self.settings.pin_inputs,
            pack_inputs=self.settings.pack_inputs,
        )
        outputs = run_segmenter(
            self.runtime.segmenter, inputs, settings=self.settings
        )
        restored_instances: list[object] = []
        for output, (
            original_height,
            original_width,
            letterbox,
        ) in zip(outputs, restore_metadata):
            restored_instances.append(
                restore_source_coordinates(
                    output["instances"],
                    letterbox,
                    original_height,
                    original_width,
                    self.runtime.target_size,
                    use_inplace_boxes=self.settings.use_inplace_boxes,
                    use_raw_to_orig_masks=self.settings.use_raw_to_orig_masks,
                )
            )
        if restored_instances:
            classify_instance_batch(
                self.runtime.classifier,
                restored_instances,
                width=batch.frames[0].width,
                height=batch.frames[0].height,
                meta_feature_set=self.runtime.meta_feature_set,
                feature_l2norm=self.runtime.feature_l2norm,
                max_instances_per_batch=self.settings.classifier_batch_size,
            )
        results: list[SegmentationFrame] = []
        for frame, instances in zip(batch.frames, restored_instances):
            drop_auxiliary_fields(instances)
            rows = instances_to_rows(
                instances.to("cpu"),
                class_names=self.runtime.class_names,
                class_ids=self.runtime.class_ids,
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


__all__ = ["Eva02CascadeAdapter"]
