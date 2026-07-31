"""MH0 implementation of the shared instance-segmentation contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from contracts import (
    FrameBatch,
    ModelDescriptor,
    SegmentationFrame,
    TaskType,
    segmentation_frame_from_rows,
)
from mask_geometry import mask_to_polygons

try:
    from .model import Mh0Runtime, infer
except ImportError:
    from model import Mh0Runtime, infer


def _normalize_result(result: Any):
    if isinstance(result, dict) and "ins_results" in result:
        result = result["ins_results"]
    boxes, segmentation = result
    if isinstance(segmentation, tuple):
        segmentation = segmentation[0]
    return boxes, segmentation


class Mh0Adapter:
    def __init__(self, runtime: Mh0Runtime, *, score_threshold: float) -> None:
        self.runtime = runtime
        self.score_threshold = float(score_threshold)
        self.class_names = tuple(
            getattr(runtime, "class_names", ("foreground",))
        )
        self.class_ids = tuple(
            getattr(runtime, "class_ids", tuple(range(len(self.class_names))))
        )
        self.descriptor = ModelDescriptor(
            model_id=f"dinov3_codino_mh0_{runtime.backend.replace('-', '_')}",
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation=(
                "tensorrt_partitioned_vitsplus_codino_mh0"
                if runtime.backend == "tensorrt-fast"
                else "pytorch_vitsplus_codino_mh0"
            )
            + ("_roi_classifier" if getattr(runtime, "classifier", None) else ""),
        )

    def infer_raw(self, batch: FrameBatch) -> Any:
        """Run the model while leaving CPU contract conversion to the caller."""

        return infer(self.runtime, batch.images)

    def convert_raw(
        self,
        batch: FrameBatch,
        raw_results: Any,
    ) -> tuple[SegmentationFrame, ...]:
        """Convert already-materialized model output to the shared contract."""

        output = []
        for frame, raw in zip(batch.frames, raw_results):
            box_results, segmentation_results = _normalize_result(raw)
            rows = []
            for class_index, boxes in enumerate(box_results):
                masks = (
                    segmentation_results[class_index]
                    if segmentation_results is not None
                    else ()
                )
                for index, box in enumerate(boxes):
                    score = float(box[4])
                    if score < self.score_threshold:
                        continue
                    x1 = float(np.clip(box[0], 0, frame.width))
                    y1 = float(np.clip(box[1], 0, frame.height))
                    x2 = float(np.clip(box[2], 0, frame.width))
                    y2 = float(np.clip(box[3], 0, frame.height))
                    polygons = []
                    if index < len(masks):
                        mask = np.asarray(masks[index], dtype=np.uint8)
                        if mask.ndim == 3:
                            mask = mask[:, :, 0]
                        polygons = mask_to_polygons(mask)
                    rows.append(
                        self._row(
                            box,
                            score=score,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            polygons=polygons,
                        )
                    )
            output.append(
                segmentation_frame_from_rows(
                    model=self.descriptor,
                    frame=frame,
                    rows=rows,
                )
            )
        return tuple(output)

    def _row(
        self,
        box,
        *,
        score: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        polygons,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "category_id": 0,
            "class_name": "foreground",
            "detector_score": score,
            "bbox_xyxy": (x1, y1, x2, y2),
            "polygons": polygons,
        }
        if len(box) >= 7 and self.class_names:
            class_index = int(round(float(box[5])))
            if not 0 <= class_index < len(self.class_names):
                raise ValueError(
                    f"MH0 classifier returned invalid class index {class_index}"
                )
            probability_count = min(
                len(self.class_names),
                max(0, len(box) - 7),
            )
            row.update(
                {
                    "classifier_class_id": int(self.class_ids[class_index]),
                    "classifier_class_name": self.class_names[class_index],
                    "class_score": float(box[6]),
                    "class_probs": [
                        float(value)
                        for value in box[7 : 7 + probability_count]
                    ],
                }
            )
        return row

    def predict(self, batch: FrameBatch) -> tuple[SegmentationFrame, ...]:
        return self.convert_raw(batch, self.infer_raw(batch))

    def synchronize(self) -> None:
        if (
            str(self.runtime.device).startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.runtime.device)

    def close(self) -> None:
        return None


__all__ = ["Mh0Adapter"]
