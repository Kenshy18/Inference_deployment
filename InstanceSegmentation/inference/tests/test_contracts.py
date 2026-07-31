from __future__ import annotations

import unittest

import numpy as np

from contracts import (
    BoundingBox,
    Classification,
    Frame,
    ModelDescriptor,
    DetectionFrame,
    FrameReference,
    Segmentation,
    TaskType,
    segmentation_frame_from_rows,
)


class ContractTest(unittest.TestCase):
    def test_rejects_invalid_box_and_score(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(10.0, 0.0, 9.0, 1.0)
        with self.assertRaises(ValueError):
            Classification(class_id=1, class_name="person", score=1.1)
        with self.assertRaises(ValueError):
            Frame(
                index=0,
                timestamp_sec=0,
                image=np.zeros((10, 10, 3), dtype=np.float32),
            )
        with self.assertRaises(ValueError):
            Segmentation(polygons=((0, 0, 1, 0, float("nan"), 1),))

    def test_task_and_result_type_must_match(self) -> None:
        segmentation_model = ModelDescriptor(
            model_id="segmenter",
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation="tests.Segmenter",
        )
        with self.assertRaises(ValueError):
            DetectionFrame(
                model=segmentation_model,
                frame=FrameReference(
                    index=0, timestamp_sec=0, width=10, height=10
                ),
                detections=(),
            )

    def test_segmentation_rows_become_contract_objects(self) -> None:
        model = ModelDescriptor(
            model_id="test_segmenter",
            task=TaskType.INSTANCE_SEGMENTATION,
            implementation="tests.FakeSegmenter",
        )
        frame = Frame(
            index=7,
            timestamp_sec=0.25,
            image=np.zeros((20, 30, 3), dtype=np.uint8),
        )
        result = segmentation_frame_from_rows(
            model=model,
            frame=frame,
            rows=[
                {
                    "bbox_xyxy": [1, 2, 11, 12],
                    "category_id": 3,
                    "class_name": "foreground",
                    "detector_score": 0.8,
                    "classifier_class_id": 11,
                    "classifier_class_name": "class-b",
                    "class_score": 0.7,
                    "class_probs": [0.3, 0.7],
                    "polygons": [[1, 2, 11, 2, 11, 12, 1, 12]],
                }
            ],
        )

        self.assertEqual(result.model.task, TaskType.INSTANCE_SEGMENTATION)
        self.assertEqual(result.frame.index, 7)
        self.assertEqual(result.frame.width, 30)
        self.assertEqual(len(result.instances), 1)
        instance = result.instances[0]
        self.assertEqual(instance.detection.bbox.width, 10.0)
        self.assertEqual(instance.detection.bbox.height, 10.0)
        self.assertEqual(instance.detection.class_id, 3)
        self.assertEqual(instance.detection.class_name, "foreground")
        self.assertEqual(instance.detection.classification.class_id, 11)
        self.assertEqual(
            instance.detection.classification.class_name,
            "class-b",
        )
        self.assertEqual(instance.detection.classification.score, 0.7)
        self.assertEqual(len(instance.segmentation.polygons), 1)


if __name__ == "__main__":
    unittest.main()
