from __future__ import annotations

import unittest

import torch

from rtdetr_head_face.postprocessing import filter_detections
from rtdetr_head_face.preprocessing import make_batch


class FaceDetectionPostprocessingTest(unittest.TestCase):
    def test_batch_spatial_metadata_uses_width_height_order(self) -> None:
        frame = torch.zeros((1080, 1920, 3), dtype=torch.uint8).numpy()

        batch, original_sizes, padding, scale, input_sizes = make_batch(
            [frame],
            (512, 896),
            torch.device("cpu"),
            None,
            False,
        )

        self.assertEqual(tuple(batch.shape), (1, 3, 512, 896))
        self.assertEqual(original_sizes.tolist(), [[1920.0, 1080.0]])
        self.assertEqual(input_sizes.tolist(), [[896.0, 512.0]])
        self.assertEqual(padding.tolist(), [[0.0, 4.0]])
        self.assertAlmostEqual(scale.item(), 896.0 / 1920.0, places=6)

    def test_filters_score_class_area_and_applies_nms(self) -> None:
        labels = torch.tensor([2, 2, 1, 2])
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
                [0.0, 0.0, 100.0, 100.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.95, 0.99])

        kept_labels, kept_boxes, kept_scores = filter_detections(
            labels,
            boxes,
            scores,
            score_threshold=0.5,
            nms_threshold=0.5,
            max_detections=10,
            max_area_ratio=0.5,
            class_filter={2},
            frame_area=10_000.0,
        )

        self.assertEqual(kept_labels.tolist(), [2])
        self.assertEqual(kept_boxes.tolist(), [[0.0, 0.0, 10.0, 10.0]])
        self.assertAlmostEqual(kept_scores.item(), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
