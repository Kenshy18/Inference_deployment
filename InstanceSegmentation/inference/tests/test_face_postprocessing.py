from __future__ import annotations

import unittest

import torch

from rtdetr_head_face.postprocessing import filter_detections


class FaceDetectionPostprocessingTest(unittest.TestCase):
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
