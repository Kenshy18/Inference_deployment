from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from contracts import Frame, FrameBatch
from face_dino_v2.adapter import (
    FaceDinoV2Adapter,
    FaceDinoV2Settings,
    ellipse_xyxy,
)


class FakeRuntime:
    def __init__(self, outputs) -> None:
        self.outputs = outputs
        self.closed = False

    def predict(self, images):
        assert len(images) == len(self.outputs)
        return self.outputs

    def synchronize(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _output(face_present: bool):
    return {
        "boxes": torch.tensor([[1.0, 2.0, 19.0, 18.0]]),
        "scores": torch.tensor([0.9]),
        "face_scores": torch.tensor([0.8]),
        "face_present": torch.tensor([face_present]),
        "ellipses": torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.0]]),
        "keypoints": torch.tensor(
            [
                [
                    [7.0, 9.0],
                    [13.0, 9.0],
                    [10.0, 10.0],
                    [8.0, 12.0],
                    [12.0, 12.0],
                ]
            ]
        ),
        "point_classes": torch.tensor([[1, 1, 2, 3, 3]]),
        "keypoint_states": torch.tensor([[2, 1, 2, 2, 0]]),
        "point_confidence": torch.tensor([[0.9, 0.8, 0.95, 0.85, 0.2]]),
        "point_valid": torch.tensor(
            [[True, True, True, True, False]]
            if face_present
            else [[False, False, False, False, False]]
        ),
        "point_class_probabilities": torch.tensor(
            [
                [
                    [0.05, 0.90, 0.03, 0.02],
                    [0.05, 0.85, 0.05, 0.05],
                    [0.02, 0.03, 0.93, 0.02],
                    [0.03, 0.02, 0.05, 0.90],
                    [0.70, 0.05, 0.05, 0.20],
                ]
            ]
        ),
        "point_state_probabilities": torch.tensor(
            [
                [
                    [0.05, 0.95],
                    [0.75, 0.25],
                    [0.05, 0.95],
                    [0.10, 0.90],
                    [0.50, 0.50],
                ]
            ]
        ),
        "ellipse_moment_masks": torch.full((1, 64, 64), 0.75),
        "ellipse_mask_boxes": torch.tensor([[4.0, 4.0, 16.0, 16.0]]),
    }


class FaceDinoV2AdapterTest(unittest.TestCase):
    def _settings(self, classes=None):
        root = Path(tempfile.gettempdir())
        return FaceDinoV2Settings(
            source_root=root,
            checkpoint=root / "checkpoint.pth",
            trt_bundle=root / "manifest.json",
            device="cpu",
            classes=classes,
        )

    def test_ellipse_axis_aligned_envelope(self) -> None:
        self.assertEqual(
            ellipse_xyxy([10.0, 10.0, 4.0, 2.0, 0.0]),
            (6.0, 8.0, 14.0, 12.0),
        )
        x1, y1, x2, y2 = ellipse_xyxy([10.0, 10.0, 4.0, 2.0, torch.pi / 2])
        self.assertAlmostEqual(x1, 8.0, places=6)
        self.assertAlmostEqual(y1, 6.0, places=6)
        self.assertAlmostEqual(x2, 12.0, places=6)
        self.assertAlmostEqual(y2, 14.0, places=6)

    def test_emits_head_and_conditional_face_in_source_coordinates(self) -> None:
        frames = FrameBatch.from_sequence(
            [
                Frame(
                    0,
                    0.0,
                    np.zeros((20, 20, 3), dtype=np.uint8),
                ),
                Frame(
                    1,
                    1 / 30,
                    np.zeros((20, 20, 3), dtype=np.uint8),
                ),
            ]
        )
        adapter = FaceDinoV2Adapter(
            self._settings(),
            runtime=FakeRuntime([_output(True), _output(False)]),
        )
        results = adapter.predict(frames)
        self.assertEqual(
            [[item.class_name for item in result.detections] for result in results],
            [["Head", "Face"], ["Head"]],
        )
        face = results[0].detections[1]
        self.assertEqual(
            (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2),
            (6.0, 8.0, 14.0, 12.0),
        )
        self.assertEqual(face.source, "ellipse_detection")
        head = results[0].detections[0]
        self.assertEqual(head.group_id, face.group_id)
        observation = head.face_observation
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertTrue(observation.present)
        self.assertEqual(5, len(observation.keypoints))
        self.assertEqual("Eye", observation.keypoints[0].class_name)
        self.assertEqual("occluded", observation.keypoints[1].state_name)
        self.assertFalse(observation.keypoints[-1].valid)
        self.assertIsNotNone(observation.mask)
        assert observation.mask is not None
        self.assertEqual((64, 64), (observation.mask.width, observation.mask.height))

    def test_class_filter_and_parser(self) -> None:
        self.assertEqual(
            FaceDinoV2Adapter.parse_classes(["Head", "2"]),
            frozenset((1, 2)),
        )
        frame = FrameBatch.from_sequence(
            [Frame(0, 0.0, np.zeros((20, 20, 3), dtype=np.uint8))]
        )
        adapter = FaceDinoV2Adapter(
            self._settings(frozenset((2,))),
            runtime=FakeRuntime([_output(True)]),
        )
        self.assertEqual(
            [item.class_name for item in adapter.predict(frame)[0].detections],
            ["Face"],
        )


if __name__ == "__main__":
    unittest.main()
