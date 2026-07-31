from __future__ import annotations

import unittest

import numpy as np
import torch

from contracts import Frame, ModelDescriptor, TaskType, segmentation_frame_from_rows
from dinov3_cascade.output import (
    drop_auxiliary_instance_fields,
    instances_to_rows as dinov3_instances_to_rows,
)
from dinov3_codino.postprocessing import detections_to_rows as codino_rows
from eva02_cascade.output import (
    drop_auxiliary_fields,
    instances_to_rows as eva02_instances_to_rows,
)


class _Boxes:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _Instances:
    def __init__(self) -> None:
        self._fields = {
            "pred_boxes": _Boxes(
                torch.tensor([[1.0, 2.0, 11.0, 12.0]], dtype=torch.float32)
            ),
            "scores": torch.tensor([0.8], dtype=torch.float32),
            "pred_multiclass_classes": torch.tensor([1], dtype=torch.int64),
            "pred_multiclass_scores": torch.tensor([0.7], dtype=torch.float32),
            "pred_multiclass_probabilities": torch.tensor(
                [[0.1, 0.7, 0.2]], dtype=torch.float32
            ),
            "pred_multiclass_logits": torch.tensor(
                [[-1.0, 1.0, 0.0]], dtype=torch.float32
            ),
            "pred_class_logits": torch.tensor(
                [[-1.0, 1.0, 0.0]], dtype=torch.float32
            ),
        }

    def __len__(self) -> int:
        return 1

    def __getattr__(self, name: str):
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]


class ClassifierProbabilityOutputTest(unittest.TestCase):
    def _assert_contract_probabilities(self, rows) -> None:
        result = segmentation_frame_from_rows(
            model=ModelDescriptor(
                model_id="test",
                task=TaskType.INSTANCE_SEGMENTATION,
                implementation="test",
            ),
            frame=Frame(
                index=0,
                timestamp_sec=0.0,
                image=np.zeros((16, 16, 3), dtype=np.uint8),
            ),
            rows=rows,
        )
        classification = result.instances[0].detection.classification
        self.assertIsNotNone(classification)
        assert classification is not None
        self.assertEqual(classification.class_id, 11)
        self.assertEqual(classification.class_name, "class-b")
        self.assertAlmostEqual(classification.score, 0.7, places=6)
        self.assertEqual(len(classification.probabilities or ()), 3)
        self.assertAlmostEqual(
            sum(classification.probabilities or ()), 1.0, places=6
        )

    def test_dinov3_preserves_probabilities_through_contract_rows(self) -> None:
        instances = _Instances()
        drop_auxiliary_instance_fields(instances)
        self.assertFalse(instances.has("pred_multiclass_logits"))
        self.assertTrue(instances.has("pred_multiclass_probabilities"))
        rows = dinov3_instances_to_rows(
            instances,
            class_names=["class-a", "class-b", "class-c"],
            class_ids=[10, 11, 12],
            score_threshold=0.0,
        )
        self._assert_contract_probabilities(rows)

    def test_eva02_preserves_probabilities_through_contract_rows(self) -> None:
        instances = _Instances()
        drop_auxiliary_fields(instances)
        self.assertFalse(instances.has("pred_class_logits"))
        self.assertTrue(instances.has("pred_multiclass_probabilities"))
        rows = eva02_instances_to_rows(
            instances,
            class_names=("class-a", "class-b", "class-c"),
            class_ids=(10, 11, 12),
            score_threshold=0.0,
        )
        self._assert_contract_probabilities(rows)

    def test_codino_keeps_detector_and_classifier_classes_separate(self) -> None:
        rows = codino_rows(
            [
                np.asarray(
                    [[1.0, 2.0, 11.0, 12.0, 0.8, 1.0, 0.7, 0.1, 0.7, 0.2]],
                    dtype=np.float32,
                )
            ],
            None,
            class_names=["女性器", "男性器", "結合部分"],
            class_ids=[1, 2, 3],
            score_threshold=0.0,
        )
        self.assertEqual(rows[0]["class_name"], "foreground")
        self.assertEqual(rows[0]["category_id"], 0)
        self.assertEqual(rows[0]["classifier_class_name"], "男性器")
        self.assertEqual(rows[0]["classifier_class_id"], 2)
        self.assertAlmostEqual(sum(rows[0]["class_probs"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
