from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dinov3_codino.classifier import RoiSpatialGapClassifier
from dinov3_codino_mh0.classifier import (
    CLASS_IDS,
    CLASS_NAMES,
    INPUT_DIM,
    POOLER_CHANNELS,
    POOLER_SIZE,
    classifier_from_checkpoint,
)
from dinov3_codino_mh0.optimization.batched_bbox import (
    _with_classifier_columns,
)


def _checkpoint(*, channels: int = POOLER_CHANNELS) -> dict[str, object]:
    input_dim = channels * POOLER_SIZE * POOLER_SIZE
    model = RoiSpatialGapClassifier(
        input_dim=input_dim,
        num_classes=3,
        use_meta=True,
        pooler_channels=channels,
        pooler_size=POOLER_SIZE,
        stem_channels=96,
        mid_channels=96,
        dw_kernel=3,
        dropout=0.05,
    )
    return {
        "artifact_status": "unit_test",
        "class_ids": list(CLASS_IDS),
        "class_names": list(CLASS_NAMES),
        "model_cfg": {
            "model_type": "spatial_gap",
            "num_classes": 3,
            "input_dim": input_dim,
            "use_meta": True,
            "pooler_channels": channels,
            "pooler_size": POOLER_SIZE,
            "gap_stem_channels": 96,
            "gap_mid_channels": 96,
            "gap_dw_kernel": 3,
            "gap_dropout": 0.05,
        },
        "model_state": model.state_dict(),
    }


class Mh0ClassifierTest(unittest.TestCase):
    def test_loader_accepts_exact_mh0_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.pt"
            torch.save(_checkpoint(), path)
            model, payload = classifier_from_checkpoint(path)
        logits = model(
            torch.randn(4, POOLER_CHANNELS, POOLER_SIZE, POOLER_SIZE),
            torch.randn(4, 5),
        )
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(payload["artifact_status"], "unit_test")
        self.assertEqual(INPUT_DIM, 192 * 14 * 14)

    def test_loader_rejects_large_codino_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save(_checkpoint(channels=256), path)
            with self.assertRaisesRegex(
                ValueError,
                r"must consume \[N,192,14,14\]",
            ):
                classifier_from_checkpoint(path)

    def test_classifier_columns_are_joined_to_same_batch_and_consumed(self) -> None:
        first_extra = torch.tensor(
            [[2.0, 0.8, 0.1, 0.1, 0.8]],
            dtype=torch.float32,
        )
        second_extra = torch.tensor(
            [
                [0.0, 0.6, 0.6, 0.3, 0.1],
                [1.0, 0.7, 0.2, 0.7, 0.1],
            ],
            dtype=torch.float32,
        )
        model = SimpleNamespace(
            _mh0_last_classifications=(first_extra, second_extra)
        )
        results = [
            (
                torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9]]),
                torch.tensor([0]),
            ),
            (
                torch.tensor(
                    [
                        [5.0, 6.0, 7.0, 8.0, 0.8],
                        [9.0, 10.0, 11.0, 12.0, 0.7],
                    ]
                ),
                torch.tensor([0, 0]),
            ),
        ]
        enriched = _with_classifier_columns(model, results)
        self.assertIsNone(model._mh0_last_classifications)
        self.assertTrue(
            torch.equal(enriched[0][0][:, 5:], first_extra)
        )
        self.assertTrue(
            torch.equal(enriched[1][0][:, 5:], second_extra)
        )


if __name__ == "__main__":
    unittest.main()
