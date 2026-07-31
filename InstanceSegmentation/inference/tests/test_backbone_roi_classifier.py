from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from backbone_roi_classifier import (
    CANONICAL_CLASS_IDS,
    CANONICAL_CLASS_NAMES,
    BackboneRoiClassifier,
)
from backbone_roi_classifier.models import build_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(root: Path) -> Path:
    config = {
        "model_type": "spatial_gap",
        "input_dim": 4 * 2 * 2,
        "num_classes": 3,
        "use_meta": True,
        "pooler_channels": 4,
        "pooler_size": 2,
        "gap_stem_channels": 4,
        "gap_mid_channels": 4,
        "gap_dw_kernel": 3,
        "gap_dropout": 0.0,
    }
    model = build_model(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.head.bias.data.copy_(torch.tensor((1.0, 3.0, 2.0)))
    checkpoint = root / "classifier.pt"
    torch.save(
        {
            "class_names": ["male", "female", "junction"],
            "class_ids": [1, 2, 3],
            "model_cfg": config,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "class_names": ["male", "female", "junction"],
                "input": {
                    "shape": [4, 2, 2],
                    "stride": 16,
                    "metadata": "geo_v2",
                },
                "classifier": {
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": _sha256(checkpoint),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


class BackboneRoiClassifierTest(unittest.TestCase):
    def test_manifest_loader_preserves_fixed_external_class_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            classifier, provenance = BackboneRoiClassifier.from_manifest(
                _write_artifacts(Path(directory))
            )
        logits = classifier(
            torch.zeros((2, 4, 2, 2)),
            torch.zeros((2, 5)),
        )
        self.assertEqual(classifier.class_names, CANONICAL_CLASS_NAMES)
        self.assertEqual(classifier.class_ids, CANONICAL_CLASS_IDS)
        self.assertEqual(tuple(logits[0].tolist()), (3.0, 1.0, 2.0))
        self.assertEqual(provenance["mode"], "single")

    def test_manifest_loader_rejects_modified_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_artifacts(root)
            with (root / "classifier.pt").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                BackboneRoiClassifier.from_manifest(manifest)

    def test_roi_align_accepts_batch_indexed_backbone_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            classifier, _ = BackboneRoiClassifier.from_manifest(
                _write_artifacts(Path(directory))
            )
        classes, scores, probabilities = classifier.classify_backbone(
            torch.zeros((2, 4, 8, 8)),
            torch.tensor(((0.0, 0.0, 16.0, 16.0), (16.0, 16.0, 32.0, 32.0))),
            torch.zeros((2, 5)),
            batch_indices=torch.tensor((0, 1)),
        )
        self.assertEqual(tuple(classes.tolist()), (0, 0))
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertEqual(tuple(probabilities.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
