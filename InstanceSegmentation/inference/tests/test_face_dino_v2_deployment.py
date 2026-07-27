from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

import torch

from face_dino_v2.deployment import (
    _DeploymentBackboneStub,
    _DropPathStub,
    _MlpStub,
    deployment_import_stubs,
    prepare_deployment_shell,
    validate_deployment_state,
)
from face_dino_v2.cudagraph import build_zero_copy_detector_backend


class _Attribute(torch.nn.Module):
    ellipse_moment_power = 2.5


class FaceDinoDeploymentTest(unittest.TestCase):
    def test_cuda_graph_rejects_multiclass_detector(self) -> None:
        detector = SimpleNamespace(
            query_head=SimpleNamespace(num_classes=2),
        )
        with self.assertRaisesRegex(ValueError, "one detector class"):
            build_zero_copy_detector_backend(detector)

    def test_import_substitutes_are_removed_after_construction(self) -> None:
        names = ("timm", "timm.models.layers", "dinov3", "dinov3.hub.backbones")
        if any(name in sys.modules for name in names):
            self.skipTest("real training dependency was already imported")
        with deployment_import_stubs():
            self.assertIn("timm.models.layers", sys.modules)
            self.assertIn("dinov3.hub.backbones", sys.modules)
        for name in names:
            self.assertNotIn(name, sys.modules)

    def test_registration_only_timm_layers_preserve_tensor_shapes(self) -> None:
        value = torch.randn(2, 3, 4)
        self.assertIs(_DropPathStub()(value), value)
        self.assertEqual(_MlpStub(4, 8)(value).shape, value.shape)

    def test_backbone_stub_fails_before_tensorrt_installation(self) -> None:
        stub = _DeploymentBackboneStub()
        self.assertEqual(stub.embed_dim, 384)
        with self.assertRaisesRegex(RuntimeError, "TensorRT backbone"):
            stub.get_intermediate_layers(torch.zeros(1))

    def test_shell_retains_only_attribute_geometry_metadata(self) -> None:
        transformer = SimpleNamespace(
            encoder=torch.nn.Linear(2, 2),
            decoder=torch.nn.Linear(2, 2),
        )
        model = SimpleNamespace(
            attribute_model=_Attribute(),
            detector=SimpleNamespace(
                neck=torch.nn.Conv2d(2, 2, 1),
                query_head=SimpleNamespace(transformer=transformer),
            ),
        )
        prepare_deployment_shell(model)
        self.assertEqual(model.attribute_model.ellipse_moment_power, 2.5)
        self.assertEqual(len(model.attribute_model.state_dict()), 0)
        self.assertIsInstance(model.detector.neck, torch.nn.Identity)
        with self.assertRaisesRegex(RuntimeError, "deployment module"):
            transformer.encoder(torch.zeros(1))

    def test_state_validation_allows_only_replaced_branches(self) -> None:
        validate_deployment_state(
            SimpleNamespace(
                missing_keys=[],
                unexpected_keys=[
                    "detector.backbone.backbone.blocks.0.weight",
                    "detector.neck.p3.weight",
                    "attribute_model.adapter.weight",
                ],
            )
        )
        with self.assertRaisesRegex(RuntimeError, "query_head.cls"):
            validate_deployment_state(
                SimpleNamespace(
                    missing_keys=[],
                    unexpected_keys=["detector.query_head.cls.weight"],
                )
            )


if __name__ == "__main__":
    unittest.main()
