from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from mmcv import Config

from dinov3_codino.optimized.deployment import (
    prepare_trt_deployment_config,
)
from dinov3_codino.trt.build_runtime_checkpoint import (
    build_runtime_checkpoint,
)


class CoDinoDeploymentTest(unittest.TestCase):
    def test_runtime_checkpoint_keeps_only_live_shell_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            output = root / "runtime.pth"
            torch.save(
                {
                    "meta": {"CLASSES": ("foreground",)},
                    "state_dict": {
                        "backbone.block.weight": torch.ones(2),
                        "neck.lateral.weight": torch.ones(3),
                        "query_head.cls_branches.0.weight": torch.ones(4),
                        "query_head.transformer.encoder.layer.weight": (
                            torch.ones(5)
                        ),
                        "query_head.transformer.decoder.layer.weight": (
                            torch.ones(6)
                        ),
                        "mask_head.semantic_convs.0.weight": torch.ones(7),
                        "roi_head.bbox.weight": torch.ones(8),
                    },
                    "optimizer": {"state": {"large": torch.ones(32)}},
                },
                source,
            )

            provenance = build_runtime_checkpoint(source, output)
            payload = torch.load(
                output,
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(
                set(payload["state_dict"]),
                {
                    "neck.lateral.weight",
                    "query_head.cls_branches.0.weight",
                    "mask_head.semantic_convs.0.weight",
                },
            )
            self.assertNotIn("optimizer", payload)
            self.assertEqual(provenance["retained_state_keys"], 3)
            self.assertEqual(
                payload["meta"]["trt_runtime_checkpoint"]["source_size"],
                source.stat().st_size,
            )

    def test_deployment_config_removes_replaced_modules(self) -> None:
        config = Config(
            {
                "model": {
                    "backbone": {"type": "Original"},
                    "rpn_head": {"type": "RPN"},
                    "roi_head": [{"type": "ROI"}],
                    "bbox_head": [{"type": "BBox"}],
                    "mask_iou_head": {"type": "MaskIoU"},
                }
            }
        )

        prepare_trt_deployment_config(config)

        self.assertEqual(
            config.model.backbone,
            {"type": "CoDinoTrtBackboneStub"},
        )
        self.assertIsNone(config.model.rpn_head)
        self.assertEqual(config.model.roi_head, [])
        self.assertEqual(config.model.bbox_head, [])
        self.assertIsNone(config.model.mask_iou_head)


if __name__ == "__main__":
    unittest.main()
