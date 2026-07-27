"""Lightweight model shell for the fixed TensorRT partition group."""

from __future__ import annotations

import torch


def prepare_trt_deployment_config(config) -> None:
    """Remove modules replaced by TensorRT before model construction."""

    from mmdet.models.builder import BACKBONES

    if BACKBONES.get("CoDinoTrtBackboneStub") is None:

        @BACKBONES.register_module()
        class CoDinoTrtBackboneStub(torch.nn.Module):
            def __init__(self, **kwargs) -> None:
                del kwargs
                super().__init__()

            def init_weights(self, *args, **kwargs) -> None:
                del args, kwargs

            def forward(self, value):
                del value
                raise RuntimeError(
                    "the TensorRT backbone was not installed on the "
                    "deployment model shell"
                )

    model = config.model
    model.backbone = {"type": "CoDinoTrtBackboneStub"}
    model.rpn_head = None
    model.roi_head = []
    model.bbox_head = []
    model.mask_iou_head = None


__all__ = ["prepare_trt_deployment_config"]
