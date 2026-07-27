"""Lightweight model shell for the fixed TensorRT partition group."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types

import torch


class _DeploymentBackboneStub(torch.nn.Module):
    """Construction-only DINOv3 replacement installed before source imports."""

    embed_dim = 1024

    def get_intermediate_layers(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "the TensorRT backbone was not installed on the Co-DINO shell"
        )


class _DropPathStub(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.drop_prob = float(drop_prob)

    def forward(self, value):
        return value


class _MlpStub(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        hidden = int(hidden_features or in_features)
        output = int(out_features or in_features)
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, value):
        return self.fc2(self.act(self.fc1(value)))


def install_timm_layer_stubs() -> bool:
    """Provide import-time timm symbols without loading training dependencies."""

    module_name = "timm.models.layers"
    if module_name in sys.modules or "timm" in sys.modules:
        return False
    timm = types.ModuleType("timm")
    timm.__path__ = []  # type: ignore[attr-defined]
    models = types.ModuleType("timm.models")
    models.__path__ = []  # type: ignore[attr-defined]
    layers = types.ModuleType(module_name)
    layers.DropPath = _DropPathStub
    layers.Mlp = _MlpStub
    layers.to_2tuple = lambda value: (
        value if isinstance(value, tuple) else (value, value)
    )
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules[module_name] = layers
    timm.models = models
    models.layers = layers
    return True


def install_dinov3_backbone_stub() -> bool:
    """Avoid importing DINOv3 because its output is replaced by TensorRT."""

    module_name = "dinov3.hub.backbones"
    if module_name in sys.modules or "dinov3" in sys.modules:
        return False
    dinov3 = types.ModuleType("dinov3")
    dinov3.__path__ = []  # type: ignore[attr-defined]
    hub = types.ModuleType("dinov3.hub")
    hub.__path__ = []  # type: ignore[attr-defined]
    backbones = types.ModuleType(module_name)

    def dinov3_vitl16(*args, **kwargs):
        del args, kwargs
        return _DeploymentBackboneStub()

    backbones.dinov3_vitl16 = dinov3_vitl16
    sys.modules["dinov3"] = dinov3
    sys.modules["dinov3.hub"] = hub
    sys.modules[module_name] = backbones
    dinov3.hub = hub
    hub.backbones = backbones
    return True


@contextmanager
def deployment_import_stubs():
    """Scope construction-only dependency substitutes to model loading."""

    names = (
        "timm.models.layers",
        "timm.models",
        "timm",
        "dinov3.hub.backbones",
        "dinov3.hub",
        "dinov3",
    )
    previous = {name: sys.modules.get(name) for name in names}
    install_timm_layer_stubs()
    install_dinov3_backbone_stub()
    try:
        yield
    finally:
        for name in names:
            prior = previous[name]
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


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


__all__ = [
    "deployment_import_stubs",
    "install_dinov3_backbone_stub",
    "install_timm_layer_stubs",
    "prepare_trt_deployment_config",
]
