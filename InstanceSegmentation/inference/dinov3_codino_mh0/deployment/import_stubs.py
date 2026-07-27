"""Construction-only dependency stubs for the MH0 TensorRT runtime."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types

import torch


class _DeploymentBackboneStub(torch.nn.Module):
    embed_dim = 384

    def get_intermediate_layers(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("the MH0 TensorRT backbone was not installed")


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


def _install_timm_stubs() -> None:
    if "timm" in sys.modules:
        return
    timm = types.ModuleType("timm")
    timm.__path__ = []  # type: ignore[attr-defined]
    models = types.ModuleType("timm.models")
    models.__path__ = []  # type: ignore[attr-defined]
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = _DropPathStub
    layers.Mlp = _MlpStub
    layers.to_2tuple = lambda value: (
        value if isinstance(value, tuple) else (value, value)
    )
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers
    timm.models = models
    models.layers = layers


def _install_dinov3_stub() -> None:
    if "dinov3" in sys.modules:
        return
    dinov3 = types.ModuleType("dinov3")
    dinov3.__path__ = []  # type: ignore[attr-defined]
    hub = types.ModuleType("dinov3.hub")
    hub.__path__ = []  # type: ignore[attr-defined]
    backbones = types.ModuleType("dinov3.hub.backbones")

    def dinov3_vits16plus(*args, **kwargs):
        del args, kwargs
        return _DeploymentBackboneStub()

    backbones.dinov3_vits16plus = dinov3_vits16plus
    backbones.dinov3_vitl16 = dinov3_vits16plus
    sys.modules["dinov3"] = dinov3
    sys.modules["dinov3.hub"] = hub
    sys.modules["dinov3.hub.backbones"] = backbones
    dinov3.hub = hub
    hub.backbones = backbones


@contextmanager
def mh0_deployment_import_stubs():
    """Avoid importing training-only timm/DINOv3 code for TensorRT."""

    names = (
        "timm.models.layers",
        "timm.models",
        "timm",
        "dinov3.hub.backbones",
        "dinov3.hub",
        "dinov3",
    )
    previous = {name: sys.modules.get(name) for name in names}
    _install_timm_stubs()
    _install_dinov3_stub()
    try:
        yield
    finally:
        for name in names:
            prior = previous[name]
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


__all__ = ["mh0_deployment_import_stubs"]
