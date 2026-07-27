"""Lightweight construction helpers for the fully TensorRT Face DINO path."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types

import torch


REPLACED_STATE_PREFIXES = (
    "detector.backbone.backbone.",
    "detector.neck.",
    "detector.query_head.transformer.encoder.",
    "detector.query_head.transformer.decoder.",
    "attribute_model.",
)


class _DeploymentBackboneStub(torch.nn.Module):
    """Construction-only DINOv3 replacement installed before source imports."""

    def __init__(self, embed_dim: int = 384) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    def get_intermediate_layers(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "the TensorRT backbone was not installed on the Face DINO shell"
        )


class _UnavailableModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("a TensorRT deployment module was not installed")


class _AttributeMetadata(torch.nn.Module):
    def __init__(self, ellipse_moment_power: float) -> None:
        super().__init__()
        self.ellipse_moment_power = float(ellipse_moment_power)


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
    """Provide registration-only timm symbols without importing training I/O."""

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
    """Avoid importing the training-only DINOv3/timm dependency graph."""

    module_name = "dinov3.hub.backbones"
    if module_name in sys.modules or "dinov3" in sys.modules:
        return False

    dinov3 = types.ModuleType("dinov3")
    dinov3.__path__ = []  # type: ignore[attr-defined]
    sys.modules["dinov3"] = dinov3
    hub = types.ModuleType("dinov3.hub")
    hub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["dinov3.hub"] = hub
    dinov3.hub = hub
    backbones = types.ModuleType(module_name)

    def dinov3_vits16plus(*args, **kwargs):
        del args, kwargs
        return _DeploymentBackboneStub(384)

    def dinov3_vitl16(*args, **kwargs):
        del args, kwargs
        return _DeploymentBackboneStub(1024)

    backbones.dinov3_vits16plus = dinov3_vits16plus
    backbones.dinov3_vitl16 = dinov3_vitl16
    sys.modules[module_name] = backbones
    hub.backbones = backbones
    return True


@contextmanager
def deployment_import_stubs():
    """Scope training-only dependency substitutes to model construction."""

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


def prepare_deployment_shell(model: torch.nn.Module) -> None:
    """Drop checkpoint branches replaced by reviewed TensorRT engines."""

    ellipse_moment_power = float(model.attribute_model.ellipse_moment_power)
    model.attribute_model = _AttributeMetadata(ellipse_moment_power)
    model.detector.neck = torch.nn.Identity()
    transformer = model.detector.query_head.transformer
    transformer.encoder = _UnavailableModule()
    transformer.decoder = _UnavailableModule()


def validate_deployment_state(incompatible) -> None:
    """Reject checkpoint drift outside explicitly replaced model branches."""

    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if "num_batches_tracked" not in key
    ]
    invalid_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(REPLACED_STATE_PREFIXES)
    ]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "Face DINO deployment checkpoint key drift: "
            f"missing={invalid_missing}, unexpected={invalid_unexpected}"
        )


__all__ = [
    "REPLACED_STATE_PREFIXES",
    "deployment_import_stubs",
    "install_dinov3_backbone_stub",
    "install_timm_layer_stubs",
    "prepare_deployment_shell",
    "validate_deployment_state",
]
