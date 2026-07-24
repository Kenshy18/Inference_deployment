"""RT-DETR model construction and batch execution."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

FRAMEWORK_SOURCE = (
    Path(os.environ["RTDETR_FRAMEWORK_SOURCE"]).expanduser().resolve()
    if os.environ.get("RTDETR_FRAMEWORK_SOURCE")
    else None
)
if FRAMEWORK_SOURCE is not None and str(FRAMEWORK_SOURCE) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SOURCE))
for _extra in reversed(
    [
        item
        for item in os.environ.get("RTDETR_EXTRA_SITE_PACKAGES", "").split(os.pathsep)
        if item
    ]
):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

# RT-DETRv4 imports a training-only FLOP profiler while importing engine.core.
try:
    import calflops as _calflops  # noqa: F401
except ModuleNotFoundError:
    _calflops_stub = types.ModuleType("calflops")

    def _unavailable_calflops(*_args, **_kwargs):
        raise RuntimeError("calflops is unavailable in the inference-only runtime")

    _calflops_stub.calculate_flops = _unavailable_calflops
    sys.modules["calflops"] = _calflops_stub

from engine.core import YAMLConfig

from .preprocessing import make_batch

DEFAULT_CONFIG = "configs/rtv2/rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"
LABEL_NAMES = {0: "VisibleBody", 1: "Head", 2: "Face"}


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if FRAMEWORK_SOURCE is not None:
        candidate = FRAMEWORK_SOURCE / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {path}")


def parse_class_filter(values: Sequence[str] | None) -> set[int] | None:
    if not values:
        return None
    name_to_id = {name.lower(): index for index, name in LABEL_NAMES.items()}
    result: set[int] = set()
    for value in values:
        result.add(
            name_to_id[value.lower()]
            if value.lower() in name_to_id
            else int(value)
        )
    return result


def _load_state_dict(checkpoint_path: Path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
        return checkpoint["ema"]["module"]
    for name in ("model", "state_dict"):
        if name in checkpoint:
            return checkpoint[name]
    raise KeyError(f"unsupported checkpoint format: {checkpoint_path}")


class DetectionModel(nn.Module):
    def __init__(self, config: YAMLConfig):
        super().__init__()
        self.model = config.model.deploy()
        self.postprocessor = config.postprocessor.deploy()

    def forward(
        self,
        images: torch.Tensor,
        original_sizes: torch.Tensor,
        padding: torch.Tensor,
        scale: torch.Tensor,
        input_sizes: torch.Tensor,
    ):
        return self.postprocessor(
            self.model(images),
            original_sizes,
            letterbox_padding=padding,
            letterbox_scale=scale,
            input_sizes=input_sizes,
        )


def build_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    size_override: Sequence[int] | None,
    use_channels_last: bool,
    use_compile: bool,
) -> tuple[nn.Module, tuple[int, int]]:
    overrides = (
        {}
        if size_override is None
        else {"eval_spatial_size": [int(size_override[0]), int(size_override[1])]}
    )
    config = YAMLConfig(str(config_path), resume=str(checkpoint_path), **overrides)
    for backbone_name in ("PResNet", "HGNetv2"):
        if backbone_name in config.yaml_cfg:
            config.yaml_cfg[backbone_name]["pretrained"] = False
    size = config.yaml_cfg.get("eval_spatial_size")
    if not size or len(size) != 2:
        raise ValueError("eval_spatial_size must be [height, width]")
    config.model.load_state_dict(_load_state_dict(checkpoint_path))
    model: nn.Module = DetectionModel(config).eval()
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
    return model, (int(size[0]), int(size[1]))


def select_precision(
    device: torch.device, precision: str
) -> tuple[torch.dtype | None, str]:
    if device.type != "cuda" or precision == "fp32":
        return None, "fp32"
    if precision in {"auto", "fp16"}:
        return torch.float16, "fp16"
    return torch.bfloat16, "bf16"


def configure_torch(device: torch.device, enable_tf32: bool) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32


def run_batch(
    model,
    frames,
    input_size,
    device,
    precision_dtype,
    autocast_dtype,
    channels_last,
):
    inputs = make_batch(
        frames,
        input_size,
        device,
        precision_dtype,
        channels_last,
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        return model(*inputs)


def warmup_model(
    model,
    input_size,
    batch_size,
    device,
    precision_dtype,
    autocast_dtype,
    warmup_iterations,
    channels_last,
) -> None:
    if warmup_iterations <= 0:
        return
    dummy = np.zeros((*input_size, 3), dtype=np.uint8)
    frames = [dummy] * batch_size
    for _ in range(warmup_iterations):
        run_batch(
            model,
            frames,
            input_size,
            device,
            precision_dtype,
            autocast_dtype,
            channels_last,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()


__all__ = [
    "DEFAULT_CONFIG",
    "LABEL_NAMES",
    "build_model",
    "configure_torch",
    "parse_class_filter",
    "resolve_existing_path",
    "run_batch",
    "select_precision",
    "warmup_model",
]
