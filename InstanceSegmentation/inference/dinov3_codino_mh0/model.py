"""Build the eager or fixed-B16 TensorRT MH0 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .optimization.fast_model import (
        SUPPORTED_BACKENDS,
        build_backend_model,
        infer_fixed_batch,
    )
    from .trt.bundle import load_engine_bundle
except ImportError:
    from optimization.fast_model import (
        SUPPORTED_BACKENDS,
        build_backend_model,
        infer_fixed_batch,
    )
    from trt.bundle import load_engine_bundle


FAMILY_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = FAMILY_ROOT / "artifacts" / "detector" / "resolved_config.py"
DEFAULT_CHECKPOINT = (
    FAMILY_ROOT
    / "artifacts"
    / "detector"
    / "video_pseudo_mh0_epoch6_ema_deploy.pth"
)
DEFAULT_TRT_BUNDLE = (
    FAMILY_ROOT
    / "artifacts"
    / "trt"
    / "fast-sm120-fixed-b16-v1"
    / "manifest.json"
)


@dataclass(slots=True)
class Mh0Runtime:
    model: object
    backend: str
    device: str
    fixed_batch_size: int


def build_runtime(
    *,
    config: Path,
    checkpoint: Path,
    backend: str,
    device: str,
    model_score_threshold: float,
    trt_bundle: Path = DEFAULT_TRT_BUNDLE,
    trt_verify: str = "engines",
) -> Mh0Runtime:
    if not config.is_file():
        raise FileNotFoundError(f"MH0 config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MH0 checkpoint not found: {checkpoint}")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported MH0 backend: {backend!r}")
    artifact_paths = None
    if backend == "tensorrt-fast":
        bundle = load_engine_bundle(trt_bundle, verify=trt_verify)
        artifact_paths = {
            **bundle.engines,
            "plugin": bundle.plugin,
            "preprocess_plugin": bundle.preprocess_plugin,
        }
    model = build_backend_model(
        config=config,
        checkpoint=checkpoint,
        backend=backend,
        device=device,
        model_score_threshold=model_score_threshold,
        artifacts=artifact_paths,
    )
    return Mh0Runtime(
        model=model,
        backend=backend,
        device=device,
        fixed_batch_size=int(model._mh0_batch_size),
    )


def infer(runtime: Mh0Runtime, frames):
    return infer_fixed_batch(
        runtime.model,
        list(frames),
        device=runtime.device,
    )[0]


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CONFIG",
    "DEFAULT_TRT_BUNDLE",
    "Mh0Runtime",
    "build_runtime",
    "infer",
]
