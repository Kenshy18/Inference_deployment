"""Build the eager or fixed-B16 TensorRT MH0 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .classifier import classifier_from_checkpoint
except ImportError:
    from classifier import classifier_from_checkpoint

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
DEFAULT_CLASSIFIER_CHECKPOINT = (
    FAMILY_ROOT / "artifacts" / "classifier" / "best.pt"
)


@dataclass(slots=True)
class Mh0Runtime:
    model: object
    classifier: object | None
    class_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    classifier_checkpoint: Path | None
    classifier_status: str | None
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
    cuda_graph: bool = False,
    classifier_checkpoint: Path | None = DEFAULT_CLASSIFIER_CHECKPOINT,
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
        cuda_graph=cuda_graph,
    )
    classifier = None
    classifier_payload: dict[str, object] = {}
    resolved_classifier = (
        None
        if classifier_checkpoint is None
        else Path(classifier_checkpoint).expanduser().resolve()
    )
    if resolved_classifier is not None:
        if not resolved_classifier.is_file():
            raise FileNotFoundError(
                f"MH0 classifier checkpoint not found: {resolved_classifier}"
            )
        classifier, classifier_payload = classifier_from_checkpoint(
            resolved_classifier,
            map_location=device,
        )
        classifier.to(device).eval()
    class_names = tuple(
        str(value)
        for value in classifier_payload.get(
            "class_names",
            ["foreground"],
        )
    )
    class_ids = tuple(
        int(value)
        for value in classifier_payload.get(
            "class_ids",
            list(range(len(class_names))),
        )
    )
    if len(class_ids) != len(class_names):
        raise ValueError(
            "MH0 classifier class_ids and class_names must have equal length"
        )
    if classifier is not None:
        model._mh0_classifier = classifier
        model._mh0_classifier_class_count = len(class_names)
    classifier_status = (
        str(classifier_payload.get("artifact_status", "trained"))
        if classifier is not None
        else None
    )
    return Mh0Runtime(
        model=model,
        classifier=classifier,
        class_names=class_names,
        class_ids=class_ids,
        classifier_checkpoint=resolved_classifier,
        classifier_status=classifier_status,
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
    "DEFAULT_CLASSIFIER_CHECKPOINT",
    "DEFAULT_CONFIG",
    "DEFAULT_TRT_BUNDLE",
    "Mh0Runtime",
    "build_runtime",
    "infer",
]
