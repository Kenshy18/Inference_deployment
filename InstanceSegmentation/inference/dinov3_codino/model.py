"""Co-DINO model loading and fixed-batch inference primitives."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from .classifier import classifier_from_checkpoint
    from .preprocessing import prepare_batch_direct
    from .trt.runtime import FixedTrtPartitionSettings, install_fixed_partitions
except ImportError:
    from classifier import classifier_from_checkpoint
    from preprocessing import prepare_batch_direct
    from trt.runtime import FixedTrtPartitionSettings, install_fixed_partitions


@dataclass(frozen=True, slots=True)
class InstanceSegmentationSettings:
    config_path: Path
    checkpoint: Path
    target_size: tuple[int, int] = (720, 1280)
    score_threshold: float = 0.1
    model_score_threshold: float = 0.05
    disable_activation_checkpointing: bool = True
    skip_backbone_initialization: bool = True
    trt_deployment_shell: bool = False

    def __post_init__(self) -> None:
        (height, width) = self.target_size
        if height <= 0 or width <= 0:
            raise ValueError("target_size dimensions must be positive")
        for (name, value) in (
            ("score_threshold", self.score_threshold),
            ("model_score_threshold", self.model_score_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def disable_activation_checkpointing(node: Any) -> None:
    if isinstance(node, (list, tuple)):
        for item in node:
            disable_activation_checkpointing(item)
        return
    if not isinstance(node, dict):
        return
    for key in list(node):
        value = node[key]
        if key == "with_cp":
            node[key] = False if isinstance(value, bool) else -1
        elif key in {"use_checkpoint", "use_act_checkpoint"}:
            node[key] = False
        else:
            disable_activation_checkpointing(value)


def set_model_score_threshold(model_config: Any, score_threshold: float) -> None:
    test_config = (
        model_config.get("test_cfg", None) if isinstance(model_config, dict) else None
    )
    if test_config is None:
        return
    items = test_config if isinstance(test_config, list) else [test_config]
    for item in items:
        if not isinstance(item, dict):
            continue
        if "max_per_img" in item or "nms" in item or "mask_thr_binary" in item:
            item["score_thr"] = float(score_threshold)
        rcnn_config = item.get("rcnn", None)
        if isinstance(rcnn_config, dict):
            rcnn_config["score_thr"] = float(score_threshold)


def build_segmenter(settings: InstanceSegmentationSettings, *, device: str):
    if not settings.config_path.is_file():
        raise FileNotFoundError(f"Config not found: {settings.config_path}")
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint}")
    if settings.trt_deployment_shell:
        try:
            from .optimized.deployment import (
                deployment_import_stubs,
                prepare_trt_deployment_config,
            )
        except ImportError:
            from optimized.deployment import (
                deployment_import_stubs,
                prepare_trt_deployment_config,
            )

        import_scope = deployment_import_stubs()
    else:
        prepare_trt_deployment_config = None
        import_scope = nullcontext()
    with import_scope:
        return _build_segmenter(
            settings,
            device=device,
            prepare_trt_deployment_config=prepare_trt_deployment_config,
        )


def _build_segmenter(
    settings: InstanceSegmentationSettings,
    *,
    device: str,
    prepare_trt_deployment_config,
):
    from mmcv import Config
    from mmdet.apis import init_detector

    config = Config.fromfile(str(settings.config_path))
    if prepare_trt_deployment_config is not None:
        prepare_trt_deployment_config(config)
    if settings.skip_backbone_initialization and "backbone" in config.model:
        if "pretrained" in config.model.backbone:
            config.model.backbone.pretrained = False
        if "weights" in config.model.backbone:
            config.model.backbone.weights = None
    if "fp16" in config:
        config.pop("fp16")
    if settings.disable_activation_checkpointing:
        disable_activation_checkpointing(config.model)
    if settings.model_score_threshold > 0:
        set_model_score_threshold(config.model, settings.model_score_threshold)
    config.load_from = None
    config.resume_from = None
    if settings.trt_deployment_shell:
        model = init_detector(config, checkpoint=None, device=device).eval()
        payload = torch.load(
            settings.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError(
                "Co-DINO TensorRT runtime checkpoint has no state_dict"
            )
        provenance = (payload.get("meta") or {}).get(
            "trt_runtime_checkpoint"
        )
        if not isinstance(provenance, dict) or provenance.get("schema") != (
            "codino-trt-runtime-checkpoint-v1"
        ):
            raise ValueError(
                "Co-DINO TensorRT runtime checkpoint provenance is invalid"
            )
        incompatible = model.load_state_dict(
            payload["state_dict"],
            strict=False,
        )
        allowed_missing = (
            "query_head.transformer.encoder.",
            "query_head.transformer.decoder.",
        )
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
            and "num_batches_tracked" not in key
        ]
        allowed_unexpected = {
            "query_head.positional_encoding._dim_t",
        }
        invalid_unexpected = [
            key
            for key in incompatible.unexpected_keys
            if key not in allowed_unexpected
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Co-DINO TensorRT runtime checkpoint key drift: "
                f"missing={invalid_missing}, "
                f"unexpected={invalid_unexpected}"
            )
        classes = (payload.get("meta") or {}).get("CLASSES")
        if classes is not None:
            model.CLASSES = classes
    else:
        model = init_detector(
            config,
            str(settings.checkpoint),
            device=device,
        ).eval()
    model.CLASSES = ("foreground",)
    return model


def infer_batch_without_classifier(
    model, frames, *, amp: str, target_size: tuple[int, int]
):
    """Run the direct instance-segmentation path without classifier columns."""
    data = prepare_batch_direct(model, frames, target_size)
    use_cuda = next(model.parameters()).is_cuda
    with torch.inference_mode():
        if amp == "off" or not use_cuda:
            return model(return_loss=False, rescale=True, **data)
        if amp == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16 requested, but this GPU does not support bf16")
            dtype = torch.bfloat16
        elif amp == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unsupported amp mode: {amp!r}")
        with torch.cuda.amp.autocast(dtype=dtype):
            return model(return_loss=False, rescale=True, **data)


@dataclass(frozen=True, slots=True)
class VideoInferenceSettings:
    amp: str = "fp16"
    batch_size: int = 2
    score_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.amp not in {"fp16", "bf16", "off"}:
            raise ValueError(f"unsupported amp mode: {self.amp!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(slots=True)
class CoDinoRuntime:
    """Loaded Co-DINO model consumed by one selected execution backend."""

    model: torch.nn.Module
    classifier: torch.nn.Module | None
    class_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    target_size: tuple[int, int]
    fixed_batch_size: int
    backend: str

    def __post_init__(self) -> None:
        self.fixed_batch_size = int(self.fixed_batch_size)
        if self.fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive")
        if self.backend not in {"pytorch", "tensorrt"}:
            raise ValueError(f"unsupported Co-DINO backend: {self.backend}")


def build_runtime(
    *,
    segmenter_settings: InstanceSegmentationSettings,
    trt_settings: FixedTrtPartitionSettings | None,
    classifier_checkpoint: Path | None,
    device: str,
    fixed_batch_size: int,
    disable_mask_iou_head: bool = True,
) -> tuple[CoDinoRuntime, dict[str, Any]]:
    """Load PyTorch Co-DINO and optionally install one TensorRT bundle."""
    model = build_segmenter(segmenter_settings, device=device)
    if disable_mask_iou_head and hasattr(model, "mask_iou_head"):
        delattr(model, "mask_iou_head")
    if trt_settings is not None:
        install_fixed_partitions(model, trt_settings)
    classifier = None
    checkpoint: dict[str, Any] = {}
    if classifier_checkpoint is not None:
        (classifier, checkpoint) = classifier_from_checkpoint(
            classifier_checkpoint, map_location=device
        )
        classifier.to(device).eval()
    class_names = tuple(
        (str(value) for value in checkpoint.get("class_names", ["foreground"]))
    )
    class_ids = tuple(
        (
            int(value)
            for value in checkpoint.get("class_ids", list(range(len(class_names))))
        )
    )
    return (
        CoDinoRuntime(
            model=model,
            classifier=classifier,
            class_names=class_names,
            class_ids=class_ids,
            target_size=segmenter_settings.target_size,
            fixed_batch_size=fixed_batch_size,
            backend="tensorrt" if trt_settings is not None else "pytorch",
        ),
        checkpoint,
    )
