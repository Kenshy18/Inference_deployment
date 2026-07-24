"""Startup assembly for one cohesive EVA02 Cascade model family."""

from __future__ import annotations

from pathlib import Path

from .classifier.contracts import contract_from_checkpoint
from .classifier.loader import classifier_from_checkpoint
from .instance_segmentation.contracts import InstanceSegmentationSettings
from .instance_segmentation.feature_export import configure_classifier_feature_export
from .instance_segmentation.model import build_segmenter
from .runtime import Eva02Runtime
from .trt.bundle import Eva02TrtBundle
from .trt.runtime import Eva02TensorRTBackbone


def build_runtime(
    *,
    segmenter_settings: InstanceSegmentationSettings,
    classifier_checkpoint: Path,
    device: str,
    trt_bundle: Eva02TrtBundle | None = None,
) -> tuple[Eva02Runtime, dict[str, object]]:
    classifier, checkpoint = classifier_from_checkpoint(
        classifier_checkpoint, map_location=device
    )
    classifier.to(device).eval()
    contract = contract_from_checkpoint(checkpoint)
    segmenter = build_segmenter(segmenter_settings, device=device)
    backend = "pytorch"
    if trt_bundle is not None:
        if trt_bundle.target_size != segmenter_settings.target_size:
            raise ValueError(
                "EVA-02 TensorRT bundle target size does not match model settings"
            )
        segmenter.backbone.net = Eva02TensorRTBackbone(
            trt_bundle.engine_path,
            expected_input_name=trt_bundle.input_name,
            expected_output_name=trt_bundle.output_name,
        ).to(device)
        backend = "tensorrt-backbone"
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    configure_classifier_feature_export(segmenter, contract.requirements)
    class_names = tuple(str(value) for value in checkpoint.get("class_names", ()))
    class_ids = tuple(int(value) for value in checkpoint.get("class_ids", ()))
    if not class_names or len(class_names) != len(class_ids):
        raise ValueError(
            "classifier checkpoint must bind matching class_names/class_ids"
        )
    return (
        Eva02Runtime(
            segmenter=segmenter,
            classifier=classifier,
            class_names=class_names,
            class_ids=class_ids,
            target_size=segmenter_settings.target_size,
            meta_feature_set=contract.meta_feature_set,
            feature_l2norm=contract.feature_l2norm,
            backend=backend,
        ),
        checkpoint,
    )


__all__ = ["build_runtime"]
