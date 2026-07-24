"""Startup assembly for one cohesive DINOv3 Cascade model family."""

from __future__ import annotations

from pathlib import Path

from .classifier.loader import classifier_from_checkpoint
from .instance_segmentation.contracts import InstanceSegmentationSettings
from .instance_segmentation.feature_export import configure_classifier_feature_export
from .instance_segmentation.model import build_instance_segmenter
from .runtime import Dinov3Runtime


def build_runtime(
    *,
    segmenter_settings: InstanceSegmentationSettings,
    classifier_checkpoint: Path | None,
    device: str,
) -> tuple[Dinov3Runtime, dict[str, object]]:
    """Bind the only approved segmenter/classifier combination once at startup."""

    classifier = None
    checkpoint: dict[str, object] = {}
    if classifier_checkpoint is not None:
        classifier, checkpoint = classifier_from_checkpoint(
            classifier_checkpoint, map_location=device
        )
        classifier.to(device).eval()
    segmenter = build_instance_segmenter(segmenter_settings, device=device)
    if classifier is not None:
        from .classifier.contracts import contract_from_checkpoint

        configure_classifier_feature_export(
            segmenter,
            contract_from_checkpoint(checkpoint).feature_requirements,
        )
    class_names = tuple(
        str(value) for value in checkpoint.get("class_names", ["foreground"])
    )
    class_ids = tuple(
        int(value)
        for value in checkpoint.get("class_ids", list(range(len(class_names))))
    )
    return (
        Dinov3Runtime(
            segmenter=segmenter,
            classifier=classifier,
            class_names=class_names,
            class_ids=class_ids,
            target_size=segmenter_settings.target_size,
        ),
        checkpoint,
    )


__all__ = ["build_runtime"]
