"""Shared DINOv3-backbone ROI classifier inference runtime."""

from .runtime import (
    CANONICAL_CLASS_IDS,
    CANONICAL_CLASS_NAMES,
    BackboneRoiClassifier,
    load_classifier_manifest,
)

__all__ = [
    "CANONICAL_CLASS_IDS",
    "CANONICAL_CLASS_NAMES",
    "BackboneRoiClassifier",
    "load_classifier_manifest",
]
