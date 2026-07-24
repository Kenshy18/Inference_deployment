"""DINOv3 + Cascade Mask R-CNN instance-segmentation implementation."""

from .feature_export import configure_classifier_feature_export

__all__ = ["configure_classifier_feature_export"]
