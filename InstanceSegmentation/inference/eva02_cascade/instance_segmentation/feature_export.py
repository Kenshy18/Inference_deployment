"""Apply the EVA02 classifier's typed feature request to Cascade ROI heads."""

from __future__ import annotations

from ..classifier.contracts import RoiFeatureRequirements


def configure_classifier_feature_export(
    model, requirements: RoiFeatureRequirements
) -> None:
    if not hasattr(model, "roi_heads"):
        raise AttributeError("EVA02 segmenter has no roi_heads")
    model.roi_heads.return_box_features = requirements.box_head
    model.roi_heads.return_box_pooler_features = requirements.box_pooler
    model.roi_heads.return_box_pooler_features_expanded = (
        requirements.expanded_box_pooler
    )
    model.roi_heads.return_mask_pooler_features = requirements.mask_pooler


__all__ = ["configure_classifier_feature_export"]
