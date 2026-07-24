"""ROI feature-export wiring owned by the family instance segmenter."""

from __future__ import annotations

from ..classifier.contracts import (
    RoiFeatureRequirements,
)


def configure_classifier_feature_export(
    model: object, requirements: RoiFeatureRequirements
) -> None:
    """Configure the existing ROI heads without adding an adapter or copy."""

    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        raise RuntimeError("model.roi_heads not found")
    setattr(roi_heads, "return_box_features", requirements.box_head)
    setattr(roi_heads, "return_box_pooler_features", requirements.box_pooler)
    setattr(
        roi_heads,
        "return_box_pooler_features_expanded",
        requirements.expanded_box_pooler,
    )
    setattr(roi_heads, "box_pooler_expanded_scale", 2.0)
    setattr(roi_heads, "return_mask_pooler_features", requirements.mask_pooler)


__all__ = ["configure_classifier_feature_export"]
