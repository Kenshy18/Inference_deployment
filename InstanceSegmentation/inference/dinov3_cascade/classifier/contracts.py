"""Static contract for the DINOv3 Cascade family classifier.

The production checkpoint selects one model shape.  Keeping that fact here
makes the segmenter-to-classifier feature contract inspectable without
importing PyTorch or Detectron2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MODEL_TYPES = frozenset(
    {
        "rich_spatial_attn_no_expanded_fusion",
        "roi_rich_spatial_attn_no_expanded_fusion",
    }
)


@dataclass(frozen=True, slots=True)
class RoiFeatureRequirements:
    """Detector tensors consumed directly by the family classifier."""

    box_head: bool = True
    box_pooler: bool = True
    expanded_box_pooler: bool = False
    mask_pooler: bool = True


@dataclass(frozen=True, slots=True)
class ClassifierCheckpointContract:
    """Validated metadata needed to construct the selected classifier."""

    model_type: str
    input_dim: int
    num_classes: int
    use_meta: bool
    feature_requirements: RoiFeatureRequirements


def contract_from_checkpoint(
    checkpoint: Mapping[str, Any]
) -> ClassifierCheckpointContract:
    """Fail closed when a checkpoint is not the family-approved architecture."""

    cfg = checkpoint.get("model_cfg") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("classifier checkpoint model_cfg must be a mapping")
    model_type = str(cfg.get("model_type", "")).strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(
            "DINOv3 Cascade classifier requires "
            "rich_spatial_attn_no_expanded_fusion; "
            f"got {model_type!r}"
        )
    input_dim = int(cfg.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"invalid classifier input_dim: {input_dim}")
    num_classes = int(cfg.get("num_classes", 0))
    if num_classes <= 0:
        class_names = checkpoint.get("class_names") or ()
        num_classes = len(class_names)
    if num_classes <= 0:
        raise ValueError(f"invalid classifier num_classes: {num_classes}")
    return ClassifierCheckpointContract(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        use_meta=bool(cfg.get("use_meta", True)),
        feature_requirements=RoiFeatureRequirements(),
    )


__all__ = [
    "ClassifierCheckpointContract",
    "MODEL_TYPES",
    "RoiFeatureRequirements",
    "contract_from_checkpoint",
]
