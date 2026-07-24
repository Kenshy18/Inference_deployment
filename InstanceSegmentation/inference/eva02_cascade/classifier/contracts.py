"""Static checkpoint and detector-feature contract for EVA02 classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MODEL_TYPES = frozenset({"rich_spatial_attn_fusion", "roi_rich_spatial_attn_fusion"})


def _normalize_meta_feature_set(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "legacy": "legacy_iou",
        "iou": "legacy_iou",
        "legacy_iou": "legacy_iou",
        "geo": "geo_v2",
        "geometry": "geo_v2",
        "geo_v2": "geo_v2",
    }
    result = aliases.get(normalized, normalized)
    if result not in {"legacy_iou", "geo_v2"}:
        raise ValueError(f"unsupported EVA02 meta_feature_set: {result!r}")
    return result


@dataclass(frozen=True, slots=True)
class RoiFeatureRequirements:
    box_head: bool = True
    box_pooler: bool = True
    expanded_box_pooler: bool = True
    mask_pooler: bool = True


@dataclass(frozen=True, slots=True)
class ClassifierCheckpointContract:
    model_type: str
    input_dim: int
    num_classes: int
    use_meta: bool
    feature_source: str
    meta_feature_set: str
    feature_l2norm: bool
    requirements: RoiFeatureRequirements


def contract_from_checkpoint(
    checkpoint: Mapping[str, Any]
) -> ClassifierCheckpointContract:
    cfg = checkpoint.get("model_cfg") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("EVA02 classifier model_cfg must be a mapping")
    model_type = str(cfg.get("model_type", "")).strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"EVA02 classifier requires rich_spatial_attn_fusion; got {model_type!r}"
        )
    input_dim = int(cfg.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"invalid classifier input_dim: {input_dim}")
    num_classes = int(cfg.get("num_classes", 0))
    if num_classes <= 0:
        num_classes = len(checkpoint.get("class_names") or ())
    if num_classes <= 0:
        raise ValueError(f"invalid classifier num_classes: {num_classes}")
    feature_source = str(cfg.get("feature_source", "pooler_flatten")).strip().lower()
    feature_source = {
        "roi_align_flatten": "pooler_flatten",
        "pooler_flatten": "pooler_flatten",
    }.get(feature_source, feature_source)
    if feature_source != "pooler_flatten":
        raise ValueError(
            f"EVA02 rich-spatial classifier requires pooler_flatten; got {feature_source!r}"
        )
    meta_feature_set = _normalize_meta_feature_set(
        cfg.get("meta_feature_set", "legacy_iou")
    )
    return ClassifierCheckpointContract(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        use_meta=bool(cfg.get("use_meta", True)),
        feature_source=feature_source,
        meta_feature_set=meta_feature_set,
        feature_l2norm=bool(cfg.get("feature_l2norm", False)),
        requirements=RoiFeatureRequirements(),
    )


__all__ = [
    "ClassifierCheckpointContract",
    "MODEL_TYPES",
    "RoiFeatureRequirements",
    "contract_from_checkpoint",
]
