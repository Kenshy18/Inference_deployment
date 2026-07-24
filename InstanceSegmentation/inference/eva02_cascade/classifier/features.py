"""Zero-copy EVA02 Cascade ROI feature access and metadata construction."""

from __future__ import annotations

import torch


META_FEATURE_SETS = frozenset({"legacy_iou", "geo_v2"})


def normalize_meta_feature_set(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "legacy": "legacy_iou",
        "iou": "legacy_iou",
        "legacy_iou": "legacy_iou",
        "geo": "geo_v2",
        "geometry": "geo_v2",
        "geo_v2": "geo_v2",
    }
    result = aliases.get(normalized, normalized)
    if result not in META_FEATURE_SETS:
        raise ValueError(f"unsupported meta feature set: {name!r}")
    return result


def _tensor_field(instances, name: str) -> torch.Tensor:
    if not instances.has(name):
        raise KeyError(f"instances has no {name!r}")
    value = getattr(instances, name)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    # The exported ROI features may arrive in autocast half precision; the
    # classifier weights are float32, and the validated runtime reads every
    # exported feature through an explicit .float() before classification.
    return tensor.float()


def _rank2(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim > 2:
        value = torch.flatten(value, start_dim=1)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got {tuple(value.shape)}")
    return value


def box_head_features(instances) -> torch.Tensor:
    return _rank2(_tensor_field(instances, "pred_box_features"), "pred_box_features")


def box_pooler_features(instances, *, expanded: bool = False) -> torch.Tensor:
    name = (
        "pred_box_pooler_features_expanded" if expanded else "pred_box_pooler_features"
    )
    value = _tensor_field(instances, name)
    if value.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got {tuple(value.shape)}")
    return value


def mask_pooler_features(instances) -> torch.Tensor:
    value = _tensor_field(instances, "pred_mask_pooler_features")
    if value.ndim != 4:
        raise ValueError(
            f"pred_mask_pooler_features must be rank-4, got {tuple(value.shape)}"
        )
    return value


def build_meta_features(
    instances,
    *,
    width: int,
    height: int,
    feature_set: str,
) -> torch.Tensor:
    count = len(instances)
    if count == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    image_area = float(max(1, int(width) * int(height)))
    boxes = instances.pred_boxes.tensor.float()
    device = boxes.device
    scores = (
        instances.scores.float().to(device, non_blocking=True)
        if instances.has("scores")
        else torch.ones((count,), dtype=torch.float32, device=device)
    )
    box_width = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    box_height = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    box_area = box_width * box_height

    mask_area = torch.zeros((count,), dtype=torch.float32, device=device)
    mask_area_ratio = torch.zeros((count,), dtype=torch.float32, device=device)
    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        if masks.ndim == 4:
            masks = masks[:, 0]
        if not hasattr(masks, "device") or masks.device == device:
            masks_float = masks.float()
            if (
                masks_float.ndim == 3
                and int(masks_float.shape[-2]) == int(height)
                and int(masks_float.shape[-1]) == int(width)
            ):
                mask_area = masks_float.flatten(start_dim=1).sum(dim=1)
            else:
                fill = (masks_float > 0.5).flatten(start_dim=1).float().mean(dim=1)
                mask_area = fill * box_area
            mask_area_ratio = mask_area / image_area

    normalized_set = normalize_meta_feature_set(feature_set)
    if normalized_set == "legacy_iou":
        fourth = torch.zeros((count,), dtype=torch.float32, device=device)
        fifth = torch.zeros((count,), dtype=torch.float32, device=device)
    else:
        fourth = torch.log((box_width + 1e-6) / (box_height + 1e-6))
        fifth = mask_area / torch.clamp(box_area, min=1e-6)
    return torch.stack(
        [scores, box_area / image_area, mask_area_ratio, fourth, fifth], dim=1
    )


__all__ = [
    "META_FEATURE_SETS",
    "box_head_features",
    "box_pooler_features",
    "build_meta_features",
    "mask_pooler_features",
    "normalize_meta_feature_set",
]
