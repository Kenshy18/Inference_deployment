"""Zero-copy Detectron instance feature access for the family classifier."""

from __future__ import annotations

import torch


def _tensor_field(instances, field_name: str) -> torch.Tensor:
    if not instances.has(field_name):
        raise KeyError(f"instances has no {field_name!r}")
    value = getattr(instances, field_name)
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def box_head_features(instances) -> torch.Tensor:
    value = _tensor_field(instances, "pred_box_features")
    if value.ndim > 2:
        value = torch.flatten(value, start_dim=1)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"pred_box_features must be rank-2, got {tuple(value.shape)}")
    return value


def box_pooler_features(instances) -> torch.Tensor:
    value = _tensor_field(instances, "pred_box_pooler_features")
    if value.ndim != 4:
        raise ValueError(
            "pred_box_pooler_features must be rank-4, " f"got {tuple(value.shape)}"
        )
    return value


def mask_pooler_features(instances) -> torch.Tensor:
    value = _tensor_field(instances, "pred_mask_pooler_features")
    if value.ndim != 4:
        raise ValueError(
            "pred_mask_pooler_features must be rank-4, " f"got {tuple(value.shape)}"
        )
    return value


def geo_v2_features(instances, target_size: int | tuple[int, int]) -> torch.Tensor:
    """Build score/area/aspect features without moving detector tensors."""

    if not instances.has("scores") or not instances.has("pred_boxes"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    scores = instances.scores.float()
    boxes = instances.pred_boxes.tensor.float()
    count = int(boxes.shape[0])
    if count == 0:
        return torch.empty((0, 5), device=boxes.device, dtype=torch.float32)
    if isinstance(target_size, int):
        target_h = target_w = int(target_size)
    else:
        target_h, target_w = (int(target_size[0]), int(target_size[1]))
    image_area = float(max(1, target_h * target_w))
    width = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    height = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    box_area = width * height
    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask_area = masks.flatten(1).float().sum(dim=1)
    else:
        mask_area = torch.zeros((count,), device=boxes.device, dtype=torch.float32)
    return torch.stack(
        [
            scores,
            box_area / image_area,
            mask_area / image_area,
            torch.log((width + 1e-6) / (height + 1e-6)),
            mask_area / torch.clamp(box_area, min=1e-6),
        ],
        dim=1,
    )


__all__ = [
    "box_head_features",
    "box_pooler_features",
    "geo_v2_features",
    "mask_pooler_features",
]
