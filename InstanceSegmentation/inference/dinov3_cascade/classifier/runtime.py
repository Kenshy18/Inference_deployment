"""Direct tensor-loop integration for DINOv3 Cascade ROI classification."""

from __future__ import annotations

import torch

from .features import (
    box_head_features,
    box_pooler_features,
    geo_v2_features,
    mask_pooler_features,
)


def classify_outputs(
    classifier: torch.nn.Module,
    outputs: list[dict[str, object]],
    target_size: int | tuple[int, int],
    *,
    pad_to: int = 0,
    pad_multiple: int = 0,
) -> int:
    """Classify a detector batch while retaining every feature on device."""

    records: list[tuple[object, int]] = []
    roi_features: list[torch.Tensor] = []
    meta_features: list[torch.Tensor] = []
    head_features: list[torch.Tensor] = []
    mask_features: list[torch.Tensor] = []
    for output in outputs:
        instances = output["instances"]
        count = int(len(instances))
        if count == 0:
            continue
        roi_features.append(box_pooler_features(instances).flatten(start_dim=1))
        meta_features.append(geo_v2_features(instances, target_size))
        head_features.append(box_head_features(instances))
        mask_features.append(mask_pooler_features(instances))
        records.append((instances, count))

    total = sum(count for _, count in records)
    if total == 0:
        return 0
    roi = torch.cat(roi_features, dim=0)
    meta = torch.cat(meta_features, dim=0)
    head = torch.cat(head_features, dim=0)
    mask = torch.cat(mask_features, dim=0)

    target_total = total
    if pad_to > 0:
        target_total = max(target_total, int(pad_to))
    if pad_multiple > 0:
        multiple = int(pad_multiple)
        target_total = ((target_total + multiple - 1) // multiple) * multiple
    if target_total > total:
        pad_count = target_total - total

        def pad(value: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [value, value.new_zeros((pad_count, *value.shape[1:]))], dim=0
            )

        roi, meta, head, mask = map(pad, (roi, meta, head, mask))

    logits = classifier(
        roi,
        meta,
        box_head_feat=head,
        mask_pooler_feat=mask,
    )[:total]
    probabilities = torch.softmax(logits.float(), dim=1)
    class_scores, classes = torch.max(probabilities, dim=1)
    offset = 0
    for instances, count in records:
        next_offset = offset + count
        instances.pred_multiclass_logits = logits[offset:next_offset]
        instances.pred_multiclass_classes = classes[offset:next_offset]
        instances.pred_multiclass_scores = class_scores[offset:next_offset]
        instances.pred_multiclass_probabilities = probabilities[offset:next_offset]
        offset = next_offset
    return total


__all__ = ["classify_outputs"]
