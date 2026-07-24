"""Direct expanded-feature classification used inside the EVA02 family loop."""

from __future__ import annotations

import torch

from .features import (
    box_head_features,
    box_pooler_features,
    build_meta_features,
    mask_pooler_features,
)


def classify_features(
    classifier: torch.nn.Module,
    *,
    roi_features: torch.Tensor,
    meta_features: torch.Tensor,
    box_head_features: torch.Tensor,
    expanded_box_pooler_features: torch.Tensor,
    mask_pooler_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = classifier(
        roi_features,
        meta_features,
        box_head_feat=box_head_features,
        box_pooler_feat_expanded=expanded_box_pooler_features,
        mask_pooler_feat=mask_pooler_features,
    )
    probabilities = torch.softmax(logits.float(), dim=1)
    scores, classes = torch.max(probabilities, dim=1)
    return classes, scores, probabilities


def classify_instances(
    classifier: torch.nn.Module,
    instances,
    *,
    width: int,
    height: int,
    meta_feature_set: str,
    feature_l2norm: bool = False,
) -> int:
    """Classify restored instances without converting family-owned tensors."""

    count = len(instances)
    if count == 0:
        return 0
    roi = box_pooler_features(instances).flatten(start_dim=1)
    if feature_l2norm:
        roi = torch.nn.functional.normalize(roi, p=2, dim=1)
    classes, scores, probabilities = classify_features(
        classifier,
        roi_features=roi,
        meta_features=build_meta_features(
            instances,
            width=width,
            height=height,
            feature_set=meta_feature_set,
        ),
        box_head_features=box_head_features(instances),
        expanded_box_pooler_features=box_pooler_features(instances, expanded=True),
        mask_pooler_features=mask_pooler_features(instances),
    )
    instances.pred_multiclass_classes = classes
    instances.pred_multiclass_scores = scores
    instances.pred_multiclass_probabilities = probabilities
    return count


def classify_instance_batch(
    classifier: torch.nn.Module,
    instances_by_frame: list[object],
    *,
    width: int,
    height: int,
    meta_feature_set: str,
    feature_l2norm: bool = False,
    max_instances_per_batch: int = 2048,
) -> int:
    """Classify all non-empty frames without crossing a tensor interface."""

    records: list[tuple[object, int]] = []
    roi_chunks: list[torch.Tensor] = []
    metadata_chunks: list[torch.Tensor] = []
    box_head_chunks: list[torch.Tensor] = []
    expanded_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    for instances in instances_by_frame:
        count = len(instances)  # type: ignore[arg-type]
        if count == 0:
            continue
        roi_chunks.append(box_pooler_features(instances).flatten(start_dim=1))
        metadata_chunks.append(
            build_meta_features(
                instances,
                width=width,
                height=height,
                feature_set=meta_feature_set,
            )
        )
        box_head_chunks.append(box_head_features(instances))
        expanded_chunks.append(box_pooler_features(instances, expanded=True))
        mask_chunks.append(mask_pooler_features(instances))
        records.append((instances, count))
    total = sum(count for _, count in records)
    if total == 0:
        return 0

    roi = torch.cat(roi_chunks, dim=0).contiguous()
    metadata = torch.cat(metadata_chunks, dim=0).contiguous()
    box_head = torch.cat(box_head_chunks, dim=0).contiguous()
    expanded = torch.cat(expanded_chunks, dim=0).contiguous()
    mask = torch.cat(mask_chunks, dim=0).contiguous()
    if feature_l2norm:
        roi = torch.nn.functional.normalize(roi, p=2, dim=1)
    limit = max(1, int(max_instances_per_batch))
    class_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    probability_chunks: list[torch.Tensor] = []
    for start in range(0, total, limit):
        end = min(start + limit, total)
        classes, scores, probabilities = classify_features(
            classifier,
            roi_features=roi[start:end],
            meta_features=metadata[start:end],
            box_head_features=box_head[start:end],
            expanded_box_pooler_features=expanded[start:end],
            mask_pooler_features=mask[start:end],
        )
        class_chunks.append(classes)
        score_chunks.append(scores)
        probability_chunks.append(probabilities)
    classes = torch.cat(class_chunks, dim=0)
    scores = torch.cat(score_chunks, dim=0)
    probabilities = torch.cat(probability_chunks, dim=0)
    offset = 0
    for instances, count in records:
        end = offset + count
        instances.pred_multiclass_classes = classes[offset:end]  # type: ignore[attr-defined]
        instances.pred_multiclass_scores = scores[offset:end]  # type: ignore[attr-defined]
        instances.pred_multiclass_probabilities = probabilities[offset:end]  # type: ignore[attr-defined]
        offset = end
    return total


__all__ = ["classify_features", "classify_instance_batch", "classify_instances"]
