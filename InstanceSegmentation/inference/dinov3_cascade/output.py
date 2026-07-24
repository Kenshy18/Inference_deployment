"""Canonical detector-row serialization for the DINOv3 Cascade family."""

from __future__ import annotations

import numpy as np
from mask_geometry import mask_to_polygons


AUXILIARY_INSTANCE_FIELDS = frozenset(
    {
        "pred_box_features",
        "pred_box_pooler_features",
        "pred_box_pooler_features_expanded",
        "pred_mask_pooler_features",
        "pred_multiclass_logits",
    }
)


def drop_auxiliary_instance_fields(instances) -> None:
    fields = getattr(instances, "_fields", None)
    if not isinstance(fields, dict):
        return
    for name in AUXILIARY_INSTANCE_FIELDS:
        fields.pop(name, None)


def instances_to_rows(
    instances,
    *,
    class_names: list[str],
    class_ids: list[int],
    score_threshold: float,
) -> list[dict[str, object]]:
    """Serialize source-coordinate instances with the validated field aliases."""

    if instances.has("scores") and score_threshold > 0:
        instances = instances[instances.scores >= score_threshold]
    if len(instances) == 0:
        return []
    boxes = (
        instances.pred_boxes.tensor.cpu().numpy()
        if instances.has("pred_boxes")
        else None
    )
    scores = instances.scores.float().cpu().numpy() if instances.has("scores") else None
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    class_indexes = (
        instances.pred_multiclass_classes.long().cpu().numpy()
        if instances.has("pred_multiclass_classes")
        else None
    )
    class_scores = (
        instances.pred_multiclass_scores.float().cpu().numpy()
        if instances.has("pred_multiclass_scores")
        else None
    )
    class_probabilities = (
        instances.pred_multiclass_probabilities.float().cpu().numpy()
        if instances.has("pred_multiclass_probabilities")
        else None
    )
    rows: list[dict[str, object]] = []
    for detection_index in range(len(instances)):
        if class_indexes is not None:
            class_index = int(class_indexes[detection_index])
            label = (
                class_names[class_index]
                if 0 <= class_index < len(class_names)
                else str(class_index)
            )
            category_id = (
                int(class_ids[class_index])
                if 0 <= class_index < len(class_ids)
                else class_index
            )
        else:
            class_index = 0
            label = class_names[0] if class_names else "foreground"
            category_id = int(class_ids[0]) if class_ids else 0
        item: dict[str, object] = {
            "label": label,
            "class_name": label,
            "category_id": category_id,
            "category_index": class_index,
        }
        if boxes is not None:
            x1, y1, x2, y2 = boxes[detection_index].tolist()
            item["bbox_xyxy"] = [float(x1), float(y1), float(x2), float(y2)]
            item["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if scores is not None:
            item["score"] = float(scores[detection_index])
            item["detector_score"] = float(scores[detection_index])
        if class_scores is not None:
            item["class_score"] = float(class_scores[detection_index])
        if class_probabilities is not None:
            item["class_probs"] = [
                float(value) for value in class_probabilities[detection_index]
            ]
        if masks is not None:
            mask = masks[detection_index]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_binary = mask.astype(np.uint8, copy=False)
            if mask_binary.max() > 1:
                mask_binary = (mask_binary > 0.5).astype(np.uint8)
            polygons = mask_to_polygons(mask_binary)
            item["segmentation"] = polygons
            item["polygons"] = polygons
        rows.append(item)
    return rows


__all__ = [
    "AUXILIARY_INSTANCE_FIELDS",
    "drop_auxiliary_instance_fields",
    "instances_to_rows",
    "mask_to_polygons",
]
