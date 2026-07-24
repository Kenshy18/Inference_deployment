"""Canonical detector rows for the EVA02 Cascade family."""

from __future__ import annotations

import numpy as np
from mask_geometry import mask_to_polygons


AUXILIARY_FIELDS = (
    "pred_class_logits",
    "pred_box_features",
    "pred_box_pooler_features",
    "pred_box_pooler_features_expanded",
    "pred_mask_pooler_features",
)


def drop_auxiliary_fields(instances) -> None:
    for name in AUXILIARY_FIELDS:
        if instances.has(name):
            instances.remove(name)


def _mask_to_polygons_in_box(
    mask: np.ndarray,
    box: np.ndarray | None,
) -> list[list[float]]:
    if box is None:
        return mask_to_polygons(mask)
    height, width = mask.shape[:2]
    x1 = max(0, min(width, int(np.floor(float(box[0])))))
    y1 = max(0, min(height, int(np.floor(float(box[1])))))
    x2 = max(0, min(width, int(np.ceil(float(box[2])))))
    y2 = max(0, min(height, int(np.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return mask_to_polygons(mask)
    return mask_to_polygons(
        mask[y1:y2, x1:x2],
        x_offset=float(x1),
        y_offset=float(y1),
    )


def instances_to_rows(
    instances,
    *,
    class_names: tuple[str, ...],
    class_ids: tuple[int, ...],
    score_threshold: float,
) -> list[dict[str, object]]:
    if instances.has("scores") and score_threshold > 0:
        instances = instances[instances.scores >= score_threshold]
    if len(instances) == 0:
        return []
    boxes = (
        instances.pred_boxes.tensor.cpu().numpy()
        if instances.has("pred_boxes")
        else None
    )
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
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
    for index in range(len(instances)):
        class_index = int(class_indexes[index]) if class_indexes is not None else 0
        if not 0 <= class_index < len(class_names):
            class_index = 0
        class_name = class_names[class_index] if class_names else "foreground"
        category_id = class_ids[class_index] if class_index < len(class_ids) else 0
        item: dict[str, object] = {
            "class_name": str(class_name),
            "category_id": int(category_id),
        }
        box = None
        if boxes is not None:
            box = boxes[index]
            x1, y1, x2, y2 = box.tolist()
            item["bbox_xyxy"] = [float(x1), float(y1), float(x2), float(y2)]
            item["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if scores is not None:
            item["score"] = float(scores[index])
        if class_scores is not None:
            item["cls_score"] = float(class_scores[index])
        if class_probabilities is not None:
            item["class_probs"] = [
                float(value) for value in class_probabilities[index]
            ]
        if masks is not None:
            mask = masks[index]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_binary = mask.astype(np.uint8, copy=False)
            if mask_binary.max() > 1:
                mask_binary = (mask_binary > 0.5).astype(np.uint8)
            item["segmentation"] = _mask_to_polygons_in_box(mask_binary, box)
        rows.append(item)
    return rows


__all__ = ["AUXILIARY_FIELDS", "drop_auxiliary_fields", "instances_to_rows"]
