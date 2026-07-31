"""Restore Co-DINO results and normalize them for the shared contract."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mask_geometry import mask_to_polygons

try:
    from .preprocessing import letterbox_params
except ImportError:
    from preprocessing import letterbox_params


def normalize_result(result: Any):
    if isinstance(result, dict) and "ins_results" in result:
        result = result["ins_results"]
    if isinstance(result, tuple) and len(result) == 2:
        (boxes, segmentations) = result
        if isinstance(segmentations, tuple) and len(segmentations) == 2:
            segmentations = segmentations[0]
        return (boxes, segmentations)
    return (result, None)


def restore_boxes(
    box_results: list[np.ndarray],
    original_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    (original_height, original_width) = original_shape
    (target_height, target_width) = target_size
    (scale, _, _, pad_top, pad_left) = letterbox_params(
        original_height, original_width, target_height, target_width
    )
    offset_x = pad_left / scale if scale > 0 else 0.0
    offset_y = pad_top / scale if scale > 0 else 0.0
    restored = []
    for boxes in box_results:
        array = boxes.copy()
        if array.size:
            array[:, [0, 2]] -= offset_x
            array[:, [1, 3]] -= offset_y
            array[:, [0, 2]] = np.clip(array[:, [0, 2]], 0, original_width - 1)
            array[:, [1, 3]] = np.clip(array[:, [1, 3]], 0, original_height - 1)
        restored.append(array)
    return restored


def restore_mask(
    mask: Any, original_shape: tuple[int, int], target_size: tuple[int, int]
) -> np.ndarray:
    from pycocotools import mask as mask_utils

    (original_height, original_width) = original_shape
    (target_height, target_width) = target_size
    if isinstance(mask, dict) and "counts" in mask:
        mask = mask_utils.decode(mask)
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    if mask_u8.shape[:2] == (original_height, original_width):
        return mask_u8.astype(bool)
    if mask_u8.shape[:2] == (target_height, target_width):
        (_, new_height, new_width, pad_top, pad_left) = letterbox_params(
            original_height, original_width, target_height, target_width
        )
        mask_u8 = mask_u8[
            pad_top : pad_top + new_height, pad_left : pad_left + new_width
        ]
    if mask_u8.shape[:2] != (original_height, original_width):
        mask_u8 = cv2.resize(
            mask_u8, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        )
    return mask_u8.astype(bool)


def restore_segmentations(
    segmentation_results: list[list[Any]] | None,
    original_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[np.ndarray]] | None:
    if segmentation_results is None:
        return None
    return [
        [restore_mask(mask, original_shape, target_size) for mask in class_masks]
        for class_masks in segmentation_results
    ]


def detections_to_rows(
    box_results: list[np.ndarray],
    segmentation_results: list[list[np.ndarray]] | None,
    *,
    class_names: list[str],
    class_ids: list[int],
    score_threshold: float,
) -> list[dict[str, object]]:
    """Serialize restored masks and classifier columns to canonical rows."""
    rows: list[dict[str, object]] = []
    for (detector_class, boxes) in enumerate(box_results):
        if boxes is None or len(boxes) == 0:
            continue
        masks = (
            segmentation_results[detector_class]
            if segmentation_results is not None
            and detector_class < len(segmentation_results)
            else []
        )
        for (detection_index, detection) in enumerate(boxes):
            detector_score = float(detection[4])
            if detector_score < score_threshold:
                continue
            if detection.shape[0] >= 7 and class_names:
                class_index = int(round(float(detection[5])))
                classifier_class_name = (
                    class_names[class_index]
                    if 0 <= class_index < len(class_names)
                    else str(class_index)
                )
                classifier_class_id = (
                    int(class_ids[class_index])
                    if 0 <= class_index < len(class_ids)
                    else class_index
                )
                class_score: float | None = float(detection[6])
                probability_count = min(
                    len(class_names), max(0, int(detection.shape[0]) - 7)
                )
                class_probabilities = (
                    [float(value) for value in detection[7 : 7 + probability_count]]
                    if probability_count
                    else None
                )
            else:
                class_index = 0
                classifier_class_name = None
                classifier_class_id = None
                class_score = None
                class_probabilities = None
            (x1, y1, x2, y2) = (float(value) for value in detection[:4])
            row: dict[str, object] = {
                "label": "foreground",
                "class_name": "foreground",
                "category_id": int(detector_class),
                "category_index": int(detector_class),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": detector_score,
                "detector_score": detector_score,
            }
            if class_score is not None:
                row["class_score"] = class_score
                row["classifier_class_id"] = classifier_class_id
                row["classifier_class_name"] = classifier_class_name
            if class_probabilities is not None:
                row["class_probs"] = class_probabilities
            if detection_index < len(masks) and masks[detection_index] is not None:
                mask = np.asarray(masks[detection_index]).astype(np.uint8, copy=False)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                if mask.max(initial=0) > 1:
                    mask = (mask > 0.5).astype(np.uint8)
                polygons = mask_to_polygons(mask)
                row["segmentation"] = polygons
                row["polygons"] = polygons
            rows.append(row)
    return rows
