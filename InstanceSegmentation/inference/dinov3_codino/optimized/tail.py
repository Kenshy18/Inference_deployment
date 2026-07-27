"""Split fixed-B2 Co-DINO core and batched mask/classifier tail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from ..classifier import (
        classifier_metadata,
        classify_mask_features,
        paste_boxes_for_masks,
        refine_stage_instance_predictions,
        scale_factor_tensor,
    )
    from ..preprocessing import move_prepared_batch
except ImportError:
    from classifier import (
        classifier_metadata,
        classify_mask_features,
        paste_boxes_for_masks,
        refine_stage_instance_predictions,
        scale_factor_tensor,
    )
    from preprocessing import move_prepared_batch


@dataclass(slots=True)
class FastCorePayload:
    query_outputs: tuple[Any, ...]
    image_metadata: list[dict[str, Any]]


def _copy_tree(value, destination=None):
    if isinstance(value, torch.Tensor):
        if (
            destination is None
            or destination.shape != value.shape
            or destination.dtype != value.dtype
            or destination.device != value.device
        ):
            destination = torch.empty_like(value)
        destination.copy_(value)
        return destination
    if isinstance(value, tuple):
        previous = destination if isinstance(destination, tuple) else ()
        return tuple(
            _copy_tree(item, previous[index] if index < len(previous) else None)
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        previous = destination if isinstance(destination, list) else []
        return [
            _copy_tree(item, previous[index] if index < len(previous) else None)
            for index, item in enumerate(value)
        ]
    return value


def capture_fast_core(
    model,
    *,
    detector_graph,
    prepared_data,
    destination=None,
) -> FastCorePayload:
    """Replay the stable core and copy outputs into one pipeline slot."""

    data = move_prepared_batch(model, prepared_data)
    image = data["img"][0]
    image_metadata = data["img_metas"][0]
    batch_input_shape = tuple(image[0].size()[-2:])
    for metadata in image_metadata:
        metadata["batch_input_shape"] = batch_input_shape
    if hasattr(model, "with_attn_mask") and not model.with_attn_mask:
        for metadata in image_metadata:
            height, width = metadata["batch_input_shape"]
            metadata["img_shape"] = [height, width, 3]
    query_outputs, _ = detector_graph.run(image, image_metadata)
    return FastCorePayload(
        query_outputs=_copy_tree(query_outputs, destination),
        image_metadata=image_metadata,
    )


def infer_fast_b2_tail(
    model,
    *,
    payload: FastCorePayload,
    target_size: tuple[int, int],
    classifier: torch.nn.Module,
    num_classifier_classes: int,
):
    """Run one combined mask/classifier tail for both fixed-B2 images."""

    from mmdet.core import bbox2result, bbox2roi

    query_outputs = payload.query_outputs
    image_metadata = payload.image_metadata
    features = query_outputs[-1]
    with_nms = model.query_head.test_cfg.get("nms", None) is not None
    results = model.query_head.get_bboxes(
        *query_outputs,
        image_metadata,
        rescale=True,
        with_nms=with_nms,
    )
    detector_classes = int(getattr(model.query_head, "num_classes", 1))
    boxes_by_image = []
    labels_by_image = []
    scaled_by_image = []
    paste_by_image = []
    scale_factors_by_image = []
    for (boxes, labels), metadata in zip(results, image_metadata):
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        boxes_by_image.append(boxes)
        labels_by_image.append(labels)
        scale_factor = scale_factor_tensor(
            metadata,
            boxes.device,
            boxes.dtype,
        )
        scale_factors_by_image.append(scale_factor)
        scaled_by_image.append(boxes[:, :4] * scale_factor)
        paste_by_image.append(
            paste_boxes_for_masks(boxes, metadata, scale_factor)
        )

    total = sum(int(boxes.shape[0]) for boxes in boxes_by_image)
    if total == 0:
        return [
            (
                bbox2result(boxes, labels, detector_classes),
                [[] for _ in range(model.mask_head.stage_num_classes[0])],
            )
            for boxes, labels in zip(boxes_by_image, labels_by_image)
        ]

    rois = bbox2roi(scaled_by_image)
    if rois.dtype != features[0].dtype:
        rois = rois.to(dtype=features[0].dtype)
    all_labels = torch.cat(labels_by_image)
    mask_result = model._mask_forward(features, rois, all_labels)
    instance_prediction = refine_stage_instance_predictions(
        mask_result["stage_instance_preds"]
    )

    masks_by_image = []
    start = 0
    for boxes, labels, paste_boxes, scale_factor, metadata in zip(
        boxes_by_image,
        labels_by_image,
        paste_by_image,
        scale_factors_by_image,
        image_metadata,
    ):
        end = start + int(boxes.shape[0])
        if end == start:
            masks_by_image.append([])
            continue
        masks_by_image.append(
            list(
                model.mask_head.get_seg_masks(
                    instance_prediction[start:end],
                    paste_boxes,
                    labels,
                    model.rcnn_test_cfg,
                    metadata["ori_shape"],
                    scale_factor,
                    True,
                )
            )
        )
        start = end

    metadata_features = torch.cat(
        [
            classifier_metadata(
                boxes,
                masks,
                metadata,
                target_size,
                boxes.device,
            )
            for boxes, masks, metadata in zip(
                boxes_by_image,
                masks_by_image,
                image_metadata,
            )
        ]
    )
    classes, scores, probabilities = classify_mask_features(
        classifier,
        mask_result["mask_feats"],
        metadata_features,
    )

    outputs = []
    start = 0
    for boxes, labels, masks in zip(
        boxes_by_image,
        labels_by_image,
        masks_by_image,
    ):
        end = start + int(boxes.shape[0])
        extra = boxes.new_zeros(
            (int(boxes.shape[0]), 2 + int(num_classifier_classes))
        )
        extra[:, 0] = classes[start:end].to(dtype=extra.dtype)
        extra[:, 1] = scores[start:end].to(dtype=extra.dtype)
        extra[:, 2 : 2 + int(num_classifier_classes)] = probabilities[
            start:end
        ].to(dtype=extra.dtype)
        outputs.append(
            (
                bbox2result(
                    torch.cat([boxes, extra], dim=1),
                    labels,
                    detector_classes,
                ),
                [masks],
            )
        )
        start = end
    return outputs


__all__ = [
    "FastCorePayload",
    "capture_fast_core",
    "infer_fast_b2_tail",
]
