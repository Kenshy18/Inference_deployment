"""Batched mask postprocessing for the fixed-batch optimized MH0 path."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch


def _as_image_lists(values: Any, count: int) -> list[torch.Tensor]:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values[index] for index in range(count)]


def simple_test_mask_batched(
    self,
    x,
    img_metas,
    det_bboxes,
    det_labels,
    rescale: bool = False,
):
    """Run RoI extraction and the mask core once across the image batch."""

    from mmdet.core import bbox2roi
    from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
        _do_paste_mask,
    )

    bbox_list = _as_image_lists(det_bboxes, len(img_metas))
    label_list = _as_image_lists(det_labels, len(img_metas))
    roi_boxes: list[torch.Tensor] = []
    entries: list[dict[str, Any]] = []

    for image_index, img_meta in enumerate(img_metas):
        boxes = bbox_list[image_index]
        labels = label_list[image_index]
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        original_shape = img_meta["ori_shape"]
        scale_factor = img_meta["scale_factor"]
        mask_out_shape = original_shape
        mask_scale_factor: Any = scale_factor
        if not rescale:
            mask_out_shape = img_meta.get("img_shape", original_shape)
            mask_scale_factor = 1.0

        if boxes.shape[0] and rescale:
            scale = self._bbox_scale_tensor(scale_factor, boxes)
            paste_boxes = boxes[:, :4] * scale
            output_boxes = paste_boxes / scale
            feature_boxes = paste_boxes
            pad_left, pad_top, input_width, input_height = (
                self._letterbox_pad_from_meta(img_meta, scale_factor)
            )
            if pad_left or pad_top:
                feature_boxes = paste_boxes.clone()
                feature_boxes[:, 0::2] += pad_left
                feature_boxes[:, 1::2] += pad_top
                feature_boxes[:, 0::2].clamp_(
                    min=0, max=max(input_width, 1)
                )
                feature_boxes[:, 1::2].clamp_(
                    min=0, max=max(input_height, 1)
                )
        else:
            feature_boxes = boxes
            paste_boxes = boxes
            output_boxes = boxes[:, :4]

        roi_boxes.append(feature_boxes)
        entries.append(
            {
                "count": int(boxes.shape[0]),
                "labels": labels,
                "paste_boxes": paste_boxes,
                "output_boxes": output_boxes,
                "mask_out_shape": mask_out_shape,
                "mask_scale_factor": mask_scale_factor,
            }
        )

    total_rois = sum(entry["count"] for entry in entries)
    if total_rois:
        mask_rois = bbox2roi(roi_boxes)
        feature_dtype = x[0].dtype
        if mask_rois.dtype != feature_dtype:
            mask_rois = mask_rois.to(dtype=feature_dtype)
        all_labels = torch.cat(
            [entry["labels"] for entry in entries if entry["count"]], 0
        )
        mask_results = self._mask_forward(x, mask_rois, all_labels)
        instance_predictions = mask_results["stage_instance_preds"][-1]
    else:
        instance_predictions = None

    threshold = self.rcnn_test_cfg.mask_thr_binary
    output_shapes = {
        tuple(entry["mask_out_shape"][:2]) for entry in entries
    }
    if (
        total_rois
        and threshold >= 0
        and len(output_shapes) == 1
    ):
        img_h, img_w = next(iter(output_shapes))
        all_boxes = torch.cat(
            [entry["output_boxes"] for entry in entries if entry["count"]],
            dim=0,
        )
        mask_predictions = instance_predictions.sigmoid()
        if mask_predictions.shape[1] > 1:
            mask_predictions = mask_predictions[
                torch.arange(total_rois, device=mask_predictions.device),
                all_labels,
            ][:, None]
        cropped_masks, spatial = _do_paste_mask(
            mask_predictions,
            all_boxes,
            img_h,
            img_w,
            skip_empty=True,
        )
        cropped_masks = (cropped_masks >= threshold).to(torch.bool)
        cropped_cpu = cropped_masks.cpu().numpy()
        y_start = int(spatial[0].start)
        y_stop = int(spatial[0].stop)
        x_start = int(spatial[1].start)
        x_stop = int(spatial[1].stop)
        full_masks = np.zeros(
            (total_rois, img_h, img_w), dtype=np.bool_
        )
        full_masks[
            :,
            y_start:y_stop,
            x_start:x_stop,
        ] = cropped_cpu
        labels_cpu = all_labels.detach().cpu().tolist()

        segmentation_results = []
        offset = 0
        for entry in entries:
            class_results = [
                [] for _ in range(self.mask_head.stage_num_classes[0])
            ]
            for index in range(entry["count"]):
                class_results[labels_cpu[offset + index]].append(
                    full_masks[offset + index]
                )
            offset += entry["count"]
            segmentation_results.append(class_results)
        return segmentation_results

    segmentation_results = []
    offset = 0
    for entry in entries:
        class_results = [
            [] for _ in range(self.mask_head.stage_num_classes[0])
        ]
        count = entry["count"]
        if count:
            end = offset + count
            masks = self.mask_head.get_seg_masks(
                instance_predictions[offset:end],
                entry["paste_boxes"],
                entry["labels"],
                self.rcnn_test_cfg,
                entry["mask_out_shape"],
                entry["mask_scale_factor"],
                rescale,
            )
            for label, mask in zip(entry["labels"], masks):
                class_results[int(label)].append(mask)
            offset = end
        segmentation_results.append(class_results)
    return segmentation_results


def install_batched_mask_test(model: torch.nn.Module) -> torch.nn.Module:
    model.simple_test_mask = types.MethodType(simple_test_mask_batched, model)
    return model


__all__ = ["install_batched_mask_test", "simple_test_mask_batched"]
