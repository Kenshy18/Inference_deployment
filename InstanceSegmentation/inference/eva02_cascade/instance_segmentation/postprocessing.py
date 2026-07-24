"""Restore EVA02 Cascade boxes and masks to source-video coordinates."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from detectron2.structures import ROIMasks

from .preprocessing import LetterboxParameters


def restore_source_coordinates(
    instances,
    letterbox: LetterboxParameters,
    original_height: int,
    original_width: int,
    target_size: int,
    *,
    use_inplace_boxes: bool,
    use_raw_to_orig_masks: bool,
):
    """Apply validated restoration with optimization choices made explicit."""

    boxes = (
        instances.pred_boxes.tensor
        if use_inplace_boxes
        else instances.pred_boxes.tensor.detach().clone()
    )
    boxes[:, [0, 2]] -= letterbox.pad_left
    boxes[:, [1, 3]] -= letterbox.pad_top
    boxes /= letterbox.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height - 1)
    if not use_inplace_boxes:
        instances.pred_boxes.tensor = boxes

    if instances.has("pred_masks"):
        masks = instances.pred_masks
        mask_device = masks.device if isinstance(masks, torch.Tensor) else boxes.device
        if len(masks) > 0:
            mask_height, mask_width = masks.shape[-2:]
            if mask_height == target_size and mask_width == target_size:
                masks_4d = masks[:, None].float() if masks.ndim == 3 else masks.float()
                masks_4d = masks_4d[
                    :,
                    :,
                    letterbox.pad_top : letterbox.pad_top + letterbox.new_h,
                    letterbox.pad_left : letterbox.pad_left + letterbox.new_w,
                ]
                if masks_4d.shape[2] > 0 and masks_4d.shape[3] > 0:
                    instances.pred_masks = (
                        functional.interpolate(
                            masks_4d,
                            size=(original_height, original_width),
                            mode="nearest",
                        )[:, 0]
                        > 0.5
                    )
                else:
                    instances.pred_masks = torch.zeros(
                        (len(masks), original_height, original_width),
                        dtype=torch.bool,
                        device=mask_device,
                    )
            elif use_raw_to_orig_masks:
                roi_masks = masks[:, 0].float() if masks.ndim == 4 else masks.float()
                instances.pred_masks = (
                    ROIMasks(roi_masks)
                    .to_bitmasks(
                        instances.pred_boxes,
                        int(original_height),
                        int(original_width),
                        0.5,
                    )
                    .tensor
                )
        else:
            instances.pred_masks = torch.zeros(
                (0, original_height, original_width),
                dtype=torch.bool,
                device=mask_device,
            )
    return instances


__all__ = ["restore_source_coordinates"]
