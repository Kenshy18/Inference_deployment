"""Restore DINOv3 Cascade detections from model to source coordinates."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .preprocessing import LetterboxParameters, TargetSize, unpack_size


def restore_source_coordinates(
    instances,
    letterbox: LetterboxParameters,
    original_height: int,
    original_width: int,
    target_size: TargetSize,
):
    """Apply the validated box/mask unletterbox rules to Detectron instances."""

    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.detach().clone()
        boxes[:, [0, 2]] -= float(letterbox.pad_left)
        boxes[:, [1, 3]] -= float(letterbox.pad_top)
        boxes /= float(letterbox.scale)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height - 1)
        instances.pred_boxes.tensor = boxes

    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        target_h, target_w = unpack_size(target_size)
        if (
            masks.ndim == 3
            and masks.shape[1] == target_h
            and masks.shape[2] == target_w
        ):
            cropped = masks[
                :,
                letterbox.pad_top : letterbox.pad_top + letterbox.new_h,
                letterbox.pad_left : letterbox.pad_left + letterbox.new_w,
            ]
            if cropped.shape[1] > 0 and cropped.shape[2] > 0:
                resized = [
                    cv2.resize(
                        mask,
                        (original_width, original_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for mask in cropped.cpu().numpy().astype(np.uint8)
                ]
                instances.pred_masks = torch.from_numpy(np.stack(resized, axis=0) > 0)
            else:
                instances.pred_masks = torch.zeros(
                    (len(masks), original_height, original_width), dtype=torch.bool
                )
    return instances


__all__ = ["restore_source_coordinates"]
