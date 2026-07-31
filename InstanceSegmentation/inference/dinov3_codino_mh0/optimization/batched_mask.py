"""Batched mask postprocessing for the fixed-batch optimized MH0 path."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch


def _store_classifier_results(
    model,
    entries: list[dict[str, Any]],
    mask_coverages: torch.Tensor,
) -> None:
    """Classify every RoI from the detector's retained backbone feature."""

    classifier = getattr(model, "_mh0_classifier", None)
    if classifier is None:
        model._mh0_last_classifications = None
        return
    counts = [int(entry["count"]) for entry in entries]
    total = sum(counts)
    class_count = int(model._mh0_classifier_class_count)
    if total == 0:
        model._mh0_last_classifications = tuple(
            entry["boxes"].new_zeros((0, 2 + class_count))
            for entry in entries
        )
        return

    boxes = torch.cat(
        [entry["output_boxes"][:, :4] for entry in entries if entry["count"]],
        dim=0,
    ).float()
    scores = torch.cat(
        [entry["boxes"][:, 4] for entry in entries if entry["count"]],
        dim=0,
    ).float()
    image_areas = torch.cat(
        [
            boxes.new_full(
                (int(entry["count"]),),
                float(
                    max(
                        1,
                        int(entry["mask_out_shape"][0])
                        * int(entry["mask_out_shape"][1]),
                    )
                ),
            )
            for entry in entries
            if entry["count"]
        ],
        dim=0,
    )
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    box_areas = widths * heights
    coverages = mask_coverages.to(
        device=boxes.device,
        dtype=torch.float32,
    ).clamp_(0, 1)
    areas = coverages * box_areas
    metadata = torch.stack(
        (
            scores,
            box_areas / image_areas,
            areas / image_areas,
            torch.log((widths + 1e-6) / (heights + 1e-6)),
            areas / box_areas.clamp_min(1e-6),
        ),
        dim=1,
    )
    feature_boxes = torch.cat(
        [entry["feature_boxes"] for entry in entries if entry["count"]],
        dim=0,
    )
    batch_indices = torch.cat(
        [
            feature_boxes.new_full((int(entry["count"]),), image_index)
            for image_index, entry in enumerate(entries)
            if entry["count"]
        ]
    )
    backbone_feature = getattr(model, "_mh0_backbone_feature", None)
    if backbone_feature is None:
        raise RuntimeError("MH0 backbone feature was not retained for classification")
    classes, scores_out, probabilities = classifier.classify_backbone(
        backbone_feature,
        feature_boxes,
        metadata,
        batch_indices=batch_indices,
    )
    packed = torch.cat(
        (
            classes.to(dtype=probabilities.dtype).unsqueeze(1),
            scores_out.unsqueeze(1),
            probabilities,
        ),
        dim=1,
    )
    model._mh0_last_classifications = tuple(
        packed[offset : offset + count]
        for offset, count in _offsets(counts)
    )


def _offsets(counts: list[int]):
    offset = 0
    for count in counts:
        yield offset, count
        offset += count


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
                "boxes": boxes,
                "labels": labels,
                "paste_boxes": paste_boxes,
                "output_boxes": output_boxes,
                "feature_boxes": feature_boxes,
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
        mask_coverages = (
            mask_predictions >= threshold
        ).float().mean(dim=(1, 2, 3))
        cropped_masks, spatial = _do_paste_mask(
            mask_predictions,
            all_boxes,
            img_h,
            img_w,
            skip_empty=True,
        )
        cropped_masks = (cropped_masks >= threshold).to(torch.bool)
        _store_classifier_results(
            self,
            entries,
            mask_coverages,
        )
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
    if total_rois:
        mask_predictions = instance_predictions.sigmoid()
        if mask_predictions.shape[1] > 1:
            mask_predictions = mask_predictions[
                torch.arange(total_rois, device=mask_predictions.device),
                all_labels,
            ][:, None]
        mask_coverages = (
            mask_predictions >= threshold
        ).float().mean(dim=(1, 2, 3))
        _store_classifier_results(
            self,
            entries,
            mask_coverages,
        )
    elif getattr(self, "_mh0_classifier", None) is not None:
        class_count = int(self._mh0_classifier_class_count)
        self._mh0_last_classifications = tuple(
            entry["boxes"].new_zeros((0, 2 + class_count))
            for entry in entries
        )
    return segmentation_results


def install_batched_mask_test(model: torch.nn.Module) -> torch.nn.Module:
    model.simple_test_mask = types.MethodType(simple_test_mask_batched, model)
    return model


__all__ = ["install_batched_mask_test", "simple_test_mask_batched"]
