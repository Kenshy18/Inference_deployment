"""Batched bbox postprocessing for fixed-batch TensorRT inference."""

from __future__ import annotations

import types

import numpy as np
import torch


def _with_classifier_columns(model, results_list):
    classifications = getattr(model, "_mh0_last_classifications", None)
    model._mh0_last_classifications = None
    if classifications is None:
        return results_list
    if len(classifications) != len(results_list):
        raise RuntimeError(
            "MH0 classifier result count does not match the image batch"
        )
    enriched = []
    for (boxes, labels), extra in zip(results_list, classifications):
        if int(boxes.shape[0]) != int(extra.shape[0]):
            raise RuntimeError(
                "MH0 classifier result count does not match detections"
            )
        enriched.append(
            (
                torch.cat(
                    (
                        boxes,
                        extra.to(device=boxes.device, dtype=boxes.dtype),
                    ),
                    dim=1,
                ),
                labels,
            )
        )
    return enriched


def _bbox_results_batched(results_list, num_classes):
    counts = [int(boxes.shape[0]) for boxes, _ in results_list]
    total = sum(counts)
    if not total:
        return [
            [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
            for _ in results_list
        ]

    boxes = torch.cat(
        [value for value, _ in results_list if value.shape[0]], dim=0
    )
    labels = torch.cat(
        [value for boxes_i, value in results_list if boxes_i.shape[0]], dim=0
    )
    packed = torch.cat(
        (boxes, labels.to(dtype=boxes.dtype).unsqueeze(1)), dim=1
    ).detach().cpu().numpy()
    outputs = []
    offset = 0
    for count in counts:
        image = packed[offset : offset + count]
        image_boxes = image[:, :-1]
        image_labels = image[:, -1].astype(np.int64, copy=False)
        outputs.append(
            [
                image_boxes[image_labels == class_index, :]
                for class_index in range(num_classes)
            ]
        )
        offset += count
    return outputs


def simple_test_query_head_batched(
    self, img, img_metas, proposals=None, rescale=False
):
    """Keep bbox results on the GPU until mask inference has completed."""

    del proposals
    batch_input_shape = tuple(img[0].size()[-2:])
    for img_meta in img_metas:
        img_meta["batch_input_shape"] = batch_input_shape
    graph_core = getattr(self, "_mh0_cuda_graph_core", None)
    if graph_core is None:
        pyramid = self.extract_feat(img, img_metas)
        query_features = self._query_features(pyramid)
        results_list, encoded = self.query_head.simple_test(
            query_features,
            img_metas,
            rescale=rescale,
            return_encoder_output=True,
        )
    else:
        pyramid, outputs = graph_core(img, img_metas)
        with_nms = self.query_head.test_cfg.get("nms", None) is not None
        results_list = self.query_head.get_bboxes(
            *outputs,
            img_metas,
            rescale=rescale,
            with_nms=with_nms,
        )
        encoded = outputs[-1]
    if not hasattr(self, "mask_head"):
        return _bbox_results_batched(
            results_list, self.query_head.num_classes
        )

    mask_features = self._mask_features(pyramid, encoded)
    det_bboxes = [boxes for boxes, _ in results_list]
    det_labels = [labels for _, labels in results_list]
    segm_results = self.simple_test_mask(
        mask_features,
        img_metas,
        det_bboxes,
        det_labels,
        rescale=rescale,
    )
    results_list = _with_classifier_columns(self, results_list)
    bbox_results = _bbox_results_batched(
        results_list, self.query_head.num_classes
    )
    return list(zip(bbox_results, segm_results))


def get_bboxes_batched(
    self,
    all_cls_scores,
    all_bbox_preds,
    enc_cls_scores,
    enc_bbox_preds,
    enc_outputs,
    img_metas,
    rescale=False,
    with_nms=False,
):
    """Apply the score threshold with one batch-wide GPU synchronization."""

    del enc_cls_scores, enc_bbox_preds, enc_outputs
    from mmcv.ops import batched_nms
    from mmdet.core import bbox_cxcywh_to_xyxy

    cls_scores = all_cls_scores[-1]
    bbox_preds = all_bbox_preds[-1]
    if not self.loss_cls.use_sigmoid:
        return self._mh0_original_get_bboxes(
            all_cls_scores,
            all_bbox_preds,
            None,
            None,
            None,
            img_metas,
            rescale=rescale,
            with_nms=with_nms,
        )

    max_per_img = self.test_cfg.get("max_per_img", self.num_query)
    if with_nms:
        max_per_img = self.num_query
    flat_scores = cls_scores.sigmoid().flatten(1)
    topk = min(max_per_img, flat_scores.shape[1])
    scores, indexes = flat_scores.topk(topk, dim=1)
    det_labels = indexes % self.num_classes
    bbox_indexes = indexes // self.num_classes
    bbox_candidates = torch.gather(
        bbox_preds,
        1,
        bbox_indexes.unsqueeze(-1).expand(-1, -1, 4),
    )

    # topk is descending, so detections above the threshold form a prefix.
    # Copy all B counts at once instead of forcing one synchronization per
    # image through boolean indexing.
    score_threshold = self.test_cfg.get("score_thr", 0)
    valid_counts = (
        (scores > score_threshold).sum(dim=1).detach().cpu().tolist()
    )

    max_valid_count = max(valid_counts, default=0)
    bbox_candidates = bbox_cxcywh_to_xyxy(
        bbox_candidates[:, :max_valid_count]
    )
    first_meta = img_metas[0]
    img_shape = first_meta["img_shape"]
    bbox_candidates[..., 0::2] *= img_shape[1]
    bbox_candidates[..., 1::2] *= img_shape[0]
    bbox_candidates[..., 0::2].clamp_(min=0, max=img_shape[1])
    bbox_candidates[..., 1::2].clamp_(min=0, max=img_shape[0])
    if rescale:
        scale_values = tuple(
            float(value) for value in first_meta["scale_factor"]
        )
        scale_cache = getattr(self, "_mh0_bbox_scale_cache", None)
        if scale_cache is None:
            scale_cache = {}
            self._mh0_bbox_scale_cache = scale_cache
        scale_key = (
            bbox_candidates.device,
            bbox_candidates.dtype,
            scale_values,
        )
        scale = scale_cache.get(scale_key)
        if scale is None:
            scale = bbox_candidates.new_tensor(scale_values)
            scale_cache[scale_key] = scale
        pad_left, pad_top, ori_width, ori_height = (
            self._letterbox_pad_from_meta(
                img_shape,
                first_meta["scale_factor"],
                first_meta,
            )
        )
        if pad_left or pad_top:
            bbox_candidates[..., 0::2] -= pad_left
            bbox_candidates[..., 1::2] -= pad_top
        bbox_candidates = bbox_candidates / scale
        if ori_width is not None and ori_height is not None:
            bbox_candidates[..., 0::2].clamp_(
                min=0, max=ori_width
            )
            bbox_candidates[..., 1::2].clamp_(
                min=0, max=ori_height
            )

    result_list = []
    for image_index, count in enumerate(valid_counts):
        image_scores = scores[image_index, :count]
        image_labels = det_labels[image_index, :count]
        image_boxes = bbox_candidates[image_index, :count]

        if count == 0:
            result_list.append(
                (image_boxes.new_zeros((0, 5)), image_labels)
            )
            continue
        if with_nms and count > 1:
            cfg = self.test_cfg
            image_boxes, keep = batched_nms(
                image_boxes.float(),
                image_scores.float(),
                image_labels,
                cfg.nms,
            )
            result_list.append(
                (
                    image_boxes[: cfg.max_per_img],
                    image_labels[keep][: cfg.max_per_img],
                )
            )
            continue

        result_list.append(
            (
                torch.cat(
                    (image_boxes.float(), image_scores.float().unsqueeze(1)),
                    dim=-1,
                ),
                image_labels,
            )
        )
    return result_list


def install_batched_bbox_test(model: torch.nn.Module) -> torch.nn.Module:
    query_head = model.query_head
    query_head._mh0_original_get_bboxes = query_head.get_bboxes
    query_head.get_bboxes = types.MethodType(get_bboxes_batched, query_head)
    model._mh0_original_simple_test_query_head = model.simple_test_query_head
    model.simple_test_query_head = types.MethodType(
        simple_test_query_head_batched, model
    )
    return model


__all__ = [
    "get_bboxes_batched",
    "install_batched_bbox_test",
    "simple_test_query_head_batched",
]
