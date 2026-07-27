"""Inference-only Face DINO path for already-filtered detector output."""

from __future__ import annotations

import types

import torch
from torch import Tensor


def _features_and_detections_prefiltered(
    self,
    images: Tensor,
    image_sizes: list[tuple[int, int]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Skip the one-class boolean gather after batched bbox decoding."""

    if self.detector_backend is not None:
        return self.detector_backend(images, image_sizes)
    metas = self.detector._metas(images, image_sizes)
    pyramid = list(self.detector.extract_feat(images, metas))
    indices = self.detector.query_level_indices
    if indices is None:
        query_features = pyramid
    else:
        query_features = [
            adapter(pyramid[index])
            for adapter, index in zip(
                self.detector.query_adapters,
                indices,
            )
        ]
    results, _ = self.detector.query_head.simple_test(
        query_features,
        metas,
        rescale=False,
        return_encoder_output=True,
    )
    # The installed detector has exactly one class. Its batched bbox decoder
    # has already score-filtered and score-sorted every result, so labels are
    # necessarily zero and the boolean gather is redundant.
    return (
        pyramid,
        [
            {
                "boxes": boxes_with_scores[:, :4],
                "scores": boxes_with_scores[:, 4],
            }
            for boxes_with_scores, _labels in results
        ],
    )


@torch.no_grad()
def _predict_prefiltered(
    self,
    images: Tensor,
    image_sizes: list[tuple[int, int]],
    score_threshold: float = 0.30,
    max_detections: int = 100,
    face_threshold: float | None = None,
    point_threshold: float | None = None,
    return_ellipse_masks: bool = False,
) -> list[dict[str, Tensor]]:
    """Run attributes without repeating detector filtering and sorting."""

    if float(score_threshold) != self._face_prefiltered_score_threshold:
        return self._face_original_predict(
            images,
            image_sizes,
            score_threshold=score_threshold,
            max_detections=max_detections,
            face_threshold=face_threshold,
            point_threshold=point_threshold,
            return_ellipse_masks=return_ellipse_masks,
        )

    from torchvision.ops import roi_align

    from face_dino_v1.geometry.ellipse import ellipse_roi_to_image
    from face_dino_v1.geometry.roi_transform import (
        expand_square_boxes,
        roi_points_to_image,
    )

    pyramid, raw_detections = self._features_and_detections(
        images,
        image_sizes,
    )
    detections: list[dict[str, Tensor]] = []
    boxes_per_image: list[Tensor] = []
    for result in raw_detections:
        order = result["scores"].argsort(descending=True)[:max_detections]
        boxes = result["boxes"][order]
        scores = result["scores"][order]
        detections.append({"boxes": boxes, "scores": scores})
        boxes_per_image.append(boxes)
    counts = [len(boxes) for boxes in boxes_per_image]
    total = sum(counts)
    if not total:
        return [
            {
                **detection,
                **self._empty(images, return_ellipse_masks),
            }
            for detection in detections
        ]
    if len(set(image_sizes)) == 1:
        height, width = image_sizes[0]
        joined_boxes = torch.cat(boxes_per_image)
        joined_square_boxes = expand_square_boxes(
            joined_boxes,
            height,
            width,
            scale=self.roi_expand,
        )
        square_boxes = list(joined_square_boxes.split(counts))
    else:
        square_boxes = [
            expand_square_boxes(
                boxes,
                height,
                width,
                scale=self.roi_expand,
            )
            for boxes, (height, width) in zip(boxes_per_image, image_sizes)
        ]
        joined_square_boxes = torch.cat(square_boxes)

    rgb = roi_align(
        images,
        square_boxes,
        output_size=(256, 256),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    p3 = roi_align(
        pyramid[0],
        square_boxes,
        output_size=(32, 32),
        spatial_scale=1.0 / 8,
        sampling_ratio=2,
        aligned=True,
    )
    p4 = roi_align(
        pyramid[1],
        square_boxes,
        output_size=(16, 16),
        spatial_scale=1.0 / 16,
        sampling_ratio=2,
        aligned=True,
    )
    p5 = roi_align(
        pyramid[2],
        square_boxes,
        output_size=(8, 8),
        spatial_scale=1.0 / 32,
        sampling_ratio=2,
        aligned=True,
    )
    output = self._run_attributes(p3, p4, p5, rgb)
    image_ellipses = ellipse_roi_to_image(
        output["ellipse"].float(),
        joined_square_boxes,
    )
    image_points = roi_points_to_image(
        output["point_coordinates"].float(),
        joined_square_boxes,
    )
    face_scores = output["face_logit"].sigmoid()
    class_probability = output["point_class_logits"].softmax(-1)
    non_background = class_probability[..., 1:]
    point_confidence, point_classes = non_background.max(-1)
    point_classes = point_classes + 1
    state_probability = output["point_state_logits"].softmax(-1)
    states = state_probability.argmax(-1) + 1
    selected_face_threshold = (
        self.face_threshold if face_threshold is None else face_threshold
    )
    selected_point_threshold = (
        self.point_threshold if point_threshold is None else point_threshold
    )
    face_present = face_scores >= selected_face_threshold
    point_valid = (
        (point_confidence >= selected_point_threshold)
        & (class_probability[..., 0] < point_confidence)
        & face_present[:, None]
    )
    moment_masks = (
        output["ellipse_mask_logits"]
        .float()
        .sigmoid()
        .pow(self.attribute_model.ellipse_moment_power)[:, 0]
        if return_ellipse_masks
        else None
    )

    answer = []
    offset = 0
    for detection, boxes in zip(detections, square_boxes):
        count = len(boxes)
        section = slice(offset, offset + count)
        result = {
            **detection,
            "face_scores": face_scores[section],
            "face_present": face_present[section],
            "ellipses": image_ellipses[section],
            "point_classes": point_classes[section],
            "keypoints": image_points[section],
            "keypoint_states": states[section],
            "point_confidence": point_confidence[section],
            "point_valid": point_valid[section],
            "point_class_probabilities": class_probability[section],
            "point_state_probabilities": state_probability[section],
        }
        if moment_masks is not None:
            result.update(
                {
                    "ellipse_moment_masks": moment_masks[section],
                    "ellipse_mask_boxes": boxes,
                }
            )
        answer.append(result)
        offset += count
    return answer


def install_prefiltered_predict(
    model: torch.nn.Module,
    *,
    score_threshold: float,
) -> torch.nn.Module:
    """Install the fast path only for the reviewed one-class bbox decoder."""

    head = model.detector.query_head
    if int(head.num_classes) != 1:
        raise ValueError("prefiltered Face DINO path requires one detector class")
    if not hasattr(head, "_face_original_get_bboxes"):
        raise RuntimeError("batched bbox decoding must be installed first")
    if not hasattr(model, "_face_original_predict"):
        model._face_original_predict = model.predict
        model._face_original_features_and_detections = (
            model._features_and_detections
        )
        model._face_prefiltered_score_threshold = float(score_threshold)
        model._features_and_detections = types.MethodType(
            _features_and_detections_prefiltered,
            model,
        )
        model.predict = types.MethodType(_predict_prefiltered, model)
    return model


__all__ = ["install_prefiltered_predict"]
