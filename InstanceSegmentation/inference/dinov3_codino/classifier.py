"""Co-DINO mask generation and DINOv3-backbone ROI classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from backbone_roi_classifier import BackboneRoiClassifier

try:
    from .preprocessing import letterbox_params, prepare_batch_direct
except ImportError:
    from preprocessing import letterbox_params, prepare_batch_direct

MODEL_TYPES = {"spatial_gap"}
META_DIM = 5


def slice_features_for_image(features, image_index: int, batch_size: int):
    if (
        isinstance(features, (list, tuple))
        and features
        and isinstance(features[0], torch.Tensor)
        and features[0].shape[0] == batch_size
    ):
        return [
            feature[image_index : image_index + 1]
            if isinstance(feature, torch.Tensor) and feature.shape[0] == batch_size
            else feature
            for feature in features
        ]
    return features


def scale_factor_tensor(
    image_metadata: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale_factor = image_metadata["scale_factor"]
    if isinstance(scale_factor, torch.Tensor):
        return scale_factor.to(device=device, dtype=dtype)
    return torch.as_tensor(scale_factor, device=device, dtype=dtype)


def paste_boxes_for_masks(
    detection_boxes: torch.Tensor,
    image_metadata: dict[str, Any],
    scale_factor: torch.Tensor,
) -> torch.Tensor:
    scaled = detection_boxes[:, :4] * scale_factor
    paste_boxes = scaled
    image_shape = image_metadata.get(
        "img_shape",
        image_metadata.get("pad_shape", image_metadata["ori_shape"]),
    )
    input_height, input_width = int(image_shape[0]), int(image_shape[1])
    original_height = int(image_metadata["ori_shape"][0])
    original_width = int(image_metadata["ori_shape"][1])
    metadata_scale = image_metadata["scale_factor"]
    if isinstance(metadata_scale, torch.Tensor):
        # This fallback is only used by callers that supply GPU-only metadata.
        # The video path retains the original CPU scale values and therefore
        # does not need to synchronize the tail stream here.
        scale_values = metadata_scale.detach().cpu().tolist()
    else:
        scale_values = metadata_scale
    scale_x = float(scale_values[0])
    scale_y = float(scale_values[1])
    new_width = int(original_width * scale_x)
    new_height = int(original_height * scale_y)
    pad_left = max((input_width - new_width) // 2, 0)
    pad_top = max((input_height - new_height) // 2, 0)
    if pad_left or pad_top:
        paste_boxes = scaled.clone()
        paste_boxes[:, 0::2] -= pad_left
        paste_boxes[:, 1::2] -= pad_top
        paste_boxes[:, 0::2].clamp_(min=0, max=max(new_width, 1))
        paste_boxes[:, 1::2].clamp_(min=0, max=max(new_height, 1))
    return paste_boxes


def refine_stage_instance_predictions(
    stage_predictions: list[torch.Tensor],
) -> torch.Tensor:
    from mmdet.models.losses.cross_entropy_loss import generate_block_target

    predictions = list(stage_predictions[1:])
    for index in range(len(predictions) - 1):
        instance_prediction = predictions[index].squeeze(1).sigmoid() >= 0.5
        non_boundary = (
            generate_block_target(instance_prediction, boundary_width=1) != 1
        ).unsqueeze(1)
        non_boundary = (
            functional.interpolate(
                non_boundary.float(),
                predictions[index + 1].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            >= 0.5
        )
        previous = functional.interpolate(
            predictions[index],
            predictions[index + 1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        if previous.dtype != predictions[index + 1].dtype:
            previous = previous.to(predictions[index + 1].dtype)
        predictions[index + 1][non_boundary] = previous[non_boundary]
    return predictions[-1]


@dataclass(frozen=True, slots=True)
class ClassifierCheckpointContract:
    model_type: str
    input_dim: int
    num_classes: int
    use_meta: bool


def contract_from_checkpoint(
    checkpoint: Mapping[str, Any]
) -> ClassifierCheckpointContract:
    cfg = checkpoint.get("model_cfg") or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("Co-DINO classifier model_cfg must be a mapping")
    model_type = str(cfg.get("model_type", "")).strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Co-DINO classifier requires spatial_gap; got {model_type!r}")
    input_dim = int(cfg.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"invalid classifier input_dim: {input_dim}")
    num_classes = int(cfg.get("num_classes", 0))
    if num_classes <= 0:
        num_classes = len(checkpoint.get("class_names") or ())
    if num_classes <= 0:
        raise ValueError(f"invalid classifier num_classes: {num_classes}")
    return ClassifierCheckpointContract(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        use_meta=bool(cfg.get("use_meta", True)),
    )


class RoiSpatialGapClassifier(nn.Module):
    """ROIAlign feature -> pointwise/depthwise conv -> GAP -> class logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        stem_channels: int = 64,
        mid_channels: int = 64,
        dw_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiSpatialGapClassifier: expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )
        stem_channels = int(stem_channels)
        mid_channels = int(mid_channels)
        dw_kernel = int(dw_kernel)
        if stem_channels <= 0 or mid_channels <= 0:
            raise ValueError(
                f"stem_channels and mid_channels must be positive: stem={stem_channels}, mid={mid_channels}"
            )
        if dw_kernel <= 0:
            raise ValueError(f"dw_kernel must be positive, got {dw_kernel}")
        padding = dw_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.pooler_channels,
                stem_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=dw_kernel,
                stride=1,
                padding=padding,
                groups=stem_channels,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = (
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
        self.head = nn.Linear(
            int(mid_channels + (META_DIM if self.use_meta else 0)), int(num_classes)
        )

    def forward(
        self, roi_feat: torch.Tensor, meta: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if roi_feat.shape[1] != self.input_dim:
                raise ValueError(
                    f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}"
                )
            value = roi_feat.contiguous().view(
                -1, self.pooler_channels, self.pooler_size, self.pooler_size
            )
        elif roi_feat.ndim == 4:
            (channels, height, width) = (
                int(roi_feat.shape[1]),
                int(roi_feat.shape[2]),
                int(roi_feat.shape[3]),
            )
            if (
                channels != self.pooler_channels
                or height != self.pooler_size
                or width != self.pooler_size
            ):
                raise ValueError(
                    f"roi_feat shape mismatch: expected [N,{self.pooler_channels},{self.pooler_size},{self.pooler_size}], got {tuple(roi_feat.shape)}"
                )
            value = roi_feat
        else:
            raise ValueError(
                f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}"
            )
        value = self.stem(value)
        value = self.depthwise(value)
        value = self.proj(value)
        features = self.dropout(self.pool(value).flatten(start_dim=1))
        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            features = torch.cat([features, meta], dim=1)
        return self.head(features)


def classifier_from_checkpoint(
    checkpoint_path: Path, map_location: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier checkpoint root must be a mapping")
    contract = contract_from_checkpoint(checkpoint)
    cfg = checkpoint.get("model_cfg") or {}
    model = RoiSpatialGapClassifier(
        input_dim=contract.input_dim,
        num_classes=contract.num_classes,
        use_meta=contract.use_meta,
        pooler_channels=int(cfg.get("pooler_channels", 256)),
        pooler_size=int(cfg.get("pooler_size", 7)),
        stem_channels=int(cfg.get("gap_stem_channels", 64)),
        mid_channels=int(cfg.get("gap_mid_channels", 64)),
        dw_kernel=int(cfg.get("gap_dw_kernel", 3)),
        dropout=float(cfg.get("gap_dropout", cfg.get("dropout", 0.0))),
    )
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise KeyError(f"checkpoint missing model_state: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return (model, checkpoint)


def classifier_from_manifest(
    manifest_path: Path,
    *,
    mode: str = "fast",
) -> tuple[BackboneRoiClassifier, dict[str, object]]:
    """Load the delivered classifier while preserving canonical SQLite classes."""

    return BackboneRoiClassifier.from_manifest(manifest_path, mode=mode)


def restore_boxes_for_classifier_metadata(
    boxes: np.ndarray, image_metadata: dict[str, Any], target_size: tuple[int, int]
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    original_height = int(image_metadata["ori_shape"][0])
    original_width = int(image_metadata["ori_shape"][1])
    (target_height, target_width) = target_size
    (scale, _, _, pad_top, pad_left) = letterbox_params(
        original_height, original_width, target_height, target_width
    )
    restored = boxes.astype(np.float32, copy=True)
    restored[:, [0, 2]] -= pad_left / scale if scale > 0 else 0.0
    restored[:, [1, 3]] -= pad_top / scale if scale > 0 else 0.0
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, max(original_width - 1, 0))
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, max(original_height - 1, 0))
    return restored


def boxes_to_model_coordinates(
    boxes_original_xyxy: torch.Tensor,
    image_metadata: dict[str, Any],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Map rescaled detector boxes into the letterboxed stride-16 input space."""

    original_height = int(image_metadata["ori_shape"][0])
    original_width = int(image_metadata["ori_shape"][1])
    target_height, target_width = target_size
    scale, _, _, pad_top, pad_left = letterbox_params(
        original_height,
        original_width,
        target_height,
        target_width,
    )
    boxes = boxes_original_xyxy[:, :4].clone()
    boxes[:, 0::2].mul_(float(scale)).add_(float(pad_left))
    boxes[:, 1::2].mul_(float(scale)).add_(float(pad_top))
    boxes[:, 0::2].clamp_(0, float(target_width))
    boxes[:, 1::2].clamp_(0, float(target_height))
    return boxes


def classifier_metadata(
    detection_boxes: torch.Tensor,
    masks: list[Any],
    image_metadata: dict[str, Any],
    target_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    count = int(detection_boxes.shape[0])
    if count == 0:
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    detection_cpu = (
        detection_boxes[:, :5].detach().float().cpu().numpy()
    )
    scores = detection_cpu[:, 4]
    boxes = restore_boxes_for_classifier_metadata(
        detection_cpu[:, :4],
        image_metadata,
        target_size,
    )
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    box_area = widths * heights
    image_area = float(
        max(
            1, int(image_metadata["ori_shape"][0]) * int(image_metadata["ori_shape"][1])
        )
    )
    mask_area = np.zeros((count,), dtype=np.float32)
    for (index, mask) in enumerate(masks[:count]):
        mask_area[index] = float(np.asarray(mask).astype(bool).sum())
    metadata = np.stack(
        [
            scores.astype(np.float32),
            (box_area / image_area).astype(np.float32),
            (mask_area / image_area).astype(np.float32),
            np.log((widths + 1e-06) / (heights + 1e-06)).astype(np.float32),
            (mask_area / np.maximum(box_area, 1e-06)).astype(np.float32),
        ],
        axis=1,
    )
    return torch.from_numpy(metadata).to(device=device, dtype=torch.float32)


def classify_mask_features(
    classifier: torch.nn.Module,
    mask_features: torch.Tensor,
    meta_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = classifier(mask_features, meta_features)
    probabilities = torch.softmax(logits.float(), dim=1)
    (scores, classes) = torch.max(probabilities, dim=1)
    return (classes, scores, probabilities)


def classify_backbone_features(
    classifier: BackboneRoiClassifier,
    backbone_feature: torch.Tensor,
    boxes_model_xyxy: torch.Tensor,
    metadata: torch.Tensor,
    *,
    batch_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return classifier.classify_backbone(
        backbone_feature,
        boxes_model_xyxy,
        metadata,
        batch_indices=batch_indices,
    )


def infer_batch_with_classifier(
    model,
    frames,
    *,
    amp: str,
    target_size: tuple[int, int],
    classifier: torch.nn.Module,
    num_classifier_classes: int,
):
    """Run the cohesive mask-head → classifier path without tensor plugins."""
    from mmdet.core import bbox2result, bbox2roi

    if not hasattr(model, "mask_head"):
        raise RuntimeError("Co-DINO classifier requires the mask head")
    data = prepare_batch_direct(model, frames, target_size)
    image = data["img"][0]
    image_metadata = data["img_metas"][0]
    batch_input_shape = tuple(image[0].size()[-2:])
    for metadata in image_metadata:
        metadata["batch_input_shape"] = batch_input_shape
    if hasattr(model, "with_attn_mask") and (not model.with_attn_mask):
        for metadata in image_metadata:
            (height, width) = metadata["batch_input_shape"]
            metadata["img_shape"] = [height, width, 3]
    use_cuda = next(model.parameters()).is_cuda
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested, but this GPU does not support bf16")
        amp_dtype = torch.bfloat16
    elif amp == "fp16":
        amp_dtype = torch.float16
    elif amp == "off":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"unsupported amp mode: {amp!r}")
    with torch.inference_mode():
        with torch.cuda.amp.autocast(
            dtype=amp_dtype, enabled=amp != "off" and use_cuda
        ):
            backbone_features = model.backbone(image)
            features = (
                model.neck(backbone_features)
                if getattr(model, "with_neck", False)
                else backbone_features
            )
            backbone_feature = backbone_features[-1]
            (results, features) = model.query_head.simple_test(
                features, image_metadata, rescale=True, return_encoder_output=True
            )
            detector_classes = int(getattr(model.query_head, "num_classes", 1))
            outputs = []
            for (image_index, ((boxes, labels), metadata)) in enumerate(
                zip(results, image_metadata)
            ):
                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)
                count = int(boxes.shape[0])
                segmentations = [
                    [] for _ in range(model.mask_head.stage_num_classes[0])
                ]
                if count == 0:
                    outputs.append(
                        (bbox2result(boxes, labels, detector_classes), segmentations)
                    )
                    continue
                image_features = slice_features_for_image(
                    features, image_index, len(image_metadata)
                )
                scale_factor = scale_factor_tensor(metadata, boxes.device, boxes.dtype)
                scaled_boxes = boxes[:, :4] * scale_factor
                paste_boxes = paste_boxes_for_masks(boxes, metadata, scale_factor)
                mask_rois = bbox2roi([scaled_boxes])
                feature_dtype = (
                    image_features[0].dtype
                    if isinstance(image_features, (list, tuple)) and image_features
                    else mask_rois.dtype
                )
                if mask_rois.dtype != feature_dtype:
                    mask_rois = mask_rois.to(dtype=feature_dtype)
                extra = boxes.new_zeros((count, 2 + int(num_classifier_classes)))
                for start in range(0, count, 150):
                    end = min(start + 150, count)
                    mask_result = model._mask_forward(
                        image_features, mask_rois[start:end], labels[start:end]
                    )
                    instance_prediction = refine_stage_instance_predictions(
                        mask_result["stage_instance_preds"]
                    )
                    chunk_masks = model.mask_head.get_seg_masks(
                        instance_prediction,
                        paste_boxes[start:end],
                        labels[start:end],
                        model.rcnn_test_cfg,
                        metadata["ori_shape"],
                        scale_factor,
                        True,
                    )
                    for (class_id, segmentation) in zip(labels[start:end], chunk_masks):
                        segmentations[int(class_id)].append(segmentation)
                    (classes, scores, probabilities) = classify_backbone_features(
                        classifier,
                        backbone_feature[
                            image_index : image_index + 1
                        ],
                        boxes_to_model_coordinates(
                            boxes[start:end], metadata, target_size
                        ),
                        classifier_metadata(
                            boxes[start:end],
                            list(chunk_masks),
                            metadata,
                            target_size,
                            boxes.device,
                        ),
                    )
                    extra[start:end, 0] = classes.to(dtype=extra.dtype)
                    extra[start:end, 1] = scores.to(dtype=extra.dtype)
                    extra[
                        start:end, 2 : 2 + int(num_classifier_classes)
                    ] = probabilities.to(dtype=extra.dtype)
                outputs.append(
                    (
                        bbox2result(
                            torch.cat([boxes, extra], dim=1), labels, detector_classes
                        ),
                        segmentations,
                    )
                )
            return outputs
