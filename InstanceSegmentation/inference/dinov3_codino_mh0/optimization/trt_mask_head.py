"""Fixed-N TensorRT adapter for the MH0 mask refinement core."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from .trt_transformer import _FixedEngine


class TensorRTMaskHead(torch.nn.Module):
    def __init__(
        self,
        original_mask_head: torch.nn.Module,
        engine_path: Path,
    ) -> None:
        super().__init__()
        self.original_mask_head = original_mask_head
        self.stage_num_classes = original_mask_head.stage_num_classes
        self.pre_upsample_last_stage = original_mask_head.pre_upsample_last_stage
        if self.stage_num_classes != [1, 1, 1]:
            raise RuntimeError(
                f"MH0 mask class layout drift: {self.stage_num_classes}"
            )
        self.fixed = _FixedEngine(
            engine_path,
            expected_inputs=("instance_feats", "sem0", "sem1"),
            expected_outputs=("stage0", "stage1", "stage2", "stage3"),
        )
        self.engine_num_rois = self.fixed.input_shapes["instance_feats"][0]
        if self.engine_num_rois < 1:
            raise RuntimeError(
                f"MH0 mask engine has invalid ROI batch: {self.engine_num_rois}"
            )

    def _semantic_rois_full(
        self,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(
            device_type=semantic_feat.device.type,
            dtype=torch.float16,
            enabled=semantic_feat.is_cuda,
        ):
            for conv in self.original_mask_head.semantic_convs:
                semantic_feat = conv(semantic_feat)
            transformed = [
                stage.relu(stage.semantic_transform_in(semantic_feat))
                for stage in self.original_mask_head.stages
            ]
        rois_fp32 = rois.float()
        return tuple(
            stage.semantic_roi_extractor([value.float()], rois_fp32)
            for stage, value in zip(
                self.original_mask_head.stages, transformed
            )
        )

    def _semantic_rois(
        self,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform only padded RoI neighborhoods on the fixed fast path."""

        roi_values = rois.detach().cpu().tolist()
        if not roi_values:
            return self._semantic_rois_full(semantic_feat, rois)

        height, width = semantic_feat.shape[-2:]
        stride = int(self.original_mask_head.semantic_out_stride)
        receptive_radius = 0
        for conv_module in self.original_mask_head.semantic_convs:
            conv = conv_module.conv
            kernel = int(conv.kernel_size[0])
            dilation = int(conv.dilation[0])
            receptive_radius += ((kernel - 1) * dilation) // 2
        # One additional pixel is needed by bilinear RoIAlign. Keep another
        # guard pixel so cropped convolution boundaries cannot affect samples.
        margin = receptive_radius + 2

        specs = []
        for row in roi_values:
            batch_index = int(row[0])
            x1, y1, x2, y2 = (float(value) for value in row[1:5])
            left = max(math.floor(x1 / stride) - margin, 0)
            top = max(math.floor(y1 / stride) - margin, 0)
            right = min(math.ceil(x2 / stride) + margin + 1, width)
            bottom = min(math.ceil(y2 / stride) + margin + 1, height)
            specs.append(
                (
                    batch_index,
                    x1,
                    y1,
                    x2,
                    y2,
                    left,
                    top,
                    right,
                    bottom,
                )
            )

        crop_height = max(row[8] - row[6] for row in specs)
        crop_width = max(row[7] - row[5] for row in specs)
        cropped_work = len(specs) * crop_height * crop_width
        full_work = int(semantic_feat.shape[0]) * height * width
        if cropped_work * 4 >= full_work * 3:
            return self._semantic_rois_full(semantic_feat, rois)

        crops = []
        adjusted_rois = []
        for roi_index, row in enumerate(specs):
            (
                batch_index,
                x1,
                y1,
                x2,
                y2,
                left,
                top,
                right,
                bottom,
            ) = row
            crop = semantic_feat[
                batch_index : batch_index + 1,
                :,
                top:bottom,
                left:right,
            ]
            pad_right = crop_width - (right - left)
            pad_bottom = crop_height - (bottom - top)
            if pad_right or pad_bottom:
                crop = functional.pad(
                    crop, (0, pad_right, 0, pad_bottom)
                )
            crops.append(crop)
            adjusted_rois.append(
                (
                    roi_index,
                    x1 - left * stride,
                    y1 - top * stride,
                    x2 - left * stride,
                    y2 - top * stride,
                )
            )

        semantic_feat = torch.cat(crops, dim=0)
        with torch.autocast(
            device_type=semantic_feat.device.type,
            dtype=torch.float16,
            enabled=semantic_feat.is_cuda,
        ):
            for conv in self.original_mask_head.semantic_convs:
                semantic_feat = conv(semantic_feat)
            transformed = [
                stage.relu(stage.semantic_transform_in(semantic_feat))
                for stage in self.original_mask_head.stages
            ]
        adjusted = rois.new_tensor(adjusted_rois, dtype=torch.float32)
        return tuple(
            stage.semantic_roi_extractor([value.float()], adjusted)
            for stage, value in zip(
                self.original_mask_head.stages, transformed
            )
        )

    @staticmethod
    def _pad_rois(value: torch.Tensor, target: int) -> torch.Tensor:
        missing = target - int(value.shape[0])
        if missing <= 0:
            return value
        return torch.cat((value, value[-1:].expand(missing, *value.shape[1:])), 0)

    def _execute_chunk(
        self,
        instance_feats: torch.Tensor,
        semantic_rois: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        valid = int(instance_feats.shape[0])
        instance_feats = self._pad_rois(instance_feats, self.engine_num_rois)
        semantic_rois = tuple(
            self._pad_rois(value, self.engine_num_rois)
            for value in semantic_rois
        )
        outputs = self.fixed.execute(
            {
                "instance_feats": instance_feats,
                "sem0": semantic_rois[0],
                "sem1": semantic_rois[1],
            }
        )
        return tuple(outputs[f"stage{index}"][:valid] for index in range(4))

    def forward(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
        roi_labels: torch.Tensor,
    ):
        del roi_labels
        num_rois = int(rois.shape[0])
        if num_rois == 0:
            empty_labels = torch.empty(
                0, dtype=torch.long, device=rois.device
            )
            return self.original_mask_head(
                instance_feats, semantic_feat, rois, empty_labels
            )
        semantic_rois = self._semantic_rois(semantic_feat, rois)
        if num_rois <= self.engine_num_rois:
            outputs = self._execute_chunk(instance_feats, semantic_rois)
            return [outputs[0], outputs[-1]], []
        chunks = [[] for _ in range(4)]
        for index in range(0, num_rois, self.engine_num_rois):
            end = min(index + self.engine_num_rois, num_rois)
            outputs = self._execute_chunk(
                instance_feats[index:end],
                tuple(value[index:end] for value in semantic_rois),
            )
            for stage, value in enumerate(outputs):
                chunks[stage].append(value.clone())
        merged = [torch.cat(values, dim=0) for values in chunks]
        return [merged[0], merged[-1]], []

    def get_seg_masks(self, *args, **kwargs):
        return self.original_mask_head.get_seg_masks(*args, **kwargs)


__all__ = ["TensorRTMaskHead"]
