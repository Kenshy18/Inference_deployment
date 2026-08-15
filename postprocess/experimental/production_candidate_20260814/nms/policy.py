"""Construct the frozen candidate without duplicating geometry algorithms."""

from __future__ import annotations

from nms.component_virtual import VirtualComponentMaskNms

from ..config import CANDIDATE, CandidateConfig


def build_policy(config: CandidateConfig = CANDIDATE) -> VirtualComponentMaskNms:
    config.validate()
    value = config.nms
    return VirtualComponentMaskNms(
        fill_all_holes=value.fill_all_holes,
        unconditional_owner_ratio_max=value.unconditional_owner_island_ratio_max,
        island_other_coverage_min=value.island_other_coverage_min,
        island_to_other_area_max=value.island_to_other_area_max,
        mask_iou_threshold=value.mask_iou_threshold,
        mask_small_iou_threshold=value.mask_small_iou_threshold,
        mask_tiny_iou_threshold=value.mask_tiny_iou_threshold,
        mask_small_area=value.small_area,
        mask_tiny_area=value.tiny_area,
        mask_containment_coverage_min=value.containment_coverage_min,
        mask_contain_ratio_max=value.contain_ratio_max,
        mask_small_contain_ratio_max=value.small_contain_ratio_max,
        mask_tiny_contain_ratio_max=value.tiny_contain_ratio_max,
    )
