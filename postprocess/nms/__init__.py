"""Adaptive non-maximum suppression."""

from .adaptive import AdaptiveNms, NmsPolicy, apply_nms, thresholds_for_area
from .components import (
    ComponentCleanupStats,
    fill_holes_and_remove_tiny_islands,
    remove_redundant_islands_candidate_v1,
    remove_redundant_surviving_islands,
    remove_small_foreground_components,
)
from .component_aware import (
    ComponentAwareMaskNms,
    ComponentAwareNmsDiagnostics,
    MaskOverlapMetrics,
    exact_mask_iou,
    exact_mask_overlap,
)
from .component_virtual import (
    DEFAULT_VIRTUAL_COMPONENT_MASK_NMS,
    DEFAULT_VIRTUAL_COMPONENT_NMS,
    VirtualComponentMaskNms,
    VirtualComponentNms,
    VirtualComponentNmsDiagnostics,
)
from .mask_adaptive import AdaptiveMaskNms

__all__ = [
    "AdaptiveNms",
    "AdaptiveMaskNms",
    "ComponentAwareMaskNms",
    "ComponentAwareNmsDiagnostics",
    "ComponentCleanupStats",
    "MaskOverlapMetrics",
    "NmsPolicy",
    "apply_nms",
    "exact_mask_iou",
    "exact_mask_overlap",
    "fill_holes_and_remove_tiny_islands",
    "remove_redundant_islands_candidate_v1",
    "remove_redundant_surviving_islands",
    "DEFAULT_VIRTUAL_COMPONENT_MASK_NMS",
    "DEFAULT_VIRTUAL_COMPONENT_NMS",
    "VirtualComponentMaskNms",
    "VirtualComponentNms",
    "VirtualComponentNmsDiagnostics",
    "remove_small_foreground_components",
    "thresholds_for_area",
]
