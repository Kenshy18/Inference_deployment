"""Production exact-mask non-maximum suppression.

Historical comparison policies live in explicitly imported modules and are
not part of this package's public runtime surface.
"""

from .component_virtual import (
    DEFAULT_PRODUCTION_NMS,
    ProductionVirtualComponentNms,
    VirtualComponentNmsDiagnostics,
)
from .config import PRODUCTION_NMS_CONFIG, ProductionNmsConfig
from .mask_adaptive import AdaptiveMaskNms
from .mask_geometry import MaskOverlapMetrics, exact_mask_iou, exact_mask_overlap
from .production import PRODUCTION_OPTIONS, ProductionMaskNmsStage

__all__ = (
    "AdaptiveMaskNms",
    "DEFAULT_PRODUCTION_NMS",
    "MaskOverlapMetrics",
    "PRODUCTION_OPTIONS",
    "PRODUCTION_NMS_CONFIG",
    "ProductionMaskNmsStage",
    "ProductionNmsConfig",
    "ProductionVirtualComponentNms",
    "VirtualComponentNmsDiagnostics",
    "exact_mask_iou",
    "exact_mask_overlap",
)
