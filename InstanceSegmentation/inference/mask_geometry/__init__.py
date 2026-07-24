"""Model-independent binary-mask geometry helpers."""

from .polygonize import DEFAULT_MAX_MASK_POINTS, mask_to_polygons

__all__ = ["DEFAULT_MAX_MASK_POINTS", "mask_to_polygons"]
