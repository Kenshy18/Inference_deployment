"""Geometry-level exact CUDA Recall/IoU convenience API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .cuda_opencv import CudaOpenCvRasterizer
from .geometry import catmull_rom_boundaries, ellipse_boundaries


class CudaGeometryMetrics:
    """Evaluate Production polygon, ellipse, and Catmull--Rom geometry."""

    def __init__(self) -> None:
        self._rasterizer = CudaOpenCvRasterizer()

    def polygons(
        self,
        references: Sequence[np.ndarray],
        predictions: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Return exact metric rows for potentially ragged polygon pairs."""

        return self._rasterizer.metrics_ragged(references, predictions)

    def ellipses(
        self,
        references: np.ndarray,
        predictions: np.ndarray,
        *,
        boundary_points: int = 96,
    ) -> np.ndarray:
        """Polygonize canonical ellipses, then evaluate exact raster metrics."""

        left = ellipse_boundaries(references, points=boundary_points)
        right = ellipse_boundaries(predictions, points=boundary_points)
        return self._rasterizer.metrics(left, right)

    def catmull_rom(
        self,
        references: np.ndarray,
        predictions: np.ndarray,
        *,
        samples_per_segment: int = 16,
    ) -> np.ndarray:
        """Sample the fixed 1/6 closed Catmull--Rom contract and evaluate it."""

        left = catmull_rom_boundaries(
            references, samples_per_segment=samples_per_segment
        )
        right = catmull_rom_boundaries(
            predictions, samples_per_segment=samples_per_segment
        )
        return self._rasterizer.metrics(left, right)


__all__ = ("CudaGeometryMetrics",)
