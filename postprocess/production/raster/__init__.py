"""Shared exact-raster contracts for Production geometry engines."""

from .cuda_opencv import CudaOpenCvRasterizer, cuda_available
from .cuda_exact import CudaExactRasterBatch, CudaProductionRasterBatch
from .cuda_interval_exact import CudaExactIntervalBatch
from .geometry import catmull_rom_boundaries, ellipse_boundaries
from .metrics import CudaGeometryMetrics

__all__ = (
    "CudaOpenCvRasterizer",
    "CudaGeometryMetrics",
    "CudaExactRasterBatch",
    "CudaProductionRasterBatch",
    "CudaExactIntervalBatch",
    "catmull_rom_boundaries",
    "cuda_available",
    "ellipse_boundaries",
)
