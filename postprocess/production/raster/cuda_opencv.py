"""CUDA rasterization matching Production's OpenCV ``fillPoly`` subset.

This is intentionally narrower than a general OpenCV drawing replacement:

* 8-bit binary masks;
* closed contours with at least three points;
* ``LINE_8`` and ``shift=0``;
* contours are painted independently and unioned into their owning case.

The scan conversion mirrors the integer/fixed-point rules used by OpenCV
4.8.1 ``CollectPolyEdges`` and ``FillEdgeCollection``.  OpenCV also draws the
polygon boundary with its 8-connected line iterator before filling the
interior, so the CUDA implementation has a separate boundary pass.  Shape
generation is deliberately outside this module.

OpenCV source used to define the compatibility contract:
https://github.com/opencv/opencv/blob/4.8.1/modules/imgproc/src/drawing.cpp
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence

import numpy as np


_MAX_POINTS = 512


_KERNEL_SOURCE = rf"""
#define XY_SHIFT 16
#define XY_ONE (1LL << XY_SHIFT)
#define MAX_POINTS {_MAX_POINTS}

__device__ __forceinline__ long long div_trunc_zero(
    const long long numerator, const long long denominator) {{
  return numerator / denominator;
}}

__device__ void draw_line8(
    unsigned char* mask,
    const int stride,
    const int width,
    const int height,
    int x0,
    int y0,
    int x1,
    int y1) {{
  // Exact LineIterator state transition used by OpenCV's LINE_8 path.  The
  // compact metric ROI contains every vertex, so clipLine is not needed here.
  int delta_x = 1;
  int delta_y = 1;
  int dx = x1 - x0;
  int dy = y1 - y0;
  if (dx < 0) {{
    // CollectPolyEdges calls Line with leftToRight=true.
    dx = -dx;
    dy = -dy;
    int swap_x = x0; x0 = x1; x1 = swap_x;
    int swap_y = y0; y0 = y1; y1 = swap_y;
  }}
  if (dy < 0) {{
    dy = -dy;
    delta_y = -1;
  }}
  bool vertical = dy > dx;
  if (vertical) {{
    int swap_d = dx; dx = dy; dy = swap_d;
    swap_d = delta_x; delta_x = delta_y; delta_y = swap_d;
  }}
  int error = dx - (dy + dy);
  int plus_delta = dx + dx;
  int minus_delta = -(dy + dy);
  int minus_shift = delta_x;
  int plus_shift = 0;
  int minus_step = 0;
  int plus_step = delta_y;
  if (vertical) {{
    int swap_s = plus_step; plus_step = plus_shift; plus_shift = swap_s;
    swap_s = minus_step; minus_step = minus_shift; minus_shift = swap_s;
  }}
  int x = x0;
  int y = y0;
  for (int index = 0; index <= dx; ++index) {{
    if ((unsigned)x < (unsigned)width && (unsigned)y < (unsigned)height) {{
      mask[y * stride + x] = 1;
    }}
    const int branch = error < 0 ? -1 : 0;
    error += minus_delta + (plus_delta & branch);
    x += minus_shift + (plus_shift & branch);
    y += minus_step + (plus_step & branch);
  }}
}}

extern "C" __global__
void round_vertices(
    const double* __restrict__ points,
    const int* __restrict__ point_counts,
    const int* __restrict__ contour_cases,
    const int* __restrict__ origins,
    const int contour_count,
    const int point_stride,
    int* __restrict__ rounded) {{
  const int contour = blockIdx.x;
  const int point = threadIdx.x + blockIdx.y * blockDim.x;
  if (contour >= contour_count || point >= point_counts[contour]) return;
  const int owner = contour_cases[contour];
  const long long source = ((long long)contour * point_stride + point) * 2LL;
  const long long target = source;
  rounded[target] = __double2int_rn(points[source] - (double)origins[owner * 2]);
  rounded[target + 1] = __double2int_rn(
      points[source + 1] - (double)origins[owner * 2 + 1]);
}}

extern "C" __global__
void paint_boundaries(
    const int* __restrict__ rounded,
    const int* __restrict__ point_counts,
    const int* __restrict__ contour_cases,
    const int* __restrict__ shapes,
    const int contour_count,
    const int point_stride,
    const int canvas_height,
    const int canvas_width,
    unsigned char* __restrict__ masks) {{
  const int contour = blockIdx.x;
  const int point = threadIdx.x + blockIdx.y * blockDim.x;
  if (contour >= contour_count) return;
  const int count = point_counts[contour];
  if (point >= count) return;
  const int owner = contour_cases[contour];
  const int width = shapes[owner * 2 + 1];
  const int height = shapes[owner * 2];
  const int next = point + 1 == count ? 0 : point + 1;
  const long long base = (long long)contour * point_stride * 2LL;
  const int x0 = rounded[base + point * 2];
  const int y0 = rounded[base + point * 2 + 1];
  const int x1 = rounded[base + next * 2];
  const int y1 = rounded[base + next * 2 + 1];
  unsigned char* mask = masks + (long long)owner * canvas_height * canvas_width;
  draw_line8(mask, canvas_width, width, height, x0, y0, x1, y1);
}}

extern "C" __global__
void fill_rows(
    const int* __restrict__ rounded,
    const int* __restrict__ point_counts,
    const int* __restrict__ contour_cases,
    const int* __restrict__ shapes,
    const int contour_count,
    const int point_stride,
    const int canvas_height,
    const int canvas_width,
    unsigned char* __restrict__ masks) {{
  const int contour = blockIdx.x;
  const int y = threadIdx.x + blockIdx.y * blockDim.x;
  if (contour >= contour_count) return;
  const int owner = contour_cases[contour];
  const int height = shapes[owner * 2];
  const int width = shapes[owner * 2 + 1];
  if (y >= height) return;
  const int count = point_counts[contour];
  const long long base = (long long)contour * point_stride * 2LL;
  long long crossings[MAX_POINTS];
  int crossing_count = 0;
  for (int point = 0; point < count; ++point) {{
    const int next = point + 1 == count ? 0 : point + 1;
    int x0 = rounded[base + point * 2];
    int y0 = rounded[base + point * 2 + 1];
    int x1 = rounded[base + next * 2];
    int y1 = rounded[base + next * 2 + 1];
    if (y0 == y1) continue;
    if (y0 > y1) {{
      int swap_v = x0; x0 = x1; x1 = swap_v;
      swap_v = y0; y0 = y1; y1 = swap_v;
    }}
    if (y < y0 || y >= y1) continue;
    const long long start_x = ((long long)x0 << XY_SHIFT) + (XY_ONE >> 1);
    const long long dx = div_trunc_zero(
        ((long long)(x1 - x0) << XY_SHIFT), (long long)(y1 - y0));
    const long long x = start_x + (long long)(y - y0) * dx;
    int insert = crossing_count;
    while (insert > 0 && crossings[insert - 1] > x) {{
      crossings[insert] = crossings[insert - 1];
      --insert;
    }}
    crossings[insert] = x;
    ++crossing_count;
  }}
  unsigned char* mask = masks + (long long)owner * canvas_height * canvas_width;
  for (int edge = 0; edge + 1 < crossing_count; edge += 2) {{
    int left = (int)(crossings[edge] >> XY_SHIFT);
    int right = (int)(crossings[edge + 1] >> XY_SHIFT);
    if (left < 0) left = 0;
    if (right >= width) right = width - 1;
    if (right < left) continue;
    unsigned char* row = mask + y * canvas_width;
    for (int x = left; x <= right; ++x) row[x] = 1;
  }}
}}

extern "C" __global__
void metric_counts(
    const unsigned char* __restrict__ references,
    const unsigned char* __restrict__ predictions,
    const int* __restrict__ shapes,
    const int case_count,
    const int canvas_height,
    const int canvas_width,
    long long* __restrict__ output) {{
  const int owner = blockIdx.x;
  const int tid = threadIdx.x;
  if (owner >= case_count) return;
  const int height = shapes[owner * 2];
  const int width = shapes[owner * 2 + 1];
  const long long base = (long long)owner * canvas_height * canvas_width;
  long long reference_area = 0;
  long long prediction_area = 0;
  long long intersection = 0;
  for (int linear = tid; linear < height * width; linear += blockDim.x) {{
    const int y = linear / width;
    const int x = linear - y * width;
    const long long offset = base + (long long)y * canvas_width + x;
    const int left = references[offset] != 0;
    const int right = predictions[offset] != 0;
    reference_area += left;
    prediction_area += right;
    intersection += left & right;
  }}
  __shared__ long long reference_rows[256];
  __shared__ long long prediction_rows[256];
  __shared__ long long intersection_rows[256];
  reference_rows[tid] = reference_area;
  prediction_rows[tid] = prediction_area;
  intersection_rows[tid] = intersection;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
    if (tid < stride) {{
      reference_rows[tid] += reference_rows[tid + stride];
      prediction_rows[tid] += prediction_rows[tid + stride];
      intersection_rows[tid] += intersection_rows[tid + stride];
    }}
    __syncthreads();
  }}
  if (tid == 0) {{
    output[owner * 3] = reference_rows[0];
    output[owner * 3 + 1] = prediction_rows[0];
    output[owner * 3 + 2] = intersection_rows[0];
  }}
}}
"""


def _import_cupy():
    site_packages = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia"
    for library in (
        site_packages / "nvjitlink/lib/libnvJitLink.so.12",
        site_packages / "cuda_nvrtc/lib/libnvrtc.so.12",
    ):
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp

    return cp


def cuda_available() -> bool:
    """Return whether the CuPy runtime can access at least one CUDA device."""

    try:
        cp = _import_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _kernels():
    cp = _import_cupy()
    module = cp.RawModule(
        code=_KERNEL_SOURCE,
        options=("--std=c++14",),
        name_expressions=(
            "round_vertices",
            "paint_boundaries",
            "fill_rows",
            "metric_counts",
        ),
    )
    return cp, tuple(
        module.get_function(name)
        for name in (
            "round_vertices",
            "paint_boundaries",
            "fill_rows",
            "metric_counts",
        )
    )


@dataclass(frozen=True, slots=True)
class PackedContours:
    """Uniform CUDA input for one or more independently painted contours."""

    points: np.ndarray
    point_counts: np.ndarray
    contour_cases: np.ndarray
    origins: np.ndarray
    shapes: np.ndarray


class CudaOpenCvRasterizer:
    """OpenCV-compatible CUDA rasterizer for the Production binary subset."""

    maximum_points = _MAX_POINTS

    @staticmethod
    def pack_single_contours(
        boundaries: np.ndarray,
        *,
        padding: int = 2,
    ) -> PackedContours:
        values = np.asarray(boundaries, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        if values.ndim != 3 or values.shape[2] != 2:
            raise ValueError("boundaries must have shape (cases, points, 2)")
        if values.shape[1] < 3 or values.shape[1] > _MAX_POINTS:
            raise ValueError(f"point count must be in [3, {_MAX_POINTS}]")
        if not np.all(np.isfinite(values)):
            raise ValueError("boundary coordinates must be finite")
        pad = max(0, int(padding))
        minimum = np.floor(np.min(values, axis=1)).astype(np.int32) - pad
        maximum = np.ceil(np.max(values, axis=1)).astype(np.int32) + pad
        shapes = np.column_stack(
            (maximum[:, 1] - minimum[:, 1] + 1, maximum[:, 0] - minimum[:, 0] + 1)
        ).astype(np.int32)
        return PackedContours(
            points=np.ascontiguousarray(values, dtype=np.float64),
            point_counts=np.full((len(values),), values.shape[1], dtype=np.int32),
            contour_cases=np.arange(len(values), dtype=np.int32),
            origins=np.ascontiguousarray(minimum, dtype=np.int32),
            shapes=np.ascontiguousarray(shapes, dtype=np.int32),
        )

    def _rasterize_packed_gpu(self, packed: PackedContours):
        cp, (round_kernel, boundary_kernel, fill_kernel, _metric_kernel) = _kernels()
        points = np.ascontiguousarray(packed.points, dtype=np.float64)
        counts = np.ascontiguousarray(packed.point_counts, dtype=np.int32)
        owners = np.ascontiguousarray(packed.contour_cases, dtype=np.int32)
        origins = np.ascontiguousarray(packed.origins, dtype=np.int32)
        shapes = np.ascontiguousarray(packed.shapes, dtype=np.int32)
        if points.ndim != 3 or points.shape[2] != 2:
            raise ValueError("packed points must have shape (contours, points, 2)")
        if len(counts) != len(points) or len(owners) != len(points):
            raise ValueError("packed contour metadata length mismatch")
        if len(shapes) != len(origins) or shapes.shape[1] != 2:
            raise ValueError("packed case metadata mismatch")
        if np.any(counts < 3) or np.any(counts > min(points.shape[1], _MAX_POINTS)):
            raise ValueError("invalid packed point count")
        if np.any(owners < 0) or np.any(owners >= len(shapes)):
            raise ValueError("invalid contour owner")
        max_height = int(np.max(shapes[:, 0], initial=1))
        max_width = int(np.max(shapes[:, 1], initial=1))
        points_gpu = cp.asarray(points)
        counts_gpu = cp.asarray(counts)
        owners_gpu = cp.asarray(owners)
        origins_gpu = cp.asarray(origins)
        shapes_gpu = cp.asarray(shapes)
        rounded_gpu = cp.empty(points.shape, dtype=cp.int32)
        masks_gpu = cp.zeros((len(shapes), max_height, max_width), dtype=cp.uint8)
        point_blocks = (points.shape[1] + 127) // 128
        round_kernel(
            (len(points), point_blocks),
            (128,),
            (
                points_gpu,
                counts_gpu,
                owners_gpu,
                origins_gpu,
                np.int32(len(points)),
                np.int32(points.shape[1]),
                rounded_gpu,
            ),
        )
        boundary_kernel(
            (len(points), point_blocks),
            (128,),
            (
                rounded_gpu,
                counts_gpu,
                owners_gpu,
                shapes_gpu,
                np.int32(len(points)),
                np.int32(points.shape[1]),
                np.int32(max_height),
                np.int32(max_width),
                masks_gpu,
            ),
        )
        row_blocks = (max_height + 127) // 128
        fill_kernel(
            (len(points), row_blocks),
            (128,),
            (
                rounded_gpu,
                counts_gpu,
                owners_gpu,
                shapes_gpu,
                np.int32(len(points)),
                np.int32(points.shape[1]),
                np.int32(max_height),
                np.int32(max_width),
                masks_gpu,
            ),
        )
        return cp, masks_gpu, shapes

    def rasterize_packed(self, packed: PackedContours) -> list[np.ndarray]:
        cp, masks_gpu, shapes = self._rasterize_packed_gpu(packed)
        masks = cp.asnumpy(masks_gpu)
        return [
            np.ascontiguousarray(masks[index, :height, :width])
            for index, (height, width) in enumerate(shapes.tolist())
        ]

    def rasterize(
        self,
        boundaries: np.ndarray,
        *,
        padding: int = 2,
    ) -> list[np.ndarray]:
        return self.rasterize_packed(
            self.pack_single_contours(boundaries, padding=padding)
        )

    @staticmethod
    def _as_boundary_batch(boundaries: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(boundaries, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        if values.ndim != 3 or values.shape[2] != 2:
            raise ValueError(f"{name} must have shape (cases, points, 2)")
        if values.shape[1] < 3 or values.shape[1] > _MAX_POINTS:
            raise ValueError(f"{name} point count must be in [3, {_MAX_POINTS}]")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} coordinates must be finite")
        return np.ascontiguousarray(values, dtype=np.float64)

    def metrics(
        self,
        references: np.ndarray,
        predictions: np.ndarray,
        *,
        padding: int = 2,
    ) -> np.ndarray:
        """Return exact ``(area, area, intersection, union, R, P, IoU)`` rows.

        References and predictions may have different point counts, but they
        must contain the same number of cases.  Raster masks and integer
        reductions remain on the GPU; only the three integer counts per case
        are copied to the host.
        """

        left = self._as_boundary_batch(references, "references")
        right = self._as_boundary_batch(predictions, "predictions")
        return self.metrics_ragged(list(left), list(right), padding=padding)

    def metrics_ragged(
        self,
        references: Sequence[np.ndarray],
        predictions: Sequence[np.ndarray],
        *,
        padding: int = 2,
    ) -> np.ndarray:
        """Exact metrics for single contours with varying point counts."""

        left_list = [
            np.asarray(value, dtype=np.float64).reshape(-1, 2) for value in references
        ]
        right_list = [
            np.asarray(value, dtype=np.float64).reshape(-1, 2) for value in predictions
        ]
        if len(left_list) != len(right_list):
            raise ValueError("references and predictions must have equal cases")
        if not left_list:
            return np.empty((0, 7), dtype=np.float64)
        for name, values in (("references", left_list), ("predictions", right_list)):
            if any(len(value) < 3 or len(value) > _MAX_POINTS for value in values):
                raise ValueError(f"{name} point count must be in [3, {_MAX_POINTS}]")
            if any(not np.all(np.isfinite(value)) for value in values):
                raise ValueError(f"{name} coordinates must be finite")
        pad = max(0, int(padding))
        minimum = np.asarray(
            [
                np.floor(
                    np.minimum(np.min(left, axis=0), np.min(right, axis=0))
                ).astype(np.int32)
                - pad
                for left, right in zip(left_list, right_list, strict=True)
            ],
            dtype=np.int32,
        )
        maximum = np.asarray(
            [
                np.ceil(np.maximum(np.max(left, axis=0), np.max(right, axis=0))).astype(
                    np.int32
                )
                + pad
                for left, right in zip(left_list, right_list, strict=True)
            ],
            dtype=np.int32,
        )
        shapes = np.column_stack(
            (maximum[:, 1] - minimum[:, 1] + 1, maximum[:, 0] - minimum[:, 0] + 1)
        ).astype(np.int32)

        def packed(values: list[np.ndarray]) -> PackedContours:
            maximum_points = max(len(value) for value in values)
            points = np.zeros((len(values), maximum_points, 2), dtype=np.float64)
            counts = np.empty((len(values),), dtype=np.int32)
            for index, value in enumerate(values):
                points[index, : len(value)] = value
                counts[index] = len(value)
            return PackedContours(
                points=points,
                point_counts=counts,
                contour_cases=np.arange(len(values), dtype=np.int32),
                origins=np.ascontiguousarray(minimum, dtype=np.int32),
                shapes=np.ascontiguousarray(shapes, dtype=np.int32),
            )

        cp, reference_gpu, _ = self._rasterize_packed_gpu(packed(left_list))
        _, prediction_gpu, _ = self._rasterize_packed_gpu(packed(right_list))
        metric_kernel = _kernels()[1][3]
        counts_gpu = cp.empty((len(left_list), 3), dtype=cp.int64)
        metric_kernel(
            (len(left_list),),
            (256,),
            (
                reference_gpu,
                prediction_gpu,
                cp.asarray(shapes),
                np.int32(len(left_list)),
                np.int32(reference_gpu.shape[1]),
                np.int32(reference_gpu.shape[2]),
                counts_gpu,
            ),
        )
        counts = cp.asnumpy(counts_gpu)
        reference_area = counts[:, 0]
        prediction_area = counts[:, 1]
        intersection = counts[:, 2]
        union = reference_area + prediction_area - intersection
        output = np.empty((len(left_list), 7), dtype=np.float64)
        output[:, :4] = np.column_stack(
            (reference_area, prediction_area, intersection, union)
        )
        output[:, 4] = np.divide(
            intersection,
            reference_area,
            out=np.ones(len(left_list), dtype=np.float64),
            where=reference_area != 0,
        )
        output[:, 5] = np.divide(
            intersection,
            prediction_area,
            out=np.ones(len(left_list), dtype=np.float64),
            where=prediction_area != 0,
        )
        output[:, 6] = np.divide(
            intersection,
            union,
            out=np.ones(len(left_list), dtype=np.float64),
            where=union != 0,
        )
        return output


__all__ = (
    "CudaOpenCvRasterizer",
    "PackedContours",
    "cuda_available",
)
