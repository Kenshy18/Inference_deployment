"""Fused exact CUDA interval evaluator for sampled closed boundaries.

One CUDA block owns a graph edge and performs float64 interpolation,
OpenCV-compatible rasterization, and integer metric reduction. References are
prepared once per bounded curve run. Large ROIs are processed in horizontal
shared-memory tiles instead of imposing a mask-size limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .cuda_opencv import _import_cupy


_MAX_POINTS = 512
_MAX_CROSSINGS = 64


_SOURCE = rf"""
#define XY_SHIFT 16
#define XY_ONE (1LL << XY_SHIFT)
#define MAX_POINTS {_MAX_POINTS}
#define MAX_CROSSINGS {_MAX_CROSSINGS}

__device__ int positive_modulo_two(const int value) {{
  const int remainder = value % 2;
  return remainder < 0 ? remainder + 2 : remainder;
}}

__device__ int origin_with_parity(const int value, const int parity) {{
  return positive_modulo_two(value) == parity ? value : value - 1;
}}

__device__ void draw_line8_tile(
    unsigned char* mask, const int width,
    const int tile_y, const int tile_height,
    int x0, int y0, int x1, int y1) {{
  int delta_x = 1, delta_y = 1;
  int dx = x1 - x0, dy = y1 - y0;
  if (dx < 0) {{
    dx = -dx; dy = -dy;
    int swap_v = x0; x0 = x1; x1 = swap_v;
    swap_v = y0; y0 = y1; y1 = swap_v;
  }}
  if (dy < 0) {{ dy = -dy; delta_y = -1; }}
  const bool vertical = dy > dx;
  if (vertical) {{
    int swap_v = dx; dx = dy; dy = swap_v;
    swap_v = delta_x; delta_x = delta_y; delta_y = swap_v;
  }}
  int error = dx - (dy + dy);
  const int plus_delta = dx + dx;
  const int minus_delta = -(dy + dy);
  int minus_shift = delta_x, plus_shift = 0;
  int minus_step = 0, plus_step = delta_y;
  if (vertical) {{
    int swap_v = plus_step; plus_step = plus_shift; plus_shift = swap_v;
    swap_v = minus_step; minus_step = minus_shift; minus_shift = swap_v;
  }}
  int x = x0, y = y0;
  for (int index = 0; index <= dx; ++index) {{
    const int local_y = y - tile_y;
    if ((unsigned)x < (unsigned)width &&
        (unsigned)local_y < (unsigned)tile_height)
      mask[local_y * width + x] = 1;
    const int branch = error < 0 ? -1 : 0;
    error += minus_delta + (plus_delta & branch);
    x += minus_shift + (plus_shift & branch);
    y += minus_step + (plus_step & branch);
  }}
}}

extern "C" __global__
void exact_intervals_tiled(
    const double* __restrict__ vectors,
    const int* __restrict__ edges,
    const unsigned char* __restrict__ references,
    const long long* __restrict__ reference_offsets,
    const int* __restrict__ origins,
    const int* __restrict__ shapes,
    const long long* __restrict__ reference_areas,
    const double* __restrict__ reference_minima,
    const int state_count,
    const int point_count,
    const double recall_floor,
    const double quadratic_weight,
    const int edge_count,
    const int shared_byte_limit,
    double* __restrict__ output,
    int* __restrict__ overflow_flags) {{
  const int edge_index = blockIdx.x;
  const int tid = threadIdx.x;
  if (edge_index >= edge_count) return;
  extern __shared__ unsigned char prediction[];
  __shared__ int rounded[MAX_POINTS * 2];
  __shared__ long long reduce_prediction[256];
  __shared__ long long reduce_intersection[256];
  __shared__ int variant_index;
  __shared__ int width;
  __shared__ int height;
  __shared__ int origin_x;
  __shared__ int origin_y;
  __shared__ int reference_origin_x;
  __shared__ int reference_origin_y;
  __shared__ int reference_width;
  __shared__ int reference_height;
  __shared__ int tile_rows;
  __shared__ int crossing_overflow;
  __shared__ long long frame_prediction;
  __shared__ long long frame_intersection;
  __shared__ double loss_total;
  __shared__ double iou_total;
  __shared__ double minimum_recall;
  __shared__ int frames_covered;
  __shared__ int keep_going;

  const int start_frame = edges[edge_index * 4];
  const int start_state = edges[edge_index * 4 + 1];
  const int end_frame = edges[edge_index * 4 + 2];
  const int end_state = edges[edge_index * 4 + 3];
  const long long values_per_state = (long long)point_count * 2LL;
  const long long values_per_frame = (long long)state_count * values_per_state;
  const double* start = vectors + (long long)start_frame * values_per_frame
      + (long long)start_state * values_per_state;
  const double* end = vectors + (long long)end_frame * values_per_frame
      + (long long)end_state * values_per_state;
  if (tid == 0) {{
    loss_total = 0.0;
    iou_total = 0.0;
    minimum_recall = 1.0;
    frames_covered = 0;
    keep_going = 1;
    overflow_flags[edge_index] = 0;
  }}
  __syncthreads();
  const int span = end_frame - start_frame;
  for (int frame = start_frame + 1; frame <= end_frame; ++frame) {{
    if (!keep_going) break;
    const bool endpoint = frame == end_frame;
    const double alpha = endpoint ? 1.0
        : (double)(frame - start_frame) / (double)span;
    const double beta = 1.0 - alpha;
    if (tid == 0) {{
      double predicted_min_x = 1.7976931348623157e+308;
      double predicted_min_y = 1.7976931348623157e+308;
      double predicted_max_x = -1.7976931348623157e+308;
      double predicted_max_y = -1.7976931348623157e+308;
      for (int point = 0; point < point_count; ++point) {{
        const int offset = point * 2;
        const double x = endpoint ? end[offset]
            : beta * start[offset] + alpha * end[offset];
        const double y = endpoint ? end[offset + 1]
            : beta * start[offset + 1] + alpha * end[offset + 1];
        predicted_min_x = fmin(predicted_min_x, x);
        predicted_min_y = fmin(predicted_min_y, y);
        predicted_max_x = fmax(predicted_max_x, x);
        predicted_max_y = fmax(predicted_max_y, y);
      }}
      const int joint_x =
          (int)floor(fmin(reference_minima[frame * 2], predicted_min_x));
      const int joint_y =
          (int)floor(fmin(reference_minima[frame * 2 + 1], predicted_min_y));
      const int parity_x = positive_modulo_two(joint_x);
      const int parity_y = positive_modulo_two(joint_y);
      variant_index = frame * 4 + parity_x + 2 * parity_y;
      origin_x = origin_with_parity((int)floor(predicted_min_x), parity_x);
      origin_y = origin_with_parity((int)floor(predicted_min_y), parity_y);
      width = (int)ceil(predicted_max_x) - origin_x + 1;
      height = (int)ceil(predicted_max_y) - origin_y + 1;
      reference_origin_x = origins[variant_index * 2];
      reference_origin_y = origins[variant_index * 2 + 1];
      reference_height = shapes[variant_index * 2];
      reference_width = shapes[variant_index * 2 + 1];
      tile_rows = width > 0 ? max(1, shared_byte_limit / width) : 0;
      crossing_overflow =
          width <= 0 || height <= 0 || width > shared_byte_limit;
      frame_prediction = 0;
      frame_intersection = 0;
    }}
    __syncthreads();
    for (int point = tid; point < point_count; point += blockDim.x) {{
      const int offset = point * 2;
      const double x = endpoint ? end[offset]
          : beta * start[offset] + alpha * end[offset];
      const double y = endpoint ? end[offset + 1]
          : beta * start[offset + 1] + alpha * end[offset + 1];
      rounded[offset] = __double2int_rn(x - (double)origin_x);
      rounded[offset + 1] = __double2int_rn(y - (double)origin_y);
    }}
    __syncthreads();
    if (crossing_overflow) {{
      if (tid == 0) {{ overflow_flags[edge_index] = 2; keep_going = 0; }}
      __syncthreads();
      break;
    }}
    const unsigned char* reference =
        references + reference_offsets[variant_index];
    for (int tile_y = 0; tile_y < height; tile_y += tile_rows) {{
      const int tile_height = min(tile_rows, height - tile_y);
      const int tile_pixels = width * tile_height;
      for (int pixel = tid; pixel < tile_pixels; pixel += blockDim.x)
        prediction[pixel] = 0;
      __syncthreads();
      for (int point = tid; point < point_count; point += blockDim.x) {{
        const int next = point + 1 == point_count ? 0 : point + 1;
        draw_line8_tile(
            prediction, width, tile_y, tile_height,
            rounded[point * 2], rounded[point * 2 + 1],
            rounded[next * 2], rounded[next * 2 + 1]);
      }}
      __syncthreads();
      for (int local_y = tid; local_y < tile_height; local_y += blockDim.x) {{
        const int y = tile_y + local_y;
        long long crossings[MAX_CROSSINGS];
        int crossing_count = 0;
        for (int point = 0; point < point_count; ++point) {{
          const int next = point + 1 == point_count ? 0 : point + 1;
          int x0 = rounded[point * 2], y0 = rounded[point * 2 + 1];
          int x1 = rounded[next * 2], y1 = rounded[next * 2 + 1];
          if (y0 == y1) continue;
          if (y0 > y1) {{
            int swap_v = x0; x0 = x1; x1 = swap_v;
            swap_v = y0; y0 = y1; y1 = swap_v;
          }}
          if (y < y0 || y >= y1) continue;
          if (crossing_count >= MAX_CROSSINGS) {{
            atomicExch(&crossing_overflow, 1);
            continue;
          }}
          const long long start_x =
              ((long long)x0 << XY_SHIFT) + (XY_ONE >> 1);
          const long long dx =
              ((long long)(x1 - x0) << XY_SHIFT) / (long long)(y1 - y0);
          const long long x = start_x + (long long)(y - y0) * dx;
          int insert = crossing_count;
          while (insert > 0 && crossings[insert - 1] > x) {{
            crossings[insert] = crossings[insert - 1];
            --insert;
          }}
          crossings[insert] = x;
          ++crossing_count;
        }}
        unsigned char* row = prediction + local_y * width;
        for (int edge = 0; edge + 1 < crossing_count; edge += 2) {{
          int left = (int)(crossings[edge] >> XY_SHIFT);
          int right = (int)(crossings[edge + 1] >> XY_SHIFT);
          left = max(left, 0); right = min(right, width - 1);
          for (int x = left; x <= right; ++x) row[x] = 1;
        }}
      }}
      __syncthreads();
      if (crossing_overflow) break;
      long long local_prediction = 0;
      for (int pixel = tid; pixel < tile_pixels; pixel += blockDim.x)
        local_prediction += prediction[pixel] != 0;
      long long local_intersection = 0;
      const int overlap_start = max(reference_origin_x, origin_x) - origin_x;
      const int overlap_end =
          min(reference_origin_x + reference_width, origin_x + width) - origin_x;
      for (int local_y = tid; local_y < tile_height; local_y += blockDim.x) {{
        const int reference_y =
            origin_y + tile_y + local_y - reference_origin_y;
        if ((unsigned)reference_y >= (unsigned)reference_height ||
            overlap_start >= overlap_end)
          continue;
        const unsigned char* predicted_row = prediction + local_y * width;
        const unsigned char* reference_row =
            reference + reference_y * reference_width;
        for (int x = overlap_start; x < overlap_end; ++x) {{
          const int reference_x = origin_x + x - reference_origin_x;
          local_intersection +=
              (predicted_row[x] != 0) & (reference_row[reference_x] != 0);
        }}
      }}
      reduce_prediction[tid] = local_prediction;
      reduce_intersection[tid] = local_intersection;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
          reduce_prediction[tid] += reduce_prediction[tid + stride];
          reduce_intersection[tid] += reduce_intersection[tid + stride];
        }}
        __syncthreads();
      }}
      if (tid == 0) {{
        frame_prediction += reduce_prediction[0];
        frame_intersection += reduce_intersection[0];
      }}
      __syncthreads();
    }}
    if (crossing_overflow) {{
      if (tid == 0) {{ overflow_flags[edge_index] = 1; keep_going = 0; }}
      __syncthreads();
      break;
    }}
    if (tid == 0) {{
      const long long gt = reference_areas[variant_index];
      const long long union_area = gt + frame_prediction - frame_intersection;
      const double recall = gt > 0
          ? (double)frame_intersection / (double)gt : 1.0;
      const double iou = union_area > 0
          ? (double)frame_intersection / (double)union_area : 1.0;
      ++frames_covered;
      minimum_recall = fmin(minimum_recall, recall);
      if (recall + 1e-12 < recall_floor) {{
        keep_going = 0;
      }} else {{
        const double loss = 1.0 - iou;
        loss_total += loss + quadratic_weight * loss * loss;
        iou_total += iou;
      }}
    }}
    __syncthreads();
  }}
  if (tid == 0) {{
    output[edge_index * 5] = loss_total;
    output[edge_index * 5 + 1] = iou_total;
    output[edge_index * 5 + 2] = minimum_recall;
    output[edge_index * 5 + 3] = (double)frames_covered;
    output[edge_index * 5 + 4] = 1.0;
  }}
}}
"""


@lru_cache(maxsize=1)
def _kernel():
    cp = _import_cupy()
    kernel = cp.RawKernel(
        _SOURCE,
        "exact_intervals_tiled",
        options=("--std=c++14", "--fmad=false"),
    )
    return cp, kernel


def _origin_with_parity(value: int, parity: int) -> int:
    return int(value) if int(value) % 2 == int(parity) else int(value) - 1


@dataclass(frozen=True, slots=True)
class _ReferenceContext:
    masks: np.ndarray
    offsets: np.ndarray
    origins: np.ndarray
    shapes: np.ndarray
    areas: np.ndarray
    minima: np.ndarray
    maximum_pixels: int


class CudaExactIntervalBatch:
    """Fused interval evaluator matching exact-double CPU raster metrics."""

    def __init__(self, references: list[np.ndarray]) -> None:
        self.references = tuple(
            np.ascontiguousarray(value, dtype=np.float64).reshape(-1, 2)
            for value in references
        )
        if any(len(value) < 3 for value in self.references):
            raise ValueError("references require at least three points")
        self._context = self._reference_context()

    def _reference_context(self) -> _ReferenceContext:
        frame_count = len(self.references)
        origins = np.empty((frame_count * 4, 2), dtype=np.int32)
        shapes = np.empty((frame_count * 4, 2), dtype=np.int32)
        areas = np.empty((frame_count * 4,), dtype=np.int64)
        minima = np.asarray(
            [np.min(reference, axis=0) for reference in self.references],
            dtype=np.float64,
        )
        masks: list[np.ndarray] = []
        offsets = np.empty((frame_count * 4,), dtype=np.int64)
        offset = 0
        maximum_pixels = 0
        for frame, reference in enumerate(self.references):
            floor_xy = np.floor(np.min(reference, axis=0)).astype(np.int32)
            ceil_xy = np.ceil(np.max(reference, axis=0)).astype(np.int32)
            for parity_y in range(2):
                for parity_x in range(2):
                    variant = frame * 4 + parity_x + 2 * parity_y
                    origin = np.asarray(
                        (
                            _origin_with_parity(int(floor_xy[0]), parity_x),
                            _origin_with_parity(int(floor_xy[1]), parity_y),
                        ),
                        dtype=np.int32,
                    )
                    width = int(ceil_xy[0] - origin[0] + 1)
                    height = int(ceil_xy[1] - origin[1] + 1)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(
                        mask,
                        [np.rint(reference - origin).astype(np.int32)],
                        1,
                    )
                    origins[variant] = origin
                    shapes[variant] = (height, width)
                    areas[variant] = int(cv2.countNonZero(mask))
                    offsets[variant] = offset
                    flat = np.ascontiguousarray(mask.reshape(-1))
                    masks.append(flat)
                    offset += len(flat)
                    maximum_pixels = max(maximum_pixels, len(flat))
        return _ReferenceContext(
            masks=np.concatenate(masks),
            offsets=offsets,
            origins=origins,
            shapes=shapes,
            areas=areas,
            minima=minima,
            maximum_pixels=maximum_pixels,
        )

    def edge_metrics(
        self,
        candidate_boundaries: np.ndarray,
        edges: np.ndarray,
        *,
        recall_floor: float,
        low_iou_quadratic_weight: float,
        threads: int = 1,
        check_topology: bool = False,
    ) -> np.ndarray:
        del threads
        if check_topology:
            raise ValueError("fused CUDA topology checking is not implemented")
        values = np.ascontiguousarray(candidate_boundaries, dtype=np.float64)
        graph = np.ascontiguousarray(edges, dtype=np.int32)
        if values.ndim != 4 or values.shape[3] != 2:
            raise ValueError("candidate boundaries must have shape (F, S, P, 2)")
        if graph.ndim != 2 or graph.shape[1] != 4:
            raise ValueError("edges must have shape (E, 4)")
        if values.shape[0] != len(self.references):
            raise ValueError("candidate frames do not match references")
        if not 3 <= values.shape[2] <= _MAX_POINTS:
            raise ValueError(f"point count must be in [3, {_MAX_POINTS}]")
        if not len(graph):
            return np.empty((0, 5), dtype=np.float64)
        context = self._context
        cp, kernel = _kernel()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        shared_limit = max(
            0,
            int(properties.get("sharedMemPerBlockOptin", 0))
            - int(kernel.shared_size_bytes)
            - int(properties.get("reservedSharedMemPerBlock", 0)),
        )
        if shared_limit < 1:
            raise RuntimeError("CUDA device has insufficient dynamic shared memory")
        kernel.max_dynamic_shared_size_bytes = int(shared_limit)
        output_gpu = cp.empty((len(graph), 5), dtype=cp.float64)
        overflow_gpu = cp.zeros((len(graph),), dtype=cp.int32)
        kernel(
            (len(graph),),
            (256,),
            (
                cp.asarray(values),
                cp.asarray(graph),
                cp.asarray(context.masks),
                cp.asarray(context.offsets),
                cp.asarray(context.origins),
                cp.asarray(context.shapes),
                cp.asarray(context.areas),
                cp.asarray(context.minima),
                np.int32(values.shape[1]),
                np.int32(values.shape[2]),
                np.float64(recall_floor),
                np.float64(low_iou_quadratic_weight),
                np.int32(len(graph)),
                np.int32(shared_limit),
                output_gpu,
                overflow_gpu,
            ),
            shared_mem=int(shared_limit),
        )
        output = cp.asnumpy(output_gpu)
        overflow = cp.asnumpy(overflow_gpu)
        if np.any(overflow):
            raise RuntimeError(
                "exact CUDA tiled raster overflow: "
                f"width={int(np.count_nonzero(overflow == 2))}, "
                f"crossings={int(np.count_nonzero(overflow == 1))}"
            )
        return output


__all__ = ("CudaExactIntervalBatch",)
