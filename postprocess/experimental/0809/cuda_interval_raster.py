"""Experimental CUDA scanline raster for Phase-2 cached interval metrics.

The kernel uses the same rounded, local integer vertices as the cached OpenCV
path, then evaluates an even/odd scanline fill.  OpenCV's boundary convention
can still differ by a few pixels.  In the precision-preserving hint mode its
ranked low-Recall frames only reorder CPU-exact checks; they never filter an
edge or provide a final objective value.
"""

from __future__ import annotations

import os
import sys
import time
import ctypes
from pathlib import Path

import numpy as np


CUDA_EXPERIMENT_SITE = Path(
    os.environ.get(
        "MASK_PIPELINE_CUDA_EXPERIMENT_SITE",
        "/home/kenshin/.local/share/mask-pipeline-cuda-experiment",
    )
)


KERNEL_SOURCE = r"""
#define MAX_RECALL_HINTS 8
extern "C" __global__
void interval_scanline(
    const float* __restrict__ vectors,
    const int* __restrict__ edges,
    const unsigned short* __restrict__ gt_prefix,
    const int* __restrict__ gt_area,
    const int* __restrict__ heights,
    const int* __restrict__ widths,
    const long long* __restrict__ prefix_offsets,
    const float* __restrict__ shifts,
    const float* __restrict__ scales,
    const int state_count,
    const int point_count,
    const float iou_weight,
    const float recall_floor,
    const int recall_hint_count,
    const int edge_count,
    double* __restrict__ frame_loss,
    double* __restrict__ recall_deficit,
    int* __restrict__ minimum_recall_frames,
    int* __restrict__ frames_covered) {
  const int edge_index = blockIdx.x;
  if (edge_index >= edge_count) return;
  const int tid = threadIdx.x;
  const int start_frame = edges[edge_index * 4 + 0];
  const int start_state = edges[edge_index * 4 + 1];
  const int end_frame = edges[edge_index * 4 + 2];
  const int end_state = edges[edge_index * 4 + 3];
  const int values_per_state = point_count * 2;
  const int values_per_frame = state_count * values_per_state;
  const float* start = vectors + start_frame * values_per_frame
      + start_state * values_per_state;
  const float* end = vectors + end_frame * values_per_frame
      + end_state * values_per_state;
  __shared__ int pred_rows[128];
  __shared__ int intersection_rows[128];
  __shared__ double total_loss;
  __shared__ double total_deficit;
  __shared__ double minimum_recalls[MAX_RECALL_HINTS];
  __shared__ int minimum_recall_indices[MAX_RECALL_HINTS];
  if (tid == 0) {
    total_loss = 0.0;
    total_deficit = 0.0;
    for (int hint = 0; hint < recall_hint_count; ++hint) {
      minimum_recalls[hint] = 2.0;
      minimum_recall_indices[hint] = end_frame;
    }
  }
  __syncthreads();
  const int denominator = max(end_frame - start_frame, 1);
  for (int frame = start_frame + 1; frame <= end_frame; ++frame) {
    const float alpha = (float)(frame - start_frame) / (float)denominator;
    const float beta = 1.0f - alpha;
    const int height = heights[frame];
    const int width = widths[frame];
    const float shift_x = shifts[frame * 2 + 0];
    const float shift_y = shifts[frame * 2 + 1];
    const float scale = scales[frame];
    int local_pred = 0;
    int local_intersection = 0;
    for (int y = tid; y < height; y += blockDim.x) {
      float crossings[64];
      int crossing_count = 0;
      for (int point = 0; point < point_count; ++point) {
        const int next = point + 1 == point_count ? 0 : point + 1;
        const float x0f =
            (beta * start[point * 2] + alpha * end[point * 2] - shift_x) * scale;
        const float y0f =
            (beta * start[point * 2 + 1] + alpha * end[point * 2 + 1] - shift_y) * scale;
        const float x1f =
            (beta * start[next * 2] + alpha * end[next * 2] - shift_x) * scale;
        const float y1f =
            (beta * start[next * 2 + 1] + alpha * end[next * 2 + 1] - shift_y) * scale;
        const int x0 = __float2int_rn(x0f);
        const int y0 = __float2int_rn(y0f);
        const int x1 = __float2int_rn(x1f);
        const int y1 = __float2int_rn(y1f);
        const bool crosses = (y0 <= y && y1 > y) || (y1 <= y && y0 > y);
        if (!crosses || crossing_count >= 64) continue;
        const float crossing = (float)x0
            + ((float)(y - y0) * (float)(x1 - x0)) / (float)(y1 - y0);
        int insert = crossing_count;
        while (insert > 0 && crossings[insert - 1] > crossing) {
          crossings[insert] = crossings[insert - 1];
          --insert;
        }
        crossings[insert] = crossing;
        ++crossing_count;
      }
      const long long prefix_base = prefix_offsets[frame]
          + (long long)y * (long long)(width + 1);
      for (int crossing = 0; crossing + 1 < crossing_count; crossing += 2) {
        int left = (int)ceilf(crossings[crossing]);
        int right = (int)floorf(crossings[crossing + 1]);
        left = max(left, 0);
        right = min(right, width - 1);
        if (right < left) continue;
        local_pred += right - left + 1;
        local_intersection += gt_prefix[prefix_base + right + 1]
            - gt_prefix[prefix_base + left];
      }
    }
    pred_rows[tid] = local_pred;
    intersection_rows[tid] = local_intersection;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        pred_rows[tid] += pred_rows[tid + stride];
        intersection_rows[tid] += intersection_rows[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      const int pred = pred_rows[0];
      const int intersection = intersection_rows[0];
      const int gt = gt_area[frame];
      const int union_area = gt + pred - intersection;
      const double recall = gt > 0 ? (double)intersection / (double)gt : 1.0;
      const double iou = union_area > 0
          ? (double)intersection / (double)union_area : 1.0;
      total_loss += (double)iou_weight * (1.0 - iou);
      total_deficit += fmax((double)recall_floor - recall, 0.0);
      for (int hint = 0; hint < recall_hint_count; ++hint) {
        if (recall < minimum_recalls[hint]) {
          for (int move = recall_hint_count - 1; move > hint; --move) {
            minimum_recalls[move] = minimum_recalls[move - 1];
            minimum_recall_indices[move] = minimum_recall_indices[move - 1];
          }
          minimum_recalls[hint] = recall;
          minimum_recall_indices[hint] = frame;
          break;
        }
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    frame_loss[edge_index] = total_loss;
    recall_deficit[edge_index] = total_deficit;
    for (int hint = 0; hint < recall_hint_count; ++hint) {
      minimum_recall_frames[edge_index * recall_hint_count + hint] =
          minimum_recall_indices[hint];
    }
    frames_covered[edge_index] = end_frame - start_frame;
  }
}
"""


def _import_cupy():
    site_packages = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia"
    for library in (
        site_packages / "nvjitlink/lib/libnvJitLink.so.12",
        site_packages / "cuda_nvrtc/lib/libnvrtc.so.12",
    ):
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    site = str(CUDA_EXPERIMENT_SITE)
    if site not in sys.path:
        sys.path.insert(0, site)
    import cupy as cp

    return cp


def evaluate_cached_intervals(
    candidate_vectors: np.ndarray,
    edges: np.ndarray,
    eval_contexts,
    *,
    iou_weight: float,
    recall_floor: float,
    chunk_edges: int = 40_000,
    return_frame_hints: bool = False,
    recall_hint_count: int = 8,
):
    cp = _import_cupy()
    vectors_np = np.ascontiguousarray(candidate_vectors, dtype=np.float32)
    edges_np = np.ascontiguousarray(edges, dtype=np.int32)
    point_count = int(vectors_np.shape[2])
    if point_count > 64:
        raise ValueError("CUDA scanline prototype supports at most 64 points")
    recall_hint_count = int(recall_hint_count)
    if not 1 <= recall_hint_count <= 8:
        raise ValueError("recall_hint_count must be in [1, 8]")
    frame_count = len(eval_contexts)
    max_height = max(int(context.shape_hw[0]) for context in eval_contexts)
    max_width = max(int(context.shape_hw[1]) for context in eval_contexts)
    if max_width > np.iinfo(np.uint16).max:
        raise ValueError("CUDA scanline prefix supports ROI widths up to 65535 pixels")
    prefix_offsets = np.empty((frame_count,), dtype=np.int64)
    prefix_parts: list[np.ndarray] = []
    gt_area = np.empty((frame_count,), dtype=np.int32)
    heights = np.empty((frame_count,), dtype=np.int32)
    widths = np.empty((frame_count,), dtype=np.int32)
    shifts = np.empty((frame_count, 2), dtype=np.float32)
    scales = np.empty((frame_count,), dtype=np.float32)
    prefix_offset = 0
    for index, context in enumerate(eval_contexts):
        mask = np.asarray(context.gt_mask, dtype=np.uint8)
        height, width = mask.shape
        prefix_offsets[index] = prefix_offset
        # A row prefix cannot exceed the ROI width. uint16 therefore remains
        # exact for normal video dimensions and halves both host/GPU storage
        # and PCIe traffic compared with int32.
        frame_prefix = np.zeros((height, width + 1), dtype=np.uint16)
        frame_prefix[:, 1:] = np.cumsum(mask, axis=1, dtype=np.uint16)
        prefix_parts.append(frame_prefix.reshape(-1))
        prefix_offset += frame_prefix.size
        gt_area[index] = int(context.gt_area)
        heights[index] = height
        widths[index] = width
        shifts[index] = np.asarray(context.shift_xy, dtype=np.float32)
        scales[index] = float(context.scale_factor)
    prefix = np.concatenate(prefix_parts) if prefix_parts else np.zeros((0,), np.uint16)
    started = time.perf_counter()
    kernel = cp.RawKernel(KERNEL_SOURCE, "interval_scanline", options=("--std=c++14",))
    vectors_gpu = cp.asarray(vectors_np)
    prefix_gpu = cp.asarray(prefix)
    gt_area_gpu = cp.asarray(gt_area)
    heights_gpu = cp.asarray(heights)
    widths_gpu = cp.asarray(widths)
    prefix_offsets_gpu = cp.asarray(prefix_offsets)
    shifts_gpu = cp.asarray(shifts)
    scales_gpu = cp.asarray(scales)
    frame_loss = np.empty((len(edges_np),), dtype=np.float64)
    recall_deficit = np.empty((len(edges_np),), dtype=np.float64)
    minimum_recall_frames = np.empty(
        (len(edges_np), recall_hint_count), dtype=np.int32
    )
    frames_covered = np.empty((len(edges_np),), dtype=np.int32)
    chunk_edges = max(1, int(chunk_edges))
    for offset in range(0, len(edges_np), chunk_edges):
        stop = min(offset + chunk_edges, len(edges_np))
        edge_gpu = cp.asarray(edges_np[offset:stop])
        loss_gpu = cp.empty((stop - offset,), dtype=cp.float64)
        deficit_gpu = cp.empty((stop - offset,), dtype=cp.float64)
        hint_gpu = cp.empty(
            (stop - offset, recall_hint_count), dtype=cp.int32
        )
        covered_gpu = cp.empty((stop - offset,), dtype=cp.int32)
        kernel(
            (stop - offset,),
            (128,),
            (
                vectors_gpu,
                edge_gpu,
                prefix_gpu,
                gt_area_gpu,
                heights_gpu,
                widths_gpu,
                prefix_offsets_gpu,
                shifts_gpu,
                scales_gpu,
                np.int32(vectors_np.shape[1]),
                np.int32(point_count),
                np.float32(iou_weight),
                np.float32(recall_floor),
                np.int32(recall_hint_count),
                np.int32(stop - offset),
                loss_gpu,
                deficit_gpu,
                hint_gpu,
                covered_gpu,
            ),
        )
        frame_loss[offset:stop] = cp.asnumpy(loss_gpu)
        recall_deficit[offset:stop] = cp.asnumpy(deficit_gpu)
        minimum_recall_frames[offset:stop] = cp.asnumpy(hint_gpu)
        frames_covered[offset:stop] = cp.asnumpy(covered_gpu)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    details = {
        "enabled": True,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "edges": int(len(edges_np)),
        "seconds": float(elapsed),
        "edges_per_second": float(len(edges_np) / max(elapsed, 1e-12)),
        "max_height": int(max_height),
        "max_width": int(max_width),
        "point_count": int(point_count),
        "recall_hint_count": int(recall_hint_count),
        "boundary_contract": "CUDA even-odd scanline approximation",
    }
    if return_frame_hints:
        return (
            frame_loss,
            recall_deficit,
            minimum_recall_frames,
            frames_covered,
            details,
        )
    return frame_loss, recall_deficit, frames_covered, details
