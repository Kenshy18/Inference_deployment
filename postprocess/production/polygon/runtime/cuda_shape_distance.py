"""CUDA batch for Phase-2 endpoint similarity-shape distances.

This module deliberately handles only the dense, independent endpoint metric.
The hard-Recall raster contract remains in the exact OpenCV C++ engine.  The
CUDA calculation uses float32 throughout; callers can compare against the C++
float64 reference before deciding whether the very small numerical difference
is acceptable for an experiment.
"""

from __future__ import annotations

import time

import numpy as np


def compute_shape_distances(
    candidate_vectors: np.ndarray,
    edges: np.ndarray,
    normalization_scale: float,
    *,
    chunk_edges: int = 131_072,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA shape-distance acceleration requested without CUDA")
    vectors_np = np.ascontiguousarray(candidate_vectors, dtype=np.float32)
    edges_np = np.ascontiguousarray(edges, dtype=np.int64)
    if vectors_np.ndim != 4 or vectors_np.shape[-1] != 2:
        raise ValueError("candidate_vectors must have shape (frames, states, points, 2)")
    if edges_np.ndim != 2 or edges_np.shape[1] != 4:
        raise ValueError("edges must have shape (E, 4)")
    chunk_edges = max(1, int(chunk_edges))
    device = torch.device("cuda")
    started = time.perf_counter()
    vectors = torch.as_tensor(vectors_np, device=device)
    output = np.empty((len(edges_np),), dtype=np.float32)
    scale_floor = max(float(normalization_scale), 1.0)
    # Warm-up/context initialization is included in the reported wall time.
    for offset in range(0, len(edges_np), chunk_edges):
        stop = min(offset + chunk_edges, len(edges_np))
        edge = torch.as_tensor(edges_np[offset:stop], device=device)
        source = vectors[edge[:, 0], edge[:, 1]]
        destination = vectors[edge[:, 2], edge[:, 3]]
        source_mean = source.mean(dim=1, keepdim=True)
        destination_mean = destination.mean(dim=1, keepdim=True)
        src = source - source_mean
        dst = destination - destination_mean
        source_variance = (src.square().sum(dim=2)).mean(dim=1)
        covariance_trace = (dst * src).sum(dim=2).mean(dim=1)
        covariance_skew = (
            dst[:, :, 1] * src[:, :, 0]
            - dst[:, :, 0] * src[:, :, 1]
        ).mean(dim=1)
        proper = torch.hypot(covariance_trace, covariance_skew)
        valid = (source_variance > 1e-9) & (proper > 0.0)
        cosine = torch.where(valid, covariance_trace / proper.clamp_min(1e-30), 1.0)
        sine = torch.where(valid, covariance_skew / proper.clamp_min(1e-30), 0.0)
        similarity_scale = torch.where(
            valid,
            proper / source_variance.clamp_min(1e-9),
            1.0,
        )
        valid_scale = torch.isfinite(similarity_scale) & (similarity_scale > 1e-9)
        similarity_scale = torch.where(valid_scale, similarity_scale, 1.0)
        aligned_x = similarity_scale[:, None] * (
            cosine[:, None] * src[:, :, 0] - sine[:, None] * src[:, :, 1]
        )
        aligned_y = similarity_scale[:, None] * (
            sine[:, None] * src[:, :, 0] + cosine[:, None] * src[:, :, 1]
        )
        residual = torch.hypot(
            dst[:, :, 0] - aligned_x,
            dst[:, :, 1] - aligned_y,
        )
        distances = residual.mean(dim=1) / scale_floor
        output[offset:stop] = distances.detach().cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return output, {
        "device": str(torch.cuda.get_device_name(device)),
        "edges": int(len(edges_np)),
        "chunk_edges": int(chunk_edges),
        "seconds": float(elapsed),
        "edges_per_second": float(len(edges_np) / max(elapsed, 1e-12)),
        "dtype": "float32",
    }
