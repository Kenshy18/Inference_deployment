"""CPU-only native exact raster batches for the Production curve runtime.

The extension is the same OpenCV 4.8 C++ evaluator used by Production's
polygon path.  This module never imports or initializes CUDA.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def native_module():
    try:
        return importlib.import_module("native_interval_metrics")
    except ImportError:
        build = (
            Path(__file__).resolve().parents[2]
            / "polygon"
            / "runtime"
            / "native_interval"
            / "build"
        )
        if not build.exists():
            return None
        value = str(build)
        if value not in sys.path:
            sys.path.insert(0, value)
        try:
            return importlib.import_module("native_interval_metrics")
        except ImportError:
            return None


class ExactRasterBatch:
    """Parity-aware exact metric cache for one single-component track."""

    def __init__(self, references: list[np.ndarray]) -> None:
        module = native_module()
        if module is None or not hasattr(module, "CachedIntervalEvaluator"):
            raise RuntimeError("native_interval_metrics is unavailable")
        count = len(references)
        # exact_frame_metrics_batch uses only exact_gt_frames.  The one-pixel
        # cached contexts satisfy the shared evaluator constructor without
        # allocating a second full-resolution reference stack.
        masks = [np.zeros((1, 1), dtype=np.uint8) for _ in range(count)]
        shifts = np.zeros((count, 2), dtype=np.float32)
        scales = np.ones((count,), dtype=np.float32)
        exact = [
            [np.ascontiguousarray(reference, dtype=np.float32)]
            for reference in references
        ]
        self._evaluator = module.CachedIntervalEvaluator(
            masks,
            shifts,
            scales,
            exact,
        )

    def metrics(
        self,
        frame_indices: np.ndarray,
        boundaries: np.ndarray,
        *,
        threads: int,
    ) -> np.ndarray:
        frames = np.ascontiguousarray(frame_indices, dtype=np.int32)
        values = np.ascontiguousarray(boundaries, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != 2:
            raise ValueError("boundaries must have shape (cases, points, 2)")
        if len(frames) != len(values):
            raise ValueError("frame indices and boundaries must have equal length")
        return np.asarray(
            self._evaluator.exact_frame_metrics_batch(
                frames,
                values,
                1,
                int(values.shape[1]),
                max(1, int(threads)),
            ),
            dtype=np.float64,
        )

    def recall_deficits(
        self,
        candidate_boundaries: np.ndarray,
        edges: np.ndarray,
        *,
        recall_floor: float,
        threads: int,
    ) -> np.ndarray:
        """Return exact first-failure Recall deficits for every graph edge.

        Catmull--Rom sampling is linear in P, so interpolating the sampled
        endpoint boundaries here is exactly equivalent (up to float32 raster
        arithmetic) to interpolating P first and sampling every intermediate
        curve.  The native evaluator stops an edge at its first Recall
        violation.  Its cached 1px metric is deliberately irrelevant; only
        the exact-deficit column is consumed.
        """
        values = np.ascontiguousarray(candidate_boundaries, dtype=np.float32)
        graph_edges = np.ascontiguousarray(edges, dtype=np.int32)
        if values.ndim != 4 or values.shape[3] != 2:
            raise ValueError(
                "candidate_boundaries must have shape (frames, states, points, 2)"
            )
        if graph_edges.ndim != 2 or graph_edges.shape[1] != 4:
            raise ValueError("edges must have shape (edge_count, 4)")
        output = np.asarray(
            self._evaluator.evaluate_edge_batch(
                values,
                graph_edges,
                1,
                int(values.shape[2]),
                1.0,
                float(recall_floor),
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                max(1, int(threads)),
                None,
                True,
                False,
                True,
                None,
            ),
            dtype=np.float64,
        )
        return np.ascontiguousarray(output[:, 8], dtype=np.float64)


class ExactDoubleRasterBatch:
    """Native batch matching the authoritative float64 + ``np.rint`` oracle."""

    def __init__(
        self,
        references: list[np.ndarray],
        *,
        maximum_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        module = native_module()
        if module is None or not hasattr(module, "ExactDoubleRasterEvaluator"):
            raise RuntimeError("native float64 raster evaluator is unavailable")
        exact = [
            [np.ascontiguousarray(reference, dtype=np.float64)]
            for reference in references
        ]
        self._evaluator = module.ExactDoubleRasterEvaluator(
            exact,
            max(0, int(maximum_cache_bytes)),
        )

    def cache_stats(self) -> dict[str, int]:
        if not hasattr(self._evaluator, "cache_stats"):
            return {}
        return {
            str(key): int(value)
            for key, value in dict(self._evaluator.cache_stats()).items()
        }

    def metrics(
        self,
        frame_indices: np.ndarray,
        boundaries: np.ndarray,
        *,
        threads: int,
    ) -> np.ndarray:
        frames = np.ascontiguousarray(frame_indices, dtype=np.int32)
        values = np.ascontiguousarray(boundaries, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != 2:
            raise ValueError("boundaries must have shape (cases, points, 2)")
        if len(frames) != len(values):
            raise ValueError("frame indices and boundaries must have equal length")
        return np.asarray(
            self._evaluator.metrics_batch(
                frames,
                values,
                1,
                int(values.shape[1]),
                max(1, int(threads)),
            ),
            dtype=np.float64,
        )

    def edge_metrics(
        self,
        candidate_boundaries: np.ndarray,
        edges: np.ndarray,
        *,
        recall_floor: float,
        low_iou_quadratic_weight: float,
        threads: int,
        check_topology: bool = True,
    ) -> np.ndarray:
        values = np.ascontiguousarray(candidate_boundaries, dtype=np.float64)
        graph_edges = np.ascontiguousarray(edges, dtype=np.int32)
        if values.ndim != 4 or values.shape[3] != 2:
            raise ValueError(
                "candidate_boundaries must have shape (frames, states, points, 2)"
            )
        if graph_edges.ndim != 2 or graph_edges.shape[1] != 4:
            raise ValueError("edges must have shape (edge_count, 4)")
        return np.asarray(
            self._evaluator.edge_metrics_batch(
                values,
                graph_edges,
                float(recall_floor),
                float(low_iou_quadratic_weight),
                max(1, int(threads)),
                bool(check_topology),
            ),
            dtype=np.float64,
        )


__all__ = (
    "ExactDoubleRasterBatch",
    "ExactRasterBatch",
    "native_module",
)
