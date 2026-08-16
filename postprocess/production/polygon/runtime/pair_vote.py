"""Exact C++ alpha batching for the frozen pair-vote optimizer.

The alpha grid, float32 interpolation, OpenCV fill rule, metric arithmetic,
and tie-breaking are unchanged. The acceleration removes repeated Python/C++
crossings and parses each local GT sequence once per alpha batch.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import importlib
import os
import time
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Any

import numpy as np

from .candidate_config import CANDIDATE, CandidateConfig


@dataclass(frozen=True)
class _FrameBinding:
    left: int
    right: int
    alpha: float
    exact_key: bool


class ExactPairVoteEvaluator:
    """Evaluate the existing pair-vote objective without changing its result."""

    def __init__(
        self,
        module: Any,
        run: Any,
        chosen_frames: list[int],
        baseline: np.ndarray,
        voted: np.ndarray,
        stats: dict[str, float | int] | None = None,
    ) -> None:
        started = time.perf_counter()
        self.module = module
        self.run = run
        self.chosen = [int(value) for value in chosen_frames]
        self.baseline = np.asarray(baseline, dtype=np.float32)
        self.voted = np.asarray(voted, dtype=np.float32)
        self.stats = stats if stats is not None else {}
        self.bindings = [
            self._binding(frame) for frame in range(len(run.frame_numbers))
        ]
        self.worker_count = max(
            1,
            int(os.environ.get("MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS", "4")),
        )
        try:
            native = importlib.import_module("native_interval_metrics")
        except ImportError:
            native = None
        self.native_batch = (
            getattr(native, "pair_vote_local_metrics_batch", None)
            if native is not None
            else None
        )
        self.native_full_batch = (
            getattr(native, "pair_vote_full_metrics_batch", None)
            if native is not None
            else None
        )
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="new-production-pair-vote",
            )
            if self.native_batch is None
            else None
        )
        self.stats["evaluator_builds"] = int(self.stats.get("evaluator_builds", 0)) + 1
        self.stats["parallel_workers"] = self.worker_count
        self.stats["native_pair_vote_batch"] = self.native_batch is not None
        if self.native_batch is not None and self.native_full_batch is not None:
            self.stats["mode"] = "native_cpp_exact_alpha_batches"
        self.stats["evaluator_build_seconds"] = float(
            self.stats.get("evaluator_build_seconds", 0.0)
        ) + (time.perf_counter() - started)

    def _binding(self, frame: int) -> _FrameBinding:
        if frame <= self.chosen[0]:
            return _FrameBinding(0, 0, 0.0, True)
        if frame >= self.chosen[-1]:
            last = len(self.chosen) - 1
            return _FrameBinding(last, last, 0.0, True)
        right = bisect.bisect_left(self.chosen, int(frame))
        if frame == self.chosen[right]:
            return _FrameBinding(right, right, 0.0, True)
        left = max(0, right - 1)
        alpha = float(
            (frame - self.chosen[left]) / max(self.chosen[right] - self.chosen[left], 1)
        )
        return _FrameBinding(left, right, alpha, False)

    def _vector_for_frame(
        self,
        vectors: np.ndarray,
        frame: int,
        *,
        replacement_pos: int | None = None,
        replacement_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        binding = self.bindings[frame]

        def key_vector(position: int) -> np.ndarray:
            if replacement_pos == position and replacement_vector is not None:
                return replacement_vector
            return vectors[position]

        if binding.left == binding.right:
            return np.asarray(key_vector(binding.left), dtype=np.float32)
        return self.module.interpolate_vectors(
            key_vector(binding.left), key_vector(binding.right), binding.alpha
        )

    @staticmethod
    def _aggregate(rows: list[dict[str, float]]):
        total_iou_loss = sum(1.0 - float(row["iou"]) for row in rows)
        total_recall = sum(float(row["recall"]) for row in rows)
        total_precision = sum(float(row["precision"]) for row in rows)
        total_gt_area = sum(float(row["gt_area"]) for row in rows)
        total_intersection = sum(float(row["intersection"]) for row in rows)
        count = max(len(rows), 1)
        return (
            rows,
            float(total_iou_loss),
            float(1.0 - total_iou_loss / count),
            float(total_recall / count),
            float(total_precision / count),
            float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0,
        )

    def full_metrics(self, vectors: np.ndarray):
        started = time.perf_counter()
        vectors = np.asarray(vectors, dtype=np.float32)
        # Scalar exact path used for the final defensive whole-track audit and
        # as the fallback when the native batch module is unavailable.
        rows = []
        for frame in range(len(self.bindings)):
            vector = self._vector_for_frame(vectors, frame)
            polygons = self.module.split_vector_to_polygons(
                vector, self.run.contour_count, self.run.anchors_per_contour
            )
            rows.append(
                self.module.compute_exact_metrics_from_polygons(
                    self.run.gt_polygons[frame], polygons
                )
            )
        self.stats["full_calls"] = int(self.stats.get("full_calls", 0)) + 1
        self.stats["evaluation_seconds"] = float(
            self.stats.get("evaluation_seconds", 0.0)
        ) + (time.perf_counter() - started)
        return self._aggregate(rows)

    def full_metrics_many(self, vectors: list[np.ndarray]) -> list[tuple[float, float]]:
        """Return (mean IoU, minimum Recall) for complete alpha trials."""
        if self.native_full_batch is None:
            output = []
            for value in vectors:
                (
                    rows,
                    _loss,
                    mean_iou,
                    _mean_recall,
                    _precision,
                    _global,
                ) = self.full_metrics(value)
                output.append(
                    (
                        float(mean_iou),
                        min((float(row["recall"]) for row in rows), default=1.0),
                    )
                )
            return output
        started = time.perf_counter()
        values = np.asarray(
            self.native_full_batch(
                self.run.gt_polygons,
                np.asarray(self.chosen, dtype=np.int32),
                np.asarray(vectors, dtype=np.float32),
                int(self.run.contour_count),
                int(self.run.anchors_per_contour),
                int(self.worker_count),
            ),
            dtype=np.float64,
        )
        self.stats["native_full_batches"] = (
            int(self.stats.get("native_full_batches", 0)) + 1
        )
        self.stats["native_full_batch_seconds"] = float(
            self.stats.get("native_full_batch_seconds", 0.0)
        ) + (time.perf_counter() - started)
        return [(float(row[0]), float(row[1])) for row in values]

    def local_metrics(
        self,
        current: np.ndarray,
        key_pos: int,
        trial_vector: np.ndarray,
    ) -> tuple[float, float]:
        started = time.perf_counter()
        left_key = max(0, key_pos - 1)
        right_key = min(len(self.chosen) - 1, key_pos + 1)
        start_frame = self.chosen[left_key]
        end_frame = self.chosen[right_key]
        iou_total = 0.0
        minimum_recall = 1.0
        for frame in range(start_frame, end_frame + 1):
            vector = self._vector_for_frame(
                current,
                frame,
                replacement_pos=key_pos,
                replacement_vector=trial_vector,
            )
            polygons = self.module.split_vector_to_polygons(
                vector, self.run.contour_count, self.run.anchors_per_contour
            )
            metrics = self.module.compute_exact_metrics_from_polygons(
                self.run.gt_polygons[frame], polygons
            )
            iou_total += float(metrics["iou"])
            minimum_recall = min(minimum_recall, float(metrics["recall"]))
        self.stats["local_calls"] = int(self.stats.get("local_calls", 0)) + 1
        self.stats["evaluation_seconds"] = float(
            self.stats.get("evaluation_seconds", 0.0)
        ) + (time.perf_counter() - started)
        return float(iou_total), float(minimum_recall)

    def local_metrics_many(
        self,
        current: np.ndarray,
        key_pos: int,
        trial_vectors: list[np.ndarray],
    ) -> list[tuple[float, float]]:
        """Evaluate independent alpha trials concurrently, without approximation."""
        if self.native_batch is not None and trial_vectors:
            started = time.perf_counter()
            values = np.asarray(
                self.native_batch(
                    self.run.gt_polygons,
                    np.asarray(self.chosen, dtype=np.int32),
                    np.asarray(current, dtype=np.float32),
                    int(key_pos),
                    np.asarray(trial_vectors, dtype=np.float32),
                    int(self.run.contour_count),
                    int(self.run.anchors_per_contour),
                    int(self.worker_count),
                ),
                dtype=np.float64,
            )
            self.stats["native_batches"] = int(self.stats.get("native_batches", 0)) + 1
            self.stats["native_batch_seconds"] = float(
                self.stats.get("native_batch_seconds", 0.0)
            ) + (time.perf_counter() - started)
            return [(float(row[0]), float(row[1])) for row in values]
        if len(trial_vectors) <= 1 or self.worker_count <= 1:
            return [
                self.local_metrics(current, key_pos, trial) for trial in trial_vectors
            ]
        # `current` is immutable for the duration of one alpha batch. OpenCV
        # and NumPy do the raster work outside the GIL, so this parallelizes
        # the exact reference evaluator without changing its arithmetic.
        assert self.executor is not None
        futures = [
            self.executor.submit(self.local_metrics, current, key_pos, trial)
            for trial in trial_vectors
        ]
        self.stats["parallel_batches"] = int(self.stats.get("parallel_batches", 0)) + 1
        return [future.result() for future in futures]

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)


def pair_vote_environment(
    config: CandidateConfig = CANDIDATE,
) -> dict[str, str]:
    """Return the frozen exact pair-vote runtime environment."""
    config.validate()
    return {
        "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
        "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": str(
            config.runtime.pair_vote_threads
        ),
    }


def apply_pair_vote_environment(
    environment: MutableMapping[str, str] | None = None,
    config: CandidateConfig = CANDIDATE,
) -> MutableMapping[str, str]:
    target = os.environ if environment is None else environment
    target.update(pair_vote_environment(config))
    return target


__all__ = (
    "ExactPairVoteEvaluator",
    "apply_pair_vote_environment",
    "pair_vote_environment",
)
