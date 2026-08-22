"""Exact CUDA frame and interval metrics over OpenCV-compatible masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from production.curve.runtime.topology import strict_self_intersection_batch

from .cuda_opencv import CudaOpenCvRasterizer
from .cuda_interval_exact import CudaExactIntervalBatch


@dataclass(frozen=True, slots=True)
class CudaExactProfile:
    metric_batches: int
    frame_cases: int
    endpoint_cases: int


class CudaExactRasterBatch:
    """Drop-in exact-raster batch for single-component geometry tracks.

    References may have varying contour lengths.  Candidate boundaries use a
    fixed point count, as required by polygon vertex correspondence and the
    sampled Catmull--Rom contract.
    """

    def __init__(
        self,
        references: list[np.ndarray],
        *,
        maximum_batch_cases: int = 4096,
    ) -> None:
        self.references = tuple(
            np.ascontiguousarray(value, dtype=np.float64).reshape(-1, 2)
            for value in references
        )
        if any(len(value) < 3 for value in self.references):
            raise ValueError("CUDA exact references require at least three points")
        self.maximum_batch_cases = max(1, int(maximum_batch_cases))
        self.rasterizer = CudaOpenCvRasterizer()
        self.profile = CudaExactProfile(0, 0, 0)

    def metrics(
        self,
        frame_indices: np.ndarray,
        boundaries: np.ndarray,
        *,
        threads: int = 1,
    ) -> np.ndarray:
        del threads
        frames = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        values = np.asarray(boundaries, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != 2 or len(values) != len(frames):
            raise ValueError("boundaries must have shape (cases, points, 2)")
        if np.any(frames < 0) or np.any(frames >= len(self.references)):
            raise ValueError("frame index is outside the reference sequence")
        parts = []
        batches = 0
        for offset in range(0, len(values), self.maximum_batch_cases):
            stop = min(len(values), offset + self.maximum_batch_cases)
            parts.append(
                self.rasterizer.metrics_ragged(
                    [self.references[int(frame)] for frame in frames[offset:stop]],
                    list(values[offset:stop]),
                )
            )
            batches += 1
        self.profile = CudaExactProfile(
            self.profile.metric_batches + batches,
            self.profile.frame_cases + len(values),
            self.profile.endpoint_cases,
        )
        return np.concatenate(parts) if parts else np.empty((0, 7), np.float64)

    def edge_metrics(
        self,
        candidate_boundaries: np.ndarray,
        edges: np.ndarray,
        *,
        recall_floor: float,
        low_iou_quadratic_weight: float,
        threads: int = 1,
        check_topology: bool = True,
    ) -> np.ndarray:
        values = np.ascontiguousarray(candidate_boundaries, dtype=np.float64)
        graph = np.ascontiguousarray(edges, dtype=np.int32)
        if values.ndim != 4 or values.shape[3] != 2:
            raise ValueError(
                "candidate_boundaries must have shape (frames, states, points, 2)"
            )
        if graph.ndim != 2 or graph.shape[1] != 4:
            raise ValueError("edges must have shape (edge_count, 4)")
        frame_count, state_count, point_count, _ = values.shape
        if frame_count != len(self.references) or point_count < 3:
            raise ValueError("candidate dimensions do not match references")
        if len(graph):
            valid = (
                (graph[:, 0] >= 0)
                & (graph[:, 0] < graph[:, 2])
                & (graph[:, 2] < frame_count)
                & (graph[:, 1] >= 0)
                & (graph[:, 1] < state_count)
                & (graph[:, 3] >= 0)
                & (graph[:, 3] < state_count)
            )
            if not bool(np.all(valid)):
                raise ValueError("edge index is outside candidate dimensions")

        # Every edge ending at the same (frame, state) shares one exact final
        # raster.  Match the CPU evaluator by calculating those endpoints once.
        endpoints = [
            (frame, state)
            for frame in range(1, frame_count)
            for state in range(state_count)
        ]
        endpoint_metrics = np.empty((frame_count, state_count, 7), np.float64)
        endpoint_topology = np.ones((frame_count, state_count), dtype=bool)
        if endpoints:
            endpoint_frames = np.asarray([row[0] for row in endpoints], np.int32)
            endpoint_values = np.asarray(
                [values[frame, state] for frame, state in endpoints], np.float64
            )
            endpoint_rows = self.metrics(
                endpoint_frames,
                endpoint_values,
                threads=threads,
            )
            for row, (frame, state) in zip(endpoint_rows, endpoints, strict=True):
                endpoint_metrics[frame, state] = row
            if check_topology:
                flags = strict_self_intersection_batch(
                    endpoint_values, threads=max(1, int(threads))
                )
                for flag, (frame, state) in zip(flags, endpoints, strict=True):
                    endpoint_topology[frame, state] = not bool(flag)

        # Rasterize all non-endpoint samples in bounded batches.  Aggregation
        # below still short-circuits at the same first invalid frame as C++.
        sample_edges: list[int] = []
        sample_frames: list[int] = []
        sample_values: list[np.ndarray] = []
        for edge_index, (start_frame, start_state, end_frame, end_state) in enumerate(
            graph.tolist()
        ):
            start = values[start_frame, start_state]
            end = values[end_frame, end_state]
            span = end_frame - start_frame
            for frame in range(start_frame + 1, end_frame):
                alpha = float(frame - start_frame) / float(span)
                beta = 1.0 - alpha
                sample_edges.append(edge_index)
                sample_frames.append(frame)
                sample_values.append(beta * start + alpha * end)
        interior_metrics = np.empty((len(sample_values), 7), np.float64)
        interior_topology = np.zeros((len(sample_values),), dtype=bool)
        if sample_values:
            sample_array = np.ascontiguousarray(sample_values, dtype=np.float64)
            interior_metrics[:] = self.metrics(
                np.asarray(sample_frames, dtype=np.int32),
                sample_array,
                threads=threads,
            )
            if check_topology:
                interior_topology[:] = strict_self_intersection_batch(
                    sample_array, threads=max(1, int(threads))
                )

        by_edge: list[list[int]] = [[] for _ in range(len(graph))]
        for sample_index, edge_index in enumerate(sample_edges):
            by_edge[edge_index].append(sample_index)
        output = np.zeros((len(graph), 5), dtype=np.float64)
        floor = float(recall_floor)
        quadratic = float(low_iou_quadratic_weight)
        for edge_index, (_sf, _ss, end_frame, end_state) in enumerate(graph.tolist()):
            loss_total = 0.0
            iou_total = 0.0
            minimum_recall = 1.0
            frames_covered = 0
            topology_valid = True
            continue_to_endpoint = True
            for sample_index in by_edge[edge_index]:
                if check_topology and interior_topology[sample_index]:
                    topology_valid = False
                    continue_to_endpoint = False
                    break
                metrics = interior_metrics[sample_index]
                frames_covered += 1
                recall = float(metrics[4])
                minimum_recall = min(minimum_recall, recall)
                if recall + 1e-12 < floor:
                    continue_to_endpoint = False
                    break
                loss = 1.0 - float(metrics[6])
                loss_total += loss + quadratic * loss * loss
                iou_total += float(metrics[6])
            if continue_to_endpoint:
                if check_topology and not endpoint_topology[end_frame, end_state]:
                    topology_valid = False
                else:
                    metrics = endpoint_metrics[end_frame, end_state]
                    frames_covered += 1
                    recall = float(metrics[4])
                    minimum_recall = min(minimum_recall, recall)
                    if recall + 1e-12 >= floor:
                        loss = 1.0 - float(metrics[6])
                        loss_total += loss + quadratic * loss * loss
                        iou_total += float(metrics[6])
            output[edge_index] = (
                loss_total,
                iou_total,
                minimum_recall,
                frames_covered,
                1.0 if topology_valid else 0.0,
            )
        self.profile = CudaExactProfile(
            self.profile.metric_batches,
            self.profile.frame_cases,
            self.profile.endpoint_cases + len(endpoints),
        )
        return output


class CudaProductionRasterBatch:
    """Production-compatible CUDA facade for frame and fused edge metrics."""

    def __init__(
        self,
        references: list[np.ndarray],
        *,
        maximum_batch_cases: int = 4096,
        frame_evaluator=None,
        edge_fallback=None,
    ) -> None:
        self._frames = CudaExactRasterBatch(
            references,
            maximum_batch_cases=maximum_batch_cases,
        )
        self._intervals = CudaExactIntervalBatch(references)
        self._frame_evaluator = frame_evaluator
        self._edge_fallback = edge_fallback

    def cache_stats(self) -> dict[str, int]:
        output = {
            "cuda_exact": 1,
            "reference_frames": len(self._frames.references),
            "cpu_frame_metrics": int(self._frame_evaluator is not None),
            "cpu_edge_fallback": int(self._edge_fallback is not None),
        }
        if self._edge_fallback is not None:
            output.update(
                {
                    f"cpu_{key}": value
                    for key, value in self._edge_fallback.cache_stats().items()
                }
            )
        return output

    def metrics(self, *args, **kwargs) -> np.ndarray:
        if self._frame_evaluator is not None:
            return self._frame_evaluator.metrics(*args, **kwargs)
        return self._frames.metrics(*args, **kwargs)

    def edge_metrics(self, *args, **kwargs) -> np.ndarray:
        try:
            return self._intervals.edge_metrics(*args, **kwargs)
        except (RuntimeError, ValueError):
            if self._edge_fallback is None:
                raise
            return self._edge_fallback.edge_metrics(*args, **kwargs)


__all__ = (
    "CudaExactProfile",
    "CudaExactRasterBatch",
    "CudaProductionRasterBatch",
)
