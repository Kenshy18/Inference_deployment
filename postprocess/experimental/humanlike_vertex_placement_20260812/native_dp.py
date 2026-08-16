"""Python bridge for the optimized temporal polygon dynamic program."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from experimental.temporal_vertex_decimation_20260812.optimizer import orient_ccw
from experimental.temporal_vertex_decimation_20260812.optimizer import (
    align_current_equal_arc,
)

def _native_module():
    build = Path(__file__).resolve().parent / "native_temporal" / "build"
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    return importlib.import_module("native_temporal_polygon")


def native_temporal_dp_sequence(
    polygons: Iterable[np.ndarray],
    target: int,
    *,
    frame_indices: Iterable[int] | None = None,
    cut_frames: Iterable[int] = (),
    temporal_weight: float = 0.1,
    distance_weight: float = 1.0,
    missing_area_weight: float = 4.0,
    excess_area_weight: float = 1.0,
    seed_mode: str = "rdp",
    contour_band_fraction: float = 0.0,
) -> np.ndarray:
    contours = [orient_ccw(value) for value in polygons]
    if not contours:
        return np.empty((0, int(target), 2), dtype=np.float64)
    frames = (
        np.arange(len(contours), dtype=np.int64)
        if frame_indices is None
        else np.asarray(list(frame_indices), dtype=np.int64)
    )
    if len(frames) != len(contours):
        raise ValueError("frame_indices and polygons must have equal length")
    cuts = {int(value) for value in cut_frames}
    starts = [0]
    for index in range(1, len(contours)):
        if int(frames[index]) != int(frames[index - 1]) + 1 or int(frames[index]) in cuts:
            starts.append(index)
    starts.append(len(contours))
    output = np.empty((len(contours), int(target), 2), dtype=np.float64)
    native = _native_module()
    for first, last in zip(starts[:-1], starts[1:], strict=True):
        segment = contours[first:last]
        if seed_mode == "rdp":
            result = native.simplify_sequence_auto(
                segment,
                int(target),
                float(temporal_weight),
                float(distance_weight),
                float(missing_area_weight),
                float(excess_area_weight),
                float(contour_band_fraction),
            )
        elif seed_mode == "production_equal_arc":
            # This is the same forward equal-arc resampling and XY-MSE phase
            # convention used by the old Production implementation.  The
            # native optimizer may slide those persistent identities along the
            # contour, but it no longer invents the identity gauge from an
            # independently changing RDP polygon in every frame.
            seed = align_current_equal_arc(segment, int(target))
            result = native.simplify_sequence(
                segment,
                seed,
                float(temporal_weight),
                float(distance_weight),
                float(missing_area_weight),
                float(excess_area_weight),
                float(contour_band_fraction),
            )
        else:
            raise ValueError(f"unknown seed_mode: {seed_mode}")
        output[first:last] = np.asarray(result, dtype=np.float64)
    return output


def native_temporal_dp_from_seed(
    polygons: Iterable[np.ndarray],
    seed: np.ndarray,
    *,
    frame_indices: Iterable[int] | None = None,
    cut_frames: Iterable[int] = (),
    temporal_weight: float = 0.003,
    distance_weight: float = 2.0,
    missing_area_weight: float = 1.0,
    excess_area_weight: float = 1.0,
    contour_band_fraction: float = 0.5,
) -> np.ndarray:
    """Refine a phase-aligned persistent seed without changing its identities."""
    contours = [orient_ccw(value) for value in polygons]
    seed = np.asarray(seed, dtype=np.float64)
    if seed.ndim != 3 or seed.shape[0] != len(contours) or seed.shape[2] != 2:
        raise ValueError("seed must have shape (frames, vertices, 2)")
    if not contours:
        return seed.copy()
    frames = (
        np.arange(len(contours), dtype=np.int64)
        if frame_indices is None
        else np.asarray(list(frame_indices), dtype=np.int64)
    )
    if len(frames) != len(contours):
        raise ValueError("frame_indices and polygons must have equal length")
    cuts = {int(value) for value in cut_frames}
    starts = [0]
    for index in range(1, len(contours)):
        if int(frames[index]) != int(frames[index - 1]) + 1 or int(frames[index]) in cuts:
            starts.append(index)
    starts.append(len(contours))
    output = np.empty_like(seed)
    native = _native_module()
    for first, last in zip(starts[:-1], starts[1:], strict=True):
        output[first:last] = np.asarray(
            native.simplify_sequence(
                contours[first:last],
                seed[first:last],
                float(temporal_weight),
                float(distance_weight),
                float(missing_area_weight),
                float(excess_area_weight),
                float(contour_band_fraction),
            ),
            dtype=np.float64,
        )
    return output
