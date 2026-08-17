"""Shared immutable-ish records passed between optimizer responsibilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackRow:
    frame: int
    track_id: str
    polygons: list[np.ndarray]
    is_gapfill: bool = False


@dataclass
class SimilarityTransform:
    scale: float
    angle_rad: float
    translation: np.ndarray


@dataclass
class InstanceRun:
    stream_id: str
    track_id: str
    run_id: int
    frame_numbers: np.ndarray
    gt_polygons: list[list[np.ndarray]]
    anchors: np.ndarray
    contour_count: int
    anchors_per_contour: int
    scale: float
    gapfilled_flags: np.ndarray | None = None
    predicted_total_points: np.ndarray | None = None
    run_target_total_points: int = 0
    emit_start_idx: int = 0
    emit_end_idx: int = -1
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_process_start: int = 0
    chunk_process_end: int = -1
    chunked_from_long_run: bool = False


@dataclass
class ShapeCandidate:
    label: str
    vector: np.ndarray
    polygons: list[np.ndarray]
    frame_loss: float
    objective: float
    recall_budget: float = 0.0
    area: float = 0.0
    center: np.ndarray | None = None
    radii: np.ndarray | None = None
    mean_radius: float = 0.0


@dataclass
class IntervalCost:
    cost: float
    shape_distance: float
    shape_update: float
    frames_covered: int
    frame_loss_mean: float = 0.0
    shape_distance_scale: float = 1.0
    shape_switch_scale: float = 1.0
    recall_budget: float = 0.0


@dataclass
class FrameEvalContext:
    gt_mask: np.ndarray
    gt_area: int
    shift_xy: np.ndarray
    shape_hw: tuple[int, int]
    scale_factor: float
    gt_center: np.ndarray
    gt_radii: np.ndarray
    gt_mean_radius: float
    gt_polygon_area: float
    scratch_pred_mask: np.ndarray | None = None
    scratch_intersection_mask: np.ndarray | None = None


__all__ = (
    "FrameEvalContext",
    "InstanceRun",
    "IntervalCost",
    "ShapeCandidate",
    "SimilarityTransform",
    "TrackRow",
)
