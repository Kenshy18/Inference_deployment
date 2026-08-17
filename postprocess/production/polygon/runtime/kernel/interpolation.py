"""Keyframe interpolation and pair-vote primitives."""

from __future__ import annotations

import argparse

import numpy as np

from .evaluation import (
    adaptive_shape_penalty_scales,
    compute_cached_metrics_from_interpolated_polygons,
    compute_cached_metrics_from_polygons,
    flatten_contours,
    frame_accuracy_loss,
    recall_budget_from_metrics,
    shape_distance,
    split_vector_to_polygons,
)
from .geometry import compute_exact_metrics_from_polygons
from .types import FrameEvalContext, InstanceRun, IntervalCost, ShapeCandidate


def build_ring_second_difference_rtr(
    contour_count: int, anchors_per_contour: int
) -> np.ndarray:
    point_count = int(contour_count) * int(anchors_per_contour)
    dim = int(point_count * 2)
    rows: list[np.ndarray] = []
    for contour_idx in range(int(contour_count)):
        base = contour_idx * int(anchors_per_contour)
        for anchor_idx in range(int(anchors_per_contour)):
            prev_idx = base + ((anchor_idx - 1) % int(anchors_per_contour))
            cur_idx = base + anchor_idx
            next_idx = base + ((anchor_idx + 1) % int(anchors_per_contour))
            for axis in range(2):
                row = np.zeros((dim,), dtype=np.float64)
                row[2 * prev_idx + axis] = 1.0
                row[2 * cur_idx + axis] = -2.0
                row[2 * next_idx + axis] = 1.0
                rows.append(row)
    if not rows:
        return np.zeros((dim, dim), dtype=np.float64)
    mat = np.asarray(rows, dtype=np.float64)
    return mat.T @ mat


def build_interpolation_weights(
    frame_count: int, chosen_frames: list[int]
) -> np.ndarray:
    key_count = int(len(chosen_frames))
    weights = np.zeros((int(frame_count), key_count), dtype=np.float64)
    chosen = [int(v) for v in chosen_frames]
    if key_count <= 0:
        return weights
    for frame_idx in range(int(frame_count)):
        if frame_idx <= chosen[0]:
            weights[frame_idx, 0] = 1.0
            continue
        if frame_idx >= chosen[-1]:
            weights[frame_idx, -1] = 1.0
            continue
        right_pos = next(
            pos for pos, keyframe in enumerate(chosen) if keyframe >= frame_idx
        )
        left_pos = max(0, right_pos - 1)
        left_frame = int(chosen[left_pos])
        right_frame = int(chosen[right_pos])
        if frame_idx == right_frame or right_frame <= left_frame:
            weights[frame_idx, right_pos] = 1.0
        else:
            alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
            weights[frame_idx, left_pos] = 1.0 - alpha
            weights[frame_idx, right_pos] = alpha
    return weights


def pair_vote_refine_keyframe_vectors(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if (
        not bool(getattr(args, "pair_vote_refine_enabled", True))
        or len(chosen_frames) <= 1
    ):
        return np.asarray(keyframe_vectors, dtype=np.float32)
    frame_count = int(len(run.frame_numbers))
    targets = np.asarray(
        [flatten_contours(run.anchors[idx]).reshape(-1) for idx in range(frame_count)],
        dtype=np.float64,
    )
    init = np.asarray(keyframe_vectors, dtype=np.float64).reshape(
        len(chosen_frames), -1
    )
    proposals: list[list[tuple[np.ndarray, float]]] = [[] for _ in chosen_frames]
    eye2 = np.eye(2, dtype=np.float64)
    for left_pos in range(len(chosen_frames) - 1):
        right_pos = left_pos + 1
        u = int(chosen_frames[left_pos])
        v = int(chosen_frames[right_pos])
        span = max(v - u, 1)
        rows = []
        local_targets = []
        for frame_idx in range(u, v + 1):
            beta = float(v - frame_idx) / float(span)
            gamma = float(frame_idx - u) / float(span)
            rows.append([beta, gamma])
            local_targets.append(targets[frame_idx])
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(local_targets, dtype=np.float64)
        gram = x.T @ x
        rhs = x.T @ y
        ab = np.linalg.solve(gram + 1e-8 * eye2, rhs)
        interval_weight = float(v - u + 1)
        proposals[left_pos].append(
            (np.asarray(ab[0], dtype=np.float32), interval_weight)
        )
        proposals[right_pos].append(
            (np.asarray(ab[1], dtype=np.float32), interval_weight)
        )
    out = init.copy()
    for idx, items in enumerate(proposals):
        if not items:
            continue
        total_w = float(sum(weight for _vec, weight in items))
        voted = sum(
            np.asarray(vec, dtype=np.float64) * float(weight) for vec, weight in items
        ) / max(total_w, 1e-8)
        out[idx] = voted
    return np.asarray(out.reshape(np.asarray(keyframe_vectors).shape), dtype=np.float32)


def interpolate_vectors(
    start_vec: np.ndarray, end_vec: np.ndarray, alpha: float
) -> np.ndarray:
    return (
        (1.0 - float(alpha)) * np.asarray(start_vec, dtype=np.float32)
        + float(alpha) * np.asarray(end_vec, dtype=np.float32)
    ).astype(np.float32)


def interpolate_polygons(
    start_polys: list[np.ndarray], end_polys: list[np.ndarray], alpha: float
) -> list[np.ndarray]:
    alpha32 = np.float32(alpha)
    beta32 = np.float32(1.0) - alpha32
    out: list[np.ndarray] = []
    for start_poly, end_poly in zip(start_polys, end_polys):
        start_pts = np.asarray(start_poly, dtype=np.float32)
        end_pts = np.asarray(end_poly, dtype=np.float32)
        out.append((beta32 * start_pts + alpha32 * end_pts).astype(np.float32))
    return out


def assign_candidate_ids_to_keyframes(
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    candidates_by_frame: list[list[ShapeCandidate]],
) -> list[int]:
    candidate_ids: list[int] = []
    for frame_idx, vector in zip(chosen_frames, keyframe_vectors):
        frame_candidates = candidates_by_frame[int(frame_idx)]
        best_cand = 0
        best_dist = float("inf")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        for cand_id, candidate in enumerate(frame_candidates):
            cand_vec = np.asarray(candidate.vector, dtype=np.float32).reshape(-1)
            dist = float(np.mean(np.square(vec - cand_vec)))
            if dist < best_dist:
                best_dist = dist
                best_cand = int(cand_id)
        candidate_ids.append(int(best_cand))
    return candidate_ids


def interval_cost_from_vectors(
    run: InstanceRun,
    start_idx: int,
    start_vec: np.ndarray,
    end_idx: int,
    end_vec: np.ndarray,
    args: argparse.Namespace,
    *,
    include_start: bool,
    eval_contexts: list[FrameEvalContext] | None = None,
    start_candidate: ShapeCandidate | None = None,
    end_candidate: ShapeCandidate | None = None,
) -> IntervalCost:
    if end_idx < start_idx:
        return IntervalCost(
            cost=float("inf"),
            shape_distance=float("inf"),
            shape_update=1.0,
            frames_covered=0,
        )
    start_polys = (
        start_candidate.polygons
        if start_candidate is not None
        else split_vector_to_polygons(
            start_vec, run.contour_count, run.anchors_per_contour
        )
    )
    end_polys = (
        end_candidate.polygons
        if end_candidate is not None
        else split_vector_to_polygons(
            end_vec, run.contour_count, run.anchors_per_contour
        )
    )
    dist = shape_distance(start_vec, end_vec, run.scale)
    update = 1.0 if dist > float(args.shape_update_threshold_ratio) else 0.0
    total = 0.0
    frames_covered = 0
    frame_loss_total = 0.0
    recall_budget_total = 0.0
    start_frame = int(start_idx if include_start else start_idx + 1)
    for frame_idx in range(start_frame, int(end_idx) + 1):
        if frame_idx == start_idx:
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_polygons(
                    eval_contexts[frame_idx], start_polys
                )
            else:
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], start_polys
                )
        elif frame_idx == end_idx:
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_polygons(
                    eval_contexts[frame_idx], end_polys
                )
            else:
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], end_polys
                )
        else:
            alpha = float((frame_idx - start_idx) / max(end_idx - start_idx, 1))
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_interpolated_polygons(
                    eval_contexts[frame_idx],
                    start_polys,
                    end_polys,
                    alpha,
                )
            else:
                pred_polys = interpolate_polygons(start_polys, end_polys, alpha)
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], pred_polys
                )
        frame_loss = float(frame_accuracy_loss(metrics, args))
        recall_budget = float(recall_budget_from_metrics(metrics))
        total += float(frame_loss)
        frame_loss_total += float(frame_loss)
        recall_budget_total += float(recall_budget)
        frames_covered += 1
    frame_loss_mean = float(frame_loss_total / max(frames_covered, 1))
    dist_scale, switch_scale = adaptive_shape_penalty_scales(frame_loss_mean, args)
    total += float(args.shape_switch_weight) * float(switch_scale) * float(update)
    total += float(args.shape_distance_weight) * float(dist_scale) * float(dist)
    return IntervalCost(
        cost=float(total),
        shape_distance=float(dist),
        shape_update=float(update),
        frames_covered=int(frames_covered),
        frame_loss_mean=float(frame_loss_mean),
        shape_distance_scale=float(dist_scale),
        shape_switch_scale=float(switch_scale),
        recall_budget=float(recall_budget_total),
    )
