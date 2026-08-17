"""Candidate-frame saliency, surrogate paths, and exact-K fallback DP."""

from __future__ import annotations

import argparse
import bisect
import math

import numpy as np

from .evaluation import recall_violation, shape_distance
from .types import InstanceRun, ShapeCandidate


def compute_saliency_scores(
    run: InstanceRun,
    fit_vectors: list[np.ndarray],
    area_series: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    length = len(fit_vectors)
    scores = np.zeros((length,), dtype=np.float64)
    area_scale = max(float(np.mean(np.asarray(area_series, dtype=np.float64))), 1.0)
    for idx in range(1, length - 1):
        prev_vec = np.asarray(fit_vectors[idx - 1], dtype=np.float64)
        cur_vec = np.asarray(fit_vectors[idx], dtype=np.float64)
        next_vec = np.asarray(fit_vectors[idx + 1], dtype=np.float64)
        second = float(
            np.linalg.norm(next_vec - 2.0 * cur_vec + prev_vec)
            / max(float(run.scale), 1.0)
        )
        jump = shape_distance(fit_vectors[idx - 1], fit_vectors[idx + 1], run.scale)
        area_peak = (
            max(
                float(area_series[idx])
                - 0.5 * float(area_series[idx - 1] + area_series[idx + 1]),
                0.0,
            )
            / area_scale
        )
        area_swing = (
            abs(float(area_series[idx + 1]) - float(area_series[idx - 1])) / area_scale
        )
        scores[idx] = (
            second
            + float(args.saliency_shape_eta) * jump
            + float(args.saliency_area_eta) * (area_peak + 0.5 * area_swing)
        )
    if length > 0:
        scores[0] = float(scores[1] if length > 1 else 0.0)
        scores[-1] = float(scores[-2] if length > 1 else 0.0)
    return scores


def compute_surrogate_prefix(
    vectors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(
        [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors],
        dtype=np.float64,
    )
    times = np.arange(q.shape[0], dtype=np.float64)[:, None]
    prefix_q = np.concatenate(
        [np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q, axis=0)], axis=0
    )
    prefix_tq = np.concatenate(
        [np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q * times, axis=0)],
        axis=0,
    )
    prefix_q2 = np.concatenate(
        [np.zeros((1,), dtype=np.float64), np.cumsum(np.sum(q * q, axis=1), axis=0)],
        axis=0,
    )
    return prefix_q, prefix_tq, prefix_q2


def surrogate_interval_cost(
    u: int,
    v: int,
    prefix_q: np.ndarray,
    prefix_tq: np.ndarray,
    prefix_q2: np.ndarray,
    vector_dim: int,
    contour_count: int,
    anchors_per_contour: int,
    scale: float,
    args: argparse.Namespace,
) -> float:
    cost, _start_vec, _end_vec = surrogate_interval_solution(
        u,
        v,
        prefix_q,
        prefix_tq,
        prefix_q2,
        vector_dim,
        contour_count,
        anchors_per_contour,
        scale,
        args,
    )
    return float(cost)


def surrogate_interval_solution(
    u: int,
    v: int,
    prefix_q: np.ndarray,
    prefix_tq: np.ndarray,
    prefix_q2: np.ndarray,
    vector_dim: int,
    contour_count: int,
    anchors_per_contour: int,
    scale: float,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray, np.ndarray]:
    if v <= u:
        zero = np.zeros((vector_dim // 2, 2), dtype=np.float32)
        return 0.0, zero, zero
    h = int(v - u)
    s0 = prefix_q[v + 1] - prefix_q[u]
    s1 = prefix_tq[v + 1] - prefix_tq[u]
    s2 = float(prefix_q2[v + 1] - prefix_q2[u])
    a = float((h + 1) * (2 * h + 1) / (6.0 * h))
    b = float((h + 1) * (h - 1) / (6.0 * h))
    c = float(a)
    gu = (float(v) * s0 - s1) / float(h)
    gv = (s1 - float(u) * s0) / float(h)
    det = max(a * c - b * b, 1e-9)
    avec = (c * gu - b * gv) / det
    bvec = (-b * gu + a * gv) / det
    quad = (
        a * float(np.dot(avec, avec))
        + 2.0 * b * float(np.dot(avec, bvec))
        + c * float(np.dot(bvec, bvec))
    )
    cross = 2.0 * float(np.dot(gu, avec) + np.dot(gv, bvec))
    sse = max(s2 - cross + quad, 0.0)
    start_vec = np.asarray(avec, dtype=np.float32).reshape(vector_dim // 2, 2)
    end_vec = np.asarray(bvec, dtype=np.float32).reshape(vector_dim // 2, 2)
    d = shape_distance(start_vec, end_vec, scale)
    return (
        float(
            sse / max(float(scale) ** 2, 1.0) + float(args.surrogate_shape_weight) * d
        ),
        start_vec,
        end_vec,
    )


def exact_k_dp(cost_fn, nodes: list[int], target_count: int, max_gap: int) -> list[int]:
    node_count = len(nodes)
    target_count = max(2, min(int(target_count), node_count))
    dp = np.full((target_count, node_count), np.inf, dtype=np.float64)
    back = np.full((target_count, node_count), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for used in range(1, target_count):
        for end_pos in range(used, node_count):
            end_node = int(nodes[end_pos])
            min_prev_pos = max(
                used - 1,
                int(bisect.bisect_left(nodes, end_node - int(max_gap), 0, end_pos)),
            )
            best_cost = float("inf")
            best_prev = -1
            for prev_pos in range(min_prev_pos, end_pos):
                prev_node = int(nodes[prev_pos])
                prev_cost = float(dp[used - 1, prev_pos])
                if not np.isfinite(prev_cost):
                    continue
                cand = prev_cost + float(cost_fn(prev_node, end_node))
                if cand < best_cost:
                    best_cost = cand
                    best_prev = int(prev_pos)
            dp[used, end_pos] = best_cost
            back[used, end_pos] = best_prev
    path = [node_count - 1]
    cur_pos = node_count - 1
    cur_used = target_count - 1
    while cur_used > 0:
        cur_pos = int(back[cur_used, cur_pos])
        if cur_pos < 0:
            return [int(nodes[0]), int(nodes[-1])]
        path.append(cur_pos)
        cur_used -= 1
    path.reverse()
    return [int(nodes[pos]) for pos in path]


def build_candidate_frame_pool(
    run: InstanceRun,
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], np.ndarray]:
    raw_vectors = [
        frame_candidates[0].vector for frame_candidates in candidates_by_frame
    ]
    area_series = np.asarray(
        [float(frame_candidates[0].area) for frame_candidates in candidates_by_frame],
        dtype=np.float64,
    )
    scores = compute_saliency_scores(run, raw_vectors, area_series, args)
    length = len(run.frame_numbers)
    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    prefix_q, prefix_tq, prefix_q2 = compute_surrogate_prefix(raw_vectors)
    vector_dim = (
        int(np.asarray(raw_vectors[0], dtype=np.float32).size) if raw_vectors else 0
    )

    surrogate_cost_cache: dict[tuple[int, int], float] = {}

    def surrogate_cost(u: int, v: int) -> float:
        key = (int(u), int(v))
        cached = surrogate_cost_cache.get(key)
        if cached is not None:
            return float(cached)
        cost = surrogate_interval_cost(
            int(u),
            int(v),
            prefix_q,
            prefix_tq,
            prefix_q2,
            vector_dim,
            run.contour_count,
            run.anchors_per_contour,
            run.scale,
            args,
        )
        surrogate_cost_cache[key] = float(cost)
        return float(cost)

    all_nodes = list(range(length))
    surrogate_path = exact_k_dp(
        surrogate_cost, all_nodes, int(target_count), dynamic_max_gap
    )
    pool_target = min(
        length,
        max(
            int(round(float(args.surrogate_pool_factor) * float(target_count))),
            int(math.ceil(math.sqrt(max(length, 1)))),
            int(target_count) + 2,
        ),
    )
    peak_target = min(
        length,
        max(0, int(round(float(args.surrogate_peak_factor) * float(target_count)))),
    )
    peak_ids = [int(idx) for idx in np.argsort(-scores)[:peak_target].tolist()]
    grid = list(range(0, length, max(1, target_interval)))
    if grid[-1] != length - 1:
        grid.append(length - 1)
    pool = {0, length - 1}
    for frame_idx in surrogate_path:
        for delta in range(
            -int(args.surrogate_neighbor_radius),
            int(args.surrogate_neighbor_radius) + 1,
        ):
            cand = int(frame_idx) + int(delta)
            if 0 <= cand < length:
                pool.add(int(cand))
    for frame_idx in peak_ids:
        pool.add(int(frame_idx))
    for frame_idx in grid:
        pool.add(int(frame_idx))
    if len(pool) < int(target_count):
        for frame_idx in np.argsort(-scores).tolist():
            pool.add(int(frame_idx))
            if len(pool) >= int(target_count):
                break
    if len(pool) < pool_target:
        for frame_idx in np.argsort(-scores).tolist():
            pool.add(int(frame_idx))
            if len(pool) >= pool_target:
                break
    return (
        sorted(int(frame_idx) for frame_idx in pool),
        [int(frame_idx) for frame_idx in surrogate_path],
        scores,
    )


__all__ = (
    "build_candidate_frame_pool",
    "compute_saliency_scores",
    "compute_surrogate_prefix",
    "exact_k_dp",
    "surrogate_interval_cost",
    "surrogate_interval_solution",
)
