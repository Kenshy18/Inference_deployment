"""Penalty-DP decoding, exact Recall repair, and lazy interpolation."""

from __future__ import annotations

import argparse
import bisect
import math

import numpy as np

from .candidates import exact_k_dp
from .evaluation import (
    compute_cached_metrics_from_polygons,
    evaluate_frame_vector_loss_budget,
    flatten_contours,
    frame_accuracy_loss,
    recall_budget_from_metrics,
    recall_budget_limit,
    recall_violation,
    scale_vector_about_centroid,
    split_vector_to_polygons,
    vector_proxy_stats,
)
from .geometry import compute_exact_metrics_from_polygons
from .interpolation import (
    assign_candidate_ids_to_keyframes,
    interpolate_polygons,
    interpolate_vectors,
    interval_cost_from_vectors,
)
from .stream import parse_float_list
from .types import FrameEvalContext, InstanceRun, IntervalCost, ShapeCandidate


def run_multistate_penalty_path(
    run: InstanceRun,
    candidate_frames: list[int],
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[
    list[int],
    list[int],
    dict[str, int],
    dict[tuple[int, int, int, int, int], IntervalCost],
    float,
]:
    if all(len(frame_candidates) == 1 for frame_candidates in candidates_by_frame):
        return run_single_state_penalty_path(
            run,
            candidate_frames,
            candidates_by_frame,
            target_count,
            args,
            eval_contexts=eval_contexts,
        )

    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    state_frames: list[int] = []
    state_candidate_ids: list[int] = []
    node_offsets: list[tuple[int, int]] = []
    cursor = 0
    for frame_idx in candidate_frames:
        start = cursor
        for cand_id in range(len(candidates_by_frame[int(frame_idx)])):
            state_frames.append(int(frame_idx))
            state_candidate_ids.append(int(cand_id))
            cursor += 1
        node_offsets.append((start, cursor))
    state_count = cursor
    cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
    edge_array_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    counters = {"interval_evals": 0, "interval_frames": 0}
    use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
    recall_penalty_weight = float(args.proxy_recall_penalty_weight)
    predecessor_nodes: list[list[int]] = []
    for node_pos, end_frame in enumerate(candidate_frames):
        valid_prev: list[int] = []
        end_frame_i = int(end_frame)
        for prev_node_pos in range(node_pos):
            if end_frame_i - int(candidate_frames[prev_node_pos]) <= int(
                dynamic_max_gap
            ):
                valid_prev.append(int(prev_node_pos))
        predecessor_nodes.append(valid_prev)

    def get_cost(
        start_frame: int,
        start_cand: int,
        end_frame: int,
        end_cand: int,
        include_start: bool,
    ) -> IntervalCost:
        key = (
            int(start_frame),
            int(start_cand),
            int(end_frame),
            int(end_cand),
            1 if include_start else 0,
        )
        info = cost_cache.get(key)
        if info is None:
            full_key = (
                int(start_frame),
                int(start_cand),
                int(end_frame),
                int(end_cand),
                1,
            )
            full_info = cost_cache.get(full_key)
            if full_info is None:
                start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                end_candidate = candidates_by_frame[int(end_frame)][int(end_cand)]
                full_info = interval_cost_from_vectors(
                    run,
                    int(start_frame),
                    start_candidate.vector,
                    int(end_frame),
                    end_candidate.vector,
                    args,
                    include_start=True,
                    eval_contexts=eval_contexts,
                    start_candidate=start_candidate,
                    end_candidate=end_candidate,
                )
                cost_cache[full_key] = full_info
                counters["interval_evals"] += 1
                counters["interval_frames"] += int(full_info.frames_covered)
            if include_start:
                info = full_info
            else:
                start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                info = IntervalCost(
                    cost=float(full_info.cost - float(start_candidate.frame_loss)),
                    shape_distance=float(full_info.shape_distance),
                    shape_update=float(full_info.shape_update),
                    frames_covered=max(int(full_info.frames_covered) - 1, 0),
                    frame_loss_mean=float(full_info.frame_loss_mean),
                    shape_distance_scale=float(full_info.shape_distance_scale),
                    shape_switch_scale=float(full_info.shape_switch_scale),
                    recall_budget=max(
                        float(full_info.recall_budget)
                        - float(start_candidate.recall_budget),
                        0.0,
                    ),
                )
                cost_cache[key] = info
        return info

    def get_edge_arrays(
        prev_node_pos: int, node_pos: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(prev_node_pos), int(node_pos))
        cached = edge_array_cache.get(key)
        if cached is not None:
            return cached
        start_frame = int(candidate_frames[prev_node_pos])
        end_frame = int(candidate_frames[node_pos])
        src_start, src_end = node_offsets[prev_node_pos]
        dst_start, dst_end = node_offsets[node_pos]
        src_count = int(src_end - src_start)
        dst_count = int(dst_end - dst_start)
        cost_arr = np.empty((src_count, dst_count), dtype=np.float64)
        budget_arr = np.empty((src_count, dst_count), dtype=np.float64)
        for src_local, src_state in enumerate(range(src_start, src_end)):
            start_cand = int(state_candidate_ids[src_state])
            for dst_local, dst_state in enumerate(range(dst_start, dst_end)):
                end_cand = int(state_candidate_ids[dst_state])
                info = get_cost(
                    start_frame, start_cand, end_frame, end_cand, include_start=False
                )
                cost_arr[src_local, dst_local] = float(info.cost)
                budget_arr[src_local, dst_local] = float(info.recall_budget)
        edge_array_cache[key] = (cost_arr, budget_arr)
        return cost_arr, budget_arr

    def decode(
        lambda_penalty: float, recall_mu: float
    ) -> tuple[list[int], list[int], float, float]:
        dp = np.full((state_count,), np.inf, dtype=np.float64)
        back = np.full((state_count,), -1, dtype=np.int32)
        raw_cost = np.full((state_count,), np.inf, dtype=np.float64)
        raw_budget = np.full((state_count,), np.inf, dtype=np.float64)
        first_start, first_end = node_offsets[0]
        for state_idx in range(first_start, first_end):
            cand_id = int(state_candidate_ids[state_idx])
            frame_loss = float(candidates_by_frame[0][cand_id].frame_loss)
            frame_budget = float(candidates_by_frame[0][cand_id].recall_budget)
            penalty = (
                float(recall_mu) * frame_budget
                if use_exact_recall_dp
                else recall_penalty_weight * frame_budget
            )
            dp[state_idx] = frame_loss + penalty + float(lambda_penalty)
            raw_cost[state_idx] = frame_loss
            raw_budget[state_idx] = frame_budget
        for node_pos in range(1, len(candidate_frames)):
            dst_start, dst_end = node_offsets[node_pos]
            prev_entries = []
            for prev_node_pos in predecessor_nodes[node_pos]:
                src_start, src_end = node_offsets[prev_node_pos]
                edge_costs, edge_budgets = get_edge_arrays(prev_node_pos, node_pos)
                prev_entries.append((src_start, src_end, edge_costs, edge_budgets))
            for dst_state in range(dst_start, dst_end):
                dst_local = int(dst_state - dst_start)
                best_cost = float("inf")
                best_raw = float("inf")
                best_budget = float("inf")
                best_prev = -1
                for src_start, src_end, edge_costs, edge_budgets in prev_entries:
                    for src_state in range(src_start, src_end):
                        prev_cost = float(dp[src_state])
                        if not np.isfinite(prev_cost):
                            continue
                        src_local = int(src_state - src_start)
                        edge_cost = float(edge_costs[src_local, dst_local])
                        edge_budget = float(edge_budgets[src_local, dst_local])
                        penalty = (
                            float(recall_mu) * edge_budget
                            if use_exact_recall_dp
                            else recall_penalty_weight * edge_budget
                        )
                        cand_cost = (
                            prev_cost + edge_cost + penalty + float(lambda_penalty)
                        )
                        cand_raw = float(raw_cost[src_state]) + edge_cost
                        cand_budget = float(raw_budget[src_state]) + edge_budget
                        if cand_cost < best_cost or (
                            abs(cand_cost - best_cost) <= 1e-9
                            and (
                                cand_budget < best_budget
                                or (
                                    abs(cand_budget - best_budget) <= 1e-9
                                    and cand_raw < best_raw
                                )
                            )
                        ):
                            best_cost = float(cand_cost)
                            best_raw = float(cand_raw)
                            best_budget = float(cand_budget)
                            best_prev = int(src_state)
                dp[dst_state] = best_cost
                raw_cost[dst_state] = best_raw
                raw_budget[dst_state] = best_budget
                back[dst_state] = int(best_prev)
        last_start, last_end = node_offsets[-1]
        best_state = -1
        best_cost = float("inf")
        best_raw = float("inf")
        best_budget = float("inf")
        for state_idx in range(last_start, last_end):
            cost = float(dp[state_idx])
            raw = float(raw_cost[state_idx])
            budget = float(raw_budget[state_idx])
            if cost < best_cost or (
                abs(cost - best_cost) <= 1e-9
                and (
                    budget < best_budget
                    or (abs(budget - best_budget) <= 1e-9 and raw < best_raw)
                )
            ):
                best_cost = cost
                best_raw = raw
                best_budget = budget
                best_state = int(state_idx)
        if best_state < 0:
            raise RuntimeError("failed to decode penalized multistate path")
        chosen_frames: list[int] = []
        chosen_candidate_ids: list[int] = []
        cur_state = best_state
        while cur_state >= 0:
            chosen_frames.append(int(state_frames[cur_state]))
            chosen_candidate_ids.append(int(state_candidate_ids[cur_state]))
            cur_state = int(back[cur_state])
        chosen_frames.reverse()
        chosen_candidate_ids.reverse()
        return chosen_frames, chosen_candidate_ids, best_raw, best_budget

    def decode_for_recall_mu(
        recall_mu: float,
    ) -> tuple[list[int], list[int], float, float, float]:
        best: tuple[list[int], list[int], float, float] | None = None
        lo = 0.0
        hi = float(args.penalty_max)
        for _ in range(max(1, int(args.penalty_binary_steps))):
            mid = 0.5 * (lo + hi)
            cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
            candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
            if best is None:
                best = candidate
            else:
                cur_frames, cur_ids, cur_raw, cur_budget = best
                cand_gap = abs(len(cand_frames) - int(target_count))
                cur_gap = abs(len(cur_frames) - int(target_count))
                if cand_gap < cur_gap or (
                    cand_gap == cur_gap
                    and (
                        len(cand_frames) < len(cur_frames)
                        or (
                            len(cand_frames) == len(cur_frames)
                            and (
                                cand_budget < cur_budget
                                or (
                                    abs(cand_budget - cur_budget) <= 1e-9
                                    and cand_raw < cur_raw
                                )
                            )
                        )
                    )
                ):
                    best = candidate
            if len(cand_frames) > int(target_count):
                lo = mid
            else:
                hi = mid
        assert best is not None
        best_frames, best_ids, best_raw, best_budget = best
        return best_frames, best_ids, best_raw, best_budget, float(hi)

    if use_exact_recall_dp:
        best_result: tuple[list[int], list[int], float, float, float] | None = None
        recall_lo = 0.0
        recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
        for _ in range(max(1, int(args.recall_budget_binary_steps))):
            recall_mid = 0.5 * (recall_lo + recall_hi)
            (
                cand_frames,
                cand_ids,
                cand_raw,
                cand_budget,
                cand_lambda,
            ) = decode_for_recall_mu(recall_mid)
            cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
            if best_result is None:
                best_result = (
                    cand_frames,
                    cand_ids,
                    cand_raw,
                    cand_budget,
                    cand_lambda,
                )
            else:
                _bf, _bi, best_raw, best_budget, best_lambda = best_result
                best_violation = recall_violation(
                    best_budget, len(run.frame_numbers), args
                )
                if cand_violation < best_violation - 1e-12 or (
                    abs(cand_violation - best_violation) <= 1e-12
                    and (
                        cand_raw < best_raw
                        or (
                            abs(cand_raw - best_raw) <= 1e-9
                            and cand_lambda < best_lambda
                        )
                    )
                ):
                    best_result = (
                        cand_frames,
                        cand_ids,
                        cand_raw,
                        cand_budget,
                        cand_lambda,
                    )
            if cand_violation > 0.0:
                recall_lo = recall_mid
            else:
                recall_hi = recall_mid
        assert best_result is not None
        best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
    else:
        (
            best_frames,
            best_ids,
            _best_raw,
            _best_budget,
            best_lambda,
        ) = decode_for_recall_mu(0.0)
    return best_frames, best_ids, counters, cost_cache, float(best_lambda)


def run_single_state_penalty_path(
    run: InstanceRun,
    candidate_frames: list[int],
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[
    list[int],
    list[int],
    dict[str, int],
    dict[tuple[int, int, int, int, int], IntervalCost],
    float,
]:
    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    node_count = int(len(candidate_frames))
    cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
    edge_cache: dict[tuple[int, int], IntervalCost] = {}
    counters = {"interval_evals": 0, "interval_frames": 0}
    use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
    recall_penalty_weight = float(args.proxy_recall_penalty_weight)

    predecessor_nodes: list[list[int]] = []
    for node_pos, end_frame in enumerate(candidate_frames):
        end_frame_i = int(end_frame)
        min_prev_pos = int(
            bisect.bisect_left(
                candidate_frames, end_frame_i - int(dynamic_max_gap), 0, node_pos
            )
        )
        predecessor_nodes.append(list(range(min_prev_pos, node_pos)))

    def get_edge_info(prev_node_pos: int, node_pos: int) -> IntervalCost:
        key = (int(prev_node_pos), int(node_pos))
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        start_frame = int(candidate_frames[prev_node_pos])
        end_frame = int(candidate_frames[node_pos])
        start_candidate = candidates_by_frame[start_frame][0]
        end_candidate = candidates_by_frame[end_frame][0]
        info = interval_cost_from_vectors(
            run,
            start_frame,
            start_candidate.vector,
            end_frame,
            end_candidate.vector,
            args,
            include_start=False,
            eval_contexts=eval_contexts,
            start_candidate=start_candidate,
            end_candidate=end_candidate,
        )
        edge_cache[key] = info
        cost_cache[(start_frame, 0, end_frame, 0, 0)] = info
        counters["interval_evals"] += 1
        counters["interval_frames"] += int(info.frames_covered)
        return info

    def decode(
        lambda_penalty: float, recall_mu: float
    ) -> tuple[list[int], list[int], float, float]:
        dp = np.full((node_count,), np.inf, dtype=np.float64)
        back = np.full((node_count,), -1, dtype=np.int32)
        raw_cost = np.full((node_count,), np.inf, dtype=np.float64)
        raw_budget = np.full((node_count,), np.inf, dtype=np.float64)

        first_candidate = candidates_by_frame[int(candidate_frames[0])][0]
        first_budget = float(first_candidate.recall_budget)
        first_penalty = (
            float(recall_mu) * first_budget
            if use_exact_recall_dp
            else recall_penalty_weight * first_budget
        )
        dp[0] = (
            float(first_candidate.frame_loss) + first_penalty + float(lambda_penalty)
        )
        raw_cost[0] = float(first_candidate.frame_loss)
        raw_budget[0] = float(first_budget)

        for node_pos in range(1, node_count):
            best_cost = float("inf")
            best_raw = float("inf")
            best_budget = float("inf")
            best_prev = -1
            for prev_node_pos in predecessor_nodes[node_pos]:
                prev_cost = float(dp[prev_node_pos])
                if not np.isfinite(prev_cost):
                    continue
                info = get_edge_info(prev_node_pos, node_pos)
                edge_budget = float(info.recall_budget)
                penalty = (
                    float(recall_mu) * edge_budget
                    if use_exact_recall_dp
                    else recall_penalty_weight * edge_budget
                )
                cand_cost = (
                    prev_cost + float(info.cost) + penalty + float(lambda_penalty)
                )
                cand_raw = float(raw_cost[prev_node_pos]) + float(info.cost)
                cand_budget = float(raw_budget[prev_node_pos]) + edge_budget
                if cand_cost < best_cost or (
                    abs(cand_cost - best_cost) <= 1e-9
                    and (
                        cand_budget < best_budget
                        or (
                            abs(cand_budget - best_budget) <= 1e-9
                            and cand_raw < best_raw
                        )
                    )
                ):
                    best_cost = float(cand_cost)
                    best_raw = float(cand_raw)
                    best_budget = float(cand_budget)
                    best_prev = int(prev_node_pos)
            dp[node_pos] = best_cost
            raw_cost[node_pos] = best_raw
            raw_budget[node_pos] = best_budget
            back[node_pos] = int(best_prev)

        last_pos = int(node_count - 1)
        if last_pos < 0 or not np.isfinite(dp[last_pos]):
            raise RuntimeError("failed to decode single-state penalized path")

        chosen_frames: list[int] = []
        cur_pos = last_pos
        while cur_pos >= 0:
            chosen_frames.append(int(candidate_frames[cur_pos]))
            cur_pos = int(back[cur_pos])
        chosen_frames.reverse()
        chosen_candidate_ids = [0] * len(chosen_frames)
        return (
            chosen_frames,
            chosen_candidate_ids,
            float(raw_cost[last_pos]),
            float(raw_budget[last_pos]),
        )

    def decode_for_recall_mu(
        recall_mu: float,
    ) -> tuple[list[int], list[int], float, float, float]:
        best: tuple[list[int], list[int], float, float] | None = None
        lo = 0.0
        hi = float(args.penalty_max)
        for _ in range(max(1, int(args.penalty_binary_steps))):
            mid = 0.5 * (lo + hi)
            cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
            candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
            if best is None:
                best = candidate
            else:
                cur_frames, _cur_ids, cur_raw, cur_budget = best
                cand_gap = abs(len(cand_frames) - int(target_count))
                cur_gap = abs(len(cur_frames) - int(target_count))
                if cand_gap < cur_gap or (
                    cand_gap == cur_gap
                    and (
                        len(cand_frames) < len(cur_frames)
                        or (
                            len(cand_frames) == len(cur_frames)
                            and (
                                cand_budget < cur_budget
                                or (
                                    abs(cand_budget - cur_budget) <= 1e-9
                                    and cand_raw < cur_raw
                                )
                            )
                        )
                    )
                ):
                    best = candidate
            if len(cand_frames) > int(target_count):
                lo = mid
            else:
                hi = mid
        assert best is not None
        best_frames, best_ids, best_raw, best_budget = best
        return best_frames, best_ids, best_raw, best_budget, float(hi)

    if use_exact_recall_dp:
        best_result: tuple[list[int], list[int], float, float, float] | None = None
        recall_lo = 0.0
        recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
        for _ in range(max(1, int(args.recall_budget_binary_steps))):
            recall_mid = 0.5 * (recall_lo + recall_hi)
            (
                cand_frames,
                cand_ids,
                cand_raw,
                cand_budget,
                cand_lambda,
            ) = decode_for_recall_mu(recall_mid)
            cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
            if best_result is None:
                best_result = (
                    cand_frames,
                    cand_ids,
                    cand_raw,
                    cand_budget,
                    cand_lambda,
                )
            else:
                _bf, _bi, best_raw, best_budget, best_lambda = best_result
                best_violation = recall_violation(
                    best_budget, len(run.frame_numbers), args
                )
                if cand_violation < best_violation - 1e-12 or (
                    abs(cand_violation - best_violation) <= 1e-12
                    and (
                        cand_raw < best_raw
                        or (
                            abs(cand_raw - best_raw) <= 1e-9
                            and cand_lambda < best_lambda
                        )
                    )
                ):
                    best_result = (
                        cand_frames,
                        cand_ids,
                        cand_raw,
                        cand_budget,
                        cand_lambda,
                    )
            if cand_violation > 0.0:
                recall_lo = recall_mid
            else:
                recall_hi = recall_mid
        assert best_result is not None
        best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
    else:
        (
            best_frames,
            best_ids,
            _best_raw,
            _best_budget,
            best_lambda,
        ) = decode_for_recall_mu(0.0)
    return best_frames, best_ids, counters, cost_cache, float(best_lambda)


def evaluate_keyframe_path(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, list[IntervalCost], float]:
    (
        total,
        _start_loss,
        interval_infos,
        total_recall_budget,
        _start_budget,
    ) = evaluate_keyframe_path_parts(
        run,
        chosen_frames,
        keyframe_vectors,
        args,
        eval_contexts=eval_contexts,
    )
    return float(total), interval_infos, float(total_recall_budget)


def evaluate_keyframe_path_parts(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, float, list[IntervalCost], float, float]:
    total = 0.0
    interval_infos: list[IntervalCost] = []
    if len(chosen_frames) <= 0:
        return float("inf"), float("inf"), interval_infos, float("inf"), float("inf")
    start_vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
    start_loss, start_budget = evaluate_frame_vector_loss_budget(
        run, int(chosen_frames[0]), start_vec, args, eval_contexts=eval_contexts
    )
    total_recall_budget = float(start_budget)
    total += float(start_loss)
    for left_idx, right_idx, left_vec, right_vec in zip(
        chosen_frames[:-1],
        chosen_frames[1:],
        keyframe_vectors[:-1],
        keyframe_vectors[1:],
    ):
        info = interval_cost_from_vectors(
            run,
            int(left_idx),
            np.asarray(left_vec, dtype=np.float32),
            int(right_idx),
            np.asarray(right_vec, dtype=np.float32),
            args,
            include_start=False,
            eval_contexts=eval_contexts,
        )
        interval_infos.append(info)
        total += float(info.cost)
        total_recall_budget += float(info.recall_budget)
    return (
        float(total),
        float(start_loss),
        interval_infos,
        float(total_recall_budget),
        float(start_budget),
    )


def exact_interpolated_metrics(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
) -> tuple[list[dict[str, float]], float, float, float, float, float]:
    metrics_rows: list[dict[str, float]] = []
    total_iou_loss = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_gt_area = 0.0
    total_intersection = 0.0
    chosen_frames_arr = [int(v) for v in chosen_frames]
    interval_pos = 0
    for frame_idx in range(len(run.frame_numbers)):
        if frame_idx <= chosen_frames_arr[0]:
            vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
        elif frame_idx >= chosen_frames_arr[-1]:
            vec = np.asarray(keyframe_vectors[-1], dtype=np.float32)
        else:
            while interval_pos + 1 < len(chosen_frames_arr) and frame_idx > int(
                chosen_frames_arr[interval_pos + 1]
            ):
                interval_pos += 1
            right_pos = int(interval_pos + 1)
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen_frames_arr[left_pos])
            right_frame = int(chosen_frames_arr[right_pos])
            if frame_idx == right_frame:
                vec = np.asarray(keyframe_vectors[right_pos], dtype=np.float32)
            else:
                alpha = float(
                    (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                )
                vec = interpolate_vectors(
                    keyframe_vectors[left_pos], keyframe_vectors[right_pos], alpha
                )
        pred_polys = split_vector_to_polygons(
            vec, run.contour_count, run.anchors_per_contour
        )
        metrics = compute_exact_metrics_from_polygons(
            run.gt_polygons[frame_idx], pred_polys
        )
        metrics_rows.append(metrics)
        total_iou_loss += 1.0 - float(metrics["iou"])
        total_recall += float(metrics["recall"])
        total_precision += float(metrics["precision"])
        total_gt_area += float(metrics["gt_area"])
        total_intersection += float(metrics["intersection"])
    mean_iou = float(1.0 - total_iou_loss / max(len(metrics_rows), 1))
    mean_recall = float(total_recall / max(len(metrics_rows), 1))
    mean_precision = float(total_precision / max(len(metrics_rows), 1))
    global_recall = (
        float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0
    )
    return (
        metrics_rows,
        float(total_iou_loss),
        float(mean_iou),
        float(mean_recall),
        float(mean_precision),
        float(global_recall),
    )


def exact_recall_solution_key(
    total_iou_loss: float, mean_recall: float, args: argparse.Namespace
) -> tuple[float, float, float]:
    violation = max(float(args.recall_min) - float(mean_recall), 0.0)
    return float(violation), float(total_iou_loss), float(-mean_recall)


def repair_keyframe_vectors_for_exact_recall(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    candidates_by_frame: list[list[ShapeCandidate]],
    args: argparse.Namespace,
) -> np.ndarray:
    if not bool(args.exact_recall_repair_enabled) or len(chosen_frames) <= 0:
        return np.asarray(keyframe_vectors, dtype=np.float32)
    current = np.asarray(keyframe_vectors, dtype=np.float32).copy()
    scale_deltas = parse_float_list(
        str(args.exact_recall_repair_scale_deltas), [0.01, 0.02, 0.04, 0.06, 0.08]
    )
    (
        metrics_rows,
        current_iou_loss,
        _current_mean_iou,
        current_mean_recall,
        _current_mean_precision,
        _current_global_recall,
    ) = exact_interpolated_metrics(run, chosen_frames, current)
    best_key = exact_recall_solution_key(current_iou_loss, current_mean_recall, args)
    if best_key[0] <= 0.0:
        return current

    for _pass in range(max(1, int(args.exact_recall_repair_max_passes))):
        frame_deficits = np.asarray(
            [
                float(row["gt_area"])
                * max(float(args.recall_min) - float(row["recall"]), 0.0)
                for row in metrics_rows
            ],
            dtype=np.float64,
        )
        if float(np.mean(frame_deficits)) <= 0.0 and best_key[0] <= 0.0:
            break
        key_scores = np.zeros((len(chosen_frames),), dtype=np.float64)
        for frame_idx, deficit in enumerate(frame_deficits.tolist()):
            if deficit <= 0.0:
                continue
            if frame_idx <= int(chosen_frames[0]):
                key_scores[0] += float(deficit)
                continue
            if frame_idx >= int(chosen_frames[-1]):
                key_scores[-1] += float(deficit)
                continue
            right_pos = next(
                pos
                for pos, keyframe in enumerate(chosen_frames)
                if keyframe >= frame_idx
            )
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen_frames[left_pos])
            right_frame = int(chosen_frames[right_pos])
            alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
            key_scores[left_pos] += (1.0 - alpha) * float(deficit)
            key_scores[right_pos] += alpha * float(deficit)
        key_order = [
            int(idx)
            for idx in np.argsort(-key_scores)[
                : max(1, int(args.exact_recall_repair_topk))
            ].tolist()
        ]
        improved = False

        trial_vectors: list[np.ndarray] = []
        for delta in scale_deltas:
            scaled_all = np.asarray(current, dtype=np.float32).copy()
            for key_idx in range(len(chosen_frames)):
                scaled_all[key_idx] = scale_vector_about_centroid(
                    scaled_all[key_idx], 1.0 + float(delta)
                )
            trial_vectors.append(scaled_all)
        for delta in scale_deltas:
            scaled = np.asarray(current, dtype=np.float32).copy()
            for key_idx in key_order:
                scaled[key_idx] = scale_vector_about_centroid(
                    scaled[key_idx], 1.0 + float(delta)
                )
            trial_vectors.append(scaled)

        for key_idx in key_order:
            frame_idx = int(chosen_frames[key_idx])
            current_area, _center, _radii, _mean_radius = vector_proxy_stats(
                current[key_idx], run.contour_count, run.anchors_per_contour
            )
            for candidate in candidates_by_frame[frame_idx]:
                if float(candidate.area) <= float(current_area) + 1e-3:
                    continue
                upgraded = np.asarray(current, dtype=np.float32).copy()
                upgraded[key_idx] = np.asarray(candidate.vector, dtype=np.float32)
                trial_vectors.append(upgraded)
            for delta in scale_deltas:
                upgraded = np.asarray(current, dtype=np.float32).copy()
                upgraded[key_idx] = scale_vector_about_centroid(
                    upgraded[key_idx], 1.0 + float(delta)
                )
                trial_vectors.append(upgraded)

        seen: list[np.ndarray] = []
        for trial in trial_vectors:
            if any(np.allclose(trial, existing, atol=1e-4) for existing in seen):
                continue
            seen.append(np.asarray(trial, dtype=np.float32))
            (
                trial_metrics,
                trial_iou_loss,
                _trial_mean_iou,
                trial_mean_recall,
                _trial_mean_precision,
                _trial_global_recall,
            ) = exact_interpolated_metrics(run, chosen_frames, trial)
            trial_key = exact_recall_solution_key(
                trial_iou_loss, trial_mean_recall, args
            )
            if trial_key < best_key:
                current = np.asarray(trial, dtype=np.float32)
                metrics_rows = trial_metrics
                best_key = trial_key
                improved = True
        if not improved:
            break
        if best_key[0] <= 0.0:
            break
    return np.asarray(current, dtype=np.float32)


class LazyInterpolatedRun:
    def __init__(
        self, run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray
    ):
        self.run = run
        self.chosen_frames = [int(v) for v in chosen_frames]
        self.keyframe_vectors = np.asarray(keyframe_vectors, dtype=np.float32)
        self.length = int(len(run.frame_numbers))

    def __len__(self) -> int:
        return int(self.length)

    def _polygons_at(self, frame_idx: int) -> list[np.ndarray]:
        idx = int(frame_idx)
        if idx < 0:
            idx += int(self.length)
        if idx < 0 or idx >= int(self.length):
            raise IndexError(frame_idx)
        chosen = self.chosen_frames
        if idx <= chosen[0]:
            vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
        elif idx >= chosen[-1]:
            vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
        else:
            right_pos = int(
                np.searchsorted(
                    np.asarray(chosen, dtype=np.int32), int(idx), side="left"
                )
            )
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen[left_pos])
            right_frame = int(chosen[right_pos])
            if idx == right_frame:
                vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
            else:
                alpha = float((idx - left_frame) / max(right_frame - left_frame, 1))
                vec = interpolate_vectors(
                    self.keyframe_vectors[left_pos],
                    self.keyframe_vectors[right_pos],
                    alpha,
                )
        return split_vector_to_polygons(
            vec, self.run.contour_count, self.run.anchors_per_contour
        )

    def __getitem__(self, frame_idx):
        if isinstance(frame_idx, slice):
            return [
                self._polygons_at(idx)
                for idx in range(*frame_idx.indices(int(self.length)))
            ]
        return self._polygons_at(int(frame_idx))

    def __iter__(self):
        if self.length <= 0:
            return
        chosen = self.chosen_frames
        interval_pos = 0
        for frame_idx in range(int(self.length)):
            if frame_idx <= chosen[0]:
                vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
            elif frame_idx >= chosen[-1]:
                vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
            else:
                while interval_pos + 1 < len(chosen) and frame_idx > int(
                    chosen[interval_pos + 1]
                ):
                    interval_pos += 1
                right_pos = int(interval_pos + 1)
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen[left_pos])
                right_frame = int(chosen[right_pos])
                if frame_idx == right_frame:
                    vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
                else:
                    alpha = float(
                        (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                    )
                    vec = interpolate_vectors(
                        self.keyframe_vectors[left_pos],
                        self.keyframe_vectors[right_pos],
                        alpha,
                    )
            yield split_vector_to_polygons(
                vec, self.run.contour_count, self.run.anchors_per_contour
            )


def interpolate_run(
    run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray
):
    length = len(run.frame_numbers)
    if length <= 0:
        return []
    return LazyInterpolatedRun(run, chosen_frames, keyframe_vectors)


class LazyUnionRows:
    def __init__(self, run: InstanceRun, interp_polygons, chosen_frames: list[int]):
        self.run = run
        self.interp_polygons = interp_polygons
        self.chosen_set = {int(v) for v in chosen_frames}
        length = int(len(run.frame_numbers))
        self.emit_start = int(max(0, min(length, int(run.emit_start_idx))))
        emit_end = length if int(run.emit_end_idx) < 0 else int(run.emit_end_idx)
        self.emit_end = int(max(self.emit_start, min(length, emit_end)))

    def __len__(self) -> int:
        return int(self.emit_end - self.emit_start)

    def __iter__(self):
        for local_idx, (frame, polygons) in enumerate(
            zip(self.run.frame_numbers.tolist(), self.interp_polygons)
        ):
            if local_idx < self.emit_start or local_idx >= self.emit_end:
                continue
            yield {
                "frame": int(frame),
                "track_id": str(self.run.track_id),
                "run_id": int(self.run.run_id),
                "polygons": [
                    np.asarray(poly, dtype=np.float32).tolist() for poly in polygons
                ],
                "has_keyframe": 1 if local_idx in self.chosen_set else 0,
                "is_gapfill": int(self.run.gapfilled_flags[local_idx])
                if self.run.gapfilled_flags is not None
                and local_idx < len(self.run.gapfilled_flags)
                else 0,
            }
