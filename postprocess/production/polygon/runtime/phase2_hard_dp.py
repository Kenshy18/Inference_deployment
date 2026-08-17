"""Exact hard-Recall multistate penalty DP for promoted Phase 2."""

from __future__ import annotations

import bisect
import json
import math
import os
import sys
import time

import numpy as np

from production.polygon.runtime.phase1_runtime import _EPSILON
from production.polygon.runtime.phase2_config import (
    CUDA_APPROX_ONLY_ENV,
    CUDA_EXACT_HINT_COUNT_ENV,
    CUDA_EXACT_HINT_ENV,
    CUDA_LAZY_DEFICIT_PENALTY_ENV,
    CUDA_LAZY_EXACT_ENV,
    CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO_ENV,
    CUDA_LAZY_FALLBACK_MIN_EDGES_ENV,
    CUDA_LAZY_FALLBACK_MIN_SECONDS_ENV,
    CUDA_LAZY_MAX_SECONDS_ENV,
    CUDA_LAZY_MIN_RETAINED_RATIO_ENV,
    CUDA_LAZY_STATE_PAIR_BATCH_ENV,
    CUDA_PREFILTER_BUDGET_ENV,
    CUDA_PREFILTER_ENV,
    CUDA_PREFILTER_SMALL_AREA_ENV,
    CUDA_PREFILTER_SMALL_BUDGET_ENV,
    CUDA_PREFILTER_VERIFY_ENV,
    CUDA_SHAPE_ENV,
    NATIVE_BATCH_ENV,
    NATIVE_BATCH_EXACT_VERIFY_ENV,
    NATIVE_BATCH_THREADS_ENV,
    NATIVE_DP_ENV,
)


def _build_dense_edge_array(
    predecessor_starts: list[int], state_count: int
) -> np.ndarray:
    """Construct the constant-state DP graph without Python tuple objects."""
    states = int(state_count)
    if states < 1:
        raise ValueError("state_count must be positive")
    state_pairs = states * states
    edge_count = sum(
        (end_pos - int(predecessor_starts[end_pos])) * state_pairs
        for end_pos in range(1, len(predecessor_starts))
    )
    edges = np.empty((edge_count, 4), dtype=np.int32)
    pair_start_states = np.repeat(np.arange(states, dtype=np.int32), states)
    pair_end_states = np.tile(np.arange(states, dtype=np.int32), states)
    offset = 0
    for end_pos in range(1, len(predecessor_starts)):
        first = int(predecessor_starts[end_pos])
        predecessor_count = int(end_pos - first)
        if predecessor_count <= 0:
            continue
        count = predecessor_count * state_pairs
        target = edges[offset : offset + count]
        target[:, 0] = np.repeat(np.arange(first, end_pos, dtype=np.int32), state_pairs)
        target[:, 1] = np.tile(pair_start_states, predecessor_count)
        target[:, 2] = int(end_pos)
        target[:, 3] = np.tile(pair_end_states, predecessor_count)
        offset += count
    if offset != edge_count:
        raise RuntimeError(
            f"dense edge construction mismatch: {offset} != {edge_count}"
        )
    return edges


def build_hard_multistate_penalty_path(module):
    def run_hard_multistate_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        runtime_args,
        eval_contexts=None,
    ):
        """Penalty DP on the exact hard-Recall feasible multistate graph.

        Production's generic multistate solver performs an outer binary search
        over a soft Recall multiplier.  Phase 2 has no soft Recall trade-off:
        ``phase1_runtime.interval_cost_from_vectors`` already maps every
        violating edge to +inf.  Removing that redundant multiplier search is
        both semantically exact and substantially faster.
        """

        if all(len(values) == 1 for values in candidates_by_frame):
            return module.run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                runtime_args,
                eval_contexts=eval_contexts,
            )
        frames = [int(value) for value in candidate_frames]
        if frames != list(range(len(run.frame_numbers))):
            raise RuntimeError("Phase 2 candidate pool must remain dense")
        node_count = len(frames)
        if node_count == 1:
            feasible = [
                (float(candidate.frame_loss), int(state))
                for state, candidate in enumerate(candidates_by_frame[frames[0]])
                if float(candidate.recall_budget) <= _EPSILON
            ]
            state = min(feasible)[1] if feasible else 0
            return (
                [frames[0]],
                [state],
                {"interval_evals": 0, "interval_frames": 0},
                {},
                0.0,
            )
        target_interval = max(
            1, int(round(1.0 / max(float(runtime_args.target_ratio), 1e-6)))
        )
        dynamic_max_gap = max(
            int(runtime_args.max_gap),
            int(
                math.ceil(
                    float(runtime_args.dynamic_max_gap_factor) * float(target_interval)
                )
            ),
        )
        predecessor_starts = [0] * node_count
        for end in range(1, node_count):
            predecessor_starts[end] = int(
                bisect.bisect_left(frames, frames[end] - dynamic_max_gap, 0, end)
            )
        edge_cache: dict[tuple[int, int, int, int], object] = {}
        counters = {"interval_evals": 0, "interval_frames": 0}
        native_batch_cache: dict[tuple[int, int, int, int], tuple[object, float]] = {}
        native_batch_profile: dict[str, object] = {
            "enabled": False,
            "threads": 0,
            "precomputed_edges": 0,
            "precompute_seconds": 0.0,
            "used_exact_failures": 0,
        }
        native_batch_requested = os.environ.get(NATIVE_BATCH_ENV, "").strip() == "1"
        native_dp_requested = os.environ.get(NATIVE_DP_ENV, "").strip() == "1"
        native_metrics = None
        native_edge_array = None
        native_edge_costs = None
        native_decode_edge_array = None
        native_decode_edge_costs = None
        native_decode_indices = None
        native_initial_losses = None
        native_incremental_decoder = None
        lazy_exact_enabled = False
        lazy_exact_requested = False
        cuda_approx_only_requested = False
        lazy_exact_verified = None
        lazy_exact_edge_offsets = None
        lazy_exact_candidate_vectors = None
        lazy_exact_evaluator = None
        lazy_exact_threads = 1
        lazy_exact_parameters = None
        lazy_exact_started = None
        lazy_dense_costs_loaded = False
        lazy_state_pair_batch = (
            os.environ.get(CUDA_LAZY_STATE_PAIR_BATCH_ENV, "0").strip() == "1"
        )
        lazy_exact_max_seconds = max(
            0.0, float(os.environ.get(CUDA_LAZY_MAX_SECONDS_ENV, "0"))
        )
        lazy_fallback_min_seconds = max(
            0.0,
            float(os.environ.get(CUDA_LAZY_FALLBACK_MIN_SECONDS_ENV, "5.0")),
        )
        lazy_fallback_min_edges = max(
            1,
            int(os.environ.get(CUDA_LAZY_FALLBACK_MIN_EDGES_ENV, "1000")),
        )
        lazy_fallback_infeasible_ratio = min(
            1.0,
            max(
                0.0,
                float(os.environ.get(CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO_ENV, "0.90")),
            ),
        )
        native_batch_exact_verify = (
            os.environ.get(NATIVE_BATCH_EXACT_VERIFY_ENV, "").strip() == "1"
        )
        if native_batch_requested:
            if not bool(getattr(module, "_phase1_native_interval_enabled", False)):
                raise RuntimeError(
                    f"{NATIVE_BATCH_ENV}=1 requires MASK_PIPELINE_PHASE1_NATIVE_INTERVAL=1"
                )
            state_counts = [len(values) for values in candidates_by_frame]
            if len(set(state_counts)) != 1:
                raise RuntimeError(
                    "native batch currently requires a constant candidate-state count"
                )
            edge_build_started = time.perf_counter()
            edge_array = _build_dense_edge_array(predecessor_starts, state_counts[0])
            edge_build_seconds = time.perf_counter() - edge_build_started
            candidate_stack_started = time.perf_counter()
            candidate_vectors = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(candidate.vector, dtype=np.float32)
                            for candidate in values
                        ],
                        axis=0,
                    )
                    for values in candidates_by_frame
                ],
                axis=0,
            )
            candidate_stack_seconds = time.perf_counter() - candidate_stack_started
            threads = max(
                1,
                int(
                    os.environ.get(
                        NATIVE_BATCH_THREADS_ENV,
                        str(min(8, os.cpu_count() or 1)),
                    )
                ),
            )
            evaluator = module._phase1_get_native_interval_evaluator(
                eval_contexts, run.gt_polygons
            )
            native_metrics = sys.modules.get("native_interval_metrics")
            if native_metrics is None:
                raise RuntimeError(
                    "native_interval_metrics disappeared after initialization"
                )
            batch_started = time.perf_counter()
            cuda_prefilter_profile: dict[str, object] = {"enabled": False}
            batch_edge_array = edge_array
            retained_indices = None
            cuda_recall_deficit = None
            cuda_recall_hint_frames = None
            cuda_exact_hint_requested = (
                os.environ.get(CUDA_EXACT_HINT_ENV, "").strip() == "1"
            )
            # Frame hints are intentionally opt-in for the all-edge exact path.
            # Adding the same CUDA pass to the already-pruned lazy path preserved
            # every output but slowed the full KPI workload, so the established
            # lazy mode must not pay that transfer/launch overhead by default.
            cuda_return_frame_hints = cuda_exact_hint_requested
            cuda_prefilter_verify = (
                os.environ.get(CUDA_PREFILTER_VERIFY_ENV, "").strip() == "1"
            )
            if os.environ.get(CUDA_PREFILTER_ENV, "").strip() == "1":
                from cuda_interval_raster import evaluate_cached_intervals

                requested_prefilter_budget = max(
                    0.0,
                    float(os.environ.get(CUDA_PREFILTER_BUDGET_ENV, "0.10")),
                )
                small_area_threshold = max(
                    0.0,
                    float(os.environ.get(CUDA_PREFILTER_SMALL_AREA_ENV, "0")),
                )
                small_area_budget = max(
                    requested_prefilter_budget,
                    float(
                        os.environ.get(
                            CUDA_PREFILTER_SMALL_BUDGET_ENV,
                            str(requested_prefilter_budget),
                        )
                    ),
                )
                reference_areas = [
                    sum(
                        abs(
                            float(
                                module.cv2.contourArea(
                                    np.asarray(polygon, dtype=np.float32)
                                )
                            )
                        )
                        for polygon in frame_polygons
                        if len(polygon) >= 3
                    )
                    for frame_polygons in run.gt_polygons
                ]
                median_reference_area = (
                    float(np.median(np.asarray(reference_areas, dtype=np.float64)))
                    if reference_areas
                    else 0.0
                )
                prefilter_budget = (
                    small_area_budget
                    if (
                        small_area_threshold > 0.0
                        and median_reference_area < small_area_threshold
                    )
                    else requested_prefilter_budget
                )
                cuda_result = evaluate_cached_intervals(
                    candidate_vectors,
                    edge_array,
                    eval_contexts,
                    iou_weight=float(runtime_args.interval_iou_weight),
                    recall_floor=float(runtime_args.recall_min),
                    return_frame_hints=bool(cuda_return_frame_hints),
                    recall_hint_count=max(
                        1,
                        min(
                            8,
                            int(os.environ.get(CUDA_EXACT_HINT_COUNT_ENV, "8")),
                        ),
                    ),
                )
                if cuda_return_frame_hints:
                    (
                        _cuda_loss,
                        cuda_recall_deficit,
                        cuda_recall_hint_frames,
                        _cuda_covered,
                        cuda_prefilter_details,
                    ) = cuda_result
                else:
                    (
                        _cuda_loss,
                        cuda_recall_deficit,
                        _cuda_covered,
                        cuda_prefilter_details,
                    ) = cuda_result
                screened_indices = np.flatnonzero(
                    np.asarray(cuda_recall_deficit) <= prefilter_budget
                )
                if not cuda_exact_hint_requested:
                    retained_indices = screened_indices
                if not cuda_prefilter_verify and not cuda_exact_hint_requested:
                    batch_edge_array = np.ascontiguousarray(
                        edge_array[retained_indices], dtype=np.int32
                    )
                cuda_prefilter_profile = {
                    **cuda_prefilter_details,
                    "enabled": True,
                    "deficit_budget": float(prefilter_budget),
                    "requested_deficit_budget": float(requested_prefilter_budget),
                    "small_area_threshold": float(small_area_threshold),
                    "small_area_budget": float(small_area_budget),
                    "median_reference_area": float(median_reference_area),
                    "retained_edges": int(len(screened_indices)),
                    "rejected_edges": int(len(edge_array) - len(screened_indices)),
                    "retained_ratio": float(
                        len(screened_indices) / max(len(edge_array), 1)
                    ),
                    "verification_mode": bool(cuda_prefilter_verify),
                    "hint_only_mode": bool(cuda_exact_hint_requested),
                }
            lazy_exact_requested = (
                os.environ.get(CUDA_LAZY_EXACT_ENV, "").strip() == "1"
            )
            cuda_approx_only_requested = (
                os.environ.get(CUDA_APPROX_ONLY_ENV, "").strip() == "1"
            )
            if lazy_exact_requested and cuda_approx_only_requested:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 and {CUDA_APPROX_ONLY_ENV}=1 "
                    "are mutually exclusive"
                )
            if cuda_exact_hint_requested and (
                lazy_exact_requested or cuda_approx_only_requested
            ):
                raise RuntimeError(
                    f"{CUDA_EXACT_HINT_ENV}=1 is mutually exclusive with "
                    f"{CUDA_LAZY_EXACT_ENV}=1 and {CUDA_APPROX_ONLY_ENV}=1"
                )
            lazy_exact_enabled = bool(lazy_exact_requested)
            lazy_min_retained_ratio = max(
                0.0,
                min(
                    1.0,
                    float(os.environ.get(CUDA_LAZY_MIN_RETAINED_RATIO_ENV, "0.60")),
                ),
            )
            retained_ratio = (
                float(len(retained_indices) / max(len(edge_array), 1))
                if retained_indices is not None
                else 1.0
            )
            lazy_auto_disabled_reason = None
            if lazy_exact_enabled and retained_ratio < lazy_min_retained_ratio:
                # A low retained ratio predicts that the approximate graph is
                # dominated by Recall failures. In that regime lazy DP tends
                # to churn through rejected paths, while the dense exact batch
                # is both faster and byte-stable.
                lazy_exact_enabled = False
                lazy_auto_disabled_reason = "low_cuda_retained_ratio"
            if lazy_exact_enabled and cuda_recall_deficit is None:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 requires {CUDA_PREFILTER_ENV}=1"
                )
            if cuda_approx_only_requested and cuda_recall_deficit is None:
                raise RuntimeError(
                    f"{CUDA_APPROX_ONLY_ENV}=1 requires {CUDA_PREFILTER_ENV}=1"
                )
            if lazy_exact_enabled and not native_dp_requested:
                raise RuntimeError(
                    f"{CUDA_LAZY_EXACT_ENV}=1 requires {NATIVE_DP_ENV}=1"
                )
            if cuda_approx_only_requested and not native_dp_requested:
                raise RuntimeError(
                    f"{CUDA_APPROX_ONLY_ENV}=1 requires {NATIVE_DP_ENV}=1"
                )
            cuda_shape_profile: dict[str, object] = {"enabled": False}
            precomputed_shape_distances = None
            use_cuda_shape = os.environ.get(CUDA_SHAPE_ENV, "").strip() == "1" and not (
                cuda_exact_hint_requested
                and not lazy_exact_requested
                and not cuda_approx_only_requested
            )
            if use_cuda_shape:
                from cuda_shape_distance import compute_shape_distances

                # Approximate/lazy CUDA constructs costs for the complete
                # graph. A prefilter may make batch_edge_array smaller, so
                # using it here would produce an incompatible shape vector.
                shape_edge_array = (
                    edge_array
                    if lazy_exact_requested or cuda_approx_only_requested
                    else batch_edge_array
                )
                (
                    precomputed_shape_distances,
                    cuda_shape_details,
                ) = compute_shape_distances(
                    candidate_vectors,
                    shape_edge_array,
                    float(run.scale),
                )
                cuda_shape_profile = {
                    "enabled": True,
                    **cuda_shape_details,
                }
            evaluation_parameters = (
                int(run.contour_count),
                int(run.anchors_per_contour),
                float(runtime_args.interval_iou_weight),
                float(runtime_args.recall_min),
                float(run.scale),
                float(runtime_args.shape_update_threshold_ratio),
                float(runtime_args.shape_switch_weight),
                float(runtime_args.shape_distance_weight),
                float(runtime_args.shape_penalty_adapt_gain),
                float(runtime_args.shape_distance_relief),
                float(runtime_args.shape_switch_relief),
                float(runtime_args.shape_distance_min_scale),
                float(runtime_args.shape_switch_min_scale),
            )
            if lazy_exact_enabled or cuda_approx_only_requested:
                # CUDA supplies dense approximate raster losses.  Only edges
                # that enter a candidate DP path are subsequently evaluated by
                # the exact OpenCV engine.  This keeps the hard Recall contract
                # on every accepted path without rasterizing the entire dense
                # graph on the CPU.
                if precomputed_shape_distances is None:
                    from cuda_shape_distance import compute_shape_distances

                    (
                        precomputed_shape_distances,
                        cuda_shape_details,
                    ) = compute_shape_distances(
                        candidate_vectors,
                        edge_array,
                        float(run.scale),
                    )
                    cuda_shape_profile = {
                        "enabled": True,
                        "implicit_for_lazy_exact": True,
                        **cuda_shape_details,
                    }
                distance = np.asarray(precomputed_shape_distances, dtype=np.float64)
                cuda_frame_loss = np.asarray(_cuda_loss, dtype=np.float64)
                covered = np.asarray(_cuda_covered, dtype=np.float64)
                frame_loss_mean = cuda_frame_loss / np.maximum(covered, 1.0)
                base = 1.0 + max(
                    float(runtime_args.shape_penalty_adapt_gain), 0.0
                ) * np.maximum(frame_loss_mean, 0.0)
                distance_scale = np.maximum(
                    float(runtime_args.shape_distance_min_scale),
                    1.0
                    / np.maximum(
                        np.power(
                            base, max(float(runtime_args.shape_distance_relief), 0.0)
                        ),
                        1e-6,
                    ),
                )
                switch_scale = np.maximum(
                    float(runtime_args.shape_switch_min_scale),
                    1.0
                    / np.maximum(
                        np.power(
                            base, max(float(runtime_args.shape_switch_relief), 0.0)
                        ),
                        1e-6,
                    ),
                )
                update = (
                    distance > float(runtime_args.shape_update_threshold_ratio)
                ).astype(np.float64)
                approximate_cost = (
                    cuda_frame_loss
                    + float(runtime_args.shape_switch_weight) * switch_scale * update
                    + float(runtime_args.shape_distance_weight)
                    * distance_scale
                    * distance
                )
                lazy_deficit_penalty = max(
                    0.0,
                    float(os.environ.get(CUDA_LAZY_DEFICIT_PENALTY_ENV, "0")),
                )
                approximate_cost += lazy_deficit_penalty * np.asarray(
                    cuda_recall_deficit, dtype=np.float64
                )
                batch_array = np.zeros((len(edge_array), 9), dtype=np.float64)
                batch_array[:, 0] = approximate_cost
                batch_array[:, 1] = distance
                batch_array[:, 2] = update
                batch_array[:, 3] = covered
                batch_array[:, 4] = frame_loss_mean
                batch_array[:, 5] = distance_scale
                batch_array[:, 6] = switch_scale
                cuda_rejected = np.asarray(cuda_recall_deficit) > float(
                    cuda_prefilter_profile["deficit_budget"]
                )
                batch_array[cuda_rejected, 0] = np.inf
                batch_array[cuda_rejected, 7] = 1.0
                batch_values = batch_array
                if lazy_exact_enabled:
                    lazy_exact_verified = np.asarray(cuda_rejected, dtype=bool).copy()
                    lazy_exact_edge_offsets = np.zeros((node_count,), dtype=np.int64)
                    running_edge_offset = 0
                    for end_pos in range(1, node_count):
                        lazy_exact_edge_offsets[end_pos] = running_edge_offset
                        running_edge_offset += (
                            (end_pos - predecessor_starts[end_pos])
                            * state_counts[end_pos]
                            * state_counts[end_pos]
                        )
                    lazy_exact_candidate_vectors = candidate_vectors
                    lazy_exact_evaluator = evaluator
                    lazy_exact_threads = int(threads)
                    lazy_exact_parameters = evaluation_parameters
            else:
                batch_values = evaluator.evaluate_edge_batch(
                    candidate_vectors,
                    batch_edge_array,
                    *evaluation_parameters,
                    int(threads),
                    precomputed_shape_distances,
                    True,
                    False,
                    bool(cuda_exact_hint_requested),
                    cuda_recall_hint_frames,
                )
            evaluated_batch_array = np.asarray(batch_values)
            if cuda_approx_only_requested:
                # Approximation-only benchmarking deliberately accepts the
                # CUDA graph as final.  The normal post-run exact audit remains
                # enabled so quality and hard-Recall drift are measured rather
                # than hidden.
                batch_array = evaluated_batch_array
            elif not lazy_exact_enabled:
                if retained_indices is None or cuda_prefilter_verify:
                    batch_array = evaluated_batch_array
                else:
                    batch_array = np.zeros((len(edge_array), 9), dtype=np.float64)
                    # Rejected edges are hard-infeasible. Only cost and the two
                    # Recall columns affect graph membership for these rows.
                    batch_array[:, 0] = np.inf
                    batch_array[:, 7] = 1.0
                    batch_array[retained_indices] = evaluated_batch_array
            if cuda_prefilter_verify and cuda_recall_deficit is not None:
                cuda_rejected = np.asarray(cuda_recall_deficit) > float(
                    cuda_prefilter_profile["deficit_budget"]
                )
                cpu_feasible = (batch_array[:, 7] <= _EPSILON) & (
                    batch_array[:, 8] <= _EPSILON
                )
                false_rejected = np.flatnonzero(cuda_rejected & cpu_feasible)
                cuda_prefilter_profile["false_rejected_feasible_edges"] = int(
                    len(false_rejected)
                )
                if len(false_rejected):
                    false_deficits = np.asarray(cuda_recall_deficit)[false_rejected]
                    cuda_prefilter_profile["false_rejected_deficit_min"] = float(
                        np.min(false_deficits)
                    )
                    cuda_prefilter_profile["false_rejected_deficit_max"] = float(
                        np.max(false_deficits)
                    )
                    cuda_prefilter_profile["false_rejected_deficit_quantiles"] = [
                        float(value)
                        for value in np.quantile(false_deficits, [0.25, 0.5, 0.75])
                    ]
                cuda_prefilter_profile["false_rejected_examples"] = [
                    [int(value) for value in edge_array[index].tolist()]
                    for index in false_rejected[:12]
                ]
                cuda_loss_array = np.asarray(_cuda_loss, dtype=np.float64)
                cuda_deficit_array = np.asarray(cuda_recall_deficit, dtype=np.float64)
                cpu_cached_loss = np.asarray(
                    batch_array[:, 4], dtype=np.float64
                ) * np.asarray(batch_array[:, 3], dtype=np.float64)
                cpu_cached_deficit = np.asarray(batch_array[:, 7], dtype=np.float64)
                frames_covered_array = np.maximum(
                    np.asarray(batch_array[:, 3], dtype=np.float64), 1.0
                )

                def error_summary(values: np.ndarray) -> dict[str, object]:
                    finite = np.asarray(values, dtype=np.float64)
                    finite = finite[np.isfinite(finite)]
                    if not len(finite):
                        return {"count": 0}
                    return {
                        "count": int(len(finite)),
                        "mean": float(np.mean(finite)),
                        "q50": float(np.quantile(finite, 0.50)),
                        "q90": float(np.quantile(finite, 0.90)),
                        "q95": float(np.quantile(finite, 0.95)),
                        "q99": float(np.quantile(finite, 0.99)),
                        "q999": float(np.quantile(finite, 0.999)),
                        "max": float(np.max(finite)),
                    }

                loss_abs_error = np.abs(cuda_loss_array - cpu_cached_loss)
                deficit_abs_error = np.abs(cuda_deficit_array - cpu_cached_deficit)
                cuda_prefilter_profile["numeric_error"] = {
                    "interval_iou_loss_absolute": error_summary(loss_abs_error),
                    "mean_frame_iou_loss_absolute": error_summary(
                        loss_abs_error / frames_covered_array
                    ),
                    "recall_deficit_sum_absolute": error_summary(deficit_abs_error),
                    "mean_frame_recall_deficit_absolute": error_summary(
                        deficit_abs_error / frames_covered_array
                    ),
                }
                exact_feasible = (batch_array[:, 7] <= _EPSILON) & (
                    batch_array[:, 8] <= _EPSILON
                )
                budget_audit = {}
                for audit_budget in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25):
                    cuda_keep = cuda_deficit_array <= float(audit_budget)
                    budget_audit[f"{audit_budget:.2f}"] = {
                        "retained_edges": int(np.count_nonzero(cuda_keep)),
                        "retained_ratio": float(np.mean(cuda_keep)),
                        "false_rejected_exact_feasible": int(
                            np.count_nonzero((~cuda_keep) & exact_feasible)
                        ),
                        "retained_exact_infeasible": int(
                            np.count_nonzero(cuda_keep & (~exact_feasible))
                        ),
                    }
                cuda_prefilter_profile["budget_audit"] = budget_audit
            if native_dp_requested:
                native_edge_array = edge_array
                native_edge_costs = np.asarray(
                    batch_array[:, 0], dtype=np.float64
                ).copy()
                native_edge_costs[
                    (batch_array[:, 7] > _EPSILON) | (batch_array[:, 8] > _EPSILON)
                ] = np.inf
                native_initial_losses = np.asarray(
                    [
                        float(candidate.frame_loss)
                        if float(candidate.recall_budget) <= _EPSILON
                        else np.inf
                        for candidate in candidates_by_frame[frames[0]]
                    ],
                    dtype=np.float64,
                )
                if retained_indices is not None and not cuda_prefilter_verify:
                    native_decode_indices = np.asarray(
                        retained_indices, dtype=np.int64
                    ).copy()
                    native_decode_edge_array = np.ascontiguousarray(
                        native_edge_array[native_decode_indices], dtype=np.int32
                    )
                    native_decode_edge_costs = np.ascontiguousarray(
                        native_edge_costs[native_decode_indices], dtype=np.float64
                    )
                else:
                    native_decode_edge_array = native_edge_array
                    native_decode_edge_costs = native_edge_costs
                if hasattr(native_metrics, "IncrementalPenaltyPathDecoder"):
                    native_incremental_decoder = (
                        native_metrics.IncrementalPenaltyPathDecoder(
                            native_decode_edge_array,
                            native_initial_losses,
                            int(node_count),
                            int(len(candidates_by_frame[frames[0]])),
                        )
                    )
            else:
                for edge, values in zip(edge_array, batch_array):
                    key = tuple(int(value) for value in edge)
                    native_batch_cache[key] = (
                        module.IntervalCost(
                            cost=float(values[0]),
                            shape_distance=float(values[1]),
                            shape_update=float(values[2]),
                            frames_covered=int(values[3]),
                            frame_loss_mean=float(values[4]),
                            shape_distance_scale=float(values[5]),
                            shape_switch_scale=float(values[6]),
                            recall_budget=float(values[7]),
                        ),
                        float(values[8]),
                    )
            native_batch_profile = {
                "enabled": True,
                "threads": int(threads),
                "native_dp": bool(native_dp_requested),
                "incremental_native_dp": bool(native_incremental_decoder is not None),
                "cuda_lazy_exact": {
                    "requested": bool(lazy_exact_requested),
                    "enabled": bool(lazy_exact_enabled),
                    "auto_disabled_reason": lazy_auto_disabled_reason,
                    "minimum_retained_ratio": float(lazy_min_retained_ratio),
                    "max_seconds_before_dense_fallback": float(lazy_exact_max_seconds),
                    "fallback_min_seconds": float(lazy_fallback_min_seconds),
                    "fallback_min_exact_edges": int(lazy_fallback_min_edges),
                    "fallback_infeasible_ratio": float(lazy_fallback_infeasible_ratio),
                    "approximate_deficit_penalty": (
                        float(lazy_deficit_penalty) if lazy_exact_enabled else 0.0
                    ),
                    "exact_edges": 0,
                    "exact_batches": 0,
                    "decode_retries": 0,
                    "frame_hints_enabled": bool(
                        lazy_exact_enabled and cuda_recall_hint_frames is not None
                    ),
                    "frame_hint_count": int(
                        cuda_recall_hint_frames.shape[1]
                        if (
                            cuda_recall_hint_frames is not None
                            and np.asarray(cuda_recall_hint_frames).ndim == 2
                        )
                        else (1 if cuda_recall_hint_frames is not None else 0)
                    ),
                },
                "cuda_approx_only": {
                    "requested": bool(cuda_approx_only_requested),
                    "enabled": bool(cuda_approx_only_requested),
                    "exact_edge_validation": False
                    if cuda_approx_only_requested
                    else None,
                },
                "cuda_exact_hint": {
                    "requested": bool(cuda_exact_hint_requested),
                    "enabled": bool(cuda_exact_hint_requested),
                    "filtered_edges": 0,
                    "hinted_edges": int(len(cuda_recall_hint_frames))
                    if cuda_recall_hint_frames is not None
                    else 0,
                },
                "cuda_shape": cuda_shape_profile,
                "cuda_prefilter": cuda_prefilter_profile,
                "precomputed_edges": int(len(edge_array)),
                "decode_edges": int(
                    len(native_decode_edge_array)
                    if native_decode_edge_array is not None
                    else len(edge_array)
                ),
                "decode_edge_ratio": float(
                    len(native_decode_edge_array) / max(len(edge_array), 1)
                    if native_decode_edge_array is not None
                    else 1.0
                ),
                "precompute_seconds": float(time.perf_counter() - batch_started),
                "edge_build_seconds": float(edge_build_seconds),
                "candidate_stack_seconds": float(candidate_stack_seconds),
                "context_statistics": dict(evaluator.context_statistics()),
                "cached_failures_precomputed": int(
                    np.count_nonzero(np.asarray(batch_values)[:, 7] > _EPSILON)
                ),
                "exact_failures_precomputed": int(
                    np.count_nonzero(np.asarray(batch_values)[:, 8] > _EPSILON)
                ),
                "used_exact_failures": 0,
                "exact_verify_edges": 0,
                "exact_verify_classification_mismatches": 0,
                "exact_verify_examples": [],
            }
            if lazy_exact_enabled:
                lazy_exact_started = time.perf_counter()

        def edge(start_pos: int, start_state: int, end_pos: int, end_state: int):
            key = (
                int(start_pos),
                int(start_state),
                int(end_pos),
                int(end_state),
            )
            value = edge_cache.get(key)
            if value is not None:
                return value
            start_frame = frames[start_pos]
            end_frame = frames[end_pos]
            left = candidates_by_frame[start_frame][start_state]
            right = candidates_by_frame[end_frame][end_state]
            precomputed_entry = native_batch_cache.get(key)
            precomputed = (
                precomputed_entry[0] if precomputed_entry is not None else None
            )
            precomputed_exact_deficit = (
                precomputed_entry[1] if precomputed_entry is not None else None
            )
            if (
                precomputed_exact_deficit is not None
                and float(precomputed_exact_deficit) > _EPSILON
            ):
                native_batch_profile["used_exact_failures"] = (
                    int(native_batch_profile["used_exact_failures"]) + 1
                )
            if (
                native_batch_exact_verify
                and precomputed_entry is not None
                and float(precomputed.recall_budget) <= _EPSILON
            ):
                reference_deficit = 0.0
                for frame_idx in range(start_frame + 1, end_frame + 1):
                    if frame_idx == end_frame:
                        polygons = right.polygons
                    else:
                        alpha = float(
                            (frame_idx - start_frame) / max(end_frame - start_frame, 1)
                        )
                        polygons = module.split_vector_to_polygons(
                            module.interpolate_vectors(
                                left.vector, right.vector, alpha
                            ),
                            run.contour_count,
                            run.anchors_per_contour,
                        )
                    metrics = module.compute_exact_metrics_from_polygons(
                        run.gt_polygons[frame_idx], polygons
                    )
                    reference_deficit += max(
                        float(runtime_args.recall_min) - float(metrics["recall"]),
                        0.0,
                    )
                    if reference_deficit > _EPSILON:
                        break
                native_batch_profile["exact_verify_edges"] = (
                    int(native_batch_profile["exact_verify_edges"]) + 1
                )
                reference_fails = reference_deficit > _EPSILON
                batch_fails = float(precomputed_exact_deficit) > _EPSILON
                if reference_fails != batch_fails:
                    native_batch_profile["exact_verify_classification_mismatches"] = (
                        int(
                            native_batch_profile[
                                "exact_verify_classification_mismatches"
                            ]
                        )
                        + 1
                    )
                    examples = native_batch_profile["exact_verify_examples"]
                    if len(examples) < 12:
                        examples.append(
                            {
                                "key": list(key),
                                "reference_deficit": float(reference_deficit),
                                "batch_deficit": float(precomputed_exact_deficit),
                            }
                        )
            value = module.interval_cost_from_vectors(
                run,
                start_frame,
                left.vector,
                end_frame,
                right.vector,
                runtime_args,
                include_start=False,
                eval_contexts=eval_contexts,
                start_candidate=left,
                end_candidate=right,
                precomputed_interval_info=precomputed,
                precomputed_exact_deficit=precomputed_exact_deficit,
            )
            edge_cache[key] = value
            counters["interval_evals"] += 1
            counters["interval_frames"] += int(value.frames_covered)
            return value

        decoded: dict[
            tuple[int, float, tuple[int, ...]],
            tuple[list[int], list[int], float, float],
        ] = {}
        decode_profiles: list[dict[str, float | int]] = []

        def record_decode(started: float, cache_before: int) -> None:
            decode_profiles.append(
                {
                    "seconds": float(time.perf_counter() - started),
                    "cache_before": int(cache_before),
                    "cache_after": int(len(edge_cache)),
                }
            )

        def emit_solver_profile() -> None:
            first = decode_profiles[0] if decode_profiles else None
            cached = [
                row
                for row in decode_profiles
                if int(row["cache_before"]) == int(row["cache_after"])
            ]
            payload = {
                "stream_id": str(run.stream_id),
                "frames": int(node_count),
                "mean_state_count": float(
                    np.mean([len(values) for values in candidates_by_frame])
                ),
                "decode_calls": int(len(decode_profiles)),
                "decode_seconds": float(
                    sum(float(row["seconds"]) for row in decode_profiles)
                ),
                "first_decode_seconds": (
                    float(first["seconds"]) if first is not None else 0.0
                ),
                "cached_decode_calls": int(len(cached)),
                "cached_decode_seconds": float(
                    sum(float(row["seconds"]) for row in cached)
                ),
                "cached_decode_mean_seconds": (
                    float(sum(float(row["seconds"]) for row in cached) / len(cached))
                    if cached
                    else 0.0
                ),
                "edge_cache_entries": int(len(edge_cache)),
                "interval_evaluations": int(counters["interval_evals"]),
                "interval_evaluation_frames": int(counters["interval_frames"]),
                "interval_kernel": dict(
                    getattr(module, "_phase1_interval_profile", {})
                ),
                "native_batch": dict(native_batch_profile),
            }
            print(
                "[phase2-dp-profile] "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

        def lazy_edge_index(
            start_pos: int,
            start_state: int,
            end_pos: int,
            end_state: int,
        ) -> int:
            if lazy_exact_edge_offsets is None:
                raise RuntimeError("lazy exact edge offsets are unavailable")
            state_count = len(candidates_by_frame[frames[0]])
            start_offset = int(start_pos) - int(predecessor_starts[end_pos])
            if start_offset < 0:
                raise RuntimeError(
                    "selected edge is outside the dense predecessor graph"
                )
            return int(
                lazy_exact_edge_offsets[end_pos]
                + start_offset * state_count * state_count
                + int(start_state) * state_count
                + int(end_state)
            )

        def fallback_lazy_to_dense_exact(reason: str) -> None:
            nonlocal lazy_dense_costs_loaded
            if not lazy_exact_enabled:
                return
            if (
                lazy_exact_candidate_vectors is None
                or lazy_exact_evaluator is None
                or lazy_exact_parameters is None
                or native_edge_array is None
                or native_edge_costs is None
                or native_decode_edge_costs is None
                or lazy_exact_verified is None
            ):
                raise RuntimeError("lazy exact dense fallback was not initialized")
            fallback_started = time.perf_counter()
            if retained_indices is None:
                dense_indices = np.arange(len(native_edge_array), dtype=np.int64)
            else:
                dense_indices = np.asarray(retained_indices, dtype=np.int64)
            # Some retained edges may already have been evaluated with the
            # pair-dependent exact Recall rasterizer before the dense-cost
            # fallback is triggered.  Preserve those decisions.  In
            # particular, an exact-infeasible edge must never be resurrected
            # by the cheaper cached-Recall pass below.
            previously_exact = np.asarray(
                lazy_exact_verified[dense_indices], dtype=bool
            ).copy()
            previous_exact_costs = np.asarray(
                native_edge_costs[dense_indices], dtype=np.float64
            ).copy()
            dense_edges = np.ascontiguousarray(
                native_edge_array[dense_indices], dtype=np.int32
            )
            dense_values = np.asarray(
                lazy_exact_evaluator.evaluate_edge_batch(
                    lazy_exact_candidate_vectors,
                    dense_edges,
                    *lazy_exact_parameters,
                    int(lazy_exact_threads),
                    None,
                    True,
                    True,
                )
            )
            dense_costs = np.asarray(dense_values[:, 0], dtype=np.float64)
            dense_costs[dense_values[:, 7] > _EPSILON] = np.inf
            dense_costs[previously_exact] = previous_exact_costs[previously_exact]
            native_edge_costs[:] = np.inf
            native_edge_costs[dense_indices] = dense_costs
            if native_decode_indices is not None:
                native_decode_edge_costs[:] = native_edge_costs[native_decode_indices]
            # The complete retained graph now has native objective costs.
            # Cached-Recall failures are final; cached-feasible edges still
            # require the pair-dependent exact Recall check if DP selects
            # them.  Repeated shortest-path validation therefore reaches the
            # same optimum as eager dense exact Recall without rasterizing
            # every feasible edge twice.
            cached_infeasible = np.asarray(dense_values[:, 7]) > _EPSILON
            lazy_exact_verified[dense_indices] = cached_infeasible | previously_exact
            lazy_dense_costs_loaded = True
            lazy_profile = native_batch_profile["cuda_lazy_exact"]
            lazy_profile["fallback_dense_exact"] = False
            lazy_profile["fallback_dense_native_costs"] = True
            lazy_profile["fallback_exact_recall_mode"] = "selected_paths"
            lazy_profile["fallback_reason"] = str(reason)
            lazy_profile["fallback_edges"] = int(len(dense_indices))
            lazy_profile["fallback_seconds"] = float(
                time.perf_counter() - fallback_started
            )
            lazy_profile["fallback_infeasible_edges"] = int(
                np.count_nonzero(~np.isfinite(dense_costs))
            )
            lazy_profile["fallback_cached_infeasible_edges"] = int(
                np.count_nonzero(cached_infeasible)
            )
            lazy_profile["fallback_preserved_exact_edges"] = int(
                np.count_nonzero(previously_exact)
            )
            lazy_profile["fallback_preserved_exact_infeasible_edges"] = int(
                np.count_nonzero(previously_exact & ~np.isfinite(previous_exact_costs))
            )
            lazy_profile["fallback_exact_pending_edges"] = int(
                np.count_nonzero(~(cached_infeasible | previously_exact))
            )

        def exactify_lazy_path(
            positions: list[int], states: list[int]
        ) -> tuple[int, int]:
            if not lazy_exact_enabled:
                return 0, int(node_count)
            if (
                lazy_exact_verified is None
                or lazy_exact_candidate_vectors is None
                or lazy_exact_evaluator is None
                or lazy_exact_parameters is None
                or native_edge_array is None
                or native_edge_costs is None
                or native_decode_edge_costs is None
            ):
                raise RuntimeError("lazy exact runtime was not initialized")
            selected_path_indices = np.asarray(
                [
                    lazy_edge_index(
                        positions[index - 1],
                        states[index - 1],
                        positions[index],
                        states[index],
                    )
                    for index in range(1, len(positions))
                ],
                dtype=np.int64,
            )
            if lazy_state_pair_batch and len(selected_path_indices):
                state_count = len(candidates_by_frame[frames[0]])
                state_pairs = state_count * state_count
                # Every state pair over a selected frame interval shares the
                # same reference-frame span.  Validate the complete small
                # block in one native batch so near-identical alternative
                # paths do not force another full-graph DP scan one edge at a
                # time.  This changes evaluation order only; costs, Recall,
                # candidates, and tie-breaking remain untouched.
                selected_indices = np.concatenate(
                    [
                        np.arange(
                            int(edge_index)
                            - (
                                int(states[path_index - 1]) * state_count
                                + int(states[path_index])
                            ),
                            int(edge_index)
                            - (
                                int(states[path_index - 1]) * state_count
                                + int(states[path_index])
                            )
                            + state_pairs,
                            dtype=np.int64,
                        )
                        for path_index, edge_index in enumerate(
                            selected_path_indices, start=1
                        )
                    ]
                )
            else:
                selected_indices = selected_path_indices
            if not len(selected_indices):
                return 0, int(node_count)
            selected_indices = np.unique(selected_indices)
            pending = selected_indices[~lazy_exact_verified[selected_indices]]
            if not len(pending):
                return 0, int(node_count)
            if lazy_exact_started is not None:
                lazy_elapsed = time.perf_counter() - lazy_exact_started
                lazy_profile = native_batch_profile["cuda_lazy_exact"]
                exact_edges_so_far = int(lazy_profile["exact_edges"])
                infeasible_ratio = float(
                    int(lazy_profile.get("exact_infeasible_edges", 0))
                    / max(exact_edges_so_far, 1)
                )
                if (
                    not lazy_dense_costs_loaded
                    and lazy_elapsed >= lazy_fallback_min_seconds
                    and exact_edges_so_far >= lazy_fallback_min_edges
                    and infeasible_ratio >= lazy_fallback_infeasible_ratio
                ):
                    fallback_lazy_to_dense_exact("high_exact_infeasible_ratio")
                    return 1, 0
                if (
                    not lazy_dense_costs_loaded
                    and lazy_exact_max_seconds > 0.0
                    and lazy_elapsed >= lazy_exact_max_seconds
                ):
                    fallback_lazy_to_dense_exact("time_budget")
                    return 1, 0
            exact_edges = np.ascontiguousarray(
                native_edge_array[pending], dtype=np.int32
            )
            exact_started = time.perf_counter()
            exact_values = np.asarray(
                lazy_exact_evaluator.evaluate_edge_batch(
                    lazy_exact_candidate_vectors,
                    exact_edges,
                    *lazy_exact_parameters,
                    int(lazy_exact_threads),
                    None,
                    True,
                    False,
                    True,
                    (
                        np.ascontiguousarray(
                            np.asarray(cuda_recall_hint_frames)[pending],
                            dtype=np.int32,
                        )
                        if cuda_recall_hint_frames is not None
                        else None
                    ),
                )
            )
            lazy_profile = native_batch_profile["cuda_lazy_exact"]
            lazy_profile["exact_evaluation_seconds"] = float(
                lazy_profile.get("exact_evaluation_seconds", 0.0)
            ) + float(time.perf_counter() - exact_started)
            exact_costs = np.asarray(exact_values[:, 0], dtype=np.float64)
            exact_costs[
                (exact_values[:, 7] > _EPSILON) | (exact_values[:, 8] > _EPSILON)
            ] = np.inf
            native_edge_costs[pending] = exact_costs
            if native_decode_indices is not None:
                compact_positions = np.searchsorted(native_decode_indices, pending)
                if np.any(compact_positions >= len(native_decode_indices)) or np.any(
                    native_decode_indices[compact_positions] != pending
                ):
                    raise RuntimeError(
                        "selected lazy edge is outside the compact decode graph"
                    )
                native_decode_edge_costs[compact_positions] = exact_costs
            else:
                compact_positions = np.asarray(pending, dtype=np.int64)
            lazy_exact_verified[pending] = True
            lazy_profile["exact_edges"] = int(lazy_profile["exact_edges"]) + int(
                len(pending)
            )
            lazy_profile["exact_batches"] = int(lazy_profile["exact_batches"]) + 1
            lazy_profile["exact_infeasible_edges"] = int(
                lazy_profile.get("exact_infeasible_edges", 0)
            ) + int(np.count_nonzero(~np.isfinite(exact_costs)))
            return int(len(pending)), int(np.min(exact_edges[:, 2]))

        def decode(penalty: float):
            decode_started = time.perf_counter()
            cache_before = len(edge_cache)
            if native_dp_requested:
                if (
                    native_metrics is None
                    or native_edge_array is None
                    or native_edge_costs is None
                    or native_decode_edge_array is None
                    or native_decode_edge_costs is None
                    or native_initial_losses is None
                ):
                    raise RuntimeError(
                        f"{NATIVE_DP_ENV}=1 requires {NATIVE_BATCH_ENV}=1"
                    )
                recompute_from = 0
                while True:
                    if native_incremental_decoder is None:
                        (
                            positions,
                            states,
                            raw_cost,
                        ) = native_metrics.decode_penalty_path(
                            native_decode_edge_array,
                            native_decode_edge_costs,
                            native_initial_losses,
                            int(node_count),
                            int(len(candidates_by_frame[frames[0]])),
                            float(penalty),
                        )
                    else:
                        positions, states, raw_cost = native_incremental_decoder.decode(
                            native_decode_edge_costs,
                            float(penalty),
                            int(recompute_from),
                        )
                    positions = [int(value) for value in positions]
                    states = [int(value) for value in states]
                    if not positions or not lazy_exact_enabled:
                        break
                    exactified, recompute_from = exactify_lazy_path(positions, states)
                    if exactified <= 0:
                        break
                    lazy_profile = native_batch_profile["cuda_lazy_exact"]
                    lazy_profile["decode_retries"] = (
                        int(lazy_profile["decode_retries"]) + 1
                    )
                if not positions:
                    fallback = (
                        list(frames),
                        [0] * len(frames),
                        float("inf"),
                        float(penalty),
                    )
                    decoded[(len(frames), float("inf"), tuple(fallback[1]))] = fallback
                    record_decode(decode_started, cache_before)
                    return fallback
                selected_frames = [frames[position] for position in positions]
                value = (
                    selected_frames,
                    states,
                    float(raw_cost),
                    float(penalty),
                )
                decoded[
                    (len(selected_frames), round(value[2], 12), tuple(states))
                ] = value
                record_decode(decode_started, cache_before)
                return value
            costs = [
                np.full(len(candidates_by_frame[frame]), np.inf, dtype=np.float64)
                for frame in frames
            ]
            raw_costs = [np.full_like(value, np.inf) for value in costs]
            counts = [np.full(len(value), 2**30, dtype=np.int32) for value in costs]
            back_pos = [np.full(len(value), -1, dtype=np.int32) for value in costs]
            back_state = [np.full(len(value), -1, dtype=np.int16) for value in costs]
            for state, candidate in enumerate(candidates_by_frame[frames[0]]):
                if float(candidate.recall_budget) <= _EPSILON:
                    raw = float(candidate.frame_loss)
                    costs[0][state] = raw + float(penalty)
                    raw_costs[0][state] = raw
                    counts[0][state] = 1
            for end_pos in range(1, node_count):
                end_state_count = len(candidates_by_frame[frames[end_pos]])
                for start_pos in range(predecessor_starts[end_pos], end_pos):
                    finite_start = np.flatnonzero(np.isfinite(costs[start_pos]))
                    if not len(finite_start):
                        continue
                    for start_state in finite_start.tolist():
                        for end_state in range(end_state_count):
                            info = edge(start_pos, start_state, end_pos, end_state)
                            if not math.isfinite(float(info.cost)):
                                continue
                            candidate_raw = float(
                                raw_costs[start_pos][start_state]
                            ) + float(info.cost)
                            candidate_count = int(counts[start_pos][start_state]) + 1
                            candidate_cost = (
                                candidate_raw + float(penalty) * candidate_count
                            )
                            current_cost = float(costs[end_pos][end_state])
                            current_raw = float(raw_costs[end_pos][end_state])
                            current_count = int(counts[end_pos][end_state])
                            if candidate_cost < current_cost - 1e-12 or (
                                abs(candidate_cost - current_cost) <= 1e-12
                                and (
                                    candidate_raw < current_raw - 1e-12
                                    or (
                                        abs(candidate_raw - current_raw) <= 1e-12
                                        and candidate_count < current_count
                                    )
                                )
                            ):
                                costs[end_pos][end_state] = candidate_cost
                                raw_costs[end_pos][end_state] = candidate_raw
                                counts[end_pos][end_state] = candidate_count
                                back_pos[end_pos][end_state] = start_pos
                                back_state[end_pos][end_state] = start_state
            final_states = np.flatnonzero(np.isfinite(costs[-1]))
            if not len(final_states):
                fallback = (
                    list(frames),
                    [0] * len(frames),
                    float("inf"),
                    float(penalty),
                )
                decoded[(len(frames), float("inf"), tuple(fallback[1]))] = fallback
                record_decode(decode_started, cache_before)
                return fallback
            final_state = min(
                final_states.tolist(),
                key=lambda state: (
                    float(costs[-1][state]),
                    float(raw_costs[-1][state]),
                    int(counts[-1][state]),
                    int(state),
                ),
            )
            positions = []
            states = []
            position = node_count - 1
            state = int(final_state)
            while position >= 0:
                positions.append(position)
                states.append(state)
                if position == 0:
                    break
                previous_position = int(back_pos[position][state])
                previous_state = int(back_state[position][state])
                if previous_position < 0 or previous_state < 0:
                    raise RuntimeError("broken Phase 2 DP predecessor chain")
                position, state = previous_position, previous_state
            positions.reverse()
            states.reverse()
            selected_frames = [frames[position] for position in positions]
            value = (
                selected_frames,
                states,
                float(raw_costs[-1][final_state]),
                float(penalty),
            )
            decoded[(len(selected_frames), round(value[2], 12), tuple(states))] = value
            record_decode(decode_started, cache_before)
            return value

        low = 0.0
        low_value = decode(low)
        if not math.isfinite(float(low_value[2])):
            emit_solver_profile()
            public_cache = {
                (frames[start], start_state, frames[end], end_state, 0): value
                for (start, start_state, end, end_state), value in edge_cache.items()
            }
            return (
                low_value[0],
                low_value[1],
                counters,
                public_cache,
                float(low_value[3]),
            )
        high = 1.0
        high_value = decode(high)
        maximum = max(float(runtime_args.penalty_max), 1.0)
        while len(high_value[0]) > int(target_count) and high < maximum:
            high = min(high * 2.0, maximum)
            high_value = decode(high)
            if high >= maximum:
                break
        if len(low_value[0]) > int(target_count):
            for _step in range(max(1, int(runtime_args.penalty_binary_steps))):
                middle = 0.5 * (low + high)
                value = decode(middle)
                if len(value[0]) > int(target_count):
                    low = middle
                    low_value = value
                else:
                    high = middle
                    high_value = value
        selected = min(
            decoded.values(),
            key=lambda value: (
                abs(len(value[0]) - int(target_count)),
                value[2],
                len(value[0]),
                tuple(value[1]),
            ),
        )
        public_cache = {
            (frames[start], start_state, frames[end], end_state, 0): value
            for (start, start_state, end, end_state), value in edge_cache.items()
        }
        if native_dp_requested:
            counters["interval_evals"] = int(len(native_edge_array))
            counters["interval_frames"] = int(
                np.sum(native_edge_array[:, 2] - native_edge_array[:, 0])
            )
            public_cache = {}
        emit_solver_profile()
        return (
            selected[0],
            selected[1],
            counters,
            public_cache,
            float(selected[3]),
        )

    return run_hard_multistate_penalty_path
