#!/usr/bin/env python3
"""Production polygon penalty-DP core with a hard per-frame Recall floor.

This private Production runtime loads the parity-frozen
Production implementation and changes only the semantics required by Phase 1:

* every prepared raw observation is a possible keyframe;
* the only shape state at a frame is Production's aligned raw polygon;
* pair-vote and post-decode shape repair are disabled;
* an interpolation edge is removed if any covered frame is below the Recall
  floor; and
* the Production quality-plus-lambda shortest path is retained, including its
  shape distance/switch terms.

No video is opened.  The input is the same prepared SQLite consumed by the
Production polygon optimizer.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

from production.polygon.runtime.algorithm_ids import PHASE1_RAW_ALGORITHM_ID


HERE = Path(__file__).resolve().parent
POSTPROCESS_ROOT = HERE.parents[2]
COMPATIBILITY_ENGINE_SOURCE = (
    POSTPROCESS_ROOT / "vendor" / "original_polygon" / "original_run_standalone.py"
)
_EPSILON = 1e-10
NATIVE_EXACT_ENV = "MASK_PIPELINE_PHASE1_NATIVE_EXACT"
NATIVE_INTERVAL_ENV = "MASK_PIPELINE_PHASE1_NATIVE_INTERVAL"
NATIVE_INTERVAL_VERIFY_ENV = "MASK_PIPELINE_PHASE1_NATIVE_INTERVAL_VERIFY"


def _load_production_runtime() -> ModuleType:
    module_name = "production_polygon_embedded_source"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name, COMPATIBILITY_ENGINE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load the parity-frozen compatibility engine: "
            f"{COMPATIBILITY_ENGINE_SOURCE}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _minimum_recall_deficit(recall: float, floor: float) -> float:
    return max(float(floor) - float(recall), 0.0)


def _patch_embedded_optimizer(module: ModuleType) -> ModuleType:
    if bool(getattr(module, "_phase1_raw_hard_recall_patched", False)):
        return module

    original_defaults = module.apply_fixed_practical_defaults
    original_candidates = module.build_frame_candidates
    original_interval_cost = module.interval_cost_from_vectors
    native_exact_enabled = os.environ.get(NATIVE_EXACT_ENV, "").strip() == "1"
    native_interval_enabled = os.environ.get(NATIVE_INTERVAL_ENV, "").strip() == "1"
    native_metrics = None
    if native_exact_enabled or native_interval_enabled:
        try:
            native_metrics = importlib.import_module("native_interval_metrics")
        except ImportError as exc:
            raise RuntimeError(
                "native interval acceleration was requested but "
                "native_interval_metrics is not importable"
            ) from exc

    if native_exact_enabled:

        def compute_exact_metrics_native(gt_polygons, pred_polygons):
            return native_metrics.exact_metrics(gt_polygons, pred_polygons)

        module.compute_exact_metrics_from_polygons = compute_exact_metrics_native
        module._phase1_native_exact_enabled = True
    else:
        module._phase1_native_exact_enabled = False
    module._phase1_native_interval_enabled = bool(native_interval_enabled)
    # The evaluator owns all per-frame GT masks. Retaining one entry per track
    # makes memory grow linearly across a full video. Only the active/most
    # recent run is reused between DP and final evaluation.
    native_interval_evaluator_cache = threading.local()
    native_verify_remaining = max(
        0, int(os.environ.get(NATIVE_INTERVAL_VERIFY_ENV, "0") or "0")
    )

    def get_native_interval_evaluator(eval_contexts, exact_gt_frames):
        if getattr(native_interval_evaluator_cache, "contexts", None) is eval_contexts:
            return native_interval_evaluator_cache.evaluator
        shifts = module.np.stack(
            [
                module.np.asarray(context.shift_xy, dtype=module.np.float32)
                for context in eval_contexts
            ],
            axis=0,
        )
        scales = module.np.asarray(
            [float(context.scale_factor) for context in eval_contexts],
            dtype=module.np.float32,
        )
        evaluator = native_metrics.CachedIntervalEvaluator(
            [context.gt_mask for context in eval_contexts],
            shifts,
            scales,
            exact_gt_frames,
        )
        native_interval_evaluator_cache.contexts = eval_contexts
        native_interval_evaluator_cache.evaluator = evaluator
        return evaluator

    def native_cached_interval_cost(*args, **kwargs):
        run, start_idx, start_vec, end_idx, end_vec, runtime_args = args[:6]
        include_start = bool(kwargs["include_start"])
        eval_contexts = kwargs.get("eval_contexts")
        if eval_contexts is None:
            return original_interval_cost(*args, **kwargs)
        if int(end_idx) < int(start_idx):
            return module.IntervalCost(
                cost=float("inf"),
                shape_distance=float("inf"),
                shape_update=1.0,
                frames_covered=0,
            )
        evaluator = get_native_interval_evaluator(eval_contexts, run.gt_polygons)
        native_profile = getattr(module, "_phase1_interval_profile", None)
        raster_started = time.perf_counter()
        (
            frame_loss_total,
            recall_budget_total,
            frames_covered,
        ) = evaluator.evaluate_vectors(
            module.np.asarray(start_vec, dtype=module.np.float32),
            module.np.asarray(end_vec, dtype=module.np.float32),
            int(run.contour_count),
            int(run.anchors_per_contour),
            int(start_idx),
            int(end_idx),
            include_start,
            float(runtime_args.interval_iou_weight),
            float(getattr(module, "_phase1_recall_floor", runtime_args.recall_min)),
        )
        if native_profile is not None:
            native_profile["native_raster_seconds"] += float(
                time.perf_counter() - raster_started
            )
        shape_started = time.perf_counter()
        distance = float(
            native_metrics.shape_distance(
                module.np.asarray(start_vec, dtype=module.np.float32),
                module.np.asarray(end_vec, dtype=module.np.float32),
                float(run.scale),
            )
        )
        update = (
            1.0 if distance > float(runtime_args.shape_update_threshold_ratio) else 0.0
        )
        frame_loss_mean = float(frame_loss_total) / max(int(frames_covered), 1)
        distance_scale, switch_scale = module.adaptive_shape_penalty_scales(
            frame_loss_mean, runtime_args
        )
        if native_profile is not None:
            native_profile["native_shape_seconds"] += float(
                time.perf_counter() - shape_started
            )
        total = float(frame_loss_total)
        total += (
            float(runtime_args.shape_switch_weight)
            * float(switch_scale)
            * float(update)
        )
        total += (
            float(runtime_args.shape_distance_weight)
            * float(distance_scale)
            * float(distance)
        )
        return module.IntervalCost(
            cost=float(total),
            shape_distance=float(distance),
            shape_update=float(update),
            frames_covered=int(frames_covered),
            frame_loss_mean=float(frame_loss_mean),
            shape_distance_scale=float(distance_scale),
            shape_switch_scale=float(switch_scale),
            recall_budget=float(recall_budget_total),
        )

    def rasterize_mask_with_context(polygons, context, out_mask=None):
        if out_mask is None:
            mask = module.np.zeros(context.shape_hw, dtype=module.np.uint8)
        else:
            mask = module.np.asarray(out_mask, dtype=module.np.uint8)
            mask.fill(0)
        # Match compute_exact_metrics_from_polygons: components are painted
        # independently.  Passing every component in one fillPoly call applies
        # an even/odd contour rule and can turn overlaps into artificial holes.
        for polygon in polygons:
            points = (
                module.np.asarray(polygon, dtype=module.np.float32)
                - context.shift_xy[None, :]
            ) * float(context.scale_factor)
            points = module.np.round(points).astype(module.np.int32)
            if len(points) >= 3:
                module.cv2.fillPoly(mask, [points], 1)
        return mask

    def rasterize_interpolated_mask_with_context(
        start_polygons, end_polygons, alpha, context, out_mask=None
    ):
        if out_mask is None:
            mask = module.np.zeros(context.shape_hw, dtype=module.np.uint8)
        else:
            mask = module.np.asarray(out_mask, dtype=module.np.uint8)
            mask.fill(0)
        alpha32 = module.np.float32(alpha)
        beta32 = module.np.float32(1.0) - alpha32
        for start_polygon, end_polygon in zip(start_polygons, end_polygons):
            points = (
                beta32 * module.np.asarray(start_polygon, dtype=module.np.float32)
                + alpha32 * module.np.asarray(end_polygon, dtype=module.np.float32)
                - context.shift_xy[None, :]
            ) * float(context.scale_factor)
            points = module.np.round(points).astype(module.np.int32)
            if len(points) >= 3:
                module.cv2.fillPoly(mask, [points], 1)
        return mask

    def apply_fixed_practical_defaults(args: argparse.Namespace) -> argparse.Namespace:
        args = original_defaults(args)
        module._phase1_recall_floor = float(args.recall_min)
        # These two stages mutate a selected raw key after the penalty DP.  A
        # true raw-only baseline must leave the selected vectors untouched.
        args.pair_vote_refine_enabled = False
        args.exact_recall_repair_enabled = False
        # More iterations improve target tracking without changing the
        # Production scalarized objective or its supported Pareto points.
        args.penalty_binary_steps = max(32, int(args.penalty_binary_steps))
        return args

    def recall_budget_from_metrics(metrics: dict[str, float]) -> float:
        return _minimum_recall_deficit(
            float(metrics["recall"]),
            float(getattr(module, "_phase1_recall_floor", 0.97)),
        )

    def recall_budget_limit(_frame_count: int, _args: argparse.Namespace) -> float:
        return 0.0

    def recall_violation(
        total_budget: float, _frame_count: int, _args: argparse.Namespace
    ) -> float:
        return max(float(total_budget), 0.0)

    def build_frame_candidates(*args, **kwargs):
        candidates_by_frame = original_candidates(*args, **kwargs)
        run = args[0]
        runtime_args = args[3]
        for frame_index, candidates in enumerate(candidates_by_frame):
            if len(candidates) != 1 or str(candidates[0].label) != "raw":
                raise RuntimeError(
                    "Phase 1 requires exactly one raw candidate per frame: "
                    f"frame_index={frame_index}, states={len(candidates)}"
                )
            exact = module.compute_exact_metrics_from_polygons(
                run.gt_polygons[frame_index], candidates[0].polygons
            )
            candidates[0].frame_loss = float(
                module.frame_accuracy_loss(exact, runtime_args)
            )
            candidates[0].objective = float(candidates[0].frame_loss)
            candidates[0].recall_budget = float(recall_budget_from_metrics(exact))
            # A resampled raw shape may itself fall microscopically below the
            # floor.  It remains a candidate position, but any edge selecting
            # it will be infeasible.  A neighboring raw-key interpolation can
            # still cover the frame without mutating the raw shape.
        return candidates_by_frame

    def build_candidate_frame_pool(run, _candidates, _target_count, _args):
        # Production normally pools about 2x the requested key count.  Under a
        # hard constraint that would prevent Recall debt from escaping into
        # additional keys.  Expose every observed/gap-filled raw frame while
        # keeping the downstream Production penalty solver unchanged.
        frames = list(range(int(len(run.frame_numbers))))
        return frames, [], module.np.zeros((len(frames),), dtype=module.np.float64)

    def interval_cost_from_vectors(*args, **kwargs):
        nonlocal native_verify_remaining
        precomputed_info = kwargs.pop("precomputed_interval_info", None)
        precomputed_exact_deficit = kwargs.pop("precomputed_exact_deficit", None)
        profile = getattr(module, "_phase1_interval_profile", None)
        if profile is None:
            profile = {
                "production_calls": 0,
                "production_seconds": 0.0,
                "exact_recheck_calls": 0,
                "exact_recheck_frames": 0,
                "exact_recheck_seconds": 0.0,
                "exact_recheck_mismatches": 0,
                "exact_mismatch_examples": [],
                "native_exact_enabled": bool(
                    getattr(module, "_phase1_native_exact_enabled", False)
                ),
                "native_interval_enabled": bool(
                    getattr(module, "_phase1_native_interval_enabled", False)
                ),
                "native_interval_verified": 0,
                "native_interval_max_abs_difference": 0.0,
                "native_raster_seconds": 0.0,
                "native_shape_seconds": 0.0,
            }
            module._phase1_interval_profile = profile
        production_started = time.perf_counter()
        if precomputed_info is not None:
            info = precomputed_info
        elif native_interval_enabled:
            info = native_cached_interval_cost(*args, **kwargs)
        else:
            info = original_interval_cost(*args, **kwargs)
        profile["production_calls"] += 1
        profile["production_seconds"] += float(time.perf_counter() - production_started)
        if (
            native_interval_enabled or precomputed_info is not None
        ) and native_verify_remaining > 0:
            reference = original_interval_cost(*args, **kwargs)
            fields = (
                "cost",
                "shape_distance",
                "shape_update",
                "frames_covered",
                "frame_loss_mean",
                "shape_distance_scale",
                "shape_switch_scale",
                "recall_budget",
            )
            max_difference = max(
                abs(float(getattr(info, field)) - float(getattr(reference, field)))
                for field in fields
            )
            profile["native_interval_verified"] += 1
            profile["native_interval_max_abs_difference"] = max(
                float(profile["native_interval_max_abs_difference"]),
                float(max_difference),
            )
            native_verify_remaining -= 1
            if max_difference > 1e-12:
                raise RuntimeError(
                    "native cached interval mismatch: "
                    f"max_abs_difference={max_difference:.17g}, "
                    f"start={args[1]}, end={args[3]}"
                )
        if precomputed_exact_deficit is not None:
            combined_deficit = max(
                float(info.recall_budget),
                float(precomputed_exact_deficit),
            )
            if combined_deficit <= _EPSILON:
                return info
            return module.IntervalCost(
                cost=float("inf"),
                shape_distance=float(info.shape_distance),
                shape_update=float(info.shape_update),
                frames_covered=int(info.frames_covered),
                frame_loss_mean=float(info.frame_loss_mean),
                shape_distance_scale=float(info.shape_distance_scale),
                shape_switch_scale=float(info.shape_switch_scale),
                recall_budget=float(combined_deficit),
            )
        if float(info.recall_budget) <= _EPSILON:
            run, start_idx, start_vec, end_idx, end_vec, _runtime_args = args[:6]
            include_start = bool(kwargs["include_start"])
            start_candidate = kwargs.get("start_candidate")
            end_candidate = kwargs.get("end_candidate")
            start_polygons = (
                start_candidate.polygons
                if start_candidate is not None
                else module.split_vector_to_polygons(
                    start_vec, run.contour_count, run.anchors_per_contour
                )
            )
            end_polygons = (
                end_candidate.polygons
                if end_candidate is not None
                else module.split_vector_to_polygons(
                    end_vec, run.contour_count, run.anchors_per_contour
                )
            )
            exact_deficit = 0.0
            first_frame = int(start_idx if include_start else start_idx + 1)
            exact_started = time.perf_counter()
            exact_frames = 0
            for frame_idx in range(first_frame, int(end_idx) + 1):
                exact_frames += 1
                if frame_idx == int(start_idx):
                    polygons = start_polygons
                elif frame_idx == int(end_idx):
                    polygons = end_polygons
                else:
                    alpha = float(
                        (frame_idx - int(start_idx))
                        / max(int(end_idx) - int(start_idx), 1)
                    )
                    polygons = module.split_vector_to_polygons(
                        module.interpolate_vectors(start_vec, end_vec, alpha),
                        run.contour_count,
                        run.anchors_per_contour,
                    )
                metrics = module.compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], polygons
                )
                exact_deficit += recall_budget_from_metrics(metrics)
                if exact_deficit > _EPSILON:
                    examples = profile["exact_mismatch_examples"]
                    eval_contexts = kwargs.get("eval_contexts")
                    if len(examples) < 8 and eval_contexts is not None:
                        cached_metrics = module.compute_cached_metrics_from_polygons(
                            eval_contexts[frame_idx], polygons
                        )
                        examples.append(
                            {
                                "frame_index": int(frame_idx),
                                "cached_recall": float(cached_metrics["recall"]),
                                "exact_recall": float(metrics["recall"]),
                                "cached_gt_area": float(cached_metrics["gt_area"]),
                                "exact_gt_area": float(metrics["gt_area"]),
                                "cached_intersection": float(
                                    cached_metrics["intersection"]
                                ),
                                "exact_intersection": float(metrics["intersection"]),
                            }
                        )
                    break
            profile["exact_recheck_calls"] += 1
            profile["exact_recheck_frames"] += int(exact_frames)
            profile["exact_recheck_seconds"] += float(
                time.perf_counter() - exact_started
            )
            if exact_deficit > _EPSILON:
                profile["exact_recheck_mismatches"] += 1
            if exact_deficit <= _EPSILON:
                return info
            info = module.IntervalCost(
                cost=float(info.cost),
                shape_distance=float(info.shape_distance),
                shape_update=float(info.shape_update),
                frames_covered=int(info.frames_covered),
                frame_loss_mean=float(info.frame_loss_mean),
                shape_distance_scale=float(info.shape_distance_scale),
                shape_switch_scale=float(info.shape_switch_scale),
                recall_budget=float(exact_deficit),
            )
        # Recall is a feasible-set definition, not a finite penalty.  Keeping
        # all diagnostic fields makes Production reporting and caches usable.
        return module.IntervalCost(
            cost=float("inf"),
            shape_distance=float(info.shape_distance),
            shape_update=float(info.shape_update),
            frames_covered=int(info.frames_covered),
            frame_loss_mean=float(info.frame_loss_mean),
            shape_distance_scale=float(info.shape_distance_scale),
            shape_switch_scale=float(info.shape_switch_scale),
            recall_budget=float(info.recall_budget),
        )

    def run_single_state_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        args,
        eval_contexts=None,
    ):
        """Production penalty path over the hard-Recall feasible raw graph."""

        target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
        dynamic_max_gap = max(
            int(args.max_gap),
            int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
        )
        frames = [int(value) for value in candidate_frames]
        node_count = len(frames)
        if node_count <= 0:
            raise RuntimeError("Phase 1 received an empty raw candidate graph")
        if frames != list(range(len(run.frame_numbers))):
            raise RuntimeError("Phase 1 candidate pool is not dense")

        first = candidates_by_frame[frames[0]][0]

        predecessor_starts = [0] * node_count
        for end in range(1, node_count):
            predecessor_starts[end] = int(
                bisect.bisect_left(
                    frames,
                    frames[end] - dynamic_max_gap,
                    0,
                    end,
                )
            )

        edge_cache: dict[tuple[int, int], object] = {}
        counters = {"interval_evals": 0, "interval_frames": 0}

        def edge(start: int, end: int):
            key = (int(start), int(end))
            cached = edge_cache.get(key)
            if cached is not None:
                return cached
            left_frame = frames[start]
            right_frame = frames[end]
            left = candidates_by_frame[left_frame][0]
            right = candidates_by_frame[right_frame][0]
            value = interval_cost_from_vectors(
                run,
                left_frame,
                left.vector,
                right_frame,
                right.vector,
                args,
                include_start=False,
                eval_contexts=eval_contexts,
                start_candidate=left,
                end_candidate=right,
            )
            edge_cache[key] = value
            counters["interval_evals"] += 1
            counters["interval_frames"] += int(value.frames_covered)
            return value

        decoded: dict[tuple[int, float], tuple[list[int], float, float]] = {}

        def decode(penalty: float) -> tuple[list[int], float, float]:
            costs = module.np.full(
                (node_count,), module.np.inf, dtype=module.np.float64
            )
            raw_costs = module.np.full(
                (node_count,), module.np.inf, dtype=module.np.float64
            )
            counts = module.np.full((node_count,), 2**30, dtype=module.np.int32)
            back = module.np.full((node_count,), -1, dtype=module.np.int32)
            if float(first.recall_budget) <= _EPSILON:
                costs[0] = float(first.frame_loss) + float(penalty)
                raw_costs[0] = float(first.frame_loss)
                counts[0] = 1

            for end in range(1, node_count):
                for start in range(predecessor_starts[end], end):
                    if not module.np.isfinite(costs[start]):
                        continue
                    info = edge(start, end)
                    if not module.np.isfinite(float(info.cost)):
                        continue
                    candidate_raw = float(raw_costs[start]) + float(info.cost)
                    candidate_count = int(counts[start]) + 1
                    candidate_cost = candidate_raw + float(penalty) * candidate_count
                    if candidate_cost < float(costs[end]) - 1e-12 or (
                        abs(candidate_cost - float(costs[end])) <= 1e-12
                        and (
                            candidate_raw < float(raw_costs[end]) - 1e-12
                            or (
                                abs(candidate_raw - float(raw_costs[end])) <= 1e-12
                                and candidate_count < int(counts[end])
                            )
                        )
                    ):
                        costs[end] = candidate_cost
                        raw_costs[end] = candidate_raw
                        counts[end] = candidate_count
                        back[end] = start

            if node_count > 1 and int(back[-1]) < 0:
                reachable = module.np.flatnonzero(module.np.isfinite(costs)).tolist()
                incoming = []
                for start in range(predecessor_starts[-1], node_count - 1):
                    info = edge(start, node_count - 1)
                    incoming.append(
                        {
                            "start": int(frames[start]),
                            "reachable": bool(module.np.isfinite(costs[start])),
                            "recall_deficit": float(info.recall_budget),
                        }
                    )
                diagnostic = {
                    "stream_id": str(run.stream_id),
                    "nodes": node_count,
                    "last_reachable": frames[reachable[-1]] if reachable else None,
                    "first_anchor_deficit": float(first.recall_budget),
                    "last_anchor_deficit": float(
                        candidates_by_frame[frames[-1]][0].recall_budget
                    ),
                    "incoming_to_last": incoming,
                }
                print(
                    "[phase1-infeasible-raw-graph] "
                    + json.dumps(diagnostic, ensure_ascii=False),
                    flush=True,
                )
                # Keep the matrix measurable without pretending a feasible
                # solution exists.  The all-raw diagnostic path is emitted;
                # its final objective is +inf and exact Recall violations are
                # reported.  No shape is repaired or silently relaxed.
                fallback = (list(frames), float("inf"), float(penalty))
                decoded[(len(frames), float("inf"))] = fallback
                return fallback
            positions: list[int] = []
            cursor = node_count - 1
            while cursor >= 0:
                positions.append(cursor)
                if cursor == 0:
                    break
                cursor = int(back[cursor])
            positions.reverse()
            selected_frames = [frames[index] for index in positions]
            value = (selected_frames, float(raw_costs[-1]), float(penalty))
            decoded[(len(selected_frames), round(float(raw_costs[-1]), 12))] = value
            return value

        # lambda=0 is essential for requested interval 1.  Historical
        # Production's midpoint-only search omitted it and could not report
        # the actual high-quality end of the supported trade-off range.
        low = 0.0
        low_value = decode(low)
        if not math.isfinite(float(low_value[1])):
            cost_cache = {
                (frames[start], 0, frames[end], 0, 0): value
                for (start, end), value in edge_cache.items()
            }
            return (
                low_value[0],
                [0] * len(low_value[0]),
                counters,
                cost_cache,
                float(low_value[2]),
            )
        high = 1.0
        high_value = decode(high)
        maximum = max(float(args.penalty_max), 1.0)
        while len(high_value[0]) > int(target_count) and high < maximum:
            high = min(high * 2.0, maximum)
            high_value = decode(high)
            if high >= maximum:
                break

        if len(low_value[0]) > int(target_count):
            for _step in range(max(1, int(args.penalty_binary_steps))):
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
                value[1],
                len(value[0]),
            ),
        )
        selected_frames, _raw_loss, selected_lambda = selected
        cost_cache = {
            (frames[start], 0, frames[end], 0, 0): value
            for (start, end), value in edge_cache.items()
        }
        return (
            selected_frames,
            [0] * len(selected_frames),
            counters,
            cost_cache,
            float(selected_lambda),
        )

    module.apply_fixed_practical_defaults = apply_fixed_practical_defaults
    module.rasterize_mask_with_context = rasterize_mask_with_context
    module.rasterize_interpolated_mask_with_context = (
        rasterize_interpolated_mask_with_context
    )
    module._phase1_get_native_interval_evaluator = get_native_interval_evaluator
    module.recall_budget_from_metrics = recall_budget_from_metrics
    module.recall_budget_limit = recall_budget_limit
    module.recall_violation = recall_violation
    module.build_frame_candidates = build_frame_candidates
    module.build_candidate_frame_pool = build_candidate_frame_pool
    module.interval_cost_from_vectors = interval_cost_from_vectors
    module.run_single_state_penalty_path = run_single_state_penalty_path
    module._phase1_raw_hard_recall_patched = True
    return module


def _write_audit(
    output_dir: Path,
    recall_floor: float,
    patched_module: ModuleType | None,
) -> dict[str, object]:
    metrics_path = output_dir / "exact" / "keyframe_exact_metrics.csv"
    recalls: list[float] = []
    ious: list[float] = []
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
        for row in metric_rows:
            recalls.append(float(row["recall"]))
            ious.append(float(row["iou"]))
    stream_rows: list[dict[str, str]] = []
    with (output_dir / "opt" / "stream_segments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        stream_rows.extend(csv.DictReader(handle))
    infeasible_keys = {
        (str(row["track_id"]), int(row["run_id"]))
        for row in stream_rows
        if (
            not math.isfinite(float(row["objective"]))
            or float(row["recall_budget_violation"]) > _EPSILON
        )
    }
    feasible_exact_recalls = [
        float(row["recall"])
        for row in metric_rows
        if (str(row["track_id"]), int(row["run_id"])) not in infeasible_keys
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    optimizer = summary["optimizer_summary"]
    audit = {
        "schema_version": 1,
        "algorithm": PHASE1_RAW_ALGORITHM_ID,
        "compatibility_engine_source": str(COMPATIBILITY_ENGINE_SOURCE),
        "recall_floor": float(recall_floor),
        "evaluated_rows": len(recalls),
        "minimum_recall": min(recalls, default=1.0),
        "mean_recall": sum(recalls) / max(len(recalls), 1),
        "mean_iou": sum(ious) / max(len(ious), 1),
        "violations": sum(value + 1e-12 < recall_floor for value in recalls),
        "constraint_satisfied": all(value + 1e-12 >= recall_floor for value in recalls),
        "pair_vote_disabled": not bool(optimizer["pair_vote_refine_enabled"]),
        "raw_state_only": abs(float(optimizer["mean_state_count"]) - 1.0) < 1e-12,
        "dense_candidate_pool": int(optimizer["candidate_frame_count_total"])
        == int(optimizer["row_count"]),
        # Production measures the call overhead even when this stage returns
        # immediately.  Audit the actual runtime flag instead of requiring an
        # impossible exactly-zero timer.
        "post_decode_shape_repair_disabled": bool(
            patched_module is not None
            and getattr(patched_module, "_phase1_exact_repair_disabled", False)
        ),
        "post_decode_shape_repair_call_seconds": float(
            optimizer["stage_seconds_total"]["exact_recall_repair_seconds"]
        ),
        "infeasible_streams": len(infeasible_keys),
        "feasible_exact_minimum_recall": min(feasible_exact_recalls, default=1.0),
        "feasible_exact_violations": sum(
            value + 1e-12 < recall_floor for value in feasible_exact_recalls
        ),
        "semantic_changes": {
            "candidate_shapes": "one aligned Production raw state per frame",
            "candidate_positions": "all prepared observations and gap-filled frames",
            "recall": "hard per-frame edge feasibility",
            "selection": "Production quality loss plus lambda per key",
            "pair_vote": "disabled",
            "post_decode_repair": "disabled",
        },
    }
    audit["implementation_contract_satisfied"] = (
        all(
            bool(audit[key])
            for key in (
                "pair_vote_disabled",
                "raw_state_only",
                "dense_candidate_pool",
                "post_decode_shape_repair_disabled",
            )
        )
        and int(audit["feasible_exact_violations"]) == 0
    )
    if not bool(audit["implementation_contract_satisfied"]):
        raise RuntimeError(f"Phase 1 contract audit failed: {audit}")
    (output_dir / "phase1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    source = _load_production_runtime()
    original_builder = source._build_embedded_polygon_v22_module
    patched_holder: list[ModuleType] = []

    def build_patched_module() -> ModuleType:
        patched = _patch_embedded_optimizer(original_builder())
        patched._phase1_exact_repair_disabled = True
        patched_holder.append(patched)
        return patched

    source._build_embedded_polygon_v22_module = build_patched_module
    os.environ["ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA"] = "1"
    source.dispatch_main()

    if len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--recall-min", type=float, default=0.97)
        known, _unknown = parser.parse_known_args(sys.argv[2:])
        audit = _write_audit(
            known.output_dir,
            known.recall_min,
            patched_holder[-1] if patched_holder else None,
        )
        print(json.dumps({"phase1_audit": audit}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
