from __future__ import annotations

import sys
import types
from pathlib import Path

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parent


def _register_inline_module(
    module_name: str, export_map: dict[str, str]
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(SELF_PATH)
    for public_name, global_name in export_map.items():
        value = globals()[global_name]
        setattr(module, public_name, value)
        setattr(module, global_name, value)
    sys.modules[module_name] = module
    return module


import argparse
import math
import time
from pathlib import Path
import numpy as np
from .optimizer import kfbase_module as base
from .dense_recall import kfdense_module as dense_base


def kftrackk_parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-recall keyframe optimizer that does not split on K1/K2 mode changes; only track, K, and frame continuity."
    )
    parser.add_argument("--input-metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-ratio", type=float, default=-1.0)
    parser.add_argument("--target-k1-ratio", type=float, default=0.1)
    parser.add_argument("--target-k2-ratio", type=float, default=0.16)
    parser.add_argument("--solver", choices=["dp", "dp_rewarded"], default="dp")
    parser.add_argument("--lambda-all", type=float, default=-1.0)
    parser.add_argument("--lambda-search-iters", type=int, default=16)
    parser.add_argument("--smooth-alpha", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.18)
    parser.add_argument("--error-scale", type=float, default=4000.0)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--local-search-radius", type=int, default=2)
    parser.add_argument(
        "--value-refine",
        choices=["none", "global_ls", "segment_ls", "residual_nudge"],
        default="global_ls",
    )
    parser.add_argument("--value-refine-ridge", type=float, default=0.001)
    parser.add_argument("--value-refine-damping", type=float, default=1.0)
    parser.add_argument("--min-segment-length", type=int, default=3)
    parser.add_argument("--theta-weight-floor", type=float, default=0.2)
    parser.add_argument("--weight-error-gain", type=float, default=1.0)
    parser.add_argument("--weight-curvature-gain", type=float, default=1.0)
    parser.add_argument("--importance-cap", type=float, default=4.0)
    parser.add_argument("--reward-error-gain", type=float, default=0.75)
    parser.add_argument("--reward-curvature-gain", type=float, default=1.25)
    parser.add_argument("--reward-cap", type=float, default=1.5)
    parser.add_argument("--auto-break-threshold", type=float, default=-1.0)
    parser.add_argument("--auto-break-min-length", type=int, default=8)
    parser.add_argument("--auto-break-min-separation", type=int, default=6)
    parser.add_argument(
        "--keyframe-value-source",
        choices=["smoothed", "raw", "confidence_blend"],
        default="confidence_blend",
    )
    parser.add_argument("--k2-slot-center-weight", type=float, default=1.0)
    parser.add_argument("--k2-slot-size-weight", type=float, default=0.65)
    parser.add_argument("--k2-slot-angle-weight", type=float, default=0.2)
    parser.add_argument("--max-streams", type=int, default=-1)
    parser.add_argument("--dense-recall-target", type=float, default=0.96)
    parser.add_argument("--dense-recall-samples", type=int, default=61)
    parser.add_argument("--dense-recall-max-inflate-log", type=float, default=1.2)
    parser.add_argument("--dense-recall-search-iters", type=int, default=20)
    return parser.parse_args(argv)


def kftrackk_resolve_target_ratio(args: argparse.Namespace) -> float:
    if float(args.target_ratio) > 0.0:
        return float(args.target_ratio)
    if abs(float(args.target_k1_ratio) - float(args.target_k2_ratio)) < 1e-09:
        return float(args.target_k1_ratio)
    return 0.5 * (float(args.target_k1_ratio) + float(args.target_k2_ratio))


def kftrackk_split_runs_track_k(
    rows: list[base.MetricRow],
) -> list[list[base.MetricRow]]:
    grouped: dict[str, list[base.MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    runs: list[list[base.MetricRow]] = []
    for _track_id, track_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        track_rows.sort(key=lambda r: r.frame)
        current: list[base.MetricRow] = []
        prev: base.MetricRow | None = None
        for row in track_rows:
            split = (
                prev is None
                or len(row.ellipse_params) != len(prev.ellipse_params)
                or row.frame != prev.frame + 1
            )
            if split:
                if current:
                    runs.append(current)
                current = [row]
            else:
                current.append(row)
            prev = row
        if current:
            runs.append(current)
    return runs


def kftrackk_build_stream_segments_track_k(
    args: argparse.Namespace, rows: list[base.MetricRow]
) -> list[base.StreamSegment]:
    runs = kftrackk_split_runs_track_k(rows)
    streams: list[base.StreamSegment] = []
    for run_id, run_rows in enumerate(runs):
        slot_count = len(run_rows[0].ellipse_params)
        if slot_count == 2:
            stabilized = base.stabilize_k2_slots(
                run_rows,
                center_weight=float(args.k2_slot_center_weight),
                size_weight=float(args.k2_slot_size_weight),
                angle_weight=float(args.k2_slot_angle_weight),
            )
        else:
            stabilized = [[list(row.ellipse_params[0])] for row in run_rows]
        frame_modes = [row.mode for row in run_rows]
        run_mode = frame_modes[0] if len(set(frame_modes)) == 1 else "MIXED"
        for slot_id in range(slot_count):
            states = np.asarray(
                [frame_slots[slot_id] for frame_slots in stabilized], dtype=np.float64
            )
            states[:, 4] = base.unwrap_angles_deg(states[:, 4])
            confidence = np.asarray(
                [
                    base.compute_confidence(
                        row,
                        floor=float(args.confidence_floor),
                        error_scale=float(args.error_scale),
                    )
                    for row in run_rows
                ],
                dtype=np.float64,
            )
            weighted_error = np.asarray(
                [row.weighted_error for row in run_rows], dtype=np.float64
            )
            stream = base.StreamSegment(
                stream_id=f"{run_rows[0].track_id}:K{slot_count}:run{run_id}:slot{slot_id}",
                track_id=run_rows[0].track_id,
                mode=run_mode,
                run_id=run_id,
                slot_id=slot_id,
                frame_numbers=np.asarray(
                    [row.frame for row in run_rows], dtype=np.int32
                ),
                raw_states=states,
                confidence=confidence,
                weighted_error=weighted_error,
            )
            setattr(stream, "frame_modes", list(frame_modes))
            setattr(stream, "ellipse_count", int(slot_count))
            streams.append(stream)
    return streams


def kftrackk_interpolated_state(
    stream: base.StreamSegment, key_q: np.ndarray, chosen: list[int]
) -> np.ndarray:
    interp_q = base.interpolate_from_key_values(
        key_q, chosen, len(stream.frame_numbers)
    )
    return base.q_to_state(
        interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
    )


def kftrackk_recall_info(
    recall: float,
    weights: np.ndarray,
    target: float,
    inflate_log_delta: float,
    attained: bool,
) -> dict[str, float | bool]:
    return {
        "dense_recall_before": float(recall),
        "dense_recall_after": float(recall),
        "inflate_log_delta": float(inflate_log_delta),
        "dense_recall_target": float(target),
        "dense_recall_attained": bool(attained),
        "source_area_sum": float(np.sum(weights)),
    }


def kftrackk_apply_uniform_key_inflation(key_q: np.ndarray, delta: float) -> np.ndarray:
    out = key_q.copy()
    out[:, 2] += float(delta)
    out[:, 3] += float(delta)
    return out


def kftrackk_approx_union_frame_recall(
    source_slots: list[np.ndarray],
    pred_slots: list[np.ndarray],
    disk_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return kftrackk_union_recall_from_source_samples(
        kftrackk_source_sample_payloads(source_slots, disk_samples), pred_slots
    )


def kftrackk_source_sample_payloads(
    source_slots: list[np.ndarray], disk_samples: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if not source_slots or len(source_slots[0]) == 0:
        return payloads
    sx = disk_samples[:, 0][None, :]
    sy = disk_samples[:, 1][None, :]
    for source_states in source_slots:
        src_theta = np.deg2rad(source_states[:, 4])[:, None]
        src_cos = np.cos(src_theta)
        src_sin = np.sin(src_theta)
        local_x = sx * source_states[:, 2][:, None]
        local_y = sy * source_states[:, 3][:, None]
        world_x = source_states[:, 0][:, None] + src_cos * local_x - src_sin * local_y
        world_y = source_states[:, 1][:, None] + src_sin * local_x + src_cos * local_y
        slot_weights = np.maximum(source_states[:, 2] * source_states[:, 3], 1e-06)
        payloads.append((world_x, world_y, slot_weights))
    return payloads


def kftrackk_union_recall_from_source_samples(
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    pred_slots: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not payloads:
        return (np.ones(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    n = int(payloads[0][0].shape[0])
    covered = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    for world_x, world_y, slot_weights in payloads:
        inside = np.zeros(world_x.shape, dtype=bool)
        for pred_states in pred_slots:
            inside |= dense_base.ellipse_membership(world_x, world_y, pred_states)
        covered += inside.mean(axis=1).astype(np.float64) * slot_weights
        weights += slot_weights
    per_frame = np.divide(
        covered, weights, out=np.ones_like(covered), where=weights > 0.0
    )
    return (per_frame, weights)


def kftrackk_union_recall_score(
    source_slots: list[np.ndarray],
    pred_slots: list[np.ndarray],
    disk_samples: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    per_frame, weights = kftrackk_approx_union_frame_recall(
        source_slots, pred_slots, disk_samples
    )
    recall = (
        float(np.average(per_frame, weights=np.maximum(weights, 1e-06)))
        if len(per_frame)
        else 1.0
    )
    return (recall, per_frame, weights)


def kftrackk_pred_slots_with_inflation(
    items: list[dict[str, object]], deltas: tuple[float, float]
) -> list[np.ndarray]:
    pred_slots: list[np.ndarray] = []
    for item, delta in zip(items, deltas, strict=True):
        pred_slots.append(
            kftrackk_interpolated_state(
                item["stream"],
                kftrackk_apply_uniform_key_inflation(item["key_q"], float(delta)),
                item["chosen"],
            )
        )
    return pred_slots


def kftrackk_union_inflation_cost(
    base_area_sums: np.ndarray, deltas: tuple[float, float]
) -> float:
    delta_arr = np.asarray(deltas, dtype=np.float64)
    return float(
        np.sum(base_area_sums * np.maximum(np.exp(2.0 * delta_arr) - 1.0, 0.0))
    )


def kftrackk_find_joint_union_inflation(
    items: list[dict[str, object]],
    source_slots: list[np.ndarray],
    target: float,
    args: argparse.Namespace,
    disk_samples: np.ndarray,
) -> tuple[tuple[float, float], float, bool]:
    max_delta = max(0.0, float(args.dense_recall_max_inflate_log))
    search_iters = max(1, int(args.dense_recall_search_iters))
    source_payloads = kftrackk_source_sample_payloads(source_slots, disk_samples)
    base_pred_slots = [
        kftrackk_interpolated_state(item["stream"], item["key_q"], item["chosen"])
        for item in items
    ]
    base_area_sums = np.asarray(
        [
            float(np.sum(np.maximum(slot[:, 2] * slot[:, 3], 1e-06)))
            for slot in base_pred_slots
        ],
        dtype=np.float64,
    )

    def inflated_pred_slots(deltas: tuple[float, float]) -> list[np.ndarray]:
        pred_slots: list[np.ndarray] = []
        for slot, delta in zip(base_pred_slots, deltas, strict=True):
            out = slot.copy()
            scale = math.exp(float(delta))
            out[:, 2] *= scale
            out[:, 3] *= scale
            pred_slots.append(out)
        return pred_slots

    def evaluate(deltas: tuple[float, float]) -> float:
        per_frame, weights = kftrackk_union_recall_from_source_samples(
            source_payloads, inflated_pred_slots(deltas)
        )
        recall = (
            float(np.average(per_frame, weights=np.maximum(weights, 1e-06)))
            if len(per_frame)
            else 1.0
        )
        return float(recall)

    def remember(
        best: tuple[tuple[float, float], float, float] | None,
        deltas: tuple[float, float],
        recall: float,
    ) -> tuple[tuple[float, float], float, float]:
        cost = kftrackk_union_inflation_cost(base_area_sums, deltas)
        candidate = (deltas, float(recall), float(cost))
        if best is None:
            return candidate
        if candidate[2] < best[2] - 1e-09:
            return candidate
        if abs(candidate[2] - best[2]) <= 1e-09 and candidate[1] > best[1]:
            return candidate
        return best

    high_recall = evaluate((max_delta, max_delta))
    if high_recall < target or max_delta <= 0.0:
        return ((max_delta, max_delta), float(high_recall), False)

    best: tuple[tuple[float, float], float, float] | None = None

    low = 0.0
    high = max_delta
    for _ in range(search_iters):
        mid = 0.5 * (low + high)
        recall = evaluate((mid, mid))
        if recall >= target:
            high = mid
            best = remember(best, (mid, mid), recall)
        else:
            low = mid
    uniform_delta = high
    uniform_recall = evaluate((uniform_delta, uniform_delta))
    best = remember(best, (uniform_delta, uniform_delta), uniform_recall)

    def minimal_partner_delta(
        slot_index: int, fixed_delta: float
    ) -> tuple[tuple[float, float], float] | None:
        endpoint = (
            (float(fixed_delta), max_delta)
            if slot_index == 0
            else (max_delta, float(fixed_delta))
        )
        if evaluate(endpoint) < target:
            return None
        low_partner = 0.0
        high_partner = max_delta
        best_recall = target
        for _ in range(search_iters):
            mid = 0.5 * (low_partner + high_partner)
            deltas = (
                (float(fixed_delta), mid)
                if slot_index == 0
                else (mid, float(fixed_delta))
            )
            recall = evaluate(deltas)
            if recall >= target:
                high_partner = mid
                best_recall = recall
            else:
                low_partner = mid
        deltas = (
            (float(fixed_delta), high_partner)
            if slot_index == 0
            else (high_partner, float(fixed_delta))
        )
        return (deltas, float(best_recall))

    def scan_fixed_deltas(lo: float, hi: float, count: int) -> None:
        nonlocal best
        if count <= 1:
            grid = np.asarray([0.5 * (lo + hi)], dtype=np.float64)
        else:
            grid = np.linspace(float(lo), float(hi), int(count), dtype=np.float64)
        for fixed in grid:
            for slot_index in (0, 1):
                result = minimal_partner_delta(slot_index, float(fixed))
                if result is not None:
                    best = remember(best, result[0], result[1])

    coarse_count = min(17, max(7, search_iters // 2 + 1))
    scan_fixed_deltas(0.0, max_delta, coarse_count)
    if best is not None:
        best_delta0, best_delta1 = best[0]
        step = max_delta / max(coarse_count - 1, 1)
        scan_fixed_deltas(
            max(0.0, best_delta0 - step), min(max_delta, best_delta0 + step), 9
        )
        scan_fixed_deltas(
            max(0.0, best_delta1 - step), min(max_delta, best_delta1 + step), 9
        )

    if best is None:
        return ((max_delta, max_delta), float(high_recall), False)
    return (best[0], float(best[1]), True)


def kftrackk_enforce_union_frame_recall_target(
    items: list[dict[str, object]], args: argparse.Namespace, disk_samples: np.ndarray
) -> None:
    items.sort(key=lambda item: int(getattr(item["stream"], "slot_id")))
    streams = [item["stream"] for item in items]
    source_slots = [stream.raw_states for stream in streams]
    target = float(args.dense_recall_target)
    pred_slots = [
        kftrackk_interpolated_state(item["stream"], item["key_q"], item["chosen"])
        for item in items
    ]
    base_recall, _per_frame, _weights = kftrackk_union_recall_score(
        source_slots, pred_slots, disk_samples
    )
    if (
        target <= 0.0
        or base_recall >= target
        or any(len(item["chosen"]) == 0 for item in items)
    ):
        for item, stream in zip(items, streams, strict=True):
            slot_weights = np.maximum(
                stream.raw_states[:, 2] * stream.raw_states[:, 3], 1e-06
            )
            item["dense_info"] = kftrackk_recall_info(
                base_recall, slot_weights, target, 0.0, base_recall >= target
            )
        return
    deltas, repaired_recall, attained = kftrackk_find_joint_union_inflation(
        items, source_slots, target, args, disk_samples
    )
    for item, stream, delta in zip(items, streams, deltas, strict=True):
        item["key_q"] = kftrackk_apply_uniform_key_inflation(
            item["key_q"], float(delta)
        )
        slot_weights = np.maximum(
            stream.raw_states[:, 2] * stream.raw_states[:, 3], 1e-06
        )
        dense_info = kftrackk_recall_info(
            base_recall, slot_weights, target, float(delta), bool(attained)
        )
        dense_info["dense_recall_after"] = float(repaired_recall)
        item["dense_info"] = dense_info


def kftrackk_optimize_streams_track_k_dense_recall(
    streams: list[base.StreamSegment], penalty: float, args: argparse.Namespace
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, float],
]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    disk_samples = dense_base.unit_disk_samples(int(args.dense_recall_samples))
    timings = {
        "stream_loop_total": 0.0,
        "anchor_setup": 0.0,
        "decode_dp": 0.0,
        "refine_local": 0.0,
        "value_refine": 0.0,
        "dense_recall_enforce": 0.0,
        "interpolate_and_state": 0.0,
        "emit_rows": 0.0,
    }
    total_t0 = time.perf_counter()
    work_items: list[dict[str, object]] = []
    for stream in streams:
        assert stream.smoothed_q is not None
        frame_modes: list[str] = list(getattr(stream, "frame_modes"))
        ellipse_count = int(getattr(stream, "ellipse_count", 1))
        t0 = time.perf_counter()
        anchor_q = base.choose_anchor_q(stream, source=str(args.keyframe_value_source))
        fit_weights = (
            stream.importance if stream.importance is not None else stream.confidence
        )
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        prefix_cache = base.get_stream_prefix_cache(stream)
        interval_costs = base.get_stream_interval_costs(
            stream, max_gap=int(args.max_gap)
        )
        timings["anchor_setup"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        chosen, objective = base.decode_keyframes_dp(
            stream.smoothed_q,
            fit_weights,
            penalty,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            rewards=stream.keyframe_reward
            if str(args.solver) == "dp_rewarded"
            else None,
            prefix_cache=prefix_cache,
            interval_costs=interval_costs,
        )
        timings["decode_dp"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        chosen = base.refine_keyframes_locally(
            stream.smoothed_q,
            fit_weights,
            chosen,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            radius=int(args.local_search_radius),
            prefix_cache=prefix_cache,
            interval_costs=interval_costs,
        )
        timings["refine_local"] += time.perf_counter() - t0
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
        t0 = time.perf_counter()
        if len(chosen) >= 2 and str(args.value_refine) != "none":
            if str(args.value_refine) == "global_ls":
                key_q = base.refine_keyframe_values_global_ls(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    ridge=float(args.value_refine_ridge),
                )
            elif str(args.value_refine) == "segment_ls":
                key_q = base.refine_keyframe_values_segment_ls(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    ridge=float(args.value_refine_ridge),
                )
            elif str(args.value_refine) == "residual_nudge":
                key_q = base.refine_keyframe_values_residual_nudge(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    damping=float(args.value_refine_damping),
                )
        timings["value_refine"] += time.perf_counter() - t0
        work_items.append(
            {
                "stream": stream,
                "frame_modes": frame_modes,
                "ellipse_count": ellipse_count,
                "fit_weights": fit_weights,
                "target_q": target_q,
                "chosen": chosen,
                "objective": objective,
                "key_q": key_q,
            }
        )
    t0 = time.perf_counter()
    k2_groups: dict[tuple[str, int, tuple[int, ...]], list[dict[str, object]]] = {}
    for item in work_items:
        stream = item["stream"]
        if int(item["ellipse_count"]) == 2:
            key = (
                str(stream.track_id),
                int(stream.run_id),
                tuple(int(frame) for frame in stream.frame_numbers.tolist()),
            )
            k2_groups.setdefault(key, []).append(item)
    handled: set[int] = set()
    for group_items in k2_groups.values():
        if len(group_items) == 2:
            kftrackk_enforce_union_frame_recall_target(group_items, args, disk_samples)
            handled.update(id(item) for item in group_items)
    for item in work_items:
        if id(item) in handled:
            continue
        stream = item["stream"]
        key_q, dense_info = dense_base.enforce_dense_recall_target(
            stream=stream,
            key_q=item["key_q"],
            chosen=item["chosen"],
            args=args,
            disk_samples=disk_samples,
        )
        item["key_q"] = key_q
        item["dense_info"] = dense_info
    timings["dense_recall_enforce"] += time.perf_counter() - t0
    for item in work_items:
        stream = item["stream"]
        frame_modes = item["frame_modes"]
        ellipse_count = int(item["ellipse_count"])
        fit_weights = item["fit_weights"]
        target_q = item["target_q"]
        chosen = item["chosen"]
        objective = float(item["objective"])
        key_q = item["key_q"]
        dense_info = item["dense_info"]
        t0 = time.perf_counter()
        interp_q = base.interpolate_from_key_values(
            key_q, chosen, len(stream.frame_numbers)
        )
        dense_state = base.q_to_state(
            interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
        )
        raw_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        weighted_rmse = float(
            math.sqrt(
                np.average(
                    np.sum((interp_q - raw_q) ** 2, axis=1),
                    weights=np.maximum(fit_weights, 1e-06),
                )
            )
        )
        segment_rows.append(
            {
                "stream_id": stream.stream_id,
                "track_id": stream.track_id,
                "mode": stream.mode,
                "run_id": stream.run_id,
                "slot_id": stream.slot_id,
                "ellipse_count": ellipse_count,
                "frame_count": int(len(stream.frame_numbers)),
                "keyframe_count": int(len(chosen)),
                "keyframe_ratio": float(
                    len(chosen) / max(len(stream.frame_numbers), 1)
                ),
                "objective": float(objective),
                "weighted_param_rmse": weighted_rmse,
                "dense_recall_before": float(dense_info["dense_recall_before"]),
                "dense_recall_after": float(dense_info["dense_recall_after"]),
                "inflate_log_delta": float(dense_info["inflate_log_delta"]),
                "dense_recall_attained": int(bool(dense_info["dense_recall_attained"])),
                "source_area_sum": float(dense_info["source_area_sum"]),
                "source_modes": ",".join(sorted(set(frame_modes))),
            }
        )
        key_states = base.q_to_state(
            key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
        )
        timings["interpolate_and_state"] += time.perf_counter() - t0
        chosen_set = set(chosen)
        t0 = time.perf_counter()
        for key_idx, local_idx in enumerate(chosen):
            keyframe_rows.append(
                {
                    "stream_id": stream.stream_id,
                    "track_id": stream.track_id,
                    "mode": frame_modes[local_idx],
                    "run_id": stream.run_id,
                    "slot_id": stream.slot_id,
                    "ellipse_count": ellipse_count,
                    "frame": int(stream.frame_numbers[local_idx]),
                    "ellipse": key_states[key_idx].tolist(),
                }
            )
        for local_idx, frame in enumerate(stream.frame_numbers):
            dense_rows.append(
                {
                    "stream_id": stream.stream_id,
                    "track_id": stream.track_id,
                    "mode": frame_modes[local_idx],
                    "run_id": stream.run_id,
                    "slot_id": stream.slot_id,
                    "ellipse_count": ellipse_count,
                    "frame": int(frame),
                    "ellipse": dense_state[local_idx].tolist(),
                    "is_keyframe": int(local_idx in chosen_set),
                }
            )
        timings["emit_rows"] += time.perf_counter() - t0
    timings["stream_loop_total"] = time.perf_counter() - total_t0
    return (keyframe_rows, dense_rows, segment_rows, timings)


def kftrackk_merge_dense_rows_to_union_track_k(
    dense_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in dense_rows:
        key = (str(row["track_id"]), int(row["run_id"]), int(row["frame"]))
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "track_id": str(row["track_id"]),
                "mode": str(row["mode"]),
                "run_id": int(row["run_id"]),
                "frame": int(row["frame"]),
                "ellipse_params": [],
                "has_keyframe": 0,
            }
            grouped[key] = entry
        elif entry["mode"] != str(row["mode"]):
            entry["mode"] = "MIXED"
        entry["ellipse_params"].append((int(row["slot_id"]), row["ellipse"]))
        entry["has_keyframe"] = int(
            max(int(entry["has_keyframe"]), int(row["is_keyframe"]))
        )
    merged: list[dict[str, object]] = []
    for key in sorted(
        grouped.keys(), key=lambda item: (int(item[0]), item[2], item[1])
    ):
        entry = grouped[key]
        ellipses = [
            ellipse
            for _slot, ellipse in sorted(
                entry["ellipse_params"], key=lambda item: item[0]
            )
        ]
        merged.append(
            {
                "track_id": entry["track_id"],
                "mode": entry["mode"],
                "run_id": entry["run_id"],
                "frame": entry["frame"],
                "ellipse_params": ellipses,
                "has_keyframe": entry["has_keyframe"],
            }
        )
    return merged


def kftrackk_main(argv: list[str] | None = None) -> None:
    args = kftrackk_parse_args(argv)
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_sec: dict[str, float] = {}
    t0 = time.perf_counter()
    rows = base.load_metric_rows(
        input_metrics,
        confidence_floor=float(args.confidence_floor),
        error_scale=float(args.error_scale),
    )
    timing_sec["load_metric_rows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    streams = kftrackk_build_stream_segments_track_k(args, rows)
    timing_sec["build_stream_segments_track_k"] = time.perf_counter() - t0
    if int(args.max_streams) > 0:
        streams = streams[: int(args.max_streams)]
    t0 = time.perf_counter()
    for stream in streams:
        base.smooth_stream_segment(stream, args)
        base.derive_stream_importance(stream, args)
    timing_sec["smooth_and_importance"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[base.StreamSegment] = []
        for stream in streams:
            broken_streams.extend(base.split_stream_on_breaks(stream, args))
        streams = broken_streams
    timing_sec["auto_break_streams"] = time.perf_counter() - t0
    target_ratio = kftrackk_resolve_target_ratio(args)
    t0 = time.perf_counter()
    penalty, ratio_summary = base.find_penalty_for_target_ratio(
        streams,
        target_ratio=target_ratio,
        fallback_penalty=0.5 if float(args.lambda_all) <= 0 else float(args.lambda_all),
        min_gap=int(args.min_gap),
        max_gap=int(args.max_gap),
        search_iters=int(args.lambda_search_iters),
        use_rewards=str(args.solver) == "dp_rewarded",
    )
    timing_sec["find_penalty_for_target_ratio"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    (
        keyframe_rows,
        dense_rows,
        segment_rows,
        inner_timings,
    ) = kftrackk_optimize_streams_track_k_dense_recall(streams, penalty, args)
    timing_sec["optimize_streams_track_k_dense_recall"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense_union_rows = kftrackk_merge_dense_rows_to_union_track_k(dense_rows)
    timing_sec["merge_dense_rows_to_union_track_k"] = time.perf_counter() - t0
    keyframe_rows.sort(
        key=lambda row: (
            int(str(row["track_id"])),
            int(row["frame"]),
            int(row["slot_id"]),
        )
    )
    dense_rows.sort(
        key=lambda row: (
            int(str(row["track_id"])),
            int(row["frame"]),
            int(row["slot_id"]),
        )
    )
    segment_rows.sort(
        key=lambda row: (
            int(str(row["track_id"])),
            int(row["run_id"]),
            int(row["slot_id"]),
        )
    )
    t0 = time.perf_counter()
    base.write_json(output_dir / "final_keyframes.json", keyframe_rows)
    base.write_json(output_dir / "interpolated_union.json", dense_union_rows)
    base.write_csv(
        output_dir / "stream_segments.csv",
        segment_rows,
        [
            "stream_id",
            "track_id",
            "mode",
            "run_id",
            "slot_id",
            "ellipse_count",
            "frame_count",
            "keyframe_count",
            "keyframe_ratio",
            "objective",
            "weighted_param_rmse",
            "dense_recall_before",
            "dense_recall_after",
            "inflate_log_delta",
            "dense_recall_attained",
            "source_area_sum",
            "source_modes",
        ],
    )
    timing_sec["write_outputs"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    total_source_area = sum((float(row["source_area_sum"]) for row in segment_rows))
    dense_recall_before = sum(
        (
            float(row["dense_recall_before"]) * float(row["source_area_sum"])
            for row in segment_rows
        )
    ) / max(total_source_area, 1e-08)
    dense_recall_after = sum(
        (
            float(row["dense_recall_after"]) * float(row["source_area_sum"])
            for row in segment_rows
        )
    ) / max(total_source_area, 1e-08)
    inflated_segments = sum(
        (1 for row in segment_rows if float(row["inflate_log_delta"]) > 1e-08)
    )
    unattained_segments = sum(
        (1 for row in segment_rows if int(row["dense_recall_attained"]) == 0)
    )
    mixed_mode_streams = sum((1 for row in segment_rows if str(row["mode"]) == "MIXED"))
    timing_sec["summarize_outputs"] = time.perf_counter() - t0
    summary = {
        "input_metrics_csv": str(input_metrics),
        "stream_count": int(len(streams)),
        "row_count": int(len(rows)),
        "ratio_summary": {
            "lambda": float(penalty),
            "target_ratio": float(target_ratio),
            **ratio_summary,
        },
        "total_keyframe_rows": int(len(keyframe_rows)),
        "total_dense_rows": int(len(dense_rows)),
        "total_union_rows": int(len(dense_union_rows)),
        "dense_recall_summary": {
            "target": float(args.dense_recall_target),
            "global_before": float(dense_recall_before),
            "global_after": float(dense_recall_after),
            "inflated_segments": int(inflated_segments),
            "unattained_segments": int(unattained_segments),
        },
        "segmentation_summary": {
            "split_policy": "track_and_ellipse_count_only",
            "mixed_mode_streams": int(mixed_mode_streams),
        },
        "timing_sec": {
            **{key: float(val) for key, val in timing_sec.items()},
            "opt_inner": {key: float(val) for key, val in inner_timings.items()},
        },
        "settings": {
            "solver": str(args.solver),
            "smooth_alpha": float(args.smooth_alpha),
            "value_refine": str(args.value_refine),
            "keyframe_value_source": str(args.keyframe_value_source),
            "dense_recall_target": float(args.dense_recall_target),
            "dense_recall_samples": int(args.dense_recall_samples),
            "dense_recall_max_inflate_log": float(args.dense_recall_max_inflate_log),
            "dense_recall_search_iters": int(args.dense_recall_search_iters),
            "min_gap": int(args.min_gap),
            "max_gap": int(args.max_gap),
        },
    }
    base.write_json(output_dir / "summary.json", summary)


kftrackk_module = _register_inline_module(
    "optimize_keyframes_trackk_dense_recall_standalone",
    {
        "parse_args": "kftrackk_parse_args",
        "resolve_target_ratio": "kftrackk_resolve_target_ratio",
        "split_runs_track_k": "kftrackk_split_runs_track_k",
        "build_stream_segments_track_k": "kftrackk_build_stream_segments_track_k",
        "optimize_streams_track_k_dense_recall": "kftrackk_optimize_streams_track_k_dense_recall",
        "merge_dense_rows_to_union_track_k": "kftrackk_merge_dense_rows_to_union_track_k",
        "main": "kftrackk_main",
    },
)
