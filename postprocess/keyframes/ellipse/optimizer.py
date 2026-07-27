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
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.linalg import solveh_banded


def kfbase_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone keyframe optimizer for routed K1/K2 ellipse sequences."
    )
    parser.add_argument("--input-metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-k1-ratio", type=float, default=0.1)
    parser.add_argument("--target-k2-ratio", type=float, default=0.16)
    parser.add_argument(
        "--solver",
        choices=[
            "dp",
            "dp_rewarded",
            "dp_candidates",
            "uniform_refine",
            "greedy_split",
            "bottom_up_merge",
            "best_first_split",
            "rdp_quantile",
            "trend_knots",
            "event_triggered",
        ],
        default="dp",
    )
    parser.add_argument("--lambda-k1", type=float, default=-1.0)
    parser.add_argument("--lambda-k2", type=float, default=-1.0)
    parser.add_argument("--lambda-search-iters", type=int, default=16)
    parser.add_argument("--smooth-alpha", type=float, default=6.0)
    parser.add_argument("--confidence-floor", type=float, default=0.18)
    parser.add_argument("--error-scale", type=float, default=4000.0)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--max-gap", type=int, default=16)
    parser.add_argument("--local-search-radius", type=int, default=2)
    parser.add_argument(
        "--value-refine",
        choices=["none", "global_ls", "segment_ls", "residual_nudge"],
        default="none",
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
    parser.add_argument("--candidate-multiplier", type=float, default=4.0)
    parser.add_argument("--candidate-min-separation", type=int, default=2)
    parser.add_argument("--candidate-uniform-support", type=int, default=6)
    parser.add_argument("--rdp-quantile", type=float, default=0.9)
    parser.add_argument("--event-quantile", type=float, default=0.9)
    parser.add_argument("--event-search-iters", type=int, default=16)
    parser.add_argument(
        "--keyframe-value-source",
        choices=["smoothed", "raw", "confidence_blend"],
        default="smoothed",
    )
    parser.add_argument("--k2-slot-center-weight", type=float, default=1.0)
    parser.add_argument("--k2-slot-size-weight", type=float, default=0.65)
    parser.add_argument("--k2-slot-angle-weight", type=float, default=0.2)
    parser.add_argument("--max-streams", type=int, default=-1)
    return parser.parse_args()


@dataclass
class kfbase_MetricRow:
    frame: int
    track_id: str
    mode: str
    weighted_error: float
    recall: float
    precision: float
    iou: float
    ellipse_params: list[list[float]]


@dataclass
class kfbase_StreamSegment:
    stream_id: str
    track_id: str
    mode: str
    run_id: int
    slot_id: int
    frame_numbers: np.ndarray
    raw_states: np.ndarray
    confidence: np.ndarray
    weighted_error: np.ndarray
    raw_q: np.ndarray | None = None
    smoothed_q: np.ndarray | None = None
    importance: np.ndarray | None = None
    keyframe_reward: np.ndarray | None = None
    break_signal: np.ndarray | None = None
    transform_scale: float = 1.0
    theta_scale: float = 1.0
    prefix_cache: "PrefixCostCache | None" = None
    interval_costs: np.ndarray | None = None
    interval_costs_max_gap: int = 0


def kfbase_canonicalize_ellipse(values: Iterable[float]) -> list[float]:
    cx, cy, a, b, theta = [float(x) for x in values]
    if b > a:
        a, b = (b, a)
        theta += 90.0
    theta = (theta + 90.0) % 180.0 - 90.0
    return [cx, cy, max(a, 1e-06), max(b, 1e-06), theta]


def kfbase_circular_angle_distance_deg(a: float, b: float) -> float:
    diff = abs((a - b + 90.0) % 180.0 - 90.0)
    return min(diff, 180.0 - diff)


def kfbase_unwrap_angles_deg(theta_deg: np.ndarray) -> np.ndarray:
    if theta_deg.size == 0:
        return theta_deg.astype(np.float64)
    out = np.zeros_like(theta_deg, dtype=np.float64)
    out[0] = float(theta_deg[0])
    for idx in range(1, len(theta_deg)):
        base = float(theta_deg[idx])
        candidates = [base - 180.0, base, base + 180.0]
        out[idx] = min(candidates, key=lambda v: abs(v - out[idx - 1]))
    return out


def kfbase_compute_confidence(
    row: kfbase_MetricRow, floor: float, error_scale: float
) -> float:
    quality = (
        max(row.iou, 0.0001) ** 0.55
        * max(row.recall, 0.0001) ** 0.3
        * max(row.precision, 0.0001) ** 0.15
        * math.exp(-max(row.weighted_error, 0.0) / max(error_scale, 1e-06))
    )
    return float(min(1.0, max(floor, quality)))


def kfbase_load_metric_rows(
    path: Path, confidence_floor: float, error_scale: float
) -> list[kfbase_MetricRow]:
    rows: list[kfbase_MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ellipse_params = json.loads(row["ellipse_params"])
            rows.append(
                kfbase_MetricRow(
                    frame=int(row["frame"]),
                    track_id=str(row["track_id"]),
                    mode=str(row["mode"]).upper(),
                    weighted_error=float(row["weighted_error"]),
                    recall=float(row["recall"]),
                    precision=float(row["precision"]),
                    iou=float(row["iou"]),
                    ellipse_params=[
                        kfbase_canonicalize_ellipse(x) for x in ellipse_params
                    ],
                )
            )
    rows.sort(key=lambda r: (int(r.track_id), r.frame))
    return rows


def kfbase_split_runs(rows: list[kfbase_MetricRow]) -> list[list[kfbase_MetricRow]]:
    grouped: dict[str, list[kfbase_MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    runs: list[list[kfbase_MetricRow]] = []
    for _track_id, track_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        track_rows.sort(key=lambda r: r.frame)
        current: list[kfbase_MetricRow] = []
        prev: kfbase_MetricRow | None = None
        for row in track_rows:
            split = (
                prev is None
                or row.mode != prev.mode
                or len(row.ellipse_params) != len(prev.ellipse_params)
                or (row.frame != prev.frame + 1)
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


def kfbase_ellipse_pair_cost(
    left: list[float],
    right: list[float],
    center_weight: float,
    size_weight: float,
    angle_weight: float,
) -> float:
    lc = np.asarray(left[:2], dtype=np.float64)
    rc = np.asarray(right[:2], dtype=np.float64)
    la, lb = (float(left[2]), float(left[3]))
    ra, rb = (float(right[2]), float(right[3]))
    center_scale = max(
        math.sqrt(max(la * lb, 1e-06)), math.sqrt(max(ra * rb, 1e-06)), 1.0
    )
    center_term = float(np.linalg.norm(lc - rc) / center_scale)
    size_term = abs(math.log(max(la, 1e-06) / max(ra, 1e-06))) + abs(
        math.log(max(lb, 1e-06) / max(rb, 1e-06))
    )
    ecc_left = 1.0 - min(la, lb) / max(la, lb)
    ecc_right = 1.0 - min(ra, rb) / max(ra, rb)
    ecc = max(0.0, 0.5 * (ecc_left + ecc_right))
    angle_term = (
        kfbase_circular_angle_distance_deg(float(left[4]), float(right[4]))
        / 45.0
        * max(0.1, ecc)
    )
    return (
        center_weight * center_term
        + size_weight * size_term
        + angle_weight * angle_term
    )


def kfbase_stabilize_k2_slots(
    rows: list[kfbase_MetricRow],
    center_weight: float,
    size_weight: float,
    angle_weight: float,
) -> list[list[list[float]]]:
    stabilized: list[list[list[float]]] = []
    prev: list[list[float]] | None = None
    for row in rows:
        current = [list(ellipse) for ellipse in row.ellipse_params]
        if prev is not None and len(prev) == 2 and (len(current) == 2):
            keep_cost = kfbase_ellipse_pair_cost(
                prev[0], current[0], center_weight, size_weight, angle_weight
            ) + kfbase_ellipse_pair_cost(
                prev[1], current[1], center_weight, size_weight, angle_weight
            )
            swap_cost = kfbase_ellipse_pair_cost(
                prev[0], current[1], center_weight, size_weight, angle_weight
            ) + kfbase_ellipse_pair_cost(
                prev[1], current[0], center_weight, size_weight, angle_weight
            )
            if swap_cost < keep_cost:
                current = [current[1], current[0]]
        stabilized.append(current)
        prev = current
    return stabilized


def kfbase_build_stream_segments(
    args: argparse.Namespace, rows: list[kfbase_MetricRow]
) -> list[kfbase_StreamSegment]:
    runs = kfbase_split_runs(rows)
    streams: list[kfbase_StreamSegment] = []
    for run_id, run_rows in enumerate(runs):
        mode = run_rows[0].mode
        if mode == "K2":
            stabilized = kfbase_stabilize_k2_slots(
                run_rows,
                center_weight=float(args.k2_slot_center_weight),
                size_weight=float(args.k2_slot_size_weight),
                angle_weight=float(args.k2_slot_angle_weight),
            )
        else:
            stabilized = [[list(row.ellipse_params[0])] for row in run_rows]
        slot_count = len(stabilized[0])
        for slot_id in range(slot_count):
            states = np.asarray(
                [frame_slots[slot_id] for frame_slots in stabilized], dtype=np.float64
            )
            states[:, 4] = kfbase_unwrap_angles_deg(states[:, 4])
            confidence = np.asarray(
                [
                    kfbase_compute_confidence(
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
            streams.append(
                kfbase_StreamSegment(
                    stream_id=f"{run_rows[0].track_id}:{mode}:run{run_id}:slot{slot_id}",
                    track_id=run_rows[0].track_id,
                    mode=mode,
                    run_id=run_id,
                    slot_id=slot_id,
                    frame_numbers=np.asarray(
                        [row.frame for row in run_rows], dtype=np.int32
                    ),
                    raw_states=states,
                    confidence=confidence,
                    weighted_error=weighted_error,
                )
            )
    return streams


def kfbase_build_second_difference_matrix(length: int) -> np.ndarray:
    if length < 3:
        return np.zeros((0, length), dtype=np.float64)
    mat = np.zeros((length - 2, length), dtype=np.float64)
    for idx in range(length - 2):
        mat[idx, idx] = 1.0
        mat[idx, idx + 1] = -2.0
        mat[idx, idx + 2] = 1.0
    return mat


def kfbase_solve_second_difference_system(
    weights: np.ndarray,
    values: np.ndarray,
    smooth_alpha: float,
) -> np.ndarray:
    """Solve ``(diag(w) + alpha * D2.T @ D2) x = diag(w) y`` in O(n).

    ``D2.T @ D2`` is symmetric positive semidefinite with half-bandwidth two.
    Materializing the dense matrix made both memory and factorization grow
    quadratically/cubically with a continuous track's length.  The banded
    representation contains the exact same five diagonals and accepts all
    value dimensions as simultaneous right-hand sides.
    """

    length = len(weights)
    if length < 3:
        return np.asarray(values, dtype=np.float64).copy()
    alpha = float(smooth_alpha)
    main = np.asarray(weights, dtype=np.float64).copy()
    main[:-2] += alpha
    main[1:-1] += 4.0 * alpha
    main[2:] += alpha
    first_upper = np.zeros(length - 1, dtype=np.float64)
    first_upper[:-1] -= 2.0 * alpha
    first_upper[1:] -= 2.0 * alpha
    second_upper = np.full(length - 2, alpha, dtype=np.float64)
    banded = np.zeros((3, length), dtype=np.float64)
    banded[2] = main
    banded[1, 1:] = first_upper
    banded[0, 2:] = second_upper
    rhs = np.asarray(weights, dtype=np.float64)[:, None] * np.asarray(
        values, dtype=np.float64
    )
    return solveh_banded(
        banded,
        rhs,
        lower=False,
        overwrite_ab=True,
        overwrite_b=True,
        check_finite=False,
    )


def kfbase_state_to_q(
    states: np.ndarray, theta_weight_floor: float
) -> tuple[np.ndarray, float, float]:
    scale = float(np.median(np.sqrt(np.maximum(states[:, 2] * states[:, 3], 1e-06))))
    scale = max(scale, 1.0)
    eccentricity = 1.0 - np.minimum(states[:, 2], states[:, 3]) / np.maximum(
        states[:, 2], states[:, 3]
    )
    theta_scale = max(
        float(theta_weight_floor),
        float(np.median(eccentricity)) if eccentricity.size else 1.0,
    )
    q = np.column_stack(
        [
            states[:, 0] / scale,
            states[:, 1] / scale,
            np.log(np.maximum(states[:, 2], 1e-06)),
            np.log(np.maximum(states[:, 3], 1e-06)),
            np.deg2rad(states[:, 4]) * theta_scale,
        ]
    ).astype(np.float64)
    return (q, scale, theta_scale)


def kfbase_q_to_state(q: np.ndarray, scale: float, theta_scale: float) -> np.ndarray:
    state = np.zeros((q.shape[0], 5), dtype=np.float64)
    state[:, 0] = q[:, 0] * scale
    state[:, 1] = q[:, 1] * scale
    state[:, 2] = np.exp(q[:, 2])
    state[:, 3] = np.exp(q[:, 3])
    state[:, 4] = np.rad2deg(q[:, 4] / max(theta_scale, 1e-06))
    for idx in range(len(state)):
        state[idx] = np.asarray(
            kfbase_canonicalize_ellipse(state[idx]), dtype=np.float64
        )
    return state


def kfbase_smooth_stream_segment(
    stream: kfbase_StreamSegment, args: argparse.Namespace
) -> None:
    q, scale, theta_scale = kfbase_state_to_q(
        stream.raw_states, theta_weight_floor=float(args.theta_weight_floor)
    )
    stream.raw_q = q
    length = len(q)
    if length < int(args.min_segment_length):
        stream.smoothed_q = q
        stream.transform_scale = scale
        stream.theta_scale = theta_scale
        return
    weights = np.maximum(stream.confidence, float(args.confidence_floor))
    smoothed = kfbase_solve_second_difference_system(
        weights,
        q,
        float(args.smooth_alpha),
    )
    stream.smoothed_q = smoothed
    stream.transform_scale = scale
    stream.theta_scale = theta_scale


def kfbase_choose_anchor_q(stream: kfbase_StreamSegment, source: str) -> np.ndarray:
    assert stream.raw_q is not None
    assert stream.smoothed_q is not None
    if source == "raw":
        return stream.raw_q
    if source == "confidence_blend":
        blend = np.clip(stream.confidence, 0.0, 1.0)[:, None]
        return blend * stream.raw_q + (1.0 - blend) * stream.smoothed_q
    return stream.smoothed_q


def kfbase_robust_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    p90 = float(np.percentile(values, 90.0))
    scale = max(p90, 1e-06)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float64)


def kfbase_compute_curvature_signal(q: np.ndarray) -> np.ndarray:
    length = len(q)
    signal = np.zeros(length, dtype=np.float64)
    if length < 3:
        return signal
    second = q[2:] - 2.0 * q[1:-1] + q[:-2]
    signal[1:-1] = np.linalg.norm(second, axis=1)
    return signal


def kfbase_derive_stream_importance(
    stream: kfbase_StreamSegment, args: argparse.Namespace
) -> None:
    assert stream.raw_q is not None
    assert stream.smoothed_q is not None
    error_norm = kfbase_robust_normalize(np.maximum(stream.weighted_error, 0.0))
    curvature_norm = kfbase_robust_normalize(
        kfbase_compute_curvature_signal(stream.smoothed_q)
    )
    importance = stream.confidence * (
        1.0
        + float(args.weight_error_gain) * error_norm
        + float(args.weight_curvature_gain) * curvature_norm
    )
    stream.importance = np.clip(importance, 1e-06, float(args.importance_cap)).astype(
        np.float64
    )
    reward = (
        float(args.reward_error_gain) * error_norm
        + float(args.reward_curvature_gain) * curvature_norm
    )
    stream.keyframe_reward = np.clip(reward, 0.0, float(args.reward_cap)).astype(
        np.float64
    )
    stream.break_signal = (curvature_norm + 0.5 * error_norm).astype(np.float64)


def kfbase_split_stream_on_breaks(
    stream: kfbase_StreamSegment, args: argparse.Namespace
) -> list[kfbase_StreamSegment]:
    threshold = float(args.auto_break_threshold)
    if (
        threshold < 0.0
        or stream.break_signal is None
        or len(stream.frame_numbers) < int(args.auto_break_min_length) * 2
    ):
        return [stream]
    min_len = int(args.auto_break_min_length)
    min_sep = int(args.auto_break_min_separation)
    candidates: list[int] = []
    last_break = -(10**9)
    signal = stream.break_signal
    for idx in range(min_len, len(signal) - min_len):
        if signal[idx] < threshold:
            continue
        left = signal[idx - 1] if idx - 1 >= 0 else -1.0
        right = signal[idx + 1] if idx + 1 < len(signal) else -1.0
        if signal[idx] < left or signal[idx] < right:
            continue
        if idx - last_break < min_sep:
            if candidates and signal[idx] > signal[candidates[-1]]:
                candidates[-1] = idx
                last_break = idx
            continue
        candidates.append(idx)
        last_break = idx
    if not candidates:
        return [stream]
    bounds = [0] + candidates + [len(stream.frame_numbers)]
    out: list[kfbase_StreamSegment] = []
    for seg_idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        if hi - lo <= 0:
            continue
        out.append(
            kfbase_StreamSegment(
                stream_id=f"{stream.stream_id}:break{seg_idx}",
                track_id=stream.track_id,
                mode=stream.mode,
                run_id=stream.run_id,
                slot_id=stream.slot_id,
                frame_numbers=stream.frame_numbers[lo:hi].copy(),
                raw_states=stream.raw_states[lo:hi].copy(),
                confidence=stream.confidence[lo:hi].copy(),
                weighted_error=stream.weighted_error[lo:hi].copy(),
                raw_q=None if stream.raw_q is None else stream.raw_q[lo:hi].copy(),
                smoothed_q=None
                if stream.smoothed_q is None
                else stream.smoothed_q[lo:hi].copy(),
                importance=None
                if stream.importance is None
                else stream.importance[lo:hi].copy(),
                keyframe_reward=None
                if stream.keyframe_reward is None
                else stream.keyframe_reward[lo:hi].copy(),
                break_signal=None
                if stream.break_signal is None
                else stream.break_signal[lo:hi].copy(),
                transform_scale=stream.transform_scale,
                theta_scale=stream.theta_scale,
            )
        )
    return out if out else [stream]


@dataclass
class kfbase_PrefixCostCache:
    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    g: np.ndarray


def kfbase_build_prefix_cache(
    q: np.ndarray, weights: np.ndarray
) -> kfbase_PrefixCostCache:
    weights = weights.astype(np.float64)
    t = np.arange(len(q), dtype=np.float64)
    s0 = np.zeros(len(q) + 1, dtype=np.float64)
    s1 = np.zeros(len(q) + 1, dtype=np.float64)
    s2 = np.zeros(len(q) + 1, dtype=np.float64)
    v0 = np.zeros((len(q) + 1, q.shape[1]), dtype=np.float64)
    v1 = np.zeros((len(q) + 1, q.shape[1]), dtype=np.float64)
    g = np.zeros(len(q) + 1, dtype=np.float64)
    s0[1:] = np.cumsum(weights)
    s1[1:] = np.cumsum(weights * t)
    s2[1:] = np.cumsum(weights * t**2)
    v0[1:] = np.cumsum(weights[:, None] * q, axis=0)
    v1[1:] = np.cumsum((weights * t)[:, None] * q, axis=0)
    g[1:] = np.cumsum(weights * np.sum(q * q, axis=1))
    return kfbase_PrefixCostCache(s0=s0, s1=s1, s2=s2, v0=v0, v1=v1, g=g)


def kfbase_get_stream_prefix_cache(
    stream: kfbase_StreamSegment,
) -> kfbase_PrefixCostCache:
    cache = stream.prefix_cache
    if cache is not None:
        return cache
    assert stream.smoothed_q is not None
    weights = stream.importance if stream.importance is not None else stream.confidence
    cache = kfbase_build_prefix_cache(stream.smoothed_q, weights)
    stream.prefix_cache = cache
    return cache


def kfbase_get_stream_interval_costs(
    stream: kfbase_StreamSegment, max_gap: int
) -> np.ndarray:
    cached = stream.interval_costs
    if cached is not None and int(stream.interval_costs_max_gap) == int(max_gap):
        return cached
    assert stream.smoothed_q is not None
    q = stream.smoothed_q
    cache = kfbase_get_stream_prefix_cache(stream)
    length = len(q)
    costs = np.full((length, int(max_gap)), np.inf, dtype=np.float64)
    # Evaluate a full diagonal (one fixed gap) at a time.  Each expression is
    # identical to ``interval_surrogate_cost`` but operates on all admissible
    # intervals in NumPy instead of entering Python O(n * max_gap) times.
    for gap in range(1, min(int(max_gap), length - 1) + 1):
        end = np.arange(gap, length, dtype=np.int64)
        start = end - gap
        span = float(gap)
        s0 = cache.s0[end + 1] - cache.s0[start]
        s1 = cache.s1[end + 1] - cache.s1[start]
        s2 = cache.s2[end + 1] - cache.s2[start]
        v0 = cache.v0[end + 1] - cache.v0[start]
        v1 = cache.v1[end + 1] - cache.v1[start]
        g = cache.g[end + 1] - cache.g[start]
        start_f = start.astype(np.float64)
        end_f = end.astype(np.float64)
        a_vec = (end_f[:, None] * v0 - v1) / span
        b_vec = (v1 - start_f[:, None] * v0) / span
        alpha = (end_f * end_f * s0 - 2.0 * end_f * s1 + s2) / (span * span)
        beta = (s2 - 2.0 * start_f * s1 + start_f * start_f * s0) / (span * span)
        gamma = (-s2 + (start_f + end_f) * s1 - start_f * end_f * s0) / (span * span)
        qi = q[start]
        qj = q[end]
        values = (
            g
            - 2.0 * np.einsum("ij,ij->i", qi, a_vec)
            - 2.0 * np.einsum("ij,ij->i", qj, b_vec)
            + alpha * np.einsum("ij,ij->i", qi, qi)
            + beta * np.einsum("ij,ij->i", qj, qj)
            + 2.0 * gamma * np.einsum("ij,ij->i", qi, qj)
        )
        costs[end, gap - 1] = np.maximum(values, 0.0)
    stream.interval_costs = costs
    stream.interval_costs_max_gap = int(max_gap)
    return costs


def kfbase_interval_surrogate_cost(
    cache: kfbase_PrefixCostCache, q: np.ndarray, start: int, end: int
) -> float:
    if end <= start:
        return 0.0
    length = float(end - start)
    s0 = cache.s0[end + 1] - cache.s0[start]
    s1 = cache.s1[end + 1] - cache.s1[start]
    s2 = cache.s2[end + 1] - cache.s2[start]
    v0 = cache.v0[end + 1] - cache.v0[start]
    v1 = cache.v1[end + 1] - cache.v1[start]
    g = cache.g[end + 1] - cache.g[start]
    qi = q[start]
    qj = q[end]
    a_vec = (end * v0 - v1) / length
    b_vec = (v1 - start * v0) / length
    alpha = (end * end * s0 - 2.0 * end * s1 + s2) / (length * length)
    beta = (s2 - 2.0 * start * s1 + start * start * s0) / (length * length)
    gamma = (-s2 + (start + end) * s1 - start * end * s0) / (length * length)
    cost = (
        g
        - 2.0 * float(np.dot(qi, a_vec))
        - 2.0 * float(np.dot(qj, b_vec))
        + alpha * float(np.dot(qi, qi))
        + beta * float(np.dot(qj, qj))
        + 2.0 * gamma * float(np.dot(qi, qj))
    )
    return float(max(cost, 0.0))


def kfbase_decode_keyframes_dp(
    q: np.ndarray,
    weights: np.ndarray,
    keyframe_penalty: float,
    min_gap: int,
    max_gap: int,
    rewards: np.ndarray | None = None,
    prefix_cache: kfbase_PrefixCostCache | None = None,
    interval_costs: np.ndarray | None = None,
) -> tuple[list[int], float]:
    length = len(q)
    if length <= 2:
        return (list(range(length)), 0.0)
    cache = (
        prefix_cache
        if prefix_cache is not None
        else kfbase_build_prefix_cache(q, weights)
    )
    dp = np.full(length, np.inf, dtype=np.float64)
    back = np.full(length, -1, dtype=np.int32)
    dp[0] = -float(keyframe_penalty)
    for end in range(1, length):
        start_low = max(0, end - max_gap)
        starts = np.arange(start_low, end, dtype=np.int32)
        gaps = end - starts
        valid = np.ones(len(starts), dtype=bool)
        if end != length - 1:
            valid &= gaps >= int(min_gap)
        else:
            valid &= (starts == 0) | (gaps >= int(min_gap))
        starts = starts[valid]
        gaps = gaps[valid]
        if starts.size == 0:
            continue
        if interval_costs is not None and int(np.max(gaps)) <= interval_costs.shape[1]:
            candidate_intervals = interval_costs[end, gaps - 1]
        else:
            candidate_intervals = np.asarray(
                [
                    kfbase_interval_surrogate_cost(cache, q, int(start), end)
                    for start in starts
                ],
                dtype=np.float64,
            )
        reward_term = 0.0 if rewards is None else float(rewards[end])
        candidate_costs = (
            dp[starts] + candidate_intervals + float(keyframe_penalty) - reward_term
        )
        best_offset = int(np.argmin(candidate_costs))
        dp[end] = float(candidate_costs[best_offset])
        back[end] = int(starts[best_offset])
    if not np.isfinite(dp[-1]):
        return (list(range(length)), float("inf"))
    chosen: list[int] = []
    cursor = length - 1
    while cursor >= 0:
        chosen.append(int(cursor))
        if cursor == 0:
            break
        cursor = int(back[cursor])
        if cursor < 0:
            return (list(range(length)), float("inf"))
    chosen.reverse()
    return (chosen, float(dp[-1]))


def kfbase_refine_keyframes_locally(
    q: np.ndarray,
    weights: np.ndarray,
    chosen: list[int],
    min_gap: int,
    max_gap: int,
    radius: int,
    prefix_cache: kfbase_PrefixCostCache | None = None,
    interval_costs: np.ndarray | None = None,
) -> list[int]:
    if len(chosen) <= 2 or radius <= 0:
        return chosen
    cache = (
        prefix_cache
        if prefix_cache is not None
        else kfbase_build_prefix_cache(q, weights)
    )
    refined = list(chosen)
    for key_idx in range(1, len(refined) - 1):
        left = refined[key_idx - 1]
        current = refined[key_idx]
        right = refined[key_idx + 1]
        best = current
        left_gap = current - left
        right_gap = right - current
        best_cost = (
            float(interval_costs[current, left_gap - 1])
            if interval_costs is not None and left_gap <= interval_costs.shape[1]
            else kfbase_interval_surrogate_cost(cache, q, left, current)
        ) + (
            float(interval_costs[right, right_gap - 1])
            if interval_costs is not None and right_gap <= interval_costs.shape[1]
            else kfbase_interval_surrogate_cost(cache, q, current, right)
        )
        low = max(left + min_gap, current - radius)
        high = min(right - min_gap, current + radius)
        for candidate in range(low, high + 1):
            if candidate == current:
                continue
            if candidate - left > max_gap or right - candidate > max_gap:
                continue
            left_gap = candidate - left
            right_gap = right - candidate
            cost = (
                float(interval_costs[candidate, left_gap - 1])
                if interval_costs is not None and left_gap <= interval_costs.shape[1]
                else kfbase_interval_surrogate_cost(cache, q, left, candidate)
            ) + (
                float(interval_costs[right, right_gap - 1])
                if interval_costs is not None and right_gap <= interval_costs.shape[1]
                else kfbase_interval_surrogate_cost(cache, q, candidate, right)
            )
            if cost < best_cost:
                best = candidate
                best_cost = cost
        refined[key_idx] = best
    return sorted(set(refined))


def kfbase_interpolate_dense_q(q: np.ndarray, keyframes: list[int]) -> np.ndarray:
    if len(keyframes) <= 1:
        return np.repeat(q[[0]], len(q), axis=0)
    out = np.zeros_like(q)
    for idx in range(len(keyframes) - 1):
        left = keyframes[idx]
        right = keyframes[idx + 1]
        if right == left:
            out[left] = q[left]
            continue
        span = right - left
        for pos in range(left, right + 1):
            alpha = (pos - left) / float(span)
            out[pos] = (1.0 - alpha) * q[left] + alpha * q[right]
    return out


def kfbase_refine_keyframe_values_global_ls(
    target_q: np.ndarray,
    base_key_q: np.ndarray,
    keyframes: list[int],
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    key_count = len(keyframes)
    if key_count <= 1:
        return base_key_q
    length, dims = target_q.shape
    diagonal = np.full(key_count, float(ridge), dtype=np.float64)
    first_upper = np.zeros(key_count - 1, dtype=np.float64)
    rhs = float(ridge) * np.asarray(base_key_q, dtype=np.float64)
    for idx in range(key_count - 1):
        left = keyframes[idx]
        right = keyframes[idx + 1]
        if right == left:
            continue
        span = float(right - left)
        # The right endpoint belongs to the following segment.  This matches
        # assignment into the original dense interpolation matrix without
        # counting a shared keyframe twice.
        rows = np.arange(left, right, dtype=np.int64)
        alpha = (rows - left) / span
        left_basis = 1.0 - alpha
        right_basis = alpha
        row_weights = np.maximum(weights[rows], 1e-08)
        weighted_left = row_weights * left_basis
        weighted_right = row_weights * right_basis
        diagonal[idx] += float(np.dot(weighted_left, left_basis))
        diagonal[idx + 1] += float(np.dot(weighted_right, right_basis))
        first_upper[idx] += float(np.dot(weighted_left, right_basis))
        rhs[idx] += weighted_left @ target_q[rows]
        rhs[idx + 1] += weighted_right @ target_q[rows]
    final_row = int(keyframes[-1])
    final_weight = float(max(weights[final_row], 1e-08))
    diagonal[-1] += final_weight
    rhs[-1] += final_weight * target_q[final_row]
    banded = np.zeros((2, key_count), dtype=np.float64)
    banded[1] = diagonal
    banded[0, 1:] = first_upper
    solved = solveh_banded(
        banded,
        rhs,
        lower=False,
        overwrite_ab=True,
        overwrite_b=True,
        check_finite=False,
    )
    if solved.shape != (key_count, dims):
        raise RuntimeError(f"unexpected global LS solution shape: {solved.shape!r}")
    return solved


def kfbase_refine_keyframe_values_segment_ls(
    target_q: np.ndarray,
    base_key_q: np.ndarray,
    keyframes: list[int],
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    key_count = len(keyframes)
    if key_count <= 1:
        return base_key_q
    dims = target_q.shape[1]
    accum = np.zeros_like(base_key_q)
    accum_w = np.zeros(key_count, dtype=np.float64)
    for seg_idx in range(key_count - 1):
        left = keyframes[seg_idx]
        right = keyframes[seg_idx + 1]
        if right <= left:
            continue
        span = float(right - left)
        rows = np.arange(left, right + 1, dtype=np.int32)
        alpha = (rows - left) / span
        basis = np.column_stack([1.0 - alpha, alpha]).astype(np.float64)
        seg_w = np.maximum(weights[rows], 1e-08)
        bw = basis * np.sqrt(seg_w)[:, None]
        gram = bw.T @ bw + float(ridge) * np.eye(2, dtype=np.float64)
        base_pair = np.stack([base_key_q[seg_idx], base_key_q[seg_idx + 1]], axis=0)
        rhs_base = float(ridge) * base_pair
        solved = np.zeros((2, dims), dtype=np.float64)
        for dim in range(dims):
            yw = target_q[rows, dim] * np.sqrt(seg_w)
            rhs = bw.T @ yw + rhs_base[:, dim]
            solved[:, dim] = np.linalg.solve(gram, rhs)
        seg_mass = float(np.sum(seg_w))
        accum[seg_idx] += seg_mass * solved[0]
        accum[seg_idx + 1] += seg_mass * solved[1]
        accum_w[seg_idx] += seg_mass
        accum_w[seg_idx + 1] += seg_mass
    out = base_key_q.copy()
    valid = accum_w > 0
    out[valid] = accum[valid] / accum_w[valid, None]
    return out


def kfbase_refine_keyframe_values_residual_nudge(
    target_q: np.ndarray,
    base_key_q: np.ndarray,
    keyframes: list[int],
    weights: np.ndarray,
    damping: float,
) -> np.ndarray:
    key_count = len(keyframes)
    if key_count <= 1:
        return base_key_q
    interp_q = kfbase_interpolate_from_key_values(base_key_q, keyframes, len(target_q))
    residual = target_q - interp_q
    a = np.zeros((len(target_q), key_count), dtype=np.float64)
    for idx in range(key_count - 1):
        left = keyframes[idx]
        right = keyframes[idx + 1]
        if right <= left:
            continue
        span = float(right - left)
        for pos in range(left, right + 1):
            alpha = (pos - left) / span
            a[pos, idx] = 1.0 - alpha
            a[pos, idx + 1] = alpha
    out = base_key_q.copy()
    damping = float(np.clip(damping, 0.0, 1.5))
    for key_idx in range(key_count):
        coeff = weights * a[:, key_idx]
        denom = float(np.sum(coeff))
        if denom <= 1e-08:
            continue
        delta = np.sum(coeff[:, None] * residual, axis=0) / denom
        out[key_idx] = base_key_q[key_idx] + damping * delta
    return out


def kfbase_interpolate_from_key_values(
    key_q: np.ndarray, keyframes: list[int], length: int
) -> np.ndarray:
    if len(keyframes) <= 1:
        return np.repeat(key_q[[0]], length, axis=0)
    out = np.zeros((length, key_q.shape[1]), dtype=np.float64)
    for idx in range(len(keyframes) - 1):
        left = keyframes[idx]
        right = keyframes[idx + 1]
        if right == left:
            out[left] = key_q[idx]
            continue
        span = right - left
        for pos in range(left, right + 1):
            alpha = (pos - left) / float(span)
            out[pos] = (1.0 - alpha) * key_q[idx] + alpha * key_q[idx + 1]
    return out


def kfbase_min_required_keyframes(length: int, max_gap: int) -> int:
    if length <= 1:
        return 1
    return max(2, int(math.ceil((length - 1) / max(max_gap, 1))) + 1)


def kfbase_max_allowed_keyframes(length: int, min_gap: int) -> int:
    if length <= 1:
        return 1
    return max(2, int(math.floor((length - 1) / max(min_gap, 1))) + 1)


def kfbase_choose_uniform_positions(
    length: int, key_count: int, min_gap: int, max_gap: int
) -> list[int]:
    if length <= 1:
        return [0]
    key_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(key_count)),
    )
    if key_count <= 2:
        return [0, length - 1]
    last = length - 1
    positions = [0]
    for idx in range(1, key_count - 1):
        remaining_segments = key_count - 1 - idx
        ideal = idx * last / float(key_count - 1)
        lower = max(positions[-1] + min_gap, last - remaining_segments * max_gap)
        upper = min(positions[-1] + max_gap, last - remaining_segments * min_gap)
        if upper < lower:
            upper = lower
        pos = int(round(ideal))
        pos = min(max(pos, int(lower)), int(upper))
        positions.append(pos)
    positions.append(last)
    return positions


def kfbase_allocate_target_keyframes(
    streams: list[kfbase_StreamSegment], target_ratio: float, min_gap: int, max_gap: int
) -> dict[str, int]:
    if not streams:
        return {}
    mins: dict[str, int] = {}
    maxs: dict[str, int] = {}
    ideals: dict[str, float] = {}
    total_min = 0
    total_max = 0
    total_frames = 0
    for stream in streams:
        length = int(len(stream.frame_numbers))
        min_count = kfbase_min_required_keyframes(length, max_gap)
        max_count = kfbase_max_allowed_keyframes(length, min_gap)
        mins[stream.stream_id] = min_count
        maxs[stream.stream_id] = max_count
        ideals[stream.stream_id] = float(
            np.clip(length * target_ratio, min_count, max_count)
        )
        total_min += min_count
        total_max += max_count
        total_frames += length
    requested_total = int(round(total_frames * max(target_ratio, 0.0)))
    target_total = min(max(requested_total, total_min), total_max)
    assigned: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    for stream in streams:
        stream_id = stream.stream_id
        ideal = ideals[stream_id]
        base = int(math.floor(ideal))
        base = min(max(base, mins[stream_id]), maxs[stream_id])
        assigned[stream_id] = base
        fractional.append((ideal - base, stream_id))
    deficit = target_total - sum(assigned.values())
    if deficit > 0:
        for _frac, stream_id in sorted(
            fractional, key=lambda item: item[0], reverse=True
        ):
            if deficit <= 0:
                break
            if assigned[stream_id] < maxs[stream_id]:
                assigned[stream_id] += 1
                deficit -= 1
    elif deficit < 0:
        for _frac, stream_id in sorted(fractional, key=lambda item: item[0]):
            if deficit >= 0:
                break
            if assigned[stream_id] > mins[stream_id]:
                assigned[stream_id] -= 1
                deficit += 1
    return assigned


def kfbase_best_split_for_segment(
    cache: kfbase_PrefixCostCache, q: np.ndarray, left: int, right: int, min_gap: int
) -> tuple[int | None, float]:
    low = left + min_gap
    high = right - min_gap
    if high < low:
        return (None, float("-inf"))
    base_cost = kfbase_interval_surrogate_cost(cache, q, left, right)
    best_idx: int | None = None
    best_gain = float("-inf")
    for candidate in range(low, high + 1):
        gain = base_cost - (
            kfbase_interval_surrogate_cost(cache, q, left, candidate)
            + kfbase_interval_surrogate_cost(cache, q, candidate, right)
        )
        if gain > best_gain:
            best_idx = candidate
            best_gain = float(gain)
    return (best_idx, best_gain)


def kfbase_residuals_for_segment(q: np.ndarray, left: int, right: int) -> np.ndarray:
    residuals = np.zeros(max(right - left + 1, 0), dtype=np.float64)
    if right <= left:
        return residuals
    span = float(right - left)
    q_left = q[left]
    q_right = q[right]
    for local_idx, pos in enumerate(range(left, right + 1)):
        alpha = (pos - left) / span
        interp = (1.0 - alpha) * q_left + alpha * q_right
        residuals[local_idx] = float(np.linalg.norm(q[pos] - interp))
    return residuals


def kfbase_best_residual_split_for_segment(
    q: np.ndarray,
    weights: np.ndarray,
    left: int,
    right: int,
    min_gap: int,
    quantile: float,
) -> tuple[int | None, float]:
    low = left + min_gap
    high = right - min_gap
    if high < low:
        return (None, float("-inf"))
    residuals = kfbase_residuals_for_segment(q, left, right)
    if residuals.size == 0:
        return (None, float("-inf"))
    seg_weights = np.maximum(weights[left : right + 1], 1e-08)
    weighted_residuals = residuals * np.sqrt(seg_weights)
    score = float(np.quantile(weighted_residuals, float(np.clip(quantile, 0.5, 0.99))))
    candidate_offset = int(np.argmax(weighted_residuals))
    candidate = left + candidate_offset
    candidate = min(max(candidate, low), high)
    return (candidate, score)


def kfbase_select_keyframes_uniform(
    q: np.ndarray, target_count: int, min_gap: int, max_gap: int
) -> list[int]:
    return kfbase_choose_uniform_positions(
        len(q), target_count, min_gap=min_gap, max_gap=max_gap
    )


def kfbase_select_keyframes_best_first_split(
    q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int
) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    chosen = kfbase_choose_uniform_positions(
        length,
        kfbase_min_required_keyframes(length, max_gap),
        min_gap=min_gap,
        max_gap=max_gap,
    )
    cache = kfbase_build_prefix_cache(q, weights)
    while len(chosen) < target_count:
        best_gain = float("-inf")
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, gain = kfbase_best_split_for_segment(
                cache, q, left, right, min_gap=min_gap
            )
            if candidate is not None and gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen


def kfbase_select_keyframes_rdp_quantile(
    q: np.ndarray,
    weights: np.ndarray,
    target_count: int,
    min_gap: int,
    max_gap: int,
    quantile: float,
) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    chosen = kfbase_choose_uniform_positions(
        length,
        kfbase_min_required_keyframes(length, max_gap),
        min_gap=min_gap,
        max_gap=max_gap,
    )
    while len(chosen) < target_count:
        best_score = float("-inf")
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, score = kfbase_best_residual_split_for_segment(
                q, weights, left, right, min_gap=min_gap, quantile=quantile
            )
            if candidate is not None and score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen


def kfbase_select_keyframes_greedy_split(
    q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int
) -> list[int]:
    del weights
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    chosen = kfbase_choose_uniform_positions(
        length,
        kfbase_min_required_keyframes(length, max_gap),
        min_gap=min_gap,
        max_gap=max_gap,
    )
    cache = kfbase_build_prefix_cache(q, np.ones(length, dtype=np.float64))
    while len(chosen) < target_count:
        best_gain = float("-inf")
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, gain = kfbase_best_split_for_segment(
                cache, q, left, right, min_gap=min_gap
            )
            if candidate is not None and gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen


def kfbase_ensure_target_count_with_peaks(
    q: np.ndarray,
    weights: np.ndarray,
    chosen: list[int],
    target_count: int,
    min_gap: int,
    max_gap: int,
) -> list[int]:
    length = len(q)
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    chosen = sorted(set((int(x) for x in chosen)))
    if len(chosen) > target_count:
        return kfbase_select_keyframes_bottom_up_merge(
            q,
            weights,
            chosen,
            target_count=target_count,
            min_gap=min_gap,
            max_gap=max_gap,
        )
    if len(chosen) == target_count:
        return chosen
    signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(
        np.maximum(weights, 1e-08)
    )
    for idx in np.argsort(signal)[::-1].tolist():
        idx = int(idx)
        if idx <= 0 or idx >= length - 1 or idx in chosen:
            continue
        if any((abs(idx - existing) < min_gap for existing in chosen)):
            continue
        chosen.append(idx)
        chosen.sort()
        if len(chosen) >= target_count:
            break
    if len(chosen) < target_count:
        for idx in kfbase_choose_uniform_positions(
            length, target_count, min_gap=min_gap, max_gap=max_gap
        ):
            if idx not in chosen:
                chosen.append(int(idx))
                chosen.sort()
            if len(chosen) >= target_count:
                break
    return sorted(set(chosen))


def kfbase_select_keyframes_trend_knots(
    q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int
) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    chosen = kfbase_choose_uniform_positions(
        length,
        kfbase_min_required_keyframes(length, max_gap),
        min_gap=min_gap,
        max_gap=max_gap,
    )
    signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(
        np.maximum(weights, 1e-08)
    )
    for idx in np.argsort(signal)[::-1].tolist():
        idx = int(idx)
        if idx <= 0 or idx >= length - 1 or idx in chosen:
            continue
        if any((abs(idx - existing) < min_gap for existing in chosen)):
            continue
        chosen.append(idx)
        chosen.sort()
        if len(chosen) >= target_count:
            break
    return kfbase_ensure_target_count_with_peaks(
        q, weights, chosen, target_count, min_gap=min_gap, max_gap=max_gap
    )


def kfbase_event_trigger_positions_for_threshold(
    q: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    min_gap: int,
    max_gap: int,
    quantile: float,
) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    chosen = [0]
    start = 0
    while start < length - 1:
        if length - 1 - start <= max_gap:
            chosen.append(length - 1)
            break
        triggered = False
        for end in range(start + min_gap, min(length, start + max_gap + 1)):
            residuals = kfbase_residuals_for_segment(q, start, end)
            seg_weights = np.sqrt(np.maximum(weights[start : end + 1], 1e-08))
            weighted = residuals * seg_weights
            score = float(np.quantile(weighted, float(np.clip(quantile, 0.5, 0.99))))
            if score <= threshold:
                continue
            low = start + min_gap
            high = end
            candidate_offset = int(np.argmax(weighted))
            candidate = start + candidate_offset
            candidate = min(max(candidate, low), high)
            if candidate <= start:
                candidate = min(start + min_gap, length - 1)
            if candidate >= length - 1:
                chosen.append(length - 1)
                return sorted(set(chosen))
            chosen.append(candidate)
            start = candidate
            triggered = True
            break
        if triggered:
            continue
        forced = min(length - 1, start + max_gap)
        if forced <= start:
            break
        if forced >= length - 1:
            chosen.append(length - 1)
            break
        chosen.append(forced)
        start = forced
    return sorted(set(chosen))


def kfbase_select_keyframes_event_triggered(
    q: np.ndarray,
    weights: np.ndarray,
    target_count: int,
    min_gap: int,
    max_gap: int,
    quantile: float,
    search_iters: int,
) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)),
    )
    base_signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(
        np.maximum(weights, 1e-08)
    )
    low = 0.0
    high = float(max(np.max(base_signal), 1.0))
    best = kfbase_choose_uniform_positions(
        length, target_count, min_gap=min_gap, max_gap=max_gap
    )
    best_gap = abs(len(best) - target_count)
    for _ in range(max(int(search_iters), 1)):
        mid = 0.5 * (low + high)
        chosen = kfbase_event_trigger_positions_for_threshold(
            q,
            weights,
            threshold=mid,
            min_gap=min_gap,
            max_gap=max_gap,
            quantile=quantile,
        )
        gap = abs(len(chosen) - target_count)
        if gap < best_gap:
            best = chosen
            best_gap = gap
        if len(chosen) > target_count:
            low = mid
        else:
            high = mid
    return kfbase_ensure_target_count_with_peaks(
        q, weights, best, target_count, min_gap=min_gap, max_gap=max_gap
    )


def kfbase_select_candidate_positions(
    stream: kfbase_StreamSegment,
    target_ratio: float,
    min_gap: int,
    max_gap: int,
    multiplier: float,
    min_separation: int,
    uniform_support: int,
) -> list[int]:
    length = len(stream.frame_numbers)
    if length <= 1:
        return [0]
    target_count = int(round(length * max(target_ratio, 0.0)))
    target_count = max(
        kfbase_min_required_keyframes(length, max_gap),
        min(kfbase_max_allowed_keyframes(length, min_gap), target_count),
    )
    desired_candidates = max(
        target_count + 2,
        int(math.ceil(target_count * max(multiplier, 1.0))),
        int(math.ceil(length / max(uniform_support, 1))),
    )
    positions = {0, length - 1}
    support_count = max(
        kfbase_min_required_keyframes(length, max(uniform_support, 1)),
        min(target_count * 2, desired_candidates),
    )
    positions.update(
        kfbase_choose_uniform_positions(
            length, support_count, min_gap=1, max_gap=max(uniform_support, 1)
        )
    )
    signal = stream.importance if stream.importance is not None else stream.confidence
    ranked = np.argsort(signal)[::-1].tolist()
    for idx in ranked:
        idx = int(idx)
        if idx <= 0 or idx >= length - 1:
            continue
        if any(
            (abs(idx - existing) < max(min_separation, 1) for existing in positions)
        ):
            continue
        positions.add(idx)
        if len(positions) >= desired_candidates:
            break
    return sorted(positions)


def kfbase_decode_keyframes_candidate_dp(
    q: np.ndarray,
    weights: np.ndarray,
    candidate_positions: list[int],
    keyframe_penalty: float,
    min_gap: int,
    max_gap: int,
    rewards: np.ndarray | None = None,
) -> tuple[list[int], float]:
    if len(candidate_positions) <= 2:
        return (list(candidate_positions), 0.0)
    cache = kfbase_build_prefix_cache(q, weights)
    n = len(candidate_positions)
    dp = np.full(n, np.inf, dtype=np.float64)
    back = np.full(n, -1, dtype=np.int32)
    dp[0] = -float(keyframe_penalty)
    for end_idx in range(1, n):
        end = candidate_positions[end_idx]
        for start_idx in range(0, end_idx):
            start = candidate_positions[start_idx]
            seg_len = end - start
            if start_idx != 0 and seg_len < min_gap:
                continue
            if end_idx != n - 1 and seg_len < min_gap:
                continue
            if seg_len > max_gap:
                continue
            interval_cost = kfbase_interval_surrogate_cost(cache, q, start, end)
            reward_term = 0.0 if rewards is None else float(rewards[end])
            candidate_cost = (
                dp[start_idx] + interval_cost + float(keyframe_penalty) - reward_term
            )
            if candidate_cost < dp[end_idx]:
                dp[end_idx] = candidate_cost
                back[end_idx] = start_idx
    if not np.isfinite(dp[-1]):
        return ([0, len(q) - 1], float("inf"))
    chosen: list[int] = []
    cursor = n - 1
    while cursor >= 0:
        chosen.append(int(candidate_positions[cursor]))
        if cursor == 0:
            break
        cursor = int(back[cursor])
        if cursor < 0:
            return ([0, len(q) - 1], float("inf"))
    chosen.reverse()
    return (chosen, float(dp[-1]))


def kfbase_decode_keyframes_candidate_budget_dp(
    q: np.ndarray,
    weights: np.ndarray,
    candidate_positions: list[int],
    target_count: int,
    min_gap: int,
    max_gap: int,
    rewards: np.ndarray | None = None,
) -> tuple[list[int], float]:
    if len(candidate_positions) <= 2:
        return (list(candidate_positions), 0.0)
    target_count = max(2, min(int(target_count), len(candidate_positions)))
    cache = kfbase_build_prefix_cache(q, weights)
    n = len(candidate_positions)
    dp = np.full((target_count, n), np.inf, dtype=np.float64)
    back = np.full((target_count, n), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for m in range(1, target_count):
        for end_idx in range(1, n):
            end = candidate_positions[end_idx]
            for start_idx in range(0, end_idx):
                if not np.isfinite(dp[m - 1, start_idx]):
                    continue
                start = candidate_positions[start_idx]
                seg_len = end - start
                if start_idx != 0 and seg_len < min_gap:
                    continue
                if end_idx != n - 1 and seg_len < min_gap:
                    continue
                if seg_len > max_gap:
                    continue
                interval_cost = kfbase_interval_surrogate_cost(cache, q, start, end)
                reward_term = 0.0 if rewards is None else float(rewards[end])
                candidate_cost = dp[m - 1, start_idx] + interval_cost - reward_term
                if candidate_cost < dp[m, end_idx]:
                    dp[m, end_idx] = candidate_cost
                    back[m, end_idx] = start_idx
    if not np.isfinite(dp[target_count - 1, n - 1]):
        fallback = kfbase_choose_uniform_positions(
            len(q), target_count, min_gap=min_gap, max_gap=max_gap
        )
        return (fallback, float("inf"))
    chosen: list[int] = []
    m = target_count - 1
    cursor = n - 1
    while cursor >= 0 and m >= 0:
        chosen.append(int(candidate_positions[cursor]))
        if m == 0:
            break
        cursor = int(back[m, cursor])
        m -= 1
        if cursor < 0:
            fallback = kfbase_choose_uniform_positions(
                len(q), target_count, min_gap=min_gap, max_gap=max_gap
            )
            return (fallback, float("inf"))
    chosen.reverse()
    return (chosen, float(dp[target_count - 1, n - 1]))


def kfbase_select_keyframes_bottom_up_merge(
    q: np.ndarray,
    weights: np.ndarray,
    initial_positions: list[int],
    target_count: int,
    min_gap: int,
    max_gap: int,
) -> list[int]:
    del weights
    chosen = sorted(set((int(x) for x in initial_positions)))
    if len(chosen) <= target_count:
        return chosen
    cache = kfbase_build_prefix_cache(q, np.ones(len(q), dtype=np.float64))
    while len(chosen) > target_count:
        best_idx: int | None = None
        best_delta = float("inf")
        for idx in range(1, len(chosen) - 1):
            left = chosen[idx - 1]
            mid = chosen[idx]
            right = chosen[idx + 1]
            if right - left > max_gap:
                continue
            if mid - left < min_gap or right - mid < min_gap:
                continue
            delta = (
                kfbase_interval_surrogate_cost(cache, q, left, right)
                - kfbase_interval_surrogate_cost(cache, q, left, mid)
                - kfbase_interval_surrogate_cost(cache, q, mid, right)
            )
            if delta < best_delta:
                best_delta = float(delta)
                best_idx = idx
        if best_idx is None:
            break
        chosen.pop(best_idx)
    return chosen


def kfbase_compute_ratio_for_lambda(
    streams: list[kfbase_StreamSegment],
    penalty: float,
    min_gap: int,
    max_gap: int,
    use_rewards: bool,
) -> tuple[float, int, int]:
    total_keyframes = 0
    total_frames = 0
    for stream in streams:
        assert stream.smoothed_q is not None
        prefix_cache = kfbase_get_stream_prefix_cache(stream)
        interval_costs = kfbase_get_stream_interval_costs(stream, max_gap=max_gap)
        chosen, _ = kfbase_decode_keyframes_dp(
            stream.smoothed_q,
            stream.importance if stream.importance is not None else stream.confidence,
            penalty,
            min_gap=min_gap,
            max_gap=max_gap,
            rewards=stream.keyframe_reward if use_rewards else None,
            prefix_cache=prefix_cache,
            interval_costs=interval_costs,
        )
        total_keyframes += len(chosen)
        total_frames += len(stream.frame_numbers)
    ratio = float(total_keyframes) / float(total_frames) if total_frames > 0 else 0.0
    return (ratio, total_keyframes, total_frames)


def kfbase_find_penalty_for_target_ratio(
    streams: list[kfbase_StreamSegment],
    target_ratio: float,
    fallback_penalty: float,
    min_gap: int,
    max_gap: int,
    search_iters: int,
    use_rewards: bool,
) -> tuple[float, dict[str, float]]:
    if target_ratio <= 0.0:
        ratio, keyframes, frames = kfbase_compute_ratio_for_lambda(
            streams,
            fallback_penalty,
            min_gap=min_gap,
            max_gap=max_gap,
            use_rewards=use_rewards,
        )
        return (
            fallback_penalty,
            {"achieved_ratio": ratio, "keyframes": keyframes, "frames": frames},
        )
    low = 0.0
    low_ratio, low_keys, low_frames = kfbase_compute_ratio_for_lambda(
        streams, low, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards
    )
    if low_ratio <= target_ratio:
        return (
            low,
            {"achieved_ratio": low_ratio, "keyframes": low_keys, "frames": low_frames},
        )
    high = max(fallback_penalty, 1.0)
    high_ratio, high_keys, high_frames = kfbase_compute_ratio_for_lambda(
        streams, high, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards
    )
    while high_ratio > target_ratio and high < 1000000.0:
        high *= 2.0
        high_ratio, high_keys, high_frames = kfbase_compute_ratio_for_lambda(
            streams, high, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards
        )
    best_penalty = high
    best_ratio = high_ratio
    best_keys = high_keys
    best_frames = high_frames
    for _ in range(search_iters):
        mid = 0.5 * (low + high)
        mid_ratio, mid_keys, mid_frames = kfbase_compute_ratio_for_lambda(
            streams, mid, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards
        )
        best_penalty, best_ratio, best_keys, best_frames = (
            mid,
            mid_ratio,
            mid_keys,
            mid_frames,
        )
        if mid_ratio > target_ratio:
            low = mid
        else:
            high = mid
    return (
        best_penalty,
        {"achieved_ratio": best_ratio, "keyframes": best_keys, "frames": best_frames},
    )


def kfbase_optimize_streams(
    streams: list[kfbase_StreamSegment], mode_penalty: float, args: argparse.Namespace
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    for stream in streams:
        assert stream.smoothed_q is not None
        anchor_q = kfbase_choose_anchor_q(
            stream, source=str(args.keyframe_value_source)
        )
        fit_weights = (
            stream.importance if stream.importance is not None else stream.confidence
        )
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        prefix_cache = kfbase_get_stream_prefix_cache(stream)
        interval_costs = kfbase_get_stream_interval_costs(
            stream, max_gap=int(args.max_gap)
        )
        if str(args.solver) == "greedy_split":
            chosen = kfbase_select_keyframes_greedy_split(
                stream.smoothed_q,
                fit_weights,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
            )
            objective = float("nan")
        elif str(args.solver) == "best_first_split":
            chosen = kfbase_select_keyframes_best_first_split(
                stream.smoothed_q,
                fit_weights,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
            )
            objective = float("nan")
        elif str(args.solver) == "rdp_quantile":
            chosen = kfbase_select_keyframes_rdp_quantile(
                stream.smoothed_q,
                fit_weights,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                quantile=float(args.rdp_quantile),
            )
            objective = float("nan")
        elif str(args.solver) == "trend_knots":
            chosen = kfbase_select_keyframes_trend_knots(
                stream.smoothed_q,
                fit_weights,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
            )
            objective = float("nan")
        elif str(args.solver) == "event_triggered":
            chosen = kfbase_select_keyframes_event_triggered(
                stream.smoothed_q,
                fit_weights,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                quantile=float(args.event_quantile),
                search_iters=int(args.event_search_iters),
            )
            objective = float("nan")
        elif str(args.solver) == "bottom_up_merge":
            target_count = int(round(mode_penalty))
            candidate_positions = kfbase_select_candidate_positions(
                stream,
                target_ratio=float(target_count) / max(len(stream.frame_numbers), 1),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                multiplier=float(args.candidate_multiplier),
                min_separation=int(args.candidate_min_separation),
                uniform_support=int(args.candidate_uniform_support),
            )
            chosen = kfbase_select_keyframes_bottom_up_merge(
                stream.smoothed_q,
                fit_weights,
                candidate_positions,
                target_count=target_count,
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
            )
            objective = float("nan")
        elif str(args.solver) == "dp_candidates":
            target_count = int(round(mode_penalty))
            candidate_positions = kfbase_select_candidate_positions(
                stream,
                target_ratio=float(target_count) / max(len(stream.frame_numbers), 1),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                multiplier=float(args.candidate_multiplier),
                min_separation=int(args.candidate_min_separation),
                uniform_support=int(args.candidate_uniform_support),
            )
            chosen, objective = kfbase_decode_keyframes_candidate_budget_dp(
                stream.smoothed_q,
                fit_weights,
                candidate_positions,
                target_count=target_count,
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                rewards=stream.keyframe_reward,
            )
        elif str(args.solver) == "uniform_refine":
            chosen = kfbase_select_keyframes_uniform(
                stream.smoothed_q,
                int(round(mode_penalty)),
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
            )
            objective = float("nan")
        else:
            chosen, objective = kfbase_decode_keyframes_dp(
                stream.smoothed_q,
                fit_weights,
                mode_penalty,
                min_gap=int(args.min_gap),
                max_gap=int(args.max_gap),
                rewards=stream.keyframe_reward
                if str(args.solver) == "dp_rewarded"
                else None,
                prefix_cache=prefix_cache,
                interval_costs=interval_costs,
            )
        chosen = kfbase_refine_keyframes_locally(
            stream.smoothed_q,
            fit_weights,
            chosen,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            radius=int(args.local_search_radius),
            prefix_cache=prefix_cache,
            interval_costs=interval_costs,
        )
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
        if len(chosen) >= 2 and str(args.value_refine) != "none":
            if str(args.value_refine) == "global_ls":
                key_q = kfbase_refine_keyframe_values_global_ls(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    ridge=float(args.value_refine_ridge),
                )
            elif str(args.value_refine) == "segment_ls":
                key_q = kfbase_refine_keyframe_values_segment_ls(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    ridge=float(args.value_refine_ridge),
                )
            elif str(args.value_refine) == "residual_nudge":
                key_q = kfbase_refine_keyframe_values_residual_nudge(
                    target_q=target_q,
                    base_key_q=key_q,
                    keyframes=chosen,
                    weights=fit_weights,
                    damping=float(args.value_refine_damping),
                )
            interp_q = kfbase_interpolate_from_key_values(
                key_q, chosen, len(stream.frame_numbers)
            )
        else:
            interp_q = kfbase_interpolate_dense_q(anchor_q, chosen)
        dense_state = kfbase_q_to_state(
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
                "frame_count": int(len(stream.frame_numbers)),
                "keyframe_count": int(len(chosen)),
                "keyframe_ratio": float(
                    len(chosen) / max(len(stream.frame_numbers), 1)
                ),
                "objective": float(objective),
                "weighted_param_rmse": weighted_rmse,
            }
        )
        key_states = kfbase_q_to_state(
            key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
        )
        for key_idx, local_idx in enumerate(chosen):
            keyframe_rows.append(
                {
                    "stream_id": stream.stream_id,
                    "track_id": stream.track_id,
                    "mode": stream.mode,
                    "run_id": stream.run_id,
                    "slot_id": stream.slot_id,
                    "frame": int(stream.frame_numbers[local_idx]),
                    "ellipse": key_states[key_idx].tolist(),
                }
            )
        for local_idx, frame in enumerate(stream.frame_numbers):
            dense_rows.append(
                {
                    "stream_id": stream.stream_id,
                    "track_id": stream.track_id,
                    "mode": stream.mode,
                    "run_id": stream.run_id,
                    "slot_id": stream.slot_id,
                    "frame": int(frame),
                    "ellipse": dense_state[local_idx].tolist(),
                    "is_keyframe": int(local_idx in set(chosen)),
                }
            )
    return (keyframe_rows, dense_rows, segment_rows)


def kfbase_merge_dense_rows_to_union(
    dense_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for row in dense_rows:
        key = (
            str(row["track_id"]),
            str(row["mode"]),
            int(row["run_id"]),
            int(row["frame"]),
        )
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
        entry["ellipse_params"].append((int(row["slot_id"]), row["ellipse"]))
        entry["has_keyframe"] = int(
            max(int(entry["has_keyframe"]), int(row["is_keyframe"]))
        )
    merged: list[dict[str, object]] = []
    for key in sorted(
        grouped.keys(), key=lambda item: (int(item[0]), item[3], item[2])
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


def kfbase_write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def kfbase_write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def kfbase_main() -> None:
    args = kfbase_parse_args()
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = kfbase_load_metric_rows(
        input_metrics,
        confidence_floor=float(args.confidence_floor),
        error_scale=float(args.error_scale),
    )
    streams = kfbase_build_stream_segments(args, rows)
    if int(args.max_streams) > 0:
        streams = streams[: int(args.max_streams)]
    for stream in streams:
        kfbase_smooth_stream_segment(stream, args)
        kfbase_derive_stream_importance(stream, args)
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[kfbase_StreamSegment] = []
        for stream in streams:
            broken_streams.extend(kfbase_split_stream_on_breaks(stream, args))
        streams = broken_streams
    k1_streams = [stream for stream in streams if stream.mode == "K1"]
    k2_streams = [stream for stream in streams if stream.mode == "K2"]
    if str(args.solver) == "dp":
        penalty_k1, ratio_summary_k1 = kfbase_find_penalty_for_target_ratio(
            k1_streams,
            target_ratio=float(args.target_k1_ratio),
            fallback_penalty=0.5
            if float(args.lambda_k1) <= 0
            else float(args.lambda_k1),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            search_iters=int(args.lambda_search_iters),
            use_rewards=False,
        )
        penalty_k2, ratio_summary_k2 = kfbase_find_penalty_for_target_ratio(
            k2_streams,
            target_ratio=float(args.target_k2_ratio),
            fallback_penalty=0.35
            if float(args.lambda_k2) <= 0
            else float(args.lambda_k2),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            search_iters=int(args.lambda_search_iters),
            use_rewards=False,
        )
    elif str(args.solver) == "dp_rewarded":
        penalty_k1, ratio_summary_k1 = kfbase_find_penalty_for_target_ratio(
            k1_streams,
            target_ratio=float(args.target_k1_ratio),
            fallback_penalty=0.5
            if float(args.lambda_k1) <= 0
            else float(args.lambda_k1),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            search_iters=int(args.lambda_search_iters),
            use_rewards=True,
        )
        penalty_k2, ratio_summary_k2 = kfbase_find_penalty_for_target_ratio(
            k2_streams,
            target_ratio=float(args.target_k2_ratio),
            fallback_penalty=0.35
            if float(args.lambda_k2) <= 0
            else float(args.lambda_k2),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            search_iters=int(args.lambda_search_iters),
            use_rewards=True,
        )
    else:
        assigned_k1 = kfbase_allocate_target_keyframes(
            k1_streams,
            target_ratio=float(args.target_k1_ratio),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
        )
        assigned_k2 = kfbase_allocate_target_keyframes(
            k2_streams,
            target_ratio=float(args.target_k2_ratio),
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
        )
        penalty_k1 = float(sum(assigned_k1.values()))
        penalty_k2 = float(sum(assigned_k2.values()))
        ratio_summary_k1 = {
            "achieved_ratio": float(
                sum(assigned_k1.values())
                / max(sum((len(stream.frame_numbers) for stream in k1_streams)), 1)
            ),
            "keyframes": int(sum(assigned_k1.values())),
            "frames": int(sum((len(stream.frame_numbers) for stream in k1_streams))),
        }
        ratio_summary_k2 = {
            "achieved_ratio": float(
                sum(assigned_k2.values())
                / max(sum((len(stream.frame_numbers) for stream in k2_streams)), 1)
            ),
            "keyframes": int(sum(assigned_k2.values())),
            "frames": int(sum((len(stream.frame_numbers) for stream in k2_streams))),
        }
    if str(args.solver) in {"dp", "dp_rewarded"}:
        keyframe_rows_k1, dense_rows_k1, segment_rows_k1 = kfbase_optimize_streams(
            k1_streams, penalty_k1, args
        )
        keyframe_rows_k2, dense_rows_k2, segment_rows_k2 = kfbase_optimize_streams(
            k2_streams, penalty_k2, args
        )
    else:
        keyframe_rows_k1: list[dict[str, object]] = []
        dense_rows_k1: list[dict[str, object]] = []
        segment_rows_k1: list[dict[str, object]] = []
        for stream in k1_streams:
            kf, dense, seg = kfbase_optimize_streams(
                [stream], float(assigned_k1[stream.stream_id]), args
            )
            keyframe_rows_k1.extend(kf)
            dense_rows_k1.extend(dense)
            segment_rows_k1.extend(seg)
        keyframe_rows_k2 = []
        dense_rows_k2 = []
        segment_rows_k2 = []
        for stream in k2_streams:
            kf, dense, seg = kfbase_optimize_streams(
                [stream], float(assigned_k2[stream.stream_id]), args
            )
            keyframe_rows_k2.extend(kf)
            dense_rows_k2.extend(dense)
            segment_rows_k2.extend(seg)
    keyframe_rows = keyframe_rows_k1 + keyframe_rows_k2
    dense_rows = dense_rows_k1 + dense_rows_k2
    segment_rows = segment_rows_k1 + segment_rows_k2
    dense_union_rows = kfbase_merge_dense_rows_to_union(dense_rows)
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
    kfbase_write_json(output_dir / "final_keyframes.json", keyframe_rows)
    kfbase_write_json(output_dir / "interpolated_union.json", dense_union_rows)
    kfbase_write_csv(
        output_dir / "stream_segments.csv",
        segment_rows,
        [
            "stream_id",
            "track_id",
            "mode",
            "run_id",
            "slot_id",
            "frame_count",
            "keyframe_count",
            "keyframe_ratio",
            "objective",
            "weighted_param_rmse",
        ],
    )
    summary = {
        "input_metrics_csv": str(input_metrics),
        "stream_count": int(len(streams)),
        "row_count": int(len(rows)),
        "mode_summary": {
            "K1": {
                "lambda": float(penalty_k1),
                "target_ratio": float(args.target_k1_ratio),
                **ratio_summary_k1,
            },
            "K2": {
                "lambda": float(penalty_k2),
                "target_ratio": float(args.target_k2_ratio),
                **ratio_summary_k2,
            },
        },
        "total_keyframe_rows": int(len(keyframe_rows)),
        "total_dense_rows": int(len(dense_rows)),
        "total_union_rows": int(len(dense_union_rows)),
        "settings": {
            "solver": str(args.solver),
            "smooth_alpha": float(args.smooth_alpha),
            "confidence_floor": float(args.confidence_floor),
            "error_scale": float(args.error_scale),
            "value_refine": str(args.value_refine),
            "value_refine_ridge": float(args.value_refine_ridge),
            "value_refine_damping": float(args.value_refine_damping),
            "weight_error_gain": float(args.weight_error_gain),
            "weight_curvature_gain": float(args.weight_curvature_gain),
            "importance_cap": float(args.importance_cap),
            "reward_error_gain": float(args.reward_error_gain),
            "reward_curvature_gain": float(args.reward_curvature_gain),
            "reward_cap": float(args.reward_cap),
            "auto_break_threshold": float(args.auto_break_threshold),
            "auto_break_min_length": int(args.auto_break_min_length),
            "auto_break_min_separation": int(args.auto_break_min_separation),
            "candidate_multiplier": float(args.candidate_multiplier),
            "candidate_min_separation": int(args.candidate_min_separation),
            "candidate_uniform_support": int(args.candidate_uniform_support),
            "rdp_quantile": float(args.rdp_quantile),
            "min_gap": int(args.min_gap),
            "max_gap": int(args.max_gap),
            "local_search_radius": int(args.local_search_radius),
            "keyframe_value_source": str(args.keyframe_value_source),
        },
    }
    kfbase_write_json(output_dir / "summary.json", summary)


kfbase_module = _register_inline_module(
    "optimize_keyframes_standalone",
    {
        "parse_args": "kfbase_parse_args",
        "MetricRow": "kfbase_MetricRow",
        "StreamSegment": "kfbase_StreamSegment",
        "canonicalize_ellipse": "kfbase_canonicalize_ellipse",
        "circular_angle_distance_deg": "kfbase_circular_angle_distance_deg",
        "unwrap_angles_deg": "kfbase_unwrap_angles_deg",
        "compute_confidence": "kfbase_compute_confidence",
        "load_metric_rows": "kfbase_load_metric_rows",
        "split_runs": "kfbase_split_runs",
        "ellipse_pair_cost": "kfbase_ellipse_pair_cost",
        "stabilize_k2_slots": "kfbase_stabilize_k2_slots",
        "build_stream_segments": "kfbase_build_stream_segments",
        "build_second_difference_matrix": "kfbase_build_second_difference_matrix",
        "solve_second_difference_system": "kfbase_solve_second_difference_system",
        "state_to_q": "kfbase_state_to_q",
        "q_to_state": "kfbase_q_to_state",
        "smooth_stream_segment": "kfbase_smooth_stream_segment",
        "choose_anchor_q": "kfbase_choose_anchor_q",
        "robust_normalize": "kfbase_robust_normalize",
        "compute_curvature_signal": "kfbase_compute_curvature_signal",
        "derive_stream_importance": "kfbase_derive_stream_importance",
        "split_stream_on_breaks": "kfbase_split_stream_on_breaks",
        "PrefixCostCache": "kfbase_PrefixCostCache",
        "build_prefix_cache": "kfbase_build_prefix_cache",
        "get_stream_prefix_cache": "kfbase_get_stream_prefix_cache",
        "get_stream_interval_costs": "kfbase_get_stream_interval_costs",
        "interval_surrogate_cost": "kfbase_interval_surrogate_cost",
        "decode_keyframes_dp": "kfbase_decode_keyframes_dp",
        "refine_keyframes_locally": "kfbase_refine_keyframes_locally",
        "interpolate_dense_q": "kfbase_interpolate_dense_q",
        "refine_keyframe_values_global_ls": "kfbase_refine_keyframe_values_global_ls",
        "refine_keyframe_values_segment_ls": "kfbase_refine_keyframe_values_segment_ls",
        "refine_keyframe_values_residual_nudge": "kfbase_refine_keyframe_values_residual_nudge",
        "interpolate_from_key_values": "kfbase_interpolate_from_key_values",
        "min_required_keyframes": "kfbase_min_required_keyframes",
        "max_allowed_keyframes": "kfbase_max_allowed_keyframes",
        "choose_uniform_positions": "kfbase_choose_uniform_positions",
        "allocate_target_keyframes": "kfbase_allocate_target_keyframes",
        "best_split_for_segment": "kfbase_best_split_for_segment",
        "residuals_for_segment": "kfbase_residuals_for_segment",
        "best_residual_split_for_segment": "kfbase_best_residual_split_for_segment",
        "select_keyframes_uniform": "kfbase_select_keyframes_uniform",
        "select_keyframes_best_first_split": "kfbase_select_keyframes_best_first_split",
        "select_keyframes_rdp_quantile": "kfbase_select_keyframes_rdp_quantile",
        "select_keyframes_greedy_split": "kfbase_select_keyframes_greedy_split",
        "ensure_target_count_with_peaks": "kfbase_ensure_target_count_with_peaks",
        "select_keyframes_trend_knots": "kfbase_select_keyframes_trend_knots",
        "event_trigger_positions_for_threshold": "kfbase_event_trigger_positions_for_threshold",
        "select_keyframes_event_triggered": "kfbase_select_keyframes_event_triggered",
        "select_candidate_positions": "kfbase_select_candidate_positions",
        "decode_keyframes_candidate_dp": "kfbase_decode_keyframes_candidate_dp",
        "decode_keyframes_candidate_budget_dp": "kfbase_decode_keyframes_candidate_budget_dp",
        "select_keyframes_bottom_up_merge": "kfbase_select_keyframes_bottom_up_merge",
        "compute_ratio_for_lambda": "kfbase_compute_ratio_for_lambda",
        "find_penalty_for_target_ratio": "kfbase_find_penalty_for_target_ratio",
        "optimize_streams": "kfbase_optimize_streams",
        "merge_dense_rows_to_union": "kfbase_merge_dense_rows_to_union",
        "write_json": "kfbase_write_json",
        "write_csv": "kfbase_write_csv",
        "main": "kfbase_main",
    },
)
