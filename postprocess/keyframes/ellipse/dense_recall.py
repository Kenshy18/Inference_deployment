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
from pathlib import Path
import numpy as np
from .optimizer import kfbase_module as base


def kfdense_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6KF_D-style keyframe optimizer with dense-ellipse recall-aware value refinement."
    )
    parser.add_argument("--input-metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-k1-ratio", type=float, default=0.1)
    parser.add_argument("--target-k2-ratio", type=float, default=0.16)
    parser.add_argument("--solver", choices=["dp", "dp_rewarded"], default="dp")
    parser.add_argument("--lambda-k1", type=float, default=-1.0)
    parser.add_argument("--lambda-k2", type=float, default=-1.0)
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
    return parser.parse_args()


def kfdense_unit_disk_samples(count: int) -> np.ndarray:
    count = max(int(count), 7)
    pts = np.zeros((count, 2), dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for idx in range(count):
        radius = math.sqrt((idx + 0.5) / count)
        theta = idx * golden
        pts[idx, 0] = radius * math.cos(theta)
        pts[idx, 1] = radius * math.sin(theta)
    return pts


def kfdense_ellipse_membership(
    points_x: np.ndarray, points_y: np.ndarray, states: np.ndarray
) -> np.ndarray:
    cx = states[:, 0][:, None]
    cy = states[:, 1][:, None]
    a = np.maximum(states[:, 2][:, None], 1e-06)
    b = np.maximum(states[:, 3][:, None], 1e-06)
    theta = np.deg2rad(states[:, 4])[:, None]
    dx = points_x - cx
    dy = points_y - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    value = (local_x / a) ** 2 + (local_y / b) ** 2
    return value <= 1.0


def kfdense_approx_dense_recall(
    source_states: np.ndarray, pred_states: np.ndarray, disk_samples: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    n = len(source_states)
    if n == 0:
        return (1.0, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    sx = disk_samples[:, 0][None, :]
    sy = disk_samples[:, 1][None, :]
    src_theta = np.deg2rad(source_states[:, 4])[:, None]
    src_cos = np.cos(src_theta)
    src_sin = np.sin(src_theta)
    local_x = sx * source_states[:, 2][:, None]
    local_y = sy * source_states[:, 3][:, None]
    world_x = source_states[:, 0][:, None] + src_cos * local_x - src_sin * local_y
    world_y = source_states[:, 1][:, None] + src_sin * local_x + src_cos * local_y
    inside = kfdense_ellipse_membership(world_x, world_y, pred_states)
    per_frame = inside.mean(axis=1).astype(np.float64)
    weights = np.maximum(source_states[:, 2] * source_states[:, 3], 1e-06)
    global_mean = float(np.average(per_frame, weights=weights))
    return (global_mean, per_frame, weights)


def kfdense_apply_uniform_inflation(key_q: np.ndarray, delta: float) -> np.ndarray:
    out = key_q.copy()
    out[:, 2] += float(delta)
    out[:, 3] += float(delta)
    return out


def kfdense_enforce_dense_recall_target(
    stream: base.StreamSegment,
    key_q: np.ndarray,
    chosen: list[int],
    args: argparse.Namespace,
    disk_samples: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    interp_q = base.interpolate_from_key_values(
        key_q, chosen, len(stream.frame_numbers)
    )
    base_state = base.q_to_state(
        interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
    )
    base_recall, _per_frame, weights = kfdense_approx_dense_recall(
        stream.raw_states, base_state, disk_samples
    )
    target = float(args.dense_recall_target)
    info: dict[str, float | bool] = {
        "dense_recall_before": float(base_recall),
        "dense_recall_after": float(base_recall),
        "inflate_log_delta": 0.0,
        "dense_recall_target": target,
        "dense_recall_attained": bool(base_recall >= target),
        "source_area_sum": float(np.sum(weights)),
    }
    if target <= 0.0 or base_recall >= target or len(chosen) == 0:
        return (key_q, info)
    high = float(args.dense_recall_max_inflate_log)
    high_q = kfdense_apply_uniform_inflation(key_q, high)
    high_interp = base.interpolate_from_key_values(
        high_q, chosen, len(stream.frame_numbers)
    )
    high_state = base.q_to_state(
        high_interp, scale=stream.transform_scale, theta_scale=stream.theta_scale
    )
    high_recall, _per_frame, _weights = kfdense_approx_dense_recall(
        stream.raw_states, high_state, disk_samples
    )
    if high_recall < target:
        info["dense_recall_after"] = float(high_recall)
        info["inflate_log_delta"] = float(high)
        info["dense_recall_attained"] = False
        return (high_q, info)
    low = 0.0
    best_q = high_q
    best_recall = high_recall
    for _ in range(int(args.dense_recall_search_iters)):
        mid = 0.5 * (low + high)
        mid_q = kfdense_apply_uniform_inflation(key_q, mid)
        mid_interp = base.interpolate_from_key_values(
            mid_q, chosen, len(stream.frame_numbers)
        )
        mid_state = base.q_to_state(
            mid_interp, scale=stream.transform_scale, theta_scale=stream.theta_scale
        )
        mid_recall, _per_frame, _weights = kfdense_approx_dense_recall(
            stream.raw_states, mid_state, disk_samples
        )
        if mid_recall >= target:
            high = mid
            best_q = mid_q
            best_recall = mid_recall
        else:
            low = mid
    info["dense_recall_after"] = float(best_recall)
    info["inflate_log_delta"] = float(high)
    info["dense_recall_attained"] = True
    return (best_q, info)


def kfdense_optimize_streams_dense_recall(
    streams: list[base.StreamSegment], mode_penalty: float, args: argparse.Namespace
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    disk_samples = kfdense_unit_disk_samples(int(args.dense_recall_samples))
    for stream in streams:
        assert stream.smoothed_q is not None
        anchor_q = base.choose_anchor_q(stream, source=str(args.keyframe_value_source))
        fit_weights = (
            stream.importance if stream.importance is not None else stream.confidence
        )
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        chosen, objective = base.decode_keyframes_dp(
            stream.smoothed_q,
            fit_weights,
            mode_penalty,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            rewards=stream.keyframe_reward
            if str(args.solver) == "dp_rewarded"
            else None,
        )
        chosen = base.refine_keyframes_locally(
            stream.smoothed_q,
            fit_weights,
            chosen,
            min_gap=int(args.min_gap),
            max_gap=int(args.max_gap),
            radius=int(args.local_search_radius),
        )
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
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
        key_q, dense_info = kfdense_enforce_dense_recall_target(
            stream=stream,
            key_q=key_q,
            chosen=chosen,
            args=args,
            disk_samples=disk_samples,
        )
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
            }
        )
        key_states = base.q_to_state(
            key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale
        )
        chosen_set = set(chosen)
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
                    "is_keyframe": int(local_idx in chosen_set),
                }
            )
    return (keyframe_rows, dense_rows, segment_rows)


def kfdense_main() -> None:
    args = kfdense_parse_args()
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = base.load_metric_rows(
        input_metrics,
        confidence_floor=float(args.confidence_floor),
        error_scale=float(args.error_scale),
    )
    streams = base.build_stream_segments(args, rows)
    if int(args.max_streams) > 0:
        streams = streams[: int(args.max_streams)]
    for stream in streams:
        base.smooth_stream_segment(stream, args)
        base.derive_stream_importance(stream, args)
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[base.StreamSegment] = []
        for stream in streams:
            broken_streams.extend(base.split_stream_on_breaks(stream, args))
        streams = broken_streams
    k1_streams = [stream for stream in streams if stream.mode == "K1"]
    k2_streams = [stream for stream in streams if stream.mode == "K2"]
    penalty_k1, ratio_summary_k1 = base.find_penalty_for_target_ratio(
        k1_streams,
        target_ratio=float(args.target_k1_ratio),
        fallback_penalty=0.5 if float(args.lambda_k1) <= 0 else float(args.lambda_k1),
        min_gap=int(args.min_gap),
        max_gap=int(args.max_gap),
        search_iters=int(args.lambda_search_iters),
        use_rewards=str(args.solver) == "dp_rewarded",
    )
    penalty_k2, ratio_summary_k2 = base.find_penalty_for_target_ratio(
        k2_streams,
        target_ratio=float(args.target_k2_ratio),
        fallback_penalty=0.35 if float(args.lambda_k2) <= 0 else float(args.lambda_k2),
        min_gap=int(args.min_gap),
        max_gap=int(args.max_gap),
        search_iters=int(args.lambda_search_iters),
        use_rewards=str(args.solver) == "dp_rewarded",
    )
    (
        keyframe_rows_k1,
        dense_rows_k1,
        segment_rows_k1,
    ) = kfdense_optimize_streams_dense_recall(k1_streams, penalty_k1, args)
    (
        keyframe_rows_k2,
        dense_rows_k2,
        segment_rows_k2,
    ) = kfdense_optimize_streams_dense_recall(k2_streams, penalty_k2, args)
    keyframe_rows = keyframe_rows_k1 + keyframe_rows_k2
    dense_rows = dense_rows_k1 + dense_rows_k2
    segment_rows = segment_rows_k1 + segment_rows_k2
    dense_union_rows = base.merge_dense_rows_to_union(dense_rows)
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
        ],
    )
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
        "dense_recall_summary": {
            "target": float(args.dense_recall_target),
            "global_before": float(dense_recall_before),
            "global_after": float(dense_recall_after),
            "inflated_segments": int(inflated_segments),
            "unattained_segments": int(unattained_segments),
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


kfdense_module = _register_inline_module(
    "optimize_keyframes_dense_recall_standalone",
    {
        "parse_args": "kfdense_parse_args",
        "unit_disk_samples": "kfdense_unit_disk_samples",
        "ellipse_membership": "kfdense_ellipse_membership",
        "approx_dense_recall": "kfdense_approx_dense_recall",
        "apply_uniform_inflation": "kfdense_apply_uniform_inflation",
        "enforce_dense_recall_target": "kfdense_enforce_dense_recall_target",
        "optimize_streams_dense_recall": "kfdense_optimize_streams_dense_recall",
        "main": "kfdense_main",
    },
)
