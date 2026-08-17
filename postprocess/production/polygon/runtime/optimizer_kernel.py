"""Numerical kernel for track-first adaptive polygon optimization.

The kernel performs:

- dense polygon input
- short-gap polygon gapfill inside each track
- track-first AI point-count prediction after gapfill
- track-segment anchor-count fixing via p90 + 1
- contour resampling / phase alignment per track segment
- raw-only per-frame shape state
- candidate-frame pooling via saliency + surrogate path
- penalty shortest-path DP with exact recall budget
- interval endpoint vote refinement (pair-vote)
- exact recall repair
- exact mask evaluation

The promoted implementation intentionally excludes:

- multi-candidate per-frame shape proposals
- interval-synthesized endpoint candidates
- soft-raster fitting
- joint keyframe gradient refinement
- local search
- polish passes
- exact-K main solver
- proxy-fast recall mode

The module is private numerical infrastructure. Pipeline orchestration,
configuration, candidate construction, topology guards, pair-vote policy and
artifact publication are owned by responsibility-specific Production modules.
"""

import argparse
import bisect
import concurrent.futures
import csv
import json
import math
import multiprocessing
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

# Stable numerical defaults for the raw-only kernel.
DEFAULT_ANCHORS_PER_CONTOUR = 48
DEFAULT_RECALL_MIN = 0.97
DEFAULT_MAX_GAP = 30
DEFAULT_DP_EVAL_SCALE = 1.0
DEFAULT_DP_EVAL_PAD = 8
DEFAULT_SURROGATE_POOL_FACTOR = 2.0
DEFAULT_SURROGATE_PEAK_FACTOR = 1.2
DEFAULT_SURROGATE_NEIGHBOR_RADIUS = 1
DEFAULT_SURROGATE_SHAPE_WEIGHT = 0.15
DEFAULT_SALIENCY_SHAPE_ETA = 0.5
DEFAULT_SALIENCY_AREA_ETA = 0.45
DEFAULT_PENALTY_BINARY_STEPS = 12
DEFAULT_PENALTY_MAX = 1024.0
DEFAULT_RECALL_BUDGET_BINARY_STEPS = 8
DEFAULT_RECALL_BUDGET_MAX_MU = 64.0
DEFAULT_PATH_RECALL_VIOLATION_WEIGHT = 64.0
DEFAULT_SHAPE_SWITCH_WEIGHT = 2.0
DEFAULT_SHAPE_DISTANCE_WEIGHT = 0.4
DEFAULT_SHAPE_UPDATE_THRESHOLD_RATIO = 0.09
DEFAULT_SHAPE_PENALTY_ADAPT_GAIN = 1.25
DEFAULT_SHAPE_DISTANCE_RELIEF = 1.10
DEFAULT_SHAPE_SWITCH_RELIEF = 0.45
DEFAULT_SHAPE_DISTANCE_MIN_SCALE = 0.10
DEFAULT_SHAPE_SWITCH_MIN_SCALE = 0.45
DEFAULT_DYNAMIC_MAX_GAP_FACTOR = 4.0
DEFAULT_INTERVAL_IOU_WEIGHT = 1.0
DEFAULT_EXACT_RECALL_REPAIR_MAX_PASSES = 4
DEFAULT_EXACT_RECALL_REPAIR_TOPK = 3
DEFAULT_EXACT_RECALL_REPAIR_SCALE_DELTAS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
DEFAULT_ADAPTIVE_ANCHOR_COUNTS = False
DEFAULT_ADAPTIVE_POINT_QUANTILE = 0.95
DEFAULT_ADAPTIVE_POINT_OFFSET = 2
DEFAULT_MIN_ANCHORS_PER_CONTOUR = 4
DEFAULT_PREDICTOR_BATCH_SIZE = 256
DEFAULT_PREDICTOR_DEVICE = "cuda"
DEFAULT_GAPFILL_ENABLED = True
DEFAULT_GAPFILL_MAX_GAP = 30
DEFAULT_GAPFILL_TEMP_POINTS = 128
DEFAULT_MAX_RUN_FRAMES = 30000
DEFAULT_RUN_OVERLAP_FRAMES = 900
DEFAULT_POINT_PREDICTOR_MODEL_DIR = (
    ROOT
    / "experiments"
    / "linear_polygon_bezier_workspace_20260410"
    / "output"
    / "mask_point_predictor_wide96_20260411"
)

from .kernel.model import (
    FEATURE_NAMES,
    ConvBNAct,
    LearnedPointPredictor,
    TinyMaskPointNet,
    build_feature_vector,
    compute_mask_descriptors,
    resize_mask_with_padding,
)
from .kernel.types import (
    FrameEvalContext,
    InstanceRun,
    IntervalCost,
    ShapeCandidate,
    SimilarityTransform,
    TrackRow,
)
from .kernel import geometry as _kernel_geometry
from .kernel import candidates as _kernel_candidates
from .kernel import stream as _kernel_stream
from .kernel.candidates import (
    build_candidate_frame_pool,
    compute_saliency_scores,
    compute_surrogate_prefix,
    exact_k_dp,
    surrogate_interval_cost,
    surrogate_interval_solution,
)
from .kernel.artifacts import (
    SqliteUnionRowStore,
    aggregate_exact_rows,
    evaluate_union_exact,
    load_rows,
    union_rows_to_pred_sqlite,
    write_compact_json_array,
    write_csv,
)
from .kernel.evaluation import (
    adaptive_shape_penalty_scales,
    build_frame_candidates,
    build_frame_eval_contexts,
    compute_cached_metrics_from_interpolated_polygons,
    compute_cached_metrics_from_polygons,
    evaluate_frame_vector_loss_budget,
    flatten_contours,
    frame_accuracy_loss,
    rasterize_interpolated_mask_with_context,
    rasterize_mask_with_context,
    recall_budget_from_metrics,
    recall_budget_limit,
    recall_violation,
    scale_vector_about_centroid,
    shape_distance,
    split_vector_to_polygons,
    vector_proxy_stats,
)
from .kernel.geometry import (
    _rotation_matrix,
    align_contour_slots,
    align_polygon_phase,
    apply_similarity_transform,
    build_local_mask_from_polygons,
    build_track_segments_with_gapfill,
    compute_exact_metrics_from_polygons,
    compute_weighted_error,
    contour_centroid,
    cyclic_shift_points,
    estimate_similarity_transform,
    interpolate_gapfill_polygons,
    normalize_closed_points,
    orient_ccw,
    parse_polygons,
    polygon_area,
    rasterize_mask_from_polygons,
    resample_closed_contour,
    signed_area,
    similarity_residuals,
    sort_polygons,
)
from .kernel.interpolation import (
    assign_candidate_ids_to_keyframes,
    build_interpolation_weights,
    build_ring_second_difference_rtr,
    interpolate_polygons,
    interpolate_vectors,
    interval_cost_from_vectors,
    pair_vote_refine_keyframe_vectors,
)
from .kernel.stream import (
    build_track_streams,
    iter_sqlite_track_rows,
    iter_track_streams_from_sqlite,
    parse_float_list,
    split_long_track_segments,
    sqlite_allowed_track_ids,
    sqlite_mask_stats_for_tracks,
)
from .kernel.solver import (
    LazyInterpolatedRun,
    LazyUnionRows,
    evaluate_keyframe_path,
    evaluate_keyframe_path_parts,
    exact_interpolated_metrics,
    exact_recall_solution_key,
    interpolate_run,
    repair_keyframe_vectors_for_exact_recall,
    run_multistate_penalty_path,
    run_single_state_penalty_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production polygon optimizer kernel with gapfill-first track-level anchor counts. "
            "Input is a SQLite file with masks(frame, track_id, polygons). "
            "Each polygons cell is a JSON array of polygons, where each polygon is [[x, y], ...]."
        )
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=1.0 / 9.0)
    parser.add_argument(
        "--anchors-per-contour",
        type=int,
        default=DEFAULT_ANCHORS_PER_CONTOUR,
        help="Fallback or maximum anchors per contour; runs select up to this cap.",
    )
    parser.add_argument(
        "--adaptive-anchor-counts",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
        help="Predict per-frame polygon counts and fix a run-wise anchor count with p90 + offset.",
    )
    parser.add_argument(
        "--point-predictor-model-dir",
        type=Path,
        default=DEFAULT_POINT_PREDICTOR_MODEL_DIR,
    )
    parser.add_argument(
        "--predictor-device", type=str, default=DEFAULT_PREDICTOR_DEVICE
    )
    parser.add_argument(
        "--predictor-batch-size", type=int, default=DEFAULT_PREDICTOR_BATCH_SIZE
    )
    parser.add_argument(
        "--adaptive-point-quantile", type=float, default=DEFAULT_ADAPTIVE_POINT_QUANTILE
    )
    parser.add_argument(
        "--adaptive-point-offset", type=int, default=DEFAULT_ADAPTIVE_POINT_OFFSET
    )
    parser.add_argument(
        "--min-anchors-per-contour", type=int, default=DEFAULT_MIN_ANCHORS_PER_CONTOUR
    )
    parser.add_argument(
        "--gapfill-enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GAPFILL_ENABLED,
    )
    parser.add_argument("--gapfill-max-gap", type=int, default=DEFAULT_GAPFILL_MAX_GAP)
    parser.add_argument(
        "--gapfill-temp-points", type=int, default=DEFAULT_GAPFILL_TEMP_POINTS
    )
    parser.add_argument("--max-run-frames", type=int, default=DEFAULT_MAX_RUN_FRAMES)
    parser.add_argument(
        "--run-overlap-frames", type=int, default=DEFAULT_RUN_OVERLAP_FRAMES
    )
    parser.add_argument("--recall-min", type=float, default=DEFAULT_RECALL_MIN)
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument("--max-tracks", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--stream-sqlite-rows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--write-pred-sqlite", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def apply_fixed_practical_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.speed_profile = "production_gapfill_track_anchor_count"
    args.solver_mode = "penalty"
    args.recall_constraint_mode = "exact_dp"
    args.proxy_recall_penalty_weight = 0.0
    args.surrogate_pool_factor = DEFAULT_SURROGATE_POOL_FACTOR
    args.surrogate_peak_factor = DEFAULT_SURROGATE_PEAK_FACTOR
    args.surrogate_neighbor_radius = DEFAULT_SURROGATE_NEIGHBOR_RADIUS
    args.surrogate_shape_weight = DEFAULT_SURROGATE_SHAPE_WEIGHT
    args.saliency_shape_eta = DEFAULT_SALIENCY_SHAPE_ETA
    args.saliency_area_eta = DEFAULT_SALIENCY_AREA_ETA
    args.interval_iou_weight = DEFAULT_INTERVAL_IOU_WEIGHT
    args.shape_switch_weight = DEFAULT_SHAPE_SWITCH_WEIGHT
    args.shape_distance_weight = DEFAULT_SHAPE_DISTANCE_WEIGHT
    args.shape_update_threshold_ratio = DEFAULT_SHAPE_UPDATE_THRESHOLD_RATIO
    args.shape_penalty_adapt_gain = DEFAULT_SHAPE_PENALTY_ADAPT_GAIN
    args.shape_distance_relief = DEFAULT_SHAPE_DISTANCE_RELIEF
    args.shape_switch_relief = DEFAULT_SHAPE_SWITCH_RELIEF
    args.shape_distance_min_scale = DEFAULT_SHAPE_DISTANCE_MIN_SCALE
    args.shape_switch_min_scale = DEFAULT_SHAPE_SWITCH_MIN_SCALE
    args.dynamic_max_gap_factor = DEFAULT_DYNAMIC_MAX_GAP_FACTOR
    args.dp_eval_scale = DEFAULT_DP_EVAL_SCALE
    args.dp_eval_pad = DEFAULT_DP_EVAL_PAD
    args.penalty_binary_steps = DEFAULT_PENALTY_BINARY_STEPS
    args.penalty_max = DEFAULT_PENALTY_MAX
    args.recall_budget_binary_steps = DEFAULT_RECALL_BUDGET_BINARY_STEPS
    args.recall_budget_max_mu = DEFAULT_RECALL_BUDGET_MAX_MU
    args.path_recall_violation_weight = DEFAULT_PATH_RECALL_VIOLATION_WEIGHT
    args.pair_vote_refine_enabled = True
    args.exact_recall_repair_enabled = True
    args.exact_recall_repair_max_passes = DEFAULT_EXACT_RECALL_REPAIR_MAX_PASSES
    args.exact_recall_repair_topk = DEFAULT_EXACT_RECALL_REPAIR_TOPK
    args.exact_recall_repair_scale_deltas = ",".join(
        str(v) for v in DEFAULT_EXACT_RECALL_REPAIR_SCALE_DELTAS
    )
    return args


def process_single_run(run: InstanceRun, args: argparse.Namespace) -> dict[str, object]:
    length = len(run.frame_numbers)
    if length <= 0:
        return {
            "union_rows": [],
            "final_keyframes": [],
            "stream_row": None,
            "interval_eval_count": 0,
            "interval_eval_frames": 0,
            "candidate_frame_count": 0,
        }
    target_count = max(2, min(length, int(round(length * float(args.target_ratio)))))
    stage_times: dict[str, float] = {}

    stage_t0 = time.perf_counter()
    eval_contexts = build_frame_eval_contexts(run, args)
    stage_times["build_eval_contexts_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    candidates_by_frame = build_frame_candidates(run, [], eval_contexts, args)
    stage_times["build_candidates_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    candidate_frames, surrogate_frames, saliency_scores = build_candidate_frame_pool(
        run, candidates_by_frame, target_count, args
    )
    stage_times["build_candidate_pool_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    (
        chosen_frames,
        chosen_candidate_ids,
        counters,
        _cache,
        _best_lambda,
    ) = run_multistate_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        args,
        eval_contexts=eval_contexts,
    )
    stage_times["solve_dp_seconds"] = float(time.perf_counter() - stage_t0)

    keyframe_vectors = np.asarray(
        [
            np.asarray(
                candidates_by_frame[int(frame_idx)][int(cand_id)].vector,
                dtype=np.float32,
            )
            for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)
        ],
        dtype=np.float32,
    )

    stage_t0 = time.perf_counter()
    keyframe_vectors = pair_vote_refine_keyframe_vectors(
        run, chosen_frames, keyframe_vectors, args
    )
    stage_times["pair_vote_refine_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    keyframe_vectors = repair_keyframe_vectors_for_exact_recall(
        run, chosen_frames, keyframe_vectors, candidates_by_frame, args
    )
    stage_times["exact_recall_repair_seconds"] = float(time.perf_counter() - stage_t0)

    chosen_candidate_ids = assign_candidate_ids_to_keyframes(
        chosen_frames, keyframe_vectors, candidates_by_frame
    )

    stage_t0 = time.perf_counter()
    objective, interval_infos, total_recall_budget = evaluate_keyframe_path(
        run, chosen_frames, keyframe_vectors, args, eval_contexts=eval_contexts
    )
    interp_polygons = interpolate_run(run, chosen_frames, keyframe_vectors)
    stage_times["final_eval_seconds"] = float(time.perf_counter() - stage_t0)

    chosen_frame_to_candidate = {
        int(frame_idx): int(cand_id)
        for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)
    }
    union_rows = LazyUnionRows(run, interp_polygons, chosen_frames)
    emit_start = int(union_rows.emit_start)
    emit_end = int(union_rows.emit_end)
    emit_frame_count = int(max(0, emit_end - emit_start))

    final_keyframes: list[dict[str, object]] = []
    for keyframe_pos, frame_idx in enumerate(chosen_frames):
        if int(frame_idx) < emit_start or int(frame_idx) >= emit_end:
            continue
        final_keyframes.append(
            {
                "track_id": str(run.track_id),
                "run_id": int(run.run_id),
                "frame": int(run.frame_numbers[frame_idx]),
                "candidate_id": int(chosen_frame_to_candidate.get(int(frame_idx), -1)),
                "polygons": [
                    np.asarray(poly, dtype=np.float32).tolist()
                    for poly in split_vector_to_polygons(
                        keyframe_vectors[keyframe_pos],
                        run.contour_count,
                        run.anchors_per_contour,
                    )
                ],
            }
        )

    shape_distance_total = (
        float(
            np.sum(
                np.asarray(
                    [info.shape_distance for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    shape_update_count = (
        float(
            np.sum(
                np.asarray(
                    [info.shape_update for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_distance = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_distance for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_update = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_update for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_interval_frame_loss = (
        float(
            np.mean(
                np.asarray(
                    [info.frame_loss_mean for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_distance_scale = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_distance_scale for info in interval_infos],
                    dtype=np.float64,
                )
            )
        )
        if interval_infos
        else 1.0
    )
    mean_shape_switch_scale = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_switch_scale for info in interval_infos],
                    dtype=np.float64,
                )
            )
        )
        if interval_infos
        else 1.0
    )
    mean_recall_budget = float(total_recall_budget / max(length, 1))
    achieved_recall_floor = float(max(0.0, 1.0 - mean_recall_budget))
    recall_budget_violation = float(recall_violation(total_recall_budget, length, args))
    keyframe_rate = float(len(chosen_frames) / max(length, 1))
    gapfilled_frame_count = (
        int(np.sum(run.gapfilled_flags.astype(np.int32)))
        if run.gapfilled_flags is not None
        else 0
    )
    shape_update_rate = float(shape_update_count / max(length, 1))
    shape_distance_rate = float(shape_distance_total / max(length, 1))
    mean_state_count = (
        float(
            np.mean(
                np.asarray(
                    [len(frame_candidates) for frame_candidates in candidates_by_frame],
                    dtype=np.float64,
                )
            )
        )
        if candidates_by_frame
        else 0.0
    )
    stream_row = {
        "stream_id": str(run.stream_id),
        "track_id": str(run.track_id),
        "run_id": int(run.run_id),
        "frame_count": int(length),
        "emit_frame_count": int(emit_frame_count),
        "chunk_index": int(run.chunk_index),
        "chunk_count": int(run.chunk_count),
        "chunk_process_start": int(run.chunk_process_start),
        "chunk_process_end": int(run.chunk_process_end),
        "gapfilled_frame_count": int(gapfilled_frame_count),
        "contour_count": int(run.contour_count),
        "anchors_per_contour": int(run.anchors_per_contour),
        "run_target_total_points": int(run.run_target_total_points),
        "predicted_total_points_p90": float(
            np.quantile(run.predicted_total_points.astype(np.float64), 0.90)
        )
        if run.predicted_total_points is not None
        and len(run.predicted_total_points) > 0
        else 0.0,
        "predicted_total_points_mean": float(
            np.mean(run.predicted_total_points.astype(np.float64))
        )
        if run.predicted_total_points is not None
        and len(run.predicted_total_points) > 0
        else 0.0,
        "candidate_frame_count": int(len(candidate_frames)),
        "mean_state_count": float(mean_state_count),
        "surrogate_frame_count": int(len(surrogate_frames)),
        "target_keyframes": int(target_count),
        "chosen_keyframes": int(len(chosen_frames)),
        "achieved_ratio": float(keyframe_rate),
        "keyframe_rate": float(keyframe_rate),
        "objective": float(objective),
        "mean_interval_frame_loss": float(mean_interval_frame_loss),
        "mean_shape_distance": float(mean_shape_distance),
        "mean_shape_update": float(mean_shape_update),
        "shape_distance_total": float(shape_distance_total),
        "shape_distance_rate": float(shape_distance_rate),
        "shape_update_count": float(shape_update_count),
        "shape_update_rate": float(shape_update_rate),
        "mean_shape_distance_scale": float(mean_shape_distance_scale),
        "mean_shape_switch_scale": float(mean_shape_switch_scale),
        "mean_recall_budget": float(mean_recall_budget),
        "achieved_recall_floor": float(achieved_recall_floor),
        "recall_budget_violation": float(recall_budget_violation),
        "interval_eval_count": int(counters["interval_evals"]),
        "interval_eval_frames": int(counters["interval_frames"]),
        "max_saliency": float(np.max(saliency_scores))
        if saliency_scores.size > 0
        else 0.0,
        **{key: float(val) for key, val in stage_times.items()},
    }
    return {
        "union_rows": union_rows,
        "final_keyframes": final_keyframes,
        "stream_row": stream_row,
        "interval_eval_count": int(counters["interval_evals"]),
        "interval_eval_frames": int(counters["interval_frames"]),
        "candidate_frame_count": int(len(candidate_frames)),
        "stage_times": {key: float(val) for key, val in stage_times.items()},
    }


def main() -> None:
    args = apply_fixed_practical_defaults(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = output_dir / "opt"
    exact_dir = output_dir / "exact"
    pred_dir = output_dir / "pred"
    pred_sqlite = pred_dir / "predictions.sqlite"
    t0 = time.perf_counter()

    predictor: LearnedPointPredictor | None = None
    if bool(args.adaptive_anchor_counts):
        predictor = LearnedPointPredictor(
            Path(args.point_predictor_model_dir), str(args.predictor_device)
        )
    effective_workers = max(1, int(args.num_workers))
    streaming_rows = bool(args.stream_sqlite_rows) and effective_workers == 1
    runs: list[InstanceRun] = []
    segmentation_stats: dict[str, int] = {}
    if not streaming_rows:
        rows = load_rows(args.input_sqlite)
        runs, segmentation_stats = build_track_streams(
            rows,
            anchors_per_contour=int(args.anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(args.predictor_batch_size),
            adaptive_anchor_counts=bool(args.adaptive_anchor_counts),
            adaptive_point_quantile=float(args.adaptive_point_quantile),
            adaptive_point_offset=int(args.adaptive_point_offset),
            min_anchors_per_contour=int(args.min_anchors_per_contour),
            gapfill_enabled=bool(args.gapfill_enabled),
            gapfill_max_gap=int(args.gapfill_max_gap),
            gapfill_temp_points=int(args.gapfill_temp_points),
            max_tracks=int(args.max_tracks),
            max_run_frames=int(args.max_run_frames),
            run_overlap_frames=int(args.run_overlap_frames),
        )

    run_count = int(len(runs))
    union_rows_all: list[dict[str, object]] = []
    union_rows: list[dict[str, object]] = []
    union_store: SqliteUnionRowStore | None = None
    union_row_count = 0
    final_keyframes: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    total_interval_evals = 0
    total_interval_frames = 0
    total_candidate_frames = 0
    total_stage_times: dict[str, float] = {}

    def collect_result(result: dict[str, object]) -> None:
        nonlocal total_interval_evals, total_interval_frames, total_candidate_frames
        final_keyframes.extend(result["final_keyframes"])
        stream_row = result["stream_row"]
        if stream_row is not None:
            stream_rows.append(stream_row)
        total_interval_evals += int(result["interval_eval_count"])
        total_interval_frames += int(result["interval_eval_frames"])
        total_candidate_frames += int(result["candidate_frame_count"])
        for key, value in result.get("stage_times", {}).items():
            total_stage_times[str(key)] = float(
                total_stage_times.get(str(key), 0.0) + float(value)
            )

    if streaming_rows:
        union_store = SqliteUnionRowStore(output_dir / ".polygon_union_rows.tmp.sqlite")
        for run in iter_track_streams_from_sqlite(
            args.input_sqlite,
            anchors_per_contour=int(args.anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(args.predictor_batch_size),
            adaptive_anchor_counts=bool(args.adaptive_anchor_counts),
            adaptive_point_quantile=float(args.adaptive_point_quantile),
            adaptive_point_offset=int(args.adaptive_point_offset),
            min_anchors_per_contour=int(args.min_anchors_per_contour),
            gapfill_enabled=bool(args.gapfill_enabled),
            gapfill_max_gap=int(args.gapfill_max_gap),
            gapfill_temp_points=int(args.gapfill_temp_points),
            max_tracks=int(args.max_tracks),
            max_run_frames=int(args.max_run_frames),
            run_overlap_frames=int(args.run_overlap_frames),
            segmentation_stats=segmentation_stats,
        ):
            run_count += 1
            result = process_single_run(run, args)
            union_store.add_rows(result["union_rows"])
            collect_result(result)
            run = None
            result = None
            __import__("gc").collect()
        if predictor is not None:
            try:
                predictor.model.to("cpu")
                torch_mod = __import__("torch")
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.synchronize()
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass
        union_store.commit()
        union_row_count = int(union_store.row_count)
    elif effective_workers == 1 or len(runs) <= 1:
        union_store = SqliteUnionRowStore(output_dir / ".polygon_union_rows.tmp.sqlite")
        for run_idx, run in enumerate(runs):
            result = process_single_run(run, args)
            union_store.add_rows(result["union_rows"])
            collect_result(result)
            runs[run_idx] = None
            result = None
            __import__("gc").collect()
        union_store.commit()
        union_row_count = int(union_store.row_count)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=mp_ctx
        ) as executor:
            results = list(executor.map(process_single_run, runs, [args] * len(runs)))
        for result in results:
            union_rows_all.extend(result["union_rows"])
            collect_result(result)
        union_rows = sorted(
            union_rows_all,
            key=lambda row: (int(row["frame"]), int(str(row["track_id"]))),
        )
        union_row_count = int(len(union_rows))
    chunk_counts_by_run: dict[tuple[str, int], int] = {}
    for row in stream_rows:
        if int(row.get("chunk_count", 1)) < 0:
            key = (str(row["track_id"]), int(row["run_id"]))
            chunk_counts_by_run[key] = max(
                int(chunk_counts_by_run.get(key, 0)), int(row["chunk_index"]) + 1
            )
    for row in stream_rows:
        if int(row.get("chunk_count", 1)) < 0:
            key = (str(row["track_id"]), int(row["run_id"]))
            chunk_count = int(chunk_counts_by_run.get(key, int(row["chunk_index"]) + 1))
            chunk_index = int(row["chunk_index"])
            row["chunk_count"] = int(chunk_count)
            row["stream_id"] = str(row["stream_id"]).replace(
                f":chunk{chunk_index + 1}:instance",
                f":chunk{chunk_index + 1}of{chunk_count}:instance",
            )
    opt_dir.mkdir(parents=True, exist_ok=True)
    if union_store is not None:
        union_store.write_union_json(opt_dir / "interpolated_union.json")
    else:
        (opt_dir / "interpolated_union.json").write_text(
            json.dumps(union_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_compact_json_array(opt_dir / "final_keyframes.json", final_keyframes)
    write_csv(
        stream_rows,
        opt_dir / "stream_segments.csv",
        [
            "stream_id",
            "track_id",
            "run_id",
            "frame_count",
            "gapfilled_frame_count",
            "contour_count",
            "anchors_per_contour",
            "run_target_total_points",
            "predicted_total_points_p90",
            "predicted_total_points_mean",
            "candidate_frame_count",
            "mean_state_count",
            "surrogate_frame_count",
            "target_keyframes",
            "chosen_keyframes",
            "achieved_ratio",
            "keyframe_rate",
            "objective",
            "mean_interval_frame_loss",
            "mean_shape_distance",
            "mean_shape_update",
            "shape_distance_total",
            "shape_distance_rate",
            "shape_update_count",
            "shape_update_rate",
            "mean_shape_distance_scale",
            "mean_shape_switch_scale",
            "mean_recall_budget",
            "achieved_recall_floor",
            "recall_budget_violation",
            "interval_eval_count",
            "interval_eval_frames",
            "max_saliency",
            "build_eval_contexts_seconds",
            "build_candidates_seconds",
            "build_candidate_pool_seconds",
            "solve_dp_seconds",
            "pair_vote_refine_seconds",
            "exact_recall_repair_seconds",
            "final_eval_seconds",
            "emit_frame_count",
            "chunk_index",
            "chunk_count",
            "chunk_process_start",
            "chunk_process_end",
        ],
    )

    optimizer_seconds = float(time.perf_counter() - t0)
    optimizer_summary = {
        "description": "Production polygon optimizer kernel with gapfill-first track-level anchor counts and pair-vote keyframe-shape refinement.",
        "input_sqlite": str(args.input_sqlite),
        "output_dir": str(opt_dir),
        "run_count": int(run_count),
        "gapfill_enabled": bool(args.gapfill_enabled),
        "gapfill_max_gap": int(args.gapfill_max_gap),
        "gapfill_temp_points": int(args.gapfill_temp_points),
        "max_run_frames": int(args.max_run_frames),
        "run_overlap_frames": int(args.run_overlap_frames),
        "num_workers": int(effective_workers),
        "stream_sqlite_rows": bool(streaming_rows),
        "row_count": int(union_row_count),
        "target_ratio": float(args.target_ratio),
        "anchors_per_contour_cap": int(args.anchors_per_contour),
        "adaptive_anchor_counts": bool(args.adaptive_anchor_counts),
        "point_predictor_model_dir": str(args.point_predictor_model_dir)
        if bool(args.adaptive_anchor_counts)
        else None,
        "predictor_device": str(args.predictor_device),
        "predictor_batch_size": int(args.predictor_batch_size),
        "adaptive_point_quantile": float(args.adaptive_point_quantile),
        "adaptive_point_offset": int(args.adaptive_point_offset),
        "min_anchors_per_contour": int(args.min_anchors_per_contour),
        "solver_mode": str(args.solver_mode),
        "recall_constraint_mode": str(args.recall_constraint_mode),
        "recall_min": float(args.recall_min),
        "pair_vote_refine_enabled": bool(args.pair_vote_refine_enabled),
        "surrogate_pool_factor": float(args.surrogate_pool_factor),
        "surrogate_peak_factor": float(args.surrogate_peak_factor),
        "surrogate_neighbor_radius": int(args.surrogate_neighbor_radius),
        "surrogate_shape_weight": float(args.surrogate_shape_weight),
        "saliency_shape_eta": float(args.saliency_shape_eta),
        "saliency_area_eta": float(args.saliency_area_eta),
        "shape_switch_weight": float(args.shape_switch_weight),
        "shape_distance_weight": float(args.shape_distance_weight),
        "shape_update_threshold_ratio": float(args.shape_update_threshold_ratio),
        "penalty_binary_steps": int(args.penalty_binary_steps),
        "recall_budget_binary_steps": int(args.recall_budget_binary_steps),
        "dp_eval_scale": float(args.dp_eval_scale),
        "dp_eval_pad": int(args.dp_eval_pad),
        "segmentation_stats": {
            key: int(val) for key, val in segmentation_stats.items()
        },
        "mean_run_anchors_per_contour": float(
            np.mean(
                np.asarray(
                    [row["anchors_per_contour"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "median_run_anchors_per_contour": float(
            np.median(
                np.asarray(
                    [row["anchors_per_contour"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "max_run_anchors_per_contour": int(
            max(int(row["anchors_per_contour"]) for row in stream_rows)
        )
        if stream_rows
        else 0,
        "mean_gapfilled_frame_count": float(
            np.mean(
                np.asarray(
                    [row["gapfilled_frame_count"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_achieved_ratio": float(
            np.mean(
                np.asarray(
                    [row["achieved_ratio"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_keyframe_rate": float(
            np.mean(
                np.asarray(
                    [row["keyframe_rate"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_state_count": float(
            np.mean(
                np.asarray(
                    [row["mean_state_count"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_interval_frame_loss": float(
            np.mean(
                np.asarray(
                    [row["mean_interval_frame_loss"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_distance": float(
            np.mean(
                np.asarray(
                    [row["mean_shape_distance"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_update": float(
            np.mean(
                np.asarray(
                    [row["mean_shape_update"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_distance_rate": float(
            np.mean(
                np.asarray(
                    [row["shape_distance_rate"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_update_rate": float(
            np.mean(
                np.asarray(
                    [row["shape_update_rate"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_recall_budget": float(
            np.mean(
                np.asarray(
                    [row["mean_recall_budget"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_achieved_recall_floor": float(
            np.mean(
                np.asarray(
                    [row["achieved_recall_floor"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 1.0,
        "mean_recall_budget_violation": float(
            np.mean(
                np.asarray(
                    [row["recall_budget_violation"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_candidate_frame_count": float(
            np.mean(
                np.asarray(
                    [row["candidate_frame_count"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "interval_eval_count": int(total_interval_evals),
        "interval_eval_frames": int(total_interval_frames),
        "candidate_frame_count_total": int(total_candidate_frames),
        "optimizer_seconds": float(optimizer_seconds),
        "stage_seconds_total": {
            key: float(val) for key, val in sorted(total_stage_times.items())
        },
        "stage_seconds_mean_per_run": {
            key: float(val / max(run_count, 1))
            for key, val in sorted(total_stage_times.items())
        },
        "artifacts": {
            "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
            "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
            "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
        },
    }
    (opt_dir / "summary.json").write_text(
        json.dumps(optimizer_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exact_summary: dict[str, object] | None = None
    if union_store is not None:
        try:
            if bool(args.evaluate_exact):
                exact_summary = union_store.evaluate_exact(args.input_sqlite, exact_dir)
            if bool(args.write_pred_sqlite):
                union_store.write_pred_sqlite(pred_sqlite)
        finally:
            union_store.close(unlink=True)
    else:
        if bool(args.evaluate_exact):
            exact_summary = evaluate_union_exact(
                union_rows, args.input_sqlite, exact_dir
            )
        if bool(args.write_pred_sqlite):
            union_rows_to_pred_sqlite(union_rows, pred_sqlite)

    summary = {
        "description": "Production polygon optimizer with gapfill-first track-level anchor counts.",
        "input_sqlite": str(args.input_sqlite),
        "output_dir": str(output_dir),
        "optimizer_summary": optimizer_summary,
        "exact_summary": exact_summary,
        "artifacts": {
            "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
            "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
            "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
            "optimizer_summary_json": str(opt_dir / "summary.json"),
            "exact_summary_json": None
            if exact_summary is None
            else str(exact_dir / "summary.json"),
            "pred_sqlite": str(pred_sqlite) if bool(args.write_pred_sqlite) else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
