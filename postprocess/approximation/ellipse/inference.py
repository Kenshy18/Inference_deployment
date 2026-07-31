from __future__ import annotations

import subprocess

try:
    import orjson
except ModuleNotFoundError:
    orjson = None


"""Ellipse approximation entrypoint.

This module accepts the tracked-SQLite contract and performs only:
- K1 exact ellipse approximation
- K1/K2 routing with the current V6 smoothing preset
- K2 V5 fallback inference
- final SQLite / metrics / summary export
"""
import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
from .runtime_fst import fst

infer__TORCH_MODULE: ModuleType | None = None
infer__K2V5_MODULE: ModuleType | None = None


def _publish_ellipse_preview(metric: dict[str, object]) -> None:
    """Publish geometry only; rendering/decoding remains off the K1/K2 path."""

    from common.live_preview import PreviewGeometry, active_postprocess_preview

    preview = active_postprocess_preview()
    if preview is None or not preview.should_sample("ellipse_approximation"):
        return
    try:
        ellipses = tuple(
            tuple(float(value) for value in ellipse[:5])
            for ellipse in json.loads(str(metric["ellipse_params"]))
        )
        preview.submit(
            PreviewGeometry(
                int(metric["frame"]),
                "ellipse_approximation",
                "ellipse approximation",
                ellipses=ellipses,
                track_id=str(metric["track_id"]),
                detail=f"{str(metric.get('mode', 'K1')).upper()} / IoU {float(metric.get('iou', 0.0)):.3f}",
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return


def infer_get_torch_module() -> ModuleType:
    global infer__TORCH_MODULE
    if infer__TORCH_MODULE is None:
        import torch

        infer__TORCH_MODULE = torch
    return infer__TORCH_MODULE


def infer_get_k2v5_module() -> ModuleType:
    global infer__K2V5_MODULE
    if infer__K2V5_MODULE is None:
        from .runtime_k2v5 import k2v5

        infer__K2V5_MODULE = k2v5
    return infer__K2V5_MODULE


infer_K1_COST_NORM_AREA_FLOOR = 1.0
infer_K2_V5_INFER_CONFIG = {
    "image_size": 192,
    "base_width": 32,
    "slot_dim": 256,
    "decoder_layers": 3,
    "num_heads": 8,
    "render_sharpness": 28.0,
}


def infer_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone ellipse approximation: exact K1 split + K2 V5 distilled model."
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k1-recall-target", type=float, default=0.99)
    parser.add_argument("--k1-exact-refine-rounds", type=int, default=1)
    parser.add_argument(
        "--k1-workers",
        type=int,
        default=0,
        help="K1 worker processes; 0 uses the available CPU count",
    )
    parser.add_argument("--k2-run-dir", type=Path, default=MODULE_DIR / "assets/k2_v5")
    parser.add_argument("--k2-device", type=str, default="cuda")
    parser.add_argument("--k2-batch-size", type=int, default=64)
    parser.add_argument("--k2-prep-workers", type=int, default=1)
    parser.add_argument(
        "--k2-precision",
        type=str,
        default="fp32",
        choices=("fp32", "fp16"),
        help="K2 model precision. fp16 is CUDA-only and falls back to fp32 on CPU.",
    )
    parser.add_argument(
        "--k2-forward-mode",
        type=str,
        default="states_only",
        choices=("states_only", "full"),
        help=(
            "states_only skips the K2 network's unused soft-mask rendering "
            "(default); full is retained for diagnostics."
        ),
    )
    parser.add_argument(
        "--k2-profile-stages",
        action="store_true",
        help="Synchronize and record K2 batch-stage timings for profiling.",
    )
    parser.add_argument(
        "--k2-cudnn-benchmark",
        type=str,
        default="off",
        choices=("on", "off"),
        help="cuDNN benchmark can speed repeated shapes but adds first-batch autotune overhead.",
    )
    parser.add_argument(
        "--k2-tf32",
        type=str,
        default="off",
        choices=("default", "on", "off"),
        help=(
            "Control TF32 for K2 CUDA matmul/cuDNN paths "
            "(default: off for CPU/GPU numerical agreement)."
        ),
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="k1n_sequence",
        choices=(
            "threshold_only",
            "threshold_soft",
            "threshold_hysteresis",
            "k1n_sequence",
            "track_dp",
            "band",
        ),
        help="K1/K2 routing mode. 'threshold_only' uses per-row K1 cost only. 'threshold_soft' applies weak temporal smoothing near the threshold. 'threshold_hysteresis' uses explicit enter/exit thresholds plus entry confirmation. 'k1n_sequence' uses only the K1 normalized cost sequence with hysteresis and protected island cleanup. 'track_dp' performs track-level non-learned DP with asymmetric switching and soft run penalties. 'band' uses the original track expansion logic.",
    )
    parser.add_argument(
        "--k1-cost-routing",
        type=str,
        default="normalized",
        choices=("raw", "normalized"),
        help="Cost scale used for K1/K2 routing. Debug columns still keep both raw and normalized costs.",
    )
    parser.add_argument("--threshold", type=int, default=5000)
    parser.add_argument("--threshold-edge", type=int, default=-1)
    parser.add_argument("--threshold-norm", type=float, default=0.18)
    parser.add_argument("--threshold-edge-norm", type=float, default=0.18)
    parser.add_argument("--k2-soft-ema-alpha", type=float, default=0.8)
    parser.add_argument("--k2-soft-band-ratio", type=float, default=0.03)
    parser.add_argument("--k2-soft-exit-ratio", type=float, default=-1.0)
    parser.add_argument("--k2-soft-strong-ratio", type=float, default=0.1)
    parser.add_argument("--k2-soft-k1-keep-cost", type=int, default=-1)
    parser.add_argument("--k2-soft-k1-keep-cost-norm", type=float, default=-1.0)
    parser.add_argument("--k2-soft-reset-gap", type=int, default=2)
    parser.add_argument("--k2-soft-merge-islands-max-len", type=int, default=0)
    parser.add_argument(
        "--k2-soft-merge-policy",
        type=str,
        default="symmetric",
        choices=("symmetric", "prefer_k2"),
    )
    parser.add_argument("--k2-hyst-enter", type=int, default=6000)
    parser.add_argument("--k2-hyst-enter-edge", type=int, default=-1)
    parser.add_argument("--k2-hyst-exit", type=int, default=4000)
    parser.add_argument("--k2-hyst-exit-edge", type=int, default=-1)
    parser.add_argument("--k2-hyst-enter-norm", type=float, default=0.20)
    parser.add_argument("--k2-hyst-enter-edge-norm", type=float, default=-1.0)
    parser.add_argument("--k2-hyst-exit-norm", type=float, default=0.14)
    parser.add_argument("--k2-hyst-exit-edge-norm", type=float, default=-1.0)
    parser.add_argument("--k2-hyst-confirm-frames", type=int, default=2)
    parser.add_argument("--k2-hyst-reset-gap", type=int, default=2)
    parser.add_argument("--k1n-seq-enter-norm", type=float, default=-1.0)
    parser.add_argument("--k1n-seq-exit-norm", type=float, default=0.13)
    parser.add_argument("--k1n-seq-strong-enter-norm", type=float, default=-1.0)
    parser.add_argument("--k1n-seq-strong-exit-norm", type=float, default=-1.0)
    parser.add_argument("--k1n-seq-protect-k2-iou-below", type=float, default=0.65)
    parser.add_argument("--k1n-seq-smooth-window", type=int, default=11)
    parser.add_argument("--k1n-seq-enter-confirm-frames", type=int, default=6)
    parser.add_argument("--k1n-seq-exit-confirm-frames", type=int, default=6)
    parser.add_argument("--k1n-seq-merge-short-k1-max-len", type=int, default=5)
    parser.add_argument("--k1n-seq-merge-short-k2-max-len", type=int, default=5)
    parser.add_argument("--k1n-seq-reset-gap", type=int, default=2)
    parser.add_argument("--k2-dp-error-weight", type=float, default=1.0)
    parser.add_argument("--k2-dp-instability-weight", type=float, default=0.4)
    parser.add_argument("--k2-dp-edge-bonus", type=float, default=0.2)
    parser.add_argument("--k2-dp-k2-bias", type=float, default=0.28)
    parser.add_argument("--k2-dp-switch-12", type=float, default=0.8)
    parser.add_argument("--k2-dp-switch-21", type=float, default=1.5)
    parser.add_argument("--k2-dp-short-k1-gamma", type=float, default=1.55)
    parser.add_argument("--k2-dp-short-k2-gamma", type=float, default=0.6)
    parser.add_argument("--k2-dp-short-k1-tau", type=float, default=1.6)
    parser.add_argument("--k2-dp-short-k2-tau", type=float, default=1.3)
    parser.add_argument("--k2-dp-short-len-cap", type=int, default=6)
    parser.add_argument("--k2-dp-reset-gap", type=int, default=2)
    parser.add_argument("--k2-dp-merge-short-k1-max-len", type=int, default=24)
    parser.add_argument("--k2-dp-merge-short-k2-max-len", type=int, default=12)
    parser.add_argument("--k2-dp-merge-short-k2-keep-cost", type=int, default=10000)
    parser.add_argument("--k2-dp-force-k2-cost", type=int, default=10000)
    parser.add_argument(
        "--k2-dp-merge-short-k2-keep-cost-norm", type=float, default=0.35
    )
    parser.add_argument("--k2-dp-force-k2-cost-norm", type=float, default=0.35)
    parser.add_argument("--k2-band-radius", type=int, default=3)
    parser.add_argument("--k2-band-error-percentile", type=float, default=92.0)
    parser.add_argument("--k2-band-instability-percentile", type=float, default=90.0)
    parser.add_argument("--k2-band-instability-floor", type=float, default=0.4)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-tracks", type=int, default=0)
    return parser


def infer_filter_rows(
    rows: list[tuple[int, str, str]], max_rows: int, max_tracks: int
) -> list[tuple[int, str, str]]:
    filtered = rows
    if max_tracks > 0:
        track_ids: list[str] = []
        seen: set[str] = set()
        for _, track_id, _ in filtered:
            if track_id in seen:
                continue
            seen.add(track_id)
            track_ids.append(track_id)
            if len(track_ids) >= max_tracks:
                break
        allowed = set(track_ids)
        filtered = [row for row in filtered if row[1] in allowed]
    if max_rows > 0:
        filtered = filtered[:max_rows]
    return filtered


def infer_compute_weighted_error_norm(weighted_error: float, gt_area: float) -> float:
    return float(weighted_error) / max(
        float(gt_area), float(infer_K1_COST_NORM_AREA_FLOOR)
    )


def infer_add_weighted_error_norm(metric_row: dict[str, object]) -> dict[str, object]:
    out = dict(metric_row)
    out["weighted_error_norm"] = infer_compute_weighted_error_norm(
        float(out.get("weighted_error", 0.0)),
        float(out.get("gt_area", 0.0)),
    )
    return out


def infer_prepare_k1_routing_metrics_lookup(
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, cost_field: str
) -> dict[tuple[int, str], dict[str, object]]:
    if cost_field == "weighted_error":
        return k1_metrics_lookup
    routing_lookup: dict[tuple[int, str], dict[str, object]] = {}
    for key, row in k1_metrics_lookup.items():
        route_cost = row.get(cost_field)
        if route_cost in (None, ""):
            if cost_field == "weighted_error_norm":
                route_cost = infer_compute_weighted_error_norm(
                    float(row.get("weighted_error", 0.0)),
                    float(row.get("gt_area", 0.0)),
                )
            else:
                route_cost = row.get("weighted_error", 0.0)
        routing_row = dict(row)
        routing_row["weighted_error_raw"] = row.get("weighted_error", 0.0)
        routing_row["weighted_error"] = float(route_cost)
        routing_row["weighted_error_routing_field"] = str(cost_field)
        routing_lookup[key] = routing_row
    return routing_lookup


def infer_write_metrics_csv(
    metric_rows: list[dict[str, object]], output_path: Path
) -> None:
    fieldnames = [
        "frame",
        "track_id",
        "mode",
        "candidate_name",
        "gt_area",
        "pred_area",
        "intersection",
        "union",
        "recall",
        "precision",
        "iou",
        "weighted_error",
        "weighted_error_norm",
        "ellipse_params",
        "branch",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([infer_add_weighted_error_norm(row) for row in metric_rows])


def infer_evaluate_mixed_metric_rows(
    metric_rows: list[dict[str, object]], *, total_gt_rows: int, total_sub_rows: int
) -> dict[str, float]:
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    mean_recall: list[float] = []
    mean_precision: list[float] = []
    mean_iou: list[float] = []
    recall_below_090 = 0
    recall_below_095 = 0
    k1_intersection = k1_union = k1_gt_area = 0
    k2_intersection = k2_union = k2_gt_area = 0
    k1_count = 0
    k2_count = 0
    for row in metric_rows:
        intersection = int(row["intersection"])
        union = int(row["union"])
        gt_area = int(row["gt_area"])
        pred_area = int(row["pred_area"])
        recall = float(row["recall"])
        precision = float(row["precision"])
        iou = float(row["iou"])
        total_intersection += intersection
        total_union += union
        total_gt_area += gt_area
        total_pred_area += pred_area
        mean_recall.append(recall)
        mean_precision.append(precision)
        mean_iou.append(iou)
        if recall < 0.9:
            recall_below_090 += 1
        if recall < 0.95:
            recall_below_095 += 1
        mode = str(row.get("mode", "k1"))
        if mode == "k2":
            k2_count += 1
            k2_intersection += intersection
            k2_union += union
            k2_gt_area += gt_area
        else:
            k1_count += 1
            k1_intersection += intersection
            k1_union += union
            k1_gt_area += gt_area
    return {
        "global_recall": total_intersection / total_gt_area if total_gt_area else 1.0,
        "global_precision": total_intersection / total_pred_area
        if total_pred_area
        else 1.0,
        "global_iou": total_intersection / total_union if total_union else 1.0,
        "mean_recall": float(np.mean(mean_recall)) if mean_recall else 1.0,
        "mean_precision": float(np.mean(mean_precision)) if mean_precision else 1.0,
        "mean_iou": float(np.mean(mean_iou)) if mean_iou else 1.0,
        "recall_below_090": int(recall_below_090),
        "recall_below_095": int(recall_below_095),
        "missing_rows": int(max(0, int(total_gt_rows) - len(metric_rows))),
        "total_gt_rows": int(total_gt_rows),
        "total_sub_rows": int(total_sub_rows),
        "k1_count": int(k1_count),
        "k2_count": int(k2_count),
        "k1_recall": k1_intersection / k1_gt_area if k1_gt_area else 0.0,
        "k1_iou": k1_intersection / k1_union if k1_union else 0.0,
        "k2_recall": k2_intersection / k2_gt_area if k2_gt_area else 0.0,
        "k2_iou": k2_intersection / k2_union if k2_union else 0.0,
    }


def infer_solve_subset(
    rows_with_index: list[tuple[int, int, str, str]],
    payloads: list[tuple[tuple[int, int], tuple[int, int], list]],
    gt_polys: list[list[np.ndarray]],
    recall_target: float,
    exact_refine_rounds: int,
    workers: int,
    branch_name: str,
) -> tuple[
    list[tuple[int, tuple[int, str, str]]], list[tuple[int, dict[str, object]]], float
]:
    if not rows_with_index:
        return ([], [], 0.0)
    fst.set_row_local_raster_cache(
        [row[3] for row in rows_with_index], payloads, gt_polys
    )
    started = time.perf_counter()
    solved_rows: list[tuple[int, tuple[int, str, str]]] = []
    solved_metrics: list[tuple[int, dict[str, object]]] = []
    if workers > 1:
        tasks = [
            (
                subset_idx,
                int(frame),
                str(track_id),
                float(recall_target),
                int(exact_refine_rounds),
            )
            for subset_idx, (
                _original_idx,
                frame,
                track_id,
                _polygons_json,
            ) in enumerate(rows_with_index)
        ]
        chunksize = max(1, len(tasks) // (workers * 8))
        # Forking while the GUI preview worker is decoding with OpenCV can
        # inherit a held native lock and leave every K1 worker asleep forever.
        # Live mode therefore uses clean interpreters plus explicit bounded
        # payloads; headless mode retains the fastest fork-inherited cache.
        from common.live_preview import active_postprocess_preview

        preview = active_postprocess_preview()
        if preview is None:
            worker = fst._solve_k1_row_worker
            worker_tasks = tasks
            executor_workers = workers
            context = multiprocessing.get_context("fork")
        else:
            worker = fst._solve_k1_payload_worker
            worker_tasks = [
                (
                    subset_idx,
                    int(frame),
                    str(track_id),
                    str(polygons_json),
                    payloads[subset_idx],
                    gt_polys[subset_idx],
                    float(recall_target),
                    int(exact_refine_rounds),
                )
                for subset_idx, (
                    _original_idx,
                    frame,
                    track_id,
                    polygons_json,
                ) in enumerate(rows_with_index)
            ]
            executor_workers = min(workers, 8)
            context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=executor_workers,
            mp_context=context,
            initializer=fst._k1_pool_init,
        ) as executor:
            for completed, result in enumerate(
                executor.map(worker, worker_tasks, chunksize=chunksize),
                start=1,
            ):
                subset_idx, row_value, metric_row, _solution_entry = result
                original_idx = rows_with_index[subset_idx][0]
                metric_row = dict(metric_row)
                metric_row["branch"] = branch_name
                metric_row["mode"] = "k1"
                metric_row = infer_add_weighted_error_norm(metric_row)
                solved_rows.append((original_idx, row_value))
                solved_metrics.append((original_idx, metric_row))
                _publish_ellipse_preview(metric_row)
                if completed % 1000 == 0 or completed == len(worker_tasks):
                    print(f"{branch_name}: processed {completed}/{len(worker_tasks)}")
    else:
        fst._k1_pool_init()
        for completed, (original_idx, frame, track_id, polygons_json) in enumerate(
            rows_with_index, start=1
        ):
            subset_idx = completed - 1
            pred_json, exact, candidate_name, ellipses = fst.solve_k1_row(
                polygons_json,
                recall_target=float(recall_target),
                exact_refine_rounds=int(exact_refine_rounds),
                prepared_payload=payloads[subset_idx],
                gt_polys=gt_polys[subset_idx],
            )
            metric_row = infer_add_weighted_error_norm(
                {
                    "frame": int(frame),
                    "track_id": str(track_id),
                    "mode": "k1",
                    "candidate_name": candidate_name,
                    "gt_area": int(exact["gt_area"]),
                    "pred_area": int(exact["pred_area"]),
                    "intersection": int(exact["intersection"]),
                    "union": int(exact["union"]),
                    "recall": float(exact["recall"]),
                    "precision": float(exact["precision"]),
                    "iou": float(exact["iou"]),
                    "weighted_error": int(exact["weighted_error"]),
                    "ellipse_params": json.dumps(
                        fst.serialize_ellipses(ellipses), ensure_ascii=True
                    ),
                    "branch": branch_name,
                }
            )
            solved_rows.append((original_idx, (int(frame), str(track_id), pred_json)))
            solved_metrics.append((original_idx, metric_row))
            _publish_ellipse_preview(metric_row)
            if completed % 1000 == 0 or completed == len(rows_with_index):
                print(f"{branch_name}: processed {completed}/{len(rows_with_index)}")
    elapsed = time.perf_counter() - started
    return (solved_rows, solved_metrics, elapsed)


def infer_build_k2_solve_band_edge_aware(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    k1_ellipses_lookup: dict[
        tuple[int, str], list[tuple[float, float, float, float, float]]
    ],
    *,
    threshold_default: float,
    threshold_edge: float,
    edge_keys: set[tuple[int, str]],
    radius: int,
    error_percentile: float,
    instability_percentile: float,
    instability_floor: float,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        errors = np.asarray(
            [float(k1_metrics_lookup[key]["weighted_error"]) for key in keys],
            dtype=np.float64,
        )
        thresholds = np.asarray(
            [
                float(threshold_edge if key in edge_keys else threshold_default)
                for key in keys
            ],
            dtype=np.float64,
        )
        seed_indices = {idx for idx, err in enumerate(errors) if err >= thresholds[idx]}
        high_error_cut = (
            float(np.percentile(errors, error_percentile))
            if len(errors) > 0
            else float(threshold_default)
        )
        instability_scores = np.zeros(len(track_rows), dtype=np.float64)
        for idx in range(1, len(track_rows)):
            prev_key = keys[idx - 1]
            curr_key = keys[idx]
            prev_ellipse = k1_ellipses_lookup[prev_key][0]
            curr_ellipse = k1_ellipses_lookup[curr_key][0]
            _, _, prev_scale = fst.composite_center_and_scale([prev_ellipse])
            _, _, curr_scale = fst.composite_center_and_scale([curr_ellipse])
            ref_scale = max(prev_scale, curr_scale, 8.0)
            center_jump = (
                float(
                    np.hypot(
                        curr_ellipse[0] - prev_ellipse[0],
                        curr_ellipse[1] - prev_ellipse[1],
                    )
                )
                / ref_scale
            )
            area_jump = abs(
                np.log(max(fst.ellipse_area(curr_ellipse), 1.0))
                - np.log(max(fst.ellipse_area(prev_ellipse), 1.0))
            )
            angle_jump = fst.angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
            instability_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
        instability_cut = (
            float(np.percentile(instability_scores, instability_percentile))
            if len(instability_scores) > 0
            else float("inf")
        )
        extra_indices = {
            idx
            for idx, err in enumerate(errors)
            if err >= max(high_error_cut, thresholds[idx] * 0.6)
        }
        extra_indices |= {
            idx
            for idx, score in enumerate(instability_scores)
            if score >= max(instability_cut, instability_floor)
        }
        source_indices = seed_indices | extra_indices
        for src_idx in source_indices:
            src_frame = int(track_rows[src_idx][0])
            for frame, track_id_value, _, _ in track_rows:
                if abs(int(frame) - src_frame) <= radius:
                    selected.add((int(frame), str(track_id_value)))
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "seed_count": len(seed_indices),
                "expanded_count": int(sum((1 for key in keys if key in selected))),
                "error_cut": float(high_error_cut),
                "instability_cut": float(instability_cut)
                if np.isfinite(instability_cut)
                else None,
            }
        )
    summary = {
        "threshold": float(threshold_default),
        "threshold_edge": float(threshold_edge),
        "radius": int(radius),
        "selected_count": len(selected),
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_build_track_local_instability_scores(
    track_rows: list[tuple[int, str, str, int]],
    k1_ellipses_lookup: dict[
        tuple[int, str], list[tuple[float, float, float, float, float]]
    ],
) -> list[float]:
    keys = [(int(frame), str(track_id)) for frame, track_id, _, _ in track_rows]
    if len(keys) <= 1:
        return [0.0] * len(keys)
    adjacent_scores = [0.0] * (len(keys) - 1)
    for idx in range(len(keys) - 1):
        prev_ellipse = k1_ellipses_lookup[keys[idx]][0]
        curr_ellipse = k1_ellipses_lookup[keys[idx + 1]][0]
        _, _, prev_scale = fst.composite_center_and_scale([prev_ellipse])
        _, _, curr_scale = fst.composite_center_and_scale([curr_ellipse])
        ref_scale = max(prev_scale, curr_scale, 8.0)
        center_jump = (
            float(
                np.hypot(
                    curr_ellipse[0] - prev_ellipse[0], curr_ellipse[1] - prev_ellipse[1]
                )
            )
            / ref_scale
        )
        area_jump = abs(
            np.log(max(fst.ellipse_area(curr_ellipse), 1.0))
            - np.log(max(fst.ellipse_area(prev_ellipse), 1.0))
        )
        angle_jump = fst.angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
        adjacent_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
    instability_scores = [0.0] * len(keys)
    for idx in range(len(keys)):
        prev_score = adjacent_scores[idx - 1] if idx > 0 else 0.0
        next_score = adjacent_scores[idx] if idx < len(adjacent_scores) else 0.0
        instability_scores[idx] = float(max(prev_score, next_score))
    return instability_scores


def infer_build_k2_track_dp_selection(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    k1_ellipses_lookup: dict[
        tuple[int, str], list[tuple[float, float, float, float, float]]
    ],
    *,
    threshold_default: float,
    threshold_edge: float,
    edge_keys: set[tuple[int, str]],
    error_weight: float,
    instability_weight: float,
    edge_bonus: float,
    k2_bias: float,
    switch_12: float,
    switch_21: float,
    short_k1_gamma: float,
    short_k2_gamma: float,
    short_k1_tau: float,
    short_k2_tau: float,
    short_len_cap: int,
    reset_gap: int,
    merge_short_k1_max_len: int,
    merge_short_k2_max_len: int,
    merge_short_k2_keep_cost: float,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    total_chunk_count = 0
    total_switch_count = 0
    total_short_k1_penalty_runs = 0
    total_short_k2_penalty_runs = 0
    total_merged_short_k1_runs = 0
    total_merged_short_k1_rows = 0
    total_merged_short_k2_runs = 0
    total_merged_short_k2_rows = 0
    bucket_cap = max(1, int(short_len_cap))
    state_stride = bucket_cap + 1
    num_states = 2 * state_stride
    max_gap = max(0, int(reset_gap))
    merge_short_k1_limit = max(0, int(merge_short_k1_max_len))
    merge_short_k2_limit = max(0, int(merge_short_k2_max_len))
    k2_to_k1_keep_cost = float(merge_short_k2_keep_cost)

    def duration_penalty(mode_idx: int, run_len_bucket: int) -> float:
        gamma = float(short_k1_gamma if mode_idx == 0 else short_k2_gamma)
        tau = max(float(short_k1_tau if mode_idx == 0 else short_k2_tau), 1e-06)
        run_len = float(run_len_bucket)
        return float(gamma * np.exp(-run_len / tau))

    def sid(mode_idx: int, run_len_bucket: int) -> int:
        return mode_idx * state_stride + (run_len_bucket - 1)

    def unpack(state_id: int) -> tuple[int, int]:
        return (state_id // state_stride, state_id % state_stride + 1)

    def decode_chunk(
        keys_chunk: list[tuple[int, str]],
        frames_chunk: list[int],
        instability_chunk: list[float],
    ) -> tuple[list[bool], int, int, int]:
        if not keys_chunk:
            return ([], 0, 0, 0)
        inst_scale = (
            float(np.percentile(np.asarray(instability_chunk, dtype=np.float64), 90.0))
            if len(instability_chunk) > 0
            else 1.0
        )
        inst_scale = max(inst_scale, 1e-06)
        unary: list[tuple[float, float]] = []
        for idx, key in enumerate(keys_chunk):
            threshold = float(threshold_edge if key in edge_keys else threshold_default)
            err = float(k1_metrics_lookup[key]["weighted_error"])
            err_margin = (err - threshold) / max(threshold, 1.0)
            inst_norm = float(instability_chunk[idx]) / inst_scale
            evidence = (
                float(error_weight) * err_margin + float(instability_weight) * inst_norm
            )
            if key in edge_keys:
                evidence += float(edge_bonus)
            unary.append((0.0, float(k2_bias) - evidence))
        back_ptr: list[list[int]] = []
        prev_cost = [float("inf")] * num_states
        for mode_idx in (0, 1):
            state_id = sid(mode_idx, 1)
            prev_cost[state_id] = unary[0][mode_idx]
        back_ptr.append([-1] * num_states)
        for t in range(1, len(keys_chunk)):
            curr_cost = [float("inf")] * num_states
            curr_back = [-1] * num_states
            for prev_state in range(num_states):
                base_cost = prev_cost[prev_state]
                if not np.isfinite(base_cost):
                    continue
                prev_mode, prev_len_bucket = unpack(prev_state)
                for mode_idx in (0, 1):
                    if mode_idx == prev_mode:
                        next_len_bucket = min(prev_len_bucket + 1, bucket_cap + 1)
                        candidate = base_cost + unary[t][mode_idx]
                    else:
                        transition_cost = float(
                            switch_12 if prev_mode == 0 else switch_21
                        )
                        candidate = (
                            base_cost
                            + transition_cost
                            + duration_penalty(prev_mode, prev_len_bucket)
                            + unary[t][mode_idx]
                        )
                        next_len_bucket = 1
                    next_state = sid(mode_idx, next_len_bucket)
                    if candidate < curr_cost[next_state]:
                        curr_cost[next_state] = candidate
                        curr_back[next_state] = prev_state
            prev_cost = curr_cost
            back_ptr.append(curr_back)
        best_state = -1
        best_cost = float("inf")
        for state_id, base_cost in enumerate(prev_cost):
            if not np.isfinite(base_cost):
                continue
            mode_idx, run_len_bucket = unpack(state_id)
            candidate = base_cost + duration_penalty(mode_idx, run_len_bucket)
            if candidate < best_cost:
                best_cost = candidate
                best_state = state_id
        state_path = [best_state]
        for t in range(len(keys_chunk) - 1, 0, -1):
            prev_state = back_ptr[t][state_path[-1]]
            state_path.append(prev_state)
        state_path.reverse()
        flags = [unpack(state_id)[0] == 1 for state_id in state_path]
        switch_count = 0
        short_k1_runs = 0
        short_k2_runs = 0
        run_mode = flags[0]
        run_len = 1
        for flag in flags[1:]:
            if flag == run_mode:
                run_len += 1
                continue
            switch_count += 1
            if run_len <= bucket_cap:
                if run_mode:
                    short_k2_runs += 1
                else:
                    short_k1_runs += 1
            run_mode = flag
            run_len = 1
        if run_len <= bucket_cap:
            if run_mode:
                short_k2_runs += 1
            else:
                short_k1_runs += 1
        return (flags, switch_count, short_k1_runs, short_k2_runs)

    def merge_short_islands(
        flags: list[bool], keys_local: list[tuple[int, str]]
    ) -> tuple[list[bool], int, int, int, int]:
        if len(flags) < 3 or (merge_short_k1_limit <= 0 and merge_short_k2_limit <= 0):
            return (list(flags), 0, 0, 0, 0)
        out = list(flags)
        merged_short_k1_runs = 0
        merged_short_k1_rows = 0
        merged_short_k2_runs = 0
        merged_short_k2_rows = 0
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(out):
                j = i + 1
                while j < len(out) and out[j] == out[i]:
                    j += 1
                run_len = j - i
                left_ok = i > 0
                right_ok = j < len(out)
                if (
                    left_ok
                    and right_ok
                    and (out[i - 1] == out[j])
                    and (out[i - 1] != out[i])
                ):
                    if not out[i] and run_len <= merge_short_k1_limit:
                        for k in range(i, j):
                            out[k] = True
                        merged_short_k1_runs += 1
                        merged_short_k1_rows += run_len
                        changed = True
                    elif out[i] and run_len <= merge_short_k2_limit:
                        max_k1_cost = max(
                            (
                                float(k1_metrics_lookup[key]["weighted_error"])
                                for key in keys_local[i:j]
                            )
                        )
                        if k2_to_k1_keep_cost < 0 or max_k1_cost <= k2_to_k1_keep_cost:
                            for k in range(i, j):
                                out[k] = False
                            merged_short_k2_runs += 1
                            merged_short_k2_rows += run_len
                            changed = True
                i = j
        return (
            out,
            merged_short_k1_runs,
            merged_short_k1_rows,
            merged_short_k2_runs,
            merged_short_k2_rows,
        )

    def merge_exact_inner_islands(
        flags: list[bool], frames_local: list[int], keys_local: list[tuple[int, str]]
    ) -> tuple[list[bool], int, int, int, int]:
        if len(flags) < 3 or (merge_short_k1_limit <= 0 and merge_short_k2_limit <= 0):
            return (list(flags), 0, 0, 0, 0)
        out = list(flags)
        merged_short_k1_runs = 0
        merged_short_k1_rows = 0
        merged_short_k2_runs = 0
        merged_short_k2_rows = 0
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(out):
                j = i + 1
                while (
                    j < len(out)
                    and out[j] == out[i]
                    and (frames_local[j] - frames_local[j - 1] == 1)
                ):
                    j += 1
                run_len = j - i
                left_ok = i > 0 and frames_local[i] - frames_local[i - 1] == 1
                right_ok = j < len(out) and frames_local[j] - frames_local[j - 1] == 1
                if (
                    left_ok
                    and right_ok
                    and (out[i - 1] == out[j])
                    and (out[i - 1] != out[i])
                ):
                    if not out[i] and run_len <= merge_short_k1_limit:
                        for k in range(i, j):
                            out[k] = True
                        merged_short_k1_runs += 1
                        merged_short_k1_rows += run_len
                        changed = True
                    elif out[i] and run_len <= merge_short_k2_limit:
                        max_k1_cost = max(
                            (
                                float(k1_metrics_lookup[key]["weighted_error"])
                                for key in keys_local[i:j]
                            )
                        )
                        if k2_to_k1_keep_cost < 0 or max_k1_cost <= k2_to_k1_keep_cost:
                            for k in range(i, j):
                                out[k] = False
                            merged_short_k2_runs += 1
                            merged_short_k2_rows += run_len
                            changed = True
                i = j
        return (
            out,
            merged_short_k1_runs,
            merged_short_k1_rows,
            merged_short_k2_runs,
            merged_short_k2_rows,
        )

    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        instability_scores = infer_build_track_local_instability_scores(
            track_rows, k1_ellipses_lookup
        )
        final_flags = [False] * len(track_rows)
        chunk_start = 0
        track_switch_count = 0
        track_short_k1_runs = 0
        track_short_k2_runs = 0
        track_merged_short_k1_runs = 0
        track_merged_short_k1_rows = 0
        track_merged_short_k2_runs = 0
        track_merged_short_k2_rows = 0
        chunk_count = 0
        for idx in range(1, len(track_rows) + 1):
            at_end = idx == len(track_rows)
            has_gap = not at_end and frames[idx] - frames[idx - 1] > max_gap
            if not at_end and (not has_gap):
                continue
            keys_chunk = keys[chunk_start:idx]
            (
                flags_chunk,
                switch_count_chunk,
                short_k1_runs_chunk,
                short_k2_runs_chunk,
            ) = decode_chunk(
                keys_chunk=keys_chunk,
                frames_chunk=frames[chunk_start:idx],
                instability_chunk=instability_scores[chunk_start:idx],
            )
            (
                flags_chunk,
                merged_short_k1_runs_chunk,
                merged_short_k1_rows_chunk,
                merged_short_k2_runs_chunk,
                merged_short_k2_rows_chunk,
            ) = merge_short_islands(flags_chunk, keys_chunk)
            final_flags[chunk_start:idx] = flags_chunk
            track_switch_count += switch_count_chunk
            track_short_k1_runs += short_k1_runs_chunk
            track_short_k2_runs += short_k2_runs_chunk
            track_merged_short_k1_runs += merged_short_k1_runs_chunk
            track_merged_short_k1_rows += merged_short_k1_rows_chunk
            track_merged_short_k2_runs += merged_short_k2_runs_chunk
            track_merged_short_k2_rows += merged_short_k2_rows_chunk
            chunk_count += 1
            chunk_start = idx
        (
            final_flags,
            merged_short_k1_runs_track_exact,
            merged_short_k1_rows_track_exact,
            merged_short_k2_runs_track_exact,
            merged_short_k2_rows_track_exact,
        ) = merge_exact_inner_islands(final_flags, frames, keys)
        track_merged_short_k1_runs += merged_short_k1_runs_track_exact
        track_merged_short_k1_rows += merged_short_k1_rows_track_exact
        track_merged_short_k2_runs += merged_short_k2_runs_track_exact
        track_merged_short_k2_rows += merged_short_k2_rows_track_exact
        selected_in_track = 0
        for key, flag in zip(keys, final_flags):
            if flag:
                selected.add(key)
                selected_in_track += 1
        total_chunk_count += chunk_count
        total_switch_count += track_switch_count
        total_short_k1_penalty_runs += track_short_k1_runs
        total_short_k2_penalty_runs += track_short_k2_runs
        total_merged_short_k1_runs += track_merged_short_k1_runs
        total_merged_short_k1_rows += track_merged_short_k1_rows
        total_merged_short_k2_runs += track_merged_short_k2_runs
        total_merged_short_k2_rows += track_merged_short_k2_rows
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "chunk_count": int(chunk_count),
                "seed_count": int(
                    sum(
                        (
                            1
                            for key in keys
                            if float(k1_metrics_lookup[key]["weighted_error"])
                            >= float(
                                threshold_edge
                                if key in edge_keys
                                else threshold_default
                            )
                        )
                    )
                ),
                "expanded_count": int(selected_in_track),
                "switch_count": int(track_switch_count),
                "short_k1_runs": int(track_short_k1_runs),
                "short_k2_runs": int(track_short_k2_runs),
                "merged_short_k1_runs": int(track_merged_short_k1_runs),
                "merged_short_k1_rows": int(track_merged_short_k1_rows),
                "merged_short_k2_runs": int(track_merged_short_k2_runs),
                "merged_short_k2_rows": int(track_merged_short_k2_rows),
                "error_cut": None,
                "instability_cut": None,
            }
        )
    summary = {
        "routing_mode": "track_dp",
        "threshold": float(threshold_default),
        "threshold_edge": float(threshold_edge),
        "selected_count": len(selected),
        "error_weight": float(error_weight),
        "instability_weight": float(instability_weight),
        "edge_bonus": float(edge_bonus),
        "k2_bias": float(k2_bias),
        "switch_12": float(switch_12),
        "switch_21": float(switch_21),
        "short_k1_gamma": float(short_k1_gamma),
        "short_k2_gamma": float(short_k2_gamma),
        "short_k1_tau": float(short_k1_tau),
        "short_k2_tau": float(short_k2_tau),
        "short_len_cap": int(bucket_cap),
        "reset_gap": int(max_gap),
        "merge_short_k1_max_len": int(merge_short_k1_limit),
        "merge_short_k2_max_len": int(merge_short_k2_limit),
        "merge_short_k2_keep_cost": float(k2_to_k1_keep_cost),
        "chunk_count": int(total_chunk_count),
        "switch_count": int(total_switch_count),
        "short_k1_penalty_runs": int(total_short_k1_penalty_runs),
        "short_k2_penalty_runs": int(total_short_k2_penalty_runs),
        "merged_short_k1_runs": int(total_merged_short_k1_runs),
        "merged_short_k1_rows": int(total_merged_short_k1_rows),
        "merged_short_k2_runs": int(total_merged_short_k2_runs),
        "merged_short_k2_rows": int(total_merged_short_k2_rows),
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_build_k2_threshold_only_selection(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    threshold_default: float,
    threshold_edge: float,
    edge_keys: set[tuple[int, str]],
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        selected_in_track = 0
        for key in keys:
            threshold = threshold_edge if key in edge_keys else threshold_default
            err = float(k1_metrics_lookup[key]["weighted_error"])
            if err >= float(threshold):
                selected.add(key)
                selected_in_track += 1
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "seed_count": int(selected_in_track),
                "expanded_count": int(selected_in_track),
                "error_cut": None,
                "instability_cut": None,
            }
        )
    summary = {
        "routing_mode": "threshold_only",
        "threshold": float(threshold_default),
        "threshold_edge": float(threshold_edge),
        "selected_count": len(selected),
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_build_k2_threshold_soft_selection(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    threshold_default: float,
    threshold_edge: float,
    edge_keys: set[tuple[int, str]],
    ema_alpha: float,
    band_ratio: float,
    exit_ratio: float,
    strong_ratio: float,
    k1_keep_cost: float,
    reset_gap: int,
    merge_islands_max_len: int,
    merge_policy: str,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    total_soft_hold = 0
    total_soft_flip = 0
    total_merged_islands = 0
    total_merged_rows = 0
    alpha = float(np.clip(ema_alpha, 0.0, 1.0))
    band = max(0.0, float(band_ratio))
    exit_band = band if float(exit_ratio) < 0.0 else max(0.0, float(exit_ratio))
    strong = max(band, float(strong_ratio))
    k1_keep_cost_threshold = float(k1_keep_cost)
    max_gap = max(0, int(reset_gap))
    merge_limit = max(0, int(merge_islands_max_len))
    merge_policy_value = str(merge_policy)

    def merge_small_islands(
        flags: list[bool], frames_local: list[int]
    ) -> tuple[list[bool], int, int]:
        if merge_limit <= 0 or len(flags) < 3:
            return (flags, 0, 0)
        out = list(flags)
        merged_islands = 0
        merged_rows = 0
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(out):
                j = i + 1
                while (
                    j < len(out)
                    and out[j] == out[i]
                    and (frames_local[j] - frames_local[j - 1] <= max_gap)
                ):
                    j += 1
                run_len = j - i
                left_ok = i > 0 and frames_local[i] - frames_local[i - 1] <= max_gap
                right_ok = (
                    j < len(out) and frames_local[j] - frames_local[j - 1] <= max_gap
                )
                if (
                    run_len <= merge_limit
                    and left_ok
                    and right_ok
                    and (out[i - 1] == out[j])
                ):
                    replacement = out[i - 1]
                    if merge_policy_value == "prefer_k2" and replacement is False:
                        i = j
                        continue
                    if replacement != out[i]:
                        for k in range(i, j):
                            out[k] = replacement
                        merged_islands += 1
                        merged_rows += run_len
                        changed = True
                i = j
        return (out, merged_islands, merged_rows)

    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        raw_scores: list[float] = []
        raw_selected: list[bool] = []
        for key in keys:
            threshold = threshold_edge if key in edge_keys else threshold_default
            err = float(k1_metrics_lookup[key]["weighted_error"])
            denom = max(float(threshold), 1.0)
            score = (err - float(threshold)) / denom
            raw_scores.append(score)
            raw_selected.append(score >= 0.0)
        ema_score: float | None = None
        prev_selected: bool | None = None
        prev_frame: int | None = None
        selected_in_track = 0
        soft_hold_in_track = 0
        soft_flip_in_track = 0
        final_flags: list[bool] = []
        for idx, key in enumerate(keys):
            frame = frames[idx]
            raw_score = raw_scores[idx]
            raw_pick = raw_selected[idx]
            if prev_frame is None or frame - prev_frame > max_gap:
                ema_score = raw_score
                prev_selected = raw_pick
            else:
                assert ema_score is not None
                ema_score = alpha * raw_score + (1.0 - alpha) * ema_score
            if raw_score >= strong:
                final_pick = True
            elif raw_score <= -strong:
                final_pick = False
            elif prev_selected:
                if ema_score <= -exit_band:
                    final_pick = False
                elif ema_score >= band:
                    final_pick = True
                else:
                    final_pick = True
                if not final_pick and k1_keep_cost_threshold >= 0:
                    err_now = float(k1_metrics_lookup[key]["weighted_error"])
                    if err_now > k1_keep_cost_threshold:
                        final_pick = True
                        if raw_pick != final_pick:
                            soft_hold_in_track += 1
            elif ema_score >= band:
                final_pick = True
            elif ema_score <= -band:
                final_pick = False
            else:
                final_pick = raw_pick if prev_selected is None else prev_selected
                if prev_selected is not None and raw_pick != final_pick:
                    if final_pick == prev_selected:
                        soft_hold_in_track += 1
                    else:
                        soft_flip_in_track += 1
            final_flags.append(final_pick)
            prev_selected = final_pick
            prev_frame = frame
        (
            final_flags,
            merged_islands_in_track,
            merged_rows_in_track,
        ) = merge_small_islands(final_flags, frames)
        for key, final_pick in zip(keys, final_flags):
            if final_pick:
                selected.add(key)
                selected_in_track += 1
        total_soft_hold += soft_hold_in_track
        total_soft_flip += soft_flip_in_track
        total_merged_islands += merged_islands_in_track
        total_merged_rows += merged_rows_in_track
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "seed_count": int(sum((1 for flag in raw_selected if flag))),
                "expanded_count": int(selected_in_track),
                "soft_hold_count": int(soft_hold_in_track),
                "soft_flip_count": int(soft_flip_in_track),
                "merged_island_count": int(merged_islands_in_track),
                "merged_island_rows": int(merged_rows_in_track),
                "error_cut": None,
                "instability_cut": None,
            }
        )
    summary = {
        "routing_mode": "threshold_soft",
        "threshold": float(threshold_default),
        "threshold_edge": float(threshold_edge),
        "selected_count": len(selected),
        "ema_alpha": float(alpha),
        "band_ratio": float(band),
        "exit_ratio": float(exit_band),
        "strong_ratio": float(strong),
        "k1_keep_cost": float(k1_keep_cost_threshold),
        "reset_gap": int(max_gap),
        "merge_policy": merge_policy_value,
        "soft_hold_count": int(total_soft_hold),
        "soft_flip_count": int(total_soft_flip),
        "merged_island_count": int(total_merged_islands),
        "merged_island_rows": int(total_merged_rows),
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_build_k2_threshold_hysteresis_selection(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    enter_default: float,
    enter_edge: float,
    exit_default: float,
    exit_edge: float,
    edge_keys: set[tuple[int, str]],
    confirm_frames: int,
    reset_gap: int,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    effective_confirm = max(1, int(confirm_frames))
    max_gap = max(0, int(reset_gap))
    total_pending_promotions = 0
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        flags = [False] * len(track_rows)
        in_k2 = False
        pending_enter: list[int] = []
        prev_frame: int | None = None
        promoted_in_track = 0
        for idx, key in enumerate(keys):
            frame = frames[idx]
            if prev_frame is not None and frame - prev_frame > max_gap:
                in_k2 = False
                pending_enter = []
            err = float(k1_metrics_lookup[key]["weighted_error"])
            enter_threshold = float(enter_edge if key in edge_keys else enter_default)
            exit_threshold = float(exit_edge if key in edge_keys else exit_default)
            if in_k2:
                if err <= exit_threshold:
                    in_k2 = False
                    pending_enter = []
                    flags[idx] = False
                else:
                    flags[idx] = True
            elif err >= enter_threshold:
                pending_enter.append(idx)
                if len(pending_enter) >= effective_confirm:
                    for pending_idx in pending_enter:
                        if not flags[pending_idx]:
                            flags[pending_idx] = True
                            promoted_in_track += 1
                    in_k2 = True
                    pending_enter = []
            else:
                pending_enter = []
            prev_frame = frame
        for key, flag in zip(keys, flags):
            if flag:
                selected.add(key)
        total_pending_promotions += promoted_in_track
        summary_tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(track_rows),
                "seed_count": int(
                    sum(
                        (
                            1
                            for key in keys
                            if float(k1_metrics_lookup[key]["weighted_error"])
                            >= float(enter_edge if key in edge_keys else enter_default)
                        )
                    )
                ),
                "expanded_count": int(sum((1 for flag in flags if flag))),
                "promoted_by_confirm": int(promoted_in_track),
                "error_cut": None,
                "instability_cut": None,
            }
        )
    summary = {
        "routing_mode": "threshold_hysteresis",
        "enter_threshold": float(enter_default),
        "enter_threshold_edge": float(enter_edge),
        "exit_threshold": float(exit_default),
        "exit_threshold_edge": float(exit_edge),
        "confirm_frames": int(effective_confirm),
        "reset_gap": int(max_gap),
        "selected_count": len(selected),
        "promoted_by_confirm": int(total_pending_promotions),
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_count_k2_switch_stats(
    flags: list[bool], frames: list[int], *, reset_gap: int
) -> dict[str, int]:
    if not flags:
        return {
            "switch_count": 0,
            "k1_run_count": 0,
            "k2_run_count": 0,
            "k1_single_frame_islands": 0,
            "k1_two_frame_islands": 0,
            "k2_single_frame_islands": 0,
            "k2_two_frame_islands": 0,
        }
    max_gap = max(0, int(reset_gap))
    runs: list[tuple[bool, int, int]] = []
    start = 0
    for idx in range(1, len(flags) + 1):
        at_end = idx == len(flags)
        has_gap = (not at_end) and (int(frames[idx]) - int(frames[idx - 1]) > max_gap)
        changed = (not at_end) and flags[idx] != flags[idx - 1]
        if at_end or has_gap or changed:
            runs.append((bool(flags[start]), start, idx))
            start = idx
    switch_count = 0
    for left, right in zip(runs, runs[1:], strict=False):
        if (
            int(frames[right[1]]) - int(frames[left[2] - 1]) <= max_gap
            and left[0] != right[0]
        ):
            switch_count += 1
    stats = {
        "switch_count": int(switch_count),
        "k1_run_count": int(sum(1 for mode, _start, _end in runs if not mode)),
        "k2_run_count": int(sum(1 for mode, _start, _end in runs if mode)),
        "k1_single_frame_islands": 0,
        "k1_two_frame_islands": 0,
        "k2_single_frame_islands": 0,
        "k2_two_frame_islands": 0,
    }
    for run_idx in range(1, len(runs) - 1):
        mode, start_idx, end_idx = runs[run_idx]
        left_mode, _left_start, left_end = runs[run_idx - 1]
        right_mode, right_start, _right_end = runs[run_idx + 1]
        if left_mode != right_mode or left_mode == mode:
            continue
        if int(frames[start_idx]) - int(frames[left_end - 1]) > max_gap:
            continue
        if int(frames[right_start]) - int(frames[end_idx - 1]) > max_gap:
            continue
        run_len = end_idx - start_idx
        if run_len == 1:
            stats["k2_single_frame_islands" if mode else "k1_single_frame_islands"] += 1
        elif run_len == 2:
            stats["k2_two_frame_islands" if mode else "k1_two_frame_islands"] += 1
    return stats


def infer_build_k2_k1n_sequence_selection(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    enter_threshold: float,
    exit_threshold: float,
    strong_enter_threshold: float,
    strong_exit_threshold: float,
    protect_k2_iou_below: float,
    smooth_window: int,
    enter_confirm_frames: int,
    exit_confirm_frames: int,
    merge_short_k1_max_len: int,
    merge_short_k2_max_len: int,
    reset_gap: int,
) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    enter = float(enter_threshold)
    exit_value = min(float(exit_threshold), enter)
    protect_iou = float(protect_k2_iou_below)
    iou_equivalent_enter = (
        (1.0 / protect_iou - 1.0) if 0.0 < protect_iou < 1.0 else enter * 1.5
    )
    strong_enter = float(
        strong_enter_threshold
        if float(strong_enter_threshold) >= 0.0
        else max(enter * 1.5, iou_equivalent_enter)
    )
    strong_enter = max(strong_enter, enter)
    strong_exit = float(
        strong_exit_threshold
        if float(strong_exit_threshold) >= 0.0
        else exit_value * 0.65
    )
    strong_exit = min(strong_exit, exit_value)
    window = max(1, int(smooth_window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    enter_confirm = max(1, int(enter_confirm_frames))
    exit_confirm = max(1, int(exit_confirm_frames))
    max_gap = max(0, int(reset_gap))
    merge_k1_limit = max(0, int(merge_short_k1_max_len))
    merge_k2_limit = max(0, int(merge_short_k2_max_len))
    totals = {
        "seed_count": 0,
        "strong_seed_count": 0,
        "pre_cleanup_selected_count": 0,
        "selected_count": 0,
        "promoted_by_confirm": 0,
        "exited_by_confirm": 0,
        "merged_short_k1_runs": 0,
        "merged_short_k1_rows": 0,
        "removed_short_k2_runs": 0,
        "removed_short_k2_rows": 0,
        "protected_short_k1_runs": 0,
        "protected_short_k2_runs": 0,
        "protected_short_k2_iou_runs": 0,
        "protected_short_k2_iou_rows": 0,
        "protected_short_k2_cost_runs": 0,
        "protected_short_k2_cost_rows": 0,
        "switch_count": 0,
        "k1_single_frame_islands": 0,
        "k1_two_frame_islands": 0,
        "k2_single_frame_islands": 0,
        "k2_two_frame_islands": 0,
    }

    def smooth_costs(costs: list[float]) -> list[float]:
        if window <= 1 or len(costs) <= 2:
            return list(costs)
        out: list[float] = []
        for idx in range(len(costs)):
            left = max(0, idx - radius)
            right = min(len(costs), idx + radius + 1)
            out.append(
                float(np.median(np.asarray(costs[left:right], dtype=np.float64)))
            )
        return out

    def split_chunks(frames_local: list[int]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        start = 0
        for idx in range(1, len(frames_local) + 1):
            at_end = idx == len(frames_local)
            has_gap = (not at_end) and int(frames_local[idx]) - int(
                frames_local[idx - 1]
            ) > max_gap
            if at_end or has_gap:
                chunks.append((start, idx))
                start = idx
        return chunks

    def apply_hysteresis(
        costs: list[float], smooth: list[float], ious: list[float]
    ) -> tuple[list[bool], int, int]:
        flags = [False] * len(costs)
        in_k2 = False
        pending_enter: list[int] = []
        pending_exit: list[int] = []
        promoted = 0
        exited = 0
        for idx, (cost, score, iou) in enumerate(zip(costs, smooth, ious, strict=True)):
            strong_k2 = cost >= strong_enter or (
                protect_iou > 0.0 and iou < protect_iou
            )
            if in_k2:
                should_exit = cost <= strong_exit or score <= exit_value
                if should_exit:
                    pending_exit.append(idx)
                else:
                    pending_exit = []
                if cost <= strong_exit or len(pending_exit) >= exit_confirm:
                    for pending_idx in pending_exit:
                        flags[pending_idx] = False
                    exited += len(pending_exit)
                    in_k2 = False
                    pending_exit = []
                    pending_enter = []
                    flags[idx] = False
                else:
                    flags[idx] = True
            else:
                should_enter = strong_k2 or score >= enter
                if should_enter:
                    pending_enter.append(idx)
                else:
                    pending_enter = []
                if strong_k2 or len(pending_enter) >= enter_confirm:
                    for pending_idx in pending_enter:
                        flags[pending_idx] = True
                    promoted += len(pending_enter)
                    in_k2 = True
                    pending_enter = []
                    pending_exit = []
                    flags[idx] = True
                else:
                    flags[idx] = False
        return (flags, promoted, exited)

    def merge_protected_short_islands(
        flags: list[bool],
        frames_local: list[int],
        costs: list[float],
        ious: list[float],
    ) -> tuple[list[bool], dict[str, int]]:
        out = list(flags)
        local_stats = {
            "merged_short_k1_runs": 0,
            "merged_short_k1_rows": 0,
            "removed_short_k2_runs": 0,
            "removed_short_k2_rows": 0,
            "protected_short_k1_runs": 0,
            "protected_short_k2_runs": 0,
            "protected_short_k2_iou_runs": 0,
            "protected_short_k2_iou_rows": 0,
            "protected_short_k2_cost_runs": 0,
            "protected_short_k2_cost_rows": 0,
        }
        if len(out) < 3 or (merge_k1_limit <= 0 and merge_k2_limit <= 0):
            return (out, local_stats)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(out):
                j = i + 1
                while (
                    j < len(out)
                    and out[j] == out[i]
                    and int(frames_local[j]) - int(frames_local[j - 1]) <= max_gap
                ):
                    j += 1
                run_len = j - i
                left_ok = (
                    i > 0 and int(frames_local[i]) - int(frames_local[i - 1]) <= max_gap
                )
                right_ok = (
                    j < len(out)
                    and int(frames_local[j]) - int(frames_local[j - 1]) <= max_gap
                )
                if (
                    left_ok
                    and right_ok
                    and out[i - 1] == out[j]
                    and out[i - 1] != out[i]
                ):
                    if (
                        (not out[i])
                        and merge_k1_limit > 0
                        and run_len <= merge_k1_limit
                    ):
                        for k in range(i, j):
                            out[k] = True
                        local_stats["merged_short_k1_runs"] += 1
                        local_stats["merged_short_k1_rows"] += run_len
                        changed = True
                    elif out[i] and merge_k2_limit > 0 and run_len <= merge_k2_limit:
                        protect_by_cost = max(costs[i:j]) >= strong_enter
                        protect_by_iou = (
                            protect_iou > 0.0 and min(ious[i:j]) < protect_iou
                        )
                        if protect_by_cost or protect_by_iou:
                            local_stats["protected_short_k2_runs"] += 1
                            if protect_by_cost:
                                local_stats["protected_short_k2_cost_runs"] += 1
                                local_stats["protected_short_k2_cost_rows"] += run_len
                            if protect_by_iou:
                                local_stats["protected_short_k2_iou_runs"] += 1
                                local_stats["protected_short_k2_iou_rows"] += run_len
                        else:
                            for k in range(i, j):
                                out[k] = False
                            local_stats["removed_short_k2_runs"] += 1
                            local_stats["removed_short_k2_rows"] += run_len
                            changed = True
                i = j
        return (out, local_stats)

    for track_id, track_rows in rows_by_track.items():
        ordered_rows = sorted(track_rows, key=lambda row: int(row[0]))
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in ordered_rows]
        frames = [int(frame) for frame, _, _, _ in ordered_rows]
        costs = [float(k1_metrics_lookup[key]["weighted_error"]) for key in keys]
        ious = [float(k1_metrics_lookup[key].get("iou", 1.0)) for key in keys]
        final_flags = [False] * len(keys)
        track_promoted = 0
        track_exited = 0
        track_cleanup = {
            "merged_short_k1_runs": 0,
            "merged_short_k1_rows": 0,
            "removed_short_k2_runs": 0,
            "removed_short_k2_rows": 0,
            "protected_short_k1_runs": 0,
            "protected_short_k2_runs": 0,
            "protected_short_k2_iou_runs": 0,
            "protected_short_k2_iou_rows": 0,
            "protected_short_k2_cost_runs": 0,
            "protected_short_k2_cost_rows": 0,
        }
        for start, end in split_chunks(frames):
            chunk_costs = costs[start:end]
            chunk_smooth = smooth_costs(chunk_costs)
            chunk_ious = ious[start:end]
            chunk_flags, promoted, exited = apply_hysteresis(
                chunk_costs, chunk_smooth, chunk_ious
            )
            chunk_flags, cleanup_stats = merge_protected_short_islands(
                chunk_flags, frames[start:end], chunk_costs, chunk_ious
            )
            final_flags[start:end] = chunk_flags
            track_promoted += promoted
            track_exited += exited
            for key, value in cleanup_stats.items():
                track_cleanup[key] += int(value)
        selected_in_track = 0
        for key, flag in zip(keys, final_flags, strict=True):
            if flag:
                selected.add(key)
                selected_in_track += 1
        seed_count = int(sum(1 for cost in costs if cost >= enter))
        strong_seed_count = int(sum(1 for cost in costs if cost >= strong_enter))
        pre_cleanup_selected_count = (
            selected_in_track
            + track_cleanup["removed_short_k2_rows"]
            - track_cleanup["merged_short_k1_rows"]
        )
        switch_stats = infer_count_k2_switch_stats(
            final_flags, frames, reset_gap=max_gap
        )
        for key in (
            "seed_count",
            "strong_seed_count",
            "pre_cleanup_selected_count",
            "selected_count",
            "promoted_by_confirm",
            "exited_by_confirm",
        ):
            if key == "seed_count":
                totals[key] += seed_count
            elif key == "strong_seed_count":
                totals[key] += strong_seed_count
            elif key == "pre_cleanup_selected_count":
                totals[key] += pre_cleanup_selected_count
            elif key == "selected_count":
                totals[key] += selected_in_track
            elif key == "promoted_by_confirm":
                totals[key] += track_promoted
            elif key == "exited_by_confirm":
                totals[key] += track_exited
        for key, value in track_cleanup.items():
            totals[key] += int(value)
        for key in (
            "switch_count",
            "k1_single_frame_islands",
            "k1_two_frame_islands",
            "k2_single_frame_islands",
            "k2_two_frame_islands",
        ):
            totals[key] += int(switch_stats[key])
        summary_tracks.append(
            {
                "track_id": str(track_id),
                "frame_count": len(keys),
                "seed_count": int(seed_count),
                "strong_seed_count": int(strong_seed_count),
                "pre_cleanup_selected_count": int(pre_cleanup_selected_count),
                "expanded_count": int(selected_in_track),
                "switch_count": int(switch_stats["switch_count"]),
                "k1_run_count": int(switch_stats["k1_run_count"]),
                "k2_run_count": int(switch_stats["k2_run_count"]),
                "k1_single_frame_islands": int(switch_stats["k1_single_frame_islands"]),
                "k1_two_frame_islands": int(switch_stats["k1_two_frame_islands"]),
                "k2_single_frame_islands": int(switch_stats["k2_single_frame_islands"]),
                "k2_two_frame_islands": int(switch_stats["k2_two_frame_islands"]),
                "promoted_by_confirm": int(track_promoted),
                "exited_by_confirm": int(track_exited),
                **{key: int(value) for key, value in track_cleanup.items()},
                "cost_min": float(min(costs)) if costs else None,
                "cost_p50": float(
                    np.percentile(np.asarray(costs, dtype=np.float64), 50.0)
                )
                if costs
                else None,
                "cost_p90": float(
                    np.percentile(np.asarray(costs, dtype=np.float64), 90.0)
                )
                if costs
                else None,
                "cost_max": float(max(costs)) if costs else None,
                "iou_min": float(min(ious)) if ious else None,
                "error_cut": None,
                "instability_cut": None,
            }
        )
    summary = {
        "routing_mode": "k1n_sequence",
        "cost_feature": "k1_cost_norm_sequence",
        "enter_threshold": float(enter),
        "exit_threshold": float(exit_value),
        "strong_enter_threshold": float(strong_enter),
        "strong_exit_threshold": float(strong_exit),
        "protect_k2_iou_below": float(protect_iou),
        "smooth_window": int(window),
        "enter_confirm_frames": int(enter_confirm),
        "exit_confirm_frames": int(exit_confirm),
        "merge_short_k1_max_len": int(merge_k1_limit),
        "merge_short_k2_max_len": int(merge_k2_limit),
        "reset_gap": int(max_gap),
        **{key: int(value) for key, value in totals.items()},
        "tracks": summary_tracks,
    }
    return (selected, summary)


def infer_cleanup_selected_k2_inner_islands(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    selected_keys: set[tuple[int, str]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    max_len: int,
    keep_cost: float,
) -> tuple[set[tuple[int, str]], dict[str, int]]:
    limit = max(0, int(max_len))
    if limit <= 0:
        return (
            set(selected_keys),
            {
                "removed_runs": 0,
                "removed_rows": 0,
                "removed_exact_runs": 0,
                "removed_exact_rows": 0,
                "removed_track_order_runs": 0,
                "removed_track_order_rows": 0,
                "removed_exact_singleton_runs": 0,
                "removed_exact_singleton_rows": 0,
            },
        )
    cleaned = set(selected_keys)
    removed_exact_runs = 0
    removed_exact_rows = 0
    removed_track_order_runs = 0
    removed_track_order_rows = 0
    removed_exact_singleton_runs = 0
    removed_exact_singleton_rows = 0
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        flags = [key in cleaned for key in keys]
        i = 0
        while i < len(flags):
            if not flags[i]:
                i += 1
                continue
            j = i + 1
            while j < len(flags) and flags[j] and (frames[j] - frames[j - 1] == 1):
                j += 1
            run_len = j - i
            left_ok = i > 0 and (not flags[i - 1]) and (frames[i] - frames[i - 1] == 1)
            right_ok = (
                j < len(flags) and (not flags[j]) and (frames[j] - frames[j - 1] == 1)
            )
            if left_ok and right_ok and (run_len <= limit):
                max_k1_cost = max(
                    (
                        float(k1_metrics_lookup[key]["weighted_error"])
                        for key in keys[i:j]
                    )
                )
                if keep_cost < 0 or max_k1_cost <= float(keep_cost):
                    for key in keys[i:j]:
                        cleaned.discard(key)
                    removed_exact_runs += 1
                    removed_exact_rows += run_len
            i = j
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        flags = [key in cleaned for key in keys]
        i = 0
        while i < len(flags):
            if not flags[i]:
                i += 1
                continue
            j = i + 1
            while j < len(flags) and flags[j]:
                j += 1
            run_len = j - i
            left_ok = i > 0 and (not flags[i - 1])
            right_ok = j < len(flags) and (not flags[j])
            if left_ok and right_ok and (run_len <= limit):
                max_k1_cost = max(
                    (
                        float(k1_metrics_lookup[key]["weighted_error"])
                        for key in keys[i:j]
                    )
                )
                if keep_cost < 0 or max_k1_cost <= float(keep_cost):
                    for key in keys[i:j]:
                        cleaned.discard(key)
                    removed_track_order_runs += 1
                    removed_track_order_rows += run_len
            i = j
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        flags = [key in cleaned for key in keys]
        i = 0
        while i < len(flags):
            if not flags[i]:
                i += 1
                continue
            j = i + 1
            while j < len(flags) and flags[j] and (frames[j] - frames[j - 1] == 1):
                j += 1
            run_len = j - i
            if run_len == 1:
                key = keys[i]
                max_k1_cost = float(k1_metrics_lookup[key]["weighted_error"])
                if keep_cost < 0 or max_k1_cost <= float(keep_cost):
                    cleaned.discard(key)
                    removed_exact_singleton_runs += 1
                    removed_exact_singleton_rows += 1
            i = j
    removed_runs = (
        removed_exact_runs + removed_track_order_runs + removed_exact_singleton_runs
    )
    removed_rows = (
        removed_exact_rows + removed_track_order_rows + removed_exact_singleton_rows
    )
    return (
        cleaned,
        {
            "removed_runs": int(removed_runs),
            "removed_rows": int(removed_rows),
            "removed_exact_runs": int(removed_exact_runs),
            "removed_exact_rows": int(removed_exact_rows),
            "removed_track_order_runs": int(removed_track_order_runs),
            "removed_track_order_rows": int(removed_track_order_rows),
            "removed_exact_singleton_runs": int(removed_exact_singleton_runs),
            "removed_exact_singleton_rows": int(removed_exact_singleton_rows),
        },
    )


def infer_promote_short_k1_runs_to_k2(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    selected_keys: set[tuple[int, str]],
    *,
    max_len: int,
) -> tuple[set[tuple[int, str]], dict[str, int]]:
    limit = max(0, int(max_len))
    if limit <= 0:
        return (set(selected_keys), {"promoted_runs": 0, "promoted_rows": 0})
    promoted = set(selected_keys)
    promoted_runs = 0
    promoted_rows = 0
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        flags = [key in promoted for key in keys]
        i = 0
        while i < len(flags):
            if flags[i]:
                i += 1
                continue
            j = i + 1
            while j < len(flags) and (not flags[j]):
                j += 1
            run_len = j - i
            left_k2 = i > 0 and flags[i - 1]
            right_k2 = j < len(flags) and flags[j]
            if run_len <= limit and (left_k2 or right_k2):
                for key in keys[i:j]:
                    promoted.add(key)
                promoted_runs += 1
                promoted_rows += run_len
            i = j
    return (
        promoted,
        {"promoted_runs": int(promoted_runs), "promoted_rows": int(promoted_rows)},
    )


def infer_force_select_high_cost_rows(
    rows_by_track: dict[str, list[tuple[int, str, str, int]]],
    selected_keys: set[tuple[int, str]],
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]],
    *,
    min_cost: float,
) -> tuple[set[tuple[int, str]], dict[str, int]]:
    threshold = float(min_cost)
    if threshold < 0:
        return (set(selected_keys), {"forced_runs": 0, "forced_rows": 0})
    forced = set(selected_keys)
    forced_runs = 0
    forced_rows = 0
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        i = 0
        while i < len(keys):
            key = keys[i]
            is_high_cost = float(k1_metrics_lookup[key]["weighted_error"]) > threshold
            if key in forced or not is_high_cost:
                i += 1
                continue
            j = i + 1
            while j < len(keys):
                key_j = keys[j]
                if (
                    key_j in forced
                    or float(k1_metrics_lookup[key_j]["weighted_error"]) <= threshold
                ):
                    break
                j += 1
            for key_run in keys[i:j]:
                forced.add(key_run)
            forced_runs += 1
            forced_rows += j - i
            i = j
    return (forced, {"forced_runs": int(forced_runs), "forced_rows": int(forced_rows)})


def infer_build_k2_input_from_payload(
    payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]],
    *,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    k2v5 = infer_get_k2v5_module()
    gt_mask, _origin = fst.rasterize_local_mask_from_payload(payload)
    gt_square, _pad, _side = k2v5.square_pad_mask(gt_mask)
    padded_mask = gt_square.astype(np.uint8, copy=False)
    signed = k2v5.build_signed_distance_channel(padded_mask)
    edge = k2v5.build_edge_channel(padded_mask)
    mask_resized = cv2.resize(
        padded_mask.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    signed_resized = cv2.resize(
        signed, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    edge_resized = cv2.resize(
        edge, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    touch_flags = k2v5.edge_touch_vector_from_row({}, gt_mask=padded_mask)
    input_image = k2v5.build_input_image(
        mask_resized, signed_resized, edge_resized, touch_flags, image_size
    )
    return (input_image.astype(np.float32, copy=False), padded_mask)


def infer_k2_v5_forward_states_only(
    model: object,
    image_tensor: "torch.Tensor",
    *,
    torch_module: object,
    k2v5_module: object,
) -> "torch.Tensor":
    torch = torch_module
    inner = model
    s1 = inner.stem(image_tensor)
    s2 = inner.stage2(s1)
    s3 = inner.stage3(s2)
    s4 = inner.stage4(s3)
    s5 = inner.stage5(s4)
    p5 = inner.lat5(s5)
    p4 = inner.fpn4(
        inner.lat4(s4)
        + torch.nn.functional.interpolate(
            p5, size=s4.shape[-2:], mode="bilinear", align_corners=False
        )
    )
    p3 = inner.fpn3(
        inner.lat3(s3)
        + torch.nn.functional.interpolate(
            p4, size=s3.shape[-2:], mode="bilinear", align_corners=False
        )
    )
    context_map = inner.context_proj(p4)
    context_tokens = context_map.flatten(2).transpose(1, 2)
    queries = inner.slot_queries.unsqueeze(0).expand(image_tensor.shape[0], -1, -1)
    global_feat = inner.global_pool(
        torch.nn.functional.interpolate(
            p3, size=context_map.shape[-2:], mode="bilinear", align_corners=False
        )
    )
    global_feat = global_feat.flatten(1).unsqueeze(1).expand(-1, 2, -1)
    queries = queries + inner.slot_refine(torch.cat([queries, global_feat], dim=-1))
    for block in inner.decoder:
        queries = block(queries, context_tokens)
    centers = inner.center_head(queries)
    chol_params = inner.chol_head(queries)
    states = k2v5_module.spd_to_normalized_states(centers, chol_params)
    return states.flatten(1)


def infer_infer_k2_v5(
    selected_rows: list[tuple[int, int, str, str]],
    payloads: list[tuple[tuple[int, int], tuple[int, int], list]],
    gt_polys: list[list[np.ndarray]],
    *,
    run_dir: Path,
    device_name: str,
    batch_size: int,
    prep_workers: int = 0,
    precision: str = "fp32",
    forward_mode: str = "states_only",
    profile_stages: bool = False,
    cudnn_benchmark: str = "off",
    tf32: str = "off",
) -> tuple[
    list[tuple[int, tuple[int, str, str]]],
    list[tuple[int, dict[str, object]]],
    float,
    dict[str, object],
]:
    if not selected_rows:
        return ([], [], 0.0, {})
    torch = infer_get_torch_module()
    k2v5 = infer_get_k2v5_module()
    checkpoint_path = run_dir / "best_exact.pt"
    device = torch.device(
        device_name
        if device_name != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = str(cudnn_benchmark) == "on"
        if str(tf32) == "on":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        elif str(tf32) == "off":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
    requested_precision = str(precision)
    effective_precision = (
        "fp16" if requested_precision == "fp16" and use_cuda else "fp32"
    )
    effective_forward_mode = str(forward_mode)
    model = k2v5.K2SlotSetSPDNet(
        in_channels=10,
        base_width=int(infer_K2_V5_INFER_CONFIG["base_width"]),
        slot_dim=int(infer_K2_V5_INFER_CONFIG["slot_dim"]),
        decoder_layers=int(infer_K2_V5_INFER_CONFIG["decoder_layers"]),
        num_heads=int(infer_K2_V5_INFER_CONFIG["num_heads"]),
        sharpness=float(infer_K2_V5_INFER_CONFIG["render_sharpness"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    if effective_precision == "fp16":
        model = model.half()
    model.eval()
    image_size = int(infer_K2_V5_INFER_CONFIG["image_size"])
    started = time.perf_counter()
    solved_rows: list[tuple[int, tuple[int, str, str]]] = []
    solved_metrics: list[tuple[int, dict[str, object]]] = []
    effective_prep_workers = max(0, int(prep_workers))
    stage_timings = {
        "prepare_batch": 0.0,
        "h2d": 0.0,
        "forward": 0.0,
        "d2h_states": 0.0,
        "postprocess_exact": 0.0,
    }

    def profile_sync() -> None:
        if profile_stages and use_cuda:
            torch.cuda.synchronize(device)

    def prepare_batch(
        batch_rows: list[tuple[int, int, str, str]],
        batch_payloads: list[tuple[tuple[int, int], tuple[int, int], list]],
        batch_gt_polys: list[list[np.ndarray]],
        executor: concurrent.futures.ThreadPoolExecutor | None,
    ) -> tuple[
        list[tuple[int, int, str, str]],
        list[tuple[tuple[int, int], tuple[int, int], list]],
        list[list[np.ndarray]],
        np.ndarray,
    ]:
        batch_count = len(batch_rows)
        image_batch = np.empty(
            (batch_count, 10, image_size, image_size), dtype=np.float32
        )
        if executor is None or batch_count <= 1:
            for local_idx, payload in enumerate(batch_payloads):
                input_image, _ = infer_build_k2_input_from_payload(
                    payload, image_size=image_size
                )
                image_batch[local_idx] = input_image
            return (batch_rows, batch_payloads, batch_gt_polys, image_batch)
        prepared = list(
            executor.map(
                lambda payload: infer_build_k2_input_from_payload(
                    payload, image_size=image_size
                )[0],
                batch_payloads,
            )
        )
        for local_idx, input_image in enumerate(prepared):
            image_batch[local_idx] = input_image
        return (batch_rows, batch_payloads, batch_gt_polys, image_batch)

    prep_executor: concurrent.futures.ThreadPoolExecutor | None = None
    prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
    if effective_prep_workers > 1:
        prep_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_prep_workers
        )
        prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with torch.inference_mode():
            batch_step = max(1, int(batch_size))
            future: concurrent.futures.Future | None = None
            for start_idx in range(0, len(selected_rows), batch_step):
                end_idx = min(len(selected_rows), start_idx + batch_step)
                batch_rows = selected_rows[start_idx:end_idx]
                batch_payloads = payloads[start_idx:end_idx]
                batch_gt_polys = gt_polys[start_idx:end_idx]
                profile_sync()
                stage_start = time.perf_counter()
                if future is None:
                    prepared_batch = prepare_batch(
                        batch_rows, batch_payloads, batch_gt_polys, prep_executor
                    )
                else:
                    prepared_batch = future.result()
                    future = None
                stage_timings["prepare_batch"] += time.perf_counter() - stage_start
                next_start = end_idx
                if next_start < len(selected_rows) and prefetch_executor is not None:
                    next_end = min(len(selected_rows), next_start + batch_step)
                    future = prefetch_executor.submit(
                        prepare_batch,
                        selected_rows[next_start:next_end],
                        payloads[next_start:next_end],
                        gt_polys[next_start:next_end],
                        prep_executor,
                    )
                batch_rows, batch_payloads, batch_gt_polys, image_batch = prepared_batch
                profile_sync()
                stage_start = time.perf_counter()
                if effective_precision == "fp16":
                    image_tensor = torch.from_numpy(image_batch).to(
                        device=device, dtype=torch.float16, non_blocking=False
                    )
                else:
                    image_tensor = torch.from_numpy(image_batch).to(
                        device, non_blocking=False
                    )
                profile_sync()
                stage_timings["h2d"] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                if effective_forward_mode == "states_only":
                    pred_states_tensor = infer_k2_v5_forward_states_only(
                        model, image_tensor, torch_module=torch, k2v5_module=k2v5
                    )
                else:
                    pred_output = model(image_tensor)
                    pred_states_tensor = pred_output["states"]
                profile_sync()
                stage_timings["forward"] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                pred_states = (
                    pred_states_tensor.view(image_tensor.shape[0], 2, 6)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                profile_sync()
                stage_timings["d2h_states"] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                for local_idx, (
                    original_idx,
                    frame,
                    track_id,
                    _polygons_json,
                ) in enumerate(batch_rows):
                    payload = batch_payloads[local_idx]
                    pred_abs = k2v5.states_to_abs_ellipses_from_payload(
                        pred_states[local_idx], payload
                    )
                    pred_polys = fst.ellipses_to_polygon_arrays(pred_abs)
                    pred_json = json.dumps(
                        [poly.astype(np.float64).tolist() for poly in pred_polys],
                        ensure_ascii=True,
                    )
                    exact = fst.compute_exact_metrics_from_polygons(
                        batch_gt_polys[local_idx], pred_polys
                    )
                    exact["weighted_error"] = int(fst.compute_weighted_error(exact))
                    metric_row = infer_add_weighted_error_norm(
                        {
                            "frame": int(frame),
                            "track_id": str(track_id),
                            "mode": "k2",
                            "candidate_name": "v5_slot_set_spd",
                            "gt_area": int(exact["gt_area"]),
                            "pred_area": int(exact["pred_area"]),
                            "intersection": int(exact["intersection"]),
                            "union": int(exact["union"]),
                            "recall": float(exact["recall"]),
                            "precision": float(exact["precision"]),
                            "iou": float(exact["iou"]),
                            "weighted_error": int(exact["weighted_error"]),
                            "ellipse_params": json.dumps(
                                fst.serialize_ellipses(pred_abs), ensure_ascii=True
                            ),
                            "branch": "k2_v5",
                        }
                    )
                    solved_rows.append(
                        (original_idx, (int(frame), str(track_id), pred_json))
                    )
                    solved_metrics.append((original_idx, metric_row))
                    _publish_ellipse_preview(metric_row)
                stage_timings["postprocess_exact"] += time.perf_counter() - stage_start
                processed = min(end_idx, len(selected_rows))
                if processed % 1000 == 0 or processed == len(selected_rows):
                    print(f"k2_v5: processed {processed}/{len(selected_rows)}")
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if prep_executor is not None:
            prep_executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    model_info = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "image_size": image_size,
        "base_width": int(infer_K2_V5_INFER_CONFIG["base_width"]),
        "slot_dim": int(infer_K2_V5_INFER_CONFIG["slot_dim"]),
        "decoder_layers": int(infer_K2_V5_INFER_CONFIG["decoder_layers"]),
        "num_heads": int(infer_K2_V5_INFER_CONFIG["num_heads"]),
        "render_sharpness": float(infer_K2_V5_INFER_CONFIG["render_sharpness"]),
        "prep_workers": effective_prep_workers,
        "precision_requested": requested_precision,
        "precision": effective_precision,
        "forward_mode": effective_forward_mode,
        "profile_stages": bool(profile_stages),
        "cudnn_benchmark": str(cudnn_benchmark),
        "tf32": str(tf32),
        "stage_timing_sec": stage_timings if profile_stages else {},
    }
    return (solved_rows, solved_metrics, elapsed, model_info)


def infer_main(argv: list[str] | None = None) -> None:
    args = infer_build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_sqlite_path = Path(args.input_sqlite)
    if not input_sqlite_path.is_file():
        raise FileNotFoundError(input_sqlite_path)
    t0 = time.perf_counter()
    load_start = time.perf_counter()
    source_rows = infer_filter_rows(
        fst.load_rows(input_sqlite_path),
        max_rows=int(args.max_rows),
        max_tracks=int(args.max_tracks),
    )
    load_rows_sec = time.perf_counter() - load_start
    row_gt_polygons: list[list[np.ndarray]] = []
    row_local_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    edge_flags: list[bool] = []
    parse_polygons_sec = 0.0
    prepare_payloads_sec = 0.0
    classify_edge_sec = 0.0
    preparation_loop_start = time.perf_counter()
    for _frame, _track_id, polygons_json in source_rows:
        t_parse = time.perf_counter()
        polys = fst.parse_polygons(polygons_json)
        parse_polygons_sec += time.perf_counter() - t_parse
        row_gt_polygons.append(polys)
        t_payload = time.perf_counter()
        payload = fst.prepare_local_raster_payload_from_polygons(polys)
        prepare_payloads_sec += time.perf_counter() - t_payload
        row_local_payloads.append(payload)
        t_classify = time.perf_counter()
        gt_mask, _origin = fst.rasterize_local_mask_from_payload(payload)
        touches = fst.detect_edge_touches(gt_mask)
        classify_edge_sec += time.perf_counter() - t_classify
        edge_flags.append(bool(any(touches.values())))
    preparation_loop_sec = time.perf_counter() - preparation_loop_start
    split_start = time.perf_counter()
    edge_rows: list[tuple[int, int, str, str]] = []
    edge_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    edge_gt_polys: list[list[np.ndarray]] = []
    nonedge_rows: list[tuple[int, int, str, str]] = []
    nonedge_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    nonedge_gt_polys: list[list[np.ndarray]] = []
    for original_idx, (row, polys, payload, is_edge) in enumerate(
        zip(source_rows, row_gt_polygons, row_local_payloads, edge_flags, strict=True)
    ):
        frame, track_id, polygons_json = row
        item = (original_idx, int(frame), str(track_id), str(polygons_json))
        if is_edge:
            edge_rows.append(item)
            edge_payloads.append(payload)
            edge_gt_polys.append(polys)
        else:
            nonedge_rows.append(item)
            nonedge_payloads.append(payload)
            nonedge_gt_polys.append(polys)
    split_rows_sec = time.perf_counter() - split_start
    workers_edge = fst.determine_k1_workers(int(args.k1_workers), len(edge_rows))
    workers_nonedge = fst.determine_k1_workers(int(args.k1_workers), len(nonedge_rows))
    edge_solved_rows, edge_solved_metrics, edge_solve_sec = infer_solve_subset(
        rows_with_index=edge_rows,
        payloads=edge_payloads,
        gt_polys=edge_gt_polys,
        recall_target=float(args.k1_recall_target),
        exact_refine_rounds=int(args.k1_exact_refine_rounds),
        workers=workers_edge,
        branch_name="edge",
    )
    nonedge_solved_rows, nonedge_solved_metrics, nonedge_solve_sec = infer_solve_subset(
        rows_with_index=nonedge_rows,
        payloads=nonedge_payloads,
        gt_polys=nonedge_gt_polys,
        recall_target=float(args.k1_recall_target),
        exact_refine_rounds=int(args.k1_exact_refine_rounds),
        workers=workers_nonedge,
        branch_name="nonedge",
    )
    merge_k1_start = time.perf_counter()
    k1_submission_rows_indexed: list[tuple[int, str, str] | None] = [None] * len(
        source_rows
    )
    k1_metric_rows_indexed: list[dict[str, object] | None] = [None] * len(source_rows)
    for original_idx, row_value in edge_solved_rows:
        k1_submission_rows_indexed[original_idx] = row_value
    for original_idx, row_value in nonedge_solved_rows:
        k1_submission_rows_indexed[original_idx] = row_value
    for original_idx, metric_row in edge_solved_metrics:
        k1_metric_rows_indexed[original_idx] = metric_row
    for original_idx, metric_row in nonedge_solved_metrics:
        k1_metric_rows_indexed[original_idx] = metric_row
    k1_metric_rows = [row for row in k1_metric_rows_indexed if row is not None]
    merge_k1_sec = time.perf_counter() - merge_k1_start
    rows_by_track: dict[str, list[tuple[int, str, str, int]]] = {}
    k1_metrics_lookup: dict[tuple[int, str], dict[str, object]] = {}
    k1_ellipses_lookup: dict[
        tuple[int, str], list[tuple[float, float, float, float, float]]
    ] = {}
    edge_keys: set[tuple[int, str]] = set()
    for idx, (frame, track_id, polygons_json) in enumerate(source_rows):
        rows_by_track.setdefault(str(track_id), []).append(
            (int(frame), str(track_id), str(polygons_json), idx)
        )
        if edge_flags[idx]:
            edge_keys.add((int(frame), str(track_id)))
    for row in k1_metric_rows:
        key = (int(row["frame"]), str(row["track_id"]))
        k1_metrics_lookup[key] = row
        k1_ellipses_lookup[key] = [
            tuple(map(float, e)) for e in json.loads(str(row["ellipse_params"]))
        ]
    k2_band_start = time.perf_counter()
    use_normalized_k1_cost = str(args.k1_cost_routing) == "normalized"
    routing_cost_field = (
        "weighted_error_norm" if use_normalized_k1_cost else "weighted_error"
    )
    routing_metrics_lookup = infer_prepare_k1_routing_metrics_lookup(
        k1_metrics_lookup, cost_field=routing_cost_field
    )
    if use_normalized_k1_cost:
        threshold_default = float(args.threshold_norm)
        effective_threshold_edge = float(
            args.threshold_norm
            if float(args.threshold_edge_norm) < 0.0
            else args.threshold_edge_norm
        )
        soft_k1_keep_cost = float(args.k2_soft_k1_keep_cost_norm)
        hyst_enter = float(args.k2_hyst_enter_norm)
        hyst_enter_edge = float(
            args.k2_hyst_enter_norm
            if float(args.k2_hyst_enter_edge_norm) < 0.0
            else args.k2_hyst_enter_edge_norm
        )
        hyst_exit = float(args.k2_hyst_exit_norm)
        hyst_exit_edge = float(
            args.k2_hyst_exit_norm
            if float(args.k2_hyst_exit_edge_norm) < 0.0
            else args.k2_hyst_exit_edge_norm
        )
        k1n_seq_enter = float(
            args.threshold_norm
            if float(args.k1n_seq_enter_norm) < 0.0
            else args.k1n_seq_enter_norm
        )
        k1n_seq_exit = float(args.k1n_seq_exit_norm)
        k1n_seq_strong_enter = float(args.k1n_seq_strong_enter_norm)
        k1n_seq_strong_exit = float(args.k1n_seq_strong_exit_norm)
        dp_merge_short_k2_keep_cost = float(args.k2_dp_merge_short_k2_keep_cost_norm)
        dp_force_k2_cost = float(args.k2_dp_force_k2_cost_norm)
    else:
        threshold_default = float(args.threshold)
        effective_threshold_edge = float(
            args.threshold if int(args.threshold_edge) < 0 else args.threshold_edge
        )
        soft_k1_keep_cost = float(args.k2_soft_k1_keep_cost)
        hyst_enter = float(args.k2_hyst_enter)
        hyst_enter_edge = float(
            args.k2_hyst_enter
            if int(args.k2_hyst_enter_edge) < 0
            else args.k2_hyst_enter_edge
        )
        hyst_exit = float(args.k2_hyst_exit)
        hyst_exit_edge = float(
            args.k2_hyst_exit
            if int(args.k2_hyst_exit_edge) < 0
            else args.k2_hyst_exit_edge
        )
        k1n_seq_enter = threshold_default
        k1n_seq_exit = hyst_exit
        k1n_seq_strong_enter = -1.0
        k1n_seq_strong_exit = -1.0
        dp_merge_short_k2_keep_cost = float(args.k2_dp_merge_short_k2_keep_cost)
        dp_force_k2_cost = float(args.k2_dp_force_k2_cost)
    if str(args.routing_mode) == "threshold_only":
        selected_keys, k2_band_summary = infer_build_k2_threshold_only_selection(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            threshold_default=threshold_default,
            threshold_edge=effective_threshold_edge,
            edge_keys=edge_keys,
        )
    elif str(args.routing_mode) == "threshold_soft":
        selected_keys, k2_band_summary = infer_build_k2_threshold_soft_selection(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            threshold_default=threshold_default,
            threshold_edge=effective_threshold_edge,
            edge_keys=edge_keys,
            ema_alpha=float(args.k2_soft_ema_alpha),
            band_ratio=float(args.k2_soft_band_ratio),
            exit_ratio=float(args.k2_soft_exit_ratio),
            strong_ratio=float(args.k2_soft_strong_ratio),
            k1_keep_cost=soft_k1_keep_cost,
            reset_gap=int(args.k2_soft_reset_gap),
            merge_islands_max_len=int(args.k2_soft_merge_islands_max_len),
            merge_policy=str(args.k2_soft_merge_policy),
        )
    elif str(args.routing_mode) == "threshold_hysteresis":
        selected_keys, k2_band_summary = infer_build_k2_threshold_hysteresis_selection(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            enter_default=hyst_enter,
            enter_edge=hyst_enter_edge,
            exit_default=hyst_exit,
            exit_edge=hyst_exit_edge,
            edge_keys=edge_keys,
            confirm_frames=int(args.k2_hyst_confirm_frames),
            reset_gap=int(args.k2_hyst_reset_gap),
        )
    elif str(args.routing_mode) == "k1n_sequence":
        selected_keys, k2_band_summary = infer_build_k2_k1n_sequence_selection(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            enter_threshold=k1n_seq_enter,
            exit_threshold=k1n_seq_exit,
            strong_enter_threshold=k1n_seq_strong_enter,
            strong_exit_threshold=k1n_seq_strong_exit,
            protect_k2_iou_below=float(args.k1n_seq_protect_k2_iou_below),
            smooth_window=int(args.k1n_seq_smooth_window),
            enter_confirm_frames=int(args.k1n_seq_enter_confirm_frames),
            exit_confirm_frames=int(args.k1n_seq_exit_confirm_frames),
            merge_short_k1_max_len=int(args.k1n_seq_merge_short_k1_max_len),
            merge_short_k2_max_len=int(args.k1n_seq_merge_short_k2_max_len),
            reset_gap=int(args.k1n_seq_reset_gap),
        )
    elif str(args.routing_mode) == "track_dp":
        selected_keys, k2_band_summary = infer_build_k2_track_dp_selection(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            k1_ellipses_lookup=k1_ellipses_lookup,
            threshold_default=threshold_default,
            threshold_edge=effective_threshold_edge,
            edge_keys=edge_keys,
            error_weight=float(args.k2_dp_error_weight),
            instability_weight=float(args.k2_dp_instability_weight),
            edge_bonus=float(args.k2_dp_edge_bonus),
            k2_bias=float(args.k2_dp_k2_bias),
            switch_12=float(args.k2_dp_switch_12),
            switch_21=float(args.k2_dp_switch_21),
            short_k1_gamma=float(args.k2_dp_short_k1_gamma),
            short_k2_gamma=float(args.k2_dp_short_k2_gamma),
            short_k1_tau=float(args.k2_dp_short_k1_tau),
            short_k2_tau=float(args.k2_dp_short_k2_tau),
            short_len_cap=int(args.k2_dp_short_len_cap),
            reset_gap=int(args.k2_dp_reset_gap),
            merge_short_k1_max_len=int(args.k2_dp_merge_short_k1_max_len),
            merge_short_k2_max_len=int(args.k2_dp_merge_short_k2_max_len),
            merge_short_k2_keep_cost=dp_merge_short_k2_keep_cost,
        )
    elif (not use_normalized_k1_cost) and effective_threshold_edge == threshold_default:
        selected_keys, k2_band_summary = fst.build_k2_solve_band(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=k1_metrics_lookup,
            k1_ellipses_lookup=k1_ellipses_lookup,
            threshold=int(args.threshold),
            radius=int(args.k2_band_radius),
            error_percentile=float(args.k2_band_error_percentile),
            instability_percentile=float(args.k2_band_instability_percentile),
            instability_floor=float(args.k2_band_instability_floor),
        )
    else:
        selected_keys, k2_band_summary = infer_build_k2_solve_band_edge_aware(
            rows_by_track=rows_by_track,
            k1_metrics_lookup=routing_metrics_lookup,
            k1_ellipses_lookup=k1_ellipses_lookup,
            threshold_default=threshold_default,
            threshold_edge=effective_threshold_edge,
            edge_keys=edge_keys,
            radius=int(args.k2_band_radius),
            error_percentile=float(args.k2_band_error_percentile),
            instability_percentile=float(args.k2_band_instability_percentile),
            instability_floor=float(args.k2_band_instability_floor),
        )
    if isinstance(k2_band_summary, dict):
        k2_band_summary.setdefault("routing_mode", str(args.routing_mode))
        k2_band_summary["k1_cost_routing"] = str(args.k1_cost_routing)
        k2_band_summary["routing_cost_field"] = routing_cost_field
        k2_band_summary["threshold_raw"] = float(args.threshold)
        k2_band_summary["threshold_edge_raw"] = float(
            args.threshold if int(args.threshold_edge) < 0 else args.threshold_edge
        )
        k2_band_summary["threshold_norm"] = float(args.threshold_norm)
        k2_band_summary["threshold_edge_norm"] = float(
            args.threshold_norm
            if float(args.threshold_edge_norm) < 0.0
            else args.threshold_edge_norm
        )
    if (
        str(args.routing_mode) == "track_dp"
        and int(args.k2_dp_merge_short_k2_max_len) > 0
    ):
        (
            selected_keys,
            removed_inner_k2_summary,
        ) = infer_cleanup_selected_k2_inner_islands(
            rows_by_track=rows_by_track,
            selected_keys=selected_keys,
            k1_metrics_lookup=routing_metrics_lookup,
            max_len=int(args.k2_dp_merge_short_k2_max_len),
            keep_cost=dp_merge_short_k2_keep_cost,
        )
        if isinstance(k2_band_summary, dict):
            k2_band_summary["selected_count"] = len(selected_keys)
            k2_band_summary["post_removed_inner_k2_runs"] = int(
                removed_inner_k2_summary["removed_runs"]
            )
            k2_band_summary["post_removed_inner_k2_rows"] = int(
                removed_inner_k2_summary["removed_rows"]
            )
            k2_band_summary["post_removed_exact_inner_k2_runs"] = int(
                removed_inner_k2_summary["removed_exact_runs"]
            )
            k2_band_summary["post_removed_exact_inner_k2_rows"] = int(
                removed_inner_k2_summary["removed_exact_rows"]
            )
            k2_band_summary["post_removed_track_order_k2_runs"] = int(
                removed_inner_k2_summary["removed_track_order_runs"]
            )
            k2_band_summary["post_removed_track_order_k2_rows"] = int(
                removed_inner_k2_summary["removed_track_order_rows"]
            )
            k2_band_summary["post_removed_exact_singleton_k2_runs"] = int(
                removed_inner_k2_summary["removed_exact_singleton_runs"]
            )
            k2_band_summary["post_removed_exact_singleton_k2_rows"] = int(
                removed_inner_k2_summary["removed_exact_singleton_rows"]
            )
    if (
        str(args.routing_mode) == "track_dp"
        and int(args.k2_dp_merge_short_k1_max_len) > 0
    ):
        selected_keys, promoted_short_k1_summary = infer_promote_short_k1_runs_to_k2(
            rows_by_track=rows_by_track,
            selected_keys=selected_keys,
            max_len=int(args.k2_dp_merge_short_k1_max_len),
        )
        if isinstance(k2_band_summary, dict):
            k2_band_summary["selected_count"] = len(selected_keys)
            k2_band_summary["post_promoted_short_k1_runs"] = int(
                promoted_short_k1_summary["promoted_runs"]
            )
            k2_band_summary["post_promoted_short_k1_rows"] = int(
                promoted_short_k1_summary["promoted_rows"]
            )
    if str(args.routing_mode) == "track_dp" and dp_force_k2_cost >= 0:
        selected_keys, forced_high_cost_summary = infer_force_select_high_cost_rows(
            rows_by_track=rows_by_track,
            selected_keys=selected_keys,
            k1_metrics_lookup=routing_metrics_lookup,
            min_cost=dp_force_k2_cost,
        )
        if isinstance(k2_band_summary, dict):
            k2_band_summary["selected_count"] = len(selected_keys)
            k2_band_summary["post_forced_high_cost_k2_runs"] = int(
                forced_high_cost_summary["forced_runs"]
            )
            k2_band_summary["post_forced_high_cost_k2_rows"] = int(
                forced_high_cost_summary["forced_rows"]
            )
    k2_band_sec = time.perf_counter() - k2_band_start
    selected_rows: list[tuple[int, int, str, str]] = []
    selected_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    selected_gt_polys: list[list[np.ndarray]] = []
    for idx, (frame, track_id, polygons_json) in enumerate(source_rows):
        key = (int(frame), str(track_id))
        if key not in selected_keys:
            continue
        selected_rows.append((idx, int(frame), str(track_id), str(polygons_json)))
        selected_payloads.append(row_local_payloads[idx])
        selected_gt_polys.append(row_gt_polygons[idx])
    k2_solved_rows, k2_solved_metrics, k2_solve_sec, model_info = infer_infer_k2_v5(
        selected_rows=selected_rows,
        payloads=selected_payloads,
        gt_polys=selected_gt_polys,
        run_dir=args.k2_run_dir,
        device_name=str(args.k2_device),
        batch_size=int(args.k2_batch_size),
        prep_workers=int(args.k2_prep_workers),
        precision=str(args.k2_precision),
        forward_mode=str(args.k2_forward_mode),
        profile_stages=bool(args.k2_profile_stages),
        cudnn_benchmark=str(args.k2_cudnn_benchmark),
        tf32=str(args.k2_tf32),
    )
    merge_final_start = time.perf_counter()
    submission_rows_indexed = list(k1_submission_rows_indexed)
    metric_rows_indexed = list(k1_metric_rows_indexed)
    for original_idx, row_value in k2_solved_rows:
        submission_rows_indexed[original_idx] = row_value
    for original_idx, metric_row in k2_solved_metrics:
        metric_rows_indexed[original_idx] = metric_row
    submission_rows = [row for row in submission_rows_indexed if row is not None]
    metric_rows = [row for row in metric_rows_indexed if row is not None]
    merge_final_sec = time.perf_counter() - merge_final_start
    eval_start = time.perf_counter()
    aggregate = infer_evaluate_mixed_metric_rows(
        metric_rows, total_gt_rows=len(source_rows), total_sub_rows=len(submission_rows)
    )
    eval_sec = time.perf_counter() - eval_start
    write_sqlite_start = time.perf_counter()
    fst.write_sqlite(
        submission_rows,
        args.output_dir / "k1_exact_k2_v5_predictions.sqlite",
        reference_sqlite=input_sqlite_path,
    )
    write_sqlite_sec = time.perf_counter() - write_sqlite_start
    write_metrics_start = time.perf_counter()
    infer_write_metrics_csv(
        k1_metric_rows, args.output_dir / "k1_candidate_metrics.csv"
    )
    infer_write_metrics_csv(metric_rows, args.output_dir / "k1_exact_k2_v5_metrics.csv")
    write_metrics_sec = time.perf_counter() - write_metrics_start
    total_sec = time.perf_counter() - t0
    total_unique_frames = len({int(frame) for frame, _, _ in source_rows})
    summary = {
        "input_sqlite": str(input_sqlite_path),
        "output_dir": str(args.output_dir),
        "config": {
            "k1_recall_target": float(args.k1_recall_target),
            "k1_exact_refine_rounds": int(args.k1_exact_refine_rounds),
            "k1_workers_argument": int(args.k1_workers),
            "k2_run_dir": str(args.k2_run_dir),
            "k2_device": str(args.k2_device),
            "k2_batch_size": int(args.k2_batch_size),
            "k2_prep_workers": int(args.k2_prep_workers),
            "k2_precision": str(args.k2_precision),
            "k2_forward_mode": str(args.k2_forward_mode),
            "k2_profile_stages": bool(args.k2_profile_stages),
            "k2_cudnn_benchmark": str(args.k2_cudnn_benchmark),
            "k2_tf32": str(args.k2_tf32),
            "routing_mode": str(args.routing_mode),
            "threshold": int(args.threshold),
            "threshold_edge": effective_threshold_edge,
            "k2_hyst_enter": int(args.k2_hyst_enter),
            "k2_hyst_enter_edge": int(
                args.k2_hyst_enter
                if int(args.k2_hyst_enter_edge) < 0
                else args.k2_hyst_enter_edge
            ),
            "k2_hyst_exit": int(args.k2_hyst_exit),
            "k2_hyst_exit_edge": int(
                args.k2_hyst_exit
                if int(args.k2_hyst_exit_edge) < 0
                else args.k2_hyst_exit_edge
            ),
            "k2_hyst_confirm_frames": int(args.k2_hyst_confirm_frames),
            "k2_hyst_reset_gap": int(args.k2_hyst_reset_gap),
            "k2_dp_error_weight": float(args.k2_dp_error_weight),
            "k2_dp_instability_weight": float(args.k2_dp_instability_weight),
            "k2_dp_edge_bonus": float(args.k2_dp_edge_bonus),
            "k2_dp_k2_bias": float(args.k2_dp_k2_bias),
            "k2_dp_switch_12": float(args.k2_dp_switch_12),
            "k2_dp_switch_21": float(args.k2_dp_switch_21),
            "k2_dp_short_k1_gamma": float(args.k2_dp_short_k1_gamma),
            "k2_dp_short_k2_gamma": float(args.k2_dp_short_k2_gamma),
            "k2_dp_short_k1_tau": float(args.k2_dp_short_k1_tau),
            "k2_dp_short_k2_tau": float(args.k2_dp_short_k2_tau),
            "k2_dp_short_len_cap": int(args.k2_dp_short_len_cap),
            "k2_dp_reset_gap": int(args.k2_dp_reset_gap),
            "k2_dp_merge_short_k1_max_len": int(args.k2_dp_merge_short_k1_max_len),
            "k2_dp_merge_short_k2_max_len": int(args.k2_dp_merge_short_k2_max_len),
            "k2_dp_merge_short_k2_keep_cost": int(args.k2_dp_merge_short_k2_keep_cost),
            "k2_dp_force_k2_cost": int(args.k2_dp_force_k2_cost),
            "k2_band_radius": int(args.k2_band_radius),
            "k2_band_error_percentile": float(args.k2_band_error_percentile),
            "k2_band_instability_percentile": float(
                args.k2_band_instability_percentile
            ),
            "k2_band_instability_floor": float(args.k2_band_instability_floor),
            "max_rows": int(args.max_rows),
            "max_tracks": int(args.max_tracks),
        },
        "counts": {
            "total_rows": len(source_rows),
            "total_unique_frames": total_unique_frames,
            "edge_rows": len(edge_rows),
            "nonedge_rows": len(nonedge_rows),
            "k2_selected_rows": len(selected_rows),
            "k1_final_rows": len(source_rows) - len(selected_rows),
        },
        "workers": {"edge": workers_edge, "nonedge": workers_nonedge},
        "timing_sec": {
            "load_rows": load_rows_sec,
            "parse_polygons_all": parse_polygons_sec,
            "prepare_payloads_all": prepare_payloads_sec,
            "classify_edge_all": classify_edge_sec,
            "input_preparation_loop_total": preparation_loop_sec,
            "split_rows": split_rows_sec,
            "edge_solve": edge_solve_sec,
            "nonedge_solve": nonedge_solve_sec,
            "merge_k1": merge_k1_sec,
            "k2_band_select": k2_band_sec,
            "k2_v5_solve": k2_solve_sec,
            "merge_final": merge_final_sec,
            "evaluate_submission": eval_sec,
            "write_sqlite": write_sqlite_sec,
            "write_metrics_csv": write_metrics_sec,
            "end_to_end_total": total_sec,
        },
        "throughput": {
            "end_to_end_rows_per_sec": len(source_rows) / max(total_sec, 1e-09),
            "end_to_end_unique_frames_per_sec": total_unique_frames
            / max(total_sec, 1e-09),
            "k2_v5_rows_per_sec": len(selected_rows) / max(k2_solve_sec, 1e-09)
            if selected_rows
            else 0.0,
        },
        "metrics": aggregate,
        "k2_band_summary": k2_band_summary,
        "k2_v5_model": model_info,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    infer_main()
