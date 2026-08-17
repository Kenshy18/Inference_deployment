#!/usr/bin/env python3
"""Production multistate search on the hard-Recall penalty DP.

The Production source and Phase-1 constraint implementation stay unchanged.
This private runtime adds a small, screened set of initial polygon states per
frame.  Pair-vote and post-decode repair remain disabled.  No video pixels are
opened; candidates are derived only from tracked SQLite polygon geometry.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

from production.polygon.runtime.algorithm_ids import (
    PHASE2_CONSTRAINED_PAIR_VOTE_ALGORITHM_ID,
    PHASE2_NO_PAIR_VOTE_ALGORITHM_ID,
    PHASE2_PER_KEY_PAIR_VOTE_ALGORITHM_ID,
    PHASE2_POST_DP_PAIR_VOTE_ALGORITHM_ID,
)
from production.polygon.runtime.diagnostics import classify_streams

from production.polygon.runtime.phase1_runtime import (
    _EPSILON,
    _load_production_runtime,
    _patch_embedded_optimizer,
)
from production.polygon.runtime.phase2_config import (
    PROFILE_ENV,
    POLYGON_CONSTRAINED_PROFILES,
    GC_INTERVAL_ENV,
    OPENCV_THREADS_ENV,
    PAIR_VOTE_ENV,
    PAIR_VOTE_CONSTRAINED_ENV,
    PAIR_VOTE_PER_KEY_ENV,
    PAIR_VOTE_SWEEPS_ENV,
    NEW_PRODUCTION_FAST_PAIR_VOTE_ENV,
    SCALE_STATE_PROFILES,
    VALID_PROFILES,
    _class_role_state_profile,
    _load_spatial_vertex_policy,
    _spatial_vertices_for_track,
)

from production.polygon.runtime.phase2_candidates import (
    _axis_vectors,
    _componentwise_scale,
    _patch_phase2_candidates,
    _temporal_vectors,
)
from production.polygon.runtime.phase2_hard_dp import _build_dense_edge_array


def _write_audit(
    output_dir: Path,
    recall_floor: float,
    patched_module: ModuleType | None,
    profile: str,
) -> dict[str, object]:
    candidate_contract = None
    if profile in POLYGON_CONSTRAINED_PROFILES:
        from production.polygon.runtime.spatial_config import CANDIDATE

        if profile == "polygon_adaptive_keyframe_v2":
            from production.polygon.runtime.candidate_config import (
                CANDIDATE as ADAPTIVE_CANDIDATE,
                with_target_interval,
            )

            requested_interval = int(
                os.environ.get(
                    "MASK_PIPELINE_PHASE2_TARGET_INTERVAL",
                    str(ADAPTIVE_CANDIDATE.temporal.target_interval),
                )
            )
            candidate_contract = with_target_interval(
                requested_interval, ADAPTIVE_CANDIDATE
            ).to_dict()
        else:
            candidate_contract = CANDIDATE.to_dict()
    metrics_path = output_dir / "exact/keyframe_exact_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    stream_rows = []
    with (output_dir / "opt/stream_segments.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        stream_rows.extend(csv.DictReader(handle))
    diagnostics = classify_streams(
        metric_rows, stream_rows, recall_floor=recall_floor, epsilon=_EPSILON
    )
    optimizer_fallback = diagnostics.optimizer_fallback
    legacy_budget_diagnostic = diagnostics.legacy_budget_diagnostic
    final_exact_infeasible = diagnostics.final_exact_infeasible
    feasible_recalls = [
        float(row["recall"])
        for row in metric_rows
        if (str(row["track_id"]), int(row["run_id"])) not in final_exact_infeasible
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    optimizer = summary["optimizer_summary"]
    selected_state_counts: dict[str, int] = {}
    selected_state_pair_counts: dict[str, int] = {}
    state_labels = {"0": "raw"}
    active_roles = (
        getattr(patched_module, "_phase2_active_role_ids", None)
        if patched_module is not None
        else None
    )
    if active_roles is not None:
        state_labels.update(
            {str(index): role_id for index, role_id in enumerate(active_roles, start=1)}
        )
    elif profile in SCALE_STATE_PROFILES:
        state_labels.update(
            {
                str(index): f"scale_{factor:.3f}"
                for index, factor in enumerate(SCALE_STATE_PROFILES[profile], start=1)
            }
        )
    final_keyframes_path = output_dir / "opt/final_keyframes.json"
    if final_keyframes_path.is_file():
        final_keys = json.loads(final_keyframes_path.read_text(encoding="utf-8"))
        for key in final_keys:
            state = str(int(key.get("candidate_id", 0)))
            selected_state_counts[state] = selected_state_counts.get(state, 0) + 1
        grouped_keys: dict[tuple[str, int], list[dict[str, object]]] = {}
        for key in final_keys:
            group = (str(key["track_id"]), int(key["run_id"]))
            grouped_keys.setdefault(group, []).append(key)
        for keys in grouped_keys.values():
            keys.sort(key=lambda value: int(value["frame"]))
            for left, right in zip(keys, keys[1:]):
                left_state = str(int(left.get("candidate_id", 0)))
                right_state = str(int(right.get("candidate_id", 0)))
                pair = (
                    f"{state_labels.get(left_state, left_state)}"
                    f"->{state_labels.get(right_state, right_state)}"
                )
                selected_state_pair_counts[pair] = (
                    selected_state_pair_counts.get(pair, 0) + 1
                )
    pair_vote_requested = os.environ.get(PAIR_VOTE_ENV, "0").strip() == "1"
    constrained_pair_vote = (
        os.environ.get(PAIR_VOTE_CONSTRAINED_ENV, "0").strip() == "1"
    )
    per_key_pair_vote = os.environ.get(PAIR_VOTE_PER_KEY_ENV, "0").strip() == "1"
    audit = {
        "schema_version": 1,
        "algorithm": (
            PHASE2_PER_KEY_PAIR_VOTE_ALGORITHM_ID
            if per_key_pair_vote
            else (
                PHASE2_CONSTRAINED_PAIR_VOTE_ALGORITHM_ID
                if constrained_pair_vote
                else (
                    PHASE2_POST_DP_PAIR_VOTE_ALGORITHM_ID
                    if pair_vote_requested
                    else PHASE2_NO_PAIR_VOTE_ALGORITHM_ID
                )
            )
        ),
        "candidate_profile": profile,
        "production_candidate_contract": candidate_contract,
        "recall_floor": float(recall_floor),
        "evaluated_rows": len(metric_rows),
        "minimum_recall": min(
            (float(row["recall"]) for row in metric_rows), default=1.0
        ),
        "mean_iou": sum(float(row["iou"]) for row in metric_rows)
        / max(len(metric_rows), 1),
        "infeasible_streams": len(final_exact_infeasible),
        "optimizer_fallback_streams": len(optimizer_fallback),
        "legacy_budget_diagnostic_streams": len(legacy_budget_diagnostic),
        "exact_recall_violations": sum(
            float(row["recall"]) + 1e-12 < float(recall_floor) for row in metric_rows
        ),
        "feasible_exact_minimum_recall": min(feasible_recalls, default=1.0),
        "feasible_exact_violations": sum(
            value + 1e-12 < float(recall_floor) for value in feasible_recalls
        ),
        "mean_state_count": float(optimizer["mean_state_count"]),
        "candidate_state_labels": state_labels,
        "selected_candidate_ids": selected_state_counts,
        "selected_candidate_pairs": dict(
            sorted(
                selected_state_pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "role_generation": (
            dict(getattr(patched_module, "_phase2_role_generation_stats", {}))
            if patched_module is not None
            else {}
        ),
        "spatial_profile": (
            dict(
                getattr(
                    patched_module,
                    "_phase2_pipeline_profile",
                    {},
                )
            )
            if patched_module is not None
            else {}
        ),
        "pair_vote_acceleration": (
            dict(getattr(patched_module, "_phase2_pair_vote_fast_stats", {}))
            if patched_module is not None
            else {}
        ),
        "topology_guard": (
            dict(
                getattr(
                    patched_module,
                    "_polygon14_topology_guard_stats",
                    {},
                )
            )
            if patched_module is not None
            else {}
        ),
        "pair_vote_disabled": not bool(optimizer["pair_vote_refine_enabled"]),
        "post_decode_shape_repair_disabled": bool(
            patched_module is not None
            and getattr(patched_module, "_phase1_exact_repair_disabled", False)
        ),
        "dense_candidate_pool": int(optimizer["candidate_frame_count_total"])
        == int(optimizer["row_count"]),
        "semantic_changes": {
            "candidate_shapes": profile,
            "spatial_polygon_representation": (
                "track-wise 14/16/18/20-point line-fit fallback with native "
                "exact Recall repair; tracked source masks remain the exact "
                "Recall reference"
                if profile in POLYGON_CONSTRAINED_PROFILES
                else "unchanged"
            ),
            "candidate_positions": "all prepared observations and gap-filled frames",
            "recall": "hard per-frame edge feasibility",
            "selection": "Production quality loss plus lambda per key",
            "pair_vote": (
                "per-key IoU-only coordinate optimization toward Production pair-vote; exact per-frame minimum Recall floor"
                if per_key_pair_vote
                else (
                    "IoU-only constrained blend of best-v4 and Production pair-vote; exact per-frame minimum Recall floor"
                    if constrained_pair_vote
                    else (
                        "Production post-DP least-squares endpoint vote enabled"
                        if pair_vote_requested
                        else "disabled"
                    )
                )
            ),
            "post_decode_repair": "disabled",
            "topology": (
                "lazy hard constraint on selected DP interpolation, "
                "pair-vote keyframes, and final dense interpolation"
                if profile in POLYGON_CONSTRAINED_PROFILES
                else "unchanged"
            ),
        },
    }
    pair_vote_mode_matches = bool(audit["pair_vote_disabled"]) != bool(
        pair_vote_requested
    )
    audit["implementation_contract_satisfied"] = (
        pair_vote_mode_matches
        and bool(audit["post_decode_shape_repair_disabled"])
        and bool(audit["dense_candidate_pool"])
        and (pair_vote_requested or int(audit["feasible_exact_violations"]) == 0)
    )
    if not bool(audit["implementation_contract_satisfied"]):
        raise RuntimeError(f"Phase 2 contract audit failed: {audit}")
    (output_dir / "phase2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    # Each class is normally scheduled as an independent process.  Leaving
    # OpenCV at its machine-wide default in every process oversubscribes the
    # 24-core host (3 x 24 threads).  This changes scheduling only; all raster
    # operations and emitted geometry remain identical.
    import cv2

    cv2.setNumThreads(
        max(1, int(os.environ.get(OPENCV_THREADS_ENV, str(os.cpu_count() or 1))))
    )
    profile = os.environ.get(PROFILE_ENV, "raw_baseline").strip()
    if profile not in VALID_PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {sorted(VALID_PROFILES)}")
    source = _load_production_runtime()
    patched_holder: list[ModuleType] = []

    def build_patched_module() -> ModuleType:
        patched = _patch_embedded_optimizer(source)
        patched._phase1_exact_repair_disabled = True
        patched = _patch_phase2_candidates(patched, profile)
        pair_vote_enabled = os.environ.get(PAIR_VOTE_ENV, "0").strip() == "1"
        constrained_pair_vote = (
            os.environ.get(PAIR_VOTE_CONSTRAINED_ENV, "0").strip() == "1"
        )
        per_key_pair_vote = os.environ.get(PAIR_VOTE_PER_KEY_ENV, "0").strip() == "1"
        if constrained_pair_vote and not pair_vote_enabled:
            raise RuntimeError("constrained pair-vote requires pair-vote enabled")
        if per_key_pair_vote and not constrained_pair_vote:
            raise RuntimeError("per-key pair-vote requires constrained pair-vote")
        production_pair_vote = patched.pair_vote_refine_keyframe_vectors
        pair_vote_fast_stats: dict[str, float | int | bool | str] = {
            "requested": bool(
                os.environ.get(NEW_PRODUCTION_FAST_PAIR_VOTE_ENV, "0").strip() == "1"
            ),
            "enabled": False,
            "mode": "reference",
        }
        patched._phase2_pair_vote_fast_stats = pair_vote_fast_stats

        if constrained_pair_vote:
            topology_guard_enabled = profile in POLYGON_CONSTRAINED_PROFILES
            if topology_guard_enabled:
                from production.polygon.runtime.topology import (
                    local_key_update_is_simple,
                    path_is_simple,
                )

                topology_guard_stats = patched._polygon14_topology_guard_stats

            def constrained_pair_vote_refine(
                run,
                chosen_frames,
                keyframe_vectors,
                args,
            ):
                baseline = patched.np.asarray(
                    keyframe_vectors, dtype=patched.np.float32
                )
                if len(chosen_frames) <= 1:
                    return baseline
                voted = patched.np.asarray(
                    production_pair_vote(run, chosen_frames, baseline, args),
                    dtype=patched.np.float32,
                )
                delta = voted - baseline
                if bool(patched.np.allclose(delta, 0.0, atol=1e-7)):
                    return baseline

                exact_evaluator = None
                if bool(pair_vote_fast_stats["requested"]):
                    if profile not in {
                        "new_production_v1",
                        "polygon14_keyframe_v1",
                        "polygon_adaptive_keyframe_v2",
                    }:
                        raise RuntimeError(
                            "fast pair-vote is restricted to the frozen "
                            "new-production temporal profiles"
                        )
                    from production.polygon.runtime.pair_vote import (
                        ExactPairVoteEvaluator,
                    )

                    exact_evaluator = ExactPairVoteEvaluator(
                        patched,
                        run,
                        [int(value) for value in chosen_frames],
                        baseline,
                        voted,
                        pair_vote_fast_stats,
                    )
                    pair_vote_fast_stats["enabled"] = True

                def full_metrics(vectors):
                    if exact_evaluator is not None:
                        return exact_evaluator.full_metrics(vectors)
                    return patched.exact_interpolated_metrics(
                        run, chosen_frames, vectors
                    )

                recall_floor = float(args.recall_min)
                evaluations: dict[float, tuple[float, float, patched.np.ndarray]] = {}

                def evaluate(alpha: float):
                    alpha = float(min(max(alpha, 0.0), 1.0))
                    cache_key = round(alpha, 12)
                    cached = evaluations.get(cache_key)
                    if cached is not None:
                        return cached
                    trial = (
                        baseline.astype(patched.np.float64)
                        + alpha * delta.astype(patched.np.float64)
                    ).astype(patched.np.float32)
                    (
                        rows,
                        _loss,
                        mean_iou,
                        _mean_recall,
                        _precision,
                        _global,
                    ) = full_metrics(trial)
                    minimum_recall = min(
                        (float(row["recall"]) for row in rows),
                        default=1.0,
                    )
                    value = (float(mean_iou), minimum_recall, trial)
                    evaluations[cache_key] = value
                    return value

                def evaluate_many(alphas):
                    normalized = [
                        float(min(max(float(alpha), 0.0), 1.0)) for alpha in alphas
                    ]
                    missing = []
                    missing_trials = []
                    if exact_evaluator is not None:
                        for alpha in normalized:
                            cache_key = round(alpha, 12)
                            if cache_key in evaluations:
                                continue
                            trial = (
                                baseline.astype(patched.np.float64)
                                + alpha * delta.astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            missing.append((cache_key, trial))
                            missing_trials.append(trial)
                        if missing_trials:
                            metrics = exact_evaluator.full_metrics_many(missing_trials)
                            for (cache_key, trial), (
                                mean_iou,
                                minimum_recall,
                            ) in zip(missing, metrics):
                                evaluations[cache_key] = (
                                    float(mean_iou),
                                    float(minimum_recall),
                                    trial,
                                )
                        return [evaluations[round(alpha, 12)] for alpha in normalized]
                    return [evaluate(alpha) for alpha in normalized]

                # Pure objective: maximize exact dense mean IoU.  Recall is a
                # hard per-frame constraint.  There is deliberately no motion,
                # shape, or temporal-smoothness penalty in this experiment.
                coarse = [index / 32.0 for index in range(33)]
                feasible: list[tuple[float, float, patched.np.ndarray]] = []
                for alpha, (mean_iou, minimum_recall, trial) in zip(
                    coarse, evaluate_many(coarse)
                ):
                    if minimum_recall + 1e-12 >= recall_floor:
                        feasible.append((mean_iou, alpha, trial))
                if not feasible:
                    if exact_evaluator is not None:
                        exact_evaluator.close()
                    return baseline
                _best_iou, best_alpha, _best_trial = max(
                    feasible, key=lambda item: (item[0], -item[1])
                )
                # Refine the best coarse neighborhood to 1/256 alpha.  The
                # raster objective is not assumed monotone or differentiable.
                refine_start = max(0.0, best_alpha - 1.0 / 32.0)
                refine_end = min(1.0, best_alpha + 1.0 / 32.0)
                refine_steps = int(round((refine_end - refine_start) * 256.0))
                refine_alphas = [
                    refine_start + index / 256.0 for index in range(refine_steps + 1)
                ]
                for alpha, (mean_iou, minimum_recall, trial) in zip(
                    refine_alphas, evaluate_many(refine_alphas)
                ):
                    if minimum_recall + 1e-12 >= recall_floor:
                        feasible.append((mean_iou, alpha, trial))

                def best_topology_valid(
                    values: list[tuple[float, float, patched.np.ndarray]],
                ):
                    if not topology_guard_enabled:
                        return max(values, key=lambda item: (item[0], -item[1]))
                    guard_started = time.perf_counter()
                    for item in sorted(
                        values,
                        key=lambda candidate: (candidate[0], -candidate[1]),
                        reverse=True,
                    ):
                        topology_guard_stats["pair_vote_paths_checked"] = (
                            int(topology_guard_stats["pair_vote_paths_checked"]) + 1
                        )
                        if path_is_simple(
                            patched,
                            run,
                            [int(value) for value in chosen_frames],
                            item[2],
                        ):
                            topology_guard_stats["pair_vote_guard_seconds"] = float(
                                topology_guard_stats["pair_vote_guard_seconds"]
                            ) + (time.perf_counter() - guard_started)
                            return item
                        topology_guard_stats["pair_vote_paths_rejected"] = (
                            int(topology_guard_stats["pair_vote_paths_rejected"]) + 1
                        )
                    topology_guard_stats["pair_vote_guard_seconds"] = float(
                        topology_guard_stats["pair_vote_guard_seconds"]
                    ) + (time.perf_counter() - guard_started)
                    return None

                selected = best_topology_valid(feasible)
                if selected is None:
                    if exact_evaluator is not None:
                        exact_evaluator.close()
                    return baseline
                if not per_key_pair_vote:
                    result = selected[2]
                else:
                    result = _per_key_refine(
                        run=run,
                        chosen_frames=chosen_frames,
                        baseline=baseline,
                        voted=voted,
                        initial=selected[2],
                        recall_floor=recall_floor,
                        exact_evaluator=exact_evaluator,
                    )
                if exact_evaluator is not None:
                    exact_evaluator.close()
                return result

            def _per_key_refine(
                *,
                run,
                chosen_frames,
                baseline,
                voted,
                initial,
                recall_floor: float,
                exact_evaluator,
            ):
                """Coordinate-ascent alpha per fixed key, with exact local gates."""
                chosen = [int(value) for value in chosen_frames]
                current = patched.np.asarray(initial, dtype=patched.np.float32).copy()
                delta = patched.np.asarray(
                    voted, dtype=patched.np.float32
                ) - patched.np.asarray(baseline, dtype=patched.np.float32)
                # Recover each initial alpha from its exact line projection.
                alphas = patched.np.zeros((len(chosen),), dtype=patched.np.float64)
                for key_pos in range(len(chosen)):
                    direction = delta[key_pos].astype(patched.np.float64).reshape(-1)
                    displacement = (
                        current[key_pos].astype(patched.np.float64)
                        - baseline[key_pos].astype(patched.np.float64)
                    ).reshape(-1)
                    denominator = float(direction @ direction)
                    if denominator > 1e-12:
                        alphas[key_pos] = float(
                            patched.np.clip(
                                float(displacement @ direction) / denominator,
                                0.0,
                                1.0,
                            )
                        )

                def local_metrics(key_pos: int, trial_vector):
                    if exact_evaluator is not None:
                        return exact_evaluator.local_metrics(
                            current, key_pos, trial_vector
                        )
                    left_key = max(0, key_pos - 1)
                    right_key = min(len(chosen) - 1, key_pos + 1)
                    start_frame = chosen[left_key]
                    end_frame = chosen[right_key]
                    iou_total = 0.0
                    minimum_recall = 1.0
                    for frame_idx in range(start_frame, end_frame + 1):
                        if frame_idx <= chosen[0]:
                            vector = trial_vector if key_pos == 0 else current[0]
                        elif frame_idx >= chosen[-1]:
                            vector = (
                                trial_vector
                                if key_pos == len(chosen) - 1
                                else current[-1]
                            )
                        else:
                            right_pos = int(
                                patched.np.searchsorted(
                                    patched.np.asarray(chosen, dtype=patched.np.int32),
                                    frame_idx,
                                    side="left",
                                )
                            )
                            left_pos = max(0, right_pos - 1)
                            if frame_idx == chosen[right_pos]:
                                vector = (
                                    trial_vector
                                    if right_pos == key_pos
                                    else current[right_pos]
                                )
                            else:
                                alpha_frame = float(
                                    (frame_idx - chosen[left_pos])
                                    / max(chosen[right_pos] - chosen[left_pos], 1)
                                )
                                left_vector = (
                                    trial_vector
                                    if left_pos == key_pos
                                    else current[left_pos]
                                )
                                right_vector = (
                                    trial_vector
                                    if right_pos == key_pos
                                    else current[right_pos]
                                )
                                vector = patched.interpolate_vectors(
                                    left_vector, right_vector, alpha_frame
                                )
                        polygons = patched.split_vector_to_polygons(
                            vector,
                            run.contour_count,
                            run.anchors_per_contour,
                        )
                        metrics = patched.compute_exact_metrics_from_polygons(
                            run.gt_polygons[frame_idx], polygons
                        )
                        iou_total += float(metrics["iou"])
                        minimum_recall = min(minimum_recall, float(metrics["recall"]))
                    return iou_total, minimum_recall

                def local_metrics_many(key_pos: int, trial_vectors):
                    if exact_evaluator is not None:
                        return exact_evaluator.local_metrics_many(
                            current, key_pos, trial_vectors
                        )
                    return [
                        local_metrics(key_pos, trial_vector)
                        for trial_vector in trial_vectors
                    ]

                # Alternate forward/backward coordinate sweeps.  Two sweeps
                # reproduce the original experiment; larger values measure
                # how close that result is to coordinate-wise saturation.
                sweep_count = max(
                    1,
                    int(os.environ.get(PAIR_VOTE_SWEEPS_ENV, "2") or "2"),
                )
                for sweep_index in range(sweep_count):
                    order = (
                        range(len(chosen))
                        if sweep_index % 2 == 0
                        else range(len(chosen) - 1, -1, -1)
                    )
                    for key_pos in order:
                        if bool(patched.np.allclose(delta[key_pos], 0.0, atol=1e-7)):
                            continue
                        coarse = [index / 16.0 for index in range(17)]
                        coarse.append(float(alphas[key_pos]))
                        candidates: list[tuple[float, float, patched.np.ndarray]] = []
                        seen = set()
                        coarse_trials = []
                        for alpha in coarse:
                            alpha = float(min(max(alpha, 0.0), 1.0))
                            cache_key = round(alpha, 12)
                            if cache_key in seen:
                                continue
                            seen.add(cache_key)
                            trial = (
                                baseline[key_pos].astype(patched.np.float64)
                                + alpha * delta[key_pos].astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            coarse_trials.append((alpha, trial))
                        coarse_metrics = local_metrics_many(
                            key_pos, [trial for _alpha, trial in coarse_trials]
                        )
                        for (alpha, trial), (iou_sum, minimum_recall) in zip(
                            coarse_trials, coarse_metrics
                        ):
                            if minimum_recall + 1e-12 >= recall_floor:
                                candidates.append((iou_sum, alpha, trial))
                        if not candidates:
                            continue
                        _coarse_iou, coarse_alpha, _coarse_trial = max(
                            candidates, key=lambda item: (item[0], -item[1])
                        )
                        refine_start = max(0.0, coarse_alpha - 1.0 / 16.0)
                        refine_end = min(1.0, coarse_alpha + 1.0 / 16.0)
                        refine_steps = int(round((refine_end - refine_start) * 128.0))
                        refine_trials = []
                        for index in range(refine_steps + 1):
                            alpha = refine_start + index / 128.0
                            cache_key = round(alpha, 12)
                            if cache_key in seen:
                                continue
                            seen.add(cache_key)
                            trial = (
                                baseline[key_pos].astype(patched.np.float64)
                                + alpha * delta[key_pos].astype(patched.np.float64)
                            ).astype(patched.np.float32)
                            refine_trials.append((alpha, trial))
                        refine_metrics = local_metrics_many(
                            key_pos, [trial for _alpha, trial in refine_trials]
                        )
                        for (alpha, trial), (iou_sum, minimum_recall) in zip(
                            refine_trials, refine_metrics
                        ):
                            if minimum_recall + 1e-12 >= recall_floor:
                                candidates.append((iou_sum, alpha, trial))
                        ordered_candidates = sorted(
                            candidates,
                            key=lambda item: (item[0], -item[1]),
                            reverse=True,
                        )
                        selected_candidate = None
                        guard_started = time.perf_counter()
                        for candidate in ordered_candidates:
                            if topology_guard_enabled:
                                topology_guard_stats[
                                    "pair_vote_local_trials_checked"
                                ] = (
                                    int(
                                        topology_guard_stats[
                                            "pair_vote_local_trials_checked"
                                        ]
                                    )
                                    + 1
                                )
                                if not local_key_update_is_simple(
                                    patched,
                                    run,
                                    chosen,
                                    current,
                                    key_pos,
                                    candidate[2],
                                ):
                                    topology_guard_stats[
                                        "pair_vote_local_trials_rejected"
                                    ] = (
                                        int(
                                            topology_guard_stats[
                                                "pair_vote_local_trials_rejected"
                                            ]
                                        )
                                        + 1
                                    )
                                    continue
                            selected_candidate = candidate
                            break
                        if topology_guard_enabled:
                            topology_guard_stats["pair_vote_guard_seconds"] = float(
                                topology_guard_stats["pair_vote_guard_seconds"]
                            ) + (time.perf_counter() - guard_started)
                        if selected_candidate is None:
                            continue
                        best_iou, best_alpha, best_trial = selected_candidate
                        current_iou, current_recall = local_metrics(
                            key_pos, current[key_pos]
                        )
                        if (
                            current_recall + 1e-12 >= recall_floor
                            and current_iou > best_iou + 1e-12
                        ):
                            continue
                        current[key_pos] = best_trial
                        alphas[key_pos] = float(best_alpha)

                # Defensive whole-track validation.  Coordinate updates are
                # locally sufficient, but never emit an unverified path.
                if exact_evaluator is not None:
                    (
                        rows,
                        _loss,
                        _iou,
                        _recall,
                        _precision,
                        _global,
                    ) = exact_evaluator.full_metrics(current)
                else:
                    (
                        rows,
                        _loss,
                        _iou,
                        _recall,
                        _precision,
                        _global,
                    ) = patched.exact_interpolated_metrics(run, chosen_frames, current)
                if (
                    min((float(row["recall"]) for row in rows), default=1.0) + 1e-12
                    < recall_floor
                ):
                    return initial
                if topology_guard_enabled and not path_is_simple(
                    patched,
                    run,
                    chosen,
                    current,
                ):
                    return initial
                return current

            patched.pair_vote_refine_keyframe_vectors = constrained_pair_vote_refine
        previous_defaults = patched.apply_fixed_practical_defaults

        def apply_pair_vote_defaults(args: argparse.Namespace) -> argparse.Namespace:
            args = previous_defaults(args)
            # Isolate pair-vote's contribution.  The DP and its candidate
            # states are unchanged, and the later mean-Recall expansion repair
            # remains disabled so it cannot hide or compensate vote effects.
            args.pair_vote_refine_enabled = bool(pair_vote_enabled)
            args.exact_recall_repair_enabled = False
            return args

        patched.apply_fixed_practical_defaults = apply_pair_vote_defaults
        patched._phase2_pair_vote_enabled = bool(pair_vote_enabled)
        patched._phase2_constrained_pair_vote_enabled = bool(constrained_pair_vote)
        patched._phase2_per_key_pair_vote_enabled = bool(per_key_pair_vote)
        patched_holder.append(patched)
        return patched

    patched = build_patched_module()
    os.environ["ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA"] = "1"
    gc_interval = max(1, int(os.environ.get(GC_INTERVAL_ENV, "1") or "1"))
    real_gc_collect = gc.collect
    gc_profile: dict[str, float | int] = {
        "requested_calls": 0,
        "executed_calls": 0,
        "skipped_calls": 0,
        "seconds": 0.0,
        "interval": int(gc_interval),
    }

    def throttled_gc_collect(*args, **kwargs):
        gc_profile["requested_calls"] = int(gc_profile["requested_calls"]) + 1
        if int(gc_profile["requested_calls"]) % int(gc_interval) != 0:
            gc_profile["skipped_calls"] = int(gc_profile["skipped_calls"]) + 1
            return 0
        started = time.perf_counter()
        result = real_gc_collect(*args, **kwargs)
        gc_profile["seconds"] = float(gc_profile["seconds"]) + (
            time.perf_counter() - started
        )
        gc_profile["executed_calls"] = int(gc_profile["executed_calls"]) + 1
        return result

    if gc_interval > 1:
        gc.collect = throttled_gc_collect
    try:
        optimizer_argv = (
            sys.argv[2:]
            if (len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize")
            else sys.argv[1:]
        )
        previous_argv = sys.argv[:]
        try:
            sys.argv = [previous_argv[0], *optimizer_argv]
            patched.main()
        finally:
            sys.argv = previous_argv
    finally:
        if gc_interval > 1:
            gc.collect = real_gc_collect
            started = time.perf_counter()
            real_gc_collect()
            gc_profile["seconds"] = float(gc_profile["seconds"]) + (
                time.perf_counter() - started
            )
            gc_profile["executed_calls"] = int(gc_profile["executed_calls"]) + 1
        print(json.dumps({"phase2_gc_profile": gc_profile}), flush=True)
    if patched_holder:
        print(
            json.dumps(
                {
                    "phase2_pipeline_profile": dict(
                        getattr(patched_holder[-1], "_phase2_pipeline_profile", {})
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--recall-min", type=float, default=0.97)
        known, _unknown = parser.parse_known_args(sys.argv[2:])
        audit = _write_audit(
            known.output_dir,
            known.recall_min,
            patched_holder[-1] if patched_holder else None,
            profile,
        )
        print(json.dumps({"phase2_audit": audit}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
