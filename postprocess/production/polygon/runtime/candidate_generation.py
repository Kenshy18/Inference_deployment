"""Generate polygon states and patch them into the hard-Recall graph."""

from __future__ import annotations

import concurrent.futures
import os
import time
from types import ModuleType

import numpy as np

from production.polygon.runtime.geometry import (
    axis_scale as _axis_scale,
    principal_basis as _principal_basis,
    rigid_align as _rigid_align,
    temporal_shapes as _temporal_shapes,
)
from production.polygon.runtime.native_runtime import _EPSILON
from production.polygon.runtime.hard_recall_dp import (
    build_hard_multistate_penalty_path,
)
from production.polygon.runtime.runtime_config import (
    CANDIDATE_FRAME_WORKERS_ENV,
    CLASS_ROLE_STATE_PROFILES,
    MIXED_STATE_PROFILES,
    PERSISTENT_LINE_FIT_BASE_ENV,
    PERSISTENT_LINE_FIT_VERTICES_ENV,
    POLYGON_CONSTRAINED_PROFILES,
    ROLE_STATE_PROFILES,
    SCALE_STATE_PROFILES,
    VALID_PROFILES,
    _class_role_state_profile,
    _spatial_vertices_for_track,
)
from production.polygon.runtime.role_candidates import build_role_candidate


def _componentwise_scale(anchors: np.ndarray, factor: float) -> np.ndarray:
    output = np.asarray(anchors, dtype=np.float64).copy()
    for slot in range(len(output)):
        center = np.mean(output[slot], axis=0)
        output[slot] = center + float(factor) * (output[slot] - center)
    return output.astype(np.float32)


def _temporal_vectors(
    run,
    frame_index: int,
    *,
    radii: tuple[int, ...],
    quantiles: tuple[float, ...],
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    central_output: list[tuple[str, np.ndarray]] = []
    recall_output: list[tuple[str, np.ndarray]] = []
    reference_slots = np.asarray(run.anchors[int(frame_index)], dtype=np.float64)
    current_frame = int(run.frame_numbers[int(frame_index)])
    for radius in radii:
        neighbour_indices = [
            index
            for index, frame in enumerate(run.frame_numbers.tolist())
            if abs(int(frame) - current_frame) <= int(radius)
        ]
        if len(neighbour_indices) < 2:
            continue
        aligned_by_slot: list[np.ndarray] = []
        for slot, reference in enumerate(reference_slots):
            aligned_by_slot.append(
                np.stack(
                    [
                        _rigid_align(
                            np.asarray(reference, dtype=np.float64),
                            np.asarray(run.anchors[index][slot], dtype=np.float64),
                        )
                        for index in neighbour_indices
                    ],
                    axis=0,
                )
            )
        central_slots = []
        for slot, aligned in enumerate(aligned_by_slot):
            central, _coverage = _temporal_shapes(
                reference_slots[slot], aligned, recall_quantile=0.90
            )
            central_slots.append(central)
        central_output.append(
            (f"temporal_central_r{radius}", np.asarray(central_slots, dtype=np.float32))
        )
        for quantile in quantiles:
            coverage_slots = []
            for slot, aligned in enumerate(aligned_by_slot):
                _central, coverage = _temporal_shapes(
                    reference_slots[slot],
                    aligned,
                    recall_quantile=float(quantile),
                )
                coverage_slots.append(coverage)
            recall_output.append(
                (
                    f"temporal_recall_r{radius}_q{int(round(100 * quantile))}",
                    np.asarray(coverage_slots, dtype=np.float32),
                )
            )
    return central_output, recall_output


def _axis_vectors(run, frame_index: int) -> list[tuple[str, np.ndarray]]:
    anchors = np.asarray(run.anchors[int(frame_index)], dtype=np.float64)
    variants = {
        "axis_major": (1.18, 1.04),
        "axis_minor": (1.04, 1.18),
        "axis_balanced": (1.12, 1.12),
    }
    output: list[tuple[str, np.ndarray]] = []
    for name, (scale_x, scale_y) in variants.items():
        slots = []
        for points in anchors:
            center, basis = _principal_basis(points)
            slots.append(
                _axis_scale(points, center, basis, float(scale_x), float(scale_y))
            )
        output.append((name, np.asarray(slots, dtype=np.float32)))
    return output


def install_candidate_generation(module: ModuleType, profile: str) -> ModuleType:
    if profile not in VALID_PROFILES:
        raise ValueError(f"unsupported Phase-2 candidate profile: {profile}")
    original_builder = module.build_frame_candidates
    role_generation_stats: dict[str, dict[str, int]] = {}
    module._phase2_role_generation_stats = role_generation_stats
    if profile in CLASS_ROLE_STATE_PROFILES:
        label = os.environ.get("MASK_PIPELINE_PHASE2_LABEL", "").strip()
        try:
            target_interval = float(
                os.environ.get("MASK_PIPELINE_PHASE2_TARGET_INTERVAL", "5")
            )
            active_role_ids = _class_role_state_profile(profile, label, target_interval)
        except KeyError as exc:
            raise ValueError(
                f"{profile} requires MASK_PIPELINE_PHASE2_LABEL in "
                f"{sorted(CLASS_ROLE_STATE_PROFILES[profile])}; got {label!r}"
            ) from exc
    else:
        active_role_ids = ROLE_STATE_PROFILES.get(profile)
    module._phase2_active_role_ids = active_role_ids
    pipeline_profile: dict[str, float | int] = {}
    module._phase2_pipeline_profile = pipeline_profile

    def add_profile_time(name: str, elapsed: float) -> None:
        pipeline_profile[name] = float(pipeline_profile.get(name, 0.0)) + float(elapsed)

    # Production's optimizer timer includes input preparation, streaming output,
    # exact QA, and artifact emission, while its per-run stage table does not.
    # Keep those phases visible so performance work targets measured costs.  The
    # wrappers are deliberately transparent and do not alter iteration order.
    original_iter_streams = module.iter_track_streams_from_sqlite

    def profiled_iter_track_streams_from_sqlite(*args, **kwargs):
        iterator = iter(original_iter_streams(*args, **kwargs))
        while True:
            started = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                add_profile_time(
                    "prepare_track_streams_seconds", time.perf_counter() - started
                )
                return
            add_profile_time(
                "prepare_track_streams_seconds", time.perf_counter() - started
            )
            pipeline_profile["prepared_track_streams"] = (
                int(pipeline_profile.get("prepared_track_streams", 0)) + 1
            )
            yield value

    module.iter_track_streams_from_sqlite = profiled_iter_track_streams_from_sqlite

    store_class = module.SqliteUnionRowStore
    for method_name, profile_name in (
        ("add_rows", "union_store_add_seconds"),
        ("write_union_json", "write_union_json_seconds"),
        ("write_pred_sqlite", "write_pred_sqlite_seconds"),
        ("evaluate_exact", "evaluate_exact_seconds"),
    ):
        original_method = getattr(store_class, method_name)

        def make_profiled_method(method, timer_name):
            def profiled_method(self, *args, **kwargs):
                started = time.perf_counter()
                result = method(self, *args, **kwargs)
                add_profile_time(timer_name, time.perf_counter() - started)
                return result

            return profiled_method

        setattr(
            store_class,
            method_name,
            make_profiled_method(original_method, profile_name),
        )

    original_compact_json = module.write_compact_json_array

    def profiled_write_compact_json_array(*args, **kwargs):
        started = time.perf_counter()
        result = original_compact_json(*args, **kwargs)
        add_profile_time("write_compact_json_seconds", time.perf_counter() - started)
        return result

    module.write_compact_json_array = profiled_write_compact_json_array

    if os.environ.get("MASK_PIPELINE_PHASE2_DEEP_PROFILE", "").strip() == "1":
        for function_name in (
            "align_contour_slots",
            "align_polygon_phase",
            "resample_closed_contour",
            "build_local_mask_from_polygons",
            "build_track_segments_with_gapfill",
            "split_long_track_segments",
        ):
            original_function = getattr(module, function_name)

            def make_profiled_function(function, timer_name):
                def profiled_function(*args, **kwargs):
                    started = time.perf_counter()
                    result = function(*args, **kwargs)
                    add_profile_time(timer_name, time.perf_counter() - started)
                    pipeline_profile[f"{timer_name}_calls"] = (
                        int(pipeline_profile.get(f"{timer_name}_calls", 0)) + 1
                    )
                    return result

                return profiled_function

            setattr(
                module,
                function_name,
                make_profiled_function(
                    original_function, f"deep_{function_name}_seconds"
                ),
            )

    def make_candidate(
        run,
        frame_index,
        label,
        anchors,
        runtime_args,
        endpoint_values=None,
    ):
        vector = module.flatten_contours(np.asarray(anchors, dtype=np.float32))
        if not np.all(np.isfinite(vector)):
            return None
        polygons = module.split_vector_to_polygons(
            vector, run.contour_count, run.anchors_per_contour
        )
        if any(len(polygon) < 3 for polygon in polygons):
            return None
        endpoint_evaluator = getattr(run, "_phase2_endpoint_evaluator", None)
        if endpoint_values is not None:
            metrics = {
                "recall": float(endpoint_values[4]),
                "iou": float(endpoint_values[6]),
            }
        elif endpoint_evaluator is None:
            metrics = module.compute_exact_metrics_from_polygons(
                run.gt_polygons[int(frame_index)], polygons
            )
        else:
            endpoint_values = endpoint_evaluator.exact_frame_metrics(
                int(frame_index),
                np.asarray(vector, dtype=np.float32),
                int(run.contour_count),
                int(run.anchors_per_contour),
            )
            # Candidate construction consumes only these two exact fields.
            # The native evaluator reuses its parity-aware GT raster cache;
            # all metric arithmetic remains identical to the scalar path.
            metrics = {
                "recall": float(endpoint_values[4]),
                "iou": float(endpoint_values[6]),
            }
        budget = float(module.recall_budget_from_metrics(metrics))
        # An added endpoint that already violates the hard floor cannot occur
        # in any feasible edge.  Do not spend quadratic DP work on it.
        if budget > _EPSILON:
            return None
        frame_loss = float(module.frame_accuracy_loss(metrics, runtime_args))
        area, center, radii, mean_radius = module.vector_proxy_stats(
            vector, run.contour_count, run.anchors_per_contour
        )
        return module.ShapeCandidate(
            label=str(label),
            vector=np.asarray(vector, dtype=np.float32),
            polygons=polygons,
            frame_loss=frame_loss,
            objective=frame_loss,
            recall_budget=budget,
            area=float(area),
            center=np.asarray(center, dtype=np.float32),
            radii=np.asarray(radii, dtype=np.float32),
            mean_radius=float(mean_radius),
        )

    def deduplicate(raw, candidates):
        output = []
        known = [np.asarray(raw.vector, dtype=np.float32)]
        scale = max(float(getattr(raw, "mean_radius", 0.0)), 1.0)
        for candidate in sorted(
            candidates,
            key=lambda value: (
                float(value.frame_loss),
                float(value.area),
                str(value.label),
            ),
        ):
            vector = np.asarray(candidate.vector, dtype=np.float32)
            distance = min(
                float(np.sqrt(np.mean(np.square(vector - previous)))) / scale
                for previous in known
            )
            if distance <= 1e-4:
                continue
            output.append(candidate)
            known.append(vector)
        return output

    def best_family(raw, values):
        valid = [value for value in values if value is not None]
        deduped = deduplicate(raw, valid)
        return deduped[:1]

    def raw_fallback(raw, label):
        """Keep the dense five-state topology when a scale is redundant/invalid."""
        return module.ShapeCandidate(
            label=str(label),
            vector=np.asarray(raw.vector, dtype=np.float32).copy(),
            polygons=[
                np.asarray(value, dtype=np.float32).copy() for value in raw.polygons
            ],
            frame_loss=float(raw.frame_loss),
            objective=float(raw.objective),
            recall_budget=float(raw.recall_budget),
            area=float(raw.area),
            center=np.asarray(raw.center, dtype=np.float32).copy(),
            radii=np.asarray(raw.radii, dtype=np.float32).copy(),
            mean_radius=float(raw.mean_radius),
        )

    def build_frame_candidates(run, contexts, eval_contexts, runtime_args):
        if profile in POLYGON_CONSTRAINED_PROFILES:
            from production.polygon.runtime.spatial_integration import (
                apply_spatial_candidate,
            )

            if not bool(getattr(module, "_phase1_native_interval_enabled", False)):
                raise RuntimeError(
                    "adaptive polygon Recall repair requires native exact interval evaluation"
                )
            run._phase2_endpoint_evaluator = (
                module._phase1_get_native_interval_evaluator(
                    eval_contexts, run.gt_polygons
                )
            )
            apply_spatial_candidate(
                run,
                pipeline_profile,
                endpoint_evaluator=run._phase2_endpoint_evaluator,
                vertices_per_component=(
                    _spatial_vertices_for_track(str(run.track_id))
                    if profile == "polygon_adaptive_keyframe_v2"
                    else 14
                ),
            )
        if (
            profile not in POLYGON_CONSTRAINED_PROFILES
            and os.environ.get(PERSISTENT_LINE_FIT_BASE_ENV, "").strip() == "1"
            and not bool(getattr(run, "_persistent_line_fit_base_applied", False))
        ):
            # Experimental replay only: change the per-frame polygon
            # representation while retaining run.gt_polygons as the exact
            # source-mask reference used by DP and pair-vote.  This lets us
            # rerun the already-frozen new_production optimizer without
            # silently changing its Recall denominator.
            if int(run.contour_count) != 1:
                raise RuntimeError(
                    "persistent-line-fit replay currently requires exactly "
                    f"one contour slot; stream={run.stream_id!r} "
                    f"contours={run.contour_count}"
                )
            target_vertices = int(
                os.environ.get(
                    PERSISTENT_LINE_FIT_VERTICES_ENV,
                    str(run.anchors_per_contour),
                )
            )
            if target_vertices != int(run.anchors_per_contour):
                raise RuntimeError(
                    "persistent-line-fit target must match the prepared "
                    "run anchor count: "
                    f"target={target_vertices} prepared={run.anchors_per_contour}"
                )
            from production.polygon.runtime.spatial_support.quality_repair import (
                persistent_line_fit_quality_guarded,
            )

            base_started = time.perf_counter()
            references = [
                np.asarray(frame_polygons[0], dtype=np.float64)
                for frame_polygons in run.gt_polygons
            ]
            sequence, repair_stats = persistent_line_fit_quality_guarded(
                references,
                target_vertices,
                dense_vertices=64,
                coverage_quantile=0.65,
                maximum_intersection_radius=0.2,
                intersection_regularization=0.01,
            )
            run.anchors = np.ascontiguousarray(
                sequence[:, None, :, :], dtype=np.float32
            )
            run.run_target_total_points = int(target_vertices)
            run._persistent_line_fit_base_applied = True
            pipeline_profile["persistent_line_fit_base_seconds"] = float(
                pipeline_profile.get("persistent_line_fit_base_seconds", 0.0)
            ) + float(time.perf_counter() - base_started)
            pipeline_profile["persistent_line_fit_base_frames"] = int(
                pipeline_profile.get("persistent_line_fit_base_frames", 0)
            ) + int(repair_stats.frames)
            pipeline_profile["persistent_line_fit_repaired_frames"] = int(
                pipeline_profile.get("persistent_line_fit_repaired_frames", 0)
            ) + int(repair_stats.repaired_frames)
            pipeline_profile["persistent_line_fit_fallback_frames"] = int(
                pipeline_profile.get("persistent_line_fit_fallback_frames", 0)
            ) + int(repair_stats.fallback_frames)
        raw_by_frame = original_builder(run, contexts, eval_contexts, runtime_args)
        if profile == "raw_baseline":
            return raw_by_frame

        # Endpoint feasibility used to cross the Python/C++ boundary once for
        # every frame/state pair.  Generate the role geometry in the original
        # deterministic order, then evaluate the independent endpoint masks in
        # one native OpenMP batch.  Candidate order, exact raster arithmetic,
        # and all downstream DP inputs remain unchanged.
        batched_role_anchors = None
        batched_role_metrics = None
        endpoint_evaluator = getattr(run, "_phase2_endpoint_evaluator", None)
        if (
            active_role_ids is not None
            and endpoint_evaluator is not None
            and hasattr(endpoint_evaluator, "exact_frame_metrics_batch")
        ):
            batch_started = time.perf_counter()
            batched_role_anchors = []
            valid_frames = []
            valid_vectors = []
            valid_positions = []
            expected_shape = tuple(np.asarray(run.anchors[0]).shape)

            def generate_role_frame(frame_index):
                frame_values = []
                generation_times = []
                for role_index, role_id in enumerate(active_role_ids):
                    generation_started = time.perf_counter()
                    generated = np.asarray(
                        build_role_candidate(run, frame_index, role_id),
                        dtype=np.float32,
                    )
                    generation_times.append(
                        float(time.perf_counter() - generation_started)
                    )
                    frame_values.append(generated)
                return frame_values, generation_times

            frame_workers = max(
                1, int(os.environ.get(CANDIDATE_FRAME_WORKERS_ENV, "1"))
            )
            if frame_workers == 1 or len(raw_by_frame) <= 1:
                generated_frames = map(generate_role_frame, range(len(raw_by_frame)))
            else:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(frame_workers, len(raw_by_frame))
                )
                generated_frames = executor.map(
                    generate_role_frame, range(len(raw_by_frame))
                )
            try:
                for frame_index, (frame_values, generation_times) in enumerate(
                    generated_frames
                ):
                    for role_index, (role_id, generated) in enumerate(
                        zip(active_role_ids, frame_values)
                    ):
                        stats = role_generation_stats.setdefault(
                            role_id,
                            {
                                "generated": 0,
                                "endpoint_feasible": 0,
                                "active": 0,
                                "fallback": 0,
                                "generation_seconds": 0.0,
                            },
                        )
                        stats["generated"] += 1
                        stats["generation_seconds"] += generation_times[role_index]
                        if generated.shape == expected_shape and np.all(
                            np.isfinite(generated)
                        ):
                            valid_frames.append(int(frame_index))
                            valid_vectors.append(generated.reshape(-1, 2))
                            valid_positions.append((int(frame_index), int(role_index)))
                    batched_role_anchors.append(frame_values)
            finally:
                if frame_workers > 1 and len(raw_by_frame) > 1:
                    executor.shutdown(wait=True)
            pipeline_profile["candidate_frame_workers"] = int(frame_workers)
            batched_role_metrics = [
                [None for _role_id in active_role_ids]
                for _frame_index in range(len(raw_by_frame))
            ]
            if valid_vectors:
                endpoint_started = time.perf_counter()
                native_values = endpoint_evaluator.exact_frame_metrics_batch(
                    np.ascontiguousarray(valid_frames, dtype=np.int32),
                    np.ascontiguousarray(np.stack(valid_vectors), dtype=np.float32),
                    int(run.contour_count),
                    int(run.anchors_per_contour),
                    max(
                        1,
                        int(
                            os.environ.get(
                                "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS", "1"
                            )
                        ),
                    ),
                )
                add_profile_time(
                    "candidate_endpoint_batch_seconds",
                    time.perf_counter() - endpoint_started,
                )
                pipeline_profile["candidate_endpoint_batch_cases"] = int(
                    pipeline_profile.get("candidate_endpoint_batch_cases", 0)
                ) + len(valid_positions)
                for position, values in zip(valid_positions, native_values):
                    frame_index, role_index = position
                    batched_role_metrics[frame_index][role_index] = values
            add_profile_time(
                "candidate_role_batch_seconds",
                time.perf_counter() - batch_started,
            )

        output = []
        for frame_index, raw_values in enumerate(raw_by_frame):
            raw = raw_values[0]
            needed = (
                {
                    "scale_best",
                    "temporal_central_best",
                    "temporal_recall_best",
                    "axis_best",
                }
                if profile == "broad_top2"
                else {profile}
            )
            families: dict[str, list[object]] = {}
            fixed_scales = {
                "scale_104": 1.04,
                "scale_108": 1.08,
                "scale_112": 1.12,
                "scale_116": 1.16,
            }
            if profile in fixed_scales:
                candidate = make_candidate(
                    run,
                    frame_index,
                    profile,
                    _componentwise_scale(
                        run.anchors[frame_index], fixed_scales[profile]
                    ),
                    runtime_args,
                )
                families[profile] = best_family(raw, [candidate])
            scale_pairs = {
                "scale_pair_104_112": (1.04, 1.12),
                "scale_pair_108_112": (1.08, 1.12),
                **SCALE_STATE_PROFILES,
            }
            if profile in scale_pairs:
                candidates = []
                for factor in scale_pairs[profile]:
                    label = f"scale_{factor:.2f}"
                    candidate = make_candidate(
                        run,
                        frame_index,
                        label,
                        _componentwise_scale(run.anchors[frame_index], factor),
                        runtime_args,
                    )
                    if profile in SCALE_STATE_PROFILES:
                        candidates.append(
                            candidate
                            if candidate is not None
                            else raw_fallback(raw, f"{label}_raw_fallback")
                        )
                    elif candidate is not None:
                        candidates.append(candidate)
                families[profile] = (
                    candidates
                    if profile in SCALE_STATE_PROFILES
                    else deduplicate(raw, candidates)
                )
            if active_role_ids is not None:
                candidates = []
                for role_index, role_id in enumerate(active_role_ids):
                    stats = role_generation_stats.setdefault(
                        role_id,
                        {
                            "generated": 0,
                            "endpoint_feasible": 0,
                            "active": 0,
                            "fallback": 0,
                            "generation_seconds": 0.0,
                        },
                    )
                    endpoint_values = None
                    if batched_role_anchors is None:
                        stats["generated"] += 1
                        generation_started = time.perf_counter()
                        generated_anchors = build_role_candidate(
                            run, frame_index, role_id
                        )
                        stats["generation_seconds"] += float(
                            time.perf_counter() - generation_started
                        )
                    else:
                        generated_anchors = batched_role_anchors[frame_index][
                            role_index
                        ]
                        endpoint_values = batched_role_metrics[frame_index][role_index]
                    candidate = make_candidate(
                        run,
                        frame_index,
                        role_id,
                        generated_anchors,
                        runtime_args,
                        endpoint_values,
                    )
                    if candidate is not None:
                        stats["endpoint_feasible"] += 1
                        scale = max(float(getattr(raw, "mean_radius", 0.0)), 1.0)
                        distance = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        np.asarray(candidate.vector, dtype=np.float32)
                                        - np.asarray(raw.vector, dtype=np.float32)
                                    )
                                )
                            )
                            / scale
                        )
                        if distance > 1e-4:
                            stats["active"] += 1
                    else:
                        stats["fallback"] += 1
                    candidates.append(
                        candidate
                        if candidate is not None
                        else raw_fallback(raw, f"{role_id}_raw_fallback")
                    )
                families[profile] = candidates
            if profile in MIXED_STATE_PROFILES:
                candidates = []
                for candidate_id in MIXED_STATE_PROFILES[profile]:
                    if candidate_id.startswith("S"):
                        factor = float(candidate_id[1:]) / 100.0
                        candidate = make_candidate(
                            run,
                            frame_index,
                            candidate_id,
                            _componentwise_scale(run.anchors[frame_index], factor),
                            runtime_args,
                        )
                    else:
                        stats = role_generation_stats.setdefault(
                            candidate_id,
                            {
                                "generated": 0,
                                "endpoint_feasible": 0,
                                "active": 0,
                                "fallback": 0,
                            },
                        )
                        stats["generated"] += 1
                        candidate = make_candidate(
                            run,
                            frame_index,
                            candidate_id,
                            build_role_candidate(run, frame_index, candidate_id),
                            runtime_args,
                        )
                        if candidate is not None:
                            stats["endpoint_feasible"] += 1
                            stats["active"] += 1
                        else:
                            stats["fallback"] += 1
                    candidates.append(
                        candidate
                        if candidate is not None
                        else raw_fallback(raw, f"{candidate_id}_raw_fallback")
                    )
                families[profile] = candidates
            if "scale_best" in needed:
                scale_candidates = [
                    make_candidate(
                        run,
                        frame_index,
                        f"scale_{factor:.2f}",
                        _componentwise_scale(run.anchors[frame_index], factor),
                        runtime_args,
                    )
                    for factor in (1.02, 1.04, 1.06, 1.10, 1.14)
                ]
                families["scale_best"] = best_family(raw, scale_candidates)
            if needed & {"temporal_central_best", "temporal_recall_best"}:
                temporal_central, temporal_recall = _temporal_vectors(
                    run,
                    frame_index,
                    radii=(2, 5, 10),
                    quantiles=(0.90, 0.95, 0.97),
                )
                if "temporal_central_best" in needed:
                    central_candidates = [
                        make_candidate(run, frame_index, label, anchors, runtime_args)
                        for label, anchors in temporal_central
                    ]
                    families["temporal_central_best"] = best_family(
                        raw, central_candidates
                    )
                if "temporal_recall_best" in needed:
                    recall_candidates = [
                        make_candidate(run, frame_index, label, anchors, runtime_args)
                        for label, anchors in temporal_recall
                    ]
                    families["temporal_recall_best"] = best_family(
                        raw, recall_candidates
                    )
            if "axis_best" in needed:
                axis_candidates = [
                    make_candidate(run, frame_index, label, anchors, runtime_args)
                    for label, anchors in _axis_vectors(run, frame_index)
                ]
                families["axis_best"] = best_family(raw, axis_candidates)
            if profile == "broad_top2":
                champions = [value[0] for value in families.values() if value]
                additions = deduplicate(raw, champions)[:2]
            else:
                additions = families[profile]
            output.append([raw, *additions])
        return output

    run_hard_multistate_penalty_path = build_hard_multistate_penalty_path(module)

    module.build_frame_candidates = build_frame_candidates
    if profile in POLYGON_CONSTRAINED_PROFILES:
        from production.polygon.runtime.topology import (
            repair_decoded_path,
        )

        topology_stats: dict[str, float | int] = {
            "dp_selected_edges_checked": 0,
            "dp_invalid_edges": 0,
            "dp_inserted_keys": 0,
            "dp_guard_seconds": 0.0,
            "pair_vote_paths_checked": 0,
            "pair_vote_paths_rejected": 0,
            "pair_vote_local_trials_checked": 0,
            "pair_vote_local_trials_rejected": 0,
            "pair_vote_guard_seconds": 0.0,
        }
        module._polygon14_topology_guard_stats = topology_stats

        def topology_guarded_penalty_path(
            run,
            candidate_frames,
            candidates_by_frame,
            target_count,
            runtime_args,
            eval_contexts=None,
        ):
            result = run_hard_multistate_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                runtime_args,
                eval_contexts=eval_contexts,
            )
            frames, states = repair_decoded_path(
                module,
                run,
                result[0],
                result[1],
                candidates_by_frame,
                runtime_args,
                eval_contexts,
                topology_stats,
            )
            return frames, states, result[2], result[3], result[4]

        module.run_multistate_penalty_path = topology_guarded_penalty_path
    else:
        module.run_multistate_penalty_path = run_hard_multistate_penalty_path
    module._phase2_candidate_profile = profile
    module._phase2_candidate_patched = True
    return module
