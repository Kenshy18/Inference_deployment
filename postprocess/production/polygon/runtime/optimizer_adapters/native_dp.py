"""Native penalty solver and exact-Recall repair adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType

from ..kernel import solver as kernel_solver


def install_native_dp_adapters(
    module: ModuleType,
    original_run_single_state_penalty_path,
    original_repair_keyframe_vectors_for_exact_recall,
) -> None:
    def ensure_native_polygon_dp_lib():
        if bool(getattr(module, "_native_polygon_dp_unavailable", False)):
            return None
        loaded = getattr(module, "_native_polygon_dp_lib", None)
        if loaded is not None:
            return loaded

        if str(
            __import__("os").environ.get("ATOSYORI_POLYGON_DISABLE_NATIVE_DP", "")
        ).strip():
            module._native_polygon_dp_unavailable = True
            return None

        native_source_path = Path(__file__).with_name("native_dp_kernel.cpp")
        native_source = native_source_path.read_text(encoding="utf-8")
        os_mod = __import__("os")
        hashlib_mod = __import__("hashlib")
        ctypes_mod = __import__("ctypes")
        digest = hashlib_mod.sha256(native_source.encode("utf-8")).hexdigest()[:16]
        build_dir = module.Path(
            os_mod.environ.get(
                "ATOSYORI_POLYGON_NATIVE_DIR", "/tmp/atosyori_polygon_native"
            )
        )
        source_path = build_dir / f"polygon_dp_{digest}.cpp"
        lib_path = build_dir / f"polygon_dp_{digest}.so"
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
            if not lib_path.exists():
                source_path.write_text(native_source, encoding="utf-8")
                tmp_lib_path = (
                    build_dir / f"polygon_dp_{digest}.{os_mod.getpid()}.tmp.so"
                )
                compiler = os_mod.environ.get("CXX")
                if not compiler:
                    # The packaged runtime invokes its Python executable
                    # directly instead of activating the environment, so its
                    # bundled compiler is not necessarily on PATH.  Resolve it
                    # beside the active interpreter before falling back to the
                    # host toolchain.
                    python_bin = module.Path(__import__("sys").executable).parent
                    bundled_compiler = python_bin / "g++"
                    compiler = (
                        str(bundled_compiler)
                        if bundled_compiler.is_file()
                        else (__import__("shutil").which("g++") or "g++")
                    )
                subprocess.run(
                    [
                        compiler,
                        "-O3",
                        "-std=c++17",
                        "-shared",
                        "-fPIC",
                        str(source_path),
                        "-o",
                        str(tmp_lib_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os_mod.replace(str(tmp_lib_path), str(lib_path))
            lib = ctypes_mod.CDLL(str(lib_path))
            fn = lib.polygon_single_state_decode
            fn.argtypes = [
                ctypes_mod.c_int,
                ctypes_mod.c_int,
                ctypes_mod.c_int,
                ctypes_mod.c_int,
                ctypes_mod.c_int,
                ctypes_mod.c_double,
                ctypes_mod.c_double,
                ctypes_mod.c_double,
                ctypes_mod.c_double,
                ctypes_mod.POINTER(ctypes_mod.c_double),
                ctypes_mod.POINTER(ctypes_mod.c_double),
                ctypes_mod.POINTER(ctypes_mod.c_int32),
                ctypes_mod.POINTER(ctypes_mod.c_int64),
                ctypes_mod.c_double,
                ctypes_mod.c_double,
                ctypes_mod.POINTER(ctypes_mod.c_int32),
                ctypes_mod.POINTER(ctypes_mod.c_int),
                ctypes_mod.POINTER(ctypes_mod.c_double),
            ]
            fn.restype = ctypes_mod.c_int
            repair_fn = lib.polygon_repair_key_scores
            repair_fn.argtypes = [
                ctypes_mod.c_int,
                ctypes_mod.c_int,
                ctypes_mod.POINTER(ctypes_mod.c_int32),
                ctypes_mod.POINTER(ctypes_mod.c_double),
                ctypes_mod.POINTER(ctypes_mod.c_double),
            ]
            repair_fn.restype = ctypes_mod.c_int
            module._native_polygon_dp_lib = lib
            module._native_polygon_dp_fn = fn
            module._native_polygon_repair_key_scores_fn = repair_fn
            return lib
        except Exception as exc:
            module._native_polygon_dp_unavailable = True
            if not bool(getattr(module, "_native_polygon_dp_warning_printed", False)):
                compiler_text = os_mod.environ.get("CXX") or str(
                    module.Path(__import__("sys").executable).parent / "g++"
                )
                print(
                    f"[polygon-optimize-warning] native DP unavailable with compiler={compiler_text!r}; "
                    f"using Python DP ({exc})",
                    flush=True,
                )
                module._native_polygon_dp_warning_printed = True
            return None

    def native_single_state_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        args,
        eval_contexts=None,
    ):
        node_count = int(len(candidate_frames))
        if ensure_native_polygon_dp_lib() is None:
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )
        if node_count <= 0:
            return original_run_single_state_penalty_path(
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
            int(
                module.math.ceil(
                    float(args.dynamic_max_gap_factor) * float(target_interval)
                )
            ),
        )
        candidate_frames_i = [int(v) for v in candidate_frames]
        pred_start = module.np.zeros((node_count,), dtype=module.np.int32)
        edge_offsets = module.np.zeros((node_count + 1,), dtype=module.np.int64)
        for node_pos in range(1, node_count):
            end_frame = int(candidate_frames_i[node_pos])
            min_prev_pos = int(
                module.bisect.bisect_left(
                    candidate_frames_i,
                    end_frame - int(dynamic_max_gap),
                    0,
                    node_pos,
                )
            )
            pred_start[node_pos] = int(min_prev_pos)
            edge_offsets[node_pos + 1] = int(
                edge_offsets[node_pos] + node_pos - min_prev_pos
            )
        compact_edge_count = int(edge_offsets[-1])
        compact_bytes = int(compact_edge_count) * 16
        os_mod = __import__("os")
        try:
            compact_limit = int(
                os_mod.environ.get(
                    "ATOSYORI_POLYGON_NATIVE_EDGE_LIMIT_BYTES",
                    os_mod.environ.get(
                        "ATOSYORI_POLYGON_NATIVE_DENSE_LIMIT_BYTES",
                        str(512 * 1024 * 1024),
                    ),
                )
            )
        except ValueError:
            compact_limit = 512 * 1024 * 1024
        if compact_bytes > max(1, int(compact_limit)):
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )
        edge_costs = module.np.full(
            (compact_edge_count,), module.np.inf, dtype=module.np.float64
        )
        edge_budgets = module.np.full(
            (compact_edge_count,), module.np.inf, dtype=module.np.float64
        )
        counters = {"interval_evals": 0, "interval_frames": 0}
        reachable = [False] * node_count
        reachable[0] = True

        for node_pos in range(1, node_count):
            end_frame = int(candidate_frames_i[node_pos])
            min_prev_pos = int(pred_start[node_pos])
            node_reachable = False
            end_candidate = candidates_by_frame[end_frame][0]
            for prev_node_pos in range(min_prev_pos, node_pos):
                if not reachable[prev_node_pos]:
                    continue
                start_frame = int(candidate_frames_i[prev_node_pos])
                start_candidate = candidates_by_frame[start_frame][0]
                info = module.interval_cost_from_vectors(
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
                edge_idx = int(
                    edge_offsets[node_pos] + prev_node_pos - min_prev_pos
                )
                edge_costs[edge_idx] = float(info.cost)
                edge_budgets[edge_idx] = float(info.recall_budget)
                counters["interval_evals"] += 1
                counters["interval_frames"] += int(info.frames_covered)
                if module.np.isfinite(float(info.cost)):
                    node_reachable = True
            reachable[node_pos] = bool(node_reachable)

        if not reachable[-1]:
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )

        ctypes_mod = __import__("ctypes")
        fn = getattr(module, "_native_polygon_dp_fn", None)
        if fn is None:
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )
        first_candidate = candidates_by_frame[int(candidate_frames_i[0])][0]
        out_path = module.np.empty((node_count,), dtype=module.np.int32)
        out_count = ctypes_mod.c_int(0)
        out_lambda = ctypes_mod.c_double(0.0)
        edge_costs = module.np.ascontiguousarray(edge_costs, dtype=module.np.float64)
        edge_budgets = module.np.ascontiguousarray(
            edge_budgets, dtype=module.np.float64
        )
        pred_start = module.np.ascontiguousarray(pred_start, dtype=module.np.int32)
        edge_offsets = module.np.ascontiguousarray(
            edge_offsets, dtype=module.np.int64
        )
        status = int(
            fn(
                ctypes_mod.c_int(node_count),
                ctypes_mod.c_int(max(2, min(int(target_count), node_count))),
                ctypes_mod.c_int(max(1, int(args.penalty_binary_steps))),
                ctypes_mod.c_int(max(1, int(args.recall_budget_binary_steps))),
                ctypes_mod.c_int(
                    1 if str(args.recall_constraint_mode) == "exact_dp" else 0
                ),
                ctypes_mod.c_double(float(args.penalty_max)),
                ctypes_mod.c_double(float(max(args.recall_budget_max_mu, 1e-6))),
                ctypes_mod.c_double(
                    float(module.recall_budget_limit(len(run.frame_numbers), args))
                ),
                ctypes_mod.c_double(float(args.proxy_recall_penalty_weight)),
                edge_costs.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_double)),
                edge_budgets.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_double)),
                pred_start.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_int32)),
                edge_offsets.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_int64)),
                ctypes_mod.c_double(float(first_candidate.frame_loss)),
                ctypes_mod.c_double(float(first_candidate.recall_budget)),
                out_path.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_int32)),
                ctypes_mod.byref(out_count),
                ctypes_mod.byref(out_lambda),
            )
        )
        if status != 0 or int(out_count.value) <= 0:
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )
        chosen_node_positions = [
            int(v) for v in out_path[: int(out_count.value)].tolist()
        ]
        chosen_frames = [int(candidate_frames_i[pos]) for pos in chosen_node_positions]
        return (
            chosen_frames,
            [0] * len(chosen_frames),
            counters,
            {},
            float(out_lambda.value),
        )

    def native_repair_key_scores(chosen_frames, frame_deficits):
        if ensure_native_polygon_dp_lib() is None:
            return None
        fn = getattr(module, "_native_polygon_repair_key_scores_fn", None)
        if fn is None:
            return None
        ctypes_mod = __import__("ctypes")
        chosen_arr = module.np.ascontiguousarray(
            [int(v) for v in chosen_frames], dtype=module.np.int32
        )
        deficits_arr = module.np.ascontiguousarray(
            frame_deficits, dtype=module.np.float64
        )
        out = module.np.zeros((len(chosen_arr),), dtype=module.np.float64)
        status = int(
            fn(
                ctypes_mod.c_int(int(len(deficits_arr))),
                ctypes_mod.c_int(int(len(chosen_arr))),
                chosen_arr.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_int32)),
                deficits_arr.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_double)),
                out.ctypes.data_as(ctypes_mod.POINTER(ctypes_mod.c_double)),
            )
        )
        if status != 0:
            return None
        return out

    def aggregate_interpolated_metric_rows(metric_rows):
        total_iou_loss = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_gt_area = 0.0
        total_intersection = 0.0
        for metrics in metric_rows:
            total_iou_loss += 1.0 - float(metrics["iou"])
            total_recall += float(metrics["recall"])
            total_precision += float(metrics["precision"])
            total_gt_area += float(metrics["gt_area"])
            total_intersection += float(metrics["intersection"])
        mean_iou = float(1.0 - total_iou_loss / max(len(metric_rows), 1))
        mean_recall = float(total_recall / max(len(metric_rows), 1))
        mean_precision = float(total_precision / max(len(metric_rows), 1))
        global_recall = (
            float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0
        )
        return (
            metric_rows,
            float(total_iou_loss),
            float(mean_iou),
            float(mean_recall),
            float(mean_precision),
            float(global_recall),
        )

    def changed_repair_frame_indices(run, chosen_frames, old_vectors, new_vectors):
        old_arr = module.np.asarray(old_vectors, dtype=module.np.float32)
        new_arr = module.np.asarray(new_vectors, dtype=module.np.float32)
        if old_arr.shape != new_arr.shape:
            return list(range(len(run.frame_numbers)))
        flat_delta = module.np.reshape(
            module.np.abs(new_arr - old_arr), (len(chosen_frames), -1)
        )
        changed_keys = module.np.flatnonzero(module.np.any(flat_delta > 0.0, axis=1))
        if len(changed_keys) <= 0:
            return []
        length = int(len(run.frame_numbers))
        chosen = [int(v) for v in chosen_frames]
        affected = set()
        key_count = len(chosen)
        for key_idx_raw in changed_keys.tolist():
            key_idx = int(key_idx_raw)
            if key_count <= 1:
                start = 0
                end = length - 1
            elif key_idx == 0:
                start = 0
                end = min(length - 1, int(chosen[1]) - 1)
            elif key_idx == key_count - 1:
                start = max(0, int(chosen[key_idx - 1]) + 1)
                end = length - 1
            else:
                start = max(0, int(chosen[key_idx - 1]) + 1)
                end = min(length - 1, int(chosen[key_idx + 1]) - 1)
            if end >= start:
                affected.update(range(start, end + 1))
        return sorted(int(v) for v in affected)

    def exact_interpolated_metrics_delta(
        run, chosen_frames, old_vectors, new_vectors, base_metrics_rows
    ):
        affected_frames = changed_repair_frame_indices(
            run, chosen_frames, old_vectors, new_vectors
        )
        if not affected_frames:
            return aggregate_interpolated_metric_rows(list(base_metrics_rows))
        if len(affected_frames) >= int(len(run.frame_numbers)):
            return module.exact_interpolated_metrics(run, chosen_frames, new_vectors)

        chosen_frames_arr = [int(v) for v in chosen_frames]
        trial_metrics = list(base_metrics_rows)
        for frame_idx in affected_frames:
            if frame_idx <= chosen_frames_arr[0]:
                vec = module.np.asarray(new_vectors[0], dtype=module.np.float32)
            elif frame_idx >= chosen_frames_arr[-1]:
                vec = module.np.asarray(new_vectors[-1], dtype=module.np.float32)
            else:
                right_pos = int(
                    module.bisect.bisect_left(chosen_frames_arr, int(frame_idx))
                )
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen_frames_arr[left_pos])
                right_frame = int(chosen_frames_arr[right_pos])
                if frame_idx == right_frame:
                    vec = module.np.asarray(
                        new_vectors[right_pos], dtype=module.np.float32
                    )
                else:
                    alpha = float(
                        (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                    )
                    vec = module.interpolate_vectors(
                        new_vectors[left_pos], new_vectors[right_pos], alpha
                    )
            pred_polys = module.split_vector_to_polygons(
                vec, run.contour_count, run.anchors_per_contour
            )
            trial_metrics[int(frame_idx)] = module.compute_exact_metrics_from_polygons(
                run.gt_polygons[int(frame_idx)], pred_polys
            )
        return aggregate_interpolated_metric_rows(trial_metrics)

    def repair_keyframe_vectors_for_exact_recall_native_key_scores(
        run,
        chosen_frames,
        keyframe_vectors,
        candidates_by_frame,
        args,
    ):
        if ensure_native_polygon_dp_lib() is None:
            return original_repair_keyframe_vectors_for_exact_recall(
                run,
                chosen_frames,
                keyframe_vectors,
                candidates_by_frame,
                args,
            )
        if not bool(args.exact_recall_repair_enabled) or len(chosen_frames) <= 0:
            return module.np.asarray(keyframe_vectors, dtype=module.np.float32)
        current = module.np.asarray(keyframe_vectors, dtype=module.np.float32).copy()
        os_mod = __import__("os")
        disable_repair_delta = bool(
            str(os_mod.environ.get("ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA", "")).strip()
        )
        use_repair_delta = not disable_repair_delta
        scale_deltas = module.parse_float_list(
            str(args.exact_recall_repair_scale_deltas), [0.01, 0.02, 0.04, 0.06, 0.08]
        )
        (
            metrics_rows,
            current_iou_loss,
            _current_mean_iou,
            current_mean_recall,
            _current_mean_precision,
            _current_global_recall,
        ) = module.exact_interpolated_metrics(run, chosen_frames, current)
        best_key = module.exact_recall_solution_key(
            current_iou_loss, current_mean_recall, args
        )
        if best_key[0] <= 0.0:
            return current

        for _pass in range(max(1, int(args.exact_recall_repair_max_passes))):
            frame_deficits = module.np.asarray(
                [
                    float(row["gt_area"])
                    * max(float(args.recall_min) - float(row["recall"]), 0.0)
                    for row in metrics_rows
                ],
                dtype=module.np.float64,
            )
            if float(module.np.mean(frame_deficits)) <= 0.0 and best_key[0] <= 0.0:
                break
            key_scores = native_repair_key_scores(chosen_frames, frame_deficits)
            if key_scores is None:
                key_scores = module.np.zeros(
                    (len(chosen_frames),), dtype=module.np.float64
                )
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
                    alpha = float(
                        (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                    )
                    key_scores[left_pos] += (1.0 - alpha) * float(deficit)
                    key_scores[right_pos] += alpha * float(deficit)
            key_order = [
                int(idx)
                for idx in module.np.argsort(-key_scores)[
                    : max(1, int(args.exact_recall_repair_topk))
                ].tolist()
            ]
            improved = False

            trial_vectors = []
            for delta in scale_deltas:
                scaled_all = module.np.asarray(current, dtype=module.np.float32).copy()
                for key_idx in range(len(chosen_frames)):
                    scaled_all[key_idx] = module.scale_vector_about_centroid(
                        scaled_all[key_idx], 1.0 + float(delta)
                    )
                trial_vectors.append(scaled_all)
            for delta in scale_deltas:
                scaled = module.np.asarray(current, dtype=module.np.float32).copy()
                for key_idx in key_order:
                    scaled[key_idx] = module.scale_vector_about_centroid(
                        scaled[key_idx], 1.0 + float(delta)
                    )
                trial_vectors.append(scaled)

            for key_idx in key_order:
                frame_idx = int(chosen_frames[key_idx])
                current_area, _center, _radii, _mean_radius = module.vector_proxy_stats(
                    current[key_idx], run.contour_count, run.anchors_per_contour
                )
                for candidate in candidates_by_frame[frame_idx]:
                    if float(candidate.area) <= float(current_area) + 1e-3:
                        continue
                    upgraded = module.np.asarray(
                        current, dtype=module.np.float32
                    ).copy()
                    upgraded[key_idx] = module.np.asarray(
                        candidate.vector, dtype=module.np.float32
                    )
                    trial_vectors.append(upgraded)
                for delta in scale_deltas:
                    upgraded = module.np.asarray(
                        current, dtype=module.np.float32
                    ).copy()
                    upgraded[key_idx] = module.scale_vector_about_centroid(
                        upgraded[key_idx], 1.0 + float(delta)
                    )
                    trial_vectors.append(upgraded)

            seen = []
            for trial in trial_vectors:
                if any(
                    module.np.allclose(trial, existing, atol=1e-4) for existing in seen
                ):
                    continue
                seen.append(module.np.asarray(trial, dtype=module.np.float32))
                if use_repair_delta:
                    (
                        trial_metrics,
                        trial_iou_loss,
                        _trial_mean_iou,
                        trial_mean_recall,
                        _trial_mean_precision,
                        _trial_global_recall,
                    ) = exact_interpolated_metrics_delta(
                        run,
                        chosen_frames,
                        current,
                        trial,
                        metrics_rows,
                    )
                else:
                    (
                        trial_metrics,
                        trial_iou_loss,
                        _trial_mean_iou,
                        trial_mean_recall,
                        _trial_mean_precision,
                        _trial_global_recall,
                    ) = module.exact_interpolated_metrics(run, chosen_frames, trial)
                trial_key = module.exact_recall_solution_key(
                    trial_iou_loss, trial_mean_recall, args
                )
                if trial_key < best_key:
                    current = module.np.asarray(trial, dtype=module.np.float32)
                    metrics_rows = trial_metrics
                    best_key = trial_key
                    improved = True
            if not improved:
                break
            if best_key[0] <= 0.0:
                break
        return module.np.asarray(current, dtype=module.np.float32)

    module.run_single_state_penalty_path = native_single_state_penalty_path
    # ``run_multistate_penalty_path`` is defined in ``kernel.solver`` and its
    # single-state fast branch resolves this module global, not the symbol
    # re-exported by optimizer_kernel.  Update both owners so Production
    # actually reaches the native implementation.
    kernel_solver.run_single_state_penalty_path = native_single_state_penalty_path
    module.repair_keyframe_vectors_for_exact_recall = (
        repair_keyframe_vectors_for_exact_recall_native_key_scores
    )


__all__ = ("install_native_dp_adapters",)
