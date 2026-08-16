#!/usr/bin/env python3
"""Production adapter for the isolated polygon optimizer kernel."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from types import ModuleType

POSTPROCESS_ROOT = Path(__file__).resolve().parents[3]


def build_optimizer_module() -> ModuleType:
    """Load and configure the isolated Production numerical kernel once."""
    module = importlib.import_module("production.polygon.runtime.optimizer_kernel")
    if bool(getattr(module, "_production_runtime_defaults_applied", False)):
        return module
    try:
        module.cv2.setNumThreads(1)
    except Exception:
        pass
    # Full-length real data can contain degenerate polygon rows. Keep the
    # fixed-dimension polygon optimizer from failing by guaranteeing that
    # contour resampling/phase alignment returns a homogeneous point count.
    original_resample_closed_contour = module.resample_closed_contour
    original_json_dumps = module.json.dumps
    original_get_context = module.multiprocessing.get_context
    original_apply_fixed_practical_defaults = module.apply_fixed_practical_defaults
    original_build_track_streams = module.build_track_streams
    original_load_rows = module.load_rows
    original_run_single_state_penalty_path = module.run_single_state_penalty_path
    original_repair_keyframe_vectors_for_exact_recall = (
        module.repair_keyframe_vectors_for_exact_recall
    )

    def resolve_point_predictor_model_dir():
        candidate = POSTPROCESS_ROOT / "models" / "polygon_point_predictor"
        if not (candidate / "best.pt").is_file():
            raise FileNotFoundError(
                f"Production polygon point predictor is missing: {candidate}"
            )
        return candidate

    module.DEFAULT_ADAPTIVE_ANCHOR_COUNTS = True
    module.DEFAULT_ADAPTIVE_POINT_OFFSET = 10
    module.DEFAULT_MIN_ANCHORS_PER_CONTOUR = 8
    module.DEFAULT_POINT_PREDICTOR_MODEL_DIR = resolve_point_predictor_model_dir()
    module.DEFAULT_PREDICTOR_DEVICE = "cuda"

    def safe_resample_closed_contour(poly, n_points):
        target = max(3, int(n_points))
        out = module.np.asarray(
            original_resample_closed_contour(poly, target), dtype=module.np.float32
        ).reshape(-1, 2)
        if len(out) == target:
            return out
        if len(out) == 0:
            return module.np.zeros((target, 2), dtype=module.np.float32)
        return module.np.repeat(out[:1], target, axis=0).astype(module.np.float32)

    def fast_align_polygon_phase(reference, poly):
        candidate = module.np.asarray(
            module.orient_ccw(poly), dtype=module.np.float32
        ).reshape(-1, 2)
        if reference is None:
            return candidate
        ref = module.np.asarray(reference, dtype=module.np.float32).reshape(-1, 2)
        if len(ref) != len(candidate):
            return safe_resample_closed_contour(candidate, len(ref))
        count = int(len(candidate))
        if count <= 1:
            return candidate

        shift_ids = module.np.arange(count, dtype=module.np.int32)
        gather = (shift_ids[None, :] + shift_ids[:, None]) % count

        def best_roll(variant):
            rolled = module.np.asarray(variant, dtype=module.np.float32)[gather]
            diff = rolled - ref[None, :, :]
            scores = module.np.mean(module.np.sum(diff * diff, axis=2), axis=1)
            best_idx = int(module.np.argmin(scores))
            return (
                float(scores[best_idx]),
                module.np.asarray(rolled[best_idx], dtype=module.np.float32).copy(),
            )

        best_score, best = best_roll(candidate)
        reverse_score, reverse_best = best_roll(candidate[::-1].copy())
        if reverse_score < best_score:
            return reverse_best
        return best

    def fast_exact_k_dp(cost_fn, nodes, target_count, max_gap):
        node_count = len(nodes)
        target_count = max(2, min(int(target_count), node_count))
        if node_count <= 0:
            return []
        if target_count <= 1:
            return [int(nodes[0])]

        nodes_i = module.np.asarray([int(v) for v in nodes], dtype=module.np.int32)
        min_prev_positions = module.np.zeros((node_count,), dtype=module.np.int32)
        max_gap_i = int(max_gap)
        max_width = 0
        for end_pos in range(1, node_count):
            end_node = int(nodes_i[end_pos])
            min_prev_pos = int(
                module.bisect.bisect_left(nodes, end_node - max_gap_i, 0, end_pos)
            )
            min_prev_positions[end_pos] = min_prev_pos
            max_width = max(max_width, int(end_pos - min_prev_pos))

        edge_costs = module.np.full(
            (node_count, max(1, int(max_width))), module.np.inf, dtype=module.np.float64
        )
        for end_pos in range(1, node_count):
            end_node = int(nodes_i[end_pos])
            min_prev_pos = int(min_prev_positions[end_pos])
            width = int(end_pos - min_prev_pos)
            if width <= 0:
                continue
            edge_costs[end_pos, :width] = module.np.asarray(
                [
                    float(cost_fn(int(nodes_i[prev_pos]), end_node))
                    for prev_pos in range(min_prev_pos, end_pos)
                ],
                dtype=module.np.float64,
            )
        try:
            for closure_cell in getattr(cost_fn, "__closure__", None) or ():
                cell_value = closure_cell.cell_contents
                if isinstance(cell_value, dict):
                    cell_value.clear()
        except Exception:
            pass

        import tempfile as tempfile_mod

        back_dtype = (
            module.np.uint16
            if max_gap_i < int(module.np.iinfo(module.np.uint16).max)
            else module.np.int32
        )
        back_itemsize = int(module.np.dtype(back_dtype).itemsize)
        row_bytes = int(node_count * back_itemsize)
        prev_dp = module.np.full((node_count,), module.np.inf, dtype=module.np.float64)
        prev_dp[0] = 0.0

        with tempfile_mod.TemporaryFile() as back_file:
            back_file.truncate(int(target_count) * row_bytes)
            for used in range(1, target_count):
                curr_dp = module.np.full(
                    (node_count,), module.np.inf, dtype=module.np.float64
                )
                back_offsets = module.np.zeros((node_count,), dtype=back_dtype)
                for end_pos in range(used, node_count):
                    min_prev_pos = max(used - 1, int(min_prev_positions[end_pos]))
                    if min_prev_pos >= end_pos:
                        continue
                    edge_offset = int(min_prev_pos - int(min_prev_positions[end_pos]))
                    edge_values = edge_costs[
                        end_pos, edge_offset : edge_offset + int(end_pos - min_prev_pos)
                    ]
                    values = prev_dp[min_prev_pos:end_pos] + edge_values
                    best_rel = int(module.np.argmin(values))
                    best_cost = float(values[best_rel])
                    if not module.np.isfinite(best_cost):
                        continue
                    best_prev = int(min_prev_pos + best_rel)
                    curr_dp[end_pos] = best_cost
                    back_offsets[end_pos] = int(end_pos - best_prev)
                back_file.seek(int(used) * row_bytes)
                back_file.write(back_offsets.tobytes(order="C"))
                prev_dp = curr_dp

            path = [node_count - 1]
            cur_pos = node_count - 1
            cur_used = target_count - 1
            while cur_used > 0:
                back_file.seek(int(cur_used) * row_bytes + int(cur_pos) * back_itemsize)
                raw = back_file.read(back_itemsize)
                if len(raw) != back_itemsize:
                    return [int(nodes[0]), int(nodes[-1])]
                offset = int(module.np.frombuffer(raw, dtype=back_dtype, count=1)[0])
                if offset <= 0:
                    return [int(nodes[0]), int(nodes[-1])]
                cur_pos = int(cur_pos - offset)
                if cur_pos < 0:
                    return [int(nodes[0]), int(nodes[-1])]
                path.append(cur_pos)
                cur_used -= 1
            path.reverse()
            return [int(nodes[pos]) for pos in path]

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

        native_source = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct DecodeResult {
    std::vector<int> path;
    double raw = std::numeric_limits<double>::infinity();
    double budget = std::numeric_limits<double>::infinity();
    double lambda = 0.0;
    bool ok = false;
};

static DecodeResult decode_once(
    int node_count,
    double lambda_penalty,
    double recall_mu,
    bool use_exact_recall_dp,
    double recall_penalty_weight,
    const double* edge_costs,
    const double* edge_budgets,
    const int32_t* pred_start,
    double first_loss,
    double first_budget
) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dp(node_count, inf);
    std::vector<double> raw_cost(node_count, inf);
    std::vector<double> raw_budget(node_count, inf);
    std::vector<int32_t> back(node_count, -1);

    const double first_penalty = (use_exact_recall_dp ? recall_mu : recall_penalty_weight) * first_budget;
    dp[0] = first_loss + first_penalty + lambda_penalty;
    raw_cost[0] = first_loss;
    raw_budget[0] = first_budget;

    for (int node_pos = 1; node_pos < node_count; ++node_pos) {
        double best_cost = inf;
        double best_raw = inf;
        double best_budget = inf;
        int best_prev = -1;
        const int begin = std::max(0, static_cast<int>(pred_start[node_pos]));
        for (int prev_node_pos = begin; prev_node_pos < node_pos; ++prev_node_pos) {
            const double prev_cost = dp[prev_node_pos];
            if (!std::isfinite(prev_cost)) {
                continue;
            }
            const int edge_idx = prev_node_pos * node_count + node_pos;
            const double edge_cost = edge_costs[edge_idx];
            if (!std::isfinite(edge_cost)) {
                continue;
            }
            const double edge_budget = edge_budgets[edge_idx];
            const double penalty = (use_exact_recall_dp ? recall_mu : recall_penalty_weight) * edge_budget;
            const double cand_cost = prev_cost + edge_cost + penalty + lambda_penalty;
            const double cand_raw = raw_cost[prev_node_pos] + edge_cost;
            const double cand_budget = raw_budget[prev_node_pos] + edge_budget;
            if (
                cand_cost < best_cost ||
                (
                    std::fabs(cand_cost - best_cost) <= 1e-9 &&
                    (
                        cand_budget < best_budget ||
                        (
                            std::fabs(cand_budget - best_budget) <= 1e-9 &&
                            cand_raw < best_raw
                        )
                    )
                )
            ) {
                best_cost = cand_cost;
                best_raw = cand_raw;
                best_budget = cand_budget;
                best_prev = prev_node_pos;
            }
        }
        dp[node_pos] = best_cost;
        raw_cost[node_pos] = best_raw;
        raw_budget[node_pos] = best_budget;
        back[node_pos] = static_cast<int32_t>(best_prev);
    }

    DecodeResult result;
    const int last_pos = node_count - 1;
    if (last_pos < 0 || !std::isfinite(dp[last_pos])) {
        return result;
    }
    std::vector<int> reversed;
    int cur_pos = last_pos;
    while (cur_pos >= 0) {
        reversed.push_back(cur_pos);
        cur_pos = static_cast<int>(back[cur_pos]);
    }
    result.path.assign(reversed.rbegin(), reversed.rend());
    result.raw = raw_cost[last_pos];
    result.budget = raw_budget[last_pos];
    result.ok = true;
    return result;
}

static DecodeResult decode_for_recall_mu(
    int node_count,
    int target_count,
    int penalty_steps,
    double penalty_max,
    double recall_mu,
    bool use_exact_recall_dp,
    double recall_penalty_weight,
    const double* edge_costs,
    const double* edge_budgets,
    const int32_t* pred_start,
    double first_loss,
    double first_budget
) {
    DecodeResult best;
    double lo = 0.0;
    double hi = penalty_max;
    const int steps = std::max(1, penalty_steps);
    for (int step = 0; step < steps; ++step) {
        const double mid = 0.5 * (lo + hi);
        DecodeResult cand = decode_once(
            node_count,
            mid,
            recall_mu,
            use_exact_recall_dp,
            recall_penalty_weight,
            edge_costs,
            edge_budgets,
            pred_start,
            first_loss,
            first_budget
        );
        if (!cand.ok) {
            hi = mid;
            continue;
        }
        cand.lambda = hi;
        if (!best.ok) {
            best = cand;
        } else {
            const int cand_gap = std::abs(static_cast<int>(cand.path.size()) - target_count);
            const int best_gap = std::abs(static_cast<int>(best.path.size()) - target_count);
            if (
                cand_gap < best_gap ||
                (
                    cand_gap == best_gap &&
                    (
                        cand.path.size() < best.path.size() ||
                        (
                            cand.path.size() == best.path.size() &&
                            (
                                cand.budget < best.budget ||
                                (
                                    std::fabs(cand.budget - best.budget) <= 1e-9 &&
                                    cand.raw < best.raw
                                )
                            )
                        )
                    )
                )
            ) {
                best = cand;
            }
        }
        if (static_cast<int>(cand.path.size()) > target_count) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (best.ok) {
        best.lambda = hi;
    }
    return best;
}

}  // namespace

extern "C" int polygon_single_state_decode(
    int node_count,
    int target_count,
    int penalty_steps,
    int recall_steps,
    int use_exact_recall_dp,
    double penalty_max,
    double recall_budget_max_mu,
    double recall_budget_limit,
    double recall_penalty_weight,
    const double* edge_costs,
    const double* edge_budgets,
    const int32_t* pred_start,
    double first_loss,
    double first_budget,
    int32_t* out_path,
    int* out_count,
    double* out_lambda
) {
    if (
        node_count <= 0 ||
        target_count <= 0 ||
        edge_costs == nullptr ||
        edge_budgets == nullptr ||
        pred_start == nullptr ||
        out_path == nullptr ||
        out_count == nullptr ||
        out_lambda == nullptr
    ) {
        return -1;
    }

    target_count = std::max(2, std::min(target_count, node_count));
    const bool exact_dp = use_exact_recall_dp != 0;
    DecodeResult best_result;
    if (exact_dp) {
        double recall_lo = 0.0;
        double recall_hi = std::max(recall_budget_max_mu, 1e-6);
        const int steps = std::max(1, recall_steps);
        for (int step = 0; step < steps; ++step) {
            const double recall_mid = 0.5 * (recall_lo + recall_hi);
            DecodeResult cand = decode_for_recall_mu(
                node_count,
                target_count,
                penalty_steps,
                penalty_max,
                recall_mid,
                exact_dp,
                recall_penalty_weight,
                edge_costs,
                edge_budgets,
                pred_start,
                first_loss,
                first_budget
            );
            if (!cand.ok) {
                recall_lo = recall_mid;
                continue;
            }
            const double cand_violation = std::max(cand.budget - recall_budget_limit, 0.0);
            if (!best_result.ok) {
                best_result = cand;
            } else {
                const double best_violation = std::max(best_result.budget - recall_budget_limit, 0.0);
                if (
                    cand_violation < best_violation - 1e-12 ||
                    (
                        std::fabs(cand_violation - best_violation) <= 1e-12 &&
                        (
                            cand.raw < best_result.raw ||
                            (
                                std::fabs(cand.raw - best_result.raw) <= 1e-9 &&
                                cand.lambda < best_result.lambda
                            )
                        )
                    )
                ) {
                    best_result = cand;
                }
            }
            if (cand_violation > 0.0) {
                recall_lo = recall_mid;
            } else {
                recall_hi = recall_mid;
            }
        }
    } else {
        best_result = decode_for_recall_mu(
            node_count,
            target_count,
            penalty_steps,
            penalty_max,
            0.0,
            exact_dp,
            recall_penalty_weight,
            edge_costs,
            edge_budgets,
            pred_start,
            first_loss,
            first_budget
        );
    }

    if (!best_result.ok) {
        return -2;
    }
    *out_count = static_cast<int>(best_result.path.size());
    *out_lambda = best_result.lambda;
    for (int idx = 0; idx < *out_count; ++idx) {
        out_path[idx] = static_cast<int32_t>(best_result.path[idx]);
    }
    return 0;
}

extern "C" int polygon_repair_key_scores(
    int frame_count,
    int key_count,
    const int32_t* chosen_frames,
    const double* frame_deficits,
    double* out_scores
) {
    if (
        frame_count < 0 ||
        key_count <= 0 ||
        chosen_frames == nullptr ||
        frame_deficits == nullptr ||
        out_scores == nullptr
    ) {
        return -1;
    }
    for (int key_idx = 0; key_idx < key_count; ++key_idx) {
        out_scores[key_idx] = 0.0;
    }
    const int first_key = static_cast<int>(chosen_frames[0]);
    const int last_key = static_cast<int>(chosen_frames[key_count - 1]);
    for (int frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        const double deficit = frame_deficits[frame_idx];
        if (deficit <= 0.0) {
            continue;
        }
        if (frame_idx <= first_key) {
            out_scores[0] += deficit;
            continue;
        }
        if (frame_idx >= last_key) {
            out_scores[key_count - 1] += deficit;
            continue;
        }
        const int32_t* begin = chosen_frames;
        const int32_t* end = chosen_frames + key_count;
        const int32_t* right_it = std::lower_bound(begin, end, static_cast<int32_t>(frame_idx));
        int right_pos = static_cast<int>(right_it - begin);
        if (right_pos <= 0) {
            out_scores[0] += deficit;
            continue;
        }
        if (right_pos >= key_count) {
            out_scores[key_count - 1] += deficit;
            continue;
        }
        const int left_pos = right_pos - 1;
        const int left_frame = static_cast<int>(chosen_frames[left_pos]);
        const int right_frame = static_cast<int>(chosen_frames[right_pos]);
        const double denom = static_cast<double>(std::max(right_frame - left_frame, 1));
        const double alpha = static_cast<double>(frame_idx - left_frame) / denom;
        out_scores[left_pos] += (1.0 - alpha) * deficit;
        out_scores[right_pos] += alpha * deficit;
    }
    return 0;
}
"""
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
                compiler = os_mod.environ.get("CXX") or "g++"
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
                compiler_text = os_mod.environ.get("CXX") or "g++"
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
        dense_bytes = int(node_count) * int(node_count) * 16
        try:
            dense_limit = int(
                __import__("os").environ.get(
                    "ATOSYORI_POLYGON_NATIVE_DENSE_LIMIT_BYTES", str(512 * 1024 * 1024)
                )
            )
        except ValueError:
            dense_limit = 512 * 1024 * 1024
        if dense_bytes > max(1, int(dense_limit)):
            return original_run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )
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
        edge_costs = module.np.full(
            (node_count, node_count), module.np.inf, dtype=module.np.float64
        )
        edge_budgets = module.np.full(
            (node_count, node_count), module.np.inf, dtype=module.np.float64
        )
        counters = {"interval_evals": 0, "interval_frames": 0}
        reachable = [False] * node_count
        reachable[0] = True

        for node_pos in range(1, node_count):
            end_frame = int(candidate_frames_i[node_pos])
            min_prev_pos = int(
                module.bisect.bisect_left(
                    candidate_frames_i, end_frame - int(dynamic_max_gap), 0, node_pos
                )
            )
            pred_start[node_pos] = int(min_prev_pos)
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
                edge_costs[prev_node_pos, node_pos] = float(info.cost)
                edge_budgets[prev_node_pos, node_pos] = float(info.recall_budget)
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

    def compact_big_list_dumps(obj, *args, **kwargs):
        if isinstance(obj, list) and kwargs.get("indent") is not None:
            kwargs = dict(kwargs)
            kwargs.pop("indent", None)
            kwargs.setdefault("separators", (",", ":"))
        return original_json_dumps(obj, *args, **kwargs)

    def polygon_cache_key(path):
        try:
            return str(module.Path(path).resolve())
        except OSError:
            return str(module.Path(path))

    def cached_load_rows(sqlite_path):
        rows = original_load_rows(sqlite_path)
        cache = getattr(module, "_polygon_loaded_rows_cache", None)
        if cache is None:
            cache = {}
            module._polygon_loaded_rows_cache = cache
        cache[polygon_cache_key(sqlite_path)] = rows
        return rows

    def cached_evaluate_union_exact(union_rows, tracked_sqlite, output_dir):
        output_dir = module.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_lookup = {
            (int(row["frame"]), str(row["track_id"])): row for row in union_rows
        }
        cache = getattr(module, "_polygon_loaded_rows_cache", {})
        rows = cache.get(polygon_cache_key(tracked_sqlite))
        if rows is None:
            rows = original_load_rows(tracked_sqlite)
            cache[polygon_cache_key(tracked_sqlite)] = rows
            module._polygon_loaded_rows_cache = cache

        result_rows = []
        for row in rows:
            pred = pred_lookup.get((int(row.frame), str(row.track_id)))
            if pred is None:
                continue
            pred_polys = [
                module.np.asarray(poly, dtype=module.np.float32).reshape(-1, 2)
                for poly in pred["polygons"]
            ]
            metrics = module.compute_exact_metrics_from_polygons(
                row.polygons, pred_polys
            )
            weighted_error = float(module.compute_weighted_error(metrics))
            result_rows.append(
                {
                    "frame": int(row.frame),
                    "track_id": str(row.track_id),
                    "run_id": int(pred.get("run_id", -1)),
                    "has_keyframe": int(pred.get("has_keyframe", 0)),
                    "gt_area": float(metrics["gt_area"]),
                    "pred_area": float(metrics["pred_area"]),
                    "intersection": float(metrics["intersection"]),
                    "union": float(metrics["union"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "iou": float(metrics["iou"]),
                    "weighted_error": weighted_error,
                }
            )

        sorted_result_rows = sorted(
            result_rows,
            key=lambda item: (int(item["frame"]), int(str(item["track_id"]))),
        )
        metrics_csv = output_dir / "keyframe_exact_metrics.csv"
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = module.csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "track_id",
                    "run_id",
                    "has_keyframe",
                    "gt_area",
                    "pred_area",
                    "intersection",
                    "union",
                    "recall",
                    "precision",
                    "iou",
                    "weighted_error",
                ],
            )
            writer.writeheader()
            writer.writerows(sorted_result_rows)
        summary = {
            "input_tracked_sqlite": str(tracked_sqlite),
            "optimized": module.aggregate_exact_rows(sorted_result_rows),
        }
        (output_dir / "summary.json").write_text(
            module.json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def fast_union_rows_to_pred_sqlite(union_rows, output_sqlite):
        output_sqlite = module.Path(output_sqlite)
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        conn = module.sqlite3.connect(str(output_sqlite))
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)"
            )
            cur.executemany(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    (
                        int(row["frame"]),
                        str(row["track_id"]),
                        module.json.dumps(row["polygons"], ensure_ascii=False),
                    )
                    for row in union_rows
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def build_track_streams_releasing_predictor(*args, **kwargs):
        release_predictor_after_build = bool(
            kwargs.pop("_release_predictor_after_build", True)
        )
        predictor = kwargs.get("predictor")
        if predictor is None and len(args) >= 3:
            predictor = args[2]
        result = original_build_track_streams(*args, **kwargs)
        if predictor is not None and release_predictor_after_build:
            try:
                predictor.model.to("cpu")
            except Exception:
                pass
            try:
                if module.torch.cuda.is_available():
                    module.torch.cuda.synchronize()
                    module.torch.cuda.empty_cache()
            except Exception:
                pass
        return result

    def memory_bounded_build_frame_eval_contexts(run, args):
        import collections as collections_mod
        import os as os_mod

        default_cache = 512
        try:
            max_items = int(
                os_mod.environ.get(
                    "ATOSYORI_POLYGON_EVAL_CONTEXT_CACHE", str(default_cache)
                )
            )
        except ValueError:
            max_items = default_cache
        max_items = max(1, int(max_items))

        class LazyFrameEvalContexts:
            def __init__(self):
                self._cache = collections_mod.OrderedDict()

            def __len__(self):
                return int(len(run.frame_numbers))

            def _build_one(self, frame_idx):
                scale_factor = float(
                    module.np.clip(float(args.dp_eval_scale), 0.1, 1.0)
                )
                pad = int(max(0, int(args.dp_eval_pad)))
                raw_vector = module.flatten_contours(run.anchors[int(frame_idx)])
                (
                    gt_polygon_area,
                    gt_center,
                    gt_radii,
                    gt_mean_radius,
                ) = module.vector_proxy_stats(
                    raw_vector,
                    run.contour_count,
                    run.anchors_per_contour,
                )
                raw_polys = module.split_vector_to_polygons(
                    module.flatten_contours(run.anchors[int(frame_idx)]),
                    run.contour_count,
                    run.anchors_per_contour,
                )
                all_polys = [
                    module.np.asarray(poly, dtype=module.np.float32)
                    for poly in run.gt_polygons[int(frame_idx)] + raw_polys
                    if len(poly) >= 3
                ]
                if all_polys:
                    all_pts = module.np.concatenate(all_polys, axis=0)
                    min_xy = (
                        module.np.floor(all_pts.min(axis=0)).astype(module.np.int32)
                        - pad
                    )
                    max_xy = (
                        module.np.ceil(all_pts.max(axis=0)).astype(module.np.int32)
                        + pad
                    )
                else:
                    min_xy = module.np.asarray([0, 0], dtype=module.np.int32)
                    max_xy = module.np.asarray([4, 4], dtype=module.np.int32)
                shift_xy = min_xy.astype(module.np.float32)
                width = int(max_xy[0] - min_xy[0] + 1)
                height = int(max_xy[1] - min_xy[1] + 1)
                shape_hw = (
                    max(1, int(module.math.ceil(height * scale_factor))),
                    max(1, int(module.math.ceil(width * scale_factor))),
                )
                context = module.FrameEvalContext(
                    gt_mask=module.np.zeros(shape_hw, dtype=module.np.uint8),
                    gt_area=0,
                    shift_xy=shift_xy,
                    shape_hw=shape_hw,
                    scale_factor=scale_factor,
                    gt_center=module.np.asarray(gt_center, dtype=module.np.float32),
                    gt_radii=module.np.asarray(gt_radii, dtype=module.np.float32),
                    gt_mean_radius=float(gt_mean_radius),
                    gt_polygon_area=float(gt_polygon_area),
                )
                gt_mask = module.rasterize_mask_with_context(
                    run.gt_polygons[int(frame_idx)], context
                )
                return module.FrameEvalContext(
                    gt_mask=gt_mask,
                    gt_area=int(gt_mask.sum()),
                    shift_xy=shift_xy,
                    shape_hw=shape_hw,
                    scale_factor=scale_factor,
                    gt_center=module.np.asarray(gt_center, dtype=module.np.float32),
                    gt_radii=module.np.asarray(gt_radii, dtype=module.np.float32),
                    gt_mean_radius=float(gt_mean_radius),
                    gt_polygon_area=float(gt_polygon_area),
                    scratch_pred_mask=module.np.zeros(shape_hw, dtype=module.np.uint8),
                    scratch_intersection_mask=module.np.zeros(
                        shape_hw, dtype=module.np.uint8
                    ),
                )

            def __getitem__(self, frame_idx):
                idx = int(frame_idx)
                if idx < 0:
                    idx += int(len(run.frame_numbers))
                if idx < 0 or idx >= int(len(run.frame_numbers)):
                    raise IndexError(idx)
                cached = self._cache.get(idx)
                if cached is not None:
                    self._cache.move_to_end(idx)
                    return cached
                context = self._build_one(idx)
                self._cache[idx] = context
                if len(self._cache) > max_items:
                    self._cache.popitem(last=False)
                return context

        return LazyFrameEvalContexts()

    def apply_fixed_practical_defaults_with_worker_mode(args):
        args = original_apply_fixed_practical_defaults(args)
        module._fork_polygon_workers = True
        return args

    def stable_polygon_get_context(method=None):
        if method == "spawn" and bool(getattr(module, "_fork_polygon_workers", False)):
            try:
                return original_get_context("fork")
            except ValueError:
                return original_get_context(method)
        return original_get_context(method)

    class PolygonMultiprocessingProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def get_context(self, method=None):
            return stable_polygon_get_context(method)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    module.resample_closed_contour = safe_resample_closed_contour
    module.align_polygon_phase = fast_align_polygon_phase
    module.exact_k_dp = fast_exact_k_dp
    module.run_single_state_penalty_path = native_single_state_penalty_path
    module.repair_keyframe_vectors_for_exact_recall = (
        repair_keyframe_vectors_for_exact_recall_native_key_scores
    )
    module.json.dumps = compact_big_list_dumps
    module.build_track_streams = build_track_streams_releasing_predictor
    module.build_frame_eval_contexts = memory_bounded_build_frame_eval_contexts
    module.load_rows = cached_load_rows
    module.evaluate_union_exact = cached_evaluate_union_exact
    module.union_rows_to_pred_sqlite = fast_union_rows_to_pred_sqlite
    module.apply_fixed_practical_defaults = (
        apply_fixed_practical_defaults_with_worker_mode
    )
    module.multiprocessing = PolygonMultiprocessingProxy(module.multiprocessing)
    module._production_runtime_defaults_applied = True
    return module
