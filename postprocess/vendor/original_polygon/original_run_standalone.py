#!/usr/bin/env python3
from __future__ import annotations

"""Production standalone single-file pipeline for shared ellipse/polygon processing.

This file embeds both the merged v7 pipeline orchestration and the practical
v22 polygon keyframe optimizer so that, aside from external model checkpoints
(e.g. K2 and polygon point-count predictor weights), the pipeline can be read
and executed from this one file alone.

Functional structure:

- shared raw preprocessing
  - NMS
  - optional cut detection
  - tracking
  - short-track removal
- ellipse branch
  - K1 exact + K2 V5 routed inference
  - ellipse keyframe optimization
  - output gap fill
- polygon branch
  - shape stabilization
  - border-aware outward polygon expansion for near-edge masks
  - track-first polygon gap fill
  - AI point-count prediction
  - polygon keyframe optimization
- shared exact evaluation / export / overlay rendering

The hidden subcommands are internal implementation details. Running the file
without a hidden subcommand executes the full pipeline entrypoint.

The production defaults are the validated fast/lossless settings:

- polygon adaptive track/run anchor-count mode is enabled by default
- polygon adaptive anchors use p95 + offset10 with a minimum of 8 anchors per contour
- K2 ellipse inference and polygon anchor-count prediction use CUDA by default
- the polygon 48-anchor setting is a cap/fallback, not a fixed output point count
- exact evaluation and final SQLite export remain enabled for production artifacts
- overlays are disabled by default and can be enabled explicitly
- polygon near-border outward mask expansion and endpoint extension remain enabled as recall safeguards

Approximation mode, target recall, and keyframe frequency are operational
policy choices. They should be specified per class through
``--class-policy-json``. The ``--default-shape-mode`` argument is only a
fallback for labels that are not covered by the class policy.
Each class can override:

- ``shape_mode`` / ``mode``: ``ellipse`` or ``polygon``
- ``target_interval`` or ``target_ratio``: keyframe density
- ``dense_recall_target``: ellipse branch dense recall target
- ``polygon_recall_min`` / ``recall_min`` / ``target_recall``: polygon branch recall floor

Minimal example:

{
  "classes": {
    "person": {
      "shape_mode": "polygon",
      "target_interval": 6,
      "polygon_recall_min": 0.97
    },
    "car": {
      "shape_mode": "ellipse",
      "target_interval": 9,
      "dense_recall_target": 0.991
    }
  }
}
"""

import csv
import json
import subprocess
import sys
import time
import textwrap
import types
from pathlib import Path

try:
    import orjson
except ModuleNotFoundError:
    orjson = None

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parent


def _register_inline_module(module_name: str, export_map: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(SELF_PATH)
    for public_name, global_name in export_map.items():
        setattr(module, public_name, globals()[global_name])
    sys.modules[module_name] = module
    return module


def _run_inline_entrypoint(func, argv: list[str]) -> None:
    previous_argv = sys.argv[:]
    sys.argv = [previous_argv[0], *argv]
    try:
        func()
    finally:
        sys.argv = previous_argv


READABLE_POLYGON_SOURCE_BEGIN = "# --- BEGIN READABLE EMBEDDED POLYGON V22 SOURCE ---"
READABLE_POLYGON_SOURCE_END = "# --- END READABLE EMBEDDED POLYGON V22 SOURCE ---"


def _load_embedded_polygon_optimizer_source() -> str:
    """Return the readable in-file polygon optimizer source.

    The optimizer is kept below as normal Python under an ``if False`` block so
    this file remains a true standalone artifact without hiding the core solver
    inside an unreadable one-line string. At runtime we execute it in a separate
    module namespace to avoid collisions with the full-pipeline helpers.
    """
    lines = SELF_PATH.read_text(encoding="utf-8").splitlines()
    try:
        start = next(idx for idx, line in enumerate(lines) if line.strip() == READABLE_POLYGON_SOURCE_BEGIN)
        end = next(
            idx
            for idx in range(start + 1, len(lines))
            if lines[idx].strip() == READABLE_POLYGON_SOURCE_END
        )
    except StopIteration as exc:
        raise RuntimeError("Readable embedded polygon optimizer source block was not found") from exc
    body = textwrap.dedent("\n".join(lines[start + 1 : end])).strip()
    return "from __future__ import annotations\n\n" + body + "\n"


def _build_embedded_polygon_v22_module() -> types.ModuleType:
    module_name = 'embedded_polygon_keyframe_v22'
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module = types.ModuleType(module_name)
    module.__file__ = str(ROOT / 'polygon_keyframe_opt' / 'standard_standalone_polygon_keyframe_multistate_v22_onefile.py')
    sys.modules[module_name] = module
    source = _load_embedded_polygon_optimizer_source()
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
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
    original_repair_keyframe_vectors_for_exact_recall = module.repair_keyframe_vectors_for_exact_recall

    def resolve_point_predictor_model_dir():
        relative = Path(
            "experiments/linear_polygon_bezier_workspace_20260410/"
            "output/mask_point_predictor_wide96_20260411"
        )
        candidates = [
            ROOT / "assets" / "polygon_point_predictor",
            Path(module.DEFAULT_POINT_PREDICTOR_MODEL_DIR),
            ROOT / relative,
            ROOT.parent / relative,
            ROOT / "standalone_k1_k2_v5_clean" / relative,
            ROOT.parent / "standalone_k1_k2_v5_clean" / relative,
        ]
        for candidate in candidates:
            if (candidate / "best.pt").exists():
                return candidate
        return Path(module.DEFAULT_POINT_PREDICTOR_MODEL_DIR)

    module.DEFAULT_ADAPTIVE_ANCHOR_COUNTS = True
    module.DEFAULT_ADAPTIVE_POINT_OFFSET = 10
    module.DEFAULT_MIN_ANCHORS_PER_CONTOUR = 8
    module.DEFAULT_POINT_PREDICTOR_MODEL_DIR = resolve_point_predictor_model_dir()
    module.DEFAULT_PREDICTOR_DEVICE = "cuda"

    def safe_resample_closed_contour(poly, n_points):
        target = max(3, int(n_points))
        out = module.np.asarray(original_resample_closed_contour(poly, target), dtype=module.np.float32).reshape(-1, 2)
        if len(out) == target:
            return out
        if len(out) == 0:
            return module.np.zeros((target, 2), dtype=module.np.float32)
        return module.np.repeat(out[:1], target, axis=0).astype(module.np.float32)

    def fast_align_polygon_phase(reference, poly):
        candidate = module.np.asarray(module.orient_ccw(poly), dtype=module.np.float32).reshape(-1, 2)
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
            return float(scores[best_idx]), module.np.asarray(rolled[best_idx], dtype=module.np.float32).copy()

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
            min_prev_pos = int(module.bisect.bisect_left(nodes, end_node - max_gap_i, 0, end_pos))
            min_prev_positions[end_pos] = min_prev_pos
            max_width = max(max_width, int(end_pos - min_prev_pos))

        edge_costs = module.np.full((node_count, max(1, int(max_width))), module.np.inf, dtype=module.np.float64)
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

        back_dtype = module.np.uint16 if max_gap_i < int(module.np.iinfo(module.np.uint16).max) else module.np.int32
        back_itemsize = int(module.np.dtype(back_dtype).itemsize)
        row_bytes = int(node_count * back_itemsize)
        prev_dp = module.np.full((node_count,), module.np.inf, dtype=module.np.float64)
        prev_dp[0] = 0.0

        with tempfile_mod.TemporaryFile() as back_file:
            back_file.truncate(int(target_count) * row_bytes)
            for used in range(1, target_count):
                curr_dp = module.np.full((node_count,), module.np.inf, dtype=module.np.float64)
                back_offsets = module.np.zeros((node_count,), dtype=back_dtype)
                for end_pos in range(used, node_count):
                    min_prev_pos = max(used - 1, int(min_prev_positions[end_pos]))
                    if min_prev_pos >= end_pos:
                        continue
                    edge_offset = int(min_prev_pos - int(min_prev_positions[end_pos]))
                    edge_values = edge_costs[end_pos, edge_offset : edge_offset + int(end_pos - min_prev_pos)]
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

        if str(__import__("os").environ.get("ATOSYORI_POLYGON_DISABLE_NATIVE_DP", "")).strip():
            module._native_polygon_dp_unavailable = True
            return None

        native_source = r'''
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
'''
        os_mod = __import__("os")
        hashlib_mod = __import__("hashlib")
        ctypes_mod = __import__("ctypes")
        digest = hashlib_mod.sha256(native_source.encode("utf-8")).hexdigest()[:16]
        build_dir = module.Path(os_mod.environ.get("ATOSYORI_POLYGON_NATIVE_DIR", "/tmp/atosyori_polygon_native"))
        source_path = build_dir / f"polygon_dp_{digest}.cpp"
        lib_path = build_dir / f"polygon_dp_{digest}.so"
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
            if not lib_path.exists():
                source_path.write_text(native_source, encoding="utf-8")
                tmp_lib_path = build_dir / f"polygon_dp_{digest}.{os_mod.getpid()}.tmp.so"
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
            dense_limit = int(__import__("os").environ.get("ATOSYORI_POLYGON_NATIVE_DENSE_LIMIT_BYTES", str(512 * 1024 * 1024)))
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
        dynamic_max_gap = max(int(args.max_gap), int(module.math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))))
        candidate_frames_i = [int(v) for v in candidate_frames]
        pred_start = module.np.zeros((node_count,), dtype=module.np.int32)
        edge_costs = module.np.full((node_count, node_count), module.np.inf, dtype=module.np.float64)
        edge_budgets = module.np.full((node_count, node_count), module.np.inf, dtype=module.np.float64)
        counters = {"interval_evals": 0, "interval_frames": 0}
        reachable = [False] * node_count
        reachable[0] = True

        for node_pos in range(1, node_count):
            end_frame = int(candidate_frames_i[node_pos])
            min_prev_pos = int(module.bisect.bisect_left(candidate_frames_i, end_frame - int(dynamic_max_gap), 0, node_pos))
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
        edge_budgets = module.np.ascontiguousarray(edge_budgets, dtype=module.np.float64)
        pred_start = module.np.ascontiguousarray(pred_start, dtype=module.np.int32)
        status = int(
            fn(
                ctypes_mod.c_int(node_count),
                ctypes_mod.c_int(max(2, min(int(target_count), node_count))),
                ctypes_mod.c_int(max(1, int(args.penalty_binary_steps))),
                ctypes_mod.c_int(max(1, int(args.recall_budget_binary_steps))),
                ctypes_mod.c_int(1 if str(args.recall_constraint_mode) == "exact_dp" else 0),
                ctypes_mod.c_double(float(args.penalty_max)),
                ctypes_mod.c_double(float(max(args.recall_budget_max_mu, 1e-6))),
                ctypes_mod.c_double(float(module.recall_budget_limit(len(run.frame_numbers), args))),
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
        chosen_node_positions = [int(v) for v in out_path[: int(out_count.value)].tolist()]
        chosen_frames = [int(candidate_frames_i[pos]) for pos in chosen_node_positions]
        return chosen_frames, [0] * len(chosen_frames), counters, {}, float(out_lambda.value)

    def native_repair_key_scores(chosen_frames, frame_deficits):
        if ensure_native_polygon_dp_lib() is None:
            return None
        fn = getattr(module, "_native_polygon_repair_key_scores_fn", None)
        if fn is None:
            return None
        ctypes_mod = __import__("ctypes")
        chosen_arr = module.np.ascontiguousarray([int(v) for v in chosen_frames], dtype=module.np.int32)
        deficits_arr = module.np.ascontiguousarray(frame_deficits, dtype=module.np.float64)
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
        global_recall = float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0
        return metric_rows, float(total_iou_loss), float(mean_iou), float(mean_recall), float(mean_precision), float(global_recall)

    def changed_repair_frame_indices(run, chosen_frames, old_vectors, new_vectors):
        old_arr = module.np.asarray(old_vectors, dtype=module.np.float32)
        new_arr = module.np.asarray(new_vectors, dtype=module.np.float32)
        if old_arr.shape != new_arr.shape:
            return list(range(len(run.frame_numbers)))
        flat_delta = module.np.reshape(module.np.abs(new_arr - old_arr), (len(chosen_frames), -1))
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

    def exact_interpolated_metrics_delta(run, chosen_frames, old_vectors, new_vectors, base_metrics_rows):
        affected_frames = changed_repair_frame_indices(run, chosen_frames, old_vectors, new_vectors)
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
                right_pos = int(module.bisect.bisect_left(chosen_frames_arr, int(frame_idx)))
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen_frames_arr[left_pos])
                right_frame = int(chosen_frames_arr[right_pos])
                if frame_idx == right_frame:
                    vec = module.np.asarray(new_vectors[right_pos], dtype=module.np.float32)
                else:
                    alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                    vec = module.interpolate_vectors(new_vectors[left_pos], new_vectors[right_pos], alpha)
            pred_polys = module.split_vector_to_polygons(vec, run.contour_count, run.anchors_per_contour)
            trial_metrics[int(frame_idx)] = module.compute_exact_metrics_from_polygons(run.gt_polygons[int(frame_idx)], pred_polys)
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
        disable_repair_delta = bool(str(os_mod.environ.get("ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA", "")).strip())
        use_repair_delta = not disable_repair_delta
        scale_deltas = module.parse_float_list(str(args.exact_recall_repair_scale_deltas), [0.01, 0.02, 0.04, 0.06, 0.08])
        metrics_rows, current_iou_loss, _current_mean_iou, current_mean_recall, _current_mean_precision, _current_global_recall = module.exact_interpolated_metrics(run, chosen_frames, current)
        best_key = module.exact_recall_solution_key(current_iou_loss, current_mean_recall, args)
        if best_key[0] <= 0.0:
            return current

        for _pass in range(max(1, int(args.exact_recall_repair_max_passes))):
            frame_deficits = module.np.asarray(
                [float(row["gt_area"]) * max(float(args.recall_min) - float(row["recall"]), 0.0) for row in metrics_rows],
                dtype=module.np.float64,
            )
            if float(module.np.mean(frame_deficits)) <= 0.0 and best_key[0] <= 0.0:
                break
            key_scores = native_repair_key_scores(chosen_frames, frame_deficits)
            if key_scores is None:
                key_scores = module.np.zeros((len(chosen_frames),), dtype=module.np.float64)
                for frame_idx, deficit in enumerate(frame_deficits.tolist()):
                    if deficit <= 0.0:
                        continue
                    if frame_idx <= int(chosen_frames[0]):
                        key_scores[0] += float(deficit)
                        continue
                    if frame_idx >= int(chosen_frames[-1]):
                        key_scores[-1] += float(deficit)
                        continue
                    right_pos = next(pos for pos, keyframe in enumerate(chosen_frames) if keyframe >= frame_idx)
                    left_pos = max(0, right_pos - 1)
                    left_frame = int(chosen_frames[left_pos])
                    right_frame = int(chosen_frames[right_pos])
                    alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                    key_scores[left_pos] += (1.0 - alpha) * float(deficit)
                    key_scores[right_pos] += alpha * float(deficit)
            key_order = [int(idx) for idx in module.np.argsort(-key_scores)[: max(1, int(args.exact_recall_repair_topk))].tolist()]
            improved = False

            trial_vectors = []
            for delta in scale_deltas:
                scaled_all = module.np.asarray(current, dtype=module.np.float32).copy()
                for key_idx in range(len(chosen_frames)):
                    scaled_all[key_idx] = module.scale_vector_about_centroid(scaled_all[key_idx], 1.0 + float(delta))
                trial_vectors.append(scaled_all)
            for delta in scale_deltas:
                scaled = module.np.asarray(current, dtype=module.np.float32).copy()
                for key_idx in key_order:
                    scaled[key_idx] = module.scale_vector_about_centroid(scaled[key_idx], 1.0 + float(delta))
                trial_vectors.append(scaled)

            for key_idx in key_order:
                frame_idx = int(chosen_frames[key_idx])
                current_area, _center, _radii, _mean_radius = module.vector_proxy_stats(current[key_idx], run.contour_count, run.anchors_per_contour)
                for candidate in candidates_by_frame[frame_idx]:
                    if float(candidate.area) <= float(current_area) + 1e-3:
                        continue
                    upgraded = module.np.asarray(current, dtype=module.np.float32).copy()
                    upgraded[key_idx] = module.np.asarray(candidate.vector, dtype=module.np.float32)
                    trial_vectors.append(upgraded)
                for delta in scale_deltas:
                    upgraded = module.np.asarray(current, dtype=module.np.float32).copy()
                    upgraded[key_idx] = module.scale_vector_about_centroid(upgraded[key_idx], 1.0 + float(delta))
                    trial_vectors.append(upgraded)

            seen = []
            for trial in trial_vectors:
                if any(module.np.allclose(trial, existing, atol=1e-4) for existing in seen):
                    continue
                seen.append(module.np.asarray(trial, dtype=module.np.float32))
                if use_repair_delta:
                    trial_metrics, trial_iou_loss, _trial_mean_iou, trial_mean_recall, _trial_mean_precision, _trial_global_recall = exact_interpolated_metrics_delta(
                        run,
                        chosen_frames,
                        current,
                        trial,
                        metrics_rows,
                    )
                else:
                    trial_metrics, trial_iou_loss, _trial_mean_iou, trial_mean_recall, _trial_mean_precision, _trial_global_recall = module.exact_interpolated_metrics(run, chosen_frames, trial)
                trial_key = module.exact_recall_solution_key(trial_iou_loss, trial_mean_recall, args)
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
        pred_lookup = {(int(row["frame"]), str(row["track_id"])): row for row in union_rows}
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
            metrics = module.compute_exact_metrics_from_polygons(row.polygons, pred_polys)
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
        (output_dir / "summary.json").write_text(module.json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def fast_union_rows_to_pred_sqlite(union_rows, output_sqlite):
        output_sqlite = module.Path(output_sqlite)
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        conn = module.sqlite3.connect(str(output_sqlite))
        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
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
        release_predictor_after_build = bool(kwargs.pop("_release_predictor_after_build", True))
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
            max_items = int(os_mod.environ.get("ATOSYORI_POLYGON_EVAL_CONTEXT_CACHE", str(default_cache)))
        except ValueError:
            max_items = default_cache
        max_items = max(1, int(max_items))

        class LazyFrameEvalContexts:
            def __init__(self):
                self._cache = collections_mod.OrderedDict()

            def __len__(self):
                return int(len(run.frame_numbers))

            def _build_one(self, frame_idx):
                scale_factor = float(module.np.clip(float(args.dp_eval_scale), 0.1, 1.0))
                pad = int(max(0, int(args.dp_eval_pad)))
                raw_vector = module.flatten_contours(run.anchors[int(frame_idx)])
                gt_polygon_area, gt_center, gt_radii, gt_mean_radius = module.vector_proxy_stats(
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
                    min_xy = module.np.floor(all_pts.min(axis=0)).astype(module.np.int32) - pad
                    max_xy = module.np.ceil(all_pts.max(axis=0)).astype(module.np.int32) + pad
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
                gt_mask = module.rasterize_mask_with_context(run.gt_polygons[int(frame_idx)], context)
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
                    scratch_intersection_mask=module.np.zeros(shape_hw, dtype=module.np.uint8),
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
    module.repair_keyframe_vectors_for_exact_recall = repair_keyframe_vectors_for_exact_recall_native_key_scores
    module.json.dumps = compact_big_list_dumps
    module.build_track_streams = build_track_streams_releasing_predictor
    module.build_frame_eval_contexts = memory_bounded_build_frame_eval_contexts
    module.load_rows = cached_load_rows
    module.evaluate_union_exact = cached_evaluate_union_exact
    module.union_rows_to_pred_sqlite = fast_union_rows_to_pred_sqlite
    module.apply_fixed_practical_defaults = apply_fixed_practical_defaults_with_worker_mode
    module.multiprocessing = PolygonMultiprocessingProxy(module.multiprocessing)
    return module


def polygon_inline_main() -> None:
    module = _build_embedded_polygon_v22_module()
    module.main()



# ==============================================================================
# Inlined from: standalone_runtime_fst.py
# ==============================================================================

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import multiprocessing
import os
import pickle
import sqlite3
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
fst_WIDTH = 1920
fst_HEIGHT = 1080
fst_POLYGON_POINTS = 96
fst_MASK_COLOR = np.array([0, 0, 255], dtype=np.float32)
fst_ELLIPSE_COLOR = (0, 0, 0)
fst__ANGLE_TABLE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
fst__KERNEL_CACHE: dict[int, np.ndarray] = {}
fst__GRID_CACHE: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
fst__ROW_POLYGONS_JSONS: list[str] = []
fst__ROW_LOCAL_RASTER_PAYLOADS: list[tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]] = []
fst__ROW_GT_POLYGONS: list[list[np.ndarray]] = []

def fst__get_unit_circle(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    cached = fst__ANGLE_TABLE.get(n_points)
    if cached is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float32)
        cached = (np.cos(angles), np.sin(angles))
        fst__ANGLE_TABLE[n_points] = cached
    return cached

def fst_ellipse_to_polygon_array(cx: float, cy: float, a: float, b: float, theta_deg: float, n_points: int=fst_POLYGON_POINTS) -> np.ndarray:
    unit_cos, unit_sin = fst__get_unit_circle(n_points)
    cos_t = math.cos(math.radians(theta_deg))
    sin_t = math.sin(math.radians(theta_deg))
    xs = a * unit_cos
    ys = b * unit_sin
    pts = np.empty((n_points, 2), dtype=np.float32)
    pts[:, 0] = cx + xs * cos_t - ys * sin_t
    pts[:, 1] = cy + xs * sin_t + ys * cos_t
    return pts

def fst_ellipse_to_polygon(cx: float, cy: float, a: float, b: float, theta_deg: float, n_points: int=fst_POLYGON_POINTS) -> list[list[float]]:
    return fst_ellipse_to_polygon_array(cx, cy, a, b, theta_deg, n_points=n_points).astype(np.float64).tolist()

def fst_parse_polygons(polygons_json: str) -> list[np.ndarray]:
    polygons = json.loads(polygons_json)
    return [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in polygons]

def fst_make_polygons_json(ellipses: list[tuple[float, float, float, float, float]]) -> str:
    polygons = [fst_ellipse_to_polygon(cx, cy, a, b, angle, n_points=fst_POLYGON_POINTS) for cx, cy, a, b, angle in ellipses]
    return json.dumps(polygons)

def fst_ellipses_to_polygon_arrays(ellipses: list[tuple[float, float, float, float, float]]) -> list[np.ndarray]:
    return [fst_ellipse_to_polygon_array(cx, cy, a, b, angle, n_points=fst_POLYGON_POINTS).astype(np.float32) for cx, cy, a, b, angle in ellipses]

def fst_rasterize_full(polygons_json: str, height: int=fst_HEIGHT, width: int=fst_WIDTH) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in fst_parse_polygons(polygons_json):
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def fst_rasterize_full_from_polygons(polygons: list[np.ndarray], height: int=fst_HEIGHT, width: int=fst_WIDTH) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def fst_prepare_local_raster_payload_from_polygons(pts: list[np.ndarray]) -> tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]:
    all_pts = np.concatenate(pts, axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    bbox_w = float(max_xy[0] - min_xy[0] + 1.0)
    bbox_h = float(max_xy[1] - min_xy[1] + 1.0)
    pad = max(24, int(max(bbox_w, bbox_h) * 0.45))
    x0 = max(0, int(math.floor(min_xy[0])) - pad)
    y0 = max(0, int(math.floor(min_xy[1])) - pad)
    x1 = min(fst_WIDTH - 1, int(math.ceil(max_xy[0])) + pad)
    y1 = min(fst_HEIGHT - 1, int(math.ceil(max_xy[1])) + pad)
    shifted = [np.round(poly - np.array([x0, y0], dtype=np.float32)).astype(np.int32) for poly in pts]
    return ((y1 - y0 + 1, x1 - x0 + 1), (x0, y0), shifted)

def fst_prepare_local_raster_payload(polygons_json: str) -> tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]:
    return fst_prepare_local_raster_payload_from_polygons(fst_parse_polygons(polygons_json))

def fst_rasterize_local_mask_from_payload(payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]) -> tuple[np.ndarray, tuple[int, int]]:
    shape, origin, shifted = payload
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, shifted, 1)
    return (mask, origin)

def fst_rasterize_polygons_to_local_mask(polygons_json: str) -> tuple[np.ndarray, tuple[int, int]]:
    return fst_rasterize_local_mask_from_payload(fst_prepare_local_raster_payload(polygons_json))

def fst_set_row_local_raster_cache(polygons_jsons: list[str], payloads: list[tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]], gt_polygons: list[list[np.ndarray]]) -> None:
    global fst__ROW_POLYGONS_JSONS, fst__ROW_LOCAL_RASTER_PAYLOADS, fst__ROW_GT_POLYGONS
    fst__ROW_POLYGONS_JSONS = polygons_jsons
    fst__ROW_LOCAL_RASTER_PAYLOADS = payloads
    fst__ROW_GT_POLYGONS = gt_polygons

def fst_normalize_ellipse(ellipse: tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
    cx, cy, a, b, angle = ellipse
    if a < b:
        a, b = (b, a)
        angle += 90.0
    angle %= 180.0
    return (float(cx), float(cy), float(max(a, 1.0)), float(max(b, 1.0)), float(angle))

def fst_fit_ellipse_from_points(points_xy: np.ndarray) -> tuple[float, float, float, float, float] | None:
    if len(points_xy) < 5:
        return None
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    return fst_normalize_ellipse((float(cx), float(cy), max(float(w) / 2.0, 1.0), max(float(h) / 2.0, 1.0), float(angle)))

def fst_fit_ellipse_from_mask(mask: np.ndarray) -> tuple[float, float, float, float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = np.concatenate(contours, axis=0).reshape(-1, 2)
    if len(points) < 5:
        ys, xs = np.where(mask > 0)
        points = np.column_stack([xs, ys]).astype(np.float32)
    return fst_fit_ellipse_from_points(points)

def fst_render_ellipses(shape: tuple[int, int], ellipses: list[tuple[float, float, float, float, float]], scales: list[float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for (cx, cy, a, b, angle), scale in zip(ellipses, scales):
        poly = np.round(fst_ellipse_to_polygon_array(cx, cy, a * scale, b * scale, angle)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
    return mask

def fst_compute_mask_metrics(gt_mask: np.ndarray, sub_mask: np.ndarray, gt_area: int | None=None) -> tuple[float, float]:
    if gt_area is None:
        gt_area = int(np.count_nonzero(gt_mask))
    intersection = int(np.count_nonzero(gt_mask & sub_mask))
    union = int(np.count_nonzero(gt_mask | sub_mask))
    recall = intersection / gt_area if gt_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return (iou, recall)

def fst_compute_exact_metrics(gt_polygons_json: str, pred_polygons_json: str) -> dict[str, float]:
    gt_polys = fst_parse_polygons(gt_polygons_json)
    pred_polys = fst_parse_polygons(pred_polygons_json)
    return fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys)

def fst_compute_exact_metrics_from_gt_polys(gt_polys: list[np.ndarray], pred_polygons_json: str) -> dict[str, float]:
    return fst_compute_exact_metrics_from_polygons(gt_polys, fst_parse_polygons(pred_polygons_json))

def fst_compute_exact_metrics_from_polygons(gt_polys: list[np.ndarray], pred_polys: list[np.ndarray]) -> dict[str, float]:
    rounded_polys = [np.round(poly).astype(np.int32) for poly in gt_polys + pred_polys]
    all_pts = np.concatenate(rounded_polys, axis=0)
    points_in_bounds = int(all_pts[:, 0].min()) >= 0 and int(all_pts[:, 1].min()) >= 0 and (int(all_pts[:, 0].max()) < fst_WIDTH) and (int(all_pts[:, 1].max()) < fst_HEIGHT)
    if points_in_bounds:
        x0 = int(all_pts[:, 0].min())
        y0 = int(all_pts[:, 1].min())
        x1 = int(all_pts[:, 0].max())
        y1 = int(all_pts[:, 1].max())
        shift = np.array([x0, y0], dtype=np.int32)
        shape = (y1 - y0 + 1, x1 - x0 + 1)
        gt_mask = np.zeros(shape, dtype=np.uint8)
        pred_mask = np.zeros(shape, dtype=np.uint8)
        gt_rounded = rounded_polys[:len(gt_polys)]
        pred_rounded = rounded_polys[len(gt_polys):]
        for poly in gt_rounded:
            cv2.fillPoly(gt_mask, [poly - shift], 1)
        for poly in pred_rounded:
            cv2.fillPoly(pred_mask, [poly - shift], 1)
    else:
        gt_mask = fst_rasterize_full_from_polygons(gt_polys)
        pred_mask = fst_rasterize_full_from_polygons(pred_polys)
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = int((gt_mask | pred_mask).sum())
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {'gt_area': float(gt_area), 'pred_area': float(pred_area), 'intersection': float(intersection), 'union': float(union), 'recall': float(recall), 'precision': float(precision), 'iou': float(iou)}

def fst_compute_weighted_error(metrics: dict[str, float]) -> int:
    fn_pixels = int(metrics['gt_area'] - metrics['intersection'])
    fp_pixels = int(metrics['pred_area'] - metrics['intersection'])
    return int(2 * fn_pixels + fp_pixels)

def fst_candidate_score(iou: float, recall: float, recall_target: float) -> float:
    return iou - 4.0 * max(0.0, recall_target - recall)

def fst_binary_search_scale(gt_mask: np.ndarray, ellipses: list[tuple[float, float, float, float, float]], fixed_scales: list[float] | None, scale_index: int | None, low: float, high: float, recall_target: float, iterations: int, gt_area: int | None=None) -> float:
    if gt_area is None:
        gt_area = int(np.count_nonzero(gt_mask))
    lo = low
    hi = high
    best = hi
    base_scales = [1.0] * len(ellipses) if fixed_scales is None else list(fixed_scales)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        scales = list(base_scales)
        if scale_index is None:
            scales = [mid] * len(ellipses)
        else:
            scales[scale_index] = mid
        sub_mask = fst_render_ellipses(gt_mask.shape, ellipses, scales)
        intersection = int(np.count_nonzero(gt_mask & sub_mask))
        recall = intersection / gt_area if gt_area > 0 else 1.0
        if recall >= recall_target:
            best = mid
            hi = mid
        else:
            lo = mid
    return best

def fst_optimize_candidate_scales(gt_mask: np.ndarray, ellipses: list[tuple[float, float, float, float, float]], recall_target: float, min_scale: float=0.35, max_scale: float=3.0) -> tuple[list[float], float, float]:
    gt_area = int(np.count_nonzero(gt_mask))
    current_high = max_scale
    for _ in range(6):
        sub_mask = fst_render_ellipses(gt_mask.shape, ellipses, [current_high] * len(ellipses))
        intersection = int(np.count_nonzero(gt_mask & sub_mask))
        recall = intersection / gt_area if gt_area > 0 else 1.0
        if recall >= recall_target:
            break
        current_high *= 1.4
    shared_scale = fst_binary_search_scale(gt_mask, ellipses, fixed_scales=None, scale_index=None, low=min_scale, high=current_high, recall_target=recall_target, iterations=12, gt_area=gt_area)
    scales = [shared_scale] * len(ellipses)
    for _ in range(2):
        changed = False
        for idx in range(len(scales)):
            improved = fst_binary_search_scale(gt_mask, ellipses, fixed_scales=scales, scale_index=idx, low=min_scale, high=scales[idx], recall_target=recall_target, iterations=10, gt_area=gt_area)
            if improved < scales[idx] - 0.001:
                scales[idx] = improved
                changed = True
        if not changed:
            break
    final_mask = fst_render_ellipses(gt_mask.shape, ellipses, scales)
    iou, recall = fst_compute_mask_metrics(gt_mask, final_mask, gt_area=gt_area)
    return (scales, iou, recall)

def fst_apply_scales_to_ellipses(ellipses: list[tuple[float, float, float, float, float]], scales: list[float]) -> list[tuple[float, float, float, float, float]]:
    return [fst_normalize_ellipse((cx, cy, a * scale, b * scale, angle)) for (cx, cy, a, b, angle), scale in zip(ellipses, scales)]

def fst_refine_ellipses_locally(gt_mask: np.ndarray, ellipses: list[tuple[float, float, float, float, float]], recall_target: float, max_rounds: int=6) -> tuple[list[tuple[float, float, float, float, float]], float, float]:
    current = [list(fst_normalize_ellipse(ellipse)) for ellipse in ellipses]
    gt_area = int(np.count_nonzero(gt_mask))

    def evaluate(candidate: list[list[float]]) -> tuple[float, float, float]:
        norm = [fst_normalize_ellipse(tuple(ellipse)) for ellipse in candidate]
        sub_mask = fst_render_ellipses(gt_mask.shape, norm, [1.0] * len(norm))
        iou, recall = fst_compute_mask_metrics(gt_mask, sub_mask, gt_area=gt_area)
        return (fst_candidate_score(iou, recall, recall_target), iou, recall)
    best_score, best_iou, best_recall = evaluate(current)
    height, width = gt_mask.shape
    step_pos = max(2.0, 0.04 * max(width, height))
    step_rad = max(2.0, 0.05 * max(width, height))
    step_angle = 12.0
    for _ in range(max_rounds):
        improved = False
        for ellipse_idx in range(len(current)):
            for param_idx in range(5):
                base_value = current[ellipse_idx][param_idx]
                deltas = (-step_pos, step_pos) if param_idx in (0, 1) else (-step_rad, step_rad) if param_idx in (2, 3) else (-step_angle, step_angle)
                local_best = None
                for delta in deltas:
                    trial = [ellipse[:] for ellipse in current]
                    trial[ellipse_idx][param_idx] = base_value + delta
                    if param_idx in (2, 3) and trial[ellipse_idx][param_idx] < 1.0:
                        continue
                    score, iou, recall = evaluate(trial)
                    if local_best is None or score > local_best[0]:
                        local_best = (score, iou, recall, trial)
                if local_best is not None and local_best[0] > best_score + 1e-06:
                    best_score, best_iou, best_recall, current = local_best
                    improved = True
        if not improved:
            step_pos *= 0.5
            step_rad *= 0.5
            step_angle *= 0.5
            if step_pos < 0.5 and step_rad < 0.5 and (step_angle < 1.0):
                break
    refined = [fst_normalize_ellipse(tuple(ellipse)) for ellipse in current]
    return (refined, best_iou, best_recall)

def fst_build_component_mask(shape: tuple[int, int], points_xy: np.ndarray, kernel_size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts_int = points_xy.astype(np.int32)
    mask[pts_int[:, 1], pts_int[:, 0]] = 1
    kernel = fst__KERNEL_CACHE.get(kernel_size)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        fst__KERNEL_CACHE[kernel_size] = kernel
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def fst__fit_axis_split_candidates(gt_mask: np.ndarray, points: np.ndarray, projection: np.ndarray, quantile: float) -> list[tuple[float, float, float, float, float]] | None:
    threshold = float(np.quantile(projection, quantile))
    ellipses: list[tuple[float, float, float, float, float]] = []
    for component in (projection <= threshold, projection > threshold):
        frac = float(component.mean())
        if frac < 0.12 or frac > 0.88:
            return None
        component_points = points[component]
        component_mask = fst_build_component_mask(gt_mask.shape, component_points, kernel_size=5)
        ellipse = fst_fit_ellipse_from_mask(component_mask)
        if ellipse is None:
            return None
        ellipses.append(ellipse)
    return ellipses

def fst_generate_principal_axis_candidates(gt_mask: np.ndarray) -> dict[tuple[str, float], list[tuple[float, float, float, float, float]]]:
    ys, xs = np.where(gt_mask > 0)
    if len(xs) < 32:
        return {}
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    denom = max(len(points) - 1, 1)
    cov = centered.T @ centered / denom
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    results: dict[tuple[str, float], list[tuple[float, float, float, float, float]]] = {}
    for quantile in (0.35, 0.5, 0.65):
        candidate = fst__fit_axis_split_candidates(gt_mask, points, centered @ axes[:, 0], quantile)
        if candidate is not None:
            results['major', quantile] = candidate
    minor = fst__fit_axis_split_candidates(gt_mask, points, centered @ axes[:, 1], 0.5)
    if minor is not None:
        results['minor', 0.5] = minor
    return results

def fst_select_distance_transform_peaks(distance_map: np.ndarray) -> list[tuple[int, int]]:
    max_value = float(distance_map.max())
    if max_value <= 0.0:
        return []
    local_max = distance_map == cv2.dilate(distance_map, np.ones((9, 9), dtype=np.float32))
    peak_mask = local_max & (distance_map >= max_value * 0.28)
    coords = np.argwhere(peak_mask)
    if len(coords) == 0:
        return []
    values = distance_map[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    min_sep = max(12.0, 0.08 * min(distance_map.shape))
    peaks: list[tuple[int, int]] = []
    for idx in order:
        y, x = coords[idx]
        if not peaks:
            peaks.append((int(x), int(y)))
            continue
        if all((math.hypot(x - px, y - py) >= min_sep for px, py in peaks)):
            peaks.append((int(x), int(y)))
        if len(peaks) == 2:
            break
    return peaks

def fst_distance_transform_candidate(gt_mask: np.ndarray) -> list[tuple[float, float, float, float, float]] | None:
    distance_map = cv2.distanceTransform(gt_mask, cv2.DIST_L2, 5)
    peaks = fst_select_distance_transform_peaks(distance_map)
    if len(peaks) < 2:
        return None
    ys, xs = np.where(gt_mask > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    seeds = np.array(peaks, dtype=np.float32)
    sq_dists = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(sq_dists, axis=1)
    ellipses = []
    for label in (0, 1):
        component_points = points[labels == label]
        if len(component_points) < 16:
            return None
        component_mask = fst_build_component_mask(gt_mask.shape, component_points, kernel_size=7)
        ellipse = fst_fit_ellipse_from_mask(component_mask)
        if ellipse is None:
            return None
        ellipses.append(ellipse)
    return ellipses

def fst_shift_ellipses_to_local(absolute_ellipses: list[tuple[float, float, float, float, float]], origin: tuple[int, int]) -> list[tuple[float, float, float, float, float]]:
    ox, oy = origin
    return [fst_normalize_ellipse((cx - ox, cy - oy, a, b, angle)) for cx, cy, a, b, angle in absolute_ellipses]

def fst_shift_ellipses_to_absolute(local_ellipses: list[tuple[float, float, float, float, float]], origin: tuple[int, int]) -> list[tuple[float, float, float, float, float]]:
    ox, oy = origin
    return [fst_normalize_ellipse((cx + ox, cy + oy, a, b, angle)) for cx, cy, a, b, angle in local_ellipses]

def fst_ensure_two_ellipses(ellipses: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    if len(ellipses) == 2:
        return ellipses
    cx, cy, _, _, angle = ellipses[0]
    return [ellipses[0], (cx, cy, 2.0, 2.0, angle)]

def fst_downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.copy()
    height, width = mask.shape
    target_h = max(8, int(round(height / factor)))
    target_w = max(8, int(round(width / factor)))
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_AREA)

def fst_detect_edge_touches(gt_mask: np.ndarray) -> dict[str, bool]:
    return {'left': bool(np.any(gt_mask[:, 0] > 0)), 'right': bool(np.any(gt_mask[:, -1] > 0)), 'top': bool(np.any(gt_mask[0, :] > 0)), 'bottom': bool(np.any(gt_mask[-1, :] > 0))}

def fst_build_initial_single_ellipse(gt_mask: np.ndarray) -> tuple[float, float, float, float, float]:
    ellipse = fst_fit_ellipse_from_mask(gt_mask)
    if ellipse is None:
        ys, xs = np.where(gt_mask > 0)
        if len(xs) == 0:
            ellipse = (0.5, 0.5, 1.0, 1.0, 0.0)
        else:
            x0 = float(xs.min())
            x1 = float(xs.max())
            y0 = float(ys.min())
            y1 = float(ys.max())
            ellipse = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(1.0, (x1 - x0 + 1.0) / 2.0), max(1.0, (y1 - y0 + 1.0) / 2.0), 0.0)
    return fst_normalize_ellipse(ellipse)

def fst_reflect_points_across_sides(points_xy: np.ndarray, shape: tuple[int, int], touches: dict[str, bool]) -> list[tuple[str, np.ndarray]]:
    if len(points_xy) == 0:
        return []
    height, width = shape
    x_last = float(width - 1)
    y_last = float(height - 1)
    candidates: list[tuple[str, np.ndarray]] = []

    def reflected_points(active_sides: tuple[str, ...]) -> np.ndarray:
        pts = points_xy.copy()
        for side in active_sides:
            if side == 'left':
                pts[:, 0] = -pts[:, 0]
            elif side == 'right':
                pts[:, 0] = 2.0 * x_last - pts[:, 0]
            elif side == 'top':
                pts[:, 1] = -pts[:, 1]
            elif side == 'bottom':
                pts[:, 1] = 2.0 * y_last - pts[:, 1]
        return pts
    for side in ('left', 'right', 'top', 'bottom'):
        if touches[side]:
            candidates.append((f'reflect_{side}', reflected_points((side,))))
    for pair in (('left', 'top'), ('left', 'bottom'), ('right', 'top'), ('right', 'bottom')):
        if all((touches[side] for side in pair)):
            candidates.append((f'reflect_{pair[0]}_{pair[1]}', reflected_points(pair)))
    return candidates

def fst_mask_contour_points(gt_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        return np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
    ys, xs = np.where(gt_mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)

def fst_build_edge_aware_initial_candidates(gt_mask: np.ndarray, base_ellipse: tuple[float, float, float, float, float]) -> list[tuple[str, tuple[float, float, float, float, float]]]:
    points_xy = fst_mask_contour_points(gt_mask)
    touches = fst_detect_edge_touches(gt_mask)
    if not any(touches.values()) or len(points_xy) < 5:
        return [('base', fst_normalize_ellipse(base_ellipse))]
    candidates: list[tuple[str, tuple[float, float, float, float, float]]] = [('base', fst_normalize_ellipse(base_ellipse))]
    seen = {tuple((round(v, 3) for v in fst_normalize_ellipse(base_ellipse)))}

    def push(name: str, ellipse: tuple[float, float, float, float, float] | None) -> None:
        if ellipse is None:
            return
        norm = fst_normalize_ellipse(ellipse)
        key = tuple((round(v, 3) for v in norm))
        if key in seen:
            return
        seen.add(key)
        candidates.append((name, norm))
    for name, reflected in fst_reflect_points_across_sides(points_xy, gt_mask.shape, touches):
        push(name, fst_fit_ellipse_from_points(np.concatenate([points_xy, reflected], axis=0)))
    cx, cy, a, b, angle = fst_normalize_ellipse(base_ellipse)
    outward_dx = 0.0
    outward_dy = 0.0
    if touches['left']:
        outward_dx -= max(2.0, a * 0.12)
    if touches['right']:
        outward_dx += max(2.0, a * 0.12)
    if touches['top']:
        outward_dy -= max(2.0, b * 0.12)
    if touches['bottom']:
        outward_dy += max(2.0, b * 0.12)
    if outward_dx != 0.0 or outward_dy != 0.0:
        for factor in (1.0, 1.8, 2.6):
            push(f'outward_shift_{factor:.1f}', (cx + outward_dx * factor, cy + outward_dy * factor, a, b, angle))
            push(f'outward_shift_{factor:.1f}_wider', (cx + outward_dx * factor, cy + outward_dy * factor, a * 1.08, b * 1.04, angle))
            push(f'outward_shift_{factor:.1f}_thinner', (cx + outward_dx * factor, cy + outward_dy * factor, max(1.0, a * 0.95), max(1.0, b * 0.98), angle))
    return candidates

def fst_evaluate_single_ellipse(gt_mask: np.ndarray, ellipse: tuple[float, float, float, float, float], gt_area: int | None=None) -> tuple[dict[str, float], np.ndarray]:
    pred_mask = fst_render_ellipses(gt_mask.shape, [ellipse], [1.0])
    if gt_area is None:
        gt_area = int(cv2.countNonZero(gt_mask))
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection = int(cv2.countNonZero(cv2.bitwise_and(gt_mask, pred_mask)))
    union = gt_area + pred_area - intersection
    iou = intersection / union if union > 0 else 1.0
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    return ({'iou': iou, 'recall': recall, 'precision': precision, 'intersection': float(intersection), 'union': float(union), 'pred_area': float(pred_area), 'gt_area': float(gt_area)}, pred_mask)

def fst_refine_edge_outward(gt_mask: np.ndarray, ellipse: tuple[float, float, float, float, float], recall_target: float, touches: dict[str, bool], max_rounds: int=12, gt_area: int | None=None) -> tuple[tuple[float, float, float, float, float], dict[str, float]]:
    current = list(fst_normalize_ellipse(ellipse))
    if gt_area is None:
        gt_area = int(cv2.countNonZero(gt_mask))

    def evaluate(candidate: list[float]) -> tuple[float, dict[str, float]]:
        norm = fst_normalize_ellipse(tuple(candidate))
        metrics, _ = fst_evaluate_single_ellipse(gt_mask, norm, gt_area=gt_area)
        score = fst_candidate_score(metrics['iou'], metrics['recall'], recall_target)
        fn_pixels = int(metrics['gt_area'] - metrics['intersection'])
        fp_pixels = int(metrics['pred_area'] - metrics['intersection'])
        score -= 1e-06 * float(2 * fn_pixels + fp_pixels)
        return (score, metrics)
    best_score, best_metrics = evaluate(current)
    step_shift_x = max(2.0, current[2] * 0.18)
    step_shift_y = max(2.0, current[3] * 0.18)
    step_radius = max(1.0, max(current[2], current[3]) * 0.08)
    x_dir = (-1.0 if touches['left'] else 0.0) + (1.0 if touches['right'] else 0.0)
    y_dir = (-1.0 if touches['top'] else 0.0) + (1.0 if touches['bottom'] else 0.0)
    for _ in range(max_rounds):
        improved = False
        trials: list[list[float]] = []
        if x_dir != 0.0:
            trials.extend([[current[0] + x_dir * step_shift_x, current[1], current[2], current[3], current[4]], [current[0] + x_dir * step_shift_x, current[1], current[2] + step_radius, current[3], current[4]], [current[0] + x_dir * step_shift_x, current[1], max(1.0, current[2] - step_radius), current[3], current[4]]])
        if y_dir != 0.0:
            trials.extend([[current[0], current[1] + y_dir * step_shift_y, current[2], current[3], current[4]], [current[0], current[1] + y_dir * step_shift_y, current[2], current[3] + step_radius, current[4]], [current[0], current[1] + y_dir * step_shift_y, current[2], max(1.0, current[3] - step_radius), current[4]]])
        if x_dir != 0.0 and y_dir != 0.0:
            trials.append([current[0] + x_dir * step_shift_x, current[1] + y_dir * step_shift_y, current[2], current[3], current[4]])
        for trial in trials:
            score, metrics = evaluate(trial)
            if score > best_score + 1e-07:
                current = trial
                best_score = score
                best_metrics = metrics
                improved = True
        if not improved:
            step_shift_x *= 0.5
            step_shift_y *= 0.5
            step_radius *= 0.5
            if step_shift_x < 0.5 and step_shift_y < 0.5 and (step_radius < 0.5):
                break
    return (fst_normalize_ellipse(tuple(current)), best_metrics)

def fst_solve_single_ellipse(gt_mask: np.ndarray, recall_target: float, refinement_rounds: int=4) -> tuple[tuple[float, float, float, float, float], dict[str, float]]:
    gt_area = int(cv2.countNonZero(gt_mask))
    base_ellipse = fst_build_initial_single_ellipse(gt_mask)
    touches = fst_detect_edge_touches(gt_mask)
    initial_candidates = fst_build_edge_aware_initial_candidates(gt_mask, base_ellipse)
    pre_ranked: list[tuple[float, str, tuple[float, float, float, float, float], dict[str, float]]] = []
    for candidate_name, ellipse in initial_candidates:
        scales, _, _ = fst_optimize_candidate_scales(gt_mask, [ellipse], recall_target=recall_target)
        baked = fst_apply_scales_to_ellipses([ellipse], scales)[0]
        metrics, _ = fst_evaluate_single_ellipse(gt_mask, baked, gt_area=gt_area)
        pre_ranked.append((fst_candidate_score(metrics['iou'], metrics['recall'], recall_target), candidate_name, baked, metrics))
    pre_ranked.sort(key=lambda item: item[0], reverse=True)
    shortlisted = pre_ranked[:min(4, len(pre_ranked))]
    best_ellipse = shortlisted[0][2]
    best_metrics = shortlisted[0][3]
    best_score = shortlisted[0][0]
    for _, _, ellipse, _ in shortlisted:
        refined = ellipse
        refined_metrics, _ = fst_evaluate_single_ellipse(gt_mask, refined, gt_area=gt_area)
        if refinement_rounds > 0:
            refined_list, _, _ = fst_refine_ellipses_locally(gt_mask, [ellipse], recall_target=recall_target, max_rounds=refinement_rounds)
            refined = refined_list[0]
            refined_metrics, _ = fst_evaluate_single_ellipse(gt_mask, refined, gt_area=gt_area)
        if any(touches.values()):
            edge_refined, edge_metrics = fst_refine_edge_outward(gt_mask, refined, recall_target=recall_target, touches=touches, max_rounds=12, gt_area=gt_area)
            if fst_candidate_score(edge_metrics['iou'], edge_metrics['recall'], recall_target) > fst_candidate_score(refined_metrics['iou'], refined_metrics['recall'], recall_target) + 1e-09:
                refined = edge_refined
                refined_metrics = edge_metrics
        if refined_metrics['recall'] < recall_target:
            refined_scales, _, _ = fst_optimize_candidate_scales(gt_mask, [refined], recall_target=recall_target)
            refined = fst_apply_scales_to_ellipses([refined], refined_scales)[0]
            refined_metrics, _ = fst_evaluate_single_ellipse(gt_mask, refined, gt_area=gt_area)
        refined_score = fst_candidate_score(refined_metrics['iou'], refined_metrics['recall'], recall_target)
        if refined_score > best_score + 1e-09:
            best_ellipse = refined
            best_metrics = refined_metrics
            best_score = refined_score
    return (best_ellipse, best_metrics)

def fst_solve_k1_row(polygons_json: str, recall_target: float, exact_refine_rounds: int, prepared_payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]] | None=None, gt_polys: list[np.ndarray] | None=None) -> tuple[str, dict[str, float], str, list[tuple[float, float, float, float, float]]]:
    if prepared_payload is None:
        gt_mask, origin = fst_rasterize_polygons_to_local_mask(polygons_json)
    else:
        gt_mask, origin = fst_rasterize_local_mask_from_payload(prepared_payload)
    touches = fst_detect_edge_touches(gt_mask)
    if any(touches.values()):
        gt_area = int(cv2.countNonZero(gt_mask))
        ellipse, _ = fst_solve_single_ellipse(gt_mask, recall_target=recall_target, refinement_rounds=4)
        ellipse, metrics = fst_refine_edge_outward(gt_mask, ellipse, recall_target=recall_target, touches=touches, max_rounds=12, gt_area=gt_area)
        if metrics['recall'] < recall_target:
            scales, _, _ = fst_optimize_candidate_scales(gt_mask, [ellipse], recall_target=recall_target)
            ellipse = fst_apply_scales_to_ellipses([ellipse], scales)[0]
        absolute = fst_shift_ellipses_to_absolute([ellipse], origin)
        pred_polys = fst_ellipses_to_polygon_arrays(absolute)
        pred_json = json.dumps([poly.astype(np.float64).tolist() for poly in pred_polys])
        exact = fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys) if gt_polys is not None else fst_compute_exact_metrics(polygons_json, pred_json)
        exact['weighted_error'] = float(fst_compute_weighted_error(exact))
        return (pred_json, exact, 'edge_aggressive', absolute)
    single = fst_fit_ellipse_from_mask(gt_mask)
    if single is None:
        ys, xs = np.where(gt_mask > 0)
        if len(xs) == 0:
            ellipse = (0.0, 0.0, 1.0, 1.0, 0.0)
        else:
            ellipse = (float(xs.mean()), float(ys.mean()), 1.0, 1.0, 0.0)
        ellipses = [ellipse]
        candidate_name = 'fallback_point'
    else:
        scales, _, _ = fst_optimize_candidate_scales(gt_mask, [single], recall_target=recall_target)
        ellipses = fst_apply_scales_to_ellipses([single], scales)
        candidate_name = 'single_fit'
    if exact_refine_rounds > 0:
        refined, refined_iou, refined_recall = fst_refine_ellipses_locally(gt_mask, ellipses, recall_target=recall_target, max_rounds=exact_refine_rounds)
        base_mask = fst_render_ellipses(gt_mask.shape, ellipses, [1.0])
        base_iou, base_recall = fst_compute_mask_metrics(gt_mask, base_mask)
        if refined_recall >= base_recall and refined_iou >= base_iou:
            ellipses = refined
            candidate_name = 'single_fit_refined'
    absolute = fst_shift_ellipses_to_absolute(ellipses, origin)
    pred_polys = fst_ellipses_to_polygon_arrays(absolute)
    pred_json = json.dumps([poly.astype(np.float64).tolist() for poly in pred_polys])
    exact = fst_compute_exact_metrics_from_polygons(gt_polys, pred_polys) if gt_polys is not None else fst_compute_exact_metrics(polygons_json, pred_json)
    exact['weighted_error'] = float(fst_compute_weighted_error(exact))
    return (pred_json, exact, candidate_name, absolute)

def fst__k1_pool_init() -> None:
    cv2.setNumThreads(1)

def fst__solve_k1_row_worker(task: tuple[int, int, str, float, int]) -> tuple[int, tuple[int, str, str], dict[str, object], tuple[tuple[int, str], dict[str, object]]]:
    idx, frame, track_id, recall_target, exact_refine_rounds = task
    polygons_json = fst__ROW_POLYGONS_JSONS[idx]
    prepared_payload = fst__ROW_LOCAL_RASTER_PAYLOADS[idx]
    gt_polys = fst__ROW_GT_POLYGONS[idx]
    pred_json, exact, candidate_name, ellipses = fst_solve_k1_row(polygons_json, recall_target=recall_target, exact_refine_rounds=exact_refine_rounds, prepared_payload=prepared_payload, gt_polys=gt_polys)
    weighted_error = int(exact['weighted_error'])
    metric_row = {'frame': frame, 'track_id': track_id, 'candidate_name': candidate_name, 'gt_area': int(exact['gt_area']), 'pred_area': int(exact['pred_area']), 'intersection': int(exact['intersection']), 'union': int(exact['union']), 'recall': float(exact['recall']), 'precision': float(exact['precision']), 'iou': float(exact['iou']), 'weighted_error': weighted_error, 'ellipse_params': json.dumps(fst_serialize_ellipses(ellipses))}
    solution = {'pred_json': pred_json, 'ellipses': ellipses, 'metrics': dict(exact), 'candidate_name': candidate_name}
    return (idx, (frame, track_id, pred_json), metric_row, ((frame, track_id), solution))

def fst_determine_k1_workers(requested_workers: int, row_count: int) -> int:
    if row_count < 32:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested_workers > 0:
        return max(1, min(requested_workers, cpu_count))
    return max(1, min(cpu_count, row_count))

def fst__precompute_k2_ranked_candidate_worker(task: tuple[int, tuple[int, str], float]) -> tuple[int, tuple[int, str], list[tuple[float, str, list[tuple[float, float, float, float, float]]]]]:
    idx, key, recall_target = task
    gt_mask, _ = fst_rasterize_local_mask_from_payload(fst__ROW_LOCAL_RASTER_PAYLOADS[idx])
    ranked = build_k2_ranked_candidates(gt_mask, recall_target=recall_target)
    return (idx, key, ranked)

def fst_determine_k2_precompute_workers(requested_workers: int, selected_count: int) -> int:
    if selected_count < 32:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested_workers > 0:
        return max(1, min(requested_workers, cpu_count))
    return max(1, min(cpu_count, selected_count))

def fst_solve_k2_selected_rows(track_rows: list[tuple[int, str, str, int]], selected_keys: set[tuple[int, str]], device: torch.device, recall_target: float, downsample_factor: int, steps: int, early_stop_patience: int, early_stop_min_delta: float, early_stop_min_steps: int, max_candidates: int, max_prev_gap: int, ranked_candidates_lookup: dict[tuple[int, str], list[tuple[float, str, list[tuple[float, float, float, float, float]]]]] | None=None, prepared_context_lookup: dict[tuple[int, str], dict[str, object]] | None=None, reverse: bool=False) -> dict[tuple[int, str], dict[str, object]]:
    ordered_rows = list(reversed(track_rows)) if reverse else list(track_rows)
    previous_solution: list[tuple[float, float, float, float, float]] | None = None
    previous_frame: int | None = None
    solved: dict[tuple[int, str], dict[str, object]] = {}
    for frame, track_id, polygons_json, row_idx in ordered_rows:
        key = (int(frame), str(track_id))
        if key not in selected_keys:
            continue
        cached_context = None if prepared_context_lookup is None else prepared_context_lookup.get(key)
        if cached_context is None:
            gt_mask, origin = fst_rasterize_local_mask_from_payload(fst__ROW_LOCAL_RASTER_PAYLOADS[row_idx])
        else:
            gt_mask = np.asarray(cached_context['gt_mask'], dtype=np.uint8)
            origin = tuple(cached_context['origin'])
        previous = None
        if previous_solution is not None and previous_frame is not None and (abs(int(frame) - previous_frame) <= max_prev_gap):
            previous = previous_solution
        if ranked_candidates_lookup is None or key not in ranked_candidates_lookup:
            candidates = build_k2_initial_candidates(gt_mask, prev_absolute_ellipses=previous, origin=origin, recall_target=recall_target, max_candidates=max_candidates)
        else:
            ranked = list(ranked_candidates_lookup[key])
            if previous is not None:
                prev_local = fst_ensure_two_ellipses(fst_shift_ellipses_to_local(previous, origin))
                ranked.append(score_k2_candidate(gt_mask, 'prev_track', prev_local, recall_target=recall_target))
                ranked.sort(key=lambda item: item[0], reverse=True)
            candidates = [(name, ellipses) for _, name, ellipses in ranked[:max_candidates]]
        ellipses, candidate_name, iou, recall = optimize_k2_candidates_gpu(gt_mask, candidates, recall_target=recall_target, device=device, downsample_factor=downsample_factor, steps=steps, early_stop_patience=early_stop_patience, early_stop_min_delta=early_stop_min_delta, early_stop_min_steps=early_stop_min_steps, prepared_context=None if cached_context is None else dict(cached_context['opt']))
        absolute_ellipses = fst_shift_ellipses_to_absolute(ellipses, origin)
        solved[key] = {'ellipses': absolute_ellipses, 'candidate_name': candidate_name, 'local_iou': iou, 'local_recall': recall}
        previous_solution = absolute_ellipses
        previous_frame = int(frame)
    return solved

def fst_load_rows(sqlite_path: Path) -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    rows = [(int(frame), str(track_id), str(polygons)) for frame, track_id, polygons in conn.execute('SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id')]
    conn.close()
    return rows

def fst_load_k1_cost_lookup(csv_path: Path) -> dict[tuple[int, str], int]:
    lookup: dict[tuple[int, str], int] = {}
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track_id = str(row['track_id'])
            lookup[(frame, track_id)] = int(float(row['weighted_error']))
    return lookup

def fst_draw_outlines(img: np.ndarray, polygons_json: str, color: tuple[int, int, int], thickness: int) -> None:
    for pts in fst_parse_polygons(polygons_json):
        cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

def fst_blend_mask(img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    idx = mask > 0
    if np.any(idx):
        img[idx] = (img[idx].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)

def fst_get_annotation_anchor(polygons_json: str, width: int, height: int) -> tuple[int, int]:
    polygons = fst_parse_polygons(polygons_json)
    all_pts = np.concatenate(polygons, axis=0)
    min_xy = all_pts.min(axis=0)
    x = int(np.clip(math.floor(float(min_xy[0])), 8, max(8, width - 8)))
    y = int(np.clip(math.floor(float(min_xy[1])) - 10, 20, max(20, height - 8)))
    return x, y

def fst_open_nvenc_writer(output_video: str, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        'ffmpeg',
        '-y',
        '-loglevel',
        'error',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        '-s',
        f'{width}x{height}',
        '-r',
        f'{fps:.8f}',
        '-i',
        '-',
        '-an',
        '-c:v',
        'h264_nvenc',
        '-preset',
        'p5',
        '-rc',
        'vbr',
        '-cq',
        '23',
        '-pix_fmt',
        'yuv420p',
        output_video,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

def fst_load_rows_by_track(sqlite_path: Path) -> dict[str, list[tuple[int, str, str]]]:
    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute('SELECT frame, track_id, polygons FROM masks ORDER BY track_id, frame').fetchall()
    conn.close()
    by_track: dict[str, list[tuple[int, str, str]]] = {}
    for frame, track_id, polygons in rows:
        by_track.setdefault(str(track_id), []).append((int(frame), str(track_id), str(polygons)))
    return by_track


def fst_load_sqlite_mask_metadata(reference_sqlite: Path | None) -> tuple[dict[tuple[int, str], dict[str, object]], dict[str, dict[str, object]], list[int]]:
    frame_track_meta: dict[tuple[int, str], dict[str, object]] = {}
    track_meta: dict[str, dict[str, object]] = {}
    cut_frames: list[int] = []
    if reference_sqlite is None:
        return frame_track_meta, track_meta, cut_frames
    ref_path = Path(reference_sqlite)
    if not ref_path.exists():
        return frame_track_meta, track_meta, cut_frames
    conn = sqlite3.connect(str(ref_path))
    conn.row_factory = sqlite3.Row
    try:
        mask_tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='masks'").fetchone()
        if mask_tables is not None:
            mask_columns = {str(row['name']) for row in conn.execute('PRAGMA table_info(masks)').fetchall()}
            if {'frame', 'track_id', 'polygons'}.issubset(mask_columns):
                select_parts = [
                    'frame',
                    'track_id',
                    ('shape_type' if 'shape_type' in mask_columns else "'polygon' AS shape_type"),
                    ('dilate_px' if 'dilate_px' in mask_columns else '0 AS dilate_px'),
                    ('feather_px' if 'feather_px' in mask_columns else '0 AS feather_px'),
                    ('mosaic_block' if 'mosaic_block' in mask_columns else '0 AS mosaic_block'),
                    ('mosaic_alias' if 'mosaic_alias' in mask_columns else '0.0 AS mosaic_alias'),
                    ('label' if 'label' in mask_columns else 'NULL AS label'),
                    ('is_endpoint_extrapolated' if 'is_endpoint_extrapolated' in mask_columns else '0 AS is_endpoint_extrapolated'),
                ]
                for row in conn.execute(f"SELECT {', '.join(select_parts)} FROM masks").fetchall():
                    track_id = str(row['track_id'])
                    meta = {
                        'shape_type': str(row['shape_type']) if row['shape_type'] is not None else 'polygon',
                        'dilate_px': int(row['dilate_px']) if row['dilate_px'] is not None else 0,
                        'feather_px': int(row['feather_px']) if row['feather_px'] is not None else 0,
                        'mosaic_block': int(row['mosaic_block']) if row['mosaic_block'] is not None else 0,
                        'mosaic_alias': float(row['mosaic_alias']) if row['mosaic_alias'] is not None else 0.0,
                        'label': str(row['label']) if row['label'] is not None else None,
                        'is_endpoint_extrapolated': int(row['is_endpoint_extrapolated']) if row['is_endpoint_extrapolated'] is not None else 0,
                    }
                    frame_track_meta[(int(row['frame']), track_id)] = meta
                    if track_id not in track_meta:
                        track_defaults = dict(meta)
                        track_defaults['is_endpoint_extrapolated'] = 0
                        track_meta[track_id] = track_defaults
                    elif track_meta[track_id].get('label') is None and meta.get('label') is not None:
                        track_meta[track_id]['label'] = meta.get('label')
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'").fetchone() is not None:
            track_columns = {str(row['name']) for row in conn.execute('PRAGMA table_info(tracks)').fetchall()}
            if 'track_id' in track_columns:
                select_sql = "SELECT track_id, {} FROM tracks".format('label' if 'label' in track_columns else 'NULL AS label')
                for row in conn.execute(select_sql).fetchall():
                    track_id = str(row['track_id'])
                    info = track_meta.setdefault(track_id, {})
                    if row['label'] is not None:
                        info['label'] = str(row['label'])
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cuts'").fetchone() is not None:
            cut_frames = [int(row['frame']) for row in conn.execute('SELECT frame FROM cuts ORDER BY frame').fetchall()]
    finally:
        conn.close()
    return frame_track_meta, track_meta, cut_frames


def fst_write_sqlite(rows: list[tuple[int, str, str]], output_path: Path, reference_sqlite: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    frame_track_meta, track_meta, cut_frames = fst_load_sqlite_mask_metadata(reference_sqlite)
    conn = sqlite3.connect(str(output_path))
    try:
        conn.execute(
            '''
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                is_endpoint_extrapolated INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(frame, track_id)
            )
            '''
        )
        conn.execute('CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)')
        conn.execute('CREATE TABLE cuts(frame INTEGER PRIMARY KEY)')
        cur = conn.cursor()
        seen_tracks: dict[str, str | None] = {}
        for frame, track_id, polygons_json in rows:
            key = (int(frame), str(track_id))
            meta = frame_track_meta.get(key)
            if meta is None:
                meta = track_meta.get(str(track_id), {})
            shape_type = str(meta.get('shape_type') or 'polygon')
            dilate_px = int(meta.get('dilate_px') or 0)
            feather_px = int(meta.get('feather_px') or 0)
            mosaic_block = int(meta.get('mosaic_block') or 0)
            mosaic_alias = float(meta.get('mosaic_alias') or 0.0)
            label = meta.get('label')
            is_endpoint_extrapolated = int(meta.get('is_endpoint_extrapolated') or 0)
            cur.execute(
                'INSERT INTO masks(frame, track_id, polygons, shape_type, dilate_px, feather_px, mosaic_block, mosaic_alias, label, is_endpoint_extrapolated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (int(frame), str(track_id), str(polygons_json), shape_type, dilate_px, feather_px, mosaic_block, mosaic_alias, label, is_endpoint_extrapolated),
            )
            if str(track_id) not in seen_tracks:
                seen_tracks[str(track_id)] = str(label) if label is not None else None
        if seen_tracks:
            cur.executemany(
                'INSERT OR REPLACE INTO tracks(track_id, label) VALUES (?, ?)',
                [(track_id, label) for track_id, label in sorted(seen_tracks.items())],
            )
        if cut_frames:
            cur.executemany('INSERT OR IGNORE INTO cuts(frame) VALUES (?)', [(int(frame),) for frame in cut_frames])
        conn.commit()
    finally:
        conn.close()

def fst_evaluate_submission(gt_rows: list[tuple[int, str, str]], submission_rows: list[tuple[int, str, str]]) -> dict[str, float]:
    sub_lookup = {(frame, track_id): polygons for frame, track_id, polygons in submission_rows}
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    k1_intersection = k1_union = k1_gt_area = k1_pred_area = 0
    k2_intersection = k2_union = k2_gt_area = k2_pred_area = 0
    recall_below_090 = 0
    recall_below_095 = 0
    missing_rows = 0
    mean_recall = []
    mean_precision = []
    mean_iou = []
    k1_count = 0
    k2_count = 0
    for frame, track_id, gt_json in gt_rows:
        pred_json = sub_lookup.get((frame, track_id))
        if pred_json is None:
            missing_rows += 1
            continue
        metrics = fst_compute_exact_metrics(gt_json, pred_json)
        total_intersection += int(metrics['intersection'])
        total_union += int(metrics['union'])
        total_gt_area += int(metrics['gt_area'])
        total_pred_area += int(metrics['pred_area'])
        mean_recall.append(metrics['recall'])
        mean_precision.append(metrics['precision'])
        mean_iou.append(metrics['iou'])
        poly_count = len(json.loads(pred_json))
        if poly_count >= 2:
            k2_count += 1
            k2_intersection += int(metrics['intersection'])
            k2_union += int(metrics['union'])
            k2_gt_area += int(metrics['gt_area'])
            k2_pred_area += int(metrics['pred_area'])
        else:
            k1_count += 1
            k1_intersection += int(metrics['intersection'])
            k1_union += int(metrics['union'])
            k1_gt_area += int(metrics['gt_area'])
            k1_pred_area += int(metrics['pred_area'])
        if metrics['recall'] < 0.9:
            recall_below_090 += 1
        if metrics['recall'] < 0.95:
            recall_below_095 += 1
    return {'global_recall': total_intersection / total_gt_area if total_gt_area else 1.0, 'global_precision': total_intersection / total_pred_area if total_pred_area else 1.0, 'global_iou': total_intersection / total_union if total_union else 1.0, 'mean_recall': float(np.mean(mean_recall)) if mean_recall else 1.0, 'mean_precision': float(np.mean(mean_precision)) if mean_precision else 1.0, 'mean_iou': float(np.mean(mean_iou)) if mean_iou else 1.0, 'recall_below_090': int(recall_below_090), 'recall_below_095': int(recall_below_095), 'missing_rows': int(missing_rows), 'total_gt_rows': len(gt_rows), 'total_sub_rows': len(submission_rows), 'k1_count': int(k1_count), 'k2_count': int(k2_count), 'k1_recall': k1_intersection / k1_gt_area if k1_gt_area else 0.0, 'k1_iou': k1_intersection / k1_union if k1_union else 0.0, 'k2_recall': k2_intersection / k2_gt_area if k2_gt_area else 0.0, 'k2_iou': k2_intersection / k2_union if k2_union else 0.0}

def fst_evaluate_k1_metric_rows(metric_rows: list[dict[str, object]], total_gt_rows: int | None=None, total_sub_rows: int | None=None) -> dict[str, float]:
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    recall_below_090 = 0
    recall_below_095 = 0
    mean_recall: list[float] = []
    mean_precision: list[float] = []
    mean_iou: list[float] = []
    for row in metric_rows:
        intersection = int(row['intersection'])
        union = int(row['union'])
        gt_area = int(row['gt_area'])
        pred_area = int(row['pred_area'])
        recall = float(row['recall'])
        precision = float(row['precision'])
        iou = float(row['iou'])
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
    if total_gt_rows is None:
        total_gt_rows = len(metric_rows)
    if total_sub_rows is None:
        total_sub_rows = len(metric_rows)
    return {'global_recall': total_intersection / total_gt_area if total_gt_area else 1.0, 'global_precision': total_intersection / total_pred_area if total_pred_area else 1.0, 'global_iou': total_intersection / total_union if total_union else 1.0, 'mean_recall': float(np.mean(mean_recall)) if mean_recall else 1.0, 'mean_precision': float(np.mean(mean_precision)) if mean_precision else 1.0, 'mean_iou': float(np.mean(mean_iou)) if mean_iou else 1.0, 'recall_below_090': int(recall_below_090), 'recall_below_095': int(recall_below_095), 'missing_rows': int(max(0, int(total_gt_rows) - len(metric_rows))), 'total_gt_rows': int(total_gt_rows), 'total_sub_rows': int(total_sub_rows), 'k1_count': int(len(metric_rows)), 'k2_count': 0, 'k1_recall': total_intersection / total_gt_area if total_gt_area else 0.0, 'k1_iou': total_intersection / total_union if total_union else 0.0, 'k2_recall': 0.0, 'k2_iou': 0.0}

def fst_summarize_weighted_errors(values: list[int], thresholds: list[int]) -> dict[str, object]:
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'threshold_counts': {}}

    def percentile(p: float) -> int:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return int(ordered[idx])
    return {'count': len(ordered), 'min': int(ordered[0]), 'p50': percentile(0.5), 'p75': percentile(0.75), 'p90': percentile(0.9), 'p95': percentile(0.95), 'p99': percentile(0.99), 'max': int(ordered[-1]), 'threshold_counts': {str(t): int(sum((v >= t for v in ordered))) for t in thresholds}}

def fst_serialize_ellipses(ellipses: list[tuple[float, float, float, float, float]]) -> list[list[float]]:
    return [[float(cx), float(cy), float(a), float(b), float(angle)] for cx, cy, a, b, angle in ellipses]

def fst_deserialize_ellipses(serialized: list[list[float]] | list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    return [(float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])) for values in serialized]

def fst_ellipse_area(ellipse: tuple[float, float, float, float, float]) -> float:
    _, _, a, b, _ = ellipse
    return float(max(a, 1.0) * max(b, 1.0))

def fst_composite_center_and_scale(ellipses: list[tuple[float, float, float, float, float]]) -> tuple[float, float, float]:
    if not ellipses:
        return (0.0, 0.0, 1.0)
    areas = np.asarray([max(fst_ellipse_area(ellipse), 1.0) for ellipse in ellipses], dtype=np.float64)
    weight_sum = float(areas.sum())
    cx = float(sum((ellipse[0] * area for ellipse, area in zip(ellipses, areas))) / weight_sum)
    cy = float(sum((ellipse[1] * area for ellipse, area in zip(ellipses, areas))) / weight_sum)
    scale = float(np.sqrt(weight_sum / max(len(ellipses), 1)))
    return (cx, cy, max(scale, 1.0))

def fst_angle_distance_deg(angle_a: float, angle_b: float) -> float:
    diff = abs((angle_a - angle_b) % 180.0)
    return float(min(diff, 180.0 - diff))

def fst_compute_local_metrics_for_local_ellipses(gt_mask: np.ndarray, local_ellipses: list[tuple[float, float, float, float, float]]) -> dict[str, float]:
    pred_mask = fst_render_ellipses(gt_mask.shape, local_ellipses, [1.0] * len(local_ellipses))
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = int((gt_mask | pred_mask).sum())
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    metrics = {'gt_area': float(gt_area), 'pred_area': float(pred_area), 'intersection': float(intersection), 'union': float(union), 'recall': float(recall), 'precision': float(precision), 'iou': float(iou)}
    metrics['weighted_error'] = float(fst_compute_weighted_error(metrics))
    return metrics

def fst_compute_local_metrics_for_absolute_ellipses(gt_mask: np.ndarray, origin: tuple[int, int], absolute_ellipses: list[tuple[float, float, float, float, float]]) -> dict[str, float]:
    return fst_compute_local_metrics_for_local_ellipses(gt_mask, fst_shift_ellipses_to_local(absolute_ellipses, origin))

def fst_build_k2_solve_band(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], k1_ellipses_lookup: dict[tuple[int, str], list[tuple[float, float, float, float, float]]], threshold: int, radius: int, error_percentile: float, instability_percentile: float, instability_floor: float) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        errors = np.asarray([float(k1_metrics_lookup[key]['weighted_error']) for key in keys], dtype=np.float64)
        seed_indices = {idx for idx, err in enumerate(errors) if err >= float(threshold)}
        high_error_cut = float(np.percentile(errors, error_percentile)) if len(errors) > 0 else float(threshold)
        instability_scores = np.zeros(len(track_rows), dtype=np.float64)
        for idx in range(1, len(track_rows)):
            prev_key = keys[idx - 1]
            curr_key = keys[idx]
            prev_ellipse = k1_ellipses_lookup[prev_key][0]
            curr_ellipse = k1_ellipses_lookup[curr_key][0]
            _, _, prev_scale = fst_composite_center_and_scale([prev_ellipse])
            _, _, curr_scale = fst_composite_center_and_scale([curr_ellipse])
            ref_scale = max(prev_scale, curr_scale, 8.0)
            center_jump = math.hypot(curr_ellipse[0] - prev_ellipse[0], curr_ellipse[1] - prev_ellipse[1]) / ref_scale
            area_jump = abs(math.log(max(fst_ellipse_area(curr_ellipse), 1.0)) - math.log(max(fst_ellipse_area(prev_ellipse), 1.0)))
            angle_jump = fst_angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
            instability_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
        instability_cut = float(np.percentile(instability_scores, instability_percentile)) if len(instability_scores) > 0 else float('inf')
        extra_indices = {idx for idx, err in enumerate(errors) if err >= max(high_error_cut, float(threshold) * 0.6)}
        extra_indices |= {idx for idx, score in enumerate(instability_scores) if score >= max(instability_cut, instability_floor)}
        source_indices = seed_indices | extra_indices
        for src_idx in source_indices:
            src_frame = int(track_rows[src_idx][0])
            for frame, track_id_value, _, _ in track_rows:
                if abs(int(frame) - src_frame) <= radius:
                    selected.add((int(frame), str(track_id_value)))
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'seed_count': len(seed_indices), 'expanded_count': int(sum((1 for key in keys if key in selected))), 'error_cut': float(high_error_cut), 'instability_cut': float(instability_cut) if np.isfinite(instability_cut) else None})
    summary = {'threshold': int(threshold), 'radius': int(radius), 'selected_count': len(selected), 'tracks': summary_tracks}
    return (selected, summary)


fst = _register_inline_module(
    'standalone_runtime_fst',
    {
    'WIDTH': 'fst_WIDTH',
    'HEIGHT': 'fst_HEIGHT',
    'POLYGON_POINTS': 'fst_POLYGON_POINTS',
    'MASK_COLOR': 'fst_MASK_COLOR',
    'ELLIPSE_COLOR': 'fst_ELLIPSE_COLOR',
    '_ANGLE_TABLE': 'fst__ANGLE_TABLE',
    '_KERNEL_CACHE': 'fst__KERNEL_CACHE',
    '_GRID_CACHE': 'fst__GRID_CACHE',
    '_ROW_POLYGONS_JSONS': 'fst__ROW_POLYGONS_JSONS',
    '_ROW_LOCAL_RASTER_PAYLOADS': 'fst__ROW_LOCAL_RASTER_PAYLOADS',
    '_ROW_GT_POLYGONS': 'fst__ROW_GT_POLYGONS',
    '_get_unit_circle': 'fst__get_unit_circle',
    'ellipse_to_polygon_array': 'fst_ellipse_to_polygon_array',
    'ellipse_to_polygon': 'fst_ellipse_to_polygon',
    'parse_polygons': 'fst_parse_polygons',
    'make_polygons_json': 'fst_make_polygons_json',
    'ellipses_to_polygon_arrays': 'fst_ellipses_to_polygon_arrays',
    'rasterize_full': 'fst_rasterize_full',
    'rasterize_full_from_polygons': 'fst_rasterize_full_from_polygons',
    'prepare_local_raster_payload_from_polygons': 'fst_prepare_local_raster_payload_from_polygons',
    'prepare_local_raster_payload': 'fst_prepare_local_raster_payload',
    'rasterize_local_mask_from_payload': 'fst_rasterize_local_mask_from_payload',
    'rasterize_polygons_to_local_mask': 'fst_rasterize_polygons_to_local_mask',
    'set_row_local_raster_cache': 'fst_set_row_local_raster_cache',
    'normalize_ellipse': 'fst_normalize_ellipse',
    'fit_ellipse_from_points': 'fst_fit_ellipse_from_points',
    'fit_ellipse_from_mask': 'fst_fit_ellipse_from_mask',
    'render_ellipses': 'fst_render_ellipses',
    'compute_mask_metrics': 'fst_compute_mask_metrics',
    'compute_exact_metrics': 'fst_compute_exact_metrics',
    'compute_exact_metrics_from_gt_polys': 'fst_compute_exact_metrics_from_gt_polys',
    'compute_exact_metrics_from_polygons': 'fst_compute_exact_metrics_from_polygons',
    'compute_weighted_error': 'fst_compute_weighted_error',
    'candidate_score': 'fst_candidate_score',
    'binary_search_scale': 'fst_binary_search_scale',
    'optimize_candidate_scales': 'fst_optimize_candidate_scales',
    'apply_scales_to_ellipses': 'fst_apply_scales_to_ellipses',
    'refine_ellipses_locally': 'fst_refine_ellipses_locally',
    'build_component_mask': 'fst_build_component_mask',
    '_fit_axis_split_candidates': 'fst__fit_axis_split_candidates',
    'generate_principal_axis_candidates': 'fst_generate_principal_axis_candidates',
    'select_distance_transform_peaks': 'fst_select_distance_transform_peaks',
    'distance_transform_candidate': 'fst_distance_transform_candidate',
    'shift_ellipses_to_local': 'fst_shift_ellipses_to_local',
    'shift_ellipses_to_absolute': 'fst_shift_ellipses_to_absolute',
    'ensure_two_ellipses': 'fst_ensure_two_ellipses',
    'downsample_mask': 'fst_downsample_mask',
    'detect_edge_touches': 'fst_detect_edge_touches',
    'build_initial_single_ellipse': 'fst_build_initial_single_ellipse',
    'reflect_points_across_sides': 'fst_reflect_points_across_sides',
    'mask_contour_points': 'fst_mask_contour_points',
    'build_edge_aware_initial_candidates': 'fst_build_edge_aware_initial_candidates',
    'evaluate_single_ellipse': 'fst_evaluate_single_ellipse',
    'refine_edge_outward': 'fst_refine_edge_outward',
    'solve_single_ellipse': 'fst_solve_single_ellipse',
    'solve_k1_row': 'fst_solve_k1_row',
    '_k1_pool_init': 'fst__k1_pool_init',
    '_solve_k1_row_worker': 'fst__solve_k1_row_worker',
    'determine_k1_workers': 'fst_determine_k1_workers',
    '_precompute_k2_ranked_candidate_worker': 'fst__precompute_k2_ranked_candidate_worker',
    'determine_k2_precompute_workers': 'fst_determine_k2_precompute_workers',
    'solve_k2_selected_rows': 'fst_solve_k2_selected_rows',
    'draw_outlines': 'fst_draw_outlines',
    'blend_mask': 'fst_blend_mask',
    'get_annotation_anchor': 'fst_get_annotation_anchor',
    'open_nvenc_writer': 'fst_open_nvenc_writer',
    'load_rows': 'fst_load_rows',
    'load_k1_cost_lookup': 'fst_load_k1_cost_lookup',
    'load_rows_by_track': 'fst_load_rows_by_track',
    'write_sqlite': 'fst_write_sqlite',
    'evaluate_submission': 'fst_evaluate_submission',
    'evaluate_k1_metric_rows': 'fst_evaluate_k1_metric_rows',
    'summarize_weighted_errors': 'fst_summarize_weighted_errors',
    'serialize_ellipses': 'fst_serialize_ellipses',
    'deserialize_ellipses': 'fst_deserialize_ellipses',
    'ellipse_area': 'fst_ellipse_area',
    'composite_center_and_scale': 'fst_composite_center_and_scale',
    'angle_distance_deg': 'fst_angle_distance_deg',
    'compute_local_metrics_for_local_ellipses': 'fst_compute_local_metrics_for_local_ellipses',
    'compute_local_metrics_for_absolute_ellipses': 'fst_compute_local_metrics_for_absolute_ellipses',
    'build_k2_solve_band': 'fst_build_k2_solve_band',
},
)

sys.modules['final_standalone_t5000'] = fst




# ==============================================================================
# Inlined from: standalone_runtime_k2v5.py
# ==============================================================================

import math
import cv2
import numpy as np
import torch
from torch import nn
import standalone_runtime_fst as fst
k2v5_ELLIPSE_CENTER_MIN = -0.75
k2v5_ELLIPSE_CENTER_MAX = 1.75
k2v5_LOG_AXIS_MIN = -9.0
k2v5_LOG_AXIS_MAX = 0.75
k2v5__COORD_GRID_CACHE: dict[int, np.ndarray] = {}
k2v5__BORDER_GRID_CACHE: dict[int, np.ndarray] = {}

def k2v5_square_pad_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    padded = np.zeros((side, side), dtype=np.uint8)
    padded[pad_top:pad_top + height, pad_left:pad_left + width] = mask
    return (padded, (pad_left, pad_top), side)

def k2v5_build_signed_distance_channel(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside - outside
    scale = float(max(mask.shape))
    return (signed / max(scale, 1.0)).astype(np.float32)

def k2v5_build_edge_channel(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return edge.astype(np.float32)

def k2v5_get_coord_grid(image_size: int) -> np.ndarray:
    cached = k2v5__COORD_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    stacked = np.stack([xx, yy], axis=0)
    k2v5__COORD_GRID_CACHE[image_size] = stacked
    return stacked

def k2v5_get_border_distance_grid(image_size: int) -> np.ndarray:
    cached = k2v5__BORDER_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    border = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy]).astype(np.float32)
    maxv = float(border.max()) if border.size else 1.0
    if maxv > 0.0:
        border /= maxv
    border = border[None, ...]
    k2v5__BORDER_GRID_CACHE[image_size] = border
    return border

def k2v5_build_touch_flag_planes(image_size: int, touch_flags: np.ndarray) -> np.ndarray:
    planes = np.broadcast_to(touch_flags.astype(np.float32)[:, None, None], (4, image_size, image_size))
    return np.asarray(planes, dtype=np.float32)

def k2v5_edge_touch_vector_from_row(row: dict[str, object], gt_mask: np.ndarray | None=None) -> np.ndarray:
    meta = row.get('source_metadata')
    if isinstance(meta, dict):
        sides = meta.get('edge_sides')
        if isinstance(sides, dict):
            return np.asarray([float(bool(sides.get('left', False))), float(bool(sides.get('right', False))), float(bool(sides.get('top', False))), float(bool(sides.get('bottom', False)))], dtype=np.float32)
    if gt_mask is None:
        return np.zeros((4,), dtype=np.float32)
    touches = fst.detect_edge_touches(gt_mask.astype(np.uint8))
    return np.asarray([float(bool(touches.get('left', False))), float(bool(touches.get('right', False))), float(bool(touches.get('top', False))), float(bool(touches.get('bottom', False)))], dtype=np.float32)

def k2v5_build_input_image(mask_resized: np.ndarray, signed_resized: np.ndarray, edge_resized: np.ndarray, touch_flags: np.ndarray, image_size: int) -> np.ndarray:
    coords = k2v5_get_coord_grid(image_size)
    border = k2v5_get_border_distance_grid(image_size)
    touch_planes = k2v5_build_touch_flag_planes(image_size, touch_flags)
    return np.concatenate([mask_resized[None, ...].astype(np.float32, copy=False), signed_resized[None, ...].astype(np.float32, copy=False), edge_resized[None, ...].astype(np.float32, copy=False), coords.astype(np.float32, copy=False), border.astype(np.float32, copy=False), touch_planes.astype(np.float32, copy=False)], axis=0)

def k2v5_states_to_abs_ellipses_from_payload(states: np.ndarray, payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]]) -> list[tuple[float, float, float, float, float]]:
    if states.ndim == 1:
        states = states.reshape(2, 6)
    (height, width), origin, _ = payload
    side = max(height, width)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    absolute: list[tuple[float, float, float, float, float]] = []
    for state in states:
        cx_n, cy_n, loga, logb, cos2, sin2 = [float(v) for v in state]
        cx_n = min(max(cx_n, k2v5_ELLIPSE_CENTER_MIN), k2v5_ELLIPSE_CENTER_MAX)
        cy_n = min(max(cy_n, k2v5_ELLIPSE_CENTER_MIN), k2v5_ELLIPSE_CENTER_MAX)
        loga = min(max(loga, k2v5_LOG_AXIS_MIN), k2v5_LOG_AXIS_MAX)
        logb = min(max(logb, k2v5_LOG_AXIS_MIN), k2v5_LOG_AXIS_MAX)
        norm = math.hypot(cos2, sin2)
        if not math.isfinite(norm) or norm < 1e-06:
            cos2, sin2 = (1.0, 0.0)
        else:
            cos2 /= norm
            sin2 /= norm
        cx = cx_n * side - pad_left
        cy = cy_n * side - pad_top
        a = math.exp(loga) * side
        b = math.exp(logb) * side
        angle = math.degrees(0.5 * math.atan2(sin2, cos2))
        absolute.append(fst.normalize_ellipse((cx + origin[0], cy + origin[1], a, b, angle)))
    return absolute

class k2v5_ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int=1, groups: int=1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class k2v5_SqueezeExcite(nn.Module):

    def __init__(self, channels: int, reduction: int=4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)

class k2v5_ResidualBlock(nn.Module):

    def __init__(self, channels: int, expansion: int=2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(k2v5_ConvBNAct(channels, hidden, 1), k2v5_ConvBNAct(hidden, hidden, 3, groups=hidden), k2v5_SqueezeExcite(hidden), nn.Conv2d(hidden, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class k2v5_SlotDecoderBlock(nn.Module):

    def __init__(self, dim: int=256, num_heads: int=8, mlp_ratio: int=4) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_q3 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q1(queries)
        c = self.norm_ctx(context)
        queries = queries + self.cross_attn(q, c, c, need_weights=False)[0]
        q = self.norm_q2(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.mlp(self.norm_q3(queries))
        return queries

def k2v5_render_soft_slots_from_spd(centers: torch.Tensor, chol_params: torch.Tensor, image_size: int, sharpness: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_slots, _ = centers.shape
    grid = torch.linspace(0.0, 1.0, image_size, device=centers.device, dtype=centers.dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    xx = xx.view(1, 1, image_size, image_size)
    yy = yy.view(1, 1, image_size, image_size)
    l11 = torch.nn.functional.softplus(chol_params[..., 0]).view(batch, n_slots, 1, 1) + 0.0001
    l21 = chol_params[..., 1].view(batch, n_slots, 1, 1)
    l22 = torch.nn.functional.softplus(chol_params[..., 2]).view(batch, n_slots, 1, 1) + 0.0001
    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22
    dx = xx - centers[..., 0].view(batch, n_slots, 1, 1)
    dy = yy - centers[..., 1].view(batch, n_slots, 1, 1)
    quad = a11 * dx * dx + 2.0 * a12 * dx * dy + a22 * dy * dy
    q = 1.0 - quad
    slot_masks = torch.sigmoid(q * sharpness)
    union = 1.0 - torch.prod(1.0 - slot_masks, dim=1)
    return (slot_masks, union)

def k2v5_spd_to_normalized_states(centers: torch.Tensor, chol_params: torch.Tensor) -> torch.Tensor:
    l11 = torch.nn.functional.softplus(chol_params[..., 0]) + 0.0001
    l21 = chol_params[..., 1]
    l22 = torch.nn.functional.softplus(chol_params[..., 2]) + 0.0001
    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22
    trace = a11 + a22
    disc = torch.sqrt(((a11 - a22) ** 2 + 4.0 * a12 * a12).clamp_min(1e-10))
    lam_min = ((trace - disc) * 0.5).clamp_min(1e-08)
    lam_max = ((trace + disc) * 0.5).clamp_min(1e-08)
    major = torch.rsqrt(lam_min).clamp_min(0.0001)
    minor = torch.rsqrt(lam_max).clamp_min(0.0001)
    det = (a11 * a22 - a12 * a12).clamp_min(1e-10)
    cov_xx = a22 / det
    cov_xy = -a12 / det
    cov_yy = a11 / det
    denom = torch.sqrt((cov_xx - cov_yy) ** 2 + (2.0 * cov_xy) ** 2).clamp_min(1e-08)
    cos2 = (cov_xx - cov_yy) / denom
    sin2 = 2.0 * cov_xy / denom
    return torch.stack([centers[..., 0], centers[..., 1], torch.log(major), torch.log(minor), cos2, sin2], dim=-1)

class k2v5_K2SlotSetSPDNet(nn.Module):

    def __init__(self, in_channels: int, base_width: int, slot_dim: int, decoder_layers: int, num_heads: int, sharpness: float) -> None:
        super().__init__()
        c1 = int(base_width)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 12
        self.sharpness = float(sharpness)
        self.stem = nn.Sequential(k2v5_ConvBNAct(in_channels, c1, 3, stride=1), k2v5_ResidualBlock(c1), k2v5_ResidualBlock(c1))
        self.stage2 = nn.Sequential(k2v5_ConvBNAct(c1, c2, 3, stride=2), k2v5_ResidualBlock(c2), k2v5_ResidualBlock(c2))
        self.stage3 = nn.Sequential(k2v5_ConvBNAct(c2, c3, 3, stride=2), k2v5_ResidualBlock(c3), k2v5_ResidualBlock(c3))
        self.stage4 = nn.Sequential(k2v5_ConvBNAct(c3, c4, 3, stride=2), k2v5_ResidualBlock(c4), k2v5_ResidualBlock(c4), k2v5_ResidualBlock(c4))
        self.stage5 = nn.Sequential(k2v5_ConvBNAct(c4, c5, 3, stride=2), k2v5_ResidualBlock(c5), k2v5_ResidualBlock(c5), k2v5_ResidualBlock(c5))
        self.lat5 = nn.Conv2d(c5, slot_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, slot_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, slot_dim, kernel_size=1)
        self.fpn4 = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.fpn3 = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.context_proj = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.slot_queries = nn.Parameter(torch.randn(2, slot_dim) * 0.02)
        self.decoder = nn.ModuleList([k2v5_SlotDecoderBlock(slot_dim, num_heads=num_heads, mlp_ratio=4) for _ in range(decoder_layers)])
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(slot_dim, slot_dim, 1), nn.SiLU(inplace=True))
        self.slot_refine = nn.Sequential(nn.Linear(slot_dim * 2, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.center_head = nn.Linear(slot_dim, 2)
        self.chol_head = nn.Linear(slot_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        p5 = self.lat5(s5)
        p4 = self.fpn4(self.lat4(s4) + torch.nn.functional.interpolate(p5, size=s4.shape[-2:], mode='bilinear', align_corners=False))
        p3 = self.fpn3(self.lat3(s3) + torch.nn.functional.interpolate(p4, size=s3.shape[-2:], mode='bilinear', align_corners=False))
        context_map = self.context_proj(p4)
        context_tokens = context_map.flatten(2).transpose(1, 2)
        queries = self.slot_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        global_feat = self.global_pool(torch.nn.functional.interpolate(p3, size=context_map.shape[-2:], mode='bilinear', align_corners=False))
        global_feat = global_feat.flatten(1).unsqueeze(1).expand(-1, 2, -1)
        queries = queries + self.slot_refine(torch.cat([queries, global_feat], dim=-1))
        for block in self.decoder:
            queries = block(queries, context_tokens)
        centers = self.center_head(queries)
        chol_params = self.chol_head(queries)
        slot_masks, union_mask = k2v5_render_soft_slots_from_spd(centers, chol_params, image_size=x.shape[-1], sharpness=self.sharpness)
        states = k2v5_spd_to_normalized_states(centers, chol_params)
        return {'states': states.reshape(x.shape[0], 12), 'centers': centers, 'chol_params': chol_params, 'slot_masks': slot_masks, 'union_mask': union_mask}

def k2v5_pair_cost(pred_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([2.0, 2.0, 1.5, 1.5, 0.5, 0.5], device=pred_states.device, dtype=pred_states.dtype)
    diff = torch.nn.functional.smooth_l1_loss(pred_states, target_states, reduction='none')
    return (diff * weights.view(1, 1, 6)).mean(dim=(1, 2))

def k2v5_states_to_geometry_tensors(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centers = states[..., :2]
    log_major = states[..., 2]
    log_minor = states[..., 3]
    cos2 = states[..., 4].clamp(-1.0, 1.0)
    sin2 = states[..., 5].clamp(-1.0, 1.0)
    cos_theta = torch.sqrt(((1.0 + cos2) * 0.5).clamp_min(1e-08))
    sin_theta = torch.sign(sin2) * torch.sqrt(((1.0 - cos2) * 0.5).clamp_min(1e-08))
    major2 = torch.exp(2.0 * log_major).clamp_min(1e-08)
    minor2 = torch.exp(2.0 * log_minor).clamp_min(1e-08)
    c2 = cos_theta * cos_theta
    s2 = sin_theta * sin_theta
    cs = cos_theta * sin_theta
    cov_xx = c2 * major2 + s2 * minor2
    cov_xy = cs * (major2 - minor2)
    cov_yy = s2 * major2 + c2 * minor2
    cov = torch.stack([cov_xx, cov_xy, cov_yy], dim=-1)
    return (centers, cov)


k2v5 = _register_inline_module(
    'standalone_runtime_k2v5',
    {
    'ELLIPSE_CENTER_MIN': 'k2v5_ELLIPSE_CENTER_MIN',
    'ELLIPSE_CENTER_MAX': 'k2v5_ELLIPSE_CENTER_MAX',
    'LOG_AXIS_MIN': 'k2v5_LOG_AXIS_MIN',
    'LOG_AXIS_MAX': 'k2v5_LOG_AXIS_MAX',
    '_COORD_GRID_CACHE': 'k2v5__COORD_GRID_CACHE',
    '_BORDER_GRID_CACHE': 'k2v5__BORDER_GRID_CACHE',
    'square_pad_mask': 'k2v5_square_pad_mask',
    'build_signed_distance_channel': 'k2v5_build_signed_distance_channel',
    'build_edge_channel': 'k2v5_build_edge_channel',
    'get_coord_grid': 'k2v5_get_coord_grid',
    'get_border_distance_grid': 'k2v5_get_border_distance_grid',
    'build_touch_flag_planes': 'k2v5_build_touch_flag_planes',
    'edge_touch_vector_from_row': 'k2v5_edge_touch_vector_from_row',
    'build_input_image': 'k2v5_build_input_image',
    'states_to_abs_ellipses_from_payload': 'k2v5_states_to_abs_ellipses_from_payload',
    'ConvBNAct': 'k2v5_ConvBNAct',
    'SqueezeExcite': 'k2v5_SqueezeExcite',
    'ResidualBlock': 'k2v5_ResidualBlock',
    'SlotDecoderBlock': 'k2v5_SlotDecoderBlock',
    'render_soft_slots_from_spd': 'k2v5_render_soft_slots_from_spd',
    'spd_to_normalized_states': 'k2v5_spd_to_normalized_states',
    'K2SlotSetSPDNet': 'k2v5_K2SlotSetSPDNet',
    'pair_cost': 'k2v5_pair_cost',
    'states_to_geometry_tensors': 'k2v5_states_to_geometry_tensors',
},
)




# ==============================================================================
# Inlined from: final_standalone_t5000_k1_exact_k2_v5.py
# ==============================================================================

"""Standard standalone inference entrypoint.

This script covers the full production path in one place:
- raw AI preprocessing from JSONL/video to tracked SQLite
- K1 exact ellipse approximation
- K1/K2 routing with the current V6 smoothing preset
- K2 V5 fallback inference
- final SQLite / metrics / summary export

If you want the standard mixed inference behavior for this bundle, use this
script directly.
"""
import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import cv2
import numpy as np
infer_ROOT = Path(__file__).resolve().parent
if str(infer_ROOT) not in sys.path:
    sys.path.insert(0, str(infer_ROOT))
try:
    import standalone_runtime_fst as fst
except ModuleNotFoundError:
    TEACHER_ROOT = infer_ROOT / 'Teacher'
    if str(TEACHER_ROOT) not in sys.path:
        sys.path.insert(0, str(TEACHER_ROOT))
    import final_standalone_t5000 as fst
infer__TORCH_MODULE: ModuleType | None = None
infer__K2V5_MODULE: ModuleType | None = None

def infer_get_torch_module() -> ModuleType:
    global infer__TORCH_MODULE
    if infer__TORCH_MODULE is None:
        import torch
        infer__TORCH_MODULE = torch
    return infer__TORCH_MODULE

def infer_get_k2v5_module() -> ModuleType:
    global infer__K2V5_MODULE
    if infer__K2V5_MODULE is None:
        try:
            import standalone_runtime_k2v5 as k2v5
        except ModuleNotFoundError:
            DISTILL_ROOT = infer_ROOT / 'Distillation'
            if str(DISTILL_ROOT) not in sys.path:
                sys.path.insert(0, str(DISTILL_ROOT))
            import train_k2_slot_set_spd_standalone_v5 as k2v5
        infer__K2V5_MODULE = k2v5
    return infer__K2V5_MODULE
infer_RAW_MAX_GAP_FRAMES = 15
infer_RAW_IOU_MIN = 0.06
infer_RAW_CENTER_DIST_MAX = 0.5
infer_RAW_AREA_RATIO_MIN = 0.2
infer_RAW_AREA_RATIO_MAX = 5.0
infer_RAW_ASPECT_RATIO_MAX = 1.3
infer_RAW_FILL_RATIO_MAX = 0.67
infer_RAW_POLY_RATIO_MIN = 0.25
infer_RAW_POLY_RATIO_MAX = 4.0
infer_RAW_SCORE_MIN = 0.2
infer_RAW_DET_SCORE_MIN = 0.35
infer_RAW_SMALL_AREA_THRESH = 5000.0
infer_RAW_TINY_AREA_THRESH = 2000.0
infer_RAW_SMALL_IOU_MIN = 0.03
infer_RAW_TINY_IOU_MIN = 0.01
infer_RAW_SMALL_CENTER_DIST_MAX = 0.8
infer_RAW_TINY_CENTER_DIST_MAX = 1.0
infer_RAW_SMALL_AREA_RATIO_MIN = 0.2
infer_RAW_SMALL_AREA_RATIO_MAX = 6.0
infer_RAW_TINY_AREA_RATIO_MIN = 0.1
infer_RAW_TINY_AREA_RATIO_MAX = 7.0
infer_RAW_SMALL_ASPECT_RATIO_MAX = 1.5
infer_RAW_TINY_ASPECT_RATIO_MAX = 1.7
infer_RAW_SMALL_SCORE_MIN = 0.15
infer_RAW_TINY_SCORE_MIN = 0.12
infer_RAW_NMS_IOU_THRESH = 0.2
infer_RAW_NMS_IOU_THRESH_SMALL = 0.1
infer_RAW_NMS_IOU_THRESH_TINY = 0.05
infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX = 8.0
infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX_SMALL = 5.0
infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX_TINY = 5.0
infer_RAW_NMS_CONTAIN_MARGIN = 2.0
infer_RAW_CUT_DIFF_THRESH = 18.0
infer_RAW_CUT_MIN_GAP_FRAMES = 15
infer_RAW_CUT_DOWNSCALE_W = 96
infer_RAW_CUT_DOWNSCALE_H = 54
infer_RAW_CUT_HYBRID_FFMPEG_CANDIDATE_THRESH = 18.0
infer_RAW_CUT_HYBRID_WINDOW_RADIUS = 3
infer_RAW_CUT_METHOD_DEFAULT = 'high_precision'
infer_RAW_CUT_METHODS = ('legacy_diff', 'high_precision')
infer_RAW_CUT_HP_MIN_DIFF = 18.0
infer_RAW_CUT_HP_NORMAL_DIFF = 30.0
infer_RAW_CUT_HP_STRONG_DIFF = 45.0
infer_RAW_CUT_HP_HARD_DIFF = 70.0
infer_RAW_CUT_HP_COLOR_CORR_MAX = 0.96
infer_RAW_CUT_HP_STRONG_COLOR_CORR_MAX = 0.98
infer_RAW_CUT_HP_SSIM_MAX = 0.45
infer_RAW_CUT_HP_STRONG_SSIM_MAX = 0.55
infer_RAW_CUT_HP_MIN_GAP_FRAMES = 45
infer_K1_COST_NORM_AREA_FLOOR = 1.0
infer_RAW_PROGRESS_EVERY = 3000
infer_K2_V5_INFER_CONFIG = {'image_size': 192, 'base_width': 32, 'slot_dim': 256, 'decoder_layers': 3, 'num_heads': 8, 'render_sharpness': 28.0}

def infer_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Standalone ellipse approximation: exact K1 split + K2 V5 distilled model.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-sqlite', type=Path)
    input_group.add_argument('--input-jsonl', type=Path, help='Raw AI detection JSONL. Runs built-in NMS, cut detection, tracking, and short-track removal before ellipse approximation.')
    parser.add_argument('--input-video', type=Path, default=None, help='Source video used for raw AI cut detection. Required for --input-jsonl unless a same-stem video is found next to the JSONL.')
    parser.add_argument('--raw-cut-detect', action=argparse.BooleanOptionalAction, default=True, help='Enable cut detection during built-in raw AI preprocessing. Enabled by default.')
    parser.add_argument('--raw-cut-method', choices=infer_RAW_CUT_METHODS, default=infer_RAW_CUT_METHOD_DEFAULT, help='Cut detector used during raw AI preprocessing.')
    parser.add_argument('--raw-remove-short-tracks-max-frames', type=int, default=10, help='Remove raw AI tracks with duration <= this many frames before ellipse approximation.')
    parser.add_argument('--raw-det-score-min', type=float, default=infer_RAW_DET_SCORE_MIN, help='Minimum raw AI detector score retained during JSONL preprocessing.')
    parser.add_argument('--class-policy-json', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--k1-recall-target', type=float, default=0.99)
    parser.add_argument('--k1-exact-refine-rounds', type=int, default=1)
    parser.add_argument('--k1-workers', type=int, default=4)
    parser.add_argument('--k2-run-dir', type=Path, default=infer_ROOT / 'assets/k2_v5')
    parser.add_argument('--k2-device', type=str, default='cuda')
    parser.add_argument('--k2-batch-size', type=int, default=64)
    parser.add_argument('--k2-prep-workers', type=int, default=1)
    parser.add_argument('--k2-precision', type=str, default='fp32', choices=('fp32', 'fp16'), help='K2 model precision. fp16 is CUDA-only and falls back to fp32 on CPU.')
    parser.add_argument('--k2-forward-mode', type=str, default='full', choices=('states_only', 'full'), help='states_only skips unused soft mask rendering during K2 inference.')
    parser.add_argument('--k2-profile-stages', action='store_true', help='Synchronize and record K2 batch-stage timings for profiling.')
    parser.add_argument('--k2-cudnn-benchmark', type=str, default='off', choices=('on', 'off'), help='cuDNN benchmark can speed repeated shapes but adds first-batch autotune overhead.')
    parser.add_argument('--k2-tf32', type=str, default='default', choices=('default', 'on', 'off'), help='Control TF32 for K2 CUDA matmul/cuDNN paths.')
    parser.add_argument('--routing-mode', type=str, default='k1n_sequence', choices=('threshold_only', 'threshold_soft', 'threshold_hysteresis', 'k1n_sequence', 'track_dp', 'band'), help="K1/K2 routing mode. 'threshold_only' uses per-row K1 cost only. 'threshold_soft' applies weak temporal smoothing near the threshold. 'threshold_hysteresis' uses explicit enter/exit thresholds plus entry confirmation. 'k1n_sequence' uses only the K1 normalized cost sequence with hysteresis and protected island cleanup. 'track_dp' performs track-level non-learned DP with asymmetric switching and soft run penalties. 'band' uses the original track expansion logic.")
    parser.add_argument('--k1-cost-routing', type=str, default='normalized', choices=('raw', 'normalized'), help='Cost scale used for K1/K2 routing. Debug columns still keep both raw and normalized costs.')
    parser.add_argument('--threshold', type=int, default=5000)
    parser.add_argument('--threshold-edge', type=int, default=-1)
    parser.add_argument('--threshold-norm', type=float, default=0.18)
    parser.add_argument('--threshold-edge-norm', type=float, default=0.18)
    parser.add_argument('--k2-soft-ema-alpha', type=float, default=0.8)
    parser.add_argument('--k2-soft-band-ratio', type=float, default=0.03)
    parser.add_argument('--k2-soft-exit-ratio', type=float, default=-1.0)
    parser.add_argument('--k2-soft-strong-ratio', type=float, default=0.1)
    parser.add_argument('--k2-soft-k1-keep-cost', type=int, default=-1)
    parser.add_argument('--k2-soft-k1-keep-cost-norm', type=float, default=-1.0)
    parser.add_argument('--k2-soft-reset-gap', type=int, default=2)
    parser.add_argument('--k2-soft-merge-islands-max-len', type=int, default=0)
    parser.add_argument('--k2-soft-merge-policy', type=str, default='symmetric', choices=('symmetric', 'prefer_k2'))
    parser.add_argument('--k2-hyst-enter', type=int, default=6000)
    parser.add_argument('--k2-hyst-enter-edge', type=int, default=-1)
    parser.add_argument('--k2-hyst-exit', type=int, default=4000)
    parser.add_argument('--k2-hyst-exit-edge', type=int, default=-1)
    parser.add_argument('--k2-hyst-enter-norm', type=float, default=0.20)
    parser.add_argument('--k2-hyst-enter-edge-norm', type=float, default=-1.0)
    parser.add_argument('--k2-hyst-exit-norm', type=float, default=0.14)
    parser.add_argument('--k2-hyst-exit-edge-norm', type=float, default=-1.0)
    parser.add_argument('--k2-hyst-confirm-frames', type=int, default=2)
    parser.add_argument('--k2-hyst-reset-gap', type=int, default=2)
    parser.add_argument('--k1n-seq-enter-norm', type=float, default=-1.0)
    parser.add_argument('--k1n-seq-exit-norm', type=float, default=0.13)
    parser.add_argument('--k1n-seq-strong-enter-norm', type=float, default=-1.0)
    parser.add_argument('--k1n-seq-strong-exit-norm', type=float, default=-1.0)
    parser.add_argument('--k1n-seq-protect-k2-iou-below', type=float, default=0.65)
    parser.add_argument('--k1n-seq-smooth-window', type=int, default=11)
    parser.add_argument('--k1n-seq-enter-confirm-frames', type=int, default=6)
    parser.add_argument('--k1n-seq-exit-confirm-frames', type=int, default=6)
    parser.add_argument('--k1n-seq-merge-short-k1-max-len', type=int, default=5)
    parser.add_argument('--k1n-seq-merge-short-k2-max-len', type=int, default=5)
    parser.add_argument('--k1n-seq-reset-gap', type=int, default=2)
    parser.add_argument('--k2-dp-error-weight', type=float, default=1.0)
    parser.add_argument('--k2-dp-instability-weight', type=float, default=0.4)
    parser.add_argument('--k2-dp-edge-bonus', type=float, default=0.2)
    parser.add_argument('--k2-dp-k2-bias', type=float, default=0.28)
    parser.add_argument('--k2-dp-switch-12', type=float, default=0.8)
    parser.add_argument('--k2-dp-switch-21', type=float, default=1.5)
    parser.add_argument('--k2-dp-short-k1-gamma', type=float, default=1.55)
    parser.add_argument('--k2-dp-short-k2-gamma', type=float, default=0.6)
    parser.add_argument('--k2-dp-short-k1-tau', type=float, default=1.6)
    parser.add_argument('--k2-dp-short-k2-tau', type=float, default=1.3)
    parser.add_argument('--k2-dp-short-len-cap', type=int, default=6)
    parser.add_argument('--k2-dp-reset-gap', type=int, default=2)
    parser.add_argument('--k2-dp-merge-short-k1-max-len', type=int, default=24)
    parser.add_argument('--k2-dp-merge-short-k2-max-len', type=int, default=12)
    parser.add_argument('--k2-dp-merge-short-k2-keep-cost', type=int, default=10000)
    parser.add_argument('--k2-dp-force-k2-cost', type=int, default=10000)
    parser.add_argument('--k2-dp-merge-short-k2-keep-cost-norm', type=float, default=0.35)
    parser.add_argument('--k2-dp-force-k2-cost-norm', type=float, default=0.35)
    parser.add_argument('--k2-band-radius', type=int, default=3)
    parser.add_argument('--k2-band-error-percentile', type=float, default=92.0)
    parser.add_argument('--k2-band-instability-percentile', type=float, default=90.0)
    parser.add_argument('--k2-band-instability-floor', type=float, default=0.4)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--max-tracks', type=int, default=0)
    return parser

@dataclass
class infer_RawTrack:
    track_id: int
    scene_id: int
    last_frame: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    area: float
    aspect: float
    poly_area: float | None
    fill_ratio: float | None

    def update(self, frame_idx: int, feat: 'RawDetFeatures') -> None:
        self.last_frame = frame_idx
        self.bbox = feat.bbox
        self.center = feat.center
        self.area = feat.area
        self.aspect = feat.aspect
        self.poly_area = feat.poly_area
        self.fill_ratio = feat.fill_ratio

class infer_RawCutDetector:

    def __init__(self) -> None:
        self.prev_small: np.ndarray | None = None
        self.last_cut_frame = -10 ** 9

    def update(self, frame_idx: int, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (infer_RAW_CUT_DOWNSCALE_W, infer_RAW_CUT_DOWNSCALE_H), interpolation=cv2.INTER_AREA)
        if self.prev_small is None:
            self.prev_small = small
            return False
        diff = float(np.mean(cv2.absdiff(small, self.prev_small)))
        self.prev_small = small
        if frame_idx - self.last_cut_frame <= infer_RAW_CUT_MIN_GAP_FRAMES:
            return False
        if diff >= infer_RAW_CUT_DIFF_THRESH:
            self.last_cut_frame = frame_idx
            return True
        return False

@dataclass
class infer_RawDetFeatures:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    area: float
    aspect: float
    poly_area: float | None
    fill_ratio: float | None

def infer_raw_to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def infer_raw_polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    x0, y0 = poly[0]
    x_prev, y_prev = (x0, y0)
    for x, y in poly[1:]:
        area += x_prev * y - x * y_prev
        x_prev, y_prev = (x, y)
    area += x_prev * y0 - x0 * y_prev
    return abs(area) * 0.5

def infer_raw_sum_polygons_area(polygons: list[list[list[float]]]) -> float:
    return sum((infer_raw_polygon_area(poly) for poly in polygons)) if polygons else 0.0

def infer_raw_finalize_polygon_points(pts: list[list[float]]) -> tuple[list[list[float]] | None, float, tuple[float, float, float, float] | None]:
    if len(pts) < 3:
        return (None, 0.0, None)
    x0, y0 = pts[0]
    min_x = max_x = x0
    min_y = max_y = y0
    area = 0.0
    x_prev, y_prev = (x0, y0)
    for x, y in pts[1:]:
        area += x_prev * y - x * y_prev
        x_prev, y_prev = (x, y)
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
    area += x_prev * y0 - x0 * y_prev
    return (pts, abs(area) * 0.5, (min_x, min_y, max_x, max_y))

def infer_raw_bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0

def infer_raw_bbox_contains(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float=0.0) -> bool:
    return a[0] - margin <= b[0] and a[1] - margin <= b[1] and (a[2] + margin >= b[2]) and (a[3] + margin >= b[3])

def infer_raw_det_bbox_area(det: dict[str, object]) -> float:
    cached = det.get('_bbox_area')
    if isinstance(cached, (int, float)):
        return float(cached)
    x1, y1, x2, y2 = det.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

def infer_raw_det_mask_area(det: dict[str, object]) -> float:
    cached = det.get('_mask_area')
    if isinstance(cached, (int, float)):
        return float(cached)
    polygons = det.get('polygons') or []
    return infer_raw_sum_polygons_area(polygons) if polygons else 0.0

def infer_raw_box_inside_bbox(box: tuple[float, float, float, float], bbox: tuple[float, float, float, float], margin: float=0.0) -> bool:
    return box[0] >= bbox[0] - margin and box[1] >= bbox[1] - margin and (box[2] <= bbox[2] + margin) and (box[3] <= bbox[3] + margin)

def infer_raw_polygon_inside_bbox(poly: list[list[float]], bbox: tuple[float, float, float, float], margin: float=0.0) -> bool:
    if len(poly) < 3:
        return False
    x1, y1, x2, y2 = bbox
    for x, y in poly:
        if x < x1 - margin or x > x2 + margin or y < y1 - margin or (y > y2 + margin):
            return False
    return True

def infer_raw_det_mask_inside_bbox(det: dict[str, object], bbox: tuple[float, float, float, float], margin: float=0.0) -> bool:
    poly_boxes = det.get('_poly_boxes')
    if isinstance(poly_boxes, list):
        return any((infer_raw_box_inside_bbox(tuple(box), bbox, margin=margin) for box in poly_boxes))
    for poly in det.get('polygons') or []:
        if infer_raw_polygon_inside_bbox(poly, bbox, margin=margin):
            return True
    return False

def infer_raw_is_contained_pair(det_i: dict[str, object], det_j: dict[str, object], bbox_i: tuple[float, float, float, float], bbox_j: tuple[float, float, float, float], margin: float=0.0) -> bool:
    if infer_raw_bbox_contains(bbox_i, bbox_j, margin) or infer_raw_bbox_contains(bbox_j, bbox_i, margin):
        return True
    if infer_raw_det_mask_inside_bbox(det_j, bbox_i, margin):
        return True
    if infer_raw_det_mask_inside_bbox(det_i, bbox_j, margin):
        return True
    return False

def infer_raw_nms_thresholds_for_area(area_ref: float) -> tuple[float, float]:
    if area_ref <= infer_RAW_TINY_AREA_THRESH:
        return (infer_RAW_NMS_IOU_THRESH_TINY, infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX_TINY)
    if area_ref <= infer_RAW_SMALL_AREA_THRESH:
        return (infer_RAW_NMS_IOU_THRESH_SMALL, infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX_SMALL)
    return (infer_RAW_NMS_IOU_THRESH, infer_RAW_NMS_CONTAIN_AREA_RATIO_MAX)

def infer_raw_apply_nms(detections: list[dict[str, object]]) -> list[dict[str, object]]:
    if not detections:
        return detections
    count = len(detections)
    bboxes = [tuple(det.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])) for det in detections]
    bbox_areas = [infer_raw_det_bbox_area(det) for det in detections]
    mask_areas = [infer_raw_det_mask_area(det) for det in detections]
    size_refs = [min(bbox_areas[idx], mask_areas[idx]) if mask_areas[idx] > 0 else bbox_areas[idx] for idx in range(count)]
    scores = [float(det.get('score') or 0.0) for det in detections]
    order = sorted(range(count), key=lambda idx: (-scores[idx], idx))
    suppressed = [False] * count
    keep_indices: list[int] = []
    for pos, idx in enumerate(order):
        if suppressed[idx]:
            continue
        keep_indices.append(idx)
        for other in order[pos + 1:]:
            if suppressed[other]:
                continue
            area_ref = min(size_refs[idx], size_refs[other])
            iou_thresh, contain_ratio_max = infer_raw_nms_thresholds_for_area(area_ref)
            area_i = bbox_areas[idx]
            area_j = bbox_areas[other]
            area_min = min(area_i, area_j)
            area_max = max(area_i, area_j)
            contains = infer_raw_is_contained_pair(detections[idx], detections[other], bboxes[idx], bboxes[other], margin=infer_RAW_NMS_CONTAIN_MARGIN)
            if contains and area_min > 0.0 and (area_max / area_min <= contain_ratio_max):
                suppressed[other] = True
                continue
            iou = infer_raw_bbox_iou(bboxes[idx], bboxes[other])
            if iou >= iou_thresh:
                suppressed[other] = True
    return [detections[idx] for idx in keep_indices]

def infer_raw_compute_features(det: dict[str, object]) -> infer_RawDetFeatures:
    x1, y1, x2, y2 = det.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    area = infer_raw_det_bbox_area(det)
    aspect = width / height if height > 0 else 0.0
    center_x = float(x1) + width * 0.5
    center_y = float(y1) + height * 0.5
    poly_area_value = infer_raw_det_mask_area(det)
    poly_area = poly_area_value if poly_area_value > 0.0 else None
    fill_ratio = poly_area / area if poly_area is not None and area > 0.0 else None
    return infer_RawDetFeatures(bbox=(float(x1), float(y1), float(x2), float(y2)), center=(center_x, center_y), area=area, aspect=aspect, poly_area=poly_area, fill_ratio=fill_ratio)

def infer_raw_compute_match_score(track: infer_RawTrack, det: infer_RawDetFeatures, frame_idx: int) -> float | None:
    gap = frame_idx - track.last_frame
    if gap < 0 or gap > infer_RAW_MAX_GAP_FRAMES:
        return None
    if track.area <= 0.0 or det.area <= 0.0:
        return None
    area_ref = 0.5 * (track.area + det.area)
    iou_min = infer_RAW_IOU_MIN
    center_dist_max = infer_RAW_CENTER_DIST_MAX
    area_ratio_min = infer_RAW_AREA_RATIO_MIN
    area_ratio_max = infer_RAW_AREA_RATIO_MAX
    aspect_ratio_max = infer_RAW_ASPECT_RATIO_MAX
    score_min = infer_RAW_SCORE_MIN
    use_fill_poly = True
    if area_ref <= infer_RAW_TINY_AREA_THRESH:
        iou_min = infer_RAW_TINY_IOU_MIN
        center_dist_max = infer_RAW_TINY_CENTER_DIST_MAX
        area_ratio_min = infer_RAW_TINY_AREA_RATIO_MIN
        area_ratio_max = infer_RAW_TINY_AREA_RATIO_MAX
        aspect_ratio_max = infer_RAW_TINY_ASPECT_RATIO_MAX
        score_min = infer_RAW_TINY_SCORE_MIN
        use_fill_poly = False
    elif area_ref <= infer_RAW_SMALL_AREA_THRESH:
        iou_min = infer_RAW_SMALL_IOU_MIN
        center_dist_max = infer_RAW_SMALL_CENTER_DIST_MAX
        area_ratio_min = infer_RAW_SMALL_AREA_RATIO_MIN
        area_ratio_max = infer_RAW_SMALL_AREA_RATIO_MAX
        aspect_ratio_max = infer_RAW_SMALL_ASPECT_RATIO_MAX
        score_min = infer_RAW_SMALL_SCORE_MIN
        use_fill_poly = False
    iou = infer_raw_bbox_iou(track.bbox, det.bbox)
    dx = det.center[0] - track.center[0]
    dy = det.center[1] - track.center[1]
    dist = math.hypot(dx, dy)
    w1 = max(0.0, track.bbox[2] - track.bbox[0])
    h1 = max(0.0, track.bbox[3] - track.bbox[1])
    w2 = max(0.0, det.bbox[2] - det.bbox[0])
    h2 = max(0.0, det.bbox[3] - det.bbox[1])
    diag = 0.5 * (math.hypot(w1, h1) + math.hypot(w2, h2))
    center_dist_norm = dist / (diag + 1e-06)
    if iou < iou_min and center_dist_norm > center_dist_max:
        return None
    area_ratio = det.area / track.area
    if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
        return None
    if track.aspect <= 0.0 or det.aspect <= 0.0:
        return None
    aspect_diff = abs(math.log(det.aspect / track.aspect))
    if aspect_diff > aspect_ratio_max:
        return None
    fill_score = 0.5
    if use_fill_poly and track.fill_ratio is not None and (det.fill_ratio is not None):
        fill_diff = abs(det.fill_ratio - track.fill_ratio)
        if fill_diff > infer_RAW_FILL_RATIO_MAX:
            return None
        fill_score = max(0.0, 1.0 - fill_diff / infer_RAW_FILL_RATIO_MAX)
    if use_fill_poly and track.poly_area is not None and (det.poly_area is not None) and (track.poly_area > 0.0) and (det.poly_area > 0.0):
        poly_ratio = det.poly_area / track.poly_area
        if poly_ratio < infer_RAW_POLY_RATIO_MIN or poly_ratio > infer_RAW_POLY_RATIO_MAX:
            return None
    center_score = max(0.0, 1.0 - center_dist_norm / center_dist_max)
    aspect_score = max(0.0, 1.0 - aspect_diff / aspect_ratio_max)
    score = 0.5 * iou + 0.3 * center_score + 0.15 * aspect_score + 0.05 * fill_score
    return score if score >= score_min else None

def infer_raw_normalize_bbox_xyxy(det: dict[str, object]) -> list[float]:
    bbox_xyxy = det.get('bbox_xyxy')
    if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
        vals = [infer_raw_to_float(x) for x in bbox_xyxy[:4]]
        if all((v is not None for v in vals)):
            x1, y1, x2, y2 = vals
            if x2 < x1:
                x1, x2 = (x2, x1)
            if y2 < y1:
                y1, y2 = (y2, y1)
            return [float(x1), float(y1), float(x2), float(y2)]
    bbox = det.get('bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        vals = [infer_raw_to_float(x) for x in bbox[:4]]
        if all((v is not None for v in vals)):
            x, y, width, height = vals
            return [float(x), float(y), float(x + max(0.0, width)), float(y + max(0.0, height))]
    return [0.0, 0.0, 0.0, 0.0]

def infer_raw_normalize_segmentation(seg_like) -> tuple[list[list[list[float]]], float, list[tuple[float, float, float, float]]]:
    polygons: list[list[list[float]]] = []
    poly_boxes: list[tuple[float, float, float, float]] = []
    total_area = 0.0
    if not isinstance(seg_like, list):
        return (polygons, total_area, poly_boxes)
    if seg_like and all((isinstance(v, (int, float)) for v in seg_like)):
        pts: list[list[float]] = []
        for idx in range(0, len(seg_like) // 2 * 2, 2):
            x = infer_raw_to_float(seg_like[idx])
            y = infer_raw_to_float(seg_like[idx + 1])
            if x is not None and y is not None:
                pts.append([x, y])
        poly, area, box = infer_raw_finalize_polygon_points(pts)
        if poly is not None and box is not None:
            polygons.append(poly)
            total_area += area
            poly_boxes.append(box)
        return (polygons, total_area, poly_boxes)
    for poly_like in seg_like:
        if not isinstance(poly_like, list):
            continue
        if poly_like and all((isinstance(v, (int, float)) for v in poly_like)):
            pts = []
            for idx in range(0, len(poly_like) // 2 * 2, 2):
                x = infer_raw_to_float(poly_like[idx])
                y = infer_raw_to_float(poly_like[idx + 1])
                if x is not None and y is not None:
                    pts.append([x, y])
            poly, area, box = infer_raw_finalize_polygon_points(pts)
            if poly is not None and box is not None:
                polygons.append(poly)
                total_area += area
                poly_boxes.append(box)
            continue
        pts = []
        for point in poly_like:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x = infer_raw_to_float(point[0])
            y = infer_raw_to_float(point[1])
            if x is not None and y is not None:
                pts.append([x, y])
        poly, area, box = infer_raw_finalize_polygon_points(pts)
        if poly is not None and box is not None:
            polygons.append(poly)
            total_area += area
            poly_boxes.append(box)
    return (polygons, total_area, poly_boxes)

def infer_normalize_raw_record(obj: dict[str, object]) -> tuple[int, list[dict[str, object]]]:
    frame_idx = int(obj.get('frame_index', obj.get('frame_idx', 0)))
    src_dets = obj.get('detections') or obj.get('instances') or []
    detections: list[dict[str, object]] = []
    for src in src_dets:
        if not isinstance(src, dict):
            continue
        det: dict[str, object] = {}
        det['class_name'] = str(src.get('class_name', src.get('label', 'unknown')))
        if 'label' in src:
            det['label'] = str(src.get('label', ''))
        for key in ('category_id', 'category_index'):
            if key in src and src.get(key) is not None:
                det[key] = src.get(key)
        for key in ('detector_score', 'class_score'):
            value = infer_raw_to_float(src.get(key))
            if value is not None:
                det[key] = value
        score = infer_raw_to_float(src.get('score'))
        if score is not None:
            det['score'] = score
        bbox_xyxy = infer_raw_normalize_bbox_xyxy(src)
        det['bbox_xyxy'] = bbox_xyxy
        if 'bbox' in src and src.get('bbox') is not None:
            det['bbox'] = src.get('bbox')
        det['_bbox_area'] = max(0.0, bbox_xyxy[2] - bbox_xyxy[0]) * max(0.0, bbox_xyxy[3] - bbox_xyxy[1])
        polygons, mask_area, poly_boxes = infer_raw_normalize_segmentation(src.get('polygons') or src.get('segmentation'))
        det['polygons'] = polygons
        det['_mask_area'] = mask_area
        det['_poly_boxes'] = poly_boxes
        detections.append(det)
    return (frame_idx, detections)

def infer_guess_input_video_path(input_jsonl: Path) -> Path | None:
    for suffix in ('.mp4', '.mov', '.mkv', '.avi', '.MP4', '.MOV', '.MKV', '.AVI'):
        candidate = input_jsonl.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None

def infer_json_loads(text: str) -> object:
    if orjson is not None:
        return orjson.loads(text)
    return json.loads(text)

def infer_parse_raw_frame_index_from_line(line: str) -> int:
    # Cut pre-scan only needs the frame number, so avoid parsing the full
    # detection payload twice on large JSONL files.
    for key in ('"frame_index"', '"frame_idx"'):
        key_pos = line.find(key)
        if key_pos < 0:
            continue
        colon_pos = line.find(':', key_pos + len(key))
        if colon_pos < 0:
            continue
        pos = colon_pos + 1
        while pos < len(line) and line[pos] in ' \t':
            pos += 1
        sign = 1
        if pos < len(line) and line[pos] == '-':
            sign = -1
            pos += 1
        start = pos
        while pos < len(line) and line[pos].isdigit():
            pos += 1
        if pos > start:
            return sign * int(line[start:pos])
    obj = infer_json_loads(line)
    if not isinstance(obj, dict):
        return 0
    return int(obj.get('frame_index', obj.get('frame_idx', 0)))

def infer_load_raw_frame_indices(jsonl_path: Path) -> list[int]:
    frame_indices: list[int] = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_indices.append(infer_parse_raw_frame_index_from_line(line))
    return frame_indices

def infer_detect_cut_frames_for_indices_exact(frame_indices: list[int], video_path: Path) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open input video for cut detection: {video_path}')
    cut_detector = infer_RawCutDetector()
    current_cap_frame: int | None = None
    max_sequential_frame_skip = 16
    cut_frames: list[int] = []
    try:
        for frame_idx in frame_indices:
            if current_cap_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx < current_cap_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx > current_cap_frame:
                gap = frame_idx - current_cap_frame
                if gap <= max_sequential_frame_skip:
                    while current_cap_frame < frame_idx:
                        ok_skip, _skip_frame = cap.read()
                        if not ok_skip:
                            raise RuntimeError(f'Failed to skip video frame before cut detection frame {frame_idx}')
                        current_cap_frame += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    current_cap_frame = frame_idx
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'Failed to read video frame for cut detection: frame={frame_idx}')
            current_cap_frame = frame_idx + 1
            if cut_detector.update(frame_idx, frame):
                cut_frames.append(int(frame_idx))
    finally:
        cap.release()
    return cut_frames

def infer_ffmpeg_cut_candidate_frames(video_path: Path) -> list[int]:
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        str(video_path),
        '-vf',
        f'scale={infer_RAW_CUT_DOWNSCALE_W}:{infer_RAW_CUT_DOWNSCALE_H}:flags=area,format=gray',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'gray',
        '-',
    ]
    frame_size = int(infer_RAW_CUT_DOWNSCALE_W * infer_RAW_CUT_DOWNSCALE_H)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    previous: np.ndarray | None = None
    frame_idx = 0
    candidates: list[int] = []
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf:
                break
            if len(buf) != frame_size:
                raise RuntimeError(f'ffmpeg cut candidate extraction returned a short frame: {len(buf)} bytes')
            current = np.frombuffer(buf, dtype=np.uint8)
            if previous is not None:
                diff = float(np.mean(np.abs(current.astype(np.int16) - previous.astype(np.int16))))
                if diff >= float(infer_RAW_CUT_HYBRID_FFMPEG_CANDIDATE_THRESH):
                    candidates.append(int(frame_idx))
            previous = current.copy()
            frame_idx += 1
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    stderr = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr is not None else ''
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f'ffmpeg cut candidate extraction failed with code {return_code}: {stderr.strip()}')
    return candidates

def infer_read_cut_small_frames(video_path: Path, frame_numbers: list[int]) -> dict[int, np.ndarray]:
    if not frame_numbers:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open input video for cut verification: {video_path}')
    current_cap_frame: int | None = None
    max_sequential_frame_skip = 16
    smalls: dict[int, np.ndarray] = {}
    try:
        for frame_idx in sorted(set(int(v) for v in frame_numbers if int(v) >= 0)):
            if current_cap_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx < current_cap_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx > current_cap_frame:
                gap = frame_idx - current_cap_frame
                if gap <= max_sequential_frame_skip:
                    while current_cap_frame < frame_idx:
                        ok_skip, _skip_frame = cap.read()
                        if not ok_skip:
                            raise RuntimeError(f'Failed to skip video frame before cut verification frame {frame_idx}')
                        current_cap_frame += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    current_cap_frame = frame_idx
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'Failed to read video frame for cut verification: frame={frame_idx}')
            current_cap_frame = frame_idx + 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smalls[int(frame_idx)] = cv2.resize(
                gray,
                (infer_RAW_CUT_DOWNSCALE_W, infer_RAW_CUT_DOWNSCALE_H),
                interpolation=cv2.INTER_AREA,
            )
    finally:
        cap.release()
    return smalls

def infer_read_cut_probe_frames(video_path: Path, frame_numbers: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not frame_numbers:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open input video for high precision cut verification: {video_path}')
    current_cap_frame: int | None = None
    max_sequential_frame_skip = 16
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for frame_idx in sorted(set(int(v) for v in frame_numbers if int(v) >= 0)):
            if current_cap_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx < current_cap_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_cap_frame = frame_idx
            elif frame_idx > current_cap_frame:
                gap = frame_idx - current_cap_frame
                if gap <= max_sequential_frame_skip:
                    while current_cap_frame < frame_idx:
                        ok_skip, _skip_frame = cap.read()
                        if not ok_skip:
                            raise RuntimeError(f'Failed to skip video frame before high precision cut verification frame {frame_idx}')
                        current_cap_frame += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    current_cap_frame = frame_idx
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'Failed to read video frame for high precision cut verification: frame={frame_idx}')
            current_cap_frame = frame_idx + 1
            small_bgr = cv2.resize(
                frame,
                (infer_RAW_CUT_DOWNSCALE_W, infer_RAW_CUT_DOWNSCALE_H),
                interpolation=cv2.INTER_AREA,
            )
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
            frames[int(frame_idx)] = (small_gray, small_bgr)
    finally:
        cap.release()
    return frames

def infer_cut_hist_correlation_gray(previous: np.ndarray, current: np.ndarray) -> float:
    prev_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten().astype('float32')
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten().astype('float32')
    return float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))

def infer_cut_hist_correlation_bgr(previous: np.ndarray, current: np.ndarray) -> float:
    prev_hist = cv2.calcHist([previous], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([current], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten().astype('float32')
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten().astype('float32')
    return float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))

def infer_cut_ssim_gray(previous: np.ndarray, current: np.ndarray) -> float:
    prev = previous.astype(np.float32)
    curr = current.astype(np.float32)
    c1 = 6.5025
    c2 = 58.5225
    kernel = (7, 7)
    sigma = 1.5
    prev_mu = cv2.GaussianBlur(prev, kernel, sigma)
    curr_mu = cv2.GaussianBlur(curr, kernel, sigma)
    prev_mu_sq = prev_mu * prev_mu
    curr_mu_sq = curr_mu * curr_mu
    prev_curr_mu = prev_mu * curr_mu
    prev_sigma = cv2.GaussianBlur(prev * prev, kernel, sigma) - prev_mu_sq
    curr_sigma = cv2.GaussianBlur(curr * curr, kernel, sigma) - curr_mu_sq
    prev_curr_sigma = cv2.GaussianBlur(prev * curr, kernel, sigma) - prev_curr_mu
    ssim_map = ((2.0 * prev_curr_mu + c1) * (2.0 * prev_curr_sigma + c2)) / (
        (prev_mu_sq + curr_mu_sq + c1) * (prev_sigma + curr_sigma + c2)
    )
    return float(np.mean(ssim_map))

def infer_high_precision_cut_candidate_score(diff_mean: float, color_corr: float, ssim: float) -> float:
    color_change = max(0.0, 1.0 - float(color_corr))
    structure_change = max(0.0, 1.0 - float(ssim))
    return float(diff_mean) * (1.0 + 0.7 * color_change + 0.5 * structure_change)

def infer_is_high_precision_cut(diff_mean: float, color_corr: float, ssim: float) -> bool:
    if diff_mean < infer_RAW_CUT_HP_MIN_DIFF:
        return False
    if diff_mean >= infer_RAW_CUT_HP_HARD_DIFF:
        return True
    if diff_mean >= infer_RAW_CUT_HP_STRONG_DIFF:
        return color_corr <= infer_RAW_CUT_HP_STRONG_COLOR_CORR_MAX and ssim <= infer_RAW_CUT_HP_STRONG_SSIM_MAX
    if diff_mean >= infer_RAW_CUT_HP_NORMAL_DIFF:
        return color_corr <= infer_RAW_CUT_HP_COLOR_CORR_MAX and ssim <= infer_RAW_CUT_HP_SSIM_MAX
    return False

def infer_detect_cut_frames_for_indices_hybrid(frame_indices: list[int], video_path: Path) -> list[int]:
    # The hybrid path is exact for the common dense-frame JSONL route: ffmpeg
    # only narrows down candidate neighborhoods, and OpenCV performs the final
    # same resize/diff/min-gap decision used by infer_RawCutDetector.
    if len(frame_indices) < 2:
        return []
    if any(int(b) - int(a) != 1 for a, b in zip(frame_indices, frame_indices[1:], strict=False)):
        raise RuntimeError('hybrid cut detection requires consecutive JSONL frame indices')
    candidate_frames = infer_ffmpeg_cut_candidate_frames(video_path)
    radius = int(infer_RAW_CUT_HYBRID_WINDOW_RADIUS)
    frame_set = set(int(v) for v in frame_indices)
    verify_frames: set[int] = set()
    candidate_eval_frames: set[int] = set()
    for candidate in candidate_frames:
        for frame_idx in range(int(candidate) - radius, int(candidate) + radius + 1):
            if frame_idx not in frame_set or frame_idx <= 0:
                continue
            candidate_eval_frames.add(frame_idx)
            verify_frames.add(frame_idx)
            verify_frames.add(frame_idx - 1)
    smalls = infer_read_cut_small_frames(video_path, sorted(verify_frames))
    exact_candidates: list[int] = []
    for frame_idx in sorted(candidate_eval_frames):
        previous = smalls.get(frame_idx - 1)
        current = smalls.get(frame_idx)
        if previous is None or current is None:
            continue
        diff = float(np.mean(cv2.absdiff(current, previous)))
        if diff >= float(infer_RAW_CUT_DIFF_THRESH):
            exact_candidates.append(int(frame_idx))
    cut_frames: list[int] = []
    last_cut_frame = -10 ** 9
    for frame_idx in exact_candidates:
        if frame_idx - last_cut_frame <= infer_RAW_CUT_MIN_GAP_FRAMES:
            continue
        last_cut_frame = int(frame_idx)
        cut_frames.append(int(frame_idx))
    return cut_frames

def infer_detect_cut_frames_for_indices_high_precision(frame_indices: list[int], video_path: Path) -> list[int]:
    if len(frame_indices) < 2:
        return []
    if any(int(b) - int(a) != 1 for a, b in zip(frame_indices, frame_indices[1:], strict=False)):
        raise RuntimeError('high precision cut detection requires consecutive JSONL frame indices')
    candidate_frames = infer_ffmpeg_cut_candidate_frames(video_path)
    radius = int(infer_RAW_CUT_HYBRID_WINDOW_RADIUS)
    frame_set = set(int(v) for v in frame_indices)
    verify_frames: set[int] = set()
    candidate_eval_frames: set[int] = set()
    for candidate in candidate_frames:
        for frame_idx in range(int(candidate) - radius, int(candidate) + radius + 1):
            if frame_idx not in frame_set or frame_idx <= 0:
                continue
            candidate_eval_frames.add(frame_idx)
            verify_frames.add(frame_idx)
            verify_frames.add(frame_idx - 1)
    probe_frames = infer_read_cut_probe_frames(video_path, sorted(verify_frames))
    exact_candidates: list[tuple[int, float]] = []
    for frame_idx in sorted(candidate_eval_frames):
        previous = probe_frames.get(frame_idx - 1)
        current = probe_frames.get(frame_idx)
        if previous is None or current is None:
            continue
        previous_gray, previous_bgr = previous
        current_gray, current_bgr = current
        diff_mean = float(np.mean(cv2.absdiff(current_gray, previous_gray)))
        color_corr = infer_cut_hist_correlation_bgr(previous_bgr, current_bgr)
        ssim = infer_cut_ssim_gray(previous_gray, current_gray)
        if infer_is_high_precision_cut(diff_mean, color_corr, ssim):
            exact_candidates.append((int(frame_idx), infer_high_precision_cut_candidate_score(diff_mean, color_corr, ssim)))
    cut_frames: list[tuple[int, float]] = []
    min_gap = int(infer_RAW_CUT_HP_MIN_GAP_FRAMES)
    for frame_idx, score in exact_candidates:
        if cut_frames and frame_idx - cut_frames[-1][0] <= min_gap:
            continue
        cut_frames.append((int(frame_idx), float(score)))
    return [frame_idx for frame_idx, _score in cut_frames]

def infer_detect_cut_frames_for_jsonl(jsonl_path: Path, video_path: Path, *, method: str=infer_RAW_CUT_METHOD_DEFAULT) -> tuple[list[int], float, str]:
    """Detect cuts before tracking without changing downstream decisions."""
    start_time = time.perf_counter()
    frame_indices = infer_load_raw_frame_indices(jsonl_path)
    method_value = str(method)
    try:
        if method_value == 'legacy_diff':
            cut_frames = infer_detect_cut_frames_for_indices_hybrid(frame_indices, video_path)
            used_method = 'ffmpeg_candidates_opencv_verify'
        elif method_value == 'high_precision':
            cut_frames = infer_detect_cut_frames_for_indices_high_precision(frame_indices, video_path)
            used_method = 'ffmpeg_candidates_opencv_high_precision'
        else:
            raise ValueError(f'Unsupported raw cut method: {method_value}')
    except Exception as exc:
        print(f'raw_preprocess: {method_value} cut detection fallback to exact OpenCV scan: {exc}', flush=True)
        cut_frames = infer_detect_cut_frames_for_indices_exact(frame_indices, video_path)
        used_method = 'opencv_exact'
    if not cut_frames and len(frame_indices) >= 2 and method_value in infer_RAW_CUT_METHODS:
        fallback_frames = infer_detect_cut_frames_for_indices_exact(frame_indices, video_path)
        if fallback_frames:
            print(
                f'raw_preprocess: {used_method} produced zero cuts; '
                f'using exact OpenCV scan fallback with {len(fallback_frames)} cuts',
                flush=True,
            )
            cut_frames = fallback_frames
            used_method = f'{used_method}+opencv_exact_empty_fallback'
    return cut_frames, float(time.perf_counter() - start_time), used_method

def infer_raw_normalize_label(label: object) -> str:
    text = str(label).strip()
    return text if text else 'unknown'


def infer_raw_policy_float(policy: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in policy:
            continue
        value = infer_raw_to_float(policy.get(key))
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def infer_raw_load_score_policy(class_policy_json: Path | None, fallback_score_min: float) -> tuple[float, dict[str, float]]:
    if class_policy_json is None:
        return float(fallback_score_min), {}
    raw = json.loads(class_policy_json.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        return float(fallback_score_min), {}
    score_keys = ('raw_det_score_min', 'confidence_min', 'score_min', 'min_score', 'confidence')
    default_score_min = float(fallback_score_min)
    default_policy = raw.get('default')
    if isinstance(default_policy, dict):
        value = infer_raw_policy_float({str(k): v for k, v in default_policy.items()}, score_keys)
        if value is not None:
            default_score_min = value
    classes_obj = raw.get('classes')
    source = classes_obj if isinstance(classes_obj, dict) else raw
    score_by_label: dict[str, float] = {}
    if isinstance(source, dict):
        for label, cfg in source.items():
            if label == 'default' or not isinstance(cfg, dict):
                continue
            value = infer_raw_policy_float({str(k): v for k, v in cfg.items()}, score_keys)
            if value is not None:
                score_by_label[infer_raw_normalize_label(label)] = value
    return default_score_min, score_by_label


def infer_raw_det_score_min_for_detection(det: dict[str, object], default_score_min: float, score_min_by_label: dict[str, float]) -> float:
    for key in ('class_name', 'label'):
        label = infer_raw_normalize_label(det.get(key, 'unknown'))
        if label in score_min_by_label:
            return float(score_min_by_label[label])
    return float(default_score_min)


def infer_build_tracked_sqlite_from_raw_jsonl(jsonl_path: Path, sqlite_path: Path, video_path: Path | None, *, remove_short_tracks_max_frames: int, enable_cut_detect: bool, raw_det_score_min: float=infer_RAW_DET_SCORE_MIN, raw_det_score_min_by_label: dict[str, float] | None=None, raw_cut_method: str=infer_RAW_CUT_METHOD_DEFAULT) -> dict[str, object]:
    tracks: dict[int, infer_RawTrack] = {}
    next_tid = 1
    total_rows = 0
    track_label_counts: dict[str, dict[str, int]] = {}
    track_label_first_seen: dict[tuple[str, str], int] = {}
    label_seen_order = 0
    current_scene_id = 0
    cut_frames: list[int] = []
    cuts_detected = 0
    cut_detection_elapsed_sec = 0.0
    cut_detection_method = 'disabled'
    precomputed_cut_frames: set[int] | None = None
    start_time = time.time()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    cap: cv2.VideoCapture | None = None
    cut_detector: infer_RawCutDetector | None = None
    current_cap_frame: int | None = None
    max_sequential_frame_skip = 16
    if enable_cut_detect:
        if video_path is None:
            raise FileNotFoundError('Cut detection requires an input video, but none was provided.')
        detected_cut_frames, cut_detection_elapsed_sec, cut_detection_method = infer_detect_cut_frames_for_jsonl(jsonl_path, video_path, method=raw_cut_method)
        precomputed_cut_frames = set(detected_cut_frames)

    def read_video_frame(target_frame_idx: int) -> tuple[bool, np.ndarray | None]:
        nonlocal current_cap_frame
        if cap is None:
            return (False, None)
        if current_cap_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            current_cap_frame = target_frame_idx
        elif target_frame_idx < current_cap_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            current_cap_frame = target_frame_idx
        elif target_frame_idx > current_cap_frame:
            gap = target_frame_idx - current_cap_frame
            if gap <= max_sequential_frame_skip:
                while current_cap_frame < target_frame_idx:
                    ok_skip, _skip_frame = cap.read()
                    if not ok_skip:
                        return (False, None)
                    current_cap_frame += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                current_cap_frame = target_frame_idx
        ok, frame = cap.read()
        if ok:
            current_cap_frame = target_frame_idx + 1
            return (True, frame)
        return (False, None)
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS masks')
    cur.execute('DROP TABLE IF EXISTS tracks')
    cur.execute('DROP TABLE IF EXISTS cuts')
    cur.execute('DROP TABLE IF EXISTS raw_tracked_masks')
    cur.execute('DROP TABLE IF EXISTS raw_tracks')
    cur.execute('\n        CREATE TABLE masks(\n            frame INTEGER NOT NULL,\n            track_id TEXT NOT NULL,\n            polygons TEXT,\n            shape_type TEXT,\n            dilate_px INTEGER NOT NULL DEFAULT 0,\n            feather_px INTEGER NOT NULL DEFAULT 0,\n            mosaic_block INTEGER NOT NULL DEFAULT 0,\n            mosaic_alias REAL NOT NULL DEFAULT 0,\n            label TEXT,\n            PRIMARY KEY(frame, track_id)\n        )\n        ')
    cur.execute('CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)')
    cur.execute('CREATE TABLE cuts(frame INTEGER PRIMARY KEY)')
    cur.execute(
        '''
        CREATE TABLE raw_tracked_masks(
            frame INTEGER NOT NULL,
            raw_track_id TEXT NOT NULL,
            raw_detection_index INTEGER NOT NULL,
            final_track_id TEXT,
            removed_by_short_track INTEGER NOT NULL DEFAULT 0,
            raw_track_length INTEGER NOT NULL DEFAULT 0,
            raw_label TEXT,
            final_label TEXT,
            polygons TEXT,
            score REAL,
            detector_score REAL,
            class_score REAL,
            category_id INTEGER,
            category_index INTEGER,
            bbox_xyxy_json TEXT,
            bbox_json TEXT,
            scene_id INTEGER,
            PRIMARY KEY(frame, raw_track_id, raw_detection_index)
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE raw_tracks(
            raw_track_id TEXT PRIMARY KEY,
            final_track_id TEXT,
            removed_by_short_track INTEGER NOT NULL DEFAULT 0,
            raw_track_length INTEGER NOT NULL DEFAULT 0,
            raw_label TEXT,
            final_label TEXT,
            scene_id INTEGER
        )
        '''
    )
    cur.execute('CREATE INDEX idx_raw_tracked_masks_track_frame ON raw_tracked_masks(raw_track_id, frame)')
    cur.execute('CREATE INDEX idx_raw_tracked_masks_final_track_frame ON raw_tracked_masks(final_track_id, frame)')
    active_track_ids: list[int] = []
    all_mask_rows_to_insert: list[tuple[int, str, str, str]] = []
    all_raw_mask_rows_to_insert: list[dict[str, object]] = []

    def raw_optional_float(value: object) -> float | None:
        number = infer_raw_to_float(value)
        if number is None:
            return None
        if not math.isfinite(float(number)):
            return None
        return float(number)

    def raw_optional_int(value: object) -> int | None:
        number = infer_raw_to_float(value)
        if number is None:
            return None
        if not math.isfinite(float(number)):
            return None
        return int(number)

    def raw_json_or_none(value: object) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    with jsonl_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = infer_json_loads(line)
            if not isinstance(obj, dict):
                continue
            frame_idx, detections = infer_normalize_raw_record(obj)
            if precomputed_cut_frames is not None:
                if frame_idx in precomputed_cut_frames:
                    current_scene_id += 1
                    cuts_detected += 1
                    cut_frames.append(frame_idx)
                    active_track_ids.clear()
            elif cap is not None and cut_detector is not None:
                ok, frame = read_video_frame(frame_idx)
                if ok and frame is not None:
                    if cut_detector.update(frame_idx, frame):
                        current_scene_id += 1
                        cuts_detected += 1
                        cut_frames.append(frame_idx)
                        active_track_ids.clear()
            score_min_by_label = raw_det_score_min_by_label or {}
            detections = [
                det
                for det in detections
                if float(det.get('score') or 0.0) >= infer_raw_det_score_min_for_detection(det, float(raw_det_score_min), score_min_by_label)
            ]
            detections = infer_raw_apply_nms(detections)
            det_features = [infer_raw_compute_features(det) for det in detections]
            if active_track_ids:
                active_track_ids = [tid for tid in active_track_ids if frame_idx - tracks[tid].last_frame <= infer_RAW_MAX_GAP_FRAMES]
            candidates: list[tuple[float, int, int]] = []
            for det_idx, feat in enumerate(det_features):
                for tid in active_track_ids:
                    track = tracks[tid]
                    score = infer_raw_compute_match_score(track, feat, frame_idx)
                    if score is not None:
                        candidates.append((score, tid, det_idx))
            candidates.sort(reverse=True)
            assigned_dets: set[int] = set()
            assigned_tracks: set[int] = set()
            det_to_track: dict[int, int] = {}
            for score, tid, det_idx in candidates:
                if det_idx in assigned_dets or tid in assigned_tracks:
                    continue
                det_to_track[det_idx] = tid
                assigned_dets.add(det_idx)
                assigned_tracks.add(tid)
            for det_idx, feat in enumerate(det_features):
                if det_idx in det_to_track:
                    tracks[det_to_track[det_idx]].update(frame_idx, feat)
                    continue
                tid = next_tid
                next_tid += 1
                tracks[tid] = infer_RawTrack(track_id=tid, scene_id=current_scene_id, last_frame=frame_idx, bbox=feat.bbox, center=feat.center, area=feat.area, aspect=feat.aspect, poly_area=feat.poly_area, fill_ratio=feat.fill_ratio)
                det_to_track[det_idx] = tid
                active_track_ids.append(tid)
            mask_rows_to_insert: list[tuple[int, str, str, str]] = []
            for det_idx, det in enumerate(detections):
                polygons = det.get('polygons') or []
                if not polygons:
                    continue
                track_id = str(det_to_track[det_idx])
                label = str(det.get('class_name', ''))
                raw_label = str(det.get('label', label))
                polygons_json = json.dumps(polygons, ensure_ascii=False)
                mask_rows_to_insert.append((int(frame_idx), track_id, polygons_json, label))
                track_obj = tracks.get(int(track_id))
                all_raw_mask_rows_to_insert.append(
                    {
                        'frame': int(frame_idx),
                        'raw_track_id': track_id,
                        'raw_detection_index': int(det_idx),
                        'raw_label': raw_label,
                        'polygons': polygons_json,
                        'score': raw_optional_float(det.get('score')),
                        'detector_score': raw_optional_float(det.get('detector_score')),
                        'class_score': raw_optional_float(det.get('class_score')),
                        'category_id': raw_optional_int(det.get('category_id')),
                        'category_index': raw_optional_int(det.get('category_index')),
                        'bbox_xyxy_json': raw_json_or_none(det.get('bbox_xyxy')),
                        'bbox_json': raw_json_or_none(det.get('bbox')),
                        'scene_id': None if track_obj is None else int(track_obj.scene_id),
                    }
                )
                label_counts = track_label_counts.setdefault(track_id, {})
                label_counts[label] = int(label_counts.get(label, 0)) + 1
                label_key = (track_id, label)
                if label_key not in track_label_first_seen:
                    track_label_first_seen[label_key] = label_seen_order
                    label_seen_order += 1
                total_rows += 1
            if mask_rows_to_insert:
                all_mask_rows_to_insert.extend(mask_rows_to_insert)
            if infer_RAW_PROGRESS_EVERY > 0 and line_no % infer_RAW_PROGRESS_EVERY == 0:
                print(f'raw_preprocess: frame {frame_idx}: {total_rows} rows, {len(tracks)} tracks, scenes={current_scene_id + 1}', flush=True)
    if cap is not None:
        cap.release()
    track_counts: dict[str, int] = {}
    for _frame, track_id, _polygons_json, _label in all_mask_rows_to_insert:
        track_counts[track_id] = track_counts.get(track_id, 0) + 1
    remove_tids = {tid for tid, count in track_counts.items() if count <= int(remove_short_tracks_max_frames)}
    keep_tids = sorted((tid for tid in track_counts if tid not in remove_tids), key=lambda value: int(value))
    id_map = {old: str(new) for new, old in enumerate(keep_tids, start=1)}
    removed_rows = sum(track_counts[tid] for tid in remove_tids)
    def majority_track_label(track_id: str) -> str:
        counts = track_label_counts.get(track_id, {})
        if not counts:
            return ''
        return max(
            counts.items(),
            key=lambda item: (int(item[1]), -int(track_label_first_seen.get((track_id, item[0]), 0))),
        )[0]
    raw_track_majority_labels = {track_id: majority_track_label(track_id) for track_id in track_counts}
    track_majority_labels = {track_id: raw_track_majority_labels.get(track_id, '') for track_id in keep_tids}
    mixed_label_tracks = sum(1 for track_id in keep_tids if len(track_label_counts.get(track_id, {})) > 1)
    relabeled_mask_rows = sum(
        1
        for _frame, track_id, _polygons_json, label in all_mask_rows_to_insert
        if track_id in id_map and str(label) != str(track_majority_labels.get(track_id, ''))
    )
    final_mask_rows_to_insert = [
        (frame, id_map[track_id], polygons_json, track_majority_labels[track_id])
        for frame, track_id, polygons_json, label in all_mask_rows_to_insert
        if track_id in id_map
    ]
    final_track_rows_to_insert = [
        (new_tid, track_majority_labels[old_tid])
        for old_tid, new_tid in id_map.items()
    ]
    raw_mask_rows_to_insert: list[tuple[object, ...]] = []
    for raw_row in all_raw_mask_rows_to_insert:
        raw_tid = str(raw_row['raw_track_id'])
        final_tid = id_map.get(raw_tid)
        raw_mask_rows_to_insert.append(
            (
                int(raw_row['frame']),
                raw_tid,
                int(raw_row['raw_detection_index']),
                final_tid,
                0 if final_tid is not None else 1,
                int(track_counts.get(raw_tid, 0)),
                raw_row.get('raw_label'),
                track_majority_labels.get(raw_tid) if final_tid is not None else None,
                raw_row.get('polygons'),
                raw_row.get('score'),
                raw_row.get('detector_score'),
                raw_row.get('class_score'),
                raw_row.get('category_id'),
                raw_row.get('category_index'),
                raw_row.get('bbox_xyxy_json'),
                raw_row.get('bbox_json'),
                raw_row.get('scene_id'),
            )
        )
    raw_track_rows_to_insert: list[tuple[object, ...]] = []
    for raw_tid in sorted(track_counts, key=lambda value: int(value)):
        track_obj = tracks.get(int(raw_tid))
        final_tid = id_map.get(raw_tid)
        raw_track_rows_to_insert.append(
            (
                raw_tid,
                final_tid,
                0 if final_tid is not None else 1,
                int(track_counts.get(raw_tid, 0)),
                raw_track_majority_labels.get(raw_tid, ''),
                track_majority_labels.get(raw_tid) if final_tid is not None else None,
                None if track_obj is None else int(track_obj.scene_id),
            )
        )
    if final_mask_rows_to_insert:
        cur.executemany("\n                    INSERT OR REPLACE INTO masks(\n                        frame, track_id, polygons, shape_type, dilate_px, feather_px, mosaic_block, mosaic_alias, label\n                    )\n                    VALUES (?, ?, ?, 'polygon', 0, 0, 0, 0, ?)\n                    ", final_mask_rows_to_insert)
    if final_track_rows_to_insert:
        cur.executemany('INSERT OR REPLACE INTO tracks(track_id, label) VALUES (?, ?)', final_track_rows_to_insert)
    if raw_mask_rows_to_insert:
        cur.executemany(
            '''
            INSERT OR REPLACE INTO raw_tracked_masks(
                frame,
                raw_track_id,
                raw_detection_index,
                final_track_id,
                removed_by_short_track,
                raw_track_length,
                raw_label,
                final_label,
                polygons,
                score,
                detector_score,
                class_score,
                category_id,
                category_index,
                bbox_xyxy_json,
                bbox_json,
                scene_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            raw_mask_rows_to_insert,
        )
    if raw_track_rows_to_insert:
        cur.executemany(
            '''
            INSERT OR REPLACE INTO raw_tracks(
                raw_track_id,
                final_track_id,
                removed_by_short_track,
                raw_track_length,
                raw_label,
                final_label,
                scene_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            raw_track_rows_to_insert,
        )
    if cut_frames:
        cur.executemany('INSERT OR IGNORE INTO cuts(frame) VALUES (?)', [(int(cut_frame),) for cut_frame in cut_frames])
    conn.commit()
    final_tracks = len(id_map)
    final_rows = len(final_mask_rows_to_insert)
    conn.close()
    return {'input_jsonl': str(jsonl_path), 'tracked_sqlite': str(sqlite_path), 'rows_before_prune': int(total_rows), 'rows_after_prune': final_rows, 'removed_short_tracks': int(len(remove_tids)), 'removed_rows': int(removed_rows), 'tracks_after_prune': final_tracks, 'raw_tracked_rows': int(len(raw_mask_rows_to_insert)), 'raw_tracks': int(len(raw_track_rows_to_insert)), 'raw_removed_rows': int(removed_rows), 'mixed_label_tracks': int(mixed_label_tracks), 'relabeled_mask_rows': int(relabeled_mask_rows), 'track_label_policy': 'majority_vote_per_track', 'cuts_detected': int(cuts_detected), 'scenes': int(current_scene_id + 1), 'cut_detect_enabled': bool(enable_cut_detect), 'raw_cut_method': str(raw_cut_method), 'cut_detection_method': str(cut_detection_method), 'cut_detection_elapsed_sec': float(cut_detection_elapsed_sec), 'remove_short_tracks_max_frames': int(remove_short_tracks_max_frames), 'raw_det_score_min': float(raw_det_score_min), 'raw_det_score_min_by_label': dict(sorted((raw_det_score_min_by_label or {}).items())), 'elapsed_sec': float(time.time() - start_time)}

def infer_prepare_input_sqlite(args: argparse.Namespace) -> tuple[Path, dict[str, object] | None]:
    if args.input_sqlite is not None:
        if not args.input_sqlite.exists():
            raise FileNotFoundError(f'Input sqlite not found: {args.input_sqlite}')
        return (args.input_sqlite, None)
    if args.input_jsonl is None:
        raise ValueError('Either --input-sqlite or --input-jsonl is required.')
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f'Input JSONL not found: {args.input_jsonl}')
    input_video = args.input_video if args.input_video is not None else infer_guess_input_video_path(args.input_jsonl)
    if bool(args.raw_cut_detect) and (input_video is None or not input_video.exists()):
        raise FileNotFoundError('Raw AI preprocessing requires a source video for cut detection. Pass --input-video or place a same-stem video next to the JSONL.')
    preprocess_dir = args.output_dir / 'preprocess'
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    tracked_sqlite = preprocess_dir / f'{args.input_jsonl.stem}.tracked.sqlite'
    raw_det_score_min, raw_det_score_min_by_label = infer_raw_load_score_policy(getattr(args, 'class_policy_json', None), float(args.raw_det_score_min))
    preprocess_summary = infer_build_tracked_sqlite_from_raw_jsonl(args.input_jsonl, tracked_sqlite, input_video, remove_short_tracks_max_frames=int(args.raw_remove_short_tracks_max_frames), enable_cut_detect=bool(args.raw_cut_detect), raw_det_score_min=float(raw_det_score_min), raw_det_score_min_by_label=raw_det_score_min_by_label, raw_cut_method=str(getattr(args, 'raw_cut_method', infer_RAW_CUT_METHOD_DEFAULT)))
    print(json.dumps(preprocess_summary, indent=2, ensure_ascii=False), flush=True)
    return (tracked_sqlite, preprocess_summary)

def preprocess_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Shared raw-AI preprocessing entrypoint for the merged ellipse/polygon standalone pipeline.'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-sqlite', type=Path)
    input_group.add_argument('--input-jsonl', type=Path)
    parser.add_argument('--input-video', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--raw-cut-detect', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--raw-cut-method', choices=infer_RAW_CUT_METHODS, default=infer_RAW_CUT_METHOD_DEFAULT)
    parser.add_argument('--raw-remove-short-tracks-max-frames', type=int, default=10)
    parser.add_argument('--raw-det-score-min', type=float, default=infer_RAW_DET_SCORE_MIN)
    parser.add_argument('--class-policy-json', type=Path, default=None)
    return parser

def preprocess_main() -> None:
    args = preprocess_build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracked_sqlite, preprocess_summary = infer_prepare_input_sqlite(args)
    summary = {
        'input': {
            'input_sqlite': str(args.input_sqlite) if args.input_sqlite is not None else None,
            'input_jsonl': str(args.input_jsonl) if args.input_jsonl is not None else None,
            'input_video': str(args.input_video) if args.input_video is not None else None,
        },
        'tracked_sqlite': str(tracked_sqlite),
        'preprocess_summary': preprocess_summary,
    }
    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary_path': str(summary_path), 'tracked_sqlite': str(tracked_sqlite)}, ensure_ascii=False, indent=2))

def infer_filter_rows(rows: list[tuple[int, str, str]], max_rows: int, max_tracks: int) -> list[tuple[int, str, str]]:
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
    return float(weighted_error) / max(float(gt_area), float(infer_K1_COST_NORM_AREA_FLOOR))

def infer_add_weighted_error_norm(metric_row: dict[str, object]) -> dict[str, object]:
    out = dict(metric_row)
    out['weighted_error_norm'] = infer_compute_weighted_error_norm(
        float(out.get('weighted_error', 0.0)),
        float(out.get('gt_area', 0.0)),
    )
    return out

def infer_prepare_k1_routing_metrics_lookup(k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, cost_field: str) -> dict[tuple[int, str], dict[str, object]]:
    if cost_field == 'weighted_error':
        return k1_metrics_lookup
    routing_lookup: dict[tuple[int, str], dict[str, object]] = {}
    for key, row in k1_metrics_lookup.items():
        route_cost = row.get(cost_field)
        if route_cost in (None, ''):
            if cost_field == 'weighted_error_norm':
                route_cost = infer_compute_weighted_error_norm(
                    float(row.get('weighted_error', 0.0)),
                    float(row.get('gt_area', 0.0)),
                )
            else:
                route_cost = row.get('weighted_error', 0.0)
        routing_row = dict(row)
        routing_row['weighted_error_raw'] = row.get('weighted_error', 0.0)
        routing_row['weighted_error'] = float(route_cost)
        routing_row['weighted_error_routing_field'] = str(cost_field)
        routing_lookup[key] = routing_row
    return routing_lookup

def infer_write_metrics_csv(metric_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ['frame', 'track_id', 'mode', 'candidate_name', 'gt_area', 'pred_area', 'intersection', 'union', 'recall', 'precision', 'iou', 'weighted_error', 'weighted_error_norm', 'ellipse_params', 'branch']
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([infer_add_weighted_error_norm(row) for row in metric_rows])

def infer_evaluate_mixed_metric_rows(metric_rows: list[dict[str, object]], *, total_gt_rows: int, total_sub_rows: int) -> dict[str, float]:
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
        intersection = int(row['intersection'])
        union = int(row['union'])
        gt_area = int(row['gt_area'])
        pred_area = int(row['pred_area'])
        recall = float(row['recall'])
        precision = float(row['precision'])
        iou = float(row['iou'])
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
        mode = str(row.get('mode', 'k1'))
        if mode == 'k2':
            k2_count += 1
            k2_intersection += intersection
            k2_union += union
            k2_gt_area += gt_area
        else:
            k1_count += 1
            k1_intersection += intersection
            k1_union += union
            k1_gt_area += gt_area
    return {'global_recall': total_intersection / total_gt_area if total_gt_area else 1.0, 'global_precision': total_intersection / total_pred_area if total_pred_area else 1.0, 'global_iou': total_intersection / total_union if total_union else 1.0, 'mean_recall': float(np.mean(mean_recall)) if mean_recall else 1.0, 'mean_precision': float(np.mean(mean_precision)) if mean_precision else 1.0, 'mean_iou': float(np.mean(mean_iou)) if mean_iou else 1.0, 'recall_below_090': int(recall_below_090), 'recall_below_095': int(recall_below_095), 'missing_rows': int(max(0, int(total_gt_rows) - len(metric_rows))), 'total_gt_rows': int(total_gt_rows), 'total_sub_rows': int(total_sub_rows), 'k1_count': int(k1_count), 'k2_count': int(k2_count), 'k1_recall': k1_intersection / k1_gt_area if k1_gt_area else 0.0, 'k1_iou': k1_intersection / k1_union if k1_union else 0.0, 'k2_recall': k2_intersection / k2_gt_area if k2_gt_area else 0.0, 'k2_iou': k2_intersection / k2_union if k2_union else 0.0}

def infer_solve_subset(rows_with_index: list[tuple[int, int, str, str]], payloads: list[tuple[tuple[int, int], tuple[int, int], list]], gt_polys: list[list[np.ndarray]], recall_target: float, exact_refine_rounds: int, workers: int, branch_name: str) -> tuple[list[tuple[int, tuple[int, str, str]]], list[tuple[int, dict[str, object]]], float]:
    if not rows_with_index:
        return ([], [], 0.0)
    fst.set_row_local_raster_cache([row[3] for row in rows_with_index], payloads, gt_polys)
    started = time.perf_counter()
    solved_rows: list[tuple[int, tuple[int, str, str]]] = []
    solved_metrics: list[tuple[int, dict[str, object]]] = []
    if workers > 1:
        tasks = [(subset_idx, int(frame), str(track_id), float(recall_target), int(exact_refine_rounds)) for subset_idx, (_original_idx, frame, track_id, _polygons_json) in enumerate(rows_with_index)]
        chunksize = max(1, len(tasks) // (workers * 8))
        ctx = multiprocessing.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=fst._k1_pool_init) as executor:
            for completed, result in enumerate(executor.map(fst._solve_k1_row_worker, tasks, chunksize=chunksize), start=1):
                subset_idx, row_value, metric_row, _solution_entry = result
                original_idx = rows_with_index[subset_idx][0]
                metric_row = dict(metric_row)
                metric_row['branch'] = branch_name
                metric_row['mode'] = 'k1'
                metric_row = infer_add_weighted_error_norm(metric_row)
                solved_rows.append((original_idx, row_value))
                solved_metrics.append((original_idx, metric_row))
                if completed % 1000 == 0 or completed == len(tasks):
                    print(f'{branch_name}: processed {completed}/{len(tasks)}')
    else:
        fst._k1_pool_init()
        for completed, (original_idx, frame, track_id, polygons_json) in enumerate(rows_with_index, start=1):
            subset_idx = completed - 1
            pred_json, exact, candidate_name, ellipses = fst.solve_k1_row(polygons_json, recall_target=float(recall_target), exact_refine_rounds=int(exact_refine_rounds), prepared_payload=payloads[subset_idx], gt_polys=gt_polys[subset_idx])
            metric_row = infer_add_weighted_error_norm({'frame': int(frame), 'track_id': str(track_id), 'mode': 'k1', 'candidate_name': candidate_name, 'gt_area': int(exact['gt_area']), 'pred_area': int(exact['pred_area']), 'intersection': int(exact['intersection']), 'union': int(exact['union']), 'recall': float(exact['recall']), 'precision': float(exact['precision']), 'iou': float(exact['iou']), 'weighted_error': int(exact['weighted_error']), 'ellipse_params': json.dumps(fst.serialize_ellipses(ellipses), ensure_ascii=True), 'branch': branch_name})
            solved_rows.append((original_idx, (int(frame), str(track_id), pred_json)))
            solved_metrics.append((original_idx, metric_row))
            if completed % 1000 == 0 or completed == len(rows_with_index):
                print(f'{branch_name}: processed {completed}/{len(rows_with_index)}')
    elapsed = time.perf_counter() - started
    return (solved_rows, solved_metrics, elapsed)

def infer_build_k2_solve_band_edge_aware(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], k1_ellipses_lookup: dict[tuple[int, str], list[tuple[float, float, float, float, float]]], *, threshold_default: float, threshold_edge: float, edge_keys: set[tuple[int, str]], radius: int, error_percentile: float, instability_percentile: float, instability_floor: float) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        errors = np.asarray([float(k1_metrics_lookup[key]['weighted_error']) for key in keys], dtype=np.float64)
        thresholds = np.asarray([float(threshold_edge if key in edge_keys else threshold_default) for key in keys], dtype=np.float64)
        seed_indices = {idx for idx, err in enumerate(errors) if err >= thresholds[idx]}
        high_error_cut = float(np.percentile(errors, error_percentile)) if len(errors) > 0 else float(threshold_default)
        instability_scores = np.zeros(len(track_rows), dtype=np.float64)
        for idx in range(1, len(track_rows)):
            prev_key = keys[idx - 1]
            curr_key = keys[idx]
            prev_ellipse = k1_ellipses_lookup[prev_key][0]
            curr_ellipse = k1_ellipses_lookup[curr_key][0]
            _, _, prev_scale = fst.composite_center_and_scale([prev_ellipse])
            _, _, curr_scale = fst.composite_center_and_scale([curr_ellipse])
            ref_scale = max(prev_scale, curr_scale, 8.0)
            center_jump = float(np.hypot(curr_ellipse[0] - prev_ellipse[0], curr_ellipse[1] - prev_ellipse[1])) / ref_scale
            area_jump = abs(np.log(max(fst.ellipse_area(curr_ellipse), 1.0)) - np.log(max(fst.ellipse_area(prev_ellipse), 1.0)))
            angle_jump = fst.angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
            instability_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
        instability_cut = float(np.percentile(instability_scores, instability_percentile)) if len(instability_scores) > 0 else float('inf')
        extra_indices = {idx for idx, err in enumerate(errors) if err >= max(high_error_cut, thresholds[idx] * 0.6)}
        extra_indices |= {idx for idx, score in enumerate(instability_scores) if score >= max(instability_cut, instability_floor)}
        source_indices = seed_indices | extra_indices
        for src_idx in source_indices:
            src_frame = int(track_rows[src_idx][0])
            for frame, track_id_value, _, _ in track_rows:
                if abs(int(frame) - src_frame) <= radius:
                    selected.add((int(frame), str(track_id_value)))
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'seed_count': len(seed_indices), 'expanded_count': int(sum((1 for key in keys if key in selected))), 'error_cut': float(high_error_cut), 'instability_cut': float(instability_cut) if np.isfinite(instability_cut) else None})
    summary = {'threshold': float(threshold_default), 'threshold_edge': float(threshold_edge), 'radius': int(radius), 'selected_count': len(selected), 'tracks': summary_tracks}
    return (selected, summary)

def infer_build_track_local_instability_scores(track_rows: list[tuple[int, str, str, int]], k1_ellipses_lookup: dict[tuple[int, str], list[tuple[float, float, float, float, float]]]) -> list[float]:
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
        center_jump = float(np.hypot(curr_ellipse[0] - prev_ellipse[0], curr_ellipse[1] - prev_ellipse[1])) / ref_scale
        area_jump = abs(np.log(max(fst.ellipse_area(curr_ellipse), 1.0)) - np.log(max(fst.ellipse_area(prev_ellipse), 1.0)))
        angle_jump = fst.angle_distance_deg(curr_ellipse[4], prev_ellipse[4]) / 45.0
        adjacent_scores[idx] = center_jump + 0.65 * area_jump + 0.35 * angle_jump
    instability_scores = [0.0] * len(keys)
    for idx in range(len(keys)):
        prev_score = adjacent_scores[idx - 1] if idx > 0 else 0.0
        next_score = adjacent_scores[idx] if idx < len(adjacent_scores) else 0.0
        instability_scores[idx] = float(max(prev_score, next_score))
    return instability_scores

def infer_build_k2_track_dp_selection(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], k1_ellipses_lookup: dict[tuple[int, str], list[tuple[float, float, float, float, float]]], *, threshold_default: float, threshold_edge: float, edge_keys: set[tuple[int, str]], error_weight: float, instability_weight: float, edge_bonus: float, k2_bias: float, switch_12: float, switch_21: float, short_k1_gamma: float, short_k2_gamma: float, short_k1_tau: float, short_k2_tau: float, short_len_cap: int, reset_gap: int, merge_short_k1_max_len: int, merge_short_k2_max_len: int, merge_short_k2_keep_cost: float) -> tuple[set[tuple[int, str]], dict[str, object]]:
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

    def decode_chunk(keys_chunk: list[tuple[int, str]], frames_chunk: list[int], instability_chunk: list[float]) -> tuple[list[bool], int, int, int]:
        if not keys_chunk:
            return ([], 0, 0, 0)
        inst_scale = float(np.percentile(np.asarray(instability_chunk, dtype=np.float64), 90.0)) if len(instability_chunk) > 0 else 1.0
        inst_scale = max(inst_scale, 1e-06)
        unary: list[tuple[float, float]] = []
        for idx, key in enumerate(keys_chunk):
            threshold = float(threshold_edge if key in edge_keys else threshold_default)
            err = float(k1_metrics_lookup[key]['weighted_error'])
            err_margin = (err - threshold) / max(threshold, 1.0)
            inst_norm = float(instability_chunk[idx]) / inst_scale
            evidence = float(error_weight) * err_margin + float(instability_weight) * inst_norm
            if key in edge_keys:
                evidence += float(edge_bonus)
            unary.append((0.0, float(k2_bias) - evidence))
        back_ptr: list[list[int]] = []
        prev_cost = [float('inf')] * num_states
        for mode_idx in (0, 1):
            state_id = sid(mode_idx, 1)
            prev_cost[state_id] = unary[0][mode_idx]
        back_ptr.append([-1] * num_states)
        for t in range(1, len(keys_chunk)):
            curr_cost = [float('inf')] * num_states
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
                        transition_cost = float(switch_12 if prev_mode == 0 else switch_21)
                        candidate = base_cost + transition_cost + duration_penalty(prev_mode, prev_len_bucket) + unary[t][mode_idx]
                        next_len_bucket = 1
                    next_state = sid(mode_idx, next_len_bucket)
                    if candidate < curr_cost[next_state]:
                        curr_cost[next_state] = candidate
                        curr_back[next_state] = prev_state
            prev_cost = curr_cost
            back_ptr.append(curr_back)
        best_state = -1
        best_cost = float('inf')
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

    def merge_short_islands(flags: list[bool], keys_local: list[tuple[int, str]]) -> tuple[list[bool], int, int, int, int]:
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
                if left_ok and right_ok and (out[i - 1] == out[j]) and (out[i - 1] != out[i]):
                    if not out[i] and run_len <= merge_short_k1_limit:
                        for k in range(i, j):
                            out[k] = True
                        merged_short_k1_runs += 1
                        merged_short_k1_rows += run_len
                        changed = True
                    elif out[i] and run_len <= merge_short_k2_limit:
                        max_k1_cost = max((float(k1_metrics_lookup[key]['weighted_error']) for key in keys_local[i:j]))
                        if k2_to_k1_keep_cost < 0 or max_k1_cost <= k2_to_k1_keep_cost:
                            for k in range(i, j):
                                out[k] = False
                            merged_short_k2_runs += 1
                            merged_short_k2_rows += run_len
                            changed = True
                i = j
        return (out, merged_short_k1_runs, merged_short_k1_rows, merged_short_k2_runs, merged_short_k2_rows)

    def merge_exact_inner_islands(flags: list[bool], frames_local: list[int], keys_local: list[tuple[int, str]]) -> tuple[list[bool], int, int, int, int]:
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
                while j < len(out) and out[j] == out[i] and (frames_local[j] - frames_local[j - 1] == 1):
                    j += 1
                run_len = j - i
                left_ok = i > 0 and frames_local[i] - frames_local[i - 1] == 1
                right_ok = j < len(out) and frames_local[j] - frames_local[j - 1] == 1
                if left_ok and right_ok and (out[i - 1] == out[j]) and (out[i - 1] != out[i]):
                    if not out[i] and run_len <= merge_short_k1_limit:
                        for k in range(i, j):
                            out[k] = True
                        merged_short_k1_runs += 1
                        merged_short_k1_rows += run_len
                        changed = True
                    elif out[i] and run_len <= merge_short_k2_limit:
                        max_k1_cost = max((float(k1_metrics_lookup[key]['weighted_error']) for key in keys_local[i:j]))
                        if k2_to_k1_keep_cost < 0 or max_k1_cost <= k2_to_k1_keep_cost:
                            for k in range(i, j):
                                out[k] = False
                            merged_short_k2_runs += 1
                            merged_short_k2_rows += run_len
                            changed = True
                i = j
        return (out, merged_short_k1_runs, merged_short_k1_rows, merged_short_k2_runs, merged_short_k2_rows)
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        frames = [int(frame) for frame, _, _, _ in track_rows]
        instability_scores = infer_build_track_local_instability_scores(track_rows, k1_ellipses_lookup)
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
            flags_chunk, switch_count_chunk, short_k1_runs_chunk, short_k2_runs_chunk = decode_chunk(keys_chunk=keys_chunk, frames_chunk=frames[chunk_start:idx], instability_chunk=instability_scores[chunk_start:idx])
            flags_chunk, merged_short_k1_runs_chunk, merged_short_k1_rows_chunk, merged_short_k2_runs_chunk, merged_short_k2_rows_chunk = merge_short_islands(flags_chunk, keys_chunk)
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
        final_flags, merged_short_k1_runs_track_exact, merged_short_k1_rows_track_exact, merged_short_k2_runs_track_exact, merged_short_k2_rows_track_exact = merge_exact_inner_islands(final_flags, frames, keys)
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
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'chunk_count': int(chunk_count), 'seed_count': int(sum((1 for key in keys if float(k1_metrics_lookup[key]['weighted_error']) >= float(threshold_edge if key in edge_keys else threshold_default)))), 'expanded_count': int(selected_in_track), 'switch_count': int(track_switch_count), 'short_k1_runs': int(track_short_k1_runs), 'short_k2_runs': int(track_short_k2_runs), 'merged_short_k1_runs': int(track_merged_short_k1_runs), 'merged_short_k1_rows': int(track_merged_short_k1_rows), 'merged_short_k2_runs': int(track_merged_short_k2_runs), 'merged_short_k2_rows': int(track_merged_short_k2_rows), 'error_cut': None, 'instability_cut': None})
    summary = {'routing_mode': 'track_dp', 'threshold': float(threshold_default), 'threshold_edge': float(threshold_edge), 'selected_count': len(selected), 'error_weight': float(error_weight), 'instability_weight': float(instability_weight), 'edge_bonus': float(edge_bonus), 'k2_bias': float(k2_bias), 'switch_12': float(switch_12), 'switch_21': float(switch_21), 'short_k1_gamma': float(short_k1_gamma), 'short_k2_gamma': float(short_k2_gamma), 'short_k1_tau': float(short_k1_tau), 'short_k2_tau': float(short_k2_tau), 'short_len_cap': int(bucket_cap), 'reset_gap': int(max_gap), 'merge_short_k1_max_len': int(merge_short_k1_limit), 'merge_short_k2_max_len': int(merge_short_k2_limit), 'merge_short_k2_keep_cost': float(k2_to_k1_keep_cost), 'chunk_count': int(total_chunk_count), 'switch_count': int(total_switch_count), 'short_k1_penalty_runs': int(total_short_k1_penalty_runs), 'short_k2_penalty_runs': int(total_short_k2_penalty_runs), 'merged_short_k1_runs': int(total_merged_short_k1_runs), 'merged_short_k1_rows': int(total_merged_short_k1_rows), 'merged_short_k2_runs': int(total_merged_short_k2_runs), 'merged_short_k2_rows': int(total_merged_short_k2_rows), 'tracks': summary_tracks}
    return (selected, summary)

def infer_build_k2_threshold_only_selection(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, threshold_default: float, threshold_edge: float, edge_keys: set[tuple[int, str]]) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        selected_in_track = 0
        for key in keys:
            threshold = threshold_edge if key in edge_keys else threshold_default
            err = float(k1_metrics_lookup[key]['weighted_error'])
            if err >= float(threshold):
                selected.add(key)
                selected_in_track += 1
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'seed_count': int(selected_in_track), 'expanded_count': int(selected_in_track), 'error_cut': None, 'instability_cut': None})
    summary = {'routing_mode': 'threshold_only', 'threshold': float(threshold_default), 'threshold_edge': float(threshold_edge), 'selected_count': len(selected), 'tracks': summary_tracks}
    return (selected, summary)

def infer_build_k2_threshold_soft_selection(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, threshold_default: float, threshold_edge: float, edge_keys: set[tuple[int, str]], ema_alpha: float, band_ratio: float, exit_ratio: float, strong_ratio: float, k1_keep_cost: float, reset_gap: int, merge_islands_max_len: int, merge_policy: str) -> tuple[set[tuple[int, str]], dict[str, object]]:
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

    def merge_small_islands(flags: list[bool], frames_local: list[int]) -> tuple[list[bool], int, int]:
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
                while j < len(out) and out[j] == out[i] and (frames_local[j] - frames_local[j - 1] <= max_gap):
                    j += 1
                run_len = j - i
                left_ok = i > 0 and frames_local[i] - frames_local[i - 1] <= max_gap
                right_ok = j < len(out) and frames_local[j] - frames_local[j - 1] <= max_gap
                if run_len <= merge_limit and left_ok and right_ok and (out[i - 1] == out[j]):
                    replacement = out[i - 1]
                    if merge_policy_value == 'prefer_k2' and replacement is False:
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
            err = float(k1_metrics_lookup[key]['weighted_error'])
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
                    err_now = float(k1_metrics_lookup[key]['weighted_error'])
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
        final_flags, merged_islands_in_track, merged_rows_in_track = merge_small_islands(final_flags, frames)
        for key, final_pick in zip(keys, final_flags):
            if final_pick:
                selected.add(key)
                selected_in_track += 1
        total_soft_hold += soft_hold_in_track
        total_soft_flip += soft_flip_in_track
        total_merged_islands += merged_islands_in_track
        total_merged_rows += merged_rows_in_track
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'seed_count': int(sum((1 for flag in raw_selected if flag))), 'expanded_count': int(selected_in_track), 'soft_hold_count': int(soft_hold_in_track), 'soft_flip_count': int(soft_flip_in_track), 'merged_island_count': int(merged_islands_in_track), 'merged_island_rows': int(merged_rows_in_track), 'error_cut': None, 'instability_cut': None})
    summary = {'routing_mode': 'threshold_soft', 'threshold': float(threshold_default), 'threshold_edge': float(threshold_edge), 'selected_count': len(selected), 'ema_alpha': float(alpha), 'band_ratio': float(band), 'exit_ratio': float(exit_band), 'strong_ratio': float(strong), 'k1_keep_cost': float(k1_keep_cost_threshold), 'reset_gap': int(max_gap), 'merge_policy': merge_policy_value, 'soft_hold_count': int(total_soft_hold), 'soft_flip_count': int(total_soft_flip), 'merged_island_count': int(total_merged_islands), 'merged_island_rows': int(total_merged_rows), 'tracks': summary_tracks}
    return (selected, summary)

def infer_build_k2_threshold_hysteresis_selection(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, enter_default: float, enter_edge: float, exit_default: float, exit_edge: float, edge_keys: set[tuple[int, str]], confirm_frames: int, reset_gap: int) -> tuple[set[tuple[int, str]], dict[str, object]]:
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
            err = float(k1_metrics_lookup[key]['weighted_error'])
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
        summary_tracks.append({'track_id': track_id, 'frame_count': len(track_rows), 'seed_count': int(sum((1 for key in keys if float(k1_metrics_lookup[key]['weighted_error']) >= float(enter_edge if key in edge_keys else enter_default)))), 'expanded_count': int(sum((1 for flag in flags if flag))), 'promoted_by_confirm': int(promoted_in_track), 'error_cut': None, 'instability_cut': None})
    summary = {'routing_mode': 'threshold_hysteresis', 'enter_threshold': float(enter_default), 'enter_threshold_edge': float(enter_edge), 'exit_threshold': float(exit_default), 'exit_threshold_edge': float(exit_edge), 'confirm_frames': int(effective_confirm), 'reset_gap': int(max_gap), 'selected_count': len(selected), 'promoted_by_confirm': int(total_pending_promotions), 'tracks': summary_tracks}
    return (selected, summary)

def infer_count_k2_switch_stats(flags: list[bool], frames: list[int], *, reset_gap: int) -> dict[str, int]:
    if not flags:
        return {
            'switch_count': 0,
            'k1_run_count': 0,
            'k2_run_count': 0,
            'k1_single_frame_islands': 0,
            'k1_two_frame_islands': 0,
            'k2_single_frame_islands': 0,
            'k2_two_frame_islands': 0,
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
        if int(frames[right[1]]) - int(frames[left[2] - 1]) <= max_gap and left[0] != right[0]:
            switch_count += 1
    stats = {
        'switch_count': int(switch_count),
        'k1_run_count': int(sum(1 for mode, _start, _end in runs if not mode)),
        'k2_run_count': int(sum(1 for mode, _start, _end in runs if mode)),
        'k1_single_frame_islands': 0,
        'k1_two_frame_islands': 0,
        'k2_single_frame_islands': 0,
        'k2_two_frame_islands': 0,
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
            stats['k2_single_frame_islands' if mode else 'k1_single_frame_islands'] += 1
        elif run_len == 2:
            stats['k2_two_frame_islands' if mode else 'k1_two_frame_islands'] += 1
    return stats

def infer_build_k2_k1n_sequence_selection(rows_by_track: dict[str, list[tuple[int, str, str, int]]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, enter_threshold: float, exit_threshold: float, strong_enter_threshold: float, strong_exit_threshold: float, protect_k2_iou_below: float, smooth_window: int, enter_confirm_frames: int, exit_confirm_frames: int, merge_short_k1_max_len: int, merge_short_k2_max_len: int, reset_gap: int) -> tuple[set[tuple[int, str]], dict[str, object]]:
    selected: set[tuple[int, str]] = set()
    summary_tracks: list[dict[str, object]] = []
    enter = float(enter_threshold)
    exit_value = min(float(exit_threshold), enter)
    protect_iou = float(protect_k2_iou_below)
    iou_equivalent_enter = (1.0 / protect_iou - 1.0) if 0.0 < protect_iou < 1.0 else enter * 1.5
    strong_enter = float(strong_enter_threshold if float(strong_enter_threshold) >= 0.0 else max(enter * 1.5, iou_equivalent_enter))
    strong_enter = max(strong_enter, enter)
    strong_exit = float(strong_exit_threshold if float(strong_exit_threshold) >= 0.0 else exit_value * 0.65)
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
        'seed_count': 0,
        'strong_seed_count': 0,
        'pre_cleanup_selected_count': 0,
        'selected_count': 0,
        'promoted_by_confirm': 0,
        'exited_by_confirm': 0,
        'merged_short_k1_runs': 0,
        'merged_short_k1_rows': 0,
        'removed_short_k2_runs': 0,
        'removed_short_k2_rows': 0,
        'protected_short_k1_runs': 0,
        'protected_short_k2_runs': 0,
        'protected_short_k2_iou_runs': 0,
        'protected_short_k2_iou_rows': 0,
        'protected_short_k2_cost_runs': 0,
        'protected_short_k2_cost_rows': 0,
        'switch_count': 0,
        'k1_single_frame_islands': 0,
        'k1_two_frame_islands': 0,
        'k2_single_frame_islands': 0,
        'k2_two_frame_islands': 0,
    }

    def smooth_costs(costs: list[float]) -> list[float]:
        if window <= 1 or len(costs) <= 2:
            return list(costs)
        out: list[float] = []
        for idx in range(len(costs)):
            left = max(0, idx - radius)
            right = min(len(costs), idx + radius + 1)
            out.append(float(np.median(np.asarray(costs[left:right], dtype=np.float64))))
        return out

    def split_chunks(frames_local: list[int]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        start = 0
        for idx in range(1, len(frames_local) + 1):
            at_end = idx == len(frames_local)
            has_gap = (not at_end) and int(frames_local[idx]) - int(frames_local[idx - 1]) > max_gap
            if at_end or has_gap:
                chunks.append((start, idx))
                start = idx
        return chunks

    def apply_hysteresis(costs: list[float], smooth: list[float], ious: list[float]) -> tuple[list[bool], int, int]:
        flags = [False] * len(costs)
        in_k2 = False
        pending_enter: list[int] = []
        pending_exit: list[int] = []
        promoted = 0
        exited = 0
        for idx, (cost, score, iou) in enumerate(zip(costs, smooth, ious, strict=True)):
            strong_k2 = cost >= strong_enter or (protect_iou > 0.0 and iou < protect_iou)
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

    def merge_protected_short_islands(flags: list[bool], frames_local: list[int], costs: list[float], ious: list[float]) -> tuple[list[bool], dict[str, int]]:
        out = list(flags)
        local_stats = {
            'merged_short_k1_runs': 0,
            'merged_short_k1_rows': 0,
            'removed_short_k2_runs': 0,
            'removed_short_k2_rows': 0,
            'protected_short_k1_runs': 0,
            'protected_short_k2_runs': 0,
            'protected_short_k2_iou_runs': 0,
            'protected_short_k2_iou_rows': 0,
            'protected_short_k2_cost_runs': 0,
            'protected_short_k2_cost_rows': 0,
        }
        if len(out) < 3 or (merge_k1_limit <= 0 and merge_k2_limit <= 0):
            return (out, local_stats)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(out):
                j = i + 1
                while j < len(out) and out[j] == out[i] and int(frames_local[j]) - int(frames_local[j - 1]) <= max_gap:
                    j += 1
                run_len = j - i
                left_ok = i > 0 and int(frames_local[i]) - int(frames_local[i - 1]) <= max_gap
                right_ok = j < len(out) and int(frames_local[j]) - int(frames_local[j - 1]) <= max_gap
                if left_ok and right_ok and out[i - 1] == out[j] and out[i - 1] != out[i]:
                    if (not out[i]) and merge_k1_limit > 0 and run_len <= merge_k1_limit:
                        for k in range(i, j):
                            out[k] = True
                        local_stats['merged_short_k1_runs'] += 1
                        local_stats['merged_short_k1_rows'] += run_len
                        changed = True
                    elif out[i] and merge_k2_limit > 0 and run_len <= merge_k2_limit:
                        protect_by_cost = max(costs[i:j]) >= strong_enter
                        protect_by_iou = protect_iou > 0.0 and min(ious[i:j]) < protect_iou
                        if protect_by_cost or protect_by_iou:
                            local_stats['protected_short_k2_runs'] += 1
                            if protect_by_cost:
                                local_stats['protected_short_k2_cost_runs'] += 1
                                local_stats['protected_short_k2_cost_rows'] += run_len
                            if protect_by_iou:
                                local_stats['protected_short_k2_iou_runs'] += 1
                                local_stats['protected_short_k2_iou_rows'] += run_len
                        else:
                            for k in range(i, j):
                                out[k] = False
                            local_stats['removed_short_k2_runs'] += 1
                            local_stats['removed_short_k2_rows'] += run_len
                            changed = True
                i = j
        return (out, local_stats)

    for track_id, track_rows in rows_by_track.items():
        ordered_rows = sorted(track_rows, key=lambda row: int(row[0]))
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in ordered_rows]
        frames = [int(frame) for frame, _, _, _ in ordered_rows]
        costs = [float(k1_metrics_lookup[key]['weighted_error']) for key in keys]
        ious = [float(k1_metrics_lookup[key].get('iou', 1.0)) for key in keys]
        final_flags = [False] * len(keys)
        track_promoted = 0
        track_exited = 0
        track_cleanup = {
            'merged_short_k1_runs': 0,
            'merged_short_k1_rows': 0,
            'removed_short_k2_runs': 0,
            'removed_short_k2_rows': 0,
            'protected_short_k1_runs': 0,
            'protected_short_k2_runs': 0,
            'protected_short_k2_iou_runs': 0,
            'protected_short_k2_iou_rows': 0,
            'protected_short_k2_cost_runs': 0,
            'protected_short_k2_cost_rows': 0,
        }
        for start, end in split_chunks(frames):
            chunk_costs = costs[start:end]
            chunk_smooth = smooth_costs(chunk_costs)
            chunk_ious = ious[start:end]
            chunk_flags, promoted, exited = apply_hysteresis(chunk_costs, chunk_smooth, chunk_ious)
            chunk_flags, cleanup_stats = merge_protected_short_islands(chunk_flags, frames[start:end], chunk_costs, chunk_ious)
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
        pre_cleanup_selected_count = selected_in_track + track_cleanup['removed_short_k2_rows'] - track_cleanup['merged_short_k1_rows']
        switch_stats = infer_count_k2_switch_stats(final_flags, frames, reset_gap=max_gap)
        for key in ('seed_count', 'strong_seed_count', 'pre_cleanup_selected_count', 'selected_count', 'promoted_by_confirm', 'exited_by_confirm'):
            if key == 'seed_count':
                totals[key] += seed_count
            elif key == 'strong_seed_count':
                totals[key] += strong_seed_count
            elif key == 'pre_cleanup_selected_count':
                totals[key] += pre_cleanup_selected_count
            elif key == 'selected_count':
                totals[key] += selected_in_track
            elif key == 'promoted_by_confirm':
                totals[key] += track_promoted
            elif key == 'exited_by_confirm':
                totals[key] += track_exited
        for key, value in track_cleanup.items():
            totals[key] += int(value)
        for key in ('switch_count', 'k1_single_frame_islands', 'k1_two_frame_islands', 'k2_single_frame_islands', 'k2_two_frame_islands'):
            totals[key] += int(switch_stats[key])
        summary_tracks.append({
            'track_id': str(track_id),
            'frame_count': len(keys),
            'seed_count': int(seed_count),
            'strong_seed_count': int(strong_seed_count),
            'pre_cleanup_selected_count': int(pre_cleanup_selected_count),
            'expanded_count': int(selected_in_track),
            'switch_count': int(switch_stats['switch_count']),
            'k1_run_count': int(switch_stats['k1_run_count']),
            'k2_run_count': int(switch_stats['k2_run_count']),
            'k1_single_frame_islands': int(switch_stats['k1_single_frame_islands']),
            'k1_two_frame_islands': int(switch_stats['k1_two_frame_islands']),
            'k2_single_frame_islands': int(switch_stats['k2_single_frame_islands']),
            'k2_two_frame_islands': int(switch_stats['k2_two_frame_islands']),
            'promoted_by_confirm': int(track_promoted),
            'exited_by_confirm': int(track_exited),
            **{key: int(value) for key, value in track_cleanup.items()},
            'cost_min': float(min(costs)) if costs else None,
            'cost_p50': float(np.percentile(np.asarray(costs, dtype=np.float64), 50.0)) if costs else None,
            'cost_p90': float(np.percentile(np.asarray(costs, dtype=np.float64), 90.0)) if costs else None,
            'cost_max': float(max(costs)) if costs else None,
            'iou_min': float(min(ious)) if ious else None,
            'error_cut': None,
            'instability_cut': None,
        })
    summary = {
        'routing_mode': 'k1n_sequence',
        'cost_feature': 'k1_cost_norm_sequence',
        'enter_threshold': float(enter),
        'exit_threshold': float(exit_value),
        'strong_enter_threshold': float(strong_enter),
        'strong_exit_threshold': float(strong_exit),
        'protect_k2_iou_below': float(protect_iou),
        'smooth_window': int(window),
        'enter_confirm_frames': int(enter_confirm),
        'exit_confirm_frames': int(exit_confirm),
        'merge_short_k1_max_len': int(merge_k1_limit),
        'merge_short_k2_max_len': int(merge_k2_limit),
        'reset_gap': int(max_gap),
        **{key: int(value) for key, value in totals.items()},
        'tracks': summary_tracks,
    }
    return (selected, summary)

def infer_cleanup_selected_k2_inner_islands(rows_by_track: dict[str, list[tuple[int, str, str, int]]], selected_keys: set[tuple[int, str]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, max_len: int, keep_cost: float) -> tuple[set[tuple[int, str]], dict[str, int]]:
    limit = max(0, int(max_len))
    if limit <= 0:
        return (set(selected_keys), {'removed_runs': 0, 'removed_rows': 0, 'removed_exact_runs': 0, 'removed_exact_rows': 0, 'removed_track_order_runs': 0, 'removed_track_order_rows': 0, 'removed_exact_singleton_runs': 0, 'removed_exact_singleton_rows': 0})
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
            right_ok = j < len(flags) and (not flags[j]) and (frames[j] - frames[j - 1] == 1)
            if left_ok and right_ok and (run_len <= limit):
                max_k1_cost = max((float(k1_metrics_lookup[key]['weighted_error']) for key in keys[i:j]))
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
                max_k1_cost = max((float(k1_metrics_lookup[key]['weighted_error']) for key in keys[i:j]))
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
                max_k1_cost = float(k1_metrics_lookup[key]['weighted_error'])
                if keep_cost < 0 or max_k1_cost <= float(keep_cost):
                    cleaned.discard(key)
                    removed_exact_singleton_runs += 1
                    removed_exact_singleton_rows += 1
            i = j
    removed_runs = removed_exact_runs + removed_track_order_runs + removed_exact_singleton_runs
    removed_rows = removed_exact_rows + removed_track_order_rows + removed_exact_singleton_rows
    return (cleaned, {'removed_runs': int(removed_runs), 'removed_rows': int(removed_rows), 'removed_exact_runs': int(removed_exact_runs), 'removed_exact_rows': int(removed_exact_rows), 'removed_track_order_runs': int(removed_track_order_runs), 'removed_track_order_rows': int(removed_track_order_rows), 'removed_exact_singleton_runs': int(removed_exact_singleton_runs), 'removed_exact_singleton_rows': int(removed_exact_singleton_rows)})

def infer_promote_short_k1_runs_to_k2(rows_by_track: dict[str, list[tuple[int, str, str, int]]], selected_keys: set[tuple[int, str]], *, max_len: int) -> tuple[set[tuple[int, str]], dict[str, int]]:
    limit = max(0, int(max_len))
    if limit <= 0:
        return (set(selected_keys), {'promoted_runs': 0, 'promoted_rows': 0})
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
    return (promoted, {'promoted_runs': int(promoted_runs), 'promoted_rows': int(promoted_rows)})

def infer_force_select_high_cost_rows(rows_by_track: dict[str, list[tuple[int, str, str, int]]], selected_keys: set[tuple[int, str]], k1_metrics_lookup: dict[tuple[int, str], dict[str, object]], *, min_cost: float) -> tuple[set[tuple[int, str]], dict[str, int]]:
    threshold = float(min_cost)
    if threshold < 0:
        return (set(selected_keys), {'forced_runs': 0, 'forced_rows': 0})
    forced = set(selected_keys)
    forced_runs = 0
    forced_rows = 0
    for track_id, track_rows in rows_by_track.items():
        keys = [(int(frame), str(track_id)) for frame, _, _, _ in track_rows]
        i = 0
        while i < len(keys):
            key = keys[i]
            is_high_cost = float(k1_metrics_lookup[key]['weighted_error']) > threshold
            if key in forced or not is_high_cost:
                i += 1
                continue
            j = i + 1
            while j < len(keys):
                key_j = keys[j]
                if key_j in forced or float(k1_metrics_lookup[key_j]['weighted_error']) <= threshold:
                    break
                j += 1
            for key_run in keys[i:j]:
                forced.add(key_run)
            forced_runs += 1
            forced_rows += j - i
            i = j
    return (forced, {'forced_runs': int(forced_runs), 'forced_rows': int(forced_rows)})

def infer_build_k2_input_from_payload(payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]], *, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    k2v5 = infer_get_k2v5_module()
    gt_mask, _origin = fst.rasterize_local_mask_from_payload(payload)
    gt_square, _pad, _side = k2v5.square_pad_mask(gt_mask)
    padded_mask = gt_square.astype(np.uint8, copy=False)
    signed = k2v5.build_signed_distance_channel(padded_mask)
    edge = k2v5.build_edge_channel(padded_mask)
    mask_resized = cv2.resize(padded_mask.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    signed_resized = cv2.resize(signed, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    edge_resized = cv2.resize(edge, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    touch_flags = k2v5.edge_touch_vector_from_row({}, gt_mask=padded_mask)
    input_image = k2v5.build_input_image(mask_resized, signed_resized, edge_resized, touch_flags, image_size)
    return (input_image.astype(np.float32, copy=False), padded_mask)

def infer_k2_v5_forward_states_only(model: object, image_tensor: 'torch.Tensor', *, torch_module: object, k2v5_module: object) -> 'torch.Tensor':
    torch = torch_module
    inner = model
    s1 = inner.stem(image_tensor)
    s2 = inner.stage2(s1)
    s3 = inner.stage3(s2)
    s4 = inner.stage4(s3)
    s5 = inner.stage5(s4)
    p5 = inner.lat5(s5)
    p4 = inner.fpn4(inner.lat4(s4) + torch.nn.functional.interpolate(p5, size=s4.shape[-2:], mode='bilinear', align_corners=False))
    p3 = inner.fpn3(inner.lat3(s3) + torch.nn.functional.interpolate(p4, size=s3.shape[-2:], mode='bilinear', align_corners=False))
    context_map = inner.context_proj(p4)
    context_tokens = context_map.flatten(2).transpose(1, 2)
    queries = inner.slot_queries.unsqueeze(0).expand(image_tensor.shape[0], -1, -1)
    global_feat = inner.global_pool(torch.nn.functional.interpolate(p3, size=context_map.shape[-2:], mode='bilinear', align_corners=False))
    global_feat = global_feat.flatten(1).unsqueeze(1).expand(-1, 2, -1)
    queries = queries + inner.slot_refine(torch.cat([queries, global_feat], dim=-1))
    for block in inner.decoder:
        queries = block(queries, context_tokens)
    centers = inner.center_head(queries)
    chol_params = inner.chol_head(queries)
    states = k2v5_module.spd_to_normalized_states(centers, chol_params)
    return states.flatten(1)

def infer_infer_k2_v5(selected_rows: list[tuple[int, int, str, str]], payloads: list[tuple[tuple[int, int], tuple[int, int], list]], gt_polys: list[list[np.ndarray]], *, run_dir: Path, device_name: str, batch_size: int, prep_workers: int=0, precision: str='fp32', forward_mode: str='full', profile_stages: bool=False, cudnn_benchmark: str='off', tf32: str='default') -> tuple[list[tuple[int, tuple[int, str, str]]], list[tuple[int, dict[str, object]]], float, dict[str, object]]:
    if not selected_rows:
        return ([], [], 0.0, {})
    torch = infer_get_torch_module()
    k2v5 = infer_get_k2v5_module()
    checkpoint_path = run_dir / 'best_exact.pt'
    device = torch.device(device_name if device_name != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.backends.cudnn.benchmark = str(cudnn_benchmark) == 'on'
        if str(tf32) == 'on':
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        elif str(tf32) == 'off':
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
    requested_precision = str(precision)
    effective_precision = 'fp16' if requested_precision == 'fp16' and use_cuda else 'fp32'
    effective_forward_mode = str(forward_mode)
    model = k2v5.K2SlotSetSPDNet(in_channels=10, base_width=int(infer_K2_V5_INFER_CONFIG['base_width']), slot_dim=int(infer_K2_V5_INFER_CONFIG['slot_dim']), decoder_layers=int(infer_K2_V5_INFER_CONFIG['decoder_layers']), num_heads=int(infer_K2_V5_INFER_CONFIG['num_heads']), sharpness=float(infer_K2_V5_INFER_CONFIG['render_sharpness'])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    if effective_precision == 'fp16':
        model = model.half()
    model.eval()
    image_size = int(infer_K2_V5_INFER_CONFIG['image_size'])
    started = time.perf_counter()
    solved_rows: list[tuple[int, tuple[int, str, str]]] = []
    solved_metrics: list[tuple[int, dict[str, object]]] = []
    effective_prep_workers = max(0, int(prep_workers))
    stage_timings = {'prepare_batch': 0.0, 'h2d': 0.0, 'forward': 0.0, 'd2h_states': 0.0, 'postprocess_exact': 0.0}

    def profile_sync() -> None:
        if profile_stages and use_cuda:
            torch.cuda.synchronize(device)

    def prepare_batch(batch_rows: list[tuple[int, int, str, str]], batch_payloads: list[tuple[tuple[int, int], tuple[int, int], list]], batch_gt_polys: list[list[np.ndarray]], executor: concurrent.futures.ThreadPoolExecutor | None) -> tuple[list[tuple[int, int, str, str]], list[tuple[tuple[int, int], tuple[int, int], list]], list[list[np.ndarray]], np.ndarray]:
        batch_count = len(batch_rows)
        image_batch = np.empty((batch_count, 10, image_size, image_size), dtype=np.float32)
        if executor is None or batch_count <= 1:
            for local_idx, payload in enumerate(batch_payloads):
                input_image, _ = infer_build_k2_input_from_payload(payload, image_size=image_size)
                image_batch[local_idx] = input_image
            return (batch_rows, batch_payloads, batch_gt_polys, image_batch)
        prepared = list(executor.map(lambda payload: infer_build_k2_input_from_payload(payload, image_size=image_size)[0], batch_payloads))
        for local_idx, input_image in enumerate(prepared):
            image_batch[local_idx] = input_image
        return (batch_rows, batch_payloads, batch_gt_polys, image_batch)
    prep_executor: concurrent.futures.ThreadPoolExecutor | None = None
    prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
    if effective_prep_workers > 1:
        prep_executor = concurrent.futures.ThreadPoolExecutor(max_workers=effective_prep_workers)
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
                    prepared_batch = prepare_batch(batch_rows, batch_payloads, batch_gt_polys, prep_executor)
                else:
                    prepared_batch = future.result()
                    future = None
                stage_timings['prepare_batch'] += time.perf_counter() - stage_start
                next_start = end_idx
                if next_start < len(selected_rows) and prefetch_executor is not None:
                    next_end = min(len(selected_rows), next_start + batch_step)
                    future = prefetch_executor.submit(prepare_batch, selected_rows[next_start:next_end], payloads[next_start:next_end], gt_polys[next_start:next_end], prep_executor)
                batch_rows, batch_payloads, batch_gt_polys, image_batch = prepared_batch
                profile_sync()
                stage_start = time.perf_counter()
                if effective_precision == 'fp16':
                    image_tensor = torch.from_numpy(image_batch).to(device=device, dtype=torch.float16, non_blocking=False)
                else:
                    image_tensor = torch.from_numpy(image_batch).to(device, non_blocking=False)
                profile_sync()
                stage_timings['h2d'] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                if effective_forward_mode == 'states_only':
                    pred_states_tensor = infer_k2_v5_forward_states_only(model, image_tensor, torch_module=torch, k2v5_module=k2v5)
                else:
                    pred_output = model(image_tensor)
                    pred_states_tensor = pred_output['states']
                profile_sync()
                stage_timings['forward'] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                pred_states = pred_states_tensor.view(image_tensor.shape[0], 2, 6).detach().float().cpu().numpy()
                profile_sync()
                stage_timings['d2h_states'] += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                for local_idx, (original_idx, frame, track_id, _polygons_json) in enumerate(batch_rows):
                    payload = batch_payloads[local_idx]
                    pred_abs = k2v5.states_to_abs_ellipses_from_payload(pred_states[local_idx], payload)
                    pred_polys = fst.ellipses_to_polygon_arrays(pred_abs)
                    pred_json = json.dumps([poly.astype(np.float64).tolist() for poly in pred_polys], ensure_ascii=True)
                    exact = fst.compute_exact_metrics_from_polygons(batch_gt_polys[local_idx], pred_polys)
                    exact['weighted_error'] = int(fst.compute_weighted_error(exact))
                    metric_row = infer_add_weighted_error_norm({'frame': int(frame), 'track_id': str(track_id), 'mode': 'k2', 'candidate_name': 'v5_slot_set_spd', 'gt_area': int(exact['gt_area']), 'pred_area': int(exact['pred_area']), 'intersection': int(exact['intersection']), 'union': int(exact['union']), 'recall': float(exact['recall']), 'precision': float(exact['precision']), 'iou': float(exact['iou']), 'weighted_error': int(exact['weighted_error']), 'ellipse_params': json.dumps(fst.serialize_ellipses(pred_abs), ensure_ascii=True), 'branch': 'k2_v5'})
                    solved_rows.append((original_idx, (int(frame), str(track_id), pred_json)))
                    solved_metrics.append((original_idx, metric_row))
                stage_timings['postprocess_exact'] += time.perf_counter() - stage_start
                processed = min(end_idx, len(selected_rows))
                if processed % 1000 == 0 or processed == len(selected_rows):
                    print(f'k2_v5: processed {processed}/{len(selected_rows)}')
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if prep_executor is not None:
            prep_executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    model_info = {'run_dir': str(run_dir), 'checkpoint_path': str(checkpoint_path), 'device': str(device), 'image_size': image_size, 'base_width': int(infer_K2_V5_INFER_CONFIG['base_width']), 'slot_dim': int(infer_K2_V5_INFER_CONFIG['slot_dim']), 'decoder_layers': int(infer_K2_V5_INFER_CONFIG['decoder_layers']), 'num_heads': int(infer_K2_V5_INFER_CONFIG['num_heads']), 'render_sharpness': float(infer_K2_V5_INFER_CONFIG['render_sharpness']), 'prep_workers': effective_prep_workers, 'precision_requested': requested_precision, 'precision': effective_precision, 'forward_mode': effective_forward_mode, 'profile_stages': bool(profile_stages), 'cudnn_benchmark': str(cudnn_benchmark), 'tf32': str(tf32), 'stage_timing_sec': stage_timings if profile_stages else {}}
    return (solved_rows, solved_metrics, elapsed, model_info)

def infer_main() -> None:
    args = infer_build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_sqlite_path, raw_preprocess_summary = infer_prepare_input_sqlite(args)
    t0 = time.perf_counter()
    load_start = time.perf_counter()
    source_rows = infer_filter_rows(fst.load_rows(input_sqlite_path), max_rows=int(args.max_rows), max_tracks=int(args.max_tracks))
    load_rows_sec = time.perf_counter() - load_start
    row_gt_polygons: list[list[np.ndarray]] = []
    row_local_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    edge_flags: list[bool] = []
    parse_polygons_sec = 0.0
    prepare_payloads_sec = 0.0
    classify_edge_sec = 0.0
    prep_loop_start = time.perf_counter()
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
    preprocess_loop_sec = time.perf_counter() - prep_loop_start
    split_start = time.perf_counter()
    edge_rows: list[tuple[int, int, str, str]] = []
    edge_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    edge_gt_polys: list[list[np.ndarray]] = []
    nonedge_rows: list[tuple[int, int, str, str]] = []
    nonedge_payloads: list[tuple[tuple[int, int], tuple[int, int], list]] = []
    nonedge_gt_polys: list[list[np.ndarray]] = []
    for original_idx, (row, polys, payload, is_edge) in enumerate(zip(source_rows, row_gt_polygons, row_local_payloads, edge_flags, strict=True)):
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
    edge_solved_rows, edge_solved_metrics, edge_solve_sec = infer_solve_subset(rows_with_index=edge_rows, payloads=edge_payloads, gt_polys=edge_gt_polys, recall_target=float(args.k1_recall_target), exact_refine_rounds=int(args.k1_exact_refine_rounds), workers=workers_edge, branch_name='edge')
    nonedge_solved_rows, nonedge_solved_metrics, nonedge_solve_sec = infer_solve_subset(rows_with_index=nonedge_rows, payloads=nonedge_payloads, gt_polys=nonedge_gt_polys, recall_target=float(args.k1_recall_target), exact_refine_rounds=int(args.k1_exact_refine_rounds), workers=workers_nonedge, branch_name='nonedge')
    merge_k1_start = time.perf_counter()
    k1_submission_rows_indexed: list[tuple[int, str, str] | None] = [None] * len(source_rows)
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
    k1_ellipses_lookup: dict[tuple[int, str], list[tuple[float, float, float, float, float]]] = {}
    edge_keys: set[tuple[int, str]] = set()
    for idx, (frame, track_id, polygons_json) in enumerate(source_rows):
        rows_by_track.setdefault(str(track_id), []).append((int(frame), str(track_id), str(polygons_json), idx))
        if edge_flags[idx]:
            edge_keys.add((int(frame), str(track_id)))
    for row in k1_metric_rows:
        key = (int(row['frame']), str(row['track_id']))
        k1_metrics_lookup[key] = row
        k1_ellipses_lookup[key] = [tuple(map(float, e)) for e in json.loads(str(row['ellipse_params']))]
    k2_band_start = time.perf_counter()
    use_normalized_k1_cost = str(args.k1_cost_routing) == 'normalized'
    routing_cost_field = 'weighted_error_norm' if use_normalized_k1_cost else 'weighted_error'
    routing_metrics_lookup = infer_prepare_k1_routing_metrics_lookup(k1_metrics_lookup, cost_field=routing_cost_field)
    if use_normalized_k1_cost:
        threshold_default = float(args.threshold_norm)
        effective_threshold_edge = float(args.threshold_norm if float(args.threshold_edge_norm) < 0.0 else args.threshold_edge_norm)
        soft_k1_keep_cost = float(args.k2_soft_k1_keep_cost_norm)
        hyst_enter = float(args.k2_hyst_enter_norm)
        hyst_enter_edge = float(args.k2_hyst_enter_norm if float(args.k2_hyst_enter_edge_norm) < 0.0 else args.k2_hyst_enter_edge_norm)
        hyst_exit = float(args.k2_hyst_exit_norm)
        hyst_exit_edge = float(args.k2_hyst_exit_norm if float(args.k2_hyst_exit_edge_norm) < 0.0 else args.k2_hyst_exit_edge_norm)
        k1n_seq_enter = float(args.threshold_norm if float(args.k1n_seq_enter_norm) < 0.0 else args.k1n_seq_enter_norm)
        k1n_seq_exit = float(args.k1n_seq_exit_norm)
        k1n_seq_strong_enter = float(args.k1n_seq_strong_enter_norm)
        k1n_seq_strong_exit = float(args.k1n_seq_strong_exit_norm)
        dp_merge_short_k2_keep_cost = float(args.k2_dp_merge_short_k2_keep_cost_norm)
        dp_force_k2_cost = float(args.k2_dp_force_k2_cost_norm)
    else:
        threshold_default = float(args.threshold)
        effective_threshold_edge = float(args.threshold if int(args.threshold_edge) < 0 else args.threshold_edge)
        soft_k1_keep_cost = float(args.k2_soft_k1_keep_cost)
        hyst_enter = float(args.k2_hyst_enter)
        hyst_enter_edge = float(args.k2_hyst_enter if int(args.k2_hyst_enter_edge) < 0 else args.k2_hyst_enter_edge)
        hyst_exit = float(args.k2_hyst_exit)
        hyst_exit_edge = float(args.k2_hyst_exit if int(args.k2_hyst_exit_edge) < 0 else args.k2_hyst_exit_edge)
        k1n_seq_enter = threshold_default
        k1n_seq_exit = hyst_exit
        k1n_seq_strong_enter = -1.0
        k1n_seq_strong_exit = -1.0
        dp_merge_short_k2_keep_cost = float(args.k2_dp_merge_short_k2_keep_cost)
        dp_force_k2_cost = float(args.k2_dp_force_k2_cost)
    if str(args.routing_mode) == 'threshold_only':
        selected_keys, k2_band_summary = infer_build_k2_threshold_only_selection(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, threshold_default=threshold_default, threshold_edge=effective_threshold_edge, edge_keys=edge_keys)
    elif str(args.routing_mode) == 'threshold_soft':
        selected_keys, k2_band_summary = infer_build_k2_threshold_soft_selection(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, threshold_default=threshold_default, threshold_edge=effective_threshold_edge, edge_keys=edge_keys, ema_alpha=float(args.k2_soft_ema_alpha), band_ratio=float(args.k2_soft_band_ratio), exit_ratio=float(args.k2_soft_exit_ratio), strong_ratio=float(args.k2_soft_strong_ratio), k1_keep_cost=soft_k1_keep_cost, reset_gap=int(args.k2_soft_reset_gap), merge_islands_max_len=int(args.k2_soft_merge_islands_max_len), merge_policy=str(args.k2_soft_merge_policy))
    elif str(args.routing_mode) == 'threshold_hysteresis':
        selected_keys, k2_band_summary = infer_build_k2_threshold_hysteresis_selection(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, enter_default=hyst_enter, enter_edge=hyst_enter_edge, exit_default=hyst_exit, exit_edge=hyst_exit_edge, edge_keys=edge_keys, confirm_frames=int(args.k2_hyst_confirm_frames), reset_gap=int(args.k2_hyst_reset_gap))
    elif str(args.routing_mode) == 'k1n_sequence':
        selected_keys, k2_band_summary = infer_build_k2_k1n_sequence_selection(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, enter_threshold=k1n_seq_enter, exit_threshold=k1n_seq_exit, strong_enter_threshold=k1n_seq_strong_enter, strong_exit_threshold=k1n_seq_strong_exit, protect_k2_iou_below=float(args.k1n_seq_protect_k2_iou_below), smooth_window=int(args.k1n_seq_smooth_window), enter_confirm_frames=int(args.k1n_seq_enter_confirm_frames), exit_confirm_frames=int(args.k1n_seq_exit_confirm_frames), merge_short_k1_max_len=int(args.k1n_seq_merge_short_k1_max_len), merge_short_k2_max_len=int(args.k1n_seq_merge_short_k2_max_len), reset_gap=int(args.k1n_seq_reset_gap))
    elif str(args.routing_mode) == 'track_dp':
        selected_keys, k2_band_summary = infer_build_k2_track_dp_selection(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, k1_ellipses_lookup=k1_ellipses_lookup, threshold_default=threshold_default, threshold_edge=effective_threshold_edge, edge_keys=edge_keys, error_weight=float(args.k2_dp_error_weight), instability_weight=float(args.k2_dp_instability_weight), edge_bonus=float(args.k2_dp_edge_bonus), k2_bias=float(args.k2_dp_k2_bias), switch_12=float(args.k2_dp_switch_12), switch_21=float(args.k2_dp_switch_21), short_k1_gamma=float(args.k2_dp_short_k1_gamma), short_k2_gamma=float(args.k2_dp_short_k2_gamma), short_k1_tau=float(args.k2_dp_short_k1_tau), short_k2_tau=float(args.k2_dp_short_k2_tau), short_len_cap=int(args.k2_dp_short_len_cap), reset_gap=int(args.k2_dp_reset_gap), merge_short_k1_max_len=int(args.k2_dp_merge_short_k1_max_len), merge_short_k2_max_len=int(args.k2_dp_merge_short_k2_max_len), merge_short_k2_keep_cost=dp_merge_short_k2_keep_cost)
    elif (not use_normalized_k1_cost) and effective_threshold_edge == threshold_default:
        selected_keys, k2_band_summary = fst.build_k2_solve_band(rows_by_track=rows_by_track, k1_metrics_lookup=k1_metrics_lookup, k1_ellipses_lookup=k1_ellipses_lookup, threshold=int(args.threshold), radius=int(args.k2_band_radius), error_percentile=float(args.k2_band_error_percentile), instability_percentile=float(args.k2_band_instability_percentile), instability_floor=float(args.k2_band_instability_floor))
    else:
        selected_keys, k2_band_summary = infer_build_k2_solve_band_edge_aware(rows_by_track=rows_by_track, k1_metrics_lookup=routing_metrics_lookup, k1_ellipses_lookup=k1_ellipses_lookup, threshold_default=threshold_default, threshold_edge=effective_threshold_edge, edge_keys=edge_keys, radius=int(args.k2_band_radius), error_percentile=float(args.k2_band_error_percentile), instability_percentile=float(args.k2_band_instability_percentile), instability_floor=float(args.k2_band_instability_floor))
    if isinstance(k2_band_summary, dict):
        k2_band_summary.setdefault('routing_mode', str(args.routing_mode))
        k2_band_summary['k1_cost_routing'] = str(args.k1_cost_routing)
        k2_band_summary['routing_cost_field'] = routing_cost_field
        k2_band_summary['threshold_raw'] = float(args.threshold)
        k2_band_summary['threshold_edge_raw'] = float(args.threshold if int(args.threshold_edge) < 0 else args.threshold_edge)
        k2_band_summary['threshold_norm'] = float(args.threshold_norm)
        k2_band_summary['threshold_edge_norm'] = float(args.threshold_norm if float(args.threshold_edge_norm) < 0.0 else args.threshold_edge_norm)
    if str(args.routing_mode) == 'track_dp' and int(args.k2_dp_merge_short_k2_max_len) > 0:
        selected_keys, removed_inner_k2_summary = infer_cleanup_selected_k2_inner_islands(rows_by_track=rows_by_track, selected_keys=selected_keys, k1_metrics_lookup=routing_metrics_lookup, max_len=int(args.k2_dp_merge_short_k2_max_len), keep_cost=dp_merge_short_k2_keep_cost)
        if isinstance(k2_band_summary, dict):
            k2_band_summary['selected_count'] = len(selected_keys)
            k2_band_summary['post_removed_inner_k2_runs'] = int(removed_inner_k2_summary['removed_runs'])
            k2_band_summary['post_removed_inner_k2_rows'] = int(removed_inner_k2_summary['removed_rows'])
            k2_band_summary['post_removed_exact_inner_k2_runs'] = int(removed_inner_k2_summary['removed_exact_runs'])
            k2_band_summary['post_removed_exact_inner_k2_rows'] = int(removed_inner_k2_summary['removed_exact_rows'])
            k2_band_summary['post_removed_track_order_k2_runs'] = int(removed_inner_k2_summary['removed_track_order_runs'])
            k2_band_summary['post_removed_track_order_k2_rows'] = int(removed_inner_k2_summary['removed_track_order_rows'])
            k2_band_summary['post_removed_exact_singleton_k2_runs'] = int(removed_inner_k2_summary['removed_exact_singleton_runs'])
            k2_band_summary['post_removed_exact_singleton_k2_rows'] = int(removed_inner_k2_summary['removed_exact_singleton_rows'])
    if str(args.routing_mode) == 'track_dp' and int(args.k2_dp_merge_short_k1_max_len) > 0:
        selected_keys, promoted_short_k1_summary = infer_promote_short_k1_runs_to_k2(rows_by_track=rows_by_track, selected_keys=selected_keys, max_len=int(args.k2_dp_merge_short_k1_max_len))
        if isinstance(k2_band_summary, dict):
            k2_band_summary['selected_count'] = len(selected_keys)
            k2_band_summary['post_promoted_short_k1_runs'] = int(promoted_short_k1_summary['promoted_runs'])
            k2_band_summary['post_promoted_short_k1_rows'] = int(promoted_short_k1_summary['promoted_rows'])
    if str(args.routing_mode) == 'track_dp' and dp_force_k2_cost >= 0:
        selected_keys, forced_high_cost_summary = infer_force_select_high_cost_rows(rows_by_track=rows_by_track, selected_keys=selected_keys, k1_metrics_lookup=routing_metrics_lookup, min_cost=dp_force_k2_cost)
        if isinstance(k2_band_summary, dict):
            k2_band_summary['selected_count'] = len(selected_keys)
            k2_band_summary['post_forced_high_cost_k2_runs'] = int(forced_high_cost_summary['forced_runs'])
            k2_band_summary['post_forced_high_cost_k2_rows'] = int(forced_high_cost_summary['forced_rows'])
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
    k2_solved_rows, k2_solved_metrics, k2_solve_sec, model_info = infer_infer_k2_v5(selected_rows=selected_rows, payloads=selected_payloads, gt_polys=selected_gt_polys, run_dir=args.k2_run_dir, device_name=str(args.k2_device), batch_size=int(args.k2_batch_size), prep_workers=int(args.k2_prep_workers), precision=str(args.k2_precision), forward_mode=str(args.k2_forward_mode), profile_stages=bool(args.k2_profile_stages), cudnn_benchmark=str(args.k2_cudnn_benchmark), tf32=str(args.k2_tf32))
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
    aggregate = infer_evaluate_mixed_metric_rows(metric_rows, total_gt_rows=len(source_rows), total_sub_rows=len(submission_rows))
    eval_sec = time.perf_counter() - eval_start
    write_sqlite_start = time.perf_counter()
    fst.write_sqlite(submission_rows, args.output_dir / 'k1_exact_k2_v5_predictions.sqlite', reference_sqlite=input_sqlite_path)
    write_sqlite_sec = time.perf_counter() - write_sqlite_start
    write_metrics_start = time.perf_counter()
    infer_write_metrics_csv(k1_metric_rows, args.output_dir / 'k1_candidate_metrics.csv')
    infer_write_metrics_csv(metric_rows, args.output_dir / 'k1_exact_k2_v5_metrics.csv')
    write_metrics_sec = time.perf_counter() - write_metrics_start
    total_sec = time.perf_counter() - t0
    total_unique_frames = len({int(frame) for frame, _, _ in source_rows})
    summary = {'input_sqlite': str(input_sqlite_path), 'input_jsonl': str(args.input_jsonl) if args.input_jsonl is not None else None, 'output_dir': str(args.output_dir), 'config': {'raw_cut_detect': bool(args.raw_cut_detect), 'raw_cut_method': str(getattr(args, 'raw_cut_method', infer_RAW_CUT_METHOD_DEFAULT)), 'raw_remove_short_tracks_max_frames': int(args.raw_remove_short_tracks_max_frames), 'raw_det_score_min': float(args.raw_det_score_min), 'k1_recall_target': float(args.k1_recall_target), 'k1_exact_refine_rounds': int(args.k1_exact_refine_rounds), 'k1_workers_argument': int(args.k1_workers), 'k2_run_dir': str(args.k2_run_dir), 'k2_device': str(args.k2_device), 'k2_batch_size': int(args.k2_batch_size), 'k2_prep_workers': int(args.k2_prep_workers), 'k2_precision': str(args.k2_precision), 'k2_forward_mode': str(args.k2_forward_mode), 'k2_profile_stages': bool(args.k2_profile_stages), 'k2_cudnn_benchmark': str(args.k2_cudnn_benchmark), 'k2_tf32': str(args.k2_tf32), 'routing_mode': str(args.routing_mode), 'threshold': int(args.threshold), 'threshold_edge': effective_threshold_edge, 'k2_hyst_enter': int(args.k2_hyst_enter), 'k2_hyst_enter_edge': int(args.k2_hyst_enter if int(args.k2_hyst_enter_edge) < 0 else args.k2_hyst_enter_edge), 'k2_hyst_exit': int(args.k2_hyst_exit), 'k2_hyst_exit_edge': int(args.k2_hyst_exit if int(args.k2_hyst_exit_edge) < 0 else args.k2_hyst_exit_edge), 'k2_hyst_confirm_frames': int(args.k2_hyst_confirm_frames), 'k2_hyst_reset_gap': int(args.k2_hyst_reset_gap), 'k2_dp_error_weight': float(args.k2_dp_error_weight), 'k2_dp_instability_weight': float(args.k2_dp_instability_weight), 'k2_dp_edge_bonus': float(args.k2_dp_edge_bonus), 'k2_dp_k2_bias': float(args.k2_dp_k2_bias), 'k2_dp_switch_12': float(args.k2_dp_switch_12), 'k2_dp_switch_21': float(args.k2_dp_switch_21), 'k2_dp_short_k1_gamma': float(args.k2_dp_short_k1_gamma), 'k2_dp_short_k2_gamma': float(args.k2_dp_short_k2_gamma), 'k2_dp_short_k1_tau': float(args.k2_dp_short_k1_tau), 'k2_dp_short_k2_tau': float(args.k2_dp_short_k2_tau), 'k2_dp_short_len_cap': int(args.k2_dp_short_len_cap), 'k2_dp_reset_gap': int(args.k2_dp_reset_gap), 'k2_dp_merge_short_k1_max_len': int(args.k2_dp_merge_short_k1_max_len), 'k2_dp_merge_short_k2_max_len': int(args.k2_dp_merge_short_k2_max_len), 'k2_dp_merge_short_k2_keep_cost': int(args.k2_dp_merge_short_k2_keep_cost), 'k2_dp_force_k2_cost': int(args.k2_dp_force_k2_cost), 'k2_band_radius': int(args.k2_band_radius), 'k2_band_error_percentile': float(args.k2_band_error_percentile), 'k2_band_instability_percentile': float(args.k2_band_instability_percentile), 'k2_band_instability_floor': float(args.k2_band_instability_floor), 'max_rows': int(args.max_rows), 'max_tracks': int(args.max_tracks)}, 'counts': {'total_rows': len(source_rows), 'total_unique_frames': total_unique_frames, 'edge_rows': len(edge_rows), 'nonedge_rows': len(nonedge_rows), 'k2_selected_rows': len(selected_rows), 'k1_final_rows': len(source_rows) - len(selected_rows)}, 'workers': {'edge': workers_edge, 'nonedge': workers_nonedge}, 'timing_sec': {'load_rows': load_rows_sec, 'parse_polygons_all': parse_polygons_sec, 'prepare_payloads_all': prepare_payloads_sec, 'classify_edge_all': classify_edge_sec, 'preprocess_loop_total': preprocess_loop_sec, 'split_rows': split_rows_sec, 'edge_solve': edge_solve_sec, 'nonedge_solve': nonedge_solve_sec, 'merge_k1': merge_k1_sec, 'k2_band_select': k2_band_sec, 'k2_v5_solve': k2_solve_sec, 'merge_final': merge_final_sec, 'evaluate_submission': eval_sec, 'write_sqlite': write_sqlite_sec, 'write_metrics_csv': write_metrics_sec, 'end_to_end_total': total_sec}, 'throughput': {'end_to_end_rows_per_sec': len(source_rows) / max(total_sec, 1e-09), 'end_to_end_unique_frames_per_sec': total_unique_frames / max(total_sec, 1e-09), 'k2_v5_rows_per_sec': len(selected_rows) / max(k2_solve_sec, 1e-09) if selected_rows else 0.0}, 'raw_preprocess': raw_preprocess_summary, 'metrics': aggregate, 'k2_band_summary': k2_band_summary, 'k2_v5_model': model_info}
    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))



# ==============================================================================
# Inlined from: keyframe_opt/optimize_keyframes_standalone.py
# ==============================================================================

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

def kfbase_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Standalone keyframe optimizer for routed K1/K2 ellipse sequences.')
    parser.add_argument('--input-metrics-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-k1-ratio', type=float, default=0.1)
    parser.add_argument('--target-k2-ratio', type=float, default=0.16)
    parser.add_argument('--solver', choices=['dp', 'dp_rewarded', 'dp_candidates', 'uniform_refine', 'greedy_split', 'bottom_up_merge', 'best_first_split', 'rdp_quantile', 'trend_knots', 'event_triggered'], default='dp')
    parser.add_argument('--lambda-k1', type=float, default=-1.0)
    parser.add_argument('--lambda-k2', type=float, default=-1.0)
    parser.add_argument('--lambda-search-iters', type=int, default=16)
    parser.add_argument('--smooth-alpha', type=float, default=6.0)
    parser.add_argument('--confidence-floor', type=float, default=0.18)
    parser.add_argument('--error-scale', type=float, default=4000.0)
    parser.add_argument('--min-gap', type=int, default=2)
    parser.add_argument('--max-gap', type=int, default=16)
    parser.add_argument('--local-search-radius', type=int, default=2)
    parser.add_argument('--value-refine', choices=['none', 'global_ls', 'segment_ls', 'residual_nudge'], default='none')
    parser.add_argument('--value-refine-ridge', type=float, default=0.001)
    parser.add_argument('--value-refine-damping', type=float, default=1.0)
    parser.add_argument('--min-segment-length', type=int, default=3)
    parser.add_argument('--theta-weight-floor', type=float, default=0.2)
    parser.add_argument('--weight-error-gain', type=float, default=1.0)
    parser.add_argument('--weight-curvature-gain', type=float, default=1.0)
    parser.add_argument('--importance-cap', type=float, default=4.0)
    parser.add_argument('--reward-error-gain', type=float, default=0.75)
    parser.add_argument('--reward-curvature-gain', type=float, default=1.25)
    parser.add_argument('--reward-cap', type=float, default=1.5)
    parser.add_argument('--auto-break-threshold', type=float, default=-1.0)
    parser.add_argument('--auto-break-min-length', type=int, default=8)
    parser.add_argument('--auto-break-min-separation', type=int, default=6)
    parser.add_argument('--candidate-multiplier', type=float, default=4.0)
    parser.add_argument('--candidate-min-separation', type=int, default=2)
    parser.add_argument('--candidate-uniform-support', type=int, default=6)
    parser.add_argument('--rdp-quantile', type=float, default=0.9)
    parser.add_argument('--event-quantile', type=float, default=0.9)
    parser.add_argument('--event-search-iters', type=int, default=16)
    parser.add_argument('--keyframe-value-source', choices=['smoothed', 'raw', 'confidence_blend'], default='smoothed')
    parser.add_argument('--k2-slot-center-weight', type=float, default=1.0)
    parser.add_argument('--k2-slot-size-weight', type=float, default=0.65)
    parser.add_argument('--k2-slot-angle-weight', type=float, default=0.2)
    parser.add_argument('--max-streams', type=int, default=-1)
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
    prefix_cache: 'PrefixCostCache | None' = None
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

def kfbase_compute_confidence(row: kfbase_MetricRow, floor: float, error_scale: float) -> float:
    quality = max(row.iou, 0.0001) ** 0.55 * max(row.recall, 0.0001) ** 0.3 * max(row.precision, 0.0001) ** 0.15 * math.exp(-max(row.weighted_error, 0.0) / max(error_scale, 1e-06))
    return float(min(1.0, max(floor, quality)))

def kfbase_load_metric_rows(path: Path, confidence_floor: float, error_scale: float) -> list[kfbase_MetricRow]:
    rows: list[kfbase_MetricRow] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ellipse_params = json.loads(row['ellipse_params'])
            rows.append(kfbase_MetricRow(frame=int(row['frame']), track_id=str(row['track_id']), mode=str(row['mode']).upper(), weighted_error=float(row['weighted_error']), recall=float(row['recall']), precision=float(row['precision']), iou=float(row['iou']), ellipse_params=[kfbase_canonicalize_ellipse(x) for x in ellipse_params]))
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
            split = prev is None or row.mode != prev.mode or len(row.ellipse_params) != len(prev.ellipse_params) or (row.frame != prev.frame + 1)
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

def kfbase_ellipse_pair_cost(left: list[float], right: list[float], center_weight: float, size_weight: float, angle_weight: float) -> float:
    lc = np.asarray(left[:2], dtype=np.float64)
    rc = np.asarray(right[:2], dtype=np.float64)
    la, lb = (float(left[2]), float(left[3]))
    ra, rb = (float(right[2]), float(right[3]))
    center_scale = max(math.sqrt(max(la * lb, 1e-06)), math.sqrt(max(ra * rb, 1e-06)), 1.0)
    center_term = float(np.linalg.norm(lc - rc) / center_scale)
    size_term = abs(math.log(max(la, 1e-06) / max(ra, 1e-06))) + abs(math.log(max(lb, 1e-06) / max(rb, 1e-06)))
    ecc_left = 1.0 - min(la, lb) / max(la, lb)
    ecc_right = 1.0 - min(ra, rb) / max(ra, rb)
    ecc = max(0.0, 0.5 * (ecc_left + ecc_right))
    angle_term = kfbase_circular_angle_distance_deg(float(left[4]), float(right[4])) / 45.0 * max(0.1, ecc)
    return center_weight * center_term + size_weight * size_term + angle_weight * angle_term

def kfbase_stabilize_k2_slots(rows: list[kfbase_MetricRow], center_weight: float, size_weight: float, angle_weight: float) -> list[list[list[float]]]:
    stabilized: list[list[list[float]]] = []
    prev: list[list[float]] | None = None
    for row in rows:
        current = [list(ellipse) for ellipse in row.ellipse_params]
        if prev is not None and len(prev) == 2 and (len(current) == 2):
            keep_cost = kfbase_ellipse_pair_cost(prev[0], current[0], center_weight, size_weight, angle_weight) + kfbase_ellipse_pair_cost(prev[1], current[1], center_weight, size_weight, angle_weight)
            swap_cost = kfbase_ellipse_pair_cost(prev[0], current[1], center_weight, size_weight, angle_weight) + kfbase_ellipse_pair_cost(prev[1], current[0], center_weight, size_weight, angle_weight)
            if swap_cost < keep_cost:
                current = [current[1], current[0]]
        stabilized.append(current)
        prev = current
    return stabilized

def kfbase_build_stream_segments(args: argparse.Namespace, rows: list[kfbase_MetricRow]) -> list[kfbase_StreamSegment]:
    runs = kfbase_split_runs(rows)
    streams: list[kfbase_StreamSegment] = []
    for run_id, run_rows in enumerate(runs):
        mode = run_rows[0].mode
        if mode == 'K2':
            stabilized = kfbase_stabilize_k2_slots(run_rows, center_weight=float(args.k2_slot_center_weight), size_weight=float(args.k2_slot_size_weight), angle_weight=float(args.k2_slot_angle_weight))
        else:
            stabilized = [[list(row.ellipse_params[0])] for row in run_rows]
        slot_count = len(stabilized[0])
        for slot_id in range(slot_count):
            states = np.asarray([frame_slots[slot_id] for frame_slots in stabilized], dtype=np.float64)
            states[:, 4] = kfbase_unwrap_angles_deg(states[:, 4])
            confidence = np.asarray([kfbase_compute_confidence(row, floor=float(args.confidence_floor), error_scale=float(args.error_scale)) for row in run_rows], dtype=np.float64)
            weighted_error = np.asarray([row.weighted_error for row in run_rows], dtype=np.float64)
            streams.append(kfbase_StreamSegment(stream_id=f'{run_rows[0].track_id}:{mode}:run{run_id}:slot{slot_id}', track_id=run_rows[0].track_id, mode=mode, run_id=run_id, slot_id=slot_id, frame_numbers=np.asarray([row.frame for row in run_rows], dtype=np.int32), raw_states=states, confidence=confidence, weighted_error=weighted_error))
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

def kfbase_state_to_q(states: np.ndarray, theta_weight_floor: float) -> tuple[np.ndarray, float, float]:
    scale = float(np.median(np.sqrt(np.maximum(states[:, 2] * states[:, 3], 1e-06))))
    scale = max(scale, 1.0)
    eccentricity = 1.0 - np.minimum(states[:, 2], states[:, 3]) / np.maximum(states[:, 2], states[:, 3])
    theta_scale = max(float(theta_weight_floor), float(np.median(eccentricity)) if eccentricity.size else 1.0)
    q = np.column_stack([states[:, 0] / scale, states[:, 1] / scale, np.log(np.maximum(states[:, 2], 1e-06)), np.log(np.maximum(states[:, 3], 1e-06)), np.deg2rad(states[:, 4]) * theta_scale]).astype(np.float64)
    return (q, scale, theta_scale)

def kfbase_q_to_state(q: np.ndarray, scale: float, theta_scale: float) -> np.ndarray:
    state = np.zeros((q.shape[0], 5), dtype=np.float64)
    state[:, 0] = q[:, 0] * scale
    state[:, 1] = q[:, 1] * scale
    state[:, 2] = np.exp(q[:, 2])
    state[:, 3] = np.exp(q[:, 3])
    state[:, 4] = np.rad2deg(q[:, 4] / max(theta_scale, 1e-06))
    for idx in range(len(state)):
        state[idx] = np.asarray(kfbase_canonicalize_ellipse(state[idx]), dtype=np.float64)
    return state

def kfbase_smooth_stream_segment(stream: kfbase_StreamSegment, args: argparse.Namespace) -> None:
    q, scale, theta_scale = kfbase_state_to_q(stream.raw_states, theta_weight_floor=float(args.theta_weight_floor))
    stream.raw_q = q
    length = len(q)
    if length < int(args.min_segment_length):
        stream.smoothed_q = q
        stream.transform_scale = scale
        stream.theta_scale = theta_scale
        return
    d2 = kfbase_build_second_difference_matrix(length)
    penalty = d2.T @ d2
    weights = np.maximum(stream.confidence, float(args.confidence_floor))
    smoothed = np.zeros_like(q)
    for dim in range(q.shape[1]):
        system = np.diag(weights) + float(args.smooth_alpha) * penalty
        rhs = weights * q[:, dim]
        smoothed[:, dim] = np.linalg.solve(system, rhs)
    stream.smoothed_q = smoothed
    stream.transform_scale = scale
    stream.theta_scale = theta_scale

def kfbase_choose_anchor_q(stream: kfbase_StreamSegment, source: str) -> np.ndarray:
    assert stream.raw_q is not None
    assert stream.smoothed_q is not None
    if source == 'raw':
        return stream.raw_q
    if source == 'confidence_blend':
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

def kfbase_derive_stream_importance(stream: kfbase_StreamSegment, args: argparse.Namespace) -> None:
    assert stream.raw_q is not None
    assert stream.smoothed_q is not None
    error_norm = kfbase_robust_normalize(np.maximum(stream.weighted_error, 0.0))
    curvature_norm = kfbase_robust_normalize(kfbase_compute_curvature_signal(stream.smoothed_q))
    importance = stream.confidence * (1.0 + float(args.weight_error_gain) * error_norm + float(args.weight_curvature_gain) * curvature_norm)
    stream.importance = np.clip(importance, 1e-06, float(args.importance_cap)).astype(np.float64)
    reward = float(args.reward_error_gain) * error_norm + float(args.reward_curvature_gain) * curvature_norm
    stream.keyframe_reward = np.clip(reward, 0.0, float(args.reward_cap)).astype(np.float64)
    stream.break_signal = (curvature_norm + 0.5 * error_norm).astype(np.float64)

def kfbase_split_stream_on_breaks(stream: kfbase_StreamSegment, args: argparse.Namespace) -> list[kfbase_StreamSegment]:
    threshold = float(args.auto_break_threshold)
    if threshold < 0.0 or stream.break_signal is None or len(stream.frame_numbers) < int(args.auto_break_min_length) * 2:
        return [stream]
    min_len = int(args.auto_break_min_length)
    min_sep = int(args.auto_break_min_separation)
    candidates: list[int] = []
    last_break = -10 ** 9
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
        out.append(kfbase_StreamSegment(stream_id=f'{stream.stream_id}:break{seg_idx}', track_id=stream.track_id, mode=stream.mode, run_id=stream.run_id, slot_id=stream.slot_id, frame_numbers=stream.frame_numbers[lo:hi].copy(), raw_states=stream.raw_states[lo:hi].copy(), confidence=stream.confidence[lo:hi].copy(), weighted_error=stream.weighted_error[lo:hi].copy(), raw_q=None if stream.raw_q is None else stream.raw_q[lo:hi].copy(), smoothed_q=None if stream.smoothed_q is None else stream.smoothed_q[lo:hi].copy(), importance=None if stream.importance is None else stream.importance[lo:hi].copy(), keyframe_reward=None if stream.keyframe_reward is None else stream.keyframe_reward[lo:hi].copy(), break_signal=None if stream.break_signal is None else stream.break_signal[lo:hi].copy(), transform_scale=stream.transform_scale, theta_scale=stream.theta_scale))
    return out if out else [stream]

@dataclass
class kfbase_PrefixCostCache:
    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    g: np.ndarray

def kfbase_build_prefix_cache(q: np.ndarray, weights: np.ndarray) -> kfbase_PrefixCostCache:
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
    s2[1:] = np.cumsum(weights * t ** 2)
    v0[1:] = np.cumsum(weights[:, None] * q, axis=0)
    v1[1:] = np.cumsum((weights * t)[:, None] * q, axis=0)
    g[1:] = np.cumsum(weights * np.sum(q * q, axis=1))
    return kfbase_PrefixCostCache(s0=s0, s1=s1, s2=s2, v0=v0, v1=v1, g=g)

def kfbase_get_stream_prefix_cache(stream: kfbase_StreamSegment) -> kfbase_PrefixCostCache:
    cache = stream.prefix_cache
    if cache is not None:
        return cache
    assert stream.smoothed_q is not None
    weights = stream.importance if stream.importance is not None else stream.confidence
    cache = kfbase_build_prefix_cache(stream.smoothed_q, weights)
    stream.prefix_cache = cache
    return cache

def kfbase_get_stream_interval_costs(stream: kfbase_StreamSegment, max_gap: int) -> np.ndarray:
    cached = stream.interval_costs
    if cached is not None and int(stream.interval_costs_max_gap) == int(max_gap):
        return cached
    assert stream.smoothed_q is not None
    q = stream.smoothed_q
    cache = kfbase_get_stream_prefix_cache(stream)
    length = len(q)
    costs = np.full((length, int(max_gap)), np.inf, dtype=np.float64)
    for end in range(1, length):
        start_low = max(0, end - int(max_gap))
        for start in range(start_low, end):
            gap = end - start
            costs[end, gap - 1] = kfbase_interval_surrogate_cost(cache, q, start, end)
    stream.interval_costs = costs
    stream.interval_costs_max_gap = int(max_gap)
    return costs

def kfbase_interval_surrogate_cost(cache: kfbase_PrefixCostCache, q: np.ndarray, start: int, end: int) -> float:
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
    cost = g - 2.0 * float(np.dot(qi, a_vec)) - 2.0 * float(np.dot(qj, b_vec)) + alpha * float(np.dot(qi, qi)) + beta * float(np.dot(qj, qj)) + 2.0 * gamma * float(np.dot(qi, qj))
    return float(max(cost, 0.0))

def kfbase_decode_keyframes_dp(q: np.ndarray, weights: np.ndarray, keyframe_penalty: float, min_gap: int, max_gap: int, rewards: np.ndarray | None=None, prefix_cache: kfbase_PrefixCostCache | None=None, interval_costs: np.ndarray | None=None) -> tuple[list[int], float]:
    length = len(q)
    if length <= 2:
        return (list(range(length)), 0.0)
    cache = prefix_cache if prefix_cache is not None else kfbase_build_prefix_cache(q, weights)
    dp = np.full(length, np.inf, dtype=np.float64)
    back = np.full(length, -1, dtype=np.int32)
    dp[0] = -float(keyframe_penalty)
    for end in range(1, length):
        start_low = max(0, end - max_gap)
        for start in range(start_low, end):
            segment_len = end - start
            if start != 0 and segment_len < min_gap:
                continue
            if end != length - 1 and segment_len < min_gap:
                continue
            if interval_costs is not None and segment_len <= interval_costs.shape[1]:
                interval_cost = float(interval_costs[end, segment_len - 1])
            else:
                interval_cost = kfbase_interval_surrogate_cost(cache, q, start, end)
            reward_term = 0.0 if rewards is None else float(rewards[end])
            candidate_cost = dp[start] + interval_cost + float(keyframe_penalty) - reward_term
            if candidate_cost < dp[end]:
                dp[end] = candidate_cost
                back[end] = start
    if not np.isfinite(dp[-1]):
        return (list(range(length)), float('inf'))
    chosen: list[int] = []
    cursor = length - 1
    while cursor >= 0:
        chosen.append(int(cursor))
        if cursor == 0:
            break
        cursor = int(back[cursor])
        if cursor < 0:
            return (list(range(length)), float('inf'))
    chosen.reverse()
    return (chosen, float(dp[-1]))

def kfbase_refine_keyframes_locally(q: np.ndarray, weights: np.ndarray, chosen: list[int], min_gap: int, max_gap: int, radius: int, prefix_cache: kfbase_PrefixCostCache | None=None, interval_costs: np.ndarray | None=None) -> list[int]:
    if len(chosen) <= 2 or radius <= 0:
        return chosen
    cache = prefix_cache if prefix_cache is not None else kfbase_build_prefix_cache(q, weights)
    refined = list(chosen)
    for key_idx in range(1, len(refined) - 1):
        left = refined[key_idx - 1]
        current = refined[key_idx]
        right = refined[key_idx + 1]
        best = current
        left_gap = current - left
        right_gap = right - current
        best_cost = (float(interval_costs[current, left_gap - 1]) if interval_costs is not None and left_gap <= interval_costs.shape[1] else kfbase_interval_surrogate_cost(cache, q, left, current)) + (float(interval_costs[right, right_gap - 1]) if interval_costs is not None and right_gap <= interval_costs.shape[1] else kfbase_interval_surrogate_cost(cache, q, current, right))
        low = max(left + min_gap, current - radius)
        high = min(right - min_gap, current + radius)
        for candidate in range(low, high + 1):
            if candidate == current:
                continue
            if candidate - left > max_gap or right - candidate > max_gap:
                continue
            left_gap = candidate - left
            right_gap = right - candidate
            cost = (float(interval_costs[candidate, left_gap - 1]) if interval_costs is not None and left_gap <= interval_costs.shape[1] else kfbase_interval_surrogate_cost(cache, q, left, candidate)) + (float(interval_costs[right, right_gap - 1]) if interval_costs is not None and right_gap <= interval_costs.shape[1] else kfbase_interval_surrogate_cost(cache, q, candidate, right))
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

def kfbase_refine_keyframe_values_global_ls(target_q: np.ndarray, base_key_q: np.ndarray, keyframes: list[int], weights: np.ndarray, ridge: float) -> np.ndarray:
    key_count = len(keyframes)
    if key_count <= 1:
        return base_key_q
    length, dims = target_q.shape
    a = np.zeros((length, key_count), dtype=np.float64)
    for idx in range(key_count - 1):
        left = keyframes[idx]
        right = keyframes[idx + 1]
        if right == left:
            a[left, idx] = 1.0
            continue
        span = float(right - left)
        for pos in range(left, right + 1):
            alpha = (pos - left) / span
            a[pos, idx] = 1.0 - alpha
            a[pos, idx + 1] = alpha
    sqrt_w = np.sqrt(np.maximum(weights, 1e-08))
    aw = a * sqrt_w[:, None]
    gram = aw.T @ aw + float(ridge) * np.eye(key_count, dtype=np.float64)
    rhs_base = float(ridge) * base_key_q
    solved = np.zeros_like(base_key_q)
    for dim in range(dims):
        bw = target_q[:, dim] * sqrt_w
        rhs = aw.T @ bw + rhs_base[:, dim]
        solved[:, dim] = np.linalg.solve(gram, rhs)
    return solved

def kfbase_refine_keyframe_values_segment_ls(target_q: np.ndarray, base_key_q: np.ndarray, keyframes: list[int], weights: np.ndarray, ridge: float) -> np.ndarray:
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

def kfbase_refine_keyframe_values_residual_nudge(target_q: np.ndarray, base_key_q: np.ndarray, keyframes: list[int], weights: np.ndarray, damping: float) -> np.ndarray:
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

def kfbase_interpolate_from_key_values(key_q: np.ndarray, keyframes: list[int], length: int) -> np.ndarray:
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

def kfbase_choose_uniform_positions(length: int, key_count: int, min_gap: int, max_gap: int) -> list[int]:
    if length <= 1:
        return [0]
    key_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(key_count)))
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

def kfbase_allocate_target_keyframes(streams: list[kfbase_StreamSegment], target_ratio: float, min_gap: int, max_gap: int) -> dict[str, int]:
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
        ideals[stream.stream_id] = float(np.clip(length * target_ratio, min_count, max_count))
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
        for _frac, stream_id in sorted(fractional, key=lambda item: item[0], reverse=True):
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

def kfbase_best_split_for_segment(cache: kfbase_PrefixCostCache, q: np.ndarray, left: int, right: int, min_gap: int) -> tuple[int | None, float]:
    low = left + min_gap
    high = right - min_gap
    if high < low:
        return (None, float('-inf'))
    base_cost = kfbase_interval_surrogate_cost(cache, q, left, right)
    best_idx: int | None = None
    best_gain = float('-inf')
    for candidate in range(low, high + 1):
        gain = base_cost - (kfbase_interval_surrogate_cost(cache, q, left, candidate) + kfbase_interval_surrogate_cost(cache, q, candidate, right))
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

def kfbase_best_residual_split_for_segment(q: np.ndarray, weights: np.ndarray, left: int, right: int, min_gap: int, quantile: float) -> tuple[int | None, float]:
    low = left + min_gap
    high = right - min_gap
    if high < low:
        return (None, float('-inf'))
    residuals = kfbase_residuals_for_segment(q, left, right)
    if residuals.size == 0:
        return (None, float('-inf'))
    seg_weights = np.maximum(weights[left:right + 1], 1e-08)
    weighted_residuals = residuals * np.sqrt(seg_weights)
    score = float(np.quantile(weighted_residuals, float(np.clip(quantile, 0.5, 0.99))))
    candidate_offset = int(np.argmax(weighted_residuals))
    candidate = left + candidate_offset
    candidate = min(max(candidate, low), high)
    return (candidate, score)

def kfbase_select_keyframes_uniform(q: np.ndarray, target_count: int, min_gap: int, max_gap: int) -> list[int]:
    return kfbase_choose_uniform_positions(len(q), target_count, min_gap=min_gap, max_gap=max_gap)

def kfbase_select_keyframes_best_first_split(q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    chosen = kfbase_choose_uniform_positions(length, kfbase_min_required_keyframes(length, max_gap), min_gap=min_gap, max_gap=max_gap)
    cache = kfbase_build_prefix_cache(q, weights)
    while len(chosen) < target_count:
        best_gain = float('-inf')
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, gain = kfbase_best_split_for_segment(cache, q, left, right, min_gap=min_gap)
            if candidate is not None and gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen

def kfbase_select_keyframes_rdp_quantile(q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int, quantile: float) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    chosen = kfbase_choose_uniform_positions(length, kfbase_min_required_keyframes(length, max_gap), min_gap=min_gap, max_gap=max_gap)
    while len(chosen) < target_count:
        best_score = float('-inf')
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, score = kfbase_best_residual_split_for_segment(q, weights, left, right, min_gap=min_gap, quantile=quantile)
            if candidate is not None and score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen

def kfbase_select_keyframes_greedy_split(q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int) -> list[int]:
    del weights
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    chosen = kfbase_choose_uniform_positions(length, kfbase_min_required_keyframes(length, max_gap), min_gap=min_gap, max_gap=max_gap)
    cache = kfbase_build_prefix_cache(q, np.ones(length, dtype=np.float64))
    while len(chosen) < target_count:
        best_gain = float('-inf')
        best_candidate: int | None = None
        for idx in range(len(chosen) - 1):
            left = chosen[idx]
            right = chosen[idx + 1]
            candidate, gain = kfbase_best_split_for_segment(cache, q, left, right, min_gap=min_gap)
            if candidate is not None and gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        if best_candidate is None or best_candidate in chosen:
            break
        chosen.append(best_candidate)
        chosen.sort()
    return chosen

def kfbase_ensure_target_count_with_peaks(q: np.ndarray, weights: np.ndarray, chosen: list[int], target_count: int, min_gap: int, max_gap: int) -> list[int]:
    length = len(q)
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    chosen = sorted(set((int(x) for x in chosen)))
    if len(chosen) > target_count:
        return kfbase_select_keyframes_bottom_up_merge(q, weights, chosen, target_count=target_count, min_gap=min_gap, max_gap=max_gap)
    if len(chosen) == target_count:
        return chosen
    signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(np.maximum(weights, 1e-08))
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
        for idx in kfbase_choose_uniform_positions(length, target_count, min_gap=min_gap, max_gap=max_gap):
            if idx not in chosen:
                chosen.append(int(idx))
                chosen.sort()
            if len(chosen) >= target_count:
                break
    return sorted(set(chosen))

def kfbase_select_keyframes_trend_knots(q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    chosen = kfbase_choose_uniform_positions(length, kfbase_min_required_keyframes(length, max_gap), min_gap=min_gap, max_gap=max_gap)
    signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(np.maximum(weights, 1e-08))
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
    return kfbase_ensure_target_count_with_peaks(q, weights, chosen, target_count, min_gap=min_gap, max_gap=max_gap)

def kfbase_event_trigger_positions_for_threshold(q: np.ndarray, weights: np.ndarray, threshold: float, min_gap: int, max_gap: int, quantile: float) -> list[int]:
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
            seg_weights = np.sqrt(np.maximum(weights[start:end + 1], 1e-08))
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

def kfbase_select_keyframes_event_triggered(q: np.ndarray, weights: np.ndarray, target_count: int, min_gap: int, max_gap: int, quantile: float, search_iters: int) -> list[int]:
    length = len(q)
    if length <= 1:
        return [0]
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), int(target_count)))
    base_signal = kfbase_robust_normalize(kfbase_compute_curvature_signal(q)) * np.sqrt(np.maximum(weights, 1e-08))
    low = 0.0
    high = float(max(np.max(base_signal), 1.0))
    best = kfbase_choose_uniform_positions(length, target_count, min_gap=min_gap, max_gap=max_gap)
    best_gap = abs(len(best) - target_count)
    for _ in range(max(int(search_iters), 1)):
        mid = 0.5 * (low + high)
        chosen = kfbase_event_trigger_positions_for_threshold(q, weights, threshold=mid, min_gap=min_gap, max_gap=max_gap, quantile=quantile)
        gap = abs(len(chosen) - target_count)
        if gap < best_gap:
            best = chosen
            best_gap = gap
        if len(chosen) > target_count:
            low = mid
        else:
            high = mid
    return kfbase_ensure_target_count_with_peaks(q, weights, best, target_count, min_gap=min_gap, max_gap=max_gap)

def kfbase_select_candidate_positions(stream: kfbase_StreamSegment, target_ratio: float, min_gap: int, max_gap: int, multiplier: float, min_separation: int, uniform_support: int) -> list[int]:
    length = len(stream.frame_numbers)
    if length <= 1:
        return [0]
    target_count = int(round(length * max(target_ratio, 0.0)))
    target_count = max(kfbase_min_required_keyframes(length, max_gap), min(kfbase_max_allowed_keyframes(length, min_gap), target_count))
    desired_candidates = max(target_count + 2, int(math.ceil(target_count * max(multiplier, 1.0))), int(math.ceil(length / max(uniform_support, 1))))
    positions = {0, length - 1}
    support_count = max(kfbase_min_required_keyframes(length, max(uniform_support, 1)), min(target_count * 2, desired_candidates))
    positions.update(kfbase_choose_uniform_positions(length, support_count, min_gap=1, max_gap=max(uniform_support, 1)))
    signal = stream.importance if stream.importance is not None else stream.confidence
    ranked = np.argsort(signal)[::-1].tolist()
    for idx in ranked:
        idx = int(idx)
        if idx <= 0 or idx >= length - 1:
            continue
        if any((abs(idx - existing) < max(min_separation, 1) for existing in positions)):
            continue
        positions.add(idx)
        if len(positions) >= desired_candidates:
            break
    return sorted(positions)

def kfbase_decode_keyframes_candidate_dp(q: np.ndarray, weights: np.ndarray, candidate_positions: list[int], keyframe_penalty: float, min_gap: int, max_gap: int, rewards: np.ndarray | None=None) -> tuple[list[int], float]:
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
            candidate_cost = dp[start_idx] + interval_cost + float(keyframe_penalty) - reward_term
            if candidate_cost < dp[end_idx]:
                dp[end_idx] = candidate_cost
                back[end_idx] = start_idx
    if not np.isfinite(dp[-1]):
        return ([0, len(q) - 1], float('inf'))
    chosen: list[int] = []
    cursor = n - 1
    while cursor >= 0:
        chosen.append(int(candidate_positions[cursor]))
        if cursor == 0:
            break
        cursor = int(back[cursor])
        if cursor < 0:
            return ([0, len(q) - 1], float('inf'))
    chosen.reverse()
    return (chosen, float(dp[-1]))

def kfbase_decode_keyframes_candidate_budget_dp(q: np.ndarray, weights: np.ndarray, candidate_positions: list[int], target_count: int, min_gap: int, max_gap: int, rewards: np.ndarray | None=None) -> tuple[list[int], float]:
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
        fallback = kfbase_choose_uniform_positions(len(q), target_count, min_gap=min_gap, max_gap=max_gap)
        return (fallback, float('inf'))
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
            fallback = kfbase_choose_uniform_positions(len(q), target_count, min_gap=min_gap, max_gap=max_gap)
            return (fallback, float('inf'))
    chosen.reverse()
    return (chosen, float(dp[target_count - 1, n - 1]))

def kfbase_select_keyframes_bottom_up_merge(q: np.ndarray, weights: np.ndarray, initial_positions: list[int], target_count: int, min_gap: int, max_gap: int) -> list[int]:
    del weights
    chosen = sorted(set((int(x) for x in initial_positions)))
    if len(chosen) <= target_count:
        return chosen
    cache = kfbase_build_prefix_cache(q, np.ones(len(q), dtype=np.float64))
    while len(chosen) > target_count:
        best_idx: int | None = None
        best_delta = float('inf')
        for idx in range(1, len(chosen) - 1):
            left = chosen[idx - 1]
            mid = chosen[idx]
            right = chosen[idx + 1]
            if right - left > max_gap:
                continue
            if mid - left < min_gap or right - mid < min_gap:
                continue
            delta = kfbase_interval_surrogate_cost(cache, q, left, right) - kfbase_interval_surrogate_cost(cache, q, left, mid) - kfbase_interval_surrogate_cost(cache, q, mid, right)
            if delta < best_delta:
                best_delta = float(delta)
                best_idx = idx
        if best_idx is None:
            break
        chosen.pop(best_idx)
    return chosen

def kfbase_compute_ratio_for_lambda(streams: list[kfbase_StreamSegment], penalty: float, min_gap: int, max_gap: int, use_rewards: bool) -> tuple[float, int, int]:
    total_keyframes = 0
    total_frames = 0
    for stream in streams:
        assert stream.smoothed_q is not None
        prefix_cache = kfbase_get_stream_prefix_cache(stream)
        interval_costs = kfbase_get_stream_interval_costs(stream, max_gap=max_gap)
        chosen, _ = kfbase_decode_keyframes_dp(stream.smoothed_q, stream.importance if stream.importance is not None else stream.confidence, penalty, min_gap=min_gap, max_gap=max_gap, rewards=stream.keyframe_reward if use_rewards else None, prefix_cache=prefix_cache, interval_costs=interval_costs)
        total_keyframes += len(chosen)
        total_frames += len(stream.frame_numbers)
    ratio = float(total_keyframes) / float(total_frames) if total_frames > 0 else 0.0
    return (ratio, total_keyframes, total_frames)

def kfbase_find_penalty_for_target_ratio(streams: list[kfbase_StreamSegment], target_ratio: float, fallback_penalty: float, min_gap: int, max_gap: int, search_iters: int, use_rewards: bool) -> tuple[float, dict[str, float]]:
    if target_ratio <= 0.0:
        ratio, keyframes, frames = kfbase_compute_ratio_for_lambda(streams, fallback_penalty, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards)
        return (fallback_penalty, {'achieved_ratio': ratio, 'keyframes': keyframes, 'frames': frames})
    low = 0.0
    low_ratio, low_keys, low_frames = kfbase_compute_ratio_for_lambda(streams, low, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards)
    if low_ratio <= target_ratio:
        return (low, {'achieved_ratio': low_ratio, 'keyframes': low_keys, 'frames': low_frames})
    high = max(fallback_penalty, 1.0)
    high_ratio, high_keys, high_frames = kfbase_compute_ratio_for_lambda(streams, high, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards)
    while high_ratio > target_ratio and high < 1000000.0:
        high *= 2.0
        high_ratio, high_keys, high_frames = kfbase_compute_ratio_for_lambda(streams, high, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards)
    best_penalty = high
    best_ratio = high_ratio
    best_keys = high_keys
    best_frames = high_frames
    for _ in range(search_iters):
        mid = 0.5 * (low + high)
        mid_ratio, mid_keys, mid_frames = kfbase_compute_ratio_for_lambda(streams, mid, min_gap=min_gap, max_gap=max_gap, use_rewards=use_rewards)
        best_penalty, best_ratio, best_keys, best_frames = (mid, mid_ratio, mid_keys, mid_frames)
        if mid_ratio > target_ratio:
            low = mid
        else:
            high = mid
    return (best_penalty, {'achieved_ratio': best_ratio, 'keyframes': best_keys, 'frames': best_frames})

def kfbase_optimize_streams(streams: list[kfbase_StreamSegment], mode_penalty: float, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    for stream in streams:
        assert stream.smoothed_q is not None
        anchor_q = kfbase_choose_anchor_q(stream, source=str(args.keyframe_value_source))
        fit_weights = stream.importance if stream.importance is not None else stream.confidence
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        prefix_cache = kfbase_get_stream_prefix_cache(stream)
        interval_costs = kfbase_get_stream_interval_costs(stream, max_gap=int(args.max_gap))
        if str(args.solver) == 'greedy_split':
            chosen = kfbase_select_keyframes_greedy_split(stream.smoothed_q, fit_weights, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
            objective = float('nan')
        elif str(args.solver) == 'best_first_split':
            chosen = kfbase_select_keyframes_best_first_split(stream.smoothed_q, fit_weights, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
            objective = float('nan')
        elif str(args.solver) == 'rdp_quantile':
            chosen = kfbase_select_keyframes_rdp_quantile(stream.smoothed_q, fit_weights, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap), quantile=float(args.rdp_quantile))
            objective = float('nan')
        elif str(args.solver) == 'trend_knots':
            chosen = kfbase_select_keyframes_trend_knots(stream.smoothed_q, fit_weights, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
            objective = float('nan')
        elif str(args.solver) == 'event_triggered':
            chosen = kfbase_select_keyframes_event_triggered(stream.smoothed_q, fit_weights, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap), quantile=float(args.event_quantile), search_iters=int(args.event_search_iters))
            objective = float('nan')
        elif str(args.solver) == 'bottom_up_merge':
            target_count = int(round(mode_penalty))
            candidate_positions = kfbase_select_candidate_positions(stream, target_ratio=float(target_count) / max(len(stream.frame_numbers), 1), min_gap=int(args.min_gap), max_gap=int(args.max_gap), multiplier=float(args.candidate_multiplier), min_separation=int(args.candidate_min_separation), uniform_support=int(args.candidate_uniform_support))
            chosen = kfbase_select_keyframes_bottom_up_merge(stream.smoothed_q, fit_weights, candidate_positions, target_count=target_count, min_gap=int(args.min_gap), max_gap=int(args.max_gap))
            objective = float('nan')
        elif str(args.solver) == 'dp_candidates':
            target_count = int(round(mode_penalty))
            candidate_positions = kfbase_select_candidate_positions(stream, target_ratio=float(target_count) / max(len(stream.frame_numbers), 1), min_gap=int(args.min_gap), max_gap=int(args.max_gap), multiplier=float(args.candidate_multiplier), min_separation=int(args.candidate_min_separation), uniform_support=int(args.candidate_uniform_support))
            chosen, objective = kfbase_decode_keyframes_candidate_budget_dp(stream.smoothed_q, fit_weights, candidate_positions, target_count=target_count, min_gap=int(args.min_gap), max_gap=int(args.max_gap), rewards=stream.keyframe_reward)
        elif str(args.solver) == 'uniform_refine':
            chosen = kfbase_select_keyframes_uniform(stream.smoothed_q, int(round(mode_penalty)), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
            objective = float('nan')
        else:
            chosen, objective = kfbase_decode_keyframes_dp(stream.smoothed_q, fit_weights, mode_penalty, min_gap=int(args.min_gap), max_gap=int(args.max_gap), rewards=stream.keyframe_reward if str(args.solver) == 'dp_rewarded' else None, prefix_cache=prefix_cache, interval_costs=interval_costs)
        chosen = kfbase_refine_keyframes_locally(stream.smoothed_q, fit_weights, chosen, min_gap=int(args.min_gap), max_gap=int(args.max_gap), radius=int(args.local_search_radius), prefix_cache=prefix_cache, interval_costs=interval_costs)
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
        if len(chosen) >= 2 and str(args.value_refine) != 'none':
            if str(args.value_refine) == 'global_ls':
                key_q = kfbase_refine_keyframe_values_global_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'segment_ls':
                key_q = kfbase_refine_keyframe_values_segment_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'residual_nudge':
                key_q = kfbase_refine_keyframe_values_residual_nudge(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, damping=float(args.value_refine_damping))
            interp_q = kfbase_interpolate_from_key_values(key_q, chosen, len(stream.frame_numbers))
        else:
            interp_q = kfbase_interpolate_dense_q(anchor_q, chosen)
        dense_state = kfbase_q_to_state(interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        raw_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        weighted_rmse = float(math.sqrt(np.average(np.sum((interp_q - raw_q) ** 2, axis=1), weights=np.maximum(fit_weights, 1e-06))))
        segment_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame_count': int(len(stream.frame_numbers)), 'keyframe_count': int(len(chosen)), 'keyframe_ratio': float(len(chosen) / max(len(stream.frame_numbers), 1)), 'objective': float(objective), 'weighted_param_rmse': weighted_rmse})
        key_states = kfbase_q_to_state(key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        for key_idx, local_idx in enumerate(chosen):
            keyframe_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame': int(stream.frame_numbers[local_idx]), 'ellipse': key_states[key_idx].tolist()})
        for local_idx, frame in enumerate(stream.frame_numbers):
            dense_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame': int(frame), 'ellipse': dense_state[local_idx].tolist(), 'is_keyframe': int(local_idx in set(chosen))})
    return (keyframe_rows, dense_rows, segment_rows)

def kfbase_merge_dense_rows_to_union(dense_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for row in dense_rows:
        key = (str(row['track_id']), str(row['mode']), int(row['run_id']), int(row['frame']))
        entry = grouped.get(key)
        if entry is None:
            entry = {'track_id': str(row['track_id']), 'mode': str(row['mode']), 'run_id': int(row['run_id']), 'frame': int(row['frame']), 'ellipse_params': [], 'has_keyframe': 0}
            grouped[key] = entry
        entry['ellipse_params'].append((int(row['slot_id']), row['ellipse']))
        entry['has_keyframe'] = int(max(int(entry['has_keyframe']), int(row['is_keyframe'])))
    merged: list[dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda item: (int(item[0]), item[3], item[2])):
        entry = grouped[key]
        ellipses = [ellipse for _slot, ellipse in sorted(entry['ellipse_params'], key=lambda item: item[0])]
        merged.append({'track_id': entry['track_id'], 'mode': entry['mode'], 'run_id': entry['run_id'], 'frame': entry['frame'], 'ellipse_params': ellipses, 'has_keyframe': entry['has_keyframe']})
    return merged

def kfbase_write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def kfbase_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def kfbase_main() -> None:
    args = kfbase_parse_args()
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = kfbase_load_metric_rows(input_metrics, confidence_floor=float(args.confidence_floor), error_scale=float(args.error_scale))
    streams = kfbase_build_stream_segments(args, rows)
    if int(args.max_streams) > 0:
        streams = streams[:int(args.max_streams)]
    for stream in streams:
        kfbase_smooth_stream_segment(stream, args)
        kfbase_derive_stream_importance(stream, args)
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[kfbase_StreamSegment] = []
        for stream in streams:
            broken_streams.extend(kfbase_split_stream_on_breaks(stream, args))
        streams = broken_streams
    k1_streams = [stream for stream in streams if stream.mode == 'K1']
    k2_streams = [stream for stream in streams if stream.mode == 'K2']
    if str(args.solver) == 'dp':
        penalty_k1, ratio_summary_k1 = kfbase_find_penalty_for_target_ratio(k1_streams, target_ratio=float(args.target_k1_ratio), fallback_penalty=0.5 if float(args.lambda_k1) <= 0 else float(args.lambda_k1), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=False)
        penalty_k2, ratio_summary_k2 = kfbase_find_penalty_for_target_ratio(k2_streams, target_ratio=float(args.target_k2_ratio), fallback_penalty=0.35 if float(args.lambda_k2) <= 0 else float(args.lambda_k2), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=False)
    elif str(args.solver) == 'dp_rewarded':
        penalty_k1, ratio_summary_k1 = kfbase_find_penalty_for_target_ratio(k1_streams, target_ratio=float(args.target_k1_ratio), fallback_penalty=0.5 if float(args.lambda_k1) <= 0 else float(args.lambda_k1), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=True)
        penalty_k2, ratio_summary_k2 = kfbase_find_penalty_for_target_ratio(k2_streams, target_ratio=float(args.target_k2_ratio), fallback_penalty=0.35 if float(args.lambda_k2) <= 0 else float(args.lambda_k2), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=True)
    else:
        assigned_k1 = kfbase_allocate_target_keyframes(k1_streams, target_ratio=float(args.target_k1_ratio), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
        assigned_k2 = kfbase_allocate_target_keyframes(k2_streams, target_ratio=float(args.target_k2_ratio), min_gap=int(args.min_gap), max_gap=int(args.max_gap))
        penalty_k1 = float(sum(assigned_k1.values()))
        penalty_k2 = float(sum(assigned_k2.values()))
        ratio_summary_k1 = {'achieved_ratio': float(sum(assigned_k1.values()) / max(sum((len(stream.frame_numbers) for stream in k1_streams)), 1)), 'keyframes': int(sum(assigned_k1.values())), 'frames': int(sum((len(stream.frame_numbers) for stream in k1_streams)))}
        ratio_summary_k2 = {'achieved_ratio': float(sum(assigned_k2.values()) / max(sum((len(stream.frame_numbers) for stream in k2_streams)), 1)), 'keyframes': int(sum(assigned_k2.values())), 'frames': int(sum((len(stream.frame_numbers) for stream in k2_streams)))}
    if str(args.solver) in {'dp', 'dp_rewarded'}:
        keyframe_rows_k1, dense_rows_k1, segment_rows_k1 = kfbase_optimize_streams(k1_streams, penalty_k1, args)
        keyframe_rows_k2, dense_rows_k2, segment_rows_k2 = kfbase_optimize_streams(k2_streams, penalty_k2, args)
    else:
        keyframe_rows_k1: list[dict[str, object]] = []
        dense_rows_k1: list[dict[str, object]] = []
        segment_rows_k1: list[dict[str, object]] = []
        for stream in k1_streams:
            kf, dense, seg = kfbase_optimize_streams([stream], float(assigned_k1[stream.stream_id]), args)
            keyframe_rows_k1.extend(kf)
            dense_rows_k1.extend(dense)
            segment_rows_k1.extend(seg)
        keyframe_rows_k2 = []
        dense_rows_k2 = []
        segment_rows_k2 = []
        for stream in k2_streams:
            kf, dense, seg = kfbase_optimize_streams([stream], float(assigned_k2[stream.stream_id]), args)
            keyframe_rows_k2.extend(kf)
            dense_rows_k2.extend(dense)
            segment_rows_k2.extend(seg)
    keyframe_rows = keyframe_rows_k1 + keyframe_rows_k2
    dense_rows = dense_rows_k1 + dense_rows_k2
    segment_rows = segment_rows_k1 + segment_rows_k2
    dense_union_rows = kfbase_merge_dense_rows_to_union(dense_rows)
    keyframe_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    dense_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    segment_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['run_id']), int(row['slot_id'])))
    kfbase_write_json(output_dir / 'final_keyframes.json', keyframe_rows)
    kfbase_write_json(output_dir / 'interpolated_union.json', dense_union_rows)
    kfbase_write_csv(output_dir / 'stream_segments.csv', segment_rows, ['stream_id', 'track_id', 'mode', 'run_id', 'slot_id', 'frame_count', 'keyframe_count', 'keyframe_ratio', 'objective', 'weighted_param_rmse'])
    summary = {'input_metrics_csv': str(input_metrics), 'stream_count': int(len(streams)), 'row_count': int(len(rows)), 'mode_summary': {'K1': {'lambda': float(penalty_k1), 'target_ratio': float(args.target_k1_ratio), **ratio_summary_k1}, 'K2': {'lambda': float(penalty_k2), 'target_ratio': float(args.target_k2_ratio), **ratio_summary_k2}}, 'total_keyframe_rows': int(len(keyframe_rows)), 'total_dense_rows': int(len(dense_rows)), 'total_union_rows': int(len(dense_union_rows)), 'settings': {'solver': str(args.solver), 'smooth_alpha': float(args.smooth_alpha), 'confidence_floor': float(args.confidence_floor), 'error_scale': float(args.error_scale), 'value_refine': str(args.value_refine), 'value_refine_ridge': float(args.value_refine_ridge), 'value_refine_damping': float(args.value_refine_damping), 'weight_error_gain': float(args.weight_error_gain), 'weight_curvature_gain': float(args.weight_curvature_gain), 'importance_cap': float(args.importance_cap), 'reward_error_gain': float(args.reward_error_gain), 'reward_curvature_gain': float(args.reward_curvature_gain), 'reward_cap': float(args.reward_cap), 'auto_break_threshold': float(args.auto_break_threshold), 'auto_break_min_length': int(args.auto_break_min_length), 'auto_break_min_separation': int(args.auto_break_min_separation), 'candidate_multiplier': float(args.candidate_multiplier), 'candidate_min_separation': int(args.candidate_min_separation), 'candidate_uniform_support': int(args.candidate_uniform_support), 'rdp_quantile': float(args.rdp_quantile), 'min_gap': int(args.min_gap), 'max_gap': int(args.max_gap), 'local_search_radius': int(args.local_search_radius), 'keyframe_value_source': str(args.keyframe_value_source)}}
    kfbase_write_json(output_dir / 'summary.json', summary)


kfbase_module = _register_inline_module(
    'optimize_keyframes_standalone',
    {
    'parse_args': 'kfbase_parse_args',
    'MetricRow': 'kfbase_MetricRow',
    'StreamSegment': 'kfbase_StreamSegment',
    'canonicalize_ellipse': 'kfbase_canonicalize_ellipse',
    'circular_angle_distance_deg': 'kfbase_circular_angle_distance_deg',
    'unwrap_angles_deg': 'kfbase_unwrap_angles_deg',
    'compute_confidence': 'kfbase_compute_confidence',
    'load_metric_rows': 'kfbase_load_metric_rows',
    'split_runs': 'kfbase_split_runs',
    'ellipse_pair_cost': 'kfbase_ellipse_pair_cost',
    'stabilize_k2_slots': 'kfbase_stabilize_k2_slots',
    'build_stream_segments': 'kfbase_build_stream_segments',
    'build_second_difference_matrix': 'kfbase_build_second_difference_matrix',
    'state_to_q': 'kfbase_state_to_q',
    'q_to_state': 'kfbase_q_to_state',
    'smooth_stream_segment': 'kfbase_smooth_stream_segment',
    'choose_anchor_q': 'kfbase_choose_anchor_q',
    'robust_normalize': 'kfbase_robust_normalize',
    'compute_curvature_signal': 'kfbase_compute_curvature_signal',
    'derive_stream_importance': 'kfbase_derive_stream_importance',
    'split_stream_on_breaks': 'kfbase_split_stream_on_breaks',
    'PrefixCostCache': 'kfbase_PrefixCostCache',
    'build_prefix_cache': 'kfbase_build_prefix_cache',
    'get_stream_prefix_cache': 'kfbase_get_stream_prefix_cache',
    'get_stream_interval_costs': 'kfbase_get_stream_interval_costs',
    'interval_surrogate_cost': 'kfbase_interval_surrogate_cost',
    'decode_keyframes_dp': 'kfbase_decode_keyframes_dp',
    'refine_keyframes_locally': 'kfbase_refine_keyframes_locally',
    'interpolate_dense_q': 'kfbase_interpolate_dense_q',
    'refine_keyframe_values_global_ls': 'kfbase_refine_keyframe_values_global_ls',
    'refine_keyframe_values_segment_ls': 'kfbase_refine_keyframe_values_segment_ls',
    'refine_keyframe_values_residual_nudge': 'kfbase_refine_keyframe_values_residual_nudge',
    'interpolate_from_key_values': 'kfbase_interpolate_from_key_values',
    'min_required_keyframes': 'kfbase_min_required_keyframes',
    'max_allowed_keyframes': 'kfbase_max_allowed_keyframes',
    'choose_uniform_positions': 'kfbase_choose_uniform_positions',
    'allocate_target_keyframes': 'kfbase_allocate_target_keyframes',
    'best_split_for_segment': 'kfbase_best_split_for_segment',
    'residuals_for_segment': 'kfbase_residuals_for_segment',
    'best_residual_split_for_segment': 'kfbase_best_residual_split_for_segment',
    'select_keyframes_uniform': 'kfbase_select_keyframes_uniform',
    'select_keyframes_best_first_split': 'kfbase_select_keyframes_best_first_split',
    'select_keyframes_rdp_quantile': 'kfbase_select_keyframes_rdp_quantile',
    'select_keyframes_greedy_split': 'kfbase_select_keyframes_greedy_split',
    'ensure_target_count_with_peaks': 'kfbase_ensure_target_count_with_peaks',
    'select_keyframes_trend_knots': 'kfbase_select_keyframes_trend_knots',
    'event_trigger_positions_for_threshold': 'kfbase_event_trigger_positions_for_threshold',
    'select_keyframes_event_triggered': 'kfbase_select_keyframes_event_triggered',
    'select_candidate_positions': 'kfbase_select_candidate_positions',
    'decode_keyframes_candidate_dp': 'kfbase_decode_keyframes_candidate_dp',
    'decode_keyframes_candidate_budget_dp': 'kfbase_decode_keyframes_candidate_budget_dp',
    'select_keyframes_bottom_up_merge': 'kfbase_select_keyframes_bottom_up_merge',
    'compute_ratio_for_lambda': 'kfbase_compute_ratio_for_lambda',
    'find_penalty_for_target_ratio': 'kfbase_find_penalty_for_target_ratio',
    'optimize_streams': 'kfbase_optimize_streams',
    'merge_dense_rows_to_union': 'kfbase_merge_dense_rows_to_union',
    'write_json': 'kfbase_write_json',
    'write_csv': 'kfbase_write_csv',
    'main': 'kfbase_main',
},
)




# ==============================================================================
# Inlined from: keyframe_opt/optimize_keyframes_dense_recall_standalone.py
# ==============================================================================

import argparse
import math
from pathlib import Path
import numpy as np
import optimize_keyframes_standalone as base

def kfdense_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='V6KF_D-style keyframe optimizer with dense-ellipse recall-aware value refinement.')
    parser.add_argument('--input-metrics-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-k1-ratio', type=float, default=0.1)
    parser.add_argument('--target-k2-ratio', type=float, default=0.16)
    parser.add_argument('--solver', choices=['dp', 'dp_rewarded'], default='dp')
    parser.add_argument('--lambda-k1', type=float, default=-1.0)
    parser.add_argument('--lambda-k2', type=float, default=-1.0)
    parser.add_argument('--lambda-search-iters', type=int, default=16)
    parser.add_argument('--smooth-alpha', type=float, default=1.0)
    parser.add_argument('--confidence-floor', type=float, default=0.18)
    parser.add_argument('--error-scale', type=float, default=4000.0)
    parser.add_argument('--min-gap', type=int, default=2)
    parser.add_argument('--max-gap', type=int, default=30)
    parser.add_argument('--local-search-radius', type=int, default=2)
    parser.add_argument('--value-refine', choices=['none', 'global_ls', 'segment_ls', 'residual_nudge'], default='global_ls')
    parser.add_argument('--value-refine-ridge', type=float, default=0.001)
    parser.add_argument('--value-refine-damping', type=float, default=1.0)
    parser.add_argument('--min-segment-length', type=int, default=3)
    parser.add_argument('--theta-weight-floor', type=float, default=0.2)
    parser.add_argument('--weight-error-gain', type=float, default=1.0)
    parser.add_argument('--weight-curvature-gain', type=float, default=1.0)
    parser.add_argument('--importance-cap', type=float, default=4.0)
    parser.add_argument('--reward-error-gain', type=float, default=0.75)
    parser.add_argument('--reward-curvature-gain', type=float, default=1.25)
    parser.add_argument('--reward-cap', type=float, default=1.5)
    parser.add_argument('--auto-break-threshold', type=float, default=-1.0)
    parser.add_argument('--auto-break-min-length', type=int, default=8)
    parser.add_argument('--auto-break-min-separation', type=int, default=6)
    parser.add_argument('--keyframe-value-source', choices=['smoothed', 'raw', 'confidence_blend'], default='confidence_blend')
    parser.add_argument('--k2-slot-center-weight', type=float, default=1.0)
    parser.add_argument('--k2-slot-size-weight', type=float, default=0.65)
    parser.add_argument('--k2-slot-angle-weight', type=float, default=0.2)
    parser.add_argument('--max-streams', type=int, default=-1)
    parser.add_argument('--dense-recall-target', type=float, default=0.96)
    parser.add_argument('--dense-recall-samples', type=int, default=61)
    parser.add_argument('--dense-recall-max-inflate-log', type=float, default=1.2)
    parser.add_argument('--dense-recall-search-iters', type=int, default=20)
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

def kfdense_ellipse_membership(points_x: np.ndarray, points_y: np.ndarray, states: np.ndarray) -> np.ndarray:
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

def kfdense_approx_dense_recall(source_states: np.ndarray, pred_states: np.ndarray, disk_samples: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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

def kfdense_enforce_dense_recall_target(stream: base.StreamSegment, key_q: np.ndarray, chosen: list[int], args: argparse.Namespace, disk_samples: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool]]:
    interp_q = base.interpolate_from_key_values(key_q, chosen, len(stream.frame_numbers))
    base_state = base.q_to_state(interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
    base_recall, _per_frame, weights = kfdense_approx_dense_recall(stream.raw_states, base_state, disk_samples)
    target = float(args.dense_recall_target)
    info: dict[str, float | bool] = {'dense_recall_before': float(base_recall), 'dense_recall_after': float(base_recall), 'inflate_log_delta': 0.0, 'dense_recall_target': target, 'dense_recall_attained': bool(base_recall >= target), 'source_area_sum': float(np.sum(weights))}
    if target <= 0.0 or base_recall >= target or len(chosen) == 0:
        return (key_q, info)
    high = float(args.dense_recall_max_inflate_log)
    high_q = kfdense_apply_uniform_inflation(key_q, high)
    high_interp = base.interpolate_from_key_values(high_q, chosen, len(stream.frame_numbers))
    high_state = base.q_to_state(high_interp, scale=stream.transform_scale, theta_scale=stream.theta_scale)
    high_recall, _per_frame, _weights = kfdense_approx_dense_recall(stream.raw_states, high_state, disk_samples)
    if high_recall < target:
        info['dense_recall_after'] = float(high_recall)
        info['inflate_log_delta'] = float(high)
        info['dense_recall_attained'] = False
        return (high_q, info)
    low = 0.0
    best_q = high_q
    best_recall = high_recall
    for _ in range(int(args.dense_recall_search_iters)):
        mid = 0.5 * (low + high)
        mid_q = kfdense_apply_uniform_inflation(key_q, mid)
        mid_interp = base.interpolate_from_key_values(mid_q, chosen, len(stream.frame_numbers))
        mid_state = base.q_to_state(mid_interp, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        mid_recall, _per_frame, _weights = kfdense_approx_dense_recall(stream.raw_states, mid_state, disk_samples)
        if mid_recall >= target:
            high = mid
            best_q = mid_q
            best_recall = mid_recall
        else:
            low = mid
    info['dense_recall_after'] = float(best_recall)
    info['inflate_log_delta'] = float(high)
    info['dense_recall_attained'] = True
    return (best_q, info)

def kfdense_optimize_streams_dense_recall(streams: list[base.StreamSegment], mode_penalty: float, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    disk_samples = kfdense_unit_disk_samples(int(args.dense_recall_samples))
    for stream in streams:
        assert stream.smoothed_q is not None
        anchor_q = base.choose_anchor_q(stream, source=str(args.keyframe_value_source))
        fit_weights = stream.importance if stream.importance is not None else stream.confidence
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        chosen, objective = base.decode_keyframes_dp(stream.smoothed_q, fit_weights, mode_penalty, min_gap=int(args.min_gap), max_gap=int(args.max_gap), rewards=stream.keyframe_reward if str(args.solver) == 'dp_rewarded' else None)
        chosen = base.refine_keyframes_locally(stream.smoothed_q, fit_weights, chosen, min_gap=int(args.min_gap), max_gap=int(args.max_gap), radius=int(args.local_search_radius))
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
        if len(chosen) >= 2 and str(args.value_refine) != 'none':
            if str(args.value_refine) == 'global_ls':
                key_q = base.refine_keyframe_values_global_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'segment_ls':
                key_q = base.refine_keyframe_values_segment_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'residual_nudge':
                key_q = base.refine_keyframe_values_residual_nudge(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, damping=float(args.value_refine_damping))
        key_q, dense_info = kfdense_enforce_dense_recall_target(stream=stream, key_q=key_q, chosen=chosen, args=args, disk_samples=disk_samples)
        interp_q = base.interpolate_from_key_values(key_q, chosen, len(stream.frame_numbers))
        dense_state = base.q_to_state(interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        raw_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        weighted_rmse = float(math.sqrt(np.average(np.sum((interp_q - raw_q) ** 2, axis=1), weights=np.maximum(fit_weights, 1e-06))))
        segment_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame_count': int(len(stream.frame_numbers)), 'keyframe_count': int(len(chosen)), 'keyframe_ratio': float(len(chosen) / max(len(stream.frame_numbers), 1)), 'objective': float(objective), 'weighted_param_rmse': weighted_rmse, 'dense_recall_before': float(dense_info['dense_recall_before']), 'dense_recall_after': float(dense_info['dense_recall_after']), 'inflate_log_delta': float(dense_info['inflate_log_delta']), 'dense_recall_attained': int(bool(dense_info['dense_recall_attained'])), 'source_area_sum': float(dense_info['source_area_sum'])})
        key_states = base.q_to_state(key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        chosen_set = set(chosen)
        for key_idx, local_idx in enumerate(chosen):
            keyframe_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame': int(stream.frame_numbers[local_idx]), 'ellipse': key_states[key_idx].tolist()})
        for local_idx, frame in enumerate(stream.frame_numbers):
            dense_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'frame': int(frame), 'ellipse': dense_state[local_idx].tolist(), 'is_keyframe': int(local_idx in chosen_set)})
    return (keyframe_rows, dense_rows, segment_rows)

def kfdense_main() -> None:
    args = kfdense_parse_args()
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = base.load_metric_rows(input_metrics, confidence_floor=float(args.confidence_floor), error_scale=float(args.error_scale))
    streams = base.build_stream_segments(args, rows)
    if int(args.max_streams) > 0:
        streams = streams[:int(args.max_streams)]
    for stream in streams:
        base.smooth_stream_segment(stream, args)
        base.derive_stream_importance(stream, args)
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[base.StreamSegment] = []
        for stream in streams:
            broken_streams.extend(base.split_stream_on_breaks(stream, args))
        streams = broken_streams
    k1_streams = [stream for stream in streams if stream.mode == 'K1']
    k2_streams = [stream for stream in streams if stream.mode == 'K2']
    penalty_k1, ratio_summary_k1 = base.find_penalty_for_target_ratio(k1_streams, target_ratio=float(args.target_k1_ratio), fallback_penalty=0.5 if float(args.lambda_k1) <= 0 else float(args.lambda_k1), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=str(args.solver) == 'dp_rewarded')
    penalty_k2, ratio_summary_k2 = base.find_penalty_for_target_ratio(k2_streams, target_ratio=float(args.target_k2_ratio), fallback_penalty=0.35 if float(args.lambda_k2) <= 0 else float(args.lambda_k2), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=str(args.solver) == 'dp_rewarded')
    keyframe_rows_k1, dense_rows_k1, segment_rows_k1 = kfdense_optimize_streams_dense_recall(k1_streams, penalty_k1, args)
    keyframe_rows_k2, dense_rows_k2, segment_rows_k2 = kfdense_optimize_streams_dense_recall(k2_streams, penalty_k2, args)
    keyframe_rows = keyframe_rows_k1 + keyframe_rows_k2
    dense_rows = dense_rows_k1 + dense_rows_k2
    segment_rows = segment_rows_k1 + segment_rows_k2
    dense_union_rows = base.merge_dense_rows_to_union(dense_rows)
    keyframe_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    dense_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    segment_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['run_id']), int(row['slot_id'])))
    base.write_json(output_dir / 'final_keyframes.json', keyframe_rows)
    base.write_json(output_dir / 'interpolated_union.json', dense_union_rows)
    base.write_csv(output_dir / 'stream_segments.csv', segment_rows, ['stream_id', 'track_id', 'mode', 'run_id', 'slot_id', 'frame_count', 'keyframe_count', 'keyframe_ratio', 'objective', 'weighted_param_rmse', 'dense_recall_before', 'dense_recall_after', 'inflate_log_delta', 'dense_recall_attained', 'source_area_sum'])
    total_source_area = sum((float(row['source_area_sum']) for row in segment_rows))
    dense_recall_before = sum((float(row['dense_recall_before']) * float(row['source_area_sum']) for row in segment_rows)) / max(total_source_area, 1e-08)
    dense_recall_after = sum((float(row['dense_recall_after']) * float(row['source_area_sum']) for row in segment_rows)) / max(total_source_area, 1e-08)
    inflated_segments = sum((1 for row in segment_rows if float(row['inflate_log_delta']) > 1e-08))
    unattained_segments = sum((1 for row in segment_rows if int(row['dense_recall_attained']) == 0))
    summary = {'input_metrics_csv': str(input_metrics), 'stream_count': int(len(streams)), 'row_count': int(len(rows)), 'mode_summary': {'K1': {'lambda': float(penalty_k1), 'target_ratio': float(args.target_k1_ratio), **ratio_summary_k1}, 'K2': {'lambda': float(penalty_k2), 'target_ratio': float(args.target_k2_ratio), **ratio_summary_k2}}, 'total_keyframe_rows': int(len(keyframe_rows)), 'total_dense_rows': int(len(dense_rows)), 'total_union_rows': int(len(dense_union_rows)), 'dense_recall_summary': {'target': float(args.dense_recall_target), 'global_before': float(dense_recall_before), 'global_after': float(dense_recall_after), 'inflated_segments': int(inflated_segments), 'unattained_segments': int(unattained_segments)}, 'settings': {'solver': str(args.solver), 'smooth_alpha': float(args.smooth_alpha), 'value_refine': str(args.value_refine), 'keyframe_value_source': str(args.keyframe_value_source), 'dense_recall_target': float(args.dense_recall_target), 'dense_recall_samples': int(args.dense_recall_samples), 'dense_recall_max_inflate_log': float(args.dense_recall_max_inflate_log), 'dense_recall_search_iters': int(args.dense_recall_search_iters), 'min_gap': int(args.min_gap), 'max_gap': int(args.max_gap)}}
    base.write_json(output_dir / 'summary.json', summary)


kfdense_module = _register_inline_module(
    'optimize_keyframes_dense_recall_standalone',
    {
    'parse_args': 'kfdense_parse_args',
    'unit_disk_samples': 'kfdense_unit_disk_samples',
    'ellipse_membership': 'kfdense_ellipse_membership',
    'approx_dense_recall': 'kfdense_approx_dense_recall',
    'apply_uniform_inflation': 'kfdense_apply_uniform_inflation',
    'enforce_dense_recall_target': 'kfdense_enforce_dense_recall_target',
    'optimize_streams_dense_recall': 'kfdense_optimize_streams_dense_recall',
    'main': 'kfdense_main',
},
)




# ==============================================================================
# Inlined from: keyframe_opt/optimize_keyframes_trackk_dense_recall_standalone.py
# ==============================================================================

import argparse
import math
import time
from pathlib import Path
import numpy as np
import optimize_keyframes_standalone as base
import optimize_keyframes_dense_recall_standalone as dense_base

def kftrackk_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dense-recall keyframe optimizer that does not split on K1/K2 mode changes; only track, K, and frame continuity.')
    parser.add_argument('--input-metrics-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-ratio', type=float, default=-1.0)
    parser.add_argument('--target-k1-ratio', type=float, default=0.1)
    parser.add_argument('--target-k2-ratio', type=float, default=0.16)
    parser.add_argument('--solver', choices=['dp', 'dp_rewarded'], default='dp')
    parser.add_argument('--lambda-all', type=float, default=-1.0)
    parser.add_argument('--lambda-search-iters', type=int, default=16)
    parser.add_argument('--smooth-alpha', type=float, default=1.0)
    parser.add_argument('--confidence-floor', type=float, default=0.18)
    parser.add_argument('--error-scale', type=float, default=4000.0)
    parser.add_argument('--min-gap', type=int, default=2)
    parser.add_argument('--max-gap', type=int, default=30)
    parser.add_argument('--local-search-radius', type=int, default=2)
    parser.add_argument('--value-refine', choices=['none', 'global_ls', 'segment_ls', 'residual_nudge'], default='global_ls')
    parser.add_argument('--value-refine-ridge', type=float, default=0.001)
    parser.add_argument('--value-refine-damping', type=float, default=1.0)
    parser.add_argument('--min-segment-length', type=int, default=3)
    parser.add_argument('--theta-weight-floor', type=float, default=0.2)
    parser.add_argument('--weight-error-gain', type=float, default=1.0)
    parser.add_argument('--weight-curvature-gain', type=float, default=1.0)
    parser.add_argument('--importance-cap', type=float, default=4.0)
    parser.add_argument('--reward-error-gain', type=float, default=0.75)
    parser.add_argument('--reward-curvature-gain', type=float, default=1.25)
    parser.add_argument('--reward-cap', type=float, default=1.5)
    parser.add_argument('--auto-break-threshold', type=float, default=-1.0)
    parser.add_argument('--auto-break-min-length', type=int, default=8)
    parser.add_argument('--auto-break-min-separation', type=int, default=6)
    parser.add_argument('--keyframe-value-source', choices=['smoothed', 'raw', 'confidence_blend'], default='confidence_blend')
    parser.add_argument('--k2-slot-center-weight', type=float, default=1.0)
    parser.add_argument('--k2-slot-size-weight', type=float, default=0.65)
    parser.add_argument('--k2-slot-angle-weight', type=float, default=0.2)
    parser.add_argument('--max-streams', type=int, default=-1)
    parser.add_argument('--dense-recall-target', type=float, default=0.96)
    parser.add_argument('--dense-recall-samples', type=int, default=61)
    parser.add_argument('--dense-recall-max-inflate-log', type=float, default=1.2)
    parser.add_argument('--dense-recall-search-iters', type=int, default=20)
    return parser.parse_args()

def kftrackk_resolve_target_ratio(args: argparse.Namespace) -> float:
    if float(args.target_ratio) > 0.0:
        return float(args.target_ratio)
    if abs(float(args.target_k1_ratio) - float(args.target_k2_ratio)) < 1e-09:
        return float(args.target_k1_ratio)
    return 0.5 * (float(args.target_k1_ratio) + float(args.target_k2_ratio))

def kftrackk_split_runs_track_k(rows: list[base.MetricRow]) -> list[list[base.MetricRow]]:
    grouped: dict[str, list[base.MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    runs: list[list[base.MetricRow]] = []
    for _track_id, track_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        track_rows.sort(key=lambda r: r.frame)
        current: list[base.MetricRow] = []
        prev: base.MetricRow | None = None
        for row in track_rows:
            split = prev is None or len(row.ellipse_params) != len(prev.ellipse_params) or row.frame != prev.frame + 1
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

def kftrackk_build_stream_segments_track_k(args: argparse.Namespace, rows: list[base.MetricRow]) -> list[base.StreamSegment]:
    runs = kftrackk_split_runs_track_k(rows)
    streams: list[base.StreamSegment] = []
    for run_id, run_rows in enumerate(runs):
        slot_count = len(run_rows[0].ellipse_params)
        if slot_count == 2:
            stabilized = base.stabilize_k2_slots(run_rows, center_weight=float(args.k2_slot_center_weight), size_weight=float(args.k2_slot_size_weight), angle_weight=float(args.k2_slot_angle_weight))
        else:
            stabilized = [[list(row.ellipse_params[0])] for row in run_rows]
        frame_modes = [row.mode for row in run_rows]
        run_mode = frame_modes[0] if len(set(frame_modes)) == 1 else 'MIXED'
        for slot_id in range(slot_count):
            states = np.asarray([frame_slots[slot_id] for frame_slots in stabilized], dtype=np.float64)
            states[:, 4] = base.unwrap_angles_deg(states[:, 4])
            confidence = np.asarray([base.compute_confidence(row, floor=float(args.confidence_floor), error_scale=float(args.error_scale)) for row in run_rows], dtype=np.float64)
            weighted_error = np.asarray([row.weighted_error for row in run_rows], dtype=np.float64)
            stream = base.StreamSegment(stream_id=f'{run_rows[0].track_id}:K{slot_count}:run{run_id}:slot{slot_id}', track_id=run_rows[0].track_id, mode=run_mode, run_id=run_id, slot_id=slot_id, frame_numbers=np.asarray([row.frame for row in run_rows], dtype=np.int32), raw_states=states, confidence=confidence, weighted_error=weighted_error)
            setattr(stream, 'frame_modes', list(frame_modes))
            setattr(stream, 'ellipse_count', int(slot_count))
            streams.append(stream)
    return streams

def kftrackk_interpolated_state(stream: base.StreamSegment, key_q: np.ndarray, chosen: list[int]) -> np.ndarray:
    interp_q = base.interpolate_from_key_values(key_q, chosen, len(stream.frame_numbers))
    return base.q_to_state(interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)

def kftrackk_recall_info(recall: float, weights: np.ndarray, target: float, inflate_log_delta: float, attained: bool) -> dict[str, float | bool]:
    return {'dense_recall_before': float(recall), 'dense_recall_after': float(recall), 'inflate_log_delta': float(inflate_log_delta), 'dense_recall_target': float(target), 'dense_recall_attained': bool(attained), 'source_area_sum': float(np.sum(weights))}

def kftrackk_apply_uniform_key_inflation(key_q: np.ndarray, delta: float) -> np.ndarray:
    out = key_q.copy()
    out[:, 2] += float(delta)
    out[:, 3] += float(delta)
    return out

def kftrackk_approx_union_frame_recall(source_slots: list[np.ndarray], pred_slots: list[np.ndarray], disk_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return kftrackk_union_recall_from_source_samples(kftrackk_source_sample_payloads(source_slots, disk_samples), pred_slots)

def kftrackk_source_sample_payloads(source_slots: list[np.ndarray], disk_samples: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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

def kftrackk_union_recall_from_source_samples(payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]], pred_slots: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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
    per_frame = np.divide(covered, weights, out=np.ones_like(covered), where=weights > 0.0)
    return (per_frame, weights)

def kftrackk_union_recall_score(source_slots: list[np.ndarray], pred_slots: list[np.ndarray], disk_samples: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    per_frame, weights = kftrackk_approx_union_frame_recall(source_slots, pred_slots, disk_samples)
    recall = float(np.average(per_frame, weights=np.maximum(weights, 1e-06))) if len(per_frame) else 1.0
    return (recall, per_frame, weights)

def kftrackk_pred_slots_with_inflation(items: list[dict[str, object]], deltas: tuple[float, float]) -> list[np.ndarray]:
    pred_slots: list[np.ndarray] = []
    for item, delta in zip(items, deltas, strict=True):
        pred_slots.append(
            kftrackk_interpolated_state(
                item['stream'],
                kftrackk_apply_uniform_key_inflation(item['key_q'], float(delta)),
                item['chosen'],
            )
        )
    return pred_slots

def kftrackk_union_inflation_cost(base_area_sums: np.ndarray, deltas: tuple[float, float]) -> float:
    delta_arr = np.asarray(deltas, dtype=np.float64)
    return float(np.sum(base_area_sums * np.maximum(np.exp(2.0 * delta_arr) - 1.0, 0.0)))

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
    base_pred_slots = [kftrackk_interpolated_state(item['stream'], item['key_q'], item['chosen']) for item in items]
    base_area_sums = np.asarray([float(np.sum(np.maximum(slot[:, 2] * slot[:, 3], 1e-06))) for slot in base_pred_slots], dtype=np.float64)

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
        per_frame, weights = kftrackk_union_recall_from_source_samples(source_payloads, inflated_pred_slots(deltas))
        recall = float(np.average(per_frame, weights=np.maximum(weights, 1e-06))) if len(per_frame) else 1.0
        return float(recall)

    def remember(best: tuple[tuple[float, float], float, float] | None, deltas: tuple[float, float], recall: float) -> tuple[tuple[float, float], float, float]:
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

    def minimal_partner_delta(slot_index: int, fixed_delta: float) -> tuple[tuple[float, float], float] | None:
        endpoint = (float(fixed_delta), max_delta) if slot_index == 0 else (max_delta, float(fixed_delta))
        if evaluate(endpoint) < target:
            return None
        low_partner = 0.0
        high_partner = max_delta
        best_recall = target
        for _ in range(search_iters):
            mid = 0.5 * (low_partner + high_partner)
            deltas = (float(fixed_delta), mid) if slot_index == 0 else (mid, float(fixed_delta))
            recall = evaluate(deltas)
            if recall >= target:
                high_partner = mid
                best_recall = recall
            else:
                low_partner = mid
        deltas = (float(fixed_delta), high_partner) if slot_index == 0 else (high_partner, float(fixed_delta))
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
        scan_fixed_deltas(max(0.0, best_delta0 - step), min(max_delta, best_delta0 + step), 9)
        scan_fixed_deltas(max(0.0, best_delta1 - step), min(max_delta, best_delta1 + step), 9)

    if best is None:
        return ((max_delta, max_delta), float(high_recall), False)
    return (best[0], float(best[1]), True)

def kftrackk_enforce_union_frame_recall_target(items: list[dict[str, object]], args: argparse.Namespace, disk_samples: np.ndarray) -> None:
    items.sort(key=lambda item: int(getattr(item['stream'], 'slot_id')))
    streams = [item['stream'] for item in items]
    source_slots = [stream.raw_states for stream in streams]
    target = float(args.dense_recall_target)
    pred_slots = [kftrackk_interpolated_state(item['stream'], item['key_q'], item['chosen']) for item in items]
    base_recall, _per_frame, _weights = kftrackk_union_recall_score(source_slots, pred_slots, disk_samples)
    if target <= 0.0 or base_recall >= target or any(len(item['chosen']) == 0 for item in items):
        for item, stream in zip(items, streams, strict=True):
            slot_weights = np.maximum(stream.raw_states[:, 2] * stream.raw_states[:, 3], 1e-06)
            item['dense_info'] = kftrackk_recall_info(base_recall, slot_weights, target, 0.0, base_recall >= target)
        return
    deltas, repaired_recall, attained = kftrackk_find_joint_union_inflation(items, source_slots, target, args, disk_samples)
    for item, stream, delta in zip(items, streams, deltas, strict=True):
        item['key_q'] = kftrackk_apply_uniform_key_inflation(item['key_q'], float(delta))
        slot_weights = np.maximum(stream.raw_states[:, 2] * stream.raw_states[:, 3], 1e-06)
        dense_info = kftrackk_recall_info(base_recall, slot_weights, target, float(delta), bool(attained))
        dense_info['dense_recall_after'] = float(repaired_recall)
        item['dense_info'] = dense_info

def kftrackk_optimize_streams_track_k_dense_recall(streams: list[base.StreamSegment], penalty: float, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    keyframe_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    disk_samples = dense_base.unit_disk_samples(int(args.dense_recall_samples))
    timings = {'stream_loop_total': 0.0, 'anchor_setup': 0.0, 'decode_dp': 0.0, 'refine_local': 0.0, 'value_refine': 0.0, 'dense_recall_enforce': 0.0, 'interpolate_and_state': 0.0, 'emit_rows': 0.0}
    total_t0 = time.perf_counter()
    work_items: list[dict[str, object]] = []
    for stream in streams:
        assert stream.smoothed_q is not None
        frame_modes: list[str] = list(getattr(stream, 'frame_modes'))
        ellipse_count = int(getattr(stream, 'ellipse_count', 1))
        t0 = time.perf_counter()
        anchor_q = base.choose_anchor_q(stream, source=str(args.keyframe_value_source))
        fit_weights = stream.importance if stream.importance is not None else stream.confidence
        target_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        prefix_cache = base.get_stream_prefix_cache(stream)
        interval_costs = base.get_stream_interval_costs(stream, max_gap=int(args.max_gap))
        timings['anchor_setup'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        chosen, objective = base.decode_keyframes_dp(stream.smoothed_q, fit_weights, penalty, min_gap=int(args.min_gap), max_gap=int(args.max_gap), rewards=stream.keyframe_reward if str(args.solver) == 'dp_rewarded' else None, prefix_cache=prefix_cache, interval_costs=interval_costs)
        timings['decode_dp'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        chosen = base.refine_keyframes_locally(stream.smoothed_q, fit_weights, chosen, min_gap=int(args.min_gap), max_gap=int(args.max_gap), radius=int(args.local_search_radius), prefix_cache=prefix_cache, interval_costs=interval_costs)
        timings['refine_local'] += time.perf_counter() - t0
        key_q = np.asarray([anchor_q[idx] for idx in chosen], dtype=np.float64)
        t0 = time.perf_counter()
        if len(chosen) >= 2 and str(args.value_refine) != 'none':
            if str(args.value_refine) == 'global_ls':
                key_q = base.refine_keyframe_values_global_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'segment_ls':
                key_q = base.refine_keyframe_values_segment_ls(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, ridge=float(args.value_refine_ridge))
            elif str(args.value_refine) == 'residual_nudge':
                key_q = base.refine_keyframe_values_residual_nudge(target_q=target_q, base_key_q=key_q, keyframes=chosen, weights=fit_weights, damping=float(args.value_refine_damping))
        timings['value_refine'] += time.perf_counter() - t0
        work_items.append({'stream': stream, 'frame_modes': frame_modes, 'ellipse_count': ellipse_count, 'fit_weights': fit_weights, 'target_q': target_q, 'chosen': chosen, 'objective': objective, 'key_q': key_q})
    t0 = time.perf_counter()
    k2_groups: dict[tuple[str, int, tuple[int, ...]], list[dict[str, object]]] = {}
    for item in work_items:
        stream = item['stream']
        if int(item['ellipse_count']) == 2:
            key = (str(stream.track_id), int(stream.run_id), tuple(int(frame) for frame in stream.frame_numbers.tolist()))
            k2_groups.setdefault(key, []).append(item)
    handled: set[int] = set()
    for group_items in k2_groups.values():
        if len(group_items) == 2:
            kftrackk_enforce_union_frame_recall_target(group_items, args, disk_samples)
            handled.update(id(item) for item in group_items)
    for item in work_items:
        if id(item) in handled:
            continue
        stream = item['stream']
        key_q, dense_info = dense_base.enforce_dense_recall_target(stream=stream, key_q=item['key_q'], chosen=item['chosen'], args=args, disk_samples=disk_samples)
        item['key_q'] = key_q
        item['dense_info'] = dense_info
    timings['dense_recall_enforce'] += time.perf_counter() - t0
    for item in work_items:
        stream = item['stream']
        frame_modes = item['frame_modes']
        ellipse_count = int(item['ellipse_count'])
        fit_weights = item['fit_weights']
        target_q = item['target_q']
        chosen = item['chosen']
        objective = float(item['objective'])
        key_q = item['key_q']
        dense_info = item['dense_info']
        t0 = time.perf_counter()
        interp_q = base.interpolate_from_key_values(key_q, chosen, len(stream.frame_numbers))
        dense_state = base.q_to_state(interp_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        raw_q = stream.raw_q if stream.raw_q is not None else stream.smoothed_q
        weighted_rmse = float(math.sqrt(np.average(np.sum((interp_q - raw_q) ** 2, axis=1), weights=np.maximum(fit_weights, 1e-06))))
        segment_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': stream.mode, 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'ellipse_count': ellipse_count, 'frame_count': int(len(stream.frame_numbers)), 'keyframe_count': int(len(chosen)), 'keyframe_ratio': float(len(chosen) / max(len(stream.frame_numbers), 1)), 'objective': float(objective), 'weighted_param_rmse': weighted_rmse, 'dense_recall_before': float(dense_info['dense_recall_before']), 'dense_recall_after': float(dense_info['dense_recall_after']), 'inflate_log_delta': float(dense_info['inflate_log_delta']), 'dense_recall_attained': int(bool(dense_info['dense_recall_attained'])), 'source_area_sum': float(dense_info['source_area_sum']), 'source_modes': ','.join(sorted(set(frame_modes)))})
        key_states = base.q_to_state(key_q, scale=stream.transform_scale, theta_scale=stream.theta_scale)
        timings['interpolate_and_state'] += time.perf_counter() - t0
        chosen_set = set(chosen)
        t0 = time.perf_counter()
        for key_idx, local_idx in enumerate(chosen):
            keyframe_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': frame_modes[local_idx], 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'ellipse_count': ellipse_count, 'frame': int(stream.frame_numbers[local_idx]), 'ellipse': key_states[key_idx].tolist()})
        for local_idx, frame in enumerate(stream.frame_numbers):
            dense_rows.append({'stream_id': stream.stream_id, 'track_id': stream.track_id, 'mode': frame_modes[local_idx], 'run_id': stream.run_id, 'slot_id': stream.slot_id, 'ellipse_count': ellipse_count, 'frame': int(frame), 'ellipse': dense_state[local_idx].tolist(), 'is_keyframe': int(local_idx in chosen_set)})
        timings['emit_rows'] += time.perf_counter() - t0
    timings['stream_loop_total'] = time.perf_counter() - total_t0
    return (keyframe_rows, dense_rows, segment_rows, timings)

def kftrackk_merge_dense_rows_to_union_track_k(dense_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in dense_rows:
        key = (str(row['track_id']), int(row['run_id']), int(row['frame']))
        entry = grouped.get(key)
        if entry is None:
            entry = {'track_id': str(row['track_id']), 'mode': str(row['mode']), 'run_id': int(row['run_id']), 'frame': int(row['frame']), 'ellipse_params': [], 'has_keyframe': 0}
            grouped[key] = entry
        elif entry['mode'] != str(row['mode']):
            entry['mode'] = 'MIXED'
        entry['ellipse_params'].append((int(row['slot_id']), row['ellipse']))
        entry['has_keyframe'] = int(max(int(entry['has_keyframe']), int(row['is_keyframe'])))
    merged: list[dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda item: (int(item[0]), item[2], item[1])):
        entry = grouped[key]
        ellipses = [ellipse for _slot, ellipse in sorted(entry['ellipse_params'], key=lambda item: item[0])]
        merged.append({'track_id': entry['track_id'], 'mode': entry['mode'], 'run_id': entry['run_id'], 'frame': entry['frame'], 'ellipse_params': ellipses, 'has_keyframe': entry['has_keyframe']})
    return merged

def kftrackk_main() -> None:
    args = kftrackk_parse_args()
    input_metrics = Path(args.input_metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_sec: dict[str, float] = {}
    t0 = time.perf_counter()
    rows = base.load_metric_rows(input_metrics, confidence_floor=float(args.confidence_floor), error_scale=float(args.error_scale))
    timing_sec['load_metric_rows'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    streams = kftrackk_build_stream_segments_track_k(args, rows)
    timing_sec['build_stream_segments_track_k'] = time.perf_counter() - t0
    if int(args.max_streams) > 0:
        streams = streams[:int(args.max_streams)]
    t0 = time.perf_counter()
    for stream in streams:
        base.smooth_stream_segment(stream, args)
        base.derive_stream_importance(stream, args)
    timing_sec['smooth_and_importance'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if float(args.auto_break_threshold) >= 0.0:
        broken_streams: list[base.StreamSegment] = []
        for stream in streams:
            broken_streams.extend(base.split_stream_on_breaks(stream, args))
        streams = broken_streams
    timing_sec['auto_break_streams'] = time.perf_counter() - t0
    target_ratio = kftrackk_resolve_target_ratio(args)
    t0 = time.perf_counter()
    penalty, ratio_summary = base.find_penalty_for_target_ratio(streams, target_ratio=target_ratio, fallback_penalty=0.5 if float(args.lambda_all) <= 0 else float(args.lambda_all), min_gap=int(args.min_gap), max_gap=int(args.max_gap), search_iters=int(args.lambda_search_iters), use_rewards=str(args.solver) == 'dp_rewarded')
    timing_sec['find_penalty_for_target_ratio'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    keyframe_rows, dense_rows, segment_rows, inner_timings = kftrackk_optimize_streams_track_k_dense_recall(streams, penalty, args)
    timing_sec['optimize_streams_track_k_dense_recall'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense_union_rows = kftrackk_merge_dense_rows_to_union_track_k(dense_rows)
    timing_sec['merge_dense_rows_to_union_track_k'] = time.perf_counter() - t0
    keyframe_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    dense_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame']), int(row['slot_id'])))
    segment_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['run_id']), int(row['slot_id'])))
    t0 = time.perf_counter()
    base.write_json(output_dir / 'final_keyframes.json', keyframe_rows)
    base.write_json(output_dir / 'interpolated_union.json', dense_union_rows)
    base.write_csv(output_dir / 'stream_segments.csv', segment_rows, ['stream_id', 'track_id', 'mode', 'run_id', 'slot_id', 'ellipse_count', 'frame_count', 'keyframe_count', 'keyframe_ratio', 'objective', 'weighted_param_rmse', 'dense_recall_before', 'dense_recall_after', 'inflate_log_delta', 'dense_recall_attained', 'source_area_sum', 'source_modes'])
    timing_sec['write_outputs'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    total_source_area = sum((float(row['source_area_sum']) for row in segment_rows))
    dense_recall_before = sum((float(row['dense_recall_before']) * float(row['source_area_sum']) for row in segment_rows)) / max(total_source_area, 1e-08)
    dense_recall_after = sum((float(row['dense_recall_after']) * float(row['source_area_sum']) for row in segment_rows)) / max(total_source_area, 1e-08)
    inflated_segments = sum((1 for row in segment_rows if float(row['inflate_log_delta']) > 1e-08))
    unattained_segments = sum((1 for row in segment_rows if int(row['dense_recall_attained']) == 0))
    mixed_mode_streams = sum((1 for row in segment_rows if str(row['mode']) == 'MIXED'))
    timing_sec['summarize_outputs'] = time.perf_counter() - t0
    summary = {'input_metrics_csv': str(input_metrics), 'stream_count': int(len(streams)), 'row_count': int(len(rows)), 'ratio_summary': {'lambda': float(penalty), 'target_ratio': float(target_ratio), **ratio_summary}, 'total_keyframe_rows': int(len(keyframe_rows)), 'total_dense_rows': int(len(dense_rows)), 'total_union_rows': int(len(dense_union_rows)), 'dense_recall_summary': {'target': float(args.dense_recall_target), 'global_before': float(dense_recall_before), 'global_after': float(dense_recall_after), 'inflated_segments': int(inflated_segments), 'unattained_segments': int(unattained_segments)}, 'segmentation_summary': {'split_policy': 'track_and_ellipse_count_only', 'mixed_mode_streams': int(mixed_mode_streams)}, 'timing_sec': {**{key: float(val) for key, val in timing_sec.items()}, 'opt_inner': {key: float(val) for key, val in inner_timings.items()}}, 'settings': {'solver': str(args.solver), 'smooth_alpha': float(args.smooth_alpha), 'value_refine': str(args.value_refine), 'keyframe_value_source': str(args.keyframe_value_source), 'dense_recall_target': float(args.dense_recall_target), 'dense_recall_samples': int(args.dense_recall_samples), 'dense_recall_max_inflate_log': float(args.dense_recall_max_inflate_log), 'dense_recall_search_iters': int(args.dense_recall_search_iters), 'min_gap': int(args.min_gap), 'max_gap': int(args.max_gap)}}
    base.write_json(output_dir / 'summary.json', summary)


kftrackk_module = _register_inline_module(
    'optimize_keyframes_trackk_dense_recall_standalone',
    {
    'parse_args': 'kftrackk_parse_args',
    'resolve_target_ratio': 'kftrackk_resolve_target_ratio',
    'split_runs_track_k': 'kftrackk_split_runs_track_k',
    'build_stream_segments_track_k': 'kftrackk_build_stream_segments_track_k',
    'optimize_streams_track_k_dense_recall': 'kftrackk_optimize_streams_track_k_dense_recall',
    'merge_dense_rows_to_union_track_k': 'kftrackk_merge_dense_rows_to_union_track_k',
    'main': 'kftrackk_main',
},
)




# ==============================================================================
# Inlined from: keyframe_opt/evaluate_keyframes_exact.py
# ==============================================================================

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
kfeval_ROOT = Path(__file__).resolve().parents[1]
if str(kfeval_ROOT) not in sys.path:
    sys.path.insert(0, str(kfeval_ROOT))
import standalone_runtime_fst as fst

def kfeval_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Exact evaluator for keyframe-optimized ellipse sequences.')
    parser.add_argument('--input-union-json', required=True)
    parser.add_argument('--input-tracked-sqlite', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-metrics-csv', default='')
    return parser.parse_args()

def kfeval_load_union_rows(path: Path) -> dict[tuple[int, str], dict[str, object]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row['frame']), str(row['track_id']))
        lookup[key] = row
    return lookup

def kfeval_aggregate_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    gt_area = sum((float(row['gt_area']) for row in rows))
    pred_area = sum((float(row['pred_area']) for row in rows))
    intersection = sum((float(row['intersection']) for row in rows))
    union = sum((float(row['union']) for row in rows))
    weighted_error = sum((float(row['weighted_error']) for row in rows))
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {'row_count': float(len(rows)), 'gt_area': float(gt_area), 'pred_area': float(pred_area), 'intersection': float(intersection), 'union': float(union), 'global_recall': float(recall), 'global_precision': float(precision), 'global_iou': float(iou), 'weighted_error_total': float(weighted_error), 'weighted_error_mean': float(weighted_error / max(len(rows), 1))}

def kfeval_load_baseline_summary(path: Path) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {'gt_area': float(row['gt_area']), 'pred_area': float(row['pred_area']), 'intersection': float(row['intersection']), 'union': float(row['union']), 'weighted_error': float(row['weighted_error'])}
            rows.append(entry)
    return kfeval_aggregate_summary(rows)

def kfeval_main() -> None:
    args = kfeval_parse_args()
    union_path = Path(args.input_union_json)
    tracked_sqlite = Path(args.input_tracked_sqlite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_lookup = kfeval_load_union_rows(union_path)
    result_rows: list[dict[str, object]] = []
    conn = sqlite3.connect(str(tracked_sqlite))
    cur = conn.cursor()
    for frame, track_id, polygons_json in cur.execute('SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)'):
        key = (int(frame), str(track_id))
        pred_entry = pred_lookup.get(key)
        if pred_entry is None:
            continue
        gt_polys = fst.parse_polygons(str(polygons_json))
        pred_polys = fst.ellipses_to_polygon_arrays([tuple(map(float, ellipse)) for ellipse in pred_entry['ellipse_params']])
        metrics = fst.compute_exact_metrics_from_polygons(gt_polys, pred_polys)
        metrics['weighted_error'] = float(fst.compute_weighted_error(metrics))
        result_rows.append({'frame': int(frame), 'track_id': str(track_id), 'mode': str(pred_entry.get('mode', '')), 'run_id': int(pred_entry.get('run_id', -1)), 'has_keyframe': int(pred_entry.get('has_keyframe', 0)), 'gt_area': float(metrics['gt_area']), 'pred_area': float(metrics['pred_area']), 'intersection': float(metrics['intersection']), 'union': float(metrics['union']), 'recall': float(metrics['recall']), 'precision': float(metrics['precision']), 'iou': float(metrics['iou']), 'weighted_error': float(metrics['weighted_error']), 'ellipse_params': json.dumps(pred_entry['ellipse_params'], ensure_ascii=False)})
    conn.close()
    result_rows.sort(key=lambda row: (int(row['frame']), int(str(row['track_id']))))
    metrics_csv = output_dir / 'keyframe_exact_metrics.csv'
    with metrics_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'track_id', 'mode', 'run_id', 'has_keyframe', 'gt_area', 'pred_area', 'intersection', 'union', 'recall', 'precision', 'iou', 'weighted_error', 'ellipse_params'])
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)
    summary = {'input_union_json': str(union_path), 'input_tracked_sqlite': str(tracked_sqlite), 'optimized': kfeval_aggregate_summary(result_rows)}
    if args.baseline_metrics_csv:
        baseline_path = Path(args.baseline_metrics_csv)
        summary['baseline'] = kfeval_load_baseline_summary(baseline_path)
        summary['delta_vs_baseline'] = {key: float(summary['optimized'][key] - summary['baseline'][key]) for key in ['global_recall', 'global_precision', 'global_iou', 'weighted_error_total', 'weighted_error_mean']}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')



# ==============================================================================
# Inlined from: keyframe_opt/fill_trackk_union_gaps.py
# ==============================================================================

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import optimize_keyframes_standalone as base

def kffill_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fill missing frames between union rows without crossing track or ellipse-count changes.')
    parser.add_argument('--input-union-json', required=True)
    parser.add_argument('--input-metrics-csv', required=True)
    parser.add_argument('--output-union-json', required=True)
    parser.add_argument('--output-metrics-csv', required=True)
    parser.add_argument('--output-summary-json', required=True)
    parser.add_argument('--max-gap', type=int, default=30)
    return parser.parse_args()

def kffill_load_union_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame'])))
    return rows

def kffill_load_metrics(path: Path) -> tuple[list[str], dict[tuple[int, str], dict[str, str]]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        lookup: dict[tuple[int, str], dict[str, str]] = {}
        for row in reader:
            lookup[int(row['frame']), str(row['track_id'])] = dict(row)
    return (fieldnames, lookup)

def kffill_linear_interp(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * float(a) + alpha * float(b)

def kffill_interpolate_angle_deg(theta0: float, theta1: float, alpha: float) -> float:
    left = float(theta0)
    right = float(theta1)
    candidates = [right - 180.0, right, right + 180.0]
    best = min(candidates, key=lambda value: abs(value - left))
    out = kffill_linear_interp(left, best, alpha)
    return base.canonicalize_ellipse([0.0, 0.0, 1.0, 1.0, out])[4]

def kffill_interpolate_ellipse(left: list[float], right: list[float], alpha: float) -> list[float]:
    cx = kffill_linear_interp(left[0], right[0], alpha)
    cy = kffill_linear_interp(left[1], right[1], alpha)
    log_a = kffill_linear_interp(math.log(max(float(left[2]), 1e-06)), math.log(max(float(right[2]), 1e-06)), alpha)
    log_b = kffill_linear_interp(math.log(max(float(left[3]), 1e-06)), math.log(max(float(right[3]), 1e-06)), alpha)
    theta = kffill_interpolate_angle_deg(float(left[4]), float(right[4]), alpha)
    return base.canonicalize_ellipse([cx, cy, math.exp(log_a), math.exp(log_b), theta])

def kffill_stabilize_right_slots(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if len(left) != 2 or len(right) != 2:
        return right
    keep_cost = base.ellipse_pair_cost(left[0], right[0], 1.0, 0.65, 0.2) + base.ellipse_pair_cost(left[1], right[1], 1.0, 0.65, 0.2)
    swap_cost = base.ellipse_pair_cost(left[0], right[1], 1.0, 0.65, 0.2) + base.ellipse_pair_cost(left[1], right[0], 1.0, 0.65, 0.2)
    if swap_cost < keep_cost:
        return [right[1], right[0]]
    return right

def kffill_choose_fill_mode(left_mode: str, right_mode: str, alpha: float) -> str:
    if left_mode == right_mode:
        return left_mode
    return left_mode if alpha < 0.5 else right_mode

def kffill_interpolate_metric_row(fieldnames: list[str], left_row: dict[str, str], right_row: dict[str, str], frame: int, track_id: str, mode: str, run_id: int, ellipse_params: list[list[float]], alpha: float) -> dict[str, str]:
    out: dict[str, str] = {}
    numeric_fields = {'gt_area', 'pred_area', 'intersection', 'union', 'recall', 'precision', 'iou', 'weighted_error'}
    for field in fieldnames:
        if field == 'frame':
            out[field] = str(int(frame))
        elif field == 'track_id':
            out[field] = str(track_id)
        elif field == 'mode':
            out[field] = str(mode)
        elif field == 'run_id':
            out[field] = str(int(run_id))
        elif field == 'has_keyframe':
            out[field] = '0'
        elif field == 'is_gap_filled':
            out[field] = '1'
        elif field == 'ellipse_params':
            out[field] = json.dumps(ellipse_params, ensure_ascii=False)
        elif field in numeric_fields and field in left_row and (field in right_row):
            out[field] = str(kffill_linear_interp(float(left_row[field]), float(right_row[field]), alpha))
        else:
            out[field] = left_row.get(field, '')
    return out

def kffill_main() -> None:
    args = kffill_parse_args()
    input_union = Path(args.input_union_json)
    input_metrics = Path(args.input_metrics_csv)
    output_union = Path(args.output_union_json)
    output_metrics = Path(args.output_metrics_csv)
    output_summary = Path(args.output_summary_json)
    output_union.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    union_rows = kffill_load_union_rows(input_union)
    fieldnames, metric_lookup = kffill_load_metrics(input_metrics)
    if 'is_gap_filled' not in fieldnames:
        fieldnames.append('is_gap_filled')
    by_track: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in union_rows:
        by_track[str(row['track_id'])].append(row)
    for rows in by_track.values():
        rows.sort(key=lambda row: int(row['frame']))
    filled_union_rows: list[dict[str, object]] = []
    filled_metric_rows: list[dict[str, str]] = []
    gap_count = 0
    filled_frame_count = 0
    for track_id, track_rows in sorted(by_track.items(), key=lambda item: int(item[0])):
        for idx, left in enumerate(track_rows):
            left_frame = int(left['frame'])
            left_mode = str(left.get('mode', ''))
            left_run_id = int(left.get('run_id', -1))
            left_ellipses = [base.canonicalize_ellipse(ellipse) for ellipse in left['ellipse_params']]
            filled_union_rows.append(left)
            metric_row = metric_lookup.get((left_frame, track_id))
            if metric_row is not None:
                metric_row = dict(metric_row)
                metric_row['is_gap_filled'] = '0'
                filled_metric_rows.append(metric_row)
            if idx + 1 >= len(track_rows):
                continue
            right = track_rows[idx + 1]
            right_frame = int(right['frame'])
            right_mode = str(right.get('mode', ''))
            gap = right_frame - left_frame - 1
            if gap <= 0 or gap > int(args.max_gap):
                continue
            if len(left['ellipse_params']) != len(right['ellipse_params']):
                continue
            left_metrics = metric_lookup.get((left_frame, track_id))
            right_metrics = metric_lookup.get((right_frame, track_id))
            if left_metrics is None or right_metrics is None:
                continue
            right_ellipses = [base.canonicalize_ellipse(ellipse) for ellipse in right['ellipse_params']]
            right_ellipses = kffill_stabilize_right_slots(left_ellipses, right_ellipses)
            gap_count += 1
            for missing_frame in range(left_frame + 1, right_frame):
                alpha = (missing_frame - left_frame) / float(right_frame - left_frame)
                mode = kffill_choose_fill_mode(left_mode, right_mode, alpha)
                ellipses = [kffill_interpolate_ellipse(left_ellipses[slot_id], right_ellipses[slot_id], alpha) for slot_id in range(len(left_ellipses))]
                filled_union_rows.append({'track_id': track_id, 'mode': mode, 'run_id': left_run_id, 'frame': int(missing_frame), 'ellipse_params': ellipses, 'has_keyframe': 0})
                filled_metric_rows.append(kffill_interpolate_metric_row(fieldnames=fieldnames, left_row=left_metrics, right_row=right_metrics, frame=missing_frame, track_id=track_id, mode=mode, run_id=left_run_id, ellipse_params=ellipses, alpha=alpha))
                filled_frame_count += 1
    filled_union_rows.sort(key=lambda row: (int(str(row['track_id'])), int(row['frame'])))
    filled_metric_rows.sort(key=lambda row: (int(row['frame']), int(str(row['track_id']))))
    output_union.write_text(json.dumps(filled_union_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    with output_metrics.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in filled_metric_rows:
            writer.writerow(row)
    summary = {'input_union_json': str(input_union), 'input_metrics_csv': str(input_metrics), 'output_union_json': str(output_union), 'output_metrics_csv': str(output_metrics), 'max_gap': int(args.max_gap), 'input_rows': int(len(union_rows)), 'output_rows': int(len(filled_union_rows)), 'filled_gaps': int(gap_count), 'filled_frames': int(filled_frame_count)}
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')



# ==============================================================================
# Inlined from: keyframe_opt/union_json_to_pred_sqlite.py
# ==============================================================================

import argparse
import json
import sqlite3
import sys
from pathlib import Path
union2sqlite_ROOT = Path(__file__).resolve().parents[1]
if str(union2sqlite_ROOT) not in sys.path:
    sys.path.insert(0, str(union2sqlite_ROOT))
import standalone_runtime_fst as fst

def union2sqlite_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert interpolated_union.json into renderer-friendly prediction SQLite.')
    parser.add_argument('--input-union-json', required=True)
    parser.add_argument('--output-sqlite', required=True)
    parser.add_argument('--reference-sqlite', default=None)
    return parser.parse_args()

def union2sqlite_main() -> None:
    args = union2sqlite_parse_args()
    input_union = Path(args.input_union_json)
    output_sqlite = Path(args.output_sqlite)
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(input_union.read_text(encoding='utf-8'))
    sqlite_rows: list[tuple[int, str, str]] = []
    for row in rows:
        polygons_json = None
        ellipse_params = row.get('ellipse_params')
        polygon_points = row.get('polygon')
        if ellipse_params:
            ellipses = [tuple(map(float, ellipse)) for ellipse in ellipse_params]
            polygons_json = fst.make_polygons_json(ellipses)
        elif polygon_points:
            polygon = [[float(point[0]), float(point[1])] for point in polygon_points]
            polygons_json = json.dumps([polygon], ensure_ascii=False, separators=(',', ':'))
        if not polygons_json:
            continue
        sqlite_rows.append((int(row['frame']), str(row['track_id']), polygons_json))
    ref_sqlite = None if not args.reference_sqlite else Path(args.reference_sqlite)
    fst.write_sqlite(sqlite_rows, output_sqlite, reference_sqlite=ref_sqlite)



# ==============================================================================
# Inlined from: render_k1_exact_k2_v5_overlay_video.py
# ==============================================================================

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from bisect import bisect_left
import cv2
import numpy as np
render_ROOT = Path(__file__).resolve().parent
render_TEACHER_ROOT = render_ROOT / 'Teacher'
if str(render_TEACHER_ROOT) not in sys.path:
    sys.path.insert(0, str(render_TEACHER_ROOT))
import final_standalone_t5000 as fst
render_PRED_FALLBACK_COLOR = (0, 215, 255)
render_GT_OUTLINE_COLOR = (255, 255, 255)
render_TEXT_BG = (18, 18, 18)
render_TEXT_FG = (245, 245, 245)
render_MODE_K1_COLOR = (0, 255, 0)
render_MODE_K2_COLOR = (255, 0, 255)

def render_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render overlay video for exact K1 + K2 V5 output.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--gt-sqlite', type=Path, required=True)
    parser.add_argument('--pred-sqlite', type=Path, required=True)
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--k1-cost-csv', type=Path, required=True, help='CSV containing the original K1 weighted_error for each frame/track row.')
    parser.add_argument('--output-video', type=Path, required=True)
    parser.add_argument('--encoder', choices=('nvenc', 'cpu'), default='nvenc')
    parser.add_argument('--mask-alpha', type=float, default=0.28)
    parser.add_argument('--gt-thickness', type=int, default=2)
    parser.add_argument('--pred-thickness', type=int, default=3)
    parser.add_argument('--k1-threshold', type=int, default=5000)
    parser.add_argument('--k1-threshold-edge', type=int, default=5000)
    parser.add_argument('--show-keyframe-progress', action='store_true')
    parser.add_argument('--progress-thickness', type=int, default=4)
    parser.add_argument('--keyframes-json', type=Path, default=None)
    return parser

def render_load_rows(sqlite_path: Path) -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return [(int(frame), str(track_id), str(polygons)) for frame, track_id, polygons in conn.execute('SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id')]
    finally:
        conn.close()

def render_load_metric_data(csv_path: Path) -> tuple[dict[tuple[int, str], dict[str, object]], dict[str, list[int]]]:
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    keyframes_by_track: dict[str, list[int]] = defaultdict(list)
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track_id = str(row['track_id'])
            has_keyframe = int(float(row.get('has_keyframe', 0)))
            key = (frame, track_id)
            weighted_error = float(row.get('weighted_error', 0.0) or 0.0)
            gt_area = float(row.get('gt_area', 0.0) or 0.0)
            weighted_error_norm_raw = row.get('weighted_error_norm')
            weighted_error_norm = (
                float(weighted_error_norm_raw)
                if weighted_error_norm_raw not in (None, '')
                else infer_compute_weighted_error_norm(weighted_error, gt_area)
            )
            lookup[key] = {'mode': str(row.get('mode', '')), 'candidate_name': str(row.get('candidate_name', '')), 'recall': float(row.get('recall', 0.0)), 'precision': float(row.get('precision', 0.0)), 'iou': float(row.get('iou', 0.0)), 'weighted_error': int(weighted_error), 'weighted_error_norm': float(weighted_error_norm), 'gt_area': float(gt_area), 'has_keyframe': has_keyframe}
            if has_keyframe:
                keyframes_by_track[track_id].append(frame)
    for frames in keyframes_by_track.values():
        frames.sort()
    return (lookup, dict(keyframes_by_track))

def render_load_slot_keyframes(path: Path | None) -> dict[tuple[str, int, int], list[int]]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding='utf-8'))
    lookup: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (str(row['track_id']), int(row.get('run_id', -1)), int(row.get('slot_id', 0)))
        lookup[key].append(int(row['frame']))
    for frames in lookup.values():
        frames.sort()
    return dict(lookup)

def render_draw_track_annotation(img: np.ndarray, anchor: tuple[int, int], track_id: str, metric_row: dict[str, object] | None, k1_weighted_error: int | None, k1_threshold: int) -> None:
    if metric_row is None:
        return
    mode = str(metric_row['mode']).upper()
    score_line = f"T{track_id} {mode} IoU:{float(metric_row['iou']):.3f} R:{float(metric_row['recall']):.3f} P:{float(metric_row['precision']):.3f}"
    if k1_weighted_error is None:
        aux_line = f"K1cost:NA/{int(k1_threshold)} n:NA {str(metric_row['candidate_name'])}"
    else:
        gt_area = float(metric_row.get('gt_area', 0.0) or 0.0)
        norm = float(metric_row.get('weighted_error_norm', infer_compute_weighted_error_norm(float(k1_weighted_error), gt_area)))
        norm_threshold = infer_compute_weighted_error_norm(float(k1_threshold), gt_area)
        aux_line = f"K1cost:{int(k1_weighted_error)}/{int(k1_threshold)} n:{norm:.3f}/{norm_threshold:.3f} {str(metric_row['candidate_name'])}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (w1, h1), b1 = cv2.getTextSize(score_line, font, font_scale, thickness)
    (w2, h2), b2 = cv2.getTextSize(aux_line, font, font_scale, thickness)
    text_w = max(w1, w2)
    text_h = h1 + h2 + b1 + b2 + 16
    x, y = anchor
    x = int(np.clip(x, 6, max(6, img.shape[1] - text_w - 12)))
    y = int(np.clip(y, text_h + 4, max(text_h + 4, img.shape[0] - 6)))
    top_left = (x - 4, y - text_h)
    bottom_right = (x + text_w + 8, y + 4)
    mode_color = render_MODE_K2_COLOR if mode == 'K2' else render_MODE_K1_COLOR
    cv2.rectangle(img, top_left, bottom_right, render_TEXT_BG, thickness=-1)
    cv2.rectangle(img, top_left, bottom_right, mode_color, thickness=2)
    cv2.putText(img, score_line, (x, y - h2 - 8), font, font_scale, render_TEXT_FG, thickness, cv2.LINE_AA)
    cv2.putText(img, aux_line, (x, y - 4), font, font_scale, mode_color, thickness, cv2.LINE_AA)

def render_get_pred_color(metric_row: dict[str, object] | None) -> tuple[int, int, int]:
    if metric_row is None:
        return render_PRED_FALLBACK_COLOR
    mode = str(metric_row.get('mode', '')).upper()
    if mode == 'K2':
        return render_MODE_K2_COLOR
    if mode == 'K1':
        return render_MODE_K1_COLOR
    return render_PRED_FALLBACK_COLOR

def render_get_progress_ratio(frame_idx: int, track_id: str, keyframes_by_track: dict[str, list[int]]) -> float | None:
    frames = keyframes_by_track.get(track_id)
    if not frames:
        return None
    pos = bisect_left(frames, frame_idx)
    if pos < len(frames) and frames[pos] == frame_idx:
        return 0.0
    if pos == 0 or pos >= len(frames):
        return None
    prev_frame = frames[pos - 1]
    next_frame = frames[pos]
    span = next_frame - prev_frame
    if span <= 0:
        return None
    return float((frame_idx - prev_frame) / span)

def render_get_slot_progress_ratio(frame_idx: int, track_id: str, run_id: int, slot_id: int, slot_keyframes: dict[tuple[str, int, int], list[int]]) -> float | None:
    frames = slot_keyframes.get((track_id, run_id, slot_id))
    if not frames:
        return None
    pos = bisect_left(frames, frame_idx)
    if pos < len(frames) and frames[pos] == frame_idx:
        return 0.0
    if pos == 0 or pos >= len(frames):
        return None
    prev_frame = frames[pos - 1]
    next_frame = frames[pos]
    span = next_frame - prev_frame
    if span <= 0:
        return None
    return float((frame_idx - prev_frame) / span)

def render_fit_progress_ellipse(polygons_json: str) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    polys = fst.parse_polygons(polygons_json)
    if not polys:
        return None
    points = np.concatenate([poly.astype(np.float32) for poly in polys if len(poly) >= 2], axis=0)
    if len(points) < 5:
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        center = (x + 0.5 * w, y + 0.5 * h)
        axes = (max(1.0, 0.5 * w), max(1.0, 0.5 * h))
        return (center, axes, 0.0)
    fitted = cv2.fitEllipse(points.reshape(-1, 1, 2))
    (cx, cy), (major, minor), angle = fitted
    return ((float(cx), float(cy)), (max(1.0, float(major) * 0.55), max(1.0, float(minor) * 0.55)), float(angle))

def render_fit_progress_ellipse_from_polygon(poly: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    points = poly.astype(np.float32)
    if len(points) < 2:
        return None
    if len(points) < 5:
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        center = (x + 0.5 * w, y + 0.5 * h)
        axes = (max(1.0, 0.5 * w), max(1.0, 0.5 * h))
        return (center, axes, 0.0)
    fitted = cv2.fitEllipse(points.reshape(-1, 1, 2))
    (cx, cy), (major, minor), angle = fitted
    return ((float(cx), float(cy)), (max(1.0, float(major) * 0.55), max(1.0, float(minor) * 0.55)), float(angle))

def render_draw_keyframe_progress(img: np.ndarray, polygons_json: str, ratio: float | None, color: tuple[int, int, int], thickness: int) -> None:
    if ratio is None:
        return
    fitted = render_fit_progress_ellipse(polygons_json)
    if fitted is None:
        return
    center, axes, angle = fitted
    outer_axes = (axes[0] + 8.0, axes[1] + 8.0)
    center_int = (int(round(center[0])), int(round(center[1])))
    axes_int = (max(1, int(round(outer_axes[0]))), max(1, int(round(outer_axes[1]))))
    cv2.ellipse(img, center_int, axes_int, angle, 0, 360, (40, 40, 40), thickness + 1, cv2.LINE_AA)
    end_angle = max(0.0, min(360.0, 360.0 * ratio))
    if end_angle > 0.0:
        cv2.ellipse(img, center_int, axes_int, angle, -90, -90 + end_angle, color, thickness, cv2.LINE_AA)
    else:
        cv2.circle(img, center_int, max(2, thickness), color, thickness=-1, lineType=cv2.LINE_AA)

def render_draw_keyframe_progress_poly(img: np.ndarray, poly: np.ndarray, ratio: float | None, color: tuple[int, int, int], thickness: int) -> None:
    if ratio is None:
        return
    fitted = render_fit_progress_ellipse_from_polygon(poly)
    if fitted is None:
        return
    center, axes, angle = fitted
    outer_axes = (axes[0] + 8.0, axes[1] + 8.0)
    center_int = (int(round(center[0])), int(round(center[1])))
    axes_int = (max(1, int(round(outer_axes[0]))), max(1, int(round(outer_axes[1]))))
    cv2.ellipse(img, center_int, axes_int, angle, 0, 360, (40, 40, 40), thickness + 1, cv2.LINE_AA)
    end_angle = max(0.0, min(360.0, 360.0 * ratio))
    if end_angle > 0.0:
        cv2.ellipse(img, center_int, axes_int, angle, -90, -90 + end_angle, color, thickness, cv2.LINE_AA)
    else:
        cv2.circle(img, center_int, max(2, thickness), color, thickness=-1, lineType=cv2.LINE_AA)

def render_render_overlay_video(video_path: Path, gt_rows: list[tuple[int, str, str]], pred_rows: list[tuple[int, str, str]], metric_lookup: dict[tuple[int, str], dict[str, object]], k1_cost_lookup: dict[tuple[int, str], int], keyframes_by_track: dict[str, list[int]], slot_keyframes: dict[tuple[str, int, int], list[int]], output_video: Path, *, encoder: str, mask_alpha: float, gt_thickness: int, pred_thickness: int, k1_threshold: int, k1_threshold_edge: int, show_keyframe_progress: bool, progress_thickness: int) -> None:
    gt_by_frame: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for frame, track_id, polygons in gt_rows:
        gt_by_frame[frame].append((track_id, polygons))
    pred_by_frame: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for frame, track_id, polygons in pred_rows:
        pred_by_frame[frame].append((track_id, polygons))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists():
        output_video.unlink()
    proc = render_open_video_writer(str(output_video), width, height, fps, encoder=encoder)
    if proc.stdin is None:
        raise RuntimeError('Failed to open ffmpeg stdin.')
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            overlay = frame.copy()
            gt_entries = gt_by_frame.get(frame_idx, [])
            pred_entries = pred_by_frame.get(frame_idx, [])
            gt_by_track = {track_id: polygons_json for track_id, polygons_json in gt_entries}
            if gt_entries:
                gt_mask = np.zeros((height, width), dtype=np.uint8)
                for _track_id, polygons_json in gt_entries:
                    gt_mask |= fst.rasterize_full(polygons_json, height=height, width=width)
                    fst.draw_outlines(overlay, polygons_json, render_GT_OUTLINE_COLOR, gt_thickness)
                fst.blend_mask(overlay, gt_mask, fst.MASK_COLOR, mask_alpha)
                for _track_id, polygons_json in gt_entries:
                    fst.draw_outlines(overlay, polygons_json, render_GT_OUTLINE_COLOR, gt_thickness)
            for track_id, polygons_json in pred_entries:
                metric_row = metric_lookup.get((frame_idx, track_id))
                pred_color = render_get_pred_color(metric_row)
                fst.draw_outlines(overlay, polygons_json, pred_color, pred_thickness)
                if show_keyframe_progress:
                    run_id = int(metric_row.get('run_id', -1)) if metric_row is not None else -1
                    polys = fst.parse_polygons(polygons_json)
                    if slot_keyframes and polys:
                        for slot_id, poly in enumerate(polys):
                            render_draw_keyframe_progress_poly(overlay, poly, render_get_slot_progress_ratio(frame_idx, track_id, run_id, slot_id, slot_keyframes), pred_color, progress_thickness)
                    else:
                        render_draw_keyframe_progress(overlay, polygons_json, render_get_progress_ratio(frame_idx, track_id, keyframes_by_track), pred_color, progress_thickness)
                anchor_polygons = gt_by_track.get(track_id, polygons_json)
                anchor = fst.get_annotation_anchor(anchor_polygons, width=width, height=height)
                gt_track_mask = fst.rasterize_full(anchor_polygons, height=height, width=width)
                edge_touch = bool(np.any(gt_track_mask[0, :]) or np.any(gt_track_mask[-1, :]) or np.any(gt_track_mask[:, 0]) or np.any(gt_track_mask[:, -1]))
                threshold = int(k1_threshold_edge if edge_touch else k1_threshold)
                render_draw_track_annotation(overlay, anchor, track_id, metric_row, k1_cost_lookup.get((frame_idx, track_id)), threshold)
            cv2.putText(overlay, f'F:{frame_idx}', (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            proc.stdin.write(overlay.tobytes())
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f'  rendered {frame_idx}/{total_frames}')
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr is not None else ''
        code = proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        if code != 0:
            raise RuntimeError(f'ffmpeg encode failed with code {code}: {stderr}')

def render_open_video_writer(output_video: str, width: int, height: int, fps: float, *, encoder: str) -> subprocess.Popen:
    if encoder == 'nvenc':
        return fst.open_nvenc_writer(output_video, width, height, fps)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps:.8f}', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', output_video]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

def render_main() -> None:
    args = render_build_parser().parse_args()
    gt_rows = render_load_rows(args.gt_sqlite)
    pred_rows = render_load_rows(args.pred_sqlite)
    metric_lookup, keyframes_by_track = render_load_metric_data(args.metrics_csv)
    k1_cost_lookup = fst.load_k1_cost_lookup(args.k1_cost_csv)
    slot_keyframes = render_load_slot_keyframes(args.keyframes_json)
    render_render_overlay_video(args.video, gt_rows, pred_rows, metric_lookup, k1_cost_lookup, keyframes_by_track, slot_keyframes, args.output_video, encoder=str(args.encoder), mask_alpha=float(args.mask_alpha), gt_thickness=int(args.gt_thickness), pred_thickness=int(args.pred_thickness), k1_threshold=int(args.k1_threshold), k1_threshold_edge=int(args.k1_threshold_edge), show_keyframe_progress=bool(args.show_keyframe_progress), progress_thickness=int(args.progress_thickness))
    summary = {'video': str(args.video), 'gt_sqlite': str(args.gt_sqlite), 'pred_sqlite': str(args.pred_sqlite), 'metrics_csv': str(args.metrics_csv), 'k1_cost_csv': str(args.k1_cost_csv), 'output_video': str(args.output_video), 'row_count': len(pred_rows), 'k1_threshold': int(args.k1_threshold), 'k1_threshold_edge': int(args.k1_threshold_edge)}
    summary_path = args.output_video.with_suffix('.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))



# =============================================================================
# Full pipeline orchestration
# =============================================================================

import argparse
import re


def pipeline_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Production standalone pipeline for shared preprocessing plus ellipse/polygon postprocess branches.'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-sqlite', type=Path)
    input_group.add_argument('--input-jsonl', type=Path)
    parser.add_argument('--input-video', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--reuse-inference-output-dir',
        type=Path,
        default=None,
        help='Reuse an existing inference output directory instead of rerunning inference.',
    )
    parser.add_argument('--intervals', default='3')
    parser.add_argument('--dense-recall-target', type=float, default=0.96)
    parser.add_argument('--polygon-recall-min', type=float, default=None, help='Default polygon exact recall floor. Class policy can override this per class.')
    parser.add_argument('--keyframe-max-gap', type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument('--gap-fill-max-gap', type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument('--dense-recall-max-inflate-log', type=float, default=1.2, help=argparse.SUPPRESS)
    parser.add_argument('--render-overlays', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overlay-encoder', choices=('cpu', 'nvenc'), default='cpu', help=argparse.SUPPRESS)
    parser.add_argument('--embed-original-masks', action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        '--k1-cost-csv',
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--raw-cut-detect', action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument('--raw-cut-method', choices=infer_RAW_CUT_METHODS, default=infer_RAW_CUT_METHOD_DEFAULT, help=argparse.SUPPRESS)
    parser.add_argument('--raw-remove-short-tracks-max-frames', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('--raw-det-score-min', type=float, default=infer_RAW_DET_SCORE_MIN, help=argparse.SUPPRESS)
    parser.add_argument(
        '--default-shape-mode',
        choices=pipeline_VALID_SHAPE_MODES,
        default='ellipse',
        help=(
            'Fallback approximation mode for labels not covered by --class-policy-json. '
            'Production mode/keyframe/recall choices should be provided per class.'
        ),
    )
    parser.add_argument(
        '--class-policy-json',
        type=Path,
        default=None,
        help=(
            'Optional JSON for class-specific approximation policy. Supported per-class keys include '
            'shape_mode/mode, target_interval or target_ratio, dense_recall_target, and '
            'polygon_recall_min/recall_min/target_recall, raw_det_score_min/confidence_min. A top-level "default" object is '
            'treated as fallback policy only.'
        ),
    )
    parser.add_argument('--polygon-border-expand', action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-trigger-px', type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-expand-ratio', type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-min-expand-px', type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-max-expand-px', type=float, default=40.0, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-influence-px', type=float, default=24.0, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-width', type=int, default=fst_WIDTH, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-border-height', type=int, default=fst_HEIGHT, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend', action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-frames', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-max-speed-px', type=float, default=1000.0, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-edge-only', action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-edge-margin-px', type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-edge-confirm-frames', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-extend-motion-frames', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('--k1-recall-target', type=float, default=0.99, help=argparse.SUPPRESS)
    parser.add_argument('--k1-exact-refine-rounds', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--k1-workers', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--k2-run-dir', type=Path, default=ROOT / 'assets/k2_v5')
    parser.add_argument('--k2-device', type=str, default='cuda')
    parser.add_argument('--k2-batch-size', type=int, default=64)
    parser.add_argument('--k2-prep-workers', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--k2-precision', type=str, default='fp32', choices=('fp32', 'fp16'), help=argparse.SUPPRESS)
    parser.add_argument('--k2-forward-mode', type=str, default='full', choices=('states_only', 'full'), help=argparse.SUPPRESS)
    parser.add_argument('--k2-profile-stages', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--k2-cudnn-benchmark', type=str, default='off', choices=('on', 'off'), help=argparse.SUPPRESS)
    parser.add_argument('--k2-tf32', type=str, default='default', choices=('default', 'on', 'off'), help=argparse.SUPPRESS)
    parser.add_argument('--routing-mode', type=str, default='k1n_sequence', help=argparse.SUPPRESS)
    parser.add_argument('--k1-cost-routing', type=str, default='normalized', choices=('raw', 'normalized'), help=argparse.SUPPRESS)
    parser.add_argument('--threshold', type=int, default=5000, help=argparse.SUPPRESS)
    parser.add_argument('--threshold-edge', type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument('--threshold-norm', type=float, default=0.18, help=argparse.SUPPRESS)
    parser.add_argument('--threshold-edge-norm', type=float, default=0.18, help=argparse.SUPPRESS)
    parser.add_argument('--k2-soft-k1-keep-cost-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k2-hyst-enter-norm', type=float, default=0.20, help=argparse.SUPPRESS)
    parser.add_argument('--k2-hyst-enter-edge-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k2-hyst-exit-norm', type=float, default=0.14, help=argparse.SUPPRESS)
    parser.add_argument('--k2-hyst-exit-edge-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-enter-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-exit-norm', type=float, default=0.13, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-strong-enter-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-strong-exit-norm', type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-protect-k2-iou-below', type=float, default=0.65, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-smooth-window', type=int, default=11, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-enter-confirm-frames', type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-exit-confirm-frames', type=int, default=6, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-merge-short-k1-max-len', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-merge-short-k2-max-len', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--k1n-seq-reset-gap', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--k2-dp-merge-short-k2-keep-cost-norm', type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument('--k2-dp-force-k2-cost-norm', type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-num-workers', type=int, default=1, help='Worker count passed to the embedded polygon keyframe optimizer.')
    parser.add_argument(
        '--polygon-adaptive-anchor-counts',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Predict per-frame polygon point counts and fix a track/run-wise minimum anchor count before DP. '
            'Enabled by default; --anchors-per-contour remains only the cap/fallback.'
        ),
    )
    parser.add_argument('--polygon-point-predictor-model-dir', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-predictor-device', type=str, default='cuda', help=argparse.SUPPRESS)
    parser.add_argument('--polygon-predictor-batch-size', type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-adaptive-point-quantile', type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-adaptive-point-offset', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-min-anchors-per-contour', type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-max-run-frames', type=int, default=30000, help=argparse.SUPPRESS)
    parser.add_argument('--polygon-run-overlap-frames', type=int, default=900, help=argparse.SUPPRESS)
    parser.add_argument('--progress-interval-sec', type=float, default=30.0, help=argparse.SUPPRESS)
    parser.add_argument('--python', default=sys.executable, help=argparse.SUPPRESS)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def pipeline_parse_intervals(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(',') if token.strip()]


def pipeline_collect_child_pids(root_pid: int) -> list[int]:
    children_by_parent: dict[int, list[int]] = {}
    for proc_dir in Path('/proc').iterdir():
        if not proc_dir.name.isdigit():
            continue
        stat_path = proc_dir / 'stat'
        try:
            stat_text = stat_path.read_text(encoding='utf-8', errors='ignore')
            tail = stat_text.rsplit(')', 1)[1].strip().split()
            parent_pid = int(tail[1])
            pid = int(proc_dir.name)
        except Exception:
            continue
        children_by_parent.setdefault(parent_pid, []).append(pid)
    stack = [int(root_pid)]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children_by_parent.get(pid, []))
    return sorted(seen)


def pipeline_process_resource_summary(root_pid: int) -> dict[str, float | int]:
    pids = pipeline_collect_child_pids(root_pid)
    rss_kb = 0
    for pid in pids:
        status_path = Path('/proc') / str(pid) / 'status'
        try:
            for line in status_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                if line.startswith('VmRSS:'):
                    rss_kb += int(line.split()[1])
                    break
        except Exception:
            continue
    cpu_percent = 0.0
    if pids:
        try:
            output = subprocess.check_output(
                ['ps', '-o', 'pcpu=', '-p', ','.join(str(pid) for pid in pids)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            cpu_percent = float(sum(float(line.strip() or 0.0) for line in output.splitlines()))
        except Exception:
            cpu_percent = 0.0
    return {
        'pid_count': int(len(pids)),
        'rss_gib': float(rss_kb / (1024.0 * 1024.0)),
        'cpu_percent': float(cpu_percent),
    }


def pipeline_gpu_resource_summary() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(',')]
    if len(parts) < 4:
        return None
    try:
        return {
            'gpu_util_percent': float(parts[0]),
            'gpu_mem_mib': float(parts[1]),
            'gpu_power_w': float(parts[2]),
            'gpu_temp_c': float(parts[3]),
        }
    except ValueError:
        return None


def pipeline_command_label(cmd: list[str]) -> str:
    for token in cmd:
        if str(token).startswith('__onefile_'):
            return str(token).replace('__onefile_', '')
    return Path(str(cmd[0])).name if cmd else 'command'


def pipeline_count_csv_data_rows(csv_path: Path) -> int:
    try:
        with csv_path.open('r', encoding='utf-8', errors='ignore') as f:
            return max(0, sum(1 for _ in f) - 1)
    except FileNotFoundError:
        return 0


def pipeline_run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    progress_interval_sec: float=30.0,
    progress_frame_count: int | None=None,
    progress_unit_count: int | None=None,
    progress_unit_label: str='rows',
) -> dict[str, object]:
    start = time.perf_counter()
    label = pipeline_command_label(cmd)
    print(f'[phase-start] {label}: {" ".join(str(part) for part in cmd)}', flush=True)
    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    process = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    last_report = start
    interval = max(5.0, float(progress_interval_sec))
    while True:
        return_code = process.poll()
        now = time.perf_counter()
        if return_code is not None:
            break
        if now - last_report >= interval:
            elapsed = now - start
            proc_stats = pipeline_process_resource_summary(process.pid)
            gpu_stats = pipeline_gpu_resource_summary()
            fields = [
                f'elapsed={elapsed:.1f}s',
                f'pid_count={proc_stats["pid_count"]}',
                f'cpu={proc_stats["cpu_percent"]:.1f}%',
                f'rss={proc_stats["rss_gib"]:.2f}GiB',
            ]
            if progress_frame_count is not None and int(progress_frame_count) > 0:
                fields.append(f'eff_fps={float(progress_frame_count) / max(elapsed, 1e-6):.2f}')
            if progress_unit_count is not None and int(progress_unit_count) > 0:
                fields.append(f'{progress_unit_label}_per_s={float(progress_unit_count) / max(elapsed, 1e-6):.2f}')
            if gpu_stats is not None:
                fields.extend(
                    [
                        f'gpu={gpu_stats["gpu_util_percent"]:.0f}%',
                        f'vram={gpu_stats["gpu_mem_mib"]:.0f}MiB',
                        f'power={gpu_stats["gpu_power_w"]:.1f}W',
                        f'gtemp={gpu_stats["gpu_temp_c"]:.0f}C',
                    ]
                )
            print(f'[phase-progress] {label}: ' + ' '.join(fields), flush=True)
            last_report = now
        time.sleep(1.0)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    elapsed = time.perf_counter() - start
    done_fields = [f'elapsed={elapsed:.1f}s']
    if progress_frame_count is not None and int(progress_frame_count) > 0:
        done_fields.append(f'eff_fps={float(progress_frame_count) / max(elapsed, 1e-6):.2f}')
    if progress_unit_count is not None and int(progress_unit_count) > 0:
        done_fields.append(f'{progress_unit_label}_per_s={float(progress_unit_count) / max(elapsed, 1e-6):.2f}')
    print(f'[phase-done] {label}: ' + ' '.join(done_fields), flush=True)
    return {
        'cmd': cmd,
        'wall_seconds': float(elapsed),
        'effective_fps': None if progress_frame_count is None else float(progress_frame_count) / max(elapsed, 1e-6),
        f'{progress_unit_label}_per_second': None if progress_unit_count is None else float(progress_unit_count) / max(elapsed, 1e-6),
    }


def pipeline_load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def pipeline_guess_input_video(args: argparse.Namespace) -> Path | None:
    if args.input_video is not None:
        return args.input_video
    if args.input_jsonl is not None:
        for suffix in ('.mp4', '.mov', '.mkv', '.avi'):
            candidate = args.input_jsonl.with_suffix(suffix)
            if candidate.exists():
                return candidate
    return None


pipeline_VALID_SHAPE_MODES = ('ellipse', 'polygon')


def pipeline_normalize_label(label: object) -> str:
    text = str(label).strip()
    return text if text else 'unknown'


def pipeline_parse_class_policy(path: Path | None) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    default_policy: dict[str, object] = {}
    class_policies: dict[str, dict[str, object]] = {}
    if path is None:
        return default_policy, class_policies
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'class policy must be a JSON object: {path}')
    if 'default' in raw and isinstance(raw['default'], dict):
        default_policy = {str(k): v for k, v in raw['default'].items()}
    classes_obj = raw.get('classes')
    if isinstance(classes_obj, dict):
        for label, cfg in classes_obj.items():
            if isinstance(cfg, dict):
                class_policies[pipeline_normalize_label(label)] = {str(k): v for k, v in cfg.items()}
    else:
        for label, cfg in raw.items():
            if label == 'default':
                continue
            if isinstance(cfg, dict):
                class_policies[pipeline_normalize_label(label)] = {str(k): v for k, v in cfg.items()}
    return default_policy, class_policies


def pipeline_load_track_labels(tracked_sqlite: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(tracked_sqlite))
    try:
        cur = conn.cursor()
        track_labels: dict[str, str] = {}
        try:
            for track_id, label in cur.execute('SELECT track_id, label FROM tracks'):
                track_labels[str(track_id)] = pipeline_normalize_label(label)
        except sqlite3.Error:
            pass
        if track_labels:
            return track_labels
        mask_columns = {str(row[1]) for row in cur.execute('PRAGMA table_info(masks)')}
        if 'label' in mask_columns:
            for track_id, label in cur.execute("SELECT track_id, MIN(COALESCE(label, '')) FROM masks GROUP BY track_id"):
                track_labels[str(track_id)] = pipeline_normalize_label(label)
        else:
            for (track_id,) in cur.execute('SELECT DISTINCT track_id FROM masks'):
                track_labels[str(track_id)] = 'unknown'
        return track_labels
    finally:
        conn.close()


def pipeline_build_branch_plan(
    *,
    tracked_sqlite: Path,
    default_shape_mode: str,
    class_policy_json: Path | None,
) -> dict[str, object]:
    def resolve_policy_mode(policy: dict[str, object] | None) -> str | None:
        if not isinstance(policy, dict):
            return None
        raw_value = policy.get('mode', policy.get('shape_mode'))
        if raw_value is None:
            return None
        candidate_mode = str(raw_value).strip().lower()
        if candidate_mode not in pipeline_VALID_SHAPE_MODES:
            raise ValueError(f'invalid shape mode in class policy: {candidate_mode}')
        return candidate_mode

    default_mode = str(default_shape_mode).strip().lower()
    if default_mode not in pipeline_VALID_SHAPE_MODES:
        raise ValueError(f'invalid default shape mode: {default_shape_mode}')
    default_policy, class_policies = pipeline_parse_class_policy(class_policy_json)
    default_policy_mode = resolve_policy_mode(default_policy)
    if default_policy_mode is not None:
        default_mode = default_policy_mode
    track_labels = pipeline_load_track_labels(tracked_sqlite)
    branch_tracks: dict[str, list[str]] = {mode: [] for mode in pipeline_VALID_SHAPE_MODES}
    label_summary: dict[str, dict[str, object]] = {}
    track_entries: list[dict[str, object]] = []
    for track_id in sorted(track_labels.keys(), key=lambda text: int(text)):
        label = pipeline_normalize_label(track_labels[track_id])
        resolved_mode = default_mode
        class_cfg = class_policies.get(label)
        candidate_mode = resolve_policy_mode(class_cfg)
        if candidate_mode is not None:
            resolved_mode = candidate_mode
        branch_tracks[resolved_mode].append(track_id)
        label_entry = label_summary.setdefault(
            label,
            {
                'track_count': 0,
                'mode': resolved_mode,
                'policy': class_cfg if class_cfg is not None else {},
            },
        )
        label_entry['track_count'] = int(label_entry['track_count']) + 1
        track_entries.append({'track_id': str(track_id), 'label': label, 'mode': resolved_mode})
    return {
        'default_mode': default_mode,
        'policy_json': str(class_policy_json) if class_policy_json is not None else None,
        'default_policy': default_policy,
        'class_policies': class_policies,
        'track_count': int(len(track_labels)),
        'label_summary': label_summary,
        'branch_track_counts': {mode: int(len(track_ids)) for mode, track_ids in branch_tracks.items()},
        'branch_tracks': branch_tracks,
        'tracks': track_entries,
    }


def pipeline_slugify_label(text: str) -> str:
    slug = re.sub(r'[^\w.-]+', '_', str(text).strip(), flags=re.UNICODE)
    slug = slug.strip('._-')
    return slug if slug else 'unknown'


def pipeline_build_policy_groups(branch_plan: dict[str, object]) -> list[dict[str, object]]:
    default_policy = dict(branch_plan.get('default_policy', {}))
    class_policies = {
        str(label): dict(cfg)
        for label, cfg in dict(branch_plan.get('class_policies', {})).items()
    }
    groups: list[dict[str, object]] = []
    used_group_ids: set[str] = set()
    for label, label_info in sorted(dict(branch_plan.get('label_summary', {})).items()):
        effective_policy = dict(default_policy)
        effective_policy.update(class_policies.get(str(label), {}))
        track_ids = [
            str(row['track_id'])
            for row in list(branch_plan.get('tracks', []))
            if str(row['label']) == str(label)
        ]
        mode = str(label_info['mode'])
        base_group_id = f'{pipeline_slugify_label(str(label))}_{mode}'
        group_id = base_group_id
        suffix = 2
        while group_id in used_group_ids:
            group_id = f'{base_group_id}_{suffix}'
            suffix += 1
        used_group_ids.add(group_id)
        groups.append(
            {
                'group_id': group_id,
                'label': str(label),
                'mode': mode,
                'track_ids': sorted(track_ids, key=lambda text: int(text)),
                'track_count': int(label_info['track_count']),
                'policy': effective_policy,
            }
        )
    return groups


def pipeline_write_subset_sqlite(input_sqlite: Path, output_sqlite: Path, keep_track_ids: list[str]) -> Path:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        output_sqlite.unlink()
    keep_set = {str(track_id) for track_id in keep_track_ids}
    src = sqlite3.connect(str(input_sqlite))
    dst = sqlite3.connect(str(output_sqlite))
    try:
        src_cur = src.cursor()
        dst_cur = dst.cursor()
        table_names = [str(row[0]) for row in src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if 'masks' not in table_names:
            raise RuntimeError(f'input sqlite does not contain masks table: {input_sqlite}')
        masks_columns = [str(row[1]) for row in src_cur.execute('PRAGMA table_info(masks)')]
        dst_cur.execute(
            'CREATE TABLE masks({})'.format(', '.join(f'"{name}"' for name in masks_columns))
        )
        mask_select = ', '.join(f'"{name}"' for name in masks_columns)
        mask_placeholders = ', '.join('?' for _ in masks_columns)
        mask_rows = src_cur.execute(f'SELECT {mask_select} FROM masks ORDER BY frame, CAST(track_id AS INTEGER)').fetchall()
        filtered_mask_rows = [row for row in mask_rows if str(row[masks_columns.index('track_id')]) in keep_set]
        if filtered_mask_rows:
            dst_cur.executemany(f'INSERT INTO masks({mask_select}) VALUES ({mask_placeholders})', filtered_mask_rows)
        if 'tracks' in table_names:
            track_columns = [str(row[1]) for row in src_cur.execute('PRAGMA table_info(tracks)')]
            dst_cur.execute(
                'CREATE TABLE tracks({})'.format(', '.join(f'"{name}"' for name in track_columns))
            )
            track_select = ', '.join(f'"{name}"' for name in track_columns)
            track_placeholders = ', '.join('?' for _ in track_columns)
            track_rows = src_cur.execute(f'SELECT {track_select} FROM tracks').fetchall()
            filtered_track_rows = [row for row in track_rows if str(row[track_columns.index('track_id')]) in keep_set]
            if filtered_track_rows:
                dst_cur.executemany(f'INSERT INTO tracks({track_select}) VALUES ({track_placeholders})', filtered_track_rows)
        if 'cuts' in table_names:
            cut_columns = [str(row[1]) for row in src_cur.execute('PRAGMA table_info(cuts)')]
            dst_cur.execute(
                'CREATE TABLE cuts({})'.format(', '.join(f'"{name}"' for name in cut_columns))
            )
            cut_select = ', '.join(f'"{name}"' for name in cut_columns)
            cut_placeholders = ', '.join('?' for _ in cut_columns)
            cut_rows = src_cur.execute(f'SELECT {cut_select} FROM cuts').fetchall()
            if cut_rows:
                dst_cur.executemany(f'INSERT INTO cuts({cut_select}) VALUES ({cut_placeholders})', cut_rows)
        dst.commit()
    finally:
        dst.close()
        src.close()
    return output_sqlite


def pipeline_prepare_group_sqlites(tracked_sqlite: Path, policy_groups: list[dict[str, object]], groups_root: Path) -> dict[str, str]:
    group_sqlites: dict[str, str] = {}
    total_tracks = sum(int(group['track_count']) for group in policy_groups)
    for group in policy_groups:
        group_id = str(group['group_id'])
        track_ids = list(group['track_ids'])
        if len(track_ids) == total_tracks:
            group_sqlites[group_id] = str(tracked_sqlite)
            continue
        subset_sqlite = groups_root / group_id / 'preprocess' / f'{group_id}.sqlite'
        pipeline_write_subset_sqlite(tracked_sqlite, subset_sqlite, track_ids)
        group_sqlites[group_id] = str(subset_sqlite)
    return group_sqlites


def pipeline_smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def pipeline_polygon_bbox(polygons: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    valid = [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in polygons if len(poly) >= 3]
    if not valid:
        return None
    pts = np.concatenate(valid, axis=0)
    return (
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    )


def pipeline_bbox_edge_sides(
    bbox: tuple[float, float, float, float] | None,
    *,
    width: int,
    height: int,
    margin_px: float,
) -> set[str]:
    if bbox is None:
        return set()
    x0, y0, x1, y1 = bbox
    margin = max(0.0, float(margin_px))
    sides: set[str] = set()
    if float(x0) <= margin:
        sides.add('left')
    if float(x1) >= float(width - 1) - margin:
        sides.add('right')
    if float(y0) <= margin:
        sides.add('top')
    if float(y1) >= float(height - 1) - margin:
        sides.add('bottom')
    return sides


def pipeline_polygons_json_edge_sides(
    polygons_json: str,
    *,
    width: int,
    height: int,
    margin_px: float,
) -> set[str]:
    try:
        polygons = fst_parse_polygons(str(polygons_json))
    except Exception:
        return set()
    return pipeline_bbox_edge_sides(
        pipeline_polygon_bbox(polygons),
        width=int(width),
        height=int(height),
        margin_px=float(margin_px),
    )


def pipeline_endpoint_edge_confirm_from_jsons(
    polygons_jsons: list[str],
    *,
    width: int,
    height: int,
    margin_px: float,
    confirm_frames: int,
) -> tuple[bool, list[str], str]:
    required = max(1, int(confirm_frames))
    if len(polygons_jsons) < required:
        return False, [], 'missing_confirm_frames'
    touched_sides: set[str] = set()
    for polygons_json in polygons_jsons[:required]:
        sides = pipeline_polygons_json_edge_sides(
            polygons_json,
            width=int(width),
            height=int(height),
            margin_px=float(margin_px),
        )
        touched_sides.update(sides)
    if not touched_sides:
        return False, [], 'not_near_edge'
    return True, sorted(touched_sides), 'ok'


def pipeline_polygon_border_expand_one(
    poly: np.ndarray,
    *,
    width: int,
    height: int,
    trigger_px: float,
    expand_ratio: float,
    min_expand_px: float,
    max_expand_px: float,
    influence_px: float,
) -> tuple[np.ndarray, bool]:
    expanded = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
    if expanded.shape[0] < 3:
        return expanded, False
    x0 = float(np.min(expanded[:, 0]))
    y0 = float(np.min(expanded[:, 1]))
    x1 = float(np.max(expanded[:, 0]))
    y1 = float(np.max(expanded[:, 1]))
    span_x = max(1.0, x1 - x0 + 1.0)
    span_y = max(1.0, y1 - y0 + 1.0)
    influence_x = max(float(influence_px), float(trigger_px) + 1.0)
    influence_y = max(float(influence_px), float(trigger_px) + 1.0)
    changed = False

    if x0 <= trigger_px:
        amount = float(np.clip(span_x * float(expand_ratio), float(min_expand_px), float(max_expand_px)))
        weights = pipeline_smoothstep(((float(trigger_px) + influence_x) - expanded[:, 0]) / influence_x)
        if float(np.max(weights)) > 0.0:
            expanded[:, 0] -= amount * weights
            changed = True

    if x1 >= float(width - 1) - trigger_px:
        amount = float(np.clip(span_x * float(expand_ratio), float(min_expand_px), float(max_expand_px)))
        weights = pipeline_smoothstep((expanded[:, 0] - (float(width - 1) - float(trigger_px) - influence_x)) / influence_x)
        if float(np.max(weights)) > 0.0:
            expanded[:, 0] += amount * weights
            changed = True

    if y0 <= trigger_px:
        amount = float(np.clip(span_y * float(expand_ratio), float(min_expand_px), float(max_expand_px)))
        weights = pipeline_smoothstep(((float(trigger_px) + influence_y) - expanded[:, 1]) / influence_y)
        if float(np.max(weights)) > 0.0:
            expanded[:, 1] -= amount * weights
            changed = True

    if y1 >= float(height - 1) - trigger_px:
        amount = float(np.clip(span_y * float(expand_ratio), float(min_expand_px), float(max_expand_px)))
        weights = pipeline_smoothstep((expanded[:, 1] - (float(height - 1) - float(trigger_px) - influence_y)) / influence_y)
        if float(np.max(weights)) > 0.0:
            expanded[:, 1] += amount * weights
            changed = True

    return expanded, changed


def pipeline_polygon_border_expand_polygons(
    polygons_json: str,
    *,
    width: int,
    height: int,
    trigger_px: float,
    expand_ratio: float,
    min_expand_px: float,
    max_expand_px: float,
    influence_px: float,
) -> tuple[str, bool, dict[str, object]]:
    try:
        polygons = fst_parse_polygons(str(polygons_json))
    except Exception:
        return str(polygons_json), False, {}
    expanded_polygons: list[np.ndarray] = []
    changed_any = False
    before_bbox = pipeline_polygon_bbox(polygons)
    for poly in polygons:
        expanded, changed = pipeline_polygon_border_expand_one(
            poly,
            width=int(width),
            height=int(height),
            trigger_px=float(trigger_px),
            expand_ratio=float(expand_ratio),
            min_expand_px=float(min_expand_px),
            max_expand_px=float(max_expand_px),
            influence_px=float(influence_px),
        )
        expanded_polygons.append(expanded)
        changed_any = bool(changed_any or changed)
    if not changed_any:
        return str(polygons_json), False, {'before_bbox': before_bbox, 'after_bbox': before_bbox}
    after_bbox = pipeline_polygon_bbox(expanded_polygons)
    expanded_json = json.dumps([poly.astype(np.float32).tolist() for poly in expanded_polygons], ensure_ascii=False)
    return expanded_json, True, {'before_bbox': before_bbox, 'after_bbox': after_bbox}


def pipeline_polygon_border_expand_sqlite(args: argparse.Namespace, input_sqlite: Path, output_sqlite: Path, summary_json: Path) -> Path:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists() and summary_json.exists() and not args.force:
        return output_sqlite
    if output_sqlite.exists():
        output_sqlite.unlink()

    src = sqlite3.connect(str(input_sqlite))
    dst = sqlite3.connect(str(output_sqlite))
    total_rows = 0
    changed_rows = 0
    side_counts: dict[str, int] = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    max_outside_px = 0.0
    try:
        src_cur = src.cursor()
        dst_cur = dst.cursor()
        table_names = [str(row[0]) for row in src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if 'masks' not in table_names:
            raise RuntimeError(f'input sqlite does not contain masks table: {input_sqlite}')
        masks_columns = [str(row[1]) for row in src_cur.execute('PRAGMA table_info(masks)')]
        dst_cur.execute('CREATE TABLE masks({})'.format(', '.join(f'"{name}"' for name in masks_columns)))
        mask_select = ', '.join(f'"{name}"' for name in masks_columns)
        mask_placeholders = ', '.join('?' for _ in masks_columns)
        polygons_idx = masks_columns.index('polygons')
        for row in src_cur.execute(f'SELECT {mask_select} FROM masks ORDER BY frame, CAST(track_id AS INTEGER)'):
            total_rows += 1
            mutable = list(row)
            before_json = str(mutable[polygons_idx])
            expanded_json, changed, meta = pipeline_polygon_border_expand_polygons(
                before_json,
                width=int(args.polygon_border_width),
                height=int(args.polygon_border_height),
                trigger_px=float(args.polygon_border_trigger_px),
                expand_ratio=float(args.polygon_border_expand_ratio),
                min_expand_px=float(args.polygon_border_min_expand_px),
                max_expand_px=float(args.polygon_border_max_expand_px),
                influence_px=float(args.polygon_border_influence_px),
            )
            if changed:
                changed_rows += 1
                mutable[polygons_idx] = expanded_json
                before_bbox = meta.get('before_bbox')
                after_bbox = meta.get('after_bbox')
                if before_bbox is not None:
                    x0, y0, x1, y1 = before_bbox
                    if x0 <= float(args.polygon_border_trigger_px):
                        side_counts['left'] += 1
                    if x1 >= float(args.polygon_border_width - 1) - float(args.polygon_border_trigger_px):
                        side_counts['right'] += 1
                    if y0 <= float(args.polygon_border_trigger_px):
                        side_counts['top'] += 1
                    if y1 >= float(args.polygon_border_height - 1) - float(args.polygon_border_trigger_px):
                        side_counts['bottom'] += 1
                if after_bbox is not None:
                    ax0, ay0, ax1, ay1 = after_bbox
                    max_outside_px = max(
                        max_outside_px,
                        max(0.0, -float(ax0)),
                        max(0.0, -float(ay0)),
                        max(0.0, float(ax1) - float(args.polygon_border_width - 1)),
                        max(0.0, float(ay1) - float(args.polygon_border_height - 1)),
                    )
            dst_cur.execute(f'INSERT INTO masks({mask_select}) VALUES ({mask_placeholders})', tuple(mutable))

        for table_name in table_names:
            if table_name == 'masks':
                continue
            cols = [str(row[1]) for row in src_cur.execute(f'PRAGMA table_info("{table_name}")')]
            dst_cur.execute('CREATE TABLE "{}"({})'.format(table_name, ', '.join(f'"{name}"' for name in cols)))
            select_cols = ', '.join(f'"{name}"' for name in cols)
            placeholders = ', '.join('?' for _ in cols)
            rows = src_cur.execute(f'SELECT {select_cols} FROM "{table_name}"').fetchall()
            if rows:
                dst_cur.executemany(f'INSERT INTO "{table_name}"({select_cols}) VALUES ({placeholders})', rows)
        dst.commit()
    finally:
        dst.close()
        src.close()

    summary = {
        'enabled': True,
        'input_sqlite': str(input_sqlite),
        'output_sqlite': str(output_sqlite),
        'width': int(args.polygon_border_width),
        'height': int(args.polygon_border_height),
        'trigger_px': float(args.polygon_border_trigger_px),
        'expand_ratio': float(args.polygon_border_expand_ratio),
        'min_expand_px': float(args.polygon_border_min_expand_px),
        'max_expand_px': float(args.polygon_border_max_expand_px),
        'influence_px': float(args.polygon_border_influence_px),
        'total_rows': int(total_rows),
        'changed_rows': int(changed_rows),
        'changed_ratio': float(changed_rows / max(total_rows, 1)),
        'side_counts': side_counts,
        'max_outside_px': float(max_outside_px),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_sqlite


def pipeline_load_cut_frames(sqlite_path: Path) -> set[int]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table_names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'cuts' not in table_names:
            return set()
        columns = [str(row[1]) for row in conn.execute('PRAGMA table_info(cuts)')]
        frame_col = next((name for name in ('frame', 'frame_index', 'cut_frame') if name in columns), None)
        if frame_col is None:
            return set()
        return {int(row[0]) for row in conn.execute(f'SELECT "{frame_col}" FROM cuts')}
    finally:
        conn.close()


def pipeline_load_mask_polygon_lookup(sqlite_path: Path) -> dict[tuple[str, int], str]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        table_names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'masks' not in table_names:
            return {}
        columns = [str(row[1]) for row in conn.execute('PRAGMA table_info(masks)')]
        if not {'frame', 'track_id', 'polygons'}.issubset(set(columns)):
            return {}
        return {
            (str(track_id), int(frame)): str(polygons_json)
            for frame, track_id, polygons_json in conn.execute('SELECT frame, track_id, polygons FROM masks')
        }
    finally:
        conn.close()


def pipeline_polygons_json_touches_k1_edge(polygons_json: str) -> bool:
    try:
        polygons = fst.parse_polygons(str(polygons_json))
        if not polygons:
            return False
        payload = fst.prepare_local_raster_payload_from_polygons(polygons)
        gt_mask, _origin = fst.rasterize_local_mask_from_payload(payload)
        touches = fst.detect_edge_touches(gt_mask)
        return bool(any(touches.values()))
    except Exception:
        return False


def pipeline_crosses_cut(frame_a: int, frame_b: int, cuts: set[int]) -> bool:
    lo, hi = sorted((int(frame_a), int(frame_b)))
    return any(lo < int(cut) <= hi for cut in cuts)


def pipeline_polygons_compatible(polys_a: list[np.ndarray], polys_b: list[np.ndarray]) -> bool:
    if len(polys_a) != len(polys_b):
        return False
    return all(np.asarray(a).shape == np.asarray(b).shape for a, b in zip(polys_a, polys_b))


def pipeline_max_vertex_speed(polys_a: list[np.ndarray], polys_b: list[np.ndarray], frame_delta: int = 1) -> float:
    if not pipeline_polygons_compatible(polys_a, polys_b):
        return float('inf')
    if not polys_a:
        return float('inf')
    dt = max(1, abs(int(frame_delta)))
    return max(float((np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32), axis=1) / float(dt)).max()) for a, b in zip(polys_a, polys_b))


def pipeline_extrapolate_polygons(polys_a: list[np.ndarray], polys_b: list[np.ndarray], step: int, *, before: bool, frame_delta: int = 1) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    dt = max(1, abs(int(frame_delta)))
    for poly_a, poly_b in zip(polys_a, polys_b):
        a = np.asarray(poly_a, dtype=np.float32)
        b = np.asarray(poly_b, dtype=np.float32)
        velocity = (b - a) / float(dt)
        if before:
            output.append(a - float(step) * velocity)
        else:
            output.append(b + float(step) * velocity)
    return output


def pipeline_linear_fit_slope_and_value(frames: list[int], values: np.ndarray, target_frame: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(frames, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    if len(t) < 2:
        return np.zeros_like(y[0], dtype=np.float32), np.asarray(y[0], dtype=np.float32)
    t_mean = float(np.mean(t))
    y_mean = np.mean(y, axis=0)
    centered = t - t_mean
    denom = float(np.sum(centered * centered))
    if denom <= 1e-6:
        return np.zeros_like(y_mean, dtype=np.float32), np.asarray(y_mean, dtype=np.float32)
    slope = np.sum(centered.reshape((-1,) + (1,) * (y.ndim - 1)) * (y - y_mean), axis=0) / denom
    fitted = y_mean + slope * (float(target_frame) - t_mean)
    return np.asarray(slope, dtype=np.float32), np.asarray(fitted, dtype=np.float32)


def pipeline_fit_extrapolate_polygons(frames: list[int], polys_seq: list[list[np.ndarray]], target_frame: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    if not polys_seq:
        return output
    contour_count = len(polys_seq[0])
    for contour_idx in range(contour_count):
        values = np.stack([np.asarray(polys[contour_idx], dtype=np.float32) for polys in polys_seq], axis=0)
        _slope, fitted = pipeline_linear_fit_slope_and_value(frames, values, target_frame)
        output.append(fitted.astype(np.float32))
    return output


def pipeline_fit_polygon_speed(frames: list[int], polys_seq: list[list[np.ndarray]]) -> float:
    if len(frames) < 2 or not polys_seq:
        return float('inf')
    max_speed = 0.0
    contour_count = len(polys_seq[0])
    for contour_idx in range(contour_count):
        values = np.stack([np.asarray(polys[contour_idx], dtype=np.float32) for polys in polys_seq], axis=0)
        slope, _fitted = pipeline_linear_fit_slope_and_value(frames, values, frames[-1])
        max_speed = max(max_speed, float(np.linalg.norm(slope, axis=1).max()))
    return max_speed


def pipeline_ellipses_compatible(ellipses_a: list[list[float]], ellipses_b: list[list[float]]) -> bool:
    return len(ellipses_a) == len(ellipses_b) and all(len(a) == 5 and len(b) == 5 for a, b in zip(ellipses_a, ellipses_b))


def pipeline_max_ellipse_speed(ellipses_a: list[list[float]], ellipses_b: list[list[float]], frame_delta: int = 1) -> float:
    if not pipeline_ellipses_compatible(ellipses_a, ellipses_b):
        return float('inf')
    dt = max(1, abs(int(frame_delta)))
    max_speed = 0.0
    for ea, eb in zip(ellipses_a, ellipses_b):
        a = np.asarray(ea[:4], dtype=np.float32)
        b = np.asarray(eb[:4], dtype=np.float32)
        max_speed = max(max_speed, float(np.max(np.abs(b - a)) / float(dt)))
    return max_speed


def pipeline_angle_delta_180(to_angle: float, from_angle: float) -> float:
    return float((float(to_angle) - float(from_angle) + 90.0) % 180.0 - 90.0)


def pipeline_extrapolate_ellipses(
    ellipses_a: list[list[float]],
    ellipses_b: list[list[float]],
    step: int,
    *,
    before: bool,
    frame_delta: int = 1,
) -> list[list[float]]:
    dt = max(1, abs(int(frame_delta)))
    output: list[list[float]] = []
    for ea, eb in zip(ellipses_a, ellipses_b):
        left = [float(v) for v in ea]
        right = [float(v) for v in eb]
        delta = [
            (right[0] - left[0]) / float(dt),
            (right[1] - left[1]) / float(dt),
            (right[2] - left[2]) / float(dt),
            (right[3] - left[3]) / float(dt),
            pipeline_angle_delta_180(right[4], left[4]) / float(dt),
        ]
        base = left if before else right
        sign = -1.0 if before else 1.0
        values = [base[idx] + sign * float(step) * delta[idx] for idx in range(5)]
        values[2] = max(1.0, values[2])
        values[3] = max(1.0, values[3])
        values[4] = (values[4] + 90.0) % 180.0 - 90.0
        output.append(values)
    return output


def pipeline_fit_extrapolate_ellipses(frames: list[int], ellipses_seq: list[list[list[float]]], target_frame: int) -> list[list[float]]:
    if not ellipses_seq:
        return []
    output: list[list[float]] = []
    slot_count = len(ellipses_seq[0])
    for slot_idx in range(slot_count):
        linear_values = np.asarray([[float(ellipses[slot_idx][param_idx]) for param_idx in range(4)] for ellipses in ellipses_seq], dtype=np.float32)
        _linear_slope, fitted_linear = pipeline_linear_fit_slope_and_value(frames, linear_values, target_frame)
        angle_ref = float(ellipses_seq[0][slot_idx][4])
        angle_values = np.asarray([pipeline_angle_delta_180(float(ellipses[slot_idx][4]), angle_ref) for ellipses in ellipses_seq], dtype=np.float32)
        _angle_slope, fitted_angle_offset = pipeline_linear_fit_slope_and_value(frames, angle_values, target_frame)
        values = [float(fitted_linear[0]), float(fitted_linear[1]), max(1.0, float(fitted_linear[2])), max(1.0, float(fitted_linear[3])), float(angle_ref + fitted_angle_offset)]
        values[4] = (values[4] + 90.0) % 180.0 - 90.0
        output.append(values)
    return output


def pipeline_fit_ellipse_speed(frames: list[int], ellipses_seq: list[list[list[float]]]) -> float:
    if len(frames) < 2 or not ellipses_seq:
        return float('inf')
    max_speed = 0.0
    slot_count = len(ellipses_seq[0])
    for slot_idx in range(slot_count):
        linear_values = np.asarray([[float(ellipses[slot_idx][param_idx]) for param_idx in range(4)] for ellipses in ellipses_seq], dtype=np.float32)
        slope, _fitted = pipeline_linear_fit_slope_and_value(frames, linear_values, frames[-1])
        max_speed = max(max_speed, float(np.max(np.abs(slope))))
    return max_speed


PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN = 'is_endpoint_extrapolated'


def pipeline_load_endpoint_extrapolated_flags(sqlite_path: Path) -> dict[tuple[int, str], int]:
    flags: dict[tuple[int, str], int] = {}
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='masks'").fetchone() is None:
            return flags
        columns = {str(row['name']) for row in conn.execute('PRAGMA table_info(masks)').fetchall()}
        if not {'frame', 'track_id', PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}.issubset(columns):
            return flags
        sql = f'SELECT frame, track_id, "{PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}" AS flag FROM masks WHERE COALESCE("{PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}", 0) != 0'
        for row in conn.execute(sql).fetchall():
            flags[(int(row['frame']), str(row['track_id']))] = int(row['flag'] or 0)
    finally:
        conn.close()
    return flags


def pipeline_apply_endpoint_extrapolated_flags(sqlite_path: Path, flags: dict[tuple[int, str], int]) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='masks'").fetchone() is None:
            return
        columns = {str(row[1]) for row in cur.execute('PRAGMA table_info(masks)').fetchall()}
        if PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN not in columns:
            cur.execute(f'ALTER TABLE masks ADD COLUMN "{PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}" INTEGER NOT NULL DEFAULT 0')
        if flags:
            cur.executemany(
                f'UPDATE masks SET "{PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}" = ? WHERE frame = ? AND track_id = ?',
                [(int(flag), int(frame), str(track_id)) for (frame, track_id), flag in sorted(flags.items())],
            )
        conn.commit()
    finally:
        conn.close()


def pipeline_copy_sqlite_tables(src: sqlite3.Connection, dst: sqlite3.Connection, *, skip_tables: set[str] | None=None) -> None:
    skip = set() if skip_tables is None else set(skip_tables)
    table_names = [str(row[0]) for row in src.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table_name in table_names:
        if table_name in skip:
            continue
        create_row = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
        if create_row is None or not create_row[0]:
            continue
        dst.execute(str(create_row[0]))
        columns = [str(row[1]) for row in src.execute(f'PRAGMA table_info("{table_name}")')]
        if not columns:
            continue
        cols = ', '.join(f'"{name}"' for name in columns)
        placeholders = ', '.join('?' for _ in columns)
        rows = src.execute(f'SELECT {cols} FROM "{table_name}"').fetchall()
        if rows:
            dst.executemany(f'INSERT INTO "{table_name}"({cols}) VALUES ({placeholders})', rows)


def pipeline_video_frame_count(video_path: Path | None) -> int | None:
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None
    finally:
        cap.release()


def pipeline_endpoint_extend_prediction_sqlite(
    args: argparse.Namespace,
    input_sqlite: Path,
    output_sqlite: Path,
    summary_json: Path,
    *,
    cuts_source_sqlite: Path,
    video_path: Path | None,
) -> Path:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists() and summary_json.exists() and not args.force:
        return output_sqlite
    if output_sqlite.exists():
        output_sqlite.unlink()

    steps = max(0, int(args.endpoint_extend_frames))
    motion_window = max(2, int(args.endpoint_extend_motion_frames))
    max_speed_px = float(args.endpoint_extend_max_speed_px)
    video_frames = pipeline_video_frame_count(video_path)
    cuts = pipeline_load_cut_frames(cuts_source_sqlite)

    src = sqlite3.connect(str(input_sqlite))
    dst = sqlite3.connect(str(output_sqlite))
    inserted_rows: list[tuple[object, ...]] = []
    events: list[dict[str, object]] = []
    skipped: dict[str, int] = defaultdict(int)
    try:
        src_cur = src.cursor()
        dst_cur = dst.cursor()
        table_names = [str(row[0]) for row in src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if 'masks' not in table_names:
            raise RuntimeError(f'input sqlite does not contain masks table: {input_sqlite}')
        mask_columns = [str(row[1]) for row in src_cur.execute('PRAGMA table_info(masks)')]
        has_endpoint_flag = PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN in mask_columns
        endpoint_flag_idx = mask_columns.index(PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN) if has_endpoint_flag else -1
        output_mask_columns = list(mask_columns)
        if not has_endpoint_flag:
            output_mask_columns.append(PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN)
        dst_cur.execute('CREATE TABLE masks({})'.format(', '.join(f'"{name}"' for name in output_mask_columns)))
        select_cols = ', '.join(f'"{name}"' for name in mask_columns)
        insert_cols = ', '.join(f'"{name}"' for name in output_mask_columns)
        placeholders = ', '.join('?' for _ in output_mask_columns)
        source_rows = src_cur.execute(f'SELECT {select_cols} FROM masks ORDER BY CAST(track_id AS INTEGER), frame').fetchall()
        if source_rows:
            source_insert_rows: list[tuple[object, ...]] = []
            for row in source_rows:
                mutable = list(row)
                if has_endpoint_flag:
                    mutable[endpoint_flag_idx] = int(mutable[endpoint_flag_idx] or 0)
                else:
                    mutable.append(0)
                source_insert_rows.append(tuple(mutable))
            dst_cur.executemany(f'INSERT INTO masks({insert_cols}) VALUES ({placeholders})', source_insert_rows)

        frame_idx = mask_columns.index('frame')
        track_idx = mask_columns.index('track_id')
        polygons_idx = mask_columns.index('polygons')
        existing = {(str(row[track_idx]), int(row[frame_idx])) for row in source_rows}
        by_track: dict[str, list[tuple[object, ...]]] = defaultdict(list)
        for row in source_rows:
            by_track[str(row[track_idx])].append(row)

        def collect_polygon_motion_rows(rows_local: list[tuple[object, ...]], *, before: bool) -> list[tuple[int, list[np.ndarray]]]:
            candidates = rows_local[:motion_window] if before else list(reversed(rows_local[-motion_window:]))
            collected: list[tuple[int, list[np.ndarray]]] = []
            base_polys: list[np.ndarray] | None = None
            for row in candidates:
                polys = fst_parse_polygons(str(row[polygons_idx]))
                if base_polys is None:
                    base_polys = polys
                elif not pipeline_polygons_compatible(base_polys, polys):
                    break
                collected.append((int(row[frame_idx]), polys))
            collected.sort(key=lambda item: item[0])
            return collected

        for track_id, track_rows in by_track.items():
            rows_sorted = sorted(track_rows, key=lambda row: int(row[frame_idx]))
            if len(rows_sorted) < 2 or steps <= 0:
                continue

            first, second = rows_sorted[0], rows_sorted[1]
            before_edge_ok = True
            before_edge_sides: list[str] = []
            if bool(args.endpoint_extend_edge_only):
                edge_ok, before_edge_sides, reason = pipeline_endpoint_edge_confirm_from_jsons(
                    [str(row[polygons_idx]) for row in rows_sorted[:max(1, int(args.endpoint_extend_edge_confirm_frames))]],
                    width=int(args.polygon_border_width),
                    height=int(args.polygon_border_height),
                    margin_px=float(args.endpoint_extend_edge_margin_px),
                    confirm_frames=int(args.endpoint_extend_edge_confirm_frames),
                )
                if not edge_ok:
                    skipped[f'before_edge_confirm_{reason}'] += 1
                    before_edge_ok = False
            if before_edge_ok:
                before_motion = collect_polygon_motion_rows(rows_sorted, before=True)
                before_frames = [frame for frame, _polys in before_motion]
                before_polys_seq = [polys for _frame, polys in before_motion]
                first_speed = pipeline_fit_polygon_speed(before_frames, before_polys_seq)
                if len(before_motion) >= 2 and first_speed <= max_speed_px:
                    inserted_count = 0
                    for step in range(1, steps + 1):
                        target_frame = int(first[frame_idx]) - step
                        if target_frame < 0:
                            skipped['before_out_of_video'] += 1
                            continue
                        if video_frames is not None and target_frame >= video_frames:
                            skipped['before_out_of_video'] += 1
                            continue
                        if (track_id, target_frame) in existing:
                            skipped['before_existing'] += 1
                            continue
                        if pipeline_crosses_cut(target_frame, int(first[frame_idx]), cuts):
                            skipped['before_cut'] += 1
                            continue
                        extrapolated = pipeline_fit_extrapolate_polygons(before_frames, before_polys_seq, target_frame)
                        mutable = list(first)
                        mutable[frame_idx] = int(target_frame)
                        mutable[polygons_idx] = json.dumps([poly.astype(np.float32).tolist() for poly in extrapolated], ensure_ascii=False)
                        if has_endpoint_flag:
                            mutable[endpoint_flag_idx] = 1
                        else:
                            mutable.append(1)
                        inserted_rows.append(tuple(mutable))
                        existing.add((track_id, target_frame))
                        inserted_count += 1
                    if inserted_count:
                        events.append({'track_id': track_id, 'side': 'before', 'source_frames': before_frames, 'motion_frame_count': len(before_frames), 'inserted': int(inserted_count), 'max_vertex_speed': float(first_speed), 'edge_sides': before_edge_sides})
                else:
                    skipped['before_incompatible_or_too_fast'] += 1

            prev_last, last = rows_sorted[-2], rows_sorted[-1]
            after_edge_ok = True
            after_edge_sides: list[str] = []
            if bool(args.endpoint_extend_edge_only):
                edge_ok, after_edge_sides, reason = pipeline_endpoint_edge_confirm_from_jsons(
                    [str(row[polygons_idx]) for row in rows_sorted[-max(1, int(args.endpoint_extend_edge_confirm_frames)):]],
                    width=int(args.polygon_border_width),
                    height=int(args.polygon_border_height),
                    margin_px=float(args.endpoint_extend_edge_margin_px),
                    confirm_frames=int(args.endpoint_extend_edge_confirm_frames),
                )
                if not edge_ok:
                    skipped[f'after_edge_confirm_{reason}'] += 1
                    after_edge_ok = False
            if after_edge_ok:
                after_motion = collect_polygon_motion_rows(rows_sorted, before=False)
                after_frames = [frame for frame, _polys in after_motion]
                after_polys_seq = [polys for _frame, polys in after_motion]
                last_speed = pipeline_fit_polygon_speed(after_frames, after_polys_seq)
                if len(after_motion) >= 2 and last_speed <= max_speed_px:
                    inserted_count = 0
                    for step in range(1, steps + 1):
                        target_frame = int(last[frame_idx]) + step
                        if target_frame < 0:
                            skipped['after_out_of_video'] += 1
                            continue
                        if video_frames is not None and target_frame >= video_frames:
                            skipped['after_out_of_video'] += 1
                            continue
                        if (track_id, target_frame) in existing:
                            skipped['after_existing'] += 1
                            continue
                        if pipeline_crosses_cut(int(last[frame_idx]), target_frame, cuts):
                            skipped['after_cut'] += 1
                            continue
                        extrapolated = pipeline_fit_extrapolate_polygons(after_frames, after_polys_seq, target_frame)
                        mutable = list(last)
                        mutable[frame_idx] = int(target_frame)
                        mutable[polygons_idx] = json.dumps([poly.astype(np.float32).tolist() for poly in extrapolated], ensure_ascii=False)
                        if has_endpoint_flag:
                            mutable[endpoint_flag_idx] = 1
                        else:
                            mutable.append(1)
                        inserted_rows.append(tuple(mutable))
                        existing.add((track_id, target_frame))
                        inserted_count += 1
                    if inserted_count:
                        events.append({'track_id': track_id, 'side': 'after', 'source_frames': after_frames, 'motion_frame_count': len(after_frames), 'inserted': int(inserted_count), 'max_vertex_speed': float(last_speed), 'edge_sides': after_edge_sides})
                else:
                    skipped['after_incompatible_or_too_fast'] += 1

        if inserted_rows:
            dst_cur.executemany(f'INSERT INTO masks({insert_cols}) VALUES ({placeholders})', inserted_rows)

        for table_name in table_names:
            if table_name == 'masks':
                continue
            columns = [str(row[1]) for row in src_cur.execute(f'PRAGMA table_info("{table_name}")')]
            dst_cur.execute('CREATE TABLE "{}"({})'.format(table_name, ', '.join(f'"{name}"' for name in columns)))
            cols = ', '.join(f'"{name}"' for name in columns)
            ph = ', '.join('?' for _ in columns)
            rows = src_cur.execute(f'SELECT {cols} FROM "{table_name}"').fetchall()
            if rows:
                dst_cur.executemany(f'INSERT INTO "{table_name}"({cols}) VALUES ({ph})', rows)
        dst.commit()
    finally:
        dst.close()
        src.close()

    before_count = sum(int(event['inserted']) for event in events if str(event['side']) == 'before')
    after_count = sum(int(event['inserted']) for event in events if str(event['side']) == 'after')
    summary = {
        'enabled': True,
        'input_sqlite': str(input_sqlite),
        'output_sqlite': str(output_sqlite),
        'cuts_source_sqlite': str(cuts_source_sqlite),
        'video': None if video_path is None else str(video_path),
        'video_frame_count': video_frames,
        'extend_frames': int(steps),
        'motion_frames': int(motion_window),
        'max_speed_px': float(max_speed_px),
        'edge_only': bool(args.endpoint_extend_edge_only),
        'edge_margin_px': float(args.endpoint_extend_edge_margin_px),
        'edge_confirm_frames': int(args.endpoint_extend_edge_confirm_frames),
        'edge_confirm_policy': 'any_frame_near_edge' if bool(args.endpoint_extend_edge_only) else 'disabled',
        'source_rows': int(len(source_rows)),
        'inserted_rows': int(len(inserted_rows)),
        'inserted_before': int(before_count),
        'inserted_after': int(after_count),
        'event_count': int(len(events)),
        'skipped': {str(key): int(value) for key, value in skipped.items()},
        'events': events,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_sqlite


def pipeline_endpoint_extend_ellipse_metrics(
    args: argparse.Namespace,
    input_metrics_csv: Path,
    output_metrics_csv: Path,
    summary_json: Path,
    *,
    cuts_source_sqlite: Path,
    video_path: Path | None,
    eval_sqlite: Path,
    output_eval_sqlite: Path,
) -> tuple[Path, Path]:
    output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    output_eval_sqlite.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if output_metrics_csv.exists() and output_eval_sqlite.exists() and summary_json.exists() and not args.force:
        return output_metrics_csv, output_eval_sqlite
    if output_eval_sqlite.exists():
        output_eval_sqlite.unlink()

    steps = max(0, int(args.endpoint_extend_frames))
    motion_window = max(2, int(args.endpoint_extend_motion_frames))
    max_speed_px = float(args.endpoint_extend_max_speed_px)
    video_frames = pipeline_video_frame_count(video_path)
    cuts = pipeline_load_cut_frames(cuts_source_sqlite)
    eval_polygon_lookup = pipeline_load_mask_polygon_lookup(eval_sqlite) if bool(args.endpoint_extend_edge_only) else {}

    with input_metrics_csv.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if 'frame' not in fieldnames or 'track_id' not in fieldnames or 'ellipse_params' not in fieldnames:
        raise RuntimeError(f'ellipse metrics csv is missing required columns: {input_metrics_csv}')

    existing = {(str(row['track_id']), int(row['frame'])) for row in rows}
    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_track[str(row['track_id'])].append(row)

    inserted_rows: list[dict[str, str]] = []
    inserted_eval: list[tuple[str, int, str, list[list[float]]]] = []
    events: list[dict[str, object]] = []
    skipped: dict[str, int] = defaultdict(int)

    def collect_ellipse_motion_rows(rows_local: list[dict[str, str]], *, before: bool) -> list[tuple[int, list[list[float]]]]:
        candidates = rows_local[:motion_window] if before else list(reversed(rows_local[-motion_window:]))
        collected: list[tuple[int, list[list[float]]]] = []
        base_ellipses: list[list[float]] | None = None
        for row in candidates:
            ellipses = json.loads(row['ellipse_params'])
            if base_ellipses is None:
                base_ellipses = ellipses
            elif not pipeline_ellipses_compatible(base_ellipses, ellipses):
                break
            collected.append((int(row['frame']), ellipses))
        collected.sort(key=lambda item: item[0])
        return collected

    def try_insert(track_id: str, endpoint: dict[str, str], *, before: bool, confirm_rows: list[dict[str, str]], motion_rows: list[dict[str, str]]) -> None:
        endpoint_frame = int(endpoint['frame'])
        motion = collect_ellipse_motion_rows(motion_rows, before=before)
        motion_frames = [frame for frame, _ellipses in motion]
        motion_ellipses_seq = [ellipses for _frame, ellipses in motion]
        speed = pipeline_fit_ellipse_speed(motion_frames, motion_ellipses_seq)
        side = 'before' if before else 'after'
        edge_sides: list[str] = []
        if bool(args.endpoint_extend_edge_only):
            polygons_jsons: list[str] = []
            missing_mask = False
            for confirm_row in confirm_rows[:max(1, int(args.endpoint_extend_edge_confirm_frames))]:
                polygons_json = eval_polygon_lookup.get((track_id, int(confirm_row['frame'])))
                if polygons_json is None:
                    missing_mask = True
                    break
                polygons_jsons.append(str(polygons_json))
            if missing_mask:
                skipped[f'{side}_edge_confirm_missing_mask'] += 1
                return
            edge_ok, edge_sides, reason = pipeline_endpoint_edge_confirm_from_jsons(
                polygons_jsons,
                width=int(args.polygon_border_width),
                height=int(args.polygon_border_height),
                margin_px=float(args.endpoint_extend_edge_margin_px),
                confirm_frames=int(args.endpoint_extend_edge_confirm_frames),
            )
            if not edge_ok:
                skipped[f'{side}_edge_confirm_{reason}'] += 1
                return
        if len(motion) < 2 or speed > max_speed_px:
            skipped[f'{side}_incompatible_or_too_fast'] += 1
            return
        inserted_count = 0
        for step in range(1, steps + 1):
            target_frame = endpoint_frame - step if before else endpoint_frame + step
            if target_frame < 0:
                skipped[f'{side}_out_of_video'] += 1
                continue
            if video_frames is not None and target_frame >= video_frames:
                skipped[f'{side}_out_of_video'] += 1
                continue
            if (track_id, target_frame) in existing:
                skipped[f'{side}_existing'] += 1
                continue
            if pipeline_crosses_cut(target_frame, endpoint_frame, cuts):
                skipped[f'{side}_cut'] += 1
                continue
            extrapolated = pipeline_fit_extrapolate_ellipses(motion_frames, motion_ellipses_seq, target_frame)
            new_row = dict(endpoint)
            new_row['frame'] = str(int(target_frame))
            new_row['ellipse_params'] = json.dumps(extrapolated, ensure_ascii=False, separators=(',', ':'))
            if 'candidate_name' in new_row:
                new_row['candidate_name'] = 'endpoint_extrapolated'
            inserted_rows.append(new_row)
            inserted_eval.append((track_id, int(target_frame), side, extrapolated))
            existing.add((track_id, target_frame))
            inserted_count += 1
        if inserted_count:
            events.append({'track_id': track_id, 'side': side, 'source_frames': motion_frames, 'motion_frame_count': len(motion_frames), 'inserted': int(inserted_count), 'max_ellipse_speed': float(speed), 'edge_sides': edge_sides})

    for track_id, track_rows in by_track.items():
        rows_sorted = sorted(track_rows, key=lambda row: int(row['frame']))
        if len(rows_sorted) < 2 or steps <= 0:
            continue
        confirm_count = max(1, int(args.endpoint_extend_edge_confirm_frames))
        try_insert(track_id, rows_sorted[0], before=True, confirm_rows=rows_sorted[:confirm_count], motion_rows=rows_sorted[:motion_window])
        try_insert(track_id, rows_sorted[-1], before=False, confirm_rows=rows_sorted[-confirm_count:], motion_rows=rows_sorted[-motion_window:])

    all_rows = rows + inserted_rows
    all_rows.sort(key=lambda row: (int(row['frame']), int(str(row['track_id']))))
    with output_metrics_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    src = sqlite3.connect(str(eval_sqlite))
    dst = sqlite3.connect(str(output_eval_sqlite))
    try:
        table_names = {str(row[0]) for row in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'masks' not in table_names:
            raise RuntimeError(f'eval sqlite does not contain masks table: {eval_sqlite}')
        mask_columns = [str(row[1]) for row in src.execute('PRAGMA table_info(masks)')]
        has_endpoint_flag = PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN in mask_columns
        endpoint_flag_idx = mask_columns.index(PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN) if has_endpoint_flag else -1
        output_mask_columns = list(mask_columns)
        if not has_endpoint_flag:
            output_mask_columns.append(PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN)
        create_row = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='masks'").fetchone()
        if create_row is None or not create_row[0]:
            dst.execute('CREATE TABLE masks({})'.format(', '.join(f'"{name}"' for name in mask_columns)))
        else:
            dst.execute(str(create_row[0]))
        if not has_endpoint_flag:
            dst.execute(f'ALTER TABLE masks ADD COLUMN "{PIPELINE_ENDPOINT_EXTRAPOLATED_COLUMN}" INTEGER NOT NULL DEFAULT 0')
        select_cols = ', '.join(f'"{name}"' for name in mask_columns)
        insert_cols = ', '.join(f'"{name}"' for name in output_mask_columns)
        placeholders = ', '.join('?' for _ in output_mask_columns)
        source_rows = src.execute(f'SELECT {select_cols} FROM masks').fetchall()
        if source_rows:
            source_insert_rows: list[tuple[object, ...]] = []
            for row in source_rows:
                mutable = list(row)
                if has_endpoint_flag:
                    mutable[endpoint_flag_idx] = int(mutable[endpoint_flag_idx] or 0)
                else:
                    mutable.append(0)
                source_insert_rows.append(tuple(mutable))
            dst.executemany(f'INSERT INTO masks({insert_cols}) VALUES ({placeholders})', source_insert_rows)
        frame_idx = mask_columns.index('frame')
        track_idx = mask_columns.index('track_id')
        polygons_idx = mask_columns.index('polygons')
        endpoint_row_lookup: dict[tuple[str, str], tuple[object, ...]] = {}
        for row in source_rows:
            endpoint_row_lookup[(str(row[track_idx]), str(row[frame_idx]))] = row
        eval_insert_rows: list[tuple[object, ...]] = []
        for track_id, target_frame, side, ellipses in inserted_eval:
            endpoint_rows = sorted(by_track.get(track_id, []), key=lambda row: int(row['frame']))
            if not endpoint_rows:
                continue
            source_metric_row = endpoint_rows[0] if side == 'before' else endpoint_rows[-1]
            template = endpoint_row_lookup.get((track_id, str(source_metric_row['frame'])))
            if template is None:
                continue
            mutable = list(template)
            mutable[frame_idx] = int(target_frame)
            mutable[polygons_idx] = fst.make_polygons_json([tuple(map(float, ellipse)) for ellipse in ellipses])
            if has_endpoint_flag:
                mutable[endpoint_flag_idx] = 1
            else:
                mutable.append(1)
            eval_insert_rows.append(tuple(mutable))
        if eval_insert_rows:
            dst.executemany(f'INSERT INTO masks({insert_cols}) VALUES ({placeholders})', eval_insert_rows)
        pipeline_copy_sqlite_tables(src, dst, skip_tables={'masks'})
        dst.commit()
    finally:
        dst.close()
        src.close()

    before_count = sum(int(event['inserted']) for event in events if str(event['side']) == 'before')
    after_count = sum(int(event['inserted']) for event in events if str(event['side']) == 'after')
    summary = {
        'enabled': True,
        'stage': 'post_ellipse_fit_pre_keyframe',
        'input_metrics_csv': str(input_metrics_csv),
        'output_metrics_csv': str(output_metrics_csv),
        'input_eval_sqlite': str(eval_sqlite),
        'output_eval_sqlite': str(output_eval_sqlite),
        'cuts_source_sqlite': str(cuts_source_sqlite),
        'video': None if video_path is None else str(video_path),
        'video_frame_count': video_frames,
        'extend_frames': int(steps),
        'motion_frames': int(motion_window),
        'max_speed_px': float(max_speed_px),
        'edge_only': bool(args.endpoint_extend_edge_only),
        'edge_margin_px': float(args.endpoint_extend_edge_margin_px),
        'edge_confirm_frames': int(args.endpoint_extend_edge_confirm_frames),
        'edge_confirm_policy': 'any_frame_near_edge' if bool(args.endpoint_extend_edge_only) else 'disabled',
        'source_rows': int(len(rows)),
        'inserted_rows': int(len(inserted_rows)),
        'inserted_before': int(before_count),
        'inserted_after': int(after_count),
        'event_count': int(len(events)),
        'skipped': {str(key): int(value) for key, value in skipped.items()},
        'events': events,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_metrics_csv, output_eval_sqlite


def pipeline_write_subset_metrics_csv(input_csv: Path, output_csv: Path, keep_track_ids: list[str]) -> Path:
    keep_set = {str(track_id) for track_id in keep_track_ids}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open('r', newline='', encoding='utf-8') as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if str(row.get('track_id', '')) in keep_set]
    with output_csv.open('w', newline='', encoding='utf-8') as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def pipeline_tracked_sqlite_path(args: argparse.Namespace, inference_output_dir: Path) -> Path:
    if args.input_sqlite is not None:
        return args.input_sqlite
    assert args.input_jsonl is not None
    return inference_output_dir / 'preprocess' / f'{args.input_jsonl.stem}.tracked.sqlite'


def pipeline_build_fallback_k1_cost_csv(metrics_csv: Path, output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open('r', encoding='utf-8', newline='') as src, output_csv.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=['frame', 'track_id', 'weighted_error', 'gt_area', 'weighted_error_norm'])
        writer.writeheader()
        for row in reader:
            weighted_error = int(float(row.get('weighted_error', 0.0)))
            gt_area = float(row.get('gt_area', 0.0) or 0.0)
            weighted_error_norm = row.get('weighted_error_norm')
            if weighted_error_norm in (None, ''):
                weighted_error_norm = infer_compute_weighted_error_norm(weighted_error, gt_area)
            writer.writerow(
                {
                    'frame': int(row['frame']),
                    'track_id': str(row['track_id']),
                    'weighted_error': weighted_error,
                    'gt_area': int(gt_area),
                    'weighted_error_norm': float(weighted_error_norm),
                }
            )
    return output_csv


def pipeline_load_k1_cost_detail_lookup(csv_path: Path) -> dict[tuple[int, str], dict[str, float]]:
    lookup: dict[tuple[int, str], dict[str, float]] = {}
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['frame']), str(row['track_id']))
            weighted_error = float(row.get('weighted_error', 0.0) or 0.0)
            gt_area = float(row.get('gt_area', 0.0) or 0.0)
            weighted_error_norm_raw = row.get('weighted_error_norm')
            weighted_error_norm = (
                float(weighted_error_norm_raw)
                if weighted_error_norm_raw not in (None, '')
                else infer_compute_weighted_error_norm(weighted_error, gt_area)
            )
            lookup[key] = {
                'weighted_error': float(weighted_error),
                'gt_area': float(gt_area),
                'weighted_error_norm': float(weighted_error_norm),
            }
    return lookup


def pipeline_merge_prediction_sqlites(input_sqlites: list[Path], output_sqlite: Path, reference_sqlite: Path | None = None) -> Path:
    merged_rows: list[tuple[int, str, str]] = []
    seen_keys: set[tuple[int, str]] = set()
    endpoint_flags: dict[tuple[int, str], int] = {}
    for input_sqlite in input_sqlites:
        endpoint_flags.update(pipeline_load_endpoint_extrapolated_flags(input_sqlite))
        for frame, track_id, polygons_json in render_load_rows(input_sqlite):
            key = (int(frame), str(track_id))
            if key in seen_keys:
                raise ValueError(f'Duplicate prediction row while merging branch SQLites: frame={frame} track_id={track_id}')
            seen_keys.add(key)
            merged_rows.append((int(frame), str(track_id), str(polygons_json)))
    merged_rows.sort(key=lambda row: (row[0], int(row[1])))
    fst.write_sqlite(merged_rows, output_sqlite, reference_sqlite=reference_sqlite)
    pipeline_apply_endpoint_extrapolated_flags(output_sqlite, endpoint_flags)
    return output_sqlite


def pipeline_add_k1_cost_columns_to_sqlite(
    sqlite_path: Path,
    *,
    reference_sqlite: Path,
    k1_cost_csv: Path,
    threshold_default: int,
    threshold_edge: int,
    threshold_default_norm: float,
    threshold_edge_norm: float,
    cost_routing: str,
) -> dict[str, object]:
    k1_cost_lookup = fst.load_k1_cost_lookup(k1_cost_csv)
    k1_cost_detail_lookup = pipeline_load_k1_cost_detail_lookup(k1_cost_csv)
    reference_polygons = pipeline_load_mask_polygon_lookup(reference_sqlite)
    threshold_default_int = int(threshold_default)
    threshold_edge_int = int(threshold_edge)
    threshold_default_norm_float = float(threshold_default_norm)
    threshold_edge_norm_float = float(threshold_edge_norm)
    use_normalized_threshold = str(cost_routing) == 'normalized'
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        columns = [str(row[1]) for row in cur.execute('PRAGMA table_info(masks)')]
        if 'k1_cost' not in columns:
            cur.execute('ALTER TABLE masks ADD COLUMN k1_cost INTEGER')
        if 'k1_cost_threshold' not in columns:
            cur.execute('ALTER TABLE masks ADD COLUMN k1_cost_threshold INTEGER')
        if 'k1_cost_gt_area' not in columns:
            cur.execute('ALTER TABLE masks ADD COLUMN k1_cost_gt_area REAL')
        if 'k1_cost_norm' not in columns:
            cur.execute('ALTER TABLE masks ADD COLUMN k1_cost_norm REAL')
        if 'k1_cost_norm_threshold' not in columns:
            cur.execute('ALTER TABLE masks ADD COLUMN k1_cost_norm_threshold REAL')

        rows = [
            (int(frame), str(track_id), str(polygons_json))
            for frame, track_id, polygons_json in cur.execute('SELECT frame, track_id, polygons FROM masks')
        ]
        updates: list[tuple[int | None, int | None, float | None, float | None, float | None, int, str]] = []
        found_cost = 0
        missing_cost = 0
        edge_threshold_rows = 0
        default_threshold_rows = 0
        for frame, track_id, polygons_json in rows:
            key = (int(frame), str(track_id))
            cost = k1_cost_lookup.get(key)
            detail = k1_cost_detail_lookup.get(key)
            if cost is None:
                missing_cost += 1
            else:
                found_cost += 1
            edge_source_json = reference_polygons.get((str(track_id), int(frame)), polygons_json)
            is_edge = pipeline_polygons_json_touches_k1_edge(edge_source_json)
            raw_threshold = threshold_edge_int if is_edge else threshold_default_int
            norm_threshold_config = threshold_edge_norm_float if is_edge else threshold_default_norm_float
            if is_edge:
                edge_threshold_rows += 1
            else:
                default_threshold_rows += 1
            gt_area = None if detail is None else float(detail.get('gt_area', 0.0))
            cost_norm = None if detail is None else float(detail.get('weighted_error_norm', 0.0))
            if use_normalized_threshold:
                threshold_norm = norm_threshold_config
                threshold = None if gt_area is None else int(round(float(threshold_norm) * max(float(gt_area), float(infer_K1_COST_NORM_AREA_FLOOR))))
            else:
                threshold = raw_threshold
                threshold_norm = (
                    None
                    if gt_area is None
                    else infer_compute_weighted_error_norm(float(threshold), float(gt_area))
                )
            updates.append((None if cost is None else int(cost), threshold, gt_area, cost_norm, threshold_norm, int(frame), str(track_id)))
        if updates:
            cur.executemany(
                'UPDATE masks SET k1_cost = ?, k1_cost_threshold = ?, k1_cost_gt_area = ?, k1_cost_norm = ?, k1_cost_norm_threshold = ? WHERE frame = ? AND track_id = ?',
                updates,
            )
        conn.commit()
    finally:
        conn.close()
    return {
        'sqlite': str(sqlite_path),
        'k1_cost_csv': str(k1_cost_csv),
        'threshold_default': int(threshold_default_int),
        'threshold_edge': int(threshold_edge_int),
        'threshold_default_norm': float(threshold_default_norm_float),
        'threshold_edge_norm': float(threshold_edge_norm_float),
        'cost_routing': str(cost_routing),
        'row_count': int(len(updates)),
        'k1_cost_rows': int(found_cost),
        'missing_k1_cost_rows': int(missing_cost),
        'edge_threshold_rows': int(edge_threshold_rows),
        'default_threshold_rows': int(default_threshold_rows),
        'columns': ['k1_cost', 'k1_cost_threshold', 'k1_cost_gt_area', 'k1_cost_norm', 'k1_cost_norm_threshold'],
    }


def pipeline_parse_optional_float(value: object) -> float | None:
    if value in (None, ''):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def pipeline_parse_optional_int(value: object) -> int | None:
    number = pipeline_parse_optional_float(value)
    if number is None:
        return None
    return int(number)


def pipeline_annotate_postprocess_metadata_to_sqlite(sqlite_path: Path, metrics_csvs: list[Path]) -> dict[str, object]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f'output sqlite does not exist: {sqlite_path}')

    lookup: dict[tuple[int, str], dict[str, object]] = {}
    csv_rows = 0
    duplicate_rows = 0
    missing_csvs: list[str] = []
    for metrics_csv in metrics_csvs:
        if not metrics_csv.exists():
            missing_csvs.append(str(metrics_csv))
            continue
        with metrics_csv.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (int(row['frame']), str(row['track_id']))
                except (KeyError, TypeError, ValueError):
                    continue
                if key in lookup:
                    duplicate_rows += 1
                lookup[key] = {
                    'has_keyframe': pipeline_parse_optional_int(row.get('has_keyframe')) or 0,
                    'is_gap_filled': pipeline_parse_optional_int(row.get('is_gap_filled')) or 0,
                    'postprocess_mode': row.get('mode'),
                    'postprocess_candidate_name': row.get('candidate_name'),
                    'postprocess_run_id': pipeline_parse_optional_int(row.get('run_id')),
                    'postprocess_recall': pipeline_parse_optional_float(row.get('recall')),
                    'postprocess_precision': pipeline_parse_optional_float(row.get('precision')),
                    'postprocess_iou': pipeline_parse_optional_float(row.get('iou')),
                    'postprocess_gt_area': pipeline_parse_optional_float(row.get('gt_area')),
                    'postprocess_pred_area': pipeline_parse_optional_float(row.get('pred_area')),
                    'postprocess_weighted_error': pipeline_parse_optional_float(row.get('weighted_error')),
                }
                csv_rows += 1

    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='masks'").fetchone() is None:
            raise RuntimeError(f'output sqlite does not contain masks table: {sqlite_path}')
        mask_columns = {str(row[1]) for row in cur.execute('PRAGMA table_info(masks)').fetchall()}
        column_defs = [
            ('has_keyframe', 'INTEGER NOT NULL DEFAULT 0'),
            ('is_gap_filled', 'INTEGER NOT NULL DEFAULT 0'),
            ('postprocess_mode', 'TEXT'),
            ('postprocess_candidate_name', 'TEXT'),
            ('postprocess_run_id', 'INTEGER'),
            ('postprocess_recall', 'REAL'),
            ('postprocess_precision', 'REAL'),
            ('postprocess_iou', 'REAL'),
            ('postprocess_gt_area', 'REAL'),
            ('postprocess_pred_area', 'REAL'),
            ('postprocess_weighted_error', 'REAL'),
        ]
        added_columns: list[str] = []
        for name, definition in column_defs:
            if name not in mask_columns:
                cur.execute(f'ALTER TABLE masks ADD COLUMN "{name}" {definition}')
                added_columns.append(name)

        rows = [
            (int(frame), str(track_id))
            for frame, track_id in cur.execute('SELECT frame, track_id FROM masks')
        ]
        updates: list[tuple[object, ...]] = []
        matched_rows = 0
        gap_filled_rows = 0
        keyframe_rows = 0
        for frame, track_id in rows:
            data = lookup.get((int(frame), str(track_id)))
            if data is None:
                updates.append((0, 0, None, None, None, None, None, None, None, None, None, int(frame), str(track_id)))
                continue
            matched_rows += 1
            gap_filled_rows += int(data['is_gap_filled'])
            keyframe_rows += int(data['has_keyframe'])
            updates.append(
                (
                    int(data['has_keyframe']),
                    int(data['is_gap_filled']),
                    data.get('postprocess_mode'),
                    data.get('postprocess_candidate_name'),
                    data.get('postprocess_run_id'),
                    data.get('postprocess_recall'),
                    data.get('postprocess_precision'),
                    data.get('postprocess_iou'),
                    data.get('postprocess_gt_area'),
                    data.get('postprocess_pred_area'),
                    data.get('postprocess_weighted_error'),
                    int(frame),
                    str(track_id),
                )
            )
        if updates:
            cur.executemany(
                '''
                UPDATE masks
                SET has_keyframe = ?,
                    is_gap_filled = ?,
                    postprocess_mode = ?,
                    postprocess_candidate_name = ?,
                    postprocess_run_id = ?,
                    postprocess_recall = ?,
                    postprocess_precision = ?,
                    postprocess_iou = ?,
                    postprocess_gt_area = ?,
                    postprocess_pred_area = ?,
                    postprocess_weighted_error = ?
                WHERE frame = ? AND track_id = ?
                ''',
                updates,
            )
        conn.commit()
        final_rows = int(len(rows))
    finally:
        conn.close()

    return {
        'enabled': True,
        'sqlite': str(sqlite_path),
        'metrics_csvs': [str(path) for path in metrics_csvs],
        'missing_metrics_csvs': missing_csvs,
        'csv_rows': int(csv_rows),
        'duplicate_metric_keys': int(duplicate_rows),
        'final_rows': int(final_rows),
        'matched_rows': int(matched_rows),
        'missing_rows': int(final_rows - matched_rows),
        'keyframe_rows': int(keyframe_rows),
        'gap_filled_rows': int(gap_filled_rows),
        'columns_added': added_columns,
    }


def pipeline_embed_original_masks_for_debug(output_sqlite: Path, original_sqlite: Path) -> dict[str, object]:
    if not output_sqlite.exists():
        raise FileNotFoundError(f'output sqlite does not exist: {output_sqlite}')
    if not original_sqlite.exists():
        raise FileNotFoundError(f'original sqlite does not exist: {original_sqlite}')

    conn = sqlite3.connect(str(output_sqlite))
    try:
        cur = conn.cursor()
        if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='masks'").fetchone() is None:
            raise RuntimeError(f'output sqlite does not contain masks table: {output_sqlite}')

        cur.execute('ATTACH DATABASE ? AS original_source', (str(original_sqlite),))
        try:
            if cur.execute("SELECT name FROM original_source.sqlite_master WHERE type='table' AND name='masks'").fetchone() is None:
                raise RuntimeError(f'original sqlite does not contain masks table: {original_sqlite}')
            source_columns = {str(row[1]) for row in cur.execute('PRAGMA original_source.table_info(masks)').fetchall()}
            required = {'frame', 'track_id', 'polygons'}
            if not required.issubset(source_columns):
                missing = ', '.join(sorted(required - source_columns))
                raise RuntimeError(f'original sqlite masks table is missing required columns: {missing}')

            mask_columns = {str(row[1]) for row in cur.execute('PRAGMA table_info(masks)').fetchall()}
            if 'has_original_mask' not in mask_columns:
                cur.execute('ALTER TABLE masks ADD COLUMN has_original_mask INTEGER NOT NULL DEFAULT 0')
            if 'mask_origin' not in mask_columns:
                cur.execute('ALTER TABLE masks ADD COLUMN mask_origin TEXT')
            mask_columns = {str(row[1]) for row in cur.execute('PRAGMA table_info(masks)').fetchall()}

            cur.execute('DROP TABLE IF EXISTS original_masks')
            cur.execute('DROP TABLE IF EXISTS raw_tracked_masks')
            cur.execute('DROP TABLE IF EXISTS raw_tracks')
            cur.execute(
                '''
                UPDATE masks
                SET has_original_mask = CASE
                    WHEN EXISTS(
                        SELECT 1
                        FROM original_source.masks original
                        WHERE original.frame = masks.frame
                          AND original.track_id = masks.track_id
                    )
                    THEN 1 ELSE 0
                END
                '''
            )
            if 'is_endpoint_extrapolated' in mask_columns:
                cur.execute(
                    '''
                    UPDATE masks
                    SET mask_origin = CASE
                        WHEN has_original_mask != 0 THEN 'tracked_original'
                        WHEN COALESCE(is_endpoint_extrapolated, 0) != 0 THEN 'endpoint_extrapolated'
                        ELSE 'generated_no_original'
                    END
                    '''
                )
            else:
                cur.execute(
                    '''
                    UPDATE masks
                    SET mask_origin = CASE
                        WHEN has_original_mask != 0 THEN 'tracked_original'
                        ELSE 'generated_no_original'
                    END
                    '''
                )

            final_rows = int(cur.execute('SELECT COUNT(*) FROM masks').fetchone()[0])
            original_rows = int(cur.execute('SELECT COUNT(*) FROM original_source.masks').fetchone()[0])
            final_with_original = int(cur.execute('SELECT COUNT(*) FROM masks WHERE has_original_mask != 0').fetchone()[0])
            final_without_original = int(cur.execute('SELECT COUNT(*) FROM masks WHERE has_original_mask = 0').fetchone()[0])
            endpoint_without_original = 0
            if 'is_endpoint_extrapolated' in mask_columns:
                endpoint_without_original = int(
                    cur.execute(
                        'SELECT COUNT(*) FROM masks WHERE has_original_mask = 0 AND COALESCE(is_endpoint_extrapolated, 0) != 0'
                    ).fetchone()[0]
                )
            original_polygons_json_bytes = int(
                cur.execute('SELECT COALESCE(SUM(LENGTH(polygons)), 0) FROM original_source.masks').fetchone()[0]
            )
            conn.commit()
        finally:
            cur.execute('DETACH DATABASE original_source')
    finally:
        conn.close()

    return {
        'enabled': True,
        'output_sqlite': str(output_sqlite),
        'original_sqlite': str(original_sqlite),
        'table': None,
        'final_rows': final_rows,
        'original_rows': original_rows,
        'final_rows_with_original': final_with_original,
        'final_rows_without_original': final_without_original,
        'endpoint_rows_without_original': endpoint_without_original,
        'original_polygons_json_bytes': original_polygons_json_bytes,
        'output_sqlite_size_bytes': int(output_sqlite.stat().st_size),
        'mask_columns_added': ['has_original_mask', 'mask_origin'],
        'original_masks_embedded': False,
    }


def pipeline_copy_sqlite_table_by_name(src_cur: sqlite3.Cursor, dst_cur: sqlite3.Cursor, table_name: str) -> int:
    if src_cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone() is None:
        return 0
    create_row = src_cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if create_row is None or not create_row[0]:
        return 0
    dst_cur.execute(str(create_row[0]))
    columns = [str(row[1]) for row in src_cur.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
    if columns:
        cols = ', '.join(f'"{name}"' for name in columns)
        rows = src_cur.execute(f'SELECT {cols} FROM "{table_name}"').fetchall()
        if rows:
            placeholders = ', '.join('?' for _ in columns)
            dst_cur.executemany(f'INSERT INTO "{table_name}"({cols}) VALUES ({placeholders})', rows)
    return int(dst_cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])


def pipeline_count_sqlite_rows(cur: sqlite3.Cursor, table_name: str) -> int:
    if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone() is None:
        return 0
    return int(cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])


def pipeline_write_tracking_pruned_sqlite(output_sqlite: Path, source_sqlite: Path) -> dict[str, object]:
    if not source_sqlite.exists():
        raise FileNotFoundError(f'source sqlite does not exist: {source_sqlite}')
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        output_sqlite.unlink()

    src = sqlite3.connect(str(source_sqlite))
    copied_tables: dict[str, int] = {}
    try:
        src_cur = src.cursor()
        source_tables = {str(row[0]) for row in src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'masks' not in source_tables:
            raise RuntimeError(f'source sqlite does not contain masks table: {source_sqlite}')
        dst = sqlite3.connect(str(output_sqlite))
        try:
            dst_cur = dst.cursor()
            for table_name in ('masks', 'tracks', 'cuts'):
                row_count = pipeline_copy_sqlite_table_by_name(src_cur, dst_cur, table_name)
                if row_count:
                    copied_tables[table_name] = int(row_count)
            if 'masks' in copied_tables:
                dst_cur.execute('CREATE INDEX IF NOT EXISTS idx_tracking_pruned_masks_track_frame ON masks(track_id, frame)')
            dst.commit()
        finally:
            dst.close()
        raw_tables_available = bool({'raw_tracked_masks', 'raw_tracks'}.issubset(source_tables))
        kept_tracks = pipeline_count_sqlite_rows(src_cur, 'tracks')
        kept_track_frame_rows = pipeline_count_sqlite_rows(src_cur, 'masks')
        kept_unique_frames = int(src_cur.execute('SELECT COUNT(DISTINCT frame) FROM masks').fetchone()[0])
        if raw_tables_available:
            raw_tracks = int(src_cur.execute('SELECT COUNT(*) FROM raw_tracks').fetchone()[0])
            raw_track_frame_rows = int(src_cur.execute('SELECT COUNT(*) FROM raw_tracked_masks').fetchone()[0])
            removed_tracks = int(src_cur.execute('SELECT COUNT(*) FROM raw_tracks WHERE COALESCE(removed_by_short_track, 0) != 0').fetchone()[0])
            removed_track_frame_rows = int(src_cur.execute('SELECT COUNT(*) FROM raw_tracked_masks WHERE COALESCE(removed_by_short_track, 0) != 0').fetchone()[0])
            raw_unique_frames = int(src_cur.execute('SELECT COUNT(DISTINCT frame) FROM raw_tracked_masks').fetchone()[0])
            affected_unique_frames = int(src_cur.execute('SELECT COUNT(DISTINCT frame) FROM raw_tracked_masks WHERE COALESCE(removed_by_short_track, 0) != 0').fetchone()[0])
            frames_removed_entirely = int(
                src_cur.execute(
                    'SELECT COUNT(*) FROM (SELECT DISTINCT frame FROM raw_tracked_masks EXCEPT SELECT DISTINCT frame FROM masks)'
                ).fetchone()[0]
            )
        else:
            raw_tracks = kept_tracks
            raw_track_frame_rows = kept_track_frame_rows
            removed_tracks = 0
            removed_track_frame_rows = 0
            raw_unique_frames = kept_unique_frames
            affected_unique_frames = 0
            frames_removed_entirely = 0
    finally:
        src.close()

    return {
        'enabled': True,
        'source_sqlite': str(source_sqlite),
        'output_sqlite': str(output_sqlite),
        'copied_tables': copied_tables,
        'short_track_prune': {
            'raw_tables_available': raw_tables_available,
            'raw_tracks_before_prune': int(raw_tracks),
            'tracks_after_prune': int(kept_tracks),
            'removed_tracks': int(removed_tracks),
            'raw_track_frame_rows_before_prune': int(raw_track_frame_rows),
            'track_frame_rows_after_prune': int(kept_track_frame_rows),
            'removed_track_frame_rows': int(removed_track_frame_rows),
            'raw_unique_frames_before_prune': int(raw_unique_frames),
            'unique_frames_after_prune': int(kept_unique_frames),
            'affected_unique_frames': int(affected_unique_frames),
            'unique_frames_removed_entirely': int(frames_removed_entirely),
        },
        'output_sqlite_size_bytes': int(output_sqlite.stat().st_size),
    }


def pipeline_evaluate_pred_sqlite_exact(tracked_sqlite: Path, pred_sqlite: Path, output_dir: Path, *, track_mode_lookup: dict[str, str] | None=None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = render_load_rows(tracked_sqlite)
    pred_rows = render_load_rows(pred_sqlite)
    pred_lookup = {(frame, track_id): polygons_json for frame, track_id, polygons_json in pred_rows}
    result_rows: list[dict[str, object]] = []
    total_intersection = total_union = total_gt_area = total_pred_area = 0
    k1_intersection = k1_union = k1_gt_area = k1_pred_area = 0
    k2_intersection = k2_union = k2_gt_area = k2_pred_area = 0
    recall_below_090 = 0
    recall_below_095 = 0
    missing_rows = 0
    mean_recall: list[float] = []
    mean_precision: list[float] = []
    mean_iou: list[float] = []
    k1_count = 0
    k2_count = 0
    for frame, track_id, gt_json in gt_rows:
        pred_json_or_none = pred_lookup.get((frame, track_id))
        pred_json = pred_json_or_none if pred_json_or_none is not None else '[]'
        gt_polys = fst.parse_polygons(gt_json)
        pred_polys = fst.parse_polygons(pred_json)
        metrics = fst.compute_exact_metrics_from_polygons(gt_polys, pred_polys)
        if pred_json_or_none is None:
            missing_rows += 1
        else:
            intersection = int(metrics['intersection'])
            union = int(metrics['union'])
            gt_area = int(metrics['gt_area'])
            pred_area = int(metrics['pred_area'])
            total_intersection += intersection
            total_union += union
            total_gt_area += gt_area
            total_pred_area += pred_area
            mean_recall.append(float(metrics['recall']))
            mean_precision.append(float(metrics['precision']))
            mean_iou.append(float(metrics['iou']))
            poly_count = len(json.loads(pred_json))
            if poly_count >= 2:
                k2_count += 1
                k2_intersection += intersection
                k2_union += union
                k2_gt_area += gt_area
                k2_pred_area += pred_area
            else:
                k1_count += 1
                k1_intersection += intersection
                k1_union += union
                k1_gt_area += gt_area
                k1_pred_area += pred_area
            if float(metrics['recall']) < 0.9:
                recall_below_090 += 1
            if float(metrics['recall']) < 0.95:
                recall_below_095 += 1
        mode = '' if track_mode_lookup is None else str(track_mode_lookup.get(str(track_id), ''))
        result_rows.append(
            {
                'frame': int(frame),
                'track_id': str(track_id),
                'mode': mode,
                'candidate_name': mode.lower() if mode else '',
                'gt_area': float(metrics['gt_area']),
                'pred_area': float(metrics['pred_area']),
                'intersection': float(metrics['intersection']),
                'union': float(metrics['union']),
                'recall': float(metrics['recall']),
                'precision': float(metrics['precision']),
                'iou': float(metrics['iou']),
                'weighted_error': float(metrics.get('weighted_error', 0.0)),
                'has_keyframe': 0,
            }
        )
    metrics_csv = output_dir / 'keyframe_exact_metrics.csv'
    with metrics_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'frame',
                'track_id',
                'mode',
                'candidate_name',
                'gt_area',
                'pred_area',
                'intersection',
                'union',
                'recall',
                'precision',
                'iou',
                'weighted_error',
                'has_keyframe',
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(result_rows, key=lambda row: (int(row['frame']), int(str(row['track_id'])))))
    aggregate = {
        'global_recall': total_intersection / total_gt_area if total_gt_area else 1.0,
        'global_precision': total_intersection / total_pred_area if total_pred_area else 1.0,
        'global_iou': total_intersection / total_union if total_union else 1.0,
        'mean_recall': float(np.mean(mean_recall)) if mean_recall else 1.0,
        'mean_precision': float(np.mean(mean_precision)) if mean_precision else 1.0,
        'mean_iou': float(np.mean(mean_iou)) if mean_iou else 1.0,
        'recall_below_090': int(recall_below_090),
        'recall_below_095': int(recall_below_095),
        'missing_rows': int(missing_rows),
        'total_gt_rows': len(gt_rows),
        'total_sub_rows': len(pred_rows),
        'k1_count': int(k1_count),
        'k2_count': int(k2_count),
        'k1_recall': k1_intersection / k1_gt_area if k1_gt_area else 0.0,
        'k1_iou': k1_intersection / k1_union if k1_union else 0.0,
        'k2_recall': k2_intersection / k2_gt_area if k2_gt_area else 0.0,
        'k2_iou': k2_intersection / k2_union if k2_union else 0.0,
    }
    summary = {
        'tracked_sqlite': str(tracked_sqlite),
        'pred_sqlite': str(pred_sqlite),
        'metrics_csv': str(metrics_csv),
        'row_count': int(len(result_rows)),
        'metrics': aggregate,
    }
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def pipeline_prepare_branch_sqlites(tracked_sqlite: Path, branch_plan: dict[str, object], branch_root: Path) -> dict[str, str | None]:
    branch_sqlites: dict[str, str | None] = {'ellipse': None, 'polygon': None}
    total_track_count = int(branch_plan['track_count'])
    for mode in pipeline_VALID_SHAPE_MODES:
        track_ids = list(branch_plan['branch_tracks'][mode])
        if not track_ids:
            continue
        if len(track_ids) == total_track_count:
            branch_sqlites[mode] = str(tracked_sqlite)
            continue
        subset_sqlite = branch_root / mode / 'preprocess' / f'{mode}_tracked.sqlite'
        pipeline_write_subset_sqlite(tracked_sqlite, subset_sqlite, track_ids)
        branch_sqlites[mode] = str(subset_sqlite)
    return branch_sqlites


def pipeline_build_settings_summary(args: argparse.Namespace, intervals: list[int]) -> dict[str, object]:
    return {
        'intervals': intervals,
        'dense_recall_target': float(args.dense_recall_target),
        'polygon_recall_min': None if args.polygon_recall_min is None else float(args.polygon_recall_min),
        'keyframe_max_gap': int(args.keyframe_max_gap),
        'gap_fill_max_gap': int(args.gap_fill_max_gap),
        'dense_recall_max_inflate_log': float(args.dense_recall_max_inflate_log),
        'render_overlays': bool(args.render_overlays),
        'overlay_encoder': str(args.overlay_encoder),
        'embed_original_masks': bool(args.embed_original_masks),
        'fallback_shape_mode': str(args.default_shape_mode),
        'default_shape_mode': str(args.default_shape_mode),
        'class_policy_json': str(args.class_policy_json) if args.class_policy_json is not None else None,
        'polygon_optimizer': 'embedded_v22',
        'polygon_border_expand': bool(args.polygon_border_expand),
        'polygon_border_trigger_px': float(args.polygon_border_trigger_px),
        'polygon_border_expand_ratio': float(args.polygon_border_expand_ratio),
        'polygon_border_min_expand_px': float(args.polygon_border_min_expand_px),
        'polygon_border_max_expand_px': float(args.polygon_border_max_expand_px),
        'polygon_border_influence_px': float(args.polygon_border_influence_px),
        'polygon_border_width': int(args.polygon_border_width),
        'polygon_border_height': int(args.polygon_border_height),
        'endpoint_extend': bool(args.endpoint_extend),
        'endpoint_extend_stage': 'post_branch_approximation_pre_keyframe',
        'endpoint_extend_frames': int(args.endpoint_extend_frames),
        'endpoint_extend_max_speed_px': float(args.endpoint_extend_max_speed_px),
        'endpoint_extend_edge_only': bool(args.endpoint_extend_edge_only),
        'endpoint_extend_edge_margin_px': float(args.endpoint_extend_edge_margin_px),
        'endpoint_extend_edge_confirm_frames': int(args.endpoint_extend_edge_confirm_frames),
        'endpoint_extend_motion_frames': int(args.endpoint_extend_motion_frames),
        'raw_det_score_min': float(args.raw_det_score_min),
        'polygon_num_workers': int(args.polygon_num_workers),
        'polygon_adaptive_anchor_counts': (
            None if args.polygon_adaptive_anchor_counts is None else bool(args.polygon_adaptive_anchor_counts)
        ),
        'progress_interval_sec': float(args.progress_interval_sec),
        'k1_workers': int(args.k1_workers),
        'k2_batch_size': int(args.k2_batch_size),
        'k2_prep_workers': int(args.k2_prep_workers),
        'k2_precision': str(args.k2_precision),
        'k2_forward_mode': str(args.k2_forward_mode),
        'k2_profile_stages': bool(args.k2_profile_stages),
        'k2_cudnn_benchmark': str(args.k2_cudnn_benchmark),
        'k2_tf32': str(args.k2_tf32),
        'routing_mode': str(args.routing_mode),
        'k1_cost_routing': str(args.k1_cost_routing),
        'threshold': int(args.threshold),
        'threshold_edge': int(args.threshold_edge),
        'threshold_norm': float(args.threshold_norm),
        'threshold_edge_norm': float(args.threshold_edge_norm),
        'k1n_seq_enter_norm': float(args.k1n_seq_enter_norm),
        'k1n_seq_exit_norm': float(args.k1n_seq_exit_norm),
        'k1n_seq_strong_enter_norm': float(args.k1n_seq_strong_enter_norm),
        'k1n_seq_strong_exit_norm': float(args.k1n_seq_strong_exit_norm),
        'k1n_seq_protect_k2_iou_below': float(args.k1n_seq_protect_k2_iou_below),
        'k1n_seq_smooth_window': int(args.k1n_seq_smooth_window),
        'k1n_seq_enter_confirm_frames': int(args.k1n_seq_enter_confirm_frames),
        'k1n_seq_exit_confirm_frames': int(args.k1n_seq_exit_confirm_frames),
        'k1n_seq_merge_short_k1_max_len': int(args.k1n_seq_merge_short_k1_max_len),
        'k1n_seq_merge_short_k2_max_len': int(args.k1n_seq_merge_short_k2_max_len),
        'k1n_seq_reset_gap': int(args.k1n_seq_reset_gap),
        'k2_dp_merge_short_k2_keep_cost_norm': float(args.k2_dp_merge_short_k2_keep_cost_norm),
        'k2_dp_force_k2_cost_norm': float(args.k2_dp_force_k2_cost_norm),
    }


def pipeline_resolve_group_target_ratio(group: dict[str, object], interval: int) -> float:
    policy = dict(group.get('policy', {}))
    if 'target_ratio' in policy:
        return float(policy['target_ratio'])
    if 'target_interval' in policy:
        value = float(policy['target_interval'])
        if value <= 0.0:
            raise ValueError(f'invalid target_interval for group {group["group_id"]}: {value}')
        return 1.0 / value
    return 1.0 / float(interval)


def pipeline_resolve_group_dense_recall_target(args: argparse.Namespace, group: dict[str, object]) -> float:
    policy = dict(group.get('policy', {}))
    if 'dense_recall_target' in policy:
        return float(policy['dense_recall_target'])
    if 'target_recall' in policy:
        return float(policy['target_recall'])
    return float(args.dense_recall_target)


def pipeline_resolve_group_polygon_recall_min(args: argparse.Namespace, group: dict[str, object]) -> float | None:
    policy = dict(group.get('policy', {}))
    if 'polygon_recall_min' in policy:
        return float(policy['polygon_recall_min'])
    if 'recall_min' in policy:
        return float(policy['recall_min'])
    if 'target_recall' in policy:
        return float(policy['target_recall'])
    if args.polygon_recall_min is not None:
        return float(args.polygon_recall_min)
    return None


def pipeline_build_zero_k1_cost_csv_from_sqlite(source_sqlite: Path, output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = render_load_rows(source_sqlite)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'track_id', 'weighted_error', 'gt_area', 'weighted_error_norm'])
        writer.writeheader()
        for frame, track_id, _polygons_json in rows:
            writer.writerow({'frame': int(frame), 'track_id': str(track_id), 'weighted_error': 0, 'gt_area': 0, 'weighted_error_norm': 0.0})
    return output_csv


def pipeline_sqlite_mask_stats(sqlite_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        rows = int(cur.execute('SELECT COUNT(*) FROM masks').fetchone()[0])
        tracks = int(cur.execute('SELECT COUNT(DISTINCT track_id) FROM masks').fetchone()[0])
        frame_min, frame_max = cur.execute('SELECT MIN(frame), MAX(frame) FROM masks').fetchone()
        return {
            'rows': rows,
            'tracks': tracks,
            'frame_min': int(frame_min) if frame_min is not None else -1,
            'frame_max': int(frame_max) if frame_max is not None else -1,
        }
    finally:
        conn.close()


def pipeline_merge_k1_cost_csvs(input_csvs: list[Path], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_rows: list[dict[str, object]] = []
    seen_keys: set[tuple[int, str]] = set()
    for input_csv in input_csvs:
        with input_csv.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['frame']), str(row['track_id']))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_rows.append(
                    {
                        'frame': int(row['frame']),
                        'track_id': str(row['track_id']),
                        'weighted_error': int(float(row.get('weighted_error', 0.0))),
                        'gt_area': int(float(row.get('gt_area', 0.0) or 0.0)),
                        'weighted_error_norm': float(row.get('weighted_error_norm', 0.0) or 0.0),
                    }
                )
    merged_rows.sort(key=lambda row: (int(row['frame']), int(str(row['track_id']))))
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'track_id', 'weighted_error', 'gt_area', 'weighted_error_norm'])
        writer.writeheader()
        writer.writerows(merged_rows)
    return output_csv


def pipeline_subcommand_cmd(python_executable: str, subcommand: str, extra_args: list[str]) -> list[str]:
    return [python_executable, str(SELF_PATH), subcommand, *extra_args]


def pipeline_ensure_preprocess(args: argparse.Namespace, pipeline_dir: Path) -> tuple[Path, dict[str, object], dict[str, object] | None]:
    if args.input_sqlite is not None:
        if not args.input_sqlite.exists():
            raise FileNotFoundError(f'Input sqlite not found: {args.input_sqlite}')
        return args.input_sqlite, {'reused': True, 'wall_seconds': 0.0}, None

    preprocess_output_dir = pipeline_dir / 'preprocess'
    summary_path = preprocess_output_dir / 'summary.json'
    if summary_path.exists() and not args.force:
        summary = pipeline_load_json(summary_path)
        tracked_sqlite = Path(str(summary['tracked_sqlite']))
        return tracked_sqlite, {'reused': True, 'wall_seconds': 0.0}, summary.get('preprocess_summary')

    cmd = pipeline_subcommand_cmd(
        args.python,
        '__onefile_preprocess',
        [
            '--input-jsonl', str(args.input_jsonl),
            '--output-dir', str(preprocess_output_dir),
            '--raw-remove-short-tracks-max-frames', str(args.raw_remove_short_tracks_max_frames),
            '--raw-det-score-min', str(args.raw_det_score_min),
            '--raw-cut-method', str(args.raw_cut_method),
        ],
    )
    cmd.append('--raw-cut-detect' if args.raw_cut_detect else '--no-raw-cut-detect')
    if args.input_video is not None:
        cmd.extend(['--input-video', str(args.input_video)])
    if args.class_policy_json is not None:
        cmd.extend(['--class-policy-json', str(args.class_policy_json)])
    preprocess_timing = pipeline_run_cmd(
        cmd,
        cwd=ROOT,
        progress_interval_sec=float(args.progress_interval_sec),
        progress_frame_count=pipeline_video_frame_count(pipeline_guess_input_video(args)),
    )
    summary = pipeline_load_json(summary_path)
    tracked_sqlite = Path(str(summary['tracked_sqlite']))
    return tracked_sqlite, preprocess_timing, summary.get('preprocess_summary')


def pipeline_ensure_inference(args: argparse.Namespace, pipeline_dir: Path, *, source_sqlite: Path | None=None) -> tuple[Path, dict[str, object], dict[str, object]]:
    if args.reuse_inference_output_dir is not None:
        inference_output_dir = args.reuse_inference_output_dir
        infer_timing = {'reused': True, 'wall_seconds': 0.0}
        infer_summary = pipeline_load_json(inference_output_dir / 'summary.json')
        return inference_output_dir, infer_timing, infer_summary

    inference_output_dir = pipeline_dir / 'inference'
    summary_path = inference_output_dir / 'summary.json'
    if summary_path.exists() and not args.force:
        return inference_output_dir, {'reused': True, 'wall_seconds': 0.0}, pipeline_load_json(summary_path)

    cmd = pipeline_subcommand_cmd(
        args.python,
        '__onefile_infer',
        [
            '--output-dir', str(inference_output_dir),
            '--k1-recall-target', str(args.k1_recall_target),
            '--k1-exact-refine-rounds', str(args.k1_exact_refine_rounds),
            '--k1-workers', str(args.k1_workers),
            '--k2-run-dir', str(args.k2_run_dir),
            '--k2-device', str(args.k2_device),
            '--k2-batch-size', str(args.k2_batch_size),
            '--k2-prep-workers', str(args.k2_prep_workers),
            '--k2-precision', str(args.k2_precision),
            '--k2-forward-mode', str(args.k2_forward_mode),
            '--k2-cudnn-benchmark', str(args.k2_cudnn_benchmark),
            '--k2-tf32', str(args.k2_tf32),
            '--routing-mode', str(args.routing_mode),
            '--k1-cost-routing', str(args.k1_cost_routing),
            '--threshold', str(args.threshold),
            '--threshold-edge', str(args.threshold_edge),
            '--threshold-norm', str(args.threshold_norm),
            '--threshold-edge-norm', str(args.threshold_edge_norm),
            '--k2-soft-k1-keep-cost-norm', str(args.k2_soft_k1_keep_cost_norm),
            '--k2-hyst-enter-norm', str(args.k2_hyst_enter_norm),
            '--k2-hyst-enter-edge-norm', str(args.k2_hyst_enter_edge_norm),
            '--k2-hyst-exit-norm', str(args.k2_hyst_exit_norm),
            '--k2-hyst-exit-edge-norm', str(args.k2_hyst_exit_edge_norm),
            '--k1n-seq-enter-norm', str(args.k1n_seq_enter_norm),
            '--k1n-seq-exit-norm', str(args.k1n_seq_exit_norm),
            '--k1n-seq-strong-enter-norm', str(args.k1n_seq_strong_enter_norm),
            '--k1n-seq-strong-exit-norm', str(args.k1n_seq_strong_exit_norm),
            '--k1n-seq-protect-k2-iou-below', str(args.k1n_seq_protect_k2_iou_below),
            '--k1n-seq-smooth-window', str(args.k1n_seq_smooth_window),
            '--k1n-seq-enter-confirm-frames', str(args.k1n_seq_enter_confirm_frames),
            '--k1n-seq-exit-confirm-frames', str(args.k1n_seq_exit_confirm_frames),
            '--k1n-seq-merge-short-k1-max-len', str(args.k1n_seq_merge_short_k1_max_len),
            '--k1n-seq-merge-short-k2-max-len', str(args.k1n_seq_merge_short_k2_max_len),
            '--k1n-seq-reset-gap', str(args.k1n_seq_reset_gap),
            '--k2-dp-merge-short-k2-keep-cost-norm', str(args.k2_dp_merge_short_k2_keep_cost_norm),
            '--k2-dp-force-k2-cost-norm', str(args.k2_dp_force_k2_cost_norm),
            '--raw-remove-short-tracks-max-frames', str(args.raw_remove_short_tracks_max_frames),
            '--raw-det-score-min', str(args.raw_det_score_min),
            '--raw-cut-method', str(args.raw_cut_method),
        ],
    )
    if args.class_policy_json is not None:
        cmd.extend(['--class-policy-json', str(args.class_policy_json)])
    if bool(args.k2_profile_stages):
        cmd.append('--k2-profile-stages')
    if source_sqlite is not None:
        cmd.extend(['--input-sqlite', str(source_sqlite)])
    elif args.input_sqlite is not None:
        cmd.extend(['--input-sqlite', str(args.input_sqlite)])
    else:
        cmd.append('--raw-cut-detect' if args.raw_cut_detect else '--no-raw-cut-detect')
        cmd.extend(['--input-jsonl', str(args.input_jsonl)])
        if args.input_video is not None:
            cmd.extend(['--input-video', str(args.input_video)])
    inference_rows = None
    if source_sqlite is not None:
        inference_rows = pipeline_sqlite_mask_stats(source_sqlite)['rows']
    elif args.input_sqlite is not None:
        inference_rows = pipeline_sqlite_mask_stats(args.input_sqlite)['rows']
    infer_timing = pipeline_run_cmd(
        cmd,
        cwd=ROOT,
        progress_interval_sec=float(args.progress_interval_sec),
        progress_frame_count=pipeline_video_frame_count(pipeline_guess_input_video(args)),
        progress_unit_count=inference_rows,
        progress_unit_label='mask_rows',
    )
    infer_summary = pipeline_load_json(summary_path)
    return inference_output_dir, infer_timing, infer_summary


def pipeline_run_render_overlay(
    args: argparse.Namespace,
    *,
    video_path: Path | None,
    tracked_sqlite: Path,
    pred_sqlite_path: Path,
    metrics_csv: Path,
    k1_cost_csv: Path,
    overlay_video: Path,
) -> dict[str, object]:
    if not args.render_overlays:
        return {'skipped': True, 'wall_seconds': 0.0}
    if video_path is None:
        raise RuntimeError('Overlay rendering requires a video path.')
    if overlay_video.exists() and not args.force:
        return {'reused': True, 'wall_seconds': 0.0}
    render_cmd = pipeline_subcommand_cmd(
        args.python,
        '__onefile_render',
        [
            '--video', str(video_path),
            '--gt-sqlite', str(tracked_sqlite),
            '--pred-sqlite', str(pred_sqlite_path),
            '--metrics-csv', str(metrics_csv),
            '--k1-cost-csv', str(k1_cost_csv),
            '--output-video', str(overlay_video),
            '--encoder', str(args.overlay_encoder),
        ],
    )
    return pipeline_run_cmd(
        render_cmd,
        cwd=ROOT,
        progress_interval_sec=float(args.progress_interval_sec),
        progress_frame_count=pipeline_video_frame_count(video_path),
    )


def pipeline_run_polygon_interval(
    args: argparse.Namespace,
    *,
    tracked_sqlite: Path,
    source_sqlite: Path,
    interval_dir: Path,
    interval: int,
    video_path: Path | None,
    k1_cost_csv: Path,
    render_overlay: bool,
    target_ratio_override: float | None=None,
    recall_min_override: float | None=None,
) -> dict[str, object]:
    polygon_dir = interval_dir / 'polygon'
    border_dir = interval_dir / 'polygon_border_expand'
    overlay_video = interval_dir / f'overlay_int_{interval}.mp4'
    ratio = float(target_ratio_override) if target_ratio_override is not None else 1.0 / float(interval)
    timings: dict[str, object] = {}
    polygon_source_sqlite = source_sqlite
    border_expand_summary: dict[str, object] | None = None
    endpoint_extend_summary: dict[str, object] | None = None

    if bool(args.polygon_border_expand):
        border_sqlite = border_dir / 'border_expanded.sqlite'
        border_summary_json = border_dir / 'summary.json'
        if not border_sqlite.exists() or not border_summary_json.exists() or args.force:
            t0 = time.perf_counter()
            polygon_source_sqlite = pipeline_polygon_border_expand_sqlite(args, source_sqlite, border_sqlite, border_summary_json)
            timings['polygon_border_expand'] = {'wall_seconds': float(time.perf_counter() - t0)}
        else:
            polygon_source_sqlite = border_sqlite
            timings['polygon_border_expand'] = {'reused': True, 'wall_seconds': 0.0}
        border_expand_summary = pipeline_load_json(border_summary_json)
    else:
        timings['polygon_border_expand'] = {'skipped': True, 'wall_seconds': 0.0}

    if bool(args.endpoint_extend):
        endpoint_dir = interval_dir / 'polygon_endpoint_extend'
        endpoint_sqlite = endpoint_dir / 'input_endpoint_extended.sqlite'
        endpoint_summary_json = endpoint_dir / 'summary.json'
        if not endpoint_sqlite.exists() or not endpoint_summary_json.exists() or args.force:
            t0 = time.perf_counter()
            polygon_source_sqlite = pipeline_endpoint_extend_prediction_sqlite(
                args,
                polygon_source_sqlite,
                endpoint_sqlite,
                endpoint_summary_json,
                cuts_source_sqlite=tracked_sqlite,
                video_path=video_path,
            )
            timings['endpoint_extend'] = {'wall_seconds': float(time.perf_counter() - t0), 'stage': 'post_polygon_prepare_pre_keyframe'}
        else:
            polygon_source_sqlite = endpoint_sqlite
            timings['endpoint_extend'] = {'reused': True, 'wall_seconds': 0.0, 'stage': 'post_polygon_prepare_pre_keyframe'}
        endpoint_extend_summary = pipeline_load_json(endpoint_summary_json)
    else:
        timings['endpoint_extend'] = {'skipped': True, 'wall_seconds': 0.0}

    if not (polygon_dir / 'summary.json').exists() or args.force:
        polygon_input_stats = pipeline_sqlite_mask_stats(polygon_source_sqlite)
        polygon_workers = max(1, int(args.polygon_num_workers))
        print(
            '[polygon-optimize-start] '
            f'rows={polygon_input_stats["rows"]} '
            f'tracks={polygon_input_stats["tracks"]} '
            f'frames={polygon_input_stats["frame_min"]}-{polygon_input_stats["frame_max"]} '
            f'target_ratio={ratio:.6f} '
            f'workers={polygon_workers} '
            f'max_run_frames={int(args.polygon_max_run_frames)} '
            f'overlap_frames={int(args.polygon_run_overlap_frames)} '
            f'adaptive_anchors={bool(args.polygon_adaptive_anchor_counts)} '
            f'recall_min={recall_min_override if recall_min_override is not None else "default"}',
            flush=True,
        )
        polygon_cmd = pipeline_subcommand_cmd(
            args.python,
            '__onefile_polygon_optimize',
            [
                '--input-sqlite', str(polygon_source_sqlite),
                '--output-dir', str(polygon_dir),
                '--target-ratio', str(ratio),
                '--anchors-per-contour', '48',
                '--evaluate-exact',
                '--write-pred-sqlite',
                '--num-workers', str(polygon_workers),
                '--predictor-device', str(args.polygon_predictor_device),
                '--predictor-batch-size', str(args.polygon_predictor_batch_size),
                '--adaptive-point-quantile', str(args.polygon_adaptive_point_quantile),
                '--adaptive-point-offset', str(args.polygon_adaptive_point_offset),
                '--min-anchors-per-contour', str(args.polygon_min_anchors_per_contour),
                '--max-run-frames', str(args.polygon_max_run_frames),
                '--run-overlap-frames', str(args.polygon_run_overlap_frames),
            ],
        )
        if args.polygon_point_predictor_model_dir is not None:
            polygon_cmd.extend(['--point-predictor-model-dir', str(args.polygon_point_predictor_model_dir)])
        if args.polygon_adaptive_anchor_counts is not None:
            polygon_cmd.append(
                '--adaptive-anchor-counts'
                if bool(args.polygon_adaptive_anchor_counts)
                else '--no-adaptive-anchor-counts'
            )
        if recall_min_override is not None:
            polygon_cmd.extend(['--recall-min', str(recall_min_override)])
        timings['polygon_optimize'] = pipeline_run_cmd(
            polygon_cmd,
            cwd=ROOT,
            progress_interval_sec=float(args.progress_interval_sec),
            progress_frame_count=pipeline_video_frame_count(video_path),
            progress_unit_count=polygon_input_stats['rows'],
            progress_unit_label='mask_rows',
        )
    else:
        timings['polygon_optimize'] = {'reused': True, 'wall_seconds': 0.0}

    polygon_summary = pipeline_load_json(polygon_dir / 'summary.json')
    pred_sqlite_path = Path(str(polygon_summary['artifacts']['pred_sqlite']))
    exact_summary_path = Path(str(polygon_summary['artifacts']['exact_summary_json']))
    exact_metrics_csv = polygon_dir / 'exact' / 'keyframe_exact_metrics.csv'

    if render_overlay:
        timings['render_overlay'] = pipeline_run_render_overlay(
            args,
            video_path=video_path,
            tracked_sqlite=tracked_sqlite,
            pred_sqlite_path=pred_sqlite_path,
            metrics_csv=exact_metrics_csv,
            k1_cost_csv=k1_cost_csv,
            overlay_video=overlay_video,
        )

    return {
        'interval': int(interval),
        'target_ratio': float(ratio),
        'mode': 'polygon',
        'border_expand_summary': border_expand_summary,
        'endpoint_extend_summary': endpoint_extend_summary,
        'optimizer_summary': polygon_summary['optimizer_summary'],
        'exact_summary': pipeline_load_json(exact_summary_path),
        'paths': {
            'polygon_dir': str(polygon_dir),
            'polygon_source_sqlite': str(polygon_source_sqlite),
            'border_expand_summary_json': str(border_dir / 'summary.json') if bool(args.polygon_border_expand) else None,
            'endpoint_extend_summary_json': str(interval_dir / 'polygon_endpoint_extend' / 'summary.json') if bool(args.endpoint_extend) else None,
            'filled_pred_sqlite': str(pred_sqlite_path),
            'filled_overlay_metrics_csv': str(exact_metrics_csv),
            'overlay_video': str(overlay_video) if render_overlay else None,
        },
        'timings': timings,
    }


def pipeline_run_ellipse_interval(
    args: argparse.Namespace,
    *,
    tracked_sqlite: Path,
    metrics_csv: Path,
    eval_tracked_sqlite: Path,
    interval_dir: Path,
    interval: int,
    video_path: Path | None,
    k1_cost_csv: Path,
    render_overlay: bool,
    target_ratio_override: float | None=None,
    dense_recall_target_override: float | None=None,
) -> dict[str, object]:
    opt_dir = interval_dir / 'opt'
    exact_dir = interval_dir / 'exact'
    filled_dir = interval_dir / 'filled'
    overlay_video = filled_dir / f'overlay_int_{interval}.mp4'
    ratio = float(target_ratio_override) if target_ratio_override is not None else 1.0 / float(interval)
    dense_recall_target = float(dense_recall_target_override) if dense_recall_target_override is not None else float(args.dense_recall_target)
    timings: dict[str, object] = {}
    optimization_metrics_csv = metrics_csv
    optimization_eval_tracked_sqlite = eval_tracked_sqlite
    endpoint_extend_summary: dict[str, object] | None = None

    if bool(args.endpoint_extend):
        endpoint_dir = interval_dir / 'ellipse_endpoint_extend'
        endpoint_metrics_csv = endpoint_dir / 'metrics_endpoint_extended.csv'
        endpoint_eval_sqlite = endpoint_dir / 'tracked_endpoint_extended.sqlite'
        endpoint_summary_json = endpoint_dir / 'summary.json'
        if not endpoint_metrics_csv.exists() or not endpoint_eval_sqlite.exists() or not endpoint_summary_json.exists() or args.force:
            t0 = time.perf_counter()
            optimization_metrics_csv, optimization_eval_tracked_sqlite = pipeline_endpoint_extend_ellipse_metrics(
                args,
                metrics_csv,
                endpoint_metrics_csv,
                endpoint_summary_json,
                cuts_source_sqlite=eval_tracked_sqlite,
                video_path=video_path,
                eval_sqlite=eval_tracked_sqlite,
                output_eval_sqlite=endpoint_eval_sqlite,
            )
            timings['endpoint_extend'] = {'wall_seconds': float(time.perf_counter() - t0), 'stage': 'post_ellipse_fit_pre_keyframe'}
        else:
            optimization_metrics_csv = endpoint_metrics_csv
            optimization_eval_tracked_sqlite = endpoint_eval_sqlite
            timings['endpoint_extend'] = {'reused': True, 'wall_seconds': 0.0, 'stage': 'post_ellipse_fit_pre_keyframe'}
        endpoint_extend_summary = pipeline_load_json(endpoint_summary_json)
    else:
        timings['endpoint_extend'] = {'skipped': True, 'wall_seconds': 0.0}

    if not (opt_dir / 'summary.json').exists() or args.force:
        opt_cmd = pipeline_subcommand_cmd(
            args.python,
            '__onefile_optimize',
            [
                '--input-metrics-csv', str(optimization_metrics_csv),
                '--output-dir', str(opt_dir),
                '--target-ratio', str(ratio),
                '--solver', 'dp',
                '--keyframe-value-source', 'confidence_blend',
                '--smooth-alpha', '1.0',
                '--value-refine', 'global_ls',
                '--dense-recall-target', str(dense_recall_target),
                '--dense-recall-max-inflate-log', str(args.dense_recall_max_inflate_log),
                '--max-gap', str(args.keyframe_max_gap),
            ],
        )
        timings['optimize'] = pipeline_run_cmd(
            opt_cmd,
            cwd=ROOT,
            progress_interval_sec=float(args.progress_interval_sec),
            progress_frame_count=pipeline_video_frame_count(video_path),
            progress_unit_count=pipeline_count_csv_data_rows(optimization_metrics_csv),
            progress_unit_label='metric_rows',
        )
    else:
        timings['optimize'] = {'reused': True, 'wall_seconds': 0.0}

    if not (exact_dir / 'summary.json').exists() or args.force:
        eval_cmd = pipeline_subcommand_cmd(
            args.python,
            '__onefile_evaluate',
            [
                '--input-union-json', str(opt_dir / 'interpolated_union.json'),
                '--input-tracked-sqlite', str(optimization_eval_tracked_sqlite),
                '--baseline-metrics-csv', str(optimization_metrics_csv),
                '--output-dir', str(exact_dir),
            ],
        )
        timings['evaluate'] = pipeline_run_cmd(
            eval_cmd,
            cwd=ROOT,
            progress_interval_sec=float(args.progress_interval_sec),
            progress_frame_count=pipeline_video_frame_count(video_path),
            progress_unit_count=pipeline_count_csv_data_rows(optimization_metrics_csv),
            progress_unit_label='metric_rows',
        )
    else:
        timings['evaluate'] = {'reused': True, 'wall_seconds': 0.0}

    fill_summary_path = filled_dir / 'fill_summary.json'
    if not fill_summary_path.exists() or args.force:
        fill_cmd = pipeline_subcommand_cmd(
            args.python,
            '__onefile_gap_fill',
            [
                '--input-union-json', str(opt_dir / 'interpolated_union.json'),
                '--input-metrics-csv', str(exact_dir / 'keyframe_exact_metrics.csv'),
                '--output-union-json', str(filled_dir / 'interpolated_union_filled.json'),
                '--output-metrics-csv', str(filled_dir / 'overlay_metrics_filled.csv'),
                '--output-summary-json', str(fill_summary_path),
                '--max-gap', str(args.gap_fill_max_gap),
            ],
        )
        timings['gap_fill'] = pipeline_run_cmd(
            fill_cmd,
            cwd=ROOT,
            progress_interval_sec=float(args.progress_interval_sec),
            progress_frame_count=pipeline_video_frame_count(video_path),
        )
    else:
        timings['gap_fill'] = {'reused': True, 'wall_seconds': 0.0}

    pred_sqlite_path = filled_dir / 'overlay_predictions_filled.sqlite'
    if not pred_sqlite_path.exists() or args.force:
        pred_cmd = pipeline_subcommand_cmd(
            args.python,
            '__onefile_union_to_sqlite',
            [
                '--input-union-json', str(filled_dir / 'interpolated_union_filled.json'),
                '--output-sqlite', str(pred_sqlite_path),
                '--reference-sqlite', str(optimization_eval_tracked_sqlite),
            ],
        )
        timings['union_to_sqlite'] = pipeline_run_cmd(
            pred_cmd,
            cwd=ROOT,
            progress_interval_sec=float(args.progress_interval_sec),
            progress_frame_count=pipeline_video_frame_count(video_path),
        )
    else:
        timings['union_to_sqlite'] = {'reused': True, 'wall_seconds': 0.0}

    if render_overlay:
        timings['render_overlay'] = pipeline_run_render_overlay(
            args,
            video_path=video_path,
            tracked_sqlite=optimization_eval_tracked_sqlite,
            pred_sqlite_path=pred_sqlite_path,
            metrics_csv=filled_dir / 'overlay_metrics_filled.csv',
            k1_cost_csv=k1_cost_csv,
            overlay_video=overlay_video,
        )

    return {
        'interval': int(interval),
        'target_ratio': float(ratio),
        'endpoint_extend_summary': endpoint_extend_summary,
        'opt_summary': pipeline_load_json(opt_dir / 'summary.json'),
        'exact_summary': pipeline_load_json(exact_dir / 'summary.json'),
        'fill_summary': pipeline_load_json(fill_summary_path),
        'paths': {
            'opt_dir': str(opt_dir),
            'exact_dir': str(exact_dir),
            'filled_dir': str(filled_dir),
            'optimization_metrics_csv': str(optimization_metrics_csv),
            'optimization_eval_tracked_sqlite': str(optimization_eval_tracked_sqlite),
            'endpoint_extend_summary_json': str(interval_dir / 'ellipse_endpoint_extend' / 'summary.json') if bool(args.endpoint_extend) else None,
            'filled_union_json': str(filled_dir / 'interpolated_union_filled.json'),
            'filled_overlay_metrics_csv': str(filled_dir / 'overlay_metrics_filled.csv'),
            'filled_pred_sqlite': str(pred_sqlite_path),
            'overlay_video': str(overlay_video) if render_overlay else None,
        },
        'timings': timings,
    }


def pipeline_write_summary(
    *,
    summary_path: Path,
    args: argparse.Namespace,
    video_path: Path | None,
    pipeline_dir: Path,
    inference_output_dir: Path,
    tracked_sqlite: Path,
    predictions_sqlite: Path | None,
    metrics_csv: Path | None,
    k1_cost_csv: Path,
    branch_plan: dict[str, object],
    branch_sqlites: dict[str, str | None],
    policy_groups: list[dict[str, object]],
    group_sqlites: dict[str, str],
    preprocess_timing: dict[str, object],
    preprocess_summary: dict[str, object] | None,
    inference_timing: dict[str, object],
    inference_summary: dict[str, object],
    source_tracked_sqlite: Path,
    intervals: list[int],
    results: dict[str, object],
) -> Path:
    summary = {
        'version': 'full_pipeline_v9_true_standalone',
        'description': 'True standalone single-file pipeline with embedded polygon v22 optimizer, shared ellipse/polygon branches, polygon border expansion, and branch-local pre-keyframe endpoint extension.',
        'input': {
            'input_sqlite': str(args.input_sqlite) if args.input_sqlite is not None else None,
            'input_jsonl': str(args.input_jsonl) if args.input_jsonl is not None else None,
            'input_video': str(video_path) if video_path is not None else None,
        },
        'output_dir': str(pipeline_dir),
        'inference_output_dir': str(inference_output_dir),
        'tracked_sqlite': str(tracked_sqlite),
        'predictions_sqlite': None if predictions_sqlite is None else str(predictions_sqlite),
        'metrics_csv': None if metrics_csv is None else str(metrics_csv),
        'k1_cost_csv': str(k1_cost_csv),
        'branch_plan': branch_plan,
        'branch_sqlites': branch_sqlites,
        'policy_groups': policy_groups,
        'group_sqlites': group_sqlites,
        'preprocess_timing': preprocess_timing,
        'preprocess_summary': preprocess_summary,
        'settings': pipeline_build_settings_summary(args, intervals),
        'inference_timing': inference_timing,
        'inference_summary': inference_summary,
        'source_tracked_sqlite': str(source_tracked_sqlite),
        'interval_results': results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary_path


def pipeline_main() -> None:
    args = pipeline_parse_args()
    pipeline_dir = args.output_dir
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    source_tracked_sqlite, preprocess_timing, preprocess_summary = pipeline_ensure_preprocess(args, pipeline_dir)
    video_path = pipeline_guess_input_video(args)
    tracked_sqlite = source_tracked_sqlite

    branch_plan = pipeline_build_branch_plan(
        tracked_sqlite=tracked_sqlite,
        default_shape_mode=str(args.default_shape_mode),
        class_policy_json=args.class_policy_json,
    )
    track_mode_lookup = {str(row['track_id']): str(row['mode']) for row in branch_plan['tracks']}
    policy_groups = pipeline_build_policy_groups(branch_plan)
    branch_root = pipeline_dir / 'branches'
    branch_sqlites = pipeline_prepare_branch_sqlites(tracked_sqlite, branch_plan, branch_root)
    groups_root = pipeline_dir / 'groups'
    group_sqlites = pipeline_prepare_group_sqlites(tracked_sqlite, policy_groups, groups_root)

    if args.render_overlays and video_path is None:
        raise RuntimeError('Overlay rendering requires --input-video, or a same-stem video next to --input-jsonl.')

    intervals = pipeline_parse_intervals(args.intervals)
    results: dict[str, object] = {}
    any_ellipse_groups = any(str(group['mode']) == 'ellipse' for group in policy_groups)
    any_polygon_groups = any(str(group['mode']) == 'polygon' for group in policy_groups)
    if any_ellipse_groups:
        inference_output_dir, infer_timing, infer_summary = pipeline_ensure_inference(args, pipeline_dir, source_sqlite=tracked_sqlite)
        metrics_csv: Path | None = inference_output_dir / 'k1_exact_k2_v5_metrics.csv'
        group_metrics_csvs: dict[str, Path] = {}
        for group in policy_groups:
            if str(group['mode']) != 'ellipse':
                continue
            group_id = str(group['group_id'])
            subset_csv = groups_root / group_id / 'inference' / f'{group_id}_metrics.csv'
            pipeline_write_subset_metrics_csv(Path(metrics_csv), subset_csv, list(group['track_ids']))
            group_metrics_csvs[group_id] = subset_csv
    else:
        inference_output_dir = pipeline_dir / 'branches' / 'polygon'
        infer_timing = {'reused': True, 'wall_seconds': 0.0, 'mode': 'polygon_passthrough'}
        infer_summary = {
            'description': 'Polygon-only mode bypasses ellipse inference and delegates to the polygon optimizer standalone.',
            'polygon_optimizer': 'embedded_v22',
        }
        metrics_csv = None
        group_metrics_csvs = {}

    if args.k1_cost_csv is not None:
        k1_cost_csv = args.k1_cost_csv
    else:
        zero_k1_cost_csv = pipeline_build_zero_k1_cost_csv_from_sqlite(
            tracked_sqlite,
            pipeline_dir / 'artifacts' / 'full_track_zero_k1_cost.csv',
        )
        if metrics_csv is not None:
            ellipse_k1_cost_csv = pipeline_build_fallback_k1_cost_csv(
                metrics_csv,
                pipeline_dir / 'artifacts' / 'ellipse_branch_k1_cost_lookup.csv',
            )
            if any_polygon_groups:
                k1_cost_csv = pipeline_merge_k1_cost_csvs(
                    [ellipse_k1_cost_csv, zero_k1_cost_csv],
                    pipeline_dir / 'artifacts' / 'mixed_branch_k1_cost_lookup.csv',
                )
            else:
                k1_cost_csv = ellipse_k1_cost_csv
        else:
            k1_cost_csv = zero_k1_cost_csv

    for interval in intervals:
        label = f'int_{interval}'
        group_results: dict[str, object] = {}
        group_target_ratios: dict[str, float] = {}
        merged_input_sqlites: list[Path] = []
        merged_metric_csvs: list[Path] = []
        timings: dict[str, object] = {}

        for group in policy_groups:
            group_id = str(group['group_id'])
            group_mode = str(group['mode'])
            group_interval_dir = groups_root / group_id / 'keyframes' / label
            group_sqlite_path = Path(str(group_sqlites[group_id]))
            target_ratio = pipeline_resolve_group_target_ratio(group, int(interval))
            group_target_ratios[group_id] = float(target_ratio)

            if group_mode == 'ellipse':
                group_result = pipeline_run_ellipse_interval(
                    args,
                    tracked_sqlite=group_sqlite_path,
                    metrics_csv=group_metrics_csvs[group_id],
                    eval_tracked_sqlite=group_sqlite_path,
                    interval_dir=group_interval_dir,
                    interval=int(interval),
                    video_path=video_path,
                    k1_cost_csv=k1_cost_csv,
                    render_overlay=False,
                    target_ratio_override=target_ratio,
                    dense_recall_target_override=pipeline_resolve_group_dense_recall_target(args, group),
                )
                pred_sqlite_path = Path(str(group_result['paths']['filled_pred_sqlite']))
            elif group_mode == 'polygon':
                group_result = pipeline_run_polygon_interval(
                    args,
                    tracked_sqlite=group_sqlite_path,
                    source_sqlite=group_sqlite_path,
                    interval_dir=group_interval_dir,
                    interval=int(interval),
                    video_path=video_path,
                    k1_cost_csv=k1_cost_csv,
                    render_overlay=False,
                    target_ratio_override=target_ratio,
                    recall_min_override=pipeline_resolve_group_polygon_recall_min(args, group),
                )
                pred_sqlite_path = Path(str(group_result['paths']['filled_pred_sqlite']))
            else:
                raise ValueError(f'Unsupported group mode: {group_mode}')

            group_result.setdefault('paths', {})['merged_input_pred_sqlite'] = str(pred_sqlite_path)

            group_results[group_id] = group_result
            merged_input_sqlites.append(pred_sqlite_path)
            group_metrics_csv = dict(group_result.get('paths', {})).get('filled_overlay_metrics_csv')
            if group_metrics_csv:
                merged_metric_csvs.append(Path(str(group_metrics_csv)))
            timings[f'group_{group_id}'] = group_result.get('timings', {})

        merged_dir = pipeline_dir / 'keyframes' / label / 'merged'
        merged_pred_sqlite_path = merged_dir / 'predictions.sqlite'
        if not merged_pred_sqlite_path.exists() or args.force:
            t0 = time.perf_counter()
            pipeline_merge_prediction_sqlites(merged_input_sqlites, merged_pred_sqlite_path, reference_sqlite=tracked_sqlite)
            timings['merge_pred_sqlite'] = {'wall_seconds': float(time.perf_counter() - t0)}
        else:
            timings['merge_pred_sqlite'] = {'reused': True, 'wall_seconds': 0.0}

        t0 = time.perf_counter()
        effective_threshold_edge = int(args.threshold if int(args.threshold_edge) < 0 else args.threshold_edge)
        effective_threshold_edge_norm = float(args.threshold_norm if float(args.threshold_edge_norm) < 0.0 else args.threshold_edge_norm)
        k1_cost_annotation_summary = pipeline_add_k1_cost_columns_to_sqlite(
            merged_pred_sqlite_path,
            reference_sqlite=tracked_sqlite,
            k1_cost_csv=k1_cost_csv,
            threshold_default=int(args.threshold),
            threshold_edge=effective_threshold_edge,
            threshold_default_norm=float(args.threshold_norm),
            threshold_edge_norm=effective_threshold_edge_norm,
            cost_routing=str(args.k1_cost_routing),
        )
        timings['annotate_k1_cost'] = {'wall_seconds': float(time.perf_counter() - t0)}

        t0 = time.perf_counter()
        postprocess_metadata_annotation_summary = pipeline_annotate_postprocess_metadata_to_sqlite(
            merged_pred_sqlite_path,
            merged_metric_csvs,
        )
        timings['annotate_postprocess_metadata'] = {'wall_seconds': float(time.perf_counter() - t0)}

        if bool(args.embed_original_masks):
            t0 = time.perf_counter()
            original_mask_annotation_summary = pipeline_embed_original_masks_for_debug(
                merged_pred_sqlite_path,
                tracked_sqlite,
            )
            timings['embed_original_masks'] = {'wall_seconds': float(time.perf_counter() - t0)}

            t0 = time.perf_counter()
            tracking_pruned_sqlite_path = merged_dir / 'ai_tracking_pruned.sqlite'
            tracking_pruned_debug_summary = pipeline_write_tracking_pruned_sqlite(
                tracking_pruned_sqlite_path,
                tracked_sqlite,
            )
            timings['write_tracking_pruned_sqlite'] = {'wall_seconds': float(time.perf_counter() - t0)}
        else:
            original_mask_annotation_summary = {'enabled': False}
            tracking_pruned_debug_summary = {'enabled': False}
            timings['embed_original_masks'] = {'skipped': True, 'wall_seconds': 0.0}
            timings['write_tracking_pruned_sqlite'] = {'skipped': True, 'wall_seconds': 0.0}

        merged_exact_dir = merged_dir / 'exact'
        merged_exact_summary_path = merged_exact_dir / 'summary.json'
        if not merged_exact_summary_path.exists() or args.force:
            t0 = time.perf_counter()
            merged_exact_summary = pipeline_evaluate_pred_sqlite_exact(
                tracked_sqlite,
                merged_pred_sqlite_path,
                merged_exact_dir,
                track_mode_lookup=track_mode_lookup,
            )
            timings['merged_exact_evaluate'] = {'wall_seconds': float(time.perf_counter() - t0)}
        else:
            merged_exact_summary = pipeline_load_json(merged_exact_summary_path)
            timings['merged_exact_evaluate'] = {'reused': True, 'wall_seconds': 0.0}

        overlay_video = merged_dir / f'overlay_{label}.mp4'
        if args.render_overlays:
            timings['render_overlay'] = pipeline_run_render_overlay(
                args,
                video_path=video_path,
                tracked_sqlite=tracked_sqlite,
                pred_sqlite_path=merged_pred_sqlite_path,
                metrics_csv=merged_exact_dir / 'keyframe_exact_metrics.csv',
                k1_cost_csv=k1_cost_csv,
                overlay_video=overlay_video,
            )

        mode_summary = 'mixed'
        if any_ellipse_groups and not any_polygon_groups:
            mode_summary = 'ellipse'
        elif any_polygon_groups and not any_ellipse_groups:
            mode_summary = 'polygon'

        results[label] = {
            'interval': int(interval),
            'target_ratio': float(1.0 / float(interval)),
            'group_target_ratios': group_target_ratios,
            'mode': mode_summary,
            'group_results': group_results,
            'k1_cost_annotation_summary': k1_cost_annotation_summary,
            'postprocess_metadata_annotation_summary': postprocess_metadata_annotation_summary,
            'original_mask_annotation_summary': original_mask_annotation_summary,
            'tracking_pruned_debug_summary': tracking_pruned_debug_summary,
            'exact_summary': merged_exact_summary,
            'paths': {
                'merged_pred_sqlite': str(merged_pred_sqlite_path),
                'ai_tracking_pruned_sqlite': str(tracking_pruned_sqlite_path) if bool(tracking_pruned_debug_summary.get('enabled', False)) else None,
                'merged_exact_metrics_csv': str(merged_exact_dir / 'keyframe_exact_metrics.csv'),
                'overlay_video': str(overlay_video) if args.render_overlays else None,
            },
            'timings': timings,
        }

    summary_path = pipeline_dir / 'summary.json'
    pipeline_write_summary(
        summary_path=summary_path,
        args=args,
        video_path=video_path,
        pipeline_dir=pipeline_dir,
        inference_output_dir=inference_output_dir,
        tracked_sqlite=tracked_sqlite,
        predictions_sqlite=None,
        metrics_csv=metrics_csv,
        k1_cost_csv=k1_cost_csv,
        branch_plan=branch_plan,
        branch_sqlites=branch_sqlites,
        policy_groups=policy_groups,
        group_sqlites=group_sqlites,
        preprocess_timing=preprocess_timing,
        preprocess_summary=preprocess_summary,
        inference_timing=infer_timing,
        inference_summary=infer_summary,
        source_tracked_sqlite=source_tracked_sqlite,
        intervals=intervals,
        results=results,
    )
    mode_label = 'mixed'
    if any_ellipse_groups and not any_polygon_groups:
        mode_label = 'ellipse'
    elif any_polygon_groups and not any_ellipse_groups:
        mode_label = 'polygon'
    print(json.dumps({'summary_path': str(summary_path), 'intervals': intervals, 'mode': mode_label}, ensure_ascii=False, indent=2))


HIDDEN_SUBCOMMANDS = {
    '__onefile_preprocess': preprocess_main,
    '__onefile_infer': infer_main,
    '__onefile_optimize': kftrackk_main,
    '__onefile_polygon_optimize': polygon_inline_main,
    '__onefile_evaluate': kfeval_main,
    '__onefile_gap_fill': kffill_main,
    '__onefile_union_to_sqlite': union2sqlite_main,
    '__onefile_render': render_main,
}


def dispatch_main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in HIDDEN_SUBCOMMANDS:
        func = HIDDEN_SUBCOMMANDS[sys.argv[1]]
        _run_inline_entrypoint(func, sys.argv[2:])
        return
    pipeline_main()


if __name__ == '__main__':
    dispatch_main()

# The block below is intentionally not executed as part of the pipeline module.
# It is parsed by Python for syntax, then extracted by
# _load_embedded_polygon_optimizer_source() and executed in the
# embedded_polygon_keyframe_v22 namespace only when polygon optimization runs.
if False:
    # --- BEGIN READABLE EMBEDDED POLYGON V22 SOURCE ---

    """Standalone polygon keyframe optimizer v22 for track-first adaptive anchors.

    This file keeps the practical v20/v21 solver core and changes the upstream
    sequence construction:

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

    This v20 variant intentionally removes:

    - multi-candidate per-frame shape proposals
    - interval-synthesized endpoint candidates
    - soft-raster fitting
    - joint keyframe gradient refinement
    - local search
    - polish passes
    - exact-K main solver
    - proxy-fast recall mode

    The goal is a readable one-file implementation of the practical path with
    gapfill-first, track-first fixed anchor counts while preserving the downstream
    fixed-dimension solver inside each contiguous track segment.
    """

    import argparse
    import bisect
    import concurrent.futures
    import csv
    import itertools
    import json
    import math
    import multiprocessing
    import sqlite3
    import time
    from dataclasses import dataclass
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from torch import nn

    ROOT = Path(__file__).resolve().parents[1]

    # Fixed practical-v20 defaults for the raw-only practical implementation.
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

    FEATURE_NAMES = [
        "area",
        "perimeter",
        "bbox_w",
        "bbox_h",
        "area_ratio",
        "compactness",
        "aspect_ratio",
        "extent",
        "solidity",
        "components",
        "holes",
        "eccentricity",
    ]


    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                'Standalone polygon keyframe optimizer v22 with gapfill-first track-level anchor counts. '
                'Input is a SQLite file with masks(frame, track_id, polygons). '
                'Each polygons cell is a JSON array of polygons, where each polygon is [[x, y], ...].'
            )
        )
        parser.add_argument('--input-sqlite', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--target-ratio', type=float, default=1.0 / 9.0)
        parser.add_argument(
            '--anchors-per-contour',
            type=int,
            default=DEFAULT_ANCHORS_PER_CONTOUR,
            help='Fallback or maximum anchors per contour. v21 selects a run-wise value up to this cap.',
        )
        parser.add_argument(
            '--adaptive-anchor-counts',
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
            help='Predict per-frame polygon counts and fix a run-wise anchor count with p90 + offset.',
        )
        parser.add_argument('--point-predictor-model-dir', type=Path, default=DEFAULT_POINT_PREDICTOR_MODEL_DIR)
        parser.add_argument('--predictor-device', type=str, default=DEFAULT_PREDICTOR_DEVICE)
        parser.add_argument('--predictor-batch-size', type=int, default=DEFAULT_PREDICTOR_BATCH_SIZE)
        parser.add_argument('--adaptive-point-quantile', type=float, default=DEFAULT_ADAPTIVE_POINT_QUANTILE)
        parser.add_argument('--adaptive-point-offset', type=int, default=DEFAULT_ADAPTIVE_POINT_OFFSET)
        parser.add_argument('--min-anchors-per-contour', type=int, default=DEFAULT_MIN_ANCHORS_PER_CONTOUR)
        parser.add_argument('--gapfill-enabled', action=argparse.BooleanOptionalAction, default=DEFAULT_GAPFILL_ENABLED)
        parser.add_argument('--gapfill-max-gap', type=int, default=DEFAULT_GAPFILL_MAX_GAP)
        parser.add_argument('--gapfill-temp-points', type=int, default=DEFAULT_GAPFILL_TEMP_POINTS)
        parser.add_argument('--max-run-frames', type=int, default=DEFAULT_MAX_RUN_FRAMES)
        parser.add_argument('--run-overlap-frames', type=int, default=DEFAULT_RUN_OVERLAP_FRAMES)
        parser.add_argument('--recall-min', type=float, default=DEFAULT_RECALL_MIN)
        parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP)
        parser.add_argument('--max-tracks', type=int, default=-1)
        parser.add_argument('--num-workers', type=int, default=1)
        parser.add_argument('--stream-sqlite-rows', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--evaluate-exact', action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--write-pred-sqlite', action=argparse.BooleanOptionalAction, default=True)
        return parser


    def apply_fixed_practical_defaults(args: argparse.Namespace) -> argparse.Namespace:
        args.speed_profile = "practical_v22_gapfill_track_anchor_count"
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
        args.exact_recall_repair_scale_deltas = ','.join(str(v) for v in DEFAULT_EXACT_RECALL_REPAIR_SCALE_DELTAS)
        return args

    @dataclass
    class TrackRow:
        frame: int
        track_id: str
        polygons: list[np.ndarray]
        is_gapfill: bool = False


    @dataclass
    class SimilarityTransform:
        scale: float
        angle_rad: float
        translation: np.ndarray


    @dataclass
    class InstanceRun:
        stream_id: str
        track_id: str
        run_id: int
        frame_numbers: np.ndarray
        gt_polygons: list[list[np.ndarray]]
        anchors: np.ndarray
        contour_count: int
        anchors_per_contour: int
        scale: float
        gapfilled_flags: np.ndarray | None = None
        predicted_total_points: np.ndarray | None = None
        run_target_total_points: int = 0
        emit_start_idx: int = 0
        emit_end_idx: int = -1
        chunk_index: int = 0
        chunk_count: int = 1
        chunk_process_start: int = 0
        chunk_process_end: int = -1
        chunked_from_long_run: bool = False


    @dataclass
    class ShapeCandidate:
        label: str
        vector: np.ndarray
        polygons: list[np.ndarray]
        frame_loss: float
        objective: float
        recall_budget: float = 0.0
        area: float = 0.0
        center: np.ndarray | None = None
        radii: np.ndarray | None = None
        mean_radius: float = 0.0


    @dataclass
    class IntervalCost:
        cost: float
        shape_distance: float
        shape_update: float
        frames_covered: int
        frame_loss_mean: float = 0.0
        shape_distance_scale: float = 1.0
        shape_switch_scale: float = 1.0
        recall_budget: float = 0.0


    @dataclass
    class FrameEvalContext:
        gt_mask: np.ndarray
        gt_area: int
        shift_xy: np.ndarray
        shape_hw: tuple[int, int]
        scale_factor: float
        gt_center: np.ndarray
        gt_radii: np.ndarray
        gt_mean_radius: float
        gt_polygon_area: float
        scratch_pred_mask: np.ndarray | None = None
        scratch_intersection_mask: np.ndarray | None = None


    def compute_mask_descriptors(mask: np.ndarray) -> dict[str, float | int]:
        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        area = float(binary.sum())
        h, w = binary.shape[:2]
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or area <= 0.0:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "bbox_w": 0.0,
                "bbox_h": 0.0,
                "area_ratio": 0.0,
                "compactness": 0.0,
                "aspect_ratio": 1.0,
                "extent": 0.0,
                "solidity": 0.0,
                "components": 0,
                "holes": 0,
                "eccentricity": 0.0,
            }
        outer = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(outer, True))
        x, y, bw, bh = cv2.boundingRect(outer)
        bbox_area = float(max(bw * bh, 1))
        hull = cv2.convexHull(outer)
        hull_area = float(max(cv2.contourArea(hull), 1.0))
        compactness = float((perimeter * perimeter) / max(4.0 * math.pi * area, 1e-6))
        ys, xs = np.nonzero(binary)
        if len(xs) >= 2:
            pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            centered = pts - pts.mean(axis=0, keepdims=True)
            cov = np.cov(centered.T)
            eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-6))[::-1]
            eccentricity = float(np.sqrt(max(0.0, 1.0 - float(eigvals[1] / eigvals[0]))))
        else:
            eccentricity = 0.0
        component_count = 0
        hole_count = 0
        if hierarchy is not None:
            for node in hierarchy[0]:
                parent = int(node[3])
                if parent < 0:
                    component_count += 1
                else:
                    hole_count += 1
        return {
            "area": area,
            "perimeter": perimeter,
            "bbox_w": float(bw),
            "bbox_h": float(bh),
            "area_ratio": float(area / max(h * w, 1)),
            "compactness": compactness,
            "aspect_ratio": float(max(bw, 1) / max(bh, 1)),
            "extent": float(area / bbox_area),
            "solidity": float(area / hull_area),
            "components": int(component_count),
            "holes": int(hole_count),
            "eccentricity": eccentricity,
        }


    def resize_mask_with_padding(mask: np.ndarray, image_size: int) -> np.ndarray:
        height, width = mask.shape[:2]
        scale = float(image_size) / float(max(height, width, 1))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        offset_y = (image_size - new_h) // 2
        offset_x = (image_size - new_w) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        return canvas


    def build_feature_vector(descriptors: dict[str, float | int], means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        values = np.asarray(
            [
                math.log1p(float(descriptors["area"])),
                math.log1p(float(descriptors["perimeter"])),
                math.log1p(float(descriptors["bbox_w"])),
                math.log1p(float(descriptors["bbox_h"])),
                float(descriptors["area_ratio"]),
                float(descriptors["compactness"]),
                math.log1p(float(descriptors["aspect_ratio"])),
                float(descriptors["extent"]),
                float(descriptors["solidity"]),
                float(descriptors["components"]),
                float(descriptors["holes"]),
                float(descriptors["eccentricity"]),
            ],
            dtype=np.float32,
        )
        return (values - means) / stds


    class ConvBNAct(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class TinyMaskPointNet(nn.Module):
        def __init__(
            self,
            *,
            feature_dim: int,
            num_classes: int,
            use_feature_branch: bool,
            width_mult: float = 1.0,
            feature_hidden_dim: int = 32,
            head_hidden_dim: int = 64,
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            self.use_feature_branch = bool(use_feature_branch)

            def ch(value: int) -> int:
                return max(8, int(round(float(value) * float(width_mult))))

            stem_ch = ch(16)
            c1 = ch(24)
            c2 = ch(32)
            c3 = ch(48)
            c4 = ch(64)
            self.stem = ConvBNAct(1, stem_ch, stride=2)
            self.encoder = nn.Sequential(
                ConvBNAct(stem_ch, c1, stride=2),
                ConvBNAct(c1, c2, stride=2),
                ConvBNAct(c2, c3, stride=2),
                ConvBNAct(c3, c4, stride=2),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.image_head = nn.Sequential(
                nn.Linear(c4, head_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=float(dropout)),
            )
            if self.use_feature_branch:
                self.feature_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_hidden_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(feature_hidden_dim, feature_hidden_dim),
                    nn.SiLU(inplace=True),
                )
                fusion_dim = head_hidden_dim + feature_hidden_dim
            else:
                self.feature_head = None
                fusion_dim = head_hidden_dim
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(head_hidden_dim, num_classes),
            )

        def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            x = self.stem(image)
            x = self.encoder(x)
            x = self.pool(x).flatten(1)
            x = self.image_head(x)
            if self.use_feature_branch:
                assert self.feature_head is not None
                x = torch.cat([x, self.feature_head(features)], dim=1)
            return self.classifier(x)


    class LearnedPointPredictor:
        def __init__(self, model_dir: Path, device_name: str) -> None:
            ckpt_path = Path(model_dir) / "best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            run_config = checkpoint["run_config"]
            self.label_min = int(run_config["label_min"])
            self.label_max = int(run_config["label_max"])
            self.image_size = int(run_config["image_size"])
            self.use_feature_branch = bool(run_config["feature_branch"])
            self.feature_means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
            self.feature_stds = np.asarray(checkpoint["feature_stds"], dtype=np.float32)
            model = TinyMaskPointNet(
                feature_dim=len(FEATURE_NAMES),
                num_classes=self.label_max - self.label_min + 1,
                use_feature_branch=self.use_feature_branch,
                width_mult=float(run_config.get("width_mult", 1.0)),
                feature_hidden_dim=int(run_config.get("feature_hidden_dim", 32)),
                head_hidden_dim=int(run_config.get("head_hidden_dim", 64)),
                dropout=float(run_config.get("dropout", 0.10)),
            )
            model.load_state_dict(checkpoint["model"])
            if str(device_name).startswith("cuda") and torch.cuda.is_available():
                self.device = torch.device(str(device_name))
            else:
                self.device = torch.device("cpu")
            self.model = model.to(self.device).eval()

        def predict_total_points_batch(
            self,
            masks: list[np.ndarray],
            descriptors_list: list[dict[str, float | int]],
            batch_size: int,
        ) -> list[int]:
            outputs: list[int] = []
            batch_size = max(1, int(batch_size))
            for start in range(0, len(masks), batch_size):
                end = min(start + batch_size, len(masks))
                image_list: list[np.ndarray] = []
                feature_list: list[np.ndarray] = []
                for mask, descriptors in zip(masks[start:end], descriptors_list[start:end], strict=False):
                    resized = resize_mask_with_padding((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, self.image_size)
                    image_list.append((resized.astype(np.float32) / 255.0)[None, :, :])
                    if self.use_feature_branch:
                        feature_list.append(build_feature_vector(descriptors, self.feature_means, self.feature_stds))
                    else:
                        feature_list.append(np.zeros((len(FEATURE_NAMES),), dtype=np.float32))
                images = torch.from_numpy(np.asarray(image_list, dtype=np.float32)).to(self.device)
                features = torch.from_numpy(np.asarray(feature_list, dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    logits = self.model(images, features)
                    pred_indices = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
                outputs.extend(int(idx + self.label_min) for idx in pred_indices.tolist())
            return outputs


    def normalize_closed_points(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) <= 1:
            return pts.copy()
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return pts.astype(np.float32, copy=True)


    def parse_polygons(polygons_json: str) -> list[np.ndarray]:
        polygons = json.loads(str(polygons_json))
        out: list[np.ndarray] = []
        for poly in polygons:
            arr = normalize_closed_points(np.asarray(poly, dtype=np.float32).reshape(-1, 2))
            if len(arr) >= 3:
                out.append(arr)
        return out


    def signed_area(poly: np.ndarray) -> float:
        pts = normalize_closed_points(poly)
        if len(pts) < 3:
            return 0.0
        xs = pts[:, 0]
        ys = pts[:, 1]
        return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


    def polygon_area(poly: np.ndarray) -> float:
        return abs(signed_area(poly))


    def orient_ccw(poly: np.ndarray) -> np.ndarray:
        pts = normalize_closed_points(poly)
        if len(pts) < 3:
            return pts
        if signed_area(pts) < 0.0:
            return pts[::-1].copy()
        return pts


    def contour_centroid(poly: np.ndarray) -> np.ndarray:
        pts = normalize_closed_points(poly)
        if len(pts) == 0:
            return np.zeros((2,), dtype=np.float32)
        return np.mean(pts, axis=0).astype(np.float32)


    def sort_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
        normalized = [orient_ccw(poly) for poly in polygons if len(normalize_closed_points(poly)) >= 3]
        normalized.sort(key=lambda poly: (polygon_area(poly), len(poly)), reverse=True)
        return normalized


    def cyclic_shift_points(poly: np.ndarray, shift: int) -> np.ndarray:
        pts = normalize_closed_points(poly)
        if len(pts) == 0:
            return pts
        return np.roll(pts, -int(shift), axis=0)


    def align_polygon_phase(reference: np.ndarray | None, poly: np.ndarray) -> np.ndarray:
        candidate = orient_ccw(poly)
        if reference is None or len(reference) != len(candidate):
            return candidate
        best = candidate
        best_score = float("inf")
        for variant in (candidate, candidate[::-1].copy()):
            for shift in range(len(variant)):
                rolled = cyclic_shift_points(variant, shift)
                score = float(np.mean(np.sum((rolled - reference) ** 2, axis=1)))
                if score < best_score:
                    best_score = score
                    best = rolled
        return best


    def resample_closed_contour(poly: np.ndarray, n_points: int) -> np.ndarray:
        pts = normalize_closed_points(poly)
        n_points = max(3, int(n_points))
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(pts) == 1:
            return np.repeat(pts, n_points, axis=0).astype(np.float32)
        nxt = np.roll(pts, -1, axis=0)
        seg_lens = np.linalg.norm(nxt - pts, axis=1)
        total = float(seg_lens.sum())
        if total <= 1e-6:
            return np.repeat(pts[:1], n_points, axis=0).astype(np.float32)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
        sample_pos = np.linspace(0.0, total, n_points, endpoint=False, dtype=np.float64)
        out = np.zeros((n_points, 2), dtype=np.float32)
        for idx, dist in enumerate(sample_pos):
            seg_idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
            seg_idx = max(0, min(seg_idx, len(pts) - 1))
            seg_start = cumulative[seg_idx]
            seg_len = max(float(seg_lens[seg_idx]), 1e-6)
            alpha = float((dist - seg_start) / seg_len)
            out[idx] = ((1.0 - alpha) * pts[seg_idx] + alpha * nxt[seg_idx]).astype(np.float32)
        return out


    def align_contour_slots(prev: list[np.ndarray] | None, current: list[np.ndarray]) -> list[np.ndarray]:
        current_sorted = sort_polygons(current)
        if prev is None or len(prev) != len(current_sorted):
            return current_sorted
        count = len(current_sorted)
        best_perm = list(range(count))
        best_cost = float("inf")
        prev_centroids = [contour_centroid(poly) for poly in prev]
        prev_areas = [polygon_area(poly) for poly in prev]
        curr_centroids = [contour_centroid(poly) for poly in current_sorted]
        curr_areas = [polygon_area(poly) for poly in current_sorted]
        for perm in itertools.permutations(range(count)):
            cost = 0.0
            for idx, src in enumerate(perm):
                center_term = float(np.linalg.norm(prev_centroids[idx] - curr_centroids[src]))
                area_term = abs(math.log(max(curr_areas[src], 1e-6) / max(prev_areas[idx], 1e-6)))
                cost += center_term + 8.0 * area_term
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        return [current_sorted[idx] for idx in best_perm]


    def build_local_mask_from_polygons(polygons: list[np.ndarray]) -> np.ndarray:
        valid_polys = [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in polygons if len(poly) >= 3]
        if not valid_polys:
            return np.zeros((1, 1), dtype=np.uint8)
        all_pts = np.concatenate(valid_polys, axis=0)
        min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
        max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
        shift_xy = min_xy.astype(np.float32)
        shape = (int(max_xy[1] - min_xy[1] + 1), int(max_xy[0] - min_xy[0] + 1))
        mask = np.zeros(shape, dtype=np.uint8)
        for poly in valid_polys:
            pts_i32 = np.round(poly - shift_xy[None, :]).astype(np.int32)
            if len(pts_i32) >= 3:
                cv2.fillPoly(mask, [pts_i32], 1)
        return mask


    def interpolate_gapfill_polygons(
        left_slots: list[np.ndarray],
        right_slots: list[np.ndarray],
        *,
        step: int,
        gap: int,
        temp_points: int,
    ) -> list[np.ndarray]:
        alpha = float(step) / float(gap + 1)
        out: list[np.ndarray] = []
        for left_poly, right_poly in zip(left_slots, right_slots, strict=False):
            left_anchor = resample_closed_contour(orient_ccw(left_poly), int(temp_points))
            right_anchor = resample_closed_contour(orient_ccw(right_poly), int(temp_points))
            right_anchor = align_polygon_phase(left_anchor, right_anchor)
            interp = ((1.0 - alpha) * left_anchor + alpha * right_anchor).astype(np.float32)
            out.append(interp)
        return out


    def build_track_segments_with_gapfill(
        rows: list[TrackRow],
        *,
        max_gap: int,
        temp_points: int,
    ) -> tuple[list[list[TrackRow]], dict[str, int]]:
        by_track: dict[str, list[TrackRow]] = {}
        for row in rows:
            by_track.setdefault(row.track_id, []).append(row)
        segments: list[list[TrackRow]] = []
        stats = {
            "source_tracks": int(len(by_track)),
            "source_rows": int(len(rows)),
            "gapfill_inserted_frames": 0,
            "gapfill_events": 0,
            "hard_split_events": 0,
        }
        for track_id, track_rows in sorted(by_track.items(), key=lambda item: int(item[0])):
            track_rows.sort(key=lambda row: row.frame)
            current_segment: list[TrackRow] = []
            prev: TrackRow | None = None
            for row in track_rows:
                current_slots = sort_polygons(row.polygons)
                if prev is None:
                    current_segment = [
                        TrackRow(
                            frame=int(row.frame),
                            track_id=str(row.track_id),
                            polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                            is_gapfill=bool(row.is_gapfill),
                        )
                    ]
                    prev = current_segment[-1]
                    continue

                prev_slots = sort_polygons(prev.polygons)
                same_contour_count = len(prev_slots) == len(current_slots)
                gap = int(row.frame) - int(prev.frame) - 1
                if same_contour_count:
                    current_slots = align_contour_slots(prev_slots, current_slots)

                if gap <= 0 and same_contour_count:
                    current_segment.append(
                        TrackRow(
                            frame=int(row.frame),
                            track_id=str(row.track_id),
                            polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                            is_gapfill=bool(row.is_gapfill),
                        )
                    )
                    prev = current_segment[-1]
                    continue

                can_gapfill = same_contour_count and gap > 0 and gap <= int(max_gap)
                if can_gapfill:
                    for step in range(1, gap + 1):
                        interp_polys = interpolate_gapfill_polygons(
                            prev_slots,
                            current_slots,
                            step=step,
                            gap=gap,
                            temp_points=int(temp_points),
                        )
                        current_segment.append(
                            TrackRow(
                                frame=int(prev.frame) + step,
                                track_id=str(track_id),
                                polygons=[np.asarray(poly, dtype=np.float32) for poly in interp_polys],
                                is_gapfill=True,
                            )
                        )
                    stats["gapfill_events"] += 1
                    stats["gapfill_inserted_frames"] += int(gap)
                    current_segment.append(
                        TrackRow(
                            frame=int(row.frame),
                            track_id=str(row.track_id),
                            polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                            is_gapfill=bool(row.is_gapfill),
                        )
                    )
                    prev = current_segment[-1]
                    continue

                if current_segment:
                    segments.append(current_segment)
                stats["hard_split_events"] += 1
                current_segment = [
                    TrackRow(
                        frame=int(row.frame),
                        track_id=str(row.track_id),
                        polygons=[np.asarray(poly, dtype=np.float32) for poly in sort_polygons(row.polygons)],
                        is_gapfill=bool(row.is_gapfill),
                    )
                ]
                prev = current_segment[-1]

            if current_segment:
                segments.append(current_segment)
        stats["segment_count"] = int(len(segments))
        return segments, stats


    def load_rows(sqlite_path: Path) -> list[TrackRow]:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            raw_rows = conn.execute("SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame").fetchall()
        finally:
            conn.close()
        rows: list[TrackRow] = []
        for frame, track_id, polygons_json in raw_rows:
            rows.append(TrackRow(frame=int(frame), track_id=str(track_id), polygons=parse_polygons(str(polygons_json))))
        return rows


    def rasterize_mask_from_polygons(
        polygons: list[np.ndarray],
        shape: tuple[int, int],
        shift_xy: np.ndarray,
    ) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return np.zeros((0, 0), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        shift = np.asarray(shift_xy, dtype=np.float32)
        for poly in polygons:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts_i32 = np.round(pts - shift[None, :]).astype(np.int32)
            cv2.fillPoly(mask, [pts_i32], 1)
        return mask


    def _rotation_matrix(angle_rad: float) -> np.ndarray:
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        return np.asarray([[c, -s], [s, c]], dtype=np.float64)


    def apply_similarity_transform(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rot = _rotation_matrix(float(transform.angle_rad))
        out = float(transform.scale) * (pts @ rot.T) + np.asarray(transform.translation, dtype=np.float64)
        return out.astype(np.float32)


    def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
        src_pts = np.asarray(src, dtype=np.float64).reshape(-1, 2)
        dst_pts = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
        if len(src_pts) == 0 or len(dst_pts) == 0:
            return SimilarityTransform(scale=1.0, angle_rad=0.0, translation=np.zeros((2,), dtype=np.float64))
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        src_var = float(np.sum(src_centered ** 2) / max(len(src_pts), 1))
        if len(src_pts) < 2 or src_var <= 1e-9:
            return SimilarityTransform(scale=1.0, angle_rad=0.0, translation=(dst_mean - src_mean).astype(np.float64))
        cov = (dst_centered.T @ src_centered) / float(len(src_pts))
        u, singular_vals, vt = np.linalg.svd(cov)
        sign_fix = np.eye(2, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0.0:
            sign_fix[-1, -1] = -1.0
        rot = u @ sign_fix @ vt
        scale = float(np.trace(np.diag(singular_vals) @ sign_fix) / max(src_var, 1e-9))
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        translation = dst_mean - scale * (rot @ src_mean)
        angle_rad = float(math.atan2(rot[1, 0], rot[0, 0]))
        return SimilarityTransform(scale=scale, angle_rad=angle_rad, translation=translation.astype(np.float64))


    def similarity_residuals(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, SimilarityTransform]:
        transform = estimate_similarity_transform(src, dst)
        aligned_src = apply_similarity_transform(src, transform)
        residual = np.asarray(dst, dtype=np.float64) - np.asarray(aligned_src, dtype=np.float64)
        return residual.astype(np.float32), transform


    def compute_exact_metrics_from_polygons(gt_polys: list[np.ndarray], pred_polys: list[np.ndarray]) -> dict[str, float]:
        if not gt_polys and not pred_polys:
            return {
                "gt_area": 0.0,
                "pred_area": 0.0,
                "intersection": 0.0,
                "union": 0.0,
                "recall": 1.0,
                "precision": 1.0,
                "iou": 1.0,
            }
        all_polys = [np.asarray(poly, dtype=np.float32) for poly in gt_polys + pred_polys if len(poly) >= 3]
        if not all_polys:
            return {
                "gt_area": 0.0,
                "pred_area": 0.0,
                "intersection": 0.0,
                "union": 0.0,
                "recall": 1.0,
                "precision": 1.0,
                "iou": 1.0,
            }
        all_pts = np.concatenate(all_polys, axis=0)
        min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
        max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
        shift_xy = min_xy.astype(np.float32)
        shape = (int(max_xy[1] - min_xy[1] + 1), int(max_xy[0] - min_xy[0] + 1))
        gt_mask = rasterize_mask_from_polygons(gt_polys, shape, shift_xy)
        pred_mask = rasterize_mask_from_polygons(pred_polys, shape, shift_xy)
        gt_area = int(gt_mask.sum())
        pred_area = int(pred_mask.sum())
        intersection = int((gt_mask & pred_mask).sum())
        union = int(gt_area + pred_area - intersection)
        recall = intersection / gt_area if gt_area > 0 else 1.0
        precision = intersection / pred_area if pred_area > 0 else 1.0
        iou = intersection / union if union > 0 else 1.0
        return {
            "gt_area": float(gt_area),
            "pred_area": float(pred_area),
            "intersection": float(intersection),
            "union": float(union),
            "recall": float(recall),
            "precision": float(precision),
            "iou": float(iou),
        }


    def compute_weighted_error(metrics: dict[str, float]) -> int:
        fn_pixels = int(round(float(metrics["gt_area"]) - float(metrics["intersection"])))
        fp_pixels = int(round(float(metrics["pred_area"]) - float(metrics["intersection"])))
        return int(2 * fn_pixels + fp_pixels)


    def write_csv(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


    def union_rows_to_pred_sqlite(union_rows: list[dict[str, object]], output_sqlite: Path) -> None:
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        conn = sqlite3.connect(str(output_sqlite))
        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
            for row in union_rows:
                cur.execute(
                    "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                    (int(row["frame"]), str(row["track_id"]), json.dumps(row["polygons"], ensure_ascii=False)),
                )
            conn.commit()
        finally:
            conn.close()


    def aggregate_exact_rows(rows: list[dict[str, object]]) -> dict[str, float]:
        gt_area = sum(float(row["gt_area"]) for row in rows)
        pred_area = sum(float(row["pred_area"]) for row in rows)
        intersection = sum(float(row["intersection"]) for row in rows)
        union = sum(float(row["union"]) for row in rows)
        weighted_error = sum(float(row["weighted_error"]) for row in rows)
        mean_recall = float(np.mean(np.asarray([float(row["recall"]) for row in rows], dtype=np.float64))) if rows else 1.0
        mean_precision = float(np.mean(np.asarray([float(row["precision"]) for row in rows], dtype=np.float64))) if rows else 1.0
        mean_iou = float(np.mean(np.asarray([float(row["iou"]) for row in rows], dtype=np.float64))) if rows else 1.0
        return {
            "row_count": float(len(rows)),
            "gt_area": float(gt_area),
            "pred_area": float(pred_area),
            "intersection": float(intersection),
            "union": float(union),
            "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
            "global_precision": float(intersection / pred_area) if pred_area > 0 else 1.0,
            "global_iou": float(intersection / union) if union > 0 else 1.0,
            "mean_recall": float(mean_recall),
            "mean_precision": float(mean_precision),
            "mean_iou": float(mean_iou),
            "weighted_error_total": float(weighted_error),
            "weighted_error_mean": float(weighted_error / max(len(rows), 1)),
        }


    def evaluate_union_exact(union_rows: list[dict[str, object]], tracked_sqlite: Path, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_lookup = {(int(row["frame"]), str(row["track_id"])): row for row in union_rows}
        result_rows: list[dict[str, object]] = []
        conn = sqlite3.connect(str(tracked_sqlite))
        try:
            cur = conn.cursor()
            for frame, track_id, polygons_json in cur.execute("SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)"):
                key = (int(frame), str(track_id))
                pred = pred_lookup.get(key)
                if pred is None:
                    continue
                gt_polys = parse_polygons(str(polygons_json))
                pred_polys = [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in pred["polygons"]]
                metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
                weighted_error = float(compute_weighted_error(metrics))
                result_rows.append(
                    {
                        "frame": int(frame),
                        "track_id": str(track_id),
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
        finally:
            conn.close()
        metrics_csv = output_dir / "keyframe_exact_metrics.csv"
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
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
            writer.writerows(sorted(result_rows, key=lambda row: (int(row["frame"]), int(str(row["track_id"])))))
        summary = {
            "input_tracked_sqlite": str(tracked_sqlite),
            "optimized": aggregate_exact_rows(result_rows),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


    def parse_float_list(text: str, default: list[float]) -> list[float]:
        values: list[float] = []
        for token in str(text).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        if not values:
            values = list(default)
        return sorted(set(float(v) for v in values))


    def split_long_track_segments(
        segments: list[list[TrackRow]],
        max_run_frames: int,
        run_overlap_frames: int,
    ) -> tuple[list[list[TrackRow]], dict[int, dict[str, int]], dict[str, int]]:
        source_lengths = [int(len(segment)) for segment in segments]
        max_source_segment_frames = int(max(source_lengths, default=0))
        max_frames = int(max_run_frames)
        requested_overlap = max(0, int(run_overlap_frames))
        disabled = max_frames <= 0
        effective_overlap = 0 if disabled else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
        emit_stride = 0 if disabled else int(max(1, max_frames - 2 * effective_overlap))

        def make_stats(
            *,
            processed_segment_count: int,
            long_segment_count: int,
            chunked_source_segment_count: int,
            chunk_output_segment_count: int,
            max_processed_segment_frames: int,
            overlap_added_rows: int,
        ) -> dict[str, int]:
            return {
                "max_run_frames": int(max_frames),
                "run_overlap_frames": int(effective_overlap),
                "source_segment_count": int(len(segments)),
                "processed_segment_count": int(processed_segment_count),
                "long_segment_count": int(long_segment_count),
                "chunked_source_segment_count": int(chunked_source_segment_count),
                "chunk_output_segment_count": int(chunk_output_segment_count),
                "max_source_segment_frames": int(max_source_segment_frames),
                "max_processed_segment_frames": int(max_processed_segment_frames),
                "emit_stride_frames": int(emit_stride),
                "overlap_added_rows": int(overlap_added_rows),
            }

        if disabled or max_source_segment_frames <= max_frames:
            return segments, {}, make_stats(
                processed_segment_count=len(segments),
                long_segment_count=sum(1 for length in source_lengths if max_frames > 0 and length > max_frames),
                chunked_source_segment_count=0,
                chunk_output_segment_count=0,
                max_processed_segment_frames=max_source_segment_frames,
                overlap_added_rows=0,
            )

        split_segments: list[list[TrackRow]] = []
        segment_meta: dict[int, dict[str, int]] = {}
        chunked_source_segment_count = 0
        chunk_output_segment_count = 0
        overlap_added_rows = 0
        max_processed_segment_frames = 0

        for source_run_id, segment in enumerate(segments):
            length = int(len(segment))
            if length <= max_frames:
                split_segments.append(segment)
                max_processed_segment_frames = max(max_processed_segment_frames, length)
                continue

            chunk_ranges: list[tuple[int, int, int, int]] = []
            for emit_start in range(0, length, emit_stride):
                emit_end = int(min(length, emit_start + emit_stride))
                if emit_start >= emit_end:
                    continue
                process_start = int(max(0, emit_start - effective_overlap))
                process_end = int(min(length, emit_end + effective_overlap))
                chunk_ranges.append((process_start, process_end, int(emit_start), emit_end))

            chunk_count = int(len(chunk_ranges))
            if chunk_count <= 1:
                split_segments.append(segment)
                max_processed_segment_frames = max(max_processed_segment_frames, length)
                continue

            chunked_source_segment_count += 1
            chunk_output_segment_count += chunk_count
            for chunk_index, (process_start, process_end, emit_start, emit_end) in enumerate(chunk_ranges):
                chunk_rows = list(segment[process_start:process_end])
                split_segments.append(chunk_rows)
                segment_meta[id(chunk_rows)] = {
                    "source_run_id": int(source_run_id),
                    "chunk_index": int(chunk_index),
                    "chunk_count": int(chunk_count),
                    "process_start": int(process_start),
                    "process_end": int(process_end),
                    "emit_start": int(emit_start - process_start),
                    "emit_end": int(emit_end - process_start),
                }
                processed_len = int(process_end - process_start)
                emitted_len = int(emit_end - emit_start)
                max_processed_segment_frames = max(max_processed_segment_frames, processed_len)
                overlap_added_rows += max(0, processed_len - emitted_len)

        return split_segments, segment_meta, make_stats(
            processed_segment_count=len(split_segments),
            long_segment_count=sum(1 for length in source_lengths if length > max_frames),
            chunked_source_segment_count=chunked_source_segment_count,
            chunk_output_segment_count=chunk_output_segment_count,
            max_processed_segment_frames=max_processed_segment_frames,
            overlap_added_rows=overlap_added_rows,
        )


    def build_track_streams(
        rows: list[TrackRow],
        anchors_per_contour: int,
        predictor: LearnedPointPredictor | None = None,
        predictor_batch_size: int = DEFAULT_PREDICTOR_BATCH_SIZE,
        adaptive_anchor_counts: bool = DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
        adaptive_point_quantile: float = DEFAULT_ADAPTIVE_POINT_QUANTILE,
        adaptive_point_offset: int = DEFAULT_ADAPTIVE_POINT_OFFSET,
        min_anchors_per_contour: int = DEFAULT_MIN_ANCHORS_PER_CONTOUR,
        gapfill_enabled: bool = DEFAULT_GAPFILL_ENABLED,
        gapfill_max_gap: int = DEFAULT_GAPFILL_MAX_GAP,
        gapfill_temp_points: int = DEFAULT_GAPFILL_TEMP_POINTS,
        max_tracks: int = -1,
        max_run_frames: int = DEFAULT_MAX_RUN_FRAMES,
        run_overlap_frames: int = DEFAULT_RUN_OVERLAP_FRAMES,
    ) -> tuple[list[InstanceRun], dict[str, int]]:
        if max_tracks > 0:
            counts: dict[str, int] = {}
            for row in rows:
                counts[row.track_id] = counts.get(row.track_id, 0) + 1
            allowed_tracks = [track_id for track_id, _count in sorted(counts.items(), key=lambda item: (-item[1], int(item[0])))][
                : int(max_tracks)
            ]
            allowed = set(allowed_tracks)
            rows = [row for row in rows if row.track_id in allowed]

        if bool(gapfill_enabled):
            segments, segmentation_stats = build_track_segments_with_gapfill(
                rows,
                max_gap=int(gapfill_max_gap),
                temp_points=int(gapfill_temp_points),
            )
        else:
            segments = []
            current: list[TrackRow] = []
            prev: TrackRow | None = None
            for row in rows:
                split = prev is None or row.track_id != prev.track_id or row.frame != prev.frame + 1 or len(row.polygons) != len(prev.polygons)
                if split:
                    if current:
                        segments.append(current)
                    current = [TrackRow(frame=row.frame, track_id=row.track_id, polygons=row.polygons, is_gapfill=row.is_gapfill)]
                else:
                    current.append(TrackRow(frame=row.frame, track_id=row.track_id, polygons=row.polygons, is_gapfill=row.is_gapfill))
                prev = row
            if current:
                segments.append(current)
            segmentation_stats = {
                "source_tracks": int(len({row.track_id for row in rows})),
                "source_rows": int(len(rows)),
                "gapfill_inserted_frames": 0,
                "gapfill_events": 0,
                "hard_split_events": 0,
                "segment_count": int(len(segments)),
            }

        segments, segment_meta, split_stats = split_long_track_segments(
            segments,
            max_run_frames=int(max_run_frames),
            run_overlap_frames=int(run_overlap_frames),
        )
        segmentation_stats.update(split_stats)

        streams: list[InstanceRun] = []
        for run_id, run_rows in enumerate(segments):
            meta = segment_meta.get(
                id(run_rows),
                {
                    "source_run_id": int(run_id),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "process_start": 0,
                    "process_end": int(len(run_rows)),
                    "emit_start": 0,
                    "emit_end": int(len(run_rows)),
                },
            )
            source_run_id = int(meta["source_run_id"])
            chunk_index = int(meta["chunk_index"])
            chunk_count = int(meta["chunk_count"])
            chunk_suffix = f":chunk{chunk_index + 1}of{chunk_count}" if chunk_count > 1 else ""
            aligned_rows: list[list[np.ndarray]] = []
            gapfilled_flags: list[bool] = []
            prev_slots: list[np.ndarray] | None = None
            for row in run_rows:
                slots = align_contour_slots(prev_slots, row.polygons)
                aligned_rows.append(slots)
                gapfilled_flags.append(bool(row.is_gapfill))
                prev_slots = slots
            contour_count = len(aligned_rows[0]) if aligned_rows else 0
            if contour_count <= 0:
                continue

            predicted_total_points: np.ndarray | None = None
            run_anchor_count = int(anchors_per_contour)
            run_target_total_points = int(contour_count * run_anchor_count)
            if bool(adaptive_anchor_counts) and predictor is not None:
                masks = [build_local_mask_from_polygons(slots) for slots in aligned_rows]
                descriptors_list = [compute_mask_descriptors(mask) for mask in masks]
                predicted_totals = predictor.predict_total_points_batch(
                    masks,
                    descriptors_list,
                    batch_size=int(predictor_batch_size),
                )
                predicted_total_points = np.asarray(predicted_totals, dtype=np.int32)
                quantile_total = int(math.ceil(float(np.quantile(predicted_total_points.astype(np.float64), float(adaptive_point_quantile)))))
                run_target_total_points = int(max(contour_count * int(min_anchors_per_contour), quantile_total + int(adaptive_point_offset)))
                run_anchor_count = int(math.ceil(run_target_total_points / max(contour_count, 1)))
                run_anchor_count = int(np.clip(run_anchor_count, int(min_anchors_per_contour), int(anchors_per_contour)))
                run_target_total_points = int(run_anchor_count * contour_count)

            frame_anchor_stack: list[np.ndarray] = []
            frame_polygons: list[list[np.ndarray]] = []
            frame_areas: list[float] = []
            prev_anchors_by_slot: list[np.ndarray | None] = [None] * contour_count
            for slots in aligned_rows:
                contour_anchors: list[np.ndarray] = []
                contour_polygons: list[np.ndarray] = []
                area_sum = 0.0
                for slot_id in range(contour_count):
                    poly = np.asarray(orient_ccw(slots[slot_id]), dtype=np.float32)
                    anchor = resample_closed_contour(poly, int(run_anchor_count))
                    anchor = align_polygon_phase(prev_anchors_by_slot[slot_id], anchor)
                    contour_anchors.append(np.asarray(anchor, dtype=np.float32))
                    contour_polygons.append(np.asarray(poly, dtype=np.float32))
                    area_sum += float(polygon_area(poly))
                    prev_anchors_by_slot[slot_id] = np.asarray(anchor, dtype=np.float32)
                frame_anchor_stack.append(np.asarray(contour_anchors, dtype=np.float32))
                frame_polygons.append(contour_polygons)
                frame_areas.append(area_sum)
            scale = float(max(math.sqrt(max(float(np.median(np.asarray(frame_areas, dtype=np.float64))), 1.0)), 1.0))
            streams.append(
                InstanceRun(
                    stream_id=f"{run_rows[0].track_id}:run{source_run_id}{chunk_suffix}:instance",
                    track_id=run_rows[0].track_id,
                    run_id=source_run_id,
                    frame_numbers=np.asarray([row.frame for row in run_rows], dtype=np.int32),
                    gt_polygons=frame_polygons,
                    anchors=np.asarray(frame_anchor_stack, dtype=np.float32),
                    contour_count=contour_count,
                    anchors_per_contour=int(run_anchor_count),
                    scale=scale,
                    gapfilled_flags=np.asarray(gapfilled_flags, dtype=np.uint8),
                    predicted_total_points=predicted_total_points,
                    run_target_total_points=int(run_target_total_points),
                    emit_start_idx=int(meta["emit_start"]),
                    emit_end_idx=int(meta["emit_end"]),
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    chunk_process_start=int(meta["process_start"]),
                    chunk_process_end=int(meta["process_end"]),
                    chunked_from_long_run=bool(chunk_count > 1),
                )
            )
        segmentation_stats["effective_stream_count"] = int(len(streams))
        return streams, segmentation_stats


    def sqlite_allowed_track_ids(sqlite_path: Path, max_tracks: int) -> list[str] | None:
        if int(max_tracks) <= 0:
            return None
        conn = sqlite3.connect(str(sqlite_path))
        try:
            rows = conn.execute(
                """
                SELECT track_id, count(*) AS n
                FROM masks
                GROUP BY track_id
                ORDER BY n DESC, CAST(track_id AS INTEGER)
                LIMIT ?
                """,
                (int(max_tracks),),
            ).fetchall()
        finally:
            conn.close()
        return [str(track_id) for track_id, _count in rows]


    def sqlite_mask_stats_for_tracks(sqlite_path: Path, allowed_track_ids: list[str] | None) -> dict[str, int]:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            if allowed_track_ids is None:
                row = conn.execute("SELECT count(*), count(DISTINCT track_id) FROM masks").fetchone()
            elif not allowed_track_ids:
                row = (0, 0)
            else:
                placeholders = ",".join("?" for _ in allowed_track_ids)
                row = conn.execute(
                    f"SELECT count(*), count(DISTINCT track_id) FROM masks WHERE track_id IN ({placeholders})",
                    tuple(str(track_id) for track_id in allowed_track_ids),
                ).fetchone()
        finally:
            conn.close()
        return {"source_rows": int(row[0] or 0), "source_tracks": int(row[1] or 0)}


    def iter_sqlite_track_rows(sqlite_path: Path, allowed_track_ids: list[str] | None):
        conn = sqlite3.connect(str(sqlite_path))
        try:
            if allowed_track_ids is None:
                rows_iter = conn.execute("SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame")
            elif not allowed_track_ids:
                rows_iter = iter(())
            else:
                placeholders = ",".join("?" for _ in allowed_track_ids)
                rows_iter = conn.execute(
                    f"SELECT frame, track_id, polygons FROM masks WHERE track_id IN ({placeholders}) ORDER BY CAST(track_id AS INTEGER), frame",
                    tuple(str(track_id) for track_id in allowed_track_ids),
                )
            for frame, track_id, polygons_json in rows_iter:
                yield TrackRow(frame=int(frame), track_id=str(track_id), polygons=parse_polygons(str(polygons_json)))
        finally:
            conn.close()


    def iter_track_streams_from_sqlite(
        sqlite_path: Path,
        *,
        anchors_per_contour: int,
        predictor: LearnedPointPredictor | None,
        predictor_batch_size: int,
        adaptive_anchor_counts: bool,
        adaptive_point_quantile: float,
        adaptive_point_offset: int,
        min_anchors_per_contour: int,
        gapfill_enabled: bool,
        gapfill_max_gap: int,
        gapfill_temp_points: int,
        max_tracks: int,
        max_run_frames: int,
        run_overlap_frames: int,
        segmentation_stats: dict[str, int],
    ):
        allowed_track_ids = sqlite_allowed_track_ids(sqlite_path, int(max_tracks))
        source_stats = sqlite_mask_stats_for_tracks(sqlite_path, allowed_track_ids)
        max_frames = int(max_run_frames)
        requested_overlap = max(0, int(run_overlap_frames))
        effective_overlap = 0 if max_frames <= 0 else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
        emit_stride = 0 if max_frames <= 0 else int(max(1, max_frames - 2 * effective_overlap))
        segmentation_stats.clear()
        segmentation_stats.update(
            {
                "source_tracks": int(source_stats["source_tracks"]),
                "source_rows": int(source_stats["source_rows"]),
                "gapfill_inserted_frames": 0,
                "gapfill_events": 0,
                "hard_split_events": 0,
                "segment_count": 0,
                "max_run_frames": int(max_frames),
                "run_overlap_frames": int(effective_overlap),
                "source_segment_count": 0,
                "processed_segment_count": 0,
                "long_segment_count": 0,
                "chunked_source_segment_count": 0,
                "chunk_output_segment_count": 0,
                "max_source_segment_frames": 0,
                "max_processed_segment_frames": 0,
                "emit_stride_frames": int(emit_stride),
                "overlap_added_rows": 0,
                "effective_stream_count": 0,
            }
        )

        buffer: list[TrackRow] = []
        buffer_start_idx = 0
        segment_len = 0
        next_emit_start = 0
        source_run_id = 0
        chunk_index = 0
        current_track_id: str | None = None
        prev: TrackRow | None = None

        def build_runs_for_chunk(
            chunk_rows: list[TrackRow],
            *,
            emit_start: int,
            emit_end: int,
            process_start: int,
            process_end: int,
            chunk_idx: int,
            chunked: bool,
        ) -> list[InstanceRun]:
            runs, _ignored_stats = build_track_streams(
                chunk_rows,
                anchors_per_contour=int(anchors_per_contour),
                predictor=predictor,
                predictor_batch_size=int(predictor_batch_size),
                adaptive_anchor_counts=bool(adaptive_anchor_counts),
                adaptive_point_quantile=float(adaptive_point_quantile),
                adaptive_point_offset=int(adaptive_point_offset),
                min_anchors_per_contour=int(min_anchors_per_contour),
                gapfill_enabled=False,
                gapfill_max_gap=int(gapfill_max_gap),
                gapfill_temp_points=int(gapfill_temp_points),
                max_tracks=-1,
                max_run_frames=0,
                run_overlap_frames=0,
                _release_predictor_after_build=False,
            )
            out: list[InstanceRun] = []
            for sub_idx, run in enumerate(runs):
                suffix = f":chunk{chunk_idx + 1}" if bool(chunked) else ""
                extra = f":part{sub_idx + 1}" if len(runs) > 1 else ""
                run.run_id = int(source_run_id)
                run.stream_id = f"{run.track_id}:run{source_run_id}{suffix}{extra}:instance"
                run.emit_start_idx = int(emit_start)
                run.emit_end_idx = int(emit_end)
                run.chunk_index = int(chunk_idx)
                run.chunk_count = -1 if bool(chunked) else 1
                run.chunk_process_start = int(process_start)
                run.chunk_process_end = int(process_end)
                run.chunked_from_long_run = bool(chunked)
                out.append(run)
            return out

        def emit_chunk(process_start: int, process_end: int, emit_start: int, emit_end: int, *, final: bool) -> list[InstanceRun]:
            nonlocal buffer, buffer_start_idx, next_emit_start, chunk_index
            start_offset = int(process_start - buffer_start_idx)
            end_offset = int(process_end - buffer_start_idx)
            chunk_rows = list(buffer[start_offset:end_offset])
            chunked = bool(segment_len > max_frames and max_frames > 0)
            emitted_len = int(emit_end - emit_start)
            processed_len = int(process_end - process_start)
            segmentation_stats["processed_segment_count"] += 1
            segmentation_stats["max_processed_segment_frames"] = int(max(segmentation_stats["max_processed_segment_frames"], processed_len))
            if chunked:
                segmentation_stats["chunk_output_segment_count"] += 1
                segmentation_stats["overlap_added_rows"] += int(max(0, processed_len - emitted_len))
            runs = build_runs_for_chunk(
                chunk_rows,
                emit_start=int(emit_start - process_start),
                emit_end=int(emit_end - process_start),
                process_start=int(process_start),
                process_end=int(process_end),
                chunk_idx=int(chunk_index),
                chunked=chunked,
            )
            segmentation_stats["effective_stream_count"] += int(len(runs))
            chunk_index += 1
            next_emit_start = int(emit_end)
            if not final:
                keep_from = int(max(0, next_emit_start - effective_overlap))
                drop_count = int(keep_from - buffer_start_idx)
                if drop_count > 0:
                    buffer = buffer[drop_count:]
                    buffer_start_idx = keep_from
            return runs

        def emit_ready_chunks(final: bool) -> list[InstanceRun]:
            out: list[InstanceRun] = []
            if segment_len <= 0:
                return out
            if max_frames <= 0:
                if final and next_emit_start < segment_len:
                    out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
                return out
            if segment_len <= max_frames:
                if final and next_emit_start < segment_len:
                    out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
                return out
            while next_emit_start < segment_len:
                emit_start = int(next_emit_start)
                emit_end = int(min(segment_len, emit_start + emit_stride))
                process_start = int(max(0, emit_start - effective_overlap))
                desired_process_end = int(emit_end + effective_overlap)
                if not final and desired_process_end > segment_len:
                    break
                process_end = int(min(segment_len, desired_process_end))
                if not final and emit_end >= segment_len:
                    break
                out.extend(emit_chunk(process_start, process_end, emit_start, emit_end, final=final))
                if final:
                    continue
            return out

        def flush_segment() -> list[InstanceRun]:
            nonlocal buffer, buffer_start_idx, segment_len, next_emit_start, source_run_id, chunk_index, prev
            if segment_len <= 0:
                return []
            segmentation_stats["segment_count"] += 1
            segmentation_stats["source_segment_count"] += 1
            segmentation_stats["max_source_segment_frames"] = int(max(segmentation_stats["max_source_segment_frames"], segment_len))
            if max_frames > 0 and segment_len > max_frames:
                segmentation_stats["long_segment_count"] += 1
                segmentation_stats["chunked_source_segment_count"] += 1
            runs = emit_ready_chunks(final=True)
            source_run_id += 1
            buffer = []
            buffer_start_idx = 0
            segment_len = 0
            next_emit_start = 0
            chunk_index = 0
            prev = None
            return runs

        def append_segment_row(row: TrackRow) -> list[InstanceRun]:
            nonlocal segment_len
            buffer.append(row)
            segment_len += 1
            return emit_ready_chunks(final=False)

        for row in iter_sqlite_track_rows(sqlite_path, allowed_track_ids):
            if current_track_id is not None and str(row.track_id) != current_track_id:
                for run in flush_segment():
                    yield run
            if current_track_id != str(row.track_id):
                current_track_id = str(row.track_id)
                prev = None

            current_slots = sort_polygons(row.polygons)
            if prev is None:
                first = TrackRow(
                    frame=int(row.frame),
                    track_id=str(row.track_id),
                    polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                    is_gapfill=bool(row.is_gapfill),
                )
                for run in append_segment_row(first):
                    yield run
                prev = first
                continue

            prev_slots = sort_polygons(prev.polygons)
            same_contour_count = len(prev_slots) == len(current_slots)
            gap = int(row.frame) - int(prev.frame) - 1
            if same_contour_count:
                current_slots = align_contour_slots(prev_slots, current_slots)

            if (not bool(gapfill_enabled)) and gap > 0:
                for run in flush_segment():
                    yield run
                current = TrackRow(
                    frame=int(row.frame),
                    track_id=str(row.track_id),
                    polygons=[np.asarray(poly, dtype=np.float32) for poly in sort_polygons(row.polygons)],
                    is_gapfill=bool(row.is_gapfill),
                )
                for run in append_segment_row(current):
                    yield run
                prev = current
                continue

            if gap <= 0 and same_contour_count:
                current = TrackRow(
                    frame=int(row.frame),
                    track_id=str(row.track_id),
                    polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                    is_gapfill=bool(row.is_gapfill),
                )
                for run in append_segment_row(current):
                    yield run
                prev = current
                continue

            can_gapfill = bool(gapfill_enabled) and same_contour_count and gap > 0 and gap <= int(gapfill_max_gap)
            if can_gapfill:
                for step in range(1, gap + 1):
                    interp_polys = interpolate_gapfill_polygons(
                        prev_slots,
                        current_slots,
                        step=step,
                        gap=gap,
                        temp_points=int(gapfill_temp_points),
                    )
                    gap_row = TrackRow(
                        frame=int(prev.frame) + step,
                        track_id=str(row.track_id),
                        polygons=[np.asarray(poly, dtype=np.float32) for poly in interp_polys],
                        is_gapfill=True,
                    )
                    for run in append_segment_row(gap_row):
                        yield run
                segmentation_stats["gapfill_events"] += 1
                segmentation_stats["gapfill_inserted_frames"] += int(gap)
                current = TrackRow(
                    frame=int(row.frame),
                    track_id=str(row.track_id),
                    polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                    is_gapfill=bool(row.is_gapfill),
                )
                for run in append_segment_row(current):
                    yield run
                prev = current
                continue

            for run in flush_segment():
                yield run
            segmentation_stats["hard_split_events"] += 1
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in sort_polygons(row.polygons)],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current

        for run in flush_segment():
            yield run


    def flatten_contours(contours: np.ndarray) -> np.ndarray:
        return np.asarray(contours, dtype=np.float32).reshape(-1, 2)


    def split_vector_to_polygons(vector: np.ndarray, contour_count: int, anchors_per_contour: int) -> list[np.ndarray]:
        vec = np.asarray(vector, dtype=np.float32).reshape(contour_count, anchors_per_contour, 2)
        return [np.asarray(vec[idx], dtype=np.float32) for idx in range(contour_count)]


    def vector_proxy_stats(vector: np.ndarray, contour_count: int, anchors_per_contour: int) -> tuple[float, np.ndarray, np.ndarray, float]:
        arr = np.asarray(vector, dtype=np.float32).reshape(contour_count, anchors_per_contour, 2)
        pts = arr.reshape(-1, 2)
        if pts.size <= 0:
            return 0.0, np.zeros((2,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 1.0
        center = np.mean(pts, axis=0).astype(np.float32)
        radii = np.linalg.norm(pts - center[None, :], axis=1).astype(np.float32)
        mean_radius = float(max(np.mean(radii, dtype=np.float64), 1e-6))
        x = arr[..., 0].astype(np.float64, copy=False)
        y = arr[..., 1].astype(np.float64, copy=False)
        x_next = np.roll(x, -1, axis=1)
        y_next = np.roll(y, -1, axis=1)
        area = float(0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1)).sum())
        return area, center, radii, mean_radius


    def scale_vector_about_centroid(vector: np.ndarray, scale_mul: float) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim == 3:
            out = np.zeros_like(arr, dtype=np.float64)
            for idx in range(arr.shape[0]):
                center = np.asarray(np.mean(arr[idx], axis=0), dtype=np.float64)
                out[idx] = center + float(scale_mul) * (arr[idx] - center)
            return out.astype(np.float32)
        pts = arr.reshape(-1, 2)
        center = np.asarray(np.mean(pts, axis=0), dtype=np.float64)
        return (center + float(scale_mul) * (pts - center)).astype(np.float32)


    def rasterize_mask_with_context(
        polygons: list[np.ndarray],
        context: FrameEvalContext,
        out_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if out_mask is None:
            mask = np.zeros(context.shape_hw, dtype=np.uint8)
        else:
            mask = np.asarray(out_mask, dtype=np.uint8)
            mask.fill(0)
        pts_list: list[np.ndarray] = []
        for poly in polygons:
            pts = (np.asarray(poly, dtype=np.float32) - context.shift_xy[None, :]) * float(context.scale_factor)
            pts = np.round(pts).astype(np.int32)
            if len(pts) >= 3:
                pts_list.append(pts)
        if pts_list:
            cv2.fillPoly(mask, pts_list, 1)
        return mask


    def rasterize_interpolated_mask_with_context(
        start_polygons: list[np.ndarray],
        end_polygons: list[np.ndarray],
        alpha: float,
        context: FrameEvalContext,
        out_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if out_mask is None:
            mask = np.zeros(context.shape_hw, dtype=np.uint8)
        else:
            mask = np.asarray(out_mask, dtype=np.uint8)
            mask.fill(0)
        pts_list: list[np.ndarray] = []
        alpha32 = np.float32(alpha)
        beta32 = np.float32(1.0) - alpha32
        for start_poly, end_poly in zip(start_polygons, end_polygons):
            start_pts = np.asarray(start_poly, dtype=np.float32)
            end_pts = np.asarray(end_poly, dtype=np.float32)
            pts = (beta32 * start_pts + alpha32 * end_pts - context.shift_xy[None, :]) * float(context.scale_factor)
            pts = np.round(pts).astype(np.int32)
            if len(pts) >= 3:
                pts_list.append(pts)
        if pts_list:
            cv2.fillPoly(mask, pts_list, 1)
        return mask


    def build_frame_eval_contexts(run: InstanceRun, args: argparse.Namespace) -> list[FrameEvalContext]:
        contexts: list[FrameEvalContext] = []
        scale_factor = float(np.clip(float(args.dp_eval_scale), 0.1, 1.0))
        pad = int(max(0, int(args.dp_eval_pad)))
        for frame_idx in range(len(run.frame_numbers)):
            raw_vector = flatten_contours(run.anchors[frame_idx])
            gt_polygon_area, gt_center, gt_radii, gt_mean_radius = vector_proxy_stats(raw_vector, run.contour_count, run.anchors_per_contour)
            raw_polys = split_vector_to_polygons(flatten_contours(run.anchors[frame_idx]), run.contour_count, run.anchors_per_contour)
            all_polys = [np.asarray(poly, dtype=np.float32) for poly in run.gt_polygons[frame_idx] + raw_polys if len(poly) >= 3]
            if all_polys:
                all_pts = np.concatenate(all_polys, axis=0)
                min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32) - pad
                max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32) + pad
            else:
                min_xy = np.asarray([0, 0], dtype=np.int32)
                max_xy = np.asarray([4, 4], dtype=np.int32)
            shift_xy = min_xy.astype(np.float32)
            width = int(max_xy[0] - min_xy[0] + 1)
            height = int(max_xy[1] - min_xy[1] + 1)
            shape_hw = (
                max(1, int(math.ceil(height * scale_factor))),
                max(1, int(math.ceil(width * scale_factor))),
            )
            context = FrameEvalContext(
                gt_mask=np.zeros(shape_hw, dtype=np.uint8),
                gt_area=0,
                shift_xy=shift_xy,
                shape_hw=shape_hw,
                scale_factor=scale_factor,
                gt_center=np.asarray(gt_center, dtype=np.float32),
                gt_radii=np.asarray(gt_radii, dtype=np.float32),
                gt_mean_radius=float(gt_mean_radius),
                gt_polygon_area=float(gt_polygon_area),
            )
            gt_mask = rasterize_mask_with_context(run.gt_polygons[frame_idx], context)
            contexts.append(
                FrameEvalContext(
                    gt_mask=gt_mask,
                    gt_area=int(gt_mask.sum()),
                    shift_xy=shift_xy,
                    shape_hw=shape_hw,
                    scale_factor=scale_factor,
                    gt_center=np.asarray(gt_center, dtype=np.float32),
                    gt_radii=np.asarray(gt_radii, dtype=np.float32),
                    gt_mean_radius=float(gt_mean_radius),
                    gt_polygon_area=float(gt_polygon_area),
                    scratch_pred_mask=np.zeros(shape_hw, dtype=np.uint8),
                    scratch_intersection_mask=np.zeros(shape_hw, dtype=np.uint8),
                )
            )
        return contexts


    def compute_cached_metrics_from_polygons(gt_context: FrameEvalContext, pred_polys: list[np.ndarray]) -> dict[str, float]:
        pred_mask = rasterize_mask_with_context(
            pred_polys,
            gt_context,
            out_mask=gt_context.scratch_pred_mask,
        )
        pred_area = int(cv2.countNonZero(pred_mask))
        intersection_mask = gt_context.scratch_intersection_mask
        if intersection_mask is None:
            intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
        cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
        intersection = int(cv2.countNonZero(intersection_mask))
        union = int(gt_context.gt_area + pred_area - intersection)
        recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
        precision = intersection / pred_area if pred_area > 0 else 1.0
        iou = intersection / union if union > 0 else 1.0
        return {
            "gt_area": float(gt_context.gt_area),
            "pred_area": float(pred_area),
            "intersection": float(intersection),
            "union": float(union),
            "recall": float(recall),
            "precision": float(precision),
            "iou": float(iou),
        }


    def compute_cached_metrics_from_interpolated_polygons(
        gt_context: FrameEvalContext,
        start_polys: list[np.ndarray],
        end_polys: list[np.ndarray],
        alpha: float,
    ) -> dict[str, float]:
        pred_mask = rasterize_interpolated_mask_with_context(
            start_polys,
            end_polys,
            alpha,
            gt_context,
            out_mask=gt_context.scratch_pred_mask,
        )
        pred_area = int(cv2.countNonZero(pred_mask))
        intersection_mask = gt_context.scratch_intersection_mask
        if intersection_mask is None:
            intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
        cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
        intersection = int(cv2.countNonZero(intersection_mask))
        union = int(gt_context.gt_area + pred_area - intersection)
        recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
        precision = intersection / pred_area if pred_area > 0 else 1.0
        iou = intersection / union if union > 0 else 1.0
        return {
            "gt_area": float(gt_context.gt_area),
            "pred_area": float(pred_area),
            "intersection": float(intersection),
            "union": float(union),
            "recall": float(recall),
            "precision": float(precision),
            "iou": float(iou),
        }


    def evaluate_frame_vector_loss_budget(
        run: InstanceRun,
        frame_idx: int,
        vector: np.ndarray,
        args: argparse.Namespace,
        eval_contexts: list[FrameEvalContext] | None = None,
    ) -> tuple[float, float]:
        pred_polys = split_vector_to_polygons(vector, run.contour_count, run.anchors_per_contour)
        if eval_contexts is not None:
            metrics = compute_cached_metrics_from_polygons(eval_contexts[int(frame_idx)], pred_polys)
        else:
            metrics = compute_exact_metrics_from_polygons(run.gt_polygons[int(frame_idx)], pred_polys)
        return float(frame_accuracy_loss(metrics, args)), float(recall_budget_from_metrics(metrics))


    def recall_budget_from_metrics(metrics: dict[str, float]) -> float:
        return max(0.0, 1.0 - float(metrics["recall"]))


    def recall_budget_limit(frame_count: int, args: argparse.Namespace) -> float:
        recall_min = float(np.clip(float(args.recall_min), 0.0, 1.0))
        return float(max(frame_count, 0)) * max(0.0, 1.0 - recall_min)


    def recall_violation(total_budget: float, frame_count: int, args: argparse.Namespace) -> float:
        return max(float(total_budget) - float(recall_budget_limit(frame_count, args)), 0.0)


    def frame_accuracy_loss(metrics: dict[str, float], args: argparse.Namespace) -> float:
        return float(args.interval_iou_weight) * (1.0 - float(metrics["iou"]))


    def adaptive_shape_penalty_scales(frame_loss_mean: float, args: argparse.Namespace) -> tuple[float, float]:
        mean_loss = max(float(frame_loss_mean), 0.0)
        gain = max(float(args.shape_penalty_adapt_gain), 0.0)
        if gain <= 0.0:
            return 1.0, 1.0
        base = 1.0 + gain * mean_loss
        distance_scale = 1.0 / max(base ** max(float(args.shape_distance_relief), 0.0), 1e-6)
        switch_scale = 1.0 / max(base ** max(float(args.shape_switch_relief), 0.0), 1e-6)
        distance_scale = max(float(args.shape_distance_min_scale), float(distance_scale))
        switch_scale = max(float(args.shape_switch_min_scale), float(switch_scale))
        return float(distance_scale), float(switch_scale)


    def build_frame_candidates(
        run: InstanceRun,
        _contexts: list[object],
        eval_contexts: list[FrameEvalContext],
        args: argparse.Namespace,
    ) -> list[list[ShapeCandidate]]:
        candidates_by_frame: list[list[ShapeCandidate]] = []
        for idx in range(len(run.frame_numbers)):
            raw_vector = flatten_contours(run.anchors[idx])
            raw_metrics = compute_cached_metrics_from_polygons(
                eval_contexts[idx],
                split_vector_to_polygons(raw_vector, run.contour_count, run.anchors_per_contour),
            )
            raw_frame_loss = frame_accuracy_loss(raw_metrics, args)
            raw_area, raw_center, raw_radii, raw_mean_radius = vector_proxy_stats(raw_vector, run.contour_count, run.anchors_per_contour)
            raw_candidate = ShapeCandidate(
                label="raw",
                vector=np.asarray(raw_vector, dtype=np.float32),
                polygons=split_vector_to_polygons(raw_vector, run.contour_count, run.anchors_per_contour),
                frame_loss=float(raw_frame_loss),
                objective=float(raw_frame_loss),
                recall_budget=float(recall_budget_from_metrics(raw_metrics)),
                area=float(raw_area),
                center=np.asarray(raw_center, dtype=np.float32),
                radii=np.asarray(raw_radii, dtype=np.float32),
                mean_radius=float(raw_mean_radius),
            )
            candidates_by_frame.append([raw_candidate])
        return candidates_by_frame


    def shape_distance(vector_a: np.ndarray, vector_b: np.ndarray, scale: float) -> float:
        residual, _ = similarity_residuals(np.asarray(vector_a, dtype=np.float32), np.asarray(vector_b, dtype=np.float32))
        norms = np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=1)
        return float(np.mean(norms) / max(float(scale), 1.0))


    def compute_saliency_scores(run: InstanceRun, fit_vectors: list[np.ndarray], area_series: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        length = len(fit_vectors)
        scores = np.zeros((length,), dtype=np.float64)
        area_scale = max(float(np.mean(np.asarray(area_series, dtype=np.float64))), 1.0)
        for idx in range(1, length - 1):
            prev_vec = np.asarray(fit_vectors[idx - 1], dtype=np.float64)
            cur_vec = np.asarray(fit_vectors[idx], dtype=np.float64)
            next_vec = np.asarray(fit_vectors[idx + 1], dtype=np.float64)
            second = float(np.linalg.norm(next_vec - 2.0 * cur_vec + prev_vec) / max(float(run.scale), 1.0))
            jump = shape_distance(fit_vectors[idx - 1], fit_vectors[idx + 1], run.scale)
            area_peak = max(float(area_series[idx]) - 0.5 * float(area_series[idx - 1] + area_series[idx + 1]), 0.0) / area_scale
            area_swing = abs(float(area_series[idx + 1]) - float(area_series[idx - 1])) / area_scale
            scores[idx] = (
                second
                + float(args.saliency_shape_eta) * jump
                + float(args.saliency_area_eta) * (area_peak + 0.5 * area_swing)
            )
        if length > 0:
            scores[0] = float(scores[1] if length > 1 else 0.0)
            scores[-1] = float(scores[-2] if length > 1 else 0.0)
        return scores


    def compute_surrogate_prefix(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray([np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors], dtype=np.float64)
        times = np.arange(q.shape[0], dtype=np.float64)[:, None]
        prefix_q = np.concatenate([np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q, axis=0)], axis=0)
        prefix_tq = np.concatenate([np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q * times, axis=0)], axis=0)
        prefix_q2 = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(np.sum(q * q, axis=1), axis=0)], axis=0)
        return prefix_q, prefix_tq, prefix_q2


    def surrogate_interval_cost(
        u: int,
        v: int,
        prefix_q: np.ndarray,
        prefix_tq: np.ndarray,
        prefix_q2: np.ndarray,
        vector_dim: int,
        contour_count: int,
        anchors_per_contour: int,
        scale: float,
        args: argparse.Namespace,
    ) -> float:
        cost, _start_vec, _end_vec = surrogate_interval_solution(
            u,
            v,
            prefix_q,
            prefix_tq,
            prefix_q2,
            vector_dim,
            contour_count,
            anchors_per_contour,
            scale,
            args,
        )
        return float(cost)


    def surrogate_interval_solution(
        u: int,
        v: int,
        prefix_q: np.ndarray,
        prefix_tq: np.ndarray,
        prefix_q2: np.ndarray,
        vector_dim: int,
        contour_count: int,
        anchors_per_contour: int,
        scale: float,
        args: argparse.Namespace,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if v <= u:
            zero = np.zeros((vector_dim // 2, 2), dtype=np.float32)
            return 0.0, zero, zero
        h = int(v - u)
        s0 = prefix_q[v + 1] - prefix_q[u]
        s1 = prefix_tq[v + 1] - prefix_tq[u]
        s2 = float(prefix_q2[v + 1] - prefix_q2[u])
        a = float((h + 1) * (2 * h + 1) / (6.0 * h))
        b = float((h + 1) * (h - 1) / (6.0 * h))
        c = float(a)
        gu = (float(v) * s0 - s1) / float(h)
        gv = (s1 - float(u) * s0) / float(h)
        det = max(a * c - b * b, 1e-9)
        avec = (c * gu - b * gv) / det
        bvec = (-b * gu + a * gv) / det
        quad = a * float(np.dot(avec, avec)) + 2.0 * b * float(np.dot(avec, bvec)) + c * float(np.dot(bvec, bvec))
        cross = 2.0 * float(np.dot(gu, avec) + np.dot(gv, bvec))
        sse = max(s2 - cross + quad, 0.0)
        start_vec = np.asarray(avec, dtype=np.float32).reshape(vector_dim // 2, 2)
        end_vec = np.asarray(bvec, dtype=np.float32).reshape(vector_dim // 2, 2)
        d = shape_distance(start_vec, end_vec, scale)
        return float(sse / max(float(scale) ** 2, 1.0) + float(args.surrogate_shape_weight) * d), start_vec, end_vec


    def exact_k_dp(cost_fn, nodes: list[int], target_count: int, max_gap: int) -> list[int]:
        node_count = len(nodes)
        target_count = max(2, min(int(target_count), node_count))
        dp = np.full((target_count, node_count), np.inf, dtype=np.float64)
        back = np.full((target_count, node_count), -1, dtype=np.int32)
        dp[0, 0] = 0.0
        for used in range(1, target_count):
            for end_pos in range(used, node_count):
                end_node = int(nodes[end_pos])
                min_prev_pos = max(used - 1, int(bisect.bisect_left(nodes, end_node - int(max_gap), 0, end_pos)))
                best_cost = float("inf")
                best_prev = -1
                for prev_pos in range(min_prev_pos, end_pos):
                    prev_node = int(nodes[prev_pos])
                    prev_cost = float(dp[used - 1, prev_pos])
                    if not np.isfinite(prev_cost):
                        continue
                    cand = prev_cost + float(cost_fn(prev_node, end_node))
                    if cand < best_cost:
                        best_cost = cand
                        best_prev = int(prev_pos)
                dp[used, end_pos] = best_cost
                back[used, end_pos] = best_prev
        path = [node_count - 1]
        cur_pos = node_count - 1
        cur_used = target_count - 1
        while cur_used > 0:
            cur_pos = int(back[cur_used, cur_pos])
            if cur_pos < 0:
                return [int(nodes[0]), int(nodes[-1])]
            path.append(cur_pos)
            cur_used -= 1
        path.reverse()
        return [int(nodes[pos]) for pos in path]


    def build_candidate_frame_pool(run: InstanceRun, candidates_by_frame: list[list[ShapeCandidate]], target_count: int, args: argparse.Namespace) -> tuple[list[int], list[int], np.ndarray]:
        raw_vectors = [frame_candidates[0].vector for frame_candidates in candidates_by_frame]
        area_series = np.asarray([float(frame_candidates[0].area) for frame_candidates in candidates_by_frame], dtype=np.float64)
        scores = compute_saliency_scores(run, raw_vectors, area_series, args)
        length = len(run.frame_numbers)
        target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
        dynamic_max_gap = max(int(args.max_gap), int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))))
        prefix_q, prefix_tq, prefix_q2 = compute_surrogate_prefix(raw_vectors)
        vector_dim = int(np.asarray(raw_vectors[0], dtype=np.float32).size) if raw_vectors else 0

        surrogate_cost_cache: dict[tuple[int, int], float] = {}

        def surrogate_cost(u: int, v: int) -> float:
            key = (int(u), int(v))
            cached = surrogate_cost_cache.get(key)
            if cached is not None:
                return float(cached)
            cost = surrogate_interval_cost(
                int(u),
                int(v),
                prefix_q,
                prefix_tq,
                prefix_q2,
                vector_dim,
                run.contour_count,
                run.anchors_per_contour,
                run.scale,
                args,
            )
            surrogate_cost_cache[key] = float(cost)
            return float(cost)

        all_nodes = list(range(length))
        surrogate_path = exact_k_dp(surrogate_cost, all_nodes, int(target_count), dynamic_max_gap)
        pool_target = min(
            length,
            max(
                int(round(float(args.surrogate_pool_factor) * float(target_count))),
                int(math.ceil(math.sqrt(max(length, 1)))),
                int(target_count) + 2,
            ),
        )
        peak_target = min(length, max(0, int(round(float(args.surrogate_peak_factor) * float(target_count)))))
        peak_ids = [int(idx) for idx in np.argsort(-scores)[:peak_target].tolist()]
        grid = list(range(0, length, max(1, target_interval)))
        if grid[-1] != length - 1:
            grid.append(length - 1)
        pool = {0, length - 1}
        for frame_idx in surrogate_path:
            for delta in range(-int(args.surrogate_neighbor_radius), int(args.surrogate_neighbor_radius) + 1):
                cand = int(frame_idx) + int(delta)
                if 0 <= cand < length:
                    pool.add(int(cand))
        for frame_idx in peak_ids:
            pool.add(int(frame_idx))
        for frame_idx in grid:
            pool.add(int(frame_idx))
        if len(pool) < int(target_count):
            for frame_idx in np.argsort(-scores).tolist():
                pool.add(int(frame_idx))
                if len(pool) >= int(target_count):
                    break
        if len(pool) < pool_target:
            for frame_idx in np.argsort(-scores).tolist():
                pool.add(int(frame_idx))
                if len(pool) >= pool_target:
                    break
        return sorted(int(frame_idx) for frame_idx in pool), [int(frame_idx) for frame_idx in surrogate_path], scores


    def build_ring_second_difference_rtr(contour_count: int, anchors_per_contour: int) -> np.ndarray:
        point_count = int(contour_count) * int(anchors_per_contour)
        dim = int(point_count * 2)
        rows: list[np.ndarray] = []
        for contour_idx in range(int(contour_count)):
            base = contour_idx * int(anchors_per_contour)
            for anchor_idx in range(int(anchors_per_contour)):
                prev_idx = base + ((anchor_idx - 1) % int(anchors_per_contour))
                cur_idx = base + anchor_idx
                next_idx = base + ((anchor_idx + 1) % int(anchors_per_contour))
                for axis in range(2):
                    row = np.zeros((dim,), dtype=np.float64)
                    row[2 * prev_idx + axis] = 1.0
                    row[2 * cur_idx + axis] = -2.0
                    row[2 * next_idx + axis] = 1.0
                    rows.append(row)
        if not rows:
            return np.zeros((dim, dim), dtype=np.float64)
        mat = np.asarray(rows, dtype=np.float64)
        return mat.T @ mat


    def build_interpolation_weights(frame_count: int, chosen_frames: list[int]) -> np.ndarray:
        key_count = int(len(chosen_frames))
        weights = np.zeros((int(frame_count), key_count), dtype=np.float64)
        chosen = [int(v) for v in chosen_frames]
        if key_count <= 0:
            return weights
        for frame_idx in range(int(frame_count)):
            if frame_idx <= chosen[0]:
                weights[frame_idx, 0] = 1.0
                continue
            if frame_idx >= chosen[-1]:
                weights[frame_idx, -1] = 1.0
                continue
            right_pos = next(pos for pos, keyframe in enumerate(chosen) if keyframe >= frame_idx)
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen[left_pos])
            right_frame = int(chosen[right_pos])
            if frame_idx == right_frame or right_frame <= left_frame:
                weights[frame_idx, right_pos] = 1.0
            else:
                alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                weights[frame_idx, left_pos] = 1.0 - alpha
                weights[frame_idx, right_pos] = alpha
        return weights


    def pair_vote_refine_keyframe_vectors(
        run: InstanceRun,
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
        args: argparse.Namespace,
    ) -> np.ndarray:
        if not bool(getattr(args, "pair_vote_refine_enabled", True)) or len(chosen_frames) <= 1:
            return np.asarray(keyframe_vectors, dtype=np.float32)
        frame_count = int(len(run.frame_numbers))
        targets = np.asarray([flatten_contours(run.anchors[idx]).reshape(-1) for idx in range(frame_count)], dtype=np.float64)
        init = np.asarray(keyframe_vectors, dtype=np.float64).reshape(len(chosen_frames), -1)
        proposals: list[list[tuple[np.ndarray, float]]] = [[] for _ in chosen_frames]
        eye2 = np.eye(2, dtype=np.float64)
        for left_pos in range(len(chosen_frames) - 1):
            right_pos = left_pos + 1
            u = int(chosen_frames[left_pos])
            v = int(chosen_frames[right_pos])
            span = max(v - u, 1)
            rows = []
            local_targets = []
            for frame_idx in range(u, v + 1):
                beta = float(v - frame_idx) / float(span)
                gamma = float(frame_idx - u) / float(span)
                rows.append([beta, gamma])
                local_targets.append(targets[frame_idx])
            x = np.asarray(rows, dtype=np.float64)
            y = np.asarray(local_targets, dtype=np.float64)
            gram = x.T @ x
            rhs = x.T @ y
            ab = np.linalg.solve(gram + 1e-8 * eye2, rhs)
            interval_weight = float(v - u + 1)
            proposals[left_pos].append((np.asarray(ab[0], dtype=np.float32), interval_weight))
            proposals[right_pos].append((np.asarray(ab[1], dtype=np.float32), interval_weight))
        out = init.copy()
        for idx, items in enumerate(proposals):
            if not items:
                continue
            total_w = float(sum(weight for _vec, weight in items))
            voted = sum(np.asarray(vec, dtype=np.float64) * float(weight) for vec, weight in items) / max(total_w, 1e-8)
            out[idx] = voted
        return np.asarray(out.reshape(np.asarray(keyframe_vectors).shape), dtype=np.float32)


    def interpolate_vectors(start_vec: np.ndarray, end_vec: np.ndarray, alpha: float) -> np.ndarray:
        return ((1.0 - float(alpha)) * np.asarray(start_vec, dtype=np.float32) + float(alpha) * np.asarray(end_vec, dtype=np.float32)).astype(np.float32)


    def interpolate_polygons(start_polys: list[np.ndarray], end_polys: list[np.ndarray], alpha: float) -> list[np.ndarray]:
        alpha32 = np.float32(alpha)
        beta32 = np.float32(1.0) - alpha32
        out: list[np.ndarray] = []
        for start_poly, end_poly in zip(start_polys, end_polys):
            start_pts = np.asarray(start_poly, dtype=np.float32)
            end_pts = np.asarray(end_poly, dtype=np.float32)
            out.append((beta32 * start_pts + alpha32 * end_pts).astype(np.float32))
        return out


    def assign_candidate_ids_to_keyframes(
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
        candidates_by_frame: list[list[ShapeCandidate]],
    ) -> list[int]:
        candidate_ids: list[int] = []
        for frame_idx, vector in zip(chosen_frames, keyframe_vectors):
            frame_candidates = candidates_by_frame[int(frame_idx)]
            best_cand = 0
            best_dist = float("inf")
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            for cand_id, candidate in enumerate(frame_candidates):
                cand_vec = np.asarray(candidate.vector, dtype=np.float32).reshape(-1)
                dist = float(np.mean(np.square(vec - cand_vec)))
                if dist < best_dist:
                    best_dist = dist
                    best_cand = int(cand_id)
            candidate_ids.append(int(best_cand))
        return candidate_ids


    def interval_cost_from_vectors(
        run: InstanceRun,
        start_idx: int,
        start_vec: np.ndarray,
        end_idx: int,
        end_vec: np.ndarray,
        args: argparse.Namespace,
        *,
        include_start: bool,
        eval_contexts: list[FrameEvalContext] | None = None,
        start_candidate: ShapeCandidate | None = None,
        end_candidate: ShapeCandidate | None = None,
    ) -> IntervalCost:
        if end_idx < start_idx:
            return IntervalCost(cost=float("inf"), shape_distance=float("inf"), shape_update=1.0, frames_covered=0)
        start_polys = start_candidate.polygons if start_candidate is not None else split_vector_to_polygons(start_vec, run.contour_count, run.anchors_per_contour)
        end_polys = end_candidate.polygons if end_candidate is not None else split_vector_to_polygons(end_vec, run.contour_count, run.anchors_per_contour)
        dist = shape_distance(start_vec, end_vec, run.scale)
        update = 1.0 if dist > float(args.shape_update_threshold_ratio) else 0.0
        total = 0.0
        frames_covered = 0
        frame_loss_total = 0.0
        recall_budget_total = 0.0
        start_frame = int(start_idx if include_start else start_idx + 1)
        for frame_idx in range(start_frame, int(end_idx) + 1):
            if frame_idx == start_idx:
                if eval_contexts is not None:
                    metrics = compute_cached_metrics_from_polygons(eval_contexts[frame_idx], start_polys)
                else:
                    metrics = compute_exact_metrics_from_polygons(run.gt_polygons[frame_idx], start_polys)
            elif frame_idx == end_idx:
                if eval_contexts is not None:
                    metrics = compute_cached_metrics_from_polygons(eval_contexts[frame_idx], end_polys)
                else:
                    metrics = compute_exact_metrics_from_polygons(run.gt_polygons[frame_idx], end_polys)
            else:
                alpha = float((frame_idx - start_idx) / max(end_idx - start_idx, 1))
                if eval_contexts is not None:
                    metrics = compute_cached_metrics_from_interpolated_polygons(
                        eval_contexts[frame_idx],
                        start_polys,
                        end_polys,
                        alpha,
                    )
                else:
                    pred_polys = interpolate_polygons(start_polys, end_polys, alpha)
                    metrics = compute_exact_metrics_from_polygons(run.gt_polygons[frame_idx], pred_polys)
            frame_loss = float(frame_accuracy_loss(metrics, args))
            recall_budget = float(recall_budget_from_metrics(metrics))
            total += float(frame_loss)
            frame_loss_total += float(frame_loss)
            recall_budget_total += float(recall_budget)
            frames_covered += 1
        frame_loss_mean = float(frame_loss_total / max(frames_covered, 1))
        dist_scale, switch_scale = adaptive_shape_penalty_scales(frame_loss_mean, args)
        total += float(args.shape_switch_weight) * float(switch_scale) * float(update)
        total += float(args.shape_distance_weight) * float(dist_scale) * float(dist)
        return IntervalCost(
            cost=float(total),
            shape_distance=float(dist),
            shape_update=float(update),
            frames_covered=int(frames_covered),
            frame_loss_mean=float(frame_loss_mean),
            shape_distance_scale=float(dist_scale),
            shape_switch_scale=float(switch_scale),
            recall_budget=float(recall_budget_total),
        )


    def run_multistate_penalty_path(
        run: InstanceRun,
        candidate_frames: list[int],
        candidates_by_frame: list[list[ShapeCandidate]],
        target_count: int,
        args: argparse.Namespace,
        eval_contexts: list[FrameEvalContext] | None = None,
    ) -> tuple[list[int], list[int], dict[str, int], dict[tuple[int, int, int, int, int], IntervalCost], float]:
        if all(len(frame_candidates) == 1 for frame_candidates in candidates_by_frame):
            return run_single_state_penalty_path(
                run,
                candidate_frames,
                candidates_by_frame,
                target_count,
                args,
                eval_contexts=eval_contexts,
            )

        target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
        dynamic_max_gap = max(int(args.max_gap), int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))))
        state_frames: list[int] = []
        state_candidate_ids: list[int] = []
        node_offsets: list[tuple[int, int]] = []
        cursor = 0
        for frame_idx in candidate_frames:
            start = cursor
            for cand_id in range(len(candidates_by_frame[int(frame_idx)])):
                state_frames.append(int(frame_idx))
                state_candidate_ids.append(int(cand_id))
                cursor += 1
            node_offsets.append((start, cursor))
        state_count = cursor
        cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
        edge_array_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        counters = {"interval_evals": 0, "interval_frames": 0}
        use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
        recall_penalty_weight = float(args.proxy_recall_penalty_weight)
        predecessor_nodes: list[list[int]] = []
        for node_pos, end_frame in enumerate(candidate_frames):
            valid_prev: list[int] = []
            end_frame_i = int(end_frame)
            for prev_node_pos in range(node_pos):
                if end_frame_i - int(candidate_frames[prev_node_pos]) <= int(dynamic_max_gap):
                    valid_prev.append(int(prev_node_pos))
            predecessor_nodes.append(valid_prev)

        def get_cost(start_frame: int, start_cand: int, end_frame: int, end_cand: int, include_start: bool) -> IntervalCost:
            key = (int(start_frame), int(start_cand), int(end_frame), int(end_cand), 1 if include_start else 0)
            info = cost_cache.get(key)
            if info is None:
                full_key = (int(start_frame), int(start_cand), int(end_frame), int(end_cand), 1)
                full_info = cost_cache.get(full_key)
                if full_info is None:
                    start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                    end_candidate = candidates_by_frame[int(end_frame)][int(end_cand)]
                    full_info = interval_cost_from_vectors(
                        run,
                        int(start_frame),
                        start_candidate.vector,
                        int(end_frame),
                        end_candidate.vector,
                        args,
                        include_start=True,
                        eval_contexts=eval_contexts,
                        start_candidate=start_candidate,
                        end_candidate=end_candidate,
                    )
                    cost_cache[full_key] = full_info
                    counters["interval_evals"] += 1
                    counters["interval_frames"] += int(full_info.frames_covered)
                if include_start:
                    info = full_info
                else:
                    start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                    info = IntervalCost(
                        cost=float(full_info.cost - float(start_candidate.frame_loss)),
                        shape_distance=float(full_info.shape_distance),
                        shape_update=float(full_info.shape_update),
                        frames_covered=max(int(full_info.frames_covered) - 1, 0),
                        frame_loss_mean=float(full_info.frame_loss_mean),
                        shape_distance_scale=float(full_info.shape_distance_scale),
                        shape_switch_scale=float(full_info.shape_switch_scale),
                        recall_budget=max(float(full_info.recall_budget) - float(start_candidate.recall_budget), 0.0),
                    )
                    cost_cache[key] = info
            return info

        def get_edge_arrays(prev_node_pos: int, node_pos: int) -> tuple[np.ndarray, np.ndarray]:
            key = (int(prev_node_pos), int(node_pos))
            cached = edge_array_cache.get(key)
            if cached is not None:
                return cached
            start_frame = int(candidate_frames[prev_node_pos])
            end_frame = int(candidate_frames[node_pos])
            src_start, src_end = node_offsets[prev_node_pos]
            dst_start, dst_end = node_offsets[node_pos]
            src_count = int(src_end - src_start)
            dst_count = int(dst_end - dst_start)
            cost_arr = np.empty((src_count, dst_count), dtype=np.float64)
            budget_arr = np.empty((src_count, dst_count), dtype=np.float64)
            for src_local, src_state in enumerate(range(src_start, src_end)):
                start_cand = int(state_candidate_ids[src_state])
                for dst_local, dst_state in enumerate(range(dst_start, dst_end)):
                    end_cand = int(state_candidate_ids[dst_state])
                    info = get_cost(start_frame, start_cand, end_frame, end_cand, include_start=False)
                    cost_arr[src_local, dst_local] = float(info.cost)
                    budget_arr[src_local, dst_local] = float(info.recall_budget)
            edge_array_cache[key] = (cost_arr, budget_arr)
            return cost_arr, budget_arr

        def decode(lambda_penalty: float, recall_mu: float) -> tuple[list[int], list[int], float, float]:
            dp = np.full((state_count,), np.inf, dtype=np.float64)
            back = np.full((state_count,), -1, dtype=np.int32)
            raw_cost = np.full((state_count,), np.inf, dtype=np.float64)
            raw_budget = np.full((state_count,), np.inf, dtype=np.float64)
            first_start, first_end = node_offsets[0]
            for state_idx in range(first_start, first_end):
                cand_id = int(state_candidate_ids[state_idx])
                frame_loss = float(candidates_by_frame[0][cand_id].frame_loss)
                frame_budget = float(candidates_by_frame[0][cand_id].recall_budget)
                penalty = float(recall_mu) * frame_budget if use_exact_recall_dp else recall_penalty_weight * frame_budget
                dp[state_idx] = frame_loss + penalty + float(lambda_penalty)
                raw_cost[state_idx] = frame_loss
                raw_budget[state_idx] = frame_budget
            for node_pos in range(1, len(candidate_frames)):
                dst_start, dst_end = node_offsets[node_pos]
                prev_entries = []
                for prev_node_pos in predecessor_nodes[node_pos]:
                    src_start, src_end = node_offsets[prev_node_pos]
                    edge_costs, edge_budgets = get_edge_arrays(prev_node_pos, node_pos)
                    prev_entries.append((src_start, src_end, edge_costs, edge_budgets))
                for dst_state in range(dst_start, dst_end):
                    dst_local = int(dst_state - dst_start)
                    best_cost = float("inf")
                    best_raw = float("inf")
                    best_budget = float("inf")
                    best_prev = -1
                    for src_start, src_end, edge_costs, edge_budgets in prev_entries:
                        for src_state in range(src_start, src_end):
                            prev_cost = float(dp[src_state])
                            if not np.isfinite(prev_cost):
                                continue
                            src_local = int(src_state - src_start)
                            edge_cost = float(edge_costs[src_local, dst_local])
                            edge_budget = float(edge_budgets[src_local, dst_local])
                            penalty = float(recall_mu) * edge_budget if use_exact_recall_dp else recall_penalty_weight * edge_budget
                            cand_cost = prev_cost + edge_cost + penalty + float(lambda_penalty)
                            cand_raw = float(raw_cost[src_state]) + edge_cost
                            cand_budget = float(raw_budget[src_state]) + edge_budget
                            if cand_cost < best_cost or (
                                abs(cand_cost - best_cost) <= 1e-9
                                and (cand_budget < best_budget or (abs(cand_budget - best_budget) <= 1e-9 and cand_raw < best_raw))
                            ):
                                best_cost = float(cand_cost)
                                best_raw = float(cand_raw)
                                best_budget = float(cand_budget)
                                best_prev = int(src_state)
                    dp[dst_state] = best_cost
                    raw_cost[dst_state] = best_raw
                    raw_budget[dst_state] = best_budget
                    back[dst_state] = int(best_prev)
            last_start, last_end = node_offsets[-1]
            best_state = -1
            best_cost = float("inf")
            best_raw = float("inf")
            best_budget = float("inf")
            for state_idx in range(last_start, last_end):
                cost = float(dp[state_idx])
                raw = float(raw_cost[state_idx])
                budget = float(raw_budget[state_idx])
                if cost < best_cost or (
                    abs(cost - best_cost) <= 1e-9 and (budget < best_budget or (abs(budget - best_budget) <= 1e-9 and raw < best_raw))
                ):
                    best_cost = cost
                    best_raw = raw
                    best_budget = budget
                    best_state = int(state_idx)
            if best_state < 0:
                raise RuntimeError("failed to decode penalized multistate path")
            chosen_frames: list[int] = []
            chosen_candidate_ids: list[int] = []
            cur_state = best_state
            while cur_state >= 0:
                chosen_frames.append(int(state_frames[cur_state]))
                chosen_candidate_ids.append(int(state_candidate_ids[cur_state]))
                cur_state = int(back[cur_state])
            chosen_frames.reverse()
            chosen_candidate_ids.reverse()
            return chosen_frames, chosen_candidate_ids, best_raw, best_budget

        def decode_for_recall_mu(recall_mu: float) -> tuple[list[int], list[int], float, float, float]:
            best: tuple[list[int], list[int], float, float] | None = None
            lo = 0.0
            hi = float(args.penalty_max)
            for _ in range(max(1, int(args.penalty_binary_steps))):
                mid = 0.5 * (lo + hi)
                cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
                candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
                if best is None:
                    best = candidate
                else:
                    cur_frames, cur_ids, cur_raw, cur_budget = best
                    cand_gap = abs(len(cand_frames) - int(target_count))
                    cur_gap = abs(len(cur_frames) - int(target_count))
                    if cand_gap < cur_gap or (
                        cand_gap == cur_gap
                        and (
                            len(cand_frames) < len(cur_frames)
                            or (len(cand_frames) == len(cur_frames) and (cand_budget < cur_budget or (abs(cand_budget - cur_budget) <= 1e-9 and cand_raw < cur_raw)))
                        )
                    ):
                        best = candidate
                if len(cand_frames) > int(target_count):
                    lo = mid
                else:
                    hi = mid
            assert best is not None
            best_frames, best_ids, best_raw, best_budget = best
            return best_frames, best_ids, best_raw, best_budget, float(hi)

        if use_exact_recall_dp:
            best_result: tuple[list[int], list[int], float, float, float] | None = None
            recall_lo = 0.0
            recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
            for _ in range(max(1, int(args.recall_budget_binary_steps))):
                recall_mid = 0.5 * (recall_lo + recall_hi)
                cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda = decode_for_recall_mu(recall_mid)
                cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
                if best_result is None:
                    best_result = (cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda)
                else:
                    _bf, _bi, best_raw, best_budget, best_lambda = best_result
                    best_violation = recall_violation(best_budget, len(run.frame_numbers), args)
                    if cand_violation < best_violation - 1e-12 or (
                        abs(cand_violation - best_violation) <= 1e-12 and (cand_raw < best_raw or (abs(cand_raw - best_raw) <= 1e-9 and cand_lambda < best_lambda))
                    ):
                        best_result = (cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda)
                if cand_violation > 0.0:
                    recall_lo = recall_mid
                else:
                    recall_hi = recall_mid
            assert best_result is not None
            best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
        else:
            best_frames, best_ids, _best_raw, _best_budget, best_lambda = decode_for_recall_mu(0.0)
        return best_frames, best_ids, counters, cost_cache, float(best_lambda)


    def run_single_state_penalty_path(
        run: InstanceRun,
        candidate_frames: list[int],
        candidates_by_frame: list[list[ShapeCandidate]],
        target_count: int,
        args: argparse.Namespace,
        eval_contexts: list[FrameEvalContext] | None = None,
    ) -> tuple[list[int], list[int], dict[str, int], dict[tuple[int, int, int, int, int], IntervalCost], float]:
        target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
        dynamic_max_gap = max(int(args.max_gap), int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))))
        node_count = int(len(candidate_frames))
        cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
        edge_cache: dict[tuple[int, int], IntervalCost] = {}
        counters = {"interval_evals": 0, "interval_frames": 0}
        use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
        recall_penalty_weight = float(args.proxy_recall_penalty_weight)

        predecessor_nodes: list[list[int]] = []
        for node_pos, end_frame in enumerate(candidate_frames):
            end_frame_i = int(end_frame)
            min_prev_pos = int(bisect.bisect_left(candidate_frames, end_frame_i - int(dynamic_max_gap), 0, node_pos))
            predecessor_nodes.append(list(range(min_prev_pos, node_pos)))

        def get_edge_info(prev_node_pos: int, node_pos: int) -> IntervalCost:
            key = (int(prev_node_pos), int(node_pos))
            cached = edge_cache.get(key)
            if cached is not None:
                return cached
            start_frame = int(candidate_frames[prev_node_pos])
            end_frame = int(candidate_frames[node_pos])
            start_candidate = candidates_by_frame[start_frame][0]
            end_candidate = candidates_by_frame[end_frame][0]
            info = interval_cost_from_vectors(
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
            edge_cache[key] = info
            cost_cache[(start_frame, 0, end_frame, 0, 0)] = info
            counters["interval_evals"] += 1
            counters["interval_frames"] += int(info.frames_covered)
            return info

        def decode(lambda_penalty: float, recall_mu: float) -> tuple[list[int], list[int], float, float]:
            dp = np.full((node_count,), np.inf, dtype=np.float64)
            back = np.full((node_count,), -1, dtype=np.int32)
            raw_cost = np.full((node_count,), np.inf, dtype=np.float64)
            raw_budget = np.full((node_count,), np.inf, dtype=np.float64)

            first_candidate = candidates_by_frame[int(candidate_frames[0])][0]
            first_budget = float(first_candidate.recall_budget)
            first_penalty = float(recall_mu) * first_budget if use_exact_recall_dp else recall_penalty_weight * first_budget
            dp[0] = float(first_candidate.frame_loss) + first_penalty + float(lambda_penalty)
            raw_cost[0] = float(first_candidate.frame_loss)
            raw_budget[0] = float(first_budget)

            for node_pos in range(1, node_count):
                best_cost = float("inf")
                best_raw = float("inf")
                best_budget = float("inf")
                best_prev = -1
                for prev_node_pos in predecessor_nodes[node_pos]:
                    prev_cost = float(dp[prev_node_pos])
                    if not np.isfinite(prev_cost):
                        continue
                    info = get_edge_info(prev_node_pos, node_pos)
                    edge_budget = float(info.recall_budget)
                    penalty = float(recall_mu) * edge_budget if use_exact_recall_dp else recall_penalty_weight * edge_budget
                    cand_cost = prev_cost + float(info.cost) + penalty + float(lambda_penalty)
                    cand_raw = float(raw_cost[prev_node_pos]) + float(info.cost)
                    cand_budget = float(raw_budget[prev_node_pos]) + edge_budget
                    if cand_cost < best_cost or (
                        abs(cand_cost - best_cost) <= 1e-9
                        and (cand_budget < best_budget or (abs(cand_budget - best_budget) <= 1e-9 and cand_raw < best_raw))
                    ):
                        best_cost = float(cand_cost)
                        best_raw = float(cand_raw)
                        best_budget = float(cand_budget)
                        best_prev = int(prev_node_pos)
                dp[node_pos] = best_cost
                raw_cost[node_pos] = best_raw
                raw_budget[node_pos] = best_budget
                back[node_pos] = int(best_prev)

            last_pos = int(node_count - 1)
            if last_pos < 0 or not np.isfinite(dp[last_pos]):
                raise RuntimeError("failed to decode single-state penalized path")

            chosen_frames: list[int] = []
            cur_pos = last_pos
            while cur_pos >= 0:
                chosen_frames.append(int(candidate_frames[cur_pos]))
                cur_pos = int(back[cur_pos])
            chosen_frames.reverse()
            chosen_candidate_ids = [0] * len(chosen_frames)
            return chosen_frames, chosen_candidate_ids, float(raw_cost[last_pos]), float(raw_budget[last_pos])

        def decode_for_recall_mu(recall_mu: float) -> tuple[list[int], list[int], float, float, float]:
            best: tuple[list[int], list[int], float, float] | None = None
            lo = 0.0
            hi = float(args.penalty_max)
            for _ in range(max(1, int(args.penalty_binary_steps))):
                mid = 0.5 * (lo + hi)
                cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
                candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
                if best is None:
                    best = candidate
                else:
                    cur_frames, _cur_ids, cur_raw, cur_budget = best
                    cand_gap = abs(len(cand_frames) - int(target_count))
                    cur_gap = abs(len(cur_frames) - int(target_count))
                    if cand_gap < cur_gap or (
                        cand_gap == cur_gap
                        and (
                            len(cand_frames) < len(cur_frames)
                            or (len(cand_frames) == len(cur_frames) and (cand_budget < cur_budget or (abs(cand_budget - cur_budget) <= 1e-9 and cand_raw < cur_raw)))
                        )
                    ):
                        best = candidate
                if len(cand_frames) > int(target_count):
                    lo = mid
                else:
                    hi = mid
            assert best is not None
            best_frames, best_ids, best_raw, best_budget = best
            return best_frames, best_ids, best_raw, best_budget, float(hi)

        if use_exact_recall_dp:
            best_result: tuple[list[int], list[int], float, float, float] | None = None
            recall_lo = 0.0
            recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
            for _ in range(max(1, int(args.recall_budget_binary_steps))):
                recall_mid = 0.5 * (recall_lo + recall_hi)
                cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda = decode_for_recall_mu(recall_mid)
                cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
                if best_result is None:
                    best_result = (cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda)
                else:
                    _bf, _bi, best_raw, best_budget, best_lambda = best_result
                    best_violation = recall_violation(best_budget, len(run.frame_numbers), args)
                    if cand_violation < best_violation - 1e-12 or (
                        abs(cand_violation - best_violation) <= 1e-12 and (cand_raw < best_raw or (abs(cand_raw - best_raw) <= 1e-9 and cand_lambda < best_lambda))
                    ):
                        best_result = (cand_frames, cand_ids, cand_raw, cand_budget, cand_lambda)
                if cand_violation > 0.0:
                    recall_lo = recall_mid
                else:
                    recall_hi = recall_mid
            assert best_result is not None
            best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
        else:
            best_frames, best_ids, _best_raw, _best_budget, best_lambda = decode_for_recall_mu(0.0)
        return best_frames, best_ids, counters, cost_cache, float(best_lambda)


    def evaluate_keyframe_path(
        run: InstanceRun,
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
        args: argparse.Namespace,
        eval_contexts: list[FrameEvalContext] | None = None,
    ) -> tuple[float, list[IntervalCost], float]:
        total, _start_loss, interval_infos, total_recall_budget, _start_budget = evaluate_keyframe_path_parts(
            run,
            chosen_frames,
            keyframe_vectors,
            args,
            eval_contexts=eval_contexts,
        )
        return float(total), interval_infos, float(total_recall_budget)


    def evaluate_keyframe_path_parts(
        run: InstanceRun,
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
        args: argparse.Namespace,
        eval_contexts: list[FrameEvalContext] | None = None,
    ) -> tuple[float, float, list[IntervalCost], float, float]:
        total = 0.0
        interval_infos: list[IntervalCost] = []
        if len(chosen_frames) <= 0:
            return float("inf"), float("inf"), interval_infos, float("inf"), float("inf")
        start_vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
        start_loss, start_budget = evaluate_frame_vector_loss_budget(run, int(chosen_frames[0]), start_vec, args, eval_contexts=eval_contexts)
        total_recall_budget = float(start_budget)
        total += float(start_loss)
        for left_idx, right_idx, left_vec, right_vec in zip(
            chosen_frames[:-1],
            chosen_frames[1:],
            keyframe_vectors[:-1],
            keyframe_vectors[1:],
        ):
            info = interval_cost_from_vectors(
                run,
                int(left_idx),
                np.asarray(left_vec, dtype=np.float32),
                int(right_idx),
                np.asarray(right_vec, dtype=np.float32),
                args,
                include_start=False,
                eval_contexts=eval_contexts,
            )
            interval_infos.append(info)
            total += float(info.cost)
            total_recall_budget += float(info.recall_budget)
        return float(total), float(start_loss), interval_infos, float(total_recall_budget), float(start_budget)


    def exact_interpolated_metrics(
        run: InstanceRun,
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
    ) -> tuple[list[dict[str, float]], float, float, float, float, float]:
        metrics_rows: list[dict[str, float]] = []
        total_iou_loss = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_gt_area = 0.0
        total_intersection = 0.0
        chosen_frames_arr = [int(v) for v in chosen_frames]
        interval_pos = 0
        for frame_idx in range(len(run.frame_numbers)):
            if frame_idx <= chosen_frames_arr[0]:
                vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
            elif frame_idx >= chosen_frames_arr[-1]:
                vec = np.asarray(keyframe_vectors[-1], dtype=np.float32)
            else:
                while interval_pos + 1 < len(chosen_frames_arr) and frame_idx > int(chosen_frames_arr[interval_pos + 1]):
                    interval_pos += 1
                right_pos = int(interval_pos + 1)
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen_frames_arr[left_pos])
                right_frame = int(chosen_frames_arr[right_pos])
                if frame_idx == right_frame:
                    vec = np.asarray(keyframe_vectors[right_pos], dtype=np.float32)
                else:
                    alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                    vec = interpolate_vectors(keyframe_vectors[left_pos], keyframe_vectors[right_pos], alpha)
            pred_polys = split_vector_to_polygons(vec, run.contour_count, run.anchors_per_contour)
            metrics = compute_exact_metrics_from_polygons(run.gt_polygons[frame_idx], pred_polys)
            metrics_rows.append(metrics)
            total_iou_loss += 1.0 - float(metrics["iou"])
            total_recall += float(metrics["recall"])
            total_precision += float(metrics["precision"])
            total_gt_area += float(metrics["gt_area"])
            total_intersection += float(metrics["intersection"])
        mean_iou = float(1.0 - total_iou_loss / max(len(metrics_rows), 1))
        mean_recall = float(total_recall / max(len(metrics_rows), 1))
        mean_precision = float(total_precision / max(len(metrics_rows), 1))
        global_recall = float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0
        return metrics_rows, float(total_iou_loss), float(mean_iou), float(mean_recall), float(mean_precision), float(global_recall)


    def exact_recall_solution_key(total_iou_loss: float, mean_recall: float, args: argparse.Namespace) -> tuple[float, float, float]:
        violation = max(float(args.recall_min) - float(mean_recall), 0.0)
        return float(violation), float(total_iou_loss), float(-mean_recall)


    def repair_keyframe_vectors_for_exact_recall(
        run: InstanceRun,
        chosen_frames: list[int],
        keyframe_vectors: np.ndarray,
        candidates_by_frame: list[list[ShapeCandidate]],
        args: argparse.Namespace,
    ) -> np.ndarray:
        if not bool(args.exact_recall_repair_enabled) or len(chosen_frames) <= 0:
            return np.asarray(keyframe_vectors, dtype=np.float32)
        current = np.asarray(keyframe_vectors, dtype=np.float32).copy()
        scale_deltas = parse_float_list(str(args.exact_recall_repair_scale_deltas), [0.01, 0.02, 0.04, 0.06, 0.08])
        metrics_rows, current_iou_loss, _current_mean_iou, current_mean_recall, _current_mean_precision, _current_global_recall = exact_interpolated_metrics(run, chosen_frames, current)
        best_key = exact_recall_solution_key(current_iou_loss, current_mean_recall, args)
        if best_key[0] <= 0.0:
            return current

        for _pass in range(max(1, int(args.exact_recall_repair_max_passes))):
            frame_deficits = np.asarray(
                [float(row["gt_area"]) * max(float(args.recall_min) - float(row["recall"]), 0.0) for row in metrics_rows],
                dtype=np.float64,
            )
            if float(np.mean(frame_deficits)) <= 0.0 and best_key[0] <= 0.0:
                break
            key_scores = np.zeros((len(chosen_frames),), dtype=np.float64)
            for frame_idx, deficit in enumerate(frame_deficits.tolist()):
                if deficit <= 0.0:
                    continue
                if frame_idx <= int(chosen_frames[0]):
                    key_scores[0] += float(deficit)
                    continue
                if frame_idx >= int(chosen_frames[-1]):
                    key_scores[-1] += float(deficit)
                    continue
                right_pos = next(pos for pos, keyframe in enumerate(chosen_frames) if keyframe >= frame_idx)
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen_frames[left_pos])
                right_frame = int(chosen_frames[right_pos])
                alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                key_scores[left_pos] += (1.0 - alpha) * float(deficit)
                key_scores[right_pos] += alpha * float(deficit)
            key_order = [int(idx) for idx in np.argsort(-key_scores)[: max(1, int(args.exact_recall_repair_topk))].tolist()]
            improved = False

            trial_vectors: list[np.ndarray] = []
            for delta in scale_deltas:
                scaled_all = np.asarray(current, dtype=np.float32).copy()
                for key_idx in range(len(chosen_frames)):
                    scaled_all[key_idx] = scale_vector_about_centroid(scaled_all[key_idx], 1.0 + float(delta))
                trial_vectors.append(scaled_all)
            for delta in scale_deltas:
                scaled = np.asarray(current, dtype=np.float32).copy()
                for key_idx in key_order:
                    scaled[key_idx] = scale_vector_about_centroid(scaled[key_idx], 1.0 + float(delta))
                trial_vectors.append(scaled)

            for key_idx in key_order:
                frame_idx = int(chosen_frames[key_idx])
                current_area, _center, _radii, _mean_radius = vector_proxy_stats(current[key_idx], run.contour_count, run.anchors_per_contour)
                for candidate in candidates_by_frame[frame_idx]:
                    if float(candidate.area) <= float(current_area) + 1e-3:
                        continue
                    upgraded = np.asarray(current, dtype=np.float32).copy()
                    upgraded[key_idx] = np.asarray(candidate.vector, dtype=np.float32)
                    trial_vectors.append(upgraded)
                for delta in scale_deltas:
                    upgraded = np.asarray(current, dtype=np.float32).copy()
                    upgraded[key_idx] = scale_vector_about_centroid(upgraded[key_idx], 1.0 + float(delta))
                    trial_vectors.append(upgraded)

            seen: list[np.ndarray] = []
            for trial in trial_vectors:
                if any(np.allclose(trial, existing, atol=1e-4) for existing in seen):
                    continue
                seen.append(np.asarray(trial, dtype=np.float32))
                trial_metrics, trial_iou_loss, _trial_mean_iou, trial_mean_recall, _trial_mean_precision, _trial_global_recall = exact_interpolated_metrics(run, chosen_frames, trial)
                trial_key = exact_recall_solution_key(trial_iou_loss, trial_mean_recall, args)
                if trial_key < best_key:
                    current = np.asarray(trial, dtype=np.float32)
                    metrics_rows = trial_metrics
                    best_key = trial_key
                    improved = True
            if not improved:
                break
            if best_key[0] <= 0.0:
                break
        return np.asarray(current, dtype=np.float32)


    class LazyInterpolatedRun:
        def __init__(self, run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray):
            self.run = run
            self.chosen_frames = [int(v) for v in chosen_frames]
            self.keyframe_vectors = np.asarray(keyframe_vectors, dtype=np.float32)
            self.length = int(len(run.frame_numbers))

        def __len__(self) -> int:
            return int(self.length)

        def _polygons_at(self, frame_idx: int) -> list[np.ndarray]:
            idx = int(frame_idx)
            if idx < 0:
                idx += int(self.length)
            if idx < 0 or idx >= int(self.length):
                raise IndexError(frame_idx)
            chosen = self.chosen_frames
            if idx <= chosen[0]:
                vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
            elif idx >= chosen[-1]:
                vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
            else:
                right_pos = int(np.searchsorted(np.asarray(chosen, dtype=np.int32), int(idx), side="left"))
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen[left_pos])
                right_frame = int(chosen[right_pos])
                if idx == right_frame:
                    vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
                else:
                    alpha = float((idx - left_frame) / max(right_frame - left_frame, 1))
                    vec = interpolate_vectors(self.keyframe_vectors[left_pos], self.keyframe_vectors[right_pos], alpha)
            return split_vector_to_polygons(vec, self.run.contour_count, self.run.anchors_per_contour)

        def __getitem__(self, frame_idx):
            if isinstance(frame_idx, slice):
                return [self._polygons_at(idx) for idx in range(*frame_idx.indices(int(self.length)))]
            return self._polygons_at(int(frame_idx))

        def __iter__(self):
            if self.length <= 0:
                return
            chosen = self.chosen_frames
            interval_pos = 0
            for frame_idx in range(int(self.length)):
                if frame_idx <= chosen[0]:
                    vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
                elif frame_idx >= chosen[-1]:
                    vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
                else:
                    while interval_pos + 1 < len(chosen) and frame_idx > int(chosen[interval_pos + 1]):
                        interval_pos += 1
                    right_pos = int(interval_pos + 1)
                    left_pos = max(0, right_pos - 1)
                    left_frame = int(chosen[left_pos])
                    right_frame = int(chosen[right_pos])
                    if frame_idx == right_frame:
                        vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
                    else:
                        alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
                        vec = interpolate_vectors(self.keyframe_vectors[left_pos], self.keyframe_vectors[right_pos], alpha)
                yield split_vector_to_polygons(vec, self.run.contour_count, self.run.anchors_per_contour)


    def interpolate_run(run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray):
        length = len(run.frame_numbers)
        if length <= 0:
            return []
        return LazyInterpolatedRun(run, chosen_frames, keyframe_vectors)


    class LazyUnionRows:
        def __init__(self, run: InstanceRun, interp_polygons, chosen_frames: list[int]):
            self.run = run
            self.interp_polygons = interp_polygons
            self.chosen_set = {int(v) for v in chosen_frames}
            length = int(len(run.frame_numbers))
            self.emit_start = int(max(0, min(length, int(run.emit_start_idx))))
            emit_end = length if int(run.emit_end_idx) < 0 else int(run.emit_end_idx)
            self.emit_end = int(max(self.emit_start, min(length, emit_end)))

        def __len__(self) -> int:
            return int(self.emit_end - self.emit_start)

        def __iter__(self):
            for local_idx, (frame, polygons) in enumerate(zip(self.run.frame_numbers.tolist(), self.interp_polygons)):
                if local_idx < self.emit_start or local_idx >= self.emit_end:
                    continue
                yield {
                    "frame": int(frame),
                    "track_id": str(self.run.track_id),
                    "run_id": int(self.run.run_id),
                    "polygons": [np.asarray(poly, dtype=np.float32).tolist() for poly in polygons],
                    "has_keyframe": 1 if local_idx in self.chosen_set else 0,
                    "is_gapfill": int(self.run.gapfilled_flags[local_idx])
                    if self.run.gapfilled_flags is not None and local_idx < len(self.run.gapfilled_flags)
                    else 0,
                }


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
        candidate_frames, surrogate_frames, saliency_scores = build_candidate_frame_pool(run, candidates_by_frame, target_count, args)
        stage_times["build_candidate_pool_seconds"] = float(time.perf_counter() - stage_t0)

        stage_t0 = time.perf_counter()
        chosen_frames, chosen_candidate_ids, counters, _cache, _best_lambda = run_multistate_penalty_path(
            run,
            candidate_frames,
            candidates_by_frame,
            target_count,
            args,
            eval_contexts=eval_contexts,
        )
        stage_times["solve_dp_seconds"] = float(time.perf_counter() - stage_t0)

        keyframe_vectors = np.asarray(
            [np.asarray(candidates_by_frame[int(frame_idx)][int(cand_id)].vector, dtype=np.float32) for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)],
            dtype=np.float32,
        )

        stage_t0 = time.perf_counter()
        keyframe_vectors = pair_vote_refine_keyframe_vectors(run, chosen_frames, keyframe_vectors, args)
        stage_times["pair_vote_refine_seconds"] = float(time.perf_counter() - stage_t0)

        stage_t0 = time.perf_counter()
        keyframe_vectors = repair_keyframe_vectors_for_exact_recall(run, chosen_frames, keyframe_vectors, candidates_by_frame, args)
        stage_times["exact_recall_repair_seconds"] = float(time.perf_counter() - stage_t0)

        chosen_candidate_ids = assign_candidate_ids_to_keyframes(chosen_frames, keyframe_vectors, candidates_by_frame)

        stage_t0 = time.perf_counter()
        objective, interval_infos, total_recall_budget = evaluate_keyframe_path(run, chosen_frames, keyframe_vectors, args, eval_contexts=eval_contexts)
        interp_polygons = interpolate_run(run, chosen_frames, keyframe_vectors)
        stage_times["final_eval_seconds"] = float(time.perf_counter() - stage_t0)

        chosen_frame_to_candidate = {int(frame_idx): int(cand_id) for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)}
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
                        for poly in split_vector_to_polygons(keyframe_vectors[keyframe_pos], run.contour_count, run.anchors_per_contour)
                    ],
                }
            )

        shape_distance_total = float(np.sum(np.asarray([info.shape_distance for info in interval_infos], dtype=np.float64))) if interval_infos else 0.0
        shape_update_count = float(np.sum(np.asarray([info.shape_update for info in interval_infos], dtype=np.float64))) if interval_infos else 0.0
        mean_shape_distance = float(np.mean(np.asarray([info.shape_distance for info in interval_infos], dtype=np.float64))) if interval_infos else 0.0
        mean_shape_update = float(np.mean(np.asarray([info.shape_update for info in interval_infos], dtype=np.float64))) if interval_infos else 0.0
        mean_interval_frame_loss = float(np.mean(np.asarray([info.frame_loss_mean for info in interval_infos], dtype=np.float64))) if interval_infos else 0.0
        mean_shape_distance_scale = float(np.mean(np.asarray([info.shape_distance_scale for info in interval_infos], dtype=np.float64))) if interval_infos else 1.0
        mean_shape_switch_scale = float(np.mean(np.asarray([info.shape_switch_scale for info in interval_infos], dtype=np.float64))) if interval_infos else 1.0
        mean_recall_budget = float(total_recall_budget / max(length, 1))
        achieved_recall_floor = float(max(0.0, 1.0 - mean_recall_budget))
        recall_budget_violation = float(recall_violation(total_recall_budget, length, args))
        keyframe_rate = float(len(chosen_frames) / max(length, 1))
        gapfilled_frame_count = int(np.sum(run.gapfilled_flags.astype(np.int32))) if run.gapfilled_flags is not None else 0
        shape_update_rate = float(shape_update_count / max(length, 1))
        shape_distance_rate = float(shape_distance_total / max(length, 1))
        mean_state_count = float(np.mean(np.asarray([len(frame_candidates) for frame_candidates in candidates_by_frame], dtype=np.float64))) if candidates_by_frame else 0.0
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
            "predicted_total_points_p90": float(np.quantile(run.predicted_total_points.astype(np.float64), 0.90)) if run.predicted_total_points is not None and len(run.predicted_total_points) > 0 else 0.0,
            "predicted_total_points_mean": float(np.mean(run.predicted_total_points.astype(np.float64))) if run.predicted_total_points is not None and len(run.predicted_total_points) > 0 else 0.0,
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
            "max_saliency": float(np.max(saliency_scores)) if saliency_scores.size > 0 else 0.0,
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


    def write_compact_json_array(output_path: Path, rows) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("[")
            first = True
            for row in rows:
                if first:
                    first = False
                else:
                    f.write(",")
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("]")


    class SqliteUnionRowStore:
        def __init__(self, store_path: Path):
            self.store_path = Path(store_path)
            if self.store_path.exists():
                self.store_path.unlink()
            self.conn = sqlite3.connect(str(self.store_path))
            self.conn.execute(
                "CREATE TABLE union_rows (frame INTEGER NOT NULL, track_id TEXT NOT NULL, track_sort INTEGER NOT NULL, row_json TEXT NOT NULL)"
            )
            self.conn.execute("CREATE INDEX idx_union_rows_order ON union_rows(frame, track_sort)")
            self.row_count = 0

        def add_rows(self, rows) -> int:
            inserted = 0

            def iter_records():
                nonlocal inserted
                for row in rows:
                    inserted += 1
                    track_id = str(row["track_id"])
                    yield (
                        int(row["frame"]),
                        track_id,
                        int(track_id),
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                    )

            self.conn.executemany(
                "INSERT INTO union_rows(frame, track_id, track_sort, row_json) VALUES (?, ?, ?, ?)",
                iter_records(),
            )
            self.row_count += int(inserted)
            return int(inserted)

        def commit(self) -> None:
            self.conn.commit()

        def iter_rows_sorted(self):
            self.commit()
            for (row_json,) in self.conn.execute("SELECT row_json FROM union_rows ORDER BY frame, track_sort"):
                yield json.loads(str(row_json))

        def write_union_json(self, output_path: Path) -> None:
            write_compact_json_array(output_path, self.iter_rows_sorted())

        def write_pred_sqlite(self, output_sqlite: Path) -> None:
            output_sqlite.parent.mkdir(parents=True, exist_ok=True)
            if output_sqlite.exists():
                output_sqlite.unlink()
            self.commit()
            out_conn = sqlite3.connect(str(output_sqlite))
            try:
                cur = out_conn.cursor()
                cur.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
                cur.executemany(
                    "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                    (
                        (
                            int(row["frame"]),
                            str(row["track_id"]),
                            json.dumps(row["polygons"], ensure_ascii=False),
                        )
                        for row in self.iter_rows_sorted()
                    ),
                )
                out_conn.commit()
            finally:
                out_conn.close()

        def evaluate_exact(self, tracked_sqlite: Path, output_dir: Path) -> dict[str, object]:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.commit()
            metrics_csv = output_dir / "keyframe_exact_metrics.csv"
            attached = False
            totals = {
                "row_count": 0.0,
                "gt_area": 0.0,
                "pred_area": 0.0,
                "intersection": 0.0,
                "union": 0.0,
                "weighted_error_total": 0.0,
                "recall_sum": 0.0,
                "precision_sum": 0.0,
                "iou_sum": 0.0,
            }
            try:
                self.conn.execute("ATTACH DATABASE ? AS tracked_eval", (str(tracked_sqlite),))
                attached = True
                rows_iter = self.conn.execute(
                    """
                    SELECT m.frame, m.track_id, m.polygons, u.row_json
                    FROM tracked_eval.masks AS m
                    JOIN union_rows AS u
                      ON u.frame = m.frame AND u.track_id = CAST(m.track_id AS TEXT)
                    ORDER BY m.frame, CAST(m.track_id AS INTEGER)
                    """
                )
                with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
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
                    for frame, track_id, polygons_json, row_json in rows_iter:
                        pred = json.loads(str(row_json))
                        gt_polys = parse_polygons(str(polygons_json))
                        pred_polys = [np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in pred["polygons"]]
                        metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
                        weighted_error = float(compute_weighted_error(metrics))
                        result_row = {
                            "frame": int(frame),
                            "track_id": str(track_id),
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
                        writer.writerow(result_row)
                        totals["row_count"] += 1.0
                        totals["gt_area"] += float(result_row["gt_area"])
                        totals["pred_area"] += float(result_row["pred_area"])
                        totals["intersection"] += float(result_row["intersection"])
                        totals["union"] += float(result_row["union"])
                        totals["weighted_error_total"] += weighted_error
                        totals["recall_sum"] += float(result_row["recall"])
                        totals["precision_sum"] += float(result_row["precision"])
                        totals["iou_sum"] += float(result_row["iou"])
            finally:
                if attached:
                    self.conn.execute("DETACH DATABASE tracked_eval")
            row_count = float(totals["row_count"])
            gt_area = float(totals["gt_area"])
            pred_area = float(totals["pred_area"])
            intersection = float(totals["intersection"])
            union = float(totals["union"])
            weighted_error = float(totals["weighted_error_total"])
            optimized = {
                "row_count": row_count,
                "gt_area": gt_area,
                "pred_area": pred_area,
                "intersection": intersection,
                "union": union,
                "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
                "global_precision": float(intersection / pred_area) if pred_area > 0 else 1.0,
                "global_iou": float(intersection / union) if union > 0 else 1.0,
                "mean_recall": float(totals["recall_sum"] / max(row_count, 1.0)),
                "mean_precision": float(totals["precision_sum"] / max(row_count, 1.0)),
                "mean_iou": float(totals["iou_sum"] / max(row_count, 1.0)),
                "weighted_error_total": weighted_error,
                "weighted_error_mean": float(weighted_error / max(row_count, 1.0)),
            }
            summary = {
                "input_tracked_sqlite": str(tracked_sqlite),
                "optimized": optimized,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        def close(self, unlink: bool = False) -> None:
            self.conn.close()
            if bool(unlink):
                try:
                    self.store_path.unlink()
                except FileNotFoundError:
                    pass


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
            predictor = LearnedPointPredictor(Path(args.point_predictor_model_dir), str(args.predictor_device))
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
                total_stage_times[str(key)] = float(total_stage_times.get(str(key), 0.0) + float(value))

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
            with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_ctx) as executor:
                results = list(executor.map(process_single_run, runs, [args] * len(runs)))
            for result in results:
                union_rows_all.extend(result["union_rows"])
                collect_result(result)
            union_rows = sorted(union_rows_all, key=lambda row: (int(row["frame"]), int(str(row["track_id"]))))
            union_row_count = int(len(union_rows))
        chunk_counts_by_run: dict[tuple[str, int], int] = {}
        for row in stream_rows:
            if int(row.get("chunk_count", 1)) < 0:
                key = (str(row["track_id"]), int(row["run_id"]))
                chunk_counts_by_run[key] = max(int(chunk_counts_by_run.get(key, 0)), int(row["chunk_index"]) + 1)
        for row in stream_rows:
            if int(row.get("chunk_count", 1)) < 0:
                key = (str(row["track_id"]), int(row["run_id"]))
                chunk_count = int(chunk_counts_by_run.get(key, int(row["chunk_index"]) + 1))
                chunk_index = int(row["chunk_index"])
                row["chunk_count"] = int(chunk_count)
                row["stream_id"] = str(row["stream_id"]).replace(f":chunk{chunk_index + 1}:instance", f":chunk{chunk_index + 1}of{chunk_count}:instance")
        opt_dir.mkdir(parents=True, exist_ok=True)
        if union_store is not None:
            union_store.write_union_json(opt_dir / "interpolated_union.json")
        else:
            (opt_dir / "interpolated_union.json").write_text(json.dumps(union_rows, ensure_ascii=False, indent=2), encoding="utf-8")
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
            "description": "Readable standalone of the practical v22 polygon keyframe optimizer with gapfill-first track-level anchor counts and pair-vote keyframe-shape refinement.",
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
            "point_predictor_model_dir": str(args.point_predictor_model_dir) if bool(args.adaptive_anchor_counts) else None,
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
            "segmentation_stats": {key: int(val) for key, val in segmentation_stats.items()},
            "mean_run_anchors_per_contour": float(np.mean(np.asarray([row["anchors_per_contour"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "median_run_anchors_per_contour": float(np.median(np.asarray([row["anchors_per_contour"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "max_run_anchors_per_contour": int(max(int(row["anchors_per_contour"]) for row in stream_rows)) if stream_rows else 0,
            "mean_gapfilled_frame_count": float(np.mean(np.asarray([row["gapfilled_frame_count"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_achieved_ratio": float(np.mean(np.asarray([row["achieved_ratio"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_keyframe_rate": float(np.mean(np.asarray([row["keyframe_rate"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_state_count": float(np.mean(np.asarray([row["mean_state_count"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_interval_frame_loss": float(np.mean(np.asarray([row["mean_interval_frame_loss"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_shape_distance": float(np.mean(np.asarray([row["mean_shape_distance"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_shape_update": float(np.mean(np.asarray([row["mean_shape_update"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_shape_distance_rate": float(np.mean(np.asarray([row["shape_distance_rate"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_shape_update_rate": float(np.mean(np.asarray([row["shape_update_rate"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_recall_budget": float(np.mean(np.asarray([row["mean_recall_budget"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_achieved_recall_floor": float(np.mean(np.asarray([row["achieved_recall_floor"] for row in stream_rows], dtype=np.float64))) if stream_rows else 1.0,
            "mean_recall_budget_violation": float(np.mean(np.asarray([row["recall_budget_violation"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "mean_candidate_frame_count": float(np.mean(np.asarray([row["candidate_frame_count"] for row in stream_rows], dtype=np.float64))) if stream_rows else 0.0,
            "interval_eval_count": int(total_interval_evals),
            "interval_eval_frames": int(total_interval_frames),
            "candidate_frame_count_total": int(total_candidate_frames),
            "optimizer_seconds": float(optimizer_seconds),
            "stage_seconds_total": {key: float(val) for key, val in sorted(total_stage_times.items())},
            "stage_seconds_mean_per_run": {key: float(val / max(run_count, 1)) for key, val in sorted(total_stage_times.items())},
            "artifacts": {
                "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
                "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
                "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
            },
        }
        (opt_dir / "summary.json").write_text(json.dumps(optimizer_summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
                exact_summary = evaluate_union_exact(union_rows, args.input_sqlite, exact_dir)
            if bool(args.write_pred_sqlite):
                union_rows_to_pred_sqlite(union_rows, pred_sqlite)

        summary = {
            "description": "Practical v22 polygon keyframe optimizer with gapfill-first track-level anchor counts.",
            "input_sqlite": str(args.input_sqlite),
            "output_dir": str(output_dir),
            "optimizer_summary": optimizer_summary,
            "exact_summary": exact_summary,
            "artifacts": {
                "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
                "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
                "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
                "optimizer_summary_json": str(opt_dir / "summary.json"),
                "exact_summary_json": None if exact_summary is None else str(exact_dir / "summary.json"),
                "pred_sqlite": str(pred_sqlite) if bool(args.write_pred_sqlite) else None,
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == '__main__':
        main()
    # --- END READABLE EMBEDDED POLYGON V22 SOURCE ---
