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
    const int64_t* edge_offsets,
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
            const int64_t edge_idx = edge_offsets[node_pos] +
                static_cast<int64_t>(prev_node_pos - begin);
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
    const int64_t* edge_offsets,
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
            edge_offsets,
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
    const int64_t* edge_offsets,
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
        edge_offsets == nullptr ||
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
                edge_offsets,
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
            edge_offsets,
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
