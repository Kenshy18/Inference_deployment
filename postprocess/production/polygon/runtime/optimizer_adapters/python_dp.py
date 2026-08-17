"""Memory-bounded Python DP adapter for the Production optimizer."""

from __future__ import annotations

from types import ModuleType

from ..kernel import candidates as kernel_candidates


def install_python_dp_adapter(module: ModuleType) -> None:
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

    module.exact_k_dp = fast_exact_k_dp
    kernel_candidates.exact_k_dp = fast_exact_k_dp


__all__ = ("install_python_dp_adapter",)
