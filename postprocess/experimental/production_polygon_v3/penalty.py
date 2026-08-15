"""Soft key-penalty refinement for a dual-Recall-safe Production V3 path."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from ..polygon_recall_optimizer.fixed_budget import (
    Keyframe,
    RawMask,
    Segment,
    _primary_component,
)
from ..polygon_recall_optimizer.pareto_dp import canonicalize_selected_path
from ..polygon_recall_optimizer.superior import BorderFrameConstraint
from .optimizer import _dual_safe_anchor, _edge_metrics


@dataclass(frozen=True)
class _Edge:
    start: int
    end: int
    quality_loss: float


@dataclass(frozen=True)
class _Graph:
    segment: Segment
    keys: tuple[Keyframe, ...]
    incoming: tuple[tuple[_Edge, ...], ...]


@dataclass(frozen=True)
class _Decoded:
    segments: dict[str, list[Segment]]
    keyframe_count: int
    quality_loss: float
    penalty: float


@dataclass(frozen=True)
class ProductionV3PenaltyResult:
    segments: dict[str, list[Segment]]
    keyframe_count: int
    target_keyframe_count: int
    target_mean_key_interval: float
    actual_mean_key_interval: float
    quality_loss: float
    selected_penalty: float
    evaluated_edges: int
    feasible_edges: int
    elapsed_seconds: float
    decoded_candidate_count: int


def _build_graph(
    segment: Segment,
    raw_by_frame: dict[int, RawMask],
    border_by_frame: dict[int, BorderFrameConstraint],
    *,
    normal_recall_floor: float,
    max_edge_span_frames: int,
) -> tuple[_Graph, int, int]:
    keys = tuple(segment.keyframes)
    if len(keys) < 2:
        return _Graph(segment, keys, tuple(() for _ in keys)), 0, 0
    incoming: list[list[_Edge]] = [[] for _ in keys]
    evaluated = 0
    feasible_count = 0
    for end in range(1, len(keys)):
        for start in range(end - 1, -1, -1):
            if keys[end].frame - keys[start].frame > max_edge_span_frames:
                break
            evaluated += 1
            feasible, iou_sum, _normal, _border, count = _edge_metrics(
                segment,
                keys[start],
                keys[end],
                raw_by_frame,
                border_by_frame,
                normal_recall_floor=normal_recall_floor,
            )
            if not feasible:
                continue
            feasible_count += 1
            incoming[end].append(
                _Edge(
                    start=start,
                    end=end,
                    quality_loss=max(float(count) - float(iou_sum), 0.0),
                )
            )
    if not incoming[-1]:
        raise RuntimeError(
            f"segment {segment.segment_id} has no feasible edge into its endpoint"
        )
    return (
        _Graph(segment, keys, tuple(tuple(values) for values in incoming)),
        evaluated,
        feasible_count,
    )


def _decode_graph(graph: _Graph, penalty: float) -> tuple[Segment, int, float]:
    if len(graph.keys) < 2:
        return graph.segment, len(graph.keys), 0.0
    costs = [float("inf")] * len(graph.keys)
    losses = [float("inf")] * len(graph.keys)
    counts = [10**9] * len(graph.keys)
    back = [-1] * len(graph.keys)
    costs[0] = 0.0
    losses[0] = 0.0
    counts[0] = 1
    for end in range(1, len(graph.keys)):
        for edge in graph.incoming[end]:
            if costs[edge.start] == float("inf"):
                continue
            candidate_loss = losses[edge.start] + edge.quality_loss
            candidate_count = counts[edge.start] + 1
            candidate_cost = candidate_loss + float(penalty) * (
                candidate_count - 1
            )
            if (
                candidate_cost < costs[end] - 1e-12
                or (
                    abs(candidate_cost - costs[end]) <= 1e-12
                    and (
                        candidate_loss < losses[end] - 1e-12
                        or (
                            abs(candidate_loss - losses[end]) <= 1e-12
                            and candidate_count < counts[end]
                        )
                    )
                )
            ):
                costs[end] = candidate_cost
                losses[end] = candidate_loss
                counts[end] = candidate_count
                back[end] = edge.start
    if back[-1] < 0:
        raise RuntimeError(
            f"segment {graph.segment.segment_id} has no complete penalty path"
        )
    indices: list[int] = []
    cursor = len(graph.keys) - 1
    while cursor >= 0:
        indices.append(cursor)
        if cursor == 0:
            break
        cursor = back[cursor]
    indices.reverse()
    selected = canonicalize_selected_path(
        tuple(graph.keys[index] for index in indices)
    )
    return (
        replace(
            graph.segment,
            interpolation_method="linear_polygon_index_v1",
            keyframes=selected,
        ),
        len(selected),
        float(losses[-1]),
    )


def _decode_all(
    identities: list[tuple[str, int]],
    graphs: list[_Graph],
    penalty: float,
) -> _Decoded:
    output: dict[str, list[Segment]] = {}
    total_keys = 0
    total_loss = 0.0
    for (track_id, _segment_id), graph in zip(identities, graphs, strict=True):
        segment, count, loss = _decode_graph(graph, penalty)
        output.setdefault(track_id, []).append(segment)
        total_keys += count
        total_loss += loss
    return _Decoded(output, total_keys, total_loss, float(penalty))


def optimize_production_v3_penalty(
    safe_segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    expanded_masks: dict[tuple[int, str], RawMask],
    border_constraints: dict[tuple[int, str], BorderFrameConstraint],
    *,
    start_frame: int,
    end_frame: int,
    normal_recall_floor: float,
    target_mean_key_interval: float,
    max_edge_span_frames: int = 30,
    penalty_search_steps: int = 36,
    penalty_max: float = 1_000_000.0,
    candidate_mode: str = "all_observations",
    max_anchor_scale: float = 1.25,
) -> ProductionV3PenaltyResult:
    """Select a soft-target path using quality loss plus lambda per key.

    Every decoded path is lambda-optimal in the feasible dual-Recall graph.
    The requested interval calibrates lambda; it never fixes a key count.
    """

    if target_mean_key_interval <= 0.0:
        raise ValueError("target_mean_key_interval must be positive")
    if candidate_mode not in {"all_observations", "safe_keys"}:
        raise ValueError(f"unsupported candidate_mode: {candidate_mode}")
    started = time.perf_counter()
    identities: list[tuple[str, int]] = []
    graphs: list[_Graph] = []
    evaluated_total = 0
    feasible_total = 0
    total_span = 0
    for track_id, values in sorted(safe_segments.items()):
        for segment in sorted(values, key=lambda item: item.segment_id):
            segment_raw = {
                frame: raw
                for (frame, candidate_track), raw in raw_masks.items()
                if candidate_track == track_id
                and segment.first_frame <= frame <= segment.last_frame
                and start_frame <= frame <= end_frame
            }
            if not segment_raw:
                continue
            segment_border = {
                frame: border_constraints[(frame, track_id)]
                for frame in segment_raw
                if (frame, track_id) in border_constraints
            }
            graph_segment = segment
            if candidate_mode == "all_observations":
                production_by_frame = {
                    key.frame: key for key in segment.keyframes
                }
                first_component = (
                    _primary_component(segment.keyframes[0])
                    if segment.keyframes
                    else None
                )
                point_count = (
                    len(first_component.values)
                    if first_component is not None
                    else 23
                )
                candidates = [
                    _dual_safe_anchor(
                        segment,
                        raw,
                        expanded_masks[(frame, track_id)],
                        border_constraints.get((frame, track_id)),
                        normal_recall_floor=normal_recall_floor,
                        point_count=point_count,
                        max_anchor_scale=max_anchor_scale,
                        production_key=production_by_frame.get(frame),
                    )
                    for frame, raw in sorted(segment_raw.items())
                ]
                if not candidates:
                    continue
                graph_segment = replace(
                    segment,
                    first_frame=candidates[0].frame,
                    last_frame=candidates[-1].frame,
                    interpolation_method="linear_polygon_index_v1",
                    keyframes=canonicalize_selected_path(tuple(candidates)),
                )
            graph, evaluated, feasible = _build_graph(
                graph_segment,
                segment_raw,
                segment_border,
                normal_recall_floor=normal_recall_floor,
                max_edge_span_frames=max_edge_span_frames,
            )
            identities.append((track_id, int(segment.segment_id)))
            graphs.append(graph)
            evaluated_total += evaluated
            feasible_total += feasible
            if graph.keys:
                total_span += graph.keys[-1].frame - graph.keys[0].frame
    if not graphs:
        raise RuntimeError("no Production V3 segments to optimize")

    target_count = int(
        round(total_span / float(target_mean_key_interval)) + len(graphs)
    )
    decoded: dict[tuple[int, float], _Decoded] = {}

    def remember(penalty: float) -> _Decoded:
        value = _decode_all(identities, graphs, penalty)
        decoded[(value.keyframe_count, value.quality_loss)] = value
        return value

    low = 0.0
    low_value = remember(low)
    high = 1.0
    high_value = remember(high)
    while high_value.keyframe_count > target_count and high < penalty_max:
        high = min(high * 2.0, penalty_max)
        high_value = remember(high)
        if high >= penalty_max:
            break
    for _step in range(max(1, int(penalty_search_steps))):
        middle = 0.5 * (low + high)
        value = remember(middle)
        if value.keyframe_count > target_count:
            low = middle
            low_value = value
        else:
            high = middle
            high_value = value

    # Only lambda-optimal paths are eligible. Closeness to the requested
    # interval is a soft calibration criterion, then raw quality breaks ties.
    selected = min(
        decoded.values(),
        key=lambda value: (
            abs(value.keyframe_count - target_count),
            value.quality_loss,
            value.keyframe_count,
        ),
    )
    actual_interval = total_span / max(
        selected.keyframe_count - len(graphs), 1
    )
    return ProductionV3PenaltyResult(
        segments=selected.segments,
        keyframe_count=selected.keyframe_count,
        target_keyframe_count=target_count,
        target_mean_key_interval=float(target_mean_key_interval),
        actual_mean_key_interval=float(actual_interval),
        quality_loss=selected.quality_loss,
        selected_penalty=selected.penalty,
        evaluated_edges=evaluated_total,
        feasible_edges=feasible_total,
        elapsed_seconds=time.perf_counter() - started,
        decoded_candidate_count=len(decoded),
    )


__all__ = ["ProductionV3PenaltyResult", "optimize_production_v3_penalty"]
