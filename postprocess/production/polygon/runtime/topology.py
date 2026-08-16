"""Topology hard gates for the polygon14 Production candidate.

The guard is deliberately lazy: only a decoded DP path and pair-vote trials
that would otherwise be selected are checked.  Valid paths are returned byte
for byte unchanged.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import numpy as np


_EPS = 1e-8


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(
        (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    )


def _on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
    return bool(
        min(float(a[0]), float(b[0])) - _EPS
        <= float(point[0])
        <= max(float(a[0]), float(b[0])) + _EPS
        and min(float(a[1]), float(b[1])) - _EPS
        <= float(point[1])
        <= max(float(a[1]), float(b[1])) + _EPS
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    if first * second < -_EPS and third * fourth < -_EPS:
        return True
    if abs(first) <= _EPS and _on_segment(a, b, c):
        return True
    if abs(second) <= _EPS and _on_segment(a, b, d):
        return True
    if abs(third) <= _EPS and _on_segment(c, d, a):
        return True
    if abs(fourth) <= _EPS and _on_segment(c, d, b):
        return True
    return False


def polygon_is_simple(points: np.ndarray) -> bool:
    """Return whether one closed ring is finite, non-degenerate and simple."""
    value = np.asarray(points, dtype=np.float64)
    count = int(len(value))
    if count < 3 or value.shape != (count, 2) or not np.all(np.isfinite(value)):
        return False
    shifted = np.roll(value, -1, axis=0)
    if np.any(np.linalg.norm(shifted - value, axis=1) <= _EPS):
        return False
    signed_twice_area = float(
        np.sum(value[:, 0] * shifted[:, 1] - shifted[:, 0] * value[:, 1])
    )
    if abs(signed_twice_area) <= _EPS:
        return False
    for first in range(count):
        a = value[first]
        b = value[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first:
                continue
            if second == (first + 1) % count:
                continue
            if (second + 1) % count == first:
                continue
            c = value[second]
            d = value[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def polygons_are_simple(polygons: Iterable[np.ndarray]) -> bool:
    return all(polygon_is_simple(polygon) for polygon in polygons)


def vector_is_simple(module, run, vector: np.ndarray) -> bool:
    return polygons_are_simple(
        module.split_vector_to_polygons(
            vector,
            int(run.contour_count),
            int(run.anchors_per_contour),
        )
    )


def first_invalid_edge_frame(
    module,
    run,
    start_frame: int,
    start_vector: np.ndarray,
    end_frame: int,
    end_vector: np.ndarray,
) -> int | None:
    """Return the first invalid integer-frame interpolation, if any."""
    start = int(start_frame)
    end = int(end_frame)
    span = max(end - start, 1)
    for frame in range(start, end + 1):
        if frame == start:
            vector = start_vector
        elif frame == end:
            vector = end_vector
        else:
            vector = module.interpolate_vectors(
                start_vector,
                end_vector,
                float(frame - start) / float(span),
            )
        if not vector_is_simple(module, run, vector):
            return int(frame)
    return None


def path_is_simple(
    module,
    run,
    chosen_frames: list[int],
    vectors: np.ndarray,
) -> bool:
    if len(chosen_frames) != len(vectors) or not len(chosen_frames):
        return False
    if len(chosen_frames) == 1:
        return vector_is_simple(module, run, vectors[0])
    return all(
        first_invalid_edge_frame(
            module,
            run,
            int(chosen_frames[index]),
            vectors[index],
            int(chosen_frames[index + 1]),
            vectors[index + 1],
        )
        is None
        for index in range(len(chosen_frames) - 1)
    )


def local_key_update_is_simple(
    module,
    run,
    chosen_frames: list[int],
    current: np.ndarray,
    key_position: int,
    trial_vector: np.ndarray,
) -> bool:
    if not vector_is_simple(module, run, trial_vector):
        return False
    position = int(key_position)
    if (
        position > 0
        and first_invalid_edge_frame(
            module,
            run,
            int(chosen_frames[position - 1]),
            current[position - 1],
            int(chosen_frames[position]),
            trial_vector,
        )
        is not None
    ):
        return False
    if (
        position + 1 < len(chosen_frames)
        and first_invalid_edge_frame(
            module,
            run,
            int(chosen_frames[position]),
            trial_vector,
            int(chosen_frames[position + 1]),
            current[position + 1],
        )
        is not None
    ):
        return False
    return True


def _finite_interval_cost(
    module,
    run,
    left_frame: int,
    left_candidate,
    right_frame: int,
    right_candidate,
    runtime_args,
    eval_contexts,
):
    info = module.interval_cost_from_vectors(
        run,
        int(left_frame),
        left_candidate.vector,
        int(right_frame),
        right_candidate.vector,
        runtime_args,
        include_start=False,
        eval_contexts=eval_contexts,
        start_candidate=left_candidate,
        end_candidate=right_candidate,
    )
    return info if math.isfinite(float(info.cost)) else None


def repair_decoded_path(
    module,
    run,
    chosen_frames: list[int],
    chosen_states: list[int],
    candidates_by_frame,
    runtime_args,
    eval_contexts,
    stats: dict[str, float | int],
) -> tuple[list[int], list[int]]:
    """Lazily split only selected DP edges whose interpolation is invalid."""
    started = time.perf_counter()
    frames = [int(value) for value in chosen_frames]
    states = [int(value) for value in chosen_states]
    cursor = 0
    while cursor + 1 < len(frames):
        left_frame = int(frames[cursor])
        right_frame = int(frames[cursor + 1])
        left = candidates_by_frame[left_frame][states[cursor]]
        right = candidates_by_frame[right_frame][states[cursor + 1]]
        stats["dp_selected_edges_checked"] = (
            int(stats.get("dp_selected_edges_checked", 0)) + 1
        )
        invalid_frame = first_invalid_edge_frame(
            module,
            run,
            left_frame,
            left.vector,
            right_frame,
            right.vector,
        )
        if invalid_frame is None:
            cursor += 1
            continue
        stats["dp_invalid_edges"] = int(stats.get("dp_invalid_edges", 0)) + 1
        if invalid_frame <= left_frame or invalid_frame >= right_frame:
            raise RuntimeError(
                "polygon14 topology guard found an invalid selected endpoint: "
                f"stream={run.stream_id!r} frame={invalid_frame}"
            )
        best: tuple[float, int] | None = None
        for state, middle in enumerate(candidates_by_frame[invalid_frame]):
            if not vector_is_simple(module, run, middle.vector):
                continue
            if (
                first_invalid_edge_frame(
                    module,
                    run,
                    left_frame,
                    left.vector,
                    invalid_frame,
                    middle.vector,
                )
                is not None
            ):
                continue
            if (
                first_invalid_edge_frame(
                    module,
                    run,
                    invalid_frame,
                    middle.vector,
                    right_frame,
                    right.vector,
                )
                is not None
            ):
                continue
            left_info = _finite_interval_cost(
                module,
                run,
                left_frame,
                left,
                invalid_frame,
                middle,
                runtime_args,
                eval_contexts,
            )
            right_info = _finite_interval_cost(
                module,
                run,
                invalid_frame,
                middle,
                right_frame,
                right,
                runtime_args,
                eval_contexts,
            )
            if left_info is None or right_info is None:
                continue
            score = float(left_info.cost) + float(right_info.cost)
            candidate = (score, int(state))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError(
                "polygon14 topology guard could not split an invalid edge "
                "without violating exact Recall: "
                f"stream={run.stream_id!r} left={left_frame} "
                f"invalid={invalid_frame} right={right_frame}"
            )
        frames.insert(cursor + 1, int(invalid_frame))
        states.insert(cursor + 1, int(best[1]))
        stats["dp_inserted_keys"] = int(stats.get("dp_inserted_keys", 0)) + 1
        # Recheck both new edges.  The cursor intentionally does not advance.
    stats["dp_guard_seconds"] = float(stats.get("dp_guard_seconds", 0.0)) + (
        time.perf_counter() - started
    )
    return frames, states
