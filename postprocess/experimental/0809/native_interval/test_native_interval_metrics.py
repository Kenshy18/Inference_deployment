#!/usr/bin/env python3
"""Parity and micro-benchmark checks for the first native prototype.

This test uses generated coordinates only. It never opens a video or SQLite.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import native_interval_metrics


METRIC_NAMES = (
    "gt_area",
    "pred_area",
    "intersection",
    "union",
    "recall",
    "precision",
    "iou",
)


def python_exact_metrics(gt_polygons, pred_polygons):
    all_polygons = [
        np.asarray(polygon, dtype=np.float32)
        for polygon in list(gt_polygons) + list(pred_polygons)
        if len(polygon) >= 3
    ]
    if not all_polygons:
        return {
            "gt_area": 0.0,
            "pred_area": 0.0,
            "intersection": 0.0,
            "union": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "iou": 1.0,
        }
    all_points = np.concatenate(all_polygons, axis=0)
    min_xy = np.floor(all_points.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_points.max(axis=0)).astype(np.int32)
    shift_xy = min_xy.astype(np.float32)
    shape = (
        int(max_xy[1] - min_xy[1] + 1),
        int(max_xy[0] - min_xy[0] + 1),
    )

    def rasterize(polygons):
        mask = np.zeros(shape, dtype=np.uint8)
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                rounded = np.round(points - shift_xy[None, :]).astype(np.int32)
                cv2.fillPoly(mask, [rounded], 1)
        return mask

    gt_mask = rasterize(gt_polygons)
    pred_mask = rasterize(pred_polygons)
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = gt_area + pred_area - intersection
    return {
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": intersection / gt_area if gt_area else 1.0,
        "precision": intersection / pred_area if pred_area else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def fixtures():
    rectangle = np.asarray([[0, 0], [9, 0], [9, 7], [0, 7]], np.float32)
    shifted = rectangle + np.asarray([2.2, -1.7], np.float32)
    triangle = np.asarray([[-3.5, 1.5], [4.5, 1.5], [0.5, 9.5]], np.float32)
    island_a = np.asarray([[0, 0], [5, 0], [5, 5], [0, 5]], np.float32)
    island_b = np.asarray([[3, 3], [8, 3], [8, 8], [3, 8]], np.float32)
    diamond = np.asarray([[4.5, -2.5], [11.5, 4.5], [4.5, 11.5], [-2.5, 4.5]], np.float32)
    return [
        ([], []),
        ([rectangle], [rectangle.copy()]),
        ([rectangle], [shifted]),
        ([triangle], [diamond]),
        ([island_a, island_b], [diamond]),
        ([island_a, island_b], [island_a, island_b]),
        ([rectangle[:2]], [diamond]),
    ]


def assert_parity() -> int:
    checked = 0
    for case_index, (gt_polygons, pred_polygons) in enumerate(fixtures()):
        expected = python_exact_metrics(gt_polygons, pred_polygons)
        actual = native_interval_metrics.exact_metrics(gt_polygons, pred_polygons)
        for name in METRIC_NAMES:
            if not np.isclose(expected[name], actual[name], rtol=0.0, atol=0.0):
                raise AssertionError(
                    f"case={case_index} metric={name} "
                    f"python={expected[name]} native={actual[name]}"
                )
        checked += 1

    rng = np.random.default_rng(20260810)
    for case_index in range(200):
        center = rng.uniform(-20.0, 80.0, size=2).astype(np.float32)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=24))
        radii = rng.uniform(5.0, 35.0, size=24)
        raw = np.column_stack(
            [np.cos(angles) * radii, np.sin(angles) * radii]
        ).astype(np.float32) + center
        pred = raw * np.float32(rng.uniform(0.9, 1.1))
        pred += rng.normal(0.0, 1.5, size=pred.shape).astype(np.float32)
        expected = python_exact_metrics([raw], [pred])
        actual = native_interval_metrics.exact_metrics([raw], [pred])
        for name in METRIC_NAMES:
            if not np.isclose(expected[name], actual[name], rtol=0.0, atol=0.0):
                raise AssertionError(
                    f"random_case={case_index} metric={name} "
                    f"python={expected[name]} native={actual[name]}"
                )
        checked += 1
    return checked


def benchmark(iterations: int):
    gt_polygons, pred_polygons = fixtures()[4]
    for _ in range(100):
        python_exact_metrics(gt_polygons, pred_polygons)
        native_interval_metrics.exact_metrics(gt_polygons, pred_polygons)

    started = time.perf_counter()
    for _ in range(iterations):
        python_exact_metrics(gt_polygons, pred_polygons)
    python_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(iterations):
        native_interval_metrics.exact_metrics(gt_polygons, pred_polygons)
    native_seconds = time.perf_counter() - started
    return {
        "iterations": iterations,
        "python_seconds": python_seconds,
        "native_seconds": native_seconds,
        "speedup": python_seconds / native_seconds,
    }


def assert_batch_parity() -> int:
    frame_count = 10
    state_count = 2
    recall_floor = 0.97
    exact_frames = []
    candidate_vectors = []
    gt_masks = []
    shifts = []
    scales = np.ones((frame_count,), dtype=np.float32)
    for frame in range(frame_count):
        center = np.asarray([30.0 + frame * 1.7, 28.0 + frame * 0.8], np.float32)
        raw = np.asarray(
            [[-9.2, -7.1], [10.3, -6.4], [9.1, 8.2], [-8.8, 7.7]],
            np.float32,
        ) + center
        expanded = center + np.float32(1.05) * (raw - center)
        gt = raw + np.asarray(
            [np.sin(frame * 0.4) * 0.8, np.cos(frame * 0.3) * 0.6],
            np.float32,
        )
        exact_frames.append([gt])
        candidate_vectors.append(np.stack([raw, expanded], axis=0))
        shift = np.floor(gt.min(axis=0)).astype(np.float32) - 12.0
        shifts.append(shift)
        shape = (48, 48)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(gt - shift).astype(np.int32)], 1)
        gt_masks.append(mask)
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)
    shifts = np.asarray(shifts, dtype=np.float32)
    edges = np.asarray(
        [
            (start, start_state, end, end_state)
            for end in range(1, frame_count)
            for start in range(max(0, end - 4), end)
            for start_state in range(state_count)
            for end_state in range(state_count)
        ],
        dtype=np.int32,
    )
    evaluator = native_interval_metrics.CachedIntervalEvaluator(
        gt_masks, shifts, scales, exact_frames
    )
    output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
        )
    )
    short_output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
            None,
            True,
        )
    )
    cost_only_output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
            None,
            True,
            True,
        )
    )
    if not np.array_equal(cost_only_output[:, :8], short_output[:, :8]):
        raise AssertionError("cost-only evaluation changed native objective fields")
    if np.any(cost_only_output[:, 8] != 0.0):
        raise AssertionError("cost-only evaluation unexpectedly ran exact Recall")
    exact_first_output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
            None,
            True,
            False,
            True,
        )
    )
    recall_hints = np.asarray(edges[:, 2], dtype=np.int32)
    hinted_output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
            None,
            True,
            False,
            True,
            recall_hints,
        )
    )
    exact_first_feasible = exact_first_output[:, 8] <= 1e-10
    hinted_feasible = hinted_output[:, 8] <= 1e-10
    if not np.array_equal(exact_first_feasible, hinted_feasible):
        raise AssertionError("Recall hints changed exact edge feasibility")
    if not np.array_equal(
        exact_first_output[exact_first_feasible, :8],
        hinted_output[hinted_feasible, :8],
    ):
        raise AssertionError("Recall hints changed exact feasible edge costs")
    recall_hint_matrix = np.stack(
        [
            np.asarray(edges[:, 2], dtype=np.int32),
            np.asarray(edges[:, 0] + 1, dtype=np.int32),
            np.asarray(edges[:, 2], dtype=np.int32),
            np.full((len(edges),), -1, dtype=np.int32),
        ],
        axis=1,
    )
    multi_hinted_output = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            4,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            2,
            None,
            True,
            False,
            True,
            recall_hint_matrix,
        )
    )
    multi_hinted_feasible = multi_hinted_output[:, 8] <= 1e-10
    if not np.array_equal(exact_first_feasible, multi_hinted_feasible):
        raise AssertionError("multiple Recall hints changed exact edge feasibility")
    if not np.array_equal(
        exact_first_output[exact_first_feasible, :8],
        multi_hinted_output[multi_hinted_feasible, :8],
    ):
        raise AssertionError("multiple Recall hints changed exact feasible edge costs")
    full_infeasible = (output[:, 7] > 1e-10) | (output[:, 8] > 1e-10)
    short_infeasible = (short_output[:, 7] > 1e-10) | (
        short_output[:, 8] > 1e-10
    )
    if not np.array_equal(full_infeasible, short_infeasible):
        raise AssertionError("short-circuit changed exact edge feasibility")
    if not np.array_equal(output[~full_infeasible], short_output[~full_infeasible]):
        raise AssertionError("short-circuit changed a feasible edge metric")
    for edge_index, (start, start_state, end, end_state) in enumerate(edges):
        start_vector = candidate_vectors[start, start_state]
        end_vector = candidate_vectors[end, end_state]
        cached_loss = 0.0
        cached_deficit = 0.0
        for frame in range(start + 1, end + 1):
            alpha32 = np.float32((frame - start) / max(end - start, 1))
            beta32 = np.float32(1.0) - alpha32
            predicted = beta32 * start_vector + alpha32 * end_vector
            pred_mask = np.zeros_like(gt_masks[frame])
            points = np.round(predicted - shifts[frame]).astype(np.int32)
            cv2.fillPoly(pred_mask, [points], 1)
            pred_area = int(cv2.countNonZero(pred_mask))
            intersection = int(cv2.countNonZero(gt_masks[frame] & pred_mask))
            union = int(gt_masks[frame].sum()) + pred_area - intersection
            recall = intersection / int(gt_masks[frame].sum())
            iou = intersection / union
            cached_loss += 1.0 - iou
            cached_deficit += max(recall_floor - recall, 0.0)
        if output[edge_index, 7] != cached_deficit:
            raise AssertionError(
                f"batch cached deficit mismatch at edge {edge_index}: "
                f"{output[edge_index, 7]} != {cached_deficit}"
            )
        if output[edge_index, 4] != cached_loss / (end - start):
            raise AssertionError(f"batch cached loss mismatch at edge {edge_index}")

        exact_deficit = 0.0
        if cached_deficit <= 1e-10:
            for frame in range(start + 1, end + 1):
                alpha = float((frame - start) / max(end - start, 1))
                predicted = (
                    (1.0 - alpha) * start_vector + alpha * end_vector
                ).astype(np.float32)
                metrics = python_exact_metrics(exact_frames[frame], [predicted])
                exact_deficit += max(recall_floor - metrics["recall"], 0.0)
                if exact_deficit > 1e-10:
                    break
        if output[edge_index, 8] != exact_deficit:
            raise AssertionError(
                f"batch exact deficit mismatch at edge {edge_index}: "
                f"{output[edge_index, 8]} != {exact_deficit}"
            )
    return len(edges)


def assert_random_exact_recall_batch_parity() -> tuple[int, int]:
    """Exercise the parity-aware exact-reference cache on irregular shapes."""
    rng = np.random.default_rng(2026081402)
    frame_count = 37
    state_count = 3
    point_count = 14
    recall_floor = 0.97
    exact_frames = []
    candidate_vectors = np.empty(
        (frame_count, state_count, point_count, 2), dtype=np.float32
    )
    gt_masks = []
    shifts = []
    scales = np.ones((frame_count,), dtype=np.float32)
    for frame in range(frame_count):
        center = np.asarray(
            [110.5 + frame * 1.25, 90.5 + np.sin(frame * 0.3) * 8.0],
            dtype=np.float32,
        )
        angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        radii = 34.0 + 7.0 * np.sin(3.0 * angles + frame * 0.17)
        gt = np.column_stack(
            [np.cos(angles) * radii, np.sin(angles) * radii]
        ).astype(np.float32)
        gt += center
        if frame % 3 == 0:
            gt = (np.round(gt * 2.0) / 2.0).astype(np.float32)
        exact_frames.append([gt])
        sampled = gt[np.linspace(0, len(gt) - 1, point_count, dtype=np.int32)]
        for state, scale in enumerate((0.985, 1.0, 1.025)):
            value = center + np.float32(scale) * (sampled - center)
            value += np.asarray(
                [0.35 * state, -0.2 * (frame % 2)], dtype=np.float32
            )
            candidate_vectors[frame, state] = value
        cached_gt = center + np.float32(0.80) * (sampled - center)
        shift = np.floor(cached_gt.min(axis=0)).astype(np.float32) - 16.0
        shifts.append(shift)
        shape = (128, 128)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(cached_gt - shift).astype(np.int32)], 1)
        gt_masks.append(mask)
    edges = np.asarray(
        [
            (start, start_state, end, end_state)
            for end in range(1, frame_count)
            for start in range(max(0, end - 9), end)
            for start_state in range(state_count)
            for end_state in range(state_count)
        ],
        dtype=np.int32,
    )
    evaluator = native_interval_metrics.CachedIntervalEvaluator(
        gt_masks,
        np.asarray(shifts, dtype=np.float32),
        scales,
        exact_frames,
    )
    endpoint_checked = 0
    for frame in range(frame_count):
        for state in range(state_count):
            expected = python_exact_metrics(
                exact_frames[frame], [candidate_vectors[frame, state]]
            )
            actual = evaluator.exact_frame_metrics(
                frame,
                candidate_vectors[frame, state],
                1,
                point_count,
            )
            for metric_index, name in enumerate(METRIC_NAMES):
                if actual[metric_index] != expected[name]:
                    raise AssertionError(
                        f"cached endpoint mismatch frame={frame} state={state} "
                        f"metric={name} expected={expected[name]} "
                        f"actual={actual[metric_index]}"
                    )
            endpoint_checked += 1
    endpoint_frames = np.repeat(
        np.arange(frame_count, dtype=np.int32), state_count
    )
    endpoint_vectors = np.ascontiguousarray(
        candidate_vectors.reshape(frame_count * state_count, point_count, 2)
    )
    endpoint_batch = np.asarray(
        evaluator.exact_frame_metrics_batch(
            endpoint_frames,
            endpoint_vectors,
            1,
            point_count,
            8,
        )
    )
    for case_index, (frame, state) in enumerate(
        (
            (frame, state)
            for frame in range(frame_count)
            for state in range(state_count)
        )
    ):
        scalar = np.asarray(
            evaluator.exact_frame_metrics(
                frame,
                candidate_vectors[frame, state],
                1,
                point_count,
            )
        )
        if not np.array_equal(endpoint_batch[case_index], scalar):
            raise AssertionError(
                f"batched endpoint mismatch frame={frame} state={state}: "
                f"batch={endpoint_batch[case_index].tolist()} "
                f"scalar={scalar.tolist()}"
            )
    values = np.asarray(
        evaluator.evaluate_edge_batch(
            candidate_vectors,
            edges,
            1,
            point_count,
            1.0,
            recall_floor,
            100.0,
            0.09,
            2.0,
            0.4,
            2.0,
            0.5,
            0.5,
            0.25,
            0.25,
            8,
            None,
            True,
        )
    )
    checked = 0
    for edge_index, (start, start_state, end, end_state) in enumerate(edges):
        if values[edge_index, 7] > 1e-10:
            if values[edge_index, 8] != 0.0:
                raise AssertionError("exact path ran after cached Recall failure")
            continue
        expected_deficit = 0.0
        left = candidate_vectors[start, start_state]
        right = candidate_vectors[end, end_state]
        for frame in range(start + 1, end + 1):
            alpha64 = (frame - start) / max(end - start, 1)
            alpha = np.float32(alpha64)
            beta = np.float32(1.0 - alpha64)
            prediction = beta * left + alpha * right
            metrics = python_exact_metrics(exact_frames[frame], [prediction])
            expected_deficit = max(
                recall_floor - float(metrics["recall"]), 0.0
            )
            if expected_deficit > 1e-10:
                break
        if values[edge_index, 8] != expected_deficit:
            raise AssertionError(
                f"random exact deficit mismatch edge={edge_index} "
                f"expected={expected_deficit} actual={values[edge_index, 8]}"
            )
        checked += 1
    if checked < 100:
        raise AssertionError(f"too few exact-cache parity edges: {checked}")
    return checked, endpoint_checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    checked = assert_parity()
    batch_edges = assert_batch_parity()
    random_exact_edges, cached_endpoint_cases = (
        assert_random_exact_recall_batch_parity()
    )
    result = {
        "implementation": native_interval_metrics.implementation,
        "parity_cases": checked,
        "batch_parity_edges": batch_edges,
        "random_exact_recall_parity_edges": random_exact_edges,
        "cached_endpoint_parity_cases": cached_endpoint_cases,
        "benchmark": benchmark(args.iterations),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
