from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from production.polygon.runtime.pair_vote import ExactPairVoteEvaluator


def _interpolate(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    return (
        (1.0 - float(alpha)) * np.asarray(left, dtype=np.float32)
        + float(alpha) * np.asarray(right, dtype=np.float32)
    ).astype(np.float32)


def _split(vector: np.ndarray, contour_count: int, anchors: int):
    value = np.asarray(vector, dtype=np.float32).reshape(contour_count, anchors, 2)
    return [np.asarray(value[index], dtype=np.float32) for index in range(contour_count)]


def _rasterize(polygons, shape, shift):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            cv2.fillPoly(
                mask,
                [np.round(points - np.asarray(shift, dtype=np.float32)).astype(np.int32)],
                1,
            )
    return mask


def _reference(gt, pred):
    points = [
        np.asarray(poly, dtype=np.float32)
        for poly in [*gt, *pred]
        if len(poly) >= 3
    ]
    all_points = np.concatenate(points, axis=0)
    minimum = np.floor(all_points.min(axis=0)).astype(np.int32)
    maximum = np.ceil(all_points.max(axis=0)).astype(np.int32)
    shape = (int(maximum[1] - minimum[1] + 1), int(maximum[0] - minimum[0] + 1))
    gt_mask = _rasterize(gt, shape, minimum.astype(np.float32))
    pred_mask = _rasterize(pred, shape, minimum.astype(np.float32))
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = gt_area + pred_area - intersection
    return {
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": intersection / gt_area,
        "precision": intersection / pred_area,
        "iou": intersection / union,
    }


def test_cached_pair_vote_metrics_are_exactly_equal_to_reference() -> None:
    module = SimpleNamespace(
        interpolate_vectors=_interpolate,
        split_vector_to_polygons=_split,
        rasterize_mask_from_polygons=_rasterize,
        compute_exact_metrics_from_polygons=_reference,
    )
    gt = [
        [np.asarray([[0 + frame, 0], [11 + frame, 0], [11 + frame, 9], [0 + frame, 9]], dtype=np.float32)]
        for frame in range(3)
    ]
    run = SimpleNamespace(
        frame_numbers=[100, 101, 102],
        gt_polygons=gt,
        contour_count=1,
        anchors_per_contour=4,
    )
    baseline = np.asarray(
        [
            [[0, 0], [10, 0], [10, 8], [0, 8]],
            [[2, 0], [12, 0], [12, 8], [2, 8]],
        ],
        dtype=np.float32,
    )
    voted = np.asarray(
        [
            [[-1, -1], [11, -1], [11, 9], [-1, 9]],
            [[1, -1], [13, -1], [13, 9], [1, 9]],
        ],
        dtype=np.float32,
    )
    evaluator = ExactPairVoteEvaluator(module, run, [0, 2], baseline, voted)
    vectors = (baseline + np.float32(0.375) * (voted - baseline)).astype(np.float32)
    cached_rows, *_ = evaluator.full_metrics(vectors)

    reference_rows = []
    for frame in range(3):
        vector = vectors[0] if frame == 0 else vectors[1] if frame == 2 else _interpolate(vectors[0], vectors[1], 0.5)
        reference_rows.append(_reference(gt[frame], _split(vector, 1, 4)))

    assert cached_rows == reference_rows

    trials = [
        (baseline + np.float32(alpha) * (voted - baseline)).astype(np.float32)
        for alpha in (0.0, 0.125, 0.375, 0.875, 1.0)
    ]
    batch = evaluator.full_metrics_many(trials)
    scalar = []
    for trial in trials:
        rows, _loss, mean_iou, _mean_recall, _precision, _global = (
            evaluator.full_metrics(trial)
        )
        scalar.append(
            (mean_iou, min(float(row["recall"]) for row in rows))
        )
    assert batch == scalar
    evaluator.close()


def test_cached_local_pair_vote_matches_explicit_reference_sum() -> None:
    module = SimpleNamespace(
        interpolate_vectors=_interpolate,
        split_vector_to_polygons=_split,
        rasterize_mask_from_polygons=_rasterize,
        compute_exact_metrics_from_polygons=_reference,
    )
    square = np.asarray([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    gt = [[square + np.asarray([frame, 0], dtype=np.float32)] for frame in range(5)]
    run = SimpleNamespace(
        frame_numbers=list(range(5)),
        gt_polygons=gt,
        contour_count=1,
        anchors_per_contour=4,
    )
    baseline = np.asarray([square, square + [2, 0], square + [4, 0]], dtype=np.float32)
    voted = np.asarray([square + [-1, -1], square + [2, -1], square + [5, -1]], dtype=np.float32)
    evaluator = ExactPairVoteEvaluator(module, run, [0, 2, 4], baseline, voted)
    current = baseline.copy()
    trial = (baseline[1] + np.float32(0.5) * (voted[1] - baseline[1])).astype(np.float32)
    cached_iou, cached_recall = evaluator.local_metrics(current, 1, trial)

    reference_iou = 0.0
    reference_recall = 1.0
    for frame in range(5):
        if frame < 2:
            vector = _interpolate(current[0], trial, frame / 2.0)
        elif frame == 2:
            vector = trial
        else:
            vector = _interpolate(trial, current[2], (frame - 2) / 2.0)
        metrics = _reference(gt[frame], _split(vector, 1, 4))
        reference_iou += metrics["iou"]
        reference_recall = min(reference_recall, metrics["recall"])

    assert cached_iou == reference_iou
    assert cached_recall == reference_recall

    trials = [
        (baseline[1] + np.float32(alpha) * (voted[1] - baseline[1])).astype(np.float32)
        for alpha in (0.0, 0.125, 0.5, 0.875, 1.0)
    ]
    batch = evaluator.local_metrics_many(current, 1, trials)
    scalar = [evaluator.local_metrics(current, 1, trial) for trial in trials]
    assert batch == scalar
    evaluator.close()
