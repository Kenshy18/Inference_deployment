"""Shape-agnostic exact mask evaluation for SQLite artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from contracts.mask_sqlite import read_mask_rows


def _decode(value: str) -> list[np.ndarray]:
    return [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in json.loads(value)
        if len(polygon) >= 3
    ]


def _metrics(
    reference: list[np.ndarray], prediction: list[np.ndarray]
) -> tuple[int, int, int, int]:
    polygons = reference + prediction
    if not polygons:
        return 0, 0, 0, 0
    points = np.concatenate(polygons, axis=0)
    minimum = np.floor(points.min(axis=0)).astype(np.int32) - 1
    maximum = np.ceil(points.max(axis=0)).astype(np.int32) + 1
    width = max(1, int(maximum[0] - minimum[0] + 1))
    height = max(1, int(maximum[1] - minimum[1] + 1))

    def rasterize(source: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in source:
            shifted = np.round(polygon - minimum).astype(np.int32)
            cv2.fillPoly(mask, [shifted], 1)
        return mask

    reference_mask = rasterize(reference)
    prediction_mask = rasterize(prediction)
    intersection = int(np.count_nonzero(reference_mask & prediction_mask))
    reference_area = int(np.count_nonzero(reference_mask))
    prediction_area = int(np.count_nonzero(prediction_mask))
    union = reference_area + prediction_area - intersection
    return reference_area, prediction_area, intersection, union


def evaluate_mask_sqlites(
    reference_sqlite: Path,
    prediction_sqlite: Path,
    output_json: Path,
) -> dict[str, float | int]:
    reference = {
        (row.frame, row.track_id): _decode(row.polygons)
        for row in read_mask_rows(reference_sqlite)
    }
    prediction = {
        (row.frame, row.track_id): _decode(row.polygons)
        for row in read_mask_rows(prediction_sqlite)
    }
    reference_area = 0
    prediction_area = 0
    intersection = 0
    union = 0
    for key in sorted(reference.keys() | prediction.keys()):
        values = _metrics(reference.get(key, []), prediction.get(key, []))
        reference_area += values[0]
        prediction_area += values[1]
        intersection += values[2]
        union += values[3]
    summary: dict[str, float | int] = {
        "row_count_reference": len(reference),
        "row_count_prediction": len(prediction),
        "reference_area": reference_area,
        "prediction_area": prediction_area,
        "intersection": intersection,
        "union": union,
        "recall": intersection / reference_area if reference_area else 1.0,
        "precision": intersection / prediction_area if prediction_area else 1.0,
        "iou": intersection / union if union else 1.0,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
