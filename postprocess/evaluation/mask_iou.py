"""Shape-agnostic exact mask evaluation for SQLite artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from contracts.mask_sqlite import MaskRow, iter_mask_rows


_OrderKey: TypeAlias = tuple[str, int]


def _decode(value: str) -> list[np.ndarray]:
    # Geometry is authored and exact-audited in float64.  Casting to float32
    # before the joint-ROI subtraction can move a half-pixel tie to the other
    # side of ``np.round`` and report a false Recall violation even though the
    # Production oracle accepted the same curve/polygon.
    return [
        np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
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
    # Keep an *even* integer padding.  ``np.round`` uses ties-to-even, so an
    # odd origin shift changes how half-pixel vertices are rasterized.  The
    # Production CPU oracle and the native exact evaluator both use a two-pixel
    # padding/parity-equivalent origin.  Using one pixel here made evaluation
    # report false Recall violations for small half-pixel masks even though the
    # exact DP and the exported geometry were identical.
    minimum = np.floor(points.min(axis=0)).astype(np.int32) - 2
    maximum = np.ceil(points.max(axis=0)).astype(np.int32) + 2
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
    *,
    selected_track_ids: Iterable[str] | None = None,
) -> dict[str, float | int]:
    """Evaluate exact raster overlap with memory independent of video length."""

    selected = (
        None
        if selected_track_ids is None
        else frozenset(str(value) for value in selected_track_ids)
    )

    def ordered_rows(path: Path) -> Iterator[tuple[_OrderKey, MaskRow]]:
        for row in iter_mask_rows(path):
            if selected is not None and row.track_id not in selected:
                continue
            # ``iter_mask_rows`` is ordered by SQLite TEXT track_id and frame.
            # The merge key must use the same order. A numeric sort key would
            # mis-pair sparse sets such as reference tracks {"10", "2"} and
            # prediction track {"2"}.
            yield ((str(row.track_id), int(row.frame)), row)

    def advance(
        iterator: Iterator[tuple[_OrderKey, MaskRow]],
    ) -> tuple[_OrderKey, MaskRow] | None:
        return next(iterator, None)

    reference = ordered_rows(reference_sqlite)
    prediction = ordered_rows(prediction_sqlite)
    reference_row = advance(reference)
    prediction_row = advance(prediction)
    reference_rows = 0
    prediction_rows = 0
    reference_area = 0
    prediction_area = 0
    intersection = 0
    union = 0
    while reference_row is not None or prediction_row is not None:
        reference_polygons: list[np.ndarray] = []
        prediction_polygons: list[np.ndarray] = []
        if prediction_row is None or (
            reference_row is not None and reference_row[0] < prediction_row[0]
        ):
            reference_polygons = _decode(reference_row[1].polygons)
            reference_rows += 1
            reference_row = advance(reference)
        elif reference_row is None or prediction_row[0] < reference_row[0]:
            prediction_polygons = _decode(prediction_row[1].polygons)
            prediction_rows += 1
            prediction_row = advance(prediction)
        else:
            reference_polygons = _decode(reference_row[1].polygons)
            prediction_polygons = _decode(prediction_row[1].polygons)
            reference_rows += 1
            prediction_rows += 1
            reference_row = advance(reference)
            prediction_row = advance(prediction)
        values = _metrics(reference_polygons, prediction_polygons)
        reference_area += values[0]
        prediction_area += values[1]
        intersection += values[2]
        union += values[3]
    summary: dict[str, float | int] = {
        "row_count_reference": reference_rows,
        "row_count_prediction": prediction_rows,
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
