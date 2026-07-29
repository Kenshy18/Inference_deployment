"""Polygon interpolation between independently selected keyframes."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from contracts.mask_sqlite import (
    MaskRow,
    read_mask_rows,
    track_sort_key,
    write_mask_sqlite,
)


class PolygonInterpolator(Protocol):
    name: str

    def interpolate(
        self,
        left: list[list[list[float]]],
        right: list[list[list[float]]],
        alpha: float,
    ) -> list[list[list[float]]]:
        """Interpolate polygons at ``alpha`` in the closed interval [0, 1]."""


def _area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) * 0.5


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-6:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, perimeter, count, endpoint=False)
    output = np.empty((count, 2), dtype=np.float64)
    for index, distance in enumerate(samples):
        segment = min(
            max(int(np.searchsorted(cumulative, distance, side="right") - 1), 0),
            len(points) - 1,
        )
        ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-6)
        output[index] = (1.0 - ratio) * points[segment] + ratio * following[segment]
    return output


def _align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    best = candidate
    best_error = float("inf")
    for variant in (candidate, candidate[::-1]):
        for shift in range(len(variant)):
            shifted = np.roll(variant, shift, axis=0)
            error = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if error < best_error:
                best = shifted
                best_error = error
    return best


@dataclass(frozen=True)
class LinearPolygonInterpolator:
    name: str = "linear_polygon"
    minimum_points: int = 8

    def interpolate(
        self,
        left: list[list[list[float]]],
        right: list[list[list[float]]],
        alpha: float,
    ) -> list[list[list[float]]]:
        if len(left) != len(right):
            return left if alpha < 0.5 else right
        left_arrays = sorted(
            (np.asarray(polygon, dtype=np.float64) for polygon in left),
            key=_area,
            reverse=True,
        )
        right_arrays = sorted(
            (np.asarray(polygon, dtype=np.float64) for polygon in right),
            key=_area,
            reverse=True,
        )
        output: list[list[list[float]]] = []
        for left_points, right_points in zip(left_arrays, right_arrays, strict=True):
            count = max(self.minimum_points, len(left_points), len(right_points))
            left_sample = _resample(left_points, count)
            right_sample = _align(left_sample, _resample(right_points, count))
            points = (1.0 - alpha) * left_sample + alpha * right_sample
            output.append(points.tolist())
        return output


def fill_keyframe_gaps_sqlite(
    keyframes_sqlite: Path,
    reference_sqlite: Path,
    output_sqlite: Path,
    *,
    interpolator: PolygonInterpolator | None = None,
    max_gap: int | None = None,
) -> Path:
    implementation = interpolator or LinearPolygonInterpolator()
    if max_gap is not None and max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    keyframes_by_track: dict[str, list[MaskRow]] = {}
    for row in read_mask_rows(keyframes_sqlite):
        keyframes_by_track.setdefault(row.track_id, []).append(row)
    targets_by_track: dict[str, list[MaskRow]] = {}
    for row in read_mask_rows(reference_sqlite):
        targets_by_track.setdefault(row.track_id, []).append(row)

    output: list[MaskRow] = []
    for track_id in sorted(targets_by_track, key=track_sort_key):
        keys = sorted(keyframes_by_track.get(track_id, []), key=lambda row: row.frame)
        if not keys:
            continue
        key_frames = [row.frame for row in keys]
        track_output: list[MaskRow] = []
        for target in sorted(targets_by_track[track_id], key=lambda row: row.frame):
            position = bisect_left(key_frames, target.frame)
            if position < len(keys) and keys[position].frame == target.frame:
                track_output.append(keys[position])
                continue
            if position == 0 or position == len(keys):
                nearest = keys[0] if position == 0 else keys[-1]
                track_output.append(
                    MaskRow(
                        target.frame,
                        track_id,
                        nearest.polygons,
                        nearest.label,
                        nearest.shape_type,
                    )
                )
                continue
            left = keys[position - 1]
            right = keys[position]
            alpha = (target.frame - left.frame) / (right.frame - left.frame)
            polygons = implementation.interpolate(
                json.loads(left.polygons), json.loads(right.polygons), alpha
            )
            track_output.append(
                MaskRow(
                    frame=target.frame,
                    track_id=track_id,
                    polygons=json.dumps(
                        polygons, ensure_ascii=False, separators=(",", ":")
                    ),
                    label=left.label if alpha < 0.5 else right.label,
                    shape_type="polygon",
                )
            )
        if max_gap:
            for index, left in enumerate(track_output):
                output.append(left)
                if index + 1 >= len(track_output):
                    continue
                right = track_output[index + 1]
                gap = right.frame - left.frame - 1
                if gap <= 0 or gap > max_gap:
                    continue
                left_polygons = json.loads(left.polygons)
                right_polygons = json.loads(right.polygons)
                for frame in range(left.frame + 1, right.frame):
                    alpha = (frame - left.frame) / (right.frame - left.frame)
                    output.append(
                        MaskRow(
                            frame=frame,
                            track_id=track_id,
                            polygons=json.dumps(
                                implementation.interpolate(
                                    left_polygons,
                                    right_polygons,
                                    alpha,
                                ),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            label=left.label if alpha < 0.5 else right.label,
                            shape_type="polygon",
                        )
                    )
        else:
            output.extend(track_output)
    return write_mask_sqlite(output_sqlite, output, reference_sqlite=reference_sqlite)
