"""Measure temporal stability and source-following fidelity for geometry runs.

The benchmark deliberately reports smoothness and source-following together.
Smoothness alone can reward a frozen or lagging mask, so every acceleration
metric has a corresponding per-frame or velocity fidelity metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Geometry:
    area: float
    centroid: np.ndarray
    shape: np.ndarray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--source-sqlite", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval-maximum", type=int, default=6)
    parser.add_argument("--shape-points", type=int, default=64)
    parser.add_argument("--shape-harmonics", type=int, default=16)
    return parser


def _normalize_contour(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points


def _signed_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    if len(points) < 3:
        return 0.0, np.zeros(2, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    signed = 0.5 * float(np.sum(cross))
    if abs(signed) <= 1e-9:
        return 0.0, points.mean(axis=0)
    centroid = np.sum((points + following) * cross[:, None], axis=0) / (6.0 * signed)
    return abs(signed), centroid


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 3:
        return np.zeros((count, 2), dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.linspace(0.0, perimeter, count, endpoint=False)
    indices = np.searchsorted(cumulative, positions, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    alpha = (positions - cumulative[indices]) / np.maximum(lengths[indices], 1e-9)
    return (1.0 - alpha[:, None]) * points[indices] + alpha[:, None] * following[indices]


def _geometry(serialized: str, shape_points: int, harmonics: int) -> Geometry:
    contours = [
        _normalize_contour(item)
        for item in json.loads(serialized)
        if len(item) >= 3
    ]
    if not contours:
        raise ValueError("mask has no valid contour")
    parts = [_signed_area_centroid(contour) for contour in contours]
    total_area = float(sum(area for area, _ in parts))
    if total_area <= 1e-9:
        raise ValueError("mask has zero polygon area")
    centroid = sum(area * center for area, center in parts) / total_area
    primary = contours[int(np.argmax([area for area, _ in parts]))]
    samples = _resample(primary, shape_points)
    centered = samples - samples.mean(axis=0)
    complex_points = centered[:, 0] + 1j * centered[:, 1]
    spectrum = np.abs(np.fft.fft(complex_points))[1 : harmonics + 1]
    norm = float(np.linalg.norm(spectrum))
    shape = spectrum / max(norm, 1e-12)
    return Geometry(total_area, centroid, shape)


def _read_source(
    path: Path, shape_points: int, harmonics: int
) -> tuple[dict[tuple[str, int], tuple[str, Geometry]], dict[str, list[int]]]:
    source: dict[tuple[str, int], tuple[str, Geometry]] = {}
    frames: dict[str, list[int]] = defaultdict(list)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "SELECT CAST(track_id AS TEXT), frame, label, polygons "
            "FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
        )
        for track_id, frame, label, polygons in cursor:
            key = (str(track_id), int(frame))
            if key in source:
                raise RuntimeError(f"duplicate source observation: {key}")
            source[key] = (
                str(label),
                _geometry(str(polygons), shape_points, harmonics),
            )
            frames[str(track_id)].append(int(frame))
    return source, frames


def _quantiles(values: Iterable[float], prefix: str) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(len(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
        f"{prefix}_q99": float(np.quantile(array, 0.99)),
        f"{prefix}_maximum": float(np.max(array)),
    }


def _summarize(bucket: dict[str, list[float]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in (
        "absolute_log_area_error",
        "centroid_tracking_error_radii",
        "shape_tracking_error",
        "log_area_velocity_error",
        "centroid_velocity_error_radii",
        "shape_velocity_error",
        "log_area_acceleration",
        "centroid_acceleration_radii",
        "shape_acceleration",
    ):
        result.update(_quantiles(bucket.get(name, ()), name))
    return result


def _run_metrics(
    path: Path,
    source: dict[tuple[str, int], tuple[str, Geometry]],
    *,
    geometry_name: str,
    target_interval: int,
    shape_points: int,
    harmonics: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    overall: dict[str, list[float]] = defaultdict(list)
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    previous: dict[str, tuple[int, Geometry, Geometry]] = {}
    previous2: dict[str, tuple[int, Geometry, Geometry]] = {}
    matched = 0

    def record(label: str, name: str, value: float) -> None:
        if math.isfinite(value):
            overall[name].append(float(value))
            by_label[label][name].append(float(value))

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "SELECT CAST(track_id AS TEXT), frame, polygons "
            "FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
        )
        for track_id_value, frame_value, polygons in cursor:
            track_id, frame = str(track_id_value), int(frame_value)
            source_row = source.get((track_id, frame))
            if source_row is None:
                continue
            label, truth = source_row
            output = _geometry(str(polygons), shape_points, harmonics)
            matched += 1
            radius = max(math.sqrt(truth.area / math.pi), 1.0)
            record(label, "absolute_log_area_error", abs(math.log(output.area / truth.area)))
            record(
                label,
                "centroid_tracking_error_radii",
                float(np.linalg.norm(output.centroid - truth.centroid)) / radius,
            )
            record(
                label,
                "shape_tracking_error",
                float(np.linalg.norm(output.shape - truth.shape)),
            )

            prior = previous.get(track_id)
            if prior is not None and frame == prior[0] + 1:
                _, prior_output, prior_truth = prior
                record(
                    label,
                    "log_area_velocity_error",
                    abs(
                        math.log(output.area / prior_output.area)
                        - math.log(truth.area / prior_truth.area)
                    ),
                )
                record(
                    label,
                    "centroid_velocity_error_radii",
                    float(
                        np.linalg.norm(
                            (output.centroid - prior_output.centroid)
                            - (truth.centroid - prior_truth.centroid)
                        )
                    )
                    / radius,
                )
                record(
                    label,
                    "shape_velocity_error",
                    float(
                        np.linalg.norm(
                            (output.shape - prior_output.shape)
                            - (truth.shape - prior_truth.shape)
                        )
                    ),
                )
                prior2 = previous2.get(track_id)
                if prior2 is not None and frame == prior2[0] + 2:
                    _, prior2_output, _prior2_truth = prior2
                    record(
                        label,
                        "log_area_acceleration",
                        abs(
                            math.log(output.area)
                            - 2.0 * math.log(prior_output.area)
                            + math.log(prior2_output.area)
                        ),
                    )
                    record(
                        label,
                        "centroid_acceleration_radii",
                        float(
                            np.linalg.norm(
                                output.centroid
                                - 2.0 * prior_output.centroid
                                + prior2_output.centroid
                            )
                        )
                        / radius,
                    )
                    record(
                        label,
                        "shape_acceleration",
                        float(
                            np.linalg.norm(
                                output.shape
                                - 2.0 * prior_output.shape
                                + prior2_output.shape
                            )
                        ),
                    )
            else:
                previous2.pop(track_id, None)
            if prior is not None:
                previous2[track_id] = prior
            previous[track_id] = (frame, output, truth)

    if matched != len(source):
        raise RuntimeError(f"{path}: matched {matched}, expected {len(source)}")
    base: dict[str, object] = {
        "geometry": geometry_name,
        "target_interval": target_interval,
        "source_observations": matched,
        "primary_contour_shape_descriptor": True,
        "shape_points": shape_points,
        "shape_harmonics": harmonics,
        **_summarize(overall),
    }
    class_rows = [
        {
            "geometry": geometry_name,
            "target_interval": target_interval,
            "label": label,
            **_summarize(bucket),
        }
        for label, bucket in sorted(by_label.items())
    ]
    return base, class_rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parser().parse_args()
    root = args.benchmark_root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, _frames = _read_source(
        args.source_sqlite.resolve(), args.shape_points, args.shape_harmonics
    )
    rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    raw_summary, raw_classes = _run_metrics(
        args.source_sqlite.resolve(),
        source,
        geometry_name="source_raw",
        target_interval=0,
        shape_points=args.shape_points,
        harmonics=args.shape_harmonics,
    )
    rows.append(raw_summary)
    class_rows.extend(raw_classes)
    for interval in range(1, args.interval_maximum + 1):
        for geometry_name, folder in (
            ("polygon", "polygon"),
            ("catmull_rom", "catmull_rom"),
        ):
            path = (
                root
                / folder
                / f"interval_{interval}"
                / "00_classwise_postprocess"
                / "predictions.sqlite"
            )
            print(f"stability {geometry_name} target={interval}: {path}", flush=True)
            summary, classes = _run_metrics(
                path,
                source,
                geometry_name=geometry_name,
                target_interval=interval,
                shape_points=args.shape_points,
                harmonics=args.shape_harmonics,
            )
            rows.append(summary)
            class_rows.extend(classes)
    _write(output / "stability_summary.csv", rows)
    _write(output / "stability_by_class.csv", class_rows)
    print(output / "stability_summary.csv")


if __name__ == "__main__":
    main()
