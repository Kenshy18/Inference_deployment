"""Analyze temporal mask geometry without reading source video frames.

This diagnostic intentionally consumes only the ``masks`` table from SQLite.
It separates consecutive-frame contour motion into translation, similarity,
full-affine, and post-affine local-deformation components, then checks whether
high-motion/reversal events are represented by the selected keyframes.

It is experimental tooling: it does not mutate pipeline artifacts or the
public result SQLite schema.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MaskSample:
    frame: int
    track_id: str
    contours: tuple[np.ndarray, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sqlite", required=True, type=Path)
    keyframes = parser.add_mutually_exclusive_group()
    keyframes.add_argument("--keyframes-sqlite", type=Path)
    keyframes.add_argument("--keyframes-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--event-radius", type=int, default=1)
    return parser


def _normalize(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(result) > 1 and np.allclose(result[0], result[-1]):
        result = result[:-1]
    return result


def _signed_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    points = _normalize(points)
    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.linspace(0.0, perimeter, count, endpoint=False)
    indices = np.searchsorted(cumulative, positions, side="right") - 1
    indices = np.clip(indices, 0, len(points) - 1)
    alpha = (positions - cumulative[indices]) / np.maximum(lengths[indices], 1e-9)
    return (
        (1.0 - alpha[:, None]) * points[indices]
        + alpha[:, None] * following[indices]
    )


def _align_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    best = candidate
    best_error = math.inf
    for variant in (candidate, candidate[::-1].copy()):
        for shift in range(len(variant)):
            rolled = np.roll(variant, -shift, axis=0)
            error = float(np.mean(np.sum((rolled - reference) ** 2, axis=1)))
            if error < best_error:
                best_error = error
                best = rolled
    return best


def _parse_contours(value: str) -> tuple[np.ndarray, ...]:
    parsed = json.loads(value)
    contours = [
        _normalize(np.asarray(contour, dtype=np.float64)) for contour in parsed
    ]
    contours = [contour for contour in contours if len(contour) >= 3]
    contours.sort(key=lambda contour: abs(_signed_area(contour)), reverse=True)
    return tuple(contours)


def _read_masks(path: Path) -> dict[str, list[MaskSample]]:
    grouped: dict[str, list[MaskSample]] = defaultdict(list)
    with sqlite3.connect(str(path)) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(masks)").fetchall()
        }
        required = {"frame", "track_id", "polygons"}
        if not required.issubset(columns):
            raise ValueError(f"{path}: masks table lacks {sorted(required - columns)}")
        for frame, track_id, polygons in connection.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY track_id, frame"
        ):
            contours = _parse_contours(str(polygons))
            if contours:
                grouped[str(track_id)].append(
                    MaskSample(int(frame), str(track_id), contours)
                )
    return grouped


def _read_keyframes(path: Path | None) -> dict[str, set[int]]:
    if path is None:
        return {}
    grouped: dict[str, set[int]] = defaultdict(set)
    with sqlite3.connect(str(path)) as connection:
        for frame, track_id in connection.execute(
            "SELECT frame, track_id FROM masks ORDER BY track_id, frame"
        ):
            grouped[str(track_id)].add(int(frame))
    return grouped


def _read_keyframes_json(path: Path | None) -> dict[str, set[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    grouped: dict[str, set[int]] = defaultdict(set)
    for row in payload:
        grouped[str(row["track_id"])].add(int(row["frame"]))
    return grouped


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _similarity_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    covariance = x.T @ y
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(singular.sum() / max(float(np.sum(x * x)), 1e-12))
    return (x @ (scale * rotation)) + target_mean


def _transition(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    translation = current.mean(axis=0) - previous.mean(axis=0)
    translated = previous + translation
    similarity = _similarity_fit(previous, current)
    design = np.column_stack((previous, np.ones(len(previous))))
    coefficients, *_ = np.linalg.lstsq(design, current, rcond=None)
    affine = design @ coefficients
    linear = coefficients[:2, :]
    singular = np.linalg.svd(linear, compute_uv=False)
    anisotropy = float(abs(math.log(max(singular[0], 1e-9) / max(singular[1], 1e-9))))
    return {
        "total_motion_px": _rms(current - previous),
        "translation_px": float(np.linalg.norm(translation)),
        "after_translation_px": _rms(current - translated),
        "after_similarity_px": _rms(current - similarity),
        "affine_component_px": _rms(affine - translated),
        "local_deformation_px": _rms(current - affine),
        "affine_anisotropy_log": anisotropy,
        "affine_determinant": float(np.linalg.det(linear)),
        "centroid_dx": float(translation[0]),
        "centroid_dy": float(translation[1]),
    }


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _near_keyframe(frame: int, keys: set[int], radius: int) -> bool:
    return any((frame + offset) in keys for offset in range(-radius, radius + 1))


def _nearest_keyframe_offset(frame: int, keys: set[int]) -> int | None:
    if not keys:
        return None
    # Prefer the earlier keyframe when the absolute distance is tied. This
    # makes a positive value unambiguously mean that representation lags the
    # event and a negative value mean that it anticipates it.
    nearest = min(keys, key=lambda key: (abs(key - frame), key > frame, key))
    return int(nearest - frame)


def analyze(
    source: dict[str, list[MaskSample]],
    keyframes: dict[str, set[int]],
    *,
    points: int,
    event_quantile: float,
    event_radius: int,
) -> dict[str, object]:
    transitions: list[dict[str, float | int | str | bool]] = []
    reversals: list[dict[str, float | int | str | bool]] = []
    incompatible_contour_pairs = 0
    for track_id, samples in source.items():
        prior_points: np.ndarray | None = None
        prior_sample: MaskSample | None = None
        prior_velocity: np.ndarray | None = None
        keys = keyframes.get(track_id, set())
        for sample in samples:
            if (
                prior_sample is None
                or sample.frame != prior_sample.frame + 1
                or len(sample.contours) != len(prior_sample.contours)
            ):
                if prior_sample is not None and len(sample.contours) != len(prior_sample.contours):
                    incompatible_contour_pairs += 1
                prior_points = None
                prior_velocity = None
            current = _resample(sample.contours[0], points)
            if prior_points is not None:
                current = _align_phase(prior_points, current)
                metrics = _transition(prior_points, current)
                velocity = np.asarray(
                    [metrics["centroid_dx"], metrics["centroid_dy"]], dtype=np.float64
                )
                entry: dict[str, float | int | str | bool] = {
                    "track_id": track_id,
                    "frame": sample.frame,
                    "keyframe_nearby": _near_keyframe(sample.frame, keys, event_radius),
                    "nearest_keyframe_offset": _nearest_keyframe_offset(
                        sample.frame, keys
                    ),
                    **metrics,
                }
                transitions.append(entry)
                if prior_velocity is not None:
                    denominator = float(np.linalg.norm(prior_velocity) * np.linalg.norm(velocity))
                    cosine = (
                        float(np.dot(prior_velocity, velocity) / denominator)
                        if denominator > 1e-9
                        else 1.0
                    )
                    acceleration = float(np.linalg.norm(velocity - prior_velocity))
                    if cosine < -0.25:
                        reversals.append(
                            {
                                "track_id": track_id,
                                "frame": sample.frame,
                                "velocity_cosine": cosine,
                                "centroid_acceleration_px": acceleration,
                                "keyframe_nearby": _near_keyframe(
                                    sample.frame, keys, event_radius
                                ),
                                "nearest_keyframe_offset": _nearest_keyframe_offset(
                                    sample.frame, keys
                                ),
                            }
                        )
                prior_velocity = velocity
            prior_points = current
            prior_sample = sample

    numeric_names = (
        "total_motion_px",
        "translation_px",
        "after_translation_px",
        "after_similarity_px",
        "affine_component_px",
        "local_deformation_px",
        "affine_anisotropy_log",
        "affine_determinant",
    )
    distributions = {
        name: _quantiles([float(row[name]) for row in transitions])
        for name in numeric_names
    }
    event_capture: dict[str, object] = {}
    covered_transitions = sum(bool(row["keyframe_nearby"]) for row in transitions)
    coverage_rate = (
        float(covered_transitions / len(transitions)) if transitions else None
    )
    for name in (
        "total_motion_px",
        "translation_px",
        "affine_component_px",
        "local_deformation_px",
    ):
        values = np.asarray([float(row[name]) for row in transitions], dtype=np.float64)
        if len(values) == 0:
            event_capture[name] = {"events": 0, "captured": 0, "capture_rate": None}
            continue
        threshold = float(np.quantile(values, event_quantile))
        events = [row for row in transitions if float(row[name]) >= threshold]
        captured = sum(bool(row["keyframe_nearby"]) for row in events)
        offsets = [
            int(row["nearest_keyframe_offset"])
            for row in events
            if row["nearest_keyframe_offset"] is not None
        ]
        event_capture[name] = {
            "threshold": threshold,
            "events": len(events),
            "captured": int(captured),
            "capture_rate": float(captured / len(events)) if events else None,
            "capture_lift_over_all_transitions": (
                float(captured / len(events)) - coverage_rate
                if events and coverage_rate is not None
                else None
            ),
            "nearest_keyframe_abs_distance": _quantiles(
                [float(abs(offset)) for offset in offsets]
            ),
            "nearest_keyframe_signed_offset_p50": (
                float(np.median(offsets)) if offsets else None
            ),
            "nearest_keyframe_after_event_rate": (
                float(sum(offset > 0 for offset in offsets) / len(offsets))
                if offsets
                else None
            ),
        }
    reversal_captured = sum(bool(row["keyframe_nearby"]) for row in reversals)
    reversal_offsets = [
        int(row["nearest_keyframe_offset"])
        for row in reversals
        if row["nearest_keyframe_offset"] is not None
    ]
    return {
        "source_tracks": len(source),
        "source_rows": int(sum(len(rows) for rows in source.values())),
        "keyframe_rows": int(sum(len(rows) for rows in keyframes.values())),
        "transitions": len(transitions),
        "incompatible_contour_pairs": incompatible_contour_pairs,
        "analysis": {
            "resampled_points": points,
            "event_quantile": event_quantile,
            "keyframe_event_radius": event_radius,
            "primary_contour_only": True,
        },
        "keyframe_proximity_coverage": {
            "covered_transitions": int(covered_transitions),
            "all_transitions": len(transitions),
            "coverage_rate": coverage_rate,
        },
        "distributions": distributions,
        "high_motion_event_capture": event_capture,
        "direction_reversals": {
            "events": len(reversals),
            "captured": int(reversal_captured),
            "capture_rate": (
                float(reversal_captured / len(reversals)) if reversals else None
            ),
            "acceleration_px": _quantiles(
                [float(row["centroid_acceleration_px"]) for row in reversals]
            ),
            "nearest_keyframe_abs_distance": _quantiles(
                [float(abs(offset)) for offset in reversal_offsets]
            ),
            "nearest_keyframe_signed_offset_p50": (
                float(np.median(reversal_offsets)) if reversal_offsets else None
            ),
            "nearest_keyframe_after_event_rate": (
                float(
                    sum(offset > 0 for offset in reversal_offsets)
                    / len(reversal_offsets)
                )
                if reversal_offsets
                else None
            ),
        },
        "worst_local_deformation": sorted(
            transitions,
            key=lambda row: float(row["local_deformation_px"]),
            reverse=True,
        )[:25],
        "worst_affine_motion": sorted(
            transitions,
            key=lambda row: float(row["affine_component_px"]),
            reverse=True,
        )[:25],
        "uncaptured_reversals": [
            row for row in reversals if not bool(row["keyframe_nearby"])
        ][:100],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.points < 8:
        raise ValueError("--points must be >= 8")
    if not 0.0 < args.event_quantile < 1.0:
        raise ValueError("--event-quantile must be between 0 and 1")
    if args.event_radius < 0:
        raise ValueError("--event-radius must be >= 0")
    result = analyze(
        _read_masks(args.source_sqlite),
        (
            _read_keyframes(args.keyframes_sqlite)
            if args.keyframes_sqlite is not None
            else _read_keyframes_json(args.keyframes_json)
        ),
        points=args.points,
        event_quantile=args.event_quantile,
        event_radius=args.event_radius,
    )
    result["inputs"] = {
        "source_sqlite": str(args.source_sqlite.resolve()),
        "keyframes_sqlite": (
            None
            if args.keyframes_sqlite is None
            else str(args.keyframes_sqlite.resolve())
        ),
        "keyframes_json": (
            None
            if args.keyframes_json is None
            else str(args.keyframes_json.resolve())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
