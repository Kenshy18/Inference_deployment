#!/usr/bin/env python3
"""Independent geometry-only audit for a superior Pareto SQLite.

No video is opened.  All comparisons use the raw polygons, editable
keyframes, segment metadata, and schema definitions stored in SQLite.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from .fixed_budget import (
    _primary_component,
    evaluate_segments,
    geometry_at,
    load_raw_masks,
    load_segments,
    summarize,
)
from .sqlite_export import schema_fingerprint
from .superior import direct_geometry_at, evaluate_direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--pareto-report", type=Path, required=True)
    parser.add_argument(
        "--reference-pareto-report", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--production-sqlite", type=Path, action="append", default=[])
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--incident-frame", type=int, action="append", default=[])
    return parser.parse_args()


def _mean_interval(segments, key_count: int) -> float:
    segment_count = 0
    span = 0
    for values in segments.values():
        for segment in values:
            if not segment.keyframes:
                continue
            segment_count += 1
            span += segment.keyframes[-1].frame - segment.keyframes[0].frame
    return span / max(key_count - segment_count, 1)


def _per_segment(evaluations) -> dict[str, dict[str, float | int]]:
    grouped = defaultdict(list)
    for item in evaluations:
        grouped[int(item.segment_id)].append(item)
    output = {}
    for segment_id, values in sorted(grouped.items()):
        output[str(segment_id)] = {
            "frame_count": len(values),
            "mean_iou": sum(item.iou for item in values) / len(values),
            "minimum_iou": min(item.iou for item in values),
            "minimum_recall": min(item.recall for item in values),
            "recall_below_floor": 0,
        }
    return output


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _best_alignment_change(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[bool, int]:
    """Return the reversal/roll needed to align an adjacent stored polygon."""

    best_error = math.inf
    best_reverse = False
    best_shift = 0
    for reverse, variant in ((False, candidate), (True, candidate[::-1])):
        for shift in range(len(variant)):
            shifted = np.roll(variant, shift, axis=0)
            error = float(np.mean(np.sum(np.square(shifted - reference), axis=1)))
            if error < best_error:
                best_error = error
                best_reverse = reverse
                best_shift = shift
    return best_reverse, best_shift


def _similarity_residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Measure non-rigid one-frame deformation after translation/rotation/scale."""

    left = reference - np.mean(reference, axis=0)
    right = candidate - np.mean(candidate, axis=0)
    u_value, _singular, vt_value = np.linalg.svd(left.T @ right)
    rotation = vt_value.T @ u_value.T
    if np.linalg.det(rotation) < 0.0:
        vt_value[-1] *= -1.0
        rotation = vt_value.T @ u_value.T
    rotated = left @ rotation
    scale = float(np.trace(rotated.T @ right) / max(float(np.sum(left * left)), 1e-9))
    residual = float(
        np.sqrt(np.mean(np.sum(np.square(scale * rotated - right), axis=1)))
    )
    return residual / math.sqrt(max(abs(_signed_area(reference)), 1e-9))


def _vertex_safety_audit(segments) -> dict[str, object]:
    """Audit the exact point-index contract without opening video pixels.

    Integer frames are the shapes an editor renders.  Dense fractional samples
    additionally prove that the linear vertex trajectories do not briefly
    cross between two integer frames.
    """

    keyframe_count = 0
    point_counts: set[int] = set()
    invalid_keyframes = 0
    keyframe_self_intersections = 0
    adjacent_winding_flips = 0
    adjacent_alignment_reversals = 0
    adjacent_alignment_shifts = 0
    integer_frames = 0
    invalid_integer_frames = 0
    integer_self_intersections = 0
    integer_winding_flips = 0
    fractional_samples = 0
    invalid_fractional_samples = 0
    fractional_self_intersections = 0
    fractional_winding_flips = 0
    non_rigid_motion: list[float] = []
    largest_motion: list[tuple[float, str, int]] = []

    for track_id, values in segments.items():
        for segment in values:
            keys = list(segment.keyframes)
            arrays: list[np.ndarray] = []
            for keyframe in keys:
                component = _primary_component(keyframe)
                if component is None:
                    arrays.append(np.empty((0, 2), dtype=np.float64))
                    continue
                points = np.asarray(component.values, dtype=np.float64)
                arrays.append(points)
                keyframe_count += 1
                point_counts.add(len(points))
                polygon = Polygon(points)
                line = LineString(np.vstack((points, points[0])))
                invalid_keyframes += int(not polygon.is_valid)
                keyframe_self_intersections += int(not line.is_simple)

            for index, (left, right) in enumerate(zip(arrays, arrays[1:])):
                if not len(left) or left.shape != right.shape:
                    continue
                left_sign = _signed_area(left)
                right_sign = _signed_area(right)
                adjacent_winding_flips += int(left_sign * right_sign < 0.0)
                reverse, shift = _best_alignment_change(left, right)
                adjacent_alignment_reversals += int(reverse)
                adjacent_alignment_shifts += int(shift != 0)

                left_frame = int(keys[index].frame)
                right_frame = int(keys[index + 1].frame)
                span = max(right_frame - left_frame, 1)
                previous: np.ndarray | None = None
                previous_sign: float | None = None
                for frame in range(left_frame, right_frame + 1):
                    # Shared interval endpoints are intentionally counted more
                    # than once; a bad boundary must not escape the gate.
                    alpha = (frame - left_frame) / span
                    points = (1.0 - alpha) * left + alpha * right
                    polygon = Polygon(points)
                    line = LineString(np.vstack((points, points[0])))
                    sign = _signed_area(points)
                    integer_frames += 1
                    invalid_integer_frames += int(not polygon.is_valid)
                    integer_self_intersections += int(not line.is_simple)
                    if previous_sign is not None:
                        integer_winding_flips += int(previous_sign * sign < 0.0)
                    if previous is not None:
                        residual = _similarity_residual(previous, points)
                        non_rigid_motion.append(residual)
                        largest_motion.append((residual, str(track_id), frame))
                    previous = points
                    previous_sign = sign

                for alpha in np.linspace(0.0, 1.0, 101):
                    points = (1.0 - alpha) * left + alpha * right
                    polygon = Polygon(points)
                    line = LineString(np.vstack((points, points[0])))
                    fractional_samples += 1
                    invalid_fractional_samples += int(not polygon.is_valid)
                    fractional_self_intersections += int(not line.is_simple)
                    fractional_winding_flips += int(
                        left_sign * _signed_area(points) < 0.0
                    )

    motion = np.asarray(non_rigid_motion, dtype=np.float64)
    largest_motion.sort(reverse=True)
    return {
        "keyframe_count": keyframe_count,
        "point_counts": sorted(point_counts),
        "invalid_keyframe_count": invalid_keyframes,
        "keyframe_self_intersection_count": keyframe_self_intersections,
        "adjacent_winding_flip_count": adjacent_winding_flips,
        "adjacent_best_alignment_reversal_count": adjacent_alignment_reversals,
        "adjacent_best_alignment_nonzero_shift_count": adjacent_alignment_shifts,
        "integer_frame_samples": integer_frames,
        "invalid_integer_frame_count": invalid_integer_frames,
        "integer_self_intersection_count": integer_self_intersections,
        "integer_winding_flip_count": integer_winding_flips,
        "fractional_samples": fractional_samples,
        "invalid_fractional_sample_count": invalid_fractional_samples,
        "fractional_self_intersection_count": fractional_self_intersections,
        "fractional_winding_flip_count": fractional_winding_flips,
        "non_rigid_motion": {
            "mean": float(np.mean(motion)) if len(motion) else 0.0,
            "q99": float(np.quantile(motion, 0.99)) if len(motion) else 0.0,
            "q999": float(np.quantile(motion, 0.999)) if len(motion) else 0.0,
            "maximum": float(np.max(motion)) if len(motion) else 0.0,
            "largest": [
                {"value": value, "track_id": track_id, "frame": frame}
                for value, track_id, frame in largest_motion[:20]
            ],
        },
    }


def main() -> int:
    args = parse_args()
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    selected = load_segments(
        args.output_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    selected_rows = evaluate_direct(raw, selected)
    selected_summary = summarize(
        selected_rows,
        selected,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    per_segment = _per_segment(selected_rows)
    for item in selected_rows:
        if item.recall + 1e-12 < args.recall_floor:
            per_segment[str(item.segment_id)]["recall_below_floor"] += 1

    # Verify editor point_index interpolation against the registered overlay
    # method on every segment frame, including frames without a raw detection.
    maximum_difference = 0.0
    nonzero_differences = 0
    compared_frames = 0
    for values in selected.values():
        for segment in values:
            for frame in range(segment.first_frame, segment.last_frame + 1):
                registered = geometry_at(segment, frame)
                direct = direct_geometry_at(segment, frame)
                difference = float(registered.symmetric_difference(direct).area)
                maximum_difference = max(maximum_difference, difference)
                nonzero_differences += int(difference > 1e-8)
                compared_frames += 1
    vertex_safety = _vertex_safety_audit(selected)

    report = json.loads(args.pareto_report.read_text(encoding="utf-8"))
    frontier = report["frontier"]
    reference_frontiers = []
    for path in args.reference_pareto_report:
        reference_report = json.loads(path.read_text(encoding="utf-8"))
        reference_frontier = reference_report["frontier"]
        failures = []
        margins = []
        for reference_point in reference_frontier:
            eligible = [
                point
                for point in frontier
                if point["keyframe_count"] <= reference_point["keyframe_count"]
            ]
            candidate = (
                max(eligible, key=lambda point: point["mean_iou"]) if eligible else None
            )
            dominated = bool(
                candidate is not None
                and candidate["mean_iou"] + 1e-12 >= reference_point["mean_iou"]
            )
            if candidate is not None:
                margins.append(candidate["mean_iou"] - reference_point["mean_iou"])
            if not dominated:
                failures.append(
                    {
                        "reference_keyframe_count": reference_point["keyframe_count"],
                        "reference_mean_iou": reference_point["mean_iou"],
                        "new_same_or_lower_key_budget": candidate,
                    }
                )
        reference_frontiers.append(
            {
                "path": str(path.resolve()),
                "point_count": len(reference_frontier),
                "all_points_dominated": not failures,
                "non_dominated_count": len(failures),
                "minimum_mean_iou_margin": min(margins, default=None),
                "failures": failures[:100],
                "failures_truncated": len(failures) > 100,
            }
        )
    production = []
    for path in args.production_sqlite:
        segments = load_segments(
            path,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        rows = evaluate_segments(raw, segments)
        summary = summarize(
            rows,
            segments,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        production_vertex_safety = _vertex_safety_audit(segments)
        selected_motion = vertex_safety["non_rigid_motion"]
        production_motion = production_vertex_safety["non_rigid_motion"]
        stability_regressions = {
            metric: float(selected_motion[metric] - production_motion[metric])
            for metric in ("mean", "q99", "q999", "maximum")
            if selected_motion[metric] > production_motion[metric] + 1e-12
        }
        keys = int(summary["keyframe_count"])
        eligible = [point for point in frontier if point["keyframe_count"] <= keys]
        same_budget = (
            max(eligible, key=lambda point: point["mean_iou"]) if eligible else None
        )
        reaches_quality = [
            point
            for point in frontier
            if point["mean_iou"] + 1e-12 >= summary["iou_mean"]
        ]
        production_per_segment = _per_segment(rows)
        segment_comparison = []
        for segment_id in sorted(
            set(production_per_segment).intersection(per_segment), key=int
        ):
            baseline_segment = production_per_segment[segment_id]
            selected_segment = per_segment[segment_id]
            segment_comparison.append(
                {
                    "segment_id": int(segment_id),
                    "production_mean_iou": baseline_segment["mean_iou"],
                    "new_mean_iou": selected_segment["mean_iou"],
                    "mean_iou_delta": (
                        selected_segment["mean_iou"] - baseline_segment["mean_iou"]
                    ),
                    "production_minimum_recall": baseline_segment["minimum_recall"],
                    "new_minimum_recall": selected_segment["minimum_recall"],
                }
            )
        production.append(
            {
                "path": str(path.resolve()),
                "keyframe_count": keys,
                "mean_key_interval": _mean_interval(segments, keys),
                "mean_iou": summary["iou_mean"],
                "minimum_recall": summary["recall_min"],
                "recall_below_floor": summary["recall_below_097"],
                "vertex_safety": production_vertex_safety,
                "new_vertex_stability_non_regression": not stability_regressions,
                "new_vertex_stability_regressions": stability_regressions,
                "new_same_or_lower_key_budget": same_budget,
                "key_iou_dominated": bool(
                    same_budget is not None
                    and same_budget["mean_iou"] + 1e-12 >= summary["iou_mean"]
                ),
                "minimum_new_keys_to_reach_iou": (
                    min(point["keyframe_count"] for point in reaches_quality)
                    if reaches_quality
                    else None
                ),
                "segment_comparison": {
                    "count": len(segment_comparison),
                    "mean_iou_regression_count": sum(
                        item["mean_iou_delta"] < -1e-12 for item in segment_comparison
                    ),
                    "worst_mean_iou_delta": min(
                        (item["mean_iou_delta"] for item in segment_comparison),
                        default=0.0,
                    ),
                    "items": segment_comparison,
                },
            }
        )

    by_identity = {(item.frame, item.track_id): item for item in selected_rows}
    incidents = []
    for frame in sorted(set(args.incident_frame)):
        matches = [
            item for (value, _track), item in by_identity.items() if value == frame
        ]
        incidents.append(
            {
                "frame": frame,
                "masks": [
                    {
                        "track_id": item.track_id,
                        "segment_id": item.segment_id,
                        "is_keyframe": item.is_keyframe,
                        "recall": item.recall,
                        "iou": item.iou,
                        "area_ratio": item.area_ratio,
                    }
                    for item in matches
                ],
            }
        )

    import sqlite3

    with sqlite3.connect(
        f"file:{args.source_sqlite.resolve()}?mode=ro", uri=True
    ) as source:
        source_schema = schema_fingerprint(source)
    with sqlite3.connect(
        f"file:{args.output_sqlite.resolve()}?mode=ro", uri=True
    ) as output:
        output_schema = schema_fingerprint(output)
        integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = list(output.execute("PRAGMA foreign_key_check"))

    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "selected": selected_summary,
        "per_segment": per_segment,
        "all_frame_vertex_contract": {
            "compared_frames": compared_frames,
            "maximum_symmetric_difference_area": maximum_difference,
            "nonzero_difference_count": nonzero_differences,
        },
        "vertex_safety": vertex_safety,
        "schema": {
            "source_fingerprint": source_schema,
            "output_fingerprint": output_schema,
            "unchanged": source_schema == output_schema,
            "integrity_check": integrity,
            "foreign_key_error_count": len(foreign_keys),
        },
        "production_comparison": production,
        "reference_frontier_comparison": reference_frontiers,
        "incidents": incidents,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    blocking_counts = {
        "editor_overlay_geometry_mismatch": nonzero_differences,
        "invalid_keyframe": vertex_safety["invalid_keyframe_count"],
        "keyframe_self_intersection": vertex_safety["keyframe_self_intersection_count"],
        "adjacent_winding_flip": vertex_safety["adjacent_winding_flip_count"],
        "adjacent_alignment_reversal": vertex_safety[
            "adjacent_best_alignment_reversal_count"
        ],
        "adjacent_alignment_shift": vertex_safety[
            "adjacent_best_alignment_nonzero_shift_count"
        ],
        "invalid_integer_frame": vertex_safety["invalid_integer_frame_count"],
        "integer_self_intersection": vertex_safety["integer_self_intersection_count"],
        "integer_winding_flip": vertex_safety["integer_winding_flip_count"],
        "invalid_fractional_sample": vertex_safety["invalid_fractional_sample_count"],
        "fractional_self_intersection": vertex_safety[
            "fractional_self_intersection_count"
        ],
        "fractional_winding_flip": vertex_safety["fractional_winding_flip_count"],
        "production_vertex_stability_regression": sum(
            not item["new_vertex_stability_non_regression"] for item in production
        ),
    }
    failures = {name: int(value) for name, value in blocking_counts.items() if value}
    if failures:
        print(
            json.dumps(
                {"vertex_safety_gate": "failed", "blocking_counts": failures},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
