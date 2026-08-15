#!/usr/bin/env python3
"""Repair polygon vertex correspondence in an exported Pareto SQLite.

The Pareto optimizer evaluates each edge after cyclic/reversal alignment, but
older experimental exports stored every selected anchor in its original point
order.  Editors that interpolate ``point_index`` directly can consequently
twist or collapse a polygon between otherwise valid keyframes.

This tool keeps the V3 schema, keyframe schedule, and exact keyframe boundaries
unchanged.  It adds collinear edge points where necessary, then aligns winding
and cyclic origin along each track segment.  An optional local Recall guard is
available for experiments, but is not part of the geometry-preserving repair.
Video pixels are never opened.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

from .fixed_budget import (
    Component,
    Keyframe,
    RawMask,
    Segment,
    _primary_component,
    geometry_from_arrays,
    load_raw_masks,
    load_segments,
)
from .sqlite_export import export_selected_sqlite, schema_fingerprint
from .temporal_consensus import build_segment_bounded_temporal_consensus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--trusted-recall-floor", type=float, default=0.97)
    parser.add_argument("--guard-floor", type=float, default=0.9702)
    parser.add_argument("--max-scale", type=float, default=1.05)
    parser.add_argument("--consensus-radius", type=int, default=2)
    parser.add_argument("--support-fraction", type=float, default=0.50)
    parser.add_argument("--target-mean-key-interval", type=float, default=10.0)
    parser.add_argument("--apply-local-recall-guard", action="store_true")
    return parser.parse_args()


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _best_alignment(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, bool, int, float]:
    best = candidate
    best_reverse = False
    best_shift = 0
    best_error = math.inf
    for reverse, variant in ((False, candidate), (True, candidate[::-1])):
        for shift in range(len(variant)):
            shifted = np.roll(variant, shift, axis=0)
            error = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if error < best_error:
                best = shifted
                best_reverse = reverse
                best_shift = shift
                best_error = error
    return best.copy(), best_reverse, best_shift, best_error


def _densify_preserving_geometry(points: np.ndarray, count: int) -> np.ndarray:
    """Add collinear edge points without changing the polygon boundary."""

    output = [np.asarray(point, dtype=np.float64) for point in points]
    if len(output) > count:
        raise ValueError(
            f"cannot preserve a {len(output)}-point polygon at point-count {count}"
        )
    while len(output) < count:
        lengths = [
            float(np.linalg.norm(output[(index + 1) % len(output)] - point))
            for index, point in enumerate(output)
        ]
        index = int(np.argmax(np.asarray(lengths, dtype=np.float64)))
        following = (index + 1) % len(output)
        midpoint = 0.5 * (output[index] + output[following])
        output.insert(index + 1, midpoint)
    return np.asarray(output, dtype=np.float64)


def _replace_primary_polygon(keyframe: Keyframe, points: np.ndarray) -> Keyframe:
    primary = _primary_component(keyframe)
    if primary is None:
        raise ValueError(f"frame {keyframe.frame} has no polygon component")
    polygon_count = sum(
        component.kind == "polygon" for _slot, component in keyframe.components
    )
    if polygon_count != 1:
        raise ValueError(
            f"frame {keyframe.frame} has {polygon_count} polygon components; "
            "the repair requires exactly one"
        )
    components = tuple(
        (
            slot,
            Component("polygon", points.tolist())
            if component is primary
            else component,
        )
        for slot, component in keyframe.components
    )
    return Keyframe(int(keyframe.frame), components)


def canonicalize_segments(
    segments: dict[str, list[Segment]], point_count: int
) -> tuple[dict[str, list[Segment]], dict[str, int]]:
    output: dict[str, list[Segment]] = {}
    reversed_count = 0
    shifted_count = 0
    keyframe_count = 0
    for track_id, values in segments.items():
        repaired_values: list[Segment] = []
        for segment in values:
            repaired_keys: list[Keyframe] = []
            reference: np.ndarray | None = None
            for keyframe in segment.keyframes:
                component = _primary_component(keyframe)
                if component is None:
                    raise ValueError(
                        f"segment {segment.segment_id} frame {keyframe.frame} "
                        "has no polygon"
                    )
                points = _densify_preserving_geometry(
                    np.asarray(component.values, dtype=np.float64), point_count
                )
                if reference is not None:
                    points, reversed_ring, shift, _error = _best_alignment(
                        reference, points
                    )
                    reversed_count += int(reversed_ring)
                    shifted_count += int(shift != 0)
                repaired_keys.append(_replace_primary_polygon(keyframe, points))
                reference = points
                keyframe_count += 1
            repaired_values.append(replace(segment, keyframes=tuple(repaired_keys)))
        output[track_id] = repaired_values
    return output, {
        "keyframe_count": keyframe_count,
        "canonicalization_reversals": reversed_count,
        "canonicalization_nonzero_shifts": shifted_count,
    }


def _scaled(points: np.ndarray, scale: float) -> np.ndarray:
    center = np.mean(points, axis=0)
    return center + float(scale) * (points - center)


def _direct_geometry(
    left: Keyframe,
    right: Keyframe,
    frame: int,
    left_scale: float = 1.0,
    right_scale: float = 1.0,
):
    left_component = _primary_component(left)
    right_component = _primary_component(right)
    if left_component is None or right_component is None:
        raise ValueError("polygon component missing during interpolation")
    left_points = _scaled(
        np.asarray(left_component.values, dtype=np.float64), left_scale
    )
    right_points = _scaled(
        np.asarray(right_component.values, dtype=np.float64), right_scale
    )
    if len(left_points) != len(right_points):
        raise ValueError("direct interpolation requires equal point counts")
    span = int(right.frame) - int(left.frame)
    alpha = 0.0 if span <= 0 else (int(frame) - int(left.frame)) / span
    return geometry_from_arrays(((1.0 - alpha) * left_points + alpha * right_points,))


def _recall(reference, predicted) -> float:
    area = float(reference.area)
    if area <= 0.0:
        return 1.0
    return float(reference.intersection(predicted).area) / area


def _interval_masks(
    trusted: dict[tuple[int, str], RawMask],
    track_id: str,
    left_frame: int,
    right_frame: int,
) -> list[RawMask]:
    return [
        value
        for (frame, candidate_track), value in trusted.items()
        if candidate_track == track_id and left_frame <= frame <= right_frame
    ]


def _interval_min_recall(
    left: Keyframe,
    right: Keyframe,
    masks: list[RawMask],
    left_scale: float,
    right_scale: float,
) -> float:
    if not masks:
        return 1.0
    return min(
        _recall(
            mask.geometry,
            _direct_geometry(
                left,
                right,
                mask.frame,
                left_scale=left_scale,
                right_scale=right_scale,
            ),
        )
        for mask in masks
    )


def _minimum_shared_scale(
    left: Keyframe,
    right: Keyframe,
    masks: list[RawMask],
    floor: float,
    max_scale: float,
) -> float:
    if _interval_min_recall(left, right, masks, 1.0, 1.0) >= floor:
        return 1.0
    steps = 100
    lower = 1.0
    upper: float | None = None
    for index in range(1, steps + 1):
        candidate = 1.0 + (max_scale - 1.0) * index / steps
        if _interval_min_recall(left, right, masks, candidate, candidate) >= floor:
            upper = candidate
            break
        lower = candidate
    if upper is None:
        achieved = _interval_min_recall(left, right, masks, max_scale, max_scale)
        raise RuntimeError(
            f"cannot satisfy Recall {floor:.6f} for interval "
            f"{left.frame}..{right.frame}; maximum at scale {max_scale:.6f} "
            f"is {achieved:.6f}"
        )
    for _iteration in range(22):
        candidate = 0.5 * (lower + upper)
        if _interval_min_recall(left, right, masks, candidate, candidate) >= floor:
            upper = candidate
        else:
            lower = candidate
    return upper


def apply_local_recall_guard(
    segments: dict[str, list[Segment]],
    trusted: dict[tuple[int, str], RawMask],
    *,
    floor: float,
    max_scale: float,
) -> tuple[dict[str, list[Segment]], dict[str, object]]:
    output: dict[str, list[Segment]] = {}
    interval_scales: list[float] = []
    applied_scales: list[float] = []
    guarded_intervals = 0
    guarded_keys = 0
    for track_id, values in segments.items():
        repaired_values: list[Segment] = []
        for segment in values:
            keys = list(segment.keyframes)
            key_scales = [1.0] * len(keys)
            for index, (left, right) in enumerate(zip(keys, keys[1:])):
                masks = _interval_masks(
                    trusted, track_id, int(left.frame), int(right.frame)
                )
                required = _minimum_shared_scale(left, right, masks, floor, max_scale)
                interval_scales.append(required)
                if required > 1.0 + 1e-10:
                    guarded_intervals += 1
                key_scales[index] = max(key_scales[index], required)
                key_scales[index + 1] = max(key_scales[index + 1], required)
            repaired_keys: list[Keyframe] = []
            for keyframe, scale in zip(keys, key_scales, strict=True):
                component = _primary_component(keyframe)
                if component is None:
                    raise ValueError("polygon component missing during scaling")
                points = _scaled(np.asarray(component.values, dtype=np.float64), scale)
                repaired_keys.append(_replace_primary_polygon(keyframe, points))
                applied_scales.append(scale)
                guarded_keys += int(scale > 1.0 + 1e-10)
            repaired_values.append(replace(segment, keyframes=tuple(repaired_keys)))
        output[track_id] = repaired_values
    return output, {
        "interval_count": len(interval_scales),
        "guarded_interval_count": guarded_intervals,
        "guarded_keyframe_count": guarded_keys,
        "interval_scale_mean": mean(interval_scales) if interval_scales else 1.0,
        "interval_scale_max": max(interval_scales, default=1.0),
        "applied_scale_mean": mean(applied_scales) if applied_scales else 1.0,
        "applied_scale_max": max(applied_scales, default=1.0),
    }


def _quantile(values: list[float], fraction: float) -> float:
    return (
        float(np.quantile(np.asarray(values, dtype=np.float64), fraction))
        if values
        else 0.0
    )


def audit_direct_interpolation(
    segments: dict[str, list[Segment]],
    trusted: dict[tuple[int, str], RawMask],
) -> dict[str, object]:
    recalls: list[float] = []
    ious: list[float] = []
    precisions: list[float] = []
    winding_flips = 0
    best_alignment_reversals = 0
    best_alignment_nonzero_shifts = 0
    point_counts: set[int] = set()
    keyframe_count = 0
    for track_id, values in segments.items():
        for segment in values:
            keys = list(segment.keyframes)
            keyframe_count += len(keys)
            for keyframe in keys:
                component = _primary_component(keyframe)
                if component is None:
                    continue
                point_counts.add(len(component.values))
            for left, right in zip(keys, keys[1:]):
                left_points = np.asarray(
                    _primary_component(left).values, dtype=np.float64
                )
                right_points = np.asarray(
                    _primary_component(right).values, dtype=np.float64
                )
                winding_flips += int(
                    _signed_area(left_points) * _signed_area(right_points) < 0.0
                )
                _aligned, reverse, shift, _error = _best_alignment(
                    left_points, right_points
                )
                best_alignment_reversals += int(reverse)
                best_alignment_nonzero_shifts += int(shift != 0)
                masks = _interval_masks(
                    trusted, track_id, int(left.frame), int(right.frame)
                )
                for mask in masks:
                    predicted = _direct_geometry(left, right, mask.frame)
                    intersection = float(predicted.intersection(mask.geometry).area)
                    reference_area = float(mask.geometry.area)
                    predicted_area = float(predicted.area)
                    union = reference_area + predicted_area - intersection
                    recalls.append(
                        intersection / reference_area if reference_area else 1.0
                    )
                    precisions.append(
                        intersection / predicted_area if predicted_area else 1.0
                    )
                    ious.append(intersection / union if union else 1.0)
    return {
        "keyframe_count": keyframe_count,
        "point_counts": sorted(point_counts),
        "adjacent_winding_flip_count": winding_flips,
        "best_alignment_reversal_count": best_alignment_reversals,
        "best_alignment_nonzero_shift_count": best_alignment_nonzero_shifts,
        "sample_count": len(recalls),
        "trusted_recall_min": min(recalls, default=1.0),
        "trusted_recall_q01": _quantile(recalls, 0.01),
        "trusted_recall_mean": mean(recalls) if recalls else 1.0,
        "trusted_recall_below_097_count": sum(value < 0.97 for value in recalls),
        "trusted_iou_min": min(ious, default=1.0),
        "trusted_iou_q01": _quantile(ious, 0.01),
        "trusted_iou_mean": mean(ious) if ious else 1.0,
        "trusted_precision_mean": mean(precisions) if precisions else 1.0,
    }


def _sqlite_metadata(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        version_row = connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()
        return {
            "size_bytes": path.stat().st_size,
            "schema_fingerprint": schema_fingerprint(connection),
            "schema_version": None if version_row is None else int(version_row[0]),
            "integrity_check": integrity,
            "foreign_key_error_count": len(foreign_keys),
        }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.trusted_recall_floor <= args.guard_floor <= 1.0:
        raise ValueError("expected 0 < trusted-recall-floor <= guard-floor <= 1")
    if args.point_count < 8:
        raise ValueError("point-count must be at least 8")
    source = args.raw_sqlite.expanduser().resolve()
    input_path = args.input_sqlite.expanduser().resolve()
    output = args.output_sqlite.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_name("vertex_alignment_report.json")
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    raw = load_raw_masks(
        source,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    original = load_segments(
        input_path,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    consensus = build_segment_bounded_temporal_consensus(
        raw,
        original,
        radius=args.consensus_radius,
        support_fraction=args.support_fraction,
    )
    canonical, canonical_stats = canonicalize_segments(original, args.point_count)
    if args.apply_local_recall_guard:
        repaired, guard_stats = apply_local_recall_guard(
            canonical,
            consensus.trusted_masks,
            floor=args.guard_floor,
            max_scale=args.max_scale,
        )
    else:
        repaired = canonical
        guard_stats = {
            "enabled": False,
            "reason": "vertex-correspondence-only repair preserves keyframe geometry",
        }
    pre_export_audit = audit_direct_interpolation(repaired, consensus.trusted_masks)
    if args.apply_local_recall_guard and (
        float(pre_export_audit["trusted_recall_min"]) + 1e-10
        < args.trusted_recall_floor
    ):
        raise RuntimeError(
            "final direct interpolation violates trusted Recall: "
            f"{pre_export_audit['trusted_recall_min']}"
        )
    if int(pre_export_audit["adjacent_winding_flip_count"]) != 0:
        raise RuntimeError("final keyframe path still contains winding flips")
    if int(pre_export_audit["best_alignment_reversal_count"]) != 0:
        raise RuntimeError("final keyframe path still needs reversal alignment")
    if int(pre_export_audit["best_alignment_nonzero_shift_count"]) != 0:
        raise RuntimeError("final keyframe path still needs cyclic shift alignment")

    export = export_selected_sqlite(
        input_path,
        output,
        repaired,
        raw,
        label=args.label,
        target_mean_key_interval=args.target_mean_key_interval,
        recall_floor=args.trusted_recall_floor,
        selection_reason="pareto_recall_constrained_vertex_aligned",
        algorithm=(
            "experimental.polygon_recall_optimizer.pareto_dp"
            "+vertex_correspondence_v1"
        ),
    )
    reloaded = load_segments(
        output,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    post_export_audit = audit_direct_interpolation(reloaded, consensus.trusted_masks)
    metadata = _sqlite_metadata(output)
    if (
        metadata["schema_fingerprint"]
        != _sqlite_metadata(input_path)["schema_fingerprint"]
    ):
        raise RuntimeError("schema fingerprint changed")
    if metadata["integrity_check"] != "ok" or metadata["foreign_key_error_count"]:
        raise RuntimeError(f"invalid output SQLite: {metadata}")
    if post_export_audit != pre_export_audit:
        raise RuntimeError("SQLite round-trip changed repaired polygon geometry")

    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "input_sqlite": str(input_path),
        "raw_sqlite": str(source),
        "output_sqlite": str(output),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "point_count": args.point_count,
        "trusted_recall_floor": args.trusted_recall_floor,
        "guard_floor": args.guard_floor,
        "local_recall_guard_requested": args.apply_local_recall_guard,
        "target_mean_key_interval": args.target_mean_key_interval,
        "canonicalization": canonical_stats,
        "local_recall_guard": guard_stats,
        "audit": post_export_audit,
        "export": export,
        "input_metadata": _sqlite_metadata(input_path),
        "output_metadata": metadata,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
