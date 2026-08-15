#!/usr/bin/env python3
"""Audit local vertex allocation changes between adjacent frames.

This deliberately uses polygon geometry only.  It detects the concrete failure
mode where a vertex changes from a meaningful corner to a redundant point on a
nearly straight run (or vice versa), even though total vertex count is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

EXPERIMENTAL = Path(__file__).resolve().parents[1]
for value in (EXPERIMENTAL, EXPERIMENTAL / "temporal_vertex_decimation_20260812"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.candidate import quality_guarded_vertex_placement  # noqa: E402
from temporal_vertex_decimation_20260812.optimizer import signed_area  # noqa: E402
from temporal_vertex_decimation_20260812.run_experiment import load_single_component_track  # noqa: E402


def _similarity_prediction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_center = np.mean(left, axis=0)
    right_center = np.mean(right, axis=0)
    left_zero = left - left_center
    right_zero = right - right_center
    covariance = left_zero.T @ right_zero
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(np.sum(left_zero * left_zero), 1e-12))
    return scale * (left_zero @ rotation) + right_center


def _straight_vertices(polygon: np.ndarray, angle_degrees: float, height_ratio: float) -> np.ndarray:
    previous = np.roll(polygon, 1, axis=0)
    following = np.roll(polygon, -1, axis=0)
    left = previous - polygon
    right = following - polygon
    cosine = np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )
    interior = np.arccos(np.clip(cosine, -1.0, 1.0))
    deviation = np.pi - interior
    chord = following - previous
    height = np.abs(
        chord[:, 0] * (polygon[:, 1] - previous[:, 1])
        - chord[:, 1] * (polygon[:, 0] - previous[:, 0])
    ) / np.maximum(np.linalg.norm(chord, axis=1), 1e-12)
    radius = math.sqrt(max(abs(signed_area(polygon)), 1.0) / math.pi)
    return (deviation <= math.radians(angle_degrees)) & (height <= height_ratio * radius)


def audit(sequence: np.ndarray, frames: list[int]) -> dict:
    events = []
    all_ratios = []
    redundant_counts = []
    straight = [_straight_vertices(value, 8.0, 0.02) for value in sequence]
    redundant_counts = [int(np.sum(value)) for value in straight]
    for frame in range(1, len(sequence)):
        if int(frames[frame]) != int(frames[frame - 1]) + 1:
            continue
        prediction = _similarity_prediction(sequence[frame - 1], sequence[frame])
        residual = np.linalg.norm(prediction - sequence[frame], axis=1)
        local_spacing = 0.5 * (
            np.linalg.norm(sequence[frame] - np.roll(sequence[frame], 1, axis=0), axis=1)
            + np.linalg.norm(np.roll(sequence[frame], -1, axis=0) - sequence[frame], axis=1)
        )
        ratio = residual / np.maximum(local_spacing, 1e-9)
        all_ratios.extend(ratio.tolist())
        toggled = straight[frame - 1] != straight[frame]
        for vertex in np.flatnonzero(toggled & (ratio >= 0.25)):
            events.append(
                {
                    "previous_frame": int(frames[frame - 1]),
                    "frame": int(frames[frame]),
                    "vertex": int(vertex),
                    "previous_straight": bool(straight[frame - 1][vertex]),
                    "current_straight": bool(straight[frame][vertex]),
                    "movement_in_local_spacings": float(ratio[vertex]),
                    "movement_pixels": float(residual[vertex]),
                    "local_spacing_pixels": float(local_spacing[vertex]),
                }
            )
    values = np.asarray(all_ratios, dtype=np.float64)
    severe = [value for value in events if value["movement_in_local_spacings"] >= 0.5]
    return {
        "adjacent_vertex_transitions": int(len(values)),
        "movement_ratio_mean": float(np.mean(values)) if len(values) else 0.0,
        "movement_ratio_q95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "movement_ratio_q99": float(np.quantile(values, 0.99)) if len(values) else 0.0,
        "movement_ratio_max": float(np.max(values)) if len(values) else 0.0,
        "ratio_over_half_spacing": int(np.sum(values >= 0.5)),
        "ratio_over_one_spacing": int(np.sum(values >= 1.0)),
        "straight_vertex_count_mean": float(np.mean(redundant_counts)),
        "straight_allocation_toggles_over_quarter_spacing": len(events),
        "straight_allocation_toggles_over_half_spacing": len(severe),
        "worst_events": sorted(
            events,
            key=lambda item: item["movement_in_local_spacings"],
            reverse=True,
        )[:30],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    frames, _track_id, _label, polygons, cuts = load_single_component_track(
        args.input_sqlite, args.track_id, -1
    )
    result = quality_guarded_vertex_placement(
        polygons,
        frame_indices=frames,
        cut_frames=cuts,
        candidate_counts=(args.vertices,),
    )
    report = {
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "track": args.track_id,
        "frames": len(frames),
        "vertices": args.vertices,
        **audit(result.polygons, frames),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
