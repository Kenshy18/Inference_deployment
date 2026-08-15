#!/usr/bin/env python3
"""Compare low vertex budgets with the actual Production polygon contract.

Only SQLite polygon geometry is read.  Source video pixels are never opened.
The comparison uses the exact Production keyframe schedule so that vertex
placement, rather than a different keyframe frequency, is being measured.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Iterable

import numpy as np

EXPERIMENTAL = Path(__file__).resolve().parents[1]
for value in (EXPERIMENTAL, EXPERIMENTAL / "temporal_vertex_decimation_20260812"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from humanlike_vertex_placement_20260812.native_dp import (  # noqa: E402
    native_temporal_dp_sequence,
)
from temporal_vertex_decimation_20260812.optimizer import (  # noqa: E402
    RasterSequenceEvaluator,
    evaluate_sequence,
)
from temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
)


DEFAULT_CASES = (
    ("55", "output/phase2_engine_profile_large_track55_20260810/input_track55.sqlite"),
    ("47", "output/phase2_engine_profile_medium_track_20260810/input_track47.sqlite"),
    ("36", "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track36.sqlite"),
    ("66", "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track66_segment1.sqlite"),
    ("97", "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track97_segment1.sqlite"),
)


def _production_keyframes(connection: sqlite3.Connection, track_id: str) -> dict[int, np.ndarray]:
    rows = connection.execute(
        """
        SELECT k.frame, p.point_index, p.x, p.y
        FROM mask_track_segments AS s
        JOIN mask_keyframes AS k ON k.segment_id = s.id
        JOIN keyframe_components AS c
          ON c.keyframe_id = k.id AND c.geometry_type = 'polygon'
        JOIN keyframe_polygon_rings AS r
          ON r.component_id = c.id AND r.ring_index = 0
        JOIN keyframe_polygon_points AS p ON p.ring_id = r.id
        WHERE s.track_id = ? AND c.slot_index = 0
        ORDER BY k.frame, p.point_index
        """,
        (str(track_id),),
    ).fetchall()
    grouped: dict[int, list[list[float]]] = {}
    for frame, _index, x, y in rows:
        grouped.setdefault(int(frame), []).append([float(x), float(y)])
    return {frame: np.asarray(points, dtype=np.float64) for frame, points in grouped.items()}


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "minimum": float(np.min(array)),
    }


def _fit_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _movement(keyframes: dict[int, np.ndarray]) -> dict[str, object]:
    ordered = sorted(keyframes.items())
    absolute_per_frame: list[float] = []
    residual_px: list[float] = []
    residual_px_per_frame: list[float] = []
    residual_radius: list[float] = []
    residual_spacing: list[float] = []
    nonzero_best_shifts = 0
    transitions = 0
    for (left_frame, left), (right_frame, right) in zip(ordered, ordered[1:], strict=False):
        if len(left) != len(right):
            continue
        gap = max(int(right_frame) - int(left_frame), 1)
        prediction = _fit_similarity(left, right)
        displacement = np.linalg.norm(right - left, axis=1) / float(gap)
        absolute_per_frame.extend(displacement.tolist())
        residual_values = np.linalg.norm(right - prediction, axis=1)
        residual_px.extend(residual_values.tolist())
        residual_px_per_frame.extend((residual_values / float(gap)).tolist())
        centered = right - np.mean(right, axis=0)
        radius = math.sqrt(max(abs(float(np.sum(
            right[:, 0] * np.roll(right[:, 1], -1)
            - np.roll(right[:, 0], -1) * right[:, 1]
        ))) * 0.5, 1.0) / math.pi)
        residual_radius.extend((residual_values / radius).tolist())
        spacing = 0.5 * (
            np.linalg.norm(right - np.roll(right, 1, axis=0), axis=1)
            + np.linalg.norm(np.roll(right, -1, axis=0) - right, axis=1)
        )
        residual = np.linalg.norm(right - prediction, axis=1) / np.maximum(spacing, 1e-9)
        residual_spacing.extend(residual.tolist())
        centered_left = prediction - np.mean(prediction, axis=0)
        centered_right = right - np.mean(right, axis=0)
        costs = [
            float(np.sum((centered_left - np.roll(centered_right, shift, axis=0)) ** 2))
            for shift in range(len(right))
        ]
        nonzero_best_shifts += int(int(np.argmin(costs)) != 0)
        transitions += 1
    return {
        "keyframe_transitions": int(transitions),
        "absolute_vertex_velocity_px_per_frame": _distribution(absolute_per_frame),
        "similarity_residual_px": _distribution(residual_px),
        "similarity_residual_px_per_frame": _distribution(residual_px_per_frame),
        "similarity_residual_in_equivalent_radius": _distribution(residual_radius),
        "similarity_residual_in_local_spacings": _distribution(residual_spacing),
        "best_cyclic_alignment_nonzero": int(nonzero_best_shifts),
    }


def _keyframe_metrics(
    evaluator: RasterSequenceEvaluator,
    frames: list[int],
    keyframes: dict[int, np.ndarray],
) -> dict[str, object]:
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    ious: list[float] = []
    recalls: list[float] = []
    for frame, polygon in sorted(keyframes.items()):
        index = frame_to_index.get(int(frame))
        if index is None:
            continue
        iou, recall = evaluator.frame_metrics(index, polygon)
        ious.append(float(iou))
        recalls.append(float(recall))
    return {
        "frames": int(len(ious)),
        "iou": _distribution(ious),
        "recall": _distribution(recalls),
        "recall_below_097": int(np.sum(np.asarray(recalls) < 0.97)),
        "iou_below_095": int(np.sum(np.asarray(ious) < 0.95)),
    }


def _interpolated_metrics(
    evaluator: RasterSequenceEvaluator,
    frames: list[int],
    keyframes: dict[int, np.ndarray],
) -> dict[str, object]:
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    ordered = sorted(
        (frame, points) for frame, points in keyframes.items() if frame in frame_to_index
    )
    ious: list[float] = []
    recalls: list[float] = []
    sampled_frames: set[int] = set()
    for (left_frame, left), (right_frame, right) in zip(ordered, ordered[1:], strict=False):
        if len(left) != len(right) or right_frame <= left_frame:
            continue
        for frame in range(left_frame, right_frame + 1):
            index = frame_to_index.get(frame)
            if index is None or frame in sampled_frames:
                continue
            alpha = (frame - left_frame) / float(right_frame - left_frame)
            polygon = (1.0 - alpha) * left + alpha * right
            iou, recall = evaluator.frame_metrics(index, polygon)
            ious.append(float(iou))
            recalls.append(float(recall))
            sampled_frames.add(frame)
    if not ious:
        return {"frames": 0}
    return {
        "frames": int(len(ious)),
        "iou": _distribution(ious),
        "recall": _distribution(recalls),
        "recall_below_097": int(np.sum(np.asarray(recalls) < 0.97)),
        "iou_below_095": int(np.sum(np.asarray(ious) < 0.95)),
    }


def _sequence_metrics(evaluator: RasterSequenceEvaluator, sequence: np.ndarray) -> dict[str, object]:
    metrics = evaluate_sequence(
        evaluator,
        sequence,
        initial_vertices=int(sequence.shape[1]),
        temporal_weight=0.05,
        tail_weight=0.20,
        vertex_weight=0.0,
        check_self_intersections=True,
    )
    return {
        "mean_iou": float(metrics.mean_iou),
        "minimum_iou": float(metrics.minimum_iou),
        "q01_iou": float(metrics.q01_iou),
        "minimum_recall": float(metrics.minimum_recall),
        "temporal_residual": float(metrics.temporal_residual),
        "self_intersections": int(metrics.self_intersections),
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Production versus low vertex budget",
        "",
        "SQLite polygon geometry only; no source video pixels were opened.",
        "",
        "## Per-track minimum feasible count",
        "",
        "| track | Production vertices | minimum new vertices | new min Recall | new min IoU | Production dense mean IoU | new dense mean IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in report["tracks"]:
        feasible = track.get("minimum_feasible")
        chosen = next(
            (row for row in track["candidates"] if row["vertices"] == feasible), None
        )
        lines.append(
            "| {track} | {production} | {feasible} | {recall} | {iou} | {prod_dense} | {new_dense} |".format(
                track=track["track_id"],
                production=track["production_vertices"],
                feasible=feasible if feasible is not None else "none",
                recall=f"{chosen['per_frame']['minimum_recall']:.6f}" if chosen else "-",
                iou=f"{chosen['per_frame']['minimum_iou']:.6f}" if chosen else "-",
                prod_dense=f"{track['production']['interpolated']['iou']['mean']:.6f}",
                new_dense=f"{chosen['production_schedule_interpolated']['iou']['mean']:.6f}" if chosen else "-",
            )
        )
    lines.extend([
        "",
        "## Candidate curves",
        "",
    ])
    for track in report["tracks"]:
        lines.extend([
            f"### Track {track['track_id']}",
            "",
            "| vertices | FPS | min Recall | min IoU | mean IoU | key min Recall | key mean IoU | dense mean IoU | residual px/frame | residual/radius |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in track["candidates"]:
            dense = row["production_schedule_interpolated"]
            keys = row["production_keyframes"]
            residual = row["movement"]["similarity_residual_px_per_frame"]
            residual_radius = row["movement"]["similarity_residual_in_equivalent_radius"]
            lines.append(
                f"| {row['vertices']} | {row['fps']:.1f} | "
                f"{row['per_frame']['minimum_recall']:.6f} | "
                f"{row['per_frame']['minimum_iou']:.6f} | "
                f"{row['per_frame']['mean_iou']:.6f} | "
                f"{keys['recall']['minimum']:.6f} | {keys['iou']['mean']:.6f} | "
                f"{dense['iou']['mean']:.6f} | {residual['mean']:.6f} | "
                f"{residual_radius['mean']:.6f} |"
            )
        prod = track["production"]
        residual = prod["movement"]["similarity_residual_px_per_frame"]
        residual_radius = prod["movement"]["similarity_residual_in_equivalent_radius"]
        keys = prod["keyframes"]
        dense = prod["interpolated"]
        lines.append(
            f"| Production ({track['production_vertices']}) | - | - | - | - | "
            f"{keys['recall']['minimum']:.6f} | {keys['iou']['mean']:.6f} | "
            f"{dense['iou']['mean']:.6f} | {residual['mean']:.6f} | "
            f"{residual_radius['mean']:.6f} |"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-sqlite", type=Path, default=Path("data/12月KPI動画.sqlite"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vertices", default="8-20")
    args = parser.parse_args()
    if "-" in args.vertices:
        first, last = (int(value) for value in args.vertices.split("-", 1))
        counts = tuple(range(first, last + 1))
    else:
        counts = tuple(int(value) for value in args.vertices.split(","))
    report: dict[str, object] = {
        "schema_version": 1,
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "production_sqlite": str(args.production_sqlite.resolve()),
        "candidate_counts": list(counts),
        "quality_gate": {"minimum_recall": 0.97, "minimum_iou": 0.95, "self_intersections": 0},
        "tracks": [],
    }
    with sqlite3.connect(args.production_sqlite) as production:
        for track_id, relative_path in DEFAULT_CASES:
            frames, _track, label, polygons, cuts = load_single_component_track(
                Path(relative_path), track_id, -1
            )
            production_all = _production_keyframes(production, track_id)
            production_keys = {
                frame: points for frame, points in production_all.items() if frame in set(frames)
            }
            if len(production_keys) < 2:
                continue
            point_counts = {len(value) for value in production_keys.values()}
            if len(point_counts) != 1:
                raise ValueError(f"Production track {track_id} changes vertex count: {point_counts}")
            evaluator = RasterSequenceEvaluator(polygons)
            track_report: dict[str, object] = {
                "track_id": track_id,
                "label": label,
                "raw_frames": len(frames),
                "evaluated_keyframes": len(production_keys),
                "production_vertices": int(next(iter(point_counts))),
                "production": {
                    "keyframes": _keyframe_metrics(evaluator, frames, production_keys),
                    "interpolated": _interpolated_metrics(evaluator, frames, production_keys),
                    "movement": _movement(production_keys),
                },
                "candidates": [],
                "minimum_feasible": None,
            }
            frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
            for count in counts:
                started = time.perf_counter()
                sequence = native_temporal_dp_sequence(
                    polygons,
                    count,
                    frame_indices=frames,
                    cut_frames=cuts,
                    temporal_weight=0.003,
                    distance_weight=2.0,
                    missing_area_weight=1.0,
                )
                seconds = time.perf_counter() - started
                per_frame = _sequence_metrics(evaluator, sequence)
                candidate_keys = {
                    frame: sequence[frame_to_index[frame]] for frame in production_keys
                }
                row = {
                    "vertices": int(count),
                    "generation_seconds": float(seconds),
                    "fps": float(len(frames) / max(seconds, 1e-12)),
                    "per_frame": per_frame,
                    "production_keyframes": _keyframe_metrics(
                        evaluator, frames, candidate_keys
                    ),
                    "production_schedule_interpolated": _interpolated_metrics(
                        evaluator, frames, candidate_keys
                    ),
                    "movement": _movement(candidate_keys),
                }
                track_report["candidates"].append(row)
                feasible = (
                    per_frame["minimum_recall"] >= 0.97
                    and per_frame["minimum_iou"] >= 0.95
                    and per_frame["self_intersections"] == 0
                )
                if feasible and track_report["minimum_feasible"] is None:
                    track_report["minimum_feasible"] = int(count)
                print(
                    f"track={track_id} vertices={count} fps={row['fps']:.1f} "
                    f"min_recall={per_frame['minimum_recall']:.6f} "
                    f"min_iou={per_frame['minimum_iou']:.6f}"
                )
            report["tracks"].append(track_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
