#!/usr/bin/env python3
"""Compare current equal-arc vertices with temporal trajectory decimation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from optimizer import (
    DecimationConfig,
    RasterSequenceEvaluator,
    align_temporal_dense,
    current_equal_arc_baseline,
    evaluate_sequence,
    optimize_temporal_vertices,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input-sqlite", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--track-id")
    value.add_argument("--max-frames", type=int, default=-1)
    value.add_argument("--initial-vertices", type=int, default=48)
    value.add_argument("--minimum-vertices", type=int, default=6)
    value.add_argument("--recall-floor", type=float, default=0.97)
    value.add_argument("--shortlist", type=int, default=10)
    value.add_argument("--temporal-weight", type=float, default=0.05)
    value.add_argument("--tail-weight", type=float, default=0.20)
    value.add_argument("--vertex-weight", type=float, default=0.02)
    value.add_argument("--local-refine-radius", type=int, default=0)
    value.add_argument("--local-refine-passes", type=int, default=1)
    value.add_argument("--native-threads", type=int, default=8)
    value.add_argument("--snapshot-counts", default="32,24,20,16,12,8")
    return value


def load_single_component_track(
    path: Path,
    track_id: str | None,
    max_frames: int,
) -> tuple[list[int], str, str, list[np.ndarray], list[int]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        selected = track_id
        if selected is None:
            row = connection.execute(
                """
                SELECT track_id, COUNT(*) AS n
                FROM masks
                GROUP BY track_id
                ORDER BY n DESC, CAST(track_id AS INTEGER)
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise ValueError("input SQLite has no masks")
            selected = str(row[0])
        rows = connection.execute(
            """
            SELECT frame, polygons, COALESCE(label, '')
            FROM masks
            WHERE CAST(track_id AS TEXT)=?
            ORDER BY frame
            """,
            (str(selected),),
        ).fetchall()
        cuts = [int(row[0]) for row in connection.execute("SELECT frame FROM cuts")]
    finally:
        connection.close()
    if max_frames > 0:
        rows = rows[: int(max_frames)]
    frames: list[int] = []
    polygons: list[np.ndarray] = []
    label = ""
    previous = None
    for frame, payload, row_label in rows:
        decoded = json.loads(str(payload))
        if len(decoded) != 1:
            raise ValueError(
                "this first experiment requires one component per frame; "
                f"frame {frame} has {len(decoded)}"
            )
        if previous is not None and int(frame) != int(previous) + 1:
            raise ValueError(
                f"track is not contiguous at {previous}->{frame}; use one segment"
            )
        if any(int(previous or frame) < cut <= int(frame) for cut in cuts):
            raise ValueError(f"track crosses a cut at frame {frame}; use one cut segment")
        frames.append(int(frame))
        polygons.append(np.asarray(decoded[0], dtype=np.float64))
        label = str(row_label)
        previous = int(frame)
    if not polygons:
        raise ValueError(f"track {selected} contains no polygons")
    return frames, str(selected), label, polygons, cuts


def write_masks_sqlite(
    path: Path,
    frames: list[int],
    track_id: str,
    label: str,
    polygons: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                shape_type TEXT NOT NULL,
                label TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO masks VALUES (?, ?, ?, 'polygon', ?)",
            (
                (
                    int(frame),
                    str(track_id),
                    json.dumps(
                        [np.asarray(polygon, dtype=np.float32).tolist()],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    str(label),
                )
                for frame, polygon in zip(frames, polygons, strict=True)
            ),
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, track_id, label, polygons, cuts = load_single_component_track(
        args.input_sqlite,
        args.track_id,
        args.max_frames,
    )
    snapshots = tuple(
        sorted(
            {
                int(value)
                for value in str(args.snapshot_counts).split(",")
                if value.strip()
            },
            reverse=True,
        )
    )
    config = DecimationConfig(
        initial_vertices=int(args.initial_vertices),
        minimum_vertices=int(args.minimum_vertices),
        recall_floor=float(args.recall_floor),
        shortlist=int(args.shortlist),
        temporal_weight=float(args.temporal_weight),
        tail_weight=float(args.tail_weight),
        vertex_weight=float(args.vertex_weight),
        local_refine_radius=int(args.local_refine_radius),
        local_refine_passes=int(args.local_refine_passes),
        native_threads=int(args.native_threads),
    )
    result = optimize_temporal_vertices(
        polygons,
        config,
        snapshot_counts=snapshots,
    )
    evaluator = RasterSequenceEvaluator(polygons)
    rows = []
    comparison_curve_seconds = 0.0
    for metrics in result.curve:
        count = int(metrics.vertices)
        started = time.perf_counter()
        baseline_polygons, baseline_metrics = current_equal_arc_baseline(
            polygons,
            count,
            evaluator=evaluator,
            initial_vertices=config.initial_vertices,
            temporal_weight=config.temporal_weight,
            tail_weight=config.tail_weight,
            vertex_weight=config.vertex_weight,
            check_self_intersections=False,
        )
        temporal_phase_polygons = align_temporal_dense(polygons, count)
        temporal_phase_metrics = evaluate_sequence(
            evaluator,
            temporal_phase_polygons,
            initial_vertices=config.initial_vertices,
            temporal_weight=config.temporal_weight,
            tail_weight=config.tail_weight,
            vertex_weight=config.vertex_weight,
            check_self_intersections=False,
        )
        comparison_curve_seconds += time.perf_counter() - started
        row = {
            "vertices": count,
            "temporal": asdict(metrics),
            "current_equal_arc": asdict(baseline_metrics),
            "temporal_phase_equal_arc": asdict(temporal_phase_metrics),
            "delta": {
                "mean_iou": metrics.mean_iou - baseline_metrics.mean_iou,
                "minimum_iou": metrics.minimum_iou - baseline_metrics.minimum_iou,
                "q01_iou": metrics.q01_iou - baseline_metrics.q01_iou,
                "minimum_recall": (
                    metrics.minimum_recall - baseline_metrics.minimum_recall
                ),
                "temporal_residual": (
                    metrics.temporal_residual
                    - baseline_metrics.temporal_residual
                ),
            },
        }
        rows.append(row)
        if count in snapshots:
            write_masks_sqlite(
                args.output_dir / f"temporal_vertices_{count}.sqlite",
                frames,
                track_id,
                label,
                result.snapshots.get(count, result.dense_aligned[:, :count]),
            )
            write_masks_sqlite(
                args.output_dir / f"current_equal_arc_{count}.sqlite",
                frames,
                track_id,
                label,
                baseline_polygons,
            )
            write_masks_sqlite(
                args.output_dir / f"temporal_phase_equal_arc_{count}.sqlite",
                frames,
                track_id,
                label,
                temporal_phase_polygons,
            )
    write_masks_sqlite(
        args.output_dir / "temporal_vertices_recommended.sqlite",
        frames,
        track_id,
        label,
        result.polygons,
    )
    write_masks_sqlite(
        args.output_dir / "temporal_vertices_greedy_terminal.sqlite",
        frames,
        track_id,
        label,
        result.greedy_terminal_polygons,
    )
    final_baseline, final_baseline_metrics = current_equal_arc_baseline(
        polygons,
        result.metrics.vertices,
        evaluator=evaluator,
        initial_vertices=config.initial_vertices,
        temporal_weight=config.temporal_weight,
        tail_weight=config.tail_weight,
        vertex_weight=config.vertex_weight,
        check_self_intersections=True,
    )
    write_masks_sqlite(
        args.output_dir / "current_equal_arc_final.sqlite",
        frames,
        track_id,
        label,
        final_baseline,
    )
    hybrid_candidates = []
    for row in rows:
        for mode in ("temporal", "temporal_phase_equal_arc"):
            metrics = row[mode]
            if float(metrics["minimum_recall"]) + 1e-12 < config.recall_floor:
                continue
            hybrid_candidates.append(
                (
                    float(metrics["objective"]),
                    -int(row["vertices"]),
                    str(mode),
                    int(row["vertices"]),
                    metrics,
                )
            )
    if not hybrid_candidates:
        raise RuntimeError("no feasible hybrid vertex solution")
    (
        _hybrid_objective,
        _hybrid_tie_break,
        hybrid_mode,
        hybrid_vertices,
        hybrid_metrics,
    ) = min(hybrid_candidates)
    if hybrid_mode == "temporal":
        if hybrid_vertices == result.metrics.vertices:
            hybrid_polygons = result.polygons
        else:
            hybrid_polygons = result.snapshots[hybrid_vertices]
    else:
        hybrid_polygons = align_temporal_dense(polygons, hybrid_vertices)
    write_masks_sqlite(
        args.output_dir / "recommended_hybrid.sqlite",
        frames,
        track_id,
        label,
        hybrid_polygons,
    )
    report = {
        "experiment": "temporal_vertex_trajectory_decimation_v1",
        "input_sqlite": str(args.input_sqlite.resolve()),
        "privacy": "SQLite polygon geometry only; video pixels were not opened",
        "track_id": track_id,
        "label": label,
        "frame_count": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "cut_count_in_source": len(cuts),
        "config": asdict(config),
        "result": {
            "active_indices": list(result.active_indices),
            "metrics": asdict(result.metrics),
            "greedy_terminal_active_indices": list(
                result.greedy_terminal_active_indices
            ),
            "greedy_terminal_metrics": asdict(result.greedy_terminal_metrics),
            "elapsed_seconds": result.elapsed_seconds,
            "exact_candidate_evaluations": result.exact_candidate_evaluations,
            "stopped_reason": result.stopped_reason,
        },
        "matching_current_baseline": asdict(final_baseline_metrics),
        "recommended_hybrid": {
            "mode": hybrid_mode,
            "vertices": hybrid_vertices,
            "metrics": hybrid_metrics,
        },
        "comparison_curve_seconds": comparison_curve_seconds,
        "curve": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["result"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
