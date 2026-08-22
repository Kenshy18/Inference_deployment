"""CLI for fitting and auditing the closed Catmull--Rom experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from production.polygon.runtime.spatial_support.optimizer import temporal_residuals
from production.polygon.runtime.spatial_support.placement import (
    align_polygon_sequence,
    rdp_fixed_count,
)

from production.curve.runtime.fitter import FitConfig, fit_sequence
from .io import available_tracks, load_track_contours
from production.curve.runtime.metrics import (
    frame_raster_metrics,
    sequence_raster_metrics,
)


def _counts(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(value.strip()) for value in text.split(","))))
    if not values or min(values) < 3:
        raise argparse.ArgumentTypeError(
            "control counts must be comma-separated integers >=3"
        )
    return values


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _baseline(contours: tuple[np.ndarray, ...], count: int, recall_floor: float):
    polygons = align_polygon_sequence(
        [rdp_fixed_count(contour, int(count)) for contour in contours]
    )
    return polygons, sequence_raster_metrics(
        list(contours), polygons, polygons, recall_floor=float(recall_floor)
    )


def _write_frame_metrics(
    path: Path,
    frames: np.ndarray,
    contours: tuple[np.ndarray, ...],
    curves: np.ndarray,
    controls: np.ndarray,
    scales: np.ndarray,
    baseline: np.ndarray,
) -> None:
    curve_residual = temporal_residuals(controls)
    baseline_residual = temporal_residuals(baseline)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame",
                "curve_iou",
                "curve_recall",
                "curve_repair_scale",
                "curve_temporal_residual",
                "polygon_iou",
                "polygon_recall",
                "polygon_temporal_residual",
            ),
        )
        writer.writeheader()
        for index, frame in enumerate(frames):
            curve_iou, curve_recall = frame_raster_metrics(
                contours[index], curves[index]
            )
            polygon_iou, polygon_recall = frame_raster_metrics(
                contours[index], baseline[index]
            )
            writer.writerow(
                {
                    "frame": int(frame),
                    "curve_iou": f"{curve_iou:.12g}",
                    "curve_recall": f"{curve_recall:.12g}",
                    "curve_repair_scale": f"{float(scales[index]):.12g}",
                    "curve_temporal_residual": f"{float(np.mean(curve_residual[index - 1])):.12g}"
                    if index
                    else "0",
                    "polygon_iou": f"{polygon_iou:.12g}",
                    "polygon_recall": f"{polygon_recall:.12g}",
                    "polygon_temporal_residual": f"{float(np.mean(baseline_residual[index - 1])):.12g}"
                    if index
                    else "0",
                }
            )


def _fit_count(
    args: argparse.Namespace, track, count: int, root: Path
) -> dict[str, object]:
    result = fit_sequence(
        list(track.contours),
        FitConfig(
            control_point_count=int(count),
            dense_contour_samples=int(args.dense_samples),
            samples_per_segment=int(args.samples_per_segment),
            recall_floor=float(args.recall_floor),
            index_refine_passes=int(args.refine_passes),
            index_refine_radius=int(args.refine_radius),
            proxy_max_frames=int(args.proxy_frames),
            scale_maximum=float(args.scale_maximum),
            scale_step=float(args.scale_step),
        ),
    )
    baseline, baseline_metrics = _baseline(
        track.contours, int(count), float(args.recall_floor)
    )
    destination = root / f"points_{count:02d}"
    destination.mkdir(parents=True)
    np.savez_compressed(
        destination / "curve_arrays.npz",
        frames=track.frames,
        controls=result.controls,
        bezier_segments=result.segments,
        sampled_curves=result.sampled_curves,
        repair_scales=result.repair_scales,
        polygon_baseline=baseline,
    )
    editor_payload = [
        {
            "frame": int(frame),
            "track_id": track.track_id,
            "control_points_P": result.controls[index].tolist(),
            "derived_bezier_segments_B0_B1_B2_B3": result.segments[index].tolist(),
        }
        for index, frame in enumerate(track.frames)
    ]
    _atomic_json(destination / "editor_curve.json", editor_payload)
    _write_frame_metrics(
        destination / "frame_metrics.csv",
        track.frames,
        track.contours,
        result.sampled_curves,
        result.controls,
        result.repair_scales,
        baseline,
    )
    summary = result.summary()
    summary.update(
        {
            "track_id": track.track_id,
            "label": track.label,
            "component_index": track.component_index,
            "source_frames": [int(track.frames[0]), int(track.frames[-1])],
            "polygon_same_point_count_baseline": asdict(baseline_metrics),
            "curve_minus_polygon_mean_iou": float(
                result.metrics.mean_iou - baseline_metrics.mean_iou
            ),
            "curve_minus_polygon_temporal_residual": float(
                result.metrics.temporal_residual - baseline_metrics.temporal_residual
            ),
        }
    )
    _atomic_json(destination / "summary.json", summary)
    return summary


def _write_comparison_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    fields = (
        "control_points",
        "curve_mean_iou",
        "curve_min_iou",
        "curve_min_recall",
        "curve_recall_violations",
        "curve_temporal_residual",
        "curve_self_intersections",
        "polygon_mean_iou",
        "polygon_min_recall",
        "polygon_temporal_residual",
        "mean_iou_gain",
        "elapsed_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            baseline = summary["polygon_same_point_count_baseline"]
            writer.writerow(
                {
                    "control_points": summary["config"]["control_point_count"],
                    "curve_mean_iou": metrics["mean_iou"],
                    "curve_min_iou": metrics["minimum_iou"],
                    "curve_min_recall": metrics["minimum_recall"],
                    "curve_recall_violations": metrics["recall_violations"],
                    "curve_temporal_residual": metrics["temporal_residual"],
                    "curve_self_intersections": metrics["self_intersections"],
                    "polygon_mean_iou": baseline["mean_iou"],
                    "polygon_min_recall": baseline["minimum_recall"],
                    "polygon_temporal_residual": baseline["temporal_residual"],
                    "mean_iou_gain": summary["curve_minus_polygon_mean_iou"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the exact editor-compatible closed Catmull-Rom curve"
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--list-tracks", action="store_true")
    parser.add_argument("--component-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--control-points", type=_counts, default=(6, 8, 10, 12))
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--dense-samples", type=int, default=192)
    parser.add_argument("--samples-per-segment", type=int, default=16)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--refine-radius", type=int, default=8)
    parser.add_argument("--proxy-frames", type=int, default=32)
    parser.add_argument("--scale-maximum", type=float, default=1.08)
    parser.add_argument("--scale-step", type=float, default=0.002)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tracks:
        print(
            json.dumps(
                available_tracks(args.input_sqlite), ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.track_id is None:
        raise SystemExit("--track-id is required unless --list-tracks is used")
    output = args.output_dir or Path("output") / "catmull_rom_bezier_experiment"
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {output} (use --force)")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    track = load_track_contours(
        args.input_sqlite,
        args.track_id,
        component_index=args.component_index,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )
    summaries = [
        _fit_count(args, track, count, output) for count in args.control_points
    ]
    comparison = {
        "input_sqlite": str(args.input_sqlite.resolve()),
        "track_id": track.track_id,
        "label": track.label,
        "frames": int(len(track.frames)),
        "control_point_results": summaries,
    }
    _atomic_json(output / "comparison.json", comparison)
    _write_comparison_csv(output / "comparison.csv", summaries)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
