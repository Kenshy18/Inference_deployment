"""End-to-end polygon-versus-curve keyframe experiment and review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from contracts.mask_sqlite import MaskRow, write_mask_sqlite

from production.curve.runtime.fitter import FitConfig, fit_sequence
from .io import load_track_contours
from production.curve.runtime.keyframe_dp import (
    KeyframeDpConfig,
    KeyframeDpResult,
    catmull_rom_renderer,
    optimize_keyframes,
    polygon_renderer,
)
from production.curve.runtime.model import bezier_segments
from production.curve.runtime.multistate_dp import (
    CurvePointRefineConfig,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)
from .review import render_review_gallery, render_review_video
from production.curve.runtime.spatial import (
    build_production_polygon_controls,
    repair_spatial_controls,
)


def _integer_list(text: str, minimum: int) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(value.strip()) for value in text.split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or min(values) < int(minimum):
        raise argparse.ArgumentTypeError(
            f"values must be comma-separated integers >= {minimum}"
        )
    return values


def _points(text: str) -> tuple[int, ...]:
    return _integer_list(text, 3)


def _intervals(text: str) -> tuple[int, ...]:
    return _integer_list(text, 1)


def _float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted(set(float(value.strip()) for value in text.split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or min(values) < 0.95:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated floats >= 0.95"
        )
    return values


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_dense_sqlite(
    path: Path,
    frames: np.ndarray,
    track_id: str,
    label: str,
    result: KeyframeDpResult,
) -> None:
    rows = [
        MaskRow(
            frame=int(frame),
            track_id=str(track_id),
            polygons=json.dumps(
                [result.dense_boundaries[index].tolist()],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            label=str(label),
            shape_type="polygon",
        )
        for index, frame in enumerate(frames)
    ]
    write_mask_sqlite(path, rows)


def _write_keyframes(
    path: Path,
    frames: np.ndarray,
    track_id: str,
    result: KeyframeDpResult,
    *,
    is_curve: bool,
) -> None:
    payload = []
    for position, index in enumerate(result.chosen_indices):
        controls = result.keyframe_controls[position]
        row = {
            "frame": int(frames[index]),
            "frame_index": int(index),
            "track_id": str(track_id),
            "control_points_P": controls.tolist(),
        }
        if is_curve:
            row["derived_bezier_segments_B0_B1_B2_B3"] = bezier_segments(
                controls
            ).tolist()
            row[
                "curve_contract"
            ] = "closed_uniform_catmull_rom_tension_1_factor_1_over_6"
        payload.append(row)
    _write_json(path, payload)


def _write_frame_metrics(
    path: Path,
    frames: np.ndarray,
    polygon: KeyframeDpResult,
    curve: KeyframeDpResult,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "frame",
            "polygon_iou",
            "curve_iou",
            "curve_minus_polygon_iou",
            "polygon_recall",
            "curve_recall",
            "polygon_precision",
            "curve_precision",
            "polygon_area_ratio",
            "curve_area_ratio",
            "polygon_is_keyframe",
            "curve_is_keyframe",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        polygon_keys = set(polygon.chosen_indices)
        curve_keys = set(curve.chosen_indices)
        for index, frame in enumerate(frames):
            writer.writerow(
                {
                    "frame": int(frame),
                    "polygon_iou": f"{polygon.audit.iou[index]:.12g}",
                    "curve_iou": f"{curve.audit.iou[index]:.12g}",
                    "curve_minus_polygon_iou": (
                        f"{curve.audit.iou[index] - polygon.audit.iou[index]:.12g}"
                    ),
                    "polygon_recall": f"{polygon.audit.recall[index]:.12g}",
                    "curve_recall": f"{curve.audit.recall[index]:.12g}",
                    "polygon_precision": f"{polygon.audit.precision[index]:.12g}",
                    "curve_precision": f"{curve.audit.precision[index]:.12g}",
                    "polygon_area_ratio": f"{polygon.audit.area_ratio[index]:.12g}",
                    "curve_area_ratio": f"{curve.audit.area_ratio[index]:.12g}",
                    "polygon_is_keyframe": int(index in polygon_keys),
                    "curve_is_keyframe": int(index in curve_keys),
                }
            )


def _run_one(
    root: Path,
    track,
    point_count: int,
    interval: int,
    polygon_controls: np.ndarray,
    curve_controls: np.ndarray,
    samples_per_segment: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], KeyframeDpResult, KeyframeDpResult]:
    shared_config = dict(
        target_interval=int(interval),
        recall_floor=float(args.recall_floor),
        maximum_gap=max(int(args.maximum_gap), 4 * int(interval)),
        pair_vote_enabled=not bool(args.no_pair_vote),
        pair_vote_sweeps=int(args.pair_vote_sweeps),
        pair_vote_global_steps=int(args.pair_vote_global_steps),
        pair_vote_local_steps=int(args.pair_vote_local_steps),
        native_cpu_threads=int(args.native_cpu_threads),
        native_batch_cases=int(args.native_batch_cases),
    )
    polygon_config = KeyframeDpConfig(**shared_config)
    curve_config = KeyframeDpConfig(
        **shared_config,
        low_iou_quadratic_weight=float(args.curve_low_iou_weight),
        quality_rescue_enabled=not bool(args.no_curve_quality_rescue),
        quality_rescue_iou_floor=float(args.curve_rescue_iou_floor),
        quality_rescue_regret_floor=float(args.curve_rescue_regret_floor),
        quality_rescue_area_ratio_cap=float(args.curve_rescue_area_ratio_cap),
        quality_rescue_maximum_extra_keys=int(args.curve_rescue_maximum_keys),
    )
    polygon = optimize_keyframes(
        list(track.contours),
        polygon_controls,
        representation="current_persistent_polygon",
        renderer=polygon_renderer,
        config=polygon_config,
    )
    if bool(args.single_state_curve):
        curve = optimize_keyframes(
            list(track.contours),
            curve_controls,
            representation="closed_uniform_catmull_rom_bezier_single_state",
            renderer=catmull_rom_renderer(samples_per_segment),
            config=curve_config,
        )
    else:
        states, state_labels = isotropic_curve_states(
            curve_controls, tuple(args.curve_state_scales)
        )
        curve = optimize_multistate_keyframes(
            list(track.contours),
            states,
            state_labels=state_labels,
            base_controls=curve_controls,
            representation="closed_uniform_catmull_rom_bezier_multistate",
            renderer=catmull_rom_renderer(samples_per_segment),
            config=curve_config,
            point_refine=CurvePointRefineConfig(
                enabled=not bool(args.no_curve_point_refine),
                sweeps=int(args.curve_point_refine_sweeps),
            ),
            interval_renderer=catmull_rom_renderer(
                int(args.curve_dp_samples_per_segment)
            ),
        )
    destination = root / f"points_{point_count:02d}" / f"interval_{interval:02d}"
    destination.mkdir(parents=True)
    _write_dense_sqlite(
        destination / "polygon_dense_review.sqlite",
        track.frames,
        track.track_id,
        track.label,
        polygon,
    )
    _write_dense_sqlite(
        destination / "curve_dense_review.sqlite",
        track.frames,
        track.track_id,
        track.label,
        curve,
    )
    _write_keyframes(
        destination / "polygon_keyframes.json",
        track.frames,
        track.track_id,
        polygon,
        is_curve=False,
    )
    _write_keyframes(
        destination / "curve_keyframes.json",
        track.frames,
        track.track_id,
        curve,
        is_curve=True,
    )
    _write_frame_metrics(
        destination / "frame_metrics.csv", track.frames, polygon, curve
    )
    polygon_summary = polygon.summary()
    curve_summary = curve.summary()
    summary = {
        "point_count": int(point_count),
        "target_interval": int(interval),
        "track_id": track.track_id,
        "label": track.label,
        "frames": int(len(track.frames)),
        "polygon": polygon_summary,
        "curve": curve_summary,
        "curve_minus_polygon": {
            "mean_iou": float(curve_summary["mean_iou"] - polygon_summary["mean_iou"]),
            "minimum_iou": float(
                curve_summary["minimum_iou"] - polygon_summary["minimum_iou"]
            ),
            "effective_interval": float(
                curve_summary["effective_interval"]
                - polygon_summary["effective_interval"]
            ),
            "area_ratio_q95": float(
                curve_summary["area_ratio_q95"] - polygon_summary["area_ratio_q95"]
            ),
            "elapsed_seconds": float(
                curve_summary["elapsed_seconds"] - polygon_summary["elapsed_seconds"]
            ),
        },
        "curve_dense_review_sqlite_note": (
            "sampled curve polygon for current overlay/editor review; "
            "curve_keyframes.json is the semantic P/B source"
        ),
        "curve_state_scales": (
            [1.0] if args.single_state_curve else list(args.curve_state_scales)
        ),
    }
    _write_json(destination / "summary.json", summary)
    return summary, polygon, curve


def _write_comparison(path: Path, summaries: list[dict[str, object]]) -> None:
    fields = (
        "point_count",
        "target_interval",
        "representation",
        "chosen_keyframes",
        "effective_interval",
        "mean_iou",
        "minimum_iou",
        "q01_iou",
        "q05_iou",
        "minimum_recall",
        "recall_violations",
        "area_ratio_mean",
        "area_ratio_q95",
        "area_ratio_maximum",
        "topology_invalid_frames",
        "pair_vote_iou_gain",
        "elapsed_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for name in ("polygon", "curve"):
                value = summary[name]
                writer.writerow(
                    {
                        "point_count": summary["point_count"],
                        "target_interval": summary["target_interval"],
                        "representation": name,
                        **{field: value[field] for field in fields[3:]},
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare current polygon DP with editor-compatible Catmull-Rom DP"
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--component-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--point-counts", type=_points, default=(8, 10, 12))
    parser.add_argument("--target-intervals", type=_intervals, default=(3, 6))
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--samples-per-segment", type=int, default=16)
    parser.add_argument(
        "--curve-dp-samples-per-segment",
        type=int,
        default=16,
        help=(
            "CPU graph sampling density. The default matches final rendering "
            "exactly; lower values are an explicit preview-only approximation."
        ),
    )
    parser.add_argument("--dense-samples", type=int, default=192)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--refine-radius", type=int, default=8)
    parser.add_argument("--proxy-frames", type=int, default=32)
    parser.add_argument("--spatial-scale-maximum", type=float, default=1.08)
    parser.add_argument("--spatial-scale-step", type=float, default=0.002)
    parser.add_argument("--maximum-gap", type=int, default=30)
    parser.add_argument(
        "--native-cpu-threads",
        type=int,
        default=8,
        help="CPU worker count for exact native raster/topology batches",
    )
    parser.add_argument(
        "--native-batch-cases",
        type=int,
        default=4096,
        help="maximum exact frame cases per CPU-native batch",
    )
    parser.add_argument("--no-pair-vote", action="store_true")
    parser.add_argument("--pair-vote-sweeps", type=int, default=2)
    parser.add_argument("--pair-vote-global-steps", type=int, default=32)
    parser.add_argument("--pair-vote-local-steps", type=int, default=8)
    parser.add_argument(
        "--curve-low-iou-weight",
        type=float,
        default=4.0,
        help="quadratic low-IoU loss used only by the curve DP/pair-vote",
    )
    parser.add_argument("--single-state-curve", action="store_true")
    parser.add_argument(
        "--curve-state-scales",
        type=_float_list,
        default=(1.0, 1.005, 1.015, 1.025, 1.035),
    )
    parser.add_argument("--no-curve-point-refine", action="store_true")
    parser.add_argument("--curve-point-refine-sweeps", type=int, default=2)
    parser.add_argument("--no-curve-quality-rescue", action="store_true")
    parser.add_argument("--curve-rescue-iou-floor", type=float, default=0.92)
    parser.add_argument("--curve-rescue-regret-floor", type=float, default=0.04)
    parser.add_argument("--curve-rescue-area-ratio-cap", type=float, default=1.20)
    parser.add_argument(
        "--curve-rescue-maximum-keys",
        type=int,
        default=12,
        help="quality-first default is 12; use 8 for the balanced Pareto profile",
    )
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--review-point-count", type=int)
    parser.add_argument("--review-target-interval", type=int)
    parser.add_argument("--review-maximum-frames", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output} (use --force)")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    track = load_track_contours(
        args.input_sqlite,
        args.track_id,
        component_index=args.component_index,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        require_contiguous=True,
    )
    summaries: list[dict[str, object]] = []
    outputs: dict[tuple[int, int], tuple[KeyframeDpResult, KeyframeDpResult]] = {}
    spatial_summaries: dict[str, object] = {}
    for point_count in args.point_counts:
        spatial_started = time.perf_counter()
        curve_fit = fit_sequence(
            list(track.contours),
            FitConfig(
                control_point_count=int(point_count),
                dense_contour_samples=int(args.dense_samples),
                samples_per_segment=int(args.samples_per_segment),
                recall_floor=float(args.recall_floor),
                index_refine_passes=int(args.refine_passes),
                index_refine_radius=int(args.refine_radius),
                proxy_max_frames=int(args.proxy_frames),
                scale_maximum=float(args.spatial_scale_maximum),
                scale_step=float(args.spatial_scale_step),
                native_cpu_threads=int(args.native_cpu_threads),
            ),
        )
        polygon_controls, polygon_build = build_production_polygon_controls(
            list(track.contours),
            int(point_count),
            recall_floor=float(args.recall_floor),
            iou_floor=0.95,
        )
        polygon_repair = repair_spatial_controls(
            list(track.contours),
            polygon_controls,
            polygon_renderer,
            recall_floor=float(args.recall_floor),
            maximum_scale=float(args.spatial_scale_maximum),
            scale_step=float(args.spatial_scale_step),
        )
        spatial_summaries[str(point_count)] = {
            "curve": curve_fit.summary(),
            "polygon_builder": polygon_build,
            "polygon_after_recall_repair": {
                "minimum_recall": polygon_repair.minimum_recall,
                "mean_iou": polygon_repair.mean_iou,
                "recall_violations": polygon_repair.recall_violations,
                "repaired_frames": polygon_repair.repaired_frames,
                "unresolved_frames": list(polygon_repair.unresolved_frames),
            },
            "shared_spatial_wall_seconds": float(time.perf_counter() - spatial_started),
        }
        for interval in args.target_intervals:
            summary, polygon, curve = _run_one(
                output,
                track,
                int(point_count),
                int(interval),
                polygon_repair.controls,
                curve_fit.controls,
                int(args.samples_per_segment),
                args,
            )
            summaries.append(summary)
            outputs[(int(point_count), int(interval))] = (polygon, curve)
    _write_comparison(output / "comparison.csv", summaries)
    review_manifest = None
    review_video = None
    if args.source_video is not None:
        review_points = int(args.review_point_count or args.point_counts[0])
        review_interval = int(args.review_target_interval or args.target_intervals[-1])
        polygon, curve = outputs[(review_points, review_interval)]
        review_manifest = render_review_gallery(
            args.source_video,
            track.frames,
            list(track.contours),
            polygon,
            curve,
            output / "review_gallery",
            maximum_frames=int(args.review_maximum_frames),
        )
        review_video = render_review_video(
            args.source_video,
            track.frames,
            list(track.contours),
            polygon,
            curve,
            output / "review_sequence.mp4",
        )
    manifest = {
        "schema_version": 1,
        "status": "experimental_not_production",
        "algorithm": ("closed_uniform_catmull_rom_tension_1_bezier_factor_1_over_6"),
        "input_sqlite": str(args.input_sqlite.resolve()),
        "source_video": (
            str(args.source_video.resolve()) if args.source_video is not None else None
        ),
        "track_id": track.track_id,
        "label": track.label,
        "component_index": track.component_index,
        "frames": int(len(track.frames)),
        "actual_frame_range": [int(track.frames[0]), int(track.frames[-1])],
        "point_counts": list(args.point_counts),
        "target_intervals": list(args.target_intervals),
        "recall_floor": float(args.recall_floor),
        "spatial": spatial_summaries,
        "comparisons": summaries,
        "review_manifest": review_manifest,
        "review_video": review_video,
        "review_sqlite_contract": (
            "dense sampled curves are stored as polygons solely for existing review tools; "
            "curve_keyframes.json contains authoritative P and derived B handles"
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
