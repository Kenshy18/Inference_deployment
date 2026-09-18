"""Exact CPU search for higher-quality Catmull--Rom DP state palettes.

This module is intentionally experimental.  It imports the frozen Production
fit/DP implementation, retains its exact Recall/topology/area guards, and only
changes the per-frame state palette and low-tail loss weight.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from production.curve.config import CURVE_PRODUCTION
from production.curve.engine import _fit_component
from production.curve.runtime.keyframe_dp import KeyframeDpConfig, catmull_rom_renderer
from production.curve.runtime.multistate_dp import (
    CurvePointRefineConfig,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)

from .benchmark_palettes import _clamped_endpoint_fit
from .io import load_track_contours


def _case(text: str) -> tuple[str, Path, str, int, int, int]:
    try:
        name, path, track, start, frames, points = text.split("|", 5)
        return name, Path(path), track, int(start), int(frames), int(points)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be NAME|SQLITE|TRACK|START|FRAMES|POINTS"
        ) from error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _similarity_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align one phase-consistent candidate to the reference pose."""

    target = np.asarray(reference, dtype=np.float64)
    source = np.asarray(candidate, dtype=np.float64)
    target_center = np.mean(target, axis=0, keepdims=True)
    source_center = np.mean(source, axis=0, keepdims=True)
    left = source - source_center
    right = target - target_center
    covariance = left.T @ right
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    denominator = float(np.sum(left * left))
    scale = float(np.sum(singular) / denominator) if denominator > 1e-12 else 1.0
    scale = float(np.clip(scale, 0.80, 1.25))
    return np.ascontiguousarray(
        scale * (left @ rotation) + target_center,
        dtype=np.float64,
    )


def _pose_consensus_states(
    controls: np.ndarray,
    radii: tuple[int, ...] = (1, 2),
) -> tuple[list[np.ndarray], list[str]]:
    """Polygon-imported temporal consensus without averaging rigid motion."""

    source = np.asarray(controls, dtype=np.float64)
    values: list[np.ndarray] = []
    labels: list[str] = []
    for radius in radii:
        output = source.copy()
        for frame in range(len(source)):
            start = max(0, frame - int(radius))
            end = min(len(source), frame + int(radius) + 1)
            aligned = np.asarray(
                [
                    _similarity_align(source[frame], source[index])
                    for index in range(start, end)
                ],
                dtype=np.float64,
            )
            output[frame] = np.median(aligned, axis=0)
        values.append(np.ascontiguousarray(output))
        labels.append(f"pose_median_r{radius}")
    return values, labels


def _endpoint_states(
    controls: np.ndarray,
    horizon: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Polygon-imported forward/backward endpoint fits for asymmetric edges."""

    source = np.asarray(controls, dtype=np.float64)
    forward = source.copy()
    backward = source.copy()
    for frame in range(len(source)):
        right = min(len(source) - 1, frame + int(horizon))
        if right - frame >= 2:
            forward[frame] = _clamped_endpoint_fit(
                source, frame, right, 0, 0.20
            )
        left = max(0, frame - int(horizon))
        if frame - left >= 2:
            backward[frame] = _clamped_endpoint_fit(
                source, left, frame, 1, 0.20
            )
    return [forward, backward], [f"forward_ls_h{horizon}", f"backward_ls_h{horizon}"]


def _curve_normal_residual_state(
    references: list[np.ndarray],
    controls: np.ndarray,
    blend: float,
) -> np.ndarray:
    """Move editable P only along its local curve normal toward raw evidence.

    Catmull--Rom handles remain derived.  The correction is circularly
    smoothed and locally clipped, so this state adds shape freedom without an
    isotropic large-mask candidate.
    """

    source = np.asarray(controls, dtype=np.float64)
    output = source.copy()
    for frame, raw in enumerate(references):
        points = source[frame]
        reference = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        tangent = following - previous
        normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-12)
        delta = reference[None, :, :] - points[:, None, :]
        nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        signed = np.sum(
            (reference[nearest] - points) * normals,
            axis=1,
        )
        chord = 0.5 * (
            np.linalg.norm(points - previous, axis=1)
            + np.linalg.norm(following - points, axis=1)
        )
        signed = np.clip(signed, -0.25 * chord, 0.25 * chord)
        signed = (
            np.roll(signed, 1) + 2.0 * signed + np.roll(signed, -1)
        ) / 4.0
        output[frame] = points + float(blend) * signed[:, None] * normals
    return np.ascontiguousarray(output)


def build_advanced_palettes(
    references: list[np.ndarray],
    controls: np.ndarray,
    target_interval: int,
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """Return ablation palettes; every palette retains the Production base."""

    source = np.asarray(controls, dtype=np.float64)
    scales, scale_labels = isotropic_curve_states(source, (1.0, 1.005, 1.025, 1.035))
    endpoints, endpoint_labels = _endpoint_states(source, int(target_interval))
    short_horizon = max(2, int(round(float(target_interval) / 2.0)))
    short_endpoints, short_endpoint_labels = _endpoint_states(source, short_horizon)
    consensus, consensus_labels = _pose_consensus_states(source)
    normal_half = _curve_normal_residual_state(references, source, 0.50)
    normal_full = _curve_normal_residual_state(references, source, 1.00)

    def stack(values: list[np.ndarray], labels: list[str]):
        return np.ascontiguousarray(np.stack(values, axis=1)), tuple(labels)

    return {
        "production4": (scales, tuple(scale_labels)),
        # Fair superset ablation: retain every Production state and add only
        # the two asymmetric endpoint fits.  Unlike polygon_endpoint4 this
        # cannot lose a feasible Production edge merely because an isotropic
        # state was replaced.
        "production_endpoint6": stack(
            [
                scales[:, 0],
                scales[:, 1],
                scales[:, 2],
                scales[:, 3],
                endpoints[0],
                endpoints[1],
            ],
            [*scale_labels, *endpoint_labels],
        ),
        "polygon_endpoint4": stack(
            [scales[:, 0], scales[:, 1], *endpoints],
            [scale_labels[0], scale_labels[1], *endpoint_labels],
        ),
        "polygon_endpoint7": stack(
            [
                scales[:, 0],
                scales[:, 1],
                short_endpoints[0],
                short_endpoints[1],
                endpoints[0],
                endpoints[1],
                0.5 * (endpoints[0] + endpoints[1]),
            ],
            [
                scale_labels[0],
                scale_labels[1],
                short_endpoint_labels[0],
                short_endpoint_labels[1],
                endpoint_labels[0],
                endpoint_labels[1],
                f"bidirectional_ls_h{target_interval}",
            ],
        ),
        "polygon_consensus4": stack(
            [scales[:, 0], scales[:, 1], *consensus],
            [scale_labels[0], scale_labels[1], *consensus_labels],
        ),
        "curve_normal4": stack(
            [scales[:, 0], scales[:, 1], normal_half, normal_full],
            [scale_labels[0], scale_labels[1], "normal_residual_050", "normal_residual_100"],
        ),
        "diverse7": stack(
            [
                scales[:, 0],
                scales[:, 1],
                endpoints[0],
                endpoints[1],
                consensus[1],
                normal_half,
                normal_full,
            ],
            [
                scale_labels[0],
                scale_labels[1],
                endpoint_labels[0],
                endpoint_labels[1],
                consensus_labels[1],
                "normal_residual_050",
                "normal_residual_100",
            ],
        ),
    }


def _dp_config(config, low_iou_weight: float) -> KeyframeDpConfig:
    return KeyframeDpConfig(
        target_interval=int(config.target_interval),
        recall_floor=float(config.recall_floor),
        maximum_gap=max(int(config.maximum_gap), 4 * int(config.target_interval)),
        pair_vote_enabled=True,
        pair_vote_sweeps=int(config.pair_vote_sweeps),
        low_iou_quadratic_weight=float(low_iou_weight),
        quality_rescue_enabled=True,
        quality_rescue_iou_floor=float(config.quality_rescue_iou_floor),
        quality_rescue_regret_floor=float(config.quality_rescue_regret_floor),
        quality_rescue_area_ratio_cap=float(config.quality_rescue_area_ratio_cap),
        quality_rescue_maximum_extra_keys=int(config.quality_rescue_maximum_extra_keys),
        quality_rescue_density_budget=bool(config.quality_rescue_density_budget),
        quality_rescue_maximum_iou_regression=float(
            config.quality_rescue_maximum_iou_regression
        ),
        quality_rescue_maximum_area_ratio_regression=float(
            config.quality_rescue_maximum_area_ratio_regression
        ),
        native_cpu_threads=int(config.native_cpu_threads),
        native_batch_cases=int(config.native_batch_cases),
        native_reference_cache_bytes=int(config.native_reference_cache_bytes),
    )


def _run_palette(
    references: list[np.ndarray],
    base: np.ndarray,
    states: np.ndarray,
    labels: tuple[str, ...],
    config,
    low_iou_weight: float,
):
    renderer = catmull_rom_renderer(int(config.samples_per_segment))
    return optimize_multistate_keyframes(
        references,
        states,
        state_labels=labels,
        base_controls=base,
        representation=f"advanced_palette:{len(labels)}",
        renderer=renderer,
        config=_dp_config(config, low_iou_weight),
        point_refine=CurvePointRefineConfig(
            enabled=True,
            sweeps=int(config.point_refine_sweeps),
            scheduler=str(config.point_refine_scheduler),
        ),
        interval_renderer=renderer,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_case, required=True)
    parser.add_argument("--target-interval", type=int, default=6)
    parser.add_argument("--native-cpu-threads", type=int, default=6)
    parser.add_argument("--palettes", default="production4,polygon_endpoint4,polygon_endpoint7,polygon_consensus4,curve_normal4,diverse7")
    parser.add_argument("--tail-weights", default="4,8")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        raise SystemExit("advanced palette search is an exact CPU experiment")
    palette_names = tuple(value.strip() for value in args.palettes.split(",") if value.strip())
    tail_weights = tuple(float(value) for value in args.tail_weights.split(","))
    config = replace(
        CURVE_PRODUCTION,
        target_interval=int(args.target_interval),
        native_cpu_threads=int(args.native_cpu_threads),
    )
    config.validate()
    started = time.perf_counter()
    cases: list[dict[str, object]] = []
    for case_name, path, track_id, start, frame_count, point_count in args.case:
        track = load_track_contours(
            path,
            track_id,
            start_frame=int(start),
            max_frames=int(frame_count),
            require_contiguous=True,
        )
        references = [np.asarray(value, dtype=np.float64) for value in track.contours]
        fit_started = time.perf_counter()
        controls, fit = _fit_component(references, int(point_count), config)
        fit_seconds = float(time.perf_counter() - fit_started)
        palettes = build_advanced_palettes(references, controls, int(args.target_interval))
        rows: list[dict[str, object]] = []
        for palette_name in palette_names:
            if palette_name not in palettes:
                raise KeyError(f"unknown palette: {palette_name}")
            states, labels = palettes[palette_name]
            for weight in tail_weights:
                run_started = time.perf_counter()
                result = _run_palette(
                    references, controls, states, labels, config, float(weight)
                )
                row = {
                    "palette": str(palette_name),
                    "low_iou_quadratic_weight": float(weight),
                    "state_count": int(states.shape[1]),
                    "state_labels": list(labels),
                    "state_usage": dict(Counter(result.chosen_state_labels)),
                    "wall_seconds": float(time.perf_counter() - run_started),
                    **result.summary(),
                }
                rows.append(row)
                # One bounded alternating-optimization experiment: feed the
                # refined first path back as an additional state and rerun DP.
                # The original states remain available, so this measures the
                # value of a second DP rather than replacing the first result.
                if palette_name == "polygon_endpoint4" and float(weight) == 4.0:
                    redp_states = np.ascontiguousarray(
                        np.concatenate(
                            (states, np.asarray(result.dense_controls)[:, None]),
                            axis=1,
                        )
                    )
                    redp_labels = (*labels, "first_pass_refined_path")
                    redp_started = time.perf_counter()
                    redp = _run_palette(
                        references,
                        controls,
                        redp_states,
                        redp_labels,
                        config,
                        float(weight),
                    )
                    rows.append(
                        {
                            "palette": "polygon_endpoint4_redp",
                            "low_iou_quadratic_weight": float(weight),
                            "state_count": int(redp_states.shape[1]),
                            "state_labels": list(redp_labels),
                            "state_usage": dict(Counter(redp.chosen_state_labels)),
                            "wall_seconds": float(
                                time.perf_counter() - redp_started
                            ),
                            **redp.summary(),
                        }
                    )
        baseline = next(
            row
            for row in rows
            if row["palette"] == "production4"
            and float(row["low_iou_quadratic_weight"]) == 4.0
        )
        for row in rows:
            row["versus_baseline"] = {
                key: float(row[key]) - float(baseline[key])
                for key in (
                    "effective_interval",
                    "mean_iou",
                    "q05_iou",
                    "q01_iou",
                    "minimum_iou",
                    "area_ratio_maximum",
                    "elapsed_seconds",
                )
            }
            row["versus_baseline"]["chosen_keyframes"] = int(
                row["chosen_keyframes"]
            ) - int(baseline["chosen_keyframes"])
        cases.append(
            {
                "case": str(case_name),
                "source": str(path.resolve()),
                "track_id": str(track_id),
                "start_frame": int(track.frames[0]),
                "frames": int(len(track.frames)),
                "points": int(point_count),
                "fit_seconds": fit_seconds,
                "fit": fit,
                "results": rows,
            }
        )
        _write_json(
            args.output_json,
            {
                "schema_version": 1,
                "status": "running",
                "config": asdict(config),
                "cases": cases,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
        )
    _write_json(
        args.output_json,
        {
            "schema_version": 1,
            "status": "complete",
            "config": asdict(config),
            "cases": cases,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
