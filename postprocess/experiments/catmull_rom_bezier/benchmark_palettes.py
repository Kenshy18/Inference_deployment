"""CPU-only ablation of Catmull--Rom DP state palettes on real tracks."""

from __future__ import annotations

import argparse
import hashlib
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
from production.curve.runtime.role_states import polygon_role_curve_states

from .io import load_track_contours


def _palette(text: str) -> tuple[str, tuple[float, ...]]:
    try:
        name, raw_values = text.split("=", 1)
        values = tuple(float(value) for value in raw_values.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "palette must be NAME=scale,scale,..."
        ) from error
    if not name or len(values) < 2 or values[:2] != (1.0, 1.005):
        raise argparse.ArgumentTypeError(
            "palette must begin with the Production fast prefix 1.0,1.005"
        )
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("palette scales must be sorted and unique")
    return name, values


def _case(text: str) -> tuple[str, Path, str, int, int, int]:
    try:
        name, path, track, start, frames, points = text.split("|", 5)
        return name, Path(path), track, int(start), int(frames), int(points)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be NAME|SQLITE|TRACK|START|FRAMES|POINTS"
        ) from error


def _roles(text: str) -> tuple[str, tuple[str, ...]]:
    try:
        name, raw_values = text.split("=", 1)
        values = tuple(
            value.strip() for value in raw_values.split(",") if value.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "role set must be NAME=ROLE,ROLE,..."
        ) from error
    if not name or not values:
        raise argparse.ArgumentTypeError("role set must contain a name and roles")
    return name, values


def _hash_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clamped_endpoint_fit(
    controls: np.ndarray,
    start: int,
    end: int,
    endpoint: int,
    maximum_correction_fraction: float,
) -> np.ndarray:
    """Fit one linear-model endpoint and bound its per-key displacement."""

    values = np.asarray(controls[start : end + 1], dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    design = np.column_stack((1.0 - alpha, alpha))
    gram = design.T @ design + 1e-8 * np.eye(2, dtype=np.float64)
    fitted = np.linalg.solve(gram, design.T @ values.reshape(len(values), -1))
    candidate = fitted[int(endpoint)].reshape(values.shape[1:])
    reference = controls[start if endpoint == 0 else end]
    centered = reference - np.mean(reference, axis=0, keepdims=True)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    maximum = max(1.0, float(maximum_correction_fraction) * radius)
    delta = candidate - reference
    rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    if rms > maximum:
        candidate = reference + (maximum / max(rms, 1e-12)) * delta
    return np.asarray(candidate, dtype=np.float64)


def _temporal_endpoint_states(
    controls: np.ndarray,
    *,
    horizon: int,
    include_scale: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build base/scale plus forward/backward linear-residual endpoints."""

    source = np.asarray(controls, dtype=np.float64)
    forward = source.copy()
    backward = source.copy()
    for frame in range(len(source)):
        right = min(len(source) - 1, frame + int(horizon))
        if right - frame >= 2:
            forward[frame] = _clamped_endpoint_fit(
                source,
                frame,
                right,
                0,
                0.20,
            )
        left = max(0, frame - int(horizon))
        if frame - left >= 2:
            backward[frame] = _clamped_endpoint_fit(
                source,
                left,
                frame,
                1,
                0.20,
            )
    fast, fast_labels = isotropic_curve_states(source, (1.0, 1.005))
    values = [fast[:, 0], fast[:, 1], forward, backward]
    labels = [*fast_labels, f"forward_ls_h{horizon}", f"backward_ls_h{horizon}"]
    if include_scale:
        scale, scale_labels = isotropic_curve_states(source, (1.035,))
        values.append(scale[:, 0])
        labels.append(scale_labels[0])
    return np.ascontiguousarray(np.stack(values, axis=1)), tuple(labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_case, required=True)
    parser.add_argument("--palette", action="append", type=_palette, required=True)
    parser.add_argument("--target-interval", type=int, default=6)
    parser.add_argument("--native-cpu-threads", type=int, default=4)
    parser.add_argument(
        "--point-refine-sweeps",
        type=int,
        default=CURVE_PRODUCTION.point_refine_sweeps,
        help="override only the CPU point-refinement sweep count",
    )
    parser.add_argument(
        "--pair-vote-sweeps",
        type=int,
        default=CURVE_PRODUCTION.pair_vote_sweeps,
        help="override only the CPU pair-vote sweep count",
    )
    parser.add_argument(
        "--temporal-endpoints",
        action="store_true",
        help="also test 4/5-state forward/backward LS endpoint candidates",
    )
    parser.add_argument("--no-quality-rescue", action="store_true")
    parser.add_argument("--no-fast-quality-probe", action="store_true")
    parser.add_argument(
        "--polygon-role-palette",
        action="store_true",
        help="also test the promoted polygon role palette projected to Catmull P",
    )
    parser.add_argument("--curve-role-palette", action="append", type=_roles)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        raise SystemExit("palette benchmark must run with CUDA hidden")
    config = replace(
        CURVE_PRODUCTION,
        target_interval=int(args.target_interval),
        native_cpu_threads=int(args.native_cpu_threads),
        point_refine_sweeps=int(args.point_refine_sweeps),
        pair_vote_sweeps=int(args.pair_vote_sweeps),
    )
    config.validate()
    renderer = catmull_rom_renderer(int(config.samples_per_segment))
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    for case_name, source, track_id, start, frame_count, point_count in args.case:
        track = load_track_contours(
            source,
            track_id,
            start_frame=int(start),
            max_frames=int(frame_count),
            require_contiguous=True,
        )
        references = [np.asarray(value, dtype=np.float64) for value in track.contours]
        fit_started = time.perf_counter()
        controls, fit = _fit_component(references, int(point_count), config)
        fit_seconds = time.perf_counter() - fit_started
        case_rows = []
        candidates: list[
            tuple[str, tuple[float, ...] | None, np.ndarray, tuple[str, ...]]
        ] = []
        for palette_name, scales in args.palette:
            fallback_controls, fallback_labels = isotropic_curve_states(
                controls, tuple(scales)
            )
            candidates.append(
                (str(palette_name), tuple(scales), fallback_controls, fallback_labels)
            )
        if args.temporal_endpoints:
            for include_scale in (False, True):
                values, labels = _temporal_endpoint_states(
                    controls,
                    horizon=int(config.target_interval),
                    include_scale=include_scale,
                )
                candidates.append(
                    (
                        "temporal5" if include_scale else "temporal4",
                        None,
                        values,
                        labels,
                    )
                )
        requested_role_sets = list(args.curve_role_palette or ())
        if args.polygon_role_palette:
            requested_role_sets.append(
                (
                    "polygon_roles",
                    (
                        "C02_125",
                        "G02",
                        "G04",
                        "A06",
                        "F3_P1",
                        "D6_P1",
                        "VF8_P1",
                    ),
                )
            )
        if requested_role_sets:
            union_roles = tuple(
                dict.fromkeys(
                    role for _name, roles in requested_role_sets for role in roles
                )
            )
            all_values, all_labels = polygon_role_curve_states(
                controls,
                track.frames,
                union_roles,
                renderer=renderer,
                samples_per_segment=int(config.samples_per_segment),
            )
            label_to_state = {label: index for index, label in enumerate(all_labels)}
            for name, roles in requested_role_sets:
                indices = (0, *(label_to_state[role] for role in roles))
                candidates.append(
                    (
                        str(name),
                        None,
                        np.ascontiguousarray(all_values[:, indices]),
                        ("raw", *roles),
                    )
                )
        for palette_name, scales, fallback_controls, fallback_labels in candidates:
            fast_controls = np.ascontiguousarray(fallback_controls[:, :2])
            fast_labels = tuple(fallback_labels[:2])
            dp_config = KeyframeDpConfig(
                target_interval=int(config.target_interval),
                recall_floor=float(config.recall_floor),
                maximum_gap=max(
                    int(config.maximum_gap),
                    4 * int(config.target_interval),
                ),
                pair_vote_enabled=True,
                pair_vote_sweeps=int(config.pair_vote_sweeps),
                low_iou_quadratic_weight=float(config.low_iou_quadratic_weight),
                path_selection_mode=str(config.path_selection_mode),
                cardinality_maximum_factor=float(config.cardinality_maximum_factor),
                shape_distance_weight=float(config.shape_distance_weight),
                quality_rescue_enabled=not bool(args.no_quality_rescue),
                quality_rescue_iou_floor=float(config.quality_rescue_iou_floor),
                quality_rescue_regret_floor=float(config.quality_rescue_regret_floor),
                quality_rescue_area_ratio_cap=float(
                    config.quality_rescue_area_ratio_cap
                ),
                quality_rescue_maximum_extra_keys=int(
                    config.quality_rescue_maximum_extra_keys
                ),
                quality_rescue_density_budget=bool(
                    config.quality_rescue_density_budget
                ),
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
            result = optimize_multistate_keyframes(
                references,
                fast_controls,
                state_labels=fast_labels,
                base_controls=controls,
                representation=f"palette_ablation:{palette_name}",
                renderer=renderer,
                config=dp_config,
                point_refine=CurvePointRefineConfig(
                    enabled=True,
                    sweeps=int(config.point_refine_sweeps),
                    scheduler=str(config.point_refine_scheduler),
                ),
                interval_renderer=renderer,
                fallback_state_controls=fallback_controls,
                fallback_state_labels=fallback_labels,
                fast_state_target_ratio=float(config.fast_state_target_ratio),
                fast_state_quality_probe=not bool(args.no_fast_quality_probe),
            )
            summary = result.summary()
            case_rows.append(
                {
                    "palette": str(palette_name),
                    "scales": None if scales is None else list(scales),
                    **summary,
                    "state_usage": dict(Counter(result.chosen_state_labels)),
                    "state_search_initial_count": int(
                        result.state_search_initial_count
                    ),
                    "state_search_final_count": int(result.state_search_final_count),
                    "state_search_fallback": bool(result.state_search_fallback),
                    "output_sha256": _hash_arrays(
                        np.asarray(result.chosen_indices, dtype=np.int64),
                        result.keyframe_controls,
                        result.dense_boundaries,
                    ),
                }
            )
        baseline = case_rows[0]
        for row in case_rows:
            row["versus_baseline"] = {
                "chosen_keyframes": int(row["chosen_keyframes"])
                - int(baseline["chosen_keyframes"]),
                "effective_interval": float(row["effective_interval"])
                - float(baseline["effective_interval"]),
                "mean_iou": float(row["mean_iou"]) - float(baseline["mean_iou"]),
                "minimum_iou": float(row["minimum_iou"])
                - float(baseline["minimum_iou"]),
                "q01_iou": float(row["q01_iou"]) - float(baseline["q01_iou"]),
                "area_ratio_maximum": float(row["area_ratio_maximum"])
                - float(baseline["area_ratio_maximum"]),
                "elapsed_seconds": float(row["elapsed_seconds"])
                - float(baseline["elapsed_seconds"]),
            }
        results.append(
            {
                "case": str(case_name),
                "source": str(Path(source).resolve()),
                "track_id": str(track_id),
                "start_frame": int(track.frames[0]),
                "frames": int(len(track.frames)),
                "points": int(point_count),
                "fit_seconds": float(fit_seconds),
                "fit": fit,
                "palettes": case_rows,
            }
        )
    payload = {
        "schema_version": 1,
        "cpu_only": True,
        "cuda_used": False,
        "target_interval": int(config.target_interval),
        "config": asdict(config),
        "elapsed_seconds": float(time.perf_counter() - started),
        "cases": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
