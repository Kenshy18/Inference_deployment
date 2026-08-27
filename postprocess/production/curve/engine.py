"""Bounded-memory CPU engine for editable closed Catmull--Rom masks."""

from __future__ import annotations

import csv
import json
import os
import resource
import sqlite3
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from classwise.sqlite import read_track_labels
from production.polygon.runtime.kernel.stream import iter_track_streams_from_sqlite

from .config import CurveProductionConfig
from .runtime.fitter import FitConfig, fit_sequence
from .runtime.keyframe_dp import (
    KeyframeDpConfig,
    _frame_metrics,
    audit_dense_path,
    catmull_rom_renderer,
)
from .runtime.model import sample_curve_sequence
from .runtime.multistate_dp import (
    CurvePointRefineConfig,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)
from .runtime.role_states import curve_role_ids, polygon_role_curve_states
from .runtime.spatial import (
    complete_spatial_recall_with_envelopes,
    repair_spatial_controls,
)
from .runtime.topology import has_strict_self_intersection
from .storage import MaskSqliteWriter


ProgressCallback = Callable[[str, float | None, float | None], None]
_SLOW_STREAM_SUMMARY_LIMIT = 32


def _compact_fit_summary(value: dict[str, object]) -> dict[str, object]:
    """Remove frame-linear diagnostics before persisting a stream audit row."""

    output = dict(value)
    raw_fit = output.get("fit")
    if not isinstance(raw_fit, dict):
        return output
    fit = dict(raw_fit)
    raw_shifts = fit.pop("temporal_phase_shifts", ())
    if isinstance(raw_shifts, (list, tuple)):
        shifts = tuple(int(item) for item in raw_shifts)
        fit["temporal_phase_shift_summary"] = {
            "frames": int(len(shifts)),
            "nonzero_frames": int(sum(value != 0 for value in shifts)),
            "maximum_absolute_shift": int(
                max((abs(value) for value in shifts), default=0)
            ),
        }
    output["fit"] = fit
    return output


def _point_count(vertex_policy: dict[str, object], track_id: str) -> int:
    tracks = vertex_policy.get("tracks")
    if not isinstance(tracks, dict) or str(track_id) not in tracks:
        raise RuntimeError(f"curve point policy is missing track {track_id}")
    value = tracks[str(track_id)]
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid curve point policy for track {track_id}")
    count = int(value["vertices_per_component"])
    if count not in {14, 16, 18, 20}:
        raise RuntimeError(f"unsupported Production curve point count: {count}")
    return count


def _phase_align(
    controls: np.ndarray,
    frames: np.ndarray,
    history: dict[int, np.ndarray],
) -> tuple[np.ndarray, int]:
    """Choose one cyclic P phase for a chunk without changing its geometry."""
    source = np.asarray(controls, dtype=np.float64)
    if not history or source.shape[1] <= 1:
        return np.ascontiguousarray(source), 0
    pairs = [
        (index, history[int(frame)])
        for index, frame in enumerate(frames)
        if int(frame) in history
    ]
    if not pairs:
        prior = [
            (frame, value) for frame, value in history.items() if frame < frames[0]
        ]
        if prior:
            _frame, value = max(prior, key=lambda item: item[0])
            pairs = [(0, value)]
    if not pairs:
        return np.ascontiguousarray(source), 0
    best_shift = 0
    best_cost = float("inf")
    for shift in range(source.shape[1]):
        costs = []
        for index, reference in pairs:
            candidate = np.roll(source[index], shift, axis=0)
            left = candidate - np.mean(candidate, axis=0, keepdims=True)
            right = reference - np.mean(reference, axis=0, keepdims=True)
            costs.append(float(np.mean(np.sum((left - right) ** 2, axis=1))))
        cost = float(np.mean(costs))
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_shift = int(shift)
    return np.ascontiguousarray(np.roll(source, best_shift, axis=1)), best_shift


def _fit_config(
    point_count: int,
    config: CurveProductionConfig,
) -> FitConfig:
    """Build the one frozen spatial-fit contract used by every curve path."""

    return FitConfig(
        control_point_count=int(point_count),
        dense_contour_samples=int(config.dense_contour_samples),
        samples_per_segment=int(config.samples_per_segment),
        recall_floor=float(config.recall_floor),
        scale_maximum=float(config.spatial_scale_maximum),
        scale_step=float(config.spatial_scale_step),
        native_cpu_threads=int(config.native_cpu_threads),
        native_batch_cases=int(config.native_batch_cases),
        native_reference_cache_bytes=int(config.native_reference_cache_bytes),
    )


def _independent_local_refits(
    references: list[np.ndarray],
    controls: np.ndarray,
    frame_indices: tuple[int, ...],
    *,
    config: CurveProductionConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Tightly refit rare spatial outliers without adding editable points.

    Persistent contour locations are normally preferable because they retain
    point identity across the whole track.  On a rare abrupt, highly concave
    frame, however, that shared placement can require a very large isotropic
    expansion merely to recover Recall.  Fit only that source contour with
    the same P count, cyclically align its point phase to the persistent
    controls, and choose the highest-IoU exact-valid blend.  The downstream
    DP still decides whether this spatial candidate needs a key and can add
    unchanged neighbouring support keys to localize the transition.
    """

    source = np.asarray(controls, dtype=np.float64)
    output = source.copy()
    renderer = catmull_rom_renderer(int(config.samples_per_segment))
    attempted = tuple(sorted(set(int(value) for value in frame_indices)))
    accepted: list[int] = []
    rejected: list[int] = []
    selected_alpha: dict[str, float] = {}
    phase_shift: dict[str, int] = {}
    maximum_control_shift = 0.0
    for frame in attempted:
        if frame < 0 or frame >= len(source):
            raise ValueError(f"local curve-refit frame is out of range: {frame}")
        reference = np.asarray(references[frame], dtype=np.float64)
        baseline = np.asarray(source[frame], dtype=np.float64)
        baseline_boundary = renderer(baseline)
        baseline_iou, baseline_recall, _precision, baseline_area = _frame_metrics(
            reference,
            baseline_boundary,
        )
        fitted = fit_sequence(
            [reference],
            _fit_config(int(source.shape[1]), config),
        )
        local = np.asarray(fitted.controls[0], dtype=np.float64)
        aligned, shift = _phase_align(
            local[None, ...],
            np.asarray((frame,), dtype=np.int64),
            {frame: baseline},
        )
        local = np.asarray(aligned[0], dtype=np.float64)
        phase_shift[str(frame)] = int(shift)
        best_controls: np.ndarray | None = None
        best_score: tuple[float, float, float] | None = None
        best_alpha = 0.0
        if baseline_recall + 1e-12 >= float(
            config.recall_floor
        ) and not has_strict_self_intersection(baseline_boundary):
            best_controls = baseline.copy()
            best_score = (
                float(baseline_iou),
                -float(baseline_area),
                0.0,
            )
        # Blending from the persistent controls preserves as much temporal
        # identity as possible.  Alpha=1 remains available when only the
        # independent fit can satisfy the exact Recall floor.
        for alpha in np.linspace(0.0, 1.0, 17, dtype=np.float64)[1:]:
            trial = (1.0 - float(alpha)) * baseline + float(alpha) * local
            boundary = renderer(trial)
            if has_strict_self_intersection(boundary):
                continue
            iou, recall, _precision, area_ratio = _frame_metrics(
                reference,
                boundary,
            )
            if recall + 1e-12 < float(config.recall_floor):
                continue
            score = (float(iou), -float(area_ratio), -float(alpha))
            if best_score is None or score > best_score:
                best_controls = trial.copy()
                best_score = score
                best_alpha = float(alpha)
        if best_controls is None or (
            baseline_recall + 1e-12 >= float(config.recall_floor)
            and best_score is not None
            and best_score[0] <= float(baseline_iou) + 1e-12
        ):
            rejected.append(frame)
            continue
        output[frame] = best_controls
        accepted.append(frame)
        selected_alpha[str(frame)] = float(best_alpha)
        maximum_control_shift = max(
            maximum_control_shift,
            float(
                np.max(
                    np.linalg.norm(
                        np.asarray(best_controls) - baseline,
                        axis=1,
                    )
                )
            ),
        )
    return np.ascontiguousarray(output), {
        "attempted_frames": list(attempted),
        "accepted_frames": list(accepted),
        "rejected_frames": list(rejected),
        "attempted_count": int(len(attempted)),
        "accepted_count": int(len(accepted)),
        "selected_alpha": selected_alpha,
        "cyclic_phase_shift": phase_shift,
        "maximum_control_shift_px": float(maximum_control_shift),
    }


def _repair_if_needed(
    references: list[np.ndarray],
    controls: np.ndarray,
    *,
    config: CurveProductionConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    renderer = catmull_rom_renderer(int(config.samples_per_segment))
    repair = repair_spatial_controls(
        references,
        controls,
        renderer,
        recall_floor=float(config.recall_floor),
        maximum_scale=float(config.emergency_scale_maximum),
        scale_step=float(config.spatial_scale_step),
    )
    if repair.recall_violations:
        # Completion is safer than aborting a multi-hour batch. This second,
        # very rare search is deliberately isolated and reported. It retains
        # P count and the exact Catmull--Rom model.
        repair = repair_spatial_controls(
            references,
            repair.controls,
            renderer,
            recall_floor=float(config.recall_floor),
            maximum_scale=2.0,
            scale_step=0.01,
        )
    repaired_controls = np.asarray(repair.controls, dtype=np.float64)
    _repaired_boundaries, repaired_audit = audit_dense_path(
        references,
        repaired_controls,
        renderer,
        threads=int(config.native_cpu_threads),
    )
    local_targets = tuple(
        int(value)
        for value in np.flatnonzero(
            np.logical_or.reduce(
                (
                    repaired_audit.recall + 1e-12 < float(config.recall_floor),
                    repaired_audit.iou + 1e-12 < float(config.quality_rescue_iou_floor),
                    repaired_audit.area_ratio
                    > float(config.quality_rescue_area_ratio_cap) + 1e-12,
                )
            )
        )
    )
    local_refit_summary: dict[str, object] = {
        "attempted_frames": [],
        "accepted_frames": [],
        "rejected_frames": [],
        "attempted_count": 0,
        "accepted_count": 0,
        "selected_alpha": {},
        "cyclic_phase_shift": {},
        "maximum_control_shift_px": 0.0,
    }
    if local_targets:
        repaired_controls, local_refit_summary = _independent_local_refits(
            references,
            repaired_controls,
            local_targets,
            config=config,
        )
        _local_boundaries, repaired_audit = audit_dense_path(
            references,
            repaired_controls,
            renderer,
            threads=int(config.native_cpu_threads),
        )

    completion_frames: tuple[int, ...] = ()
    maximum_completion_area_ratio = 1.0
    minimum_recall = float(np.min(repaired_audit.recall))
    mean_iou = float(np.mean(repaired_audit.iou))
    unresolved = tuple(
        int(value)
        for value in np.flatnonzero(
            repaired_audit.recall + 1e-12 < float(config.recall_floor)
        )
    )
    if unresolved:
        completion = complete_spatial_recall_with_envelopes(
            references,
            repaired_controls,
            renderer,
            unresolved_frames=unresolved,
            recall_floor=float(config.recall_floor),
        )
        repaired_controls = completion.controls
        completion_frames = completion.replaced_frames
        maximum_completion_area_ratio = float(completion.maximum_area_ratio)
        minimum_recall = float(completion.minimum_recall)
        mean_iou = float(completion.mean_iou)
    return repaired_controls, {
        "repaired_frames": int(repair.repaired_frames),
        "minimum_recall": minimum_recall,
        "mean_iou": mean_iou,
        "emergency_repair": True,
        "independent_local_refit": local_refit_summary,
        "completion_envelope_frames": list(completion_frames),
        "completion_envelope_count": int(len(completion_frames)),
        "maximum_completion_area_ratio": maximum_completion_area_ratio,
    }


def _fit_component(
    references: list[np.ndarray],
    point_count: int,
    config: CurveProductionConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    fitted = fit_sequence(
        references,
        _fit_config(int(point_count), config),
    )
    controls = fitted.controls
    repair_summary: dict[str, object] = {
        "emergency_repair": False,
        "repaired_frames": 0,
        "completion_envelope_frames": [],
        "completion_envelope_count": 0,
        "maximum_completion_area_ratio": 1.0,
        "independent_local_refit": {
            "attempted_frames": [],
            "accepted_frames": [],
            "rejected_frames": [],
            "attempted_count": 0,
            "accepted_count": 0,
            "selected_alpha": {},
            "cyclic_phase_shift": {},
            "maximum_control_shift_px": 0.0,
        },
    }
    if fitted.unresolved_recall_frames:
        controls, repair_summary = _repair_if_needed(
            references,
            controls,
            config=config,
        )
    return np.ascontiguousarray(controls), {
        "fit": fitted.summary(),
        "post_fit_repair": repair_summary,
    }


class _Audit:
    def __init__(self) -> None:
        self.frames = 0
        self.keys = 0
        self.component_observations = 0
        self.iou: list[np.ndarray] = []
        self.recall_min = 1.0
        self.recall_violations = 0
        self.topology_invalid = 0
        self.area_maximum = 0.0
        self.pair_trials = 0
        self.pair_accepted = 0
        self.pair_gain = 0.0
        self.point_trials = 0
        self.point_accepted = 0
        self.point_gain = 0.0
        self.dp_seconds = 0.0
        self.pair_vote_seconds = 0.0
        self.point_refine_seconds = 0.0
        self.final_audit_seconds = 0.0
        self.state_search_fallbacks = 0
        self.state_search_probe_seconds = 0.0
        self.lazy_topology_decodes = 0
        self.lazy_topology_checked_edges = 0
        self.lazy_topology_checked_frames = 0
        self.lazy_topology_rejected_edges = 0
        self.refinement_guard_triggers = 0
        self.refinement_guard_minimum_alpha = 1.0

    def add_component(
        self,
        component_audit,
        emit_start: int,
        emit_end: int,
        *,
        recall_floor: float,
    ) -> None:
        selection = slice(int(emit_start), int(emit_end))
        iou = np.asarray(component_audit.iou[selection], dtype=np.float64)
        recall = np.asarray(component_audit.recall[selection], dtype=np.float64)
        area = np.asarray(component_audit.area_ratio[selection], dtype=np.float64)
        topology = np.asarray(component_audit.topology_valid[selection], dtype=bool)
        self.component_observations += len(iou)
        self.iou.append(iou)
        self.recall_min = min(self.recall_min, float(np.min(recall)))
        self.recall_violations += int(
            np.count_nonzero(recall + 1e-12 < float(recall_floor))
        )
        self.topology_invalid += int(np.count_nonzero(~topology))
        self.area_maximum = max(self.area_maximum, float(np.max(area)))

    def add_result(self, result, emit_start: int, emit_end: int) -> None:
        self.add_component(
            result.audit,
            emit_start,
            emit_end,
            recall_floor=float(result.config.recall_floor),
        )
        self.pair_trials += int(result.pair_vote_trials)
        self.pair_accepted += int(result.pair_vote_accepted)
        self.pair_gain += float(result.pair_vote_iou_gain)
        self.point_trials += int(result.point_refine_trials)
        self.point_accepted += int(result.point_refine_accepted)
        self.point_gain += float(result.point_refine_iou_gain)
        self.dp_seconds += float(result.dp_seconds)
        self.pair_vote_seconds += float(result.pair_vote_seconds)
        self.point_refine_seconds += float(result.point_refine_seconds)
        self.final_audit_seconds += float(result.final_audit_seconds)
        self.state_search_fallbacks += int(bool(result.state_search_fallback))
        self.state_search_probe_seconds += float(result.state_search_probe_seconds)
        self.lazy_topology_decodes += int(result.native_lazy_topology_decodes)
        self.lazy_topology_checked_edges += int(
            result.native_lazy_topology_checked_edges
        )
        self.lazy_topology_checked_frames += int(
            result.native_lazy_topology_checked_frames
        )
        self.lazy_topology_rejected_edges += int(
            result.native_lazy_topology_rejected_edges
        )
        self.refinement_guard_triggers += int(bool(result.refinement_guard_triggered))
        self.refinement_guard_minimum_alpha = min(
            self.refinement_guard_minimum_alpha,
            float(result.refinement_guard_alpha),
        )

    def summary(self) -> dict[str, object]:
        values = np.concatenate(self.iou) if self.iou else np.ones((0,))
        return {
            "emitted_component_frames": int(self.frames),
            "component_observations": int(self.component_observations),
            "keyframes": int(self.keys),
            "effective_interval": float(self.frames / max(self.keys, 1)),
            "mean_iou": float(np.mean(values)) if len(values) else 1.0,
            "minimum_iou": float(np.min(values)) if len(values) else 1.0,
            "q01_iou": float(np.quantile(values, 0.01)) if len(values) else 1.0,
            "q05_iou": float(np.quantile(values, 0.05)) if len(values) else 1.0,
            "minimum_recall": float(self.recall_min),
            "recall_violations": int(self.recall_violations),
            "topology_invalid_frames": int(self.topology_invalid),
            "maximum_area_ratio": float(self.area_maximum),
            "pair_vote_trials": int(self.pair_trials),
            "pair_vote_accepted": int(self.pair_accepted),
            "pair_vote_iou_gain": float(self.pair_gain),
            "point_refine_trials": int(self.point_trials),
            "point_refine_accepted": int(self.point_accepted),
            "point_refine_iou_gain": float(self.point_gain),
            "dp_seconds": float(self.dp_seconds),
            "pair_vote_seconds": float(self.pair_vote_seconds),
            "point_refine_seconds": float(self.point_refine_seconds),
            "final_audit_seconds": float(self.final_audit_seconds),
            "state_search_fallbacks": int(self.state_search_fallbacks),
            "state_search_probe_seconds": float(self.state_search_probe_seconds),
            "lazy_topology_decodes": int(self.lazy_topology_decodes),
            "lazy_topology_checked_edges": int(self.lazy_topology_checked_edges),
            "lazy_topology_checked_frames": int(self.lazy_topology_checked_frames),
            "lazy_topology_rejected_edges": int(self.lazy_topology_rejected_edges),
            "refinement_guard_triggers": int(self.refinement_guard_triggers),
            "refinement_guard_minimum_alpha": float(
                self.refinement_guard_minimum_alpha
            ),
        }


def _append_passthrough(
    tracked: Path,
    labels: dict[str, str],
    passthrough_track_ids: tuple[str, ...],
    dense: MaskSqliteWriter,
    keys: MaskSqliteWriter,
) -> int:
    count = 0
    allowed = set(str(value) for value in passthrough_track_ids)
    if not allowed:
        return 0
    with sqlite3.connect(f"file:{Path(tracked).resolve()}?mode=ro", uri=True) as db:
        for frame, track_id, polygons, label in db.execute(
            "SELECT frame,track_id,polygons,COALESCE(label,'') FROM masks "
            "ORDER BY track_id,frame"
        ):
            resolved = labels.get(str(track_id), str(label))
            if str(track_id) not in allowed:
                continue
            for writer in (dense, keys):
                writer.append(
                    frame=int(frame),
                    track_id=str(track_id),
                    polygons=str(polygons),
                    label=resolved,
                )
            count += 1
    return count


def run_curve_optimizer(
    tracked_sqlite: Path,
    preparation: dict[str, object],
    output_root: Path,
    *,
    config: CurveProductionConfig,
    progress_callback: ProgressCallback | None = None,
    max_tracks: int = 0,
) -> dict[str, object]:
    """Run all prepared genital streams without materializing them in memory."""
    config.validate()
    started = time.perf_counter()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tracked = Path(tracked_sqlite).resolve()
    labels = read_track_labels(tracked)
    vertex_policy = preparation.get("vertex_policy")
    if not isinstance(vertex_policy, dict):
        raise RuntimeError("curve optimizer requires the Production point policy")
    dense_writer = MaskSqliteWriter(root / "predictions.sqlite", tracked)
    key_writer = MaskSqliteWriter(root / "keyframes.sqlite", tracked)
    audit = _Audit()
    stream_count = 0
    slowest_stream_summaries: list[dict[str, object]] = []
    stream_audit_path = root / "curve_stream_audit.jsonl"
    stream_audit_temporary = root / ".curve_stream_audit.jsonl.tmp"
    stream_audit_temporary.unlink(missing_ok=True)
    stream_audit = stream_audit_temporary.open("w", encoding="utf-8")
    component_metrics_path = root / "curve_component_metrics.csv"
    component_metrics_temporary = root / ".curve_component_metrics.csv.tmp"
    component_metrics_temporary.unlink(missing_ok=True)
    component_metrics_file = component_metrics_temporary.open(
        "w", encoding="utf-8", newline=""
    )
    component_metrics = csv.writer(component_metrics_file)
    component_metrics.writerow(
        (
            "stream_id",
            "label",
            "track_id",
            "frame",
            "component",
            "is_keyframe",
            "points_per_component",
            "iou",
            "recall",
            "precision",
            "area_ratio",
            "topology_valid",
        )
    )
    segmentation: dict[str, dict[str, int]] = {}
    fallback_multi_component_streams = 0
    emergency_repairs = 0
    independent_local_refit_attempts = 0
    independent_local_refit_accepted = 0
    maximum_independent_local_refit_shift = 0.0
    completion_envelope_frames = 0
    maximum_completion_area_ratio = 1.0
    phase_history: dict[tuple[str, int, int], dict[int, np.ndarray]] = {}
    previous_track_id: str | None = None
    maximum_phase_history_frames = 0
    total_rows = sum(
        int(value.get("input_rows", 0))
        for value in preparation.get("classes", {}).values()
        if isinstance(value, dict)
    )
    emitted_rows = 0
    try:
        for label in preparation.get("active_labels", []):
            class_value = preparation["classes"][label]
            source = Path(str(class_value["endpoint_sqlite"])).resolve()
            stats: dict[str, int] = {}
            for run in iter_track_streams_from_sqlite(
                source,
                anchors_per_contour=14,
                gapfill_enabled=True,
                gapfill_max_gap=int(config.gapfill_max_gap),
                gapfill_temp_points=128,
                max_tracks=int(max_tracks),
                max_run_frames=int(config.max_run_frames),
                run_overlap_frames=int(config.run_overlap_frames),
                segmentation_stats=stats,
                prepare_anchors=False,
            ):
                run_started = time.perf_counter()
                current_track_id = str(run.track_id)
                if (
                    previous_track_id is not None
                    and current_track_id != previous_track_id
                ):
                    # Streams are ordered by track.  Cross-chunk phase history
                    # is useful only while that one track is active; retaining
                    # every completed track would make an eight-hour batch grow
                    # linearly in memory.
                    for history_key in tuple(phase_history):
                        if history_key[0] == previous_track_id:
                            del phase_history[history_key]
                previous_track_id = current_track_id
                point_count = _point_count(vertex_policy, run.track_id)
                emit_start = max(0, int(run.emit_start_idx))
                emit_end = (
                    len(run.frame_numbers)
                    if int(run.emit_end_idx) < 0
                    else min(len(run.frame_numbers), int(run.emit_end_idx))
                )
                if emit_end <= emit_start:
                    continue
                component_controls: list[np.ndarray] = []
                component_boundaries: list[np.ndarray] = []
                component_audits = []
                fit_summaries: list[dict[str, object]] = []
                component_results = []
                for component in range(int(run.contour_count)):
                    references = [
                        np.asarray(frame[component], dtype=np.float64)
                        for frame in run.gt_polygons
                    ]
                    controls, fit_summary = _fit_component(
                        references,
                        point_count,
                        config,
                    )
                    post_fit_repair = fit_summary["post_fit_repair"]
                    if post_fit_repair.get("emergency_repair"):
                        emergency_repairs += 1
                    local_refit = post_fit_repair.get("independent_local_refit", {})
                    if isinstance(local_refit, dict):
                        independent_local_refit_attempts += int(
                            local_refit.get("attempted_count", 0)
                        )
                        independent_local_refit_accepted += int(
                            local_refit.get("accepted_count", 0)
                        )
                        maximum_independent_local_refit_shift = max(
                            maximum_independent_local_refit_shift,
                            float(local_refit.get("maximum_control_shift_px", 0.0)),
                        )
                    completion_envelope_frames += int(
                        post_fit_repair.get("completion_envelope_count", 0)
                    )
                    maximum_completion_area_ratio = max(
                        maximum_completion_area_ratio,
                        float(
                            post_fit_repair.get(
                                "maximum_completion_area_ratio",
                                1.0,
                            )
                        ),
                    )
                    history_key = (str(run.track_id), int(run.contour_count), component)
                    history = phase_history.setdefault(history_key, {})
                    controls, phase_shift = _phase_align(
                        controls,
                        run.frame_numbers,
                        history,
                    )
                    fit_summary["cross_chunk_phase_shift"] = int(phase_shift)
                    if int(run.contour_count) == 1:
                        role_ids = curve_role_ids(label)
                        if role_ids:
                            all_states, all_state_labels = polygon_role_curve_states(
                                controls,
                                run.frame_numbers,
                                role_ids,
                                renderer=catmull_rom_renderer(
                                    int(config.samples_per_segment)
                                ),
                                samples_per_segment=int(config.samples_per_segment),
                            )
                            states = np.ascontiguousarray(all_states[:, :2])
                            state_labels = tuple(all_state_labels[:2])
                            fallback_states = all_states
                            fallback_state_labels = all_state_labels
                            fast_target_ratio = float(config.fast_state_target_ratio)
                            fast_quality_probe = False
                        else:
                            states, state_labels = isotropic_curve_states(
                                controls,
                                tuple(config.fast_state_scales),
                            )
                            (
                                fallback_states,
                                fallback_state_labels,
                            ) = isotropic_curve_states(
                                controls,
                                tuple(config.state_scales),
                            )
                            fast_target_ratio = float(config.fast_state_target_ratio)
                            fast_quality_probe = True
                        dp_config = KeyframeDpConfig(
                            target_interval=int(config.target_interval),
                            recall_floor=float(config.recall_floor),
                            maximum_gap=max(
                                int(config.maximum_gap),
                                4 * int(config.target_interval),
                            ),
                            pair_vote_enabled=True,
                            pair_vote_sweeps=int(config.pair_vote_sweeps),
                            low_iou_quadratic_weight=float(
                                config.low_iou_quadratic_weight
                            ),
                            path_selection_mode=str(config.path_selection_mode),
                            cardinality_maximum_factor=float(
                                config.cardinality_maximum_factor
                            ),
                            shape_distance_weight=float(config.shape_distance_weight),
                            # Recall/topology are hard graph constraints. IoU
                            # remains the cost traded against the requested
                            # key cardinality; no post-DP key insertion changes
                            # that selected Pareto point.
                            quality_rescue_enabled=False,
                            quality_rescue_iou_floor=float(
                                config.quality_rescue_iou_floor
                            ),
                            quality_rescue_regret_floor=float(
                                config.quality_rescue_regret_floor
                            ),
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
                            native_reference_cache_bytes=int(
                                config.native_reference_cache_bytes
                            ),
                        )
                        renderer = catmull_rom_renderer(int(config.samples_per_segment))
                        result = optimize_multistate_keyframes(
                            references,
                            states,
                            state_labels=state_labels,
                            base_controls=controls,
                            representation=(
                                "closed_uniform_catmull_rom_bezier_multistate"
                            ),
                            renderer=renderer,
                            config=dp_config,
                            point_refine=CurvePointRefineConfig(
                                enabled=True,
                                sweeps=int(config.point_refine_sweeps),
                                scheduler=str(config.point_refine_scheduler),
                            ),
                            interval_renderer=renderer,
                            fallback_state_controls=fallback_states,
                            fallback_state_labels=fallback_state_labels,
                            fast_state_target_ratio=fast_target_ratio,
                            fast_state_quality_probe=fast_quality_probe,
                        )
                        dense_controls = np.asarray(result.dense_controls)
                        selected = {
                            int(index)
                            for index in result.chosen_indices
                            if emit_start <= int(index) < emit_end
                        }
                        selected.update((emit_start, emit_end - 1))
                        component_results.append((result, selected))
                        component_audits.append(result.audit)
                    else:
                        dense_controls = controls
                        selected = set(range(emit_start, emit_end))
                        renderer = catmull_rom_renderer(int(config.samples_per_segment))
                        _boundaries, component_audit = audit_dense_path(
                            references,
                            dense_controls,
                            renderer,
                            threads=int(config.native_cpu_threads),
                        )
                        component_audits.append(component_audit)
                    dense_controls, output_shift = _phase_align(
                        dense_controls,
                        run.frame_numbers,
                        history,
                    )
                    fit_summary["output_phase_shift"] = int(output_shift)
                    boundaries = sample_curve_sequence(
                        dense_controls,
                        int(config.samples_per_segment),
                    )
                    component_controls.append(dense_controls)
                    component_boundaries.append(boundaries)
                    fit_summaries.append(fit_summary)
                    for frame, value in zip(
                        run.frame_numbers,
                        dense_controls,
                        strict=True,
                    ):
                        history[int(frame)] = np.asarray(value, dtype=np.float64)
                    if len(history) > 2 * int(config.max_run_frames):
                        keep = sorted(history)[-int(config.max_run_frames) :]
                        phase_history[history_key] = {
                            frame: history[frame] for frame in keep
                        }
                if int(run.contour_count) > 1:
                    fallback_multi_component_streams += 1
                selected_keys = (
                    sorted(component_results[0][1])
                    if component_results
                    else list(range(emit_start, emit_end))
                )
                for index in range(emit_start, emit_end):
                    dense_writer.append(
                        frame=int(run.frame_numbers[index]),
                        track_id=str(run.track_id),
                        polygons=[
                            boundary[index].tolist()
                            for boundary in component_boundaries
                        ],
                        label=str(label),
                    )
                for index in selected_keys:
                    key_writer.append(
                        frame=int(run.frame_numbers[index]),
                        track_id=str(run.track_id),
                        polygons=[
                            controls[index].tolist() for controls in component_controls
                        ],
                        label=str(label),
                    )
                emitted = int(emit_end - emit_start)
                audit.frames += emitted
                audit.keys += int(len(selected_keys))
                for result, _selected in component_results:
                    audit.add_result(result, emit_start, emit_end)
                if not component_results:
                    for component_audit in component_audits:
                        audit.add_component(
                            component_audit,
                            emit_start,
                            emit_end,
                            recall_floor=float(config.recall_floor),
                        )
                selected_key_set = set(int(value) for value in selected_keys)
                for component, component_audit in enumerate(component_audits):
                    for index in range(emit_start, emit_end):
                        component_metrics.writerow(
                            (
                                str(run.stream_id),
                                str(label),
                                str(run.track_id),
                                int(run.frame_numbers[index]),
                                int(component),
                                int(index in selected_key_set),
                                int(point_count),
                                float(component_audit.iou[index]),
                                float(component_audit.recall[index]),
                                float(component_audit.precision[index]),
                                float(component_audit.area_ratio[index]),
                                int(bool(component_audit.topology_valid[index])),
                            )
                        )
                emitted_rows += emitted
                stream_summary = {
                    "stream_id": str(run.stream_id),
                    "track_id": str(run.track_id),
                    "label": str(label),
                    "frames_processed": int(len(run.frame_numbers)),
                    "frames_emitted": emitted,
                    "components": int(run.contour_count),
                    "points_per_component": int(point_count),
                    "keyframes": int(len(selected_keys)),
                    "chunk_index": int(run.chunk_index),
                    "chunked": bool(run.chunked_from_long_run),
                    "fit": [_compact_fit_summary(value) for value in fit_summaries],
                    "state_search": [
                        {
                            "initial_states": int(result.state_search_initial_count),
                            "final_states": int(result.state_search_final_count),
                            "fallback": bool(result.state_search_fallback),
                            "probe_seconds": float(result.state_search_probe_seconds),
                            "native_reference_cache": dict(
                                result.native_reference_cache
                            ),
                        }
                        for result, _selected in component_results
                    ],
                    "optimization": [
                        {
                            "dp_seconds": float(result.dp_seconds),
                            "pair_vote_seconds": float(result.pair_vote_seconds),
                            "point_refine_seconds": float(result.point_refine_seconds),
                            "final_audit_seconds": float(result.final_audit_seconds),
                            "graph_edges": int(result.edge_evaluations),
                            "graph_reused_edges": int(result.native_graph_reused_edges),
                            "shape_distance_unique_pairs": int(
                                result.native_shape_distance_unique_pairs
                            ),
                            "lazy_topology_decodes": int(
                                result.native_lazy_topology_decodes
                            ),
                            "lazy_topology_checked_edges": int(
                                result.native_lazy_topology_checked_edges
                            ),
                            "lazy_topology_checked_frames": int(
                                result.native_lazy_topology_checked_frames
                            ),
                            "lazy_topology_rejected_edges": int(
                                result.native_lazy_topology_rejected_edges
                            ),
                            "quality_rescue_inserted": int(
                                len(result.quality_rescue_inserted_indices)
                            ),
                            "refinement_guard_triggered": bool(
                                result.refinement_guard_triggered
                            ),
                            "refinement_guard_alpha": float(
                                result.refinement_guard_alpha
                            ),
                        }
                        for result, _selected in component_results
                    ],
                    "elapsed_seconds": float(time.perf_counter() - run_started),
                }
                stream_audit.write(
                    json.dumps(
                        stream_summary,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream_count += 1
                slowest_stream_summaries.append(stream_summary)
                slowest_stream_summaries.sort(
                    key=lambda value: float(value["elapsed_seconds"]),
                    reverse=True,
                )
                del slowest_stream_summaries[_SLOW_STREAM_SUMMARY_LIMIT:]
                maximum_phase_history_frames = max(
                    maximum_phase_history_frames,
                    sum(len(value) for value in phase_history.values()),
                )
                if progress_callback is not None:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    progress_callback(
                        f"curve:{label}:track={run.track_id}",
                        min(1.0, emitted_rows / max(total_rows, 1)),
                        emitted_rows / elapsed,
                    )
            segmentation[str(label)] = dict(stats)
        passthrough = _append_passthrough(
            tracked,
            labels,
            tuple(str(value) for value in preparation.get("passthrough_track_ids", [])),
            dense_writer,
            key_writer,
        )
        key_writer.add_curve_metadata(point_policy=vertex_policy.get("summary", {}))
        dense_path = dense_writer.finalize()
        key_path = key_writer.finalize()
        stream_audit.flush()
        stream_audit.close()
        os.replace(stream_audit_temporary, stream_audit_path)
        component_metrics_file.flush()
        component_metrics_file.close()
        os.replace(component_metrics_temporary, component_metrics_path)
    except BaseException:
        stream_audit.close()
        stream_audit_temporary.unlink(missing_ok=True)
        component_metrics_file.close()
        component_metrics_temporary.unlink(missing_ok=True)
        dense_writer.abort()
        key_writer.abort()
        raise
    elapsed = time.perf_counter() - started
    raster_backend = (
        os.environ.get("MASK_CURVE_EXACT_RASTER_BACKEND", "cpu").strip().lower()
    )
    cuda_used = raster_backend in {"cuda", "cuda_hybrid"}
    summary = {
        "schema_version": 1,
        "algorithm": "closed_uniform_catmull_rom_tension_1_factor_1_over_6",
        "raster_backend": raster_backend,
        "cpu_only": not cuda_used,
        "cuda_initialized": cuda_used,
        "target_interval": int(config.target_interval),
        "runtime_config": asdict(config),
        "predictions_sqlite": str(dense_path),
        "keyframes_sqlite": str(key_path),
        "prediction_rows": int(dense_writer.rows),
        "keyframe_rows": int(key_writer.rows),
        "passthrough_rows": int(passthrough),
        "streams": int(stream_count),
        "multi_component_key_every_frame_streams": int(
            fallback_multi_component_streams
        ),
        "emergency_spatial_repairs": int(emergency_repairs),
        "independent_local_refit_attempts": int(independent_local_refit_attempts),
        "independent_local_refit_accepted": int(independent_local_refit_accepted),
        "maximum_independent_local_refit_shift_px": float(
            maximum_independent_local_refit_shift
        ),
        "completion_envelope_frames": int(completion_envelope_frames),
        "maximum_completion_area_ratio": float(maximum_completion_area_ratio),
        "segmentation": segmentation,
        "audit": audit.summary(),
        "elapsed_seconds": float(elapsed),
        "emitted_fps": float(emitted_rows / max(elapsed, 1e-9)),
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "maximum_phase_history_frames": int(maximum_phase_history_frames),
        "stream_audit_jsonl": str(stream_audit_path),
        "component_metrics_csv": str(component_metrics_path),
        "stream_summaries": slowest_stream_summaries,
        "stream_summaries_are_slowest": True,
        "stream_summaries_truncated": bool(
            stream_count > len(slowest_stream_summaries)
        ),
    }
    manifest = root / "curve_engine_manifest.json"
    manifest.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["manifest"] = str(manifest)
    return summary


__all__ = ("run_curve_optimizer",)
