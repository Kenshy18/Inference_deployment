"""Deterministic track-sharded orchestration for Production curves."""

from __future__ import annotations

import concurrent.futures
import csv
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import time
from typing import Callable

import numpy as np

from classwise.sqlite import read_track_labels

from .config import CurveProductionConfig
from .engine import _SLOW_STREAM_SUMMARY_LIMIT, _append_passthrough, run_curve_optimizer
from .storage import MaskSqliteWriter


ProgressCallback = Callable[[str, float | None, float | None], None]


def _available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _track_counts(path: Path, max_tracks: int) -> list[tuple[str, int]]:
    with sqlite3.connect(
        f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True
    ) as database:
        suffix = " LIMIT ?" if int(max_tracks) > 0 else ""
        parameters = (int(max_tracks),) if int(max_tracks) > 0 else ()
        rows = database.execute(
            "SELECT track_id,count(*) AS n FROM masks GROUP BY track_id "
            "ORDER BY n DESC,CAST(track_id AS INTEGER)" + suffix,
            parameters,
        )
        return [(str(track_id), int(count)) for track_id, count in rows]


def _shard_preparations(
    preparation: dict[str, object], requested_workers: int, max_tracks: int
) -> list[tuple[dict[str, object], int, int, int]]:
    labels = tuple(str(value) for value in preparation.get("active_labels", ()))
    classes = preparation.get("classes", {})
    if not isinstance(classes, dict):
        raise RuntimeError("curve preparation classes must be a mapping")
    vertex_policy = preparation.get("vertex_policy", {})
    track_policy = (
        vertex_policy.get("tracks", {})
        if isinstance(vertex_policy, dict)
        else {}
    )
    work: list[tuple[int, int, str, str]] = []
    for label in labels:
        value = classes.get(label)
        if not isinstance(value, dict):
            raise RuntimeError(f"curve preparation is missing label {label}")
        for track_id, rows in _track_counts(
            Path(str(value["endpoint_sqlite"])), int(max_tracks)
        ):
            policy = (
                track_policy.get(str(track_id), {})
                if isinstance(track_policy, dict)
                else {}
            )
            points = (
                int(policy.get("vertices_per_component", 14))
                if isinstance(policy, dict)
                else 14
            )
            # Exact curve fitting and refinement cost grows substantially with
            # the number of editable points.  Full-corpus timings show that a
            # +2-point step costs about x1.56 per frame.  Use that stable,
            # geometry-derived signal only for scheduling; it cannot alter a
            # candidate, metric, keyframe, or serialized mask.
            point_factor = math.exp(0.22 * max(0, points - 14))
            estimated_cost = max(1, int(round(int(rows) * point_factor * 100.0)))
            work.append((estimated_cost, int(rows), label, track_id))
    if not work:
        return []
    row_lookup = {
        (label, track_id): int(rows) for _cost, rows, label, track_id in work
    }
    worker_count = min(max(1, int(requested_workers)), len(work))
    bins: list[dict[str, list[str]]] = [{} for _ in range(worker_count)]
    loads = [0] * worker_count
    counts = [0] * worker_count
    row_loads = [0] * worker_count
    for cost, rows, label, track_id in sorted(
        work, key=lambda value: (-value[0], value[2], value[3])
    ):
        target = min(range(worker_count), key=lambda index: (loads[index], index))
        bins[target].setdefault(label, []).append(track_id)
        loads[target] += int(cost)
        row_loads[target] += int(rows)
        counts[target] += 1
    output = []
    for assignments, row_load, count, estimated_load in zip(
        bins, row_loads, counts, loads, strict=True
    ):
        worker = dict(preparation)
        worker["active_labels"] = [
            label for label in labels if assignments.get(label)
        ]
        worker_classes: dict[str, object] = {}
        for label in worker["active_labels"]:
            value = dict(classes[label])
            value["allowed_track_ids"] = list(assignments[label])
            value["input_rows"] = sum(
                row_lookup[(label, track_id)] for track_id in assignments[label]
            )
            worker_classes[label] = value
        worker["classes"] = worker_classes
        worker["passthrough_track_ids"] = []
        output.append(
            (worker, int(row_load), int(count), int(estimated_load))
        )
    return output


def _run_label_worker(
    tracked_sqlite: str,
    preparation: dict[str, object],
    output_root: str,
    config: CurveProductionConfig,
    max_tracks: int,
) -> dict[str, object]:
    return run_curve_optimizer(
        Path(tracked_sqlite),
        preparation,
        Path(output_root),
        config=config,
        max_tracks=int(max_tracks),
    )


def _copy_masks(source: Path, writer: MaskSqliteWriter) -> None:
    with sqlite3.connect(
        f"file:{Path(source).resolve()}?mode=ro&immutable=1", uri=True
    ) as database:
        for frame, track_id, polygons, label in database.execute(
            "SELECT frame,track_id,polygons,COALESCE(label,'') FROM masks "
            "ORDER BY track_id,frame"
        ):
            writer.append(
                frame=int(frame),
                track_id=str(track_id),
                polygons=str(polygons),
                label=str(label),
            )


def _merge_component_metrics(
    sources: list[Path], output: Path, recall_floor: float
) -> tuple[dict[str, object], int]:
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    ious: list[float] = []
    recall_minimum = 1.0
    recall_violations = 0
    topology_invalid = 0
    maximum_area_ratio = 0.0
    rows = 0
    fieldnames: list[str] | None = None
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = None
        for source in sources:
            with source.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or ())
                    writer = csv.DictWriter(target, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames or ()) != fieldnames:
                    raise RuntimeError("curve component metric schemas differ")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    iou = float(row["iou"])
                    recall = float(row["recall"])
                    area_ratio = float(row["area_ratio"])
                    ious.append(iou)
                    recall_minimum = min(recall_minimum, recall)
                    recall_violations += int(
                        recall + 1e-12 < float(recall_floor)
                    )
                    topology_invalid += int(not bool(int(row["topology_valid"])))
                    maximum_area_ratio = max(maximum_area_ratio, area_ratio)
                    rows += 1
    os.replace(temporary, output)
    values = np.asarray(ious, dtype=np.float64)
    return {
        "component_observations": int(rows),
        "mean_iou": float(np.mean(values)) if len(values) else 1.0,
        "minimum_iou": float(np.min(values)) if len(values) else 1.0,
        "q01_iou": float(np.quantile(values, 0.01)) if len(values) else 1.0,
        "q05_iou": float(np.quantile(values, 0.05)) if len(values) else 1.0,
        "minimum_recall": float(recall_minimum),
        "recall_violations": int(recall_violations),
        "topology_invalid_frames": int(topology_invalid),
        "maximum_area_ratio": float(maximum_area_ratio),
    }, int(rows)


def _merge_audits(
    workers: list[dict[str, object]], metric_summary: dict[str, object]
) -> dict[str, object]:
    audits = [dict(value["audit"]) for value in workers]
    frames = sum(int(value["emitted_component_frames"]) for value in audits)
    keys = sum(int(value["keyframes"]) for value in audits)
    output = dict(metric_summary)
    output.update(
        {
            "emitted_component_frames": int(frames),
            "keyframes": int(keys),
            "effective_interval": float(frames / max(keys, 1)),
        }
    )
    summed = (
        "pair_vote_trials",
        "pair_vote_accepted",
        "pair_vote_iou_gain",
        "point_refine_trials",
        "point_refine_accepted",
        "point_refine_iou_gain",
        "fit_seconds",
        "stream_seconds",
        "dp_seconds",
        "pair_vote_seconds",
        "point_refine_seconds",
        "final_audit_seconds",
        "state_search_fallbacks",
        "state_search_probe_seconds",
        "lazy_topology_decodes",
        "lazy_topology_checked_edges",
        "lazy_topology_checked_frames",
        "lazy_topology_rejected_edges",
        "refinement_guard_triggers",
    )
    for key in summed:
        output[key] = sum(value.get(key, 0) for value in audits)
    output["other_stream_seconds"] = max(
        0.0,
        float(output["stream_seconds"])
        - float(output["fit_seconds"])
        - float(output["dp_seconds"])
        - float(output["pair_vote_seconds"])
        - float(output["point_refine_seconds"])
        - float(output["final_audit_seconds"]),
    )
    output["refinement_guard_minimum_alpha"] = min(
        (float(value.get("refinement_guard_minimum_alpha", 1.0)) for value in audits),
        default=1.0,
    )
    return output


def _merge_segmentation(
    workers: list[dict[str, object]], labels: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    additive = {
        "source_tracks",
        "source_rows",
        "gapfill_inserted_frames",
        "gapfill_events",
        "hard_split_events",
        "segment_count",
        "source_segment_count",
        "processed_segment_count",
        "long_segment_count",
        "chunked_source_segment_count",
        "chunk_output_segment_count",
        "overlap_added_rows",
        "effective_stream_count",
    }
    maxima = {"max_source_segment_frames", "max_processed_segment_frames"}
    output: dict[str, dict[str, int]] = {}
    for label in labels:
        values = [
            dict(worker.get("segmentation", {}).get(label, {}))
            for worker in workers
            if label in worker.get("segmentation", {})
        ]
        if not values:
            continue
        keys = set().union(*(value.keys() for value in values))
        merged: dict[str, int] = {}
        for key in keys:
            numbers = [int(value.get(key, 0)) for value in values]
            if key in additive:
                merged[key] = int(sum(numbers))
            elif key in maxima:
                merged[key] = int(max(numbers, default=0))
            else:
                merged[key] = int(max(numbers, default=0))
        output[label] = merged
    return output


def _merge_jsonl(sources: list[Path], output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as target:
        for source in sources:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, target, length=1024 * 1024)
    os.replace(temporary, output)


def run_curve_optimizer_parallel(
    tracked_sqlite: Path,
    preparation: dict[str, object],
    output_root: Path,
    *,
    config: CurveProductionConfig,
    optimizer_workers: int,
    progress_callback: ProgressCallback | None = None,
    max_tracks: int = 0,
) -> dict[str, object]:
    """Run independent tracks concurrently and merge in canonical order."""

    labels = tuple(str(value) for value in preparation.get("active_labels", ()))
    requested = max(1, int(optimizer_workers))
    shards = _shard_preparations(preparation, requested, int(max_tracks))
    worker_count = len(shards)
    if worker_count <= 1:
        summary = run_curve_optimizer(
            tracked_sqlite,
            preparation,
            output_root,
            config=config,
            progress_callback=progress_callback,
            max_tracks=max_tracks,
        )
        summary["execution_mode"] = "serial"
        summary["optimizer_workers"] = 1
        summary["label_workers"] = 1
        return summary

    started = time.perf_counter()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    workers_root = root / "optimizer_workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    available_cpus = _available_cpu_count()
    native_threads = min(
        int(config.native_cpu_threads),
        max(1, available_cpus // worker_count),
    )
    worker_config = replace(config, native_cpu_threads=int(native_threads))
    worker_config.validate()
    futures: dict[concurrent.futures.Future, int] = {}
    by_worker: dict[int, dict[str, object]] = {}
    total_rows = sum(
        int(value.get("input_rows", 0))
        for value in preparation.get("classes", {}).values()
        if isinstance(value, dict)
    )
    completed_rows = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        for index, (
            worker_preparation,
            _row_load,
            _track_count,
            _estimated_load,
        ) in enumerate(shards):
            future = pool.submit(
                _run_label_worker,
                str(Path(tracked_sqlite).resolve()),
                worker_preparation,
                str(workers_root / f"{index:02d}"),
                worker_config,
                0,
            )
            futures[future] = index
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            by_worker[index] = future.result()
            completed_rows += int(shards[index][1])
            if progress_callback is not None:
                elapsed = max(time.perf_counter() - started, 1e-9)
                progress_callback(
                    f"curve:worker={index + 1}:parallel-complete",
                    min(1.0, completed_rows / max(total_rows, 1)),
                    completed_rows / elapsed,
                )
    parallel_seconds = float(time.perf_counter() - started)
    workers = [by_worker[index] for index in range(worker_count)]

    merge_started = time.perf_counter()
    tracked = Path(tracked_sqlite).resolve()
    dense_writer = MaskSqliteWriter(root / "predictions.sqlite", tracked)
    key_writer = MaskSqliteWriter(root / "keyframes.sqlite", tracked)
    try:
        for value in workers:
            _copy_masks(Path(str(value["predictions_sqlite"])), dense_writer)
            _copy_masks(Path(str(value["keyframes_sqlite"])), key_writer)
        passthrough = _append_passthrough(
            tracked,
            read_track_labels(tracked),
            tuple(str(value) for value in preparation.get("passthrough_track_ids", ())),
            dense_writer,
            key_writer,
        )
        vertex_policy = preparation.get("vertex_policy", {})
        point_policy = (
            vertex_policy.get("summary", {})
            if isinstance(vertex_policy, dict)
            else {}
        )
        key_writer.add_curve_metadata(point_policy=point_policy)
        dense_path = dense_writer.finalize()
        key_path = key_writer.finalize()
    except BaseException:
        dense_writer.abort()
        key_writer.abort()
        raise

    component_metrics_path = root / "curve_component_metrics.csv"
    metric_summary, _metric_rows = _merge_component_metrics(
        [Path(str(value["component_metrics_csv"])) for value in workers],
        component_metrics_path,
        float(config.recall_floor),
    )
    stream_audit_path = root / "curve_stream_audit.jsonl"
    _merge_jsonl(
        [Path(str(value["stream_audit_jsonl"])) for value in workers],
        stream_audit_path,
    )
    audit = _merge_audits(workers, metric_summary)
    slowest = sorted(
        (
            summary
            for value in workers
            for summary in value.get("stream_summaries", ())
        ),
        key=lambda value: float(value["elapsed_seconds"]),
        reverse=True,
    )[:_SLOW_STREAM_SUMMARY_LIMIT]
    emitted_rows = sum(
        int(value["prediction_rows"]) - int(value.get("passthrough_rows", 0))
        for value in workers
    )
    elapsed = float(time.perf_counter() - started)
    raster_backend = os.environ.get(
        "MASK_CURVE_EXACT_RASTER_BACKEND", "cpu"
    ).strip().lower()
    cuda_used = raster_backend in {"cuda", "cuda_hybrid"}
    summary = {
        "schema_version": 1,
        "algorithm": "closed_uniform_catmull_rom_tension_1_factor_1_over_6",
        "raster_backend": raster_backend,
        "cpu_only": not cuda_used,
        "cuda_initialized": cuda_used,
        "execution_mode": "track_sharded",
        "optimizer_workers": int(worker_count),
        # Retained for consumers of the first parallel prototype manifest.
        "label_workers": int(worker_count),
        "native_threads_per_worker": int(native_threads),
        "native_threads_per_label": int(native_threads),
        "target_interval": int(config.target_interval),
        "runtime_config": asdict(config),
        "worker_runtime_config": asdict(worker_config),
        "predictions_sqlite": str(dense_path),
        "keyframes_sqlite": str(key_path),
        "prediction_rows": int(dense_writer.rows),
        "keyframe_rows": int(key_writer.rows),
        "passthrough_rows": int(passthrough),
        "streams": sum(int(value["streams"]) for value in workers),
        "multi_component_key_every_frame_streams": sum(
            int(value["multi_component_key_every_frame_streams"]) for value in workers
        ),
        "emergency_spatial_repairs": sum(
            int(value["emergency_spatial_repairs"]) for value in workers
        ),
        "independent_local_refit_attempts": sum(
            int(value["independent_local_refit_attempts"]) for value in workers
        ),
        "independent_local_refit_accepted": sum(
            int(value["independent_local_refit_accepted"]) for value in workers
        ),
        "maximum_independent_local_refit_shift_px": max(
            float(value["maximum_independent_local_refit_shift_px"])
            for value in workers
        ),
        "completion_envelope_frames": sum(
            int(value["completion_envelope_frames"]) for value in workers
        ),
        "maximum_completion_area_ratio": max(
            float(value["maximum_completion_area_ratio"]) for value in workers
        ),
        "segmentation": _merge_segmentation(workers, labels),
        "audit": audit,
        "elapsed_seconds": elapsed,
        "parallel_compute_seconds": parallel_seconds,
        "merge_seconds": float(time.perf_counter() - merge_started),
        "emitted_fps": float(emitted_rows / max(elapsed, 1e-9)),
        "maximum_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            + sum(int(value.get("maximum_rss_kib", 0)) for value in workers)
        ),
        "maximum_phase_history_frames": max(
            int(value["maximum_phase_history_frames"]) for value in workers
        ),
        "stream_audit_jsonl": str(stream_audit_path),
        "component_metrics_csv": str(component_metrics_path),
        "stream_summaries": slowest,
        "stream_summaries_are_slowest": True,
        "stream_summaries_truncated": bool(
            sum(int(value["streams"]) for value in workers) > len(slowest)
        ),
        "worker_elapsed_seconds": {
            str(index): float(by_worker[index]["elapsed_seconds"])
            for index in range(worker_count)
        },
        "worker_input_rows": {
            str(index): int(shards[index][1]) for index in range(worker_count)
        },
        "worker_track_counts": {
            str(index): int(shards[index][2]) for index in range(worker_count)
        },
        "worker_estimated_cost": {
            str(index): int(shards[index][3]) for index in range(worker_count)
        },
    }
    manifest = root / "curve_engine_manifest.json"
    manifest.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["manifest"] = str(manifest)

    # Worker SQLite files duplicate the merged result and can be hundreds of
    # megabytes on long videos.  Remove only this private stage subtree after
    # the atomic merged outputs and manifest are complete.
    shutil.rmtree(workers_root)
    return summary


__all__ = ("run_curve_optimizer_parallel",)
