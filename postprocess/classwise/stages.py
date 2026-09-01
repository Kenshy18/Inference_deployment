"""Composite stage that routes disjoint tracks through existing pipelines."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.runner import PipelineRunner
from contracts.stages import StageContext, StageResult

from .curve_parallel import (
    CurveGroupJob,
    available_cpu_count,
    curve_track_costs,
    partition_curve_tracks,
    run_curve_group_process,
)
from .curve_scheduling import allocate_curve_group_shards
from .pipeline_factory import build_nested_pipeline
from .policy import (
    ClassPostprocessSettings,
    PRODUCTION_POLYGON_MAX_GAP,
    load_class_postprocess_policy,
)
from .sqlite import (
    RoutedGroup,
    count_masks,
    filter_tracked_sqlite,
    merge_routed_outputs,
    read_track_labels,
)


@dataclass(frozen=True)
class ClasswisePostprocessStage:
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "classwise_postprocess"
    requires: frozenset[str] = frozenset(
        {"tracked_sqlite", "class_postprocess_policy_json"}
    )
    provides: frozenset[str] = frozenset({"predictions_sqlite", "classwise_manifest"})

    def run(self, context: StageContext) -> StageResult:
        started = time.perf_counter()
        context.report_progress("classwise:preparing", 0.01)
        fallback = ClassPostprocessSettings(
            shape_mode="polygon",
            keyframe_interval=int(
                self.options.get(
                    "default_keyframe_interval",
                    6,
                )
            ),
            max_gap=PRODUCTION_POLYGON_MAX_GAP,
        )
        policy = load_class_postprocess_policy(
            context.artifacts["class_postprocess_policy_json"],
            fallback=fallback,
        )
        tracked = context.artifacts["tracked_sqlite"]
        track_labels = read_track_labels(tracked)
        # Each semantic class has an independent keyframe budget. Do not
        # coalesce labels merely because their interval happens to match.
        tracks_by_group: dict[tuple[str, ClassPostprocessSettings], list[str]] = {}
        for track_id, label in sorted(track_labels.items()):
            settings = policy.resolve(label)
            tracks_by_group.setdefault((label, settings), []).append(track_id)

        geometry_options = dict(
            self.options.get(
                "geometry_options",
                self.options.get("polygon_options", {}),
            )
        )
        geometry_mode = str(self.options.get("geometry_mode", "polygon"))
        if geometry_mode not in {"polygon", "catmull_rom"}:
            raise ValueError(f"unsupported Production geometry: {geometry_mode}")
        semantic_groups = sorted(
            tracks_by_group,
            key=lambda value: (
                value[1].keyframe_interval,
                value[0],
            ),
        )
        mask_counts_by_track: dict[str, int] = {}
        curve_costs_by_track: dict[str, int] = {}
        curve_cost_metric: str | None = None
        available_cpus = available_cpu_count()
        if geometry_mode == "catmull_rom":
            requested_workers = int(
                geometry_options.get(
                    "parallel_workers",
                    min(6, available_cpus),
                )
            )
        else:
            requested_workers = int(self.options.get("classwise_workers", 3))
        if geometry_mode == "catmull_rom":
            # Every route reads the same immutable tracked SQLite. Scan its
            # compact index once rather than making concurrent class workers
            # perform identical full-table GROUP BY queries.
            with sqlite3.connect(
                f"file:{Path(tracked).resolve()}?mode=ro", uri=True
            ) as connection:
                mask_counts_by_track = {
                    str(track_id): int(count)
                    for track_id, count in connection.execute(
                        "SELECT track_id,COUNT(*) FROM masks GROUP BY track_id"
                    )
                }
                curve_costs_by_track, curve_cost_metric = curve_track_costs(
                    connection,
                    mask_counts_by_track,
                )
        maximum_curve_shards_per_class = max(
            1,
            min(
                8,
                int(geometry_options.get("parallel_shards_per_class", 8)),
            ),
        )
        semantic_track_ids = tuple(
            tuple(tracks_by_group[group]) for group in semantic_groups
        )
        curve_shards_by_group = (
            allocate_curve_group_shards(
                semantic_track_ids,
                curve_costs_by_track,
                process_budget=min(max(1, requested_workers), available_cpus),
                maximum_shards_per_group=maximum_curve_shards_per_class,
            )
            if geometry_mode == "catmull_rom"
            else tuple(1 for _group in semantic_groups)
        )
        work_groups: list[
            tuple[str, ClassPostprocessSettings, int, tuple[str, ...]]
        ] = []
        for group_index, (label, settings) in enumerate(semantic_groups):
            track_ids = tuple(tracks_by_group[(label, settings)])
            partitions = (
                partition_curve_tracks(
                    track_ids,
                    curve_costs_by_track,
                    curve_shards_by_group[group_index],
                )
                if geometry_mode == "catmull_rom"
                else (track_ids,)
            )
            work_groups.extend(
                (label, settings, shard_index, partition)
                for shard_index, partition in enumerate(partitions)
            )
        group_count = max(1, len(work_groups))
        workers = min(max(1, requested_workers), group_count, available_cpus)
        requested_curve_threads = int(geometry_options.get("native_cpu_threads", 0))
        if requested_curve_threads > 0:
            curve_cpu_threads = requested_curve_threads
        else:
            # Native exact batches are deterministic across thread counts.
            # Share the affinity-visible CPU budget between balanced track
            # shards so no semantic class can strand the remaining cores.
            curve_cpu_threads = max(
                1,
                min(12, available_cpus // max(workers, 1)),
            )
        progress_lock = threading.Lock()
        progress_by_index = {index: 0.0 for index in range(len(work_groups))}

        def report_group_progress(
            index: int,
            label: str,
            detail: str,
            fraction: float | None,
            fps: float | None,
        ) -> None:
            with progress_lock:
                if fraction is not None:
                    progress_by_index[index] = max(
                        progress_by_index[index],
                        min(1.0, max(0.0, float(fraction))),
                    )
                aggregate = sum(progress_by_index.values()) / group_count
                context.report_progress(
                    f"classwise:{label}:{detail}",
                    0.02 + 0.94 * aggregate,
                    fps,
                )

        def run_group(
            item: tuple[
                int,
                tuple[str, ClassPostprocessSettings, int, tuple[str, ...]],
            ],
        ) -> tuple[RoutedGroup, dict[str, object]]:
            index, (label, settings, shard_index, track_ids) = item
            group_started = time.perf_counter()
            group_id = f"{index:02d}_{geometry_mode}_k{settings.keyframe_interval}"
            group_root = context.stage_dir / "groups" / group_id
            projected = group_root / "tracked.sqlite"
            if geometry_mode == "catmull_rom":
                # The curve stage streams this route directly from the shared
                # read-only source into its lean preparation DB. Avoid a full
                # tracked-database backup for every semantic class.
                projected = Path(tracked)
                input_masks = sum(
                    int(mask_counts_by_track.get(track_id, 0)) for track_id in track_ids
                )
            elif set(track_ids) == set(track_labels):
                projected = Path(tracked)
                input_masks = count_masks(projected)
            else:
                input_masks = filter_tracked_sqlite(
                    tracked,
                    projected,
                    track_ids=track_ids,
                )
            nested_root = group_root / "pipeline"
            nested_inputs = {"tracked_sqlite": projected}
            if context.artifacts.get("input_video") is not None:
                nested_inputs["input_video"] = context.artifacts["input_video"]
            manifest = PipelineRunner(
                build_nested_pipeline(
                    settings,
                    geometry_options=geometry_options,
                    geometry_mode=geometry_mode,
                    curve_cpu_threads=curve_cpu_threads,
                    selected_track_ids=track_ids,
                ),
                nested_root,
                progress_callback=(
                    lambda detail, fraction, fps: report_group_progress(
                        index,
                        label,
                        detail,
                        fraction,
                        fps,
                    )
                ),
            ).run(nested_inputs)
            predictions = (
                Path(str(manifest["artifacts"]["predictions_sqlite"]))
                .expanduser()
                .resolve()
            )
            routed_group = RoutedGroup(
                group_id=group_id,
                labels=(label,),
                track_ids=track_ids,
                settings=settings,
                predictions_sqlite=predictions,
            )
            output_masks = 0
            for stage in manifest["stages"]:
                if stage["id"] == "output_validation":
                    output_masks = int(stage["metadata"]["masks"])
                    break
            group_manifest = {
                "id": group_id,
                "labels": [label],
                "geometry_mode": geometry_mode,
                "semantic_shard_index": int(shard_index),
                "track_ids": list(track_ids),
                "settings": settings.as_dict(),
                "input_masks": input_masks,
                "tracked_input": str(Path(projected).resolve()),
                "shared_read_only_source": bool(geometry_mode == "catmull_rom"),
                "output_masks": output_masks,
                "pipeline_manifest": str(nested_root / "pipeline_manifest.json"),
                "predictions_sqlite": str(predictions),
                "elapsed_seconds": time.perf_counter() - group_started,
            }
            report_group_progress(index, label, "complete", 1.0, None)
            return routed_group, group_manifest

        indexed_groups = list(enumerate(work_groups))
        execution_worker_mode = "serial"
        if workers == 1:
            results = [run_group(item) for item in indexed_groups]
        elif geometry_mode == "catmull_rom":
            # Most exact raster work releases the GIL, but candidate setup,
            # DP, rescue, SQLite and audit orchestration do not. Independent
            # processes let those Python sections execute concurrently.
            # ``spawn`` avoids inheriting OpenCV/OpenMP worker state.
            execution_worker_mode = "process_spawn"
            jobs: list[CurveGroupJob] = []
            for index, (label, settings, shard_index, track_ids) in indexed_groups:
                group_id = (
                    f"{index:02d}_{geometry_mode}_k{settings.keyframe_interval}"
                )
                group_root = context.stage_dir / "groups" / group_id
                jobs.append(
                    CurveGroupJob(
                        index=index,
                        group_id=group_id,
                        label=label,
                        shard_index=shard_index,
                        settings=settings,
                        geometry_options=dict(geometry_options),
                        curve_cpu_threads=curve_cpu_threads,
                        track_ids=track_ids,
                        tracked=Path(tracked).resolve(),
                        input_video=(
                            Path(context.artifacts["input_video"]).resolve()
                            if context.artifacts.get("input_video") is not None
                            else None
                        ),
                        group_root=group_root,
                        input_masks=sum(
                            int(mask_counts_by_track.get(track_id, 0))
                            for track_id in track_ids
                        ),
                    )
                )
            completed: dict[int, tuple[RoutedGroup, dict[str, object]]] = {}

            def refresh_worker_progress() -> None:
                latest: tuple[float, str, str, float | None] | None = None
                with progress_lock:
                    for job in jobs:
                        progress_path = job.group_root / "worker_progress.json"
                        try:
                            payload = json.loads(
                                progress_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            continue
                        raw_fraction = payload.get("fraction")
                        if raw_fraction is not None:
                            progress_by_index[job.index] = max(
                                progress_by_index[job.index],
                                min(1.0, max(0.0, float(raw_fraction))),
                            )
                        updated = float(payload.get("updated_monotonic", 0.0))
                        candidate = (
                            updated,
                            job.label,
                            str(payload.get("detail", "running")),
                            (
                                None
                                if payload.get("fps") is None
                                else float(payload["fps"])
                            ),
                        )
                        if latest is None or candidate[0] > latest[0]:
                            latest = candidate
                    aggregate = sum(progress_by_index.values()) / group_count
                detail = "classwise:curve-workers:running"
                fps = None
                if latest is not None:
                    _updated, label, worker_detail, fps = latest
                    detail = f"classwise:{label}:{worker_detail}"
                context.report_progress(
                    detail,
                    0.02 + 0.94 * aggregate,
                    fps,
                )

            spawn_context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_context,
            ) as executor:
                pending = {
                    executor.submit(run_curve_group_process, job): job
                    for job in jobs
                }
                while pending:
                    done, _not_done = concurrent.futures.wait(
                        pending,
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        refresh_worker_progress()
                        continue
                    for future in done:
                        job = pending.pop(future)
                        result_index, routed_group, group_manifest = future.result()
                        completed[result_index] = (routed_group, group_manifest)
                        report_group_progress(
                            result_index,
                            job.label,
                            "complete",
                            1.0,
                            None,
                        )
            results = [completed[index] for index in range(len(indexed_groups))]
        else:
            execution_worker_mode = "thread"
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="classwise-postprocess",
            ) as executor:
                results = list(executor.map(run_group, indexed_groups))
        routed = [result[0] for result in results]
        group_manifests = [result[1] for result in results]

        output = context.stage_dir / "predictions.sqlite"
        context.report_progress("classwise:merging", 0.97)
        merge_summary = merge_routed_outputs(
            tracked,
            tuple(routed),
            output,
            policy=policy,
            track_labels=track_labels,
        )
        elapsed = time.perf_counter() - started
        classwise_manifest = context.stage_dir / "classwise_manifest.json"
        manifest_value = {
            "schema_version": 1,
            "policy": policy.as_dict(),
            "policy_source": str(context.artifacts["class_postprocess_policy_json"]),
            "tracked_sqlite": str(tracked),
            "predictions_sqlite": str(output),
            "groups": group_manifests,
            "execution": {
                "classwise_workers": workers,
                "parallel": workers > 1,
                "worker_mode": execution_worker_mode,
                "geometry_mode": geometry_mode,
                "curve_cpu_threads_per_group": (
                    curve_cpu_threads if geometry_mode == "catmull_rom" else None
                ),
                "curve_parallel_workers": (
                    workers if geometry_mode == "catmull_rom" else None
                ),
                "curve_maximum_shards_per_class": (
                    maximum_curve_shards_per_class
                    if geometry_mode == "catmull_rom"
                    else None
                ),
                "curve_shards_by_semantic_group": (
                    [
                        {
                            "label": str(group[0]),
                            "target_interval": int(group[1].keyframe_interval),
                            "shards": int(curve_shards_by_group[index]),
                        }
                        for index, group in enumerate(semantic_groups)
                    ]
                    if geometry_mode == "catmull_rom"
                    else None
                ),
                "curve_work_cost_metric": curve_cost_metric,
            },
            "merge": merge_summary,
            "elapsed_seconds": elapsed,
        }
        classwise_manifest.write_text(
            json.dumps(manifest_value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context.report_progress("classwise:validating", 0.99)
        return StageResult(
            {
                "predictions_sqlite": output,
                "classwise_manifest": classwise_manifest,
            },
            {
                "policy": policy.as_dict(),
                "groups": len(group_manifests),
                "classwise_workers": workers,
                "group_summaries": group_manifests,
                **merge_summary,
                "elapsed_seconds": elapsed,
            },
        )


__all__ = ["ClasswisePostprocessStage"]
