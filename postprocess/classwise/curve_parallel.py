"""Deterministic CPU scheduling for independent Catmull--Rom track routes."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from common.runner import PipelineRunner

from .pipeline_factory import build_nested_pipeline
from .policy import ClassPostprocessSettings
from .sqlite import RoutedGroup


def available_cpu_count() -> int:
    """Respect orchestrator/cgroup affinity before falling back to host CPUs."""

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def partition_curve_tracks(
    track_ids: tuple[str, ...],
    costs: dict[str, int],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Stably balance independent tracks without changing their semantics."""

    requested = max(1, int(shard_count))
    count = min(requested, len(track_ids))
    if count <= 1:
        return (tuple(track_ids),)
    bins: list[list[str]] = [[] for _index in range(count)]
    loads = [0 for _index in range(count)]
    ordered = sorted(
        (str(track_id) for track_id in track_ids),
        key=lambda track_id: (-int(costs.get(track_id, 0)), track_id),
    )
    for track_id in ordered:
        destination = min(range(count), key=lambda index: (loads[index], index))
        bins[destination].append(track_id)
        loads[destination] += int(costs.get(track_id, 0))
    return tuple(tuple(sorted(values)) for values in bins)


def curve_track_costs(
    connection: sqlite3.Connection,
    mask_counts: dict[str, int],
) -> tuple[dict[str, int], str]:
    """Estimate exact-raster work from native-resolution detection boxes."""

    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'"
        )
    }
    if "raw_tracked_masks" not in tables:
        return dict(mask_counts), "mask_rows"
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(raw_tracked_masks)")
    }
    required = {
        "final_track_id",
        "removed_by_short_track",
        "bbox_xyxy_json",
    }
    if not required.issubset(columns):
        return dict(mask_counts), "mask_rows"
    try:
        rows = tuple(
            connection.execute(
                """
                SELECT final_track_id,
                       SUM(MAX(
                           1.0,
                           (json_extract(bbox_xyxy_json, '$[2]')
                            - json_extract(bbox_xyxy_json, '$[0]'))
                           *
                           (json_extract(bbox_xyxy_json, '$[3]')
                            - json_extract(bbox_xyxy_json, '$[1]'))
                       ))
                FROM raw_tracked_masks
                WHERE removed_by_short_track=0
                  AND final_track_id IS NOT NULL
                  AND bbox_xyxy_json IS NOT NULL
                GROUP BY final_track_id
                """
            )
        )
    except sqlite3.OperationalError:
        # Some external SQLite builds omit JSON1. A performance hint must not
        # make an otherwise valid input unusable.
        return dict(mask_counts), "mask_rows"
    measured = {
        str(track_id): max(1, int(round(float(value))))
        for track_id, value in rows
        if value is not None
    }
    measured_frames = sum(
        int(mask_counts.get(track_id, 0)) for track_id in measured
    )
    average_area = max(
        1,
        int(round(sum(measured.values()) / max(measured_frames, 1))),
    )
    return (
        {
            track_id: int(measured.get(track_id, count * average_area))
            for track_id, count in mask_counts.items()
        },
        "source_bbox_area_sum",
    )


@dataclass(frozen=True, slots=True)
class CurveGroupJob:
    """Pickle-safe description of one disjoint curve worker."""

    index: int
    group_id: str
    label: str
    shard_index: int
    settings: ClassPostprocessSettings
    geometry_options: dict[str, object]
    curve_cpu_threads: int
    track_ids: tuple[str, ...]
    tracked: Path
    input_video: Path | None
    group_root: Path
    input_masks: int


def run_curve_group_process(
    job: CurveGroupJob,
) -> tuple[int, RoutedGroup, dict[str, object]]:
    """Run one curve shard in a fresh process and return merge metadata."""

    group_started = time.perf_counter()
    nested_root = job.group_root / "pipeline"
    progress_path = job.group_root / "worker_progress.json"
    progress_temporary = job.group_root / ".worker_progress.json.tmp"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_temporary.unlink(missing_ok=True)
    last_progress_write = 0.0

    def publish_progress(
        detail: str,
        fraction: float | None,
        fps: float | None,
    ) -> None:
        nonlocal last_progress_write
        now = time.monotonic()
        if fraction != 1.0 and now - last_progress_write < 0.25:
            return
        payload = {
            "detail": str(detail),
            "fraction": None if fraction is None else float(fraction),
            "fps": None if fps is None else float(fps),
            "updated_monotonic": float(now),
        }
        progress_temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(progress_temporary, progress_path)
        last_progress_write = now

    nested_inputs = {"tracked_sqlite": Path(job.tracked)}
    if job.input_video is not None:
        nested_inputs["input_video"] = Path(job.input_video)
    manifest = PipelineRunner(
        build_nested_pipeline(
            job.settings,
            geometry_options=dict(job.geometry_options),
            geometry_mode="catmull_rom",
            curve_cpu_threads=int(job.curve_cpu_threads),
            selected_track_ids=tuple(job.track_ids),
        ),
        nested_root,
        progress_callback=publish_progress,
    ).run(nested_inputs)
    publish_progress("complete", 1.0, None)
    predictions = (
        Path(str(manifest["artifacts"]["predictions_sqlite"])).expanduser().resolve()
    )
    output_masks = 0
    for stage in manifest["stages"]:
        if stage["id"] == "output_validation":
            output_masks = int(stage["metadata"]["masks"])
            break
    routed = RoutedGroup(
        group_id=str(job.group_id),
        labels=(str(job.label),),
        track_ids=tuple(job.track_ids),
        settings=job.settings,
        predictions_sqlite=predictions,
    )
    group_manifest = {
        "id": str(job.group_id),
        "labels": [str(job.label)],
        "geometry_mode": "catmull_rom",
        "semantic_shard_index": int(job.shard_index),
        "track_ids": list(job.track_ids),
        "settings": job.settings.as_dict(),
        "input_masks": int(job.input_masks),
        "tracked_input": str(Path(job.tracked).resolve()),
        "shared_read_only_source": True,
        "output_masks": int(output_masks),
        "pipeline_manifest": str(nested_root / "pipeline_manifest.json"),
        "predictions_sqlite": str(predictions),
        "elapsed_seconds": time.perf_counter() - group_started,
        "worker_pid": int(os.getpid()),
        "native_cpu_threads": int(job.curve_cpu_threads),
        "worker_progress_json": str(progress_path),
    }
    return int(job.index), routed, group_manifest


__all__ = (
    "CurveGroupJob",
    "available_cpu_count",
    "curve_track_costs",
    "partition_curve_tracks",
    "run_curve_group_process",
)
