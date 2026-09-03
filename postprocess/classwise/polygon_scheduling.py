"""Allocate one bounded polygon worker budget across classwise routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from production.polygon.runtime.scheduling import (
    allocate_label_workers,
    screened_adaptive_process_budget,
)

from .policy import ClassPostprocessSettings


PolygonWorkGroup = tuple[str, ClassPostprocessSettings, int, tuple[str, ...]]


@dataclass(frozen=True)
class PolygonWorkerPlan:
    """Globally screened process allocation for isolated semantic routes."""

    workers_by_group: dict[int, int]
    process_budget: int
    screened_process_budget: int

    def worker_count(self, group_index: int) -> int:
        return int(self.workers_by_group[group_index])


def plan_polygon_group_workers(
    work_groups: Sequence[PolygonWorkGroup],
    mask_counts_by_track: Mapping[str, int],
    geometry_options: Mapping[str, object],
    *,
    available_cpus: int,
) -> PolygonWorkerPlan:
    """Distribute the host-safe polygon process budget by observation load."""

    if not work_groups:
        return PolygonWorkerPlan({}, 0, 0)

    configured_cap = max(1, int(geometry_options.get("optimizer_workers", 9)))
    maximum_single_group = screened_adaptive_process_budget(
        cpu_count=available_cpus,
        active_label_count=1,
    )
    screened_budget = min(
        screened_adaptive_process_budget(
            cpu_count=available_cpus,
            active_label_count=len(work_groups),
        ),
        configured_cap * len(work_groups),
    )
    workloads = {
        str(index): sum(
            int(mask_counts_by_track.get(track_id, 0)) for track_id in track_ids
        )
        for index, (_label, _settings, _shard, track_ids) in enumerate(work_groups)
    }
    allocated = allocate_label_workers(workloads, process_budget=screened_budget)
    workers_by_group = {
        index: max(
            1,
            min(
                configured_cap,
                maximum_single_group,
                int(allocated.get(str(index), 1)),
            ),
        )
        for index in range(len(work_groups))
    }
    return PolygonWorkerPlan(
        workers_by_group=workers_by_group,
        process_budget=sum(workers_by_group.values()),
        screened_process_budget=screened_budget,
    )


def polygon_worker_manifest(
    plan: PolygonWorkerPlan,
    work_groups: Sequence[PolygonWorkGroup],
    mask_counts_by_track: Mapping[str, int],
) -> list[dict[str, object]]:
    """Describe the effective allocation without exposing scheduler internals."""

    return [
        {
            "group_index": index,
            "label": str(group[0]),
            "input_masks": sum(
                int(mask_counts_by_track.get(track_id, 0)) for track_id in group[3]
            ),
            "optimizer_workers": plan.worker_count(index),
        }
        for index, group in enumerate(work_groups)
    ]


__all__ = [
    "PolygonWorkGroup",
    "PolygonWorkerPlan",
    "plan_polygon_group_workers",
    "polygon_worker_manifest",
]
