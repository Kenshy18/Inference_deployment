"""Global process-budget allocation for classwise curve routes."""

from __future__ import annotations

from .curve_parallel import partition_curve_tracks
from .policy import ClassPostprocessSettings


def build_classwise_work_groups(
    semantic_groups: list[tuple[str, ClassPostprocessSettings]],
    tracks_by_group: dict[tuple[str, ClassPostprocessSettings], list[str]],
    *,
    geometry_mode: str,
    curve_costs_by_track: dict[str, int],
    curve_shards_by_group: tuple[int, ...],
) -> list[tuple[str, ClassPostprocessSettings, int, tuple[str, ...]]]:
    """Expand semantic groups into deterministic curve shards or polygon routes."""

    work_groups: list[
        tuple[str, ClassPostprocessSettings, int, tuple[str, ...]]
    ] = []
    for index, (label, settings) in enumerate(semantic_groups):
        track_ids = tuple(tracks_by_group[(label, settings)])
        partitions = (
            partition_curve_tracks(
                track_ids,
                curve_costs_by_track,
                curve_shards_by_group[index],
            )
            if geometry_mode == "catmull_rom"
            else (track_ids,)
        )
        work_groups.extend(
            (label, settings, shard_index, partition)
            for shard_index, partition in enumerate(partitions)
        )
    return work_groups


def allocate_curve_group_shards(
    tracks_by_group: tuple[tuple[str, ...], ...],
    costs: dict[str, int],
    *,
    process_budget: int,
    maximum_shards_per_group: int = 8,
) -> tuple[int, ...]:
    """Allocate spare workers to the busiest semantic route.

    Every semantic group keeps one route so its keyframe policy remains
    isolated. Remaining routes go to the group with the largest estimated
    work per current shard.
    """

    if not tracks_by_group:
        return ()
    budget = max(1, int(process_budget))
    cap = max(1, int(maximum_shards_per_group))
    shards = [1 for _tracks in tracks_by_group]
    maximum = [min(cap, max(1, len(tracks))) for tracks in tracks_by_group]
    totals = [
        sum(max(1, int(costs.get(str(track_id), 1))) for track_id in tracks)
        for tracks in tracks_by_group
    ]
    remaining = max(0, budget - len(tracks_by_group))
    while remaining:
        eligible = [
            index
            for index in range(len(shards))
            if shards[index] < maximum[index]
        ]
        if not eligible:
            break
        selected = max(
            eligible,
            key=lambda index: (
                float(totals[index]) / float(shards[index]),
                -index,
            ),
        )
        shards[selected] += 1
        remaining -= 1
    return tuple(int(value) for value in shards)


def resolve_curve_cpu_threads(
    requested_threads: int,
    *,
    available_cpus: int,
    concurrent_workers: int,
) -> int:
    """Share deterministic native exact threads between concurrent shards."""

    if requested_threads > 0:
        return int(requested_threads)
    return max(1, min(12, available_cpus // max(concurrent_workers, 1)))


__all__ = (
    "allocate_curve_group_shards",
    "build_classwise_work_groups",
    "resolve_curve_cpu_threads",
)
