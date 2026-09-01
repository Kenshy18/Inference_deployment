"""Global process-budget allocation for classwise curve routes."""

from __future__ import annotations


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


__all__ = ("allocate_curve_group_shards",)
