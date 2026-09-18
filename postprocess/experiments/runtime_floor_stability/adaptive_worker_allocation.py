"""Deterministic fixed-budget worker allocation for runtime-floor experiments."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Mapping


def count_observations(source: Path) -> int:
    """Return the prepared observation count without loading mask geometry."""

    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as database:
        row = database.execute("SELECT COUNT(*) FROM masks").fetchone()
    return int(row[0] if row else 0)


def allocate_fixed_budget(
    workloads: Mapping[str, int],
    *,
    process_budget: int,
) -> dict[str, int]:
    """Allocate a fixed process budget proportionally with one slot per label.

    Empty labels receive no process.  Active labels first receive one process;
    remaining processes are assigned one at a time to the label furthest below
    its ideal proportional allocation.  Stable input order breaks exact ties,
    making the schedule deterministic.
    """

    budget = int(process_budget)
    if budget < 1:
        raise ValueError("process_budget must be >= 1")
    active = [(str(label), int(rows)) for label, rows in workloads.items() if rows > 0]
    if not active:
        return {}
    if len(active) > budget:
        raise ValueError("process_budget must cover every active label")
    total = sum(rows for _, rows in active)
    ideal = {label: budget * rows / total for label, rows in active}
    allocation = {label: 1 for label, _ in active}
    stable_order = {label: index for index, (label, _) in enumerate(active)}
    for _ in range(budget - len(active)):
        label = max(
            (label for label, _ in active),
            key=lambda value: (
                ideal[value] - allocation[value],
                -stable_order[value],
            ),
        )
        allocation[label] += 1
    if sum(allocation.values()) != budget:
        raise AssertionError("worker allocation did not preserve the process budget")
    return allocation


def recommended_process_budget(*, cpu_count: int, active_label_count: int) -> int:
    """Return the screened host budget used by the current experiment.

    On the 24-core deployment host, eight processes were the fastest budget
    that kept at least 12 GiB available on the balanced KPI corpus.  Nine
    processes crossed the 10 GiB stop line.  A single active label saturated at
    six processes, so giving it the full eight only increased memory.
    """

    cpus = max(1, int(cpu_count))
    labels = max(0, int(active_label_count))
    if labels == 0:
        return 0
    shared_budget = max(labels, cpus // 3)
    if labels == 1:
        return max(1, min(shared_budget, cpus // 4))
    return shared_budget


def allocation_imbalance(
    workloads: Mapping[str, int], allocation: Mapping[str, int]
) -> float:
    """Return the largest rows-per-worker divided by the smallest positive one."""

    loads = [
        int(rows) / int(allocation[label])
        for label, rows in workloads.items()
        if rows > 0 and int(allocation.get(label, 0)) > 0
    ]
    if len(loads) < 2:
        return 1.0
    return max(loads) / min(loads)


__all__ = (
    "allocate_fixed_budget",
    "allocation_imbalance",
    "count_observations",
    "recommended_process_budget",
)
