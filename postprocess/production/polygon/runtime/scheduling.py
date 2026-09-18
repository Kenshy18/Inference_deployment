"""Hardware-aware worker budgeting for the polygon optimizer.

The numerical result is independent of this schedule.  The scheduler only
prevents nested class/process/OpenMP parallelism from oversubscribing the host.
Requested values remain hard upper bounds so diagnostics can still reduce
parallelism explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping


def available_cpu_count() -> int:
    """Return CPUs available to this process, including affinity limits."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


@dataclass(frozen=True, slots=True)
class PolygonRuntimeSchedule:
    cpu_count: int
    label_workers: int
    optimizer_workers_per_label: int
    native_threads_per_optimizer: int
    maximum_optimizer_processes: int
    maximum_native_threads: int


def balanced_polygon_schedule(
    *,
    cpu_count: int,
    label_count: int,
    requested_label_workers: int,
    requested_optimizer_workers: int,
    requested_native_threads: int,
) -> PolygonRuntimeSchedule:
    """Return a bounded nested-parallel schedule.

    Measurements on the 24-core deployment host found that approximately
    ``0.75 * CPUs`` optimizer processes and ``1.5 * CPUs`` native worker
    threads keep the CUDA/CPU pipeline saturated without the 4--6x exact-
    evaluator stalls seen under the former 3 x 9 x 4 schedule.
    """

    cpus = max(1, int(cpu_count))
    labels = max(1, int(label_count))
    label_workers = min(max(1, int(requested_label_workers)), labels, cpus)
    process_budget = max(label_workers, int(math.floor(0.75 * cpus)))
    optimizer_workers = min(
        max(1, int(requested_optimizer_workers)),
        max(1, process_budget // label_workers),
    )
    native_budget = max(1, int(math.floor(1.5 * cpus)))
    native_threads = min(
        max(1, int(requested_native_threads)),
        max(1, native_budget // (label_workers * optimizer_workers)),
    )
    return PolygonRuntimeSchedule(
        cpu_count=cpus,
        label_workers=label_workers,
        optimizer_workers_per_label=optimizer_workers,
        native_threads_per_optimizer=native_threads,
        maximum_optimizer_processes=label_workers * optimizer_workers,
        maximum_native_threads=label_workers * optimizer_workers * native_threads,
    )


def screened_adaptive_process_budget(*, cpu_count: int, active_label_count: int) -> int:
    """Return the memory-screened process budget for adaptive label scheduling.

    The 24-core deployment host saturated at six processes for one active
    label.  With two or three labels, eight processes improved the runtime
    floor while keeping at least 12 GiB available on the balanced KPI corpus;
    nine processes crossed the 10-GiB safety line.
    """

    cpus = max(1, int(cpu_count))
    labels = max(0, int(active_label_count))
    if labels == 0:
        return 0
    if labels == 1:
        return min(cpus, 6)
    # Eight is the largest budget that passed the balanced long-video memory
    # screen.  Do not infer a larger memory allowance merely from a larger CPU
    # affinity: deployment hosts can expose more cores without more WSL RAM.
    return max(labels, min(cpus, 8))


def allocate_label_workers(
    workloads: Mapping[str, int],
    *,
    process_budget: int,
) -> dict[str, int]:
    """Allocate a fixed process budget in proportion to prepared row counts."""

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
    return allocation


__all__ = (
    "PolygonRuntimeSchedule",
    "allocate_label_workers",
    "available_cpu_count",
    "balanced_polygon_schedule",
    "screened_adaptive_process_budget",
)
