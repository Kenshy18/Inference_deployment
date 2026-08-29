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


__all__ = (
    "PolygonRuntimeSchedule",
    "available_cpu_count",
    "balanced_polygon_schedule",
)
