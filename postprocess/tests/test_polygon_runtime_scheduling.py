from __future__ import annotations

from production.polygon.runtime import scheduling
from production.polygon.runtime.scheduling import balanced_polygon_schedule


def test_balanced_schedule_matches_24_core_three_label_budget() -> None:
    schedule = balanced_polygon_schedule(
        cpu_count=24,
        label_count=3,
        requested_label_workers=3,
        requested_optimizer_workers=9,
        requested_native_threads=4,
    )
    assert schedule.label_workers == 3
    assert schedule.optimizer_workers_per_label == 6
    assert schedule.native_threads_per_optimizer == 2
    assert schedule.maximum_optimizer_processes == 18
    assert schedule.maximum_native_threads == 36


def test_balanced_schedule_uses_spare_capacity_for_one_label() -> None:
    schedule = balanced_polygon_schedule(
        cpu_count=24,
        label_count=1,
        requested_label_workers=3,
        requested_optimizer_workers=9,
        requested_native_threads=4,
    )
    assert schedule.label_workers == 1
    assert schedule.optimizer_workers_per_label == 9
    assert schedule.native_threads_per_optimizer == 4


def test_balanced_schedule_never_exceeds_explicit_limits() -> None:
    schedule = balanced_polygon_schedule(
        cpu_count=24,
        label_count=3,
        requested_label_workers=2,
        requested_optimizer_workers=3,
        requested_native_threads=1,
    )
    assert schedule.label_workers == 2
    assert schedule.optimizer_workers_per_label == 3
    assert schedule.native_threads_per_optimizer == 1


def test_available_cpu_count_honours_process_affinity(monkeypatch) -> None:
    monkeypatch.setattr(scheduling.os, "sched_getaffinity", lambda _pid: {2, 4, 6})
    monkeypatch.setattr(scheduling.os, "cpu_count", lambda: 64)
    assert scheduling.available_cpu_count() == 3


def test_available_cpu_count_falls_back_when_affinity_is_unavailable(
    monkeypatch,
) -> None:
    def unavailable(_pid: int) -> set[int]:
        raise OSError("affinity unavailable")

    monkeypatch.setattr(scheduling.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(scheduling.os, "cpu_count", lambda: 12)
    assert scheduling.available_cpu_count() == 12


def test_balanced_schedule_does_not_oversubscribe_low_core_label_workers() -> None:
    one_cpu = balanced_polygon_schedule(
        cpu_count=1,
        label_count=3,
        requested_label_workers=3,
        requested_optimizer_workers=9,
        requested_native_threads=4,
    )
    two_cpus = balanced_polygon_schedule(
        cpu_count=2,
        label_count=3,
        requested_label_workers=3,
        requested_optimizer_workers=9,
        requested_native_threads=4,
    )
    assert one_cpu.label_workers == 1
    assert one_cpu.maximum_optimizer_processes == 1
    assert one_cpu.maximum_native_threads == 1
    assert two_cpus.label_workers == 2
    assert two_cpus.maximum_optimizer_processes == 2
    assert two_cpus.maximum_native_threads == 2
