from __future__ import annotations

import argparse
from pathlib import Path

from production.polygon.runtime import run as runtime_run
from production.polygon.runtime import scheduling
from production.polygon.runtime.scheduling import (
    allocate_label_workers,
    balanced_polygon_schedule,
    screened_adaptive_process_budget,
)


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


def test_adaptive_budget_matches_screened_24_core_limit() -> None:
    assert screened_adaptive_process_budget(cpu_count=24, active_label_count=3) == 8
    assert screened_adaptive_process_budget(cpu_count=24, active_label_count=2) == 8
    assert screened_adaptive_process_budget(cpu_count=24, active_label_count=1) == 6


def test_adaptive_budget_does_not_scale_past_memory_screened_cap() -> None:
    assert screened_adaptive_process_budget(cpu_count=64, active_label_count=3) == 8
    assert screened_adaptive_process_budget(cpu_count=64, active_label_count=1) == 6
    assert screened_adaptive_process_budget(cpu_count=4, active_label_count=3) == 4


def test_adaptive_allocation_tracks_prepared_row_load() -> None:
    assert allocate_label_workers(
        {"女性器": 21916, "男性器": 2634, "結合部分": 4131},
        process_budget=8,
    ) == {"女性器": 6, "男性器": 1, "結合部分": 1}
    assert allocate_label_workers(
        {"女性器": 9132, "男性器": 6499, "結合部分": 8872},
        process_budget=8,
    ) == {"女性器": 3, "男性器": 2, "結合部分": 3}
    assert allocate_label_workers(
        {"女性器": 1939, "男性器": 64},
        process_budget=8,
    ) == {"女性器": 7, "男性器": 1}


def _run_arguments(*, labels: str, adaptive: bool) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=Path("/tmp/source"),
        output_root=Path("/tmp/output"),
        labels=labels,
        label_workers=3,
        num_workers=9,
        adaptive_worker_allocation=adaptive,
        total_worker_budget=0,
        pair_vote_threads=4,
        native_batch_threads=4,
        interval_evaluation="cuda_lazy_exact",
        cuda_lazy_frame_hints=True,
        cuda_exact_hint_count=8,
        max_tracks=0,
        force=False,
        profile=runtime_run.ADAPTIVE_PROFILE_ID,
    )


def _run_arguments_with_cap(*, labels: str, cap: int) -> argparse.Namespace:
    arguments = _run_arguments(labels=labels, adaptive=True)
    arguments.num_workers = cap
    return arguments


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_adaptive_run_command_uses_screened_budget(monkeypatch) -> None:
    monkeypatch.setattr(runtime_run, "available_cpu_count", lambda: 24)
    command = runtime_run.build_command(
        _run_arguments(labels="女性器,男性器,結合部分", adaptive=True),
        3,
        Path("/tmp/run"),
    )
    assert "--adaptive-worker-allocation" in command
    assert _command_value(command, "--total-worker-budget") == "8"
    assert _command_value(command, "--num-workers") == "9"
    assert _command_value(command, "--native-batch-threads") == "4"


def test_adaptive_single_label_command_caps_at_six(monkeypatch) -> None:
    monkeypatch.setattr(runtime_run, "available_cpu_count", lambda: 24)
    command = runtime_run.build_command(
        _run_arguments(labels="男性器", adaptive=True),
        3,
        Path("/tmp/run"),
    )
    assert _command_value(command, "--total-worker-budget") == "6"


def test_adaptive_budget_respects_user_per_label_cap(monkeypatch) -> None:
    monkeypatch.setattr(runtime_run, "available_cpu_count", lambda: 24)
    command = runtime_run.build_command(
        _run_arguments_with_cap(labels="男性器", cap=3),
        3,
        Path("/tmp/run"),
    )
    assert _command_value(command, "--total-worker-budget") == "3"
    assert _command_value(command, "--num-workers") == "3"


def test_nonadaptive_run_command_preserves_equal_schedule(monkeypatch) -> None:
    monkeypatch.setattr(runtime_run, "available_cpu_count", lambda: 24)
    command = runtime_run.build_command(
        _run_arguments(labels="女性器,男性器,結合部分", adaptive=False),
        3,
        Path("/tmp/run"),
    )
    assert "--adaptive-worker-allocation" not in command
    assert _command_value(command, "--num-workers") == "6"
    assert _command_value(command, "--native-batch-threads") == "2"
