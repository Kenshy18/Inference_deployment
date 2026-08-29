#!/usr/bin/env python3
"""Measure full-corpus Catmull--Rom scheduler throughput and variance."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import time

from classwise.curve_parallel import curve_track_costs, partition_curve_tracks
from classwise.sqlite import read_track_labels
from production.config import PRODUCTION
from production.curve.config import CURVE_PRODUCTION
from production.curve.engine import run_curve_optimizer
from production.curve.preparation import prepare_curve_source
from production.polygon.runtime_bridge import build_runtime_config


WIDTH = 1920
HEIGHT = 1080
VIDEO_FRAMES = 23510


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-interval", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    return parser


def _worker(
    source_value: str,
    track_ids: tuple[str, ...],
    output_value: str,
    target_interval: int,
    native_threads: int,
) -> dict[str, object]:
    source = Path(source_value)
    output = Path(output_value)
    preparation_config = replace(
        PRODUCTION,
        target_interval=int(target_interval),
        interval_evaluation="native_exact",
    )
    curve_config = replace(
        CURVE_PRODUCTION,
        target_interval=int(target_interval),
        native_cpu_threads=int(native_threads),
    )
    started = time.perf_counter()
    preparation = prepare_curve_source(
        source,
        output / "preparation",
        width=WIDTH,
        height=HEIGHT,
        input_video=None,
        config=build_runtime_config(preparation_config),
        selected_track_ids=track_ids,
    )
    prepared_seconds = time.perf_counter() - started
    engine = run_curve_optimizer(
        source,
        preparation,
        output / "runtime",
        config=curve_config,
    )
    audit = engine["audit"]
    if int(audit["recall_violations"]) or int(audit["topology_invalid_frames"]):
        raise RuntimeError(f"curve audit failure: {audit}")
    return {
        "wall_seconds": float(time.perf_counter() - started),
        "prepared_seconds": float(prepared_seconds),
        "prediction_rows": int(engine["prediction_rows"]),
        "streams": int(engine["streams"]),
        "minimum_recall": float(audit["minimum_recall"]),
        "mean_iou": float(audit["mean_iou"]),
        "minimum_iou": float(audit["minimum_iou"]),
        "keyframes": int(audit["keyframes"]),
        "actual_mean_interval": float(audit["effective_interval"]),
    }


def _groups(source: Path, shards_per_class: int) -> tuple[tuple[str, ...], ...]:
    labels = read_track_labels(source)
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        counts = {
            str(track_id): int(count)
            for track_id, count in db.execute(
                "SELECT track_id,COUNT(*) FROM masks GROUP BY track_id"
            )
        }
        costs, _metric = curve_track_costs(db, counts)
    groups: list[tuple[str, ...]] = []
    for label in PRODUCTION.labels:
        ids = tuple(sorted(track for track, value in labels.items() if value == label))
        if ids:
            groups.extend(partition_curve_tracks(ids, costs, shards_per_class))
    return tuple(group for group in groups if group)


def _run_schedule(
    source: Path,
    output: Path,
    *,
    target_interval: int,
    shards_per_class: int,
    max_workers: int,
    native_threads: int,
) -> dict[str, object]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = _groups(source, shards_per_class)
    workers = min(int(max_workers), len(groups))
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _worker,
                str(source),
                track_ids,
                str(output / f"group_{index:02d}"),
                int(target_interval),
                int(native_threads),
            )
            for index, track_ids in enumerate(groups)
        ]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - started
    rows = sum(int(result["prediction_rows"]) for result in results)
    return {
        "wall_seconds": float(wall),
        "video_fps": float(VIDEO_FRAMES / max(wall, 1e-9)),
        "observation_fps": float(rows / max(wall, 1e-9)),
        "prediction_rows": int(rows),
        "groups": int(len(groups)),
        "process_workers": int(workers),
        "native_threads": int(native_threads),
        "maximum_native_threads": int(workers * native_threads),
        "minimum_recall": float(
            min(result["minimum_recall"] for result in results)
        ),
        "mean_iou": float(
            sum(result["mean_iou"] * result["prediction_rows"] for result in results)
            / max(rows, 1)
        ),
        "minimum_iou": float(min(result["minimum_iou"] for result in results)),
        "keyframes": int(sum(result["keyframes"] for result in results)),
        "actual_mean_interval": float(
            sum(
                result["actual_mean_interval"] * result["prediction_rows"]
                for result in results
            )
            / max(rows, 1)
        ),
        "worker_results": results,
    }


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    schedules = (
        ("production_2shards_6x4", 2, 6, 4),
        ("coarse_1shard_3x8", 1, 3, 8),
    )
    results: list[dict[str, object]] = []
    for repeat in range(1, int(args.repeats) + 1):
        for name, shards, workers, threads in schedules:
            print(f"[run] {name} repeat={repeat}", flush=True)
            measured = _run_schedule(
                source,
                root / "runs" / f"{name}_repeat{repeat}",
                target_interval=args.target_interval,
                shards_per_class=shards,
                max_workers=workers,
                native_threads=threads,
            )
            row = {
                "name": name,
                "repeat": int(repeat),
                "target_interval": int(args.target_interval),
                **measured,
            }
            results.append(row)
            print(
                f"[result] {name} repeat={repeat} "
                f"wall={measured['wall_seconds']:.3f}s "
                f"fps={measured['video_fps']:.2f} "
                f"iou={measured['mean_iou']:.6f} "
                f"recall={measured['minimum_recall']:.6f}",
                flush=True,
            )
            (root / "benchmark_results.json").write_text(
                json.dumps(
                    {"schema_version": 1, "source": str(source), "results": results},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
