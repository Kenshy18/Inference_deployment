#!/usr/bin/env python3
"""Compare exact polygon worker transports on many short runs.

The numerical kernel and all Production optimization parameters are identical.
Only multiprocessing transport changes:

* spawn: pickle each InstanceRun into a fresh worker process;
* fork: inherit immutable InstanceRuns copy-on-write and send integer indexes.

The parent runs variants serially, samples aggregate process-tree RSS, and
enforces a system-memory floor so this experiment cannot repeat the earlier WSL
memory exhaustion incident.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from production.config import PRODUCTION, RUNTIME_POLYGON_PROFILE_ID
from production.polygon.materialize import materialize_outputs
from production.polygon.runtime_bridge import (
    build_runtime_config,
    optimize,
    prepare_inputs,
)


LABEL = "女性器"
WIDTH = 1920
HEIGHT = 1080
SOURCE_TRACK = "26"
OBSERVATIONS = 480


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-count", type=int, default=60)
    parser.add_argument("--optimizer-workers", type=int, default=6)
    parser.add_argument("--minimum-available-memory-mib", type=int, default=8192)
    parser.add_argument("--methods", default="spawn,fork")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--policy", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("spawn", "fork"), help=argparse.SUPPRESS)
    return parser


def _create_many_short_runs(source: Path, output: Path, track_count: int) -> None:
    if track_count < 2 or track_count > OBSERVATIONS:
        raise ValueError(f"track-count must be in [2,{OBSERVATIONS}]")
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as database:
        rows = database.execute(
            "SELECT frame,polygons FROM masks WHERE track_id=? ORDER BY frame LIMIT ?",
            (SOURCE_TRACK, OBSERVATIONS),
        ).fetchall()
    if len(rows) != OBSERVATIONS:
        raise RuntimeError(f"expected {OBSERVATIONS} source rows, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with sqlite3.connect(output) as database:
        database.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                PRIMARY KEY(frame,track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY,label TEXT);
            """
        )
        generated = []
        for index, (_frame, polygons) in enumerate(rows):
            partition = min(track_count - 1, index * track_count // OBSERVATIONS)
            track_id = str(10000 + partition)
            # Each short track gets a local contiguous timeline.  The optimizer
            # cost depends on run length and geometry, not absolute video time.
            local_frame = index - math.floor(partition * OBSERVATIONS / track_count)
            generated.append((local_frame, track_id, str(polygons), LABEL))
        database.executemany(
            """
            INSERT INTO masks(
                frame,track_id,polygons,shape_type,dilate_px,feather_px,
                mosaic_block,mosaic_alias,label
            ) VALUES (?,?,?,'polygon',0,0,0,0,?)
            """,
            generated,
        )
        track_ids = sorted({row[1] for row in generated}, key=int)
        database.executemany(
            "INSERT INTO tracks(track_id,label) VALUES (?,?)",
            ((track_id, LABEL) for track_id in track_ids),
        )
        database.commit()
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("generated source failed SQLite integrity_check")


def _force_vertices(policy_path: Path, vertices: int = 14) -> None:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    tracks = policy.get("tracks", {})
    for value in tracks.values():
        value["vertices_per_component"] = int(vertices)
    allowed = [14, 16, 18, 20]
    rows = sum(int(value.get("rows", 0)) for value in tracks.values())
    summary = policy.setdefault("summary", {})
    summary["tracks_by_vertices"] = {
        str(point_count): len(tracks) if point_count == vertices else 0
        for point_count in allowed
    }
    summary["track_rows_by_vertices"] = {
        str(point_count): rows if point_count == vertices else 0
        for point_count in allowed
    }
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _memory_available_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _tree_rss_mib(process: psutil.Process) -> float:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += int(member.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024.0 * 1024.0)


def _run_variant(args: argparse.Namespace, method: str, prepared: Path, policy: Path) -> dict:
    variant = args.output / method
    if variant.exists():
        shutil.rmtree(variant)
    result_path = variant / "result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source", str(args.source),
        "--output", str(variant),
        "--track-count", str(args.track_count),
        "--optimizer-workers", str(args.optimizer_workers),
        "--worker",
        "--source-root", str(prepared),
        "--policy", str(policy),
        "--method", method,
    ]
    environment = os.environ.copy()
    environment["MASK_PIPELINE_POLYGON_WORKER_START_METHOD"] = method
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    ps_process = psutil.Process(process.pid)
    peak_rss = 0.0
    minimum_available = _memory_available_mib()
    log_lines: list[str] = []
    try:
        while process.poll() is None:
            peak_rss = max(peak_rss, _tree_rss_mib(ps_process))
            available = _memory_available_mib()
            minimum_available = min(minimum_available, available)
            if available < args.minimum_available_memory_mib:
                os.killpg(process.pid, signal.SIGTERM)
                raise MemoryError(
                    f"{method} crossed MemAvailable floor: {available:.1f} MiB"
                )
            time.sleep(0.25)
        if process.stdout is not None:
            log_lines.extend(process.stdout.read().splitlines())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    (variant / "worker.log").parent.mkdir(parents=True, exist_ok=True)
    (variant / "worker.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{method} worker failed:\n" + "\n".join(log_lines[-50:]))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "parent_wall_seconds": time.perf_counter() - started,
            "peak_process_tree_rss_mib": peak_rss,
            "minimum_available_memory_mib": minimum_available,
        }
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _worker(args: argparse.Namespace) -> int:
    if args.source_root is None or args.policy is None or args.method is None:
        raise ValueError("worker arguments are incomplete")
    config = replace(PRODUCTION, target_interval=3)
    started = time.perf_counter()
    optimizer = optimize(
        args.source_root,
        args.output / "optimizer",
        labels=(LABEL,),
        max_tracks=0,
        force=True,
        config=config,
        optimizer_workers=args.optimizer_workers,
    )
    optimizer_wall = time.perf_counter() - started
    materialize_started = time.perf_counter()
    runtime = build_runtime_config(config, optimizer_workers=args.optimizer_workers)
    predictions = args.output / "predictions.sqlite"
    keyframes = args.output / "keyframes.sqlite"
    materialize_outputs(
        Path(str(optimizer["phase2_root"])),
        args.output.parent / "source.sqlite",
        predictions,
        keyframes,
        config=config,
        runtime_profile=runtime.polygon_profile_id,
    )
    materialize_wall = time.perf_counter() - materialize_started
    runtime_root = (
        Path(str(optimizer["phase2_root"]))
        / RUNTIME_POLYGON_PROFILE_ID
        / LABEL
        / "runtime"
    )
    summary = json.loads(
        (runtime_root / "opt" / "summary.json").read_text(encoding="utf-8")
    )
    with (runtime_root / "exact" / "keyframe_exact_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        recalls = [float(row["recall"]) for row in csv.DictReader(handle)]
    payload = {
        "method": args.method,
        "track_count": args.track_count,
        "observations": OBSERVATIONS,
        "optimizer_workers": args.optimizer_workers,
        "optimizer_wall_seconds": optimizer_wall,
        "materialize_wall_seconds": materialize_wall,
        "optimizer_seconds": summary["optimizer_seconds"],
        "worker_start_method": summary["worker_start_method"],
        "worker_transfer_mode": summary["worker_transfer_mode"],
        "run_count": summary["run_count"],
        "minimum_recall": min(recalls) if recalls else 1.0,
        "predictions": str(predictions),
        "final_keyframes": str(runtime_root / "opt" / "final_keyframes.json"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def _prediction_rows(path: Path) -> list[tuple[int, str, str]]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        return [
            (int(frame), str(track_id), str(polygons))
            for frame, track_id, polygons in database.execute(
                "SELECT frame,track_id,polygons FROM masks "
                "ORDER BY frame,CAST(track_id AS INTEGER)"
            )
        ]


def main() -> int:
    args = _parser().parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.worker:
        return _worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "source.sqlite"
    _create_many_short_runs(args.source, generated, args.track_count)
    config = replace(PRODUCTION, target_interval=3)
    prepared_root, _preparation = prepare_inputs(
        generated,
        args.output / "preparation",
        width=WIDTH,
        height=HEIGHT,
        input_video=None,
        config=config,
    )
    prepared_root = Path(prepared_root).resolve()
    policy = prepared_root / "vertex_policy.json"
    _force_vertices(policy)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if set(methods) - {"spawn", "fork"}:
        raise ValueError(f"unsupported methods: {methods}")
    results = [_run_variant(args, method, prepared_root, policy) for method in methods]
    comparison = None
    if {result["method"] for result in results} == {"spawn", "fork"}:
        by_method = {result["method"]: result for result in results}
        spawn = by_method["spawn"]
        fork = by_method["fork"]
        prediction_equal = _prediction_rows(Path(spawn["predictions"])) == _prediction_rows(
            Path(fork["predictions"])
        )
        keyframes_equal = Path(spawn["final_keyframes"]).read_bytes() == Path(
            fork["final_keyframes"]
        ).read_bytes()
        comparison = {
            "prediction_rows_exact_equal": prediction_equal,
            "final_keyframes_bytes_exact_equal": keyframes_equal,
            "optimizer_speedup": spawn["optimizer_wall_seconds"]
            / fork["optimizer_wall_seconds"],
            "parent_speedup": spawn["parent_wall_seconds"]
            / fork["parent_wall_seconds"],
            "peak_rss_change_pct": 100.0
            * (
                fork["peak_process_tree_rss_mib"]
                / spawn["peak_process_tree_rss_mib"]
                - 1.0
            ),
        }
    payload = {
        "schema_version": 1,
        "source": str(args.source),
        "generated_source": str(generated),
        "track_count": args.track_count,
        "observations": OBSERVATIONS,
        "results": results,
        "comparison": comparison,
    }
    destination = args.output / "benchmark_results.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
