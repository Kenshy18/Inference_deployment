#!/usr/bin/env python3
"""Benchmark per-label DP worker allocations with a fixed process budget.

This experiment keeps the optimizer and output format unchanged.  It only
changes how a fixed number of worker processes is divided across labels.
Each label is launched independently so allocations such as 4/1/1 can be
measured without modifying the Production coordinator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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

from production.config import LABELS, RUNTIME_POLYGON_PROFILE_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vertex-policy", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-frames", type=int, required=True)
    parser.add_argument(
        "--labels",
        default=",".join(LABELS),
        help="comma-separated active labels; defaults to all Production labels",
    )
    parser.add_argument(
        "--allocations",
        default="4,1,1",
        help=(
            "semicolon-separated worker allocations in LABELS order; "
            "for example 4,1,1;3,2,1"
        ),
    )
    parser.add_argument("--minimum-available-memory-mib", type=int, default=8192)
    return parser


def _available_memory_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("MemAvailable is missing")


def _tree_rss_mib(processes: list[psutil.Process]) -> float:
    seen: set[int] = set()
    total = 0
    for process in processes:
        for member in [process, *process.children(recursive=True)]:
            if member.pid in seen:
                continue
            seen.add(member.pid)
            try:
                total += int(member.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return total / (1024.0 * 1024.0)


def _environment(vertex_policy: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
            "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": "4",
            "MASK_PIPELINE_SPATIAL_VERTEX_POLICY_JSON": str(vertex_policy),
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_SECONDS": "0.5",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET": "0.10",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_AREA": "0",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_BUDGET": "0.10",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_EDGES": "1024",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO": "1.0",
            "MASK_PIPELINE_PHASE2_CANDIDATE_FRAME_WORKERS": "1",
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(POSTPROCESS), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment


def _command(
    source_root: Path,
    output_root: Path,
    label: str,
    workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(POSTPROCESS / "production/polygon/runtime/coordinator.py"),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--profiles",
        RUNTIME_POLYGON_PROFILE_ID,
        "--labels",
        label,
        "--target-interval",
        "3",
        "--recall-floor",
        "0.97",
        "--anchors-per-contour",
        "20",
        "--num-workers",
        str(workers),
        # Preserve the same OpenCV thread division as the 3-label baseline.
        "--label-workers",
        "3",
        "--native-batch-threads",
        "4",
        "--gc-interval",
        "8",
        "--max-run-frames",
        "30000",
        "--run-overlap-frames",
        "900",
        "--keyframe-max-gap",
        "30",
        "--cuda-lazy-exact",
        "--cuda-lazy-frame-hints",
        "--cuda-exact-hint-count",
        "8",
        "--pair-vote-per-key",
        "--pair-vote-sweeps",
        "2",
        "--force",
    ]


def _runtime_root(root: Path, label: str) -> Path:
    return root / label / RUNTIME_POLYGON_PROFILE_ID / label / "runtime"


def _prediction_rows(path: Path) -> list[tuple[int, str, str]]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        return [
            (int(frame), str(track), str(polygons))
            for frame, track, polygons in database.execute(
                "SELECT frame,track_id,polygons FROM masks "
                "ORDER BY frame,CAST(track_id AS INTEGER)"
            )
        ]


def _label_metrics(root: Path, label: str) -> dict[str, object]:
    runtime = _runtime_root(root, label)
    summary = json.loads((runtime / "opt/summary.json").read_text(encoding="utf-8"))
    with (runtime / "exact/keyframe_exact_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    recalls = [float(row["recall"]) for row in rows]
    ious = [float(row["iou"]) for row in rows]
    return {
        "rows": len(rows),
        "runs": int(summary["run_count"]),
        "optimizer_seconds": float(summary["optimizer_seconds"]),
        "minimum_recall": min(recalls) if recalls else 1.0,
        "mean_iou": sum(ious) / max(len(ious), 1),
        "predictions": str(runtime / "pred/predictions.sqlite"),
        "keyframes": str(runtime / "opt/final_keyframes.json"),
    }


def _baseline_runtime(root: Path, label: str) -> Path:
    return root / RUNTIME_POLYGON_PROFILE_ID / label / "runtime"


def _compare_label(
    baseline_root: Path, candidate: dict[str, object], label: str
) -> dict:
    baseline = _baseline_runtime(baseline_root, label)
    candidate_predictions = Path(str(candidate["predictions"]))
    candidate_keyframes = Path(str(candidate["keyframes"]))
    return {
        "prediction_rows_exact_equal": _prediction_rows(
            baseline / "pred/predictions.sqlite"
        )
        == _prediction_rows(candidate_predictions),
        "keyframes_bytes_exact_equal": (
            baseline / "opt/final_keyframes.json"
        ).read_bytes()
        == candidate_keyframes.read_bytes(),
    }


def _run_allocation(args: argparse.Namespace, allocation: tuple[int, ...]) -> dict:
    name = "workers_" + "_".join(map(str, allocation))
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    environment = _environment(args.vertex_policy)
    processes: list[subprocess.Popen[str]] = []
    logs = []
    started = time.perf_counter()
    try:
        for label, workers in zip(args.active_labels, allocation, strict=True):
            label_root = root / label
            label_root.mkdir(parents=True, exist_ok=True)
            log = (root / f"{label}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    _command(args.source_root, label_root, label, workers),
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        ps_processes = [psutil.Process(process.pid) for process in processes]
        peak_rss = 0.0
        minimum_available = _available_memory_mib()
        while any(process.poll() is None for process in processes):
            peak_rss = max(peak_rss, _tree_rss_mib(ps_processes))
            available = _available_memory_mib()
            minimum_available = min(minimum_available, available)
            if available < args.minimum_available_memory_mib:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                raise MemoryError(
                    f"{name} crossed MemAvailable floor: {available:.1f} MiB"
                )
            time.sleep(0.25)
        returncodes = [process.wait() for process in processes]
    finally:
        for log in logs:
            log.close()
    wall = time.perf_counter() - started
    if any(returncodes):
        failures = []
        for label, returncode in zip(args.active_labels, returncodes, strict=True):
            if returncode:
                log_path = root / f"{label}.log"
                failures.append(
                    f"{label} rc={returncode}\n"
                    + "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
                )
        raise RuntimeError("\n".join(failures))
    labels = {label: _label_metrics(root, label) for label in args.active_labels}
    comparisons = {
        label: _compare_label(args.baseline_root, labels[label], label)
        for label in args.active_labels
    }
    recalls = [float(values["minimum_recall"]) for values in labels.values()]
    weighted_iou_numerator = sum(
        int(values["rows"]) * float(values["mean_iou"]) for values in labels.values()
    )
    row_count = sum(int(values["rows"]) for values in labels.values())
    return {
        "name": name,
        "allocation": dict(zip(args.active_labels, allocation, strict=True)),
        "maximum_processes": sum(allocation),
        "wall_seconds": wall,
        "video_fps": args.video_frames / wall,
        "peak_process_tree_rss_mib": peak_rss,
        "minimum_available_memory_mib": minimum_available,
        "minimum_recall": min(recalls),
        "mean_iou": weighted_iou_numerator / max(row_count, 1),
        "labels": labels,
        "comparisons": comparisons,
        "all_outputs_exact_equal": all(
            value["prediction_rows_exact_equal"]
            and value["keyframes_bytes_exact_equal"]
            for value in comparisons.values()
        ),
        "root": str(root),
    }


def main() -> int:
    args = _parser().parse_args()
    args.source_root = args.source_root.resolve()
    args.vertex_policy = args.vertex_policy.resolve()
    args.baseline_root = args.baseline_root.resolve()
    args.output = args.output.resolve()
    args.active_labels = [
        value.strip() for value in args.labels.split(",") if value.strip()
    ]
    if not args.active_labels or any(
        label not in LABELS for label in args.active_labels
    ):
        raise ValueError(f"labels must be selected from {LABELS}")
    allocations = []
    for raw in (part.strip() for part in args.allocations.split(";") if part.strip()):
        allocation = tuple(int(value) for value in raw.split(","))
        if len(allocation) != len(args.active_labels) or min(allocation) < 1:
            raise ValueError(f"invalid allocation: {raw}")
        allocations.append(allocation)
    results = [_run_allocation(args, allocation) for allocation in allocations]
    payload = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "video_frames": args.video_frames,
        "baseline_root": str(args.baseline_root),
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "benchmark_results.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
