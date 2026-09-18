#!/usr/bin/env python3
"""Benchmark equal-process polygon label/optimizer layouts on prepared data."""

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
    parser.add_argument("--baseline-wall-seconds", type=float, required=True)
    parser.add_argument("--baseline-peak-rss-mib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-frames", type=int, required=True)
    parser.add_argument("--layouts", default="2x3,1x6")
    parser.add_argument(
        "--labels",
        default=",".join(LABELS),
        help="comma-separated active labels; defaults to all Production labels",
    )
    parser.add_argument("--minimum-available-memory-mib", type=int, default=8192)
    return parser


def _available_memory_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("MemAvailable is missing")


def _tree_rss_mib(process: psutil.Process) -> float:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += int(member.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024.0 * 1024.0)


def _runtime_root(root: Path, label: str) -> Path:
    return root / RUNTIME_POLYGON_PROFILE_ID / label / "runtime"


def _metrics(
    root: Path,
    wall: float,
    peak_rss: float,
    minimum_available: float,
    labels: list[str],
) -> dict:
    label_rows: dict[str, dict[str, object]] = {}
    all_recalls: list[float] = []
    all_ious: list[float] = []
    for label in labels:
        runtime = _runtime_root(root, label)
        summary = json.loads((runtime / "opt/summary.json").read_text(encoding="utf-8"))
        with (runtime / "exact/keyframe_exact_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        recalls = [float(row["recall"]) for row in rows]
        ious = [float(row["iou"]) for row in rows]
        all_recalls.extend(recalls)
        all_ious.extend(ious)
        label_rows[label] = {
            "rows": len(rows),
            "runs": int(summary["run_count"]),
            "optimizer_seconds": float(summary["optimizer_seconds"]),
            "worker_start_method": summary["worker_start_method"],
            "worker_transfer_mode": summary["worker_transfer_mode"],
            "minimum_recall": min(recalls) if recalls else 1.0,
            "mean_iou": sum(ious) / max(len(ious), 1),
            "predictions": str(runtime / "pred/predictions.sqlite"),
            "keyframes": str(runtime / "opt/final_keyframes.json"),
        }
    return {
        "wall_seconds": float(wall),
        "video_fps": 0.0,
        "peak_process_tree_rss_mib": float(peak_rss),
        "minimum_available_memory_mib": float(minimum_available),
        "minimum_recall": min(all_recalls) if all_recalls else 1.0,
        "mean_iou": sum(all_ious) / max(len(all_ious), 1),
        "labels": label_rows,
    }


def _run(args: argparse.Namespace, label_workers: int, optimizer_workers: int) -> dict:
    name = f"label{label_workers}_optimizer{optimizer_workers}"
    output = args.output / name
    command = [
        sys.executable,
        str(POSTPROCESS / "production/polygon/runtime/coordinator.py"),
        "--source-root",
        str(args.source_root),
        "--output-root",
        str(output),
        "--profiles",
        RUNTIME_POLYGON_PROFILE_ID,
        "--labels",
        ",".join(args.active_labels),
        "--target-interval",
        "3",
        "--recall-floor",
        "0.97",
        "--anchors-per-contour",
        "20",
        "--num-workers",
        str(optimizer_workers),
        "--label-workers",
        str(label_workers),
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
    environment = os.environ.copy()
    environment.update(
        {
            "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
            "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": "4",
            "MASK_PIPELINE_SPATIAL_VERTEX_POLICY_JSON": str(args.vertex_policy),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output / f"{name}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        ps_process = psutil.Process(process.pid)
        peak_rss = 0.0
        minimum_available = _available_memory_mib()
        while process.poll() is None:
            peak_rss = max(peak_rss, _tree_rss_mib(ps_process))
            available = _available_memory_mib()
            minimum_available = min(minimum_available, available)
            if available < args.minimum_available_memory_mib:
                os.killpg(process.pid, signal.SIGTERM)
                raise MemoryError(
                    f"{name} crossed MemAvailable floor: {available:.1f} MiB"
                )
            time.sleep(0.25)
        returncode = process.wait()
    wall = time.perf_counter() - started
    if returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-50:])
        raise RuntimeError(f"{name} failed:\n{tail}")
    result = _metrics(output, wall, peak_rss, minimum_available, args.active_labels)
    result.update(
        {
            "name": name,
            "label_workers": label_workers,
            "optimizer_workers_per_label": optimizer_workers,
            "maximum_processes": label_workers * optimizer_workers,
            "video_fps": args.video_frames / wall,
            "root": str(output),
        }
    )
    return result


def _prediction_rows(path: Path) -> list[tuple[int, str, str]]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        return [
            (int(frame), str(track), str(polygons))
            for frame, track, polygons in database.execute(
                "SELECT frame,track_id,polygons FROM masks "
                "ORDER BY frame,CAST(track_id AS INTEGER)"
            )
        ]


def _compare(base: dict, candidate: dict, labels_to_compare: list[str]) -> dict:
    labels = {}
    for label in labels_to_compare:
        left = base["labels"][label]
        right = candidate["labels"][label]
        labels[label] = {
            "prediction_rows_exact_equal": _prediction_rows(Path(left["predictions"]))
            == _prediction_rows(Path(right["predictions"])),
            "keyframes_bytes_exact_equal": Path(left["keyframes"]).read_bytes()
            == Path(right["keyframes"]).read_bytes(),
        }
    return {
        "wall_speedup": base["wall_seconds"] / candidate["wall_seconds"],
        "fps_change_pct": 100.0 * (candidate["video_fps"] / base["video_fps"] - 1.0),
        "peak_rss_change_pct": 100.0
        * (
            candidate["peak_process_tree_rss_mib"] / base["peak_process_tree_rss_mib"]
            - 1.0
        ),
        "labels": labels,
        "all_outputs_exact_equal": all(
            value["prediction_rows_exact_equal"]
            and value["keyframes_bytes_exact_equal"]
            for value in labels.values()
        ),
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
    baseline = _metrics(
        args.baseline_root,
        args.baseline_wall_seconds,
        args.baseline_peak_rss_mib,
        float("nan"),
        args.active_labels,
    )
    baseline.update(
        {
            "name": f"label{len(args.active_labels)}_optimizer2_baseline",
            "label_workers": len(args.active_labels),
            "optimizer_workers_per_label": 2,
            "maximum_processes": len(args.active_labels) * 2,
            "video_fps": args.video_frames / args.baseline_wall_seconds,
            "root": str(args.baseline_root),
        }
    )
    results = [baseline]
    for raw in (value.strip() for value in args.layouts.split(",") if value.strip()):
        left, right = raw.lower().split("x", 1)
        results.append(_run(args, int(left), int(right)))
    comparisons = {
        result["name"]: _compare(baseline, result, args.active_labels)
        for result in results[1:]
    }
    payload = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "video_frames": args.video_frames,
        "results": results,
        "comparisons_to_baseline": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "benchmark_results.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
