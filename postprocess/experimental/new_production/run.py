#!/usr/bin/env python3
"""Run the frozen new-production polygon profile at selected intervals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE2_RUNNER = ROOT / "postprocess/experimental/0809/run_phase2.py"
DEFAULT_SOURCE = ROOT / "output/production_raw_only_0809_20260809"
DEFAULT_OUTPUT = ROOT / "output/new_production_benchmark_20260812"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intervals", default="1,3,6")
    parser.add_argument("--engine", choices=("reference", "optimized"), default="optimized")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--labels", default="女性器,男性器,結合部分")
    parser.add_argument("--label-workers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pair-vote-threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    if not intervals or any(value < 1 for value in intervals):
        raise ValueError("intervals must contain positive integers")
    root = args.output_root.expanduser().resolve() / args.engine
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for interval in intervals:
        output = root / f"interval_{interval}"
        command = [
            sys.executable,
            str(PHASE2_RUNNER),
            "--source-root", str(args.source_root.expanduser().resolve()),
            "--output-root", str(output),
            "--profiles", "new_production_v1",
            "--labels", args.labels,
            "--target-interval", str(interval),
            "--recall-floor", "0.97",
            "--num-workers", str(max(1, int(args.num_workers))),
            "--label-workers", str(max(1, int(args.label_workers))),
            "--predictor-device", "cpu",
            "--cuda-fast",
            "--native-batch-threads", "8",
            "--gc-interval", "8",
            "--pair-vote-per-key",
            "--pair-vote-sweeps", "2",
        ]
        if args.force:
            command.append("--force")
        environment = os.environ.copy()
        environment["MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE"] = (
            "1" if args.engine == "optimized" else "0"
        )
        environment["MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS"] = str(
            max(1, int(args.pair_vote_threads))
        )
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        wall = time.perf_counter() - started
        if process.returncode != 0:
            return int(process.returncode)
        matrix = json.loads((output / "phase2_matrix.json").read_text(encoding="utf-8"))
        aggregate = matrix["completed_profiles"][-1]
        class_rows = [
            row
            for row in matrix.get("rows", [])
            if row.get("candidate_profile") == "new_production_v1"
        ]
        runs.append(
            {
                "interval": interval,
                "engine": args.engine,
                "pair_vote_threads": max(1, int(args.pair_vote_threads)),
                "wall_seconds": wall,
                "profile_wall_seconds": aggregate["profile_wall_seconds"],
                "video_fps": aggregate["video_fps"],
                "actual_mean_interval": aggregate["actual_mean_interval"],
                "iou_mean": aggregate["iou_mean"],
                "iou_min": min(
                    (float(row["iou_min"]) for row in class_rows), default=1.0
                ),
                "iou_q01_by_class_min": aggregate["iou_q01_by_class_min"],
                "iou_q05_by_class_min": min(
                    (float(row["iou_q05"]) for row in class_rows), default=1.0
                ),
                "recall_min": aggregate["recall_min"],
                "recall_violations": aggregate["recall_violations"],
                "keyframes": aggregate["keyframes"],
            }
        )
    payload = {
        "schema_version": 1,
        "profile": "new_production_v1",
        "engine": args.engine,
        "privacy": "SQLite polygon geometry only; video pixels were not opened.",
        "runs": runs,
    }
    (root / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
