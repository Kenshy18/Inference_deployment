#!/usr/bin/env python3
"""Run the parity-frozen adaptive-polygon Production optimizer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .spatial_config import ADAPTIVE_PROFILE_ID, CANDIDATE, PROFILE_ID


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PHASE2_RUNNER = HERE / "run_phase2.py"
DEFAULT_SOURCE = ROOT / "output/production_polygon_source"
DEFAULT_OUTPUT = ROOT / "output/production_polygon_optimizer"
LABELS = ("女性器", "男性器", "結合部分")


def _candidate_contract(profile: str, target_interval: int) -> dict[str, object]:
    if profile == ADAPTIVE_PROFILE_ID:
        from .candidate_config import (
            CANDIDATE as ADAPTIVE_CANDIDATE,
            with_target_interval,
        )

        value = with_target_interval(int(target_interval), ADAPTIVE_CANDIDATE).to_dict()
    else:
        value = CANDIDATE.to_dict()
    return json.loads(json.dumps(value, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build persistent track-adaptive 14/16/18/20-point polygons "
            "and optimize their temporal keyframes under exact per-frame "
            "Recall constraints."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intervals", default="1,3,6")
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--label-workers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pair-vote-threads", type=int, default=2)
    parser.add_argument("--native-batch-threads", type=int, default=8)
    parser.add_argument(
        "--interval-evaluation",
        choices=("native_exact",),
        default="native_exact",
    )
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--profile",
        choices=(PROFILE_ID, ADAPTIVE_PROFILE_ID),
        default=PROFILE_ID,
    )
    parser.add_argument(
        "--vertex-policy",
        type=Path,
        help="required track-level vertex policy for the adaptive profile",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, interval: int, output: Path) -> list[str]:
    profile = str(getattr(args, "profile", PROFILE_ID))
    adaptive = profile == ADAPTIVE_PROFILE_ID
    command = [
        sys.executable,
        str(PHASE2_RUNNER),
        "--source-root",
        str(args.source_root.expanduser().resolve()),
        "--output-root",
        str(output),
        "--profiles",
        profile,
        "--labels",
        args.labels,
        "--target-interval",
        str(interval),
        "--recall-floor",
        str(CANDIDATE.temporal_recall_floor),
        "--anchors-per-contour",
        str(20 if adaptive else CANDIDATE.vertices_per_component),
        "--min-anchors-per-contour",
        str(14 if adaptive else CANDIDATE.vertices_per_component),
        "--no-adaptive-anchor-counts",
        "--num-workers",
        str(max(1, int(args.num_workers))),
        "--label-workers",
        str(max(1, int(args.label_workers))),
        "--max-tracks",
        str(max(0, int(args.max_tracks))),
        "--predictor-device",
        "cpu",
        "--native-exact",
        "--native-batch-threads",
        str(max(1, int(getattr(args, "native_batch_threads", 8)))),
        "--gc-interval",
        "8",
        "--pair-vote-per-key",
        "--pair-vote-sweeps",
        str(CANDIDATE.pair_vote_sweeps),
    ]
    if args.force:
        command.append("--force")
    return command


def _exact_quality(
    interval_root: Path,
    labels: list[str],
    target_interval: int,
    profile: str = PROFILE_ID,
) -> dict[str, object]:
    minimum_recall = 1.0
    rows = 0
    violations = 0
    audits: dict[str, str] = {}
    for label in labels:
        runtime = interval_root / profile / label / "runtime"
        audit_path = runtime / "phase2_audit.json"
        metrics_path = runtime / "exact/keyframe_exact_metrics.csv"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_contract = _candidate_contract(profile, target_interval)
        if audit.get("production_candidate_contract") != expected_contract:
            raise RuntimeError(f"candidate contract mismatch: {audit_path}")
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                recall = float(row["recall"])
                minimum_recall = min(minimum_recall, recall)
                rows += 1
                violations += int(
                    recall + 1e-12 < float(CANDIDATE.temporal_recall_floor)
                )
        audits[label] = str(audit_path)
    return {
        "evaluated_rows": rows,
        "minimum_recall": minimum_recall,
        "recall_violations": violations,
        "audits": audits,
    }


def main() -> int:
    args = parse_args()
    intervals = [
        int(value.strip()) for value in args.intervals.split(",") if value.strip()
    ]
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    if not intervals or any(value < 1 for value in intervals):
        raise ValueError("intervals must contain positive integers")
    if not labels:
        raise ValueError("labels must contain at least one non-empty class name")
    unsupported = tuple(label for label in labels if label not in LABELS)
    if unsupported:
        raise ValueError(
            f"unsupported Production labels: {unsupported}; expected a subset of {LABELS}"
        )
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.profile == ADAPTIVE_PROFILE_ID:
        if args.vertex_policy is None:
            raise ValueError("--vertex-policy is required for the adaptive profile")
        vertex_policy = args.vertex_policy.expanduser().resolve()
        if not vertex_policy.is_file():
            raise FileNotFoundError(vertex_policy)
    else:
        vertex_policy = None
    runs: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment["MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE"] = "1"
    environment["MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS"] = str(
        max(1, int(args.pair_vote_threads))
    )
    if vertex_policy is not None:
        environment["MASK_PIPELINE_SPATIAL_VERTEX_POLICY_JSON"] = str(vertex_policy)
    for interval in intervals:
        interval_root = root / f"interval_{interval}"
        command = build_command(args, interval, interval_root)
        started = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        wall = time.perf_counter() - started
        if process.returncode != 0:
            return int(process.returncode)
        matrix_path = interval_root / "phase2_matrix.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        aggregate = matrix["completed_profiles"][-1]
        quality = _exact_quality(interval_root, labels, interval, args.profile)
        runs.append(
            {
                "target_interval": interval,
                "candidate": _candidate_contract(args.profile, interval),
                "wall_seconds": wall,
                "actual_mean_interval": aggregate["actual_mean_interval"],
                "mean_iou": aggregate["iou_mean"],
                "keyframes": aggregate["keyframes"],
                "exact_quality": quality,
                "matrix": str(matrix_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "production",
        "candidate_contracts_by_target_interval": {
            str(interval): _candidate_contract(args.profile, interval)
            for interval in intervals
        },
        "polygon_profile": args.profile,
        "vertex_policy": None if vertex_policy is None else str(vertex_policy),
        "privacy": "SQLite mask geometry only; no video frames were opened.",
        "exact_recall_gate": "repair_then_audit_and_publish",
        "sqlite_output_schema_changed": False,
        "source_root": str(args.source_root.expanduser().resolve()),
        "runs": runs,
    }
    manifest_path = root / "production_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
