#!/usr/bin/env python3
"""Production coordinator for multistate polygon optimization."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
POSTPROCESS = ROOT / "postprocess"
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from production.polygon.runtime import run_phase1 as phase1
from production.polygon.runtime.phase2_runtime import (
    PAIR_VOTE_CONSTRAINED_ENV,
    PAIR_VOTE_ENV,
    PAIR_VOTE_PER_KEY_ENV,
    PAIR_VOTE_SWEEPS_ENV,
    PROFILE_ENV,
    VALID_PROFILES,
)


RUNTIME = HERE / "phase2_runtime.py"
DEFAULT_OUTPUT = ROOT / "output/production_polygon_phase2"
DEFAULT_PROFILES = (
    "scale_best",
    "temporal_central_best",
    "temporal_recall_best",
    "axis_best",
    "broad_top2",
)
LABELS = phase1.LABELS
_UNSUPPORTED_CUDA_ENABLE_ENVIRONMENT = (
    "MASK_PIPELINE_PHASE2_CUDA_SHAPE",
    "MASK_PIPELINE_PHASE2_CUDA_PREFILTER",
    "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_VERIFY",
    "MASK_PIPELINE_PHASE2_CUDA_EXACT_HINT",
    "MASK_PIPELINE_PHASE2_CUDA_LAZY_EXACT",
    "MASK_PIPELINE_PHASE2_CUDA_APPROX_ONLY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=phase1.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--target-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-workers", type=int, default=3)
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="screen only the longest N tracks per class; 0 evaluates all tracks",
    )
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument(
        "--cuda-fast",
        action="store_true",
        help="use the screened CUDA dense-graph evaluator and native DP",
    )
    parser.add_argument(
        "--native-exact",
        action="store_true",
        help="use exact C++ interval evaluation and native DP (diagnostic)",
    )
    parser.add_argument(
        "--cuda-lazy-exact",
        action="store_true",
        help=(
            "screen the dense graph on CUDA, then exactly validate selected "
            "edges before accepting the path"
        ),
    )
    parser.add_argument(
        "--cuda-hint-exact",
        action="store_true",
        help=(
            "use CUDA only to prioritize low-Recall frames for the exact "
            "CPU evaluator; no edge is filtered by the approximation"
        ),
    )
    parser.add_argument("--anchors-per-contour", type=int, default=48)
    parser.add_argument("--min-anchors-per-contour", type=int, default=8)
    parser.add_argument(
        "--adaptive-anchor-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--native-batch-threads", type=int, default=8)
    parser.add_argument("--gc-interval", type=int, default=8)
    parser.add_argument(
        "--predictor-model-dir",
        type=Path,
        default=phase1.DEFAULT_PREDICTOR,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--pair-vote",
        action="store_true",
        help=(
            "enable Production post-DP pair-vote while keeping DP inputs and "
            "post-decode Recall repair unchanged/disabled"
        ),
    )
    parser.add_argument(
        "--pair-vote-constrained",
        action="store_true",
        help=(
            "blend toward Production pair-vote by exact mean-IoU maximization "
            "subject only to per-frame minimum Recall"
        ),
    )
    parser.add_argument(
        "--pair-vote-per-key",
        action="store_true",
        help=(
            "optimize one pair-vote blend alpha per fixed key by exact local "
            "IoU coordinate ascent under the per-frame Recall floor"
        ),
    )
    parser.add_argument(
        "--pair-vote-sweeps",
        type=int,
        default=2,
        help="number of alternating coordinate sweeps for per-key pair-vote",
    )
    return parser.parse_args()


def command(source: Path, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME),
        "__onefile_polygon_optimize",
        "--input-sqlite",
        str(source),
        "--output-dir",
        str(output),
        "--target-ratio",
        str(1.0 / float(args.target_interval)),
        "--anchors-per-contour",
        str(args.anchors_per_contour),
        "--point-predictor-model-dir",
        str(args.predictor_model_dir.resolve()),
        "--predictor-device",
        str(args.predictor_device),
        "--predictor-batch-size",
        "256",
        "--adaptive-point-quantile",
        "0.95",
        "--adaptive-point-offset",
        "10",
        "--min-anchors-per-contour",
        str(args.min_anchors_per_contour),
        "--gapfill-max-gap",
        "15",
        "--max-run-frames",
        "30000",
        "--run-overlap-frames",
        "900",
        "--recall-min",
        str(args.recall_floor),
        "--max-gap",
        "30",
        "--num-workers",
        str(args.num_workers),
        "--max-tracks",
        str(args.max_tracks),
        "--stream-sqlite-rows",
        "--evaluate-exact",
        "--write-pred-sqlite",
        "--gapfill-enabled",
    ] + (
        ["--adaptive-anchor-counts"]
        if args.adaptive_anchor_counts
        else ["--no-adaptive-anchor-counts"]
    )


def run_cell(
    source: Path,
    label: str,
    profile: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = args.output_root / profile / label
    report = root / "metrics.json"
    if report.is_file() and not args.force:
        return json.loads(report.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    output = root / "runtime"
    environment = os.environ.copy()
    for variable in _UNSUPPORTED_CUDA_ENABLE_ENVIRONMENT:
        environment.pop(variable, None)
    environment[PROFILE_ENV] = profile
    pair_vote_enabled = bool(
        args.pair_vote or args.pair_vote_constrained or args.pair_vote_per_key
    )
    environment[PAIR_VOTE_ENV] = "1" if pair_vote_enabled else "0"
    environment[PAIR_VOTE_CONSTRAINED_ENV] = (
        "1" if (args.pair_vote_constrained or args.pair_vote_per_key) else "0"
    )
    environment[PAIR_VOTE_PER_KEY_ENV] = "1" if args.pair_vote_per_key else "0"
    environment[PAIR_VOTE_SWEEPS_ENV] = str(max(1, args.pair_vote_sweeps))
    environment["MASK_PIPELINE_PHASE2_LABEL"] = str(label)
    environment["MASK_PIPELINE_PHASE2_TARGET_INTERVAL"] = str(args.target_interval)
    environment["MASK_PIPELINE_PHASE2_OPENCV_THREADS"] = str(
        max(1, int((os.cpu_count() or 1) / max(1, args.label_workers)))
    )
    python_paths = [str(POSTPROCESS)]
    if args.native_exact:
        python_paths.insert(0, str(HERE / "native_interval/build"))
        environment.update(
            {
                "MASK_PIPELINE_PHASE1_NATIVE_EXACT": "1",
                "MASK_PIPELINE_PHASE1_NATIVE_INTERVAL": "1",
                "MASK_PIPELINE_PHASE2_NATIVE_BATCH": "1",
                "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS": str(
                    args.native_batch_threads
                ),
                "MASK_PIPELINE_PHASE2_NATIVE_DP": "1",
                "MASK_PIPELINE_PHASE2_GC_INTERVAL": str(args.gc_interval),
            }
        )
    environment["PYTHONPATH"] = os.pathsep.join(
        [*python_paths, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.perf_counter()
    with (root / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command(source, output, args),
            cwd=POSTPROCESS,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Phase 2 failed: profile={profile} label={label}; {root/'run.log'}"
        )
    metrics = phase1._metrics(
        output,
        source,
        label,
        args.target_interval,
        wall,
        audit_name="phase2_audit.json",
    )
    metrics["candidate_profile"] = profile
    report.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.predictor_model_dir = args.predictor_model_dir.expanduser().resolve()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    unknown_profiles = sorted(set(profiles) - VALID_PROFILES)
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {unknown_profiles}")
    if any(label not in LABELS for label in labels):
        raise ValueError(f"labels must be selected from {LABELS}")
    if args.num_workers < 1 or args.label_workers < 1:
        raise ValueError("worker counts must be >= 1")
    if args.native_batch_threads < 1 or args.gc_interval < 1:
        raise ValueError("native-batch-threads and gc-interval must be >= 1")
    if args.max_tracks < 0:
        raise ValueError("max-tracks must be >= 0")
    if not 0.0 < args.recall_floor <= 1.0:
        raise ValueError("recall-floor must be in (0, 1]")
    if args.target_interval < 1:
        raise ValueError("target-interval must be >= 1")
    if args.cuda_fast or args.cuda_lazy_exact or args.cuda_hint_exact:
        raise ValueError(
            "the deployed Production runtime supports CPU native_exact only"
        )
    if not args.native_exact:
        raise ValueError("the deployed Production runtime requires --native-exact")
    if (
        sum(
            bool(value)
            for value in (
                args.cuda_fast,
                args.cuda_lazy_exact,
                args.cuda_hint_exact,
                args.native_exact,
            )
        )
        > 1
    ):
        raise ValueError(
            "--cuda-fast, --cuda-lazy-exact, --cuda-hint-exact, and "
            "--native-exact are mutually exclusive"
        )
    if args.anchors_per_contour < 1 or args.min_anchors_per_contour < 1:
        raise ValueError("anchor counts must be >= 1")
    if args.min_anchors_per_contour > args.anchors_per_contour:
        raise ValueError("min-anchors-per-contour cannot exceed anchors-per-contour")
    if (
        sum(
            bool(value)
            for value in (
                args.pair_vote,
                args.pair_vote_constrained,
                args.pair_vote_per_key,
            )
        )
        > 1
    ):
        raise ValueError("pair-vote mode flags are mutually exclusive")
    sources = phase1._discover_inputs(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    profile_reports = []
    for profile in profiles:
        print(f"[phase2-profile] {profile}", flush=True)
        started = time.perf_counter()
        workers = min(int(args.label_workers), len(labels))
        if workers == 1:
            rows = [run_cell(sources[label], label, profile, args) for label in labels]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(
                    executor.map(
                        lambda label: run_cell(sources[label], label, profile, args),
                        labels,
                    )
                )
        elapsed = time.perf_counter() - started
        aggregate = phase1._aggregate(rows, args.target_interval)
        aggregate["candidate_profile"] = profile
        aggregate["profile_wall_seconds"] = elapsed
        aggregate["observation_rows_per_second"] = float(
            aggregate["observation_rows"]
        ) / max(elapsed, 1e-9)
        aggregate["video_fps"] = (
            23510.0 / max(elapsed, 1e-9) if args.max_tracks == 0 else None
        )
        profile_reports.append(aggregate)
        all_rows.extend(rows)
        print(
            f"[phase2-result] {profile} wall={elapsed:.3f}s "
            f"actual={aggregate['actual_mean_interval']:.3f} "
            f"iou={aggregate['iou_mean']:.6f} "
            f"feasible_iou={aggregate['feasible_iou_mean']:.6f} "
            f"infeasible={aggregate['infeasible_streams']}",
            flush=True,
        )
        payload = {
            "schema_version": 1,
            "production": True,
            "privacy": "SQLite geometry only; no video frame was opened.",
            "target_interval": int(args.target_interval),
            "recall_floor": args.recall_floor,
            "pair_vote": bool(
                args.pair_vote or args.pair_vote_constrained or args.pair_vote_per_key
            ),
            "pair_vote_mode": (
                "per_key_iou_recall"
                if args.pair_vote_per_key
                else (
                    "constrained_iou_recall"
                    if args.pair_vote_constrained
                    else ("production_post_dp" if args.pair_vote else "off")
                )
            ),
            "profiles": profiles,
            "completed_profiles": profile_reports,
            "rows": all_rows,
            "execution": {
                "label_workers": workers,
                "dp_workers_per_label": args.num_workers,
                "maximum_concurrent_dp_workers": workers * args.num_workers,
                "max_tracks_per_label": args.max_tracks,
                "cuda_fast": bool(args.cuda_fast),
                "cuda_lazy_exact": bool(args.cuda_lazy_exact),
                "cuda_hint_exact": bool(args.cuda_hint_exact),
                "native_exact": bool(args.native_exact),
                "anchors_per_contour": int(args.anchors_per_contour),
                "min_anchors_per_contour": int(args.min_anchors_per_contour),
                "adaptive_anchor_counts": bool(args.adaptive_anchor_counts),
                "native_batch_threads": int(args.native_batch_threads),
                "gc_interval": int(args.gc_interval),
            },
        }
        (args.output_root / "phase2_matrix.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if profile_reports:
            columns = list(profile_reports[0])
            with (args.output_root / "phase2_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(profile_reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
