#!/usr/bin/env python3
"""Run the clean Production raw-only and forced minimum-Recall matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from metrics import LABELS, evaluate_sqlite, write_metrics


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
DEFAULT_OUTPUT = ROOT / "output" / "production_raw_only_0809_20260809"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, default=ROOT / "data/12月KPI動画.sqlite")
    parser.add_argument("--source-video", type=Path, default=ROOT / "data/12月KPI動画.mp4")
    parser.add_argument(
        "--cuts-json",
        type=Path,
        default=ROOT
        / "output/production_vs_old_pareto_score030_20260806/production10_work/03_cut_detection/cuts.json",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intervals", default="1,3,5,8,10")
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--guard-margin", type=float, default=0.002)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--max-anchor-scale", type=float, default=1.50)
    parser.add_argument("--polygon-workers", type=int, default=4)
    parser.add_argument(
        "--mode", choices=("all", "baseline", "guard"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run(command: list[str], log_path: Path, environment: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - started


def _policy(interval: int) -> dict[str, object]:
    value = {
        "shape_mode": "polygon",
        "keyframe_interval": int(interval),
        "max_gap": 15,
    }
    return {
        "schema_version": 1,
        "default": dict(value),
        "classes": {label: dict(value) for label in LABELS},
    }


def _result_from_manifest(work: Path) -> Path:
    manifest_path = work / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = Path(manifest["artifacts"]["result_sqlite"])
    return result if result.is_absolute() else (ROOT / result).resolve()


def _raw_ranges(source: Path) -> dict[str, tuple[int, int]]:
    from experimental.polygon_recall_optimizer.fixed_budget import load_raw_masks

    output = {}
    for label in LABELS:
        rows = load_raw_masks(source, label=label, start_frame=0, end_frame=2**31 - 1)
        frames = [frame for frame, _track in rows]
        if not frames:
            raise RuntimeError(f"no raw masks for {label}")
        output[label] = (min(frames), max(frames))
    return output


def _baseline(
    interval: int,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> tuple[Path, dict[str, object]]:
    root = args.output_root / f"interval_{interval}" / "production_raw"
    work = root / "work"
    report_path = root / "metrics.json"
    if report_path.is_file() and not args.force:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return Path(payload["result_sqlite"]), payload
    root.mkdir(parents=True, exist_ok=True)
    policy_path = root / "all_polygon_policy.json"
    policy_path.write_text(
        json.dumps(_policy(interval), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(POSTPROCESS / "run_pipeline.py"),
        "--input-sqlite",
        str(args.source_sqlite.resolve()),
        "--input-video",
        str(args.source_video.resolve()),
        "--output-dir",
        str(work.resolve()),
        "--class-postprocess-policy-json",
        str(policy_path.resolve()),
        "--score-min",
        "0.3",
        "--precomputed-cuts-json",
        str(args.cuts_json.resolve()),
        "--polygon-num-workers",
        str(args.polygon_workers),
        "--face-mask-target",
        "none",
    ]
    wall = _run(command, root / "run.log", environment)
    result = _result_from_manifest(work)
    # Use the raw masks and tracking assignments copied into this exact run's
    # result. Track IDs from a different historical run are not comparable.
    payload = evaluate_sqlite(result, result, recall_floor=args.recall_floor)
    payload.update(
        {
            "variant": "production_raw",
            "requested_interval": interval,
            "pipeline_wall_seconds": wall,
            "policy": _policy(interval),
            "result_sqlite": str(result.resolve()),
        }
    )
    classwise = json.loads(
        (work / "04_classwise_postprocess/classwise_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    payload["classwise_wall_seconds"] = float(classwise["elapsed_seconds"])
    payload["production_groups"] = classwise["groups"]
    summaries = []
    for group in classwise["groups"]:
        manifest = Path(group["pipeline_manifest"])
        if not manifest.is_absolute():
            manifest = (ROOT / manifest).resolve()
        group_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        polygon_stage = next(
            stage
            for stage in group_manifest["stages"]
            if stage["id"] == "polygon_optimization"
        )
        optimizer = polygon_stage["metadata"]["optimizer"]
        summaries.append(
            {
                "group_id": group["id"],
                "labels": group["labels"],
                "input_masks": group["input_masks"],
                "group_elapsed_seconds": group["elapsed_seconds"],
                "stage_elapsed_seconds": polygon_stage["elapsed_seconds"],
                "mean_state_count": optimizer["mean_state_count"],
                "optimizer_seconds": optimizer["optimizer_seconds"],
                "stage_seconds_total": optimizer["stage_seconds_total"],
            }
        )
    payload["production_polygon_summaries"] = summaries
    write_metrics(report_path, payload)
    return result, payload


def _hard_guard(
    interval: int,
    baseline: Path,
    baseline_payload: dict[str, object],
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, object]:
    root = args.output_root / f"interval_{interval}" / "production_raw_hard_recall"
    report_path = root / "metrics.json"
    if report_path.is_file() and not args.force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    current = baseline.resolve()
    ranges = _raw_ranges(baseline)
    reports = []
    guard_wall = 0.0
    previous_temporary: Path | None = None
    for index, label in enumerate(LABELS):
        label_dir = root / f"{index:02d}_{label}"
        output = root / f".guard_{index}.sqlite"
        first, last = ranges[label]
        command = [
            sys.executable,
            "-m",
            "experimental.polygon_recall_optimizer.run_production_guard",
            "--source-sqlite",
            str(baseline.resolve()),
            "--baseline-sqlite",
            str(current),
            "--output-dir",
            str(label_dir),
            "--output-sqlite",
            str(output),
            "--label",
            label,
            "--start-frame",
            str(first),
            "--end-frame",
            str(last),
            "--recall-floor",
            str(args.recall_floor),
            "--guard-margin",
            str(args.guard_margin),
            "--point-count",
            str(args.point_count),
            "--max-anchor-scale",
            str(args.max_anchor_scale),
        ]
        elapsed = _run(command, label_dir / "run.log", environment)
        guard_wall += elapsed
        report = json.loads(
            (label_dir / "comparison_report.json").read_text(encoding="utf-8")
        )
        reports.append(report)
        if previous_temporary is not None and previous_temporary.is_file():
            previous_temporary.unlink()
        previous_temporary = output
        current = output.resolve()
    final_sqlite = root / "result.sqlite"
    if final_sqlite.exists():
        final_sqlite.unlink()
    shutil.move(str(current), final_sqlite)
    payload = evaluate_sqlite(
        baseline, final_sqlite, recall_floor=args.recall_floor
    )
    payload.update(
        {
            "variant": "production_raw_hard_recall",
            "requested_interval": interval,
            "baseline_result_sqlite": str(baseline.resolve()),
            "result_sqlite": str(final_sqlite.resolve()),
            "baseline_pipeline_wall_seconds": float(
                baseline_payload["pipeline_wall_seconds"]
            ),
            "hard_guard_wall_seconds": guard_wall,
            "hard_guard_optimizer_seconds": sum(
                float(report["optimizer"]["elapsed_seconds"])
                for report in reports
            ),
            "total_wall_seconds": float(baseline_payload["pipeline_wall_seconds"])
            + guard_wall,
            "guard_reports": reports,
        }
    )
    write_metrics(report_path, payload)
    return payload


def _collect_matrix(output_root: Path) -> list[dict[str, object]]:
    """Collect every completed cell so split/resumed runs keep one full matrix."""
    rows: list[dict[str, object]] = []
    for interval_dir in sorted(
        output_root.glob("interval_*"),
        key=lambda path: int(path.name.removeprefix("interval_")),
    ):
        for variant in ("production_raw", "production_raw_hard_recall"):
            report = interval_dir / variant / "metrics.json"
            if report.is_file():
                rows.append(json.loads(report.read_text(encoding="utf-8")))
    return rows


def main() -> int:
    args = parse_args()
    args.source_sqlite = args.source_sqlite.expanduser().resolve()
    args.source_video = args.source_video.expanduser().resolve()
    args.cuts_json = args.cuts_json.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for path in (args.source_sqlite, args.source_video, args.cuts_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    intervals = [int(value) for value in args.intervals.split(",") if value.strip()]
    if any(value < 1 for value in intervals):
        raise ValueError("intervals must be >= 1")
    if not 0.0 < args.recall_floor <= 1.0:
        raise ValueError("recall-floor must be in (0, 1]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(POSTPROCESS), str(ROOT / "overlay/src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for interval in intervals:
        if args.mode == "guard":
            baseline_report = (
                args.output_root / f"interval_{interval}/production_raw/metrics.json"
            )
            if not baseline_report.is_file():
                raise FileNotFoundError(
                    f"baseline is required before guard mode: {baseline_report}"
                )
            baseline_payload = json.loads(
                baseline_report.read_text(encoding="utf-8")
            )
            baseline = Path(baseline_payload["result_sqlite"])
        else:
            baseline, baseline_payload = _baseline(
                interval, args, environment
            )
        if args.mode in {"all", "guard"}:
            _hard_guard(
                interval,
                baseline,
                baseline_payload,
                args,
                environment,
            )
    write_metrics(
        args.output_root / "matrix.json",
        {
            "experimental": True,
            "privacy": "SQLite geometry only; video pixels were not opened by evaluation.",
            "intervals": intervals,
            "labels": list(LABELS),
            "recall_floor": args.recall_floor,
            "rows": _collect_matrix(args.output_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
