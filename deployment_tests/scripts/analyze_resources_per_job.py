#!/usr/bin/env python3
"""Summarize GUI resource samples per orchestrator job.

The aggregate sampler spans heterogeneous videos and resolutions.  Treating
all inference or postprocess samples as one time series creates a false leak
signal whenever a later job is larger than an earlier one.  This analyzer
uses the immutable GUI job ID embedded in each command line and reports each
job independently.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


JOB_ID = re.compile(r"(20\d\d-\d\d-\d\dT\d\d-\d\d-\d\d-\d{3}Z)")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(len(ordered) * fraction))]


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
    }


def phase(row: dict[str, object]) -> str:
    commands = "\n".join(str(process["command"]) for process in row["processes"])
    if "postprocess/run_pipeline.py" in commands:
        return "postprocess"
    if "overlay_renderer" in commands or "overlay_native" in commands:
        return "overlay"
    if "InstanceSegmentation/inference/" in commands:
        return "inference"
    if "-m orchestration" in commands:
        return "orchestration"
    return "wrapper"


def row_job_ids(row: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for process in row["processes"]:
        ids.update(JOB_ID.findall(str(process["command"])))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    ambiguous_rows = 0
    idle_rows = 0
    for row in rows:
        ids = row_job_ids(row)
        if not ids:
            idle_rows += 1
            continue
        if len(ids) > 1:
            ambiguous_rows += 1
        for job_id in ids:
            grouped.setdefault(job_id, []).append(row)

    jobs = []
    issues = []
    for job_id, items in sorted(grouped.items(), key=lambda item: item[1][0]["elapsed_seconds"]):
        pss = [float(item.get("pipeline_pss_kib") or 0) / 1024 for item in items]
        rss = [float(item.get("pipeline_rss_kib") or 0) / 1024 for item in items]
        fds = [float(item.get("pipeline_fd_count") or 0) for item in items]
        phases: dict[str, list[float]] = {}
        for item, value in zip(items, pss):
            phases.setdefault(phase(item), []).append(value)
        # Samples are roughly two seconds apart. A return to zero is checked
        # globally below; the final in-job sample may legitimately be a copy
        # wrapper and is not itself a leak test.
        job = {
            "job_id": job_id,
            "first_elapsed_seconds": float(items[0]["elapsed_seconds"]),
            "last_elapsed_seconds": float(items[-1]["elapsed_seconds"]),
            "observed_duration_seconds": float(items[-1]["elapsed_seconds"])
            - float(items[0]["elapsed_seconds"]),
            "sample_count": len(items),
            "pss_mib": stats(pss),
            "rss_mib": stats(rss),
            "fd_count": stats(fds),
            "phase_sample_counts": {name: len(values) for name, values in phases.items()},
            "phase_peak_pss_mib": {name: max(values) for name, values in phases.items()},
        }
        if max(fds, default=0) > 256:
            issues.append(f"{job_id}: file descriptor peak {max(fds):.0f}")
        jobs.append(job)

    active_flags = [bool(row_job_ids(row)) for row in rows]
    completed_intervals = 0
    intervals_returning_to_zero = 0
    in_active = False
    for index, active in enumerate(active_flags):
        if active:
            in_active = True
        elif in_active:
            completed_intervals += 1
            if float(rows[index].get("pipeline_pss_kib") or 0) == 0:
                intervals_returning_to_zero += 1
            in_active = False
    all_completed_returned = (
        completed_intervals > 0 and completed_intervals == intervals_returning_to_zero
    )
    if not all_completed_returned:
        issues.append(
            "not every completed active interval returned pipeline PSS to zero: "
            f"{intervals_returning_to_zero}/{completed_intervals}"
        )

    gpu_rows = [row["gpu"] for row in rows if "error" not in row.get("gpu", {})]
    report = {
        "schema_version": 1,
        "sample_count": len(rows),
        "job_count": len(jobs),
        "idle_sample_count": idle_rows,
        "ambiguous_sample_count": ambiguous_rows,
        "completed_active_intervals": completed_intervals,
        "intervals_returning_pipeline_pss_to_zero": intervals_returning_to_zero,
        "all_completed_intervals_returned_to_zero": all_completed_returned,
        "jobs": jobs,
        "gpu": {
            "temperature_c": stats([float(row["temperature_c"]) for row in gpu_rows]),
            "memory_used_mib": stats([float(row["memory_used_mib"]) for row in gpu_rows]),
            "thermal_slowdown_values": sorted(
                {str(row.get("thermal_slowdown")) for row in gpu_rows}
            ),
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "issues": issues}, ensure_ascii=False))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
