#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(len(ordered) * fraction))]


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def slope_per_hour(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    per_second = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return per_second * 3600


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
    return "idle"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(phase(row), []).append(row)
    phase_reports = {}
    issues = []
    for name, items in grouped.items():
        rss_mib = [float(item["pipeline_rss_kib"]) / 1024 for item in items]
        pss_mib = [
            float(item["pipeline_pss_kib"]) / 1024
            for item in items
            if item.get("pipeline_pss_kib") is not None
        ]
        primary_mib = pss_mib if len(pss_mib) == len(items) and any(pss_mib) else rss_mib
        points = [
            (float(item["elapsed_seconds"]), value)
            for item, value in zip(items, primary_mib)
        ]
        trimmed = points[len(points) // 10 :] if len(points) >= 20 else points
        slope = slope_per_hour(trimmed)
        first = primary_mib[: max(1, len(primary_mib) // 4)]
        last = primary_mib[-max(1, len(primary_mib) // 4) :]
        median_delta_percent = (
            (statistics.median(last) - statistics.median(first)) / statistics.median(first) * 100
            if first and statistics.median(first) > 0
            else None
        )
        # A pipeline phase may contain bounded substage transitions (for
        # example NMS -> tracking -> classwise GPU processing).  Regressing
        # across that step reports an enormous, but fictitious, leak rate.
        # Keep the whole-phase values for diagnostics and gate linear growth
        # on the stable tail.  A genuine per-frame leak continues through
        # this tail; a one-time model/cache allocation does not.
        stable_start = math.floor(len(points) * 0.6) if len(points) >= 20 else 0
        stable_points = points[stable_start:]
        stable_values = primary_mib[stable_start:]
        stable_slope = slope_per_hour(stable_points)
        stable_quartile = max(1, len(stable_values) // 4)
        stable_first = stable_values[:stable_quartile]
        stable_last = stable_values[-stable_quartile:]
        stable_median_delta_percent = (
            (statistics.median(stable_last) - statistics.median(stable_first))
            / statistics.median(stable_first)
            * 100
            if stable_first and statistics.median(stable_first) > 0
            else None
        )
        phase_reports[name] = {
            "rss_mib": summary(rss_mib),
            "pss_mib": summary(pss_mib),
            "primary_memory_metric": "pss" if primary_mib is pss_mib else "rss",
            "memory_slope_mib_per_hour_after_warmup": slope,
            "first_to_last_quartile_median_percent": median_delta_percent,
            "stable_tail_fraction": 0.4 if len(points) >= 20 else 1.0,
            "stable_tail_memory_slope_mib_per_hour": stable_slope,
            "stable_tail_quartile_median_percent": stable_median_delta_percent,
            "fd_count": summary(
                [float(item["pipeline_fd_count"]) for item in items if item.get("pipeline_fd_count") is not None]
            ),
        }
        if (
            name in {"inference", "postprocess", "overlay"}
            and stable_slope is not None
        ):
            if stable_slope > 128 and (stable_median_delta_percent or 0) > 10:
                issues.append(
                    f"{name}: stable-tail RSS slope {stable_slope:.1f} MiB/hour "
                    f"and quartile delta {stable_median_delta_percent:.1f}%"
                )
    gpu_rows = [row["gpu"] for row in rows if "error" not in row["gpu"]]
    swap_used = [
        (float(row["system"]["swap_total_kib"]) - float(row["system"]["swap_free_kib"])) / 1024
        for row in rows
    ]
    thermal = sorted({str(row.get("thermal_slowdown")) for row in gpu_rows})
    if any(value.lower() not in {"not active", "n/a", "none"} for value in thermal):
        issues.append(f"thermal slowdown observed: {thermal}")
    report = {
        "schema_version": 1,
        "sample_count": len(rows),
        "elapsed_seconds": float(rows[-1]["elapsed_seconds"]) if rows else 0,
        "phase_resources": phase_reports,
        "system": {
            "swap_used_mib": summary(swap_used),
            "minimum_mem_available_mib": min(
                (float(row["system"]["mem_available_kib"]) / 1024 for row in rows),
                default=None,
            ),
        },
        "gpu": {
            "utilization_percent": summary([float(row["utilization_percent"]) for row in gpu_rows]),
            "memory_used_mib": summary([float(row["memory_used_mib"]) for row in gpu_rows]),
            "temperature_c": summary([float(row["temperature_c"]) for row in gpu_rows]),
            "power_w": summary([float(row["power_w"]) for row in gpu_rows]),
            "thermal_slowdown_values": thermal,
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "issues": issues}, ensure_ascii=False))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
