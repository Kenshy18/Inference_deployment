#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PIPELINE_MARKERS = (
    "-m orchestration",
    "InstanceSegmentation/inference/",
    "postprocess/run_pipeline.py",
    "postprocess/vendor/original_polygon/",
    "-m overlay_renderer",
    "overlay/native/build/overlay_native",
    "wsl-runner.py",
)


def process_snapshot() -> list[dict[str, object]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,rss=,vsz=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = line.strip().split(None, 5)
        if len(values) != 6 or not any(marker in values[5] for marker in PIPELINE_MARKERS):
            continue
        pid = int(values[0])
        pss_kib = None
        try:
            for smaps_line in Path(f"/proc/{pid}/smaps_rollup").read_text(
                encoding="ascii"
            ).splitlines():
                if smaps_line.startswith("Pss:"):
                    pss_kib = int(smaps_line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        try:
            fd_count = len(list(Path(f"/proc/{pid}/fd").iterdir()))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            fd_count = None
        rows.append(
            {
                "pid": pid,
                "ppid": int(values[1]),
                "rss_kib": int(values[2]),
                "pss_kib": pss_kib,
                "fd_count": fd_count,
                "vsz_kib": int(values[3]),
                "elapsed_seconds": int(values[4]),
                "command": values[5],
            }
        )
    return rows


def memory_snapshot() -> dict[str, int]:
    output: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        output[key] = int(value.strip().split()[0])
    return output


def gpu_snapshot() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_throttle_reasons.hw_thermal_slowdown",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if result.returncode or len(values) < 5:
        return {"error": result.stderr.strip() or f"exit {result.returncode}"}
    return {
        "utilization_percent": float(values[0]),
        "memory_used_mib": float(values[1]),
        "temperature_c": float(values[2]),
        "power_w": float(values[3]),
        "thermal_slowdown": values[4],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=4 * 60 * 60)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("a", encoding="utf-8", buffering=1) as handle:
        while not args.stop_file.exists() and time.monotonic() - started < args.max_seconds:
            processes = process_snapshot()
            memory = memory_snapshot()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "pipeline_rss_kib": sum(int(item["rss_kib"]) for item in processes),
                "pipeline_pss_kib": sum(
                    int(item["pss_kib"])
                    for item in processes
                    if item["pss_kib"] is not None
                ),
                "pipeline_fd_count": sum(
                    int(item["fd_count"])
                    for item in processes
                    if item["fd_count"] is not None
                ),
                "processes": processes,
                "system": {
                    "mem_total_kib": memory.get("MemTotal", 0),
                    "mem_available_kib": memory.get("MemAvailable", 0),
                    "swap_total_kib": memory.get("SwapTotal", 0),
                    "swap_free_kib": memory.get("SwapFree", 0),
                    "load_average": os.getloadavg(),
                },
                "gpu": gpu_snapshot(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(max(0.2, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
