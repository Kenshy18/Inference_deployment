"""Run the CPU-only Production curve on an archived V3 tracking corpus.

This is a resumable validation harness, not a Production dependency.  Each run
is published by ``run_pipeline.py`` into its own directory and the batch
manifest is atomically refreshed after every child process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--target-interval", type=int, default=6)
    parser.add_argument("--exclude", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        raise SystemExit("V3 curve validation requires CUDA_VISIBLE_DEVICES='' or -1")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    excluded = {str(value) for value in args.exclude}
    run_dirs = tuple(
        path.parent.parent
        for path in sorted(source_root.glob("*/shared/shared_manifest.json"))
        if path.parent.parent.name not in excluded
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "target_interval": int(args.target_interval),
        "source_root": str(source_root),
        "policy_json": str(args.policy_json.expanduser().resolve()),
        "runs": [],
    }
    manifest_path = output_root / "batch_manifest.json"
    for run_dir in run_dirs:
        shared = json.loads(
            (run_dir / "shared" / "shared_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        run_id = str(run_dir.name)
        run_output = output_root / run_id
        complete = run_output / "pipeline_manifest.json"
        row: dict[str, object] = {
            "run_id": run_id,
            "tracked_sqlite": str(Path(shared["tracking"]["tracked_sqlite"])),
            "input_video": str(Path(shared["run"]["video"])),
            "source_video_frames": int(shared["run"]["video_metadata_frames"]),
            "output_root": str(run_output),
        }
        payload["runs"].append(row)
        if complete.is_file():
            row["status"] = "reused_complete"
            row["elapsed_seconds"] = 0.0
            _write_json(manifest_path, payload)
            continue
        command = (
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "run_pipeline.py"),
            "--input-sqlite",
            str(shared["tracking"]["tracked_sqlite"]),
            "--input-video",
            str(shared["run"]["video"]),
            "--output-dir",
            str(run_output),
            "--mask-geometry",
            "catmull_rom",
            "--class-postprocess-policy-json",
            str(args.policy_json.expanduser().resolve()),
            "--keyframe-interval",
            str(int(args.target_interval)),
        )
        log_path = output_root / f"{run_id}.log"
        row["command"] = list(command)
        row["log"] = str(log_path)
        row["status"] = "running"
        _write_json(manifest_path, payload)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        row["elapsed_seconds"] = float(time.perf_counter() - started)
        row["returncode"] = int(completed.returncode)
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        _write_json(manifest_path, payload)
        if completed.returncode != 0:
            return int(completed.returncode)
    payload["complete"] = True
    payload["elapsed_seconds"] = float(
        sum(float(row.get("elapsed_seconds", 0.0)) for row in payload["runs"])
    )
    _write_json(manifest_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
