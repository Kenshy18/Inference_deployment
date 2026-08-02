#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*values: str) -> str:
    result = subprocess.run(values, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    disk = shutil.disk_usage(args.output.parent)
    files = {}
    for name, path in {
        "installed_exe": args.exe,
        "installer": args.installer,
        "portable": args.portable,
        "release_manifest": args.release_manifest,
    }.items():
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_head": command("git", "-C", str(args.workspace), "rev-parse", "HEAD"),
        "git_status": command("git", "-C", str(args.workspace), "status", "--short"),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": command("free", "-h"),
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,temperature.gpu",
            "--format=csv,noheader",
        ),
        "disk": {
            "path": str(args.output.parent),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"git_head": report["git_head"], "files": files}, ensure_ascii=False))
    return 0 if all(item["exists"] for item in files.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
