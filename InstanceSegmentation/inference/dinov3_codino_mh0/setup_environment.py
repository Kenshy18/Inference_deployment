#!/usr/bin/env python3
"""Prepare the self-contained MH0 source and shared runtime directories."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
VENDOR = ROOT / "vendor"


def _extract(archive: Path, destination: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"vendor archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.force and RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    _extract(VENDOR / "codino.tar.gz", RUNTIME / "src")
    _extract(VENDOR / "dinov3_root.tar.gz", RUNTIME / "src")
    _extract(VENDOR / "inference_common.tar.gz", RUNTIME / "shared")
    if not args.extract_only:
        subprocess.run(
            [
                str(args.python.expanduser().resolve()),
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements.txt"),
            ],
            check=True,
        )
    lock = {
        "schema": "dinov3-codino-mh0-runtime-v1",
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "sources": [
            str(RUNTIME / "src" / "codino"),
            str(RUNTIME / "src" / "dinov3_root"),
        ],
        "shared": str(RUNTIME / "shared"),
    }
    (RUNTIME / "environment-lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] prepared {RUNTIME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
