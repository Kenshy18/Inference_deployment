#!/usr/bin/env python3
"""Install and verify an asset pack into a fresh Git clone."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from asset_tools import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--link", action="store_true", help="local QA only; install hard links")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    root = args.root.expanduser().resolve()
    payload = json.loads((source / "ASSET_PACK.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ASSET_PACK.json")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if payload.get("canonical_commit") != commit:
        raise ValueError(
            "asset pack/repository commit mismatch: "
            f"pack={payload.get('canonical_commit')} repository={commit}"
        )
    for record in payload["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe asset path: {relative}")
        origin = source / relative
        target = root / relative
        if not origin.is_file() or origin.stat().st_size != record["size"]:
            raise ValueError(f"invalid asset pack file: {origin}")
        if sha256_file(origin) != record["sha256"]:
            raise ValueError(f"asset pack hash mismatch: {origin}")
        if target.exists():
            if (
                target.is_file()
                and target.stat().st_size == record["size"]
                and sha256_file(target) == record["sha256"]
            ):
                continue
            raise FileExistsError(f"refusing to replace existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.link:
            os.link(origin, target)
        else:
            shutil.copy2(origin, target)
    print(f"[PASS] installed {len(payload['files'])} verified files into {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
