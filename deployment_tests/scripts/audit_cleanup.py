#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


RESIDUE_PATTERNS = (
    "*.orchestrating-*",
    ".orchestrating-*",
    "*.tmp",
    "*.partial",
    "*.sqlite-wal",
    "*.sqlite-shm",
    "*.wal",
    "*.shm",
)


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def residue_files(root: Path) -> list[dict[str, object]]:
    found: dict[Path, dict[str, object]] = {}
    for pattern in RESIDUE_PATTERNS:
        for path in root.rglob(pattern):
            found[path] = {
                "path": str(path),
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": tree_size(path),
            }
    return sorted(found.values(), key=lambda item: str(item["path"]))


def run_report(manifest: Path) -> dict[str, object]:
    root = manifest.parent
    stages = []
    for path in sorted(root.iterdir()):
        if not path.name[:2].isdigit():
            continue
        stages.append(
            {
                "name": path.name,
                "size_bytes": tree_size(path),
                "files": sum(1 for item in path.rglob("*") if item.is_file()),
            }
        )
    public = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            public.append({"name": path.name, "size_bytes": path.stat().st_size})
    return {
        "root": str(root),
        "size_bytes": tree_size(root),
        "manifest_size_bytes": manifest.stat().st_size,
        "stages": stages,
        "root_files": public,
        "residue": residue_files(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp/mask-pipeline-studio"))
    args = parser.parse_args()

    manifests = sorted(args.root.rglob("run_manifest.json"))
    runs = [run_report(manifest) for manifest in manifests]
    temp_entries = []
    if args.temp_root.exists():
        for path in sorted(args.temp_root.iterdir()):
            temp_entries.append(
                {
                    "path": str(path),
                    "size_bytes": tree_size(path),
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "empty": not any(path.iterdir()) if path.is_dir() else path.stat().st_size == 0,
                }
            )
    issues = []
    for run in runs:
        if run["residue"]:
            issues.append(f"temporary residue under {run['root']}: {len(run['residue'])}")
    nonempty_temp = [item for item in temp_entries if not item["empty"]]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "run_count": len(runs),
        "runs": runs,
        "temp_root": str(args.temp_root),
        "temp_entries": temp_entries,
        "nonempty_temp_entry_count": len(nonempty_temp),
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "issues": len(issues), "nonempty_temp": len(nonempty_temp)}))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
