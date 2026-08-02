#!/usr/bin/env python3
"""Create a relocatable production asset pack outside the Git repository."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from asset_tools import iter_included_files, selected_artifacts, sha256_file, verify_artifact


def copy_file(source: Path, target: Path, *, link: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=("core", "all"), default="all")
    parser.add_argument("--link", action="store_true", help="local QA only; create hard links")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    artifacts = selected_artifacts(args.profile, "runtime")
    errors = [
        error
        for item in artifacts
        for error in verify_artifact(root, item, full_hash=True)
    ]
    if errors:
        raise RuntimeError("asset source is invalid:\n" + "\n".join(errors))
    destination.mkdir(parents=True)
    records: list[dict[str, object]] = []
    copied: set[Path] = set()
    for item in artifacts:
        source = root / item["path"]
        if item.get("type", "file") == "directory":
            files = iter_included_files(source, item.get("include"))
        else:
            files = [source]
        for source_file in files:
            relative = source_file.relative_to(root)
            if relative in copied:
                continue
            copied.add(relative)
            target = destination / relative
            copy_file(source_file, target, link=args.link)
            records.append(
                {
                    "path": relative.as_posix(),
                    "size": source_file.stat().st_size,
                    "sha256": sha256_file(source_file),
                }
            )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pack = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_commit": commit,
        "profile": args.profile,
        "hard_linked_local_qa_pack": args.link,
        "files": records,
    }
    (destination / "ASSET_PACK.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PASS] asset pack: {destination} ({len(records)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
