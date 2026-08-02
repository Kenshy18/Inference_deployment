#!/usr/bin/env python3
"""Validate manually installed weights and prepared runtime artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from asset_tools import selected_artifacts, verify_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=("core", "all"), default="core")
    parser.add_argument("--stage", choices=("source", "runtime"), default="runtime")
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    artifacts = selected_artifacts(args.profile, args.stage)
    errors = [
        error
        for item in artifacts
        for error in verify_artifact(root, item, full_hash=args.full_hash)
    ]
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        f"[PASS] {len(artifacts)} external asset groups: "
        f"profile={args.profile} stage={args.stage} root={root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
