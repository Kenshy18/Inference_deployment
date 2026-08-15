#!/usr/bin/env python3
"""Verify that optimized new-production artifacts equal the reference run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LABELS = ("女性器", "男性器", "結合部分")
FILES = (
    "opt/final_keyframes.json",
    "opt/interpolated_union.json",
    "exact/keyframe_exact_metrics.csv",
    "pred/predictions.sqlite",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = []
    all_equal = True
    for label in LABELS:
        for relative in FILES:
            reference = args.reference / "new_production_v1" / label / "runtime" / relative
            optimized = args.optimized / "new_production_v1" / label / "runtime" / relative
            reference_hash = _sha256(reference)
            optimized_hash = _sha256(optimized)
            equal = reference_hash == optimized_hash
            all_equal = all_equal and equal
            comparisons.append(
                {
                    "label": label,
                    "artifact": relative,
                    "equal": equal,
                    "reference_sha256": reference_hash,
                    "optimized_sha256": optimized_hash,
                    "reference_bytes": reference.stat().st_size,
                    "optimized_bytes": optimized.stat().st_size,
                }
            )
    payload = {"all_equal": all_equal, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
