#!/usr/bin/env python3
"""Validate runtime-only acceleration against a frozen polygon reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .config import LABELS, POLYGON_PROFILE_ID
from .validation import audit_sqlite, compare_sqlite_tables


EXACT_ARTIFACTS = (
    "runtime/opt/final_keyframes.json",
    "runtime/opt/interpolated_union.json",
    "runtime/exact/keyframe_exact_metrics.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_root(path: Path, label: str) -> Path:
    return path / POLYGON_PROFILE_ID / label


def validate(reference: Path, candidate: Path) -> dict[str, object]:
    labels: dict[str, object] = {}
    passed = True
    for label in LABELS:
        reference_label = _run_root(reference, label)
        candidate_label = _run_root(candidate, label)
        artifacts: dict[str, object] = {}
        for relative in EXACT_ARTIFACTS:
            reference_path = reference_label / relative
            candidate_path = candidate_label / relative
            reference_hash = _sha256(reference_path)
            candidate_hash = _sha256(candidate_path)
            equal = reference_hash == candidate_hash
            artifacts[relative] = {
                "equal": equal,
                "reference_sha256": reference_hash,
                "candidate_sha256": candidate_hash,
            }
            passed = passed and equal
        reference_sqlite = reference_label / "runtime/pred/predictions.sqlite"
        candidate_sqlite = candidate_label / "runtime/pred/predictions.sqlite"
        semantic = compare_sqlite_tables(
            reference_sqlite,
            candidate_sqlite,
            tables=("masks",),
            float_tolerance=0.0,
        )
        reference_audit = audit_sqlite(reference_sqlite)
        candidate_audit = audit_sqlite(candidate_sqlite)
        passed = passed and semantic.equal and reference_audit.ok and candidate_audit.ok
        labels[label] = {
            "artifacts": artifacts,
            "predictions_sqlite": {
                "semantic": semantic.to_dict(),
                "reference_audit": reference_audit.to_dict(),
                "candidate_audit": candidate_audit.to_dict(),
            },
        }
    return {
        "schema_version": 1,
        "passed": passed,
        "privacy": "SQLite/JSON geometry only; no video frames were opened.",
        "reference_root": str(reference.resolve()),
        "candidate_root": str(candidate.resolve()),
        "labels": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.reference_root, args.candidate_root)
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
