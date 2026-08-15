#!/usr/bin/env python3
"""Independent integrity and exact-metric audit for the 1..6 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


LABELS = ("女性器", "男性器", "結合部分")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(root: Path, arm: str, interval: int) -> tuple[list[Path], list[Path], list[Path]]:
    if arm == "legacy_production":
        stage = next((root / arm / f"interval_{interval}").glob("*_polygon_optimization"))
        return (
            [stage / "vendor_output/exact/keyframe_exact_metrics.csv"],
            [stage / "vendor_output/opt/final_keyframes.json"],
            [stage / "predictions.sqlite"],
        )
    base = root / arm / f"interval_{interval}/polygon14_keyframe_v1"
    return (
        [base / label / "runtime/exact/keyframe_exact_metrics.csv" for label in LABELS],
        [base / label / "runtime/opt/final_keyframes.json" for label in LABELS],
        [base / label / "runtime/pred/predictions.sqlite" for label in LABELS],
    )


def validate(root: Path, comparison: dict[str, object]) -> dict[str, object]:
    checks = []
    for row in comparison["rows"]:
        arm = str(row["arm"])
        interval = int(row["target_interval"])
        exacts, keypaths, predpaths = paths(root, arm, interval)
        recalls: list[float] = []
        ious: list[float] = []
        for path in exacts:
            with path.open(encoding="utf-8", newline="") as handle:
                for item in csv.DictReader(handle):
                    recalls.append(float(item["recall"]))
                    ious.append(float(item["iou"]))
        keyframes = sum(
            len(json.loads(path.read_text(encoding="utf-8"))) for path in keypaths
        )
        prediction_rows = 0
        for path in predpaths:
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
                prediction_rows += int(db.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
        result = Path(str(row["result_sqlite"]))
        with sqlite3.connect(f"file:{result.resolve()}?mode=ro", uri=True) as db:
            integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_keys = len(db.execute("PRAGMA foreign_key_check").fetchall())
        passed = all(
            (
                len(ious) == int(row["evaluated_rows"]),
                keyframes == int(row["keyframes"]),
                prediction_rows == int(row["prediction_rows"]),
                abs(min(recalls) - float(row["recall_min"])) < 1e-12,
                abs(sum(ious) / len(ious) - float(row["iou_mean"])) < 1e-12,
                sum(value < 0.97 - 1e-9 for value in recalls)
                == int(row["recall_violations"]),
                integrity == "ok",
                foreign_keys == 0,
                sha256(result) == row["result_sqlite_sha256"],
            )
        )
        checks.append(
            {
                "arm": arm,
                "target_interval": interval,
                "passed": passed,
                "exact_rows": len(ious),
                "keyframes": keyframes,
                "prediction_rows": prediction_rows,
                "integrity": integrity,
                "foreign_key_errors": foreign_keys,
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError("independent comparison validation failed")
    rows = comparison["rows"]
    return {
        "status": "share_with_caveats",
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "independent_exact_recalculation_passed": True,
        "all_sqlite_integrity_ok": True,
        "all_foreign_keys_ok": True,
        "single_schema_fingerprint": len({row["result_schema_sha256"] for row in rows}) == 1,
        "candidate_recall_gate_6_of_6": all(
            int(row["recall_violations"]) == 0
            for row in rows
            if row["arm"] == "production_candidate_20260814"
        ),
        "invalid_polygon_rings": sum(int(row["invalid_polygon_rings"]) for row in rows),
        "limitations": [
            "No human GT; arm-local references differ after NMS/tracking.",
            "Single KPI video.",
            "Common-raw metric uses AI detections as reference.",
            "Browser QA unavailable; structural HTML QA passed.",
        ],
        "privacy": "No video frames decoded or uploaded.",
        "cells": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    report = validate(args.root.resolve(), comparison)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cells"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
