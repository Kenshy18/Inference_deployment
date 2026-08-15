#!/usr/bin/env python3
"""Fail-fast contract audit for every completed 0809 matrix cell."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "output" / "production_raw_only_0809_20260809"
LABELS = ("女性器", "男性器", "結合部分")
VARIANTS = ("production_raw", "production_raw_hard_recall")


def _topology(path: Path) -> dict[str, int | str]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        keys, multi, maximum = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(n > 1), 0), COALESCE(MAX(n), 0)
            FROM (
                SELECT k.id, COUNT(DISTINCT c.slot_index) AS n
                FROM mask_keyframes k
                JOIN mask_track_segments s ON s.id=k.segment_id
                JOIN tracks t ON t.track_id=s.track_id
                JOIN keyframe_components c ON c.keyframe_id=k.id
                WHERE t.domain='genital'
                GROUP BY k.id
            )
            """
        ).fetchone()
    return {
        "integrity": integrity,
        "foreign_key_errors": int(foreign_keys),
        "keyframes_with_components": int(keys),
        "multi_component_keyframes": int(multi),
        "max_components": int(maximum),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    failures: list[str] = []
    rows = []
    fingerprints = []
    for interval in (1, 3, 5, 8, 10):
        for variant in VARIANTS:
            report_path = output_root / f"interval_{interval}" / variant / "metrics.json"
            if not report_path.is_file():
                failures.append(f"missing metrics: interval={interval} variant={variant}")
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = Path(report["result_sqlite"])
            topology = _topology(result)
            aggregate = report["aggregate"]
            contract = report["sqlite"]
            fingerprints.append(contract["schema_fingerprint"])
            cell = f"interval={interval} variant={variant}"
            if abs(float(aggregate["coverage_ratio"]) - 1.0) > 1e-12:
                failures.append(f"{cell}: incomplete geometry coverage")
            if topology["integrity"] != "ok" or topology["foreign_key_errors"]:
                failures.append(f"{cell}: SQLite integrity failure")
            policies = {
                (item["label"], item["shape_mode"], item["keyframe_interval"], item["max_gap"])
                for item in contract["policies"]
                if item["label"] in LABELS
            }
            expected_policies = {(label, "polygon", interval, 15) for label in LABELS}
            if policies != expected_policies:
                failures.append(f"{cell}: class policy mismatch: {sorted(policies)}")
            shapes = {
                (item["label"], item["shape_type"])
                for item in contract["segment_shape_counts"]
                if item["label"] in LABELS
            }
            if shapes != {(label, "polygon") for label in LABELS}:
                failures.append(f"{cell}: non-polygon output: {sorted(shapes)}")
            if variant == "production_raw":
                states = [
                    float(item["mean_state_count"])
                    for item in report["production_polygon_summaries"]
                ]
                if len(states) != 3 or any(abs(value - 1.0) > 1e-12 for value in states):
                    failures.append(f"{cell}: Production candidate states are not raw-only")
            else:
                if int(aggregate["recall_violations"]) != 0:
                    failures.append(f"{cell}: hard Recall violation remains")
                if float(aggregate["recall_min"]) + 1e-12 < 0.97:
                    failures.append(f"{cell}: minimum Recall below 0.97")
            # This dataset has one legitimate two-component key. It must not be flattened.
            if topology["multi_component_keyframes"] != 1 or topology["max_components"] != 2:
                failures.append(f"{cell}: connected-component topology changed: {topology}")
            rows.append(
                {
                    "interval": interval,
                    "variant": variant,
                    "result_sqlite": str(result.resolve()),
                    "coverage": aggregate["coverage_ratio"],
                    "recall_min": aggregate["recall_min"],
                    "recall_violations": aggregate["recall_violations"],
                    **topology,
                }
            )
    if fingerprints and any(item != fingerprints[0] for item in fingerprints[1:]):
        failures.append("schema fingerprints differ across matrix cells")
    payload = {
        "passed": not failures,
        "completed_cells": len(rows),
        "expected_cells": 10,
        "single_schema_fingerprint": bool(fingerprints) and not any(
            item != fingerprints[0] for item in fingerprints[1:]
        ),
        "failures": failures,
        "rows": rows,
    }
    (output_root / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("passed", "completed_cells", "failures")}, ensure_ascii=False))
    return 0 if payload["passed"] and len(rows) == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
