#!/usr/bin/env python3
"""Build one deterministic medium-length whole-track source per class."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import run_phase1
from build_half_track_source import copy_subset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=650)
    args = parser.parse_args()
    sources = run_phase1._discover_inputs(args.source_root.resolve())
    prepared = args.output_root.resolve() / "prepared"
    groups = []
    reports = []
    for index, label in enumerate(run_phase1.LABELS):
        source = sources[label]
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as db:
            rows = [
                (str(track), int(count))
                for track, count in db.execute(
                    "SELECT CAST(track_id AS TEXT), COUNT(*) FROM masks "
                    "GROUP BY CAST(track_id AS TEXT)"
                )
            ]
        selected_track, selected_rows = min(
            rows,
            key=lambda value: (
                abs(int(value[1]) - int(args.target_rows)),
                int(value[1]) < 200,
                str(value[0]),
            ),
        )
        subset = prepared / f"{index:02d}_{label}.sqlite"
        report = copy_subset(source, subset, [selected_track])
        report.update(
            {
                "label": label,
                "target_rows": int(args.target_rows),
                "selected_track": selected_track,
                "selected_rows": selected_rows,
            }
        )
        reports.append(report)
        manifest = prepared / f"{index:02d}_{label}_pipeline_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "stages": [
                        {
                            "id": "polygon_optimization",
                            "metadata": {"optimizer": {"input_sqlite": str(subset)}},
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        groups.append({"labels": [label], "pipeline_manifest": str(manifest)})
    classwise = (
        args.output_root.resolve()
        / "interval_10/production_raw/work/04_classwise_postprocess/classwise_manifest.json"
    )
    classwise.parent.mkdir(parents=True, exist_ok=True)
    classwise.write_text(
        json.dumps({"groups": groups}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "privacy": "SQLite geometry only; no video frame was decoded.",
        "target_rows": int(args.target_rows),
        "classes": reports,
        "total_rows": sum(int(value["subset_rows"]) for value in reports),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "representative_source_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
