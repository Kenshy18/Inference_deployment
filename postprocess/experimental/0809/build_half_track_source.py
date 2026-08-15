#!/usr/bin/env python3
"""Build a deterministic, whole-track half-size Phase-2 input set."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import run_phase1


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=run_phase1.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    return parser.parse_args()


def closest_subset(track_rows: list[tuple[str, int]], target: int) -> list[str]:
    """Subset-sum whole tracks to the row count closest to ``target``."""
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for track_id, count in track_rows:
        updates = {
            total + int(count): (*selected, str(track_id))
            for total, selected in list(reachable.items())
        }
        for total, selected in updates.items():
            reachable.setdefault(total, selected)
    best = min(reachable, key=lambda total: (abs(total - target), total > target, total))
    return list(reachable[best])


def copy_subset(source: Path, output: Path, selected: list[str]) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        source_rows = int(src.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
        source_tracks = int(src.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
        with sqlite3.connect(output) as dst:
            src.backup(dst)
    with sqlite3.connect(output) as db:
        marks = ",".join("?" for _ in selected)
        with db:
            db.execute(
                f"DELETE FROM masks WHERE CAST(track_id AS TEXT) NOT IN ({marks})",
                selected,
            )
            db.execute(
                f"DELETE FROM tracks WHERE CAST(track_id AS TEXT) NOT IN ({marks})",
                selected,
            )
            db.execute(
                f"DELETE FROM raw_tracked_masks WHERE final_track_id IS NULL "
                f"OR CAST(final_track_id AS TEXT) NOT IN ({marks})",
                selected,
            )
            db.execute(
                f"DELETE FROM raw_tracks WHERE final_track_id IS NULL "
                f"OR CAST(final_track_id AS TEXT) NOT IN ({marks})",
                selected,
            )
        db.execute("VACUUM")
        subset_rows = int(db.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
        subset_tracks = int(db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
        integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
        frame_range = db.execute("SELECT MIN(frame), MAX(frame) FROM masks").fetchone()
    if integrity != "ok":
        raise RuntimeError(f"subset SQLite failed integrity check: {output}: {integrity}")
    return {
        "source_sqlite": str(source),
        "subset_sqlite": str(output),
        "source_rows": source_rows,
        "subset_rows": subset_rows,
        "row_fraction": subset_rows / max(source_rows, 1),
        "source_tracks": source_tracks,
        "subset_tracks": subset_tracks,
        "selected_track_ids": selected,
        "first_frame": frame_range[0],
        "last_frame": frame_range[1],
        "integrity_check": integrity,
    }


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    sources = run_phase1._discover_inputs(args.source_root)
    prepared = args.output_root / "prepared"
    reports = []
    groups = []
    for index, label in enumerate(run_phase1.LABELS):
        source = sources[label]
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as db:
            track_rows = [
                (str(track_id), int(count))
                for track_id, count in db.execute(
                    "SELECT CAST(track_id AS TEXT), COUNT(*) FROM masks "
                    "GROUP BY CAST(track_id AS TEXT) ORDER BY CAST(track_id AS TEXT)"
                )
            ]
        target = round(sum(count for _track, count in track_rows) * args.fraction)
        selected = closest_subset(track_rows, target)
        subset = prepared / f"{index:02d}_{label}.sqlite"
        report = copy_subset(source, subset, selected)
        report["label"] = label
        report["target_rows"] = target
        reports.append(report)
        pipeline_manifest = prepared / f"{index:02d}_{label}_pipeline_manifest.json"
        pipeline_manifest.write_text(
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
        groups.append({"labels": [label], "pipeline_manifest": str(pipeline_manifest)})
    manifest = (
        args.output_root
        / "interval_10/production_raw/work/04_classwise_postprocess/classwise_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"groups": groups}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "privacy": "SQLite geometry only; no video frame was decoded.",
        "fraction_requested": args.fraction,
        "classes": reports,
        "total_source_rows": sum(item["source_rows"] for item in reports),
        "total_subset_rows": sum(item["subset_rows"] for item in reports),
    }
    payload["total_row_fraction"] = payload["total_subset_rows"] / payload["total_source_rows"]
    (args.output_root / "half_source_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
