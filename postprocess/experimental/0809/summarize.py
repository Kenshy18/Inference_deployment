#!/usr/bin/env python3
"""Build the compact matrix CSV/Markdown from completed 0809 cells."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "output" / "production_raw_only_0809_20260809"
VARIANTS = ("production_raw", "production_raw_hard_recall")


def _load_rows(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for interval in (1, 3, 5, 8, 10):
        for variant in VARIANTS:
            path = output_root / f"interval_{interval}" / variant / "metrics.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _flat(payload: dict[str, object]) -> dict[str, object]:
    aggregate = payload["aggregate"]
    contract = payload["sqlite"]
    frames = int(contract.get("video_frames", 0))
    if not frames:
        with sqlite3.connect(
            f"file:{Path(payload['result_sqlite']).resolve()}?mode=ro", uri=True
        ) as connection:
            frames = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
    baseline_wall = float(
        payload.get("pipeline_wall_seconds", payload.get("baseline_pipeline_wall_seconds", 0.0))
    )
    guard_wall = float(payload.get("hard_guard_wall_seconds", 0.0))
    guard_optimizer = float(
        payload.get(
            "hard_guard_optimizer_seconds",
            sum(
                float(report["optimizer"]["elapsed_seconds"])
                for report in payload.get("guard_reports", [])
            ),
        )
    )
    total_wall = float(payload.get("total_wall_seconds", baseline_wall))
    summaries = payload.get("production_polygon_summaries", [])
    return {
        "variant": payload["variant"],
        "requested_interval": int(payload["requested_interval"]),
        "video_frames": frames,
        "pipeline_seconds": baseline_wall,
        "pipeline_fps": frames / baseline_wall if baseline_wall and frames else 0.0,
        "guard_seconds": guard_wall,
        "guard_optimizer_seconds": guard_optimizer,
        "total_seconds": total_wall,
        "total_fps": frames / total_wall if total_wall and frames else 0.0,
        "keyframes": int(aggregate["keyframe_count"]),
        "actual_interval": float(aggregate["mean_temporal_key_interval"]),
        "iou_mean": float(aggregate["iou_mean"]),
        "iou_q01": float(aggregate["iou_q01"]),
        "iou_min": float(aggregate["iou_min"]),
        "recall_mean": float(aggregate["recall_mean"]),
        "recall_q01": float(aggregate["recall_q01"]),
        "recall_min": float(aggregate["recall_min"]),
        "recall_violations": int(aggregate["recall_violations"]),
        "precision_mean": float(aggregate["precision_mean"]),
        "area_ratio_mean": float(aggregate["area_ratio_mean"]),
        "area_ratio_q99": float(aggregate["area_ratio_q99"]),
        "coverage": float(aggregate["coverage_ratio"]),
        "mean_state_count": (
            max(float(item["mean_state_count"]) for item in summaries)
            if summaries else ""
        ),
        "integrity": contract["integrity_check"],
        "foreign_key_errors": int(contract["foreign_key_errors"]),
        "result_sqlite": payload["result_sqlite"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    payloads = _load_rows(output_root)
    rows = [_flat(payload) for payload in payloads]
    if not rows:
        raise SystemExit("no completed metrics found")
    csv_path = output_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_rows: list[dict[str, object]] = []
    for payload in payloads:
        for label, values in payload["classes"].items():
            class_rows.append(
                {
                    "variant": payload["variant"],
                    "requested_interval": payload["requested_interval"],
                    "label": label,
                    "raw_observations": values["raw_observations"],
                    "coverage": values["coverage_ratio"],
                    "keyframes": values["keyframe_count"],
                    "actual_interval": values["mean_temporal_key_interval"],
                    "iou_mean": values["iou_mean"],
                    "iou_q01": values["iou_q01"],
                    "iou_min": values["iou_min"],
                    "recall_mean": values["recall_mean"],
                    "recall_q01": values["recall_q01"],
                    "recall_min": values["recall_min"],
                    "recall_violations": values["recall_violations"],
                    "precision_mean": values["precision_mean"],
                    "area_ratio_mean": values["area_ratio_mean"],
                    "area_ratio_q99": values["area_ratio_q99"],
                }
            )
    class_csv_path = output_root / "comparison_by_class.csv"
    with class_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    lines = [
        "# 0809 Production raw-only / hard minimum-Recall comparison",
        "",
        "All three genital classes use polygon output. Production input states are raw-only.",
        "The hard-Recall variant is a diagnostic post-pass that retains Production keys,",
        "adjusts failing keys, and inserts the minimum additional keys needed for per-observation Recall >= 0.97.",
        "",
        "|variant|requested|actual|keys|pipeline s/fps|guard wall/optimizer s|total s/fps|IoU mean/q01/min|Recall mean/q01/min|violations|area mean/q99|",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    names = {"production_raw": "Production", "production_raw_hard_recall": "Hard Recall"}
    for row in rows:
        lines.append(
            "|{name}|{requested_interval}|{actual_interval:.3f}|{keyframes}|{pipeline_seconds:.2f}/{pipeline_fps:.2f}|"
            "{guard_seconds:.2f}/{guard_optimizer_seconds:.2f}|{total_seconds:.2f}/{total_fps:.2f}|{iou_mean:.4f}/{iou_q01:.4f}/{iou_min:.4f}|"
            "{recall_mean:.4f}/{recall_q01:.4f}/{recall_min:.4f}|{recall_violations}|"
            "{area_ratio_mean:.3f}/{area_ratio_q99:.3f}|".format(name=names[row["variant"]], **row)
        )
    lines += [
        "",
        "## Audit",
        "",
        f"Completed cells: {len(rows)}/10.",
        "Geometry coverage, SQLite integrity, foreign keys, polygon-only policies, schema fingerprints,",
        "and Production mean state count are recorded in each `metrics.json`.",
        "No video pixels are opened by the evaluator.",
    ]
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "matrix.json").write_text(
        json.dumps({"rows": payloads}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completed": len(rows),
                "csv": str(csv_path),
                "class_csv": str(class_csv_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
