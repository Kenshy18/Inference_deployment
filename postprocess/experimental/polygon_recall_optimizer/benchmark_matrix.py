#!/usr/bin/env python3
"""Compare Production and superior Pareto SQLite outputs at target intervals.

The benchmark is geometry-only: it never opens video pixels.  Every output is
reloaded from SQLite and evaluated against the same raw masks so exported
geometry, interpolation metadata, and schema integrity are part of the result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from .audit_superior import _vertex_safety_audit
from .fixed_budget import evaluate_segments, load_raw_masks, load_segments, summarize
from .sqlite_export import schema_fingerprint
from .superior import evaluate_direct


TARGETS = (1, 3, 5, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--superior-dir", type=Path, required=True)
    parser.add_argument("--production-log-dir", type=Path, required=True)
    parser.add_argument("--production-current-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    return parser.parse_args()


def _quantile(values: list[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def _achievement_score(target: float, actual: float) -> float:
    """Symmetric closeness score: 100 is exact, over/undershoot are equivalent."""

    return 100.0 * min(target / actual, actual / target)


def _mean_interval(segments, keyframe_count: int) -> float:
    segment_count = 0
    total_span = 0
    for values in segments.values():
        for segment in values:
            if not segment.keyframes:
                continue
            segment_count += 1
            total_span += segment.keyframes[-1].frame - segment.keyframes[0].frame
    return total_span / max(keyframe_count - segment_count, 1)


def _read_manifest(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8")
    marker = '{\n  "schema_version": 1,\n  "pipeline"'
    offset = text.find(marker)
    if offset < 0:
        raise ValueError(f"pipeline manifest not found in {log_path}")
    return json.loads(text[offset:])


def _production_timing(args: argparse.Namespace, target: int) -> dict[str, object]:
    if target in {1, 10}:
        root = args.production_current_dir / f"interval_{target}"
        log_path = root / "run.log"
        time_path = root / "time.txt"
    else:
        log_path = args.production_log_dir / f"interval_{target}.log"
        time_path = None
    manifest = _read_manifest(log_path)
    classwise = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "classwise_postprocess"
    )
    male = next(
        group
        for group in classwise["metadata"]["group_summaries"]
        if group["labels"] == [args.label]
    )
    wall_seconds = None
    max_rss_kib = None
    if time_path is not None:
        time_text = time_path.read_text(encoding="utf-8")
        wall_match = re.search(
            r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)",
            time_text,
        )
        rss_match = re.search(
            r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text
        )
        if wall_match:
            fields = [float(value) for value in wall_match.group(1).split(":")]
            wall_seconds = 0.0
            for value in fields:
                wall_seconds = wall_seconds * 60.0 + value
        if rss_match:
            max_rss_kib = int(rss_match.group(1))
    return {
        "male_polygon_group_seconds": float(male["elapsed_seconds"]),
        "classwise_seconds": float(classwise["elapsed_seconds"]),
        "pipeline_stage_sum_seconds": float(
            sum(float(stage["elapsed_seconds"]) for stage in manifest["stages"])
        ),
        "measured_wall_seconds": wall_seconds,
        "max_rss_kib": max_rss_kib,
        "source": str(log_path.resolve()),
    }


def _superior_timing(root: Path) -> dict[str, object]:
    report = json.loads((root / "superior_pareto_report.json").read_text())
    time_text = (root / "time.txt").read_text(encoding="utf-8")
    wall_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", time_text
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
    fields = [float(value) for value in wall_match.group(1).split(":")]
    wall_seconds = 0.0
    for value in fields:
        wall_seconds = wall_seconds * 60.0 + value
    return {
        "optimizer_seconds": float(report["optimizer"]["seconds"]),
        "sqlite_complete_wall_seconds": wall_seconds,
        "max_rss_kib": int(rss_match.group(1)),
        "edge_evaluations": int(report["optimizer"]["edge_evaluations"]),
        "feasible_edges": int(report["optimizer"]["feasible_edges"]),
        "frontier_size": int(report["optimizer"]["frontier_size"]),
        "source": str((root / "time.txt").resolve()),
    }


def _sqlite_audit(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        return {
            "size_bytes": path.stat().st_size,
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[
                0
            ],
            "foreign_key_error_count": len(
                list(connection.execute("PRAGMA foreign_key_check"))
            ),
            "schema_fingerprint": schema_fingerprint(connection),
        }


def _evaluate(
    path: Path,
    raw,
    *,
    label: str,
    start_frame: int,
    end_frame: int,
    target: int,
    recall_floor: float,
) -> dict[str, object]:
    segments = load_segments(
        path, label=label, start_frame=start_frame, end_frame=end_frame
    )
    overlay_rows = evaluate_segments(raw, segments)
    direct_rows = evaluate_direct(raw, segments)
    summary = summarize(
        direct_rows, segments, start_frame=start_frame, end_frame=end_frame
    )
    overlay_summary = summarize(
        overlay_rows, segments, start_frame=start_frame, end_frame=end_frame
    )
    recalls = [float(item.recall) for item in direct_rows]
    ious = [float(item.iou) for item in direct_rows]
    precisions = [float(item.precision) for item in direct_rows]
    area_ratios = [float(item.area_ratio) for item in direct_rows]
    actual = _mean_interval(segments, int(summary["keyframe_count"]))
    vertex = _vertex_safety_audit(segments)
    return {
        "path": str(path.resolve()),
        "target_interval": target,
        "keyframe_count": int(summary["keyframe_count"]),
        "actual_interval": actual,
        "target_absolute_error": abs(actual - target),
        "target_absolute_percentage_error": 100.0 * abs(actual - target) / target,
        "target_achievement_score": _achievement_score(float(target), actual),
        "observed_mask_frames": len(direct_rows),
        "recall_mean": float(np.mean(recalls)),
        "recall_min": min(recalls),
        "recall_q01": _quantile(recalls, 0.01),
        "recall_q05": _quantile(recalls, 0.05),
        "recall_below_floor_count": sum(
            value + 1e-12 < recall_floor for value in recalls
        ),
        "recall_below_floor_rate": sum(
            value + 1e-12 < recall_floor for value in recalls
        )
        / len(recalls),
        "iou_mean": float(np.mean(ious)),
        "iou_min": min(ious),
        "iou_q01": _quantile(ious, 0.01),
        "iou_q05": _quantile(ious, 0.05),
        "precision_mean": float(np.mean(precisions)),
        "precision_q05": _quantile(precisions, 0.05),
        "area_ratio_mean": float(np.mean(area_ratios)),
        "area_ratio_q95": _quantile(area_ratios, 0.95),
        "centroid_error_mean_px": float(summary["centroid_error_mean_px"]),
        "centroid_error_q95_px": float(summary["centroid_error_q95_px"]),
        "predicted_adjacent_iou_mean": float(summary["predicted_adjacent_iou_mean"]),
        "predicted_area_log_delta_mean": float(
            summary["predicted_area_log_delta_mean"]
        ),
        "predicted_area_log_delta_q95": float(summary["predicted_area_log_delta_q95"]),
        "predicted_centroid_acceleration_mean_px": float(
            summary["predicted_centroid_acceleration_mean_px"]
        ),
        "predicted_centroid_acceleration_q95_px": float(
            summary["predicted_centroid_acceleration_q95_px"]
        ),
        "overlay_direct_iou_mean_delta": float(
            overlay_summary["iou_mean"] - summary["iou_mean"]
        ),
        "vertex_safety": vertex,
        "sqlite": _sqlite_audit(path),
    }


def main() -> int:
    args = parse_args()
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    rows: list[dict[str, object]] = []
    frontier_hashes: list[str] = []
    for target in TARGETS:
        production_path = (
            args.production_dir / f"12月KPI動画_旧Production_目標間隔{target}.sqlite"
        )
        superior_root = args.superior_dir / f"interval_{target}"
        superior_path = (
            superior_root / f"12月KPI動画_新Pareto_目標間隔{target}_Recall097.sqlite"
        )
        production = _evaluate(
            production_path,
            raw,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target=target,
            recall_floor=args.recall_floor,
        )
        production["algorithm"] = "Production"
        production["timing"] = _production_timing(args, target)
        rows.append(production)
        superior = _evaluate(
            superior_path,
            raw,
            label=args.label,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target=target,
            recall_floor=args.recall_floor,
        )
        superior["algorithm"] = "Superior Pareto"
        superior["timing"] = _superior_timing(superior_root)
        rows.append(superior)
        report = json.loads((superior_root / "superior_pareto_report.json").read_text())
        frontier_hashes.append(
            hashlib.sha256(
                json.dumps(
                    report["frontier"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )

    schema_fingerprints = {item["sqlite"]["schema_fingerprint"] for item in rows}
    reference_report = json.loads(
        (args.superior_dir / "interval_10" / "superior_pareto_report.json").read_text()
    )
    frontier = reference_report["frontier"]
    same_budget = []
    for target in TARGETS:
        production = next(
            item
            for item in rows
            if item["algorithm"] == "Production" and item["target_interval"] == target
        )
        candidate = max(
            (
                point
                for point in frontier
                if point["keyframe_count"] <= production["keyframe_count"]
            ),
            key=lambda point: point["mean_iou"],
        )
        same_budget.append(
            {
                "target_interval": target,
                "keyframe_budget": production["keyframe_count"],
                "actual_interval": production["actual_interval"],
                "production_mean_iou": production["iou_mean"],
                "production_min_recall": production["recall_min"],
                "superior_keyframes": candidate["keyframe_count"],
                "superior_mean_iou": candidate["mean_iou"],
                "superior_min_recall": candidate["min_recall"],
                "mean_iou_delta": candidate["mean_iou"] - production["iou_mean"],
            }
        )
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "scope": {
            "source_sqlite": str(args.source_sqlite.resolve()),
            "label": args.label,
            "frame_range": [args.start_frame, args.end_frame],
            "raw_mask_count": len(raw),
            "recall_floor": args.recall_floor,
            "targets": list(TARGETS),
        },
        "definitions": {
            "actual_interval": "sum(last_key-first_key) / sum(key_count-1), segment-bounded",
            "target_achievement_score": "100 * min(target/actual, actual/target)",
            "quality_reference": "same exported raw AI masks for both algorithms",
            "quality_path": "stored point_index interpolation reloaded from SQLite",
        },
        "rows": rows,
        "same_keyframe_budget_comparison": same_budget,
        "validation": {
            "all_integrity_ok": all(
                item["sqlite"]["integrity_check"] == "ok" for item in rows
            ),
            "foreign_key_error_total": sum(
                item["sqlite"]["foreign_key_error_count"] for item in rows
            ),
            "schema_fingerprint_count": len(schema_fingerprints),
            "frontier_hashes": frontier_hashes,
            "superior_frontier_deterministic": len(set(frontier_hashes)) == 1,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for item in rows:
        flat_rows.append(
            {
                key: value
                for key, value in item.items()
                if not isinstance(value, (dict, list))
            }
            | {
                "non_rigid_motion_mean": item["vertex_safety"]["non_rigid_motion"][
                    "mean"
                ],
                "non_rigid_motion_q99": item["vertex_safety"]["non_rigid_motion"][
                    "q99"
                ],
                "invalid_integer_frames": item["vertex_safety"][
                    "invalid_integer_frame_count"
                ],
            }
        )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload["validation"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
