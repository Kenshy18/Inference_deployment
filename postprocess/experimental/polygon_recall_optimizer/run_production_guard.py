#!/usr/bin/env python3
"""Run and audit the minimal Production + raw-Recall safety experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from .fixed_budget import (
    FrameEvaluation,
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from .production_recall_guard import guard_production_recall
from .sqlite_export import export_selected_sqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep Production polygon keyframes and add only the minimum local "
            "changes required for a dense raw-mask Recall floor."
        )
    )
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--guard-margin", type=float, default=0.002)
    parser.add_argument("--point-count", type=int, default=23)
    parser.add_argument("--max-anchor-scale", type=float, default=1.50)
    return parser.parse_args()


def _tail_metrics(
    evaluations: list[FrameEvaluation], floor: float
) -> dict[str, object]:
    recalls = np.asarray([item.recall for item in evaluations], dtype=np.float64)
    ious = np.asarray([item.iou for item in evaluations], dtype=np.float64)
    precisions = np.asarray([item.precision for item in evaluations], dtype=np.float64)
    area_ratios = np.asarray(
        [item.area_ratio for item in evaluations], dtype=np.float64
    )
    return {
        "recall_min": float(np.min(recalls)),
        "recall_q001": float(np.quantile(recalls, 0.001)),
        "recall_q01": float(np.quantile(recalls, 0.01)),
        "recall_violation_count": int(np.sum(recalls + 1e-12 < floor)),
        "iou_min": float(np.min(ious)),
        "iou_q001": float(np.quantile(ious, 0.001)),
        "iou_q01": float(np.quantile(ious, 0.01)),
        "iou_mean": float(np.mean(ious)),
        "precision_min": float(np.min(precisions)),
        "precision_q01": float(np.quantile(precisions, 0.01)),
        "precision_mean": float(np.mean(precisions)),
        "area_ratio_q01": float(np.quantile(area_ratios, 0.01)),
        "area_ratio_q99": float(np.quantile(area_ratios, 0.99)),
        "area_ratio_mean": float(np.mean(area_ratios)),
    }


def _key_metrics(segments, *, start_frame: int, end_frame: int, frame_count: int):
    key_count = 0
    interval_span = 0
    interval_count = 0
    for values in segments.values():
        for segment in values:
            frames = [
                keyframe.frame
                for keyframe in segment.keyframes
                if start_frame <= keyframe.frame <= end_frame
            ]
            key_count += len(frames)
            if len(frames) >= 2:
                interval_span += frames[-1] - frames[0]
                interval_count += len(frames) - 1
    return {
        "keyframe_count": key_count,
        "key_frequency_per_observation": key_count / max(frame_count, 1),
        "observations_per_key": frame_count / max(key_count, 1),
        "mean_temporal_key_interval": (
            interval_span / interval_count if interval_count else 0.0
        ),
    }


def _variant_summary(
    evaluations: list[FrameEvaluation],
    segments,
    *,
    start_frame: int,
    end_frame: int,
    recall_floor: float,
) -> dict[str, object]:
    result = summarize(
        evaluations,
        segments,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    result.update(_tail_metrics(evaluations, recall_floor))
    result.update(
        _key_metrics(
            segments,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_count=len(evaluations),
        )
    )
    return result


def _delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    fields = (
        "keyframe_count",
        "mean_temporal_key_interval",
        "recall_min",
        "recall_q01",
        "recall_violation_count",
        "iou_min",
        "iou_q01",
        "iou_mean",
        "precision_mean",
        "area_ratio_mean",
        "excess_area_ratio_mean",
        "predicted_area_log_delta_mean",
        "centroid_residual_acceleration_mean_px",
    )
    output = {}
    for field in fields:
        output[field] = float(after[field]) - float(before[field])
    output["keyframe_count_percent"] = (
        100.0
        * (float(after["keyframe_count"]) - float(before["keyframe_count"]))
        / max(float(before["keyframe_count"]), 1.0)
    )
    return output


def _frame_rows(
    baseline: list[FrameEvaluation], guarded: list[FrameEvaluation]
) -> list[dict[str, object]]:
    old = {(item.frame, item.track_id): item for item in baseline}
    new = {(item.frame, item.track_id): item for item in guarded}
    rows = []
    for identity in sorted(old.keys() & new.keys()):
        before = old[identity]
        after = new[identity]
        rows.append(
            {
                "frame": identity[0],
                "track_id": identity[1],
                "segment_id": after.segment_id,
                "production_is_keyframe": int(before.is_keyframe),
                "guarded_is_keyframe": int(after.is_keyframe),
                "production_recall": before.recall,
                "guarded_recall": after.recall,
                "production_iou": before.iou,
                "guarded_iou": after.iou,
                "production_precision": before.precision,
                "guarded_precision": after.precision,
                "production_area_ratio": before.area_ratio,
                "guarded_area_ratio": after.area_ratio,
            }
        )
    return rows


def _restore_requested_class_policy(
    baseline_sqlite: Path, output_sqlite: Path, *, label: str
) -> None:
    """Keep the user-requested policy; achieved spacing belongs in metrics."""

    with sqlite3.connect(
        f"file:{baseline_sqlite.resolve()}?mode=ro", uri=True
    ) as source:
        row = source.execute(
            """
            SELECT label, policy_source, shape_mode, keyframe_interval, max_gap
            FROM class_postprocess_policies WHERE label=?
            """,
            (label,),
        ).fetchone()
    if row is None:
        return
    with sqlite3.connect(output_sqlite) as destination:
        destination.execute(
            """
            INSERT INTO class_postprocess_policies(
                label, policy_source, shape_mode, keyframe_interval, max_gap
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                policy_source=excluded.policy_source,
                shape_mode=excluded.shape_mode,
                keyframe_interval=excluded.keyframe_interval,
                max_gap=excluded.max_gap
            """,
            row,
        )
        destination.commit()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    baseline = load_segments(
        args.baseline_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    if not raw or not baseline:
        raise SystemExit("no matching raw masks or Production polygon segments")

    baseline_evaluations = evaluate_segments(raw, baseline)
    result = guard_production_recall(
        baseline,
        raw,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
        guard_margin=args.guard_margin,
        point_count=args.point_count,
        max_anchor_scale=args.max_anchor_scale,
    )
    guarded_evaluations = evaluate_segments(raw, result.segments)
    violations = [
        item for item in guarded_evaluations if item.recall + 1e-12 < args.recall_floor
    ]
    if violations:
        worst = min(violations, key=lambda item: item.recall)
        raise RuntimeError(
            f"dense Recall guard failed at frame={worst.frame} "
            f"track={worst.track_id}: {worst.recall:.12f}"
        )

    baseline_summary = _variant_summary(
        baseline_evaluations,
        baseline,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )
    guarded_summary = _variant_summary(
        guarded_evaluations,
        result.segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )
    sqlite_export = export_selected_sqlite(
        args.baseline_sqlite,
        args.output_sqlite,
        result.segments,
        raw,
        label=args.label,
        target_mean_key_interval=baseline_summary["mean_temporal_key_interval"],
        recall_floor=args.recall_floor,
        selection_reason="production_recall_guard",
        algorithm="experimental.polygon_recall_optimizer.production_recall_guard",
    )
    _restore_requested_class_policy(
        args.baseline_sqlite, args.output_sqlite, label=args.label
    )

    # Reload the actual deliverable through the same consumer path and repeat
    # the hard constraint check.  This catches export/reader mismatches.
    exported_segments = load_segments(
        args.output_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    exported_evaluations = evaluate_segments(raw, exported_segments)
    exported_summary = _variant_summary(
        exported_evaluations,
        exported_segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=args.recall_floor,
    )
    if int(exported_summary["recall_violation_count"]) != 0:
        raise RuntimeError("exported SQLite violates the requested Recall floor")
    if (
        abs(float(exported_summary["iou_mean"]) - float(guarded_summary["iou_mean"]))
        > 1e-12
    ):
        raise RuntimeError("exported SQLite reconstruction differs from memory result")

    rows = _frame_rows(baseline_evaluations, exported_evaluations)
    with (args.output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "experimental": True,
        "method": "Production keyframes plus dense raw-observation Recall guard",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "baseline_sqlite": str(args.baseline_sqlite.resolve()),
        "output_sqlite": str(args.output_sqlite.resolve()),
        "label": args.label,
        "frame_range": [args.start_frame, args.end_frame],
        "recall_floor": args.recall_floor,
        "internal_guard_floor": min(0.9999, args.recall_floor + args.guard_margin),
        "production_preservation": {
            "all_production_key_positions_retained": True,
            "adjusted_production_key_shapes": result.adjusted_production_keys,
            "added_recall_keys": result.added_recall_keys,
            "temporal_reference_used": False,
        },
        "optimizer": {
            "elapsed_seconds": result.elapsed_seconds,
            "evaluated_edges": result.evaluated_edges,
            "feasible_edges": result.feasible_edges,
            "selection": "fewest additions, then highest mean raw IoU",
        },
        "production": baseline_summary,
        "production_recall_guard": exported_summary,
        "delta_guard_minus_production": _delta(baseline_summary, exported_summary),
        "sqlite_export": sqlite_export,
        "segments": list(result.segment_diagnostics),
    }
    (args.output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "production_keys": baseline_summary["keyframe_count"],
                "guarded_keys": exported_summary["keyframe_count"],
                "added_keys": result.added_recall_keys,
                "adjusted_keys": result.adjusted_production_keys,
                "production_min_recall": baseline_summary["recall_min"],
                "guarded_min_recall": exported_summary["recall_min"],
                "production_iou_mean": baseline_summary["iou_mean"],
                "guarded_iou_mean": exported_summary["iou_mean"],
                "output_sqlite": str(args.output_sqlite),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
