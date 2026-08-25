"""Build the canonical portable-report artifact from validated summaries."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("output/geometry_tradeoff_kpi_15min_20260825/analysis")


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _pick(rows: list[dict[str, str]], geometry: str, interval: int) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["geometry"] == geometry and int(row["target_interval"]) == interval
    )


def main() -> None:
    summary = _rows("summary.csv")
    by_class = _rows("summary_by_class.csv")
    manifest = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    polygon = [row for row in summary if row["geometry"] == "polygon"]
    curve = [row for row in summary if row["geometry"] == "catmull_rom"]
    polygon3 = _pick(summary, "polygon", 3)
    curve4 = _pick(summary, "catmull_rom", 4)
    curve7 = _pick(summary, "catmull_rom", 7)
    polygon7 = _pick(summary, "polygon", 7)
    median_polygon_fps = statistics.median(_number(row, "source_video_fps") for row in polygon)
    median_curve_fps = statistics.median(_number(row, "source_video_fps") for row in curve)
    speed_gain = median_curve_fps / median_polygon_fps - 1.0
    equal_iou_gain = _number(curve4, "iou_mean") - _number(polygon3, "iou_mean")
    equal_tail_gain = _number(curve4, "iou_q05") - _number(polygon3, "iou_q05")
    equal_area_gain = _number(curve4, "area_ratio_q95") - _number(
        polygon3, "area_ratio_q95"
    )

    table_rows = []
    for row in summary:
        table_rows.append(
            {
                "geometry": "Catmull–Rom" if row["geometry"] == "catmull_rom" else "Polygon",
                "target": int(row["target_interval"]),
                "actual_interval": _number(row, "effective_interval"),
                "fps": _number(row, "source_video_fps"),
                "mean_iou": _number(row, "iou_mean"),
                "q05_iou": _number(row, "iou_q05"),
                "min_iou": _number(row, "iou_minimum"),
                "min_recall": _number(row, "recall_minimum"),
                "area_q95": _number(row, "area_ratio_q95"),
                "area_max": _number(row, "area_ratio_maximum"),
                "mean_points": _number(row, "editable_points_mean_per_keyframe"),
                "keyframes": int(row["keyframe_rows"]),
            }
        )
    chart_rows = []
    for row in table_rows:
        for metric, value in (
            ("Actual interval", row["actual_interval"]),
            ("Throughput", row["fps"]),
            ("Mean IoU", row["mean_iou"]),
            ("5th percentile IoU", row["q05_iou"]),
            ("95th percentile area ratio", row["area_q95"]),
            ("Mean editable points", row["mean_points"]),
        ):
            chart_rows.append(
                {
                    "geometry": row["geometry"],
                    "target": row["target"],
                    "metric": metric,
                    "value": value,
                }
            )
    class_rows = [
        {
            "geometry": "Catmull–Rom" if row["geometry"] == "catmull_rom" else "Polygon",
            "target": int(row["target_interval"]),
            "class": row["label"],
            "mean_iou": _number(row, "iou_mean"),
            "q05_iou": _number(row, "iou_q05"),
            "min_iou": _number(row, "iou_minimum"),
            "area_q95": _number(row, "area_ratio_q95"),
            "keyframes": int(row["keyframe_rows"]),
        }
        for row in by_class
    ]

    source_id = "geometry_benchmark_analysis"
    report = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Polygon vs Catmull–Rom: real-video geometry trade-off",
            "description": (
                "Fourteen-run benchmark across target keyframe intervals 1–7 on "
                "the V3 KPI masklet."
            ),
            "generatedAt": manifest["generated_at"],
            "cards": [
                {
                    "id": "headline",
                    "description": "Validated benchmark headlines.",
                    "dataset": "headline",
                    "sourceId": source_id,
                    "metrics": [
                        {"label": "Polygon median FPS", "field": "polygon_fps", "format": "number"},
                        {"label": "Curve median FPS", "field": "curve_fps", "format": "number"},
                        {"label": "Curve speed gain", "field": "speed_gain", "format": "percent", "signed": True},
                        {"label": "Recall violations", "field": "recall_violations", "format": "number"},
                    ],
                }
            ],
            "charts": [
                {
                    "id": "target_attainment",
                    "title": "Curve quality comes with an interval ceiling",
                    "subtitle": "Polygon continues toward sparse targets; Catmull–Rom plateaus near 3.4.",
                    "type": "line",
                    "dataset": "target_attainment",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "target", "type": "ordinal", "label": "Target interval"},
                        "y": {"field": "actual_interval", "type": "quantitative", "label": "Actual effective interval"},
                        "color": {"field": "geometry", "type": "nominal", "label": "Geometry"},
                    },
                },
                {
                    "id": "speed",
                    "title": "End-to-end postprocess throughput",
                    "subtitle": "Video frames divided by classwise wall time; 200 FPS reference is discussed in text.",
                    "type": "line",
                    "dataset": "speed",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "target", "type": "ordinal", "label": "Target interval"},
                        "y": {"field": "fps", "type": "quantitative", "label": "Video FPS"},
                        "color": {"field": "geometry", "type": "nominal", "label": "Geometry"},
                    },
                },
                {
                    "id": "quality",
                    "title": "Mean IoU across the common source population",
                    "subtitle": "Both methods use the same 24,503 original track-frame masks.",
                    "type": "line",
                    "dataset": "quality",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "target", "type": "ordinal", "label": "Target interval"},
                        "y": {"field": "mean_iou", "type": "quantitative", "label": "Mean IoU", "format": "percent"},
                        "color": {"field": "geometry", "type": "nominal", "label": "Geometry"},
                    },
                },
                {
                    "id": "tail_quality",
                    "title": "Low-tail IoU exposes the largest separation",
                    "subtitle": "The 5th percentile is more sensitive than the mean to local breakage.",
                    "type": "line",
                    "dataset": "quality",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "target", "type": "ordinal", "label": "Target interval"},
                        "y": {"field": "q05_iou", "type": "quantitative", "label": "5th percentile IoU", "format": "percent"},
                        "color": {"field": "geometry", "type": "nominal", "label": "Geometry"},
                    },
                },
                {
                    "id": "expansion",
                    "title": "Curve suppresses high-tail expansion",
                    "subtitle": "95th-percentile output/source area ratio; lower is tighter subject to Recall ≥ 0.97.",
                    "type": "line",
                    "dataset": "quality",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "target", "type": "ordinal", "label": "Target interval"},
                        "y": {"field": "area_q95", "type": "quantitative", "label": "Area ratio p95"},
                        "color": {"field": "geometry", "type": "nominal", "label": "Geometry"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "complete_results",
                    "title": "All 14 benchmark points",
                    "subtitle": "Exact values supporting every headline conclusion.",
                    "dataset": "results",
                    "sourceId": source_id,
                    "defaultSort": {"field": "target", "direction": "asc"},
                    "columns": [
                        {"field": "geometry", "label": "Geometry", "type": "text"},
                        {"field": "target", "label": "Target", "format": "number"},
                        {"field": "actual_interval", "label": "Actual interval", "format": "number"},
                        {"field": "fps", "label": "FPS", "format": "number"},
                        {"field": "mean_iou", "label": "Mean IoU", "format": "percent"},
                        {"field": "q05_iou", "label": "IoU p05", "format": "percent"},
                        {"field": "min_iou", "label": "Min IoU", "format": "percent"},
                        {"field": "min_recall", "label": "Min Recall", "format": "percent"},
                        {"field": "area_q95", "label": "Area p95", "format": "number"},
                        {"field": "mean_points", "label": "Points/key", "format": "number"},
                        {"field": "keyframes", "label": "Keys", "format": "number"},
                    ],
                },
                {
                    "id": "class_results",
                    "title": "Class-level quality",
                    "subtitle": "Use this table to locate class-specific degradation.",
                    "dataset": "class_results",
                    "sourceId": source_id,
                    "defaultSort": {"field": "target", "direction": "asc"},
                    "columns": [
                        {"field": "geometry", "label": "Geometry", "type": "text"},
                        {"field": "target", "label": "Target", "format": "number"},
                        {"field": "class", "label": "Class", "type": "text"},
                        {"field": "mean_iou", "label": "Mean IoU", "format": "percent"},
                        {"field": "q05_iou", "label": "IoU p05", "format": "percent"},
                        {"field": "min_iou", "label": "Min IoU", "format": "percent"},
                        {"field": "area_q95", "label": "Area p95", "format": "number"},
                        {"field": "keyframes", "label": "Keys", "format": "number"},
                    ],
                },
            ],
            "sources": [
                {
                    "id": source_id,
                    "label": "Validated geometry benchmark summary",
                    "path": "output/geometry_tradeoff_kpi_15min_20260825/analysis/summary.csv",
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Polygon vs Catmull–Rom: real-video geometry trade-off",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "Fourteen complete runs were evaluated on one 23,510-frame (13:04.45) "
                        "V3 KPI video. The curve implementation is the stronger high-quality "
                        "representation, while polygon remains necessary when an actual interval "
                        "above roughly 3.4 is required."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["headline"]},
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "body": (
                        "## Key findings\n\n"
                        f"- Median throughput is **{median_curve_fps:.1f} FPS for Catmull–Rom** "
                        f"versus **{median_polygon_fps:.1f} FPS for polygon** ({speed_gain:+.1%}).\n"
                        f"- At almost equal effective interval 3 (curve target 4 = "
                        f"{_number(curve4, 'effective_interval'):.3f}; polygon target 3 = "
                        f"{_number(polygon3, 'effective_interval'):.3f}), curve improves mean IoU "
                        f"by **{equal_iou_gain:+.2%}**, p05 IoU by **{equal_tail_gain:+.2%}**, "
                        f"and reduces area-ratio p95 by **{-equal_area_gain:.3f}**.\n"
                        f"- Catmull–Rom reaches only **{_number(curve7, 'effective_interval'):.3f}** "
                        f"at target 7; polygon reaches **{_number(polygon7, 'effective_interval'):.3f}**.\n"
                        "- Every run has **zero Recall < 0.97 violations** and **zero topology-invalid outputs**.\n"
                        "- Both methods use the same track budget distribution: 73×14, 7×16, "
                        "6×18, and 7×20 editable points per component."
                    ),
                },
                {"id": "attainment_chart", "type": "chart", "chartId": "target_attainment"},
                {"id": "speed_chart", "type": "chart", "chartId": "speed"},
                {"id": "quality_chart", "type": "chart", "chartId": "quality"},
                {"id": "tail_chart", "type": "chart", "chartId": "tail_quality"},
                {"id": "expansion_chart", "type": "chart", "chartId": "expansion"},
                {"id": "all_results", "type": "table", "tableId": "complete_results"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        "- Source: 93 V3 tracks and 24,503 original track-frame observations.\n"
                        "- Quality population: original observations only. The 587 gap-filled rows "
                        "are excluded from IoU, Recall, and area statistics.\n"
                        "- Frequency population: all 25,090 materialized rows. Effective interval "
                        "is materialized rows divided by keyframe rows.\n"
                        "- IoU and Recall use exact raster evaluation against the V3 source mask. "
                        "Area ratio is output area divided by source-mask area.\n"
                        "- FPS is 23,510 video frames divided by classwise wall time."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Methodology\n\n"
                        "All three classes used the same target in each run. Polygon used the "
                        "quality-preserving CUDA lazy-exact path with two outer class workers and "
                        "three optimizer workers per class. Catmull–Rom used six balanced CPU "
                        "shards with four native threads each and CUDA disabled. Each output SQLite "
                        "was checked for existence, population equality, Recall violations, and "
                        "topology failures."
                    ),
                },
                {"id": "class_table", "type": "table", "tableId": "class_results"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and robustness\n\n"
                        "This is one real 13-minute video, not a multi-video confidence interval. "
                        "The reference is the AI source mask rather than human ground truth, so a "
                        "temporally sensible correction can score worse. Same target values are not "
                        "equal-sparsity comparisons after target 3. Polygon uses GPU while curve is "
                        "CPU-only, so speed is an operational pipeline result rather than a pure "
                        "representation microbenchmark. Maximum area ratios are unstable for tiny "
                        "source masks; p95 is the primary expansion statistic."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Next steps\n\n"
                        "1. Use Catmull–Rom for high-quality settings whose required actual interval "
                        "is at most about 3.4.\n"
                        "2. Keep polygon available for sparse settings 4–7 until curve DP can widen "
                        "its feasible interval without reducing the local-quality guard.\n"
                        "3. Review `review_candidates.csv`, especially curve-negative deltas and both "
                        "methods' lowest-IoU rows, before Production promotion.\n"
                        "4. Repeat the frozen protocol on multiple V3 videos before choosing GUI defaults."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Can the curve keyframe-rescue policy be relaxed selectively so actual "
                        "intervals 4–7 become reachable while preserving its 0.85 local IoU floor? "
                        "Do the same results hold against human-reviewed masks and on edge-contact, "
                        "fast-reversal, and multi-component strata?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": manifest["generated_at"],
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "polygon_fps": median_polygon_fps,
                        "curve_fps": median_curve_fps,
                        "speed_gain": speed_gain,
                        "recall_violations": 0,
                    }
                ],
                "results": table_rows,
                "target_attainment": [
                    {
                        "geometry": row["geometry"],
                        "target": row["target"],
                        "actual_interval": row["actual_interval"],
                    }
                    for row in table_rows
                ],
                "speed": [
                    {"geometry": row["geometry"], "target": row["target"], "fps": row["fps"]}
                    for row in table_rows
                ],
                "quality": table_rows,
                "class_results": class_rows,
                "tradeoff_long": chart_rows,
            },
        },
        "sources": [
            {
                "id": source_id,
                "query": {
                    "engine": "snapshot",
                    "description": (
                        "Aggregates exact per-frame metrics from 14 completed runs; joins curve and "
                        "polygon results to the same source (track_id, frame) population."
                    ),
                    "sql": "SELECT * FROM validated_geometry_benchmark_summary",
                    "code": (
                        "python postprocess/experiments/geometry_tradeoff_kpi/analyze.py"
                    ),
                    "executed_at": manifest["generated_at"],
                },
            }
        ],
        "package_info": {
            "originUrl": "artifact://geometry-tradeoff-kpi-20260825",
            "controls": {"edit": False, "refresh": False},
        },
    }
    path = ROOT / "artifact.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
