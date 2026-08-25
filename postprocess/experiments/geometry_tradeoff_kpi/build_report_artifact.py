"""Build a portable technical report from validated geometry summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("output/geometry_tradeoff_kpi_15min_20260825/analysis"),
    )
    return parser


def _rows(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pick(rows: list[dict[str, str]], geometry: str, interval: int) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["geometry"] == geometry and int(row["target_interval"]) == interval
    )


def _equal_interval_deltas(
    polygon: list[dict[str, str]],
    curve: list[dict[str, str]],
) -> list[dict[str, float | int]]:
    """Linearly compare the two measured Pareto polylines at curve x values."""

    ordered_polygon = sorted(
        polygon,
        key=lambda row: _number(row, "effective_interval"),
    )
    result: list[dict[str, float | int]] = []
    for curve_row in sorted(
        curve,
        key=lambda row: _number(row, "effective_interval"),
    ):
        value = _number(curve_row, "effective_interval")
        for left, right in zip(ordered_polygon, ordered_polygon[1:], strict=False):
            left_value = _number(left, "effective_interval")
            right_value = _number(right, "effective_interval")
            if not left_value <= value <= right_value:
                continue
            alpha = (value - left_value) / max(right_value - left_value, 1e-12)
            row: dict[str, float | int] = {
                "curve_target": int(curve_row["target_interval"]),
                "effective_interval": value,
            }
            for metric in ("iou_mean", "iou_q05", "area_ratio_q95"):
                polygon_value = _number(left, metric) + alpha * (
                    _number(right, metric) - _number(left, metric)
                )
                row[f"curve_{metric}"] = _number(curve_row, metric)
                row[f"polygon_interpolated_{metric}"] = polygon_value
                row[f"curve_minus_polygon_{metric}"] = (
                    _number(curve_row, metric) - polygon_value
                )
            result.append(row)
            break
    return result


def main() -> None:
    root = _parser().parse_args().analysis_dir.resolve()
    summary = _rows(root, "summary.csv")
    by_class = _rows(root, "summary_by_class.csv")
    stability = _rows(root, "stability_summary.csv")
    manifest = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    polygon = [row for row in summary if row["geometry"] == "polygon"]
    curve = [row for row in summary if row["geometry"] == "catmull_rom"]
    targets = sorted({int(row["target_interval"]) for row in summary})
    maximum_target = max(targets)
    final_polygon = _pick(summary, "polygon", maximum_target)
    final_curve = _pick(summary, "catmull_rom", maximum_target)
    equal_interval = _equal_interval_deltas(polygon, curve)
    _write_csv(root / "equal_interval_summary.csv", equal_interval)

    median_polygon_fps = statistics.median(
        _number(row, "source_video_fps") for row in polygon
    )
    median_curve_fps = statistics.median(
        _number(row, "source_video_fps") for row in curve
    )
    speed_gain = median_curve_fps / median_polygon_fps - 1.0
    equal_mean_gains = [
        float(row["curve_minus_polygon_iou_mean"]) for row in equal_interval
    ]
    equal_tail_gains = [
        float(row["curve_minus_polygon_iou_q05"]) for row in equal_interval
    ]
    equal_area_gains = [
        float(row["curve_minus_polygon_area_ratio_q95"]) for row in equal_interval
    ]

    table_rows = [
        {
            "geometry": (
                "Catmull–Rom" if row["geometry"] == "catmull_rom" else "Polygon"
            ),
            "target": int(row["target_interval"]),
            "actual_interval": _number(row, "effective_interval"),
            "fps": _number(row, "source_video_fps"),
            "mean_iou": _number(row, "iou_mean"),
            "q05_iou": _number(row, "iou_q05"),
            "min_iou": _number(row, "iou_minimum"),
            "mean_recall": _number(row, "recall_mean"),
            "min_recall": _number(row, "recall_minimum"),
            "area_mean": _number(row, "area_ratio_mean"),
            "area_q95": _number(row, "area_ratio_q95"),
            "area_max": _number(row, "area_ratio_maximum"),
            "mean_points": _number(row, "editable_points_mean_per_keyframe"),
            "keyframes": int(row["keyframe_rows"]),
        }
        for row in summary
    ]
    stability_index = {
        (row["geometry"], int(row["target_interval"])): row
        for row in stability
        if row["geometry"] != "source_raw"
    }
    for row in table_rows:
        key = (
            "catmull_rom" if row["geometry"] == "Catmull–Rom" else "polygon",
            int(row["target"]),
        )
        stable = stability_index[key]
        row.update(
            {
                "area_accel_q95": _number(stable, "log_area_acceleration_q95"),
                "centroid_accel_q95": _number(
                    stable, "centroid_acceleration_radii_q95"
                ),
                "shape_accel_q95": _number(stable, "shape_acceleration_q95"),
                "area_velocity_error_q95": _number(
                    stable, "log_area_velocity_error_q95"
                ),
                "centroid_velocity_error_q95": _number(
                    stable, "centroid_velocity_error_radii_q95"
                ),
                "shape_velocity_error_q95": _number(
                    stable, "shape_velocity_error_q95"
                ),
            }
        )
    _write_csv(root / "combined_summary.csv", table_rows)
    class_rows = [
        {
            "geometry": (
                "Catmull–Rom" if row["geometry"] == "catmull_rom" else "Polygon"
            ),
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

    same_target_wins = sum(
        _number(_pick(summary, "catmull_rom", target), "iou_mean")
        > _number(_pick(summary, "polygon", target), "iou_mean")
        for target in targets
    )
    class_wins = 0
    class_comparisons = 0
    for target in targets:
        labels = {
            row["label"] for row in by_class if int(row["target_interval"]) == target
        }
        for label in labels:
            compared = {
                row["geometry"]: row
                for row in by_class
                if int(row["target_interval"]) == target and row["label"] == label
            }
            class_comparisons += 1
            class_wins += int(
                _number(compared["catmull_rom"], "iou_mean")
                > _number(compared["polygon"], "iou_mean")
            )

    source_id = "geometry_benchmark_analysis"
    source_path = str(root.relative_to(Path.cwd().resolve()) / "combined_summary.csv")
    source = {
        "id": source_id,
        "label": "Validated quality, stability and speed benchmark summary",
        "path": source_path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "description": (
                "Exact per-frame quality plus temporal stability metrics joined "
                "to the common source (track_id, frame) population."
            ),
            "sql": (
                "SELECT CAST(track_id AS TEXT) AS track_id, frame, label " "FROM masks"
            ),
            "tables_used": ["masks"],
            "code": (
                "python postprocess/experiments/geometry_tradeoff_kpi/analyze.py "
                f"--benchmark-root {Path(manifest['benchmark_root']).relative_to(Path.cwd().resolve())} "
                f"--interval-maximum {maximum_target}; python "
                "postprocess/experiments/geometry_tradeoff_kpi/analyze_stability.py "
                f"--benchmark-root {Path(manifest['benchmark_root']).relative_to(Path.cwd().resolve())} "
                f"--source-sqlite {Path(manifest['source_sqlite']).relative_to(Path.cwd().resolve())} "
                f"--interval-maximum {maximum_target}"
            ),
            "executed_at": manifest["generated_at"],
            "filters": [
                "quality excludes 587 deterministic gap-fill rows",
                f"target intervals 1 through {maximum_target}",
            ],
            "metric_definitions": {
                "effective_interval": "25,090 materialized rows / keyframe rows",
                "iou": "exact raster intersection / union against the source AI mask",
                "recall": "exact raster intersection / source AI-mask area",
                "area_ratio": "output raster area / source AI-mask area",
                "centroid_acceleration_radii_q95": (
                    "95th percentile norm of the output centroid second difference, "
                    "divided by the source-mask equivalent radius"
                ),
                "centroid_velocity_error_radii_q95": (
                    "95th percentile norm of output minus source centroid velocity, "
                    "divided by the source-mask equivalent radius"
                ),
                "shape_acceleration_q95": (
                    "95th percentile second difference of a translation-, rotation- "
                    "and phase-invariant primary-contour Fourier descriptor"
                ),
            },
        },
    }

    cards = [
        {
            "id": "curve_median_fps",
            "description": "Median across targets 1–6.",
            "dataset": "headline",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Catmull–Rom median FPS",
                    "field": "curve_fps",
                    "format": "number",
                },
                {
                    "label": "vs polygon",
                    "field": "speed_gain",
                    "format": "percent",
                    "signed": True,
                },
            ],
        },
        {
            "id": "equal_interval_iou",
            "description": "Piecewise-linear comparison at equal effective interval.",
            "dataset": "headline",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Mean IoU advantage",
                    "field": "equal_iou_gain",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "target6_interval",
            "description": "Achieved effective interval at target 6.",
            "dataset": "headline",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Catmull–Rom actual interval",
                    "field": "curve_final_interval",
                    "format": "number",
                },
                {
                    "label": "Polygon",
                    "field": "polygon_final_interval",
                    "format": "number",
                },
            ],
        },
        {
            "id": "recall_violations",
            "description": "Across all 12 completed runs.",
            "dataset": "headline",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Recall violations",
                    "field": "recall_violations",
                    "format": "number",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "pareto_mean",
            "title": "Mean IoU versus effective keyframe interval",
            "subtitle": "24,503 common source observations; farther right uses fewer keys.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "actual_interval",
                    "type": "quantitative",
                    "label": "Effective interval",
                },
                "y": {
                    "field": "mean_iou",
                    "type": "quantitative",
                    "label": "Mean IoU",
                    "format": "percent",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
        {
            "id": "pareto_tail",
            "title": "5th-percentile IoU versus effective interval",
            "subtitle": "Lower-tail quality across the same 24,503 observations.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "actual_interval",
                    "type": "quantitative",
                    "label": "Effective interval",
                },
                "y": {
                    "field": "q05_iou",
                    "type": "quantitative",
                    "label": "IoU p05",
                    "format": "percent",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
        {
            "id": "area",
            "title": "95th-percentile area ratio versus effective interval",
            "subtitle": "Lower is tighter while every run maintains minimum Recall 0.97.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "actual_interval",
                    "type": "quantitative",
                    "label": "Effective interval",
                },
                "y": {
                    "field": "area_q95",
                    "type": "quantitative",
                    "label": "Area ratio p95",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
        {
            "id": "speed",
            "title": "End-to-end postprocess throughput",
            "subtitle": "23,510 video frames divided by classwise wall time.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "target",
                    "type": "ordinal",
                    "label": "Target interval",
                },
                "y": {
                    "field": "fps",
                    "type": "quantitative",
                    "label": "Video FPS",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
        {
            "id": "stability_motion",
            "title": "Centroid acceleration versus effective interval",
            "subtitle": "95th percentile, normalized by source-mask radius; lower is smoother.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "actual_interval",
                    "type": "quantitative",
                    "label": "Effective interval",
                },
                "y": {
                    "field": "centroid_accel_q95",
                    "type": "quantitative",
                    "label": "Centroid acceleration p95",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
        {
            "id": "motion_fidelity",
            "title": "Centroid-velocity error versus effective interval",
            "subtitle": "95th percentile difference from source-mask motion; lower follows motion better.",
            "type": "line",
            "dataset": "quality",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "actual_interval",
                    "type": "quantitative",
                    "label": "Effective interval",
                },
                "y": {
                    "field": "centroid_velocity_error_q95",
                    "type": "quantitative",
                    "label": "Centroid velocity error p95",
                },
                "color": {
                    "field": "geometry",
                    "type": "nominal",
                    "label": "Geometry",
                },
            },
        },
    ]
    tables = [
        {
            "id": "complete_results",
            "title": f"All {len(table_rows)} benchmark points",
            "subtitle": "Targets 1–6, exact values for the common KPI input.",
            "dataset": "results",
            "sourceId": source_id,
            "defaultSort": {"field": "target", "direction": "asc"},
            "columns": [
                {"field": "geometry", "label": "Geometry", "type": "text"},
                {"field": "target", "label": "Target", "format": "number"},
                {
                    "field": "actual_interval",
                    "label": "Actual interval",
                    "format": "number",
                },
                {"field": "fps", "label": "FPS", "format": "number"},
                {"field": "mean_iou", "label": "Mean IoU", "format": "percent"},
                {"field": "q05_iou", "label": "IoU p05", "format": "percent"},
                {"field": "min_iou", "label": "Min IoU", "format": "percent"},
                {
                    "field": "mean_recall",
                    "label": "Mean Recall",
                    "format": "percent",
                },
                {
                    "field": "min_recall",
                    "label": "Min Recall",
                    "format": "percent",
                },
                {"field": "area_mean", "label": "Area mean", "format": "number"},
                {"field": "area_q95", "label": "Area p95", "format": "number"},
                {"field": "area_max", "label": "Area max", "format": "number"},
                {
                    "field": "centroid_accel_q95",
                    "label": "Motion accel p95",
                    "format": "number",
                },
                {
                    "field": "centroid_velocity_error_q95",
                    "label": "Motion error p95",
                    "format": "number",
                },
                {"field": "keyframes", "label": "Keys", "format": "number"},
            ],
        },
        {
            "id": "class_results",
            "title": "Class-level quality",
            "subtitle": "Three classes at every target and geometry.",
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
        {
            "id": "equal_interval",
            "title": "Equal-effective-interval comparison",
            "subtitle": "Polygon values are linearly interpolated between measured points.",
            "dataset": "equal_interval",
            "sourceId": source_id,
            "defaultSort": {"field": "curve_target", "direction": "asc"},
            "columns": [
                {
                    "field": "curve_target",
                    "label": "Curve target",
                    "format": "number",
                },
                {
                    "field": "effective_interval",
                    "label": "Actual interval",
                    "format": "number",
                },
                {
                    "field": "curve_minus_polygon_iou_mean",
                    "label": "Mean IoU delta",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "curve_minus_polygon_iou_q05",
                    "label": "p05 IoU delta",
                    "format": "percent",
                    "movement": True,
                },
                {
                    "field": "curve_minus_polygon_area_ratio_q95",
                    "label": "Area p95 delta",
                    "format": "number",
                    "movement": True,
                },
            ],
        },
    ]

    title = "Polygon vs Catmull–Rom: target intervals 1–6"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Technical summary\n\n"
                "Catmull–Rom now provides the complete interval 1–6 trade-off and "
                "forms the stronger measured frontier. At equal effective interval, "
                f"its mean-IoU advantage is **{min(equal_mean_gains):+.2%} to "
                f"{max(equal_mean_gains):+.2%}**, while its p05 IoU is "
                f"**{min(equal_tail_gains):+.2%} to {max(equal_tail_gains):+.2%}** "
                "higher and its p95 area ratio is lower. It also has "
                f"**{speed_gain:+.1%}** higher median operational throughput. "
                "Both methods retain zero Recall violations and zero invalid topology."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "frontier_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## The curve frontier is consistently above the polygon frontier\n\n"
                f"Catmull–Rom has higher mean IoU at **{same_target_wins}/{len(targets)}** "
                f"same-target comparisons and **{class_wins}/{class_comparisons}** "
                "class-by-target comparisons. Same-target points are not perfectly "
                "equal in key count, so the equal-effective-interval interpolation is "
                "the stricter comparison; it still favors the curve at every shared "
                "interval from about 2.0 through 5.3."
            ),
        },
        {"id": "pareto_mean_chart", "type": "chart", "chartId": "pareto_mean"},
        {
            "id": "tail_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Lower-tail quality and expansion also favor Catmull–Rom\n\n"
                "The result is not driven only by the mean. At equal effective "
                f"interval, p05 IoU improves by {min(equal_tail_gains):+.2%} to "
                f"{max(equal_tail_gains):+.2%}, while area-ratio p95 changes by "
                f"{max(equal_area_gains):+.3f} to {min(equal_area_gains):+.3f}. "
                "Negative area deltas mean less mask inflation. Individual minima "
                "remain unstable on tiny masks, so p05 and p95 are the primary tail "
                "measures."
            ),
        },
        {"id": "pareto_tail_chart", "type": "chart", "chartId": "pareto_tail"},
        {"id": "area_chart", "type": "chart", "chartId": "area"},
        {
            "id": "speed_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Catmull–Rom is faster except at the every-frame endpoint\n\n"
                f"Median throughput is {median_curve_fps:.1f} FPS versus "
                f"{median_polygon_fps:.1f} FPS. Polygon is faster only at target 1, "
                "where Catmull–Rom writes every materialized row as a key. At targets "
                "2–6, the curve path is faster despite using exact CPU rasterization; "
                "the polygon benchmark uses its CUDA lazy-exact path."
            ),
        },
        {"id": "speed_chart", "type": "chart", "chartId": "speed"},
        {
            "id": "stability_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Smoothness and motion fidelity expose a real trade-off\n\n"
                "Polygon has lower p95 temporal acceleration, so its output is the "
                "smoother of the two. Catmull–Rom, however, has lower p95 centroid-"
                "velocity error against the source sequence at every target, and it "
                "also has higher IoU. The extra curve motion is therefore mostly "
                "better tracking rather than ungrounded jitter. These two metrics "
                "must be read together; acceleration alone would reward a frozen or "
                "lagging mask."
            ),
        },
        {"id": "stability_chart", "type": "chart", "chartId": "stability_motion"},
        {"id": "motion_fidelity_chart", "type": "chart", "chartId": "motion_fidelity"},
        {"id": "equal_interval_table", "type": "table", "tableId": "equal_interval"},
        {"id": "all_results", "type": "table", "tableId": "complete_results"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## Scope and metric definitions\n\n"
                "The source is one 23,510-frame (13:04.45), 93-track V3 KPI run. "
                "IoU, Recall and area use the same 24,503 original track-frame masks; "
                "587 deterministic gap-fill rows are excluded. Effective interval "
                "uses all 25,090 materialized rows divided by keyframe rows. Target "
                "interval is a soft request under the hard per-frame Recall 0.97 and "
                "topology constraints. Stability uses consecutive original observations: "
                "p95 log-area acceleration, centroid acceleration normalized by source "
                "radius, and a rotation/translation/phase-invariant Fourier contour "
                "descriptor. Corresponding source-velocity errors prevent static masks "
                "from being scored as stable."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Reused runs were accepted only under matching contracts\n\n"
                "Polygon targets 1–6 reuse completed runs because the source SQLite, "
                "Production polygon implementation and metric contract are unchanged. "
                "Catmull–Rom targets 1–6 were rerun after the role-state change. Every "
                "SQLite passed `PRAGMA integrity_check`, contained 25,090 masks, 93 "
                "tracks and three class policies. The paired analysis joins only exact "
                "`(track_id, frame)` matches."
            ),
        },
        {"id": "class_table", "type": "table", "tableId": "class_results"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and robustness\n\n"
                "This is one real video rather than a multi-video confidence interval, "
                "and the reference is the AI source mask rather than human ground truth. "
                "The equal-interval comparison linearly interpolates between measured "
                "polygon points and is descriptive, not inferential. Speed runs were on "
                "the same host but not interleaved; polygon uses GPU and curve uses CPU. "
                "Stability is descriptive and uses the largest contour for its shape "
                "descriptor; IoU/Recall still use the complete exact raster mask. "
                "Sparse curve targets intentionally permit low local IoU instead of "
                "silently adding keys, so visual review of the saved low-tail candidates "
                "remains required."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Use Catmull–Rom as the leading candidate across target intervals 1–6.\n"
                "2. Review the saved target-5/6 low-IoU and high-expansion candidates before promotion.\n"
                "3. Repeat the frozen comparison on multiple V3 videos and edge-contact strata.\n"
                "4. Keep polygon available as a compatibility fallback until that visual review passes."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Do human reviewers prefer the smoother curve on frames where raw-mask "
                "IoU is lower? Does the measured frontier advantage persist on multiple "
                "videos, very small masks and prolonged screen-edge contact?"
            ),
        },
    ]

    report = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                f"{len(table_rows)}-run comparison over target intervals 1–{maximum_target}."
            ),
            "generatedAt": manifest["generated_at"],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": manifest["generated_at"],
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "curve_fps": median_curve_fps,
                        "polygon_fps": median_polygon_fps,
                        "speed_gain": speed_gain,
                        "equal_iou_gain": statistics.mean(equal_mean_gains),
                        "curve_final_interval": _number(
                            final_curve, "effective_interval"
                        ),
                        "polygon_final_interval": _number(
                            final_polygon, "effective_interval"
                        ),
                        "recall_violations": sum(
                            int(row["recall_violations_below_0_97"]) for row in summary
                        ),
                    }
                ],
                "results": table_rows,
                "quality": table_rows,
                "class_results": class_rows,
                "equal_interval": equal_interval,
            },
        },
        "sources": [source],
        "package_info": {
            "originUrl": "artifact://geometry-tradeoff-kpi-20260826",
            "controls": {"edit": False, "refresh": False},
        },
    }
    path = root / "artifact.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
