#!/usr/bin/env python3
"""Compare Production polygon and Catmull--Rom speed across one raw corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polygon-root", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(root: Path) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for path in sorted((root / "datasets").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["result_json"] = str(path.resolve())
        values[str(result["dataset"])] = result
    return values


def _bbox_profile(path: Path, width: int, height: int) -> tuple[float, float]:
    denominator = max(float(width * height), 1.0)
    with sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True
    ) as database:
        values = np.fromiter(
            (
                max(0.0, float(x2 - x1) * float(y2 - y1)) / denominator
                for x1, y1, x2, y2 in database.execute(
                    "SELECT x1,y1,x2,y2 FROM detections"
                )
            ),
            dtype=np.float64,
        )
    if not values.size:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.quantile(values, 0.95))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def _correlation(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    x = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["costMsPerFrame"]) for row in rows], dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        pearson = spearman = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return {
        "factor": field,
        "pearsonVsCostMsPerFrame": pearson,
        "spearmanVsCostMsPerFrame": spearman,
        "videos": len(rows),
    }


def _row(result: dict[str, object], geometry: str) -> dict[str, object]:
    profile = dict(result["profile"])
    quality = dict(result["quality"])
    frames = int(profile["frames"])
    tracked_rows = int(result["tracking"]["rows_after_prune"])
    tracks = int(result["tracking"]["tracks_after_prune"])
    runs = int(quality.get("run_count", 0))
    point_rows = dict(quality.get("vertex_rows", {}))
    point_total = max(sum(int(value) for value in point_rows.values()), 1)
    bbox_mean, bbox_p95 = _bbox_profile(
        Path(str(profile["source"])), int(profile["width"]), int(profile["height"])
    )
    pipeline_fps = float(result["pipeline_timeline_fps"])
    geometry_fps = float(result["geometry_timeline_fps"])
    return {
        "dataset": str(result["dataset"]),
        "cohort": str(result["cohort"]),
        "stratum": str(result.get("stratum", "")),
        "geometry": geometry,
        "frames": frames,
        "durationSeconds": float(profile["duration_seconds"]),
        "rawDetections": int(profile["segmentation_detections"]),
        "detectionsPerFrame": float(profile["detections_per_frame"]),
        "trackedRows": tracked_rows,
        "trackedRowsPerFrame": tracked_rows / max(frames, 1),
        "tracks": tracks,
        "tracksPer10kFrames": 10000.0 * tracks / max(frames, 1),
        "runs": runs,
        "runsPer10kFrames": 10000.0 * runs / max(frames, 1),
        "meanPointsPerDetection": float(profile["mean_points_per_detection"]),
        "meanBboxAreaPct": 100.0 * bbox_mean,
        "p95BboxAreaPct": 100.0 * bbox_p95,
        "highPointSharePct": 100.0
        * sum(int(point_rows.get(str(value), 0)) for value in (16, 18, 20))
        / point_total,
        "pipelineSeconds": float(result["pipeline_wall_seconds"]),
        "geometrySeconds": float(result["geometry_seconds"]),
        "pipelineFps": pipeline_fps,
        "geometryFps": geometry_fps,
        "costMsPerFrame": 1000.0 / pipeline_fps,
        "peakRssGiB": float(result["peak_process_tree_rss_mib"]) / 1024.0,
        "minimumAvailableMemoryGiB": float(
            result["minimum_system_available_memory_mib"]
        )
        / 1024.0,
        "actualInterval": float(quality.get("actual_interval", 0.0)),
        "meanIoU": float(quality.get("mean_iou", 0.0)),
        "p05IoU": float(quality.get("p05_iou", 0.0)),
        "minimumRecall": float(quality.get("minimum_recall", 0.0)),
        "recallViolations": int(quality.get("recall_violations", 0)),
        "resultJson": str(result["result_json"]),
    }


def _summary(
    rows: list[dict[str, object]],
    cohort: str,
    geometry: str,
    *,
    population: str = "all",
) -> dict[str, object]:
    values = [
        row
        for row in rows
        if row["cohort"] == cohort and row["geometry"] == geometry
    ]
    if population == "operational":
        values = [
            row
            for row in values
            if "very sparse" not in str(row["stratum"])
            and "startup-only" not in str(row["stratum"])
        ]
    fps = np.asarray([float(row["pipelineFps"]) for row in values])
    geometry_fps = np.asarray([float(row["geometryFps"]) for row in values])
    frames = sum(int(row["frames"]) for row in values)
    seconds = sum(float(row["pipelineSeconds"]) for row in values)
    geometry_seconds = sum(float(row["geometrySeconds"]) for row in values)
    return {
        "cohort": cohort,
        "geometry": geometry,
        "population": population,
        "videos": len(values),
        "frames": frames,
        "hours": sum(float(row["durationSeconds"]) for row in values) / 3600.0,
        "weightedPipelineFps": frames / seconds,
        "weightedGeometryFps": frames / geometry_seconds,
        "meanPipelineFps": float(np.mean(fps)),
        "medianPipelineFps": float(np.median(fps)),
        "p10PipelineFps": float(np.quantile(fps, 0.10)),
        "p90PipelineFps": float(np.quantile(fps, 0.90)),
        "minimumPipelineFps": float(np.min(fps)),
        "maximumPipelineFps": float(np.max(fps)),
        "maxToMinRatio": float(np.max(fps) / np.min(fps)),
        "coefficientOfVariation": float(np.std(fps, ddof=1) / np.mean(fps)),
        "minimumGeometryFps": float(np.min(geometry_fps)),
        "maximumGeometryFps": float(np.max(geometry_fps)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    polygon = _load(args.polygon_root)
    curve = _load(args.curve_root)
    names = sorted(set(polygon) | set(curve))
    missing = {
        "polygon": sorted(set(curve) - set(polygon)),
        "catmull_rom": sorted(set(polygon) - set(curve)),
    }
    incomplete = {
        geometry: sorted(
            name
            for name, result in values.items()
            if result.get("status") != "complete"
        )
        for geometry, values in (("polygon", polygon), ("catmull_rom", curve))
    }
    completion = []
    for geometry, values in (("polygon", polygon), ("catmull_rom", curve)):
        for cohort in sorted({str(result["cohort"]) for result in values.values()}):
            cohort_values = [
                result for result in values.values() if str(result["cohort"]) == cohort
            ]
            complete = sum(result.get("status") == "complete" for result in cohort_values)
            completion.append(
                {
                    "geometry": geometry,
                    "cohort": cohort,
                    "datasets": len(cohort_values),
                    "complete": complete,
                    "failed": len(cohort_values) - complete,
                    "completionRate": complete / max(len(cohort_values), 1),
                    "failedWallSeconds": sum(
                        float(result.get("pipeline_wall_seconds", 0.0))
                        for result in cohort_values
                        if result.get("status") != "complete"
                    ),
                }
            )
    usable = [
        name
        for name in names
        if name in polygon
        and name in curve
        and polygon[name].get("status") == "complete"
        and curve[name].get("status") == "complete"
    ]
    rows = [
        _row(result, geometry)
        for geometry, values in (("polygon", polygon), ("catmull_rom", curve))
        for result in values.values()
        if result.get("status") == "complete"
    ]
    rows.sort(key=lambda row: (str(row["cohort"]), str(row["dataset"]), str(row["geometry"])))
    _write_csv(args.output / "per_video_geometry_speed.csv", rows)

    cohorts = sorted({str(row["cohort"]) for row in rows})
    summaries = [
        _summary(rows, cohort, geometry, population=population)
        for cohort in cohorts
        for geometry in ("polygon", "catmull_rom")
        for population in ("all", "operational")
    ]
    _write_csv(args.output / "speed_variance_summary.csv", summaries)

    factors = (
        "detectionsPerFrame",
        "trackedRowsPerFrame",
        "tracksPer10kFrames",
        "runsPer10kFrames",
        "meanPointsPerDetection",
        "meanBboxAreaPct",
        "p95BboxAreaPct",
        "highPointSharePct",
        "frames",
    )
    primary_correlations = {
        geometry: [
            _correlation(
                [
                    row
                    for row in rows
                    if row["cohort"] == "primary_v3"
                    and row["geometry"] == geometry
                ],
                factor,
            )
            for factor in factors
        ]
        for geometry in ("polygon", "catmull_rom")
    }
    primary_operational_correlations = {
        geometry: [
            _correlation(
                [
                    row
                    for row in rows
                    if row["cohort"] == "primary_v3"
                    and row["geometry"] == geometry
                    and "very sparse" not in str(row["stratum"])
                ],
                factor,
            )
            for factor in factors
        ]
        for geometry in ("polygon", "catmull_rom")
    }
    paired = []
    for name in usable:
        p = next(row for row in rows if row["dataset"] == name and row["geometry"] == "polygon")
        c = next(row for row in rows if row["dataset"] == name and row["geometry"] == "catmull_rom")
        paired.append(
            {
                "dataset": name,
                "cohort": p["cohort"],
                "polygonPipelineFps": p["pipelineFps"],
                "curvePipelineFps": c["pipelineFps"],
                "curveVsPolygonFpsRatio": float(c["pipelineFps"]) / float(p["pipelineFps"]),
                "polygonGeometryFps": p["geometryFps"],
                "curveGeometryFps": c["geometryFps"],
            }
        )
    _write_csv(args.output / "paired_geometry_speed.csv", paired)
    _write_csv(args.output / "completion_summary.csv", completion)
    payload = {
        "schemaVersion": 1,
        "targetInterval": 3,
        "missing": missing,
        "incomplete": incomplete,
        "completion": completion,
        "pairedDatasets": len(usable),
        "summaries": summaries,
        "primaryV3Correlations": primary_correlations,
        "primaryV3OperationalCorrelations": primary_operational_correlations,
        "paired": paired,
        "rows": rows,
        "sources": {
            "polygon": str(args.polygon_root.resolve()),
            "catmull_rom": str(args.curve_root.resolve()),
        },
    }
    output = args.output / "geometry_speed_variance_analysis.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
