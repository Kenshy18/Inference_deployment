#!/usr/bin/env python3
"""Build reviewed, cohort-separated analysis rows from raw-corpus benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _results(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "datasets").glob("*/result.json"))
    }


def _bbox_profile(source: Path, width: int, height: int) -> tuple[float, float]:
    denominator = max(float(width * height), 1.0)
    with sqlite3.connect(
        f"file:{source.resolve()}?mode=ro&immutable=1", uri=True
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
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


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
        "sampleVideos": len(rows),
    }


def _summary(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    fps = np.asarray([float(row["pipelineFps"]) for row in rows], dtype=np.float64)
    geometry_fps = np.asarray(
        [float(row["geometryFps"]) for row in rows], dtype=np.float64
    )
    total_frames = int(sum(int(row["frames"]) for row in rows))
    wall_seconds = float(sum(float(row["pipelineSeconds"]) for row in rows))
    geometry_seconds = float(sum(float(row["geometrySeconds"]) for row in rows))
    return {
        "cohort": label,
        "videos": len(rows),
        "frames": total_frames,
        "hours": float(sum(float(row["durationSeconds"]) for row in rows) / 3600.0),
        "weightedPipelineFps": total_frames / wall_seconds,
        "weightedGeometryFps": total_frames / geometry_seconds,
        "unweightedMeanPipelineFps": float(np.mean(fps)),
        "medianPipelineFps": float(np.median(fps)),
        "p10PipelineFps": float(np.quantile(fps, 0.10)),
        "p90PipelineFps": float(np.quantile(fps, 0.90)),
        "minimumPipelineFps": float(np.min(fps)),
        "maximumPipelineFps": float(np.max(fps)),
        "coefficientOfVariation": float(np.std(fps, ddof=1) / np.mean(fps)),
        "minimumGeometryFps": float(np.min(geometry_fps)),
        "maximumGeometryFps": float(np.max(geometry_fps)),
    }


def _row(result: dict[str, object]) -> dict[str, object]:
    profile = result["profile"]
    quality = result["quality"]
    frames = int(profile["frames"])
    vertex_rows = quality.get("vertex_rows", {})
    total_vertex_rows = max(sum(int(value) for value in vertex_rows.values()), 1)
    bbox_mean, bbox_p95 = _bbox_profile(
        Path(profile["source"]), int(profile["width"]), int(profile["height"])
    )
    return {
        "dataset": result["dataset"],
        "cohort": result["cohort"],
        "stratum": result["stratum"],
        "model": profile["model"],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "fpsMetadata": float(profile["fps"]),
        "frames": frames,
        "durationSeconds": float(profile["duration_seconds"]),
        "rawDetections": int(profile["segmentation_detections"]),
        "detectionsPerFrame": float(profile["detections_per_frame"]),
        "trackedRows": int(result["tracking"]["rows_after_prune"]),
        "trackedRowsPerFrame": float(result["tracking"]["rows_after_prune"]) / frames,
        "tracks": int(result["tracking"]["tracks_after_prune"]),
        "tracksPer10kFrames": 10000.0
        * float(result["tracking"]["tracks_after_prune"])
        / frames,
        "runs": int(quality.get("run_count", 0)),
        "runsPer10kFrames": 10000.0 * float(quality.get("run_count", 0)) / frames,
        "meanPointsPerDetection": float(profile["mean_points_per_detection"]),
        "meanBboxAreaPct": 100.0 * bbox_mean,
        "p95BboxAreaPct": 100.0 * bbox_p95,
        "highVertexSharePct": 100.0
        * sum(int(vertex_rows.get(str(value), 0)) for value in (16, 18, 20))
        / total_vertex_rows,
        "pipelineSeconds": float(result["pipeline_wall_seconds"]),
        "geometrySeconds": float(result["geometry_seconds"]),
        "pipelineFps": float(result["pipeline_timeline_fps"]),
        "geometryFps": float(result["geometry_timeline_fps"]),
        "costMsPerFrame": 1000.0 / float(result["pipeline_timeline_fps"]),
        "peakProcessTreeRssGiB": float(result["peak_process_tree_rss_mib"]) / 1024.0,
        "minimumAvailableMemoryGiB": float(
            result["minimum_system_available_memory_mib"]
        )
        / 1024.0,
        "actualInterval": float(quality.get("actual_interval", 0.0)),
        "meanIoU": float(quality.get("mean_iou", 0.0)),
        "p05IoU": float(quality.get("p05_iou", 0.0)),
        "p01IoU": float(quality.get("p01_iou", 0.0)),
        "minimumIoU": float(quality.get("minimum_iou", 0.0)),
        "minimumRecall": float(quality.get("minimum_recall", 0.0)),
        "recallViolations": int(quality.get("recall_violations", 0)),
        "exactRecallViolations": int(quality.get("exact_recall_violations", 0)),
        "areaRatioP95": float(quality.get("area_ratio_p95", 0.0)),
        "areaRatioP99": float(quality.get("area_ratio_p99", 0.0)),
        "areaRatioMax": float(quality.get("area_ratio_max", 0.0)),
        "source": str(profile["source"]),
        "result": str(
            Path(result["log"]).parent / "result.json"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    primary_results = _results(args.benchmark_root)
    repeats = _results(args.repeat_root)
    rows = [_row(result) for result in primary_results.values()]
    rows.sort(key=lambda row: (str(row["cohort"]), float(row["pipelineFps"])))
    primary = [row for row in rows if row["cohort"] == "primary_v3"]
    supplemental = [
        row for row in rows if row["cohort"] == "supplemental_model_media"
    ]
    operational = [row for row in primary if row["dataset"] != "v3_sdam_sparse"]

    repeat_rows = []
    for name, repeat in sorted(repeats.items()):
        baseline = primary_results[name]
        quality_equal = baseline["quality"] == repeat["quality"]
        baseline_fps = float(baseline["pipeline_timeline_fps"])
        repeat_fps = float(repeat["pipeline_timeline_fps"])
        repeat_rows.append(
            {
                "dataset": name,
                "baselinePipelineFps": baseline_fps,
                "repeatPipelineFps": repeat_fps,
                "pipelineFpsDeltaPct": 100.0 * (repeat_fps / baseline_fps - 1.0),
                "baselineGeometryFps": float(baseline["geometry_timeline_fps"]),
                "repeatGeometryFps": float(repeat["geometry_timeline_fps"]),
                "qualityExactlyEqual": quality_equal,
            }
        )

    factors = [
        "detectionsPerFrame",
        "trackedRowsPerFrame",
        "tracksPer10kFrames",
        "runsPer10kFrames",
        "meanPointsPerDetection",
        "meanBboxAreaPct",
        "p95BboxAreaPct",
        "highVertexSharePct",
        "frames",
    ]
    correlations = [_correlation(primary, field) for field in factors]
    stage_rows = []
    total_frames = sum(int(row["frames"]) for row in primary)
    total_wall = sum(float(row["pipelineSeconds"]) for row in primary)
    stage_ids = sorted(
        {
            stage
            for name, result in primary_results.items()
            if result["cohort"] == "primary_v3"
            for stage in result["stages"]
        }
    )
    for stage in stage_ids:
        seconds = sum(
            float(result["stages"].get(stage, {}).get("elapsed_seconds", 0.0))
            for result in primary_results.values()
            if result["cohort"] == "primary_v3"
        )
        stage_rows.append(
            {
                "stage": stage,
                "seconds": seconds,
                "shareOfPipelineWallPct": 100.0 * seconds / total_wall,
                "equivalentFps": total_frames / seconds if seconds else None,
            }
        )
    stage_rows.sort(key=lambda row: float(row["seconds"]), reverse=True)

    evaluated = sum(
        int(primary_results[row["dataset"]]["quality"]["evaluated_rows"])
        for row in primary
    )
    weighted_iou = sum(
        int(primary_results[row["dataset"]]["quality"]["evaluated_rows"])
        * float(row["meanIoU"])
        for row in primary
    ) / max(evaluated, 1)
    max_repeat_delta = max(
        abs(float(row["pipelineFpsDeltaPct"])) for row in repeat_rows
    )
    analysis = {
        "schemaVersion": 1,
        "scope": {
            "targetInterval": 3,
            "optimizerWorkersPerLabel": 2,
            "minimumAvailableMemoryGuardMiB": 8192,
            "primaryModel": "dinov3_codino",
            "primaryIndependentVideos": len(primary),
            "primaryFrames": sum(int(row["frames"]) for row in primary),
            "primaryHours": sum(float(row["durationSeconds"]) for row in primary)
            / 3600.0,
        },
        "cohortSummaries": [
            _summary(primary, "primary_v3"),
            _summary(operational, "primary_v3_excluding_intentional_sparse_stress"),
            _summary(supplemental, "supplemental_model_media"),
        ],
        "qualitySummary": {
            "evaluatedRows": evaluated,
            "weightedMeanIoU": weighted_iou,
            "minimumRecall": min(float(row["minimumRecall"]) for row in primary),
            "totalRecallViolations": sum(
                int(row["recallViolations"]) for row in primary
            ),
            "totalExactRecallViolations": sum(
                int(row["exactRecallViolations"]) for row in primary
            ),
            "minimumIoUAcrossVideos": min(float(row["minimumIoU"]) for row in primary),
            "maximumAreaRatioAcrossVideos": max(
                float(row["areaRatioMax"]) for row in primary
            ),
        },
        "validation": {
            "allPrimaryRunsComplete": all(
                primary_results[row["dataset"]]["status"] == "complete"
                for row in primary
            ),
            "allPrimaryRecallConstraintsSatisfied": all(
                int(row["recallViolations"]) == 0
                and int(row["exactRecallViolations"]) == 0
                for row in primary
            ),
            "repeatQualityExactlyEqual": all(
                bool(row["qualityExactlyEqual"]) for row in repeat_rows
            ),
            "maximumAbsoluteRepeatPipelineFpsDeltaPct": max_repeat_delta,
            "memoryGuardTriggers": sum(
                primary_results[row["dataset"]]["status"]
                == "memory_guard_triggered"
                for row in primary
            ),
        },
        "rows": rows,
        "repeatRows": repeat_rows,
        "correlations": correlations,
        "stageRows": stage_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (args.output / "dataset_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(args.output / "analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
