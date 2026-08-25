"""Aggregate the real-video polygon versus Catmull--Rom benchmark.

Quality is compared only on source observations shared by both methods.  Keyframe
frequency and editable-point burden are measured on the materialized outputs,
including deterministic gap fill.  This distinction prevents gap-filled curve
rows from biasing the quality comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RECALL_FLOOR = 0.97
VIDEO_FRAMES = 23_510
VIDEO_FPS = 29.97002997002997


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("output/geometry_tradeoff_kpi_15min_20260825"),
    )
    parser.add_argument(
        "--source-sqlite",
        type=Path,
        default=Path(
            "output/nms_virtual_component_mask_v4_fixed_downstream_kpi_20260814/"
            "virtual_component_mask_v4/tracked.sqlite"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_rows(path: Path) -> tuple[set[tuple[str, int]], dict[str, str]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT CAST(track_id AS TEXT), frame, label FROM masks"
        ).fetchall()
    keys = {(str(track_id), int(frame)) for track_id, frame, _label in rows}
    labels: dict[str, str] = {}
    for track_id, _frame, label in rows:
        previous = labels.setdefault(str(track_id), str(label))
        if previous != str(label):
            raise RuntimeError(f"track {track_id} changes label")
    if len(keys) != len(rows):
        raise RuntimeError("source masks contain duplicate (track_id, frame) rows")
    return keys, labels


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty quality rows")
    iou = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    recall = np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
    area = np.asarray([float(row["area_ratio"]) for row in rows], dtype=np.float64)
    return {
        "quality_rows": int(len(rows)),
        "iou_mean": float(np.mean(iou)),
        "iou_median": float(np.median(iou)),
        "iou_q01": _quantile(iou, 0.01),
        "iou_q05": _quantile(iou, 0.05),
        "iou_minimum": float(np.min(iou)),
        "recall_mean": float(np.mean(recall)),
        "recall_minimum": float(np.min(recall)),
        "recall_violations_below_0_97": int(
            np.count_nonzero(recall + 1e-12 < RECALL_FLOOR)
        ),
        "area_ratio_mean": float(np.mean(area)),
        "area_ratio_q95": _quantile(area, 0.95),
        "area_ratio_q99": _quantile(area, 0.99),
        "area_ratio_maximum": float(np.max(area)),
        "area_ratio_over_1_10": int(np.count_nonzero(area > 1.10 + 1e-12)),
        "area_ratio_over_1_20": int(np.count_nonzero(area > 1.20 + 1e-12)),
    }


def _point_budget_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, int] = {}
    frame_distribution: Counter[int] = Counter()
    for row in rows:
        points = int(row["points_per_component"])
        track_id = str(row["track_id"])
        previous = by_track.setdefault(track_id, points)
        if previous != points:
            raise RuntimeError(f"point budget changes within track {track_id}")
        frame_distribution[points] += 1
    track_values = np.asarray(list(by_track.values()), dtype=np.float64)
    frame_values = np.asarray(
        [int(row["points_per_component"]) for row in rows], dtype=np.float64
    )
    return {
        "tracks": int(len(by_track)),
        "track_points_mean": float(np.mean(track_values)),
        "track_points_median": float(np.median(track_values)),
        "track_point_budget_distribution": dict(
            sorted(Counter(by_track.values()).items())
        ),
        "source_frame_points_mean": float(np.mean(frame_values)),
        "source_frame_point_budget_distribution": dict(sorted(frame_distribution.items())),
    }


def _point_count_from_polygons(serialized: str | None) -> tuple[int, int]:
    if not serialized:
        return 0, 0
    polygons = json.loads(serialized)
    components = [component for component in polygons if component]
    return sum(len(component) for component in components), len(components)


def _keyframe_stats(paths: Iterable[Path]) -> dict[str, Any]:
    point_counts: list[int] = []
    component_counts: list[int] = []
    by_label: Counter[str] = Counter()
    for path in paths:
        with sqlite3.connect(path) as connection:
            for polygons, label in connection.execute(
                "SELECT polygons, label FROM masks"
            ):
                points, components = _point_count_from_polygons(polygons)
                point_counts.append(points)
                component_counts.append(components)
                by_label[str(label)] += 1
    values = np.asarray(point_counts, dtype=np.float64)
    components = np.asarray(component_counts, dtype=np.float64)
    return {
        "keyframe_rows": int(len(point_counts)),
        "editable_points_total": int(sum(point_counts)),
        "editable_points_mean_per_keyframe": float(np.mean(values)),
        "editable_points_median_per_keyframe": float(np.median(values)),
        "editable_points_q95_per_keyframe": _quantile(values, 0.95),
        "editable_points_maximum_per_keyframe": int(np.max(values)),
        "components_mean_per_keyframe": float(np.mean(components)),
        "keyframe_rows_by_label": dict(sorted(by_label.items())),
    }


def _polygon_run(
    root: Path,
    interval: int,
    source_keys: set[tuple[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run = root / "polygon" / f"interval_{interval}"
    classwise = _json(run / "00_classwise_postprocess" / "classwise_manifest.json")
    quality_rows: list[dict[str, Any]] = []
    keyframe_paths: list[Path] = []
    topology_invalid = 0
    topology_edges_checked = 0
    exact_audit_rows = 0
    phase_totals: Counter[str] = Counter()
    interval_eval_count = 0
    interval_eval_frames = 0
    track_points: dict[str, int] = {}
    for group in classwise["groups"]:
        label = str(group["labels"][0])
        group_root = run / "00_classwise_postprocess" / "groups" / str(group["id"])
        policy_path = group_root / "pipeline/00_polygon_optimization/preparation/vertex_policy.json"
        policy = _json(policy_path)
        for track_id, item in policy["tracks"].items():
            track_points[str(track_id)] = int(item["vertices_per_component"])
        keyframe_paths.append(group_root / "pipeline/00_polygon_optimization/keyframes.sqlite")
        metrics_paths = list(group_root.rglob("runtime/exact/keyframe_exact_metrics.csv"))
        audit_paths = list(group_root.rglob("runtime/phase2_audit.json"))
        if len(metrics_paths) != 1 or len(audit_paths) != 1:
            raise RuntimeError(f"unexpected polygon artifacts in {group_root}")
        with metrics_paths[0].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["track_id"]), int(row["frame"]))
                if key not in source_keys:
                    raise RuntimeError(f"polygon quality row is not in source: {key}")
                quality_rows.append(
                    {
                        "geometry": "polygon",
                        "target_interval": interval,
                        "label": label,
                        "track_id": key[0],
                        "frame": key[1],
                        "iou": float(row["iou"]),
                        "recall": float(row["recall"]),
                        "precision": float(row["precision"]),
                        "area_ratio": float(row["pred_area"])
                        / max(float(row["gt_area"]), 1.0),
                        "points_per_component": track_points[key[0]],
                    }
                )
        audit = _json(audit_paths[0])
        runtime_summary = _json(audit_paths[0].parent / "summary.json")
        optimizer_summary = runtime_summary["optimizer_summary"]
        interval_eval_count += int(optimizer_summary["interval_eval_count"])
        interval_eval_frames += int(optimizer_summary["interval_eval_frames"])
        phase_totals.update(
            {
                key: float(value)
                for key, value in optimizer_summary["stage_seconds_total"].items()
            }
        )
        exact_audit_rows += int(audit["evaluated_rows"])
        topology = audit["topology_guard"]
        topology_invalid += int(topology["dp_invalid_edges"])
        topology_invalid += int(topology["pair_vote_paths_rejected"])
        topology_edges_checked += int(topology["dp_selected_edges_checked"])
        topology_edges_checked += int(topology["pair_vote_paths_checked"])
        if int(audit["exact_recall_violations"]) != 0:
            raise RuntimeError(f"polygon Recall violation in {audit_paths[0]}")
    if len(quality_rows) != len(source_keys):
        raise RuntimeError(
            f"polygon interval {interval}: {len(quality_rows)} rows, "
            f"expected {len(source_keys)}"
        )
    if len({(row["track_id"], row["frame"]) for row in quality_rows}) != len(
        quality_rows
    ):
        raise RuntimeError(f"polygon interval {interval} has duplicate quality rows")
    keys = _keyframe_stats(keyframe_paths)
    output_rows = int(classwise["merge"]["output_masks"])
    summary = {
        "geometry": "polygon",
        "target_interval": interval,
        "source_video_frames": VIDEO_FRAMES,
        "materialized_rows": output_rows,
        **keys,
        "effective_interval": float(output_rows / keys["keyframe_rows"]),
        "wall_seconds": float(classwise["elapsed_seconds"]),
        "source_video_fps": float(VIDEO_FRAMES / classwise["elapsed_seconds"]),
        "mask_rows_per_second": float(output_rows / classwise["elapsed_seconds"]),
        **_quality_summary(quality_rows),
        **_point_budget_summary(quality_rows),
        "topology_invalid_or_rejected": int(topology_invalid),
        "topology_paths_checked": int(topology_edges_checked),
        "exact_audit_rows": int(exact_audit_rows),
        "quality_gapfill_rows_excluded": 0,
        "interval_eval_count": int(interval_eval_count),
        "interval_eval_frames": int(interval_eval_frames),
        "dp_seconds_sum": float(phase_totals["solve_dp_seconds"]),
        "pair_vote_seconds_sum": float(phase_totals["pair_vote_refine_seconds"]),
        "point_refine_seconds_sum": 0.0,
        "candidate_build_seconds_sum": float(phase_totals["build_candidates_seconds"]),
        "final_audit_seconds_sum": float(phase_totals["final_eval_seconds"]),
        "quality_rescue_inserted_keys": 0,
        "cuda_used": True,
        "execution_profile": "2 class workers x 3 optimizer workers; CUDA lazy exact",
        "sqlite": str(
            (run / "00_classwise_postprocess/predictions.sqlite").resolve()
        ),
    }
    return summary, quality_rows, _class_summaries(summary, quality_rows)


def _curve_run(
    root: Path,
    interval: int,
    source_keys: set[tuple[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run = root / "catmull_rom" / f"interval_{interval}"
    classwise = _json(run / "00_classwise_postprocess" / "classwise_manifest.json")
    quality_rows: list[dict[str, Any]] = []
    all_metric_rows = 0
    gapfill_metric_rows = 0
    topology_invalid = 0
    keyframe_paths: list[Path] = []
    engine_keyframes = 0
    phase_totals: Counter[str] = Counter()
    quality_rescue_inserted = 0
    for group in classwise["groups"]:
        group_root = run / "00_classwise_postprocess" / "groups" / str(group["id"])
        engine_paths = list(group_root.rglob("curve_engine_manifest.json"))
        if len(engine_paths) != 1:
            raise RuntimeError(f"unexpected curve engine artifacts in {group_root}")
        engine = _json(engine_paths[0])
        metrics_path = Path(str(engine["component_metrics_csv"]))
        if not metrics_path.exists():
            metrics_path = engine_paths[0].parent / metrics_path.name
        keyframe_path = Path(str(engine["keyframes_sqlite"]))
        if not keyframe_path.exists():
            keyframe_path = engine_paths[0].parent / keyframe_path.name
        keyframe_paths.append(keyframe_path)
        engine_keyframes += int(engine["keyframe_rows"])
        topology_invalid += int(engine["audit"]["topology_invalid_frames"])
        phase_totals.update(
            {
                "dp_seconds": float(engine["audit"]["dp_seconds"]),
                "pair_vote_seconds": float(engine["audit"]["pair_vote_seconds"]),
                "point_refine_seconds": float(engine["audit"]["point_refine_seconds"]),
                "final_audit_seconds": float(engine["audit"]["final_audit_seconds"]),
            }
        )
        audit_path = Path(str(engine["stream_audit_jsonl"]))
        if not audit_path.exists():
            audit_path = engine_paths[0].parent / audit_path.name
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                stream = json.loads(line)
                quality_rescue_inserted += sum(
                    int(item.get("quality_rescue_inserted", 0))
                    for item in stream.get("optimization", [])
                )
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                all_metric_rows += 1
                key = (str(row["track_id"]), int(row["frame"]))
                if key not in source_keys:
                    gapfill_metric_rows += 1
                    continue
                quality_rows.append(
                    {
                        "geometry": "catmull_rom",
                        "target_interval": interval,
                        "label": str(row["label"]),
                        "track_id": key[0],
                        "frame": key[1],
                        "iou": float(row["iou"]),
                        "recall": float(row["recall"]),
                        "precision": float(row["precision"]),
                        "area_ratio": float(row["area_ratio"]),
                        "points_per_component": int(row["points_per_component"]),
                    }
                )
    if len(quality_rows) != len(source_keys):
        raise RuntimeError(
            f"curve interval {interval}: {len(quality_rows)} rows, "
            f"expected {len(source_keys)}"
        )
    if len({(row["track_id"], row["frame"]) for row in quality_rows}) != len(
        quality_rows
    ):
        raise RuntimeError(f"curve interval {interval} has duplicate quality rows")
    keys = _keyframe_stats(keyframe_paths)
    if keys["keyframe_rows"] != engine_keyframes:
        raise RuntimeError(
            f"curve interval {interval}: keyframe SQLite and engine disagree"
        )
    output_rows = int(classwise["merge"]["output_masks"])
    if all_metric_rows != output_rows or gapfill_metric_rows != output_rows - len(source_keys):
        raise RuntimeError(f"curve interval {interval}: metric population mismatch")
    summary = {
        "geometry": "catmull_rom",
        "target_interval": interval,
        "source_video_frames": VIDEO_FRAMES,
        "materialized_rows": output_rows,
        **keys,
        "effective_interval": float(output_rows / keys["keyframe_rows"]),
        "wall_seconds": float(classwise["elapsed_seconds"]),
        "source_video_fps": float(VIDEO_FRAMES / classwise["elapsed_seconds"]),
        "mask_rows_per_second": float(output_rows / classwise["elapsed_seconds"]),
        **_quality_summary(quality_rows),
        **_point_budget_summary(quality_rows),
        "topology_invalid_or_rejected": int(topology_invalid),
        "topology_paths_checked": None,
        "exact_audit_rows": int(all_metric_rows),
        "quality_gapfill_rows_excluded": int(gapfill_metric_rows),
        "interval_eval_count": None,
        "interval_eval_frames": None,
        "dp_seconds_sum": float(phase_totals["dp_seconds"]),
        "pair_vote_seconds_sum": float(phase_totals["pair_vote_seconds"]),
        "point_refine_seconds_sum": float(phase_totals["point_refine_seconds"]),
        "candidate_build_seconds_sum": None,
        "final_audit_seconds_sum": float(phase_totals["final_audit_seconds"]),
        "quality_rescue_inserted_keys": int(quality_rescue_inserted),
        "cuda_used": False,
        "execution_profile": "6 balanced CPU shards x 4 native threads; CUDA disabled",
        "sqlite": str(
            (run / "00_classwise_postprocess/predictions.sqlite").resolve()
        ),
    }
    return summary, quality_rows, _class_summaries(summary, quality_rows)


def _class_summaries(
    overall: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["label"])].append(row)
    result: list[dict[str, Any]] = []
    keyframes_by_label = overall["keyframe_rows_by_label"]
    for label, selected in sorted(buckets.items()):
        keys = int(keyframes_by_label.get(label, 0))
        result.append(
            {
                "geometry": overall["geometry"],
                "target_interval": overall["target_interval"],
                "label": label,
                "materialized_rows": len(selected),
                "keyframe_rows": keys,
                "effective_interval_on_source_rows": float(len(selected) / max(keys, 1)),
                **_quality_summary(selected),
            }
        )
    return result


def _rounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, float):
                converted[key] = round(value, 9) if math.isfinite(value) else value
            elif isinstance(value, (dict, list)):
                converted[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                converted[key] = value
        result.append(converted)
    return result


def _timecode(frame: int) -> str:
    seconds = frame / VIDEO_FPS
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    display_frame = int(round((seconds - math.floor(seconds)) * VIDEO_FPS))
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}:{display_frame:02d}"


def _paired_outputs(
    quality_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    for row in quality_rows:
        key = (
            int(row["target_interval"]),
            str(row["geometry"]),
            str(row["track_id"]),
            int(row["frame"]),
        )
        indexed[key] = row
    paired: list[dict[str, Any]] = []
    for interval in range(1, 8):
        for _target, _geometry, track_id, frame in sorted(
            key for key in indexed if key[0] == interval and key[1] == "polygon"
        ):
            polygon = indexed[(interval, "polygon", track_id, frame)]
            curve = indexed[(interval, "catmull_rom", track_id, frame)]
            paired.append(
                {
                    "target_interval": interval,
                    "frame": frame,
                    "timecode": _timecode(frame),
                    "track_id": track_id,
                    "label": polygon["label"],
                    "polygon_iou": float(polygon["iou"]),
                    "curve_iou": float(curve["iou"]),
                    "curve_minus_polygon_iou": float(curve["iou"])
                    - float(polygon["iou"]),
                    "polygon_recall": float(polygon["recall"]),
                    "curve_recall": float(curve["recall"]),
                    "polygon_area_ratio": float(polygon["area_ratio"]),
                    "curve_area_ratio": float(curve["area_ratio"]),
                    "curve_minus_polygon_area_ratio": float(curve["area_ratio"])
                    - float(polygon["area_ratio"]),
                    "polygon_points": int(polygon["points_per_component"]),
                    "curve_points": int(curve["points_per_component"]),
                }
            )
    paired_summary: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    for interval in range(1, 8):
        selected = [row for row in paired if row["target_interval"] == interval]
        iou_delta = np.asarray(
            [float(row["curve_minus_polygon_iou"]) for row in selected],
            dtype=np.float64,
        )
        area_delta = np.asarray(
            [float(row["curve_minus_polygon_area_ratio"]) for row in selected],
            dtype=np.float64,
        )
        paired_summary.append(
            {
                "target_interval": interval,
                "rows": len(selected),
                "curve_minus_polygon_iou_mean": float(np.mean(iou_delta)),
                "curve_minus_polygon_iou_q05": _quantile(iou_delta, 0.05),
                "curve_minus_polygon_iou_q95": _quantile(iou_delta, 0.95),
                "curve_iou_better_rows": int(np.count_nonzero(iou_delta > 1e-12)),
                "polygon_iou_better_rows": int(np.count_nonzero(iou_delta < -1e-12)),
                "curve_iou_better_share": float(np.mean(iou_delta > 1e-12)),
                "curve_minus_polygon_area_ratio_mean": float(np.mean(area_delta)),
            }
        )
        reasons = (
            ("polygon_low_iou", sorted(selected, key=lambda row: row["polygon_iou"])[:20]),
            ("curve_low_iou", sorted(selected, key=lambda row: row["curve_iou"])[:20]),
            (
                "polygon_high_expansion",
                sorted(selected, key=lambda row: row["polygon_area_ratio"], reverse=True)[:20],
            ),
            (
                "curve_high_expansion",
                sorted(selected, key=lambda row: row["curve_area_ratio"], reverse=True)[:20],
            ),
            (
                "curve_largest_gain",
                sorted(
                    selected,
                    key=lambda row: row["curve_minus_polygon_iou"],
                    reverse=True,
                )[:20],
            ),
            (
                "polygon_largest_gain",
                sorted(selected, key=lambda row: row["curve_minus_polygon_iou"])[:20],
            ),
        )
        for reason, rows in reasons:
            for rank, row in enumerate(rows, 1):
                review_candidates.append({"reason": reason, "rank": rank, **row})
    return paired_summary, review_candidates


def _sqlite_integrity(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for summary in summaries:
        path = Path(str(summary["sqlite"]))
        with sqlite3.connect(path) as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            masks = int(connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
            tracks = int(connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
            policies = int(
                connection.execute(
                    "SELECT COUNT(*) FROM class_postprocess_policies"
                ).fetchone()[0]
            )
        checks.append(
            {
                "geometry": summary["geometry"],
                "target_interval": summary["target_interval"],
                "integrity": integrity,
                "mask_rows": masks,
                "tracks": tracks,
                "class_policies": policies,
                "bytes": path.stat().st_size,
                "sqlite": str(path),
            }
        )
    return checks


def main() -> None:
    args = _parser().parse_args()
    root = args.benchmark_root.resolve()
    source = args.source_sqlite.resolve()
    output = (args.output_dir or (root / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_keys, source_labels = _source_rows(source)
    summaries: list[dict[str, Any]] = []
    by_class: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []
    for interval in range(1, 8):
        for loader in (_polygon_run, _curve_run):
            summary, quality, class_rows = loader(root, interval, source_keys)
            summaries.append(summary)
            all_quality.extend(quality)
            by_class.extend(class_rows)
    summaries.sort(key=lambda row: (int(row["target_interval"]), str(row["geometry"])))
    by_class.sort(
        key=lambda row: (
            int(row["target_interval"]),
            str(row["geometry"]),
            str(row["label"]),
        )
    )
    _write_csv(output / "summary.csv", _rounded_rows(summaries))
    _write_csv(output / "summary_by_class.csv", _rounded_rows(by_class))
    paired_summary, review_candidates = _paired_outputs(all_quality)
    _write_csv(output / "paired_delta_summary.csv", _rounded_rows(paired_summary))
    _write_csv(output / "review_candidates.csv", _rounded_rows(review_candidates))
    sqlite_checks = _sqlite_integrity(summaries)
    _write_csv(output / "sqlite_integrity.csv", _rounded_rows(sqlite_checks))

    # Long-form datasets keep report and notebook chart definitions simple.
    tradeoff_rows: list[dict[str, Any]] = []
    for row in summaries:
        geometry = "Catmull–Rom" if row["geometry"] == "catmull_rom" else "Polygon"
        for metric, value in (
            ("Actual effective interval", row["effective_interval"]),
            ("Mean IoU", row["iou_mean"]),
            ("5th percentile IoU", row["iou_q05"]),
            ("1st percentile IoU", row["iou_q01"]),
            ("95th percentile area ratio", row["area_ratio_q95"]),
            ("Mean editable points/key", row["editable_points_mean_per_keyframe"]),
            ("Throughput (video FPS)", row["source_video_fps"]),
        ):
            tradeoff_rows.append(
                {
                    "target_interval": row["target_interval"],
                    "geometry": geometry,
                    "metric": metric,
                    "value": float(value),
                }
            )
    _write_csv(output / "tradeoff_long.csv", _rounded_rows(tradeoff_rows))

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "benchmark_root": str(root),
        "source_sqlite": str(source),
        "source_video": str((Path("data") / "12月KPI動画.mp4").resolve()),
        "source_video_frames": VIDEO_FRAMES,
        "source_video_fps": VIDEO_FPS,
        "source_video_seconds": VIDEO_FRAMES / VIDEO_FPS,
        "source_observations": len(source_keys),
        "source_tracks": len(source_labels),
        "materialized_observations": 25_090,
        "gapfill_observations": 587,
        "quality_population_contract": (
            "Both methods are compared on the same 24,503 original (track, frame) "
            "observations. The 587 gap-filled rows are excluded from quality metrics."
        ),
        "frequency_population_contract": (
            "Effective interval and editable points are measured on all 25,090 "
            "materialized output rows, including gap fill."
        ),
        "summaries": summaries,
        "by_class": by_class,
        "paired_summary": paired_summary,
        "sqlite_integrity": sqlite_checks,
        "validation": {
            "expected_run_count": 14,
            "actual_run_count": len(summaries),
            "all_quality_rows_equal_source": all(
                int(row["quality_rows"]) == len(source_keys) for row in summaries
            ),
            "all_recall_constraints_satisfied": all(
                int(row["recall_violations_below_0_97"]) == 0 for row in summaries
            ),
            "all_topology_valid": all(
                int(row["topology_invalid_or_rejected"]) == 0 for row in summaries
            ),
            "all_outputs_exist": all(Path(str(row["sqlite"])).is_file() for row in summaries),
            "all_sqlite_integrity_checks_passed": all(
                row["integrity"] == "ok"
                and int(row["mask_rows"]) == 25_090
                and int(row["tracks"]) == 93
                and int(row["class_policies"]) == 3
                for row in sqlite_checks
            ),
        },
    }
    _atomic_json(output / "summary.json", payload)
    print(json.dumps(payload["validation"], ensure_ascii=False))
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
