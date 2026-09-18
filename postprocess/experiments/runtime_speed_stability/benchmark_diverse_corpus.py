#!/usr/bin/env python3
"""Benchmark the current Production polygon scheduler on diverse real tracks.

The parent process launches one fresh worker process per dataset so reported
peak RSS and wall time are not contaminated by earlier cases.  Both timeline
FPS and observation FPS are reported because sparse videos can look fast when
only video-frame count is used, while a single long track can look slow even
though it contains relatively few observations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from production.config import PRODUCTION, RUNTIME_POLYGON_PROFILE_ID
from production.polygon.runtime_bridge import optimize, prepare_inputs


DATASETS: dict[str, dict[str, object]] = {
    "kpi_full_multiclass": {
        "source": ROOT
        / "output/production_candidate_20260814_kpi_parity/04_tracking/tracked.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 23510,
        "stratum": "full three-class corpus; many tracks",
    },
    "simple_3min_long_tracks": {
        "source": ROOT
        / "output/postprocess_quality_analysis_20260804/baseline_3min_polygon/04_tracking/tracked.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 5290,
        "stratum": "few relatively long tracks",
    },
    "heyzo_canonical1080_many_tracks": {
        "source": ROOT
        / "output/postprocess_quality_analysis_20260804/multi_input/heyzo3545_ellipse/04_tracking/tracked.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 2400,
        "coordinate_scale": 1.5,
        "stratum": "720p source canonicalized to the deployed 1080p workspace; many short tracks",
    },
    "kpi_excerpt_many_tracks": {
        "source": ROOT
        / "output/postprocess_quality_analysis_20260804/multi_input/kpi_ellipse/04_tracking/tracked.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 2400,
        "stratum": "1080p excerpt; many short tracks",
    },
    "release_qa_sparse_with_cuts": {
        "source": ROOT
        / "output/release_final_qa_20260818/postprocess_preview_join_fix/04_tracking/tracked.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 900,
        "stratum": "sparse short clip with scene cuts",
    },
    "single_large_long_track": {
        "source": ROOT
        / "output/runtime_speed_stability_20260830/long_track_chunking/track26_1519.sqlite",
        "width": 1920,
        "height": 1080,
        "video_frames": 1519,
        "stratum": "one long, large-mask, 20-vertex track",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--target-interval", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=tuple(DATASETS), help=argparse.SUPPRESS)
    return parser


def _source_characteristics(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        rows = database.execute(
            "SELECT frame,track_id,label FROM masks ORDER BY frame,track_id"
        ).fetchall()
        has_cuts = database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cuts'"
        ).fetchone()
        cuts = (
            int(database.execute("SELECT COUNT(*) FROM cuts").fetchone()[0])
            if has_cuts
            else 0
        )
    frames = [int(row[0]) for row in rows]
    tracks: dict[str, int] = {}
    label_rows: dict[str, int] = {}
    label_tracks: dict[str, set[str]] = {}
    for _frame, track, label in rows:
        key = str(track)
        value = str(label)
        tracks[key] = tracks.get(key, 0) + 1
        label_rows[value] = label_rows.get(value, 0) + 1
        label_tracks.setdefault(value, set()).add(key)
    lengths = np.asarray(tuple(tracks.values()), dtype=np.float64)
    return {
        "observation_rows": int(len(rows)),
        "active_mask_frames": int(len(set(frames))),
        "first_mask_frame": int(min(frames)) if frames else None,
        "last_mask_frame": int(max(frames)) if frames else None,
        "tracks": int(len(tracks)),
        "active_labels": int(len(label_rows)),
        "cuts": cuts,
        "track_length_median": float(np.median(lengths)) if len(lengths) else 0.0,
        "track_length_p95": float(np.quantile(lengths, 0.95)) if len(lengths) else 0.0,
        "track_length_max": int(np.max(lengths)) if len(lengths) else 0,
        "rows_by_label": label_rows,
        "tracks_by_label": {
            label: len(values) for label, values in label_tracks.items()
        },
    }


def _scaled_tracked_source(source: Path, output: Path, scale: float) -> Path:
    """Create the same coordinate enlargement used before Production postprocess."""

    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.sqlite")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)

    def scaled_polygons(raw: str) -> str:
        value = json.loads(raw)
        container = value
        if isinstance(value, dict):
            polygons = value.get("polygons", value.get("points", []))
        else:
            polygons = value
        converted = [
            [[float(point[0]) * scale, float(point[1]) * scale] for point in polygon]
            for polygon in polygons or []
        ]
        if isinstance(container, dict):
            key = "polygons" if "polygons" in container else "points"
            container[key] = converted
            return json.dumps(container, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(converted, ensure_ascii=False, separators=(",", ":"))

    with sqlite3.connect(temporary) as database:
        for table in ("masks", "raw_tracked_masks"):
            exists = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                continue
            rows = database.execute(f"SELECT rowid,polygons FROM {table}").fetchall()
            database.executemany(
                f"UPDATE {table} SET polygons=? WHERE rowid=?",
                ((scaled_polygons(str(raw)), int(rowid)) for rowid, raw in rows),
            )
        database.commit()
    temporary.replace(output)
    return output


def _exact_metrics(phase2_root: Path, labels: tuple[str, ...]) -> dict[str, object]:
    ious: list[float] = []
    recalls: list[float] = []
    area_ratios: list[float] = []
    per_label: dict[str, dict[str, object]] = {}
    stage_work: dict[str, float] = {}
    interval_eval_count = 0
    interval_eval_frames = 0
    run_count = 0
    candidate_frame_count = 0
    maximum_workers = 0
    for label in labels:
        runtime = phase2_root / RUNTIME_POLYGON_PROFILE_ID / label / "runtime"
        label_ious: list[float] = []
        label_recalls: list[float] = []
        label_areas: list[float] = []
        with (runtime / "exact/keyframe_exact_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                iou = float(row["iou"])
                recall = float(row["recall"])
                area_ratio = float(row["pred_area"]) / max(float(row["gt_area"]), 1.0)
                label_ious.append(iou)
                label_recalls.append(recall)
                label_areas.append(area_ratio)
        ious.extend(label_ious)
        recalls.extend(label_recalls)
        area_ratios.extend(label_areas)
        summary = json.loads((runtime / "opt/summary.json").read_text(encoding="utf-8"))
        run_count += int(summary.get("run_count", 0))
        interval_eval_count += int(summary.get("interval_eval_count", 0))
        interval_eval_frames += int(summary.get("interval_eval_frames", 0))
        candidate_frame_count += int(summary.get("candidate_frame_count_total", 0))
        maximum_workers = max(maximum_workers, int(summary.get("num_workers", 0)))
        for key, value in summary.get("stage_seconds_total", {}).items():
            stage_work[key] = stage_work.get(key, 0.0) + float(value)
        per_label[label] = {
            "rows": len(label_ious),
            "mean_iou": float(np.mean(label_ious)),
            "p01_iou": float(np.quantile(label_ious, 0.01)),
            "minimum_iou": float(np.min(label_ious)),
            "minimum_recall": float(np.min(label_recalls)),
        }
    iou_array = np.asarray(ious, dtype=np.float64)
    recall_array = np.asarray(recalls, dtype=np.float64)
    area_array = np.asarray(area_ratios, dtype=np.float64)
    return {
        "evaluated_rows": int(len(ious)),
        "mean_iou": float(np.mean(iou_array)),
        "p05_iou": float(np.quantile(iou_array, 0.05)),
        "p01_iou": float(np.quantile(iou_array, 0.01)),
        "minimum_iou": float(np.min(iou_array)),
        "minimum_recall": float(np.min(recall_array)),
        "recall_violations": int(np.sum(recall_array + 1e-12 < 0.97)),
        "area_ratio_p95": float(np.quantile(area_array, 0.95)),
        "area_ratio_p99": float(np.quantile(area_array, 0.99)),
        "area_ratio_max": float(np.max(area_array)),
        "per_label": per_label,
        "run_count": run_count,
        "interval_eval_count": interval_eval_count,
        "interval_eval_frames": interval_eval_frames,
        "candidate_frame_count": candidate_frame_count,
        "maximum_optimizer_workers_per_label": maximum_workers,
        "stage_work_seconds": stage_work,
    }


def _worker(args: argparse.Namespace) -> int:
    if args.dataset is None:
        raise ValueError("--dataset is required in worker mode")
    spec = DATASETS[args.dataset]
    original_source = Path(spec["source"])
    if not original_source.is_file():
        raise FileNotFoundError(original_source)
    dataset_root = args.output.resolve() / "datasets" / args.dataset
    if args.force and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    result_path = dataset_root / "result.json"
    if result_path.is_file() and not args.force:
        print(result_path)
        return 0

    coordinate_scale = float(spec.get("coordinate_scale", 1.0))
    source = (
        _scaled_tracked_source(
            original_source,
            dataset_root / "derived_input" / "tracked_1920x1080.sqlite",
            coordinate_scale,
        )
        if coordinate_scale != 1.0
        else original_source
    )
    characteristics = _source_characteristics(source)
    config = replace(PRODUCTION, target_interval=int(args.target_interval))
    preparation_started = time.perf_counter()
    source_root, preparation = prepare_inputs(
        source,
        dataset_root / "preparation",
        width=int(spec["width"]),
        height=int(spec["height"]),
        input_video=None,
        config=config,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    active_labels = tuple(str(value) for value in preparation["active_labels"])
    run = optimize(
        Path(source_root),
        dataset_root / "optimizer",
        labels=active_labels,
        max_tracks=0,
        force=True,
        config=config,
    )
    phase2_root = Path(run["phase2_root"])
    quality = _exact_metrics(phase2_root, active_labels)
    matrix = json.loads((phase2_root / "phase2_matrix.json").read_text(encoding="utf-8"))
    aggregate = matrix["completed_profiles"][-1]
    execution = matrix.get("execution", {})
    optimizer_seconds = float(run["wall_seconds"])
    video_frames = int(spec["video_frames"])
    rss_kib = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    policy = preparation["vertex_policy"]["summary"]
    result = {
        "schema_version": 1,
        "dataset": args.dataset,
        "stratum": spec["stratum"],
        "source": str(source.resolve()),
        "original_source": str(original_source.resolve()),
        "coordinate_scale": coordinate_scale,
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "video_frames": video_frames,
        "target_interval": int(args.target_interval),
        "source_characteristics": characteristics,
        "preparation_seconds": float(preparation_seconds),
        "optimizer_wall_seconds": optimizer_seconds,
        "end_to_end_polygon_seconds": float(preparation_seconds + optimizer_seconds),
        "timeline_fps": float(video_frames / max(optimizer_seconds, 1e-9)),
        "observation_fps": float(
            characteristics["observation_rows"] / max(optimizer_seconds, 1e-9)
        ),
        "active_mask_frame_fps": float(
            characteristics["active_mask_frames"] / max(optimizer_seconds, 1e-9)
        ),
        "end_to_end_timeline_fps": float(
            video_frames / max(preparation_seconds + optimizer_seconds, 1e-9)
        ),
        "peak_child_rss_mib": float(rss_kib / 1024.0),
        "actual_mean_interval": float(aggregate["actual_mean_interval"]),
        "keyframes": int(aggregate["keyframes"]),
        "quality": quality,
        "vertex_policy_summary": policy,
        "execution": execution,
        "phase2_matrix": str((phase2_root / "phase2_matrix.json").resolve()),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result_path, flush=True)
    return 0


def _parent(args: argparse.Namespace) -> int:
    selected = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    unknown = tuple(value for value in selected if value not in DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for name in selected:
        result_path = root / "datasets" / name / "result.json"
        if result_path.is_file() and not args.force:
            print(f"[reuse] {name}", flush=True)
        else:
            print(f"[run] {name}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--dataset",
                name,
                "--output",
                str(root),
                "--target-interval",
                str(int(args.target_interval)),
            ]
            if args.force:
                command.append("--force")
            log_path = root / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    check=False,
                )
            if process.returncode:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                )
                raise RuntimeError(f"dataset failed: {name}\n{tail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(result)
        print(
            f"[result] {name}: timeline={result['timeline_fps']:.2f} FPS, "
            f"observations={result['observation_fps']:.2f}/s, "
            f"wall={result['optimizer_wall_seconds']:.2f}s, "
            f"recall={result['quality']['minimum_recall']:.6f}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "target_interval": int(args.target_interval),
        "results": results,
    }
    (root / "benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "dataset",
        "stratum",
        "video_frames",
        "observations",
        "tracks",
        "active_labels",
        "track_length_max",
        "vertices20_row_share",
        "optimizer_wall_seconds",
        "timeline_fps",
        "observation_fps",
        "end_to_end_timeline_fps",
        "actual_mean_interval",
        "mean_iou",
        "p01_iou",
        "minimum_iou",
        "minimum_recall",
        "recall_violations",
        "area_ratio_p95",
        "area_ratio_max",
        "peak_child_rss_mib",
    )
    with (root / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            source = result["source_characteristics"]
            quality = result["quality"]
            policy = result["vertex_policy_summary"]
            rows = max(1, int(source["observation_rows"]))
            row20 = int(policy["track_rows_by_vertices"].get("20", 0))
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "stratum": result["stratum"],
                    "video_frames": result["video_frames"],
                    "observations": source["observation_rows"],
                    "tracks": source["tracks"],
                    "active_labels": source["active_labels"],
                    "track_length_max": source["track_length_max"],
                    "vertices20_row_share": row20 / rows,
                    "optimizer_wall_seconds": result["optimizer_wall_seconds"],
                    "timeline_fps": result["timeline_fps"],
                    "observation_fps": result["observation_fps"],
                    "end_to_end_timeline_fps": result["end_to_end_timeline_fps"],
                    "actual_mean_interval": result["actual_mean_interval"],
                    "mean_iou": quality["mean_iou"],
                    "p01_iou": quality["p01_iou"],
                    "minimum_iou": quality["minimum_iou"],
                    "minimum_recall": quality["minimum_recall"],
                    "recall_violations": quality["recall_violations"],
                    "area_ratio_p95": quality["area_ratio_p95"],
                    "area_ratio_max": quality["area_ratio_max"],
                    "peak_child_rss_mib": result["peak_child_rss_mib"],
                }
            )
    print(root / "benchmark_results.json")
    return 0


def main() -> int:
    args = _parser().parse_args()
    if int(args.target_interval) < 1:
        raise ValueError("--target-interval must be >= 1")
    return _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
