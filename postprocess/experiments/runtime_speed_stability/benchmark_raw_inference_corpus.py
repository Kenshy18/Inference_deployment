#!/usr/bin/env python3
"""Benchmark current Production postprocess across diverse raw AI outputs.

The primary corpus contains independent, full or intentionally bounded V3
inference outputs.  Supplemental cases exercise an older compact model and
media-coordinate variants, and are reported separately so they cannot inflate
the independent-scene sample size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import resource
import signal
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNTIME_PYTHON = Path(
    "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10"
)
POSTPROCESS_CLI = ROOT / "postprocess/run_pipeline.py"


def _v3(name: str) -> Path:
    return ROOT / "output/v3_postprocessed_target_interval_3_20260817/sqlite" / name


DATASETS: dict[str, dict[str, object]] = {
    "v3_heyzo3554": {
        "source": _v3(
            "HEYZO-3554 小野寺まり おのてらまり 熟女泡姫のテクてイカせてアケル美女コレクションVol.sqlite"
        ),
        "cohort": "primary_v3",
        "stratum": "720p/24fps; long; high mask density",
    },
    "v3_heyzo3560": {
        "source": _v3(
            "HEYZO-3560 夏目りんか なつめりんか 欲求不満な私を好きにしてくたさい - 無修正アタルト動画.sqlite"
        ),
        "cohort": "primary_v3",
        "stratum": "720p/24fps; long; high mask density",
    },
    "v3_heyzo3549": {
        "source": _v3(
            "HEYZO-3549 浜田希 はまたのそみ 激しめイラマか好き - 無修正アタルト動画 HEYZO -.sqlite"
        ),
        "cohort": "primary_v3",
        "stratum": "720p/24fps; long; medium mask density",
    },
    "v3_heyzo3545": {
        "source": _v3(
            "HEYZO-3545 乙葉いおり おとはいおり 性の悩みはホクかトヒュっと解決しますおしゃふりは浮気し.sqlite"
        ),
        "cohort": "primary_v3",
        "stratum": "720p/24fps; long; medium mask density",
    },
    "v3_white0210": {
        "source": _v3("3月以降解析白カン動画-0210.sqlite"),
        "cohort": "primary_v3",
        "stratum": "1080p/29.97fps; 30min; high multi-class density",
    },
    "v3_kpi": {
        "source": _v3("12月KPI動画.sqlite"),
        "cohort": "primary_v3",
        "stratum": "1080p/29.97fps; diverse KPI scenes",
    },
    "v3_white_axel": {
        "source": _v3("アクセル様２月解析用白カン01.26.sqlite"),
        "cohort": "primary_v3",
        "stratum": "1080p/29.97fps; female-class dominant",
    },
    "v3_sdam_sparse": {
        "source": _v3("SDAM-151_AIモザイク_アクセル様.sqlite"),
        "cohort": "primary_v3",
        "stratum": "1080p/29.97fps; bounded 10min; very sparse masks",
    },
    "v3_joined_male": {
        "source": _v3("連結済み_長時間動画.sqlite"),
        "cohort": "primary_v3",
        "stratum": "720p/24fps; 10min; male-class dominant",
    },
    "lite_kpi_2400": {
        "source": ROOT
        / "output/postprocess_quality_analysis_20260804/multi_input/kpi_2400.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; 1080p KPI excerpt",
    },
    "lite_heyzo3545_2400": {
        "source": ROOT
        / "output/postprocess_quality_analysis_20260804/multi_input/heyzo3545_2400.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; 720p excerpt",
    },
    "lite_progressive_24p": {
        "source": ROOT
        / "output/interlace_validation_20260803/runs/progressive/progressive_24p.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; progressive 720p/24fps",
    },
    "lite_interlaced_tff": {
        "source": ROOT
        / "output/interlace_validation_20260803/runs_final/tff/interlaced_tff_24i.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; TFF interlaced source",
    },
    "lite_interlaced_bff": {
        "source": ROOT
        / "output/interlace_validation_20260803/runs_final/bff/interlaced_bff_24i.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; BFF interlaced source",
    },
    "lite_4k_short": {
        "source": ROOT
        / "output/deployment_change_validation_20260804/run_4k/progressive_4k_2s.sqlite",
        "cohort": "supplemental_model_media",
        "stratum": "older compact model; 4K/24fps; startup-only stress",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--target-interval", type=int, default=3)
    parser.add_argument(
        "--mask-geometry",
        choices=("polygon", "catmull_rom"),
        default="polygon",
        help="Production geometry implementation to benchmark",
    )
    parser.add_argument(
        "--optimizer-workers",
        type=int,
        default=0,
        help=(
            "explicit polygon optimizer process cap; zero uses the current "
            "Production adaptive hardware-aware default"
        ),
    )
    parser.add_argument(
        "--minimum-available-memory-mib",
        type=int,
        default=8192,
        help="abort the current run below this system MemAvailable floor",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="retain generated canonical and pipeline SQLite artifacts",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=tuple(DATASETS), help=argparse.SUPPRESS)
    return parser


def _tables(database: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _raw_profile(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True) as db:
        tables = _tables(db)
        required = {"frames", "detections", "segmentations"}
        if not required.issubset(tables):
            raise ValueError(f"raw inference tables missing from {path}")
        frame_count, first_frame, last_frame = db.execute(
            "SELECT COUNT(*),MIN(frame_index),MAX(frame_index) FROM frames"
        ).fetchone()
        dimensions = db.execute(
            "SELECT width,height,COUNT(*) FROM frames GROUP BY width,height"
        ).fetchall()
        if len(dimensions) != 1:
            raise ValueError(f"mixed frame dimensions in {path}: {dimensions}")
        width, height, _ = dimensions[0]
        fps = 0.0
        video_path = ""
        if "videos" in tables:
            video = db.execute(
                "SELECT path,fps FROM videos ORDER BY id LIMIT 1"
            ).fetchone()
            if video is not None:
                video_path, fps = str(video[0]), float(video[1])
        segmentation_rows = int(
            db.execute("SELECT COUNT(*) FROM segmentations").fetchone()[0]
        )
        polygon_rows = (
            int(db.execute("SELECT COUNT(*) FROM segmentation_polygons").fetchone()[0])
            if "segmentation_polygons" in tables
            else 0
        )
        point_rows = (
            int(db.execute("SELECT COUNT(*) FROM segmentation_points").fetchone()[0])
            if "segmentation_points" in tables
            else 0
        )
        class_rows = []
        score_rows: list[float] = []
        if "classifications" in tables:
            class_rows = [
                {"label": str(label), "detections": int(count)}
                for label, count in db.execute(
                    "SELECT class_name,COUNT(*) FROM classifications "
                    "GROUP BY class_name ORDER BY class_name"
                )
            ]
            score_rows = [
                float(row[0])
                for row in db.execute("SELECT score FROM classifications")
                if row[0] is not None
            ]
        model = "unknown"
        runtime_model = "unknown"
        backend = "unknown"
        if "model_executions" in tables:
            model_row = db.execute(
                "SELECT model_id,runtime_model_id,backend FROM model_executions "
                "WHERE role='instance_segmentation' ORDER BY id LIMIT 1"
            ).fetchone()
            if model_row is not None:
                model, runtime_model, backend = map(str, model_row)
        cuts = (
            int(db.execute("SELECT COUNT(*) FROM cuts").fetchone()[0])
            if "cuts" in tables
            else 0
        )
    scores = np.asarray(score_rows, dtype=np.float64)
    duration = float(frame_count) / fps if fps > 0 else None
    return {
        "source": str(path.resolve()),
        "source_size_mib": path.stat().st_size / 1024**2,
        "video_path": video_path,
        "frames": int(frame_count),
        "first_frame": None if first_frame is None else int(first_frame),
        "last_frame": None if last_frame is None else int(last_frame),
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "duration_seconds": duration,
        "segmentation_detections": segmentation_rows,
        "detections_per_frame": segmentation_rows / max(int(frame_count), 1),
        "polygons": polygon_rows,
        "polygon_points": point_rows,
        "mean_points_per_detection": point_rows / max(segmentation_rows, 1),
        "class_rows": class_rows,
        "score_p05": float(np.quantile(scores, 0.05)) if len(scores) else None,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_p95": float(np.quantile(scores, 0.95)) if len(scores) else None,
        "model": model,
        "runtime_model": runtime_model,
        "backend": backend,
        "stored_cuts": cuts,
    }


def _write_cuts(source: Path, output: Path) -> None:
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro&immutable=1", uri=True) as db:
        tables = _tables(db)
        frames = (
            [int(row[0]) for row in db.execute("SELECT frame FROM cuts ORDER BY frame")]
            if "cuts" in tables
            else []
        )
        method = "reused_from_integrated_source"
        elapsed = 0.0
        if "cut_detection_metadata" in tables:
            row = db.execute(
                "SELECT method,elapsed_seconds FROM cut_detection_metadata "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                method = str(row[0]) or method
                elapsed = float(row[1] or 0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": frames,
                "method": method,
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_source(source: Path, output: Path, profile: dict[str, object]) -> tuple[Path, float]:
    width = int(profile["width"])
    height = int(profile["height"])
    if width * 9 != height * 16 or (width, height) == (1920, 1080):
        return source, 0.0
    started = time.perf_counter()
    from orchestration.rescale_result_sqlite import (
        VideoGeometry,
        rescale_inference_sqlite_for_postprocess,
    )

    frames = int(profile["frames"])
    fps = float(profile["fps"])
    if output.exists():
        output.unlink()
    rescale_inference_sqlite_for_postprocess(
        source,
        output,
        inference=VideoGeometry(width, height, fps, frames),
        workspace=VideoGeometry(1920, 1080, fps, frames),
        workspace_video=Path(str(profile["video_path"] or source)),
    )
    return output, time.perf_counter() - started


def _stage(manifest: dict[str, object], stage_id: str) -> dict[str, object] | None:
    for row in manifest.get("stages", []):
        if row.get("id") == stage_id:
            return row
    return None


def _process_tree_rss_kib(root_pid: int) -> int:
    """Return aggregate RSS for root_pid and all currently live descendants."""

    processes: dict[int, tuple[int, int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent = 0
            rss = 0
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
            processes[pid] = (parent, rss)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _available_memory_kib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _quality_from_production_manifests(
    production_manifest_paths: list[Path],
) -> dict[str, object]:
    ious: list[float] = []
    recalls: list[float] = []
    areas: list[float] = []
    run_count = 0
    keyframes = 0
    vertex_rows = {"14": 0, "16": 0, "18": 0, "20": 0}
    exact_recall_violations = 0
    for production_manifest_path in production_manifest_paths:
        production_manifest = json.loads(
            production_manifest_path.read_text(encoding="utf-8")
        )
        exact_recall_violations += int(production_manifest["exact_recall_violations"])
        summary = production_manifest["vertex_policy"]["summary"]
        for key, value in summary.get("track_rows_by_vertices", {}).items():
            vertex_rows[str(key)] = vertex_rows.get(str(key), 0) + int(value)
        keyframes += int(production_manifest["materialization"]["keyframes"])
        phase2_root = Path(production_manifest["optimizer"]["phase2_root"])
        for label in production_manifest["optimizer"]["active_labels"]:
            runtime = phase2_root / "polygon_adaptive_keyframe_v2" / label / "runtime"
            summary_path = runtime / "opt/summary.json"
            if summary_path.is_file():
                run_count += int(json.loads(summary_path.read_text()).get("run_count", 0))
            with (runtime / "exact/keyframe_exact_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    ious.append(float(row["iou"]))
                    recalls.append(float(row["recall"]))
                    areas.append(float(row["pred_area"]) / max(float(row["gt_area"]), 1.0))
    if not ious:
        return {
            "evaluated_rows": 0,
            "keyframes": keyframes,
            "run_count": run_count,
            "exact_recall_violations": exact_recall_violations,
            "vertex_rows": vertex_rows,
        }
    iou = np.asarray(ious, dtype=np.float64)
    recall = np.asarray(recalls, dtype=np.float64)
    area = np.asarray(areas, dtype=np.float64)
    return {
        "evaluated_rows": len(ious),
        "keyframes": keyframes,
        "actual_interval": len(ious) / max(keyframes, 1),
        "run_count": run_count,
        "mean_iou": float(np.mean(iou)),
        "p05_iou": float(np.quantile(iou, 0.05)),
        "p01_iou": float(np.quantile(iou, 0.01)),
        "minimum_iou": float(np.min(iou)),
        "minimum_recall": float(np.min(recall)),
        "recall_violations": int(np.sum(recall + 1e-12 < 0.97)),
        "exact_recall_violations": exact_recall_violations,
        "area_ratio_p95": float(np.quantile(area, 0.95)),
        "area_ratio_p99": float(np.quantile(area, 0.99)),
        "area_ratio_max": float(np.max(area)),
        "vertex_rows": vertex_rows,
    }


def _quality_from_curve_manifests(
    production_manifest_paths: list[Path],
) -> dict[str, object]:
    """Return curve quality in the same normalized shape as polygon quality."""

    ious: list[float] = []
    recalls: list[float] = []
    areas: list[float] = []
    output_rows = 0
    keyframes = 0
    run_count = 0
    topology_invalid = 0
    point_rows = {"14": 0, "16": 0, "18": 0, "20": 0}
    audit_recall_violations = 0
    for production_manifest_path in production_manifest_paths:
        production_manifest = json.loads(
            production_manifest_path.read_text(encoding="utf-8")
        )
        engine = dict(production_manifest["engine"])
        output_rows += int(engine.get("prediction_rows", 0))
        keyframes += int(engine.get("keyframe_rows", 0))
        run_count += int(engine.get("streams", 0))
        audit = dict(engine.get("audit", {}))
        audit_recall_violations += int(audit.get("recall_violations", 0))
        topology_invalid += int(audit.get("topology_invalid_frames", 0))
        metrics_path = Path(str(engine["component_metrics_csv"]))
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ious.append(float(row["iou"]))
                recalls.append(float(row["recall"]))
                areas.append(float(row["area_ratio"]))
                points = str(int(row["points_per_component"]))
                point_rows[points] = point_rows.get(points, 0) + 1
                topology_invalid += int(row["topology_valid"]) == 0
    if not ious:
        return {
            "evaluated_rows": 0,
            "keyframes": keyframes,
            "run_count": run_count,
            "exact_recall_violations": audit_recall_violations,
            "topology_invalid": topology_invalid,
            "vertex_rows": point_rows,
        }
    iou = np.asarray(ious, dtype=np.float64)
    recall = np.asarray(recalls, dtype=np.float64)
    area = np.asarray(areas, dtype=np.float64)
    return {
        "evaluated_rows": len(ious),
        "output_rows": output_rows,
        "keyframes": keyframes,
        "actual_interval": output_rows / max(keyframes, 1),
        "run_count": run_count,
        "mean_iou": float(np.mean(iou)),
        "p05_iou": float(np.quantile(iou, 0.05)),
        "p01_iou": float(np.quantile(iou, 0.01)),
        "minimum_iou": float(np.min(iou)),
        "minimum_recall": float(np.min(recall)),
        "recall_violations": int(np.sum(recall + 1e-12 < 0.97)),
        "exact_recall_violations": audit_recall_violations,
        "topology_invalid": topology_invalid,
        "area_ratio_p95": float(np.quantile(area, 0.95)),
        "area_ratio_p99": float(np.quantile(area, 0.99)),
        "area_ratio_max": float(np.max(area)),
        "vertex_rows": point_rows,
    }


def _quality_and_geometry_stage(
    manifest: dict[str, object],
    geometry: str,
) -> tuple[dict[str, object], dict[str, object]]:
    stage_id = "polygon_optimization" if geometry == "polygon" else "curve_optimization"
    artifact_id = (
        "production_polygon_manifest"
        if geometry == "polygon"
        else "production_curve_manifest"
    )
    summarize = (
        _quality_from_production_manifests
        if geometry == "polygon"
        else _quality_from_curve_manifests
    )
    classwise = _stage(manifest, "classwise_postprocess")
    if classwise is not None:
        classwise_manifest = json.loads(
            Path(classwise["artifacts"]["classwise_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        production_manifests: list[Path] = []
        for group in classwise_manifest.get("groups", []):
            pipeline = json.loads(
                Path(group["pipeline_manifest"]).read_text(encoding="utf-8")
            )
            geometry_stage = _stage(pipeline, stage_id)
            if geometry_stage is not None:
                production_manifests.append(
                    Path(geometry_stage["artifacts"][artifact_id])
                )
        return summarize(production_manifests), classwise

    geometry_stage = _stage(manifest, stage_id)
    if geometry_stage is None:
        raise RuntimeError(
            f"neither classwise_postprocess nor {stage_id} stage exists"
        )
    return (
        summarize([Path(geometry_stage["artifacts"][artifact_id])]),
        geometry_stage,
    )


def _worker(args: argparse.Namespace) -> int:
    if args.dataset is None:
        raise ValueError("--dataset is required in worker mode")
    spec = DATASETS[args.dataset]
    source = Path(spec["source"])
    dataset_root = args.output / "datasets" / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    result_path = dataset_root / "result.json"
    if result_path.is_file() and not args.force:
        print(result_path)
        return 0
    if args.force:
        for child in (dataset_root / "pipeline", dataset_root / "canonical.sqlite"):
            if child.is_dir():
                shutil.rmtree(child)
            elif child.exists():
                child.unlink()

    profile = _raw_profile(source)
    cuts = dataset_root / "cuts.json"
    _write_cuts(source, cuts)
    full_started = time.perf_counter()
    canonical, canonical_seconds = _canonical_source(
        source, dataset_root / "canonical.sqlite", profile
    )
    pipeline_root = dataset_root / "pipeline"
    command = [
        str(RUNTIME_PYTHON),
        str(POSTPROCESS_CLI),
        "--input-sqlite",
        str(canonical),
        "--output-dir",
        str(pipeline_root),
        "--precomputed-cuts-json",
        str(cuts),
        "--keyframe-interval",
        str(args.target_interval),
        "--mask-geometry",
        str(args.mask_geometry),
        "--score-min",
        "0.3",
    ]
    if args.mask_geometry == "polygon" and args.optimizer_workers > 0:
        command.extend(
            ["--polygon-optimizer-workers", str(args.optimizer_workers)]
        )
    log_path = dataset_root / "pipeline.log"
    peak_tree_rss_kib = 0
    minimum_available_kib = _available_memory_kib()
    memory_guard_triggered = False
    with log_path.open("w", encoding="utf-8") as log:
        pipeline_started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            peak_tree_rss_kib = max(
                peak_tree_rss_kib, _process_tree_rss_kib(os.getpid())
            )
            available_kib = _available_memory_kib()
            if available_kib:
                minimum_available_kib = min(minimum_available_kib, available_kib)
                if available_kib < args.minimum_available_memory_mib * 1024:
                    memory_guard_triggered = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
            time.sleep(0.25)
        returncode = process.wait()
        peak_tree_rss_kib = max(
            peak_tree_rss_kib, _process_tree_rss_kib(os.getpid())
        )
        pipeline_seconds = time.perf_counter() - pipeline_started
    full_seconds = time.perf_counter() - full_started
    if returncode or memory_guard_triggered:
        failure = {
            "schema_version": 1,
            "dataset": args.dataset,
            "cohort": spec["cohort"],
            "stratum": spec["stratum"],
            "status": (
                "memory_guard_triggered" if memory_guard_triggered else "failed"
            ),
            "returncode": returncode,
            "profile": profile,
            "canonicalization_seconds": canonical_seconds,
            "pipeline_wall_seconds": pipeline_seconds,
            "full_wall_seconds": full_seconds,
            "peak_process_tree_rss_mib": peak_tree_rss_kib / 1024.0,
            "minimum_system_available_memory_mib": minimum_available_kib / 1024.0,
            "log": str(log_path),
        }
        if memory_guard_triggered:
            if canonical != source and canonical.is_file():
                canonical.unlink()
            if pipeline_root.is_dir():
                shutil.rmtree(pipeline_root)
        result_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(result_path)
        return 0

    manifest_path = pipeline_root / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {
        str(row["id"]): {
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "metadata": row.get("metadata", {}),
        }
        for row in manifest["stages"]
    }
    quality, geometry_stage = _quality_and_geometry_stage(
        manifest, str(args.mask_geometry)
    )
    frames = int(profile["frames"])
    geometry_seconds = float(geometry_stage["elapsed_seconds"])
    tracking_metadata = stages.get("tracking", {}).get("metadata", {})
    nms_metadata = stages.get("nms", {}).get("metadata", {})
    rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = {
        "schema_version": 1,
        "dataset": args.dataset,
        "cohort": spec["cohort"],
        "stratum": spec["stratum"],
        "status": "complete",
        "mask_geometry": str(args.mask_geometry),
        "target_interval": args.target_interval,
        "optimizer_workers_per_label": (
            args.optimizer_workers
            if args.mask_geometry == "polygon" and args.optimizer_workers > 0
            else None
        ),
        "profile": profile,
        "canonical_source": str(canonical),
        "coordinate_scale": 1920 / int(profile["width"]) if int(profile["width"]) != 1920 else 1.0,
        "canonicalization_seconds": canonical_seconds,
        "pipeline_wall_seconds": pipeline_seconds,
        "full_wall_seconds": full_seconds,
        "pipeline_timeline_fps": frames / pipeline_seconds,
        "full_timeline_fps": frames / full_seconds,
        "geometry_seconds": geometry_seconds,
        "geometry_timeline_fps": frames / geometry_seconds,
        "geometry_observation_fps": int(quality.get("evaluated_rows", 0)) / geometry_seconds,
        "peak_child_rss_mib": rss_kib / 1024.0,
        "peak_process_tree_rss_mib": peak_tree_rss_kib / 1024.0,
        "minimum_system_available_memory_mib": minimum_available_kib / 1024.0,
        "stages": stages,
        "tracking": {
            "rows_before_prune": int(tracking_metadata.get("rows_before_prune", 0)),
            "rows_after_prune": int(tracking_metadata.get("rows_after_prune", 0)),
            "tracks_after_prune": int(tracking_metadata.get("tracks_after_prune", 0)),
            "removed_short_tracks": int(tracking_metadata.get("removed_short_tracks", 0)),
            "cuts": int(tracking_metadata.get("cuts_detected", 0)),
        },
        "nms": {
            "detections_in": int(nms_metadata.get("detections_in", 0)),
            "detections_out": int(nms_metadata.get("detections_out", 0)),
            "holes_filled": int(nms_metadata.get("holes_filled", 0)),
            "tiny_islands_removed": int(nms_metadata.get("tiny_islands_removed", 0)),
            "main_owners_suppressed": int(nms_metadata.get("main_owners_suppressed", 0)),
            "island_main_suppressed": int(nms_metadata.get("island_main_suppressed", 0)),
        },
        "quality": quality,
        "artifacts_retained": bool(args.keep_artifacts),
        "manifest": str(manifest_path) if args.keep_artifacts else None,
        "result_sqlite": (
            str(manifest["artifacts"]["result_sqlite"])
            if args.keep_artifacts
            else None
        ),
        "log": str(log_path),
    }
    if not args.keep_artifacts:
        if canonical != source and canonical.is_file():
            canonical.unlink()
        if pipeline_root.is_dir():
            shutil.rmtree(pipeline_root)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result_path)
    return 0


def _selected(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return names


def _parent(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    names = _selected(args.datasets)
    inventory = []
    for name in names:
        spec = DATASETS[name]
        source = Path(spec["source"])
        if not source.is_file():
            inventory.append(
                {
                    "dataset": name,
                    "cohort": spec["cohort"],
                    "stratum": spec["stratum"],
                    "status": "missing",
                    "source": str(source),
                }
            )
            continue
        inventory.append(
            {
                "dataset": name,
                "cohort": spec["cohort"],
                "stratum": spec["stratum"],
                "status": "available",
                **_raw_profile(source),
            }
        )
    inventory_path = args.output / "raw_corpus_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {"schema_version": 1, "datasets": inventory},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.inventory_only:
        print(inventory_path)
        return 0

    results = []
    for index, name in enumerate(names, 1):
        if not Path(DATASETS[name]["source"]).is_file():
            continue
        print(f"[raw-corpus] {index}/{len(names)} {name}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--target-interval",
            str(args.target_interval),
            "--mask-geometry",
            str(args.mask_geometry),
            "--optimizer-workers",
            str(args.optimizer_workers),
            "--minimum-available-memory-mib",
            str(args.minimum_available_memory_mib),
            "--worker",
            "--dataset",
            name,
        ]
        if args.force:
            command.append("--force")
        if args.keep_artifacts:
            command.append("--keep-artifacts")
        subprocess.run(command, cwd=ROOT, check=True)
        result_path = args.output / "datasets" / name / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") == "memory_guard_triggered"
            and args.optimizer_workers > 1
        ):
            print(
                f"[raw-corpus] {name}: memory guard triggered; retrying with 1 worker",
                flush=True,
            )
            retry = list(command)
            retry[retry.index("--optimizer-workers") + 1] = "1"
            if "--force" not in retry:
                retry.append("--force")
            subprocess.run(retry, cwd=ROOT, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(result)
    output = args.output / "benchmark_results.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_interval": args.target_interval,
                "optimizer_workers_per_label": args.optimizer_workers,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.target_interval < 1:
        raise ValueError("--target-interval must be >= 1")
    if args.optimizer_workers < 0:
        raise ValueError("--optimizer-workers must be >= 0")
    if args.minimum_available_memory_mib < 1024:
        raise ValueError("--minimum-available-memory-mib must be >= 1024")
    return _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
