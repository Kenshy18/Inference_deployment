#!/usr/bin/env python3
"""Measure the speed/quality trade-off of polygon long-track chunking.

Production deliberately keeps ``max_run_frames=30000`` so a normal track is
optimized as one global DP problem.  That preserves global decisions but leaves
only one process busy when a class contains one long track.  This experiment
uses a real 1,519-frame track and lowers the run length while retaining overlap.
It never changes the Production defaults or imports this module at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np

from production.config import PRODUCTION, RUNTIME_POLYGON_PROFILE_ID
from production.polygon.runtime_bridge import prepare_inputs


LABEL = "女性器"
TRACK_ID = "26"
WIDTH = 1920
HEIGHT = 1080
TARGET_INTERVAL = 3
DEFAULT_VARIANTS = (
    ("global", 30000, 900),
    ("chunk1200_o90", 1200, 90),
    ("chunk800_o90", 800, 90),
    ("chunk600_o60", 600, 60),
    ("chunk400_o60", 400, 60),
)
PARALLEL_VARIANTS = (
    ("candidate1_native4", 1, 4),
    ("candidate2_native4", 2, 4),
    ("candidate4_native4", 4, 4),
    ("candidate8_native4", 8, 4),
    ("candidate1_native8", 1, 8),
    ("candidate2_native8", 2, 8),
    ("candidate4_native8", 4, 8),
)
MAX_GAP_VARIANTS = (
    ("maxgap30", 30),
    ("maxgap20", 20),
    ("maxgap16", 16),
    ("maxgap14", 14),
    ("maxgap12", 12),
    ("maxgap10", 10),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=1519)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--native-threads", type=int, default=4)
    parser.add_argument(
        "--suite",
        choices=("chunking", "parallel", "max_gap"),
        default="chunking",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _make_track_source(source: Path, output: Path, observations: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
        rows = list(
            db.execute(
                "SELECT frame,polygons FROM masks WHERE track_id=? "
                "ORDER BY frame LIMIT ?",
                (TRACK_ID, int(observations)),
            )
        )
    if len(rows) != int(observations):
        raise RuntimeError(f"expected {observations} source rows, got {len(rows)}")
    first_frame = int(rows[0][0])
    with sqlite3.connect(output) as db:
        db.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                PRIMARY KEY(frame,track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY,label TEXT);
            """
        )
        db.executemany(
            """
            INSERT INTO masks(
                frame,track_id,polygons,shape_type,dilate_px,feather_px,
                mosaic_block,mosaic_alias,label
            ) VALUES (?,? ,?,'polygon',0,0,0,0,?)
            """,
            (
                (int(frame) - first_frame, TRACK_ID, str(polygons), LABEL)
                for frame, polygons in rows
            ),
        )
        db.execute(
            "INSERT INTO tracks(track_id,label) VALUES (?,?)", (TRACK_ID, LABEL)
        )
        db.commit()
        integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise RuntimeError(f"generated source failed integrity: {integrity}")


def _run_variant(
    source_root: Path,
    output: Path,
    policy_path: Path,
    *,
    max_run_frames: int,
    overlap_frames: int,
    workers: int,
    native_threads: int,
    candidate_frame_workers: int = 1,
    keyframe_max_gap: int = 30,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "production/polygon/runtime/coordinator.py"
        ),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output),
        "--profiles",
        RUNTIME_POLYGON_PROFILE_ID,
        "--labels",
        LABEL,
        "--target-interval",
        str(TARGET_INTERVAL),
        "--recall-floor",
        "0.97",
        "--keyframe-max-gap",
        str(int(keyframe_max_gap)),
        "--anchors-per-contour",
        "20",
        "--num-workers",
        str(int(workers)),
        "--label-workers",
        "1",
        "--native-batch-threads",
        str(int(native_threads)),
        "--gc-interval",
        "8",
        "--max-run-frames",
        str(int(max_run_frames)),
        "--run-overlap-frames",
        str(int(overlap_frames)),
        "--cuda-lazy-exact",
        "--cuda-lazy-frame-hints",
        "--cuda-exact-hint-count",
        "8",
        "--pair-vote-per-key",
        "--pair-vote-sweeps",
        "2",
        "--force",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
            "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": "4",
            "MASK_PIPELINE_SPATIAL_VERTEX_POLICY_JSON": str(policy_path),
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_SECONDS": "0.5",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET": "0.10",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_AREA": "0",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_SMALL_BUDGET": "0.10",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_MIN_EDGES": "1024",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_FALLBACK_INFEASIBLE_RATIO": "1.0",
            "MASK_PIPELINE_PHASE2_CANDIDATE_FRAME_WORKERS": str(
                int(candidate_frame_workers)
            ),
        }
    )
    postprocess = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(postprocess), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    log_path = output.parent / f"{output.name}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=postprocess.parent,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.perf_counter() - started
    if process.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        )
        raise RuntimeError(f"variant failed: {output.name}\n{tail}")

    runtime = output / RUNTIME_POLYGON_PROFILE_ID / LABEL / "runtime"
    summary = json.loads((runtime / "opt/summary.json").read_text(encoding="utf-8"))
    matrix = json.loads((output / "phase2_matrix.json").read_text(encoding="utf-8"))
    aggregate = matrix["completed_profiles"][0]
    with (runtime / "exact/keyframe_exact_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        exact = list(csv.DictReader(handle))
    recalls = np.asarray([float(row["recall"]) for row in exact], dtype=np.float64)
    ious = np.asarray([float(row["iou"]) for row in exact], dtype=np.float64)
    area_ratios = np.asarray(
        [
            float(row["pred_area"]) / max(float(row["gt_area"]), 1.0)
            for row in exact
        ],
        dtype=np.float64,
    )
    stage = summary.get("stage_seconds_total", {})
    prediction = runtime / "pred/predictions.sqlite"
    return {
        "wall_seconds": float(wall),
        "observation_fps": float(len(exact) / max(wall, 1e-9)),
        "prediction": str(prediction),
        "run_count": int(summary["run_count"]),
        "worker_start_method": summary.get("worker_start_method"),
        "worker_transfer_mode": summary.get("worker_transfer_mode"),
        "optimizer_seconds": float(summary["optimizer_seconds"]),
        "stage_seconds": {str(key): float(value) for key, value in stage.items()},
        "keyframes": int(aggregate["keyframes"]),
        "actual_mean_interval": float(aggregate["actual_mean_interval"]),
        "mean_iou": float(np.mean(ious)),
        "p01_iou": float(np.quantile(ious, 0.01)),
        "minimum_iou": float(np.min(ious)),
        "minimum_recall": float(np.min(recalls)),
        "recall_violations": int(np.sum(recalls + 1e-12 < 0.97)),
        "area_ratio_p95": float(np.quantile(area_ratios, 0.95)),
        "area_ratio_max": float(np.max(area_ratios)),
        "evaluated_rows": int(len(exact)),
    }


def _load_prediction(path: Path) -> dict[int, str]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(masks)")}
        if not {"frame", "polygons"}.issubset(columns):
            raise RuntimeError(f"unexpected prediction schema: {path}")
        rows = db.execute("SELECT frame,polygons FROM masks ORDER BY frame").fetchall()
    return {int(frame): str(polygons) for frame, polygons in rows}


def _polygons(raw: str) -> list[np.ndarray]:
    value = json.loads(raw)
    if isinstance(value, dict):
        value = value.get("polygons", value.get("points", []))
    output: list[np.ndarray] = []
    for polygon in value or []:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] >= 2:
            output.append(points[:, :2])
    return output


def _mask_iou(left_raw: str, right_raw: str) -> float:
    left = _polygons(left_raw)
    right = _polygons(right_raw)
    all_polygons = left + right
    if not all_polygons:
        return 1.0
    points = np.concatenate(all_polygons, axis=0)
    minimum = np.floor(np.min(points, axis=0)).astype(np.int64) - 2
    maximum = np.ceil(np.max(points, axis=0)).astype(np.int64) + 2
    width = max(1, int(maximum[0] - minimum[0] + 1))
    height = max(1, int(maximum[1] - minimum[1] + 1))

    def rasterize(polygons: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted = [
            np.rint(polygon - minimum[None, :]).astype(np.int32)
            for polygon in polygons
        ]
        if shifted:
            cv2.fillPoly(mask, shifted, 1)
        return mask

    left_mask = rasterize(left)
    right_mask = rasterize(right)
    intersection = int(np.count_nonzero(left_mask & right_mask))
    union = int(np.count_nonzero(left_mask | right_mask))
    return float(intersection / union) if union else 1.0


def _compare_predictions(baseline: Path, candidate: Path) -> dict[str, object]:
    left = _load_prediction(baseline)
    right = _load_prediction(candidate)
    frames = sorted(set(left) | set(right))
    if set(left) != set(right):
        raise RuntimeError("prediction frame sets differ")
    values = np.asarray(
        [_mask_iou(left[frame], right[frame]) for frame in frames],
        dtype=np.float64,
    )
    changed = np.flatnonzero(values < 1.0 - 1e-12)
    worst = np.argsort(values)[:20]
    return {
        "mean_iou_vs_global": float(np.mean(values)),
        "p01_iou_vs_global": float(np.quantile(values, 0.01)),
        "minimum_iou_vs_global": float(np.min(values)),
        "changed_frames": int(changed.size),
        "changed_fraction": float(changed.size / max(len(values), 1)),
        "worst_frames": [
            {"frame": int(frames[index]), "iou_vs_global": float(values[index])}
            for index in worst
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.observations < 2:
        raise ValueError("observations must be >= 2")
    if args.workers < 1 or args.native_threads < 1:
        raise ValueError("worker counts must be positive")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "track26_1519.sqlite"
    prepared = root / "prepared"
    if args.force or not source.is_file():
        _make_track_source(args.source.expanduser().resolve(), source, args.observations)
    if args.force and prepared.exists():
        shutil.rmtree(prepared)
    source_root = prepared / "phase2_source"
    if not source_root.exists():
        source_root, preparation = prepare_inputs(
            source,
            prepared,
            width=WIDTH,
            height=HEIGHT,
            input_video=None,
            config=PRODUCTION,
        )
        source_root = Path(source_root).resolve()
        (root / "preparation_summary.json").write_text(
            json.dumps(preparation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    source_root = Path(source_root).resolve()
    policy_path = source_root / "vertex_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)

    results: list[dict[str, object]] = []
    if args.suite == "chunking":
        variants = [
            (name, max_run_frames, overlap_frames, 1, args.native_threads, 30)
            for name, max_run_frames, overlap_frames in DEFAULT_VARIANTS
        ]
    elif args.suite == "parallel":
        variants = [
            (name, 30000, 900, candidate_workers, native_threads, 30)
            for name, candidate_workers, native_threads in PARALLEL_VARIANTS
        ]
    else:
        variants = [
            (name, 30000, 900, 2, 4, keyframe_max_gap)
            for name, keyframe_max_gap in MAX_GAP_VARIANTS
        ]
    for (
        name,
        max_run_frames,
        overlap_frames,
        candidate_frame_workers,
        native_threads,
        keyframe_max_gap,
    ) in variants:
        print(f"[variant] {name}", flush=True)
        metrics = _run_variant(
            source_root,
            root / "runs" / name,
            policy_path,
            max_run_frames=max_run_frames,
            overlap_frames=overlap_frames,
            workers=args.workers,
            native_threads=native_threads,
            candidate_frame_workers=candidate_frame_workers,
            keyframe_max_gap=keyframe_max_gap,
        )
        metrics.update(
            {
                "name": name,
                "max_run_frames": int(max_run_frames),
                "overlap_frames": int(overlap_frames),
                "candidate_frame_workers": int(candidate_frame_workers),
                "native_threads": int(native_threads),
                "keyframe_max_gap": int(keyframe_max_gap),
            }
        )
        results.append(metrics)
        print(
            f"[result] {name} wall={metrics['wall_seconds']:.3f}s "
            f"fps={metrics['observation_fps']:.2f} runs={metrics['run_count']} "
            f"iou={metrics['mean_iou']:.6f} recall={metrics['minimum_recall']:.6f}",
            flush=True,
        )

    baseline = Path(str(results[0]["prediction"]))
    for metrics in results:
        metrics["comparison_to_global"] = _compare_predictions(
            baseline, Path(str(metrics["prediction"]))
        )
    output = {
        "schema_version": 1,
        "source": str(source),
        "observations": int(args.observations),
        "target_interval": TARGET_INTERVAL,
        "workers": int(args.workers),
        "suite": str(args.suite),
        "variants": results,
    }
    path = root / "benchmark_results.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
