#!/usr/bin/env python3
"""Validate speed-stability controls on the complete V3 KPI mask corpus."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np

from production.config import LABELS, PRODUCTION, RUNTIME_POLYGON_PROFILE_ID
from production.polygon.runtime_bridge import prepare_inputs


WIDTH = 1920
HEIGHT = 1080
VARIANTS = (
    ("production", 1, 30),
    ("stable_fast", 2, 16),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", default="3,6")
    parser.add_argument("--video-frames", type=int, default=23510)
    parser.add_argument("--force-preparation", action="store_true")
    return parser


def _run(
    source_root: Path,
    output: Path,
    policy_path: Path,
    *,
    target_interval: int,
    candidate_workers: int,
    max_gap: int,
    optimizer_workers: int = 9,
    label_workers: int = 3,
    native_threads: int = 4,
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    coordinator = (
        Path(__file__).resolve().parents[2]
        / "production/polygon/runtime/coordinator.py"
    )
    command = [
        sys.executable,
        str(coordinator),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output),
        "--profiles",
        RUNTIME_POLYGON_PROFILE_ID,
        "--labels",
        ",".join(LABELS),
        "--target-interval",
        str(int(target_interval)),
        "--recall-floor",
        "0.97",
        "--anchors-per-contour",
        "20",
        "--num-workers",
        str(int(optimizer_workers)),
        "--label-workers",
        str(int(label_workers)),
        "--native-batch-threads",
        str(int(native_threads)),
        "--gc-interval",
        "8",
        "--max-run-frames",
        "30000",
        "--run-overlap-frames",
        "900",
        "--keyframe-max-gap",
        str(int(max_gap)),
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
                int(candidate_workers)
            ),
        }
    )
    if environment_overrides:
        environment.update(
            {str(key): str(value) for key, value in environment_overrides.items()}
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
    if process.returncode:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        )
        raise RuntimeError(f"full-corpus run failed: {output}\n{tail}")
    matrix = json.loads((output / "phase2_matrix.json").read_text(encoding="utf-8"))
    aggregate = matrix["completed_profiles"][0]
    ious: list[float] = []
    recalls: list[float] = []
    area_ratios: list[float] = []
    summaries: dict[str, object] = {}
    prediction_paths: dict[str, str] = {}
    keyframe_paths: dict[str, str] = {}
    for label in LABELS:
        runtime = output / RUNTIME_POLYGON_PROFILE_ID / label / "runtime"
        with (runtime / "exact/keyframe_exact_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                ious.append(float(row["iou"]))
                recalls.append(float(row["recall"]))
                area_ratios.append(
                    float(row["pred_area"]) / max(float(row["gt_area"]), 1.0)
                )
        summaries[label] = json.loads(
            (runtime / "opt/summary.json").read_text(encoding="utf-8")
        )
        prediction_paths[label] = str(runtime / "pred/predictions.sqlite")
        keyframe_paths[label] = str(runtime / "opt/final_keyframes.json")
    iou_array = np.asarray(ious, dtype=np.float64)
    recall_array = np.asarray(recalls, dtype=np.float64)
    area_array = np.asarray(area_ratios, dtype=np.float64)
    return {
        "wall_seconds": float(wall),
        "observation_rows": int(len(ious)),
        "observation_fps": float(len(ious) / max(wall, 1e-9)),
        "keyframes": int(aggregate["keyframes"]),
        "actual_mean_interval": float(aggregate["actual_mean_interval"]),
        "mean_iou": float(np.mean(iou_array)),
        "p01_iou": float(np.quantile(iou_array, 0.01)),
        "minimum_iou": float(np.min(iou_array)),
        "minimum_recall": float(np.min(recall_array)),
        "recall_violations": int(np.sum(recall_array + 1e-12 < 0.97)),
        "area_ratio_p95": float(np.quantile(area_array, 0.95)),
        "area_ratio_max": float(np.max(area_array)),
        "summaries": summaries,
        "predictions": prediction_paths,
        "keyframe_paths": keyframe_paths,
    }


def _prediction_rows(path: Path) -> dict[tuple[int, str], str]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT frame,track_id,polygons FROM masks ORDER BY frame,track_id"
        ).fetchall()
    return {(int(frame), str(track)): str(polygons) for frame, track, polygons in rows}


def _polygons(raw: str) -> list[np.ndarray]:
    value = json.loads(raw)
    if isinstance(value, dict):
        value = value.get("polygons", value.get("points", []))
    return [
        points[:, :2]
        for polygon in value or []
        for points in (np.asarray(polygon, dtype=np.float64),)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] >= 2
    ]


def _iou(left_raw: str, right_raw: str) -> float:
    left = _polygons(left_raw)
    right = _polygons(right_raw)
    polygons = left + right
    if not polygons:
        return 1.0
    points = np.concatenate(polygons, axis=0)
    minimum = np.floor(points.min(axis=0)).astype(np.int64) - 2
    maximum = np.ceil(points.max(axis=0)).astype(np.int64) + 2
    shape = (
        max(1, int(maximum[1] - minimum[1] + 1)),
        max(1, int(maximum[0] - minimum[0] + 1)),
    )

    def rasterize(values: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        shifted = [
            np.rint(value - minimum[None, :]).astype(np.int32) for value in values
        ]
        if shifted:
            cv2.fillPoly(mask, shifted, 1)
        return mask

    left_mask = rasterize(left)
    right_mask = rasterize(right)
    union = int(np.count_nonzero(left_mask | right_mask))
    return (
        float(np.count_nonzero(left_mask & right_mask) / union) if union else 1.0
    )


def _compare(base: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    exact_equal_rows = 0
    row_count = 0
    changed_ious: list[tuple[float, str, int, str]] = []
    keyframes_equal = True
    for label in LABELS:
        left = _prediction_rows(Path(str(base["predictions"][label])))
        right = _prediction_rows(Path(str(candidate["predictions"][label])))
        if set(left) != set(right):
            raise RuntimeError(f"prediction key mismatch for {label}")
        row_count += len(left)
        for key, left_raw in left.items():
            right_raw = right[key]
            if left_raw == right_raw:
                exact_equal_rows += 1
            else:
                changed_ious.append((_iou(left_raw, right_raw), label, key[0], key[1]))
        left_keys = json.loads(Path(str(base["keyframe_paths"][label])).read_text())
        right_keys = json.loads(
            Path(str(candidate["keyframe_paths"][label])).read_text()
        )
        keyframes_equal = keyframes_equal and left_keys == right_keys
    changed_ious.sort(key=lambda value: value[0])
    values = np.asarray([value[0] for value in changed_ious], dtype=np.float64)
    return {
        "rows": int(row_count),
        "exact_equal_rows": int(exact_equal_rows),
        "changed_rows": int(len(changed_ious)),
        "keyframes_exact_equal": bool(keyframes_equal),
        "changed_mean_iou": float(np.mean(values)) if len(values) else 1.0,
        "changed_minimum_iou": float(np.min(values)) if len(values) else 1.0,
        "worst_changed_rows": [
            {
                "iou": float(iou),
                "label": label,
                "frame": int(frame),
                "track_id": track,
            }
            for iou, label, frame, track in changed_ious[:30]
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    intervals = tuple(
        int(value.strip()) for value in args.intervals.split(",") if value.strip()
    )
    if not intervals or min(intervals) < 1:
        raise ValueError("intervals must contain positive integers")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared"
    source_root = prepared / "phase2_source"
    if args.force_preparation and prepared.exists():
        shutil.rmtree(prepared)
    if not source_root.exists():
        source_root, preparation = prepare_inputs(
            args.source.expanduser().resolve(),
            prepared,
            width=WIDTH,
            height=HEIGHT,
            input_video=None,
            config=replace(PRODUCTION, target_interval=intervals[0]),
        )
        (root / "preparation_summary.json").write_text(
            json.dumps(preparation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    source_root = Path(source_root).resolve()
    policy_path = source_root / "vertex_policy.json"
    results: list[dict[str, object]] = []
    by_interval: dict[int, dict[str, dict[str, object]]] = {}
    for interval in intervals:
        by_interval[interval] = {}
        for name, candidate_workers, max_gap in VARIANTS:
            print(f"[run] interval={interval} variant={name}", flush=True)
            metrics = _run(
                source_root,
                root / "runs" / f"interval{interval}_{name}",
                policy_path,
                target_interval=interval,
                candidate_workers=candidate_workers,
                max_gap=max_gap,
            )
            metrics.update(
                {
                    "target_interval": int(interval),
                    "variant": name,
                    "candidate_workers": int(candidate_workers),
                    "keyframe_max_gap": int(max_gap),
                    "video_fps": float(args.video_frames / metrics["wall_seconds"]),
                }
            )
            results.append(metrics)
            by_interval[interval][name] = metrics
            print(
                f"[result] interval={interval} {name} "
                f"wall={metrics['wall_seconds']:.3f}s "
                f"video_fps={metrics['video_fps']:.2f} "
                f"iou={metrics['mean_iou']:.6f} "
                f"recall={metrics['minimum_recall']:.6f}",
                flush=True,
            )
    comparisons = {
        str(interval): _compare(
            by_interval[interval]["production"],
            by_interval[interval]["stable_fast"],
        )
        for interval in intervals
    }
    payload = {
        "schema_version": 1,
        "source": str(args.source.expanduser().resolve()),
        "video_frames": int(args.video_frames),
        "intervals": list(intervals),
        "results": results,
        "comparisons": comparisons,
    }
    path = root / "benchmark_results.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
