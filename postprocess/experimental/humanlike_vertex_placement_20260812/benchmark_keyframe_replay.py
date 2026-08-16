#!/usr/bin/env python3
"""Replay the frozen keyframe DP/pair-vote on 14-point line-fit polygons.

The old Production key count is used as each track's soft target.  Both old
and replayed outputs are evaluated against the same tracked source masks.  No
video file is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POSTPROCESS = ROOT / "postprocess"
RUNTIME = POSTPROCESS / "experimental/0809/phase2_runtime.py"
PRODUCTION = ROOT / "data/12月KPI動画.sqlite"
CASES = (
    ("55", "男性器", ROOT / "output/phase2_engine_profile_large_track55_20260810/input_track55.sqlite"),
    ("47", "男性器", ROOT / "output/phase2_engine_profile_medium_track_20260810/input_track47.sqlite"),
    ("36", "男性器", ROOT / "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track36.sqlite"),
    ("66", "結合部分", ROOT / "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track66_segment1.sqlite"),
    ("97", "結合部分", ROOT / "output/humanlike_vertex_placement_20260812/additional_geometry_tracks/input_track97_segment1.sqlite"),
)

for value in (POSTPROCESS, POSTPROCESS / "experimental"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experimental.humanlike_vertex_placement_20260812.compare_production_vertex_budget import (  # noqa: E402
    _production_keyframes,
)
from experimental.temporal_vertex_decimation_20260812.run_experiment import (  # noqa: E402
    load_single_component_track,
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
    }


def _read_exact(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "iou": _distribution([float(row["iou"]) for row in rows]),
        "recall": _distribution([float(row["recall"]) for row in rows]),
        "recall_violations": int(
            sum(float(row["recall"]) + 1e-12 < 0.97 for row in rows)
        ),
    }


def _frame_metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    polygons = [
        np.asarray(value, dtype=np.float32)
        for value in (reference, prediction)
        if len(value) >= 3
    ]
    points = np.concatenate(polygons, axis=0)
    minimum = np.floor(points.min(axis=0)).astype(np.int32)
    maximum = np.ceil(points.max(axis=0)).astype(np.int32)
    shape = (
        int(maximum[1] - minimum[1] + 1),
        int(maximum[0] - minimum[0] + 1),
    )
    masks = []
    for polygon in (reference, prediction):
        mask = np.zeros(shape, dtype=np.uint8)
        vertices = np.round(
            np.asarray(polygon, dtype=np.float32) - minimum[None, :]
        ).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 1)
        masks.append(mask)
    gt_area = int(masks[0].sum())
    pred_area = int(masks[1].sum())
    intersection = int((masks[0] & masks[1]).sum())
    union = gt_area + pred_area - intersection
    return (
        float(intersection / union) if union else 1.0,
        float(intersection / gt_area) if gt_area else 1.0,
    )


def _exact_keyframe_metrics(
    frames: list[int], references: list[np.ndarray], keyframes: dict[int, np.ndarray]
) -> dict[str, object]:
    lookup = {int(frame): index for index, frame in enumerate(frames)}
    values = [
        _frame_metrics(references[lookup[frame]], polygon)
        for frame, polygon in sorted(keyframes.items())
        if frame in lookup
    ]
    ious = [value[0] for value in values]
    recalls = [value[1] for value in values]
    return {
        "iou": _distribution(ious),
        "recall": _distribution(recalls),
        "recall_violations": int(sum(value + 1e-12 < 0.97 for value in recalls)),
    }


def _exact_interpolated_metrics(
    frames: list[int],
    references: list[np.ndarray],
    keyframes: dict[int, np.ndarray],
    *,
    first_frame: int,
    last_frame: int,
) -> dict[str, object]:
    lookup = {int(frame): index for index, frame in enumerate(frames)}
    ordered = sorted(keyframes.items())
    ious: list[float] = []
    recalls: list[float] = []
    position = 0
    for frame in range(int(first_frame), int(last_frame) + 1):
        if frame not in lookup:
            continue
        while position + 1 < len(ordered) and frame > ordered[position + 1][0]:
            position += 1
        if frame <= ordered[0][0]:
            polygon = ordered[0][1]
        elif frame >= ordered[-1][0]:
            polygon = ordered[-1][1]
        else:
            left_frame, left = ordered[position]
            right_frame, right = ordered[position + 1]
            alpha = (frame - left_frame) / max(right_frame - left_frame, 1)
            polygon = (1.0 - alpha) * left + alpha * right
        iou, recall = _frame_metrics(references[lookup[frame]], polygon)
        ious.append(iou)
        recalls.append(recall)
    return {
        "iou": _distribution(ious),
        "recall": _distribution(recalls),
        "recall_violations": int(sum(value + 1e-12 < 0.97 for value in recalls)),
    }


def _new_keyframes(path: Path, track: str) -> dict[int, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for row in rows:
        if str(row["track_id"]) != str(track):
            continue
        polygons = row["polygons"]
        if len(polygons) != 1:
            raise ValueError(f"track {track} replay unexpectedly has {len(polygons)} slots")
        output[int(row["frame"])] = np.asarray(polygons[0], dtype=np.float64)
    return output


def _command(source: Path, output: Path, target_ratio: float, vertices: int) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME),
        "__onefile_polygon_optimize",
        "--input-sqlite", str(source),
        "--output-dir", str(output),
        "--target-ratio", str(target_ratio),
        "--anchors-per-contour", str(vertices),
        "--point-predictor-model-dir", str(POSTPROCESS / "models/polygon_point_predictor"),
        "--predictor-device", "cpu",
        "--predictor-batch-size", "256",
        "--adaptive-point-quantile", "0.95",
        "--adaptive-point-offset", "0",
        "--min-anchors-per-contour", str(vertices),
        "--gapfill-max-gap", "15",
        "--max-run-frames", "30000",
        "--run-overlap-frames", "900",
        "--recall-min", "0.97",
        "--max-gap", "30",
        "--num-workers", "1",
        "--stream-sqlite-rows",
        "--evaluate-exact",
        "--write-pred-sqlite",
        "--gapfill-enabled",
    ]


def _environment(label: str, interval: float, vertices: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MASK_PIPELINE_PHASE2_CANDIDATES": "new_production_v1",
            "MASK_PIPELINE_PHASE2_LABEL": label,
            "MASK_PIPELINE_PHASE2_TARGET_INTERVAL": str(interval),
            "MASK_PIPELINE_PHASE2_PAIR_VOTE": "1",
            "MASK_PIPELINE_PHASE2_PAIR_VOTE_CONSTRAINED": "1",
            "MASK_PIPELINE_PHASE2_PAIR_VOTE_PER_KEY": "1",
            "MASK_PIPELINE_PHASE2_PAIR_VOTE_SWEEPS": "2",
            "MASK_PIPELINE_NEW_PRODUCTION_FAST_PAIR_VOTE": "1",
            "MASK_PIPELINE_NEW_PRODUCTION_PAIR_VOTE_THREADS": "2",
            "MASK_PIPELINE_PERSISTENT_LINE_FIT_BASE": "1",
            "MASK_PIPELINE_PERSISTENT_LINE_FIT_VERTICES": str(vertices),
            "MASK_PIPELINE_PHASE1_NATIVE_EXACT": "1",
            "MASK_PIPELINE_PHASE1_NATIVE_INTERVAL": "1",
            "MASK_PIPELINE_PHASE2_NATIVE_BATCH": "1",
            "MASK_PIPELINE_PHASE2_NATIVE_BATCH_THREADS": "8",
            "MASK_PIPELINE_PHASE2_NATIVE_DP": "1",
            "MASK_PIPELINE_PHASE2_GC_INTERVAL": "8",
            "MASK_PIPELINE_PHASE2_CUDA_SHAPE": "1",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER": "1",
            "MASK_PIPELINE_PHASE2_CUDA_PREFILTER_BUDGET": "0.10",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_EXACT": "1",
            "MASK_PIPELINE_PHASE2_CUDA_LAZY_MIN_RETAINED_RATIO": "0",
            "MASK_PIPELINE_PHASE2_OPENCV_THREADS": "8",
        }
    )
    python_paths = (
        "/home/kenshin/.local/share/mask-pipeline-cuda-experiment",
        str(POSTPROCESS / "experimental/0809/native_interval/build"),
        str(POSTPROCESS),
        str(ROOT / "overlay/src"),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output/humanlike_vertex_placement_20260812/keyframe_replay_vs_production",
    )
    parser.add_argument("--vertices", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with sqlite3.connect(PRODUCTION) as production_connection:
        for track, label, source in CASES:
            frames, _track, _label, references, _cuts = load_single_component_track(
                source, track, -1
            )
            frame_set = set(frames)
            old_keys = {
                frame: polygon
                for frame, polygon in _production_keyframes(
                    production_connection, track
                ).items()
                if frame in frame_set
            }
            target_ratio = len(old_keys) / len(frames)
            target_interval = len(frames) / len(old_keys)
            output = root / f"track{track}"
            required = output / "phase2_audit.json"
            started = time.perf_counter()
            if args.force or not required.is_file():
                output.mkdir(parents=True, exist_ok=True)
                with (output / "run.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        _command(source, output, target_ratio, args.vertices),
                        cwd=ROOT,
                        env=_environment(label, target_interval, args.vertices),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode:
                    raise RuntimeError(f"track {track} replay failed; see {output/'run.log'}")
            wall = time.perf_counter() - started
            new_keys = _new_keyframes(output / "opt/final_keyframes.json", track)
            common_first = max(min(old_keys), min(new_keys))
            common_last = min(max(old_keys), max(new_keys))
            old_dense = _exact_interpolated_metrics(
                frames,
                references,
                old_keys,
                first_frame=common_first,
                last_frame=common_last,
            )
            new_dense = _exact_interpolated_metrics(
                frames,
                references,
                new_keys,
                first_frame=common_first,
                last_frame=common_last,
            )
            old_key_metrics = _exact_keyframe_metrics(frames, references, old_keys)
            new_key_metrics = _exact_keyframe_metrics(frames, references, new_keys)
            audit = json.loads(required.read_text(encoding="utf-8"))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "track": track,
                    "label": label,
                    "frames": len(frames),
                    "target_old_keys": len(old_keys),
                    "new_keys": len(new_keys),
                    "old_interval": target_interval,
                    "new_interval": len(frames) / len(new_keys),
                    "common_evaluated_frames": int(common_last - common_first + 1),
                    "old_vertices": sorted({len(value) for value in old_keys.values()}),
                    "new_vertices": sorted({len(value) for value in new_keys.values()}),
                    "old_keyframes": old_key_metrics,
                    "new_keyframes": new_key_metrics,
                    "old_dense": old_dense,
                    "new_dense": new_dense,
                    "new_exact_all_frames": _read_exact(
                        output / "exact/keyframe_exact_metrics.csv"
                    ),
                    "new_audit": {
                        "minimum_recall": audit["minimum_recall"],
                        "recall_violations": audit["feasible_exact_violations"],
                        "final_exact_recall_valid": bool(
                            audit["minimum_recall"] + 1e-12 >= 0.97
                        ),
                        # Lazy exactification may leave the rejected CUDA
                        # path's diagnostic objective marked infeasible.  It
                        # is not the final exact artifact's quality verdict.
                        "approximate_path_metadata_infeasible_streams": audit[
                            "infeasible_streams"
                        ],
                    },
                    "optimizer_seconds": summary["optimizer_summary"]["optimizer_seconds"],
                    "fresh_wall_seconds": wall,
                }
            )
            print(
                f"track={track} keys={len(new_keys)}/{len(old_keys)} "
                f"old_iou={old_dense['iou']['mean']:.6f} "
                f"new_iou={new_dense['iou']['mean']:.6f} "
                f"min_recall={audit['minimum_recall']:.6f}",
                flush=True,
            )
    total_frames = sum(row["frames"] for row in rows)
    common_frames = sum(row["common_evaluated_frames"] for row in rows)
    weighted = lambda path: sum(
        row["common_evaluated_frames"] * row[path[0]][path[1]]["mean"]
        for row in rows
    ) / common_frames
    report = {
        "schema_version": 1,
        "privacy": "SQLite polygon geometry only; no source video was opened",
        "comparison": "same tracks; old Production key count used as replay soft target",
        "recall_reference": "tracked source mask in the input SQLite for both stages",
        "vertices": args.vertices,
        "tracks": rows,
        "aggregate": {
            "frames": total_frames,
            "common_evaluated_frames": common_frames,
            "old_keys": sum(row["target_old_keys"] for row in rows),
            "new_keys": sum(row["new_keys"] for row in rows),
            "old_dense_weighted_mean_iou": weighted(("old_dense", "iou")),
            "new_dense_weighted_mean_iou": weighted(("new_dense", "iou")),
            "old_dense_minimum_recall": min(
                row["old_dense"]["recall"]["minimum"] for row in rows
            ),
            "new_dense_minimum_recall": min(
                row["new_dense"]["recall"]["minimum"] for row in rows
            ),
            "new_recall_violations": sum(
                row["new_exact_all_frames"]["recall_violations"] for row in rows
            ),
            "optimizer_seconds": sum(row["optimizer_seconds"] for row in rows),
        },
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
