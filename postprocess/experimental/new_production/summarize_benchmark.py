#!/usr/bin/env python3
"""Summarize reference/optimized full-data runs and 20-minute projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_FRAMES = 23_510
SOURCE_FPS = 30_000 / 1_001
TARGET_SECONDS = 20 * 60


def _load(root: Path, engine: str, interval: int) -> dict[str, object]:
    matrix = json.loads(
        (root / engine / f"interval_{interval}" / "phase2_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = matrix["completed_profiles"][-1]
    rows = matrix["rows"]
    # Class jobs run concurrently. Their maximum wall time is the stable
    # processing wall independent of coordinator/report-writing jitter.
    wall = max(float(row["wall_seconds"]) for row in rows)
    return {
        "engine": engine,
        "interval": interval,
        "wall_seconds": wall,
        "video_fps": VIDEO_FRAMES / wall,
        "estimated_20min_seconds": (TARGET_SECONDS * SOURCE_FPS) / (VIDEO_FRAMES / wall),
        "actual_mean_interval": float(aggregate["actual_mean_interval"]),
        "keyframes": int(aggregate["keyframes"]),
        "iou_mean": float(aggregate["iou_mean"]),
        "iou_min": min(float(row["iou_min"]) for row in rows),
        "iou_q01_by_class_min": float(aggregate["iou_q01_by_class_min"]),
        "iou_q05_by_class_min": min(float(row["iou_q05"]) for row in rows),
        "recall_min": float(aggregate["recall_min"]),
        "recall_violations": int(aggregate["recall_violations"]),
        "class_rows": [
            {
                "label": row["label"],
                "wall_seconds": float(row["wall_seconds"]),
                "pair_vote_seconds": float(row["pair_vote_seconds"]),
            }
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for interval in (1, 3, 6):
        reference = _load(args.root, "reference", interval)
        optimized = _load(args.root, "optimized", interval)
        parity = json.loads(
            (args.root / f"parity_interval_{interval}.json").read_text(
                encoding="utf-8"
            )
        )["all_equal"]
        results.append(
            {
                "interval": interval,
                "reference": reference,
                "optimized": optimized,
                "speedup_percent": 100.0
                * (float(reference["wall_seconds"]) - float(optimized["wall_seconds"]))
                / float(reference["wall_seconds"]),
                "artifacts_byte_identical": bool(parity),
            }
        )
    payload = {
        "schema_version": 1,
        "profile": "new_production_v1",
        "source_video_frames": VIDEO_FRAMES,
        "source_fps_for_projection": SOURCE_FPS,
        "scope": "polygon keyframe optimization only",
        "privacy": "SQLite polygon geometry only; video pixels were not opened.",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# new_production benchmark (2026-08-12)",
        "",
        "SQLite polygon geometry only; no video frame was decoded.",
        "",
        "| target | actual | keys | mean IoU | min / q01 / q05 IoU | ref | optimized | speedup | 20 min estimate | byte parity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in results:
        ref = item["reference"]
        opt = item["optimized"]
        estimate = float(opt["estimated_20min_seconds"])
        lines.append(
            "| {interval} | {actual:.3f} | {keys:,} | {iou:.6f} | "
            "{imin:.6f} / {q01:.6f} / {q05:.6f} | {ref:.2f}s | "
            "{opt:.2f}s | {speedup:.1f}% | {minutes:d}:{seconds:02d} | {parity} |".format(
                interval=item["interval"],
                actual=float(opt["actual_mean_interval"]),
                keys=int(opt["keyframes"]),
                iou=float(opt["iou_mean"]),
                imin=float(opt["iou_min"]),
                q01=float(opt["iou_q01_by_class_min"]),
                q05=float(opt["iou_q05_by_class_min"]),
                ref=float(ref["wall_seconds"]),
                opt=float(opt["wall_seconds"]),
                speedup=float(item["speedup_percent"]),
                minutes=int(estimate // 60),
                seconds=int(round(estimate % 60)),
                parity="yes" if item["artifacts_byte_identical"] else "NO",
            )
        )
    lines.extend(
        [
            "",
            "The 20-minute estimate assumes the same 30000/1001 fps and mask density.",
            "It covers this polygon keyframe optimization stage, not inference or overlay.",
            "The exact CPU audit still sees the frozen baseline's known CUDA/OpenCV boundary cases; the optimized engine neither adds nor removes them.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
