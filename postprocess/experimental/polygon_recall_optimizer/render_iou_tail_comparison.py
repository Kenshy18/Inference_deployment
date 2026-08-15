#!/usr/bin/env python3
"""Render the lowest-IoU tail of two local polygon SQLite results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .fixed_budget import evaluate_segments, load_raw_masks, load_segments
from .render_recall_comparison import _key_stats, _panel, _timecode, _write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--production-sqlite", type=Path, required=True)
    parser.add_argument("--guarded-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--hold-seconds", type=float, default=0.75)
    parser.add_argument("--production-title", default="Production interval 10")
    parser.add_argument(
        "--guarded-title", default="Production + Recall 0.97 guard"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("tail-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.raw_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    production_segments = load_segments(
        args.production_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    guarded_segments = load_segments(
        args.guarded_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    production = evaluate_segments(raw, production_segments)
    guarded = evaluate_segments(raw, guarded_segments)
    production_by_identity = {(item.frame, item.track_id): item for item in production}
    guarded_by_identity = {(item.frame, item.track_id): item for item in guarded}
    count = max(1, int(math.ceil(len(guarded) * args.tail_fraction)))
    selected = sorted(
        guarded,
        key=lambda item: (item.iou, item.recall, item.frame, item.track_id),
    )[:count]
    cutoff = max(item.iou for item in selected)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded: dict[int, np.ndarray] = {}
    for frame_number in sorted({item.frame for item in selected}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {frame_number}")
        decoded[frame_number] = frame
    capture.release()

    image_dir = args.output_dir / "guarded_iou_bottom_1pct"
    image_dir.mkdir(parents=True, exist_ok=True)
    slides: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    percentage = 100.0 * args.tail_fraction
    for rank, guarded_item in enumerate(selected, start=1):
        identity = (guarded_item.frame, guarded_item.track_id)
        production_item = production_by_identity[identity]
        source = decoded[guarded_item.frame]
        category = f"guarded IoU bottom {percentage:g}%  rank={rank}/{count}"
        slide = np.concatenate(
            [
                _panel(
                    source,
                    production_item,
                    title=args.production_title,
                    fps=fps,
                    category=category,
                ),
                _panel(
                    source,
                    guarded_item,
                    title=args.guarded_title,
                    fps=fps,
                    category=category,
                ),
            ],
            axis=1,
        )
        cv2.putText(
            slide,
            "Pink=final  Cyan=raw AI boundary  Red=raw area missed",
            (550, 530),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        image_path = (
            image_dir / f"{rank:03d}_iou_{guarded_item.iou:.6f}_frame_"
            f"{guarded_item.frame}_track_{guarded_item.track_id}.png"
        )
        if not cv2.imwrite(str(image_path), slide):
            raise RuntimeError(f"failed to write {image_path}")
        slides.append(slide)
        rows.append(
            {
                "rank": rank,
                "frame": guarded_item.frame,
                "timecode": _timecode(guarded_item.frame, fps),
                "track_id": guarded_item.track_id,
                "production_is_keyframe": int(production_item.is_keyframe),
                "guarded_is_keyframe": int(guarded_item.is_keyframe),
                "production_recall": production_item.recall,
                "guarded_recall": guarded_item.recall,
                "production_iou": production_item.iou,
                "guarded_iou": guarded_item.iou,
                "production_precision": production_item.precision,
                "guarded_precision": guarded_item.precision,
                "image": str(image_path.resolve()),
            }
        )

    video_path = args.output_dir / "guarded_iou_bottom_1pct_comparison.mp4"
    _write_video(video_path, slides, fps=30.0, hold_seconds=args.hold_seconds)
    csv_path = args.output_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "privacy": "All video decoding/rendering was local; no pixels were uploaded.",
        "selection": "lowest guarded raw-observation IoU",
        "panel_titles": {
            "left": args.production_title,
            "right": args.guarded_title,
        },
        "tail_fraction": args.tail_fraction,
        "evaluated_mask_instances": len(guarded),
        "sample_count": count,
        "guarded_iou_min": min(item.iou for item in selected),
        "guarded_iou_cutoff": cutoff,
        "video": str(video_path.resolve()),
        "samples_csv": str(csv_path.resolve()),
        "images": str(image_dir.resolve()),
        "production": _key_stats(
            production_segments, args.start_frame, args.end_frame, len(raw)
        ),
        "guarded": _key_stats(
            guarded_segments, args.start_frame, args.end_frame, len(raw)
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
