#!/usr/bin/env python3
"""Locally render a raw/final comparison around one temporal area jump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
)
from ..polygon_recall_optimizer.render_recall_comparison import (
    _draw_geometry,
    _timecode,
    _write_video,
)


def _safe_panel(
    source: np.ndarray,
    evaluation,
    *,
    title: str,
    fps: float,
    category: str,
) -> np.ndarray:
    """Draw metadata outside the video so top-edge masks stay visible."""

    image = source.copy()
    _draw_geometry(
        image,
        evaluation.predicted_geometry,
        fill_color=(180, 50, 230),
        alpha=0.42,
        outline_color=(255, 180, 255),
        thickness=3,
    )
    missed = evaluation.raw_geometry.difference(evaluation.predicted_geometry)
    _draw_geometry(
        image,
        missed,
        fill_color=(20, 20, 245),
        alpha=0.78,
        outline_color=(20, 20, 255),
        thickness=2,
    )
    _draw_geometry(
        image,
        evaluation.raw_geometry,
        fill_color=None,
        alpha=0.0,
        outline_color=(255, 255, 0),
        thickness=3,
    )
    resized = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    panel = np.zeros((630, 960, 3), dtype=np.uint8)
    panel[90:, :] = resized
    first = (
        f"{title}  frame={evaluation.frame}  "
        f"TC={_timecode(evaluation.frame, fps)}  track={evaluation.track_id}"
    )
    second = (
        f"Recall={evaluation.recall:.6f}  IoU={evaluation.iou:.6f}  {category}"
    )
    cv2.putText(
        panel,
        first,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        second,
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--reference-sqlite", type=Path, required=True)
    parser.add_argument("--result-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--label", default="男性器")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_masks(
        args.raw_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    reference_segments = load_segments(
        args.reference_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    result_segments = load_segments(
        args.result_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    reference = {
        (item.frame, item.track_id): item
        for item in evaluate_segments(raw, reference_segments)
        if item.track_id == args.track_id
    }
    result = {
        (item.frame, item.track_id): item
        for item in evaluate_segments(raw, result_segments)
        if item.track_id == args.track_id
    }

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    slides: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    previous_result_area: float | None = None
    for frame in range(args.start_frame, args.end_frame + 1):
        ok, source = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {frame}")
        identity = (frame, args.track_id)
        raw_mask = raw.get(identity)
        left = reference.get(identity)
        right = result.get(identity)
        if raw_mask is None or left is None or right is None:
            continue
        raw_area = float(raw_mask.geometry.area)
        reference_area = float(left.predicted_geometry.area)
        result_area = float(right.predicted_geometry.area)
        growth = (
            None
            if previous_result_area is None
            else result_area / max(previous_result_area, 1e-9)
        )
        category_left = (
            f"raw={raw_area:.0f} final={reference_area:.0f} "
            f"final/raw={reference_area / max(raw_area, 1e-9):.2f}x"
        )
        category_right = (
            f"raw={raw_area:.0f} final={result_area:.0f} "
            f"final/raw={result_area / max(raw_area, 1e-9):.2f}x "
            f"growth={'-' if growth is None else f'{growth:.2f}x'}"
        )
        slide = np.concatenate(
            [
                _safe_panel(
                    source,
                    left,
                    title="New Pareto reference",
                    fps=fps,
                    category=category_left,
                ),
                _safe_panel(
                    source,
                    right,
                    title="Temporal7 + alternating DP2",
                    fps=fps,
                    category=category_right,
                ),
            ],
            axis=1,
        )
        legend = np.zeros((40, slide.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            legend,
            "Pink=final mask  Cyan=raw AI boundary  Red=raw pixels missed",
            (505, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        slide = np.concatenate([slide, legend], axis=0)
        image_path = image_dir / f"frame_{frame:06d}.png"
        if not cv2.imwrite(str(image_path), slide):
            raise RuntimeError(f"failed to write {image_path}")
        slides.append(slide)
        records.append(
            {
                "frame": frame,
                "track_id": args.track_id,
                "raw_area": raw_area,
                "reference_area": reference_area,
                "result_area": result_area,
                "result_growth_ratio": growth,
                "reference_recall": left.recall,
                "reference_iou": left.iou,
                "result_recall": right.recall,
                "result_iou": right.iou,
                "image": str(image_path.resolve()),
            }
        )
        previous_result_area = result_area
    capture.release()
    if not slides:
        raise RuntimeError("no matching frames were rendered")

    actual_path = args.output_dir / "area_jump_actual_speed.mp4"
    writer = cv2.VideoWriter(
        str(actual_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (slides[0].shape[1], slides[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {actual_path}")
    for slide in slides:
        writer.write(slide)
    writer.release()
    slow_path = args.output_dir / "area_jump_slow.mp4"
    _write_video(slow_path, slides, fps=30.0, hold_seconds=0.45)

    chosen = slides[:4]
    contact_path = args.output_dir / "area_jump_contact_sheet.png"
    if len(chosen) == 4:
        contact = np.concatenate(
            [
                np.concatenate(chosen[:2], axis=1),
                np.concatenate(chosen[2:], axis=1),
            ],
            axis=0,
        )
        if not cv2.imwrite(str(contact_path), contact):
            raise RuntimeError(f"failed to write {contact_path}")

    report = {
        "privacy": "Video decoding/rendering stayed local; no pixels were uploaded.",
        "frame_range": [args.start_frame, args.end_frame],
        "track_id": args.track_id,
        "source_fps": fps,
        "actual_speed_video": str(actual_path.resolve()),
        "slow_video": str(slow_path.resolve()),
        "contact_sheet": str(contact_path.resolve()),
        "frames": str(image_dir.resolve()),
        "records": records,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
