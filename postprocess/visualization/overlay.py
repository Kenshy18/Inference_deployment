"""Simple overlay renderer depending only on public SQLite artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from contracts.mask_sqlite import read_mask_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt-sqlite", type=Path, required=True)
    parser.add_argument("--pred-sqlite", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--mask-alpha", type=float, default=0.28)
    parser.add_argument("--gt-thickness", type=int, default=2)
    parser.add_argument("--pred-thickness", type=int, default=3)
    return parser


def _by_frame(path: Path) -> dict[int, list[list[np.ndarray]]]:
    rows: dict[int, list[list[np.ndarray]]] = defaultdict(list)
    for row in read_mask_rows(path):
        polygons = [
            np.round(np.asarray(polygon)).astype(np.int32).reshape(-1, 1, 2)
            for polygon in json.loads(row.polygons)
        ]
        rows[row.frame].append(polygons)
    return rows


def render_main() -> None:
    args = _parser().parse_args()
    ground_truth = _by_frame(args.gt_sqlite)
    predictions = _by_frame(args.pred_sqlite)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create video: {args.output_video}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tint = frame.copy()
            for instance in predictions.get(frame_index, []):
                cv2.fillPoly(tint, instance, (0, 215, 255))
                cv2.polylines(
                    frame,
                    instance,
                    True,
                    (0, 215, 255),
                    args.pred_thickness,
                )
            frame = cv2.addWeighted(
                tint, args.mask_alpha, frame, 1.0 - args.mask_alpha, 0.0
            )
            for instance in ground_truth.get(frame_index, []):
                cv2.polylines(
                    frame,
                    instance,
                    True,
                    (255, 255, 255),
                    args.gt_thickness,
                )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
