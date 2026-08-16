#!/usr/bin/env python3
"""Render SQLite-only mask comparisons without decoding private video frames."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--validated-sqlite", type=Path, required=True)
    parser.add_argument("--cuda-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--focus-frame", type=int, required=True)
    return parser.parse_args()


def load(path: Path, track_id: str, start: int, end: int) -> dict[int, list[np.ndarray]]:
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT frame, polygons FROM masks "
            "WHERE CAST(track_id AS TEXT)=? AND frame BETWEEN ? AND ? ORDER BY frame",
            (str(track_id), int(start), int(end)),
        ).fetchall()
    return {
        int(frame): [np.asarray(polygon, dtype=np.float32) for polygon in json.loads(value)]
        for frame, value in rows
    }


def raster_canonical(
    polygons: list[np.ndarray], origin: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize after rounding in the global coordinate system.

    Rounding before the integer ROI translation makes the raw-mask raster
    invariant to the bounds of the prediction it is compared with.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    integer_origin = origin.astype(np.int32)
    for polygon in polygons:
        points = np.rint(polygon).astype(np.int32) - integer_origin
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def raster_legacy(
    polygons: list[np.ndarray], origin: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Reproduce the former prediction-dependent local-coordinate rounding."""
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.rint(polygon - origin).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def metrics(raw: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(raw & pred))
    raw_area = int(np.count_nonzero(raw))
    pred_area = int(np.count_nonzero(pred))
    union = raw_area + pred_area - intersection
    return {
        "raw_area": raw_area,
        "pred_area": pred_area,
        "intersection": intersection,
        "recall": intersection / raw_area if raw_area else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def colorize(mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.40) -> np.ndarray:
    image = np.full((*mask.shape, 3), 24, dtype=np.uint8)
    image[mask > 0] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(image, alpha, np.full_like(image, 24), 1.0 - alpha, 0)


def panel(
    title: str,
    raw: np.ndarray,
    pred: np.ndarray | None,
    *,
    scale: int,
    mode: str,
) -> np.ndarray:
    if mode == "raw":
        body = colorize(raw, (255, 210, 0), 0.72)
        stat = {"raw_area": int(np.count_nonzero(raw))}
    elif mode == "prediction":
        assert pred is not None
        body = colorize(pred, (180, 40, 235), 0.68)
        raw_contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(body, raw_contours, -1, (255, 220, 0), 1, cv2.LINE_AA)
        stat = metrics(raw, pred)
    else:
        assert pred is not None
        body = np.full((*raw.shape, 3), 24, dtype=np.uint8)
        overlap = (raw > 0) & (pred > 0)
        missed = (raw > 0) & (pred == 0)
        excess = (raw == 0) & (pred > 0)
        body[overlap] = (60, 190, 60)
        body[missed] = (30, 30, 245)
        body[excess] = (220, 80, 220)
        stat = metrics(raw, pred)
    body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    header = np.full((92, body.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    if pred is None:
        line = f"raw area={stat['raw_area']} px"
    else:
        line = (
            f"recall={stat['recall']:.6f}  IoU={stat['iou']:.6f}  "
            f"raw={stat['raw_area']:.0f} pred={stat['pred_area']:.0f}"
        )
    cv2.putText(header, line, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def render_frame(
    frame: int,
    track_id: str,
    raw_polygons: list[np.ndarray],
    validated_polygons: list[np.ndarray],
    cuda_polygons: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    all_points = np.concatenate([*raw_polygons, *validated_polygons, *cuda_polygons], axis=0)
    globally_rounded = np.rint(all_points).astype(np.int32)
    minimum = np.min(globally_rounded, axis=0) - 10
    maximum = np.max(globally_rounded, axis=0) + 10
    width, height = np.maximum(maximum - minimum + 1, 1)
    raw = raster_canonical(raw_polygons, minimum, (int(height), int(width)))
    validated = raster_canonical(validated_polygons, minimum, raw.shape)
    cuda = raster_canonical(cuda_polygons, minimum, raw.shape)
    scale = max(4, min(12, 420 // max(raw.shape)))
    panels = [
        panel("Raw AI mask", raw, None, scale=scale, mode="raw"),
        panel("CUDA + C++ validation", raw, validated, scale=scale, mode="prediction"),
        panel("CUDA only", raw, cuda, scale=scale, mode="prediction"),
        panel("CUDA-only difference", raw, cuda, scale=scale, mode="difference"),
    ]
    max_height = max(value.shape[0] for value in panels)
    padded = [
        cv2.copyMakeBorder(value, 0, max_height - value.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(16, 16, 16))
        for value in panels
    ]
    slide = np.concatenate(padded, axis=1)
    footer = np.full((54, slide.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        footer,
        f"frame={frame} track={track_id} | cyan=raw boundary | difference: green=overlap red=missed magenta=excess",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([slide, footer], axis=0), {
        "frame": int(frame),
        **{f"validated_{key}": value for key, value in metrics(raw, validated).items()},
        **{f"cuda_{key}": value for key, value in metrics(raw, cuda).items()},
    }


def pair_rasters(
    raw_polygons: list[np.ndarray], pred_polygons: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the prediction-dependent local ROI used by exact_metrics."""
    all_points = np.concatenate([*raw_polygons, *pred_polygons], axis=0)
    minimum = np.floor(np.min(all_points, axis=0)).astype(np.int32)
    maximum = np.ceil(np.max(all_points, axis=0)).astype(np.int32)
    width, height = np.maximum(maximum - minimum + 1, 1)
    shape = (int(height), int(width))
    return raster_legacy(raw_polygons, minimum, shape), raster_legacy(pred_polygons, minimum, shape)


def render_metric_grid_frame(
    frame: int,
    raw_polygons: list[np.ndarray],
    validated_polygons: list[np.ndarray],
    cuda_polygons: list[np.ndarray],
) -> np.ndarray:
    validated_raw, validated = pair_rasters(raw_polygons, validated_polygons)
    cuda_raw, cuda = pair_rasters(raw_polygons, cuda_polygons)
    panels = [
        panel("Validated evaluator raw grid", validated_raw, None, scale=10, mode="raw"),
        panel("Validated evaluator difference", validated_raw, validated, scale=10, mode="difference"),
        panel("CUDA-only evaluator raw grid", cuda_raw, None, scale=10, mode="raw"),
        panel("CUDA-only evaluator difference", cuda_raw, cuda, scale=10, mode="difference"),
    ]
    max_height = max(value.shape[0] for value in panels)
    padded = [
        cv2.copyMakeBorder(value, 0, max_height - value.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(16, 16, 16))
        for value in panels
    ]
    slide = np.concatenate(padded, axis=1)
    footer = np.full((58, slide.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        footer,
        f"frame={frame} | each pair uses its own exact-metric ROI; raw raster area may change with prediction bounds",
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([slide, footer], axis=0)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load(args.raw_sqlite, args.track_id, args.start_frame, args.end_frame)
    validated = load(args.validated_sqlite, args.track_id, args.start_frame, args.end_frame)
    cuda = load(args.cuda_sqlite, args.track_id, args.start_frame, args.end_frame)
    frames = sorted(set(raw) & set(validated) & set(cuda))
    if not frames:
        raise RuntimeError("no common mask frames")
    image_dir = args.output_dir / "frames"
    image_dir.mkdir(exist_ok=True)
    slides: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for frame in frames:
        slide, row = render_frame(
            frame, str(args.track_id), raw[frame], validated[frame], cuda[frame]
        )
        path = image_dir / f"frame_{frame:06d}.png"
        if not cv2.imwrite(str(path), slide):
            raise RuntimeError(f"failed to write {path}")
        row["image"] = str(path.resolve())
        rows.append(row)
        slides.append(slide)
        if frame == args.focus_frame:
            cv2.imwrite(str(args.output_dir / f"focus_frame_{frame}.png"), slide)
            metric_grid = render_metric_grid_frame(
                frame, raw[frame], validated[frame], cuda[frame]
            )
            cv2.imwrite(
                str(args.output_dir / f"focus_frame_{frame}_metric_grid.png"),
                metric_grid,
            )
    height = max(image.shape[0] for image in slides)
    width = max(image.shape[1] for image in slides)
    video_path = args.output_dir / "comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
    for image in slides:
        writer.write(cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(16, 16, 16)))
    writer.release()
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    report = {
        "privacy": "SQLite polygon geometry only; no video frame was decoded.",
        "frames": frames,
        "focus_frame": int(args.focus_frame),
        "focus_image": str((args.output_dir / f"focus_frame_{args.focus_frame}.png").resolve()),
        "focus_metric_grid_image": str(
            (args.output_dir / f"focus_frame_{args.focus_frame}_metric_grid.png").resolve()
        ),
        "sequence_video": str(video_path.resolve()),
        "metrics_csv": str((args.output_dir / "metrics.csv").resolve()),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
