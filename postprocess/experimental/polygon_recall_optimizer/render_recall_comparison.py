#!/usr/bin/env python3
"""Render local-only old/new recall comparison frames.

The script intentionally performs all video decoding and drawing locally.  It
never uploads pixels and does not require an interactive image viewer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .fixed_budget import evaluate_segments, load_raw_masks, load_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--old-sqlite", type=Path, required=True)
    parser.add_argument("--new-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument(
        "--new-range-max-recall",
        type=float,
        help=(
            "When set, near-floor samples are restricted to "
            "recall_floor <= Recall <= this value"
        ),
    )
    parser.add_argument(
        "--stratify-new-range",
        action="store_true",
        help="Sample evenly across the configured new-Recall range",
    )
    parser.add_argument("--low-count", type=int, default=20)
    parser.add_argument("--floor-count", type=int, default=10)
    parser.add_argument("--minimum-frame-gap", type=int, default=15)
    parser.add_argument("--hold-seconds", type=float, default=1.25)
    return parser.parse_args()


def _select_diverse(rows, count: int, minimum_gap: int):
    if count <= 0:
        return []
    selected = []
    for row in rows:
        if all(abs(row.frame - existing.frame) >= minimum_gap for existing in selected):
            selected.append(row)
            if len(selected) == count:
                return selected
    if len(selected) < count:
        for row in rows:
            if row not in selected:
                selected.append(row)
                if len(selected) == count:
                    break
    return selected


def _select_stratified_range(
    rows,
    count: int,
    minimum_gap: int,
    lower: float,
    upper: float,
):
    """Select samples close to evenly spaced Recall targets across a range."""

    if count <= 0:
        return []
    available = list(rows)
    selected = []
    targets = [
        lower + (upper - lower) * (index + 0.5) / count for index in range(count)
    ]
    for target in targets:
        diverse = [
            item
            for item in available
            if all(
                abs(item.frame - existing.frame) >= minimum_gap for existing in selected
            )
        ]
        candidates = diverse or available
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda item: (
                abs(item.recall - target),
                item.frame,
                item.track_id,
            ),
        )
        selected.append(chosen)
        available.remove(chosen)
    return selected


def _polygons(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for part in geometry.geoms if isinstance(part, Polygon)]
    return []


def _contours(geometry) -> list[np.ndarray]:
    output = []
    for polygon in _polygons(geometry):
        points = np.rint(np.asarray(polygon.exterior.coords[:-1])).astype(np.int32)
        if len(points) >= 3:
            output.append(points.reshape(-1, 1, 2))
    return output


def _draw_geometry(
    image: np.ndarray,
    geometry,
    *,
    fill_color: tuple[int, int, int] | None,
    alpha: float,
    outline_color: tuple[int, int, int] | None,
    thickness: int,
) -> None:
    contours = _contours(geometry)
    if not contours:
        return
    if fill_color is not None:
        layer = image.copy()
        cv2.fillPoly(layer, contours, fill_color)
        cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, image)
    if outline_color is not None:
        cv2.polylines(image, contours, True, outline_color, thickness, cv2.LINE_AA)


def _timecode(frame: int, fps: float) -> str:
    rounded_fps = max(1, round(fps))
    total_seconds, frame_part = divmod(frame, rounded_fps)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame_part:02d}"


def _panel(
    source: np.ndarray,
    evaluation,
    *,
    title: str,
    fps: float,
    category: str,
) -> np.ndarray:
    image = source.copy()
    # Final reconstructed mask.
    _draw_geometry(
        image,
        evaluation.predicted_geometry,
        fill_color=(180, 50, 230),
        alpha=0.42,
        outline_color=(255, 180, 255),
        thickness=3,
    )
    # Raw AI pixels missed by the reconstructed mask are the privacy-critical
    # part of this comparison.
    missed = evaluation.raw_geometry.difference(evaluation.predicted_geometry)
    _draw_geometry(
        image,
        missed,
        fill_color=(20, 20, 245),
        alpha=0.78,
        outline_color=(20, 20, 255),
        thickness=2,
    )
    # Raw AI mask boundary.
    _draw_geometry(
        image,
        evaluation.raw_geometry,
        fill_color=None,
        alpha=0.0,
        outline_color=(255, 255, 0),
        thickness=3,
    )
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (960, 78), (10, 10, 10), -1)
    first = (
        f"{title}  frame={evaluation.frame}  "
        f"TC={_timecode(evaluation.frame, fps)}  track={evaluation.track_id}"
    )
    second = f"Recall={evaluation.recall:.6f}  IoU={evaluation.iou:.6f}  {category}"
    cv2.putText(
        image,
        first,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        second,
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _write_video(
    path: Path, slides: list[np.ndarray], fps: float, hold_seconds: float
) -> None:
    if not slides:
        return
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (slides[0].shape[1], slides[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    repeats = max(1, round(fps * hold_seconds))
    for slide in slides:
        for _ in range(repeats):
            writer.write(slide)
    writer.release()


def _key_stats(segments, start_frame: int, end_frame: int, raw_count: int):
    values = [
        segment for track_segments in segments.values() for segment in track_segments
    ]
    key_count = sum(
        start_frame <= keyframe.frame <= end_frame
        for segment in values
        for keyframe in segment.keyframes
    )
    total_span = sum(
        segment.keyframes[-1].frame - segment.keyframes[0].frame
        for segment in values
        if segment.keyframes
    )
    interval_count = max(key_count - len(values), 1)
    return {
        "segment_count": len(values),
        "keyframe_count": key_count,
        "key_frequency": key_count / max(raw_count, 1),
        "mean_key_interval": total_span / interval_count,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_masks(
        args.raw_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    old_segments = load_segments(
        args.old_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    new_segments = load_segments(
        args.new_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    old = evaluate_segments(raw, old_segments)
    new = evaluate_segments(raw, new_segments)
    old_by_identity = {(item.frame, item.track_id): item for item in old}
    new_by_identity = {(item.frame, item.track_id): item for item in new}

    old_candidates = sorted(
        (item for item in old if item.recall <= args.recall_floor),
        key=lambda item: (item.recall, item.frame, item.track_id),
    )
    new_candidates = sorted(
        (
            item
            for item in new
            if item.recall >= args.recall_floor
            and (
                args.new_range_max_recall is None
                or item.recall <= args.new_range_max_recall
            )
        ),
        key=lambda item: (
            abs(item.recall - args.recall_floor),
            item.frame,
            item.track_id,
        ),
    )
    groups = {}
    if args.low_count > 0:
        groups[f"old_below_recall_floor_{args.low_count}"] = _select_diverse(
            old_candidates, args.low_count, args.minimum_frame_gap
        )
    if args.floor_count > 0:
        if args.stratify_new_range:
            if args.new_range_max_recall is None:
                raise ValueError("--stratify-new-range requires --new-range-max-recall")
            selected_new = _select_stratified_range(
                new_candidates,
                args.floor_count,
                args.minimum_frame_gap,
                args.recall_floor,
                args.new_range_max_recall,
            )
        else:
            selected_new = _select_diverse(
                new_candidates, args.floor_count, args.minimum_frame_gap
            )
        groups[f"new_near_recall_floor_{args.floor_count}"] = selected_new
    if not groups:
        raise ValueError("at least one sample count must be positive")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows = []
    for group_name, selected in groups.items():
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        slides = []
        for index, chosen in enumerate(selected, start=1):
            identity = (chosen.frame, chosen.track_id)
            old_item = old_by_identity[identity]
            new_item = new_by_identity[identity]
            capture.set(cv2.CAP_PROP_POS_FRAMES, chosen.frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {chosen.frame}")
            category = (
                "selected: OLD Recall <= floor"
                if group_name.startswith("old_")
                else "selected: NEW Recall near floor"
            )
            slide = np.concatenate(
                [
                    _panel(
                        frame,
                        old_item,
                        title="OLD Production",
                        fps=fps,
                        category=category,
                    ),
                    _panel(
                        frame, new_item, title="NEW Pareto", fps=fps, category=category
                    ),
                ],
                axis=1,
            )
            legend = "Pink=final  Cyan=raw AI boundary  Red=raw area missed"
            cv2.putText(
                slide,
                legend,
                (550, 530),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = (
                group_dir
                / f"{index:03d}_frame_{chosen.frame}_track_{chosen.track_id}.png"
            )
            if not cv2.imwrite(str(path), slide):
                raise RuntimeError(f"failed to write {path}")
            slides.append(slide)
            rows.append(
                {
                    "group": group_name,
                    "sample_index": index,
                    "frame": chosen.frame,
                    "timecode": _timecode(chosen.frame, fps),
                    "track_id": chosen.track_id,
                    "old_recall": old_item.recall,
                    "new_recall": new_item.recall,
                    "old_iou": old_item.iou,
                    "new_iou": new_item.iou,
                    "image": str(path),
                }
            )
        _write_video(
            args.output_dir / f"{group_name}.mp4",
            slides,
            fps=30.0,
            hold_seconds=args.hold_seconds,
        )
    capture.release()

    with (args.output_dir / "samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "privacy": "All video decoding/rendering was local; no pixels were uploaded.",
        "video": str(args.video.resolve()),
        "raw_sqlite": str(args.raw_sqlite.resolve()),
        "old_sqlite": str(args.old_sqlite.resolve()),
        "new_sqlite": str(args.new_sqlite.resolve()),
        "recall_floor": args.recall_floor,
        "new_range_max_recall": args.new_range_max_recall,
        "stratified_new_range": args.stratify_new_range,
        "old": _key_stats(old_segments, args.start_frame, args.end_frame, len(raw)),
        "new": _key_stats(new_segments, args.start_frame, args.end_frame, len(raw)),
        "groups": {name: len(items) for name, items in groups.items()},
        "samples_csv": str((args.output_dir / "samples.csv").resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
