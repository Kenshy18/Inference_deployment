#!/usr/bin/env python3
"""Render the KPI nested-duplicate regression at frames 4275..4280.

The source video is decoded only by local OpenCV.  No image is uploaded or
opened through an AI image tool.  Every output has three panels:

* left: scored V3 AI masks before NMS;
* middle: legacy Production NMS;
* right: component-aware Mask-IoU candidate v2.

Full-frame and common-ROI crops are emitted so the temporal context and the
nested mask boundaries can both be reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import iter_detection_records  # noqa: E402
from nms.adaptive import _bbox_iou  # noqa: E402
from nms.component_aware import (  # noqa: E402
    _overlap_slices,
    _raster_mask,
    exact_mask_iou,
)
from experimental.nms_ablation_20260813.render_component_candidate_v2_review_gallery import (  # noqa: E402,E501
    _draw_dashed_polygon,
    _put,
    seek_frame,
)


DEFAULT_ROOT = (
    REPOSITORY_ROOT
    / "output/nms_component_candidate_v2_fixed_downstream_kpi_corrected_20260813"
)
DEFAULT_RAW = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_ablation_20260813/inputs/"
    "v3__kpi_2025_12/scored.jsonl"
)
DEFAULT_VIDEO = REPOSITORY_ROOT / "data/新しいフォルダー/12月KPI動画.mp4"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_kpi_nested_4275_4280_20260813"
)
FRAMES = tuple(range(4275, 4281))

# Fixed role colours make the sequence easier to follow even though source IDs
# change every frame.  OpenCV uses BGR.
HIGH_SCORE_COLOR = (35, 210, 255)
LOW_SCORE_COLOR = (235, 80, 225)
OTHER_COLOR = (110, 220, 120)


def _id(detection: dict[str, Any], index: int) -> int | str:
    value = detection.get("source_detection_id")
    if value is None:
        return f"index:{index}"
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _load_frames(path: Path) -> dict[int, dict[str, Any]]:
    wanted = set(FRAMES)
    records: dict[int, dict[str, Any]] = {}
    for record in iter_detection_records(path):
        frame = int(record["frame_index"])
        if frame in wanted:
            records[frame] = record
        if frame > FRAMES[-1] or len(records) == len(wanted):
            break
    missing = sorted(wanted - set(records))
    if missing:
        raise RuntimeError(f"missing frames in {path}: {missing}")
    return records


def _role_colors(raw: dict[str, Any]) -> dict[int | str, tuple[int, int, int]]:
    indexed = list(enumerate(raw["detections"]))
    indexed.sort(key=lambda row: (-float(row[1].get("score") or 0.0), row[0]))
    result: dict[int | str, tuple[int, int, int]] = {}
    for rank, (index, detection) in enumerate(indexed):
        color = (
            HIGH_SCORE_COLOR
            if rank == 0
            else LOW_SCORE_COLOR
            if rank == 1
            else OTHER_COLOR
        )
        result[_id(detection, index)] = color
    return result


def _paint_mask(
    canvas: np.ndarray,
    detection: dict[str, Any],
    color: tuple[int, int, int],
    alpha: float = 0.30,
) -> None:
    raster = _raster_mask(detection)
    if raster is None or raster.area <= 0:
        return
    height, width = canvas.shape[:2]
    left = max(0, int(raster.left))
    top = max(0, int(raster.top))
    right = min(width - 1, int(raster.right))
    bottom = min(height - 1, int(raster.bottom))
    if right < left or bottom < top:
        return
    source = raster.mask[
        top - raster.top : bottom - raster.top + 1,
        left - raster.left : right - raster.left + 1,
    ]
    target = canvas[top : bottom + 1, left : right + 1]
    selected = source != 0
    if not np.any(selected):
        return
    foreground = np.asarray(color, dtype=np.float32)
    target[selected] = np.clip(
        target[selected].astype(np.float32) * (1.0 - alpha) + foreground * alpha,
        0,
        255,
    ).astype(np.uint8)


def _render_content(
    image: np.ndarray,
    raw: dict[str, Any],
    output: dict[str, Any],
    colors: dict[int | str, tuple[int, int, int]],
) -> np.ndarray:
    canvas = image.copy()
    raw_map = {
        _id(detection, index): detection
        for index, detection in enumerate(raw["detections"])
    }
    output_map = {
        _id(detection, index): detection
        for index, detection in enumerate(output["detections"])
    }

    for detection_id, detection in output_map.items():
        color = colors.get(detection_id, OTHER_COLOR)
        _paint_mask(canvas, detection, color)
    for detection_id, detection in output_map.items():
        color = colors.get(detection_id, OTHER_COLOR)
        for polygon in detection.get("polygons") or []:
            points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
            if len(points) >= 3:
                cv2.polylines(canvas, [points], True, color, 4, cv2.LINE_AA)
        bbox = detection.get("bbox_xyxy") or [10, 30, 0, 0]
        _put(
            canvas,
            f"D{detection_id} score={float(detection.get('score') or 0.0):.3f}",
            (max(8, int(float(bbox[0]))), max(26, int(float(bbox[1])) - 8)),
            scale=0.60,
            color=color,
        )

    for detection_id, detection in raw_map.items():
        if detection_id in output_map:
            continue
        color = colors.get(detection_id, OTHER_COLOR)
        for polygon in detection.get("polygons") or []:
            _draw_dashed_polygon(canvas, polygon, color, thickness=4, dash=11.0)
    return canvas


def _raw_bounds(
    raw: dict[str, Any], width: int, height: int
) -> tuple[int, int, int, int]:
    points: list[np.ndarray] = []
    for detection in raw["detections"]:
        for polygon in detection.get("polygons") or []:
            array = np.asarray(polygon, dtype=np.float64)
            if len(array) >= 3:
                points.append(array)
    if not points:
        return (0, 0, width, height)
    joined = np.concatenate(points, axis=0)
    minimum = joined.min(axis=0)
    maximum = joined.max(axis=0)
    object_width = max(1.0, float(maximum[0] - minimum[0]))
    object_height = max(1.0, float(maximum[1] - minimum[1]))
    padding = max(80.0, 0.65 * max(object_width, object_height))
    left = max(0, int(np.floor(minimum[0] - padding)))
    top = max(0, int(np.floor(minimum[1] - padding)))
    right = min(width, int(np.ceil(maximum[0] + padding + 1)))
    bottom = min(height, int(np.ceil(maximum[1] + padding + 1)))
    return (left, top, right, bottom)


def _resize_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    if panel_width <= 0 or image.shape[1] == panel_width:
        return image
    scale = panel_width / image.shape[1]
    target_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (panel_width, target_height), interpolation=interpolation)


def _title_panel(image: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((54, image.shape[1], 3), 14, dtype=np.uint8)
    _put(bar, title, (14, 36), scale=0.66)
    return np.vstack([bar, image])


def _pair_metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    detections = list(raw["detections"])
    if len(detections) < 2:
        return {
            "mask_iou": None,
            "small_mask_coverage": None,
            "large_to_small_area_ratio": None,
            "bbox_iou": None,
        }
    ordered = sorted(
        detections, key=lambda detection: -float(detection.get("score") or 0.0)
    )
    first = _raster_mask(ordered[0])
    second = _raster_mask(ordered[1])
    intersection = 0
    if first is not None and second is not None:
        slices = _overlap_slices(first, second)
        if slices is not None:
            first_slice, second_slice = slices
            intersection = int(
                np.count_nonzero(
                    (first.mask[first_slice] != 0) & (second.mask[second_slice] != 0)
                )
            )
    areas = [
        first.area if first is not None else 0,
        second.area if second is not None else 0,
    ]
    small = min(areas)
    large = max(areas)
    first_bbox = tuple(map(float, ordered[0].get("bbox_xyxy", [0, 0, 0, 0])))
    second_bbox = tuple(map(float, ordered[1].get("bbox_xyxy", [0, 0, 0, 0])))
    return {
        "mask_iou": float(exact_mask_iou(first, second)),
        "small_mask_coverage": float(intersection / small) if small > 0 else None,
        "large_to_small_area_ratio": float(large / small) if small > 0 else None,
        "bbox_iou": float(_bbox_iou(first_bbox, second_bbox)),
    }


def _header(
    width: int,
    frame: int,
    raw: dict[str, Any],
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    metrics: dict[str, float | None],
) -> np.ndarray:
    raw_ids = [
        _id(detection, index) for index, detection in enumerate(raw["detections"])
    ]
    legacy_ids = [
        _id(detection, index) for index, detection in enumerate(legacy["detections"])
    ]
    candidate_ids = [
        _id(detection, index) for index, detection in enumerate(candidate["detections"])
    ]
    if metrics["mask_iou"] is None:
        metric_line = "single detection in this frame; no nested pair"
    else:
        metric_line = (
            f"Mask-IoU={metrics['mask_iou']:.4f} | "
            f"small coverage={metrics['small_mask_coverage']:.2%} | "
            f"large/small area={metrics['large_to_small_area_ratio']:.3f} | "
            f"bbox-IoU={metrics['bbox_iou']:.4f}"
        )
    lines = (
        f"KPI V3 nested duplicate audit | frame={frame}",
        f"RAW IDs={raw_ids} | LEGACY kept={legacy_ids} | CANDIDATE kept={candidate_ids}",
        metric_line,
        "gold=higher-score mask | magenta=lower-score mask | solid+fill=kept | dashed=suppressed",
    )
    header = np.full((118, width, 3), 10, dtype=np.uint8)
    for index, line in enumerate(lines):
        _put(header, line, (14, 25 + index * 27), scale=0.52)
    return header


def _compose(
    contents: tuple[np.ndarray, np.ndarray, np.ndarray],
    titles: tuple[str, str, str],
    *,
    crop: tuple[int, int, int, int] | None,
    panel_width: int,
    frame: int,
    raw: dict[str, Any],
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    metrics: dict[str, float | None],
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for content, title in zip(contents, titles, strict=True):
        if crop is not None:
            left, top, right, bottom = crop
            content = content[top:bottom, left:right]
        panels.append(_title_panel(_resize_panel(content, panel_width), title))
    common_height = min(panel.shape[0] for panel in panels)
    panels = [panel[:common_height] for panel in panels]
    body = np.concatenate(panels, axis=1)
    return np.vstack(
        [_header(body.shape[1], frame, raw, legacy, candidate, metrics), body]
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"failed to write image: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--legacy-jsonl",
        type=Path,
        default=DEFAULT_ROOT / "legacy_production/nms.jsonl",
    )
    parser.add_argument(
        "--candidate-jsonl",
        type=Path,
        default=DEFAULT_ROOT / "component_mask_v2/nms.jsonl",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-panel-width", type=int, default=960)
    parser.add_argument("--roi-panel-width", type=int, default=960)
    args = parser.parse_args()

    paths = {
        "raw": args.raw_jsonl.expanduser().resolve(),
        "legacy": args.legacy_jsonl.expanduser().resolve(),
        "candidate": args.candidate_jsonl.expanduser().resolve(),
        "video": args.video.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    raw_records = _load_frames(paths["raw"])
    legacy_records = _load_frames(paths["legacy"])
    candidate_records = _load_frames(paths["candidate"])
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    manifest: list[dict[str, Any]] = []
    roi_sequence: list[np.ndarray] = []
    try:
        for frame in FRAMES:
            raw = raw_records[frame]
            legacy = legacy_records[frame]
            candidate = candidate_records[frame]
            image = seek_frame(paths["video"], frame)
            colors = _role_colors(raw)
            contents = (
                _render_content(image, raw, raw, colors),
                _render_content(image, raw, legacy, colors),
                _render_content(image, raw, candidate, colors),
            )
            titles = (
                "LEFT: V3 AI RAW (before NMS)",
                "MIDDLE: legacy Production NMS",
                "RIGHT: component candidate v2",
            )
            metrics = _pair_metrics(raw)
            crop = _raw_bounds(raw, image.shape[1], image.shape[0])
            full = _compose(
                contents,
                titles,
                crop=None,
                panel_width=int(args.full_panel_width),
                frame=frame,
                raw=raw,
                legacy=legacy,
                candidate=candidate,
                metrics=metrics,
            )
            roi = _compose(
                contents,
                titles,
                crop=crop,
                panel_width=int(args.roi_panel_width),
                frame=frame,
                raw=raw,
                legacy=legacy,
                candidate=candidate,
                metrics=metrics,
            )
            full_relative = Path("01_full_frame") / f"frame_{frame:06d}.jpg"
            roi_relative = Path("02_roi_zoom") / f"frame_{frame:06d}.jpg"
            _write_image(staging / full_relative, full)
            _write_image(staging / roi_relative, roi)
            roi_sequence.append(_resize_panel(roi, 1920))
            manifest.append(
                {
                    "frame_index": frame,
                    "raw_ids": [
                        _id(detection, index)
                        for index, detection in enumerate(raw["detections"])
                    ],
                    "legacy_ids": [
                        _id(detection, index)
                        for index, detection in enumerate(legacy["detections"])
                    ],
                    "candidate_ids": [
                        _id(detection, index)
                        for index, detection in enumerate(candidate["detections"])
                    ],
                    **metrics,
                    "roi_xyxy": list(crop),
                    "full_frame_image": str(output / full_relative),
                    "roi_zoom_image": str(output / roi_relative),
                }
            )

        sequence_width = min(image.shape[1] for image in roi_sequence)
        sequence = np.vstack(
            [
                _resize_panel(image, sequence_width)
                if image.shape[1] != sequence_width
                else image
                for image in roi_sequence
            ]
        )
        _write_image(staging / "00_sequence_roi_frames_4275_4280.jpg", sequence)

        with (staging / "manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "\n".join(
                [
                    "# KPI nested duplicate: frames 4275-4280",
                    "",
                    "Each JPEG has three panels: left is V3 AI raw masks, middle is legacy Production NMS, and right is component candidate v2.",
                    "Gold is the higher-score detection and magenta is the lower-score detection. Solid fill is retained; dashed outline is suppressed.",
                    "Frame 4279 contains only one detection and is included to show the temporal transition.",
                    "",
                    "- `00_sequence_roi_frames_4275_4280.jpg`: all six ROI comparisons in temporal order",
                    "- `01_full_frame/`: full-frame context",
                    "- `02_roi_zoom/`: enlarged common ROI",
                    "- `manifest.csv` / `manifest.json`: exact IDs and overlap metrics",
                    "",
                    "Privacy: source frames were decoded and written only by local OpenCV. No image was uploaded or opened through an AI image tool.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "frames": list(FRAMES),
                "images": len(FRAMES) * 2 + 1,
                "privacy": "local OpenCV only; no AI image viewing or upload",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
