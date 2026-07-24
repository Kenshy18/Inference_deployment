"""Render side-by-side visualization overlays for polygon vertex limits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskSample:
    frame: int
    mask_id: str
    polygons: tuple[np.ndarray, ...]

    @property
    def vertex_count(self) -> int:
        return sum(len(polygon) for polygon in self.polygons)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[100, 150, 300, 500])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=360)
    parser.add_argument(
        "--minimum-original-vertices",
        type=int,
        help="Minimum source vertices for selected samples; defaults to max(caps).",
    )
    return parser.parse_args()


def _decode_polygons(value: str) -> tuple[np.ndarray, ...]:
    output: list[np.ndarray] = []
    for polygon in json.loads(value):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            output.append(points)
    return tuple(output)


def _read_samples(path: Path) -> list[MaskSample]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT frame, mask_id, polygons FROM masks ORDER BY frame, mask_id"
        ).fetchall()
    finally:
        connection.close()
    return [
        MaskSample(int(frame), str(mask_id), _decode_polygons(str(polygons)))
        for frame, mask_id, polygons in rows
    ]


def _select_diverse_samples(
    samples: list[MaskSample], count: int, minimum_vertices: int
) -> list[MaskSample]:
    valid = [sample for sample in samples if sample.vertex_count >= 3]
    candidates = [
        sample for sample in valid if sample.vertex_count >= minimum_vertices
    ]
    if len(candidates) < min(count, len(valid)):
        candidates = valid
    if not candidates:
        raise RuntimeError("no valid polygons found")
    candidates.sort(key=lambda sample: sample.vertex_count)
    low = candidates[0].vertex_count
    high = candidates[-1].vertex_count
    anchors = np.linspace(low, high, min(count, len(candidates)))
    selected: list[MaskSample] = []
    used: set[tuple[int, str]] = set()
    for anchor in anchors:
        available = [
            sample
            for sample in candidates
            if (sample.frame, sample.mask_id) not in used
        ]
        chosen = min(
            available,
            key=lambda sample: abs(sample.vertex_count - float(anchor)),
        )
        selected.append(chosen)
        used.add((chosen.frame, chosen.mask_id))
    return selected


def _rdp_with_cap(points: np.ndarray, cap: int) -> np.ndarray:
    if len(points) <= cap:
        return points.copy()
    contour = points.reshape(-1, 1, 2)
    zero = cv2.approxPolyDP(contour, 0.0, True).reshape(-1, 2)
    if len(zero) <= cap:
        return zero
    perimeter = float(cv2.arcLength(contour, True))
    low = 0.0
    high = max(perimeter, 1.0)
    best = cv2.approxPolyDP(contour, high, True).reshape(-1, 2)
    for _ in range(48):
        epsilon = (low + high) / 2.0
        candidate = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(candidate) > cap:
            low = epsilon
        else:
            high = epsilon
            best = candidate
    if len(best) < 3:
        return points.copy()
    return best


def _simplify(
    polygons: tuple[np.ndarray, ...], cap: int
) -> tuple[np.ndarray, ...]:
    total = sum(len(polygon) for polygon in polygons)
    if total <= cap:
        return tuple(polygon.copy() for polygon in polygons)
    remaining = cap
    output: list[np.ndarray] = []
    for index, polygon in enumerate(polygons):
        if index == len(polygons) - 1:
            component_cap = remaining
        else:
            component_cap = max(3, round(cap * len(polygon) / total))
            component_cap = min(component_cap, remaining - 3)
        simplified = _rdp_with_cap(polygon, max(3, component_cap))
        output.append(simplified)
        remaining -= len(simplified)
    return tuple(output)


def _as_contours(polygons: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    return [
        np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        for polygon in polygons
    ]


def _mask(
    polygons: tuple[np.ndarray, ...], width: int, height: int
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, _as_contours(polygons), 1)
    return output


def _metrics(
    original: tuple[np.ndarray, ...],
    simplified: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> tuple[float, float]:
    original_mask = _mask(original, width, height)
    simplified_mask = _mask(simplified, width, height)
    intersection = int(np.count_nonzero(original_mask & simplified_mask))
    union = int(np.count_nonzero(original_mask | simplified_mask))
    original_area = int(np.count_nonzero(original_mask))
    simplified_area = int(np.count_nonzero(simplified_mask))
    iou = intersection / union if union else 1.0
    area_delta = (
        100.0 * (simplified_area - original_area) / original_area
        if original_area
        else 0.0
    )
    return iou, area_delta


def _crop_box(
    polygons: tuple[np.ndarray, ...], width: int, height: int
) -> tuple[int, int, int, int]:
    points = np.concatenate(polygons, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = max(16.0, float(max(span)) * 0.12)
    x1 = max(0, math.floor(float(minimum[0]) - padding))
    y1 = max(0, math.floor(float(minimum[1]) - padding))
    x2 = min(width, math.ceil(float(maximum[0]) + padding) + 1)
    y2 = min(height, math.ceil(float(maximum[1]) + padding) + 1)
    return x1, y1, x2, y2


def _tint_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> None:
    pixels = mask.astype(bool)
    if not np.any(pixels):
        return
    blended = (
        image[pixels].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    )
    image[pixels] = np.clip(blended, 0, 255).astype(np.uint8)


def _render_overlay(
    frame: np.ndarray,
    original: tuple[np.ndarray, ...],
    simplified: tuple[np.ndarray, ...] | None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    output = frame.copy()
    original_mask = _mask(original, width, height)
    if simplified is None:
        _tint_mask(output, original_mask, (255, 180, 0), 0.28)
        cv2.polylines(
            output, _as_contours(original), True, (255, 255, 255), 2, cv2.LINE_AA
        )
        return output

    simplified_mask = _mask(simplified, width, height)
    shared = original_mask & simplified_mask
    missed = original_mask & (1 - simplified_mask)
    extra = simplified_mask & (1 - original_mask)
    _tint_mask(output, shared, (0, 210, 255), 0.22)
    _tint_mask(output, missed, (0, 0, 255), 0.62)
    _tint_mask(output, extra, (255, 0, 255), 0.62)
    cv2.polylines(
        output, _as_contours(original), True, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.polylines(
        output, _as_contours(simplified), True, (0, 215, 255), 2, cv2.LINE_AA
    )
    return output


def _fit_panel(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.full((size, size, 3), 24, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _title_panel(panel: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header = np.full((62, panel.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, panel])


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame_index}")
    return frame


def main() -> None:
    args = _parse_args()
    if any(cap < 3 for cap in args.caps):
        raise ValueError("all caps must be at least 3")
    minimum_vertices = (
        args.minimum_original_vertices
        if args.minimum_original_vertices is not None
        else max(args.caps)
    )
    samples = _select_diverse_samples(
        _read_samples(args.sqlite), args.samples, minimum_vertices
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    rendered_rows: list[np.ndarray] = []
    try:
        for sample_index, sample in enumerate(samples, start=1):
            frame = _read_frame(capture, sample.frame)
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = _crop_box(sample.polygons, width, height)
            original_overlay = _render_overlay(frame, sample.polygons, None)
            panels = [
                _title_panel(
                    _fit_panel(original_overlay[y1:y2, x1:x2], args.panel_size),
                    f"Shape {sample_index}: original",
                    f"frame {sample.frame} | {sample.vertex_count} vertices",
                )
            ]
            for cap in args.caps:
                simplified = _simplify(sample.polygons, cap)
                actual = sum(len(polygon) for polygon in simplified)
                iou, area_delta = _metrics(
                    sample.polygons, simplified, width, height
                )
                overlay = _render_overlay(frame, sample.polygons, simplified)
                panels.append(
                    _title_panel(
                        _fit_panel(overlay[y1:y2, x1:x2], args.panel_size),
                        f"max {cap}: {actual} vertices",
                        f"IoU {iou:.5f} | area {area_delta:+.3f}%",
                    )
                )
                metrics_rows.append(
                    {
                        "shape": sample_index,
                        "frame": sample.frame,
                        "mask_id": sample.mask_id,
                        "original_vertices": sample.vertex_count,
                        "vertex_cap": cap,
                        "actual_vertices": actual,
                        "iou": f"{iou:.8f}",
                        "area_delta_percent": f"{area_delta:.6f}",
                    }
                )
            row = np.hstack(panels)
            rendered_rows.append(row)
            cv2.imwrite(
                str(
                    args.output_dir
                    / f"shape_{sample_index:02d}_frame_{sample.frame:04d}.png"
                ),
                row,
            )
    finally:
        capture.release()

    legend = np.full((76, rendered_rows[0].shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        legend,
        "Polygon vertex comparison (RDP)",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        legend,
        "white: original edge | yellow: simplified edge | red: missed | magenta: extra",
        (16, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    grid = np.vstack([legend, *rendered_rows])
    cv2.imwrite(str(args.output_dir / "vertex_comparison_grid.png"), grid)

    with (args.output_dir / "comparison_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    manifest = {
        "video": str(args.video.resolve()),
        "sqlite": str(args.sqlite.resolve()),
        "caps": args.caps,
        "selected_samples": [
            {
                "shape": index,
                "frame": sample.frame,
                "mask_id": sample.mask_id,
                "original_vertices": sample.vertex_count,
            }
            for index, sample in enumerate(samples, start=1)
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
