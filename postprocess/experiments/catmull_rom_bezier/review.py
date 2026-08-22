"""Video-backed review overlays for polygon versus Catmull--Rom DP."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from production.curve.runtime.keyframe_dp import KeyframeDpResult


RAW_COLOR = (80, 220, 80)
POLYGON_COLOR = (220, 80, 220)
CURVE_COLOR = (255, 210, 40)
POINT_COLOR = (40, 180, 255)


def select_review_indices(
    polygon: KeyframeDpResult,
    curve: KeyframeDpResult,
    maximum: int = 16,
) -> list[int]:
    count = len(curve.audit.iou)
    selected: set[int] = {0, count - 1}

    def add_order(values: np.ndarray, *, descending: bool, limit: int) -> None:
        order = np.argsort(values)
        if descending:
            order = order[::-1]
        selected.update(int(value) for value in order[:limit])

    add_order(curve.audit.iou, descending=False, limit=3)
    add_order(curve.audit.iou - polygon.audit.iou, descending=False, limit=3)
    add_order(curve.audit.iou - polygon.audit.iou, descending=True, limit=3)
    add_order(curve.audit.area_ratio, descending=True, limit=3)
    for left, right in zip(
        curve.chosen_indices[:-1], curve.chosen_indices[1:], strict=True
    ):
        selected.add(int(round((left + right) / 2)))
    ranked = sorted(
        selected,
        key=lambda index: (
            curve.audit.iou[index],
            -(curve.audit.area_ratio[index]),
            index,
        ),
    )
    if len(ranked) > int(maximum):
        ranked = ranked[: int(maximum)]
    return sorted(ranked)


def _read_frames(video: Path, frame_numbers: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video}")
    output: dict[int, np.ndarray] = {}
    try:
        for frame in sorted(set(int(value) for value in frame_numbers)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"cannot decode frame {frame} from {video}")
            output[frame] = image
    finally:
        capture.release()
    return output


def _overlay(
    image: np.ndarray,
    raw: np.ndarray,
    boundary: np.ndarray,
    controls: np.ndarray | None,
    *,
    color: tuple[int, int, int],
    title: str,
    detail: str,
) -> np.ndarray:
    header_height = 64
    output = np.full(
        (image.shape[0] + header_height, image.shape[1], 3), 20, dtype=np.uint8
    )
    output[header_height:] = image
    tint = output.copy()
    offset = np.asarray((0, header_height), dtype=np.float64)
    candidate = np.rint(boundary + offset).astype(np.int32)
    raw_points = np.rint(raw + offset).astype(np.int32)
    cv2.fillPoly(tint, [candidate], color)
    cv2.addWeighted(tint, 0.20, output, 0.80, 0.0, output)
    cv2.polylines(output, [raw_points], True, RAW_COLOR, 2, cv2.LINE_AA)
    cv2.polylines(output, [candidate], True, color, 3, cv2.LINE_AA)
    if controls is not None:
        shifted_controls = np.rint(controls + offset).astype(np.int32)
        for index, point in enumerate(shifted_controls):
            cv2.circle(output, tuple(point), 5, POINT_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                f"P{index}",
                (int(point[0]) + 5, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(
        output, (0, 0), (output.shape[1], header_height - 1), (20, 20, 20), -1
    )
    cv2.putText(
        output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        output,
        detail,
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return output


def _roi_bounds(
    image_shape: tuple[int, ...], arrays: list[np.ndarray], margin: int = 45
) -> tuple[int, int, int, int]:
    points = np.concatenate([np.asarray(value).reshape(-1, 2) for value in arrays])
    low = np.floor(np.min(points, axis=0)).astype(int) - int(margin)
    high = np.ceil(np.max(points, axis=0)).astype(int) + int(margin)
    height, width = image_shape[:2]
    x0, y0 = max(0, low[0]), max(0, low[1])
    x1, y1 = min(width, high[0] + 1), min(height, high[1] + 1)
    if x1 - x0 < 360:
        extra = (360 - (x1 - x0)) // 2 + 1
        x0, x1 = max(0, x0 - extra), min(width, x1 + extra)
    if y1 - y0 < 280:
        extra = (280 - (y1 - y0)) // 2 + 1
        y0, y1 = max(0, y0 - extra), min(height, y1 + extra)
    return int(x0), int(y0), int(x1), int(y1)


def _three_panel(
    image: np.ndarray,
    raw: np.ndarray,
    polygon: KeyframeDpResult,
    curve: KeyframeDpResult,
    index: int,
    frame_number: int,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    origin = np.asarray((x0, y0), dtype=np.float64)
    crop_image = image[y0:y1, x0:x1]
    crop_raw = np.asarray(raw, dtype=np.float64) - origin
    crop_polygon = polygon.dense_boundaries[index] - origin
    crop_curve = curve.dense_boundaries[index] - origin
    polygon_controls = polygon.dense_controls[index] - origin
    curve_controls = curve.dense_controls[index] - origin
    polygon_key = "KEY" if index in polygon.chosen_indices else "interp"
    curve_key = "KEY" if index in curve.chosen_indices else "interp"
    panels = [
        _overlay(
            crop_image,
            crop_raw,
            crop_raw,
            None,
            color=RAW_COLOR,
            title=f"AI RAW | frame {frame_number}",
            detail="GREEN = source mask",
        ),
        _overlay(
            crop_image,
            crop_raw,
            crop_polygon,
            polygon_controls,
            color=POLYGON_COLOR,
            title="POLYGON + SAME DP",
            detail=(
                f"IoU {polygon.audit.iou[index]:.3f} | "
                f"R {polygon.audit.recall[index]:.3f} | "
                f"A {polygon.audit.area_ratio[index]:.3f}x | {polygon_key}"
            ),
        ),
        _overlay(
            crop_image,
            crop_raw,
            crop_curve,
            curve_controls,
            color=CURVE_COLOR,
            title="CATMULL-BEZIER + SAME DP",
            detail=(
                f"IoU {curve.audit.iou[index]:.3f} | "
                f"R {curve.audit.recall[index]:.3f} | "
                f"A {curve.audit.area_ratio[index]:.3f}x | {curve_key}"
            ),
        ),
    ]
    return np.concatenate(panels, axis=1)


def render_review_gallery(
    video: Path,
    actual_frames: np.ndarray,
    references: list[np.ndarray],
    polygon: KeyframeDpResult,
    curve: KeyframeDpResult,
    output_dir: Path,
    *,
    maximum_frames: int = 16,
) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    indices = select_review_indices(polygon, curve, maximum_frames)
    frame_numbers = [int(actual_frames[index]) for index in indices]
    decoded = _read_frames(Path(video), frame_numbers)
    manifest: list[dict[str, object]] = []
    rendered_images: list[np.ndarray] = []
    for index, frame_number in zip(indices, frame_numbers, strict=True):
        image = decoded[frame_number]
        raw = np.asarray(references[index], dtype=np.float64)
        polygon_boundary = polygon.dense_boundaries[index]
        curve_boundary = curve.dense_boundaries[index]
        bounds = _roi_bounds(image.shape, [raw, polygon_boundary, curve_boundary])
        crop = _three_panel(image, raw, polygon, curve, index, frame_number, bounds)
        path = output / f"frame_{frame_number:06d}.jpg"
        if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {path}")
        rendered_images.append(crop)
        manifest.append(
            {
                "frame_index": int(index),
                "frame": int(frame_number),
                "path": str(path),
                "polygon_iou": float(polygon.audit.iou[index]),
                "curve_iou": float(curve.audit.iou[index]),
                "curve_minus_polygon_iou": float(
                    curve.audit.iou[index] - polygon.audit.iou[index]
                ),
                "polygon_recall": float(polygon.audit.recall[index]),
                "curve_recall": float(curve.audit.recall[index]),
                "polygon_area_ratio": float(polygon.audit.area_ratio[index]),
                "curve_area_ratio": float(curve.audit.area_ratio[index]),
            }
        )
    if rendered_images:
        width = max(image.shape[1] for image in rendered_images)
        normalized = []
        for image in rendered_images:
            if image.shape[1] == width:
                normalized.append(image)
            else:
                scale = width / image.shape[1]
                normalized.append(
                    cv2.resize(
                        image,
                        (width, int(round(image.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                )
        sheet = np.concatenate(normalized, axis=0)
        cv2.imwrite(
            str(output / "contact_sheet.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(manifest[0]) if manifest else ()
        )
        if manifest:
            writer.writeheader()
            writer.writerows(manifest)
    return manifest


def render_review_video(
    video: Path,
    actual_frames: np.ndarray,
    references: list[np.ndarray],
    polygon: KeyframeDpResult,
    curve: KeyframeDpResult,
    output_path: Path,
) -> dict[str, object]:
    """Render every evaluated frame with a fixed ROI for temporal review."""
    frame_numbers = [int(value) for value in actual_frames]
    decoded = _read_frames(Path(video), frame_numbers)
    first_image = decoded[frame_numbers[0]]
    arrays = list(references)
    arrays.extend(list(polygon.dense_boundaries))
    arrays.extend(list(curve.dense_boundaries))
    bounds = _roi_bounds(first_image.shape, arrays, margin=60)

    probe = cv2.VideoCapture(str(video))
    fps = float(probe.get(cv2.CAP_PROP_FPS)) if probe.isOpened() else 0.0
    probe.release()
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    first_panel = _three_panel(
        first_image,
        references[0],
        polygon,
        curve,
        0,
        frame_numbers[0],
        bounds,
    )
    # Common MP4 encoders require an even luma/chroma grid and silently crop
    # odd dimensions.  Make that crop explicit so the decode audit is exact.
    encoded_height = int(first_panel.shape[0] // 2 * 2)
    encoded_width = int(first_panel.shape[1] // 2 * 2)
    first_panel = np.ascontiguousarray(first_panel[:encoded_height, :encoded_width])
    path = Path(output_path)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(first_panel.shape[1]), int(first_panel.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open review video writer: {path}")
    try:
        writer.write(first_panel)
        for index in range(1, len(frame_numbers)):
            panel = _three_panel(
                decoded[frame_numbers[index]],
                references[index],
                polygon,
                curve,
                index,
                frame_numbers[index],
                bounds,
            )
            writer.write(np.ascontiguousarray(panel[:encoded_height, :encoded_width]))
    finally:
        writer.release()
    verify = cv2.VideoCapture(str(path))
    if not verify.isOpened():
        raise RuntimeError(f"cannot reopen review video: {path}")
    decoded_count = 0
    while True:
        ok, frame = verify.read()
        if not ok:
            break
        if frame is None or frame.shape[:2] != first_panel.shape[:2]:
            raise RuntimeError(f"invalid frame in review video: {path}")
        decoded_count += 1
    verify.release()
    if decoded_count != len(frame_numbers):
        raise RuntimeError(
            f"review video frame mismatch: {decoded_count} != {len(frame_numbers)}"
        )
    return {
        "path": str(path),
        "frames": int(decoded_count),
        "fps": float(fps),
        "width": int(first_panel.shape[1]),
        "height": int(first_panel.shape[0]),
        "fixed_roi_xyxy": list(bounds),
    }


__all__ = (
    "render_review_gallery",
    "render_review_video",
    "select_review_indices",
)
