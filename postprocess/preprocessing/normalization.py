"""Canonical detector JSONL normalization and validation.

The contract intentionally accepts both historic detector spellings:
`frame_index`/`detections` from DINOv3 and `frame_idx`/`instances` from EVA02.
Validation is streaming so long videos do not require loading the JSONL into
memory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DetectionJsonlContractError(ValueError):
    """Raised when a detector JSONL file cannot be consumed safely."""


@dataclass
class DetectionJsonlStats:
    frame_records: int = 0
    detections: int = 0
    detections_with_bbox: int = 0
    detections_with_mask: int = 0
    detections_with_label: int = 0
    empty_frames: int = 0
    first_frame_index: int | None = None
    last_frame_index: int | None = None
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise DetectionJsonlContractError(
            f"{field} must be an integer: {value!r}"
        ) from exc


def _to_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise DetectionJsonlContractError(
            f"{field} must be numeric: {value!r}"
        ) from exc


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _to_int(value, field)


def _optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _to_float(value, field)


def _normalize_bbox_xyxy(src: dict[str, Any]) -> list[float] | None:
    bbox_xyxy = src.get("bbox_xyxy")
    if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
        x1, y1, x2, y2 = [_to_float(v, "bbox_xyxy") for v in bbox_xyxy[:4]]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    bbox = src.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, width, height = [_to_float(v, "bbox") for v in bbox[:4]]
        return [x, y, x + max(0.0, width), y + max(0.0, height)]
    return None


def _normalize_polygons(value: object) -> list[list[list[float]]]:
    if not isinstance(value, list) or not value:
        raise DetectionJsonlContractError(
            "non-empty polygons/segmentation list is required for each detection"
        )
    is_flat_polygon = all(isinstance(item, (int, float)) for item in value)
    is_point_polygon = all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (int, float))
        and isinstance(item[1], (int, float))
        for item in value
    )
    candidates = [value] if is_flat_polygon or is_point_polygon else value
    polygons: list[list[list[float]]] = []
    for candidate in candidates:
        if not isinstance(candidate, list) or not candidate:
            continue
        if all(isinstance(item, (int, float)) for item in candidate):
            if len(candidate) < 6 or len(candidate) % 2:
                raise DetectionJsonlContractError(
                    "flat polygon coordinates require at least three x/y pairs"
                )
            polygon = [
                [
                    _to_float(candidate[index], "polygons"),
                    _to_float(candidate[index + 1], "polygons"),
                ]
                for index in range(0, len(candidate), 2)
            ]
        else:
            polygon = []
            for point in candidate:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    raise DetectionJsonlContractError(
                        "polygon points must be [x, y] pairs"
                    )
                polygon.append(
                    [
                        _to_float(point[0], "polygons"),
                        _to_float(point[1], "polygons"),
                    ]
                )
        if len(polygon) < 3:
            raise DetectionJsonlContractError(
                "each polygon requires at least three points"
            )
        if not all(math.isfinite(value) for point in polygon for value in point):
            raise DetectionJsonlContractError(
                "polygon coordinates must be finite"
            )
        polygons.append(polygon)
    if not polygons:
        raise DetectionJsonlContractError(
            "at least one valid polygon is required for each detection"
        )
    return polygons


def _bbox_from_polygons(
    polygons: list[list[list[float]]],
) -> list[float]:
    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _has_mask(src: dict[str, Any]) -> bool:
    polygons = src.get("polygons")
    segmentation = src.get("segmentation")
    return (
        isinstance(polygons, list)
        and bool(polygons)
        or isinstance(segmentation, list)
        and bool(segmentation)
    )


def normalize_detection(src: Any) -> dict[str, Any]:
    if not isinstance(src, dict):
        raise DetectionJsonlContractError(
            f"detection must be an object: {type(src).__name__}"
        )

    label = src.get("class_name", src.get("label", "unknown"))
    out: dict[str, Any] = {
        "class_name": str(label),
        "label": str(src.get("label", label)),
    }

    polygons = _normalize_polygons(
        src.get("polygons", src.get("segmentation"))
    )
    bbox_xyxy = _normalize_bbox_xyxy(src) or _bbox_from_polygons(polygons)
    if bbox_xyxy is not None:
        out["bbox_xyxy"] = bbox_xyxy
        out["bbox"] = [
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ]

    for key in ("source_detection_id", "category_id", "category_index"):
        value = _optional_int(src.get(key), key)
        if value is not None:
            out[key] = value

    for key in ("score", "detector_score", "class_score"):
        value = _optional_float(src.get(key), key)
        if value is not None:
            out[key] = value
    if "class_score" not in out:
        value = _optional_float(src.get("cls_score"), "cls_score")
        if value is not None:
            out["class_score"] = value

    out["polygons"] = polygons
    out["segmentation"] = polygons

    return out


def normalize_frame_record(src: Any) -> dict[str, Any]:
    if not isinstance(src, dict):
        raise DetectionJsonlContractError(
            f"frame record must be an object: {type(src).__name__}"
        )
    frame_value = src.get("frame_index", src.get("frame_idx"))
    if frame_value is None:
        raise DetectionJsonlContractError("frame_index or frame_idx is required")
    frame_index = _to_int(frame_value, "frame_index")

    raw_detections = src.get("detections", src.get("instances", []))
    if raw_detections is None:
        raw_detections = []
    if not isinstance(raw_detections, list):
        raise DetectionJsonlContractError("detections/instances must be a list")

    out: dict[str, Any] = {
        "frame_index": frame_index,
        "frame_idx": frame_index,
        "detections": [normalize_detection(item) for item in raw_detections],
    }
    out["instances"] = out["detections"]

    for key in ("width", "height"):
        value = _optional_int(src.get(key), key)
        if value is not None:
            out[key] = value
    value = _optional_float(src.get("time_sec"), "time_sec")
    if value is not None:
        out["time_sec"] = value
    return out


def summarize_detection_jsonl(path: Path) -> DetectionJsonlStats:
    stats = DetectionJsonlStats()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = normalize_frame_record(json.loads(text))
            except json.JSONDecodeError as exc:
                raise DetectionJsonlContractError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            except DetectionJsonlContractError as exc:
                raise DetectionJsonlContractError(f"{path}:{line_no}: {exc}") from exc

            frame_index = int(record["frame_index"])
            stats.frame_records += 1
            stats.first_frame_index = (
                frame_index
                if stats.first_frame_index is None
                else min(stats.first_frame_index, frame_index)
            )
            stats.last_frame_index = (
                frame_index
                if stats.last_frame_index is None
                else max(stats.last_frame_index, frame_index)
            )

            width = record.get("width")
            height = record.get("height")
            if isinstance(width, int):
                stats.min_width = (
                    width if stats.min_width is None else min(stats.min_width, width)
                )
                stats.max_width = (
                    width if stats.max_width is None else max(stats.max_width, width)
                )
            if isinstance(height, int):
                stats.min_height = (
                    height
                    if stats.min_height is None
                    else min(stats.min_height, height)
                )
                stats.max_height = (
                    height
                    if stats.max_height is None
                    else max(stats.max_height, height)
                )

            detections = list(record["detections"])
            stats.detections += len(detections)
            if not detections:
                stats.empty_frames += 1
            for det in detections:
                if "bbox_xyxy" in det:
                    stats.detections_with_bbox += 1
                if _has_mask(det):
                    stats.detections_with_mask += 1
                if det.get("class_name") or det.get("label"):
                    stats.detections_with_label += 1

    if stats.frame_records == 0:
        raise DetectionJsonlContractError(f"{path}: no frame records found")
    return stats


def normalize_detection_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    """Normalize model-specific JSONL into the canonical detection contract."""

    source = Path(input_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("normalization output must differ from input")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = detections = 0
    with source.open("r", encoding="utf-8") as source_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(source_handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                normalized = normalize_frame_record(json.loads(text))
            except Exception as exc:
                raise DetectionJsonlContractError(
                    f"{source}:{line_number}: {exc}"
                ) from exc
            canonical = {
                "frame_index": int(normalized["frame_index"]),
                "detections": normalized["detections"],
            }
            for key in ("width", "height", "time_sec"):
                if key in normalized:
                    canonical[key] = normalized[key]
            output_handle.write(
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            frames += 1
            detections += len(canonical["detections"])
    return {"frames": frames, "detections": detections}
