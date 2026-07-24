"""Canonical detection input preparation owned by tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from contracts.detections import iter_detection_records


def _polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return (
        abs(
            sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
        )
        * 0.5
    )


def _polygon_bbox(
    polygon: list[list[float]],
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_polygons(value: object) -> list[list[list[float]]]:
    if not isinstance(value, list) or not value:
        return []
    candidates = (
        [value] if all(isinstance(item, (int, float)) for item in value) else value
    )
    polygons: list[list[list[float]]] = []
    for candidate in candidates:
        if not isinstance(candidate, list) or not candidate:
            continue
        if all(isinstance(item, (int, float)) for item in candidate):
            polygon = [
                [float(candidate[index]), float(candidate[index + 1])]
                for index in range(0, len(candidate) - 1, 2)
            ]
        else:
            polygon = [
                [float(point[0]), float(point[1])]
                for point in candidate
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def prepare_detection(detection: dict[str, Any]) -> dict[str, Any]:
    output = dict(detection)
    polygons = _normalize_polygons(output.get("polygons"))
    output["polygons"] = polygons
    output["segmentation"] = polygons
    if "bbox_xyxy" in output:
        bbox = list(map(float, output["bbox_xyxy"]))
    else:
        boxes = [_polygon_bbox(polygon) for polygon in polygons]
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    output["bbox_xyxy"] = bbox
    output["_bbox_area"] = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    output["_mask_area"] = sum(_polygon_area(polygon) for polygon in polygons)
    return output


def iter_tracking_records(
    path: Path,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    for record in iter_detection_records(path):
        yield int(record["frame_index"]), [
            prepare_detection(detection) for detection in record["detections"]
        ]
