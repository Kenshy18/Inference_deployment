"""Adaptive non-maximum suppression for canonical mask detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class NmsPolicy(Protocol):
    """Replaceable detection suppression policy."""

    name: str

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return retained detections in deterministic order."""


def _bbox_area(detection: dict[str, Any]) -> float:
    cached = detection.get("_bbox_area")
    if isinstance(cached, (int, float)):
        return float(cached)
    x1, y1, x2, y2 = detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


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


def _mask_area(detection: dict[str, Any]) -> float:
    cached = detection.get("_mask_area")
    if isinstance(cached, (int, float)):
        return float(cached)
    return sum(_polygon_area(polygon) for polygon in detection.get("polygons") or [])


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_contains(a: tuple[float, ...], b: tuple[float, ...], margin: float) -> bool:
    return (
        a[0] - margin <= b[0]
        and a[1] - margin <= b[1]
        and a[2] + margin >= b[2]
        and a[3] + margin >= b[3]
    )


def _mask_inside_bbox(
    detection: dict[str, Any], bbox: tuple[float, ...], margin: float
) -> bool:
    x1, y1, x2, y2 = bbox
    for polygon in detection.get("polygons") or []:
        if polygon and all(
            x1 - margin <= float(point[0]) <= x2 + margin
            and y1 - margin <= float(point[1]) <= y2 + margin
            for point in polygon
        ):
            return True
    return False


def _contained_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    first_bbox: tuple[float, ...],
    second_bbox: tuple[float, ...],
    margin: float,
) -> bool:
    return (
        _bbox_contains(first_bbox, second_bbox, margin)
        or _bbox_contains(second_bbox, first_bbox, margin)
        or _mask_inside_bbox(second, first_bbox, margin)
        or _mask_inside_bbox(first, second_bbox, margin)
    )


@dataclass(frozen=True)
class AdaptiveNms:
    """Area-aware NMS matching the validated preprocessing defaults."""

    name: str = "adaptive"
    iou_threshold: float = 0.20
    small_iou_threshold: float = 0.10
    tiny_iou_threshold: float = 0.05
    small_area: float = 5000.0
    tiny_area: float = 2000.0
    contain_ratio_max: float = 8.0
    small_contain_ratio_max: float = 5.0
    tiny_contain_ratio_max: float = 5.0
    contain_margin: float = 2.0

    def thresholds_for_area(self, area: float) -> tuple[float, float]:
        if area <= self.tiny_area:
            return self.tiny_iou_threshold, self.tiny_contain_ratio_max
        if area <= self.small_area:
            return self.small_iou_threshold, self.small_contain_ratio_max
        return self.iou_threshold, self.contain_ratio_max

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return []
        bboxes = [
            tuple(map(float, detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])))
            for detection in detections
        ]
        bbox_areas = [_bbox_area(detection) for detection in detections]
        mask_areas = [_mask_area(detection) for detection in detections]
        size_refs = [
            min(bbox_area, mask_area) if mask_area > 0.0 else bbox_area
            for bbox_area, mask_area in zip(bbox_areas, mask_areas, strict=True)
        ]
        order = sorted(
            range(len(detections)),
            key=lambda index: (-float(detections[index].get("score") or 0.0), index),
        )
        suppressed: set[int] = set()
        retained: list[int] = []
        for position, index in enumerate(order):
            if index in suppressed:
                continue
            retained.append(index)
            for other in order[position + 1 :]:
                if other in suppressed:
                    continue
                threshold, contain_limit = self.thresholds_for_area(
                    min(size_refs[index], size_refs[other])
                )
                area_min = min(bbox_areas[index], bbox_areas[other])
                area_max = max(bbox_areas[index], bbox_areas[other])
                contained = _contained_pair(
                    detections[index],
                    detections[other],
                    bboxes[index],
                    bboxes[other],
                    self.contain_margin,
                )
                if (
                    contained
                    and area_min > 0.0
                    and area_max / area_min <= contain_limit
                ):
                    suppressed.add(other)
                elif _bbox_iou(bboxes[index], bboxes[other]) >= threshold:
                    suppressed.add(other)
        return [detections[index] for index in retained]


DEFAULT_NMS = AdaptiveNms()


def thresholds_for_area(area: float) -> tuple[float, float]:
    return DEFAULT_NMS.thresholds_for_area(float(area))


def apply_nms(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return DEFAULT_NMS.apply(detections)
