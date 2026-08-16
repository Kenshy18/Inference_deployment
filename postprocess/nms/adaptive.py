"""Adaptive non-maximum suppression for canonical mask detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .components import (
    remove_redundant_islands_candidate_v1,
    remove_small_foreground_components,
)


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
    island_cleanup_policy: str = "disabled"
    remove_small_islands: bool = False
    small_island_ratio_max: float = 0.10
    fill_all_holes: bool = True
    island_unconditional_owner_ratio_max: float = 0.01
    island_other_coverage_min: float = 0.90
    island_to_other_area_max: float = 0.30

    def thresholds_for_area(self, area: float) -> tuple[float, float]:
        if area <= self.tiny_area:
            return self.tiny_iou_threshold, self.tiny_contain_ratio_max
        if area <= self.small_area:
            return self.small_iou_threshold, self.small_contain_ratio_max
        return self.iou_threshold, self.contain_ratio_max

    def pair_suppression_reason(
        self,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> str | None:
        """Return the legacy reason for suppressing ``second`` by ``first``.

        The predicate is deliberately symmetric in geometry; caller ordering
        supplies the score/stable-index priority.  Exposing the predicate lets
        component-aware policies reuse the validated Production rule without
        reimplementing it or changing legacy output.
        """
        first_bbox = tuple(map(float, first.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])))
        second_bbox = tuple(map(float, second.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])))
        first_bbox_area = _bbox_area(first)
        second_bbox_area = _bbox_area(second)
        first_mask_area = _mask_area(first)
        second_mask_area = _mask_area(second)
        first_size = (
            min(first_bbox_area, first_mask_area)
            if first_mask_area > 0.0
            else first_bbox_area
        )
        second_size = (
            min(second_bbox_area, second_mask_area)
            if second_mask_area > 0.0
            else second_bbox_area
        )
        threshold, contain_limit = self.thresholds_for_area(
            min(first_size, second_size)
        )
        area_min = min(first_bbox_area, second_bbox_area)
        area_max = max(first_bbox_area, second_bbox_area)
        if (
            _contained_pair(
                first,
                second,
                first_bbox,
                second_bbox,
                self.contain_margin,
            )
            and area_min > 0.0
            and area_max / area_min <= contain_limit
        ):
            return "contained"
        if _bbox_iou(first_bbox, second_bbox) >= threshold:
            return "bbox_iou"
        return None

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return []
        if self.island_cleanup_policy == "production_candidate_v1":
            detections = remove_redundant_islands_candidate_v1(
                detections,
                fill_all_holes=self.fill_all_holes,
                unconditional_owner_ratio_max=(
                    self.island_unconditional_owner_ratio_max
                ),
                other_coverage_min=self.island_other_coverage_min,
                island_to_other_area_max=self.island_to_other_area_max,
            )
        elif self.island_cleanup_policy != "disabled":
            raise ValueError(
                f"unsupported island cleanup policy: {self.island_cleanup_policy}"
            )
        elif self.remove_small_islands:
            detections = [
                remove_small_foreground_components(
                    detection,
                    ratio_max=self.small_island_ratio_max,
                )
                for detection in detections
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
                if (
                    self.pair_suppression_reason(detections[index], detections[other])
                    is not None
                ):
                    suppressed.add(other)
        return [detections[index] for index in retained]


DEFAULT_NMS = AdaptiveNms()


def thresholds_for_area(area: float) -> tuple[float, float]:
    return DEFAULT_NMS.thresholds_for_area(float(area))


def apply_nms(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return DEFAULT_NMS.apply(detections)
