"""Exact mask geometry shared by Production NMS policies.

This module owns only geometry and overlap calculations.  It deliberately has
no suppression thresholds or stage orchestration, so topology, policy and I/O
can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .components import _geometry


def polygon_area(polygon: list[list[float]]) -> float:
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


def bbox_area(detection: dict[str, Any]) -> float:
    cached = detection.get("_bbox_area")
    if isinstance(cached, (int, float)):
        return float(cached)
    x1, y1, x2, y2 = detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    return max(0.0, float(x2) - float(x1)) * max(
        0.0, float(y2) - float(y1)
    )


def mask_area(detection: dict[str, Any]) -> float:
    cached = detection.get("_mask_area")
    if isinstance(cached, (int, float)):
        return float(cached)
    return sum(polygon_area(item) for item in detection.get("polygons") or [])


def association_geometry(detection: dict[str, Any]) -> dict[str, Any]:
    """Return transient pre-cleanup geometry used only for tracking."""
    output: dict[str, Any] = {}
    bbox = detection.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        output["_association_bbox_xyxy"] = [float(value) for value in bbox]
    output["_association_mask_area"] = float(
        sum(polygon_area(item) for item in detection.get("polygons") or [])
    )
    return output


@dataclass(frozen=True)
class RasterMask:
    mask: np.ndarray
    left: int
    top: int
    right: int
    bottom: int
    area: int


@dataclass(frozen=True)
class MaskOverlapMetrics:
    intersection: int
    union: int
    first_area: int
    second_area: int

    @property
    def iou(self) -> float:
        return self.intersection / self.union if self.union > 0 else 0.0

    @property
    def first_coverage(self) -> float:
        return self.intersection / self.first_area if self.first_area > 0 else 0.0

    @property
    def second_coverage(self) -> float:
        return self.intersection / self.second_area if self.second_area > 0 else 0.0

    @property
    def smaller_coverage(self) -> float:
        smaller = min(self.first_area, self.second_area)
        return self.intersection / smaller if smaller > 0 else 0.0

    @property
    def smaller_to_larger_area_ratio(self) -> float:
        larger = max(self.first_area, self.second_area)
        return min(self.first_area, self.second_area) / larger if larger > 0 else 0.0


def raster_mask(detection: dict[str, Any]) -> RasterMask | None:
    geometry = _geometry(detection)
    if geometry is None:
        return None
    points = np.concatenate(geometry.polygons, axis=0)
    left, top = np.floor(points.min(axis=0)).astype(int) - 1
    right, bottom = np.ceil(points.max(axis=0)).astype(int) + 1
    width = int(right - left + 1)
    height = int(bottom - top + 1)
    if width <= 0 or height <= 0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    offset = np.array([left, top], dtype=np.float32)
    for index in sorted(
        range(len(geometry.polygons)), key=lambda value: geometry.depths[value]
    ):
        contour = np.rint(geometry.polygons[index] - offset).astype(np.int32)
        value = 1 if geometry.depths[index] % 2 == 0 else 0
        cv2.fillPoly(mask, [contour], value)
    return RasterMask(
        mask=mask,
        left=int(left),
        top=int(top),
        right=int(right),
        bottom=int(bottom),
        area=int(np.count_nonzero(mask)),
    )


def overlap_slices(
    first: RasterMask, second: RasterMask
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right < left or bottom < top:
        return None
    return (
        (
            slice(top - first.top, bottom - first.top + 1),
            slice(left - first.left, right - first.left + 1),
        ),
        (
            slice(top - second.top, bottom - second.top + 1),
            slice(left - second.left, right - second.left + 1),
        ),
    )


def exact_mask_overlap(
    first: RasterMask | None, second: RasterMask | None
) -> MaskOverlapMetrics:
    if first is None or second is None or first.area <= 0 or second.area <= 0:
        return MaskOverlapMetrics(
            intersection=0,
            union=0,
            first_area=first.area if first is not None else 0,
            second_area=second.area if second is not None else 0,
        )
    slices = overlap_slices(first, second)
    if slices is None:
        return MaskOverlapMetrics(
            intersection=0,
            union=first.area + second.area,
            first_area=first.area,
            second_area=second.area,
        )
    first_slice, second_slice = slices
    intersection = int(
        np.count_nonzero(
            (first.mask[first_slice] != 0) & (second.mask[second_slice] != 0)
        )
    )
    return MaskOverlapMetrics(
        intersection=intersection,
        union=first.area + second.area - intersection,
        first_area=first.area,
        second_area=second.area,
    )


def exact_mask_iou(first: RasterMask | None, second: RasterMask | None) -> float:
    return exact_mask_overlap(first, second).iou


# Private aliases keep archived experiments readable while Production imports
# the explicit public names above.
_association_geometry = association_geometry
_bbox_area = bbox_area
_mask_area = mask_area
_overlap_slices = overlap_slices
_polygon_area = polygon_area
_RasterMask = RasterMask
_raster_mask = raster_mask


__all__ = (
    "MaskOverlapMetrics",
    "RasterMask",
    "association_geometry",
    "bbox_area",
    "exact_mask_iou",
    "exact_mask_overlap",
    "mask_area",
    "overlap_slices",
    "polygon_area",
    "raster_mask",
)
