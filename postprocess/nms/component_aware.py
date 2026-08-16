"""Mask-IoU NMS with connected-component-aware topology cleanup.

This is an opt-in Production candidate.  The existing adaptive bbox policy is
kept intact so both policies can be evaluated from the same raw detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .components import (
    _geometry,
    fill_holes_and_remove_tiny_islands,
    remove_redundant_surviving_islands,
)


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


def _association_geometry(detection: dict[str, Any]) -> dict[str, Any]:
    """Attach the geometry used only for temporal association.

    Survivor-island cleanup is a local final-mask correction.  Letting that
    small correction also alter greedy tracking can flip an otherwise stable
    assignment.  Preserve the geometry immediately before that cleanup while
    keeping the public polygons/bbox free to hold the cleaned final mask.
    These private fields are consumed by tracking and are never persisted to
    SQLite.
    """
    output: dict[str, Any] = {}
    bbox = detection.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        output["_association_bbox_xyxy"] = [float(value) for value in bbox]
    polygons = detection.get("polygons") or []
    output["_association_mask_area"] = float(
        sum(_polygon_area(polygon) for polygon in polygons)
    )
    return output


@dataclass(frozen=True)
class ComponentAwareNmsDiagnostics:
    input_detections: int = 0
    holes_filled: int = 0
    tiny_islands_removed: int = 0
    bbox_overlap_pairs: int = 0
    mask_iou_pairs: int = 0
    nms_suppressed: int = 0
    redundant_islands_removed: int = 0
    output_detections: int = 0

    def as_dict(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class _RasterMask:
    mask: np.ndarray
    left: int
    top: int
    right: int
    bottom: int
    area: int


@dataclass(frozen=True)
class MaskOverlapMetrics:
    """Exact native-pixel overlap between two rasterized masks."""

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


def _raster_mask(detection: dict[str, Any]) -> _RasterMask | None:
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
    # Parent contours must be drawn before their holes and nested foreground.
    for index in sorted(
        range(len(geometry.polygons)), key=lambda value: geometry.depths[value]
    ):
        contour = np.rint(geometry.polygons[index] - offset).astype(np.int32)
        value = 1 if geometry.depths[index] % 2 == 0 else 0
        cv2.fillPoly(mask, [contour], value)
    area = int(np.count_nonzero(mask))
    return _RasterMask(
        mask=mask,
        left=int(left),
        top=int(top),
        right=int(right),
        bottom=int(bottom),
        area=area,
    )


def _overlap_slices(
    first: _RasterMask, second: _RasterMask
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right < left or bottom < top:
        return None
    first_slice = (
        slice(top - first.top, bottom - first.top + 1),
        slice(left - first.left, right - first.left + 1),
    )
    second_slice = (
        slice(top - second.top, bottom - second.top + 1),
        slice(left - second.left, right - second.left + 1),
    )
    return first_slice, second_slice


def exact_mask_overlap(
    first: _RasterMask | None, second: _RasterMask | None
) -> MaskOverlapMetrics:
    """Return raster-exact overlap metrics in native pixel coordinates."""
    if first is None or second is None or first.area <= 0 or second.area <= 0:
        return MaskOverlapMetrics(
            intersection=0,
            union=0,
            first_area=first.area if first is not None else 0,
            second_area=second.area if second is not None else 0,
        )
    slices = _overlap_slices(first, second)
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
    union = first.area + second.area - intersection
    return MaskOverlapMetrics(
        intersection=intersection,
        union=union,
        first_area=first.area,
        second_area=second.area,
    )


def exact_mask_iou(first: _RasterMask | None, second: _RasterMask | None) -> float:
    """Return raster-exact mask IoU in the masks' native pixel coordinates."""
    return exact_mask_overlap(first, second).iou


@dataclass(frozen=True)
class ComponentAwareMaskNms:
    """Frozen 2026-08-13 topology/NMS Production candidate.

    Order of operations:

    1. fill every hole and remove owner-relative islands <=1%;
    2. greedy, score-ordered full-instance mask-IoU NMS;
    3. among survivors only, remove redundant virtual islands by 80% coverage
       and 50% island/covering-instance area thresholds.

    Bboxes are used only as a broad-phase overlap test.  They never suppress a
    detection.  Output detections retain the canonical schema.
    """

    name: str = "component_aware_mask_candidate_v2"
    mask_iou_threshold: float = 0.70
    fill_all_holes: bool = True
    unconditional_owner_ratio_max: float = 0.01
    island_other_coverage_min: float = 0.80
    island_to_other_area_max: float = 0.50

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retained, _ = self.apply_with_diagnostics(detections)
        return retained

    def apply_with_diagnostics(
        self, detections: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], ComponentAwareNmsDiagnostics]:
        if not detections:
            return [], ComponentAwareNmsDiagnostics()
        preprocessed, topology = fill_holes_and_remove_tiny_islands(
            detections,
            fill_all_holes=self.fill_all_holes,
            unconditional_owner_ratio_max=self.unconditional_owner_ratio_max,
        )
        if len(preprocessed) == 1:
            return preprocessed, ComponentAwareNmsDiagnostics(
                input_detections=1,
                holes_filled=topology.holes_filled,
                tiny_islands_removed=topology.tiny_islands_removed,
                output_detections=1,
            )
        rasters = [_raster_mask(detection) for detection in preprocessed]
        order = sorted(
            range(len(preprocessed)),
            key=lambda index: (
                -float(preprocessed[index].get("score") or 0.0),
                index,
            ),
        )
        suppressed: set[int] = set()
        retained_indices: list[int] = []
        bbox_overlap_pairs = 0
        mask_iou_pairs = 0
        for position, index in enumerate(order):
            if index in suppressed:
                continue
            retained_indices.append(index)
            first = rasters[index]
            if first is None or first.area <= 0:
                continue
            for other in order[position + 1 :]:
                if other in suppressed:
                    continue
                second = rasters[other]
                if second is None or second.area <= 0:
                    continue
                if _overlap_slices(first, second) is None:
                    continue
                bbox_overlap_pairs += 1
                mask_iou_pairs += 1
                if exact_mask_iou(first, second) >= self.mask_iou_threshold:
                    suppressed.add(other)

        survivors = [preprocessed[index] for index in retained_indices]
        cleaned, component_stats = remove_redundant_surviving_islands(
            survivors,
            other_coverage_min=self.island_other_coverage_min,
            island_to_other_area_max=self.island_to_other_area_max,
        )
        # Most detections are unchanged by survivor-island cleanup.  Add the
        # private association hint only to masks whose public geometry really
        # changed, keeping the streaming JSONL overhead negligible.
        cleaned = [
            after if after is before else {**after, **_association_geometry(before)}
            for before, after in zip(survivors, cleaned, strict=True)
        ]
        return cleaned, ComponentAwareNmsDiagnostics(
            input_detections=len(detections),
            holes_filled=topology.holes_filled,
            tiny_islands_removed=topology.tiny_islands_removed,
            bbox_overlap_pairs=bbox_overlap_pairs,
            mask_iou_pairs=mask_iou_pairs,
            nms_suppressed=len(suppressed),
            redundant_islands_removed=(component_stats.redundant_islands_removed),
            output_detections=len(cleaned),
        )


DEFAULT_COMPONENT_AWARE_NMS = ComponentAwareMaskNms()
