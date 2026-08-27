"""Connected-component cleanup for canonical polygon detections.

The production candidate in this module is deliberately independent from NMS
retention.  It removes geometry that is redundant in the *raw instance set*;
which of those instances NMS should retain is a separate policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> bool:
    if abs(_cross(start, end, point)) > tolerance:
        return False
    return bool(
        min(float(start[0]), float(end[0])) - tolerance
        <= float(point[0])
        <= max(float(start[0]), float(end[0])) + tolerance
        and min(float(start[1]), float(end[1])) - tolerance
        <= float(point[1])
        <= max(float(start[1]), float(end[1])) + tolerance
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> bool:
    values = (
        _cross(first_start, first_end, second_start),
        _cross(first_start, first_end, second_end),
        _cross(second_start, second_end, first_start),
        _cross(second_start, second_end, first_end),
    )
    if (
        (values[0] > tolerance and values[1] < -tolerance)
        or (values[0] < -tolerance and values[1] > tolerance)
    ) and (
        (values[2] > tolerance and values[3] < -tolerance)
        or (values[2] < -tolerance and values[3] > tolerance)
    ):
        return True
    return any(
        abs(value) <= tolerance and _point_on_segment(point, start, end)
        for value, point, start, end in (
            (values[0], second_start, first_start, first_end),
            (values[1], second_end, first_start, first_end),
            (values[2], first_start, second_start, second_end),
            (values[3], first_end, second_start, second_end),
        )
    )


def _contours_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    for first_index in range(len(first)):
        first_start = first[first_index]
        first_end = first[(first_index + 1) % len(first)]
        for second_index in range(len(second)):
            if _segments_intersect(
                first_start,
                first_end,
                second[second_index],
                second[(second_index + 1) % len(second)],
            ):
                return True
    return False


def _polygon_relation(polygons: list[np.ndarray]) -> tuple[list[int], list[int | None]]:
    """Return nesting depth and the smallest *fully containing* parent.

    Testing only one child vertex can misclassify intersecting contours as a
    hole.  V3 normally contains proper nested contours, but the cleanup policy
    must fail safe for malformed/crossing polygons: such a contour remains an
    independent foreground component instead of being filled as a false hole.
    """
    areas = [abs(float(cv2.contourArea(polygon))) for polygon in polygons]
    bounds = [
        (
            float(np.min(polygon[:, 0])),
            float(np.min(polygon[:, 1])),
            float(np.max(polygon[:, 0])),
            float(np.max(polygon[:, 1])),
        )
        for polygon in polygons
    ]
    parents: list[int | None] = [None] * len(polygons)
    for child_index, child in enumerate(polygons):
        candidates: list[int] = []
        for parent_index, parent in enumerate(polygons):
            if parent_index == child_index or areas[parent_index] <= areas[child_index]:
                continue
            child_box = bounds[child_index]
            parent_box = bounds[parent_index]
            if not (
                parent_box[0] <= child_box[0]
                and parent_box[1] <= child_box[1]
                and parent_box[2] >= child_box[2]
                and parent_box[3] >= child_box[3]
            ):
                continue
            if all(
                cv2.pointPolygonTest(
                    parent,
                    (float(point[0]), float(point[1])),
                    False,
                )
                > 0
                for point in child
            ) and not _contours_intersect(parent, child):
                candidates.append(parent_index)
        if candidates:
            parents[child_index] = min(candidates, key=lambda index: areas[index])

    depths: list[int] = []
    for index in range(len(polygons)):
        depth = 0
        seen = {index}
        parent = parents[index]
        while parent is not None and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    return depths, parents


def _is_descendant(index: int, roots: set[int], parents: list[int | None]) -> bool:
    current: int | None = index
    seen: set[int] = set()
    while current is not None and current not in seen:
        if current in roots:
            return True
        seen.add(current)
        current = parents[current]
    return False


@dataclass(frozen=True)
class _Geometry:
    polygons: list[np.ndarray]
    depths: list[int]
    parents: list[int | None]
    areas: list[float]
    foreground: list[int]
    largest_foreground: int


@dataclass(frozen=True)
class ComponentCleanupStats:
    """Topology-only counters used by Production NMS diagnostics."""

    holes_filled: int = 0
    tiny_islands_removed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "holes_filled": self.holes_filled,
            "tiny_islands_removed": self.tiny_islands_removed,
        }


def _geometry(detection: dict[str, Any]) -> _Geometry | None:
    source = detection.get("polygons") or []
    if not source:
        return None
    polygons: list[np.ndarray] = []
    for polygon in source:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all():
            return None
        polygons.append(points)
    depths, parents = _polygon_relation(polygons)
    areas = [abs(float(cv2.contourArea(polygon))) for polygon in polygons]
    foreground = [index for index, depth in enumerate(depths) if depth % 2 == 0]
    if not foreground:
        return None
    largest = max(foreground, key=lambda index: areas[index])
    return _Geometry(polygons, depths, parents, areas, foreground, largest)


def _descendants(root: int, parents: list[int | None]) -> set[int]:
    return {
        index for index in range(len(parents)) if _is_descendant(index, {root}, parents)
    }


def _net_area(geometry: _Geometry, indices: set[int] | None = None) -> float:
    selected = range(len(geometry.polygons)) if indices is None else indices
    return max(
        0.0,
        sum(
            geometry.areas[index] * (1.0 if geometry.depths[index] % 2 == 0 else -1.0)
            for index in selected
        ),
    )


def _component_coverage(
    owner: _Geometry,
    root: int,
    other: _Geometry,
    *,
    other_root: int | None = None,
) -> float:
    """Return the visible component fraction covered by another instance.

    Rasterization is restricted to the component bounding box, so cost scales
    with the island rather than the full video frame.  Nesting parity preserves
    holes in both masks.
    """
    indices = _descendants(root, owner.parents)
    points = np.concatenate([owner.polygons[index] for index in indices], axis=0)
    low = np.floor(points.min(axis=0)).astype(int) - 1
    high = np.ceil(points.max(axis=0)).astype(int) + 1
    width = int(high[0] - low[0] + 1)
    height = int(high[1] - low[1] + 1)
    if width <= 0 or height <= 0:
        return 0.0

    component = np.zeros((height, width), np.uint8)
    for index in sorted(indices, key=lambda value: owner.depths[value]):
        contour = np.rint(owner.polygons[index] - low).astype(np.int32)
        value = 1 if owner.depths[index] % 2 == 0 else 0
        cv2.fillPoly(component, [contour], value)
    component_area = int(np.count_nonzero(component))
    if component_area <= 0:
        return 0.0

    other_mask = np.zeros_like(component)
    other_indices = (
        set(range(len(other.polygons)))
        if other_root is None
        else _descendants(other_root, other.parents)
    )
    for index in sorted(other_indices, key=lambda value: other.depths[value]):
        contour = np.rint(other.polygons[index] - low).astype(np.int32)
        value = 1 if other.depths[index] % 2 == 0 else 0
        cv2.fillPoly(other_mask, [contour], value)
    intersection = int(np.count_nonzero((component != 0) & (other_mask != 0)))
    return intersection / component_area


def _with_removed_components(
    detection: dict[str, Any], geometry: _Geometry, removed_roots: set[int]
) -> dict[str, Any]:
    if not removed_roots:
        return detection
    retained_indices = [
        index
        for index in range(len(geometry.polygons))
        if not _is_descendant(index, removed_roots, geometry.parents)
    ]
    if not retained_indices:
        retained_indices = [geometry.largest_foreground]
    retained = [
        geometry.polygons[index].astype(float).tolist() for index in retained_indices
    ]
    xs = [point[0] for polygon in retained for point in polygon]
    ys = [point[1] for polygon in retained for point in polygon]
    box = [min(xs), min(ys), max(xs), max(ys)]
    output = dict(detection)
    output["polygons"] = retained
    output["segmentation"] = retained
    output["bbox_xyxy"] = box
    output["bbox"] = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    output.pop("_bbox_area", None)
    output.pop("_mask_area", None)
    return output


def fill_holes_and_remove_tiny_islands(
    detections: list[dict[str, Any]],
    *,
    fill_all_holes: bool = True,
    unconditional_owner_ratio_max: float = 0.01,
) -> tuple[list[dict[str, Any]], ComponentCleanupStats]:
    """Apply only the topology cleanup that is safe *before* NMS.

    Holes are filled and secondary foreground components no larger than one
    percent of their owner's largest component are removed.  No decision in
    this function depends on another detection: cross-instance island cleanup
    must happen only after mask-IoU NMS has selected the surviving instances.

    Inputs are never mutated and no private metadata is written to a
    detection, preserving the canonical JSON/SQLite contract.
    """
    if not detections:
        return [], ComponentCleanupStats()
    cleaned: list[dict[str, Any]] = []
    holes_filled = 0
    tiny_islands_removed = 0
    for detection in detections:
        geometry = _geometry(detection)
        if geometry is None:
            cleaned.append(detection)
            continue
        removed_roots: set[int] = set()
        if fill_all_holes:
            holes = {
                index for index, depth in enumerate(geometry.depths) if depth % 2 == 1
            }
            removed_roots.update(holes)
            holes_filled += len(holes)
        main_area = geometry.areas[geometry.largest_foreground]
        if main_area > 0.0 and unconditional_owner_ratio_max >= 0.0:
            tiny = {
                index
                for index in geometry.foreground
                if index != geometry.largest_foreground
                and geometry.areas[index] / main_area <= unconditional_owner_ratio_max
            }
            removed_roots.update(tiny)
            tiny_islands_removed += len(tiny)
        cleaned.append(_with_removed_components(detection, geometry, removed_roots))
    return cleaned, ComponentCleanupStats(
        holes_filled=holes_filled,
        tiny_islands_removed=tiny_islands_removed,
    )
