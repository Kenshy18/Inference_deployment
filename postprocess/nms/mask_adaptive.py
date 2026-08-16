"""Size-aware exact-mask suppression policy used by Production NMS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mask_geometry import (
    MaskOverlapMetrics,
    bbox_area,
    exact_mask_overlap,
    mask_area,
    overlap_slices,
    raster_mask,
)


@dataclass(frozen=True)
class AdaptiveMaskNms:
    """Score-ordered, size-aware NMS using exact mask geometry.

    ``containment_coverage_min`` catches a mask that is nearly contained in a
    larger mask even when symmetric IoU is low.  The size-dependent maximum
    large/small ratio mirrors Production's adaptive containment guard.
    """

    name: str = "adaptive_mask_v1"
    iou_threshold: float = 0.20
    small_iou_threshold: float = 0.10
    tiny_iou_threshold: float = 0.05
    small_area: float = 5000.0
    tiny_area: float = 2000.0
    containment_coverage_min: float = 0.80
    contain_ratio_max: float = 8.0
    small_contain_ratio_max: float = 5.0
    tiny_contain_ratio_max: float = 5.0

    def thresholds_for_area(self, area: float) -> tuple[float, float]:
        if area <= self.tiny_area:
            return self.tiny_iou_threshold, self.tiny_contain_ratio_max
        if area <= self.small_area:
            return self.small_iou_threshold, self.small_contain_ratio_max
        return self.iou_threshold, self.contain_ratio_max

    def suppression_reason_from_metrics(
        self,
        metrics: MaskOverlapMetrics,
        *,
        threshold_area: float | None = None,
    ) -> str | None:
        if metrics.intersection <= 0:
            return None
        smaller_area = (
            min(metrics.first_area, metrics.second_area)
            if threshold_area is None
            else float(threshold_area)
        )
        threshold, contain_limit = self.thresholds_for_area(float(smaller_area))
        raster_smaller_area = min(metrics.first_area, metrics.second_area)
        larger_to_smaller = (
            max(metrics.first_area, metrics.second_area) / raster_smaller_area
            if raster_smaller_area > 0
            else float("inf")
        )
        if (
            metrics.smaller_coverage >= self.containment_coverage_min
            and larger_to_smaller <= contain_limit
        ):
            return "mask_contained"
        if metrics.iou >= threshold:
            return "mask_iou"
        return None

    @staticmethod
    def detection_threshold_area(detection: dict[str, Any]) -> float:
        """Return the size used by validated Production's adaptive bands.

        Exact mask comparisons use native-pixel rasters, but the historical
        2,000/5,000 px size bands are defined from continuous contour/bbox
        geometry.  Using raster area for those bands can move a half-pixel
        contour across a band boundary and create an isolated NMS decision.
        """
        box_area = bbox_area(detection)
        polygon_area = mask_area(detection)
        return min(box_area, polygon_area) if polygon_area > 0.0 else box_area

    def pair_threshold_area(
        self, first: dict[str, Any], second: dict[str, Any]
    ) -> float:
        return min(
            self.detection_threshold_area(first),
            self.detection_threshold_area(second),
        )

    def pair_metrics(
        self, first: dict[str, Any], second: dict[str, Any]
    ) -> MaskOverlapMetrics:
        return exact_mask_overlap(raster_mask(first), raster_mask(second))

    def pair_suppression_reason(
        self, first: dict[str, Any], second: dict[str, Any]
    ) -> str | None:
        return self.suppression_reason_from_metrics(
            self.pair_metrics(first, second),
            threshold_area=self.pair_threshold_area(first, second),
        )

    def apply(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return []
        rasters = [raster_mask(detection) for detection in detections]
        threshold_areas = [
            self.detection_threshold_area(detection) for detection in detections
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
            first = rasters[index]
            if first is None or first.area <= 0:
                continue
            for other in order[position + 1 :]:
                if other in suppressed:
                    continue
                second = rasters[other]
                if second is None or second.area <= 0:
                    continue
                if overlap_slices(first, second) is None:
                    continue
                metrics = exact_mask_overlap(first, second)
                if (
                    self.suppression_reason_from_metrics(
                        metrics,
                        threshold_area=min(
                            threshold_areas[index], threshold_areas[other]
                        ),
                    )
                    is not None
                ):
                    suppressed.add(other)
        return [detections[index] for index in retained]


DEFAULT_ADAPTIVE_MASK_NMS = AdaptiveMaskNms()
