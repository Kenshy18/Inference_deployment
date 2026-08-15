"""Greedy temporal association for canonical mask detections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionFeatures:
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    area: float
    aspect: float
    polygon_area: float | None
    fill_ratio: float | None


@dataclass
class TrackState:
    track_id: int
    scene_id: int
    last_frame: int
    features: DetectionFeatures

    def update(self, frame_index: int, features: DetectionFeatures) -> None:
        self.last_frame = frame_index
        self.features = features


@dataclass(frozen=True)
class AssociationConfig:
    max_gap_frames: int = 15
    iou_min: float = 0.06
    center_distance_max: float = 0.50
    area_ratio_min: float = 0.20
    area_ratio_max: float = 5.0
    aspect_log_difference_max: float = 1.30
    fill_ratio_difference_max: float = 0.67
    polygon_ratio_min: float = 0.25
    polygon_ratio_max: float = 4.0
    score_min: float = 0.20
    small_area: float = 5000.0
    tiny_area: float = 2000.0
    small_iou_min: float = 0.03
    tiny_iou_min: float = 0.01
    small_center_distance_max: float = 0.80
    tiny_center_distance_max: float = 1.00
    small_area_ratio_min: float = 0.20
    small_area_ratio_max: float = 6.0
    tiny_area_ratio_min: float = 0.10
    tiny_area_ratio_max: float = 7.0
    small_aspect_log_difference_max: float = 1.50
    tiny_aspect_log_difference_max: float = 1.70
    small_score_min: float = 0.15
    tiny_score_min: float = 0.12


def detection_features(detection: dict[str, Any]) -> DetectionFeatures:
    association_bbox = detection.get(
        "_association_bbox_xyxy", detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    )
    x1, y1, x2, y2 = map(float, association_bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    polygon_area = (
        float(
            detection.get("_association_mask_area", detection.get("_mask_area")) or 0.0
        )
        or None
    )
    return DetectionFeatures(
        bbox=(x1, y1, x2, y2),
        center=(x1 + width * 0.5, y1 + height * 0.5),
        area=area,
        aspect=width / height if height > 0.0 else 0.0,
        polygon_area=polygon_area,
        fill_ratio=polygon_area / area
        if polygon_area is not None and area > 0
        else None,
    )


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-6)


def match_score(
    track: TrackState,
    detection: DetectionFeatures,
    frame_index: int,
    config: AssociationConfig,
) -> float | None:
    gap = frame_index - track.last_frame
    previous = track.features
    if gap < 0 or gap > config.max_gap_frames:
        return None
    if previous.area <= 0.0 or detection.area <= 0.0:
        return None
    area_reference = 0.5 * (previous.area + detection.area)
    iou_min = config.iou_min
    center_distance_max = config.center_distance_max
    area_ratio_min = config.area_ratio_min
    area_ratio_max = config.area_ratio_max
    aspect_difference_max = config.aspect_log_difference_max
    score_min = config.score_min
    compare_polygon_shape = True
    if area_reference <= config.tiny_area:
        iou_min = config.tiny_iou_min
        center_distance_max = config.tiny_center_distance_max
        area_ratio_min = config.tiny_area_ratio_min
        area_ratio_max = config.tiny_area_ratio_max
        aspect_difference_max = config.tiny_aspect_log_difference_max
        score_min = config.tiny_score_min
        compare_polygon_shape = False
    elif area_reference <= config.small_area:
        iou_min = config.small_iou_min
        center_distance_max = config.small_center_distance_max
        area_ratio_min = config.small_area_ratio_min
        area_ratio_max = config.small_area_ratio_max
        aspect_difference_max = config.small_aspect_log_difference_max
        score_min = config.small_score_min
        compare_polygon_shape = False

    iou = _iou(previous.bbox, detection.bbox)
    distance = math.hypot(
        detection.center[0] - previous.center[0],
        detection.center[1] - previous.center[1],
    )
    previous_diagonal = math.hypot(
        previous.bbox[2] - previous.bbox[0],
        previous.bbox[3] - previous.bbox[1],
    )
    current_diagonal = math.hypot(
        detection.bbox[2] - detection.bbox[0],
        detection.bbox[3] - detection.bbox[1],
    )
    normalized_distance = distance / (
        0.5 * (previous_diagonal + current_diagonal) + 1e-6
    )
    if iou < iou_min and normalized_distance > center_distance_max:
        return None

    area_ratio = detection.area / previous.area
    if not area_ratio_min <= area_ratio <= area_ratio_max:
        return None
    if previous.aspect <= 0.0 or detection.aspect <= 0.0:
        return None
    aspect_difference = abs(math.log(detection.aspect / previous.aspect))
    if aspect_difference > aspect_difference_max:
        return None

    fill_score = 0.5
    if (
        compare_polygon_shape
        and previous.fill_ratio is not None
        and detection.fill_ratio is not None
    ):
        difference = abs(detection.fill_ratio - previous.fill_ratio)
        if difference > config.fill_ratio_difference_max:
            return None
        fill_score = max(0.0, 1.0 - difference / config.fill_ratio_difference_max)
    if (
        compare_polygon_shape
        and previous.polygon_area is not None
        and detection.polygon_area is not None
    ):
        polygon_ratio = detection.polygon_area / max(previous.polygon_area, 1e-6)
        if not config.polygon_ratio_min <= polygon_ratio <= config.polygon_ratio_max:
            return None

    center_score = max(0.0, 1.0 - normalized_distance / center_distance_max)
    aspect_score = max(0.0, 1.0 - aspect_difference / aspect_difference_max)
    score = 0.5 * iou + 0.3 * center_score + 0.15 * aspect_score + 0.05 * fill_score
    return score if score >= score_min else None


def associate(
    active_tracks: list[TrackState],
    detections: list[DetectionFeatures],
    frame_index: int,
    config: AssociationConfig,
) -> dict[int, int]:
    """Return ``detection index -> existing track id`` assignments."""

    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for track in active_tracks:
            score = match_score(track, detection, frame_index, config)
            if score is not None:
                candidates.append((score, track.track_id, detection_index))
    candidates.sort(reverse=True)
    assignments: dict[int, int] = {}
    assigned_tracks: set[int] = set()
    for _score, track_id, detection_index in candidates:
        if detection_index in assignments or track_id in assigned_tracks:
            continue
        assignments[detection_index] = track_id
        assigned_tracks.add(track_id)
    return assignments
