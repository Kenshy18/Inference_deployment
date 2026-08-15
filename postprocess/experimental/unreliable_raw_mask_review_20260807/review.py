#!/usr/bin/env python3
"""Rank and locally render suspicious raw segmentation masks.

Candidate scoring reads only SQLite geometry and metadata.  When a source video
is supplied, OpenCV decodes frames locally to create a human-review artifact.
No pixels leave the machine and this module is not connected to production.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from shapely import affinity, make_valid
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class Observation:
    detection_id: int
    frame: int
    timestamp_sec: float
    scene_id: int
    raw_track_id: str
    raw_track_length: int
    removed_by_short_track: bool
    label: str
    score: float
    geometry: object
    component_count: int
    satellite_ratio: float
    area: float
    center_x: float
    center_y: float
    pca_angle: float
    touches_border: bool


@dataclass(frozen=True)
class Candidate:
    detection_id: int
    frame: int
    timestamp_sec: float
    timecode: str
    scene_id: int
    raw_track_id: str
    raw_track_length: int
    removed_by_short_track: bool
    label: str
    score: float
    area: float
    component_count: int
    satellite_ratio: float
    touches_border: bool
    neighbour_count: int
    temporal_iou: float | None
    temporal_added_ratio: float | None
    temporal_missing_ratio: float | None
    area_ratio: float | None
    center_residual_norm: float | None
    angle_residual_deg: float | None
    hausdorff_norm: float | None
    risk_score: float
    primary_reason: str
    reasons: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--labels", default="男性器,女性器,結合部分")
    parser.add_argument("--risk-threshold", type=float, default=1.35)
    parser.add_argument("--max-candidates", type=int, default=90)
    parser.add_argument("--minimum-event-gap", type=int, default=12)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--review-fps", type=float, default=15.0)
    parser.add_argument(
        "--kept-only",
        action="store_true",
        help=(
            "Restrict analysis to observations retained by short-track filtering "
            "and assigned to a final track"
        ),
    )
    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        help="Write one chronological before/target/after image per candidate",
    )
    parser.add_argument("--contact-context-frames", type=int, default=4)
    return parser.parse_args()


def _polygonal(value):
    if value.is_empty:
        return GeometryCollection()
    valid = make_valid(value)
    if isinstance(valid, (Polygon, MultiPolygon)):
        return valid
    parts = [
        part
        for part in getattr(valid, "geoms", ())
        if isinstance(part, (Polygon, MultiPolygon))
    ]
    return unary_union(parts) if parts else GeometryCollection()


def _geometry(arrays: list[np.ndarray]):
    values = []
    for points in arrays:
        if len(points) >= 3:
            polygon = _polygonal(Polygon(points))
            if not polygon.is_empty and polygon.area > 0.0:
                values.append(polygon)
    return _polygonal(unary_union(values)) if values else GeometryCollection()


def _angle(geometry) -> float:
    if geometry.is_empty:
        return 0.0
    polygon = (
        geometry
        if isinstance(geometry, Polygon)
        else max(geometry.geoms, key=lambda value: float(value.area))
    )
    points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if len(points) < 3:
        return 0.0
    centered = points - np.asarray([geometry.centroid.x, geometry.centroid.y])
    covariance = np.cov(centered, rowvar=False)
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(_values))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _axis_distance(left: float, right: float) -> float:
    values = [abs(left - right + offset) for offset in (-math.pi, 0.0, math.pi)]
    return min(values)


def _load_observations(
    path: Path,
    *,
    labels: set[str],
    start_frame: int,
    end_frame: int | None,
    kept_only: bool,
) -> tuple[list[Observation], dict[str, float]]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        video = connection.execute(
            "SELECT fps, width, height FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
        if video is None:
            raise ValueError("SQLite has no videos row")
        fps, width, height = float(video[0]), int(video[1]), int(video[2])
        upper = (
            int(end_frame)
            if end_frame is not None
            else int(
                connection.execute("SELECT MAX(frame_index) FROM frames").fetchone()[0]
            )
        )
        placeholders = ",".join("?" for _ in labels)
        kept_clause = (
            "AND a.removed_by_short_track=0 AND a.final_track_id IS NOT NULL"
            if kept_only
            else ""
        )
        rows = connection.execute(
            f"""
            SELECT d.id, f.frame_index, f.timestamp_sec, a.scene_id,
                   a.raw_track_id, a.raw_track_length,
                   a.removed_by_short_track, a.raw_label,
                   COALESCE(a.selected_score, d.score),
                   sp.polygon_index, pt.point_index, pt.x, pt.y
            FROM tracking_assignments a
            JOIN detections d ON d.id=a.source_detection_id
            JOIN frames f ON f.id=d.frame_id
            JOIN segmentation_polygons sp ON sp.detection_id=d.id
            JOIN segmentation_points pt ON pt.polygon_id=sp.id
            WHERE a.raw_label IN ({placeholders})
              AND f.frame_index BETWEEN ? AND ?
              {kept_clause}
            ORDER BY d.id, sp.polygon_index, pt.point_index
            """,
            (*sorted(labels), int(start_frame), upper),
        )

        grouped: dict[int, dict[str, object]] = {}
        for row in rows:
            detection_id = int(row[0])
            value = grouped.setdefault(
                detection_id,
                {
                    "frame": int(row[1]),
                    "timestamp": float(row[2]),
                    "scene": int(row[3]),
                    "track": str(row[4]),
                    "length": int(row[5]),
                    "removed": bool(row[6]),
                    "label": str(row[7]),
                    "score": float(row[8]),
                    "polygons": defaultdict(list),
                },
            )
            value["polygons"][int(row[9])].append((float(row[11]), float(row[12])))

    observations: list[Observation] = []
    for detection_id, value in grouped.items():
        arrays = [
            np.asarray(points, dtype=np.float64)
            for _index, points in sorted(value["polygons"].items())
        ]
        geometry = _geometry(arrays)
        if geometry.is_empty or geometry.area <= 0.0:
            continue
        component_areas = sorted(
            [float(abs(Polygon(points).area)) for points in arrays], reverse=True
        )
        satellite = (
            sum(component_areas[1:]) / max(sum(component_areas), 1e-9)
            if len(component_areas) > 1
            else 0.0
        )
        min_x, min_y, max_x, max_y = geometry.bounds
        observations.append(
            Observation(
                detection_id=detection_id,
                frame=int(value["frame"]),
                timestamp_sec=float(value["timestamp"]),
                scene_id=int(value["scene"]),
                raw_track_id=str(value["track"]),
                raw_track_length=int(value["length"]),
                removed_by_short_track=bool(value["removed"]),
                label=str(value["label"]),
                score=float(value["score"]),
                geometry=geometry,
                component_count=len(arrays),
                satellite_ratio=float(satellite),
                area=float(geometry.area),
                center_x=float(geometry.centroid.x),
                center_y=float(geometry.centroid.y),
                pca_angle=_angle(geometry),
                touches_border=(
                    min_x <= 10.0
                    or min_y <= 10.0
                    or max_x >= width - 11.0
                    or max_y >= height - 11.0
                ),
            )
        )
    return observations, {"fps": fps, "width": width, "height": height}


def _translate_center(geometry, x: float, y: float):
    return affinity.translate(
        geometry,
        xoff=float(x) - float(geometry.centroid.x),
        yoff=float(y) - float(geometry.centroid.y),
    )


def _iou(left, right) -> float:
    intersection = float(left.intersection(right).area)
    union = float(left.area + right.area - intersection)
    return intersection / union if union else 1.0


def _timecode(frame: int, fps: float) -> str:
    nominal = max(1, round(fps))
    seconds, part = divmod(int(frame), nominal)
    hours, remainder = divmod(seconds, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}:{part:02d}"


def _score_track(values: list[Observation], fps: float) -> list[Candidate]:
    values = sorted(values, key=lambda value: (value.frame, value.detection_id))
    output: list[Candidate] = []
    for index, current in enumerate(values):
        neighbours = [
            item
            for item in values
            if item.detection_id != current.detection_id
            and 0 < abs(item.frame - current.frame) <= 3
        ]
        previous = [item for item in neighbours if item.frame < current.frame]
        following = [item for item in neighbours if item.frame > current.frame]
        temporal_iou = None
        added_ratio = None
        missing_ratio = None
        area_ratio = None
        center_residual = None
        angle_residual = None
        hausdorff_norm = None
        if previous and following:
            left = max(previous, key=lambda item: item.frame)
            right = min(following, key=lambda item: item.frame)
            alpha = (current.frame - left.frame) / max(right.frame - left.frame, 1)
            expected_x = (1.0 - alpha) * left.center_x + alpha * right.center_x
            expected_y = (1.0 - alpha) * left.center_y + alpha * right.center_y
            aligned_left = _translate_center(left.geometry, expected_x, expected_y)
            aligned_right = _translate_center(right.geometry, expected_x, expected_y)
            support_union = _polygonal(aligned_left.union(aligned_right))
            support_core = _polygonal(aligned_left.intersection(aligned_right))
            current_centered = _translate_center(
                current.geometry, expected_x, expected_y
            )
            temporal_iou = median(
                [_iou(current_centered, aligned_left), _iou(current_centered, aligned_right)]
            )
            support_area = max(float(support_union.area), 1e-9)
            core_area = max(float(support_core.area), 1e-9)
            added_ratio = float(current_centered.difference(support_union).area) / support_area
            missing_ratio = float(support_core.difference(current_centered).area) / core_area
            expected_area = math.exp(
                (1.0 - alpha) * math.log(max(left.area, 1e-9))
                + alpha * math.log(max(right.area, 1e-9))
            )
            area_ratio = current.area / max(expected_area, 1e-9)
            center_residual = math.hypot(
                current.center_x - expected_x, current.center_y - expected_y
            ) / math.sqrt(max(expected_area, 1.0))
            right_angle = min(
                (right.pca_angle - math.pi, right.pca_angle, right.pca_angle + math.pi),
                key=lambda value: abs(value - left.pca_angle),
            )
            expected_angle = (1.0 - alpha) * left.pca_angle + alpha * right_angle
            angle_residual = math.degrees(
                _axis_distance(current.pca_angle, expected_angle)
            )
            hausdorff_norm = float(current_centered.hausdorff_distance(support_core)) / math.sqrt(
                max(expected_area, 1.0)
            )

        reasons: list[str] = []
        risk = 0.0
        if temporal_iou is not None:
            risk += 2.2 * min(1.5, max(0.0, 0.78 - temporal_iou) / 0.48)
            if temporal_iou < 0.60:
                reasons.append("low_temporal_iou")
        if area_ratio is not None:
            log_change = abs(math.log(max(area_ratio, 1e-6)))
            risk += 1.35 * min(1.5, log_change / math.log(1.55))
            if area_ratio >= 1.35:
                reasons.append("sudden_expansion")
            elif area_ratio <= 0.74:
                reasons.append("sudden_contraction")
        if added_ratio is not None and added_ratio >= 0.18:
            risk += min(1.2, added_ratio)
            reasons.append("unsupported_added_area")
        if missing_ratio is not None and missing_ratio >= 0.20:
            risk += 0.65 * min(1.5, missing_ratio)
            reasons.append("missing_supported_area")
        if center_residual is not None and center_residual >= 0.18:
            risk += min(1.2, center_residual)
            reasons.append("centroid_jump")
        if angle_residual is not None and angle_residual >= 28.0:
            risk += 0.45 * min(1.5, angle_residual / 45.0)
            reasons.append("axis_jump")
        if hausdorff_norm is not None and hausdorff_norm >= 0.35:
            risk += 0.45 * min(1.5, hausdorff_norm)
            reasons.append("boundary_jump")
        if current.score < 0.50:
            risk += 0.55 * min(1.0, (0.50 - current.score) / 0.20)
            reasons.append("low_confidence")
        if current.removed_by_short_track:
            risk += 1.15
            reasons.append("short_track_removed")
        if current.component_count > 1 and current.satellite_ratio >= 0.04:
            risk += 0.35
            reasons.append("multiple_components")
        if current.touches_border and (
            temporal_iou is None or temporal_iou < 0.72
        ):
            risk += 0.55
            reasons.append("border_unstable")
        if not reasons:
            reasons.append("temporally_supported")

        if "short_track_removed" in reasons and temporal_iou is None:
            primary = "short_track_low_context"
        elif "sudden_expansion" in reasons or "unsupported_added_area" in reasons:
            primary = "possible_false_expansion"
        elif "sudden_contraction" in reasons or "missing_supported_area" in reasons:
            primary = "possible_false_contraction"
        elif "centroid_jump" in reasons:
            primary = "spatial_jump"
        elif "border_unstable" in reasons:
            primary = "border_instability"
        elif "low_temporal_iou" in reasons:
            primary = "mixed_shape_outlier"
        else:
            primary = reasons[0]

        output.append(
            Candidate(
                detection_id=current.detection_id,
                frame=current.frame,
                timestamp_sec=current.timestamp_sec,
                timecode=_timecode(current.frame, fps),
                scene_id=current.scene_id,
                raw_track_id=current.raw_track_id,
                raw_track_length=current.raw_track_length,
                removed_by_short_track=current.removed_by_short_track,
                label=current.label,
                score=current.score,
                area=current.area,
                component_count=current.component_count,
                satellite_ratio=current.satellite_ratio,
                touches_border=current.touches_border,
                neighbour_count=len(neighbours),
                temporal_iou=temporal_iou,
                temporal_added_ratio=added_ratio,
                temporal_missing_ratio=missing_ratio,
                area_ratio=area_ratio,
                center_residual_norm=center_residual,
                angle_residual_deg=angle_residual,
                hausdorff_norm=hausdorff_norm,
                risk_score=float(risk),
                primary_reason=primary,
                reasons="|".join(dict.fromkeys(reasons)),
            )
        )
    return output


def score_observations(
    observations: list[Observation], fps: float
) -> list[Candidate]:
    groups: dict[tuple[int, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        groups[(observation.scene_id, observation.raw_track_id)].append(observation)
    output: list[Candidate] = []
    for values in groups.values():
        output.extend(_score_track(values, fps))
    return sorted(output, key=lambda value: (-value.risk_score, value.frame))


def select_candidates(
    candidates: list[Candidate],
    *,
    threshold: float,
    maximum: int,
    minimum_gap: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    per_reason: dict[str, int] = defaultdict(int)
    reason_caps = {
        "possible_false_expansion": 30,
        "possible_false_contraction": 18,
        "mixed_shape_outlier": 18,
        "spatial_jump": 14,
        "border_instability": 12,
        "short_track_low_context": 20,
    }
    for candidate in candidates:
        if candidate.risk_score < threshold:
            continue
        cap = reason_caps.get(candidate.primary_reason, 12)
        if per_reason[candidate.primary_reason] >= cap:
            continue
        if any(
            candidate.scene_id == prior.scene_id
            and candidate.raw_track_id == prior.raw_track_id
            and abs(candidate.frame - prior.frame) < minimum_gap
            for prior in selected
        ):
            continue
        selected.append(candidate)
        per_reason[candidate.primary_reason] += 1
        if len(selected) >= maximum:
            break
    return sorted(selected, key=lambda value: value.frame)


def _polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [
        part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon)
    ]


def _contours(geometry) -> list[np.ndarray]:
    output = []
    for polygon in _polygons(geometry):
        points = np.rint(np.asarray(polygon.exterior.coords[:-1])).astype(np.int32)
        if len(points) >= 3:
            output.append(points.reshape(-1, 1, 2))
    return output


def _draw_geometry(
    image: np.ndarray,
    geometry,
    *,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    contours = _contours(geometry)
    if not contours:
        return
    if alpha > 0.0:
        layer = image.copy()
        cv2.fillPoly(layer, contours, color)
        cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, image)
    cv2.polylines(image, contours, True, color, thickness, cv2.LINE_AA)


def _candidate_support(
    candidate: Candidate,
    by_track: dict[tuple[int, str], list[Observation]],
):
    values = by_track[(candidate.scene_id, candidate.raw_track_id)]
    current = next(
        item for item in values if item.detection_id == candidate.detection_id
    )
    previous = [
        item for item in values if 0 < current.frame - item.frame <= 3
    ]
    following = [
        item for item in values if 0 < item.frame - current.frame <= 3
    ]
    if not previous or not following:
        return GeometryCollection()
    left = max(previous, key=lambda item: item.frame)
    right = min(following, key=lambda item: item.frame)
    alpha = (current.frame - left.frame) / max(right.frame - left.frame, 1)
    x = (1.0 - alpha) * left.center_x + alpha * right.center_x
    y = (1.0 - alpha) * left.center_y + alpha * right.center_y
    return _polygonal(
        _translate_center(left.geometry, x, y).union(
            _translate_center(right.geometry, x, y)
        )
    )


def render_review(
    video: Path,
    output: Path,
    selected: list[Candidate],
    observations: list[Observation],
    *,
    context_frames: int,
    review_fps: float,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    width, height = 960, 540
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), review_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {output}")
    by_detection = {item.detection_id: item for item in observations}
    by_track: dict[tuple[int, str], list[Observation]] = defaultdict(list)
    by_frame_track: dict[tuple[int, int, str], list[Observation]] = defaultdict(list)
    for item in observations:
        by_track[(item.scene_id, item.raw_track_id)].append(item)
        by_frame_track[(item.frame, item.scene_id, item.raw_track_id)].append(item)
    frames_written = 0
    for rank, candidate in enumerate(selected, start=1):
        target = by_detection[candidate.detection_id]
        support = _candidate_support(candidate, by_track)
        first = max(0, candidate.frame - context_frames)
        last = candidate.frame + context_frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame_index in range(first, last + 1):
            ok, image = capture.read()
            if not ok:
                break
            if frame_index == candidate.frame:
                _draw_geometry(
                    image, support, color=(255, 255, 0), alpha=0.0, thickness=5
                )
                _draw_geometry(
                    image, target.geometry, color=(20, 20, 255), alpha=0.48, thickness=5
                )
            else:
                for item in by_frame_track.get(
                    (frame_index, candidate.scene_id, candidate.raw_track_id), ()
                ):
                    _draw_geometry(
                        image,
                        item.geometry,
                        color=(80, 200, 255),
                        alpha=0.20,
                        thickness=3,
                    )
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (width, 104), (8, 8, 8), -1)
            lines = [
                f"CANDIDATE {rank:03d}/{len(selected):03d}  frame={candidate.frame}  TC={candidate.timecode}",
                f"det={candidate.detection_id} track={candidate.raw_track_id} label={candidate.label} score={candidate.score:.3f} risk={candidate.risk_score:.2f}",
                f"{candidate.primary_reason}  area_ratio={_fmt(candidate.area_ratio)} temp_iou={_fmt(candidate.temporal_iou)} added={_fmt(candidate.temporal_added_ratio)}",
            ]
            for line_index, line in enumerate(lines):
                cv2.putText(
                    image,
                    line,
                    (14, 29 + 32 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(image)
            frames_written += 1
        # Add a brief black separator to make event boundaries obvious.
        separator = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(1, round(review_fps * 0.20))):
            writer.write(separator)
            frames_written += 1
    writer.release()
    capture.release()
    return {
        "path": str(output),
        "candidate_count": len(selected),
        "frames_written": frames_written,
        "review_fps": review_fps,
        "duration_seconds": frames_written / max(review_fps, 1e-9),
    }


def render_contact_sheets(
    video: Path,
    output_dir: Path,
    selected: list[Candidate],
    observations: list[Observation],
    *,
    context_frames: int,
    fps: float,
) -> dict[str, object]:
    """Render chronological 3-column contact sheets for human review."""

    if context_frames < 1:
        raise ValueError("contact context frames must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    panel_width, panel_height = 640, 360
    columns = 3
    panel_count = 2 * context_frames + 1
    rows = int(math.ceil(panel_count / columns))
    header_height = 112
    by_frame_track: dict[tuple[int, int, str], list[Observation]] = defaultdict(list)
    for item in observations:
        by_frame_track[(item.frame, item.scene_id, item.raw_track_id)].append(item)
    written: list[dict[str, object]] = []
    for rank, candidate in enumerate(selected, start=1):
        first = max(0, candidate.frame - context_frames)
        requested = list(range(first, candidate.frame + context_frames + 1))
        canvas = np.zeros(
            (header_height + rows * panel_height, columns * panel_width, 3),
            dtype=np.uint8,
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        decoded = 0
        for panel_index, frame_index in enumerate(requested):
            ok, frame = capture.read()
            if not ok:
                break
            values = by_frame_track.get(
                (frame_index, candidate.scene_id, candidate.raw_track_id), ()
            )
            for item in values:
                is_target = item.detection_id == candidate.detection_id
                _draw_geometry(
                    frame,
                    item.geometry,
                    color=(20, 20, 255) if is_target else (40, 210, 255),
                    alpha=0.50 if is_target else 0.34,
                    thickness=5 if is_target else 3,
                )
            panel = cv2.resize(
                frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(panel, (0, 0), (panel_width, 38), (8, 8, 8), -1)
            offset = frame_index - candidate.frame
            prefix = "TARGET" if offset == 0 else f"t{offset:+d}"
            cv2.putText(
                panel,
                f"{prefix}  frame={frame_index}  TC={_timecode(frame_index, fps)}",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if offset == 0:
                cv2.rectangle(
                    panel,
                    (2, 2),
                    (panel_width - 3, panel_height - 3),
                    (20, 20, 255),
                    5,
                )
            row_index, column_index = divmod(panel_index, columns)
            y0 = header_height + row_index * panel_height
            x0 = column_index * panel_width
            canvas[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
            decoded += 1

        header_lines = [
            f"CANDIDATE {rank:03d}/{len(selected):03d}  target={candidate.timecode}  frame={candidate.frame}  detection={candidate.detection_id}",
            f"track={candidate.raw_track_id} length={candidate.raw_track_length} label={candidate.label} score={candidate.score:.3f} risk={candidate.risk_score:.2f}",
            f"{candidate.primary_reason}  area_ratio={_fmt(candidate.area_ratio)} temporal_iou={_fmt(candidate.temporal_iou)} added={_fmt(candidate.temporal_added_ratio)}",
        ]
        for line_index, line in enumerate(header_lines):
            cv2.putText(
                canvas,
                line,
                (14, 30 + 34 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        safe_timecode = candidate.timecode.replace(":", "-")
        filename = (
            f"candidate_{rank:03d}_{safe_timecode}_frame{candidate.frame}_"
            f"det{candidate.detection_id}_{candidate.primary_reason}.jpg"
        )
        path = output_dir / filename
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write contact sheet: {path}")
        written.append(
            {
                "candidate_number": rank,
                "image_file": filename,
                "decoded_panel_count": decoded,
                **_serializable(candidate),
            }
        )
    capture.release()
    index_path = output_dir / "image_index.csv"
    if written:
        with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(written[0]))
            writer.writeheader()
            writer.writerows(written)
    return {
        "directory": str(output_dir),
        "image_count": len(written),
        "panel_count_per_image": panel_count,
        "image_width": columns * panel_width,
        "image_height": header_height + rows * panel_height,
        "index_csv": str(index_path),
    }


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _serializable(candidate: Candidate) -> dict[str, object]:
    value = asdict(candidate)
    for key, item in list(value.items()):
        if isinstance(item, float) and not math.isfinite(item):
            value[key] = None
    return value


def _write_csv(path: Path, values: list[Candidate], *, review: bool) -> None:
    rows = [_serializable(value) for value in values]
    fieldnames = list(rows[0]) if rows else list(Candidate.__dataclass_fields__)
    if review:
        fieldnames += ["human_unreliable", "human_reason", "human_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if review:
                row.update(
                    human_unreliable="",
                    human_reason="",
                    human_notes="",
                )
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    labels = {value.strip() for value in args.labels.split(",") if value.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, metadata = _load_observations(
        args.sqlite,
        labels=labels,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        kept_only=args.kept_only,
    )
    candidates = score_observations(observations, metadata["fps"])
    selected = select_candidates(
        candidates,
        threshold=args.risk_threshold,
        maximum=args.max_candidates,
        minimum_gap=args.minimum_event_gap,
    )
    _write_csv(args.output_dir / "all_observations.csv", candidates, review=False)
    _write_csv(
        args.output_dir / "review_candidates.csv", selected, review=True
    )

    reason_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        reason_counts[item.primary_reason] += 1
    render = None
    if args.video is not None and not args.contact_sheets:
        render = render_review(
            args.video,
            args.output_dir / "unreliable_candidates_review.mp4",
            selected,
            observations,
            context_frames=args.context_frames,
            review_fps=args.review_fps,
        )
    contact_sheets = None
    if args.contact_sheets:
        if args.video is None:
            raise ValueError("--contact-sheets requires --video")
        contact_sheets = render_contact_sheets(
            args.video,
            args.output_dir / "contact_sheets",
            selected,
            observations,
            context_frames=args.contact_context_frames,
            fps=metadata["fps"],
        )
    summary = {
        "privacy": (
            "Candidate scoring used SQLite geometry only. Video decoding, when "
            "requested, occurred locally solely to write the review artifact."
        ),
        "source_sqlite": str(args.sqlite.resolve()),
        "source_video": None if args.video is None else str(args.video.resolve()),
        "metadata": metadata,
        "labels": sorted(labels),
        "kept_only": bool(args.kept_only),
        "observation_count": len(observations),
        "risk_threshold": args.risk_threshold,
        "candidate_count_before_temporal_dedup": sum(
            item.risk_score >= args.risk_threshold for item in candidates
        ),
        "selected_candidate_count": len(selected),
        "selected_reason_counts": dict(sorted(reason_counts.items())),
        "risk_quantiles": {
            str(fraction): float(
                np.quantile(
                    np.asarray([item.risk_score for item in candidates]), fraction
                )
            )
            for fraction in (0.5, 0.9, 0.95, 0.99)
        },
        "render": render,
        "contact_sheets": contact_sheets,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
