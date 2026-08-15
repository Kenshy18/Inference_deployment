"""Asymmetric temporal consensus for noisy instance-mask observations.

The AI mask at one frame is an observation, not ground truth.  This module
builds a trusted reference from neighbouring observations without opening the
video.  A one-frame contraction keeps temporally supported area, whereas a
one-frame expansion is accepted only where neighbouring masks support it.

The implementation is deliberately experimental and does not participate in
the production stage registry or alter the SQLite contract.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from .fixed_budget import RawMask, Segment, _polygonal


@dataclass(frozen=True)
class TemporalMaskDiagnostic:
    frame: int
    track_id: str
    reliability: float
    observation_iou: float
    missing_ratio: float
    added_ratio: float
    area_ratio: float
    classification: str
    neighbour_count: int
    support_required: int


@dataclass(frozen=True)
class TemporalConsensusResult:
    trusted_masks: dict[tuple[int, str], RawMask]
    diagnostics: dict[tuple[int, str], TemporalMaskDiagnostic]


@dataclass(frozen=True)
class _Pose:
    center_x: float
    center_y: float
    angle: float
    scale: float


def _largest_polygon(geometry) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda value: float(value.area), default=None)
    polygons = [
        part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon)
    ]
    return max(polygons, key=lambda value: float(value.area), default=None)


def _primary_points(geometry) -> np.ndarray:
    polygon = _largest_polygon(geometry)
    if polygon is None or polygon.is_empty:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)


def _pose(geometry) -> _Pose:
    polygon = _largest_polygon(geometry)
    if polygon is None or polygon.is_empty:
        return _Pose(0.0, 0.0, 0.0, 1.0)
    points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    center = np.asarray([geometry.centroid.x, geometry.centroid.y], dtype=np.float64)
    centered = points - center
    if len(centered) >= 3:
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        angle = math.atan2(float(axis[1]), float(axis[0]))
    else:
        angle = 0.0
    return _Pose(
        center_x=float(center[0]),
        center_y=float(center[1]),
        angle=angle,
        scale=math.sqrt(max(float(geometry.area), 1e-6)),
    )


def _unwrap_near(angle: float, reference: float) -> float:
    # A polygon principal axis is pi-periodic, not 2*pi-periodic.
    candidates = (angle - math.pi, angle, angle + math.pi)
    return min(candidates, key=lambda value: abs(value - reference))


def _interpolated_pose(
    frame: int,
    frames: list[int],
    poses: dict[int, _Pose],
) -> _Pose:
    previous = [candidate for candidate in frames if candidate < frame]
    following = [candidate for candidate in frames if candidate > frame]
    if previous and following:
        left_frame = previous[-1]
        right_frame = following[0]
    elif len(following) >= 2:
        # One-sided linear extrapolation at a track/scene start.
        left_frame = following[0]
        right_frame = following[1]
    elif len(previous) >= 2:
        # One-sided linear extrapolation at a track/scene end.
        left_frame = previous[-2]
        right_frame = previous[-1]
    else:
        return poses[frame]
    left = poses[left_frame]
    right = poses[right_frame]
    span = max(right_frame - left_frame, 1)
    alpha = (frame - left_frame) / span
    right_angle = _unwrap_near(right.angle, left.angle)
    return _Pose(
        center_x=(1.0 - alpha) * left.center_x + alpha * right.center_x,
        center_y=(1.0 - alpha) * left.center_y + alpha * right.center_y,
        angle=(1.0 - alpha) * left.angle + alpha * right_angle,
        scale=math.exp(
            (1.0 - alpha) * math.log(max(left.scale, 1e-6))
            + alpha * math.log(max(right.scale, 1e-6))
        ),
    )


def _map_pose(geometry, source: _Pose, target: _Pose):
    if geometry.is_empty:
        return geometry
    moved = affinity.translate(
        geometry,
        xoff=-source.center_x,
        yoff=-source.center_y,
    )
    moved = affinity.rotate(moved, -math.degrees(source.angle), origin=(0.0, 0.0))
    ratio = target.scale / max(source.scale, 1e-6)
    moved = affinity.scale(moved, xfact=ratio, yfact=ratio, origin=(0.0, 0.0))
    moved = affinity.rotate(moved, math.degrees(target.angle), origin=(0.0, 0.0))
    return _polygonal(
        affinity.translate(moved, xoff=target.center_x, yoff=target.center_y)
    )


def _at_least_k(geometries: list, required: int):
    """Return area covered by at least ``required`` input geometries."""

    usable = [value for value in geometries if not value.is_empty]
    if not usable:
        return GeometryCollection()
    requested = min(max(1, int(required)), len(usable))
    if requested == 1:
        return _polygonal(unary_union(usable))
    intersections = []
    for combination in itertools.combinations(usable, requested):
        candidate = combination[0]
        for other in combination[1:]:
            candidate = candidate.intersection(other)
            if candidate.is_empty:
                break
        if not candidate.is_empty:
            intersections.append(candidate)
    return (
        _polygonal(unary_union(intersections))
        if intersections
        else GeometryCollection()
    )


def _iou(left, right) -> float:
    intersection = float(left.intersection(right).area)
    union = float(left.area + right.area - intersection)
    return intersection / union if union else 1.0


def build_asymmetric_temporal_consensus(
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    radius: int = 2,
    minimum_neighbours: int = 2,
    support_fraction: float = 0.50,
    boundary_tolerance_ratio: float = 0.015,
    minimum_boundary_tolerance_px: float = 1.5,
    anomaly_ratio: float = 0.12,
) -> TemporalConsensusResult:
    """Build motion-aligned, asymmetric trusted masks per track.

    Neighbours are mapped into a pose predicted from the closest observations
    on both sides.  Temporally supported area is retained when the current raw
    mask contracts.  Current-frame expansion is included only inside a small
    tolerance around the supported region.  Track ends with insufficient
    temporal evidence intentionally fall back to the raw observation.
    """

    by_track: dict[str, dict[int, RawMask]] = defaultdict(dict)
    for (frame, track_id), raw in raw_masks.items():
        by_track[track_id][frame] = raw

    trusted: dict[tuple[int, str], RawMask] = {}
    diagnostics: dict[tuple[int, str], TemporalMaskDiagnostic] = {}
    for track_id, observations in by_track.items():
        frames = sorted(observations)
        raw_poses = {frame: _pose(observations[frame].geometry) for frame in frames}
        # PCA axes have no direction: theta and theta+pi describe the same
        # axis.  Resolve that ambiguity once along the track.  Without this,
        # an asymmetric polygon may be spuriously flipped by 180 degrees when
        # neighbouring masks are mapped into a common pose.
        poses: dict[int, _Pose] = {}
        previous_angle: float | None = None
        for frame in frames:
            current = raw_poses[frame]
            angle = (
                current.angle
                if previous_angle is None
                else _unwrap_near(current.angle, previous_angle)
            )
            poses[frame] = _Pose(
                current.center_x,
                current.center_y,
                angle,
                current.scale,
            )
            previous_angle = angle
        frame_set = set(frames)
        for frame in frames:
            raw = observations[frame]
            neighbour_frames = [
                candidate
                for candidate in range(frame - int(radius), frame + int(radius) + 1)
                if candidate != frame and candidate in frame_set
            ]
            if len(neighbour_frames) < int(minimum_neighbours):
                trusted[(frame, track_id)] = raw
                diagnostics[(frame, track_id)] = TemporalMaskDiagnostic(
                    frame=frame,
                    track_id=track_id,
                    reliability=0.50,
                    observation_iou=1.0,
                    missing_ratio=0.0,
                    added_ratio=0.0,
                    area_ratio=1.0,
                    classification="insufficient_context",
                    neighbour_count=len(neighbour_frames),
                    support_required=0,
                )
                continue

            expected_pose = _interpolated_pose(frame, frames, poses)
            aligned = [
                _map_pose(
                    observations[candidate].geometry,
                    poses[candidate],
                    expected_pose,
                )
                for candidate in neighbour_frames
            ]
            required = min(
                len(aligned),
                max(2, int(math.ceil(len(aligned) * support_fraction))),
            )
            consensus = _at_least_k(aligned, required)
            if consensus.is_empty:
                consensus = raw.geometry

            tolerance = max(
                float(minimum_boundary_tolerance_px),
                float(boundary_tolerance_ratio)
                * math.sqrt(max(float(consensus.area), 1.0)),
            )
            # Retain the neighbour-supported consensus on contractions.  On
            # expansions, accept only the part close to temporal support.
            supported_current = raw.geometry.intersection(consensus.buffer(tolerance))
            trusted_geometry = _polygonal(consensus.union(supported_current))
            if trusted_geometry.is_empty:
                trusted_geometry = raw.geometry

            consensus_area = max(float(consensus.area), 1e-9)
            missing_ratio = (
                float(consensus.difference(raw.geometry).area) / consensus_area
            )
            added_ratio = (
                float(raw.geometry.difference(consensus).area) / consensus_area
            )
            observation_iou = _iou(raw.geometry, consensus)
            if missing_ratio >= anomaly_ratio and added_ratio < anomaly_ratio:
                classification = "sudden_contraction"
            elif added_ratio >= anomaly_ratio and missing_ratio < anomaly_ratio:
                classification = "sudden_expansion"
            elif added_ratio >= anomaly_ratio and missing_ratio >= anomaly_ratio:
                classification = "mixed_shape_outlier"
            else:
                classification = "temporally_supported"
            anomaly = min(1.0, missing_ratio + added_ratio)
            reliability = float(
                np.clip(observation_iou * (1.0 - 0.5 * anomaly), 0.0, 1.0)
            )
            points = _primary_points(trusted_geometry)
            if len(points) < 3:
                trusted_geometry = raw.geometry
                points = np.asarray(raw.primary_points, dtype=np.float64)
            trusted[(frame, track_id)] = RawMask(
                frame=frame,
                track_id=track_id,
                geometry=trusted_geometry,
                primary_points=points,
                score=raw.score,
            )
            diagnostics[(frame, track_id)] = TemporalMaskDiagnostic(
                frame=frame,
                track_id=track_id,
                reliability=reliability,
                observation_iou=observation_iou,
                missing_ratio=missing_ratio,
                added_ratio=added_ratio,
                area_ratio=float(raw.geometry.area) / consensus_area,
                classification=classification,
                neighbour_count=len(neighbour_frames),
                support_required=required,
            )
    return TemporalConsensusResult(trusted_masks=trusted, diagnostics=diagnostics)


def build_segment_bounded_temporal_consensus(
    raw_masks: dict[tuple[int, str], RawMask],
    segments: dict[str, list[Segment]],
    **kwargs,
) -> TemporalConsensusResult:
    """Build consensus independently inside every scene/track segment."""

    trusted: dict[tuple[int, str], RawMask] = {}
    diagnostics: dict[tuple[int, str], TemporalMaskDiagnostic] = {}
    for track_id, values in segments.items():
        for segment in values:
            local = {
                identity: raw
                for identity, raw in raw_masks.items()
                if identity[1] == track_id
                and segment.first_frame <= identity[0] <= segment.last_frame
            }
            if not local:
                continue
            result = build_asymmetric_temporal_consensus(local, **kwargs)
            trusted.update(result.trusted_masks)
            diagnostics.update(result.diagnostics)
    # Preserve observations not covered by the supplied segment selection.
    for identity, raw in raw_masks.items():
        if identity in trusted:
            continue
        trusted[identity] = raw
        diagnostics[identity] = TemporalMaskDiagnostic(
            frame=identity[0],
            track_id=identity[1],
            reliability=0.0,
            observation_iou=1.0,
            missing_ratio=0.0,
            added_ratio=0.0,
            area_ratio=1.0,
            classification="outside_selected_segment",
            neighbour_count=0,
            support_required=0,
        )
    return TemporalConsensusResult(trusted_masks=trusted, diagnostics=diagnostics)
