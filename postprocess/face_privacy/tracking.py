"""Small, model-agnostic tracker for rich face observations.

The tracker deliberately does not smooth detector geometry.  It only assigns
stable subject IDs using the Head box, keeps a bounded set of active tracks,
and exposes enough association metadata for audit and conservative short-track
filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceTrackingConfig:
    max_gap_frames: int = 5
    high_score_threshold: float = 0.50
    low_score_threshold: float = 0.05
    iou_min: float = 0.08
    center_distance_max: float = 0.85
    area_ratio_min: float = 0.20
    area_ratio_max: float = 5.0
    match_score_min: float = 0.18
    short_track_max_hits: int = 2
    short_track_keep_score: float = 0.90

    def validate(self) -> None:
        if self.max_gap_frames < 0:
            raise ValueError("face tracking max_gap_frames must be non-negative")
        if not 0.0 <= self.low_score_threshold <= self.high_score_threshold <= 1.0:
            raise ValueError("face tracking scores must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= self.iou_min <= 1.0:
            raise ValueError("face tracking iou_min must be between 0 and 1")
        if self.center_distance_max <= 0.0:
            raise ValueError("face tracking center_distance_max must be positive")
        if self.area_ratio_min <= 0.0 or self.area_ratio_max < self.area_ratio_min:
            raise ValueError("face tracking area ratio bounds are invalid")
        if not 0.0 <= self.match_score_min <= 1.0:
            raise ValueError("face tracking match_score_min must be between 0 and 1")
        if self.short_track_max_hits < 0:
            raise ValueError("face tracking short_track_max_hits must be non-negative")
        if not 0.0 <= self.short_track_keep_score <= 1.0:
            raise ValueError(
                "face tracking short_track_keep_score must be between 0 and 1"
            )


@dataclass(frozen=True)
class FaceTrackObservation:
    observation_id: int
    anchor_detection_id: int
    frame: int
    bbox: BBox
    head_score: float
    face_score: float

    @property
    def association_score(self) -> float:
        return max(float(self.head_score), float(self.face_score))


@dataclass(frozen=True)
class FaceTrackAssignment:
    observation: FaceTrackObservation
    raw_track_id: str
    scene_id: int
    association_stage: str
    association_score: float | None


@dataclass(frozen=True)
class FaceTrackSummary:
    raw_track_id: str
    final_track_id: str | None
    scene_id: int
    start_frame: int
    end_frame: int
    observed_frames: int
    maximum_score: float
    mean_score: float
    removed_by_short_track: bool
    termination_reason: str


@dataclass
class _Track:
    serial: int
    scene_id: int
    start_frame: int
    last_frame: int
    bbox: BBox
    hits: int
    score_sum: float
    maximum_score: float

    @property
    def raw_track_id(self) -> str:
        return f"face:raw:{self.scene_id}:{self.serial}"

    @property
    def final_track_id(self) -> str:
        return f"face:{self.scene_id}:{self.serial}"

    def update(self, observation: FaceTrackObservation) -> None:
        self.last_frame = observation.frame
        self.bbox = observation.bbox
        self.hits += 1
        score = observation.association_score
        self.score_sum += score
        self.maximum_score = max(self.maximum_score, score)


def _bbox_metrics(first: BBox, second: BBox) -> tuple[float, float, float]:
    first_width = max(0.0, first[2] - first[0])
    first_height = max(0.0, first[3] - first[1])
    second_width = max(0.0, second[2] - second[0])
    second_height = max(0.0, second[3] - second[1])
    first_area = first_width * first_height
    second_area = second_width * second_height
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    union = first_area + second_area - intersection
    iou = intersection / max(union, 1e-6)
    first_center = (
        first[0] + first_width * 0.5,
        first[1] + first_height * 0.5,
    )
    second_center = (
        second[0] + second_width * 0.5,
        second[1] + second_height * 0.5,
    )
    center_distance = math.hypot(
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    )
    scale = 0.5 * (
        math.hypot(first_width, first_height) + math.hypot(second_width, second_height)
    )
    normalized_distance = center_distance / max(scale, 1e-6)
    area_ratio = second_area / max(first_area, 1e-6)
    return iou, normalized_distance, area_ratio


def _match_score(
    track: _Track,
    observation: FaceTrackObservation,
    config: FaceTrackingConfig,
) -> float | None:
    iou, distance, area_ratio = _bbox_metrics(track.bbox, observation.bbox)
    if iou < config.iou_min and distance > config.center_distance_max:
        return None
    if not config.area_ratio_min <= area_ratio <= config.area_ratio_max:
        return None
    center_score = max(0.0, 1.0 - distance / config.center_distance_max)
    scale_score = math.exp(-abs(math.log(max(area_ratio, 1e-6))))
    score = 0.65 * iou + 0.25 * center_score + 0.10 * scale_score
    return score if score >= config.match_score_min else None


def _associate(
    tracks: Sequence[_Track],
    observations: Sequence[FaceTrackObservation],
    config: FaceTrackingConfig,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    if not tracks or not observations:
        return [], set(range(len(tracks))), set(range(len(observations)))
    invalid = 1_000_000.0
    costs = np.full((len(tracks), len(observations)), invalid, dtype=np.float64)
    scores: dict[tuple[int, int], float] = {}
    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            score = _match_score(track, observation, config)
            if score is None:
                continue
            costs[track_index, observation_index] = 1.0 - score
            scores[(track_index, observation_index)] = score
    rows, columns = linear_sum_assignment(costs)
    matches: list[tuple[int, int, float]] = []
    matched_tracks: set[int] = set()
    matched_observations: set[int] = set()
    for track_index, observation_index in zip(rows.tolist(), columns.tolist()):
        score = scores.get((track_index, observation_index))
        if score is None:
            continue
        matches.append((track_index, observation_index, score))
        matched_tracks.add(track_index)
        matched_observations.add(observation_index)
    return (
        matches,
        set(range(len(tracks))) - matched_tracks,
        set(range(len(observations))) - matched_observations,
    )


class FaceTracker:
    """Streaming two-stage Head-box association with bounded active state."""

    def __init__(self, config: FaceTrackingConfig) -> None:
        config.validate()
        self.config = config
        self.scene_id = 0
        self._next_serial = 1
        self._active: list[_Track] = []

    def _summary(self, track: _Track, reason: str) -> FaceTrackSummary:
        removed = (
            track.hits <= self.config.short_track_max_hits
            and track.maximum_score < self.config.short_track_keep_score
        )
        return FaceTrackSummary(
            raw_track_id=track.raw_track_id,
            final_track_id=None if removed else track.final_track_id,
            scene_id=track.scene_id,
            start_frame=track.start_frame,
            end_frame=track.last_frame,
            observed_frames=track.hits,
            maximum_score=track.maximum_score,
            mean_score=track.score_sum / max(track.hits, 1),
            removed_by_short_track=removed,
            termination_reason=reason,
        )

    def _new_track(self, observation: FaceTrackObservation) -> _Track:
        score = observation.association_score
        track = _Track(
            serial=self._next_serial,
            scene_id=self.scene_id,
            start_frame=observation.frame,
            last_frame=observation.frame,
            bbox=observation.bbox,
            hits=1,
            score_sum=score,
            maximum_score=score,
        )
        self._next_serial += 1
        self._active.append(track)
        return track

    def update(
        self,
        frame: int,
        observations: Sequence[FaceTrackObservation],
        *,
        is_cut: bool = False,
    ) -> tuple[list[FaceTrackAssignment], list[FaceTrackSummary]]:
        completed: list[FaceTrackSummary] = []
        if is_cut:
            completed.extend(self._summary(track, "cut") for track in self._active)
            self._active.clear()
            self.scene_id += 1
        retained: list[_Track] = []
        for track in self._active:
            if frame - track.last_frame > self.config.max_gap_frames:
                completed.append(self._summary(track, "max_gap"))
            else:
                retained.append(track)
        self._active = retained

        high = [
            observation
            for observation in observations
            if observation.association_score >= self.config.high_score_threshold
        ]
        low = [
            observation
            for observation in observations
            if self.config.low_score_threshold
            <= observation.association_score
            < self.config.high_score_threshold
        ]
        ignored = [
            observation
            for observation in observations
            if observation.association_score < self.config.low_score_threshold
        ]
        assignments: list[FaceTrackAssignment] = []
        first_matches, unmatched_tracks, unmatched_high = _associate(
            self._active,
            high,
            self.config,
        )
        for track_index, observation_index, score in first_matches:
            track = self._active[track_index]
            observation = high[observation_index]
            track.update(observation)
            assignments.append(
                FaceTrackAssignment(
                    observation,
                    track.raw_track_id,
                    track.scene_id,
                    "high",
                    score,
                )
            )

        remaining_tracks = [self._active[index] for index in sorted(unmatched_tracks)]
        second_matches, _unmatched_second_tracks, unmatched_low = _associate(
            remaining_tracks,
            low,
            self.config,
        )
        for track_index, observation_index, score in second_matches:
            track = remaining_tracks[track_index]
            observation = low[observation_index]
            track.update(observation)
            assignments.append(
                FaceTrackAssignment(
                    observation,
                    track.raw_track_id,
                    track.scene_id,
                    "low",
                    score,
                )
            )

        for observation_index in sorted(unmatched_high):
            observation = high[observation_index]
            track = self._new_track(observation)
            assignments.append(
                FaceTrackAssignment(
                    observation,
                    track.raw_track_id,
                    track.scene_id,
                    "new_high",
                    None,
                )
            )
        for observation_index in sorted(unmatched_low):
            observation = low[observation_index]
            track = self._new_track(observation)
            assignments.append(
                FaceTrackAssignment(
                    observation,
                    track.raw_track_id,
                    track.scene_id,
                    "new_low",
                    None,
                )
            )
        for observation in ignored:
            track = self._new_track(observation)
            assignments.append(
                FaceTrackAssignment(
                    observation,
                    track.raw_track_id,
                    track.scene_id,
                    "new_below_low",
                    None,
                )
            )
        assignments.sort(key=lambda value: value.observation.observation_id)
        return assignments, completed

    def finish(self) -> list[FaceTrackSummary]:
        completed = [self._summary(track, "end_of_stream") for track in self._active]
        self._active.clear()
        return completed


def track_face_observations(
    observations: Iterable[FaceTrackObservation],
    *,
    cuts: set[int] | frozenset[int] = frozenset(),
    config: FaceTrackingConfig = FaceTrackingConfig(),
) -> tuple[list[FaceTrackAssignment], list[FaceTrackSummary]]:
    """Convenience wrapper used by tests and small callers."""

    tracker = FaceTracker(config)
    assignments: list[FaceTrackAssignment] = []
    summaries: list[FaceTrackSummary] = []
    frame_observations: list[FaceTrackObservation] = []
    current_frame: int | None = None
    pending_cuts = iter(sorted(cuts))
    next_cut = next(pending_cuts, None)

    def flush(frame: int, values: Sequence[FaceTrackObservation]) -> None:
        nonlocal next_cut
        while next_cut is not None and next_cut <= frame:
            _unused, completed = tracker.update(next_cut, (), is_cut=True)
            summaries.extend(completed)
            next_cut = next(pending_cuts, None)
        frame_assignments, completed = tracker.update(frame, values)
        assignments.extend(frame_assignments)
        summaries.extend(completed)

    for observation in observations:
        if current_frame is None:
            current_frame = observation.frame
        if observation.frame != current_frame:
            flush(current_frame, frame_observations)
            frame_observations = []
            current_frame = observation.frame
        frame_observations.append(observation)
    if current_frame is not None:
        flush(current_frame, frame_observations)
    summaries.extend(tracker.finish())
    return assignments, summaries


__all__ = [
    "FaceTrackAssignment",
    "FaceTrackObservation",
    "FaceTrackSummary",
    "FaceTracker",
    "FaceTrackingConfig",
    "track_face_observations",
]
