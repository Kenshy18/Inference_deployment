"""Mask-SQLite input helpers for the isolated curve experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.mask_sqlite import read_mask_rows, track_sort_key
from production.polygon.runtime.spatial_support.optimizer import orient_ccw


@dataclass(frozen=True, slots=True)
class TrackContours:
    track_id: str
    label: str
    frames: np.ndarray
    contours: tuple[np.ndarray, ...]
    component_index: int


def available_tracks(path: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[object]] = {}
    for row in read_mask_rows(Path(path)):
        grouped.setdefault(row.track_id, []).append(row)
    return [
        {
            "track_id": track_id,
            "label": next((row.label for row in rows if row.label), ""),
            "first_frame": min(row.frame for row in rows),
            "last_frame": max(row.frame for row in rows),
            "frames": len(rows),
        }
        for track_id, rows in sorted(
            grouped.items(), key=lambda item: track_sort_key(item[0])
        )
    ]


def load_track_contours(
    path: Path,
    track_id: str,
    *,
    component_index: int = 0,
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_frames: int = 0,
    require_contiguous: bool = False,
) -> TrackContours:
    """Load one explicit component without silently changing its identity."""
    selected = []
    for row in read_mask_rows(Path(path)):
        if row.track_id != str(track_id):
            continue
        if start_frame is not None and row.frame < int(start_frame):
            continue
        if end_frame is not None and row.frame > int(end_frame):
            continue
        selected.append(row)
    if max_frames > 0:
        selected = selected[: int(max_frames)]
    if not selected:
        raise ValueError(f"track {track_id!r} has no rows in the requested range")
    if bool(require_contiguous):
        frames = np.asarray([row.frame for row in selected], dtype=np.int64)
        gaps = np.flatnonzero(np.diff(frames) != 1)
        if len(gaps):
            position = int(gaps[0])
            raise ValueError(
                "keyframe comparison requires a contiguous run; "
                f"first gap is {int(frames[position])}->{int(frames[position + 1])}. "
                "Select one contiguous range or use a prepared gap-filled SQLite."
            )

    contours: list[np.ndarray] = []
    for row in selected:
        payload = json.loads(row.polygons)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"frame {row.frame}: polygons is empty")
        if int(component_index) < 0 or int(component_index) >= len(payload):
            raise ValueError(
                f"frame {row.frame}: component {component_index} is absent "
                f"(available: 0..{len(payload) - 1})"
            )
        contour = np.asarray(payload[int(component_index)], dtype=np.float64)
        contours.append(orient_ccw(contour))
    label = next((row.label for row in selected if row.label), "")
    return TrackContours(
        track_id=str(track_id),
        label=label,
        frames=np.asarray([row.frame for row in selected], dtype=np.int64),
        contours=tuple(contours),
        component_index=int(component_index),
    )


__all__ = ("TrackContours", "available_tracks", "load_track_contours")
