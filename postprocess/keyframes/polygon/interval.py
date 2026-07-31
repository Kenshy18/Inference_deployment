"""Replaceable polygon keyframe selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from contracts.mask_sqlite import (
    MaskRow,
    read_mask_rows,
    track_sort_key,
    write_mask_sqlite,
)


class KeyframeSelector(Protocol):
    name: str

    def select(self, rows: list[MaskRow]) -> list[MaskRow]:
        """Return a deterministic subset containing track endpoints."""


@dataclass(frozen=True)
class IntervalKeyframeSelector:
    name: str = "fixed_interval"
    interval_frames: int = 3

    def select(self, rows: list[MaskRow]) -> list[MaskRow]:
        if self.interval_frames < 1:
            raise ValueError("interval_frames must be >= 1")
        by_track: dict[str, list[MaskRow]] = {}
        for row in rows:
            by_track.setdefault(row.track_id, []).append(row)
        selected: list[MaskRow] = []
        for track_id in sorted(by_track, key=track_sort_key):
            track_rows = sorted(by_track[track_id], key=lambda row: row.frame)
            if not track_rows:
                continue
            keep = [track_rows[0]]
            last_frame = track_rows[0].frame
            for row in track_rows[1:-1]:
                if row.frame - last_frame >= self.interval_frames:
                    keep.append(row)
                    last_frame = row.frame
            if len(track_rows) > 1 and keep[-1].frame != track_rows[-1].frame:
                keep.append(track_rows[-1])
            selected.extend(keep)
        return selected


def select_keyframes_sqlite(
    input_sqlite: Path,
    output_sqlite: Path,
    *,
    selector: KeyframeSelector | None = None,
) -> Path:
    implementation = selector or IntervalKeyframeSelector()
    rows = implementation.select(read_mask_rows(input_sqlite))
    from common.live_preview import PreviewGeometry, active_postprocess_preview

    preview = active_postprocess_preview()
    if preview is not None:
        for row in rows:
            if not preview.should_sample("keyframe_selection"):
                continue
            polygons = json.loads(row.polygons)
            preview.submit(
                PreviewGeometry(
                    row.frame,
                    "keyframe_selection",
                    "keyframe selection",
                    polygons=tuple(
                        tuple((float(point[0]), float(point[1])) for point in polygon)
                        for polygon in polygons
                    ),
                    track_id=row.track_id,
                    detail=f"interval {getattr(implementation, 'interval_frames', '?')}",
                    is_keyframe=True,
                )
            )
    return write_mask_sqlite(output_sqlite, rows, reference_sqlite=input_sqlite)
