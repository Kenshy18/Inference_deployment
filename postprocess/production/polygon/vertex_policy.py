"""Deterministic track-level polygon vertex-count selection.

The policy intentionally reads the tracked mask SQLite before border and
endpoint preparation.  Edge safeguards must never make a track cross a size
threshold and thereby change its editable representation.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from contracts.mask_sqlite import read_mask_rows

from .runtime.candidate_config import CANDIDATE, CandidateConfig


def select_vertex_count(
    occupancy: float,
    config: CandidateConfig = CANDIDATE,
) -> int:
    """Return 14/16/18/20 using strict upper-threshold crossings."""
    value = float(occupancy)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"screen occupancy must be finite and non-negative: {value}")
    counts = config.spatial.allowed_vertices_per_component
    for index, threshold in enumerate(config.spatial.screen_occupancy_thresholds):
        if value <= float(threshold):
            return int(counts[index])
    return int(counts[-1])


def _foreground_area(polygons_json: str) -> float:
    """Measure total continuous foreground area after topology cleanup.

    Candidate NMS fills true holes before tracking and stores disconnected
    foreground components as separate contours.  Summing their continuous
    contour areas therefore matches the audited q99.9 policy without a
    resolution-dependent raster allocation.
    """
    polygons = json.loads(polygons_json)
    return float(
        sum(
            abs(
                float(
                    cv2.contourArea(
                        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    )
                )
            )
            for polygon in polygons
            if len(polygon) >= 3
        )
    )


def build_vertex_policy(
    tracked_sqlite: Path,
    output_json: Path,
    *,
    width: int,
    height: int,
    track_labels: dict[str, str],
    config: CandidateConfig = CANDIDATE,
) -> dict[str, object]:
    """Compute and persist one immutable vertex count for every target track."""
    config.validate()
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("vertex policy requires positive video dimensions")
    frame_area = float(int(width) * int(height))
    areas: dict[str, list[float]] = defaultdict(list)
    for row in read_mask_rows(Path(tracked_sqlite)):
        track_id = str(row.track_id)
        if track_labels.get(track_id) not in config.labels:
            continue
        areas[track_id].append(_foreground_area(row.polygons))

    quantile = float(config.spatial.track_area_quantile)
    tracks: dict[str, dict[str, object]] = {}
    count_tracks = {
        str(vertices): 0 for vertices in config.spatial.allowed_vertices_per_component
    }
    count_rows = {
        str(vertices): 0 for vertices in config.spatial.allowed_vertices_per_component
    }
    for track_id in sorted(
        areas, key=lambda value: (int(value) if value.isdigit() else 10**30, value)
    ):
        values = np.asarray(areas[track_id], dtype=np.float64)
        q_area = float(np.quantile(values, quantile, method="linear"))
        occupancy = q_area / frame_area
        vertices = select_vertex_count(occupancy, config)
        tracks[track_id] = {
            "label": str(track_labels[track_id]),
            "rows": int(len(values)),
            "q_area_px2": q_area,
            "screen_occupancy": occupancy,
            "vertices_per_component": vertices,
        }
        count_tracks[str(vertices)] += 1
        count_rows[str(vertices)] += int(len(values))

    missing = sorted(
        track_id
        for track_id, label in track_labels.items()
        if label in config.labels and track_id not in tracks
    )
    if missing:
        raise RuntimeError(f"tracked genital tracks have no mask rows: {missing}")
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": config.profile_id,
        "polygon_profile_id": config.polygon_profile_id,
        "source_sqlite": str(Path(tracked_sqlite).resolve()),
        "source_stage": config.spatial.vertex_selection_source,
        "area_definition": "sum_abs_continuous_foreground_contour_area",
        "quantile": quantile,
        "quantile_method": "linear",
        "width": int(width),
        "height": int(height),
        "frame_area_px2": int(width) * int(height),
        "thresholds": list(config.spatial.screen_occupancy_thresholds),
        "allowed_vertices": list(config.spatial.allowed_vertices_per_component),
        "threshold_comparison": config.spatial.vertex_selection_comparison,
        "tracks": tracks,
        "summary": {
            "tracks": int(len(tracks)),
            "track_rows": int(sum(len(value) for value in areas.values())),
            "tracks_by_vertices": count_tracks,
            "track_rows_by_vertices": count_rows,
        },
    }
    output = Path(output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


__all__ = ("build_vertex_policy", "select_vertex_count")
