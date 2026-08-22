"""Convert classwise optimizer artifacts to public mask SQLite contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from contracts.mask_sqlite import MaskRow, iter_mask_rows, write_mask_sqlite

from ..config import ProductionConfig


def _label_maps(reference: Path) -> tuple[dict[tuple[int, str], str], dict[str, str]]:
    """Keep frame-level labels only for the rare track whose label changes."""

    exact: dict[tuple[int, str], str] = {}
    tracks: dict[str, str] = {}
    variable: set[str] = set()
    source = Path(reference).resolve()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        for track_id, minimum, maximum in connection.execute(
            "SELECT track_id,MIN(COALESCE(label,'')),MAX(COALESCE(label,'')) "
            "FROM masks GROUP BY track_id ORDER BY track_id"
        ):
            key = str(track_id)
            minimum_label = str(minimum)
            maximum_label = str(maximum)
            if minimum_label != maximum_label:
                variable.add(key)
        # Preserve the old materializer's first-non-empty track fallback while
        # avoiding polygon JSON allocation.  The exact per-frame map is only
        # retained for tracks whose label actually changes.
        for frame, track_id, label in connection.execute(
            "SELECT frame,track_id,COALESCE(label,'') FROM masks "
            "ORDER BY track_id,frame"
        ):
            key = str(track_id)
            value = str(label)
            if value:
                tracks.setdefault(key, value)
            if key in variable:
                exact[(int(frame), key)] = value
    return exact, tracks


def materialize_outputs(
    phase2_root: Path,
    tracked_sqlite: Path,
    predictions_sqlite: Path,
    keyframes_sqlite: Path,
    *,
    config: ProductionConfig,
    runtime_profile: str,
) -> dict[str, object]:
    """Merge independent semantic-class jobs without changing schema."""
    exact_labels, track_labels = _label_maps(tracked_sqlite)
    class_counts: dict[str, dict[str, int]] = {
        label: {"prediction_rows": 0, "keyframes": 0} for label in config.labels
    }
    prediction_rows = 0
    keyframe_rows = 0
    passthrough_rows = 0
    passthrough_labels: set[str] = set()

    def resolved_label(frame: int, track_id: str, fallback: str) -> str:
        return exact_labels.get(
            (int(frame), str(track_id)),
            track_labels.get(str(track_id), str(fallback)),
        )

    def dense_rows() -> Iterator[MaskRow]:
        nonlocal prediction_rows, passthrough_rows
        for label in config.labels:
            prediction = (
                Path(phase2_root)
                / runtime_profile
                / label
                / "runtime/pred/predictions.sqlite"
            )
            if not prediction.is_file():
                continue
            for row in iter_mask_rows(prediction):
                prediction_rows += 1
                class_counts[label]["prediction_rows"] += 1
                yield MaskRow(
                    frame=row.frame,
                    track_id=row.track_id,
                    polygons=row.polygons,
                    label=resolved_label(row.frame, row.track_id, label),
                    shape_type="polygon",
                )
        for row in iter_mask_rows(tracked_sqlite):
            if row.label in config.labels:
                continue
            passthrough_rows += 1
            passthrough_labels.add(row.label)
            prediction_rows += 1
            yield row

    def key_rows() -> Iterator[MaskRow]:
        nonlocal keyframe_rows
        for label in config.labels:
            keyframes = (
                Path(phase2_root)
                / runtime_profile
                / label
                / "runtime/opt/final_keyframes.json"
            )
            payload = (
                []
                if not keyframes.is_file()
                else json.loads(keyframes.read_text(encoding="utf-8"))
            )
            class_counts[label]["keyframes"] = int(len(payload))
            for value in payload:
                frame = int(value["frame"])
                track_id = str(value["track_id"])
                keyframe_rows += 1
                yield MaskRow(
                    frame=frame,
                    track_id=track_id,
                    polygons=json.dumps(
                        value["polygons"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    label=resolved_label(frame, track_id, label),
                    shape_type="polygon",
                )
        for row in iter_mask_rows(tracked_sqlite):
            if row.label in config.labels:
                continue
            keyframe_rows += 1
            yield row

    # Production's learned/optimized candidate palette is deliberately frozen
    # to the three genital labels.  Additional detector labels must not abort
    # the whole job or disappear: preserve their tracked polygons exactly and
    # expose each observed row as an editable keyframe.
    write_mask_sqlite(
        predictions_sqlite,
        dense_rows(),
        reference_sqlite=tracked_sqlite,
    )
    write_mask_sqlite(
        keyframes_sqlite,
        key_rows(),
        reference_sqlite=tracked_sqlite,
    )
    with sqlite3.connect(keyframes_sqlite) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS polygon_keyframe_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT OR REPLACE INTO polygon_keyframe_metadata(key,value) VALUES (?,?)",
            (
                ("interpolation_method", "linear_polygon_index_v1"),
                ("profile", config.profile_id),
                ("vertices_per_component", "adaptive_by_track"),
                (
                    "allowed_vertices_per_component",
                    json.dumps(config.allowed_vertices_per_component),
                ),
                (
                    "minimum_vertices_per_component",
                    str(min(config.allowed_vertices_per_component)),
                ),
                (
                    "maximum_vertices_per_component",
                    str(max(config.allowed_vertices_per_component)),
                ),
                ("track_area_quantile", str(config.track_area_quantile)),
                (
                    "screen_occupancy_thresholds",
                    json.dumps(config.screen_occupancy_thresholds),
                ),
                ("vertex_selection_source", config.vertex_selection_source),
                (
                    "exact_recall_policy",
                    "best_of_persistent_or_direct_rdp_then_uniform_scale_and_audit",
                ),
                (
                    "exact_recall_repair_max_scale",
                    str(config.spatial_recall_repair_max_scale),
                ),
            ),
        )
    return {
        "prediction_rows": int(prediction_rows),
        "keyframes": int(keyframe_rows),
        "classes": class_counts,
        "passthrough_rows": int(passthrough_rows),
        "passthrough_labels": sorted(passthrough_labels),
    }


__all__ = ("materialize_outputs",)
