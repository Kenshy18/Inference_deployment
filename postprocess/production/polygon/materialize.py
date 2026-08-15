"""Convert classwise optimizer artifacts to public mask SQLite contracts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite

from ..config import ProductionConfig


def _label_maps(reference: Path) -> tuple[dict[tuple[int, str], str], dict[str, str]]:
    exact: dict[tuple[int, str], str] = {}
    tracks: dict[str, str] = {}
    for row in read_mask_rows(reference):
        exact[(row.frame, row.track_id)] = row.label
        if row.label:
            tracks.setdefault(row.track_id, row.label)
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
    """Merge three independent semantic-class jobs without changing schema."""
    exact_labels, track_labels = _label_maps(tracked_sqlite)
    dense_rows: list[MaskRow] = []
    key_rows: list[MaskRow] = []
    class_counts: dict[str, dict[str, int]] = {}
    for label in config.labels:
        root = Path(phase2_root) / runtime_profile / label / "runtime"
        prediction = root / "pred/predictions.sqlite"
        keyframes = root / "opt/final_keyframes.json"
        label_dense = [] if not prediction.is_file() else read_mask_rows(prediction)
        for row in label_dense:
            resolved_label = exact_labels.get(
                (row.frame, row.track_id), track_labels.get(row.track_id, label)
            )
            dense_rows.append(
                MaskRow(
                    frame=row.frame,
                    track_id=row.track_id,
                    polygons=row.polygons,
                    label=resolved_label,
                    shape_type="polygon",
                )
            )
        payload = [] if not keyframes.is_file() else json.loads(
            keyframes.read_text(encoding="utf-8")
        )
        for value in payload:
            frame = int(value["frame"])
            track_id = str(value["track_id"])
            key_rows.append(
                MaskRow(
                    frame=frame,
                    track_id=track_id,
                    polygons=json.dumps(
                        value["polygons"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    label=exact_labels.get(
                        (frame, track_id), track_labels.get(track_id, label)
                    ),
                    shape_type="polygon",
                )
            )
        class_counts[label] = {
            "prediction_rows": len(label_dense),
            "keyframes": len(payload),
        }
    dense_rows.sort(key=lambda row: (int(row.frame), str(row.track_id)))
    key_rows.sort(key=lambda row: (int(row.frame), str(row.track_id)))
    write_mask_sqlite(predictions_sqlite, dense_rows, reference_sqlite=tracked_sqlite)
    write_mask_sqlite(keyframes_sqlite, key_rows, reference_sqlite=tracked_sqlite)
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
                ("vertices_per_component", str(config.vertices_per_component)),
            ),
        )
    return {
        "prediction_rows": len(dense_rows),
        "keyframes": len(key_rows),
        "classes": class_counts,
    }


__all__ = ("materialize_outputs",)
