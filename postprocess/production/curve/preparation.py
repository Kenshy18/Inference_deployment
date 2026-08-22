"""Lean, read-only input preparation for the Production curve engine.

The polygon runtime still needs its historical Phase-2 compatibility tree.
The curve engine consumes the prepared class SQLite paths directly, so making
the same compatibility copies only increases long-video wall time and disk
usage.  This module keeps the shared border, endpoint and vertex semantics but
does not create absent-class databases or the legacy Phase-2 shim.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from classwise.sqlite import read_track_labels
from production.polygon.input_geometry import (
    apply_border_expansion,
    apply_endpoint_extension,
)
from production.polygon.runtime.candidate_config import CandidateConfig
from production.polygon.vertex_policy import build_vertex_policy

from .storage import MaskSqliteWriter


def _sqlite_stats(path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
        }
        counts = {
            table: int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in ("masks", "tracks")
            if table in tables
        }
        integrity = tuple(str(row[0]) for row in db.execute("PRAGMA integrity_check"))
        foreign_keys = tuple(db.execute("PRAGMA foreign_key_check"))
    if integrity != ("ok",) or foreign_keys:
        raise RuntimeError(
            f"invalid curve source: integrity={integrity}, "
            f"foreign_keys={len(foreign_keys)}"
        )
    return {
        "counts": counts,
        "integrity_check": "ok",
        "foreign_key_errors": 0,
        "size_bytes": Path(path).stat().st_size,
        "minimal_curve_projection": True,
    }


def _write_minimal_projection(
    source: Path,
    output: Path,
    *,
    track_ids: tuple[str, ...],
    track_labels: dict[str, str],
) -> int:
    """Stream only geometry needed by border/endpoint/curve processing."""

    allowed = {str(track_id) for track_id in track_ids}
    writer = MaskSqliteWriter(output, source)
    try:
        with sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True) as db:
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(masks)")}
            label_sql = "COALESCE(label,'')" if "label" in columns else "''"
            for frame, track_id, polygons, row_label in db.execute(
                "SELECT frame,track_id,polygons," + label_sql + " FROM masks "
                "ORDER BY track_id,frame"
            ):
                key = str(track_id)
                if key not in allowed:
                    continue
                writer.append(
                    frame=int(frame),
                    track_id=key,
                    polygons=str(polygons),
                    label=track_labels.get(key, str(row_label)),
                )
        result = writer.finalize()
    except BaseException:
        writer.abort()
        raise

    # Endpoint extension only needs cut positions from non-mask provenance.
    # Copy that small table when present instead of cloning the entire tracked
    # database (raw detections can be hundreds of times larger than masks).
    with sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True) as src:
        cut_sql_row = src.execute(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name='cuts'"
        ).fetchone()
        cut_rows = tuple(src.execute("SELECT * FROM cuts")) if cut_sql_row else ()
    if cut_sql_row and cut_sql_row[0]:
        with sqlite3.connect(result) as dst:
            dst.execute(str(cut_sql_row[0]))
            if cut_rows:
                placeholders = ",".join("?" for _value in cut_rows[0])
                dst.executemany(
                    f"INSERT INTO cuts VALUES ({placeholders})",
                    cut_rows,
                )
    return int(writer.rows)


def prepare_curve_source(
    tracked_sqlite: Path,
    output_root: Path,
    *,
    width: int,
    height: int,
    input_video: Path | None,
    config: CandidateConfig,
    selected_track_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Prepare only classes that exist and return direct engine inputs."""

    config.validate()
    tracked = Path(tracked_sqlite).resolve()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_track_labels = read_track_labels(tracked)
    if selected_track_ids is None:
        routed_track_ids = tuple(all_track_labels)
    else:
        routed_track_ids = tuple(
            dict.fromkeys(str(value) for value in selected_track_ids)
        )
        missing = sorted(set(routed_track_ids) - set(all_track_labels))
        if missing:
            raise RuntimeError(f"curve route contains unknown tracks: {missing}")
    selected_set = set(routed_track_ids)
    track_labels = {
        track_id: label
        for track_id, label in all_track_labels.items()
        if track_id in selected_set
    }
    vertex_policy_path = root / "vertex_policy.json"
    vertex_policy = build_vertex_policy(
        tracked,
        vertex_policy_path,
        width=int(width),
        height=int(height),
        track_labels=track_labels,
        config=config,
    )
    tracks_by_label = {
        label: tuple(
            track_id
            for track_id, assigned in sorted(track_labels.items())
            if assigned == label
        )
        for label in config.labels
    }
    active_labels = tuple(label for label in config.labels if tracks_by_label[label])
    settings = config.preparation
    classes: dict[str, object] = {}
    for index, label in enumerate(active_labels):
        class_root = root / "classes" / f"{index:02d}_{label}"
        class_root.mkdir(parents=True, exist_ok=True)
        class_track_ids = tracks_by_label[label]
        projected = class_root / "tracked.sqlite"
        input_rows = _write_minimal_projection(
            tracked,
            projected,
            track_ids=class_track_ids,
            track_labels=track_labels,
        )
        projection = _sqlite_stats(projected)
        border = class_root / "border_expanded.sqlite"
        endpoint = class_root / "endpoint_extended.sqlite"
        _, border_stats = apply_border_expansion(
            projected,
            border,
            width=int(width),
            height=int(height),
            trigger_px=settings.border_trigger_px,
            expand_ratio=settings.border_expand_ratio,
            min_expand_px=settings.border_min_expand_px,
            max_expand_px=settings.border_max_expand_px,
            influence_px=settings.border_influence_px,
            corner_support=settings.border_corner_support,
        )
        _, endpoint_stats = apply_endpoint_extension(
            border,
            endpoint,
            video=input_video,
            extend_frames=settings.endpoint_extend_frames,
            motion_frames=settings.endpoint_motion_frames,
            max_speed_px=settings.endpoint_max_speed_px,
        )
        classes[label] = {
            "active": True,
            "track_ids": list(class_track_ids),
            "input_rows": int(input_rows),
            "projection": projection,
            "projected_sqlite": str(projected),
            "border_sqlite": str(border),
            "endpoint_sqlite": str(endpoint),
            "border": border_stats,
            "endpoint": endpoint_stats,
        }
    return {
        "tracked_sqlite": str(tracked),
        "width": int(width),
        "height": int(height),
        "input_video": (
            None if input_video is None else str(Path(input_video).resolve())
        ),
        "classes": classes,
        "active_labels": list(active_labels),
        "selected_track_ids": list(routed_track_ids),
        "passthrough_track_ids": [
            track_id
            for track_id in routed_track_ids
            if track_labels[track_id] not in config.labels
        ],
        "vertex_policy": vertex_policy,
        "vertex_policy_json": str(vertex_policy_path),
        "compatibility_copies_skipped": True,
    }


__all__ = ("prepare_curve_source",)
