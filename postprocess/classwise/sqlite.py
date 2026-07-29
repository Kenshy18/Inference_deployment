"""SQLite projection and merge helpers for class-aware postprocessing."""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .policy import ClassPostprocessPolicy, ClassPostprocessSettings


@dataclass(frozen=True, slots=True)
class RoutedGroup:
    group_id: str
    labels: tuple[str, ...]
    track_ids: tuple[str, ...]
    settings: ClassPostprocessSettings
    predictions_sqlite: Path


def read_track_labels(path: Path) -> dict[str, str]:
    """Return one stable label for every track that owns at least one mask."""

    source = Path(path).expanduser().resolve()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        has_tracks = "tracks" in tables
        track_label = (
            "NULLIF(t.label, '')" if has_tracks else "NULL"
        )
        track_join = (
            "LEFT JOIN tracks t ON t.track_id=m.track_id" if has_tracks else ""
        )
        rows = connection.execute(
            f"""
            SELECT m.track_id,
                   COALESCE({track_label}, MAX(NULLIF(m.label, '')), '')
            FROM masks m
            {track_join}
            GROUP BY m.track_id
            ORDER BY m.track_id
            """
        )
        return {str(track_id): str(label) for track_id, label in rows}


def count_masks(path: Path) -> int:
    """Return the active mask row count without materializing mask geometry."""

    source = Path(path).expanduser().resolve()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0])


def _temporary_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")


def _remove_sqlite_temporary_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def filter_tracked_sqlite(
    source: Path,
    output: Path,
    *,
    track_ids: tuple[str, ...],
) -> int:
    """Copy a tracked SQLite while projecting its active masks to selected tracks."""

    resolved_source = Path(source).expanduser().resolve()
    resolved_output = Path(output).expanduser().resolve()
    if resolved_source == resolved_output:
        raise ValueError("filtered SQLite output must differ from its source")
    if not track_ids:
        raise ValueError("classwise route must contain at least one track")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(resolved_output)
    _remove_sqlite_temporary_files(temporary)
    try:
        with sqlite3.connect(
            f"file:{resolved_source}?mode=ro", uri=True
        ) as source_connection:
            with sqlite3.connect(temporary) as output_connection:
                source_connection.backup(output_connection)
        with sqlite3.connect(temporary) as connection:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            )
            if journal_mode.lower() != "delete":
                raise RuntimeError(
                    "failed to finalize routed SQLite journal: "
                    f"{journal_mode}"
                )
            connection.execute(
                "CREATE TEMP TABLE selected_tracks(track_id TEXT PRIMARY KEY)"
            )
            connection.executemany(
                "INSERT INTO selected_tracks(track_id) VALUES (?)",
                ((track_id,) for track_id in track_ids),
            )
            connection.execute(
                """
                DELETE FROM masks
                WHERE track_id NOT IN (SELECT track_id FROM selected_tracks)
                """
            )
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "tracks" in tables:
                connection.execute(
                    """
                    DELETE FROM tracks
                    WHERE track_id NOT IN (SELECT track_id FROM selected_tracks)
                    """
                )
            count = int(
                connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
            )
            integrity = str(
                connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if integrity != "ok":
                raise RuntimeError(
                    f"filtered tracked SQLite integrity failed: {integrity}"
                )
            connection.commit()
        os.replace(temporary, resolved_output)
        return count
    finally:
        _remove_sqlite_temporary_files(temporary)


def merge_routed_outputs(
    reference: Path,
    groups: tuple[RoutedGroup, ...],
    output: Path,
    *,
    policy: ClassPostprocessPolicy,
    track_labels: Mapping[str, str],
) -> dict[str, object]:
    """Merge disjoint routed tracks and persist per-mask policy provenance."""

    resolved_reference = Path(reference).expanduser().resolve()
    resolved_output = Path(output).expanduser().resolve()
    if resolved_reference == resolved_output:
        raise ValueError("classwise output must differ from tracked input")
    settings_by_track: dict[str, ClassPostprocessSettings] = {}
    for group in groups:
        for track_id in group.track_ids:
            previous = settings_by_track.setdefault(track_id, group.settings)
            if previous != group.settings:
                raise RuntimeError(
                    f"track {track_id!r} was routed with conflicting policies"
                )
    if set(settings_by_track) != set(track_labels):
        missing = sorted(set(track_labels) - set(settings_by_track))
        extra = sorted(set(settings_by_track) - set(track_labels))
        raise RuntimeError(
            "classwise routing does not cover the tracked input: "
            f"missing={missing}, extra={extra}"
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(resolved_output)
    _remove_sqlite_temporary_files(temporary)
    input_masks = count_masks(resolved_reference)
    group_counts: dict[str, int] = {}
    try:
        routed_tracks = {
            track_id for group in groups for track_id in group.track_ids
        }
        direct_single_group = (
            len(groups) == 1 and routed_tracks == set(track_labels)
        )
        backup_source = (
            groups[0].predictions_sqlite
            if direct_single_group
            else resolved_reference
        )
        with sqlite3.connect(
            f"file:{Path(backup_source).resolve()}?mode=ro", uri=True
        ) as source_connection:
            with sqlite3.connect(temporary) as output_connection:
                source_connection.backup(output_connection)
        with sqlite3.connect(temporary) as connection:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            )
            if journal_mode.lower() != "delete":
                raise RuntimeError(
                    "failed to finalize merged SQLite journal: "
                    f"{journal_mode}"
                )
            connection.execute(
                "ATTACH DATABASE ? AS source_reference",
                (str(resolved_reference),),
            )
            if not direct_single_group:
                connection.execute("DELETE FROM masks")
                connection.commit()
            for group in groups:
                connection.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS allowed_route_tracks(
                        track_id TEXT PRIMARY KEY
                    )
                    """
                )
                connection.execute("DELETE FROM temp.allowed_route_tracks")
                connection.executemany(
                    "INSERT INTO temp.allowed_route_tracks(track_id) VALUES (?)",
                    ((track_id,) for track_id in group.track_ids),
                )
                connection.commit()
                if direct_single_group:
                    schema = "main"
                else:
                    connection.execute(
                        "ATTACH DATABASE ? AS route_db",
                        (str(group.predictions_sqlite.resolve()),),
                    )
                    schema = "route_db"
                try:
                    unexpected = connection.execute(
                        f"""
                        SELECT m.track_id
                        FROM {schema}.masks m
                        LEFT JOIN temp.allowed_route_tracks r
                          ON r.track_id=m.track_id
                        WHERE r.track_id IS NULL
                        LIMIT 1
                        """
                    ).fetchone()
                    if unexpected is not None:
                        raise RuntimeError(
                            f"group {group.group_id!r} emitted an unrouted "
                            f"track {str(unexpected[0])!r}"
                        )
                    group_counts[group.group_id] = int(
                        connection.execute(
                            f"SELECT COUNT(*) FROM {schema}.masks"
                        ).fetchone()[0]
                    )
                    if not direct_single_group:
                        collision = connection.execute(
                            """
                            SELECT incoming.frame, incoming.track_id
                            FROM route_db.masks incoming
                            JOIN main.masks existing
                              ON existing.frame=incoming.frame
                             AND existing.track_id=incoming.track_id
                            LIMIT 1
                            """
                        ).fetchone()
                        if collision is not None:
                            raise RuntimeError(
                                "classwise output collision at "
                                f"({int(collision[0])}, {str(collision[1])!r})"
                            )
                        connection.execute(
                            """
                            INSERT INTO main.masks(
                                frame, track_id, polygons, shape_type,
                                dilate_px, feather_px, mosaic_block,
                                mosaic_alias, label
                            )
                            SELECT
                                frame, track_id, polygons, shape_type,
                                dilate_px, feather_px, mosaic_block,
                                mosaic_alias, label
                            FROM route_db.masks
                            """
                        )
                        connection.commit()
                finally:
                    if not direct_single_group:
                        connection.execute("DETACH DATABASE route_db")
            connection.executescript(
                """
                DROP TABLE IF EXISTS class_postprocess_policies;
                DROP TABLE IF EXISTS mask_postprocess_provenance;
                CREATE TABLE class_postprocess_policies(
                    label TEXT PRIMARY KEY,
                    policy_source TEXT NOT NULL,
                    shape_mode TEXT NOT NULL,
                    keyframe_interval INTEGER NOT NULL,
                    max_gap INTEGER NOT NULL
                );
                CREATE TABLE mask_postprocess_provenance(
                    frame INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    policy_source TEXT NOT NULL,
                    shape_mode TEXT NOT NULL,
                    keyframe_interval INTEGER NOT NULL,
                    max_gap INTEGER NOT NULL,
                    is_gap_filled INTEGER NOT NULL,
                    PRIMARY KEY(frame, track_id),
                    CHECK(shape_mode IN ('polygon', 'ellipse')),
                    CHECK(keyframe_interval >= 1),
                    CHECK(max_gap >= 0),
                    CHECK(is_gap_filled IN (0, 1))
                );
                CREATE INDEX idx_mask_postprocess_provenance_label_frame
                    ON mask_postprocess_provenance(label, frame);
                CREATE TEMP TABLE routed_track_policies(
                    track_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    policy_source TEXT NOT NULL,
                    shape_mode TEXT NOT NULL,
                    keyframe_interval INTEGER NOT NULL,
                    max_gap INTEGER NOT NULL
                );
                """
            )
            connection.executemany(
                """
                INSERT INTO temp.routed_track_policies(
                    track_id, label, policy_source, shape_mode,
                    keyframe_interval, max_gap
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        track_id,
                        track_labels[track_id],
                        (
                            "class"
                            if track_labels[track_id] in policy.classes
                            else "default"
                        ),
                        settings.shape_mode,
                        settings.keyframe_interval,
                        settings.max_gap,
                    )
                    for track_id, settings in settings_by_track.items()
                ),
            )
            labels = sorted(set(track_labels.values()) | set(policy.classes))
            connection.executemany(
                """
                INSERT INTO class_postprocess_policies(
                    label, policy_source, shape_mode, keyframe_interval, max_gap
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        label,
                        "class" if label in policy.classes else "default",
                        policy.resolve(label).shape_mode,
                        policy.resolve(label).keyframe_interval,
                        policy.resolve(label).max_gap,
                    )
                    for label in labels
                ),
            )
            unrouted = connection.execute(
                """
                SELECT m.track_id
                FROM masks m
                LEFT JOIN temp.routed_track_policies p
                  ON p.track_id=m.track_id
                WHERE p.track_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if unrouted is not None:
                raise RuntimeError(
                    "classwise output contains track without policy: "
                    f"{str(unrouted[0])!r}"
                )
            connection.execute(
                """
                INSERT INTO mask_postprocess_provenance(
                    frame, track_id, label, policy_source, shape_mode,
                    keyframe_interval, max_gap, is_gap_filled
                )
                SELECT
                    m.frame,
                    m.track_id,
                    p.label,
                    p.policy_source,
                    p.shape_mode,
                    p.keyframe_interval,
                    p.max_gap,
                    CASE WHEN EXISTS(
                        SELECT 1
                        FROM source_reference.masks source
                        WHERE source.frame=m.frame
                          AND source.track_id=m.track_id
                    ) THEN 0 ELSE 1 END
                FROM masks m
                JOIN temp.routed_track_policies p
                  ON p.track_id=m.track_id
                """
            )
            output_masks = int(
                connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
            )
            gap_filled = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(is_gap_filled), 0)
                    FROM mask_postprocess_provenance
                    """
                ).fetchone()[0]
            )
            connection.commit()
            connection.execute("DETACH DATABASE source_reference")
            integrity = str(
                connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if integrity != "ok":
                raise RuntimeError(
                    f"classwise merged SQLite integrity failed: {integrity}"
                )
            connection.commit()
        os.replace(temporary, resolved_output)
    finally:
        _remove_sqlite_temporary_files(temporary)

    return {
        "input_masks": input_masks,
        "output_masks": output_masks,
        "gap_filled_masks": gap_filled,
        "groups": group_counts,
    }


__all__ = [
    "RoutedGroup",
    "count_masks",
    "filter_tracked_sqlite",
    "merge_routed_outputs",
    "read_track_labels",
]
