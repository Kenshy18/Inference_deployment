"""Per-class input preparation for the Production polygon stage."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from .input_geometry import (
    apply_border_expansion,
    apply_endpoint_extension,
)
from classwise.sqlite import filter_tracked_sqlite, read_track_labels

from .runtime.candidate_config import CANDIDATE, CandidateConfig
from .vertex_policy import build_vertex_policy


def _finalize_class_projection(
    path: Path,
    label: str | None,
) -> dict[str, object]:
    """Drop unrelated raw provenance and reclaim pages in a temporary input.

    ``filter_tracked_sqlite`` projects the active ``masks`` and ``tracks``
    tables. Phase 2 does not consume raw provenance, but retaining all classes
    there makes every class copy as large as the original tracked database.
    Keep only the matching final label so the projection remains internally
    coherent and does not accumulate multi-gigabyte redundant files.
    """
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }
        with connection:
            if "raw_tracked_masks" in tables:
                if label is None:
                    connection.execute("DELETE FROM raw_tracked_masks")
                else:
                    connection.execute(
                        "DELETE FROM raw_tracked_masks "
                        "WHERE final_label IS NULL OR final_label <> ?",
                        (label,),
                    )
            if "raw_tracks" in tables:
                if label is None:
                    connection.execute("DELETE FROM raw_tracks")
                else:
                    connection.execute(
                        "DELETE FROM raw_tracks "
                        "WHERE final_label IS NULL OR final_label <> ?",
                        (label,),
                    )
        connection.execute("VACUUM")
        counts = {
            table: int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            )
            for table in ("masks", "tracks", "raw_tracked_masks", "raw_tracks")
            if table in tables
        }
        integrity = tuple(
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        )
        foreign_keys = tuple(connection.execute("PRAGMA foreign_key_check"))
    if integrity != ("ok",) or foreign_keys:
        raise RuntimeError(
            f"invalid class projection for {label}: "
            f"integrity={integrity}, foreign_keys={len(foreign_keys)}"
        )
    return {
        "counts": counts,
        "integrity_check": "ok",
        "foreign_key_errors": 0,
        "size_bytes": path.stat().st_size,
    }


def _empty_class_projection(source: Path, output: Path) -> dict[str, object]:
    """Create a schema-identical empty input for an absent genital class."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    shutil.copyfile(source, output)
    with sqlite3.connect(output) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }
        with connection:
            for table in ("masks", "tracks"):
                if table in tables:
                    connection.execute(f'DELETE FROM "{table}"')
    return _finalize_class_projection(output, None)


def prepare_classwise_source(
    tracked_sqlite: Path,
    output_root: Path,
    *,
    width: int,
    height: int,
    input_video: Path | None,
    config: CandidateConfig = CANDIDATE,
) -> tuple[Path, dict[str, object]]:
    """Split classes, apply approved edge safeguards, and write Phase-2 shim."""
    config.validate()
    tracked = Path(tracked_sqlite).resolve()
    root = Path(output_root).resolve()
    track_labels = read_track_labels(tracked)
    vertex_policy_path = root / "vertex_policy.json"
    if config.spatial.adaptive_vertex_policy:
        vertex_policy = build_vertex_policy(
            tracked,
            vertex_policy_path,
            width=int(width),
            height=int(height),
            track_labels=track_labels,
            config=config,
        )
    else:
        vertex_policy = {
            "enabled": False,
            "reason": "legacy_fixed_14_profile",
            "vertices_per_component": 14,
        }
    tracks_by_label = {
        label: tuple(
            track_id
            for track_id, assigned in sorted(track_labels.items())
            if assigned == label
        )
        for label in config.labels
    }
    active_labels = tuple(label for label in config.labels if tracks_by_label[label])
    prepared_by_label: dict[str, Path] = {}
    classes: dict[str, object] = {}
    settings = config.preparation
    for index, label in enumerate(config.labels):
        class_root = root / "classes" / f"{index:02d}_{label}"
        projected = class_root / "tracked.sqlite"
        projected.parent.mkdir(parents=True, exist_ok=True)
        if tracks_by_label[label]:
            input_rows = filter_tracked_sqlite(
                tracked,
                projected,
                track_ids=tracks_by_label[label],
            )
            projection = _finalize_class_projection(projected, label)
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
        else:
            input_rows = 0
            projection = _empty_class_projection(tracked, projected)
            border = projected
            endpoint = projected
            border_stats = {"enabled": False, "reason": "class_absent"}
            endpoint_stats = {"enabled": False, "reason": "class_absent"}
        prepared_by_label[label] = endpoint
        classes[label] = {
            "active": bool(tracks_by_label[label]),
            "track_ids": list(tracks_by_label[label]),
            "input_rows": input_rows,
            "projection": projection,
            "projected_sqlite": str(projected),
            "border_sqlite": str(border),
            "endpoint_sqlite": str(endpoint),
            "border": border_stats,
            "endpoint": endpoint_stats,
        }

    source_root = root / "phase2_source"
    source_root.mkdir(parents=True, exist_ok=True)
    optimizer_vertex_policy = source_root / "vertex_policy.json"
    if config.spatial.adaptive_vertex_policy:
        shutil.copyfile(vertex_policy_path, optimizer_vertex_policy)
    work = source_root / "interval_10/production_raw/work/04_classwise_postprocess"
    work.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, object]] = []
    for index, label in enumerate(config.labels):
        group = work / "groups" / f"{index:02d}_{label}"
        group.mkdir(parents=True, exist_ok=True)
        pipeline_manifest = group / "pipeline_manifest.json"
        pipeline_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stages": [
                        {
                            "id": "polygon_optimization",
                            "metadata": {
                                "optimizer": {
                                    "input_sqlite": str(
                                        prepared_by_label[label].resolve()
                                    )
                                }
                            },
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        groups.append(
            {
                "id": f"{index:02d}_{label}",
                "labels": [label],
                "pipeline_manifest": str(pipeline_manifest.resolve()),
            }
        )
    classwise_manifest = work / "classwise_manifest.json"
    classwise_manifest.write_text(
        json.dumps(
            {"schema_version": 1, "groups": groups},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root, {
        "tracked_sqlite": str(tracked),
        "width": int(width),
        "height": int(height),
        "input_video": None
        if input_video is None
        else str(Path(input_video).resolve()),
        "classes": classes,
        "active_labels": list(active_labels),
        "vertex_policy": vertex_policy,
        "vertex_policy_json": (
            str(optimizer_vertex_policy)
            if config.spatial.adaptive_vertex_policy
            else None
        ),
        "source_root": str(source_root),
        "classwise_manifest": str(classwise_manifest),
    }
