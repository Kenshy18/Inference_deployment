"""Export an experimental Pareto solution without changing the V3 contract.

The exporter copies a validated keyframe-primary V3 SQLite with SQLite's backup
API, then replaces only the selected track-segment keyframes.  Table, view,
index, and trigger definitions are fingerprinted before and after the update so
an experimental algorithm can never silently become a schema migration.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .fixed_budget import RawMask, Segment


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, COALESCE(sql, '')
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name, tbl_name
        """
    )
    encoded = json.dumps(list(rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _segment_ids(segments: dict[str, list[Segment]]) -> list[int]:
    return sorted(
        {int(segment.segment_id) for values in segments.values() for segment in values}
    )


def _placeholders(values: list[int]) -> str:
    if not values:
        raise ValueError("no selected segments to export")
    return ",".join("?" for _ in values)


def _delete_keyframe_tree(
    connection: sqlite3.Connection,
    segment_ids: list[int],
) -> None:
    marks = _placeholders(segment_ids)
    component_query = (
        "SELECT c.id FROM keyframe_components c "
        "JOIN mask_keyframes k ON k.id=c.keyframe_id "
        f"WHERE k.segment_id IN ({marks})"
    )
    ring_query = (
        "SELECT r.id FROM keyframe_polygon_rings r "
        f"WHERE r.component_id IN ({component_query})"
    )
    key_query = f"SELECT id FROM mask_keyframes WHERE segment_id IN ({marks})"
    connection.execute(
        f"DELETE FROM keyframe_polygon_points WHERE ring_id IN ({ring_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM keyframe_polygon_rings WHERE component_id IN ({component_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM keyframe_ellipses WHERE component_id IN ({component_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM keyframe_rectangles WHERE component_id IN ({component_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM keyframe_components WHERE keyframe_id IN ({key_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM mask_geometry_provenance WHERE keyframe_id IN ({key_query})",
        segment_ids,
    )
    connection.execute(
        f"DELETE FROM mask_keyframes WHERE segment_id IN ({marks})",
        segment_ids,
    )


def _insert_component_geometry(
    connection: sqlite3.Connection,
    *,
    keyframe_id: int,
    slot_index: int,
    kind: str,
    values,
) -> None:
    cursor = connection.execute(
        """
        INSERT INTO keyframe_components(keyframe_id, slot_index, geometry_type)
        VALUES (?, ?, ?)
        """,
        (keyframe_id, int(slot_index), str(kind)),
    )
    component_id = int(cursor.lastrowid)
    if kind == "polygon":
        ring = connection.execute(
            """
            INSERT INTO keyframe_polygon_rings(
                component_id, ring_index, ring_role
            ) VALUES (?, 0, 'exterior')
            """,
            (component_id,),
        )
        ring_id = int(ring.lastrowid)
        points = [(float(x), float(y)) for x, y in values]
        if len(points) < 3:
            raise ValueError("a polygon keyframe needs at least three points")
        connection.executemany(
            """
            INSERT INTO keyframe_polygon_points(ring_id, point_index, x, y)
            VALUES (?, ?, ?, ?)
            """,
            (
                (ring_id, point_index, point[0], point[1])
                for point_index, point in enumerate(points)
            ),
        )
        return
    numeric = tuple(float(value) for value in values)
    if len(numeric) != 5:
        raise ValueError(f"{kind} geometry requires five values")
    if kind == "ellipse":
        connection.execute(
            """
            INSERT INTO keyframe_ellipses(
                component_id, cx, cy, radius_x, radius_y, theta_radians
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (component_id, *numeric),
        )
        return
    if kind == "rectangle":
        connection.execute(
            """
            INSERT INTO keyframe_rectangles(
                component_id, cx, cy, half_width, half_height, theta_radians
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (component_id, *numeric),
        )
        return
    raise ValueError(f"unsupported keyframe geometry: {kind}")


def _refresh_contract_counts(connection: sqlite3.Connection) -> None:
    sources = {
        "final_annotations": "mask_keyframes",
        "native_polygon_keyframes": "keyframe_polygon_points",
        "native_ellipse_keyframes": "keyframe_ellipses",
        "native_rectangle_keyframes": "keyframe_rectangles",
    }
    for name, table in sources.items():
        count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        connection.execute(
            "UPDATE result_capabilities SET row_count=? WHERE name=?",
            (count, name),
        )
        connection.execute(
            "UPDATE result_components SET row_count=? WHERE name=?",
            (count, name),
        )


def export_selected_sqlite(
    baseline_path: Path,
    output_path: Path,
    segments: dict[str, list[Segment]],
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    label: str,
    target_mean_key_interval: float | None,
    recall_floor: float,
    selection_reason: str = "pareto_recall_constrained",
    algorithm: str = "experimental.polygon_recall_optimizer.pareto_dp",
) -> dict[str, object]:
    """Copy ``baseline_path`` and replace selected keys transactionally."""

    baseline = Path(baseline_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing SQLite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(f"file:{baseline}?mode=ro", uri=True) as source:
        before = schema_fingerprint(source)
        with sqlite3.connect(output) as destination:
            source.backup(destination)

    ids = _segment_ids(segments)
    for values in segments.values():
        for segment in values:
            frames = [int(keyframe.frame) for keyframe in segment.keyframes]
            if (
                not frames
                or frames[0] != int(segment.first_frame)
                or frames[-1] != int(segment.last_frame)
            ):
                raise ValueError(
                    f"segment {segment.segment_id} is only partially covered by "
                    "the selected keyframes; refusing a destructive replacement"
                )
    inserted_keyframes = 0
    inserted_components = 0
    with sqlite3.connect(output) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        with connection:
            _delete_keyframe_tree(connection, ids)
            for track_id, values in sorted(segments.items()):
                for segment in sorted(values, key=lambda value: value.segment_id):
                    connection.execute(
                        """
                        UPDATE mask_track_segments
                        SET interpolation_method=?
                        WHERE id=?
                        """,
                        (
                            str(segment.interpolation_method),
                            int(segment.segment_id),
                        ),
                    )
                    for keyframe_index, keyframe in enumerate(segment.keyframes):
                        raw = raw_masks.get((int(keyframe.frame), str(track_id)))
                        confidence = None if raw is None else float(raw.score)
                        cursor = connection.execute(
                            """
                            INSERT INTO mask_keyframes(
                                segment_id, frame, keyframe_index,
                                selection_reason, source_detection_id,
                                confidence, quality_score
                            ) VALUES (?, ?, ?, ?, NULL, ?, NULL)
                            """,
                            (
                                int(segment.segment_id),
                                int(keyframe.frame),
                                int(keyframe_index),
                                str(selection_reason),
                                confidence,
                            ),
                        )
                        keyframe_id = int(cursor.lastrowid)
                        for slot_index, component in keyframe.components:
                            _insert_component_geometry(
                                connection,
                                keyframe_id=keyframe_id,
                                slot_index=slot_index,
                                kind=component.kind,
                                values=component.values,
                            )
                            inserted_components += 1
                        connection.execute(
                            """
                            INSERT INTO mask_geometry_provenance(
                                keyframe_id, source_kind, source_detection_id,
                                source_face_observation_id, algorithm,
                                parameters_json
                            ) VALUES (?, 'postprocess_polygon', NULL, NULL, ?, ?)
                            """,
                            (
                                keyframe_id,
                                str(algorithm),
                                json.dumps(
                                    {
                                        "minimum_recall": float(recall_floor),
                                        "target_mean_key_interval": (
                                            None
                                            if target_mean_key_interval is None
                                            else float(target_mean_key_interval)
                                        ),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            ),
                        )
                        inserted_keyframes += 1
            if target_mean_key_interval is not None:
                policy_interval = max(1, round(target_mean_key_interval))
                connection.execute(
                    """
                    UPDATE class_postprocess_policies
                    SET keyframe_interval=? WHERE label=?
                    """,
                    (policy_interval, label),
                )
                connection.execute(
                    """
                    UPDATE mask_postprocess_provenance
                    SET keyframe_interval=? WHERE label=?
                    """,
                    (policy_interval, label),
                )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE annotation_state
                SET revision=revision+1, updated_at_utc=? WHERE id=1
                """,
                (now,),
            )
            _refresh_contract_counts(connection)

        after = schema_fingerprint(connection)
        if after != before:
            raise RuntimeError("SQLite schema changed during keyframe export")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        if integrity != "ok" or foreign_keys:
            raise RuntimeError(
                f"invalid exported SQLite: integrity={integrity}, "
                f"foreign_key_errors={len(foreign_keys)}"
            )
        annotation_revision = int(
            connection.execute(
                "SELECT revision FROM annotation_state WHERE id=1"
            ).fetchone()[0]
        )

    return {
        "path": str(output),
        "schema_fingerprint_before": before,
        "schema_fingerprint_after": after,
        "schema_unchanged": before == after,
        "segment_count": len(ids),
        "inserted_keyframes": inserted_keyframes,
        "inserted_components": inserted_components,
        "integrity_check": integrity,
        "foreign_key_error_count": len(foreign_keys),
        "annotation_revision": annotation_revision,
    }
