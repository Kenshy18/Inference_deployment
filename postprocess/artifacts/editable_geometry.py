"""Import authoritative native keyframes into the stable result SQLite.

Dense masks are staging data only.  These helpers preserve independently
selected authoring keyframes and native geometry so downstream software never
has to fit an ellipse back from a polygonized rendering cache.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _track_sort_key(track_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                float(points[index][0]) * float(points[(index + 1) % len(points)][1])
                - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
                for index in range(len(points))
            )
        )
        * 0.5
    )


def _decode_polygons(value: object) -> list[list[list[float]]]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("keyframe polygons must decode to a list")
    polygons: list[list[list[float]]] = []
    for raw_polygon in decoded:
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            raise ValueError("each keyframe polygon requires at least three points")
        polygon: list[list[float]] = []
        for raw_point in raw_polygon:
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise ValueError("polygon point must be [x, y]")
            x, y = float(raw_point[0]), float(raw_point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("polygon coordinates must be finite")
            polygon.append([x, y])
        polygons.append(polygon)
    # The production interpolator associates disconnected polygons by area.
    # Persist that stable order as slot_index for downstream readers.
    polygons.sort(key=_polygon_area, reverse=True)
    return polygons


def _cuts(connection: sqlite3.Connection) -> list[int]:
    return [
        int(row[0])
        for row in connection.execute("SELECT frame FROM cuts ORDER BY frame")
    ]


def _scene_id(cuts: list[int], frame: int) -> int:
    return bisect.bisect_right(cuts, int(frame))


def _label(connection: sqlite3.Connection, track_id: str) -> str:
    row = connection.execute(
        "SELECT COALESCE(label, '') FROM tracks WHERE track_id=?",
        (track_id,),
    ).fetchone()
    return "" if row is None else str(row[0])


def _confidence(
    connection: sqlite3.Connection,
    frame: int,
    track_id: str,
) -> float | None:
    row = connection.execute(
        """
        SELECT MAX(COALESCE(detector_score, score, class_score))
        FROM raw_tracked_masks
        WHERE frame=? AND final_track_id=?
        """,
        (int(frame), str(track_id)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    value = float(row[0])
    return value if math.isfinite(value) else None


def _upsert_track(
    connection: sqlite3.Connection,
    track_id: str,
    *,
    label: str,
    domain: str,
    confidence: float | None,
) -> None:
    connection.execute(
        """
        INSERT INTO tracks(
            track_id, label, domain, confidence, status
        ) VALUES (?, ?, ?, ?, 'active')
        ON CONFLICT(track_id) DO UPDATE SET
            label=CASE
                WHEN excluded.label <> '' THEN excluded.label
                ELSE tracks.label
            END,
            domain=excluded.domain,
            confidence=COALESCE(excluded.confidence, tracks.confidence)
        """,
        (track_id, label, domain, confidence),
    )


def _insert_segment(
    connection: sqlite3.Connection,
    *,
    track_id: str,
    scene_id: int,
    start_frame: int,
    end_frame: int,
    shape_type: str,
    interpolation_method: str,
    component_count: int,
    source_run_key: str,
    segment_reason: str,
) -> int:
    connection.execute(
        """
        INSERT OR IGNORE INTO mask_track_segments(
            track_id, scene_id, start_frame, end_frame, shape_type,
            interpolation_method, component_count, source_run_key,
            segment_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            scene_id,
            start_frame,
            end_frame,
            shape_type,
            interpolation_method,
            component_count,
            source_run_key,
            segment_reason,
        ),
    )
    row = connection.execute(
        """
        SELECT id FROM mask_track_segments
        WHERE track_id=? AND source_run_key=?
        """,
        (track_id, source_run_key),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to create editable mask segment")
    return int(row[0])


def _insert_keyframe(
    connection: sqlite3.Connection,
    *,
    segment_id: int,
    frame: int,
    keyframe_index: int,
    selection_reason: str,
    confidence: float | None,
) -> int:
    connection.execute(
        """
        INSERT OR IGNORE INTO mask_keyframes(
            segment_id, frame, keyframe_index, selection_reason, confidence
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            segment_id,
            int(frame),
            int(keyframe_index),
            selection_reason,
            confidence,
        ),
    )
    row = connection.execute(
        "SELECT id FROM mask_keyframes WHERE segment_id=? AND frame=?",
        (segment_id, int(frame)),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to create editable mask keyframe")
    return int(row[0])


def _insert_component(
    connection: sqlite3.Connection,
    *,
    keyframe_id: int,
    slot_index: int,
    geometry_type: str,
) -> int:
    connection.execute(
        """
        INSERT OR IGNORE INTO keyframe_components(
            keyframe_id, slot_index, geometry_type
        ) VALUES (?, ?, ?)
        """,
        (keyframe_id, int(slot_index), geometry_type),
    )
    row = connection.execute(
        """
        SELECT id FROM keyframe_components
        WHERE keyframe_id=? AND slot_index=?
        """,
        (keyframe_id, int(slot_index)),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to create keyframe geometry component")
    if str(row[0]) == "":
        raise RuntimeError("invalid keyframe component id")
    return int(row[0])


def _insert_polygon_geometry(
    connection: sqlite3.Connection,
    keyframe_id: int,
    polygons: list[list[list[float]]],
) -> None:
    for slot_index, polygon in enumerate(polygons):
        component_id = _insert_component(
            connection,
            keyframe_id=keyframe_id,
            slot_index=slot_index,
            geometry_type="polygon",
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO keyframe_polygon_rings(
                component_id, ring_index, ring_role
            ) VALUES (?, 0, 'exterior')
            """,
            (component_id,),
        )
        ring_id = int(
            connection.execute(
                """
                SELECT id FROM keyframe_polygon_rings
                WHERE component_id=? AND ring_index=0
                """,
                (component_id,),
            ).fetchone()[0]
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO keyframe_polygon_points(
                ring_id, point_index, x, y
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (ring_id, point_index, point[0], point[1])
                for point_index, point in enumerate(polygon)
            ),
        )


def _final_track_runs(
    connection: sqlite3.Connection,
    track_id: str,
    cuts: list[int],
) -> list[tuple[int, int, int]]:
    """Return contiguous runs as ``(start, end, maximum_polygon_count)``.

    A change in disconnected-polygon count is not a segment boundary.  The
    production interpolator handles that transition by selecting the nearer
    keyframe geometry.  Splitting there would promote an already-interpolated
    frame to a new keyframe and make a second interpolation drift from the
    original final mask.
    """

    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    previous: int | None = None
    component_count: int | None = None
    previous_scene: int | None = None
    for frame, polygons_json, shape_type in connection.execute(
        """
        SELECT frame, polygons, COALESCE(shape_type, 'polygon')
        FROM masks
        WHERE track_id=?
        ORDER BY frame
        """,
        (track_id,),
    ):
        if str(shape_type) != "polygon":
            continue
        decoded_count = len(_decode_polygons(polygons_json))
        current = int(frame)
        current_scene = _scene_id(cuts, current)
        boundary = start is not None and (
            previous is None
            or current != previous + 1
            or current_scene != previous_scene
        )
        if boundary:
            assert start is not None and previous is not None
            assert component_count is not None
            runs.append((start, previous, component_count))
            start = current
            component_count = decoded_count
        elif start is None:
            start = current
            component_count = decoded_count
        else:
            assert component_count is not None
            component_count = max(component_count, decoded_count)
        previous = current
        previous_scene = current_scene
    if start is not None and previous is not None and component_count is not None:
        runs.append((start, previous, component_count))
    return runs


def _polygon_key_rows(path: Path) -> dict[str, dict[int, tuple[str, str]]]:
    source = Path(path).expanduser().resolve()
    by_track: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        for frame, track_id, polygons, label in connection.execute(
            """
            SELECT frame, track_id, polygons, COALESCE(label, '')
            FROM masks ORDER BY track_id, frame
            """
        ):
            by_track[str(track_id)][int(frame)] = (str(polygons), str(label))
    return by_track


def import_polygon_keyframes(
    connection: sqlite3.Connection,
    path: Path,
    *,
    source_prefix: str = "polygon",
    only_unrepresented_tracks: bool = False,
) -> dict[str, int]:
    cuts = _cuts(connection)
    keys_by_track = _polygon_key_rows(path)
    represented_tracks = (
        {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT track_id FROM mask_track_segments"
            )
        }
        if only_unrepresented_tracks
        else set()
    )
    segment_count = 0
    keyframe_count = 0
    component_count = 0
    for track_id in sorted(keys_by_track, key=_track_sort_key):
        if track_id in represented_tracks:
            continue
        source_keys = keys_by_track[track_id]
        runs = _final_track_runs(connection, track_id, cuts)
        if not runs and source_keys:
            frames = sorted(source_keys)
            first_polygons = _decode_polygons(source_keys[frames[0]][0])
            runs = [(frames[0], frames[-1], len(first_polygons))]
        label = next(
            (value[1] for value in source_keys.values() if value[1]),
            _label(connection, track_id),
        )
        track_confidences = [
            value
            for frame in source_keys
            if (value := _confidence(connection, frame, track_id)) is not None
        ]
        _upsert_track(
            connection,
            track_id,
            label=label,
            domain="genital",
            confidence=max(track_confidences, default=None),
        )
        for run_index, (start, end, expected_components) in enumerate(runs):
            rows = {
                frame: value
                for frame, value in source_keys.items()
                if start <= frame <= end
            }
            for endpoint in (start, end):
                if endpoint in rows:
                    continue
                row = connection.execute(
                    """
                    SELECT polygons, COALESCE(label, '')
                    FROM masks WHERE frame=? AND track_id=?
                    """,
                    (endpoint, track_id),
                ).fetchone()
                if row is not None:
                    rows[endpoint] = (str(row[0]), str(row[1]))
            if not rows:
                continue
            segment_id = _insert_segment(
                connection,
                track_id=track_id,
                scene_id=_scene_id(cuts, start),
                start_frame=start,
                end_frame=end,
                shape_type="polygon",
                interpolation_method="linear_polygon_aligned_v1",
                component_count=expected_components,
                source_run_key=f"{source_prefix}:{track_id}:{run_index}:{start}:{end}",
                segment_reason="continuous_topology",
            )
            segment_count += 1
            ordered = sorted(rows.items())
            for keyframe_index, (frame, (polygons_json, _row_label)) in enumerate(
                ordered
            ):
                keyframe_id = _insert_keyframe(
                    connection,
                    segment_id=segment_id,
                    frame=frame,
                    keyframe_index=keyframe_index,
                    selection_reason=(
                        "fixed_interval" if frame in source_keys else "segment_endpoint"
                    ),
                    confidence=_confidence(connection, frame, track_id),
                )
                polygons = _decode_polygons(polygons_json)
                _insert_polygon_geometry(connection, keyframe_id, polygons)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO mask_geometry_provenance(
                        keyframe_id, source_kind, algorithm, parameters_json
                    ) VALUES (?, 'postprocess_polygon', ?, ?)
                    """,
                    (
                        keyframe_id,
                        "keyframes.polygon.interval",
                        json.dumps(
                            {"source_artifact": str(Path(path).resolve())},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                keyframe_count += 1
                component_count += len(polygons)
    return {
        "segments": segment_count,
        "keyframes": keyframe_count,
        "components": component_count,
    }


def _ellipse_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ellipse keyframes artifact must be a JSON list")
    return [dict(row) for row in payload]


def import_ellipse_keyframes(
    connection: sqlite3.Connection,
    path: Path,
    *,
    source_prefix: str = "ellipse",
) -> dict[str, int]:
    cuts = _cuts(connection)
    grouped: dict[
        tuple[str, str, int],
        dict[int, list[tuple[int, list[float]]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in _ellipse_rows(path):
        ellipse = [float(value) for value in row["ellipse"]]
        if len(ellipse) != 5 or not all(math.isfinite(value) for value in ellipse):
            raise ValueError("ellipse keyframe must contain five finite parameters")
        grouped[
            (
                str(row["track_id"]),
                str(row.get("mode", "")),
                int(row.get("run_id", 0)),
            )
        ][int(row["frame"])].append((int(row.get("slot_id", 0)), ellipse))

    segment_count = 0
    keyframe_count = 0
    component_count = 0
    for (track_id, mode, run_id), frames in sorted(
        grouped.items(),
        key=lambda item: (_track_sort_key(item[0][0]), item[0][2], item[0][1]),
    ):
        ordered_frames = sorted(frames)
        start, end = ordered_frames[0], ordered_frames[-1]
        slots = {
            slot_index
            for components in frames.values()
            for slot_index, _ellipse in components
        }
        label = _label(connection, track_id)
        confidences = [
            value
            for frame in ordered_frames
            if (value := _confidence(connection, frame, track_id)) is not None
        ]
        _upsert_track(
            connection,
            track_id,
            label=label,
            domain="genital",
            confidence=max(confidences, default=None),
        )
        segment_id = _insert_segment(
            connection,
            track_id=track_id,
            scene_id=_scene_id(cuts, start),
            start_frame=start,
            end_frame=end,
            shape_type="ellipse",
            interpolation_method="ellipse_log_axes_short_angle_v1",
            component_count=max(slots, default=-1) + 1,
            source_run_key=(
                f"{source_prefix}:{track_id}:{mode}:{run_id}:{start}:{end}"
            ),
            segment_reason=f"ellipse_{mode or 'unknown'}_run",
        )
        segment_count += 1
        for keyframe_index, frame in enumerate(ordered_frames):
            keyframe_id = _insert_keyframe(
                connection,
                segment_id=segment_id,
                frame=frame,
                keyframe_index=keyframe_index,
                selection_reason="adaptive",
                confidence=_confidence(connection, frame, track_id),
            )
            for slot_index, ellipse in sorted(frames[frame]):
                cx, cy, radius_x, radius_y, theta_degrees = ellipse
                component_id = _insert_component(
                    connection,
                    keyframe_id=keyframe_id,
                    slot_index=slot_index,
                    geometry_type="ellipse",
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO keyframe_ellipses(
                        component_id, cx, cy, radius_x, radius_y, theta_radians
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        component_id,
                        cx,
                        cy,
                        radius_x,
                        radius_y,
                        math.radians(theta_degrees),
                    ),
                )
                component_count += 1
            connection.execute(
                """
                INSERT OR REPLACE INTO mask_geometry_provenance(
                    keyframe_id, source_kind, algorithm, parameters_json
                ) VALUES (?, 'postprocess_ellipse', ?, ?)
                """,
                (
                    keyframe_id,
                    "keyframes.ellipse.dense",
                    json.dumps(
                        {
                            "source_artifact": str(Path(path).resolve()),
                            "mode": mode,
                            "run_id": run_id,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
            keyframe_count += 1
    return {
        "segments": segment_count,
        "keyframes": keyframe_count,
        "components": component_count,
    }


def import_classwise_keyframes(
    connection: sqlite3.Connection,
    path: Path,
) -> dict[str, int]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    totals = {"segments": 0, "keyframes": 0, "components": 0}
    for group in manifest.get("groups", []):
        group_id = str(group.get("id", "group"))
        pipeline_manifest = Path(str(group["pipeline_manifest"])).resolve()
        nested = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
        artifacts = dict(nested.get("artifacts", {}))
        if artifacts.get("keyframes_sqlite"):
            summary = import_polygon_keyframes(
                connection,
                Path(str(artifacts["keyframes_sqlite"])),
                source_prefix=f"classwise:{group_id}:polygon",
            )
        elif artifacts.get("keyframes_json"):
            summary = import_ellipse_keyframes(
                connection,
                Path(str(artifacts["keyframes_json"])),
                source_prefix=f"classwise:{group_id}:ellipse",
            )
        else:
            continue
        for key, value in summary.items():
            totals[key] += value
    return totals


def import_face_privacy_keyframes(
    connection: sqlite3.Connection,
    *,
    source_schema: str,
) -> dict[str, int]:
    source_tables = {
        str(row[0])
        for row in connection.execute(
            f"SELECT name FROM {source_schema}.sqlite_master WHERE type='table'"
        )
    }
    if "face_mask_geometries" not in source_tables:
        return {"segments": 0, "keyframes": 0, "components": 0}
    cuts = _cuts(connection)
    count = 0
    for (
        frame,
        track_id,
        geometry_type,
        cx,
        cy,
        half_width,
        half_height,
        theta_radians,
        label,
        confidence,
        source_observation_id,
        derivation,
        algorithm_version,
    ) in connection.execute(
        f"""
        SELECT g.frame, g.track_id, g.geometry_type, g.cx, g.cy,
               g.half_width, g.half_height, g.theta_radians,
               COALESCE(t.label, ''),
               p.confidence, p.source_observation_id, p.derivation,
               p.algorithm_version
        FROM {source_schema}.face_mask_geometries g
        LEFT JOIN {source_schema}.tracks t ON t.track_id=g.track_id
        JOIN {source_schema}.mask_provenance p
          ON p.frame=g.frame AND p.track_id=g.track_id
        ORDER BY g.frame, g.track_id
        """
    ):
        frame_value = int(frame)
        track_value = str(track_id)
        confidence_value = float(confidence)
        _upsert_track(
            connection,
            track_value,
            label=str(label),
            domain="face_privacy",
            confidence=confidence_value,
        )
        segment_id = _insert_segment(
            connection,
            track_id=track_value,
            scene_id=_scene_id(cuts, frame_value),
            start_frame=frame_value,
            end_frame=frame_value,
            shape_type=str(geometry_type),
            interpolation_method="none",
            component_count=1,
            source_run_key=f"face_privacy:{track_value}:{frame_value}",
            segment_reason="single_face_observation",
        )
        keyframe_id = _insert_keyframe(
            connection,
            segment_id=segment_id,
            frame=frame_value,
            keyframe_index=0,
            selection_reason="derived_face_geometry",
            confidence=confidence_value,
        )
        component_id = _insert_component(
            connection,
            keyframe_id=keyframe_id,
            slot_index=0,
            geometry_type=str(geometry_type),
        )
        if str(geometry_type) == "ellipse":
            connection.execute(
                """
                INSERT OR REPLACE INTO keyframe_ellipses(
                    component_id, cx, cy, radius_x, radius_y, theta_radians
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    component_id,
                    float(cx),
                    float(cy),
                    float(half_width),
                    float(half_height),
                    float(theta_radians),
                ),
            )
        else:
            connection.execute(
                """
                INSERT OR REPLACE INTO keyframe_rectangles(
                    component_id, cx, cy, half_width, half_height,
                    theta_radians
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    component_id,
                    float(cx),
                    float(cy),
                    float(half_width),
                    float(half_height),
                    float(theta_radians),
                ),
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO mask_geometry_provenance(
                keyframe_id, source_kind, source_face_observation_id,
                algorithm, parameters_json
            ) VALUES (?, 'face_privacy', ?, ?, ?)
            """,
            (
                keyframe_id,
                int(source_observation_id),
                str(algorithm_version),
                json.dumps(
                    {"derivation": str(derivation)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        count += 1
    return {"segments": count, "keyframes": count, "components": count}


def import_editable_geometry(
    connection: sqlite3.Connection,
    *,
    polygon_keyframes_sqlite: Path | None = None,
    ellipse_keyframes_json: Path | None = None,
    classwise_manifest: Path | None = None,
    face_source_schema: str | None = None,
) -> dict[str, int]:
    totals = {"segments": 0, "keyframes": 0, "components": 0}
    sources: Iterable[tuple[str, Path | None]] = (
        ("polygon", polygon_keyframes_sqlite),
        ("ellipse", ellipse_keyframes_json),
        ("classwise", classwise_manifest),
    )
    for kind, source in sources:
        if source is None:
            continue
        if kind == "polygon":
            summary = import_polygon_keyframes(connection, source)
        elif kind == "ellipse":
            summary = import_ellipse_keyframes(connection, source)
        else:
            summary = import_classwise_keyframes(connection, source)
        for key, value in summary.items():
            totals[key] += value
    if face_source_schema is not None:
        summary = import_face_privacy_keyframes(
            connection,
            source_schema=face_source_schema,
        )
        for key, value in summary.items():
            totals[key] += value
    return totals


__all__ = [
    "import_classwise_keyframes",
    "import_editable_geometry",
    "import_ellipse_keyframes",
    "import_face_privacy_keyframes",
    "import_polygon_keyframes",
]
