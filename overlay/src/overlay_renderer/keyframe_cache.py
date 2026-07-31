"""Materialize ephemeral dense overlay rows from keyframe-primary V3 SQLite."""

from __future__ import annotations

import bisect
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Point = tuple[float, float]
Polygon = list[Point]


@dataclass(frozen=True)
class Component:
    kind: str
    values: tuple[float, ...] | Polygon


@dataclass(frozen=True)
class Keyframe:
    frame: int
    components: tuple[tuple[int, Component], ...]


def is_keyframe_primary(path: Path) -> bool:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return False
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "result_schema_info" not in tables:
            return False
        info = dict(connection.execute("SELECT key, value FROM result_schema_info"))
        return info.get("compatibility_profile") == "keyframe-primary-v3"


def _ellipse_polygon(values: tuple[float, ...], *, points: int = 96) -> Polygon:
    cx, cy, radius_x, radius_y, theta = values
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return [
        (
            cx
            + radius_x * math.cos(phase) * cosine
            - radius_y * math.sin(phase) * sine,
            cy
            + radius_x * math.cos(phase) * sine
            + radius_y * math.sin(phase) * cosine,
        )
        for phase in (
            2.0 * math.pi * index / max(12, points) for index in range(max(12, points))
        )
    ]


def _rectangle_polygon(values: tuple[float, ...]) -> Polygon:
    cx, cy, half_width, half_height, theta = values
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return [
        (
            cx + x * cosine - y * sine,
            cy + x * sine + y * cosine,
        )
        for x, y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


def _perimeter_samples(points: Polygon, count: int) -> Polygon:
    following = points[1:] + points[:1]
    lengths = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, following, strict=True)
    ]
    perimeter = sum(lengths)
    if perimeter <= 1e-6:
        return [points[0]] * count
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    output: Polygon = []
    segment = 0
    for index in range(count):
        distance = perimeter * index / count
        while segment + 1 < len(cumulative) and cumulative[segment + 1] <= distance:
            segment += 1
        ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-6)
        first = points[segment]
        second = following[segment]
        output.append(
            (
                (1.0 - ratio) * first[0] + ratio * second[0],
                (1.0 - ratio) * first[1] + ratio * second[1],
            )
        )
    return output


def _align(reference: Polygon, candidate: Polygon) -> Polygon:
    best = candidate
    best_error = float("inf")
    for variant in (candidate, list(reversed(candidate))):
        for shift in range(len(variant)):
            shifted = variant[shift:] + variant[:shift]
            error = sum(
                (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2
                for left, right in zip(reference, shifted, strict=True)
            ) / len(reference)
            if error < best_error:
                best = shifted
                best_error = error
    return best


def _interpolate_angle(left: float, right: float, alpha: float) -> float:
    delta = (right - left + math.pi / 2.0) % math.pi - math.pi / 2.0
    return left + alpha * delta


def _interpolate_component(
    left: Component,
    right: Component,
    alpha: float,
) -> Component:
    if left.kind != right.kind:
        return left if alpha < 0.5 else right
    if left.kind == "polygon":
        left_points = list(left.values)  # type: ignore[arg-type]
        right_points = list(right.values)  # type: ignore[arg-type]
        count = max(8, len(left_points), len(right_points))
        left_sample = _perimeter_samples(left_points, count)
        right_sample = _align(left_sample, _perimeter_samples(right_points, count))
        return Component(
            "polygon",
            [
                (
                    (1.0 - alpha) * first[0] + alpha * second[0],
                    (1.0 - alpha) * first[1] + alpha * second[1],
                )
                for first, second in zip(left_sample, right_sample, strict=True)
            ],
        )
    left_values = tuple(left.values)  # type: ignore[arg-type]
    right_values = tuple(right.values)  # type: ignore[arg-type]
    if left.kind == "ellipse":
        return Component(
            "ellipse",
            (
                (1.0 - alpha) * left_values[0] + alpha * right_values[0],
                (1.0 - alpha) * left_values[1] + alpha * right_values[1],
                math.exp(
                    (1.0 - alpha) * math.log(max(left_values[2], 1e-6))
                    + alpha * math.log(max(right_values[2], 1e-6))
                ),
                math.exp(
                    (1.0 - alpha) * math.log(max(left_values[3], 1e-6))
                    + alpha * math.log(max(right_values[3], 1e-6))
                ),
                _interpolate_angle(left_values[4], right_values[4], alpha),
            ),
        )
    return Component(
        "rectangle",
        tuple(
            (1.0 - alpha) * first + alpha * second
            for first, second in zip(left_values[:4], right_values[:4], strict=True)
        )
        + (_interpolate_angle(left_values[4], right_values[4], alpha),),
    )


def _polygon_area(points: Polygon) -> float:
    following = points[1:] + points[:1]
    return (
        abs(
            sum(
                first[0] * second[1] - second[0] * first[1]
                for first, second in zip(points, following, strict=True)
            )
        )
        * 0.5
    )


def _interpolate_polygon_keyframes(
    left: Keyframe,
    right: Keyframe,
    alpha: float,
) -> tuple[Component, ...]:
    """Match the postprocess polygon interpolator's area-based slot pairing."""

    left_components = [
        component for _slot, component in left.components if component.kind == "polygon"
    ]
    right_components = [
        component
        for _slot, component in right.components
        if component.kind == "polygon"
    ]
    if len(left_components) != len(right_components):
        return tuple(left_components if alpha < 0.5 else right_components)
    left_arrays = sorted(
        (
            np.asarray(component.values, dtype=np.float64)
            for component in left_components
        ),
        key=_numpy_polygon_area,
        reverse=True,
    )
    right_arrays = sorted(
        (
            np.asarray(component.values, dtype=np.float64)
            for component in right_components
        ),
        key=_numpy_polygon_area,
        reverse=True,
    )
    output: list[Component] = []
    for left_points, right_points in zip(left_arrays, right_arrays, strict=True):
        count = max(8, len(left_points), len(right_points))
        left_sample = _numpy_resample(left_points, count)
        right_sample = _numpy_align(left_sample, _numpy_resample(right_points, count))
        output.append(
            Component(
                "polygon",
                ((1.0 - alpha) * left_sample + alpha * right_sample).tolist(),
            )
        )
    return tuple(output)


def _numpy_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) * 0.5


def _numpy_resample(points: np.ndarray, count: int) -> np.ndarray:
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-6:
        return np.repeat(points[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, perimeter, count, endpoint=False)
    output = np.empty((count, 2), dtype=np.float64)
    for index, distance in enumerate(samples):
        segment = min(
            max(
                int(np.searchsorted(cumulative, distance, side="right") - 1),
                0,
            ),
            len(points) - 1,
        )
        ratio = (distance - cumulative[segment]) / max(lengths[segment], 1e-6)
        output[index] = (1.0 - ratio) * points[segment] + ratio * following[segment]
    return output


def _numpy_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    best = candidate
    best_error = float("inf")
    for variant in (candidate, candidate[::-1]):
        for shift in range(len(variant)):
            shifted = np.roll(variant, shift, axis=0)
            error = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if error < best_error:
                best = shifted
                best_error = error
    return best


def _components_at(
    keyframes: list[Keyframe],
    frame: int,
    interpolation_method: str,
) -> tuple[Component, ...]:
    frames = [keyframe.frame for keyframe in keyframes]
    position = bisect.bisect_left(frames, frame)
    if position < len(keyframes) and frames[position] == frame:
        return tuple(component for _slot, component in keyframes[position].components)
    polygon_segment = all(
        component.kind == "polygon"
        for keyframe in keyframes
        for _slot, component in keyframe.components
    )
    if polygon_segment:
        if position == 0:
            return tuple(component for _slot, component in keyframes[0].components)
        if position == len(keyframes):
            return tuple(component for _slot, component in keyframes[-1].components)
        left = keyframes[position - 1]
        right = keyframes[position]
        alpha = (frame - left.frame) / (right.frame - left.frame)
        return _interpolate_polygon_keyframes(left, right, alpha)

    # Ellipse and rectangle components can be absent at individual
    # keyframes, so interpolate each stable slot independently.
    by_slot: dict[int, list[tuple[int, Component]]] = {}
    for keyframe in keyframes:
        for slot, component in keyframe.components:
            by_slot.setdefault(slot, []).append((keyframe.frame, component))
    output: list[tuple[int, Component]] = []
    for slot, samples in sorted(by_slot.items()):
        frames = [sample_frame for sample_frame, _component in samples]
        position = bisect.bisect_left(frames, frame)
        if position < len(samples) and frames[position] == frame:
            output.append((slot, samples[position][1]))
            continue
        if position == 0:
            output.append((slot, samples[0][1]))
            continue
        if position == len(samples):
            output.append((slot, samples[-1][1]))
            continue
        left_frame, left = samples[position - 1]
        right_frame, right = samples[position]
        alpha = (frame - left_frame) / (right_frame - left_frame)
        output.append((slot, _interpolate_component(left, right, alpha)))
    return tuple(component for _slot, component in output)


def _load_keyframes(
    connection: sqlite3.Connection,
    segment_id: int,
) -> list[Keyframe]:
    rows = connection.execute(
        """
        SELECT k.id AS keyframe_id, k.frame, c.id AS component_id,
               c.slot_index, c.geometry_type,
               e.cx, e.cy, e.radius_x, e.radius_y, e.theta_radians,
               r.cx, r.cy, r.half_width, r.half_height, r.theta_radians,
               pr.ring_role, pp.point_index, pp.x, pp.y
        FROM mask_keyframes k
        JOIN keyframe_components c ON c.keyframe_id=k.id
        LEFT JOIN keyframe_ellipses e ON e.component_id=c.id
        LEFT JOIN keyframe_rectangles r ON r.component_id=c.id
        LEFT JOIN keyframe_polygon_rings pr
          ON pr.component_id=c.id AND pr.ring_role='exterior'
        LEFT JOIN keyframe_polygon_points pp ON pp.ring_id=pr.id
        WHERE k.segment_id=?
        ORDER BY k.frame, c.slot_index, pr.ring_index, pp.point_index
        """,
        (segment_id,),
    )
    grouped: dict[int, dict[int, Component | list[Point]]] = {}
    for row in rows:
        frame = int(row[1])
        slot = int(row[3])
        kind = str(row[4])
        slots = grouped.setdefault(frame, {})
        if kind == "polygon":
            points = slots.setdefault(slot, [])
            assert isinstance(points, list)
            if row[17] is not None:
                points.append((float(row[17]), float(row[18])))
        elif kind == "ellipse":
            slots[slot] = Component(
                kind, tuple(float(row[index]) for index in range(5, 10))
            )
        else:
            slots[slot] = Component(
                kind, tuple(float(row[index]) for index in range(10, 15))
            )
    output: list[Keyframe] = []
    for frame, slots in sorted(grouped.items()):
        components: list[tuple[int, Component]] = []
        for slot, value in sorted(slots.items()):
            components.append(
                (
                    slot,
                    value
                    if isinstance(value, Component)
                    else Component("polygon", value),
                )
            )
        output.append(Keyframe(frame, tuple(components)))
    return output


def _component_polygons(
    components: tuple[Component, ...],
    *,
    face_label: str | None = None,
) -> list[Polygon]:
    output: list[Polygon] = []
    for component in components:
        if component.kind == "polygon":
            output.append(list(component.values))  # type: ignore[arg-type]
        elif component.kind == "ellipse":
            output.append(
                _ellipse_polygon(
                    tuple(component.values),  # type: ignore[arg-type]
                    points=(
                        64
                        if face_label is not None and face_label.casefold() == "eyes"
                        else 96
                    ),
                )
            )
        else:
            output.append(
                _rectangle_polygon(tuple(component.values))  # type: ignore[arg-type]
            )
    return output


def _polygon_components_for_final_frame(
    keyframes: list[Keyframe],
    observed_components: dict[int, tuple[Component, ...]],
    observed_frames: list[int],
    frame: int,
    interpolation_method: str,
) -> tuple[Component, ...]:
    """Reproduce the polygon pipeline's observed-frame then gap-fill order."""

    if not observed_frames:
        return _components_at(keyframes, frame, interpolation_method)
    exact = observed_components.get(frame)
    if exact is not None:
        return exact
    position = bisect.bisect_left(observed_frames, frame)
    if position == 0:
        return observed_components[observed_frames[0]]
    if position == len(observed_frames):
        return observed_components[observed_frames[-1]]
    left_frame = observed_frames[position - 1]
    right_frame = observed_frames[position]
    alpha = (frame - left_frame) / (right_frame - left_frame)
    left = Keyframe(
        left_frame,
        tuple(enumerate(observed_components[left_frame])),
    )
    right = Keyframe(
        right_frame,
        tuple(enumerate(observed_components[right_frame])),
    )
    return _interpolate_polygon_keyframes(left, right, alpha)


def _create_cache_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE masks(
            frame INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            polygons TEXT NOT NULL,
            shape_type TEXT NOT NULL,
            label TEXT,
            is_keyframe INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(frame, track_id)
        ) WITHOUT ROWID;
        CREATE TABLE mask_ellipses(
            frame INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            cx REAL NOT NULL,
            cy REAL NOT NULL,
            radius_x REAL NOT NULL,
            radius_y REAL NOT NULL,
            theta_radians REAL NOT NULL,
            point_count INTEGER NOT NULL,
            label TEXT,
            is_keyframe INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(frame, track_id, slot_index)
        ) WITHOUT ROWID;
        CREATE TABLE mask_rectangles(
            frame INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            cx REAL NOT NULL,
            cy REAL NOT NULL,
            half_width REAL NOT NULL,
            half_height REAL NOT NULL,
            theta_radians REAL NOT NULL,
            label TEXT,
            is_keyframe INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(frame, track_id, slot_index)
        ) WITHOUT ROWID;
        CREATE TABLE cache_info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE cuts(frame INTEGER PRIMARY KEY) WITHOUT ROWID;
        """
    )


def _load_cut_frames(
    connection: sqlite3.Connection,
    start_frame: int,
    end_frame: int,
) -> list[int]:
    has_cuts = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cuts'"
    ).fetchone()
    if has_cuts is None:
        return []
    return [
        int(row[0])
        for row in connection.execute(
            "SELECT frame FROM cuts WHERE frame BETWEEN ? AND ? ORDER BY frame",
            (start_frame, end_frame),
        )
    ]


def _split_keyframes_at_cuts(
    keyframes: list[Keyframe],
    cuts: list[int],
) -> list[list[Keyframe]]:
    """Return connected keyframe runs; a cut frame starts a new scene."""

    groups: list[list[Keyframe]] = []
    for keyframe in keyframes:
        scene = bisect.bisect_right(cuts, keyframe.frame)
        if not groups or bisect.bisect_right(cuts, groups[-1][0].frame) != scene:
            groups.append([])
        groups[-1].append(keyframe)
    return groups


def _materialize_final(
    source: sqlite3.Connection,
    output: sqlite3.Connection,
    start_frame: int,
    end_frame: int | None,
    *,
    compact_typed: bool,
    mask_domain: str | None,
) -> int:
    limit = (2**31 - 1) if end_frame is None else end_frame
    insert_sql = "INSERT OR REPLACE INTO masks VALUES (?, ?, ?, ?, ?, ?)"
    batch: list[tuple[object, ...]] = []
    ellipse_batch: list[tuple[object, ...]] = []
    rectangle_batch: list[tuple[object, ...]] = []
    total = 0
    # A worker shard can start after a cut while its source segment contains
    # keyframes on both sides.  Load the tiny global cut list for grouping;
    # only range-local cuts are copied to the disposable shard below.
    cuts = _load_cut_frames(source, 0, 2**31 - 1)

    def flush_typed() -> None:
        if ellipse_batch:
            output.executemany(
                "INSERT OR REPLACE INTO mask_ellipses "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ellipse_batch,
            )
            ellipse_batch.clear()
        if rectangle_batch:
            output.executemany(
                "INSERT OR REPLACE INTO mask_rectangles "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rectangle_batch,
            )
            rectangle_batch.clear()

    domain_clause = "1=1" if mask_domain is None else "t.domain=?"
    parameters: tuple[object, ...] = (
        (start_frame, limit)
        if mask_domain is None
        else (mask_domain, start_frame, limit)
    )
    rows = source.execute(
        f"""
        SELECT s.id, s.track_id, COALESCE(t.label, ''), t.domain,
               s.start_frame, s.end_frame, s.interpolation_method
        FROM mask_track_segments s
        JOIN tracks t ON t.track_id=s.track_id
        WHERE {domain_clause}
          AND s.end_frame>=? AND s.start_frame<=?
        ORDER BY s.start_frame, s.id
        """,
        parameters,
    )
    for segment_id, track_id, label, domain, first, last, method in rows:
        keyframes = _load_keyframes(source, int(segment_id))
        if not keyframes:
            continue
        keyframe_frames = {keyframe.frame for keyframe in keyframes}
        polygon_segment = all(
            component.kind == "polygon"
            for keyframe in keyframes
            for _slot, component in keyframe.components
        )
        observed_frames: list[int] = []
        observed_components: dict[int, tuple[Component, ...]] = {}
        if polygon_segment:
            observed_frames = [
                int(row[0])
                for row in source.execute(
                    """
                    SELECT DISTINCT frame
                    FROM tracking_assignments
                    WHERE removed_by_short_track=0
                      AND final_track_id=?
                      AND frame BETWEEN ? AND ?
                    ORDER BY frame
                    """,
                    (str(track_id), int(first), int(last)),
                )
            ]
            observed_components = {
                observed_frame: _components_at(keyframes, observed_frame, str(method))
                for observed_frame in observed_frames
            }
        for connected_keyframes in _split_keyframes_at_cuts(keyframes, cuts):
            # Only interpolate between actual keyframes.  Segment metadata may
            # be broader, but extrapolating into that range makes masks flash
            # or leak across disconnected observations.
            first_frame = max(
                start_frame,
                int(first),
                connected_keyframes[0].frame,
            )
            last_frame = min(
                limit,
                int(last),
                connected_keyframes[-1].frame,
            )
            connected_observed_frames = [
                frame
                for frame in observed_frames
                if first_frame <= frame <= last_frame
            ]
            connected_observed_components = {
                frame: observed_components[frame]
                for frame in connected_observed_frames
            }
            for frame in range(first_frame, last_frame + 1):
                components = (
                    _polygon_components_for_final_frame(
                        connected_keyframes,
                        connected_observed_components,
                        connected_observed_frames,
                        frame,
                        str(method),
                    )
                    if polygon_segment
                    else _components_at(connected_keyframes, frame, str(method))
                )
                if not components:
                    continue
                if compact_typed and all(
                    component.kind in {"ellipse", "rectangle"}
                    for component in components
                ):
                    for slot_index, component in enumerate(components):
                        values = tuple(component.values)  # type: ignore[arg-type]
                        common = (
                            frame,
                            str(track_id),
                            slot_index,
                            *values,
                        )
                        if component.kind == "ellipse":
                            ellipse_batch.append(
                                (
                                    *common,
                                    (
                                        64
                                        if str(domain) == "face_privacy"
                                        and str(label).casefold() == "eyes"
                                        else 96
                                    ),
                                    str(label),
                                    1 if frame in keyframe_frames else 0,
                                )
                            )
                        else:
                            rectangle_batch.append(
                                (
                                    *common,
                                    str(label),
                                    1 if frame in keyframe_frames else 0,
                                )
                            )
                    total += 1
                    if len(ellipse_batch) + len(rectangle_batch) >= 2048:
                        flush_typed()
                    continue
                polygons = _component_polygons(
                    components,
                    face_label=(
                        str(label) if str(domain) == "face_privacy" else None
                    ),
                )
                if not polygons:
                    continue
                batch.append(
                    (
                        frame,
                        str(track_id),
                        json.dumps(polygons, separators=(",", ":")),
                        components[0].kind,
                        str(label),
                        1 if frame in keyframe_frames else 0,
                    )
                )
                total += 1
                if len(batch) >= 1024:
                    output.executemany(insert_sql, batch)
                    batch.clear()
    if batch:
        output.executemany(insert_sql, batch)
    flush_typed()
    return total


def _materialize_tracked(
    source: sqlite3.Connection,
    output: sqlite3.Connection,
    start_frame: int,
    end_frame: int | None,
) -> int:
    limit = (2**31 - 1) if end_frame is None else end_frame
    rows = source.execute(
        """
        SELECT a.frame, a.final_track_id, COALESCE(a.final_label, ''),
               sp.polygon_index, pt.point_index, pt.x, pt.y
        FROM tracking_assignments a
        JOIN segmentation_polygons sp
          ON sp.detection_id=a.source_detection_id
        JOIN segmentation_points pt ON pt.polygon_id=sp.id
        WHERE a.removed_by_short_track=0
          AND a.final_track_id IS NOT NULL
          AND a.frame BETWEEN ? AND ?
        ORDER BY a.frame, a.final_track_id,
                 sp.polygon_index, pt.point_index
        """,
        (start_frame, limit),
    )
    current: tuple[int, str] | None = None
    label = ""
    polygons: list[Polygon] = []
    polygon_index: int | None = None
    polygon: Polygon = []
    batch: list[tuple[object, ...]] = []
    total = 0

    def flush() -> None:
        nonlocal polygon, polygons, polygon_index, total
        if current is None:
            return
        if polygon:
            polygons.append(polygon)
        if polygons:
            batch.append(
                (
                    current[0],
                    current[1],
                    json.dumps(polygons, separators=(",", ":")),
                    "polygon",
                    label,
                    0,
                )
            )
            total += 1
        polygon = []
        polygons = []
        polygon_index = None

    for frame, track_id, row_label, next_polygon_index, _point, x, y in rows:
        key = (int(frame), str(track_id))
        if current is not None and key != current:
            flush()
            if len(batch) >= 1024:
                output.executemany(
                    "INSERT OR REPLACE INTO masks VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
                batch.clear()
        current = key
        label = str(row_label)
        next_index = int(next_polygon_index)
        if polygon_index is not None and next_index != polygon_index:
            polygons.append(polygon)
            polygon = []
        polygon_index = next_index
        polygon.append((float(x), float(y)))
    flush()
    if batch:
        output.executemany(
            "INSERT OR REPLACE INTO masks VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
    return total


def materialize_overlay_cache(
    source_sqlite: Path,
    output_sqlite: Path,
    *,
    mode: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    compact_typed: bool = False,
    mask_domain: str | None = None,
) -> dict[str, object]:
    """Build a disposable dense cache for an existing overlay renderer."""

    if mode not in {"tracked", "final"}:
        raise ValueError("keyframe cache mode must be tracked or final")
    if mask_domain not in {None, "genital", "face_privacy"}:
        raise ValueError("unsupported mask domain")
    source_path = Path(source_sqlite).expanduser().resolve()
    output_path = Path(output_sqlite).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("overlay cache must differ from source SQLite")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    try:
        with sqlite3.connect(
            f"file:{source_path}?mode=ro", uri=True
        ) as source, sqlite3.connect(temporary) as output:
            source.execute("PRAGMA query_only=ON")
            output.execute("PRAGMA journal_mode=OFF")
            output.execute("PRAGMA synchronous=OFF")
            _create_cache_schema(output)
            copied_cuts = _load_cut_frames(
                source,
                start_frame,
                (2**31 - 1) if end_frame is None else end_frame,
            )
            output.executemany(
                "INSERT INTO cuts(frame) VALUES (?)",
                ((frame,) for frame in copied_cuts),
            )
            rows = (
                _materialize_tracked(source, output, start_frame, end_frame)
                if mode == "tracked"
                else _materialize_final(
                    source,
                    output,
                    start_frame,
                    end_frame,
                    compact_typed=compact_typed,
                    mask_domain=mask_domain,
                )
            )
            output.executemany(
                "INSERT INTO cache_info(key, value) VALUES (?, ?)",
                (
                    ("source_sqlite", str(source_path)),
                    ("mode", mode),
                    ("start_frame", str(start_frame)),
                    ("end_frame", "" if end_frame is None else str(end_frame)),
                    ("compact_typed", "1" if compact_typed else "0"),
                    ("mask_domain", "" if mask_domain is None else mask_domain),
                ),
            )
            output.commit()
            if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("overlay cache integrity check failed")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_sqlite": str(source_path),
        "cache_sqlite": str(output_path),
        "mode": mode,
        "compact_typed": compact_typed,
        "mask_domain": mask_domain,
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "size_bytes": output_path.stat().st_size,
    }


def _materialize_cache_job(
    arguments: tuple[Path, Path, str, int, int, bool, str | None],
) -> dict[str, object]:
    (
        source,
        output,
        mode,
        start_frame,
        end_frame,
        compact_typed,
        mask_domain,
    ) = arguments
    return materialize_overlay_cache(
        source,
        output,
        mode=mode,
        start_frame=start_frame,
        end_frame=end_frame,
        compact_typed=compact_typed,
        mask_domain=mask_domain,
    )


def materialize_overlay_cache_shards(
    source_sqlite: Path,
    output_directory: Path,
    *,
    mode: str,
    frame_ranges: list[tuple[int, int]],
    workers: int,
    mask_domain: str | None = None,
) -> dict[str, object]:
    """Build independent frame-range caches concurrently.

    Each native renderer worker reads only its own shard.  This avoids both
    serial keyframe expansion and a second merge/write of the dense cache.
    """

    if mode not in {"tracked", "final"}:
        raise ValueError("keyframe cache mode must be tracked or final")
    if workers < 1:
        raise ValueError("cache workers must be positive")
    if mask_domain not in {None, "genital", "face_privacy"}:
        raise ValueError("unsupported mask domain")
    if not frame_ranges:
        raise ValueError("at least one cache frame range is required")
    previous_end: int | None = None
    for start_frame, end_frame in frame_ranges:
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("invalid cache frame range")
        if previous_end is not None and start_frame <= previous_end:
            raise ValueError("cache frame ranges must be ordered and disjoint")
        previous_end = end_frame

    source = Path(source_sqlite).expanduser().resolve()
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    jobs = [
        (
            source,
            directory / f"masks-{index:02d}.sqlite",
            mode,
            start_frame,
            end_frame,
            True,
            mask_domain,
        )
        for index, (start_frame, end_frame) in enumerate(frame_ranges)
    ]
    started = time.perf_counter()
    if len(jobs) == 1 or workers == 1:
        shards = [_materialize_cache_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            shards = list(executor.map(_materialize_cache_job, jobs))
    return {
        "source_sqlite": str(source),
        "mode": mode,
        "strategy": "parallel-frame-range-shards",
        "workers": min(workers, len(jobs)),
        "mask_domain": mask_domain,
        "ranges": [
            {"start_frame": start, "end_frame": end} for start, end in frame_ranges
        ],
        "rows": sum(int(shard["rows"]) for shard in shards),
        "seconds": time.perf_counter() - started,
        "size_bytes": sum(int(shard["size_bytes"]) for shard in shards),
        "shards": shards,
    }


__all__ = [
    "is_keyframe_primary",
    "materialize_overlay_cache",
    "materialize_overlay_cache_shards",
]
