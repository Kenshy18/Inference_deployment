"""Original polygon recall safeguards before the v22 optimizer."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from contracts.mask_sqlite import MaskRow, read_mask_rows, write_mask_sqlite


def _parse(value: str) -> list[np.ndarray]:
    return [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in json.loads(value)
        if len(polygon) >= 3
    ]


def _dump(polygons: list[np.ndarray]) -> str:
    return json.dumps(
        [polygon.astype(np.float32).tolist() for polygon in polygons],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _bbox(polygons: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    if not polygons:
        return None
    points = np.concatenate(polygons, axis=0)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _expand_polygon(
    polygon: np.ndarray,
    *,
    width: int,
    height: int,
    trigger_px: float,
    expand_ratio: float,
    min_expand_px: float,
    max_expand_px: float,
    influence_px: float,
) -> tuple[np.ndarray, bool]:
    expanded = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    if len(expanded) < 3:
        return expanded, False
    x0, y0 = np.min(expanded, axis=0)
    x1, y1 = np.max(expanded, axis=0)
    span_x = max(1.0, float(x1 - x0 + 1.0))
    span_y = max(1.0, float(y1 - y0 + 1.0))
    influence = max(float(influence_px), float(trigger_px) + 1.0)
    changed = False
    if float(x0) <= trigger_px:
        amount = float(np.clip(span_x * expand_ratio, min_expand_px, max_expand_px))
        expanded[:, 0] -= amount * _smoothstep(
            ((trigger_px + influence) - expanded[:, 0]) / influence
        )
        changed = True
    if float(x1) >= float(width - 1) - trigger_px:
        amount = float(np.clip(span_x * expand_ratio, min_expand_px, max_expand_px))
        expanded[:, 0] += amount * _smoothstep(
            (expanded[:, 0] - (float(width - 1) - trigger_px - influence))
            / influence
        )
        changed = True
    if float(y0) <= trigger_px:
        amount = float(np.clip(span_y * expand_ratio, min_expand_px, max_expand_px))
        expanded[:, 1] -= amount * _smoothstep(
            ((trigger_px + influence) - expanded[:, 1]) / influence
        )
        changed = True
    if float(y1) >= float(height - 1) - trigger_px:
        amount = float(np.clip(span_y * expand_ratio, min_expand_px, max_expand_px))
        expanded[:, 1] += amount * _smoothstep(
            (expanded[:, 1] - (float(height - 1) - trigger_px - influence))
            / influence
        )
        changed = True
    return expanded, changed


def apply_border_expansion(
    source: Path,
    output: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    trigger_px: float = 10.0,
    expand_ratio: float = 0.10,
    min_expand_px: float = 6.0,
    max_expand_px: float = 40.0,
    influence_px: float = 24.0,
) -> tuple[Path, dict[str, object]]:
    rows: list[MaskRow] = []
    changed_rows = 0
    side_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for row in read_mask_rows(source):
        polygons = _parse(row.polygons)
        before = _bbox(polygons)
        changed = False
        expanded: list[np.ndarray] = []
        for polygon in polygons:
            value, item_changed = _expand_polygon(
                polygon,
                width=width,
                height=height,
                trigger_px=trigger_px,
                expand_ratio=expand_ratio,
                min_expand_px=min_expand_px,
                max_expand_px=max_expand_px,
                influence_px=influence_px,
            )
            expanded.append(value)
            changed = changed or item_changed
        if changed:
            changed_rows += 1
            if before is not None:
                x0, y0, x1, y1 = before
                side_counts["left"] += int(x0 <= trigger_px)
                side_counts["right"] += int(x1 >= width - 1 - trigger_px)
                side_counts["top"] += int(y0 <= trigger_px)
                side_counts["bottom"] += int(y1 >= height - 1 - trigger_px)
        rows.append(
            MaskRow(
                row.frame,
                row.track_id,
                _dump(expanded) if changed else row.polygons,
                row.label,
                row.shape_type,
            )
        )
    write_mask_sqlite(output, rows, reference_sqlite=source)
    return output, {
        "enabled": True,
        "total_rows": len(rows),
        "changed_rows": changed_rows,
        "changed_ratio": changed_rows / max(len(rows), 1),
        "side_counts": side_counts,
        "width": width,
        "height": height,
    }


def _cuts(source: Path) -> set[int]:
    with sqlite3.connect(source) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "cuts" not in tables:
            return set()
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(cuts)")}
        frame_column = next(
            (name for name in ("frame", "frame_index", "cut_frame") if name in columns),
            None,
        )
        if frame_column is None:
            return set()
        return {
            int(row[0])
            for row in connection.execute(f'SELECT "{frame_column}" FROM cuts')
        }


def _crosses_cut(left: int, right: int, cuts: set[int]) -> bool:
    low, high = sorted((left, right))
    return any(low < cut <= high for cut in cuts)


def _compatible(sequence: list[list[np.ndarray]]) -> bool:
    if not sequence:
        return False
    reference = sequence[0]
    return all(
        len(value) == len(reference)
        and all(a.shape == b.shape for a, b in zip(reference, value, strict=True))
        for value in sequence[1:]
    )


def _fit(sequence_frames: list[int], sequence: list[list[np.ndarray]], target: int) -> list[np.ndarray]:
    times = np.asarray(sequence_frames, dtype=np.float32)
    centered = times - float(np.mean(times))
    denominator = float(np.sum(centered * centered))
    output: list[np.ndarray] = []
    for contour in range(len(sequence[0])):
        values = np.stack([item[contour] for item in sequence], axis=0)
        mean = np.mean(values, axis=0)
        slope = (
            np.zeros_like(mean)
            if denominator <= 1e-6
            else np.sum(centered[:, None, None] * (values - mean), axis=0)
            / denominator
        )
        output.append((mean + slope * (float(target) - float(np.mean(times)))).astype(np.float32))
    return output


def _speed(sequence_frames: list[int], sequence: list[list[np.ndarray]]) -> float:
    fitted_next = _fit(sequence_frames, sequence, sequence_frames[-1] + 1)
    fitted_last = _fit(sequence_frames, sequence, sequence_frames[-1])
    return max(
        float(np.max(np.linalg.norm(a - b, axis=1)))
        for a, b in zip(fitted_next, fitted_last, strict=True)
    )


def apply_endpoint_extension(
    source: Path,
    output: Path,
    *,
    video: Path | None = None,
    extend_frames: int = 5,
    motion_frames: int = 10,
    max_speed_px: float = 1000.0,
) -> tuple[Path, dict[str, object]]:
    frame_count: int | None = None
    if video is not None:
        capture = cv2.VideoCapture(str(video))
        try:
            value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = value if value > 0 else None
        finally:
            capture.release()
    source_rows = read_mask_rows(source)
    rows_by_track: dict[str, list[MaskRow]] = defaultdict(list)
    for row in source_rows:
        rows_by_track[row.track_id].append(row)
    cuts = _cuts(source)
    existing = {(row.frame, row.track_id) for row in source_rows}
    inserted: list[MaskRow] = []
    events: list[dict[str, object]] = []
    for track_id, values in rows_by_track.items():
        ordered = sorted(values, key=lambda row: row.frame)
        if len(ordered) < 2 or extend_frames <= 0:
            continue
        for before in (True, False):
            motion = ordered[:motion_frames] if before else ordered[-motion_frames:]
            sequence = [_parse(row.polygons) for row in motion]
            if len(sequence) < 2 or not _compatible(sequence):
                continue
            speed = _speed([row.frame for row in motion], sequence)
            if speed > max_speed_px:
                continue
            endpoint = ordered[0] if before else ordered[-1]
            count = 0
            for step in range(1, extend_frames + 1):
                target = endpoint.frame - step if before else endpoint.frame + step
                if target < 0 or (frame_count is not None and target >= frame_count):
                    continue
                if (target, track_id) in existing:
                    continue
                if _crosses_cut(target, endpoint.frame, cuts):
                    continue
                polygons = _fit([row.frame for row in motion], sequence, target)
                inserted.append(
                    MaskRow(target, track_id, _dump(polygons), endpoint.label, "polygon")
                )
                existing.add((target, track_id))
                count += 1
            if count:
                events.append(
                    {
                        "track_id": track_id,
                        "side": "before" if before else "after",
                        "inserted": count,
                        "max_vertex_speed": speed,
                    }
                )
    output_rows = sorted(
        [*source_rows, *inserted], key=lambda row: (row.frame, row.track_id)
    )
    write_mask_sqlite(output, output_rows, reference_sqlite=source)
    return output, {
        "enabled": True,
        "source_rows": len(source_rows),
        "inserted_rows": len(inserted),
        "extend_frames": extend_frames,
        "motion_frames": motion_frames,
        "max_speed_px": max_speed_px,
        "video_frame_count": frame_count,
        "events": events,
    }


__all__ = ["apply_border_expansion", "apply_endpoint_extension"]
