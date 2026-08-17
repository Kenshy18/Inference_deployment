"""Streaming input, gapfill, and track segmentation for polygon optimization."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    align_contour_slots,
    align_polygon_phase,
    build_local_mask_from_polygons,
    build_track_segments_with_gapfill,
    interpolate_gapfill_polygons,
    orient_ccw,
    parse_polygons,
    polygon_area,
    resample_closed_contour,
    sort_polygons,
)
from .defaults import (
    DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
    DEFAULT_ADAPTIVE_POINT_OFFSET,
    DEFAULT_ADAPTIVE_POINT_QUANTILE,
    DEFAULT_GAPFILL_ENABLED,
    DEFAULT_GAPFILL_MAX_GAP,
    DEFAULT_GAPFILL_TEMP_POINTS,
    DEFAULT_MAX_RUN_FRAMES,
    DEFAULT_MIN_ANCHORS_PER_CONTOUR,
    DEFAULT_PREDICTOR_BATCH_SIZE,
    DEFAULT_RUN_OVERLAP_FRAMES,
)
from .model import LearnedPointPredictor, compute_mask_descriptors
from .types import InstanceRun, TrackRow


def parse_float_list(text: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        values = list(default)
    return sorted(set(float(v) for v in values))


def split_long_track_segments(
    segments: list[list[TrackRow]],
    max_run_frames: int,
    run_overlap_frames: int,
) -> tuple[list[list[TrackRow]], dict[int, dict[str, int]], dict[str, int]]:
    source_lengths = [int(len(segment)) for segment in segments]
    max_source_segment_frames = int(max(source_lengths, default=0))
    max_frames = int(max_run_frames)
    requested_overlap = max(0, int(run_overlap_frames))
    disabled = max_frames <= 0
    effective_overlap = (
        0 if disabled else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
    )
    emit_stride = 0 if disabled else int(max(1, max_frames - 2 * effective_overlap))

    def make_stats(
        *,
        processed_segment_count: int,
        long_segment_count: int,
        chunked_source_segment_count: int,
        chunk_output_segment_count: int,
        max_processed_segment_frames: int,
        overlap_added_rows: int,
    ) -> dict[str, int]:
        return {
            "max_run_frames": int(max_frames),
            "run_overlap_frames": int(effective_overlap),
            "source_segment_count": int(len(segments)),
            "processed_segment_count": int(processed_segment_count),
            "long_segment_count": int(long_segment_count),
            "chunked_source_segment_count": int(chunked_source_segment_count),
            "chunk_output_segment_count": int(chunk_output_segment_count),
            "max_source_segment_frames": int(max_source_segment_frames),
            "max_processed_segment_frames": int(max_processed_segment_frames),
            "emit_stride_frames": int(emit_stride),
            "overlap_added_rows": int(overlap_added_rows),
        }

    if disabled or max_source_segment_frames <= max_frames:
        return (
            segments,
            {},
            make_stats(
                processed_segment_count=len(segments),
                long_segment_count=sum(
                    1
                    for length in source_lengths
                    if max_frames > 0 and length > max_frames
                ),
                chunked_source_segment_count=0,
                chunk_output_segment_count=0,
                max_processed_segment_frames=max_source_segment_frames,
                overlap_added_rows=0,
            ),
        )

    split_segments: list[list[TrackRow]] = []
    segment_meta: dict[int, dict[str, int]] = {}
    chunked_source_segment_count = 0
    chunk_output_segment_count = 0
    overlap_added_rows = 0
    max_processed_segment_frames = 0

    for source_run_id, segment in enumerate(segments):
        length = int(len(segment))
        if length <= max_frames:
            split_segments.append(segment)
            max_processed_segment_frames = max(max_processed_segment_frames, length)
            continue

        chunk_ranges: list[tuple[int, int, int, int]] = []
        for emit_start in range(0, length, emit_stride):
            emit_end = int(min(length, emit_start + emit_stride))
            if emit_start >= emit_end:
                continue
            process_start = int(max(0, emit_start - effective_overlap))
            process_end = int(min(length, emit_end + effective_overlap))
            chunk_ranges.append((process_start, process_end, int(emit_start), emit_end))

        chunk_count = int(len(chunk_ranges))
        if chunk_count <= 1:
            split_segments.append(segment)
            max_processed_segment_frames = max(max_processed_segment_frames, length)
            continue

        chunked_source_segment_count += 1
        chunk_output_segment_count += chunk_count
        for chunk_index, (
            process_start,
            process_end,
            emit_start,
            emit_end,
        ) in enumerate(chunk_ranges):
            chunk_rows = list(segment[process_start:process_end])
            split_segments.append(chunk_rows)
            segment_meta[id(chunk_rows)] = {
                "source_run_id": int(source_run_id),
                "chunk_index": int(chunk_index),
                "chunk_count": int(chunk_count),
                "process_start": int(process_start),
                "process_end": int(process_end),
                "emit_start": int(emit_start - process_start),
                "emit_end": int(emit_end - process_start),
            }
            processed_len = int(process_end - process_start)
            emitted_len = int(emit_end - emit_start)
            max_processed_segment_frames = max(
                max_processed_segment_frames, processed_len
            )
            overlap_added_rows += max(0, processed_len - emitted_len)

    return (
        split_segments,
        segment_meta,
        make_stats(
            processed_segment_count=len(split_segments),
            long_segment_count=sum(
                1 for length in source_lengths if length > max_frames
            ),
            chunked_source_segment_count=chunked_source_segment_count,
            chunk_output_segment_count=chunk_output_segment_count,
            max_processed_segment_frames=max_processed_segment_frames,
            overlap_added_rows=overlap_added_rows,
        ),
    )


def build_track_streams(
    rows: list[TrackRow],
    anchors_per_contour: int,
    predictor: LearnedPointPredictor | None = None,
    predictor_batch_size: int = DEFAULT_PREDICTOR_BATCH_SIZE,
    adaptive_anchor_counts: bool = DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
    adaptive_point_quantile: float = DEFAULT_ADAPTIVE_POINT_QUANTILE,
    adaptive_point_offset: int = DEFAULT_ADAPTIVE_POINT_OFFSET,
    min_anchors_per_contour: int = DEFAULT_MIN_ANCHORS_PER_CONTOUR,
    gapfill_enabled: bool = DEFAULT_GAPFILL_ENABLED,
    gapfill_max_gap: int = DEFAULT_GAPFILL_MAX_GAP,
    gapfill_temp_points: int = DEFAULT_GAPFILL_TEMP_POINTS,
    max_tracks: int = -1,
    max_run_frames: int = DEFAULT_MAX_RUN_FRAMES,
    run_overlap_frames: int = DEFAULT_RUN_OVERLAP_FRAMES,
) -> tuple[list[InstanceRun], dict[str, int]]:
    if max_tracks > 0:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.track_id] = counts.get(row.track_id, 0) + 1
        allowed_tracks = [
            track_id
            for track_id, _count in sorted(
                counts.items(), key=lambda item: (-item[1], int(item[0]))
            )
        ][: int(max_tracks)]
        allowed = set(allowed_tracks)
        rows = [row for row in rows if row.track_id in allowed]

    if bool(gapfill_enabled):
        segments, segmentation_stats = build_track_segments_with_gapfill(
            rows,
            max_gap=int(gapfill_max_gap),
            temp_points=int(gapfill_temp_points),
        )
    else:
        segments = []
        current: list[TrackRow] = []
        prev: TrackRow | None = None
        for row in rows:
            split = (
                prev is None
                or row.track_id != prev.track_id
                or row.frame != prev.frame + 1
                or len(row.polygons) != len(prev.polygons)
            )
            if split:
                if current:
                    segments.append(current)
                current = [
                    TrackRow(
                        frame=row.frame,
                        track_id=row.track_id,
                        polygons=row.polygons,
                        is_gapfill=row.is_gapfill,
                    )
                ]
            else:
                current.append(
                    TrackRow(
                        frame=row.frame,
                        track_id=row.track_id,
                        polygons=row.polygons,
                        is_gapfill=row.is_gapfill,
                    )
                )
            prev = row
        if current:
            segments.append(current)
        segmentation_stats = {
            "source_tracks": int(len({row.track_id for row in rows})),
            "source_rows": int(len(rows)),
            "gapfill_inserted_frames": 0,
            "gapfill_events": 0,
            "hard_split_events": 0,
            "segment_count": int(len(segments)),
        }

    segments, segment_meta, split_stats = split_long_track_segments(
        segments,
        max_run_frames=int(max_run_frames),
        run_overlap_frames=int(run_overlap_frames),
    )
    segmentation_stats.update(split_stats)

    streams: list[InstanceRun] = []
    for run_id, run_rows in enumerate(segments):
        meta = segment_meta.get(
            id(run_rows),
            {
                "source_run_id": int(run_id),
                "chunk_index": 0,
                "chunk_count": 1,
                "process_start": 0,
                "process_end": int(len(run_rows)),
                "emit_start": 0,
                "emit_end": int(len(run_rows)),
            },
        )
        source_run_id = int(meta["source_run_id"])
        chunk_index = int(meta["chunk_index"])
        chunk_count = int(meta["chunk_count"])
        chunk_suffix = (
            f":chunk{chunk_index + 1}of{chunk_count}" if chunk_count > 1 else ""
        )
        aligned_rows: list[list[np.ndarray]] = []
        gapfilled_flags: list[bool] = []
        prev_slots: list[np.ndarray] | None = None
        for row in run_rows:
            slots = align_contour_slots(prev_slots, row.polygons)
            aligned_rows.append(slots)
            gapfilled_flags.append(bool(row.is_gapfill))
            prev_slots = slots
        contour_count = len(aligned_rows[0]) if aligned_rows else 0
        if contour_count <= 0:
            continue

        predicted_total_points: np.ndarray | None = None
        run_anchor_count = int(anchors_per_contour)
        run_target_total_points = int(contour_count * run_anchor_count)
        if bool(adaptive_anchor_counts) and predictor is not None:
            masks = [build_local_mask_from_polygons(slots) for slots in aligned_rows]
            descriptors_list = [compute_mask_descriptors(mask) for mask in masks]
            predicted_totals = predictor.predict_total_points_batch(
                masks,
                descriptors_list,
                batch_size=int(predictor_batch_size),
            )
            predicted_total_points = np.asarray(predicted_totals, dtype=np.int32)
            quantile_total = int(
                math.ceil(
                    float(
                        np.quantile(
                            predicted_total_points.astype(np.float64),
                            float(adaptive_point_quantile),
                        )
                    )
                )
            )
            run_target_total_points = int(
                max(
                    contour_count * int(min_anchors_per_contour),
                    quantile_total + int(adaptive_point_offset),
                )
            )
            run_anchor_count = int(
                math.ceil(run_target_total_points / max(contour_count, 1))
            )
            run_anchor_count = int(
                np.clip(
                    run_anchor_count,
                    int(min_anchors_per_contour),
                    int(anchors_per_contour),
                )
            )
            run_target_total_points = int(run_anchor_count * contour_count)

        frame_anchor_stack: list[np.ndarray] = []
        frame_polygons: list[list[np.ndarray]] = []
        frame_areas: list[float] = []
        prev_anchors_by_slot: list[np.ndarray | None] = [None] * contour_count
        for slots in aligned_rows:
            contour_anchors: list[np.ndarray] = []
            contour_polygons: list[np.ndarray] = []
            area_sum = 0.0
            for slot_id in range(contour_count):
                poly = np.asarray(orient_ccw(slots[slot_id]), dtype=np.float32)
                anchor = resample_closed_contour(poly, int(run_anchor_count))
                anchor = align_polygon_phase(prev_anchors_by_slot[slot_id], anchor)
                contour_anchors.append(np.asarray(anchor, dtype=np.float32))
                contour_polygons.append(np.asarray(poly, dtype=np.float32))
                area_sum += float(polygon_area(poly))
                prev_anchors_by_slot[slot_id] = np.asarray(anchor, dtype=np.float32)
            frame_anchor_stack.append(np.asarray(contour_anchors, dtype=np.float32))
            frame_polygons.append(contour_polygons)
            frame_areas.append(area_sum)
        scale = float(
            max(
                math.sqrt(
                    max(
                        float(np.median(np.asarray(frame_areas, dtype=np.float64))), 1.0
                    )
                ),
                1.0,
            )
        )
        streams.append(
            InstanceRun(
                stream_id=f"{run_rows[0].track_id}:run{source_run_id}{chunk_suffix}:instance",
                track_id=run_rows[0].track_id,
                run_id=source_run_id,
                frame_numbers=np.asarray(
                    [row.frame for row in run_rows], dtype=np.int32
                ),
                gt_polygons=frame_polygons,
                anchors=np.asarray(frame_anchor_stack, dtype=np.float32),
                contour_count=contour_count,
                anchors_per_contour=int(run_anchor_count),
                scale=scale,
                gapfilled_flags=np.asarray(gapfilled_flags, dtype=np.uint8),
                predicted_total_points=predicted_total_points,
                run_target_total_points=int(run_target_total_points),
                emit_start_idx=int(meta["emit_start"]),
                emit_end_idx=int(meta["emit_end"]),
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_process_start=int(meta["process_start"]),
                chunk_process_end=int(meta["process_end"]),
                chunked_from_long_run=bool(chunk_count > 1),
            )
        )
    segmentation_stats["effective_stream_count"] = int(len(streams))
    return streams, segmentation_stats


def sqlite_allowed_track_ids(sqlite_path: Path, max_tracks: int) -> list[str] | None:
    if int(max_tracks) <= 0:
        return None
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id, count(*) AS n
            FROM masks
            GROUP BY track_id
            ORDER BY n DESC, CAST(track_id AS INTEGER)
            LIMIT ?
            """,
            (int(max_tracks),),
        ).fetchall()
    finally:
        conn.close()
    return [str(track_id) for track_id, _count in rows]


def sqlite_mask_stats_for_tracks(
    sqlite_path: Path, allowed_track_ids: list[str] | None
) -> dict[str, int]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if allowed_track_ids is None:
            row = conn.execute(
                "SELECT count(*), count(DISTINCT track_id) FROM masks"
            ).fetchone()
        elif not allowed_track_ids:
            row = (0, 0)
        else:
            placeholders = ",".join("?" for _ in allowed_track_ids)
            row = conn.execute(
                f"SELECT count(*), count(DISTINCT track_id) FROM masks WHERE track_id IN ({placeholders})",
                tuple(str(track_id) for track_id in allowed_track_ids),
            ).fetchone()
    finally:
        conn.close()
    return {"source_rows": int(row[0] or 0), "source_tracks": int(row[1] or 0)}


def iter_sqlite_track_rows(sqlite_path: Path, allowed_track_ids: list[str] | None):
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if allowed_track_ids is None:
            rows_iter = conn.execute(
                "SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
            )
        elif not allowed_track_ids:
            rows_iter = iter(())
        else:
            placeholders = ",".join("?" for _ in allowed_track_ids)
            rows_iter = conn.execute(
                f"SELECT frame, track_id, polygons FROM masks WHERE track_id IN ({placeholders}) ORDER BY CAST(track_id AS INTEGER), frame",
                tuple(str(track_id) for track_id in allowed_track_ids),
            )
        for frame, track_id, polygons_json in rows_iter:
            yield TrackRow(
                frame=int(frame),
                track_id=str(track_id),
                polygons=parse_polygons(str(polygons_json)),
            )
    finally:
        conn.close()


def iter_track_streams_from_sqlite(
    sqlite_path: Path,
    *,
    anchors_per_contour: int,
    predictor: LearnedPointPredictor | None,
    predictor_batch_size: int,
    adaptive_anchor_counts: bool,
    adaptive_point_quantile: float,
    adaptive_point_offset: int,
    min_anchors_per_contour: int,
    gapfill_enabled: bool,
    gapfill_max_gap: int,
    gapfill_temp_points: int,
    max_tracks: int,
    max_run_frames: int,
    run_overlap_frames: int,
    segmentation_stats: dict[str, int],
):
    allowed_track_ids = sqlite_allowed_track_ids(sqlite_path, int(max_tracks))
    source_stats = sqlite_mask_stats_for_tracks(sqlite_path, allowed_track_ids)
    max_frames = int(max_run_frames)
    requested_overlap = max(0, int(run_overlap_frames))
    effective_overlap = (
        0
        if max_frames <= 0
        else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
    )
    emit_stride = (
        0 if max_frames <= 0 else int(max(1, max_frames - 2 * effective_overlap))
    )
    segmentation_stats.clear()
    segmentation_stats.update(
        {
            "source_tracks": int(source_stats["source_tracks"]),
            "source_rows": int(source_stats["source_rows"]),
            "gapfill_inserted_frames": 0,
            "gapfill_events": 0,
            "hard_split_events": 0,
            "segment_count": 0,
            "max_run_frames": int(max_frames),
            "run_overlap_frames": int(effective_overlap),
            "source_segment_count": 0,
            "processed_segment_count": 0,
            "long_segment_count": 0,
            "chunked_source_segment_count": 0,
            "chunk_output_segment_count": 0,
            "max_source_segment_frames": 0,
            "max_processed_segment_frames": 0,
            "emit_stride_frames": int(emit_stride),
            "overlap_added_rows": 0,
            "effective_stream_count": 0,
        }
    )

    buffer: list[TrackRow] = []
    buffer_start_idx = 0
    segment_len = 0
    next_emit_start = 0
    source_run_id = 0
    chunk_index = 0
    current_track_id: str | None = None
    prev: TrackRow | None = None

    def build_runs_for_chunk(
        chunk_rows: list[TrackRow],
        *,
        emit_start: int,
        emit_end: int,
        process_start: int,
        process_end: int,
        chunk_idx: int,
        chunked: bool,
    ) -> list[InstanceRun]:
        runs, _ignored_stats = build_track_streams(
            chunk_rows,
            anchors_per_contour=int(anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(predictor_batch_size),
            adaptive_anchor_counts=bool(adaptive_anchor_counts),
            adaptive_point_quantile=float(adaptive_point_quantile),
            adaptive_point_offset=int(adaptive_point_offset),
            min_anchors_per_contour=int(min_anchors_per_contour),
            gapfill_enabled=False,
            gapfill_max_gap=int(gapfill_max_gap),
            gapfill_temp_points=int(gapfill_temp_points),
            max_tracks=-1,
            max_run_frames=0,
            run_overlap_frames=0,
            _release_predictor_after_build=False,
        )
        out: list[InstanceRun] = []
        for sub_idx, run in enumerate(runs):
            suffix = f":chunk{chunk_idx + 1}" if bool(chunked) else ""
            extra = f":part{sub_idx + 1}" if len(runs) > 1 else ""
            run.run_id = int(source_run_id)
            run.stream_id = f"{run.track_id}:run{source_run_id}{suffix}{extra}:instance"
            run.emit_start_idx = int(emit_start)
            run.emit_end_idx = int(emit_end)
            run.chunk_index = int(chunk_idx)
            run.chunk_count = -1 if bool(chunked) else 1
            run.chunk_process_start = int(process_start)
            run.chunk_process_end = int(process_end)
            run.chunked_from_long_run = bool(chunked)
            out.append(run)
        return out

    def emit_chunk(
        process_start: int,
        process_end: int,
        emit_start: int,
        emit_end: int,
        *,
        final: bool,
    ) -> list[InstanceRun]:
        nonlocal buffer, buffer_start_idx, next_emit_start, chunk_index
        start_offset = int(process_start - buffer_start_idx)
        end_offset = int(process_end - buffer_start_idx)
        chunk_rows = list(buffer[start_offset:end_offset])
        chunked = bool(segment_len > max_frames and max_frames > 0)
        emitted_len = int(emit_end - emit_start)
        processed_len = int(process_end - process_start)
        segmentation_stats["processed_segment_count"] += 1
        segmentation_stats["max_processed_segment_frames"] = int(
            max(segmentation_stats["max_processed_segment_frames"], processed_len)
        )
        if chunked:
            segmentation_stats["chunk_output_segment_count"] += 1
            segmentation_stats["overlap_added_rows"] += int(
                max(0, processed_len - emitted_len)
            )
        runs = build_runs_for_chunk(
            chunk_rows,
            emit_start=int(emit_start - process_start),
            emit_end=int(emit_end - process_start),
            process_start=int(process_start),
            process_end=int(process_end),
            chunk_idx=int(chunk_index),
            chunked=chunked,
        )
        segmentation_stats["effective_stream_count"] += int(len(runs))
        chunk_index += 1
        next_emit_start = int(emit_end)
        if not final:
            keep_from = int(max(0, next_emit_start - effective_overlap))
            drop_count = int(keep_from - buffer_start_idx)
            if drop_count > 0:
                buffer = buffer[drop_count:]
                buffer_start_idx = keep_from
        return runs

    def emit_ready_chunks(final: bool) -> list[InstanceRun]:
        out: list[InstanceRun] = []
        if segment_len <= 0:
            return out
        if max_frames <= 0:
            if final and next_emit_start < segment_len:
                out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
            return out
        if segment_len <= max_frames:
            if final and next_emit_start < segment_len:
                out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
            return out
        while next_emit_start < segment_len:
            emit_start = int(next_emit_start)
            emit_end = int(min(segment_len, emit_start + emit_stride))
            process_start = int(max(0, emit_start - effective_overlap))
            desired_process_end = int(emit_end + effective_overlap)
            if not final and desired_process_end > segment_len:
                break
            process_end = int(min(segment_len, desired_process_end))
            if not final and emit_end >= segment_len:
                break
            out.extend(
                emit_chunk(
                    process_start, process_end, emit_start, emit_end, final=final
                )
            )
            if final:
                continue
        return out

    def flush_segment() -> list[InstanceRun]:
        nonlocal buffer, buffer_start_idx, segment_len, next_emit_start, source_run_id, chunk_index, prev
        if segment_len <= 0:
            return []
        segmentation_stats["segment_count"] += 1
        segmentation_stats["source_segment_count"] += 1
        segmentation_stats["max_source_segment_frames"] = int(
            max(segmentation_stats["max_source_segment_frames"], segment_len)
        )
        if max_frames > 0 and segment_len > max_frames:
            segmentation_stats["long_segment_count"] += 1
            segmentation_stats["chunked_source_segment_count"] += 1
        runs = emit_ready_chunks(final=True)
        source_run_id += 1
        buffer = []
        buffer_start_idx = 0
        segment_len = 0
        next_emit_start = 0
        chunk_index = 0
        prev = None
        return runs

    def append_segment_row(row: TrackRow) -> list[InstanceRun]:
        nonlocal segment_len
        buffer.append(row)
        segment_len += 1
        return emit_ready_chunks(final=False)

    for row in iter_sqlite_track_rows(sqlite_path, allowed_track_ids):
        if current_track_id is not None and str(row.track_id) != current_track_id:
            for run in flush_segment():
                yield run
        if current_track_id != str(row.track_id):
            current_track_id = str(row.track_id)
            prev = None

        current_slots = sort_polygons(row.polygons)
        if prev is None:
            first = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(first):
                yield run
            prev = first
            continue

        prev_slots = sort_polygons(prev.polygons)
        same_contour_count = len(prev_slots) == len(current_slots)
        gap = int(row.frame) - int(prev.frame) - 1
        if same_contour_count:
            current_slots = align_contour_slots(prev_slots, current_slots)

        if (not bool(gapfill_enabled)) and gap > 0:
            for run in flush_segment():
                yield run
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[
                    np.asarray(poly, dtype=np.float32)
                    for poly in sort_polygons(row.polygons)
                ],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        if gap <= 0 and same_contour_count:
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        can_gapfill = (
            bool(gapfill_enabled)
            and same_contour_count
            and gap > 0
            and gap <= int(gapfill_max_gap)
        )
        if can_gapfill:
            for step in range(1, gap + 1):
                interp_polys = interpolate_gapfill_polygons(
                    prev_slots,
                    current_slots,
                    step=step,
                    gap=gap,
                    temp_points=int(gapfill_temp_points),
                )
                gap_row = TrackRow(
                    frame=int(prev.frame) + step,
                    track_id=str(row.track_id),
                    polygons=[
                        np.asarray(poly, dtype=np.float32) for poly in interp_polys
                    ],
                    is_gapfill=True,
                )
                for run in append_segment_row(gap_row):
                    yield run
            segmentation_stats["gapfill_events"] += 1
            segmentation_stats["gapfill_inserted_frames"] += int(gap)
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        for run in flush_segment():
            yield run
        segmentation_stats["hard_split_events"] += 1
        current = TrackRow(
            frame=int(row.frame),
            track_id=str(row.track_id),
            polygons=[
                np.asarray(poly, dtype=np.float32)
                for poly in sort_polygons(row.polygons)
            ],
            is_gapfill=bool(row.is_gapfill),
        )
        for run in append_segment_row(current):
            yield run
        prev = current

    for run in flush_segment():
        yield run
