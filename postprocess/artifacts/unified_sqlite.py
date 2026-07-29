"""Build one stable public SQLite across every inference/postprocess mode."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
import hashlib
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from .editable_geometry import import_editable_geometry, import_polygon_keyframes


INFERENCE_SCHEMA_NAME = "instance-segmentation-unified-inference"
RESULT_SCHEMA_NAME = "video-mask-integrated-result"
RESULT_SCHEMA_VERSION = "3"
RESULT_CONTRACT_REVISION = "5"
RESULT_COMPATIBILITY_PROFILE = "keyframe-primary-v3"

RESULT_OWNED_TABLES = (
    "result_schema_info",
    "result_capabilities",
    "result_components",
    "processing_stage_runs",
    "processing_runs",
    # Dense tables are staging-only in V3.  They are dropped after native
    # editable geometry and tracking references have been imported.
    "tracked_masks",
    "tracked_tracks",
    "masks",
    "annotation_state",
    "face_track_interpolations",
    "face_tracking_assignments",
    "face_tracks",
    "tracking_assignments",
    "mask_geometry_provenance",
    "keyframe_polygon_points",
    "keyframe_polygon_rings",
    "keyframe_rectangles",
    "keyframe_ellipses",
    "keyframe_components",
    "mask_keyframes",
    "mask_track_segments",
    "tracks",
    "cuts",
    "cut_detection_metadata",
    "raw_tracked_masks",
    "raw_tracks",
    "class_postprocess_policies",
    "mask_postprocess_provenance",
    "mask_provenance",
)
RESULT_OWNED_VIEWS = (
    "editable_keyframe_components",
    "editable_polygon_vertices",
)

RESULT_REQUIRED_TABLES = frozenset(
    {
        "schema_info",
        "videos",
        "runs",
        "run_metadata",
        "model_executions",
        "model_metadata",
        "result_schema_info",
        "result_capabilities",
        "video_streams",
        "frames",
        "detections",
        "classifications",
        "classification_probabilities",
        "segmentations",
        "segmentation_polygons",
        "segmentation_points",
        "face_observations",
        "face_keypoints",
        "face_masks",
        "face_keypoint_class_probabilities",
        "face_keypoint_state_probabilities",
        "processing_runs",
        "processing_stage_runs",
        "annotation_state",
        "face_track_interpolations",
        "face_tracking_assignments",
        "face_tracks",
        "tracking_assignments",
        "raw_tracks",
        "class_postprocess_policies",
        "mask_postprocess_provenance",
        "mask_provenance",
        "tracks",
        "cuts",
        "cut_detection_metadata",
        "mask_track_segments",
        "mask_keyframes",
        "keyframe_components",
        "keyframe_ellipses",
        "keyframe_rectangles",
        "keyframe_polygon_rings",
        "keyframe_polygon_points",
        "mask_geometry_provenance",
    }
)

RESULT_SCHEMA_SQL = """
CREATE TABLE result_schema_info(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE result_capabilities(
    name TEXT PRIMARY KEY,
    available INTEGER NOT NULL CHECK(available IN (0, 1)),
    row_count INTEGER NOT NULL CHECK(row_count >= 0),
    source_table TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE result_components(
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(
        status IN (
            'complete', 'empty', 'not_requested', 'unsupported', 'failed'
        )
    ),
    row_count INTEGER NOT NULL CHECK(row_count >= 0),
    source_table TEXT NOT NULL,
    producer_stage_run_id INTEGER,
    details_json TEXT NOT NULL
);
CREATE TABLE processing_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT,
    completed_at_utc TEXT,
    resolved_config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    software_version TEXT,
    git_commit TEXT,
    UNIQUE(kind, config_hash)
);
CREATE TABLE processing_stage_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    processing_run_id INTEGER NOT NULL,
    stage_index INTEGER NOT NULL,
    stage_id TEXT NOT NULL,
    implementation TEXT NOT NULL,
    device TEXT,
    options_json TEXT NOT NULL,
    elapsed_seconds REAL,
    status TEXT NOT NULL,
    UNIQUE(processing_run_id, stage_index),
    FOREIGN KEY(processing_run_id) REFERENCES processing_runs(id)
);
CREATE TABLE annotation_state(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    revision INTEGER NOT NULL CHECK(revision >= 0),
    updated_at_utc TEXT,
    authoritative_geometry TEXT NOT NULL
        CHECK(authoritative_geometry = 'mask_keyframes'),
    dense_cache_policy TEXT NOT NULL
        CHECK(dense_cache_policy = 'not_materialized')
);
CREATE TABLE tracked_masks(
    frame INTEGER NOT NULL,
    track_id TEXT NOT NULL,
    polygons TEXT,
    shape_type TEXT,
    dilate_px INTEGER NOT NULL DEFAULT 0,
    feather_px INTEGER NOT NULL DEFAULT 0,
    mosaic_block INTEGER NOT NULL DEFAULT 0,
    mosaic_alias REAL NOT NULL DEFAULT 0,
    label TEXT,
    PRIMARY KEY(frame, track_id)
);
CREATE TABLE tracked_tracks(
    track_id TEXT PRIMARY KEY,
    label TEXT
);
CREATE TABLE masks(
    frame INTEGER NOT NULL,
    track_id TEXT NOT NULL,
    polygons TEXT,
    shape_type TEXT,
    dilate_px INTEGER NOT NULL DEFAULT 0,
    feather_px INTEGER NOT NULL DEFAULT 0,
    mosaic_block INTEGER NOT NULL DEFAULT 0,
    mosaic_alias REAL NOT NULL DEFAULT 0,
    label TEXT,
    PRIMARY KEY(frame, track_id)
);
CREATE TABLE tracks(
    track_id TEXT PRIMARY KEY,
    label TEXT,
    domain TEXT NOT NULL DEFAULT 'genital'
        CHECK(domain IN ('genital', 'face_privacy', 'other')),
    class_id INTEGER,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'active',
    source_model_execution_id INTEGER
);
CREATE TABLE mask_track_segments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    scene_id INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    shape_type TEXT NOT NULL CHECK(
        shape_type IN ('polygon', 'ellipse', 'rectangle')
    ),
    interpolation_method TEXT NOT NULL,
    component_count INTEGER NOT NULL CHECK(component_count >= 1),
    source_run_key TEXT NOT NULL,
    segment_reason TEXT NOT NULL,
    UNIQUE(track_id, source_run_key),
    CHECK(start_frame >= 0 AND end_frame >= start_frame),
    FOREIGN KEY(track_id) REFERENCES tracks(track_id)
);
CREATE TABLE mask_keyframes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL,
    frame INTEGER NOT NULL CHECK(frame >= 0),
    keyframe_index INTEGER NOT NULL CHECK(keyframe_index >= 0),
    selection_reason TEXT NOT NULL,
    source_detection_id INTEGER,
    confidence REAL,
    quality_score REAL,
    UNIQUE(segment_id, frame),
    UNIQUE(segment_id, keyframe_index),
    FOREIGN KEY(segment_id) REFERENCES mask_track_segments(id),
    FOREIGN KEY(source_detection_id) REFERENCES detections(id)
);
CREATE TABLE keyframe_components(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyframe_id INTEGER NOT NULL,
    slot_index INTEGER NOT NULL CHECK(slot_index >= 0),
    geometry_type TEXT NOT NULL CHECK(
        geometry_type IN ('polygon', 'ellipse', 'rectangle')
    ),
    UNIQUE(keyframe_id, slot_index),
    FOREIGN KEY(keyframe_id) REFERENCES mask_keyframes(id)
);
CREATE TABLE keyframe_ellipses(
    component_id INTEGER PRIMARY KEY,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    radius_x REAL NOT NULL CHECK(radius_x > 0),
    radius_y REAL NOT NULL CHECK(radius_y > 0),
    theta_radians REAL NOT NULL,
    FOREIGN KEY(component_id) REFERENCES keyframe_components(id)
);
CREATE TABLE keyframe_rectangles(
    component_id INTEGER PRIMARY KEY,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    half_width REAL NOT NULL CHECK(half_width > 0),
    half_height REAL NOT NULL CHECK(half_height > 0),
    theta_radians REAL NOT NULL,
    FOREIGN KEY(component_id) REFERENCES keyframe_components(id)
);
CREATE TABLE keyframe_polygon_rings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id INTEGER NOT NULL,
    ring_index INTEGER NOT NULL CHECK(ring_index >= 0),
    ring_role TEXT NOT NULL CHECK(ring_role IN ('exterior', 'hole')),
    UNIQUE(component_id, ring_index),
    FOREIGN KEY(component_id) REFERENCES keyframe_components(id)
);
CREATE TABLE keyframe_polygon_points(
    ring_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL CHECK(point_index >= 0),
    x REAL NOT NULL,
    y REAL NOT NULL,
    PRIMARY KEY(ring_id, point_index),
    FOREIGN KEY(ring_id) REFERENCES keyframe_polygon_rings(id)
);
CREATE TABLE mask_geometry_provenance(
    keyframe_id INTEGER PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_detection_id INTEGER,
    source_face_observation_id INTEGER,
    algorithm TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    FOREIGN KEY(keyframe_id) REFERENCES mask_keyframes(id),
    FOREIGN KEY(source_detection_id) REFERENCES detections(id),
    FOREIGN KEY(source_face_observation_id) REFERENCES face_observations(id)
);
CREATE TABLE cuts(
    frame INTEGER PRIMARY KEY
);
CREATE TABLE cut_detection_metadata(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    schema_version INTEGER NOT NULL,
    method TEXT NOT NULL CHECK(length(method) > 0),
    elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0),
    cut_count INTEGER NOT NULL CHECK(cut_count >= 0),
    frame_semantics TEXT NOT NULL
        CHECK(frame_semantics = 'first_frame_of_new_scene')
);
CREATE TABLE raw_tracked_masks(
    frame INTEGER NOT NULL,
    raw_track_id TEXT NOT NULL,
    raw_detection_index INTEGER NOT NULL,
    source_detection_id INTEGER,
    final_track_id TEXT,
    removed_by_short_track INTEGER NOT NULL DEFAULT 0,
    raw_track_length INTEGER NOT NULL DEFAULT 0,
    raw_label TEXT,
    final_label TEXT,
    polygons TEXT,
    score REAL,
    detector_score REAL,
    class_score REAL,
    category_id INTEGER,
    category_index INTEGER,
    bbox_xyxy_json TEXT,
    bbox_json TEXT,
    scene_id INTEGER,
    PRIMARY KEY(frame, raw_track_id, raw_detection_index)
);
CREATE TABLE tracking_assignments(
    source_detection_id INTEGER PRIMARY KEY,
    frame INTEGER NOT NULL CHECK(frame >= 0),
    raw_track_id TEXT NOT NULL,
    raw_detection_index INTEGER NOT NULL CHECK(raw_detection_index >= 0),
    final_track_id TEXT,
    removed_by_short_track INTEGER NOT NULL CHECK(
        removed_by_short_track IN (0, 1)
    ),
    raw_track_length INTEGER NOT NULL CHECK(raw_track_length >= 0),
    raw_label TEXT,
    final_label TEXT,
    selected_score REAL,
    scene_id INTEGER NOT NULL CHECK(scene_id >= 0),
    UNIQUE(frame, raw_track_id, raw_detection_index),
    FOREIGN KEY(source_detection_id) REFERENCES detections(id)
);
CREATE TABLE face_tracks(
    raw_track_id TEXT PRIMARY KEY,
    final_track_id TEXT,
    scene_id INTEGER NOT NULL CHECK(scene_id >= 0),
    start_frame INTEGER NOT NULL CHECK(start_frame >= 0),
    end_frame INTEGER NOT NULL CHECK(end_frame >= start_frame),
    observed_frames INTEGER NOT NULL CHECK(observed_frames >= 1),
    maximum_score REAL NOT NULL,
    mean_score REAL NOT NULL,
    removed_by_short_track INTEGER NOT NULL CHECK(
        removed_by_short_track IN (0, 1)
    ),
    termination_reason TEXT NOT NULL,
    UNIQUE(final_track_id)
);
CREATE TABLE face_tracking_assignments(
    observation_id INTEGER PRIMARY KEY,
    anchor_detection_id INTEGER NOT NULL,
    frame INTEGER NOT NULL CHECK(frame >= 0),
    raw_track_id TEXT NOT NULL,
    final_track_id TEXT,
    removed_by_short_track INTEGER NOT NULL CHECK(
        removed_by_short_track IN (0, 1)
    ),
    association_stage TEXT NOT NULL,
    association_score REAL,
    head_score REAL NOT NULL,
    face_score REAL NOT NULL,
    head_x1 REAL NOT NULL,
    head_y1 REAL NOT NULL,
    head_x2 REAL NOT NULL,
    head_y2 REAL NOT NULL,
    scene_id INTEGER NOT NULL CHECK(scene_id >= 0),
    FOREIGN KEY(observation_id) REFERENCES face_observations(id),
    FOREIGN KEY(anchor_detection_id) REFERENCES detections(id),
    FOREIGN KEY(raw_track_id) REFERENCES face_tracks(raw_track_id)
);
CREATE TABLE face_track_interpolations(
    frame INTEGER NOT NULL CHECK(frame >= 0),
    final_track_id TEXT NOT NULL,
    scene_id INTEGER NOT NULL CHECK(scene_id >= 0),
    previous_observation_id INTEGER NOT NULL,
    next_observation_id INTEGER NOT NULL,
    head_x1 REAL NOT NULL,
    head_y1 REAL NOT NULL,
    head_x2 REAL NOT NULL,
    head_y2 REAL NOT NULL,
    interpolation_method TEXT NOT NULL
        CHECK(interpolation_method = 'linear-two-sided'),
    PRIMARY KEY(frame, final_track_id),
    FOREIGN KEY(previous_observation_id) REFERENCES face_observations(id),
    FOREIGN KEY(next_observation_id) REFERENCES face_observations(id)
);
CREATE TABLE raw_tracks(
    raw_track_id TEXT PRIMARY KEY,
    final_track_id TEXT,
    removed_by_short_track INTEGER NOT NULL DEFAULT 0,
    raw_track_length INTEGER NOT NULL DEFAULT 0,
    raw_label TEXT,
    final_label TEXT,
    scene_id INTEGER
);
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
CREATE TABLE mask_provenance(
    frame INTEGER NOT NULL,
    track_id TEXT NOT NULL,
    mask_kind TEXT NOT NULL,
    source_observation_id INTEGER,
    source_observation_id_end INTEGER,
    is_interpolated INTEGER NOT NULL DEFAULT 0 CHECK(
        is_interpolated IN (0, 1)
    ),
    derivation TEXT NOT NULL,
    confidence REAL NOT NULL,
    algorithm_version TEXT NOT NULL,
    PRIMARY KEY(frame, track_id)
);
CREATE INDEX idx_tracked_masks_track_frame
    ON tracked_masks(track_id, frame);
CREATE INDEX idx_masks_track_frame
    ON masks(track_id, frame);
CREATE INDEX idx_mask_segments_track_frame
    ON mask_track_segments(track_id, start_frame, end_frame);
CREATE INDEX idx_mask_keyframes_segment_frame
    ON mask_keyframes(segment_id, frame);
CREATE INDEX idx_keyframe_components_keyframe
    ON keyframe_components(keyframe_id, slot_index);
CREATE INDEX idx_keyframe_polygon_points_ring
    ON keyframe_polygon_points(ring_id, point_index);
CREATE INDEX idx_raw_tracked_masks_track_frame
    ON raw_tracked_masks(raw_track_id, frame);
CREATE INDEX idx_raw_tracked_masks_final_track_frame
    ON raw_tracked_masks(final_track_id, frame);
CREATE INDEX idx_tracking_assignments_final_track_frame
    ON tracking_assignments(final_track_id, frame);
CREATE INDEX idx_tracking_assignments_raw_track_frame
    ON tracking_assignments(raw_track_id, frame);
CREATE INDEX idx_face_tracking_assignments_final_track_frame
    ON face_tracking_assignments(final_track_id, frame);
CREATE INDEX idx_face_tracking_assignments_raw_track_frame
    ON face_tracking_assignments(raw_track_id, frame);
CREATE INDEX idx_face_track_interpolations_frame
    ON face_track_interpolations(frame, final_track_id);
CREATE INDEX idx_mask_postprocess_provenance_label_frame
    ON mask_postprocess_provenance(label, frame);
CREATE INDEX idx_mask_provenance_source
    ON mask_provenance(source_observation_id);
CREATE VIEW editable_keyframe_components AS
SELECT
    t.track_id,
    t.domain,
    t.label,
    s.id AS segment_id,
    s.scene_id,
    s.start_frame,
    s.end_frame,
    s.interpolation_method,
    k.id AS keyframe_id,
    k.frame,
    k.keyframe_index,
    k.selection_reason,
    k.confidence,
    c.id AS component_id,
    c.slot_index,
    c.geometry_type,
    e.cx AS ellipse_cx,
    e.cy AS ellipse_cy,
    e.radius_x AS ellipse_radius_x,
    e.radius_y AS ellipse_radius_y,
    e.theta_radians AS ellipse_theta_radians,
    r.cx AS rectangle_cx,
    r.cy AS rectangle_cy,
    r.half_width AS rectangle_half_width,
    r.half_height AS rectangle_half_height,
    r.theta_radians AS rectangle_theta_radians
FROM mask_keyframes k
JOIN mask_track_segments s ON s.id=k.segment_id
JOIN tracks t ON t.track_id=s.track_id
JOIN keyframe_components c ON c.keyframe_id=k.id
LEFT JOIN keyframe_ellipses e ON e.component_id=c.id
LEFT JOIN keyframe_rectangles r ON r.component_id=c.id;
CREATE VIEW editable_polygon_vertices AS
SELECT
    c.keyframe_id,
    c.id AS component_id,
    c.slot_index,
    rings.ring_index,
    rings.ring_role,
    points.point_index,
    points.x,
    points.y
FROM keyframe_components c
JOIN keyframe_polygon_rings rings ON rings.component_id=c.id
JOIN keyframe_polygon_points points ON points.ring_id=rings.id
WHERE c.geometry_type='polygon';
"""

# Unified inference v2 files are still accepted as reusable inputs.  These
# additive objects make their public result surface match schema-v3 outputs.
INFERENCE_COMPATIBILITY_SQL = """
CREATE TABLE IF NOT EXISTS videos(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    reported_frame_count INTEGER,
    fps REAL,
    width INTEGER,
    height INTEGER
);
CREATE TABLE IF NOT EXISTS video_streams(
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    stream_index INTEGER NOT NULL,
    codec_name TEXT,
    width INTEGER,
    height INTEGER,
    fps_num INTEGER,
    fps_den INTEGER,
    time_base_num INTEGER,
    time_base_den INTEGER,
    frame_count INTEGER,
    rotation INTEGER,
    pixel_format TEXT,
    color_range TEXT,
    color_primaries TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(video_id, stream_index)
);
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_metadata(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_executions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    runtime_model_id TEXT NOT NULL,
    task TEXT NOT NULL,
    backend TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_metadata(
    model_execution_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL,
    PRIMARY KEY(model_execution_id, key)
);
CREATE TABLE IF NOT EXISTS classification_probabilities(
    detection_id INTEGER NOT NULL,
    class_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(detection_id, class_index)
);
CREATE TABLE IF NOT EXISTS face_observations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_detection_id INTEGER NOT NULL UNIQUE,
    head_detection_id INTEGER UNIQUE,
    face_detection_id INTEGER UNIQUE,
    face_score REAL NOT NULL,
    face_present INTEGER NOT NULL,
    geometry_type TEXT,
    ellipse_cx REAL,
    ellipse_cy REAL,
    ellipse_major_radius REAL,
    ellipse_minor_radius REAL,
    ellipse_theta_radians REAL
);
CREATE TABLE IF NOT EXISTS face_keypoints(
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    state INTEGER NOT NULL,
    state_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid INTEGER NOT NULL,
    PRIMARY KEY(observation_id, point_index)
);
CREATE TABLE IF NOT EXISTS face_masks(
    observation_id INTEGER PRIMARY KEY,
    encoding TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    box_x1 REAL NOT NULL,
    box_y1 REAL NOT NULL,
    box_x2 REAL NOT NULL,
    box_y2 REAL NOT NULL,
    data BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS face_keypoint_class_probabilities(
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    class_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(observation_id, point_index, class_index)
);
CREATE TABLE IF NOT EXISTS face_keypoint_state_probabilities(
    observation_id INTEGER NOT NULL,
    point_index INTEGER NOT NULL,
    state_index INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY(observation_id, point_index, state_index)
);
"""

MASK_COLUMNS = (
    "frame",
    "track_id",
    "polygons",
    "shape_type",
    "dilate_px",
    "feather_px",
    "mosaic_block",
    "mosaic_alias",
    "label",
)

FINAL_COPY_COLUMNS = {
    "processing_runs": (
        "id",
        "kind",
        "name",
        "status",
        "created_at_utc",
        "completed_at_utc",
        "resolved_config_json",
        "config_hash",
        "software_version",
        "git_commit",
    ),
    "processing_stage_runs": (
        "id",
        "processing_run_id",
        "stage_index",
        "stage_id",
        "implementation",
        "device",
        "options_json",
        "elapsed_seconds",
        "status",
    ),
    "masks": MASK_COLUMNS,
    "tracks": ("track_id", "label"),
    "cuts": ("frame",),
    "cut_detection_metadata": (
        "id",
        "schema_version",
        "method",
        "elapsed_seconds",
        "cut_count",
        "frame_semantics",
    ),
    "raw_tracked_masks": (
        "frame",
        "raw_track_id",
        "raw_detection_index",
        "source_detection_id",
        "final_track_id",
        "removed_by_short_track",
        "raw_track_length",
        "raw_label",
        "final_label",
        "polygons",
        "score",
        "detector_score",
        "class_score",
        "category_id",
        "category_index",
        "bbox_xyxy_json",
        "bbox_json",
        "scene_id",
    ),
    "raw_tracks": (
        "raw_track_id",
        "final_track_id",
        "removed_by_short_track",
        "raw_track_length",
        "raw_label",
        "final_label",
        "scene_id",
    ),
    "face_tracks": (
        "raw_track_id",
        "final_track_id",
        "scene_id",
        "start_frame",
        "end_frame",
        "observed_frames",
        "maximum_score",
        "mean_score",
        "removed_by_short_track",
        "termination_reason",
    ),
    "face_tracking_assignments": (
        "observation_id",
        "anchor_detection_id",
        "frame",
        "raw_track_id",
        "final_track_id",
        "removed_by_short_track",
        "association_stage",
        "association_score",
        "head_score",
        "face_score",
        "head_x1",
        "head_y1",
        "head_x2",
        "head_y2",
        "scene_id",
    ),
    "face_track_interpolations": (
        "frame",
        "final_track_id",
        "scene_id",
        "previous_observation_id",
        "next_observation_id",
        "head_x1",
        "head_y1",
        "head_x2",
        "head_y2",
        "interpolation_method",
    ),
    "class_postprocess_policies": (
        "label",
        "policy_source",
        "shape_mode",
        "keyframe_interval",
        "max_gap",
    ),
    "mask_postprocess_provenance": (
        "frame",
        "track_id",
        "label",
        "policy_source",
        "shape_mode",
        "keyframe_interval",
        "max_gap",
        "is_gap_filled",
    ),
    "mask_provenance": (
        "frame",
        "track_id",
        "mask_kind",
        "source_observation_id",
        "source_observation_id_end",
        "is_interpolated",
        "derivation",
        "confidence",
        "algorithm_version",
    ),
    "mask_track_segments": (
        "id",
        "track_id",
        "scene_id",
        "start_frame",
        "end_frame",
        "shape_type",
        "interpolation_method",
        "component_count",
        "source_run_key",
        "segment_reason",
    ),
    "mask_keyframes": (
        "id",
        "segment_id",
        "frame",
        "keyframe_index",
        "selection_reason",
        "source_detection_id",
        "confidence",
        "quality_score",
    ),
    "keyframe_components": (
        "id",
        "keyframe_id",
        "slot_index",
        "geometry_type",
    ),
    "keyframe_ellipses": (
        "component_id",
        "cx",
        "cy",
        "radius_x",
        "radius_y",
        "theta_radians",
    ),
    "keyframe_rectangles": (
        "component_id",
        "cx",
        "cy",
        "half_width",
        "half_height",
        "theta_radians",
    ),
    "keyframe_polygon_rings": (
        "id",
        "component_id",
        "ring_index",
        "ring_role",
    ),
    "keyframe_polygon_points": (
        "ring_id",
        "point_index",
        "x",
        "y",
    ),
    "mask_geometry_provenance": (
        "keyframe_id",
        "source_kind",
        "source_detection_id",
        "source_face_observation_id",
        "algorithm",
        "parameters_json",
    ),
}


def _tables(connection: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT name
            FROM {schema}.sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
        )
    }


def _views(connection: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT name FROM {schema}.sqlite_master WHERE type='view'"
        )
    }


def _columns(
    connection: sqlite3.Connection,
    table: str,
    schema: str = "main",
) -> set[str]:
    quoted = table.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA {schema}.table_info("{quoted}")')
    }


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def _ensure_inference_compatibility(connection: sqlite3.Connection) -> None:
    connection.executescript(INFERENCE_COMPATIBILITY_SQL)
    if "group_id" not in _columns(connection, "detections"):
        connection.execute("ALTER TABLE detections ADD COLUMN group_id INTEGER")
    for video_id, frame_count, fps, width, height in connection.execute(
        """
        SELECT id, reported_frame_count, fps, width, height
        FROM videos
        """
    ):
        rate = (
            Fraction(float(fps)).limit_denominator(1_000_000)
            if fps is not None and float(fps) > 0
            else None
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO video_streams(
                id, video_id, stream_index, width, height,
                fps_num, fps_den, frame_count, metadata_json
            ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, '{}')
            """,
            (
                int(video_id),
                int(video_id),
                width,
                height,
                None if rate is None else rate.numerator,
                None if rate is None else rate.denominator,
                frame_count,
            ),
        )


def _reset_result_schema(connection: sqlite3.Connection) -> None:
    existing_views = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )
    }
    for view in RESULT_OWNED_VIEWS:
        if view in existing_views:
            quoted = view.replace('"', '""')
            connection.execute(f'DROP VIEW "{quoted}"')
    existing = _tables(connection)
    for table in RESULT_OWNED_TABLES:
        if table in existing:
            quoted = table.replace('"', '""')
            connection.execute(f'DROP TABLE "{quoted}"')
    connection.executescript(RESULT_SCHEMA_SQL)


def _copy_projected_table(
    connection: sqlite3.Connection,
    *,
    source_schema: str,
    table: str,
    columns: tuple[str, ...],
    target_table: str | None = None,
) -> int:
    if table not in _tables(connection, source_schema):
        return 0
    missing = set(columns) - _columns(connection, table, source_schema)
    if missing:
        raise ValueError(f"{source_schema}.{table} columns missing: {sorted(missing)}")
    target = table if target_table is None else target_table
    names = ", ".join(f'"{name}"' for name in columns)
    connection.execute(
        f'INSERT INTO main."{target}"({names}) '
        f'SELECT {names} FROM {source_schema}."{table}"'
    )
    return int(
        connection.execute(f'SELECT COUNT(*) FROM main."{target}"').fetchone()[0]
    )


def _compact_to_keyframe_primary(connection: sqlite3.Connection) -> dict[str, int]:
    """Replace duplicated mask snapshots with stable source references.

    The raw detector geometry remains in ``segmentations``.  Tracking keeps
    only its decision/identity mapping, while ``mask_keyframes`` is the sole
    authoritative final geometry.
    """

    raw_rows = _row_count(connection, "raw_tracked_masks")
    missing_source_ids = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM raw_tracked_masks
            WHERE source_detection_id IS NULL
            """
        ).fetchone()[0]
    )
    if missing_source_ids:
        raise ValueError(
            "tracked rows are missing source_detection_id; rerun normalization, "
            "NMS, and tracking with the keyframe-primary V3 pipeline"
        )
    connection.execute(
        """
        INSERT INTO tracking_assignments(
            source_detection_id, frame, raw_track_id, raw_detection_index,
            final_track_id, removed_by_short_track, raw_track_length,
            raw_label, final_label, selected_score, scene_id
        )
        SELECT source_detection_id, frame, raw_track_id, raw_detection_index,
               final_track_id, removed_by_short_track, raw_track_length,
               raw_label, final_label,
               COALESCE(score, class_score, detector_score), scene_id
        FROM raw_tracked_masks
        ORDER BY frame, raw_track_id, raw_detection_index
        """
    )
    invalid_links = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM tracking_assignments a
            LEFT JOIN detections d ON d.id=a.source_detection_id
            LEFT JOIN frames f ON f.id=d.frame_id
            WHERE d.id IS NULL OR f.frame_index<>a.frame
            """
        ).fetchone()[0]
    )
    if invalid_links:
        raise ValueError(
            f"tracking assignments contain {invalid_links} invalid detection links"
        )
    connection.execute(
        """
        INSERT INTO annotation_state(
            id, revision, updated_at_utc, authoritative_geometry,
            dense_cache_policy
        ) VALUES (1, 0, ?, 'mask_keyframes', 'not_materialized')
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )
    dense_counts = {
        "materialized_tracked_masks_removed": _row_count(connection, "tracked_masks"),
        "materialized_final_masks_removed": _row_count(connection, "masks"),
        "duplicated_tracking_polygons_removed": raw_rows,
    }
    connection.executescript(
        """
        DROP TABLE tracked_masks;
        DROP TABLE tracked_tracks;
        DROP TABLE masks;
        DROP TABLE raw_tracked_masks;
        """
    )
    return {
        "tracking_assignments": _row_count(connection, "tracking_assignments"),
        **dense_counts,
    }


def _model_rows(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    if "model_executions" not in _tables(connection):
        return []
    return [
        (str(role), str(model_id), str(runtime_model_id))
        for role, model_id, runtime_model_id in connection.execute(
            "SELECT role, model_id, runtime_model_id FROM model_executions"
        )
    ]


def _row_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _capability_row_count(
    connection: sqlite3.Connection,
    name: str,
    source_table: str,
) -> int:
    if name == "face_detection":
        if "model_executions" not in _tables(connection):
            return 0
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM detections AS d
                JOIN model_executions AS m
                  ON m.id=d.model_execution_id
                WHERE m.role='face_detection'
                """
            ).fetchone()[0]
        )
    return _row_count(connection, source_table)


def _write_result_metadata(
    connection: sqlite3.Connection,
    *,
    tracked_available: bool,
    final_available: bool,
    cut_available: bool,
    classwise_available: bool,
    face_privacy_available: bool,
    editable_keyframes_available: bool,
) -> None:
    connection.executemany(
        "INSERT INTO result_schema_info(key, value) VALUES (?, ?)",
        (
            ("schema_name", RESULT_SCHEMA_NAME),
            ("schema_version", RESULT_SCHEMA_VERSION),
            ("contract_revision", RESULT_CONTRACT_REVISION),
            ("compatibility_profile", RESULT_COMPATIBILITY_PROFILE),
            ("missing_components", "capability_rows"),
            ("frame_index_base", "0"),
            ("coordinate_space", "source_display_pixels"),
            ("angle_unit", "radians"),
            ("cut_semantics", "first_frame_of_new_scene"),
            ("raw_data", "unified-inference-schema"),
            ("tracked_data", "tracking_assignments"),
            ("face_tracked_data", "face_tracking_assignments"),
            ("final_data", "mask_keyframes"),
            ("editable_data", "mask_keyframes"),
            ("materialized_dense_masks", "none"),
        ),
    )
    models = _model_rows(connection)
    segmentation_models = [
        model_id
        for role, model_id, _runtime_model_id in models
        if role == "instance_segmentation"
    ]
    face_models = [
        model_id
        for role, model_id, _runtime_model_id in models
        if role == "face_detection"
    ]
    rich_face_available = any(
        "face_dino_v2" in value
        for role, model_id, runtime_model_id in models
        if role == "face_detection"
        for value in (model_id, runtime_model_id)
    )
    cut_method = None
    if _row_count(connection, "cut_detection_metadata"):
        cut_method = str(
            connection.execute(
                "SELECT method FROM cut_detection_metadata WHERE id=1"
            ).fetchone()[0]
        )
    capabilities = (
        (
            "raw_inference",
            True,
            _row_count(connection, "frames"),
            "frames",
            {"schema": INFERENCE_SCHEMA_NAME},
        ),
        (
            "instance_segmentation",
            bool(segmentation_models) or bool(_row_count(connection, "segmentations")),
            _row_count(connection, "segmentations"),
            "segmentations",
            {"models": segmentation_models},
        ),
        (
            "face_detection",
            bool(face_models),
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM detections AS d
                    JOIN model_executions AS m
                      ON m.id=d.model_execution_id
                    WHERE m.role='face_detection'
                    """
                ).fetchone()[0]
            )
            if "model_executions" in _tables(connection)
            else 0,
            "detections",
            {"models": face_models},
        ),
        (
            "rich_face_geometry",
            rich_face_available,
            _row_count(connection, "face_observations"),
            "face_observations",
            {"models": face_models},
        ),
        (
            "tracking_assignments",
            tracked_available,
            _row_count(connection, "tracking_assignments"),
            "tracking_assignments",
            {"geometry_source": "segmentations"},
        ),
        (
            "face_tracking",
            bool(_row_count(connection, "face_tracking_assignments")),
            _row_count(connection, "face_tracking_assignments"),
            "face_tracking_assignments",
            {
                "geometry_source": "head_boxes",
                "smoothing": False,
                "assignment": "two_stage_hungarian",
                "interpolated_boxes": _row_count(
                    connection,
                    "face_track_interpolations",
                ),
            },
        ),
        (
            "final_annotations",
            final_available,
            _row_count(connection, "mask_keyframes"),
            "mask_keyframes",
            {
                "authoritative": True,
                "materialized_dense_masks": False,
                "geometry_contract": "typed-native-v1",
            },
        ),
        (
            "cut_detection",
            cut_available,
            _row_count(connection, "cuts"),
            "cuts",
            {"method": cut_method},
        ),
        (
            "classwise_postprocess",
            classwise_available,
            _row_count(connection, "mask_postprocess_provenance"),
            "mask_postprocess_provenance",
            {},
        ),
        (
            "face_privacy_masks",
            face_privacy_available,
            _row_count(connection, "mask_provenance"),
            "mask_provenance",
            {},
        ),
        (
            "native_polygon_keyframes",
            editable_keyframes_available
            and bool(_row_count(connection, "keyframe_polygon_points")),
            _row_count(connection, "keyframe_polygon_points"),
            "keyframe_polygon_points",
            {"coordinate_space": "source_display_pixels"},
        ),
        (
            "native_ellipse_keyframes",
            editable_keyframes_available
            and bool(_row_count(connection, "keyframe_ellipses")),
            _row_count(connection, "keyframe_ellipses"),
            "keyframe_ellipses",
            {"angle_unit": "radians"},
        ),
        (
            "native_rectangle_keyframes",
            editable_keyframes_available
            and bool(_row_count(connection, "keyframe_rectangles")),
            _row_count(connection, "keyframe_rectangles"),
            "keyframe_rectangles",
            {"angle_unit": "radians"},
        ),
    )
    connection.executemany(
        """
        INSERT INTO result_capabilities(
            name, available, row_count, source_table, details_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            (
                name,
                int(available),
                row_count,
                source_table,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
            )
            for name, available, row_count, source_table, details in capabilities
        ),
    )
    for name, available, row_count, source_table, details in capabilities:
        if available:
            status = "complete" if row_count else "empty"
        elif name == "rich_face_geometry" and face_models:
            status = "unsupported"
        else:
            status = "not_requested"
        connection.execute(
            """
            INSERT INTO result_components(
                name, status, row_count, source_table, details_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                status,
                row_count,
                source_table,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
            ),
        )


def validate_integrated_result(path: Path) -> dict[str, Any]:
    """Validate the stable raw/tracked/final result contract."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        tables = _tables(connection)
        missing = RESULT_REQUIRED_TABLES - tables
        if missing:
            raise ValueError(
                f"{source}: integrated result tables missing: {sorted(missing)}"
            )
        missing_views = set(RESULT_OWNED_VIEWS) - _views(connection)
        if missing_views:
            raise ValueError(
                f"{source}: integrated result views missing: "
                f"{sorted(missing_views)}"
            )
        inference_info = dict(connection.execute("SELECT key, value FROM schema_info"))
        if inference_info.get("schema_name") != INFERENCE_SCHEMA_NAME:
            raise ValueError(f"{source}: unexpected inference schema")
        result_info = dict(
            connection.execute("SELECT key, value FROM result_schema_info")
        )
        if (
            result_info.get("schema_name") != RESULT_SCHEMA_NAME
            or result_info.get("schema_version") != RESULT_SCHEMA_VERSION
            or result_info.get("contract_revision") != RESULT_CONTRACT_REVISION
            or result_info.get("compatibility_profile") != RESULT_COMPATIBILITY_PROFILE
        ):
            raise ValueError(f"{source}: unsupported integrated result contract")
        capability_rows = list(
            connection.execute(
                """
                SELECT name, available, row_count, source_table, details_json
                FROM result_capabilities ORDER BY name
                """
            )
        )
        expected_capabilities = {
            "raw_inference",
            "instance_segmentation",
            "face_detection",
            "rich_face_geometry",
            "tracking_assignments",
            "face_tracking",
            "final_annotations",
            "cut_detection",
            "classwise_postprocess",
            "face_privacy_masks",
            "native_polygon_keyframes",
            "native_ellipse_keyframes",
            "native_rectangle_keyframes",
        }
        actual_capabilities = {str(row[0]) for row in capability_rows}
        if actual_capabilities != expected_capabilities:
            raise ValueError(
                f"{source}: result capabilities mismatch: "
                f"{sorted(actual_capabilities)}"
            )
        for name, _available, row_count, source_table, details_json in capability_rows:
            actual_count = _capability_row_count(
                connection,
                str(name),
                str(source_table),
            )
            if int(row_count) != actual_count:
                raise ValueError(f"{source}: capability {name!r} row_count mismatch")
            try:
                details = json.loads(str(details_json))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: capability {name!r} details_json is invalid"
                ) from exc
            if not isinstance(details, dict):
                raise ValueError(
                    f"{source}: capability {name!r} details_json must be an object"
                )
        component_rows = list(
            connection.execute(
                """
                SELECT name, status, row_count, source_table, details_json
                FROM result_components ORDER BY name
                """
            )
        )
        if {str(row[0]) for row in component_rows} != expected_capabilities:
            raise ValueError(f"{source}: result components mismatch")
        for name, status, row_count, source_table, details_json in component_rows:
            if str(status) not in {
                "complete",
                "empty",
                "not_requested",
                "unsupported",
                "failed",
            }:
                raise ValueError(f"{source}: invalid component status for {name!r}")
            actual_count = _capability_row_count(
                connection,
                str(name),
                str(source_table),
            )
            if int(row_count) != actual_count:
                raise ValueError(f"{source}: component {name!r} row_count mismatch")
            if not isinstance(json.loads(str(details_json)), dict):
                raise ValueError(f"{source}: component {name!r} details must be object")
        invalid_keyframe = connection.execute(
            """
            SELECT k.id
            FROM mask_keyframes k
            JOIN mask_track_segments s ON s.id=k.segment_id
            WHERE k.frame < s.start_frame OR k.frame > s.end_frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_keyframe is not None:
            raise ValueError(
                f"{source}: keyframe {invalid_keyframe[0]} is outside its segment"
            )
        invalid_component = connection.execute(
            """
            SELECT c.id
            FROM keyframe_components c
            JOIN mask_keyframes k ON k.id=c.keyframe_id
            JOIN mask_track_segments s ON s.id=k.segment_id
            LEFT JOIN keyframe_ellipses e ON e.component_id=c.id
            LEFT JOIN keyframe_rectangles r ON r.component_id=c.id
            LEFT JOIN keyframe_polygon_rings pr ON pr.component_id=c.id
            LEFT JOIN keyframe_polygon_points pp ON pp.ring_id=pr.id
            GROUP BY c.id
            HAVING c.slot_index >= s.component_count
               OR (c.geometry_type='ellipse' AND COUNT(DISTINCT e.component_id) <> 1)
               OR (c.geometry_type='rectangle' AND COUNT(DISTINCT r.component_id) <> 1)
               OR (c.geometry_type='polygon' AND COUNT(pp.point_index) < 3)
            LIMIT 1
            """
        ).fetchone()
        if invalid_component is not None:
            raise ValueError(
                f"{source}: invalid typed keyframe component " f"{invalid_component[0]}"
            )
        dense_tables = {
            "masks",
            "tracked_masks",
            "raw_tracked_masks",
            "tracked_tracks",
        } & tables
        if dense_tables:
            raise ValueError(
                f"{source}: V3 contains duplicated dense tables: "
                f"{sorted(dense_tables)}"
            )
        annotation_state = connection.execute(
            """
            SELECT revision, authoritative_geometry, dense_cache_policy
            FROM annotation_state WHERE id=1
            """
        ).fetchone()
        if annotation_state is None or tuple(annotation_state[1:]) != (
            "mask_keyframes",
            "not_materialized",
        ):
            raise ValueError(f"{source}: invalid annotation_state")
        invalid_tracking_link = connection.execute(
            """
            SELECT a.source_detection_id
            FROM tracking_assignments a
            LEFT JOIN detections d ON d.id=a.source_detection_id
            LEFT JOIN frames f ON f.id=d.frame_id
            WHERE d.id IS NULL OR f.frame_index<>a.frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_tracking_link is not None:
            raise ValueError(
                f"{source}: invalid tracking source detection "
                f"{invalid_tracking_link[0]}"
            )
        invalid_face_tracking_link = connection.execute(
            """
            SELECT a.observation_id
            FROM face_tracking_assignments a
            LEFT JOIN face_observations fo ON fo.id=a.observation_id
            LEFT JOIN detections d ON d.id=a.anchor_detection_id
            LEFT JOIN frames f ON f.id=d.frame_id
            WHERE fo.id IS NULL OR d.id IS NULL OR f.frame_index<>a.frame
            LIMIT 1
            """
        ).fetchone()
        if invalid_face_tracking_link is not None:
            raise ValueError(
                f"{source}: invalid face tracking observation "
                f"{invalid_face_tracking_link[0]}"
            )
        invalid_face_interpolation = connection.execute(
            """
            SELECT i.frame, i.final_track_id
            FROM face_track_interpolations i
            LEFT JOIN face_observations previous
              ON previous.id=i.previous_observation_id
            LEFT JOIN face_observations following
              ON following.id=i.next_observation_id
            WHERE previous.id IS NULL OR following.id IS NULL
               OR i.head_x2<i.head_x1 OR i.head_y2<i.head_y1
            LIMIT 1
            """
        ).fetchone()
        if invalid_face_interpolation is not None:
            raise ValueError(
                f"{source}: invalid face interpolation "
                f"{tuple(invalid_face_interpolation)}"
            )
        counts = {
            "frames": _row_count(connection, "frames"),
            "detections": _row_count(connection, "detections"),
            "segmentations": _row_count(connection, "segmentations"),
            "face_observations": _row_count(connection, "face_observations"),
            "tracking_assignments": _row_count(connection, "tracking_assignments"),
            "face_tracking_assignments": _row_count(
                connection, "face_tracking_assignments"
            ),
            "face_track_interpolations": _row_count(
                connection, "face_track_interpolations"
            ),
            "final_annotations": _row_count(connection, "mask_keyframes"),
            "cuts": _row_count(connection, "cuts"),
            "mask_segments": _row_count(connection, "mask_track_segments"),
            "mask_keyframes": _row_count(connection, "mask_keyframes"),
            "ellipse_keyframes": _row_count(connection, "keyframe_ellipses"),
            "polygon_keyframe_points": _row_count(
                connection, "keyframe_polygon_points"
            ),
            "rectangle_keyframes": _row_count(connection, "keyframe_rectangles"),
        }
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"{source}: integrity check failed: {integrity}")
        return {
            "schema_name": RESULT_SCHEMA_NAME,
            "schema_version": int(RESULT_SCHEMA_VERSION),
            "contract_revision": int(RESULT_CONTRACT_REVISION),
            "compatibility_profile": RESULT_COMPATIBILITY_PROFILE,
            "annotation_revision": int(annotation_state[0]),
            "inference_schema_version": int(inference_info["schema_version"]),
            "capabilities": {
                str(name): {
                    "available": bool(available),
                    "row_count": int(row_count),
                    "source_table": str(source_table),
                    "details": json.loads(str(details_json)),
                }
                for (
                    name,
                    available,
                    row_count,
                    source_table,
                    details_json,
                ) in capability_rows
            },
            **counts,
            "size_bytes": source.stat().st_size,
        }


def record_processing_run(
    result_sqlite: Path,
    *,
    kind: str,
    name: str,
    resolved_config: object,
    stages: list[dict[str, object]],
    status: str = "complete",
    git_commit: str | None = None,
) -> dict[str, object]:
    """Idempotently embed a resolved pipeline and its stage executions."""

    target = Path(result_sqlite).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    config_json = json.dumps(
        resolved_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(target) as connection:
        if {
            "processing_runs",
            "processing_stage_runs",
        } - _tables(connection):
            raise ValueError(f"{target}: processing metadata tables are absent")
        connection.execute(
            """
            INSERT INTO processing_runs(
                kind, name, status, created_at_utc, completed_at_utc,
                resolved_config_json, config_hash, software_version, git_commit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, config_hash) DO UPDATE SET
                name=excluded.name,
                status=excluded.status,
                completed_at_utc=excluded.completed_at_utc,
                resolved_config_json=excluded.resolved_config_json,
                software_version=excluded.software_version,
                git_commit=COALESCE(excluded.git_commit, processing_runs.git_commit)
            """,
            (
                str(kind),
                str(name),
                str(status),
                now,
                now,
                config_json,
                config_hash,
                f"result-contract-{RESULT_CONTRACT_REVISION}",
                git_commit,
            ),
        )
        run_id = int(
            connection.execute(
                """
                SELECT id FROM processing_runs
                WHERE kind=? AND config_hash=?
                """,
                (str(kind), config_hash),
            ).fetchone()[0]
        )
        connection.execute(
            "DELETE FROM processing_stage_runs WHERE processing_run_id=?",
            (run_id,),
        )
        for stage_index, stage in enumerate(stages):
            options = stage.get("options", {})
            connection.execute(
                """
                INSERT INTO processing_stage_runs(
                    processing_run_id, stage_index, stage_id, implementation,
                    device, options_json, elapsed_seconds, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    stage_index,
                    str(stage.get("id", stage.get("name", stage_index))),
                    str(stage.get("implementation", "")),
                    (None if stage.get("device") is None else str(stage.get("device"))),
                    json.dumps(
                        options,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                    (
                        None
                        if stage.get("elapsed_seconds") is None
                        else float(stage["elapsed_seconds"])
                    ),
                    str(stage.get("status", "complete")),
                ),
            )
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(
                f"{target}: integrity check failed after metadata update"
            )
    return {
        "processing_run_id": run_id,
        "kind": str(kind),
        "config_hash": config_hash,
        "stages": len(stages),
    }


def build_integrated_result(
    inference_sqlite: Path,
    tracked_sqlite: Path | None,
    final_sqlite: Path | None,
    output_sqlite: Path,
    *,
    polygon_keyframes_sqlite: Path | None = None,
    ellipse_keyframes_json: Path | None = None,
    classwise_manifest: Path | None = None,
) -> dict[str, Any]:
    """Atomically publish a stable SQLite; unavailable components stay empty."""

    inference = Path(inference_sqlite).expanduser().resolve()
    tracked = (
        None if tracked_sqlite is None else Path(tracked_sqlite).expanduser().resolve()
    )
    final = None if final_sqlite is None else Path(final_sqlite).expanduser().resolve()
    polygon_keys = (
        None
        if polygon_keyframes_sqlite is None
        else Path(polygon_keyframes_sqlite).expanduser().resolve()
    )
    ellipse_keys = (
        None
        if ellipse_keyframes_json is None
        else Path(ellipse_keyframes_json).expanduser().resolve()
    )
    classwise = (
        None
        if classwise_manifest is None
        else Path(classwise_manifest).expanduser().resolve()
    )
    output = Path(output_sqlite).expanduser().resolve()
    if not inference.is_file():
        raise FileNotFoundError(inference)
    for source in (tracked, final):
        if source is not None and not source.is_file():
            raise FileNotFoundError(source)
    for source in (polygon_keys, ellipse_keys, classwise):
        if source is not None and not source.is_file():
            raise FileNotFoundError(source)
    sources = {source for source in (inference, tracked, final) if source is not None}
    if output in sources:
        raise ValueError("integrated output must differ from all source SQLite files")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    _remove_sqlite_files(temporary)
    copied_tables: list[str] = []
    source_final_tables: set[str] = set()
    source_tracked_tables: set[str] = set()
    editable_summary = {"segments": 0, "keyframes": 0, "components": 0}
    compact_summary: dict[str, int] = {}
    try:
        with sqlite3.connect(
            f"file:{inference}?mode=ro", uri=True
        ) as inference_connection:
            inference_info = dict(
                inference_connection.execute("SELECT key, value FROM schema_info")
            )
            if inference_info.get("schema_name") != INFERENCE_SCHEMA_NAME:
                raise ValueError(f"{inference}: unsupported inference schema")
            with sqlite3.connect(temporary) as output_connection:
                inference_connection.backup(output_connection)

        with sqlite3.connect(temporary) as connection:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            )
            if journal_mode.lower() != "delete":
                raise RuntimeError(
                    f"failed to set integrated SQLite journal mode: {journal_mode}"
                )
            _ensure_inference_compatibility(connection)
            _reset_result_schema(connection)

            if final is not None:
                connection.execute("ATTACH DATABASE ? AS final_db", (str(final),))
                source_final_tables = _tables(connection, "final_db")
                for table, columns in FINAL_COPY_COLUMNS.items():
                    if table == "mask_provenance" and table in source_final_tables:
                        source_columns = _columns(connection, table, "final_db")
                        if {
                            "source_observation_id_end",
                            "is_interpolated",
                        } - source_columns:
                            legacy_columns = (
                                "frame",
                                "track_id",
                                "mask_kind",
                                "source_observation_id",
                                "derivation",
                                "confidence",
                                "algorithm_version",
                            )
                            missing = set(legacy_columns) - source_columns
                            if missing:
                                raise ValueError(
                                    "final_db.mask_provenance columns missing: "
                                    f"{sorted(missing)}"
                                )
                            connection.execute(
                                """
                                INSERT INTO mask_provenance(
                                    frame, track_id, mask_kind,
                                    source_observation_id,
                                    source_observation_id_end,
                                    is_interpolated, derivation,
                                    confidence, algorithm_version
                                )
                                SELECT frame, track_id, mask_kind,
                                       source_observation_id,
                                       source_observation_id, 0,
                                       derivation, confidence,
                                       algorithm_version
                                FROM final_db.mask_provenance
                                """
                            )
                            copied_tables.append(table)
                            continue
                    if _copy_projected_table(
                        connection,
                        source_schema="final_db",
                        table=table,
                        columns=columns,
                    ):
                        copied_tables.append(table)
                if "masks" not in source_final_tables:
                    raise ValueError(f"{final}: final masks table is absent")
                if "tracks" not in source_final_tables:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO tracks(track_id, label)
                        SELECT DISTINCT track_id, label FROM masks
                        """
                    )
                connection.execute(
                    """
                    UPDATE tracks SET domain='face_privacy'
                    WHERE track_id LIKE 'face:%'
                    """
                )

            tracked_source_schema = None
            if tracked is not None:
                connection.execute(
                    "ATTACH DATABASE ? AS tracked_db",
                    (str(tracked),),
                )
                source_tracked_tables = _tables(connection, "tracked_db")
                tracked_source_schema = "tracked_db"
            elif final is not None and "tracked_masks" in source_final_tables:
                tracked_source_schema = "final_db"
                source_tracked_tables = source_final_tables

            if tracked_source_schema is not None:
                source_mask_table = (
                    "tracked_masks"
                    if "tracked_masks" in source_tracked_tables
                    else "masks"
                )
                _copy_projected_table(
                    connection,
                    source_schema=tracked_source_schema,
                    table=source_mask_table,
                    columns=MASK_COLUMNS,
                    target_table="tracked_masks",
                )
                source_track_table = (
                    "tracked_tracks"
                    if "tracked_tracks" in source_tracked_tables
                    else "tracks"
                )
                if source_track_table in source_tracked_tables:
                    source_columns = _columns(
                        connection,
                        source_track_table,
                        tracked_source_schema,
                    )
                    if {"track_id", "label"} <= source_columns:
                        connection.execute(
                            f"""
                            INSERT INTO tracked_tracks(track_id, label)
                            SELECT track_id, label
                            FROM {tracked_source_schema}."{source_track_table}"
                            """
                        )
                if not _row_count(connection, "tracked_tracks"):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO tracked_tracks(track_id, label)
                        SELECT DISTINCT track_id, label FROM tracked_masks
                        """
                    )

            editable_requested = (
                any(
                    source is not None
                    for source in (polygon_keys, ellipse_keys, classwise)
                )
                or "mask_keyframes" in source_final_tables
            )
            editable_summary = import_editable_geometry(
                connection,
                polygon_keyframes_sqlite=polygon_keys,
                ellipse_keyframes_json=ellipse_keys,
                classwise_manifest=classwise,
                face_source_schema=(
                    "final_db"
                    if final is not None
                    and "face_mask_geometries" in source_final_tables
                    else None
                ),
            )
            # A caller may provide only a legacy/materialized final SQLite,
            # without a separate native-keyframe artifact.  Preserve those
            # annotations by treating each stored frame as an explicit
            # polygon keyframe.  Tracks already imported from native polygon,
            # ellipse, rectangle, or face geometry are left untouched.
            if final is not None and _row_count(connection, "masks"):
                fallback_summary = import_polygon_keyframes(
                    connection,
                    final,
                    source_prefix="materialized-final-fallback",
                    only_unrepresented_tracks=True,
                )
                for key, value in fallback_summary.items():
                    editable_summary[key] += value
            if editable_summary["keyframes"]:
                editable_requested = True

            cut_method = None
            if _row_count(connection, "cut_detection_metadata"):
                cut_method = str(
                    connection.execute(
                        "SELECT method FROM cut_detection_metadata WHERE id=1"
                    ).fetchone()[0]
                )
            cut_available = cut_method is not None and cut_method.lower() not in {
                "disabled",
                "not_run",
                "unavailable",
            }
            compact_summary = _compact_to_keyframe_primary(connection)
            _write_result_metadata(
                connection,
                tracked_available=bool(compact_summary["tracking_assignments"]),
                final_available=editable_requested,
                cut_available=cut_available,
                classwise_available=(
                    "class_postprocess_policies" in source_final_tables
                    or "mask_postprocess_provenance" in source_final_tables
                ),
                face_privacy_available="mask_provenance" in source_final_tables,
                editable_keyframes_available=editable_requested,
            )
            connection.commit()
            if tracked is not None:
                connection.execute("DETACH DATABASE tracked_db")
            if final is not None:
                connection.execute("DETACH DATABASE final_db")
            connection.execute("VACUUM")
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise RuntimeError(
                    f"integrated SQLite integrity check failed: {integrity}"
                )
            connection.commit()
        os.replace(temporary, output)
    finally:
        _remove_sqlite_files(temporary)

    validated = validate_integrated_result(output)
    return {
        **validated,
        "input_inference_sqlite": str(inference),
        "input_tracked_sqlite": None if tracked is None else str(tracked),
        "input_final_sqlite": None if final is None else str(final),
        "input_polygon_keyframes_sqlite": (
            None if polygon_keys is None else str(polygon_keys)
        ),
        "input_ellipse_keyframes_json": (
            None if ellipse_keys is None else str(ellipse_keys)
        ),
        "input_classwise_manifest": None if classwise is None else str(classwise),
        "editable_geometry_import": editable_summary,
        "keyframe_primary_compaction": compact_summary,
        "copied_postprocess_tables": copied_tables,
    }


__all__ = [
    "RESULT_COMPATIBILITY_PROFILE",
    "RESULT_CONTRACT_REVISION",
    "RESULT_REQUIRED_TABLES",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "build_integrated_result",
    "record_processing_run",
    "validate_integrated_result",
]
