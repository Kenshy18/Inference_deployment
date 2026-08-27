"""Static SQLite schema and copy projections for integrated result artifacts.

This module is deliberately data-only.  Schema evolution belongs here; atomic file
publication, row copying, validation, and run recording stay in ``unified_sqlite``.
"""

from __future__ import annotations

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


__all__ = (
    "FINAL_COPY_COLUMNS",
    "INFERENCE_COMPATIBILITY_SQL",
    "INFERENCE_SCHEMA_NAME",
    "MASK_COLUMNS",
    "RESULT_COMPATIBILITY_PROFILE",
    "RESULT_CONTRACT_REVISION",
    "RESULT_OWNED_TABLES",
    "RESULT_OWNED_VIEWS",
    "RESULT_REQUIRED_TABLES",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_SQL",
    "RESULT_SCHEMA_VERSION",
)
