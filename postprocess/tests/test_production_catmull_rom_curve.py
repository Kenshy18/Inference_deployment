"""Production closed Catmull--Rom geometry and optimizer regression tests."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import production.curve.runtime.multistate_dp as multistate_dp

from production.curve.config import CurveProductionConfig
from production.curve.engine import (
    _compact_fit_summary,
    _repair_if_needed,
    run_curve_optimizer,
)
from production.curve.runtime.curve_fit import (
    _smooth_temporal_correction,
    catmull_rom_basis_matrix,
    contour_arc_targets,
    fit_whole_curve_controls,
)
from production.curve.runtime.fitter import (
    FitConfig,
    _stabilize_temporal_phase,
    fit_sequence,
)
from production.curve.runtime.keyframe_dp import (
    KeyframeDpConfig,
    _frame_metrics,
    catmull_rom_renderer,
    interpolate_controls,
    optimize_keyframes,
    render_control_sequence,
)
from production.curve.runtime.model import (
    bezier_segments,
    evaluate_cubic,
    normalize_control_points,
    sample_closed_curve,
    sample_curve_sequence,
)
from production.curve.runtime.multistate_dp import (
    CurvePointRefineConfig,
    _MultistateIntervalEvaluator,
    _NodePath,
    _first_node_losses,
    _guard_refined_controls,
    isotropic_curve_states,
    optimize_multistate_keyframes,
)
from production.curve.runtime.native_cpu import ExactDoubleRasterBatch, native_module
from production.curve.runtime.topology import (
    has_strict_self_intersection,
    strict_self_intersection_batch,
)
from production.curve.stage import _cleanup_preparation_sqlites
from production.polygon.runtime.spatial_support.optimizer import (
    has_self_intersection,
    temporal_residuals,
)
from production.polygon.runtime.kernel.stream import split_long_track_segments
from production.polygon.runtime.kernel.types import InstanceRun, TrackRow


def _ellipse(center: tuple[float, float], radii=(34.0, 20.0), count=96):
    theta = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
    return np.column_stack(
        (
            float(center[0]) + float(radii[0]) * np.cos(theta),
            float(center[1]) + float(radii[1]) * np.sin(theta),
        )
    )


def test_production_quality_rescue_guards_are_explicit_and_validated() -> None:
    config = CurveProductionConfig()
    assert config.maximum_gap == 24
    assert config.path_selection_mode == "fixed_cardinality"
    assert config.cardinality_maximum_factor == 2.0
    assert config.low_iou_quadratic_weight == 16.0
    assert config.shape_distance_weight == 0.4
    assert config.quality_rescue_maximum_extra_keys == 0
    assert not config.quality_rescue_density_budget
    assert config.quality_rescue_maximum_iou_regression == 0.005
    assert config.quality_rescue_maximum_area_ratio_regression == 0.01
    config.validate()
    for field in (
        "quality_rescue_maximum_iou_regression",
        "quality_rescue_maximum_area_ratio_regression",
    ):
        values = {field: -0.001}
        try:
            CurveProductionConfig(**values).validate()
        except ValueError:
            pass
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"negative {field} was accepted")


def test_supported_target_intervals_share_the_frozen_exact_contract() -> None:
    for target in range(1, 7):
        config = CurveProductionConfig(target_interval=target)
        config.validate()
        assert config.maximum_gap == 24
        assert config.recall_floor == 0.97
        assert config.samples_per_segment == 16
        assert config.state_scales == (1.0, 1.005, 1.025, 1.035)
        assert config.path_selection_mode == "fixed_cardinality"


def test_fit_summary_compaction_is_bounded_and_does_not_mutate_input() -> None:
    shifts = [0, -2, 0, 3] * 10_000
    source = {
        "fit": {
            "temporal_phase_shifts": shifts,
            "boundary_rms_after": 0.25,
        },
        "post_fit_repair": {"repaired_frames": 2},
    }
    compact = _compact_fit_summary(source)
    assert source["fit"]["temporal_phase_shifts"] is shifts
    assert "temporal_phase_shifts" not in compact["fit"]
    assert compact["fit"]["temporal_phase_shift_summary"] == {
        "frames": 40_000,
        "nonzero_frames": 20_000,
        "maximum_absolute_shift": 3,
    }
    assert len(json.dumps(compact)) < 512


def test_curve_preparation_cleanup_is_scoped_and_removes_sqlite_sidecars(
    tmp_path,
) -> None:
    root = tmp_path / "preparation"
    source = root / "classes" / "00_test" / "endpoint.sqlite"
    source.parent.mkdir(parents=True)
    for candidate in (source, tmp_path / "outside.sqlite"):
        candidate.write_bytes(b"sqlite")
    Path(f"{source}-wal").write_bytes(b"wal")
    Path(f"{source}-shm").write_bytes(b"shm")
    removed = _cleanup_preparation_sqlites(
        {"classes": {"test": {"endpoint_sqlite": str(source)}}},
        root,
    )
    assert removed == {"files": 3, "bytes": 12}
    assert not source.exists()
    assert not Path(f"{source}-wal").exists()
    assert not Path(f"{source}-shm").exists()
    assert (tmp_path / "outside.sqlite").read_bytes() == b"sqlite"

    try:
        _cleanup_preparation_sqlites(
            {
                "classes": {
                    "test": {"endpoint_sqlite": str(tmp_path / "outside.sqlite")}
                }
            },
            root,
        )
    except RuntimeError as exc:
        assert "escaped" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("curve cleanup accepted a path outside its stage root")


def test_long_track_chunks_are_bounded_and_emit_every_frame_once() -> None:
    polygon = _ellipse((80.0, 60.0), count=24)
    source = [
        TrackRow(
            frame=frame,
            track_id="1",
            polygons=[polygon],
            is_gapfill=False,
        )
        for frame in range(1_300)
    ]
    chunks, metadata, summary = split_long_track_segments(
        [source],
        max_run_frames=600,
        run_overlap_frames=30,
    )
    assert summary["chunk_output_segment_count"] == 3
    assert summary["max_processed_segment_frames"] <= 600
    emitted: list[int] = []
    for chunk in chunks:
        value = metadata[id(chunk)]
        process_start = int(value["process_start"])
        emitted.extend(
            range(
                process_start + int(value["emit_start"]),
                process_start + int(value["emit_end"]),
            )
        )
    assert emitted == list(range(1_300))


def test_engine_audits_and_exports_every_multi_component_observation(
    tmp_path,
    monkeypatch,
) -> None:
    """The conservative multi-component fallback must not escape hard audit."""

    tracked = tmp_path / "tracked.sqlite"
    with sqlite3.connect(tracked) as connection:
        connection.executescript(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT NOT NULL,
                label TEXT,
                PRIMARY KEY(frame, track_id)
            );
            CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT);
            INSERT INTO tracks(track_id,label) VALUES ('7','女性器');
            INSERT INTO masks(frame,track_id,polygons,label)
            VALUES (0,'7','[]','女性器');
            """
        )

    frames = np.asarray((0, 1, 2), dtype=np.int64)
    references: list[list[np.ndarray]] = []
    component_controls: list[np.ndarray] = []
    for component in range(2):
        controls = np.asarray(
            [
                _ellipse(
                    (
                        (40.0 if component == 0 else 80.0) + frame,
                        50.0 if component == 0 else 55.0,
                    ),
                    radii=(12.0, 8.0) if component == 0 else (9.0, 6.0),
                    count=14,
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        component_controls.append(controls)
    for index in range(len(frames)):
        references.append(
            [
                sample_closed_curve(component_controls[component][index], 16)
                for component in range(2)
            ]
        )
    run = InstanceRun(
        stream_id="7:run0:instance",
        track_id="7",
        run_id=0,
        frame_numbers=frames,
        gt_polygons=references,
        anchors=np.concatenate(component_controls, axis=1),
        contour_count=2,
        anchors_per_contour=14,
        scale=1.0,
    )
    controls_to_return = iter(component_controls)

    def fake_fit(_references, _point_count, _config):
        return next(controls_to_return), {
            "fit": {},
            "post_fit_repair": {
                "emergency_repair": False,
                "completion_envelope_count": 0,
                "maximum_completion_area_ratio": 1.0,
            },
        }

    monkeypatch.setattr(
        "production.curve.engine.iter_track_streams_from_sqlite",
        lambda *_args, **_kwargs: iter((run,)),
    )
    monkeypatch.setattr("production.curve.engine._fit_component", fake_fit)
    preparation = {
        "active_labels": ["女性器"],
        "classes": {"女性器": {"endpoint_sqlite": str(tracked), "input_rows": 3}},
        "passthrough_track_ids": [],
        "vertex_policy": {
            "tracks": {"7": {"vertices_per_component": 14}},
            "summary": {},
        },
    }
    summary = run_curve_optimizer(
        tracked,
        preparation,
        tmp_path / "curve",
        config=CurveProductionConfig(),
    )

    assert summary["prediction_rows"] == 3
    assert summary["keyframe_rows"] == 3
    assert summary["multi_component_key_every_frame_streams"] == 1
    assert summary["audit"]["emitted_component_frames"] == 3
    assert summary["audit"]["component_observations"] == 6
    assert summary["audit"]["recall_violations"] == 0
    assert summary["audit"]["topology_invalid_frames"] == 0
    with Path(summary["component_metrics_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {int(row["component"]) for row in rows} == {0, 1}
    assert {int(row["is_keyframe"]) for row in rows} == {1}
    assert min(float(row["recall"]) for row in rows) >= 0.97


def test_independent_local_refit_precedes_conservative_spatial_envelope() -> None:
    reference = _ellipse((120.0, 80.0), radii=(48.0, 23.0), count=192)
    theta = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
    tiny = np.column_stack((120.0 + np.cos(theta), 80.0 + np.sin(theta)))
    repaired, summary = _repair_if_needed(
        [reference],
        tiny[None],
        config=CurveProductionConfig(native_cpu_threads=1),
    )
    boundary = catmull_rom_renderer(16)(repaired[0])
    iou, recall, _precision, area_ratio = _frame_metrics(reference, boundary)
    local_refit = summary["independent_local_refit"]
    assert local_refit["attempted_frames"] == [0]
    assert local_refit["accepted_frames"] == [0]
    assert summary["completion_envelope_count"] == 0
    assert summary["completion_envelope_frames"] == []
    assert recall >= 0.97
    assert iou > 0.95
    assert area_ratio < 1.08
    assert not has_strict_self_intersection(boundary)


def test_conservative_spatial_envelope_remains_a_non_stopping_last_resort(
    monkeypatch,
) -> None:
    reference = _ellipse((120.0, 80.0), radii=(48.0, 23.0), count=192)
    theta = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
    tiny = np.column_stack((120.0 + np.cos(theta), 80.0 + np.sin(theta)))
    monkeypatch.setattr(
        "production.curve.engine.fit_sequence",
        lambda _references, _config: SimpleNamespace(controls=tiny[None]),
    )
    repaired, summary = _repair_if_needed(
        [reference],
        tiny[None],
        config=CurveProductionConfig(native_cpu_threads=1),
    )
    boundary = catmull_rom_renderer(16)(repaired[0])
    iou, recall, _precision, area_ratio = _frame_metrics(reference, boundary)
    assert summary["independent_local_refit"]["rejected_frames"] == [0]
    assert summary["completion_envelope_count"] == 1
    assert summary["completion_envelope_frames"] == [0]
    assert recall >= 0.97
    assert iou > 0.60
    assert area_ratio < 1.7
    assert not has_strict_self_intersection(boundary)


def test_refinement_guard_backs_off_a_locally_destructive_move() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    renderer = catmull_rom_renderer(16)
    baseline_keys = np.asarray((base, base + (8.0, 0.0)))
    baseline_dense = np.asarray([base + (2.0 * frame, 0.0) for frame in range(5)])
    references = [renderer(points) for points in baseline_dense]
    refined = baseline_keys.copy()
    center = np.mean(refined[1], axis=0, keepdims=True)
    refined[1] = center + 0.70 * (refined[1] - center)
    config = KeyframeDpConfig(
        target_interval=5,
        recall_floor=0.97,
        pair_vote_enabled=False,
        quality_rescue_maximum_iou_regression=0.005,
        quality_rescue_maximum_area_ratio_regression=0.01,
        native_cpu_batches=False,
        native_cpu_threads=1,
    )
    keys, _dense, _boundaries, audit, triggered, alpha = _guard_refined_controls(
        references,
        (0, 4),
        baseline_keys,
        refined,
        renderer,
        config,
        None,
    )
    assert triggered
    assert 0.0 <= alpha < 1.0
    assert np.min(audit.recall) >= 0.97
    assert np.all(audit.iou + 0.005 + 1e-12 >= 1.0)
    assert not np.array_equal(keys, refined)


def test_refinement_guard_does_not_move_subfloor_debt_to_another_frame(
    monkeypatch,
) -> None:
    baseline_keys = np.zeros((2, 3, 2), dtype=np.float64)
    refined_keys = np.ones((2, 3, 2), dtype=np.float64)
    chosen = (0, 2)
    baseline_dense = multistate_dp.materialize_controls(
        3,
        chosen,
        baseline_keys,
    )

    def audit(iou: tuple[float, ...]):
        return SimpleNamespace(
            iou=np.asarray(iou, dtype=np.float64),
            recall=np.ones((3,), dtype=np.float64),
            precision=np.ones((3,), dtype=np.float64),
            area_ratio=np.ones((3,), dtype=np.float64),
            topology_valid=np.ones((3,), dtype=bool),
        )

    baseline_audit = audit((0.849, 0.855, 0.95))
    shifted_audit = audit((0.854, 0.849, 0.95))

    def fake_audit_dense_path(_references, dense, _renderer, **_kwargs):
        selected = (
            baseline_audit
            if np.array_equal(np.asarray(dense), baseline_dense)
            else shifted_audit
        )
        return np.zeros((3, 3, 2), dtype=np.float64), selected

    monkeypatch.setattr(
        multistate_dp,
        "audit_dense_path",
        fake_audit_dense_path,
    )
    (
        keys,
        _dense,
        _boundaries,
        result,
        triggered,
        alpha,
    ) = multistate_dp._guard_refined_controls(
        [np.zeros((3, 2), dtype=np.float64) for _frame in range(3)],
        chosen,
        baseline_keys,
        refined_keys,
        lambda values: values,
        KeyframeDpConfig(
            target_interval=2,
            recall_floor=0.97,
            quality_rescue_iou_floor=0.85,
            quality_rescue_maximum_iou_regression=0.005,
            native_cpu_batches=False,
            native_cpu_threads=1,
        ),
        None,
    )
    assert triggered
    assert alpha == 0.0
    np.testing.assert_array_equal(keys, baseline_keys)
    np.testing.assert_array_equal(result.iou, baseline_audit.iou)


def test_refinement_guard_allows_redistribution_above_quality_floor(
    monkeypatch,
) -> None:
    baseline_keys = np.zeros((2, 3, 2), dtype=np.float64)
    refined_keys = np.ones((2, 3, 2), dtype=np.float64)
    chosen = (0, 2)
    baseline_dense = multistate_dp.materialize_controls(
        3,
        chosen,
        baseline_keys,
    )

    def audit(iou: tuple[float, ...]):
        return SimpleNamespace(
            iou=np.asarray(iou, dtype=np.float64),
            recall=np.ones((3,), dtype=np.float64),
            precision=np.ones((3,), dtype=np.float64),
            area_ratio=np.ones((3,), dtype=np.float64),
            topology_valid=np.ones((3,), dtype=bool),
        )

    baseline_audit = audit((0.95, 0.90, 0.90))
    redistributed_audit = audit((0.90, 0.95, 0.90))

    def fake_audit_dense_path(_references, dense, _renderer, **_kwargs):
        selected = (
            baseline_audit
            if np.array_equal(np.asarray(dense), baseline_dense)
            else redistributed_audit
        )
        return np.zeros((3, 3, 2), dtype=np.float64), selected

    monkeypatch.setattr(
        multistate_dp,
        "audit_dense_path",
        fake_audit_dense_path,
    )
    (
        keys,
        _dense,
        _boundaries,
        result,
        triggered,
        alpha,
    ) = multistate_dp._guard_refined_controls(
        [np.zeros((3, 2), dtype=np.float64) for _frame in range(3)],
        chosen,
        baseline_keys,
        refined_keys,
        lambda values: values,
        KeyframeDpConfig(
            target_interval=2,
            recall_floor=0.97,
            quality_rescue_iou_floor=0.85,
            quality_rescue_maximum_iou_regression=0.005,
            native_cpu_batches=False,
            native_cpu_threads=1,
        ),
        None,
    )
    assert not triggered
    assert alpha == 1.0
    np.testing.assert_array_equal(keys, refined_keys)
    np.testing.assert_array_equal(result.iou, redistributed_audit.iou)


def test_duplicate_closing_point_is_removed() -> None:
    points = np.asarray(((0, 0), (4, 0), (3, 2), (0, 0)), dtype=np.float64)
    normalized = normalize_control_points(points)
    assert normalized.shape == (3, 2)
    np.testing.assert_array_equal(normalized, points[:-1])


def test_segments_use_exact_one_sixth_formula() -> None:
    points = np.asarray(((0, 0), (6, 0), (6, 6), (0, 6)), dtype=np.float64)
    segments = bezier_segments(points)
    expected_first = np.asarray(
        (
            (0, 0),
            (1, -1),
            (5, -1),
            (6, 0),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(segments[0], expected_first, atol=0.0, rtol=0.0)


def test_curve_interpolates_every_point_and_is_c1_continuous() -> None:
    points = np.asarray(((1, 0), (5, 1), (6, 5), (2, 7), (-1, 4)), dtype=np.float64)
    segments = bezier_segments(points)
    for index, segment in enumerate(segments):
        values = evaluate_cubic(segment, np.asarray((0.0, 1.0)))
        np.testing.assert_allclose(values[0], points[index], atol=1e-12)
        np.testing.assert_allclose(
            values[1], points[(index + 1) % len(points)], atol=1e-12
        )
        outgoing = 3.0 * (segment[3] - segment[2])
        following = segments[(index + 1) % len(points)]
        incoming = 3.0 * (following[1] - following[0])
        np.testing.assert_allclose(outgoing, incoming, atol=1e-12)


def test_sampling_has_no_duplicated_terminal_point() -> None:
    points = np.asarray(((0, 0), (5, 0), (4, 3), (0, 4)), dtype=np.float64)
    sampled = sample_closed_curve(points, samples_per_segment=7)
    assert sampled.shape == (28, 2)
    np.testing.assert_allclose(sampled[::7], points, atol=1e-12)
    assert not np.allclose(sampled[0], sampled[-1])


def test_native_catmull_control_batches_match_materialized_boundaries() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    controls = np.asarray(
        [
            np.column_stack(
                (
                    90.0 + frame + (28.0 + 2.0 * np.sin(3.0 * angles)) * np.cos(angles),
                    70.0 + (22.0 + 1.5 * np.cos(2.0 * angles)) * np.sin(angles),
                )
            )
            for frame in range(6)
        ],
        dtype=np.float64,
    )
    references = list(sample_curve_sequence(controls, 16))
    raster = ExactDoubleRasterBatch(references)
    frames = np.arange(len(controls), dtype=np.int32)
    materialized = raster.metrics(
        frames,
        sample_curve_sequence(controls, 16),
        threads=2,
    )
    fused = raster.catmull_metrics(
        frames,
        controls,
        samples_per_segment=16,
        threads=2,
        check_topology=True,
    )
    np.testing.assert_array_equal(materialized, fused[:, :7])
    np.testing.assert_array_equal(fused[:, 7], np.ones(len(controls)))

    scales = np.asarray((1.0, 1.012, 1.04, 1.08), dtype=np.float64)
    centers = np.mean(controls, axis=1, keepdims=True)
    scale_controls = centers[:, None] + scales[None, :, None, None] * (
        controls[:, None] - centers[:, None]
    )
    scale_boundaries = sample_curve_sequence(
        scale_controls.reshape(-1, controls.shape[1], 2),
        16,
    )
    expected = raster.metrics(
        np.repeat(frames, len(scales)),
        scale_boundaries,
        threads=2,
    ).reshape(len(controls), len(scales), 7)
    actual = raster.catmull_scale_metrics(
        controls,
        scales,
        samples_per_segment=16,
        threads=2,
        check_topology=True,
    )
    np.testing.assert_array_equal(expected, actual[:, :, :7])
    np.testing.assert_array_equal(actual[:, :, 7], np.ones(actual.shape[:2]))


def test_linear_basis_exactly_matches_fixed_handle_curve() -> None:
    rng = np.random.default_rng(20260821)
    for count in (3, 6, 14):
        points = rng.normal(size=(count, 2))
        for samples in (2, 7, 16):
            basis = catmull_rom_basis_matrix(count, samples)
            np.testing.assert_allclose(np.sum(basis, axis=1), 1.0, atol=1e-15)
            np.testing.assert_allclose(
                basis @ points,
                sample_closed_curve(points, samples),
                atol=1e-12,
                rtol=1e-12,
            )


def test_whole_curve_fit_reduces_assigned_arc_error_deterministically() -> None:
    dense = np.asarray(
        [
            _ellipse((80.0 + frame, 60.0), radii=(34.0, 20.0), count=96)
            for frame in range(5)
        ]
    )
    indices = np.asarray((0, 5, 20, 37, 52, 70, 84), dtype=np.int32)
    initial = dense[:, indices]
    first, stats = fit_whole_curve_controls(
        dense,
        indices,
        initial,
        samples_per_segment=12,
        anchor_weight=1.0,
        temporal_correction_weight=2.0,
        maximum_correction_chord_fraction=0.4,
    )
    second, second_stats = fit_whole_curve_controls(
        dense,
        indices,
        initial,
        samples_per_segment=12,
        anchor_weight=1.0,
        temporal_correction_weight=2.0,
        maximum_correction_chord_fraction=0.4,
    )
    targets = contour_arc_targets(dense, indices, 12)
    basis = catmull_rom_basis_matrix(len(indices), 12)
    before = basis @ initial[0] - targets[0]
    after = basis @ first[0] - targets[0]
    assert float(np.mean(after * after)) < float(np.mean(before * before))
    assert stats.boundary_rms_after < stats.boundary_rms_before
    assert stats == second_stats
    np.testing.assert_array_equal(first, second)


def test_temporal_correction_tridiagonal_solver_matches_dense_system() -> None:
    rng = np.random.default_rng(20260822)
    value = rng.normal(size=(37, 20, 2))
    weight = 2.0
    frame_count = len(value)
    system = np.eye(frame_count, dtype=np.float64)
    diagonal = np.full((frame_count,), 2.0 * weight, dtype=np.float64)
    diagonal[0] = weight
    diagonal[-1] = weight
    system[np.arange(frame_count), np.arange(frame_count)] += diagonal
    system[np.arange(frame_count - 1), np.arange(1, frame_count)] = -weight
    system[np.arange(1, frame_count), np.arange(frame_count - 1)] = -weight
    expected = np.linalg.solve(system, value.reshape(frame_count, -1)).reshape(
        value.shape
    )
    actual = _smooth_temporal_correction(value, weight)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)


def test_track_fit_is_deterministic_and_keeps_point_correspondence() -> None:
    references = [
        _ellipse((80.0 + 2.0 * frame, 60.0 + frame), radii=(34.0, 20.0))
        for frame in range(5)
    ]
    config = FitConfig(
        control_point_count=8,
        dense_contour_samples=64,
        samples_per_segment=8,
        recall_floor=0.95,
        index_refine_passes=1,
        index_refine_radius=3,
        proxy_max_frames=5,
        scale_maximum=1.08,
        scale_step=0.01,
    )
    first = fit_sequence(references, config)
    second = fit_sequence(references, config)
    np.testing.assert_array_equal(first.controls, second.controls)
    np.testing.assert_array_equal(first.segments, second.segments)
    assert first.persistent_dense_indices == second.persistent_dense_indices
    assert first.metrics.recall_violations == 0
    assert first.metrics.self_intersections == 0
    # After subtracting known translation, a point keeps the same contour role.
    normalized = first.controls - np.asarray(
        [[[2.0 * frame, float(frame)]] for frame in range(5)]
    )
    assert float(np.max(np.std(normalized, axis=0))) < 0.6


def test_temporal_phase_stabilization_removes_cyclic_jumps_only() -> None:
    base = _ellipse((80.0, 60.0), radii=(34.0, 19.0), count=96)
    sampled = np.asarray(
        (
            np.roll(base, 23, axis=0) + (0.0, 0.0),
            np.roll(base, 61, axis=0) + (5.0, 2.0),
            np.roll(base, 7, axis=0) + (10.0, 4.0),
        )
    )
    aligned, shifts = _stabilize_temporal_phase(sampled)
    centered = aligned - np.mean(aligned, axis=1, keepdims=True)
    for frame in centered:
        np.testing.assert_allclose(frame, centered[1], atol=1e-12)
    assert shifts[1] == 0
    for before, after in zip(sampled, aligned):
        # A cyclic roll changes correspondence but not the closed point set.
        np.testing.assert_allclose(
            np.sort(before - np.mean(before, axis=0), axis=0),
            np.sort(after - np.mean(after, axis=0), axis=0),
            atol=1e-12,
        )


def test_dp_interpolates_p_then_derives_the_requested_curve() -> None:
    first = np.asarray(((0, 0), (8, 0), (8, 5), (0, 5)), dtype=np.float64)
    second = np.asarray(((2, 1), (11, 0), (9, 8), (-1, 6)), dtype=np.float64)
    midpoint = interpolate_controls(first, second, 0.5)
    rendered = catmull_rom_renderer(12)(midpoint)
    np.testing.assert_allclose(rendered, sample_closed_curve(midpoint, 12), atol=0.0)
    segments = bezier_segments(midpoint)
    np.testing.assert_allclose(segments[:, 0], midpoint, atol=0.0)
    np.testing.assert_allclose(segments[:, 3], np.roll(midpoint, -1, axis=0), atol=0.0)


def test_hard_recall_dp_reaches_target_when_curve_motion_is_linear() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [base + np.asarray((2.0 * frame, float(frame))) for frame in range(8)]
    )
    renderer = catmull_rom_renderer(12)
    references = [renderer(frame) for frame in controls]
    result = optimize_keyframes(
        references,
        controls,
        representation="catmull_rom",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=4,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
        ),
    )
    assert result.chosen_indices == (0, 7)
    assert result.summary()["recall_violations"] == 0
    assert result.summary()["minimum_iou"] == 1.0


def test_hard_recall_overrides_unachievable_key_target() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    translations = (0.0, 0.0, 28.0, 0.0, 0.0)
    controls = np.asarray([base + np.asarray((shift, 0.0)) for shift in translations])
    renderer = catmull_rom_renderer(12)
    references = [renderer(frame) for frame in controls]
    result = optimize_keyframes(
        references,
        controls,
        representation="catmull_rom",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=5,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
        ),
    )
    assert len(result.chosen_indices) > result.target_keyframes
    assert 2 in result.chosen_indices
    assert result.summary()["recall_violations"] == 0


def test_quality_rescue_can_use_unchanged_support_keys_to_localize_shape() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    center = np.mean(base, axis=0, keepdims=True)
    controls = np.asarray(
        [center + scale * (base - center) for scale in (1.3, 1.3, 1.0, 1.3, 1.3)]
    )
    renderer = catmull_rom_renderer(16)
    references = [renderer(frame) for frame in controls]
    baseline = optimize_keyframes(
        references,
        controls,
        representation="catmull_rom",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=5,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
            low_iou_quadratic_weight=4.0,
        ),
    )
    rescued = optimize_keyframes(
        references,
        controls,
        representation="catmull_rom",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=5,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
            low_iou_quadratic_weight=4.0,
            quality_rescue_enabled=True,
            quality_rescue_iou_floor=0.95,
            quality_rescue_regret_floor=0.02,
            quality_rescue_area_ratio_cap=1.10,
            quality_rescue_maximum_extra_keys=3,
        ),
    )
    assert baseline.chosen_indices == (0, 4)
    assert baseline.audit.iou[2] < 0.60
    assert rescued.chosen_indices == (0, 1, 2, 3, 4)
    assert rescued.quality_rescue_inserted_indices == (1, 2, 3)
    np.testing.assert_allclose(rescued.audit.iou, 1.0, atol=0.0)
    np.testing.assert_allclose(rescued.audit.recall, 1.0, atol=0.0)

    unbounded = optimize_keyframes(
        references,
        controls,
        representation="catmull_rom",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=5,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
            low_iou_quadratic_weight=4.0,
            quality_rescue_enabled=True,
            quality_rescue_iou_floor=0.95,
            quality_rescue_regret_floor=0.02,
            quality_rescue_area_ratio_cap=1.10,
            # Zero deliberately means that the quality guard, rather than an
            # arbitrary insertion quota, decides when rescue is complete.
            quality_rescue_maximum_extra_keys=0,
        ),
    )
    assert unbounded.chosen_indices == rescued.chosen_indices
    assert unbounded.quality_rescue_inserted_indices == (1, 2, 3)
    np.testing.assert_allclose(unbounded.audit.iou, 1.0, atol=0.0)


def test_multistate_dp_selects_exact_curve_scale_without_free_handles() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [base + np.asarray((2.0 * frame, float(frame))) for frame in range(8)]
    )
    states, labels = isotropic_curve_states(controls, (1.0, 1.02))
    renderer = catmull_rom_renderer(12)
    references = [renderer(frame) for frame in states[:, 1]]
    result = optimize_multistate_keyframes(
        references,
        states,
        state_labels=labels,
        base_controls=controls,
        representation="catmull_rom_multistate",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=4,
            recall_floor=0.97,
            maximum_gap=30,
            pair_vote_enabled=False,
        ),
        point_refine=CurvePointRefineConfig(enabled=False),
    )
    assert result.chosen_indices == (0, 7)
    assert result.chosen_state_labels == ("scale_1.020", "scale_1.020")
    assert result.summary()["minimum_iou"] == 1.0
    assert result.summary()["recall_violations"] == 0
    assert result.summary()["topology_invalid_frames"] == 0


def test_fixed_cardinality_curve_dp_keeps_requested_pareto_point() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [base + np.asarray((1.5 * frame, 0.5 * frame)) for frame in range(10)]
    )
    states, labels = isotropic_curve_states(controls, (1.0, 1.02))
    renderer = catmull_rom_renderer(12)
    references = [renderer(frame) for frame in states[:, 1]]
    result = optimize_multistate_keyframes(
        references,
        states,
        state_labels=labels,
        base_controls=controls,
        representation="catmull_rom_multistate",
        renderer=renderer,
        config=KeyframeDpConfig(
            target_interval=3,
            recall_floor=0.97,
            maximum_gap=12,
            path_selection_mode="fixed_cardinality",
            pair_vote_enabled=False,
            quality_rescue_enabled=False,
        ),
        point_refine=CurvePointRefineConfig(enabled=False),
    )
    assert result.target_keyframes == 3
    assert len(result.chosen_indices) == 3
    assert result.quality_rescue_inserted_indices == ()
    assert result.summary()["recall_violations"] == 0


def test_native_cardinality_decoder_relaxes_only_upward() -> None:
    module = native_module()
    assert module is not None and hasattr(module, "decode_cardinality_path")
    # The only route to the final frame has three keys. Asking for two must
    # therefore return three, never a less safe one-key path.
    edges = np.asarray(((0, 0, 1, 0), (1, 0, 2, 0)), dtype=np.int32)
    costs = np.asarray((1.0, 1.0), dtype=np.float64)
    frames, states, raw_cost, count = module.decode_cardinality_path(
        edges,
        costs,
        np.asarray((0.0,), dtype=np.float64),
        3,
        1,
        2,
        3,
    )
    assert tuple(frames) == (0, 1, 2)
    assert tuple(states) == (0, 0, 0)
    assert raw_cost == 2.0
    assert count == 3


def test_isotropic_state_shape_distances_are_deduplicated_exactly() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [
            base
            + np.asarray((2.0 * frame, float(frame)))
            + 0.1 * frame * np.sin(np.arange(len(base)))[:, None]
            for frame in range(8)
        ]
    )
    states, _labels = isotropic_curve_states(
        controls,
        (1.0, 1.005, 1.015, 1.035),
    )
    edges = np.asarray(
        [
            (start, start_state, end, end_state)
            for end in range(1, len(controls))
            for start in range(max(0, end - 4), end)
            for end_state in range(states.shape[1])
            for start_state in range(states.shape[1])
        ],
        dtype=np.int32,
    )
    renderer = catmull_rom_renderer(12)
    references = [renderer(value) for value in controls]
    evaluator = _MultistateIntervalEvaluator(
        references,
        states,
        renderer,
        KeyframeDpConfig(maximum_gap=4),
    )
    actual = evaluator._shape_distances_for_edges(edges)
    expected = np.asarray(
        [
            float(
                np.mean(
                    temporal_residuals(
                        np.stack(
                            (
                                states[start, start_state],
                                states[end, end_state],
                            )
                        )
                    )[0]
                )
            )
            for start, start_state, end, end_state in edges
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    assert evaluator._states_share_similarity_shape
    assert evaluator._shape_distance_unique_pairs < len(edges) // 8


def test_fallback_graph_reuses_fast_edges_without_changing_costs() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [base + np.asarray((2.0 * frame, float(frame))) for frame in range(8)]
    )
    fast, _fast_labels = isotropic_curve_states(controls, (1.0, 1.005))
    full, _full_labels = isotropic_curve_states(
        controls,
        (1.0, 1.005, 1.015, 1.025, 1.035),
    )
    renderer = catmull_rom_renderer(12)
    references = [renderer(value) for value in controls]
    raster = ExactDoubleRasterBatch(references)
    config = KeyframeDpConfig(maximum_gap=4, native_cpu_threads=2)
    seed = _MultistateIntervalEvaluator(
        references,
        fast,
        renderer,
        config,
        raster,
    )
    assert seed.precompute_all(4)
    reused = _MultistateIntervalEvaluator(
        references,
        full,
        renderer,
        config,
        raster,
        seed,
    )
    direct = _MultistateIntervalEvaluator(
        references,
        full,
        renderer,
        config,
        raster,
    )
    assert reused.precompute_all(4)
    assert direct.precompute_all(4)
    np.testing.assert_array_equal(
        reused._native_decode_edges,
        direct._native_decode_edges,
    )
    np.testing.assert_array_equal(
        reused._native_decode_costs,
        direct._native_decode_costs,
    )
    assert reused._native_reused_edge_count == seed._native_edge_count


def test_sweep_topology_matches_production_strict_crossing_semantics() -> None:
    rng = np.random.default_rng(20260821)
    shapes = []
    for count in (3, 4, 8, 14, 20, 64, 128):
        for _ in range(20):
            shapes.append(rng.normal(size=(count, 2)))
    for shape in shapes:
        assert has_strict_self_intersection(shape) == has_self_intersection(shape)


def test_sweep_topology_accepts_simple_curve_and_rejects_bow_tie() -> None:
    simple = sample_closed_curve(
        np.asarray(((0, 0), (8, 0), (8, 5), (0, 5)), dtype=np.float64), 16
    )
    bow_tie = np.asarray(((0, 0), (8, 8), (0, 8), (8, 0)), dtype=np.float64)
    assert not has_strict_self_intersection(simple)
    assert has_strict_self_intersection(bow_tie)


def test_native_topology_batch_exactly_matches_scalar_sweep() -> None:
    rng = np.random.default_rng(2026082201)
    values = np.asarray(
        [rng.normal(size=(128, 2)) for _ in range(40)],
        dtype=np.float64,
    )
    expected = np.asarray(
        [has_strict_self_intersection(value) for value in values],
        dtype=bool,
    )
    actual = strict_self_intersection_batch(values, threads=4)
    np.testing.assert_array_equal(actual, expected)


def test_lazy_topology_rejects_selected_crossing_edge_like_eager_graph() -> None:
    """The exact lazy gate must solve the same constrained DP as eager checks.

    Both endpoint curves are simple, but their index-wise midpoint crosses.
    A direct 0->2 edge is therefore invalid while the 0->1->2 route is valid.
    This is the failure mode that the Production lazy topology gate exists to
    catch without checking every unselected graph edge.
    """

    first = (
        np.asarray(
            (
                (-20.8385357387, -16.2386381486),
                (-12.5779779505, -15.5953404636),
                (8.4606834576, -25.4762975843),
                (9.0906962551, -13.7263525915),
                (16.5469038174, -0.6842814052),
            ),
            dtype=np.float64,
        )
        + 50.0
    )
    last = (
        np.asarray(
            (
                (9.1349539604, -16.5021598958),
                (16.8335671676, -0.1111056913),
                (-23.9676621854, -15.4265268191),
                (-12.2998133384, -17.1655435926),
                (6.8539476708, -30.5448832298),
            ),
            dtype=np.float64,
        )
        + 50.0
    )
    renderer = catmull_rom_renderer(16)
    assert not has_strict_self_intersection(renderer(first))
    assert not has_strict_self_intersection(renderer(last))
    assert has_strict_self_intersection(renderer(0.5 * (first + last)))

    controls = np.asarray((first, first, last), dtype=np.float64)[:, None]
    references = [renderer(first), renderer(first), renderer(last)]
    config = KeyframeDpConfig(
        maximum_gap=2,
        native_cpu_threads=2,
        recall_floor=0.0,
        pair_vote_enabled=False,
        shape_distance_weight=0.0,
    )
    raster = ExactDoubleRasterBatch(references)
    evaluator = _MultistateIntervalEvaluator(
        references,
        controls,
        renderer,
        config,
        raster,
    )
    assert evaluator.precompute_all(2)
    direct = (0, 0, 2, 0)
    assert evaluator._invalid_path_edges(_NodePath((0, 2), (0, 0), 0.0)) == (direct,)
    first_losses = _first_node_losses(references, controls, renderer, config)
    lazy_path = evaluator.decode_native(3, 1, first_losses, 100.0)
    assert lazy_path is not None
    assert lazy_path.frames == (0, 1, 2)
    assert lazy_path.states == (0, 0, 0)
    assert evaluator._lazy_topology_decodes == 2
    assert evaluator._lazy_topology_checked_edges == 3
    assert evaluator._lazy_topology_checked_frames == 4
    assert evaluator._lazy_topology_rejected_edges == 1
    assert not np.isfinite(
        evaluator._native_decode_costs[evaluator._native_edge_offset(*direct)]
    )

    eager = raster.edge_metrics(
        render_control_sequence(renderer, controls),
        evaluator._native_decode_edges,
        recall_floor=0.0,
        low_iou_quadratic_weight=float(config.low_iou_quadratic_weight),
        threads=2,
        check_topology=True,
    )
    spans = evaluator._native_decode_edges[:, 2] - evaluator._native_decode_edges[:, 0]
    eager_costs = np.where(
        (eager[:, 4] > 0.5) & (np.rint(eager[:, 3]).astype(np.int32) == spans),
        eager[:, 0],
        np.inf,
    )
    eager_frames, eager_states, _eager_cost = native_module().decode_penalty_path(
        evaluator._native_decode_edges,
        np.ascontiguousarray(eager_costs, dtype=np.float64),
        first_losses,
        3,
        1,
        100.0,
    )
    assert tuple(eager_frames) == lazy_path.frames
    assert tuple(eager_states) == lazy_path.states
    checked_frames = evaluator._lazy_topology_checked_frames
    repeated = evaluator.decode_native(3, 1, first_losses, 100.0)
    assert repeated is not None and repeated.frames == lazy_path.frames
    assert evaluator._lazy_topology_checked_frames == checked_frames


def test_native_edge_offset_matches_every_stable_graph_row() -> None:
    base = np.asarray(
        ((50, 30), (70, 28), (82, 43), (72, 61), (48, 63), (38, 45)),
        dtype=np.float64,
    )
    controls = np.asarray(
        [base + np.asarray((2.0 * frame, float(frame))) for frame in range(9)]
    )
    states, _labels = isotropic_curve_states(controls, (1.0, 1.01, 1.03))
    renderer = catmull_rom_renderer(8)
    references = [renderer(value) for value in controls]
    evaluator = _MultistateIntervalEvaluator(
        references,
        states,
        renderer,
        KeyframeDpConfig(maximum_gap=4, native_cpu_threads=2),
        ExactDoubleRasterBatch(references),
    )
    assert evaluator.precompute_all(4)
    for expected_index, row in enumerate(evaluator._native_decode_edges):
        assert evaluator._native_edge_offset(*(int(value) for value in row)) == (
            expected_index
        )
