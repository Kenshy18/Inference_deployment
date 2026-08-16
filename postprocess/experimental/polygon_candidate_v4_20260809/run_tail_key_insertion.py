#!/usr/bin/env python3
"""Add a few quality-driven keys after DP without fixing the key budget."""

from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import shapely

from experimental.alternating_temporal_pareto.refinement import (
    _candidate_segment,
    _local_loss,
)
from experimental.polygon_candidate_v4_20260809.low_dim_refinement import (
    _coordinate_optimize,
)
from experimental.polygon_candidate_v4_20260809.run_full_v4 import (
    _audit,
    _write_keyframes,
)
from experimental.polygon_candidate_v4_20260809.run_refinement_checkpoint import (
    _load_checkpoint,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.pareto_dp import (
    _build_pair_vote_sources,
    _keyframe_geometry,
    _make_feasible_anchors,
    _make_temporal7_anchors,
    _merge_anchor_sets,
)
from experimental.polygon_recall_optimizer.sqlite_export import export_selected_sqlite
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    build_border_safety_constraints,
    expand_border_constraints,
    supported_single_component_segments,
    video_dimensions,
)
from overlay_renderer.keyframe_cache import Keyframe, _components_at


def _interpolated_keyframe(segment, frame: int) -> Keyframe:
    components = _components_at(
        list(segment.keyframes), int(frame), segment.interpolation_method
    )
    return Keyframe(
        int(frame), tuple((index, component) for index, component in enumerate(components))
    )


def _problem_targets(segments, rows, *, threshold: float, maximum: int):
    segment_by_id = {
        int(segment.segment_id): segment
        for values in segments.values()
        for segment in values
    }
    ranked = sorted(
        (
            row
            for row in rows
            if not row.is_keyframe and row.iou < threshold
        ),
        key=lambda row: (row.iou, -row.area_ratio),
    )
    used = set()
    output = []
    for row in ranked:
        segment = segment_by_id[int(row.segment_id)]
        frames = [key.frame for key in segment.keyframes]
        position = bisect.bisect_left(frames, row.frame)
        if position == 0 or position == len(frames):
            continue
        identity = (int(row.segment_id), frames[position - 1], frames[position])
        if identity in used:
            continue
        used.add(identity)
        output.append((row.frame, row.track_id, row.iou, row.area_ratio))
        if len(output) >= maximum:
            break
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-insertions", type=int, default=10)
    parser.add_argument("--normal-controls", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--export-sqlite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    loaded = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    source_segments, topology_fallbacks = supported_single_component_segments(loaded)
    current = _load_checkpoint(args.checkpoint, source_segments)
    quality = {item: raw for item, raw in quality.items() if item[1] in current}
    source_by_id = {
        int(segment.segment_id): segment
        for values in source_segments.values()
        for segment in values
    }
    width, height = video_dimensions(args.source_sqlite)
    visible = shapely.box(0.0, 0.0, float(width), float(height))
    config = BorderExpansionConfig()
    constraints, expansion = expand_border_constraints(
        quality, width=width, height=height, config=config
    )
    borders, border_preparation = build_border_safety_constraints(
        quality,
        constraints,
        width=width,
        height=height,
        config=config,
        local_recall_floor=0.97,
    )
    baseline_rows, baseline_audit = _audit(
        current,
        quality,
        borders,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    targets = _problem_targets(
        current,
        baseline_rows,
        threshold=args.iou_threshold,
        maximum=args.max_insertions,
    )
    started = time.perf_counter()
    records = []
    for frame, track_id, initial_iou, initial_area_ratio in targets:
        segment = next(
            segment
            for segment in current[track_id]
            if segment.first_frame <= frame <= segment.last_frame
        )
        source_segment = source_by_id[int(segment.segment_id)]
        quality_by_frame = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in quality.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        constraint_by_frame = {
            candidate_frame: raw
            for (candidate_frame, candidate_track), raw in constraints.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        border_by_frame = {
            candidate_frame: value
            for (candidate_frame, candidate_track), value in borders.items()
            if candidate_track == track_id
            and segment.first_frame <= candidate_frame <= segment.last_frame
        }
        raw_by_frame = constraint_by_frame
        pair_sources = _build_pair_vote_sources(
            sorted(raw_by_frame),
            quality_by_frame,
            point_count=23,
            max_edge_span_frames=30,
        )
        baseline = _interpolated_keyframe(segment, frame)
        baseline_loss, feasible, _count = _local_loss(
            segment,
            baseline,
            quality_by_frame,
            constraint_by_frame,
            border_by_frame,
            recall_floor=0.97,
            width=width,
            height=height,
            regularization_source=baseline,
        )
        if not feasible:
            raise RuntimeError(f"existing path unexpectedly failed at {frame}:{track_id}")
        legacy = _make_feasible_anchors(
            source_segment,
            raw_by_frame[frame],
            quality_raw=quality_by_frame[frame],
            recall_floor=0.97,
            point_count=23,
            max_anchor_scale=1.25,
            anchor_state_count=4,
            anchor_expansion=0.30,
            anchor_iou_floor=0.0,
            anchor_relative_iou_margin=None,
            anchor_point_strategy="uniform",
            max_anchor_hausdorff_px=None,
            stored_vertex_contract=True,
            extra_sources=pair_sources.get(frame),
            border_constraint=border_by_frame.get(frame),
            visible_rectangle=visible,
        )
        temporal = _make_temporal7_anchors(
            frame,
            quality_by_frame,
            raw_by_frame[frame],
            quality_by_frame[frame],
            recall_floor=0.97,
            point_count=23,
            max_anchor_scale=1.25,
            window_radii=(2, 5, 10),
            recall_quantile=0.97,
            border_constraint=border_by_frame.get(frame),
            visible_rectangle=visible,
            candidate_names=frozenset(
                ("short_recall", "medium_recall", "long_recall")
            ),
        )
        candidates = _merge_anchor_sets(legacy, temporal)
        scored = []
        for candidate in candidates:
            loss, candidate_feasible, _evaluated = _local_loss(
                segment,
                candidate,
                quality_by_frame,
                constraint_by_frame,
                border_by_frame,
                recall_floor=0.97,
                width=width,
                height=height,
                regularization_source=baseline,
            )
            if candidate_feasible:
                scored.append((loss, candidate))
        if not scored:
            continue
        candidate_loss, candidate = min(scored, key=lambda item: item[0])
        refined = _coordinate_optimize(
            segment,
            candidate,
            quality_by_frame,
            constraint_by_frame,
            border_by_frame,
            recall_floor=0.97,
            width=width,
            height=height,
            normal_control_count=args.normal_controls,
            rounds=args.rounds,
        )
        if refined is not None and refined.loss < candidate_loss:
            candidate_loss = refined.loss
            candidate = refined.keyframe
        if candidate_loss + 1e-8 >= baseline_loss:
            continue
        updated = _candidate_segment(segment, frame, candidate)
        current[track_id] = [
            updated if value.segment_id == segment.segment_id else value
            for value in current[track_id]
        ]
        records.append(
            {
                "frame": int(frame),
                "track_id": str(track_id),
                "initial_iou": float(initial_iou),
                "initial_area_ratio": float(initial_area_ratio),
                "baseline_loss": float(baseline_loss),
                "selected_loss": float(candidate_loss),
                "gain": float(baseline_loss - candidate_loss),
                "selected_area": float(_keyframe_geometry(candidate).area),
            }
        )
    elapsed = time.perf_counter() - started
    _rows, final_audit = _audit(
        current,
        quality,
        borders,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    _write_keyframes(args.output_dir / "final_keyframes.json", current)
    export = None
    if args.export_sqlite:
        export = export_selected_sqlite(
            args.source_sqlite,
            args.output_dir / "tail_inserted.sqlite",
            current,
            quality,
            label=args.label,
            target_mean_key_interval=None,
            recall_floor=0.97,
            selection_reason="candidate_v4_adaptive_tail_key",
            algorithm="experimental.polygon_candidate_v4_20260809",
        )
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "configuration": {
            **vars(args),
            "source_sqlite": str(args.source_sqlite),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
        },
        "topology_fallbacks": topology_fallbacks,
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "target_count": len(targets),
        "accepted_insertions": len(records),
        "elapsed_seconds": elapsed,
        "targets": targets,
        "records": records,
        "baseline": baseline_audit,
        "final": final_audit,
        "export": export,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "accepted": len(records),
                "baseline_iou": baseline_audit["quality"]["iou_mean"],
                "final_iou": final_audit["quality"]["iou_mean"],
                "baseline_min_iou": baseline_audit["tail"]["iou_min"],
                "final_min_iou": final_audit["tail"]["iou_min"],
                "seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
