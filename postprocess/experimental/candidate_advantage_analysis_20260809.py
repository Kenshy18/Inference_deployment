#!/usr/bin/env python3
"""Ablate legacy/temporal anchor states on one difficult segment.

This diagnostic intentionally invokes both historical and independent border
builders.  It never opens video pixels and does not alter production code.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from shapely.geometry import box

from overlay_renderer.keyframe_cache import _numpy_resample

from experimental.alternating_temporal_pareto.independent_border import (
    build_independent_border_constraints,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    _raw_keyframe,
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.pareto_dp import (
    _build_pair_vote_sources,
    _geometry_iou,
    _keyframe_geometry,
    _make_feasible_anchors,
    optimize_segment_pareto,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    build_border_safety_constraints,
    expand_border_constraints,
    video_dimensions,
)


def _point_at(frontier, target: float, span: int):
    return min(
        frontier,
        key=lambda point: (
            abs(span / max(point.keyframe_count - 1, 1) - target),
            -point.mean_iou,
        ),
    )


def _summarize_frontier(frontier, span: int) -> dict[str, object]:
    return {
        str(target): {
            "keyframe_count": point.keyframe_count,
            "mean_key_interval": span / max(point.keyframe_count - 1, 1),
            "mean_iou": point.mean_iou,
            "min_recall": point.min_recall,
        }
        for target in (1.0, 3.0, 5.0, 8.0, 9.0, 10.0, 15.0)
        for point in [_point_at(frontier, target, span)]
    }


def _raw_topology(segment, quality, point_count: int):
    keys = []
    for key in segment.keyframes:
        raw = quality.get((key.frame, segment.track_id))
        keys.append(key if raw is None else _raw_keyframe(raw, point_count=point_count))
    return replace(segment, keyframes=tuple(keys))


def _classify_selected_states(
    segment,
    point,
    constraints,
    quality,
    borders,
    *,
    visible,
    point_count: int,
) -> dict[str, object]:
    by_frame = {
        frame: raw
        for (frame, track_id), raw in quality.items()
        if track_id == segment.track_id
        and segment.first_frame <= frame <= segment.last_frame
    }
    pair_sources = _build_pair_vote_sources(
        sorted(by_frame),
        by_frame,
        point_count=point_count,
        max_edge_span_frames=30,
    )
    counts = {"base": 0, "expanded": 0, "pair_vote": 0, "unknown": 0}
    rows = []
    for selected in point.keyframes:
        identity = (selected.frame, segment.track_id)
        common = dict(
            segment=segment,
            raw=constraints[identity],
            quality_raw=quality[identity],
            recall_floor=0.97,
            point_count=point_count,
            max_anchor_scale=1.25,
            anchor_iou_floor=0.0,
            stored_vertex_contract=True,
            border_constraint=borders.get(identity),
            visible_rectangle=visible,
        )
        full = _make_feasible_anchors(
            **common,
            anchor_state_count=4,
            anchor_expansion=0.30,
            extra_sources=pair_sources.get(selected.frame),
        )
        without_pair = _make_feasible_anchors(
            **common,
            anchor_state_count=4,
            anchor_expansion=0.30,
            extra_sources=None,
        )
        unexpanded = _make_feasible_anchors(
            **common,
            anchor_state_count=4,
            anchor_expansion=0.0,
            extra_sources=None,
        )
        geometry = _keyframe_geometry(selected)
        full_index = next(
            (
                index
                for index, candidate in enumerate(full)
                if _geometry_iou(geometry, _keyframe_geometry(candidate)) >= 0.9999
            ),
            -1,
        )
        if not any(
            _geometry_iou(geometry, _keyframe_geometry(candidate)) >= 0.9999
            for candidate in without_pair
        ):
            kind = "pair_vote"
        elif any(
            _geometry_iou(geometry, _keyframe_geometry(candidate)) >= 0.9999
            for candidate in unexpanded
        ):
            kind = "base"
        else:
            kind = "expanded"
        if full_index < 0:
            kind = "unknown"
        counts[kind] += 1
        rows.append(
            {
                "frame": int(selected.frame),
                "kind": kind,
                "full_candidate_index": full_index,
                "area": float(geometry.area),
            }
        )
    return {"counts": counts, "keys": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", default="61")
    parser.add_argument("--segment-start", type=int, default=14583)
    parser.add_argument("--segment-end", type=int, default=15187)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--edge-processes", type=int, default=5)
    args = parser.parse_args()
    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.segment_start,
        end_frame=args.segment_end,
    )
    segments = load_segments(
        args.baseline_sqlite,
        label=args.label,
        start_frame=args.segment_start,
        end_frame=args.segment_end,
    )
    segment = next(
        value
        for value in segments[args.track_id]
        if value.first_frame == args.segment_start
        and value.last_frame == args.segment_end
    )
    raw_segment = _raw_topology(segment, quality, 23)
    width, height = video_dimensions(args.source_sqlite)
    visible = box(0.0, 0.0, float(width), float(height))
    border_config = BorderExpansionConfig()
    historical_constraints, historical_preparation = expand_border_constraints(
        quality, width=width, height=height, config=border_config
    )
    historical_borders, historical_border_preparation = (
        build_border_safety_constraints(
            quality,
            historical_constraints,
            width=width,
            height=height,
            config=border_config,
            local_recall_floor=0.97,
        )
    )
    independent_constraints, independent_borders, independent_preparation = (
        build_independent_border_constraints(
            quality,
            width=width,
            height=height,
            config=border_config,
            local_recall_floor=0.97,
        )
    )
    variants = [
        ("legacy_full_historical", segment, historical_constraints, historical_borders, "legacy", 4, 0.30, True),
        ("legacy_full_independent", segment, independent_constraints, independent_borders, "legacy", 4, 0.30, True),
        ("legacy_raw_topology_full", raw_segment, historical_constraints, historical_borders, "legacy", 4, 0.30, True),
        ("legacy_no_pair", segment, historical_constraints, historical_borders, "legacy", 4, 0.30, False),
        ("legacy_pair_no_expansion", segment, historical_constraints, historical_borders, "legacy", 2, 0.0, True),
        ("legacy_base_only", segment, historical_constraints, historical_borders, "legacy", 1, 0.0, False),
        ("temporal7_historical", segment, historical_constraints, historical_borders, "temporal7", 4, 0.30, False),
        ("temporal7_independent", segment, independent_constraints, independent_borders, "temporal7", 4, 0.30, False),
    ]
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "segment": {
            "track_id": args.track_id,
            "first_frame": segment.first_frame,
            "last_frame": segment.last_frame,
            "span": segment.last_frame - segment.first_frame,
        },
        "historical_border_preparation": historical_border_preparation,
        "historical_expansion_preparation": historical_preparation,
        "independent_border_preparation": independent_preparation,
        "variants": {},
    }
    for (
        name,
        current_segment,
        constraints,
        borders,
        candidate_mode,
        state_count,
        expansion,
        pair_vote,
    ) in variants:
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        frontier, edge_count, feasible_count, anchor_count = optimize_segment_pareto(
            current_segment,
            constraints,
            quality_masks=quality,
            border_constraints=borders,
            visible_bounds=(0.0, 0.0, float(width), float(height)),
            start_frame=args.segment_start,
            end_frame=args.segment_end,
            recall_floor=0.97,
            max_edge_span_frames=30,
            point_count=23,
            max_anchor_scale=1.25,
            anchor_state_count=state_count,
            anchor_expansion=expansion,
            edge_processes=args.edge_processes,
            stored_vertex_contract=True,
            pair_vote_states=pair_vote,
            candidate_mode=candidate_mode,
        )
        elapsed = time.perf_counter() - started
        span = segment.last_frame - segment.first_frame
        variant = {
            "seconds": elapsed,
            "candidate_mode": candidate_mode,
            "production_key_shapes_available": current_segment is segment,
            "historical_border_transform": constraints is historical_constraints,
            "anchor_state_count": state_count,
            "anchor_expansion": expansion,
            "pair_vote_states": pair_vote,
            "anchor_states": anchor_count,
            "edge_evaluations": edge_count,
            "feasible_edges": feasible_count,
            "feasible_edge_ratio": feasible_count / max(edge_count, 1),
            "frontier_size": len(frontier),
            "minimum_keys": frontier[0].keyframe_count,
            "maximum_keys": frontier[-1].keyframe_count,
            "minimum_key_interval": span / max(frontier[-1].keyframe_count - 1, 1),
            "maximum_key_interval": span / max(frontier[0].keyframe_count - 1, 1),
            "targets": _summarize_frontier(frontier, span),
        }
        if name == "legacy_full_historical":
            variant["selected_state_classes_at_interval5"] = (
                _classify_selected_states(
                    segment,
                    _point_at(frontier, 5.0, span),
                    constraints,
                    quality,
                    borders,
                    visible=visible,
                    point_count=23,
                )
            )
        report["variants"][name] = variant
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"DONE {name} seconds={elapsed:.3f} "
            f"target5={variant['targets']['5.0']}",
            flush=True,
        )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
