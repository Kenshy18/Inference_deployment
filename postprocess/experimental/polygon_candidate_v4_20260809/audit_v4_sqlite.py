#!/usr/bin/env python3
"""Independent geometry/schema audit for a V4 experimental SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from shapely.geometry import box

from experimental.polygon_candidate_v4_20260809.run_low_dim_experiment import _tail
from experimental.polygon_recall_optimizer.audit_superior import _vertex_safety_audit
from experimental.polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from experimental.polygon_recall_optimizer.sqlite_export import schema_fingerprint
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    audit_border_safety,
    build_border_safety_constraints,
    compare_geometry_paths,
    evaluate_direct,
    expand_border_constraints,
    supported_single_component_segments,
    video_dimensions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument(
        "--allow-preserved-topology-fallbacks", action="store_true"
    )
    args = parser.parse_args()

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    loaded = load_segments(
        args.output_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    segments, topology_fallbacks = supported_single_component_segments(loaded)
    quality = {item: raw for item, raw in quality.items() if item[1] in segments}
    width, height = video_dimensions(args.source_sqlite)
    visible = box(0.0, 0.0, float(width), float(height))
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
        local_recall_floor=args.recall_floor,
    )
    direct = evaluate_direct(
        quality,
        segments,
        visible_rectangle=visible,
        border_constraints=borders,
    )
    overlay = evaluate_segments(
        quality,
        segments,
        visible_rectangle=visible,
        border_constraints=borders,
    )
    summary = summarize(
        direct,
        segments,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    agreement = compare_geometry_paths(direct, overlay)
    border = audit_border_safety(borders, segments)
    vertex = _vertex_safety_audit(segments)
    interpolation_methods = sorted(
        {segment.interpolation_method for values in segments.values() for segment in values}
    )
    point_counts = sorted(
        {
            len(component.values)
            for values in segments.values()
            for segment in values
            for keyframe in segment.keyframes
            for _slot, component in keyframe.components
            if component.kind == "polygon"
        }
    )
    with sqlite3.connect(
        f"file:{args.source_sqlite.resolve()}?mode=ro", uri=True
    ) as source:
        source_fingerprint = schema_fingerprint(source)
    with sqlite3.connect(
        f"file:{args.output_sqlite.resolve()}?mode=ro", uri=True
    ) as output:
        output_fingerprint = schema_fingerprint(output)
        integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = list(output.execute("PRAGMA foreign_key_check"))
    blocking = {
        "recall_below_floor": int(summary["recall_below_097"]),
        "border_failed": int(not border["passed"]),
        "geometry_path_mismatch": int(
            agreement["nonzero_difference_count"]
        ),
        "schema_changed": int(source_fingerprint != output_fingerprint),
        "integrity_failed": int(integrity != "ok"),
        "foreign_key_errors": len(foreign_keys),
        "topology_fallbacks": (
            0
            if args.allow_preserved_topology_fallbacks
            else len(topology_fallbacks)
        ),
        "wrong_interpolation_contract": int(
            interpolation_methods != ["linear_polygon_index_v1"]
        ),
        "wrong_polygon_point_count": int(point_counts != [23]),
        "invalid_keyframes": int(vertex["invalid_keyframe_count"]),
        "self_intersecting_keyframes": int(
            vertex["keyframe_self_intersection_count"]
        ),
        "adjacent_winding_flips": int(vertex["adjacent_winding_flip_count"]),
        "adjacent_alignment_reversals": int(
            vertex["adjacent_best_alignment_reversal_count"]
        ),
        "adjacent_alignment_shifts": int(
            vertex["adjacent_best_alignment_nonzero_shift_count"]
        ),
        "invalid_integer_frames": int(vertex["invalid_integer_frame_count"]),
        "integer_self_intersections": int(
            vertex["integer_self_intersection_count"]
        ),
        "integer_winding_flips": int(vertex["integer_winding_flip_count"]),
        "invalid_fractional_samples": int(
            vertex["invalid_fractional_sample_count"]
        ),
        "fractional_self_intersections": int(
            vertex["fractional_self_intersection_count"]
        ),
        "fractional_winding_flips": int(vertex["fractional_winding_flip_count"]),
    }
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "output_sqlite": str(args.output_sqlite.resolve()),
        "quality": summary,
        "tail": _tail(direct),
        "border": border,
        "path_agreement": agreement,
        "vertex_safety": vertex,
        "interpolation_methods": interpolation_methods,
        "polygon_point_counts": point_counts,
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "schema": {
            "source_fingerprint": source_fingerprint,
            "output_fingerprint": output_fingerprint,
            "unchanged": source_fingerprint == output_fingerprint,
            "integrity_check": integrity,
            "foreign_key_error_count": len(foreign_keys),
        },
        "blocking": blocking,
        "passed": not any(blocking.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"], "blocking": blocking}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
