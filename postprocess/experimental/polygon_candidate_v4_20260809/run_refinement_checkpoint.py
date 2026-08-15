#!/usr/bin/env python3
"""Sweep low-dimensional refinement without repeating the expensive DP stage."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from shapely.geometry import box

from experimental.polygon_candidate_v4_20260809.low_dim_refinement import (
    refine_path_sequential,
)
from experimental.polygon_candidate_v4_20260809.run_full_v4 import (
    _audit,
    _distributed_targets,
    _write_keyframes,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.sqlite_export import export_selected_sqlite
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    build_border_safety_constraints,
    expand_border_constraints,
    supported_single_component_segments,
    video_dimensions,
)
from overlay_renderer.keyframe_cache import Component, Keyframe


def _load_checkpoint(path: Path, segments):
    grouped: dict[tuple[str, int], list[Keyframe]] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        keyframe = Keyframe(
            int(row["frame"]),
            tuple(
                (
                    int(component["slot"]),
                    Component(str(component["kind"]), component["values"]),
                )
                for component in row["components"]
            ),
        )
        grouped.setdefault(
            (str(row["track_id"]), int(row["segment_id"])), []
        ).append(keyframe)
    output = {}
    for track_id, values in segments.items():
        output[track_id] = []
        for segment in values:
            keys = grouped.get((str(track_id), int(segment.segment_id)))
            if keys is None:
                raise ValueError(
                    f"checkpoint has no keys for track={track_id} "
                    f"segment={segment.segment_id}"
                )
            output[track_id].append(
                replace(
                    segment,
                    interpolation_method="linear_polygon_index_v1",
                    keyframes=tuple(sorted(keys, key=lambda item: item.frame)),
                )
            )
    unused = set(grouped) - {
        (str(track_id), int(segment.segment_id))
        for track_id, values in segments.items()
        for segment in values
    }
    if unused:
        raise ValueError(f"checkpoint contains {len(unused)} unknown segments")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--target-interval", type=float, default=5.0)
    parser.add_argument("--sequential-targets", type=int, required=True)
    parser.add_argument("--normal-controls", type=int, default=6)
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
    supported, topology_fallbacks = supported_single_component_segments(loaded)
    quality = {item: raw for item, raw in quality.items() if item[1] in supported}
    stage1 = _load_checkpoint(args.checkpoint, supported)
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
        local_recall_floor=0.97,
    )
    stage1_rows, stage1_audit = _audit(
        stage1,
        quality,
        borders,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    targets = _distributed_targets(stage1, stage1_rows, args.sequential_targets)
    started = time.perf_counter()
    refined = refine_path_sequential(
        stage1,
        quality,
        constraints,
        borders,
        targets,
        recall_floor=0.97,
        width=width,
        height=height,
        normal_control_count=args.normal_controls,
        rounds=args.rounds,
    )
    final_rows, final_audit = _audit(
        refined.segments,
        quality,
        borders,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    audit_seconds = time.perf_counter() - started - refined.elapsed_seconds
    _write_keyframes(args.output_dir / "final_keyframes.json", refined.segments)
    export = None
    if args.export_sqlite:
        export = export_selected_sqlite(
            args.source_sqlite,
            args.output_dir / "refined.sqlite",
            refined.segments,
            quality,
            label=args.label,
            target_mean_key_interval=args.target_interval,
            recall_floor=0.97,
            selection_reason="candidate_v4_checkpoint_refinement",
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
        "stage1": stage1_audit,
        "refinement": {
            "requested_targets": args.sequential_targets,
            "target_count": len(targets),
            "accepted_keys": refined.accepted_keys,
            "elapsed_seconds": refined.elapsed_seconds,
            "audit_seconds": audit_seconds,
            "records": refined.records,
            "audit": final_audit,
        },
        "export": export,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage1_iou": stage1_audit["quality"]["iou_mean"],
                "final_iou": final_audit["quality"]["iou_mean"],
                "accepted": refined.accepted_keys,
                "seconds": refined.elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
