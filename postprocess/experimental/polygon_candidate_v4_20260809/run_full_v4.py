#!/usr/bin/env python3
"""Full-range V4 candidate and optional fixed-key C2 validation."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

from shapely.geometry import box

from experimental.polygon_candidate_v4_20260809.low_dim_refinement import (
    refine_path_sequential,
)
from experimental.polygon_candidate_v4_20260809.run_low_dim_experiment import (
    _rank_key_targets_scored,
    _tail,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    evaluate_segments,
    load_raw_masks,
    load_segments,
    summarize,
)
from experimental.polygon_recall_optimizer.pareto_dp import optimize_pareto_frontier
from experimental.polygon_recall_optimizer.sqlite_export import export_selected_sqlite
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


def _audit(segments, quality, borders, visible, *, start_frame, end_frame):
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
        direct, segments, start_frame=start_frame, end_frame=end_frame
    )
    border = audit_border_safety(borders, segments)
    agreement = compare_geometry_paths(direct, overlay)
    if summary["recall_min"] + 1e-8 < 0.97 or not border["passed"]:
        raise RuntimeError("full V4 audit failed a hard constraint")
    if agreement["symmetric_difference_max_area"] > 1e-7:
        raise RuntimeError("editor/Overlay reconstruction disagrees")
    return direct, {
        "quality": summary,
        "tail": _tail(direct),
        "worst_frames": _worst_frames(direct),
        "border": border,
        "path_agreement": agreement,
    }


def _worst_frames(rows, limit: int = 20):
    """Persist enough tail context to audit failures without rerunning DP."""

    def record(row):
        return {
            "frame": int(row.frame),
            "track_id": str(row.track_id),
            "segment_id": int(row.segment_id),
            "is_keyframe": bool(row.is_keyframe),
            "iou": float(row.iou),
            "recall": float(row.recall),
            "precision": float(row.precision),
            "area_ratio": float(row.area_ratio),
        }

    return {
        "lowest_iou": [
            record(row) for row in sorted(rows, key=lambda item: item.iou)[:limit]
        ],
        "largest_area_ratio": [
            record(row)
            for row in sorted(rows, key=lambda item: item.area_ratio, reverse=True)[:limit]
        ],
    }


def _write_keyframes(path: Path, segments) -> None:
    rows = []
    for track_id, values in segments.items():
        for segment in values:
            for keyframe in segment.keyframes:
                rows.append(
                    {
                        "track_id": str(track_id),
                        "segment_id": int(segment.segment_id),
                        "frame": int(keyframe.frame),
                        "components": [
                            {
                                "slot": int(slot),
                                "kind": component.kind,
                                "values": component.values,
                            }
                            for slot, component in keyframe.components
                        ],
                    }
                )
    path.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _distributed_targets(segments, rows, maximum: int):
    if maximum <= 0:
        return []
    row_by_segment = {}
    for row in rows:
        row_by_segment.setdefault(int(row.segment_id), []).append(row)
    key_total = sum(
        len(segment.keyframes)
        for values in segments.values()
        for segment in values
    )
    ranked_by_segment = []
    for track_id, values in segments.items():
        for segment in values:
            allocation = max(
                1,
                int(math.ceil(maximum * len(segment.keyframes) / max(key_total, 1))),
            )
            ranked_by_segment.append(
                _rank_key_targets_scored(
                    segment,
                    row_by_segment.get(int(segment.segment_id), []),
                    allocation,
                )
            )
    # First cover every non-empty segment, then spend the remaining budget on
    # the globally most difficult keys.  This avoids both dictionary-order
    # starvation and the quality loss of strict equal round-robin allocation.
    output = [
        (candidates[0][1], candidates[0][2])
        for candidates in ranked_by_segment
        if candidates
    ][:maximum]
    remaining = sorted(
        (candidate for candidates in ranked_by_segment for candidate in candidates[1:]),
        reverse=True,
    )
    output.extend(
        (frame, track_id)
        for _score, frame, track_id in remaining[: max(0, maximum - len(output))]
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--target-interval", type=float, default=5.0)
    parser.add_argument("--candidate-mode", default="legacy_temporal_recall")
    parser.add_argument("--temporal-recall-quantile", type=float, default=0.97)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--edge-processes", type=int, default=5)
    parser.add_argument("--sequential-targets", type=int, default=0)
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
    all_segments = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    segments, topology_fallbacks = supported_single_component_segments(all_segments)
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
        local_recall_floor=0.97,
    )
    started = time.perf_counter()
    result = optimize_pareto_frontier(
        segments,
        constraints,
        quality_masks=quality,
        border_constraints=borders,
        visible_bounds=(0.0, 0.0, float(width), float(height)),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recall_floor=0.97,
        max_edge_span_frames=30,
        point_count=23,
        max_anchor_scale=1.25,
        anchor_state_count=4,
        anchor_expansion=0.30,
        selection="target_interval",
        target_mean_key_interval=args.target_interval,
        workers=args.workers,
        edge_processes=args.edge_processes,
        stored_vertex_contract=True,
        pair_vote_states=True,
        solver_mode="target_only",
        candidate_mode=args.candidate_mode,
        temporal_recall_quantile=args.temporal_recall_quantile,
    )
    stage1_seconds = time.perf_counter() - started
    stage1_rows, stage1_audit = _audit(
        result.segments,
        quality,
        borders,
        visible,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    _write_keyframes(args.output_dir / "stage1_keyframes.json", result.segments)
    final_segments = result.segments
    sequential_report = None
    if args.sequential_targets > 0:
        targets = _distributed_targets(
            result.segments, stage1_rows, args.sequential_targets
        )
        sequential = refine_path_sequential(
            result.segments,
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
        final_segments = sequential.segments
        _rows, sequential_audit = _audit(
            final_segments,
            quality,
            borders,
            visible,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        sequential_report = {
            "target_count": len(targets),
            "accepted_keys": sequential.accepted_keys,
            "elapsed_seconds": sequential.elapsed_seconds,
            "pipeline_seconds": stage1_seconds + sequential.elapsed_seconds,
            "records": sequential.records,
            "audit": sequential_audit,
        }
    _write_keyframes(args.output_dir / "final_keyframes.json", final_segments)
    export = None
    if args.export_sqlite:
        output_sqlite = args.output_dir / (
            f"v4_{args.candidate_mode}_interval{args.target_interval:g}.sqlite"
        )
        export = export_selected_sqlite(
            args.source_sqlite,
            output_sqlite,
            final_segments,
            quality,
            label=args.label,
            target_mean_key_interval=args.target_interval,
            recall_floor=0.97,
            selection_reason="candidate_v4_full_validation",
            algorithm="experimental.polygon_candidate_v4_20260809",
        )
    selected = result.selected
    report = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "source_sqlite": str(args.source_sqlite.resolve()),
        "configuration": vars(args) | {
            "source_sqlite": str(args.source_sqlite),
            "output_dir": str(args.output_dir),
        },
        "topology_fallbacks": topology_fallbacks,
        "raw_observations": len(quality),
        "segment_count": sum(len(values) for values in segments.values()),
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "stage1": {
            "wall_seconds": stage1_seconds,
            "optimizer_seconds": result.elapsed_seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "anchor_states": result.anchor_state_total,
            "edge_evaluations": result.edge_evaluations,
            "feasible_edges": result.feasible_edges,
            "feasible_edge_ratio": result.feasible_edges
            / max(result.edge_evaluations, 1),
            "keyframe_count": selected.keyframe_count,
            "mean_key_interval": selected.mean_key_interval,
            "mean_iou": selected.mean_iou,
            "min_recall": selected.min_recall,
            "audit": stage1_audit,
        },
        "sequential": sequential_report,
        "export": export,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage1_iou": selected.mean_iou,
                "stage1_seconds": stage1_seconds,
                "sequential_iou": (
                    None
                    if sequential_report is None
                    else sequential_report["audit"]["quality"]["iou_mean"]
                ),
                "pipeline_seconds": (
                    stage1_seconds
                    if sequential_report is None
                    else sequential_report["pipeline_seconds"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
