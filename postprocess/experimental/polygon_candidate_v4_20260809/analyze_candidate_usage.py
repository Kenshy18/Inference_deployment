#!/usr/bin/env python3
"""Classify selected stage-1 anchors by their generating candidate family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import shapely

from experimental.polygon_candidate_v4_20260809.run_refinement_checkpoint import (
    _load_checkpoint,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.pareto_dp import (
    _build_pair_vote_sources,
    _geometry_iou,
    _keyframe_geometry,
    _make_feasible_anchors,
    _make_temporal7_anchors,
)
from experimental.polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    build_border_safety_constraints,
    expand_border_constraints,
    supported_single_component_segments,
    video_dimensions,
)


def _best_similarity(selected, candidates) -> float:
    geometry = _keyframe_geometry(selected)
    return max(
        (_geometry_iou(geometry, _keyframe_geometry(candidate)) for candidate in candidates),
        default=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--temporal-recall-quantile", type=float, default=0.97)
    args = parser.parse_args()

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
    selected = _load_checkpoint(args.checkpoint, supported)
    quality = {item: raw for item, raw in quality.items() if item[1] in supported}
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
    selected_by_segment = {
        (str(track_id), int(segment.segment_id)): segment
        for track_id, values in selected.items()
        for segment in values
    }
    records = []
    counts: Counter[str] = Counter()
    for track_id, values in supported.items():
        for segment in values:
            chosen = selected_by_segment[(str(track_id), int(segment.segment_id))]
            raw_by_frame = {
                frame: raw
                for (frame, candidate_track), raw in constraints.items()
                if candidate_track == track_id
                and segment.first_frame <= frame <= segment.last_frame
            }
            quality_by_frame = {
                frame: quality.get((frame, track_id), raw)
                for frame, raw in raw_by_frame.items()
            }
            pair_sources = _build_pair_vote_sources(
                sorted(raw_by_frame),
                quality_by_frame,
                point_count=23,
                max_edge_span_frames=30,
            )
            for keyframe in chosen.keyframes:
                frame = int(keyframe.frame)
                border = borders.get((frame, track_id))
                legacy = _make_feasible_anchors(
                    segment,
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
                    border_constraint=border,
                    visible_rectangle=visible,
                )
                similarities = {"legacy": _best_similarity(keyframe, legacy)}
                for window in ("short", "medium", "long"):
                    try:
                        candidates = _make_temporal7_anchors(
                            frame,
                            quality_by_frame,
                            raw_by_frame[frame],
                            quality_by_frame[frame],
                            recall_floor=0.97,
                            point_count=23,
                            max_anchor_scale=1.25,
                            window_radii=(2, 5, 10),
                            recall_quantile=args.temporal_recall_quantile,
                            border_constraint=border,
                            visible_rectangle=visible,
                            candidate_names=frozenset((f"{window}_recall",)),
                        )
                    except RuntimeError:
                        candidates = []
                    similarities[f"{window}_recall"] = _best_similarity(
                        keyframe, candidates
                    )
                best = max(similarities, key=similarities.get)
                exact = [
                    name for name, score in similarities.items() if score >= 0.999999
                ]
                family = "+".join(exact) if exact else f"nearest:{best}"
                counts[family] += 1
                records.append(
                    {
                        "frame": frame,
                        "track_id": str(track_id),
                        "segment_id": int(segment.segment_id),
                        "classification": family,
                        "best_family": best,
                        "best_similarity": similarities[best],
                        "similarities": similarities,
                    }
                )
    payload = {
        "privacy": "SQLite geometry only; no video pixels were opened.",
        "configuration": {
            **vars(args),
            "source_sqlite": str(args.source_sqlite),
            "checkpoint": str(args.checkpoint),
            "output_json": str(args.output_json),
        },
        "topology_fallbacks": topology_fallbacks,
        "expansion_preparation": expansion,
        "border_preparation": border_preparation,
        "selected_key_count": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "nearest_similarity": {
            "minimum": min(record["best_similarity"] for record in records),
            "below_0999": sum(
                record["best_similarity"] < 0.999 for record in records
            ),
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": payload["classification_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
