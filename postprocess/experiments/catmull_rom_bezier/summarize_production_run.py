"""Summarize one completed Production Catmull--Rom classwise run.

The engine emits one exact component row for every output component/frame.
This utility combines disjoint class shards without re-rasterizing masks and
therefore remains cheap enough for long-corpus validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize(root: Path, *, source_video_frames: int | None = None) -> dict[str, object]:
    source = Path(root).expanduser().resolve()
    classwise_path = source / "00_classwise_postprocess" / "classwise_manifest.json"
    classwise = json.loads(classwise_path.read_text(encoding="utf-8"))
    iou: list[float] = []
    recall: list[float] = []
    area: list[float] = []
    topology_invalid = 0
    component_rows = 0
    key_component_rows = 0
    point_rows: Counter[int] = Counter()
    point_tracks: dict[str, int] = {}
    engine_rows: list[dict[str, object]] = []
    quality_iou_floors: set[float] = set()
    quality_area_caps: set[float] = set()
    quality_rescue_inserted = 0
    quality_rescue_maximum_per_stream = 0
    quality_rescue_streams_above_legacy_192 = 0
    multi_component_streams: set[str] = set()
    metric_stream_ids: list[str] = []
    for group in classwise["groups"]:
        pipeline = json.loads(
            Path(str(group["pipeline_manifest"])).read_text(encoding="utf-8")
        )
        engine_path = None
        for stage in pipeline["stages"]:
            if stage["id"] == "curve_optimization":
                engine_path = Path(
                    str(stage["metadata"]["engine"]["manifest"])
                ).resolve()
                break
        if engine_path is None:
            raise RuntimeError(f"curve engine manifest missing for {group['id']}")
        engine = json.loads(engine_path.read_text(encoding="utf-8"))
        engine_rows.append(engine)
        runtime_config = engine["runtime_config"]
        quality_iou_floors.add(float(runtime_config["quality_rescue_iou_floor"]))
        quality_area_caps.add(float(runtime_config["quality_rescue_area_ratio_cap"]))
        stream_audit_path = Path(str(engine["stream_audit_jsonl"])).resolve()
        with stream_audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                stream = json.loads(line)
                stream_id = str(stream["stream_id"])
                if int(stream["components"]) > 1:
                    multi_component_streams.add(stream_id)
                inserted = int(
                    sum(
                        int(item.get("quality_rescue_inserted", 0))
                        for item in stream.get("optimization", [])
                    )
                )
                quality_rescue_inserted += inserted
                quality_rescue_maximum_per_stream = max(
                    quality_rescue_maximum_per_stream,
                    inserted,
                )
                quality_rescue_streams_above_legacy_192 += int(inserted > 192)
        metrics_path = Path(str(engine["component_metrics_csv"])).resolve()
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                component_rows += 1
                key_component_rows += int(row["is_keyframe"])
                iou.append(float(row["iou"]))
                recall.append(float(row["recall"]))
                area.append(float(row["area_ratio"]))
                topology_invalid += int(row["topology_valid"]) == 0
                points = int(row["points_per_component"])
                point_rows[points] += 1
                metric_stream_ids.append(str(row["stream_id"]))
                track_key = f"{row['label']}:{row['track_id']}"
                previous = point_tracks.setdefault(track_key, points)
                if previous != points:
                    raise RuntimeError(f"point count changed within {track_key}")
    ious = np.asarray(iou, dtype=np.float64)
    recalls = np.asarray(recall, dtype=np.float64)
    areas = np.asarray(area, dtype=np.float64)
    if len(quality_iou_floors) != 1 or len(quality_area_caps) != 1:
        raise RuntimeError("curve groups used inconsistent local-quality guards")
    quality_iou_floor = quality_iou_floors.pop()
    quality_area_cap = quality_area_caps.pop()
    quality_bad = np.logical_or(
        ious + 1e-12 < quality_iou_floor,
        areas > quality_area_cap + 1e-12,
    )
    multi_component = np.asarray(
        [stream_id in multi_component_streams for stream_id in metric_stream_ids],
        dtype=bool,
    )
    output_rows = int(sum(int(group["output_masks"]) for group in classwise["groups"]))
    key_rows = int(sum(int(engine["keyframe_rows"]) for engine in engine_rows))
    elapsed = float(classwise["elapsed_seconds"])
    video_frames = None if source_video_frames is None else int(source_video_frames)
    return {
        "schema_version": 1,
        "root": str(source),
        "geometry_mode": "catmull_rom",
        "cpu_only": True,
        "cuda_used": False,
        "target_interval": int(classwise["policy"]["default"]["keyframe_interval"]),
        "output_rows": output_rows,
        "keyframe_rows": key_rows,
        "effective_interval": float(output_rows / max(key_rows, 1)),
        "component_observations": int(component_rows),
        "key_component_observations": int(key_component_rows),
        "iou": {
            "mean": float(np.mean(ious)),
            "q01": float(np.quantile(ious, 0.01)),
            "q05": float(np.quantile(ious, 0.05)),
            "minimum": float(np.min(ious)),
        },
        "recall": {
            "minimum": float(np.min(recalls)),
            "violations_below_0_97": int(np.count_nonzero(recalls + 1e-12 < 0.97)),
        },
        "area_ratio": {
            "mean": float(np.mean(areas)),
            "q95": float(np.quantile(areas, 0.95)),
            "q99": float(np.quantile(areas, 0.99)),
            "maximum": float(np.max(areas)),
        },
        "topology_invalid_component_frames": int(topology_invalid),
        "local_quality_guard": {
            "iou_floor": float(quality_iou_floor),
            "area_ratio_cap": float(quality_area_cap),
            "violating_component_frames": int(np.count_nonzero(quality_bad)),
            "single_component_violating_frames": int(
                np.count_nonzero(np.logical_and(quality_bad, ~multi_component))
            ),
            "multi_component_violating_frames": int(
                np.count_nonzero(np.logical_and(quality_bad, multi_component))
            ),
        },
        "point_count": {
            "tracks": dict(sorted(Counter(point_tracks.values()).items())),
            "component_frames": dict(sorted(point_rows.items())),
        },
        "fallbacks": {
            "multi_component_key_every_frame_streams": int(
                sum(
                    int(engine["multi_component_key_every_frame_streams"])
                    for engine in engine_rows
                )
            ),
            "emergency_spatial_repairs": int(
                sum(int(engine["emergency_spatial_repairs"]) for engine in engine_rows)
            ),
            "independent_local_refit_attempts": int(
                sum(
                    int(engine.get("independent_local_refit_attempts", 0))
                    for engine in engine_rows
                )
            ),
            "independent_local_refit_accepted": int(
                sum(
                    int(engine.get("independent_local_refit_accepted", 0))
                    for engine in engine_rows
                )
            ),
            "maximum_independent_local_refit_shift_px": float(
                max(
                    float(
                        engine.get("maximum_independent_local_refit_shift_px", 0.0)
                    )
                    for engine in engine_rows
                )
            ),
            "completion_envelope_frames": int(
                sum(int(engine["completion_envelope_frames"]) for engine in engine_rows)
            ),
            "maximum_completion_area_ratio": float(
                max(float(engine["maximum_completion_area_ratio"]) for engine in engine_rows)
            ),
        },
        "optimization": {
            "quality_rescue_inserted_keys": int(quality_rescue_inserted),
            "quality_rescue_maximum_per_stream": int(
                quality_rescue_maximum_per_stream
            ),
            "quality_rescue_streams_above_legacy_192_cap": int(
                quality_rescue_streams_above_legacy_192
            ),
            "state_search_fallbacks": int(
                sum(int(engine["audit"]["state_search_fallbacks"]) for engine in engine_rows)
            ),
            "refinement_guard_triggers": int(
                sum(
                    int(engine["audit"]["refinement_guard_triggers"])
                    for engine in engine_rows
                )
            ),
            "refinement_guard_minimum_alpha": float(
                min(
                    float(engine["audit"]["refinement_guard_minimum_alpha"])
                    for engine in engine_rows
                )
            ),
            "dp_seconds_sum": float(
                sum(float(engine["audit"]["dp_seconds"]) for engine in engine_rows)
            ),
            "pair_vote_seconds_sum": float(
                sum(
                    float(engine["audit"]["pair_vote_seconds"])
                    for engine in engine_rows
                )
            ),
            "point_refine_seconds_sum": float(
                sum(
                    float(engine["audit"]["point_refine_seconds"])
                    for engine in engine_rows
                )
            ),
            "final_audit_seconds_sum": float(
                sum(
                    float(engine["audit"]["final_audit_seconds"])
                    for engine in engine_rows
                )
            ),
        },
        "runtime": {
            "classwise_wall_seconds": elapsed,
            "output_mask_fps": float(output_rows / max(elapsed, 1e-12)),
            "source_video_frames": video_frames,
            "source_video_fps": (
                None
                if video_frames is None
                else float(video_frames / max(elapsed, 1e-12))
            ),
            "maximum_worker_rss_kib": int(
                max(int(engine["maximum_rss_kib"]) for engine in engine_rows)
            ),
            "maximum_phase_history_frames": int(
                max(int(engine["maximum_phase_history_frames"]) for engine in engine_rows)
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-video-frames", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize(
        args.run_root,
        source_video_frames=args.source_video_frames,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
