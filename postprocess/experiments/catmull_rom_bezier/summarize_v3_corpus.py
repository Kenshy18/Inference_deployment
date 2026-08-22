"""Create one exact, source-backed report for a completed V3 curve corpus."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.catmull_rom_bezier.summarize_production_run import summarize


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _engine_manifests(run_root: Path) -> tuple[dict[str, object], ...]:
    classwise = json.loads(
        (
            run_root / "00_classwise_postprocess" / "classwise_manifest.json"
        ).read_text(encoding="utf-8")
    )
    engines: list[dict[str, object]] = []
    for group in classwise["groups"]:
        pipeline = json.loads(
            Path(str(group["pipeline_manifest"])).read_text(encoding="utf-8")
        )
        curve_stage = next(
            stage for stage in pipeline["stages"] if stage["id"] == "curve_optimization"
        )
        engine_path = Path(str(curve_stage["metadata"]["engine"]["manifest"]))
        engines.append(json.loads(engine_path.read_text(encoding="utf-8")))
    return tuple(engines)


def summarize_corpus(batch_manifest: Path) -> dict[str, object]:
    batch_path = Path(batch_manifest).expanduser().resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    output_root = batch_path.parent
    run_summaries: list[dict[str, object]] = []
    iou: list[float] = []
    recall: list[float] = []
    area: list[float] = []
    points_by_tracks: Counter[int] = Counter()
    points_by_component_frames: Counter[int] = Counter()
    for row in batch["runs"]:
        run_root = output_root / str(row["run_id"])
        if not (run_root / "pipeline_manifest.json").is_file():
            raise RuntimeError(f"incomplete curve run: {run_root}")
        summary = summarize(
            run_root,
            source_video_frames=int(row["source_video_frames"]),
        )
        _write_json(run_root / "curve_summary.json", summary)
        run_summaries.append({"run_id": str(row["run_id"]), **summary})
        points_by_tracks.update(
            {
                int(points): int(count)
                for points, count in summary["point_count"]["tracks"].items()
            }
        )
        points_by_component_frames.update(
            {
                int(points): int(count)
                for points, count in summary["point_count"][
                    "component_frames"
                ].items()
            }
        )
        for engine in _engine_manifests(run_root):
            metrics_path = Path(str(engine["component_metrics_csv"]))
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for metric in csv.DictReader(handle):
                    iou.append(float(metric["iou"]))
                    recall.append(float(metric["recall"]))
                    area.append(float(metric["area_ratio"]))
    ious = np.asarray(iou, dtype=np.float64)
    recalls = np.asarray(recall, dtype=np.float64)
    areas = np.asarray(area, dtype=np.float64)
    quality_iou_floors = {
        float(row["local_quality_guard"]["iou_floor"]) for row in run_summaries
    }
    quality_area_caps = {
        float(row["local_quality_guard"]["area_ratio_cap"])
        for row in run_summaries
    }
    if len(quality_iou_floors) != 1 or len(quality_area_caps) != 1:
        raise RuntimeError("V3 runs used inconsistent local-quality guards")
    quality_iou_floor = quality_iou_floors.pop()
    quality_area_cap = quality_area_caps.pop()
    total_wall = float(
        sum(float(row["runtime"]["classwise_wall_seconds"]) for row in run_summaries)
    )
    total_output_rows = int(sum(int(row["output_rows"]) for row in run_summaries))
    total_source_frames = int(
        sum(int(row["runtime"]["source_video_frames"]) for row in run_summaries)
    )
    total_keys = int(sum(int(row["keyframe_rows"]) for row in run_summaries))
    recall_violations = int(np.count_nonzero(recalls + 1e-12 < 0.97))
    topology_invalid = int(
        sum(int(row["topology_invalid_component_frames"]) for row in run_summaries)
    )
    local_quality_bad = np.logical_or(
        ious + 1e-12 < quality_iou_floor,
        areas > quality_area_cap + 1e-12,
    )
    return {
        "schema_version": 1,
        "status": "production_validation",
        "geometry_mode": "catmull_rom",
        "algorithm": "production_catmull_rom_cpu_exact_v1",
        "cpu_only": True,
        "cuda_used": False,
        "batch_manifest": str(batch_path),
        "target_interval": int(batch["target_interval"]),
        "runs": int(len(run_summaries)),
        "source_video_frames": total_source_frames,
        "output_rows": total_output_rows,
        "component_observations": int(len(ious)),
        "keyframe_rows": total_keys,
        "effective_interval": float(total_output_rows / max(total_keys, 1)),
        "iou": {
            "mean": float(np.mean(ious)),
            "q01": float(np.quantile(ious, 0.01)),
            "q05": float(np.quantile(ious, 0.05)),
            "minimum": float(np.min(ious)),
        },
        "recall": {
            "minimum": float(np.min(recalls)),
            "violations_below_0_97": recall_violations,
        },
        "area_ratio": {
            "mean": float(np.mean(areas)),
            "q95": float(np.quantile(areas, 0.95)),
            "q99": float(np.quantile(areas, 0.99)),
            "maximum": float(np.max(areas)),
        },
        "local_quality_guard": {
            "iou_floor": float(quality_iou_floor),
            "area_ratio_cap": float(quality_area_cap),
            "violating_component_frames": int(np.count_nonzero(local_quality_bad)),
            "single_component_violating_frames": int(
                sum(
                    int(row["local_quality_guard"][
                        "single_component_violating_frames"
                    ])
                    for row in run_summaries
                )
            ),
            "multi_component_violating_frames": int(
                sum(
                    int(row["local_quality_guard"][
                        "multi_component_violating_frames"
                    ])
                    for row in run_summaries
                )
            ),
        },
        "topology_invalid_component_frames": topology_invalid,
        "point_count": {
            "tracks": dict(sorted(points_by_tracks.items())),
            "component_frames": dict(sorted(points_by_component_frames.items())),
        },
        "fallbacks": {
            "multi_component_key_every_frame_streams": int(
                sum(
                    int(row["fallbacks"]["multi_component_key_every_frame_streams"])
                    for row in run_summaries
                )
            ),
            "emergency_spatial_repairs": int(
                sum(
                    int(row["fallbacks"]["emergency_spatial_repairs"])
                    for row in run_summaries
                )
            ),
            "independent_local_refit_attempts": int(
                sum(
                    int(row["fallbacks"]["independent_local_refit_attempts"])
                    for row in run_summaries
                )
            ),
            "independent_local_refit_accepted": int(
                sum(
                    int(row["fallbacks"]["independent_local_refit_accepted"])
                    for row in run_summaries
                )
            ),
            "maximum_independent_local_refit_shift_px": float(
                max(
                    float(
                        row["fallbacks"][
                            "maximum_independent_local_refit_shift_px"
                        ]
                    )
                    for row in run_summaries
                )
            ),
            "completion_envelope_frames": int(
                sum(
                    int(row["fallbacks"]["completion_envelope_frames"])
                    for row in run_summaries
                )
            ),
        },
        "optimization": {
            "quality_rescue_inserted_keys": int(
                sum(
                    int(row["optimization"]["quality_rescue_inserted_keys"])
                    for row in run_summaries
                )
            ),
            "quality_rescue_maximum_per_stream": int(
                max(
                    int(row["optimization"]["quality_rescue_maximum_per_stream"])
                    for row in run_summaries
                )
            ),
            "quality_rescue_streams_above_legacy_192_cap": int(
                sum(
                    int(row["optimization"][
                        "quality_rescue_streams_above_legacy_192_cap"
                    ])
                    for row in run_summaries
                )
            ),
        },
        "runtime": {
            "summed_classwise_wall_seconds": total_wall,
            "output_mask_fps": float(total_output_rows / max(total_wall, 1e-12)),
            "source_video_fps": float(total_source_frames / max(total_wall, 1e-12)),
            "maximum_worker_rss_kib": int(
                max(int(row["runtime"]["maximum_worker_rss_kib"]) for row in run_summaries)
            ),
            "maximum_phase_history_frames": int(
                max(
                    int(row["runtime"]["maximum_phase_history_frames"])
                    for row in run_summaries
                )
            ),
        },
        "hard_gates_passed": bool(recall_violations == 0 and topology_invalid == 0),
        "production_acceptance_passed": bool(
            recall_violations == 0
            and topology_invalid == 0
            and not np.any(local_quality_bad)
        ),
        "reference_semantics": "tracked_AI_masks_not_human_ground_truth",
        "per_run": run_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize_corpus(args.batch_manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("summarize_corpus",)
