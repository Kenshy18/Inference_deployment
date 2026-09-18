#!/usr/bin/env python3
"""Reconcile fixed and adaptive Production runtime benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]

CASES = {
    "white_axel": (
        ROOT
        / "output/runtime_floor_stability_20260901/white_axel_baseline/datasets/v3_white_axel/result.json",
        ROOT
        / "output/runtime_floor_stability_20260901/full_pipeline_adaptive/datasets/v3_white_axel/result.json",
    ),
    "white0210": (
        ROOT
        / "output/runtime_speed_raw_inference_corpus_20260831/datasets/v3_white0210/result.json",
        ROOT
        / "output/runtime_floor_stability_20260901/production_default_long_validation/datasets/v3_white0210/result.json",
    ),
    "kpi": (
        ROOT
        / "output/runtime_speed_raw_inference_corpus_20260831/datasets/v3_kpi/result.json",
        ROOT
        / "output/runtime_floor_stability_20260901/production_default_balanced_validation/datasets/v3_kpi/result.json",
    ),
    "compact_kpi_smoke": (
        ROOT
        / "output/runtime_speed_raw_inference_corpus_20260831/datasets/lite_kpi_2400/result.json",
        ROOT
        / "output/runtime_floor_stability_20260901/production_default_smoke/datasets/lite_kpi_2400/result.json",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_identity(profile: dict[str, object]) -> dict[str, object]:
    return {
        key: profile.get(key)
        for key in (
            "source",
            "frames",
            "first_frame",
            "last_frame",
            "width",
            "height",
            "fps",
            "segmentation_detections",
            "polygon_points",
            "stored_cuts",
        )
    }


def _distribution(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "minimum": min(values),
        "maximum": max(values),
        "maximum_to_minimum": max(values) / min(values),
        "mean": mean,
        "population_standard_deviation": statistics.pstdev(values),
        "population_coefficient_of_variation": statistics.pstdev(values) / mean,
    }


def main() -> int:
    args = _parser().parse_args()
    rows: list[dict[str, object]] = []
    for name, (old_path, new_path) in CASES.items():
        old = _load(old_path)
        new = _load(new_path)
        old_profile = _profile_identity(dict(old["profile"]))
        new_profile = _profile_identity(dict(new["profile"]))
        old_fps = float(old["pipeline_timeline_fps"])
        new_fps = float(new["pipeline_timeline_fps"])
        old_geometry_fps = float(old["geometry_timeline_fps"])
        new_geometry_fps = float(new["geometry_timeline_fps"])
        rows.append(
            {
                "case": name,
                "cohort": old.get("cohort"),
                "old_result": str(old_path),
                "new_result": str(new_path),
                "profile_identity_equal": old_profile == new_profile,
                "profile_identity": new_profile,
                "quality_exact_equal": old.get("quality") == new.get("quality"),
                "old_pipeline_fps": old_fps,
                "new_pipeline_fps": new_fps,
                "pipeline_speedup": new_fps / old_fps,
                "old_polygon_fps": old_geometry_fps,
                "new_polygon_fps": new_geometry_fps,
                "polygon_speedup": new_geometry_fps / old_geometry_fps,
                "new_peak_process_tree_rss_mib": new.get("peak_process_tree_rss_mib"),
                "new_minimum_system_available_memory_mib": new.get(
                    "minimum_system_available_memory_mib"
                ),
                "quality": new.get("quality"),
            }
        )

    primary = [row for row in rows if row["case"] != "compact_kpi_smoke"]
    old_values = [float(row["old_pipeline_fps"]) for row in primary]
    new_values = [float(row["new_pipeline_fps"]) for row in primary]
    payload = {
        "schema_version": 1,
        "metric": "full postprocess timeline frames / pipeline wall seconds",
        "primary_population": [str(row["case"]) for row in primary],
        "supplemental_population": ["compact_kpi_smoke"],
        "cases": rows,
        "primary_distribution": {
            "old": _distribution(old_values),
            "adaptive": _distribution(new_values),
            "floor_gain": min(new_values) / min(old_values) - 1.0,
            "mean_gain": statistics.fmean(new_values) / statistics.fmean(old_values)
            - 1.0,
        },
        "validation": {
            "all_source_profiles_equal": all(
                bool(row["profile_identity_equal"]) for row in rows
            ),
            "all_recorded_quality_exact_equal": all(
                bool(row["quality_exact_equal"]) for row in rows
            ),
            "all_recall_violations_zero": all(
                int(dict(row["quality"])["recall_violations"]) == 0 for row in rows
            ),
        },
        "caveats": [
            "The three primary cases are deliberately stratified, not a random sample of every possible video.",
            "Full-pipeline old/new runs were not all interleaved; controlled phase-2 exact-output comparisons separately validate the scheduling effect.",
            "The eight-process memory cap was screened on the 30-GiB WSL deployment host and is not evidence for a larger process budget.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
