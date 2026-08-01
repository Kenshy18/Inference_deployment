"""Benchmark simplified polygon stages against original/restored v22."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .compare_original_postprocess import (
    DEFAULT_ORIGINAL_ROOT,
    DEFAULT_PYTHON,
    POSTPROCESS_ROOT,
    _json_dump,
    _run,
    compare_mask_sqlites,
    extract_real_fixture,
    run_polygon_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tracked-sqlite", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--frames-per-track", type=int, default=300)
    parser.add_argument("--tracks", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    if root.exists():
        if not args.force:
            raise FileExistsError(root)
        shutil.rmtree(root)
    root.mkdir(parents=True)
    fixture_info = extract_real_fixture(
        args.source_tracked_sqlite.expanduser().resolve(),
        root / "fixture" / "tracked.sqlite",
        frames_per_track=args.frames_per_track,
        track_limit=args.tracks,
    )
    fixture = Path(fixture_info["path"])
    v22 = run_polygon_comparison(
        python=args.python.expanduser().resolve(),
        original_root=args.original_root.expanduser().resolve(),
        fixture=fixture,
        work_dir=root / "v22",
        device=args.device,
    )
    simple_config = root / "simplified_pipeline.json"
    _json_dump(
        simple_config,
        {
            "name": "simplified_rdp_baseline",
            "stages": [
                {
                    "id": "polygon_approximation",
                    "implementation": "approximation.polygon.rdp",
                    "options": {"epsilon_ratio": 0.01, "minimum_epsilon_px": 0.5},
                },
                {
                    "id": "keyframe_selection",
                    "implementation": "keyframes.polygon.interval",
                    "options": {"interval_frames": 3},
                },
                {
                    "id": "mask_gap_fill",
                    "implementation": "gap_fill.polygon.linear",
                    "options": {"minimum_points": 8, "max_gap": 30},
                },
                {"id": "exact_evaluation", "implementation": "evaluation.mask_iou"},
                {"id": "output_validation", "implementation": "artifacts.validate"},
            ],
        },
    )
    simple_root = root / "simplified"
    simple_seconds = _run(
        [
            str(args.python.expanduser().resolve()),
            "run_pipeline.py",
            "--input-sqlite",
            str(fixture),
            "--output-dir",
            str(simple_root),
            "--pipeline-config",
            str(simple_config),
        ],
        cwd=POSTPROCESS_ROOT,
        log_path=root / "simplified.log",
    )
    simple_manifest = json.loads(
        (simple_root / "pipeline_manifest.json").read_text(encoding="utf-8")
    )
    simple_predictions = Path(simple_manifest["artifacts"]["predictions_sqlite"])
    original_predictions = Path(v22["artifacts"]["original_predictions"])
    report = {
        "fixture": fixture_info,
        "row_count": sum(track["rows"] for track in fixture_info["tracks"]),
        "seconds": {
            "simplified": simple_seconds,
            "original_v22": v22["original_seconds"],
            "restored_v22": v22["current_seconds"],
        },
        "simplified_vs_input": compare_mask_sqlites(fixture, simple_predictions),
        "original_v22_vs_input": v22["original_vs_input"],
        "restored_v22_vs_input": v22["current_vs_input"],
        "simplified_vs_original_v22": compare_mask_sqlites(
            original_predictions, simple_predictions
        ),
        "restored_vs_original_v22": v22["current_vs_original"],
        "keyframes": v22["keyframes"],
        "artifacts": {
            "fixture": str(fixture),
            "simplified": str(simple_predictions),
            **v22["artifacts"],
        },
    }
    _json_dump(root / "benchmark.json", report)
    simple_vertices = report["simplified_vs_original_v22"][
        "prediction_vertices_per_row"
    ]
    restored_vertices = report["restored_vs_original_v22"][
        "prediction_vertices_per_row"
    ]
    lines = [
        "# Polygon restoration benchmark",
        "",
        f"Rows: `{report['row_count']}`",
        "",
        "| Implementation | Seconds | Input recall | Input IoU | Mean vertices/row | Median vertices/row |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Simplified RDP | {simple_seconds:.3f} | {report['simplified_vs_input']['global_recall']:.6f} | {report['simplified_vs_input']['global_iou']:.6f} | {simple_vertices['mean']:.3f} | {simple_vertices['median']:.3f} |",
        f"| Original v22 | {v22['original_seconds']:.3f} | {v22['original_vs_input']['global_recall']:.6f} | {v22['original_vs_input']['global_iou']:.6f} | {restored_vertices['mean']:.3f} | {restored_vertices['median']:.3f} |",
        f"| Restored v22 | {v22['current_seconds']:.3f} | {v22['current_vs_input']['global_recall']:.6f} | {v22['current_vs_input']['global_iou']:.6f} | {restored_vertices['mean']:.3f} | {restored_vertices['median']:.3f} |",
        "",
        f"Restored/original IoU: `{v22['current_vs_original']['global_iou']:.9f}`.",
        f"Restored/original keyframe Jaccard: `{v22['keyframes']['jaccard']:.9f}`.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
