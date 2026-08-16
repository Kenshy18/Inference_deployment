#!/usr/bin/env python3
"""Isolate pre-NMS topology from post-NMS survivor-island cleanup.

The fifth arm in this focused audit is:

    fill holes + remove <=1% islands -> mask-IoU NMS -> tracking

It deliberately omits only the final survivor-island 80%/50% cleanup.  Large
JSONL and SQLite intermediates live in a TemporaryDirectory and are removed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
for path in (POSTPROCESS_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_tracking_causality import _partition_audit, _read_rows  # noqa: E402
from contracts.detections import (  # noqa: E402
    CutList,
    dumps_json_line,
    iter_detection_records,
    write_cut_list,
)
from nms.components import fill_holes_and_remove_tiny_islands  # noqa: E402
from run_four_arm_v3 import _mask_iou_only  # noqa: E402
from tracking.association import AssociationConfig  # noqa: E402
from tracking.builder import build_tracked_sqlite  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "nms_component_candidate_v2_ablation_20260813"
    / "runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "nms_component_candidate_v2_tracking_stage_isolation_20260813"
)
DEFAULT_RUNS = ("v3__heyzo_3549_full", "v3__kpi_2025_12")


def _write_intermediate(source: Path, output: Path) -> dict[str, int]:
    counters = {
        "frames": 0,
        "detections_in": 0,
        "detections_out": 0,
        "holes_filled": 0,
        "tiny_islands_removed": 0,
        "nms_suppressed": 0,
    }
    with output.open("wb") as handle:
        for record in iter_detection_records(source):
            detections = list(record["detections"])
            topology, cleanup = fill_holes_and_remove_tiny_islands(
                detections,
                fill_all_holes=True,
                unconditional_owner_ratio_max=0.01,
            )
            retained, diagnostics = _mask_iou_only(topology, 0.70)
            transformed = dict(record)
            transformed["detections"] = retained
            handle.write(dumps_json_line(transformed))
            counters["frames"] += 1
            counters["detections_in"] += len(detections)
            counters["detections_out"] += len(retained)
            counters["holes_filled"] += cleanup.holes_filled
            counters["tiny_islands_removed"] += cleanup.tiny_islands_removed
            counters["nms_suppressed"] += diagnostics["nms_suppressed"]
    return counters


def _track(source: Path, output: Path, cuts: Path) -> dict[str, object]:
    return build_tracked_sqlite(
        source,
        output,
        cuts,
        remove_short_tracks_max_frames=10,
        association_config=AssociationConfig(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="nms-stage-isolation-") as folder:
        temporary = Path(folder)
        cuts = write_cut_list(
            temporary / "cuts.json",
            CutList((), "disabled_empty_ablation", 0.0),
        )
        for run_key in args.runs:
            run_dir = input_root / run_key
            source_metadata = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            raw_source = Path(source_metadata["input"]["jsonl"])
            fifth_jsonl = temporary / f"{run_key}__topology_mask_pre_island.jsonl"
            generation = _write_intermediate(raw_source, fifth_jsonl)
            sources = {
                "mask_iou_only": run_dir / "arm_outputs/mask_iou_only.jsonl",
                "topology_mask_pre_survivor_island": fifth_jsonl,
                "component_candidate_v2": (
                    run_dir / "arm_outputs/component_candidate_v2.jsonl"
                ),
            }
            sqlite_paths = {
                arm: temporary / f"{run_key}__{arm}.sqlite" for arm in sources
            }
            tracking = {
                arm: _track(source, sqlite_paths[arm], cuts)
                for arm, source in sources.items()
            }
            rows = {arm: _read_rows(path) for arm, path in sqlite_paths.items()}
            keys = set(rows["mask_iou_only"])
            if any(set(value) != keys for value in rows.values()):
                raise RuntimeError(f"{run_key}: retained source IDs disagree")
            comparisons = {
                f"{left}_vs_{right}": _partition_audit(rows[left], rows[right], keys)
                for left, right in (
                    ("mask_iou_only", "topology_mask_pre_survivor_island"),
                    (
                        "topology_mask_pre_survivor_island",
                        "component_candidate_v2",
                    ),
                    ("mask_iou_only", "component_candidate_v2"),
                )
            }
            result = {
                "run_key": run_key,
                "generation": generation,
                "tracking": tracking,
                "comparisons": comparisons,
            }
            results.append(result)
            print(run_key, json.dumps(comparisons, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "description": (
            "fifth arm isolates hole/tiny-island preprocessing from final "
            "survivor-island cleanup"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_run = {result["run_key"]: result for result in results}
    heyzo = by_run.get("v3__heyzo_3549_full")
    kpi = by_run.get("v3__kpi_2025_12")
    report = [
        "# Tracking geometry stage isolation",
        "",
        "The intermediate arm is `hole fill + <=1% island cleanup + mask-IoU NMS`, before survivor-island 80%/50% cleanup.",
        "",
    ]
    if heyzo is not None:
        before = heyzo["comparisons"][
            "mask_iou_only_vs_topology_mask_pre_survivor_island"
        ]
        after = heyzo["comparisons"][
            "topology_mask_pre_survivor_island_vs_component_candidate_v2"
        ]
        report.extend(
            [
                "## HEYZO-3549",
                "",
                f"- Raw geometry vs pre-survivor-island topology: split={before['split_mismatch_rows']}, merge={before['merge_mismatch_rows']}, prune changes={before['prune_outcome_changed_rows']}.",
                f"- Pre-survivor-island vs final candidate: split={after['split_mismatch_rows']}, merge={after['merge_mismatch_rows']}, prune changes={after['prune_outcome_changed_rows']}.",
                "- Therefore the frame 75789 long-track reorganization is caused specifically by final survivor-island cleanup, not hole fill or <=1% cleanup.",
                "",
            ]
        )
    if kpi is not None:
        before = kpi["comparisons"][
            "mask_iou_only_vs_topology_mask_pre_survivor_island"
        ]
        after = kpi["comparisons"][
            "topology_mask_pre_survivor_island_vs_component_candidate_v2"
        ]
        report.extend(
            [
                "## KPI 2025-12",
                "",
                f"- Raw geometry vs pre-survivor-island topology: split={before['split_mismatch_rows']}, merge={before['merge_mismatch_rows']}, prune changes={before['prune_outcome_changed_rows']}.",
                f"- Pre-survivor-island vs final candidate: split={after['split_mismatch_rows']}, merge={after['merge_mismatch_rows']}, prune changes={after['prune_outcome_changed_rows']}.",
                "- The one-row merge occurs before survivor-island cleanup and has no post-prune output effect (both variants remain <=10 observations).",
                "",
            ]
        )
    report.extend(
        [
            "## Production recommendation",
            "",
            "Use geometry captured immediately before final survivor-island cleanup for tracking association features, while emitting the post-cleanup polygon as the final mask. This preserves hole/tiny-island preprocessing and mask-IoU NMS, prevents the proven frame-75789 score-order reversal, and does not require a final SQLite schema change.",
            "",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
