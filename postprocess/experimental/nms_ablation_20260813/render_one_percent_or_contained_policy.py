#!/usr/bin/env python3
"""Render the V3 island policy requested on 2026-08-13.

Delete a secondary foreground component when either:

* its area is at most 1% of the owner's largest foreground component; or
* its area is at most 5%, it does not share a screen edge with the owner's
  main component, and another raw instance completely covers it.

The renderer is an audit utility only.  It never modifies source SQLite files.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from render_conservative_island_policy_audit import (
    DEFAULT_TOPOLOGY,
    OUTPUT as CONSERVATIVE_OUTPUT,
    PALETTE,
    ROOT,
    contact_sheet_pages,
    draw_panel,
    load_groups,
    open_ro,
    removal_sets,
    seek_frame,
)


OUTPUT = ROOT / "output/small_island_policy_1pct_or_covered90_v3_20260813"
CAUSAL_ANALYSIS = (
    ROOT / "output/small_island_nms_causal_analysis_v3_20260813/analysis.json"
)
UNCONDITIONAL_RATIO_MAX = 0.01
CONDITIONAL_RATIO_MAX = 0.05
CONTAINMENT_MIN = 0.90


def maximum_other_instance_coverage(item: dict[str, object]) -> float:
    return max(
        (
            float(overlap.get("candidate_coverage", 0.0))
            for overlap in list(item.get("overlaps", []))
        ),
        default=0.0,
    )


def classify(item: dict[str, object]) -> tuple[str, str, float]:
    ratio = float(item["ratio"])
    coverage = maximum_other_instance_coverage(item)
    if ratio <= UNCONDITIONAL_RATIO_MAX:
        return "deleted", "ratio_at_most_1_percent", coverage
    if (
        ratio <= CONDITIONAL_RATIO_MAX
        and not list(item.get("common_edge", []))
        and coverage >= CONTAINMENT_MIN
    ):
        return "deleted", "contained_non_edge_at_most_5_percent", coverage
    if list(item.get("common_edge", [])):
        return "kept", "shared_screen_edge", coverage
    if ratio > CONDITIONAL_RATIO_MAX:
        return "kept", "ratio_above_5_percent", coverage
    return "kept", "not_fully_contained", coverage


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folders = {
        "deleted": OUTPUT / "01_deleted",
        "kept": OUTPUT / "02_kept",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.jpg"):
            stale.unlink()
    for stale in OUTPUT.glob("contact_sheet_*.jpg"):
        stale.unlink()

    causal = json.loads(CAUSAL_ANALYSIS.read_text(encoding="utf-8"))
    metadata_rows = json.loads(
        (CONSERVATIVE_OUTPUT / "manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        (
            str(row["run_key"]),
            int(row["frame"]),
            int(row["detection_id"]),
            int(row["polygon_index"]),
        ): row
        for row in metadata_rows
    }

    classified: list[dict[str, object]] = []
    for row in causal:
        key = (
            str(row["run"]),
            int(row["frame"]),
            int(row["did"]),
            int(row["pidx"]),
        )
        source = metadata[key]
        action, reason, coverage = classify(row)
        classified.append(
            {
                **source,
                "action": action,
                "reason": reason,
                "maximum_other_instance_coverage": coverage,
                "common_edges": list(row.get("common_edge", [])),
            }
        )

    action_map = {
        (
            str(item["run_key"]),
            int(item["detection_id"]),
            int(item["polygon_index"]),
        ): str(item["action"])
        for item in classified
    }
    topology = open_ro(DEFAULT_TOPOLOGY)
    counters: Counter[str] = Counter()
    manifest: list[dict[str, object]] = []

    for item in sorted(
        classified,
        key=lambda value: (
            str(value["action"]),
            str(value["reason"]),
            float(value["ratio"]),
            str(value["run_key"]),
            int(value["frame"]),
        ),
    ):
        run_key = str(item["run_key"])
        frame = int(item["frame"])
        owner_id = int(item["detection_id"])
        polygon_index = int(item["polygon_index"])
        detection_ids = [
            int(row["detection_id"])
            for row in topology.execute(
                """SELECT detection_id FROM mask_topology
                   WHERE run_key=? AND frame=?
                   ORDER BY detection_index,detection_id""",
                (run_key, frame),
            )
        ]
        instances = {
            detection_id: load_groups(
                Path(str(item["inference_sqlite"])),
                DEFAULT_TOPOLOGY,
                run_key,
                detection_id,
            )
            for detection_id in detection_ids
        }
        colors = {
            detection_id: PALETTE[index % len(PALETTE)]
            for index, detection_id in enumerate(detection_ids)
        }
        no_removals = {detection_id: set() for detection_id in detection_ids}
        removals = removal_sets(instances, action_map, run_key)
        image = seek_frame(Path(str(item["input_video"])), frame)
        ratio_percent = float(item["ratio"]) * 100.0
        coverage_percent = float(item["maximum_other_instance_coverage"]) * 100.0
        detail = (
            f"run={run_key} frame={frame} D{owner_id} P{polygon_index} "
            f"ratio={ratio_percent:.3f}% other_cover={coverage_percent:.2f}% "
            f"shared_edge={item['common_edges'] or 'none'}"
        )
        before = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            polygon_index,
            no_removals,
            title="BEFORE: all raw instances; owner=WHITE, island=YELLOW",
            detail=detail,
        )
        after = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            polygon_index,
            removals,
            title=(
                f"AFTER: {str(item['action']).upper()} / "
                f"{str(item['reason']).replace('_', ' ')}"
            ),
            detail=(
                "DELETE when ratio<=1%, OR ratio<=5% + no shared edge + "
                "other instance coverage>=90%"
            ),
        )
        comparison = np.concatenate([before, after], axis=1)
        action = str(item["action"])
        counters[action] += 1
        filename = (
            f"sample_{counters[action]:03d}_{item['video_slug']}_f{frame}_"
            f"D{owner_id}_P{polygon_index}_ratio_{ratio_percent:.3f}pct_"
            f"cover_{coverage_percent:.2f}pct.jpg"
        )
        path = folders[action] / filename
        cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append(
            {
                **item,
                "ratio_percent": ratio_percent,
                "coverage_percent": coverage_percent,
                "frame_detection_ids": detection_ids,
                "path": str(path),
            }
        )

    sheets = {
        action: contact_sheet_pages(
            sorted(folder.glob("*.jpg")), OUTPUT, action
        )
        for action, folder in folders.items()
    }
    topology.close()

    summary = {
        "total_components": len(classified),
        "actions": dict(Counter(str(item["action"]) for item in classified)),
        "reasons": dict(Counter(str(item["reason"]) for item in classified)),
        "parameters": {
            "unconditional_ratio_max": UNCONDITIONAL_RATIO_MAX,
            "conditional_ratio_max": CONDITIONAL_RATIO_MAX,
            "containment_min": CONTAINMENT_MIN,
            "edge_rule": "protect only when island and owner main share an edge",
        },
        "contact_sheets": sheets,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "action",
            "reason",
            "run_key",
            "video_slug",
            "frame",
            "detection_id",
            "polygon_index",
            "ratio_percent",
            "coverage_percent",
            "common_edges",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
