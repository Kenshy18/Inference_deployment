#!/usr/bin/env python3
"""Render every V3 raw-mask hole before and after filling that hole.

This is a read-only audit utility. Source videos and SQLite files are never
modified. Each output uses a full video frame and assigns a distinct color to
every raw instance in that frame.
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
    PALETTE,
    ROOT,
    contact_sheet_pages,
    draw_panel,
    load_groups,
    open_ro,
    seek_frame,
)


OUTPUT = ROOT / "output/hole_fill_audit_v3_20260813"


def ratio_folder(ratio: float) -> str:
    if ratio <= 0.01:
        return "01_at_most_1pct"
    if ratio <= 0.03:
        return "02_1_to_3pct"
    if ratio <= 0.05:
        return "03_3_to_5pct"
    return "04_above_5pct"


def descendants(
    groups: list[dict[str, object]], root: int
) -> set[int]:
    parents = {
        int(group["polygon_index"]): group["parent"] for group in groups
    }
    removed: set[int] = set()
    for group in groups:
        index = int(group["polygon_index"])
        current: int | None = index
        seen: set[int] = set()
        while current is not None and current not in seen:
            if current == root:
                removed.add(index)
                break
            seen.add(current)
            parent = parents.get(current)
            current = None if parent is None else int(parent)
    return removed


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folders = {
        key: OUTPUT / key
        for key in (
            "01_at_most_1pct",
            "02_1_to_3pct",
            "03_3_to_5pct",
            "04_above_5pct",
        )
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.jpg"):
            stale.unlink()
    for stale in OUTPUT.glob("contact_sheet_*.jpg"):
        stale.unlink()

    topology = open_ro(DEFAULT_TOPOLOGY)
    holes = [
        dict(row)
        for row in topology.execute(
            """SELECT r.run_key,r.video_slug,r.input_video,r.inference_sqlite,
                      m.frame,c.detection_id,c.polygon_index,c.absolute_area,
                      c.parent_polygon_index,p.absolute_area AS parent_area,
                      c.absolute_area/p.absolute_area AS ratio
               FROM contour_topology c
               JOIN contour_topology p
                 ON p.run_key=c.run_key
                AND p.detection_id=c.detection_id
                AND p.polygon_index=c.parent_polygon_index
               JOIN mask_topology m
                 ON m.run_key=c.run_key AND m.detection_id=c.detection_id
               JOIN audit_runs r ON r.run_key=c.run_key
               WHERE r.model_key='v3' AND c.role='hole'
               ORDER BY ratio,r.run_key,m.frame,c.detection_id,c.polygon_index"""
        )
    ]

    counters: Counter[str] = Counter()
    manifest: list[dict[str, object]] = []
    for item in holes:
        run_key = str(item["run_key"])
        frame = int(item["frame"])
        owner_id = int(item["detection_id"])
        polygon_index = int(item["polygon_index"])
        ratio = float(item["ratio"])
        folder_key = ratio_folder(ratio)
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
        before_removals = {detection_id: set() for detection_id in detection_ids}
        after_removals = {detection_id: set() for detection_id in detection_ids}
        after_removals[owner_id] = descendants(
            instances[owner_id], polygon_index
        )
        image = seek_frame(Path(str(item["input_video"])), frame)
        detail = (
            f"run={run_key} frame={frame} D{owner_id} P{polygon_index} "
            f"hole={float(item['absolute_area']):.1f}px2 / "
            f"parent={float(item['parent_area']):.1f}px2 "
            f"ratio={ratio * 100.0:.3f}%"
        )
        before = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            polygon_index,
            before_removals,
            title="BEFORE: raw mask; target hole=YELLOW",
            detail=detail,
        )
        after = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            polygon_index,
            after_removals,
            title="AFTER: target hole filled; former boundary=YELLOW DASHED",
            detail="Only the selected hole and contours nested inside it are removed",
        )
        comparison = np.concatenate([before, after], axis=1)
        counters[folder_key] += 1
        filename = (
            f"sample_{counters[folder_key]:03d}_{item['video_slug']}_f{frame}_"
            f"D{owner_id}_P{polygon_index}_ratio_{ratio * 100.0:.3f}pct.jpg"
        )
        path = folders[folder_key] / filename
        cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append(
            {
                **item,
                "ratio_percent": ratio * 100.0,
                "frame_detection_ids": detection_ids,
                "folder": folder_key,
                "path": str(path),
            }
        )

    sheets = {
        key: contact_sheet_pages(sorted(folder.glob("*.jpg")), OUTPUT, key)
        for key, folder in folders.items()
    }
    topology.close()
    summary = {
        "total_holes": len(holes),
        "counts": dict(Counter(str(item["folder"]) for item in manifest)),
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
            "folder",
            "run_key",
            "video_slug",
            "frame",
            "detection_id",
            "polygon_index",
            "absolute_area",
            "parent_area",
            "ratio_percent",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
