#!/usr/bin/env python3
"""Render every V3 secondary component under the conservative cleanup rule."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from render_island_threshold_audit import (
    DEFAULT_TOPOLOGY,
    PALETTE,
    ROOT,
    draw_panel,
    load_groups,
    open_ro,
    seek_frame,
)


OUTPUT = ROOT / "output/small_island_conservative_policy_v3_20260813"
RATIO_MAX = 0.05
TEMPORAL_RADIUS = 2
EDGE_MARGIN = 8.0


def bbox_iou(first: dict[str, object], second: dict[str, object]) -> float:
    intersection = max(
        0.0, min(float(first["x2"]), float(second["x2"]))
        - max(float(first["x1"]), float(second["x1"]))
    ) * max(
        0.0, min(float(first["y2"]), float(second["y2"]))
        - max(float(first["y1"]), float(second["y1"]))
    )
    first_area = max(0.0, float(first["x2"]) - float(first["x1"])) * max(
        0.0, float(first["y2"]) - float(first["y1"])
    )
    second_area = max(0.0, float(second["x2"]) - float(second["x1"])) * max(
        0.0, float(second["y2"]) - float(second["y1"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def center_distance(first: dict[str, object], second: dict[str, object]) -> float:
    return math.hypot(
        (
            float(first["x1"]) + float(first["x2"])
            - float(second["x1"]) - float(second["x2"])
        )
        * 0.5,
        (
            float(first["y1"]) + float(first["y2"])
            - float(second["y1"]) - float(second["y2"])
        )
        * 0.5,
    )


def diagonal(item: dict[str, object]) -> float:
    return math.hypot(
        float(item["x2"]) - float(item["x1"]),
        float(item["y2"]) - float(item["y1"]),
    )


def similar(first: dict[str, object], second: dict[str, object]) -> bool:
    first_main = dict(first["main"])
    second_main = dict(second["main"])
    main_ok = bbox_iou(first_main, second_main) >= 0.25 or center_distance(
        first_main, second_main
    ) <= 0.25 * max(diagonal(first_main), diagonal(second_main))
    area_ratio = float(first["absolute_area"]) / max(
        1e-9, float(second["absolute_area"])
    )
    island_ok = (
        bbox_iou(first, second) >= 0.05
        or center_distance(first, second)
        <= max(8.0, 0.60 * max(diagonal(first), diagonal(second)))
    ) and 0.20 <= area_ratio <= 5.0
    return main_ok and island_ok


def edge_set(
    item: dict[str, object], width: int, height: int
) -> set[str]:
    result: set[str] = set()
    if float(item["x1"]) <= EDGE_MARGIN:
        result.add("left")
    if float(item["y1"]) <= EDGE_MARGIN:
        result.add("top")
    if float(item["x2"]) >= width - EDGE_MARGIN:
        result.add("right")
    if float(item["y2"]) >= height - EDGE_MARGIN:
        result.add("bottom")
    return result


def load_components(
    topology: sqlite3.Connection,
) -> tuple[list[dict[str, object]], dict[str, tuple[int, int]]]:
    run_rows = list(
        topology.execute("SELECT * FROM audit_runs WHERE model_key='v3'")
    )
    dimensions: dict[str, tuple[int, int]] = {}
    for run in run_rows:
        inference = sqlite3.connect(str(run["inference_sqlite"]))
        width, height = inference.execute(
            "SELECT width,height FROM frames LIMIT 1"
        ).fetchone()
        inference.close()
        dimensions[str(run["run_key"])] = (int(width), int(height))

    items: list[dict[str, object]] = []
    for run in run_rows:
        run_key = str(run["run_key"])
        grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
        rows = topology.execute(
            """SELECT m.frame,c.detection_id,c.polygon_index,c.absolute_area,
                      c.x1,c.y1,c.x2,c.y2
               FROM contour_topology c
               JOIN mask_topology m USING(run_key,detection_id)
               WHERE c.run_key=? AND c.role='foreground'
               ORDER BY c.detection_id,c.absolute_area DESC""",
            (run_key,),
        )
        for row in rows:
            grouped[int(row["detection_id"])].append(dict(row))
        for detection_id, groups in grouped.items():
            if len(groups) < 2:
                continue
            main = groups[0]
            for group in groups[1:]:
                items.append(
                    {
                        **group,
                        "run_key": run_key,
                        "model_key": "v3",
                        "video_slug": str(run["video_slug"]),
                        "input_video": str(run["input_video"]),
                        "inference_sqlite": str(run["inference_sqlite"]),
                        "main": main,
                        "ratio": float(group["absolute_area"])
                        / float(main["absolute_area"]),
                    }
                )
    return items, dimensions


def classify(
    items: list[dict[str, object]], dimensions: dict[str, tuple[int, int]]
) -> list[dict[str, object]]:
    by_run_frame: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for item in items:
        by_run_frame[(str(item["run_key"]), int(item["frame"]))].append(item)

    classified: list[dict[str, object]] = []
    for item in items:
        temporal_hits: list[dict[str, object]] = []
        for delta in range(-TEMPORAL_RADIUS, TEMPORAL_RADIUS + 1):
            if delta == 0:
                continue
            for candidate in by_run_frame.get(
                (str(item["run_key"]), int(item["frame"]) + delta), []
            ):
                if similar(item, candidate):
                    temporal_hits.append(
                        {
                            "frame_delta": delta,
                            "detection_id": int(candidate["detection_id"]),
                            "polygon_index": int(candidate["polygon_index"]),
                            "ratio": float(candidate["ratio"]),
                        }
                    )
        width, height = dimensions[str(item["run_key"])]
        item_edges = edge_set(item, width, height)
        main_edges = edge_set(dict(item["main"]), width, height)
        common_edges = sorted(item_edges & main_edges)
        if float(item["ratio"]) > RATIO_MAX:
            action, reason = "kept", "ratio_above_5_percent"
        elif temporal_hits:
            action, reason = "kept", "temporal_support"
        elif common_edges:
            action, reason = "kept", "same_frame_edge"
        else:
            action, reason = "deleted", "small_isolated_non_edge"
        classified.append(
            {
                **item,
                "action": action,
                "reason": reason,
                "temporal_hits": temporal_hits,
                "item_edges": sorted(item_edges),
                "main_edges": sorted(main_edges),
                "common_edges": common_edges,
            }
        )
    return classified


def removal_sets(
    instances: dict[int, list[dict[str, object]]],
    action_map: dict[tuple[str, int, int], str],
    run_key: str,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for detection_id, groups in instances.items():
        roots = {
            int(group["polygon_index"])
            for group in groups
            if action_map.get(
                (run_key, detection_id, int(group["polygon_index"]))
            )
            == "deleted"
        }
        parents = {
            int(group["polygon_index"]): group["parent"] for group in groups
        }
        removed: set[int] = set()
        for group in groups:
            index: int | None = int(group["polygon_index"])
            seen: set[int] = set()
            while index is not None and index not in seen:
                if index in roots:
                    removed.add(int(group["polygon_index"]))
                    break
                seen.add(index)
                parent = parents.get(index)
                index = None if parent is None else int(parent)
        result[detection_id] = removed
    return result


def contact_sheet_pages(
    paths: list[Path], output_dir: Path, prefix: str, page_size: int = 20
) -> list[str]:
    outputs: list[str] = []
    for page, start in enumerate(range(0, len(paths), page_size), 1):
        selected = paths[start : start + page_size]
        thumbnails: list[np.ndarray] = []
        for path in selected:
            image = cv2.imread(str(path))
            if image is None:
                continue
            width = 1200
            height = max(1, int(image.shape[0] * width / image.shape[1]))
            thumbnails.append(
                cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            )
        if not thumbnails:
            continue
        cell_height = max(item.shape[0] for item in thumbnails)
        rows = (len(thumbnails) + 1) // 2
        sheet = np.full((rows * cell_height, 2400, 3), 12, np.uint8)
        for index, thumbnail in enumerate(thumbnails):
            row, column = divmod(index, 2)
            sheet[
                row * cell_height : row * cell_height + thumbnail.shape[0],
                column * 1200 : (column + 1) * 1200,
            ] = thumbnail
        output = output_dir / f"contact_sheet_{prefix}_page_{page:02d}.jpg"
        cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs.append(str(output))
    return outputs


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    topology = open_ro(DEFAULT_TOPOLOGY)
    items, dimensions = load_components(topology)
    classified = classify(items, dimensions)
    action_map = {
        (
            str(item["run_key"]),
            int(item["detection_id"]),
            int(item["polygon_index"]),
        ): str(item["action"])
        for item in classified
    }
    folders = {
        "deleted": OUTPUT / "01_deleted",
        "temporal_support": OUTPUT / "02_kept_temporal_support",
        "same_frame_edge": OUTPUT / "03_kept_edge_protection",
        "ratio_above_5_percent": OUTPUT / "04_kept_above_5_percent",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.jpg"):
            stale.unlink()
    for stale in OUTPUT.glob("contact_sheet_*.jpg"):
        stale.unlink()

    manifest: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    for item in sorted(
        classified,
        key=lambda value: (
            str(value["action"]), str(value["reason"]),
            float(value["ratio"]), str(value["run_key"]), int(value["frame"]),
        ),
    ):
        run_key = str(item["run_key"])
        frame = int(item["frame"])
        owner_id = int(item["detection_id"])
        frame_rows = list(
            topology.execute(
                """SELECT detection_id FROM mask_topology
                   WHERE run_key=? AND frame=? ORDER BY detection_index,detection_id""",
                (run_key, frame),
            )
        )
        detection_ids = [int(row["detection_id"]) for row in frame_rows]
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
        removals = removal_sets(instances, action_map, run_key)
        image = seek_frame(Path(str(item["input_video"])), frame)
        ratio_percent = float(item["ratio"]) * 100.0
        temporal_deltas = [
            int(hit["frame_delta"]) for hit in list(item["temporal_hits"])
        ]
        detail = (
            f"run={run_key} frame={frame} D{owner_id} P{item['polygon_index']} "
            f"ratio={ratio_percent:.3f}% temporal={temporal_deltas or 'none'} "
            f"common_edge={item['common_edges'] or 'none'}"
        )
        before = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            int(item["polygon_index"]),
            {detection_id: set() for detection_id in detection_ids},
            title="BEFORE: all raw instances; owner=WHITE, island=YELLOW",
            detail=detail,
        )
        after = draw_panel(
            image,
            instances,
            colors,
            owner_id,
            int(item["polygon_index"]),
            removals,
            title=(
                f"AFTER: {str(item['action']).upper()} / "
                f"{str(item['reason']).replace('_', ' ')}"
            ),
            detail="Rule: ratio<=5%, no temporal support, no shared edge => DELETE",
        )
        comparison = np.concatenate([before, after], axis=1)
        folder_key = (
            "deleted" if item["action"] == "deleted" else str(item["reason"])
        )
        folder = folders[folder_key]
        counters[folder_key] += 1
        filename = (
            f"sample_{counters[folder_key]:03d}_{item['video_slug']}_f{frame}_"
            f"D{owner_id}_P{item['polygon_index']}_ratio_{ratio_percent:.3f}pct.jpg"
        )
        path = folder / filename
        cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append(
            {
                **item,
                "ratio_percent": ratio_percent,
                "frame_detection_ids": detection_ids,
                "path": str(path),
            }
        )

    sheets: dict[str, list[str]] = {}
    for key, folder in folders.items():
        sheets[key] = contact_sheet_pages(
            sorted(folder.glob("*.jpg")), OUTPUT, key
        )

    topology.close()
    summary = {
        "total_components": len(classified),
        "counts": dict(Counter(str(item["reason"]) for item in classified)),
        "actions": dict(Counter(str(item["action"]) for item in classified)),
        "parameters": {
            "ratio_max": RATIO_MAX,
            "temporal_radius": TEMPORAL_RADIUS,
            "edge_margin_px": EDGE_MARGIN,
        },
        "contact_sheets": sheets,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "action", "reason", "run_key", "video_slug", "frame", "detection_id",
            "polygon_index", "absolute_area", "ratio_percent", "common_edges",
            "temporal_hits", "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
