#!/usr/bin/env python3
"""Render before/after overlays for the 10% small-island cleanup rule."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from run_ablation import DEFAULT_TOPOLOGY, ROOT, open_ro, put, seek_frame


DEFAULT_OUTPUT = ROOT / "output/small_island_threshold_audit_20260813"
INFERENCE = "inference"

BINS = [
    ("00_0_to_0p1pct", 0.0, 0.001, 6),
    ("01_0p1_to_1pct", 0.001, 0.01, 6),
    ("02_1_to_3pct", 0.01, 0.03, 6),
    ("03_3_to_5pct", 0.03, 0.05, 6),
    ("04_5_to_8pct", 0.05, 0.08, 8),
    # Render every unique near-boundary case.
    ("05_8_to_10pct_removed", 0.08, 0.10, -1),
    ("06_10_to_12pct_kept", 0.10, 0.12, -1),
    ("07_12_to_15pct", 0.12, 0.15, 6),
    ("08_15_to_25pct", 0.15, 0.25, 6),
    ("09_25_to_50pct", 0.25, 0.50, 6),
    ("10_50_to_100pct", 0.50, 1.01, 6),
]

PALETTE = [
    (55, 70, 245),
    (245, 120, 35),
    (45, 210, 95),
    (210, 70, 220),
    (35, 200, 240),
    (225, 185, 45),
    (85, 145, 250),
    (180, 105, 250),
]


def candidates(
    connection: sqlite3.Connection, *, model_key: str | None = None
) -> list[dict[str, object]]:
    query = """
    WITH foreground AS (
      SELECT c.run_key,c.detection_id,c.polygon_index,c.absolute_area,
             MAX(c.absolute_area) OVER(
               PARTITION BY c.run_key,c.detection_id
             ) AS largest_area
      FROM contour_topology c
      WHERE c.role='foreground'
    )
    SELECT f.run_key,r.model_key,r.video_slug,r.input_video,r.inference_sqlite,
           m.frame,m.score,m.class_name,f.detection_id,f.polygon_index,
           f.absolute_area,f.largest_area,
           f.absolute_area/f.largest_area AS ratio
    FROM foreground f
    JOIN mask_topology m
      ON m.run_key=f.run_key AND m.detection_id=f.detection_id
    JOIN audit_runs r ON r.run_key=f.run_key
    WHERE f.largest_area>0 AND f.absolute_area<f.largest_area
    ORDER BY ratio,f.run_key,m.frame,f.detection_id,f.polygon_index
    """
    rows = [dict(row) for row in connection.execute(query)]
    if model_key is not None:
        rows = [row for row in rows if str(row["model_key"]) == model_key]
    return rows


def choose_diverse(
    items: list[dict[str, object]], limit: int, *, closest_to: float | None = None
) -> list[dict[str, object]]:
    ordered = sorted(
        items,
        key=(
            (lambda item: abs(float(item["ratio"]) - float(closest_to)))
            if closest_to is not None
            else (lambda item: float(item["ratio"]))
        ),
    )
    # The same source occurs in overlapping full/clip/concatenated audit runs.
    # Geometry identity removes these duplicates while retaining model diversity.
    unique: list[dict[str, object]] = []
    seen_geometry: set[tuple[object, ...]] = set()
    for item in ordered:
        key = (
            str(item["model_key"]),
            int(item["detection_id"]),
            int(item["polygon_index"]),
            round(float(item["absolute_area"]), 3),
            round(float(item["largest_area"]), 3),
        )
        if key in seen_geometry:
            continue
        seen_geometry.add(key)
        unique.append(item)
    if limit < 0 or len(unique) <= limit:
        return unique

    selected: list[dict[str, object]] = []
    used_runs: set[str] = set()
    used_videos: set[tuple[str, str]] = set()
    for item in unique:
        video_key = (str(item["model_key"]), str(item["video_slug"]))
        if video_key in used_videos:
            continue
        selected.append(item)
        used_runs.add(str(item["run_key"]))
        used_videos.add(video_key)
        if len(selected) == limit:
            return selected
    for item in unique:
        if item in selected:
            continue
        if str(item["run_key"]) in used_runs:
            continue
        selected.append(item)
        used_runs.add(str(item["run_key"]))
        if len(selected) == limit:
            return selected
    for item in unique:
        if item not in selected:
            selected.append(item)
            if len(selected) == limit:
                break
    return selected


def load_groups(
    inference_path: Path,
    topology_path: Path,
    run_key: str,
    detection_id: int,
) -> list[dict[str, object]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("ATTACH DATABASE ? AS src", (str(inference_path),))
    connection.execute("ATTACH DATABASE ? AS topo", (str(topology_path),))
    rows = connection.execute(
        """SELECT p.polygon_index,t.role,t.nesting_depth,t.parent_polygon_index,
                  t.absolute_area,s.point_index,s.x,s.y
           FROM src.segmentation_polygons p
           JOIN src.segmentation_points s ON s.polygon_id=p.id
           JOIN topo.contour_topology t
             ON t.run_key=? AND t.detection_id=p.detection_id
            AND t.polygon_index=p.polygon_index
           WHERE p.detection_id=?
           ORDER BY p.polygon_index,s.point_index""",
        (run_key, detection_id),
    )
    result: list[dict[str, object]] = []
    current: int | None = None
    metadata: dict[str, object] = {}
    points: list[tuple[float, float]] = []
    for row in rows:
        polygon_index = int(row["polygon_index"])
        if current is not None and polygon_index != current:
            result.append({**metadata, "points": np.asarray(points, np.float32)})
            points = []
        current = polygon_index
        metadata = {
            "polygon_index": polygon_index,
            "role": str(row["role"]),
            "depth": int(row["nesting_depth"]),
            "parent": (
                None
                if row["parent_polygon_index"] is None
                else int(row["parent_polygon_index"])
            ),
            "area": float(row["absolute_area"]),
        }
        points.append((float(row["x"]), float(row["y"])))
    if current is not None:
        result.append({**metadata, "points": np.asarray(points, np.float32)})
    connection.close()
    return result


def descendants_of_small_components(
    groups: list[dict[str, object]], ratio_max: float
) -> set[int]:
    foreground = [group for group in groups if group["role"] == "foreground"]
    largest = max(float(group["area"]) for group in foreground)
    roots = {
        int(group["polygon_index"])
        for group in foreground
        if float(group["area"]) < largest
        and float(group["area"]) / largest <= ratio_max
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
            value = parents.get(index)
            index = None if value is None else int(value)
    return removed


def selected_bounds(group: dict[str, object], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    points = np.asarray(group["points"])
    low = np.floor(points.min(axis=0)).astype(int)
    high = np.ceil(points.max(axis=0)).astype(int)
    width, height = max(1, high[0] - low[0]), max(1, high[1] - low[1])
    padding = max(24, int(max(width, height) * 1.5))
    x1 = max(0, low[0] - padding)
    y1 = max(0, low[1] - padding)
    x2 = min(image_shape[1], high[0] + padding + 1)
    y2 = min(image_shape[0], high[1] + padding + 1)
    return x1, y1, x2, y2


def instance_mask(
    shape: tuple[int, int], groups: list[dict[str, object]], removed: set[int]
) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for group in sorted(groups, key=lambda item: int(item["depth"])):
        if int(group["polygon_index"]) in removed:
            continue
        contour = np.rint(group["points"]).astype(np.int32)
        if len(contour) < 3:
            continue
        cv2.fillPoly(mask, [contour], 1 if group["role"] == "foreground" else 0)
    return mask


def dashed_polyline(
    image: np.ndarray, contour: np.ndarray, color: tuple[int, int, int], thickness: int
) -> None:
    points = contour.reshape(-1, 2)
    if len(points) < 2:
        return
    for index in range(0, len(points), 2):
        cv2.line(
            image,
            tuple(int(value) for value in points[index]),
            tuple(int(value) for value in points[(index + 1) % len(points)]),
            color,
            thickness,
            cv2.LINE_AA,
        )


def label_instance(
    image: np.ndarray,
    detection_id: int,
    groups: list[dict[str, object]],
    color: tuple[int, int, int],
) -> None:
    foreground = [group for group in groups if group["role"] == "foreground"]
    if not foreground:
        return
    main = max(foreground, key=lambda group: float(group["area"]))
    points = np.asarray(main["points"])
    center = np.mean(points, axis=0).astype(int)
    text = f"D{detection_id}"
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    x = int(np.clip(center[0], 4, max(4, image.shape[1] - width - 10)))
    y = int(np.clip(center[1], height + 8, max(height + 8, image.shape[0] - 8)))
    cv2.rectangle(image, (x - 4, y - height - 6), (x + width + 5, y + 5), color, -1)
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (15, 15, 15), 2, cv2.LINE_AA,
    )


def draw_panel(
    image: np.ndarray,
    instances: dict[int, list[dict[str, object]]],
    colors: dict[int, tuple[int, int, int]],
    owner_id: int,
    selected_index: int,
    removals: dict[int, set[int]],
    *,
    title: str,
    detail: str,
) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    for detection_id, groups in instances.items():
        mask = instance_mask(image.shape[:2], groups, removals.get(detection_id, set())).astype(bool)
        overlay[mask] = colors[detection_id]
    cv2.addWeighted(overlay, 0.46, canvas, 0.54, 0, dst=canvas)

    # Only the island owner receives a strong whole-instance boundary.
    owner_groups = instances[owner_id]
    owner_removed = removals.get(owner_id, set())
    for group in owner_groups:
        if int(group["polygon_index"]) in owner_removed:
            continue
        contour = np.rint(group["points"]).astype(np.int32)
        if len(contour) >= 3:
            cv2.polylines(canvas, [contour], True, (245, 245, 245), 5, cv2.LINE_AA)

    selected = next(
        group for group in owner_groups if int(group["polygon_index"]) == selected_index
    )
    contour = np.rint(selected["points"]).astype(np.int32)
    if selected_index not in owner_removed:
        cv2.polylines(canvas, [contour], True, (20, 235, 255), 4, cv2.LINE_AA)
    else:
        dashed_polyline(canvas, contour, (20, 235, 255), 4)

    for detection_id, groups in instances.items():
        label_instance(canvas, detection_id, groups, colors[detection_id])

    x1, y1, x2, y2 = selected_bounds(selected, image.shape[:2])
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (20, 235, 255), 3)

    crop = canvas[y1:y2, x1:x2]
    inset_width = min(380, max(220, image.shape[1] // 3))
    inset_height = min(300, max(180, image.shape[0] // 3))
    scale = min(inset_width / max(1, crop.shape[1]), inset_height / max(1, crop.shape[0]))
    resized = cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    ih, iw = resized.shape[:2]
    ix1, iy1 = canvas.shape[1] - iw - 16, 16
    canvas[iy1 : iy1 + ih, ix1 : ix1 + iw] = resized
    cv2.rectangle(canvas, (ix1 - 3, iy1 - 3), (ix1 + iw + 2, iy1 + ih + 2), (20, 235, 255), 3)

    header = np.full((130, canvas.shape[1], 3), 14, np.uint8)
    put(header, title, (16, 30), 0.64)
    put(header, detail, (16, 66), 0.48)
    x = 16
    for detection_id in instances:
        label = f"D{detection_id}"
        (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        if x + width + 48 > header.shape[1]:
            break
        cv2.rectangle(header, (x, 91), (x + 24, 115), colors[detection_id], -1)
        put(header, label, (x + 31, 111), 0.46)
        x += width + 68
    return np.vstack([header, canvas])


def write_contact_sheet(paths: list[Path], output: Path, *, columns: int = 2) -> None:
    if not paths:
        return
    thumbnails: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        width = 1200
        height = max(1, int(image.shape[0] * width / image.shape[1]))
        thumbnails.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    if not thumbnails:
        return
    cell_height = max(item.shape[0] for item in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * 1200, 3), 12, np.uint8)
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        sheet[row * cell_height : row * cell_height + thumbnail.shape[0],
              column * 1200 : (column + 1) * 1200] = thumbnail
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("v3", "v3lite"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="render every unique component in every area bin",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    topology = open_ro(DEFAULT_TOPOLOGY)
    all_items = candidates(topology, model_key=args.model)
    manifest: list[dict[str, object]] = []
    distribution: list[dict[str, object]] = []

    for bin_name, lower, upper, limit in BINS:
        members = [
            item
            for item in all_items
            if float(item["ratio"]) > lower or lower == 0.0
            if float(item["ratio"]) <= upper
        ]
        distribution.append(
            {
                "bin": bin_name,
                "lower_exclusive": lower,
                "upper_inclusive": upper,
                "component_count": len(members),
            }
        )
        closest = 0.10 if lower >= 0.08 and upper <= 0.12 else None
        selected = choose_diverse(
            members, -1 if args.all_components else limit, closest_to=closest
        )
        folder = output / bin_name
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.jpg"):
            stale.unlink()

        for number, item in enumerate(selected, 1):
            frame_rows = list(
                topology.execute(
                    """SELECT detection_id,score FROM mask_topology
                       WHERE run_key=? AND frame=?
                       ORDER BY detection_index,detection_id""",
                    (str(item["run_key"]), int(item["frame"])),
                )
            )
            detection_ids = [int(row["detection_id"]) for row in frame_rows]
            instances = {
                detection_id: load_groups(
                    Path(str(item["inference_sqlite"])),
                    DEFAULT_TOPOLOGY,
                    str(item["run_key"]),
                    detection_id,
                )
                for detection_id in detection_ids
            }
            colors = {
                detection_id: PALETTE[index % len(PALETTE)]
                for index, detection_id in enumerate(detection_ids)
            }
            image = seek_frame(Path(str(item["input_video"])), int(item["frame"]))
            removals = {
                detection_id: descendants_of_small_components(groups, 0.10)
                for detection_id, groups in instances.items()
            }
            owner_id = int(item["detection_id"])
            removed = removals[owner_id]
            ratio = float(item["ratio"])
            action = "REMOVED" if int(item["polygon_index"]) in removed else "KEPT"
            detail = (
                f"run={item['run_key']} frame={item['frame']} D{item['detection_id']} "
                f"island={float(item['absolute_area']):.1f}px2 / main={float(item['largest_area']):.1f}px2 "
                f"ratio={ratio * 100:.3f}%"
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
                title=f"AFTER 10% RULE: {action}",
                detail="Removed island position remains as a YELLOW DASHED outline",
            )
            comparison = np.concatenate([before, after], axis=1)
            filename = (
                f"sample_{number:02d}_{item['model_key']}_{item['video_slug']}_"
                f"f{item['frame']}_D{item['detection_id']}_ratio_{ratio * 100:.3f}pct.jpg"
            )
            path = folder / filename
            cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append(
                {
                    **item,
                    "ratio_percent": ratio * 100.0,
                    "action_at_10_percent": action.lower(),
                    "frame_detection_ids": detection_ids,
                    "owner_detection_id": owner_id,
                    "bin": bin_name,
                    "path": str(path),
                }
            )
        write_contact_sheet(
            sorted(folder.glob("*.jpg")), output / f"contact_sheet_{bin_name}.jpg"
        )
        print(f"{bin_name}: population={len(members)} rendered={len(selected)}", flush=True)

    topology.close()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "bin", "model_key", "video_slug", "run_key", "frame", "detection_id",
            "polygon_index", "absolute_area", "largest_area", "ratio_percent",
            "action_at_10_percent", "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
