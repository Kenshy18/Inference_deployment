#!/usr/bin/env python3
"""Render local-only review sheets for holes, islands and cross-instance overlap.

Each sample contains the actual source-video frame, the AI raw mask, the
postprocessed final mask and a diagnostic comparison.  Source frames never
leave the local machine.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from render_actual_frame_overlays import (
    all_points,
    contact_sheet,
    crop_panel,
    draw_groups,
    draw_topology_mask,
    draw_two_masks,
    mask_from_groups,
    open_ro,
    put,
    raw_contours,
    read_frames,
    save,
)
from audit_additional_failure_modes import _load_contours, _safe_polygon


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "output/instance_mask_topology_20260806"
TOPOLOGY = EXPERIMENT / "topology.sqlite"
TRACE = EXPERIMENT / "postprocess_trace.sqlite"
OUTPUT = ROOT / "output/instance_topology_three_core_families_20260812"
SAMPLES_PER_FAMILY = 10


def result_summary(run_key: str) -> dict[str, object]:
    path = EXPERIMENT / f"postprocess_trace.{run_key}.summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def predictions_path(run_key: str) -> Path:
    manifest = Path(str(result_summary(run_key)["postprocess_manifest"]))
    return manifest.parent / "05_polygon_optimization" / "predictions.sqlite"


def final_components(run_key: str, track_id: str, frame: int) -> list[dict[str, object]]:
    connection = open_ro(predictions_path(run_key))
    row = connection.execute(
        "SELECT polygons FROM masks WHERE frame=? AND track_id=?",
        (int(frame), str(track_id)),
    ).fetchone()
    connection.close()
    if row is None or not row["polygons"]:
        return []
    polygons = json.loads(str(row["polygons"]))
    return [
        {
            "slot": index,
            "role": "exterior",
            "points": np.asarray(points, dtype=np.float32),
        }
        for index, points in enumerate(polygons)
        if len(points) >= 3
    ]


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left.astype(bool)
    right = right.astype(bool)
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    left_area = int(np.count_nonzero(left))
    right_area = int(np.count_nonzero(right))
    return {
        "iou": intersection / max(union, 1),
        "left_area": left_area,
        "right_area": right_area,
        "right_to_left_area": right_area / max(left_area, 1),
    }


def draw_difference(
    image: np.ndarray,
    raw_groups: list[dict[str, object]],
    final_groups: list[dict[str, object]],
) -> tuple[np.ndarray, dict[str, float]]:
    raw_mask = mask_from_groups(image.shape[:2], raw_groups).astype(bool)
    final_mask = mask_from_groups(image.shape[:2], final_groups).astype(bool)
    output = image.copy()
    overlay = output.copy()
    overlay[raw_mask & final_mask] = (80, 220, 90)   # green: shared
    overlay[raw_mask & ~final_mask] = (60, 60, 255)  # red: raw only
    overlay[~raw_mask & final_mask] = (255, 160, 40) # blue: final only
    cv2.addWeighted(overlay, 0.52, output, 0.48, 0.0, dst=output)
    for group in raw_groups:
        contour = np.rint(group["points"]).astype(np.int32)
        if len(contour) >= 3:
            color = (0, 255, 255) if group["role"] == "hole" else (60, 60, 255)
            cv2.polylines(output, [contour], True, color, 3, cv2.LINE_AA)
    for group in final_groups:
        contour = np.rint(group["points"]).astype(np.int32)
        if len(contour) >= 3:
            cv2.polylines(output, [contour], True, (255, 160, 40), 2, cv2.LINE_AA)
    put(output, "green=shared  red=raw-only  blue=final-only", (18, 34), 0.58)
    return output, _metrics(raw_mask, final_mask)


def draw_pair_difference(
    image: np.ndarray,
    first: list[dict[str, object]],
    second: list[dict[str, object]],
    *,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    left = mask_from_groups(image.shape[:2], first).astype(bool)
    right = mask_from_groups(image.shape[:2], second).astype(bool)
    output = image.copy()
    overlay = output.copy()
    overlay[left] = (60, 80, 255)
    overlay[right] = (255, 140, 40)
    overlay[left & right] = (245, 245, 245)
    cv2.addWeighted(overlay, 0.52, output, 0.48, 0.0, dst=output)
    for groups, color, name in (
        (first, (60, 80, 255), first_name),
        (second, (255, 140, 40), second_name),
    ):
        points = [group["points"] for group in groups if len(group["points"])]
        if not points:
            continue
        for group in groups:
            contour = np.rint(group["points"]).astype(np.int32)
            cv2.polylines(output, [contour], True, color, 3, cv2.LINE_AA)
        center = np.mean(np.concatenate(points, axis=0), axis=0).astype(int)
        put(output, name, (int(center[0]), int(center[1])), 0.52)
    metrics = _metrics(left, right)
    smaller = min(metrics["left_area"], metrics["right_area"])
    overlap = int(np.count_nonzero(left & right))
    metrics["smaller_containment"] = overlap / max(smaller, 1)
    return output, metrics


def _candidate_rows(kind: str) -> list[dict[str, object]]:
    trace = open_ro(TRACE)
    topology = open_ro(TOPOLOGY)
    topo = {
        (str(row["run_key"]), int(row["detection_id"])): dict(row)
        for row in topology.execute("SELECT * FROM mask_topology")
    }
    if kind == "holes":
        where = "hole_count>0"
        severity_key = "largest_hole_to_outer_ratio"
    else:
        where = "foreground_component_count>1 AND hole_count=0"
        severity_key = "second_to_largest_ratio"
    rows = []
    for row in trace.execute(
        f"""SELECT * FROM detection_outcomes
            WHERE disposition='retained' AND exact_keyframe=1 AND {where}"""
    ):
        item = dict(row)
        item.update(topo[(str(row["run_key"]), int(row["detection_id"]))])
        item["severity"] = float(item.get(severity_key) or 0.0)
        rows.append(item)
    trace.close()
    topology.close()
    rows.sort(key=lambda item: (-float(item["severity"]), -float(item.get("score") or 0.0)))
    return rows


def _diverse_selection(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Prefer distinct runs/tracks, then admit a second sample per run."""
    by_run: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    seen_track: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["run_key"]), str(row["final_track_id"]))
        if key in seen_track:
            continue
        seen_track.add(key)
        by_run[str(row["run_key"])].append(row)
    selected: list[dict[str, object]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for run_key in sorted(by_run):
            if depth < len(by_run[run_key]):
                item = by_run[run_key][depth]
                if final_components(
                    str(item["run_key"]), str(item["final_track_id"]), int(item["frame"])
                ):
                    selected.append(item)
                    added = True
                    if len(selected) == count:
                        break
        if not added:
            break
        depth += 1
    return selected


def _all_duplicate_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Recompute all audited duplicate pairs; the old report capped examples at 50 globally."""
    trace = open_ro(TRACE)
    output: list[dict[str, object]] = []
    for run_key in sorted(runs):
        retained = list(
            trace.execute(
                """SELECT detection_id,frame,final_track_id
                   FROM detection_outcomes
                   WHERE run_key=? AND disposition='retained'
                   ORDER BY frame,detection_id""",
                (run_key,),
            )
        )
        by_frame: dict[int, list[sqlite3.Row]] = collections.defaultdict(list)
        for row in retained:
            by_frame[int(row["frame"])].append(row)
        candidate_ids = sorted({
            int(row["detection_id"])
            for values in by_frame.values() if len(values) > 1
            for row in values
        })
        inference = open_ro(Path(str(runs[run_key]["inference_sqlite"])))
        contours = _load_contours(inference, set(candidate_ids))
        info: dict[int, sqlite3.Row] = {}
        for start in range(0, len(candidate_ids), 700):
            chunk = candidate_ids[start : start + 700]
            if not chunk:
                continue
            for row in inference.execute(
                f"SELECT id,score,x1,y1,x2,y2 FROM detections WHERE id IN ({','.join('?' for _ in chunk)})",
                chunk,
            ):
                info[int(row["id"])] = row
        for frame, values in by_frame.items():
            for left_pos, left_row in enumerate(values):
                for right_row in values[left_pos + 1 :]:
                    left_id = int(left_row["detection_id"])
                    right_id = int(right_row["detection_id"])
                    left_info, right_info = info.get(left_id), info.get(right_id)
                    if left_info is None or right_info is None:
                        continue
                    overlap_w = max(
                        0.0,
                        min(left_info["x2"], right_info["x2"])
                        - max(left_info["x1"], right_info["x1"]),
                    )
                    overlap_h = max(
                        0.0,
                        min(left_info["y2"], right_info["y2"])
                        - max(left_info["y1"], right_info["y1"]),
                    )
                    if overlap_w * overlap_h <= 0:
                        continue
                    left_polys = [
                        _safe_polygon(points) for points in contours.get(left_id, {}).values()
                    ]
                    right_polys = [
                        _safe_polygon(points) for points in contours.get(right_id, {}).values()
                    ]
                    left_polys = [value for value in left_polys if value is not None and value.is_valid]
                    right_polys = [value for value in right_polys if value is not None and value.is_valid]
                    if not left_polys or not right_polys:
                        continue
                    left_shape = left_polys[0]
                    for value in left_polys[1:]:
                        left_shape = left_shape.union(value)
                    right_shape = right_polys[0]
                    for value in right_polys[1:]:
                        right_shape = right_shape.union(value)
                    intersection = float(left_shape.intersection(right_shape).area)
                    if intersection <= 0:
                        continue
                    iou = intersection / max(float(left_shape.union(right_shape).area), 1e-9)
                    containment = intersection / max(min(float(left_shape.area), float(right_shape.area)), 1e-9)
                    if containment < 0.80 and iou < 0.20:
                        continue
                    output.append({
                        "run_key": run_key,
                        "frame": frame,
                        "detection_ids": [left_id, right_id],
                        "track_ids": [str(left_row["final_track_id"]), str(right_row["final_track_id"])],
                        "scores": [float(left_info["score"]), float(right_info["score"])],
                        "mask_iou": iou,
                        "smaller_containment": containment,
                    })
        inference.close()
    trace.close()
    return output


def _duplicate_selection(runs: dict[str, dict[str, object]], count: int) -> list[dict[str, object]]:
    rows = _all_duplicate_rows(runs)
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, object]]] = collections.defaultdict(list)
    for row in rows:
        groups[(str(row["run_key"]), tuple(map(str, row["track_ids"])))].append(row)
    for values in groups.values():
        values.sort(key=lambda item: (-float(item["smaller_containment"]), int(item["frame"])))
    selected: list[dict[str, object]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for key in sorted(groups):
            values = groups[key]
            if depth >= len(values):
                continue
            item = values[depth]
            tracks = list(map(str, item["track_ids"]))
            if all(final_components(str(item["run_key"]), track, int(item["frame"])) for track in tracks):
                selected.append(item)
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def _load_raw(
    topology: sqlite3.Connection,
    run: dict[str, object],
    run_key: str,
    detection_id: int,
) -> list[dict[str, object]]:
    inference = open_ro(Path(str(run["inference_sqlite"])))
    groups = raw_contours(inference, topology, run_key, detection_id)
    inference.close()
    return groups


def _hole_fill_fraction(
    shape: tuple[int, int],
    raw_groups: list[dict[str, object]],
    final_groups: list[dict[str, object]],
) -> float:
    hole_groups = [group for group in raw_groups if group["role"] == "hole"]
    if not hole_groups:
        return 0.0
    hole_mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(
        hole_mask,
        [np.rint(group["points"]).astype(np.int32) for group in hole_groups],
        1,
    )
    final_mask = mask_from_groups(shape, final_groups)
    return float(np.count_nonzero((hole_mask > 0) & (final_mask > 0))) / max(
        int(np.count_nonzero(hole_mask)), 1
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("01_holes", "02_islands", "03_cross_instance_overlap"):
        family_dir = OUTPUT / name
        family_dir.mkdir(parents=True, exist_ok=True)
        for stale in family_dir.glob("*.jpg"):
            stale.unlink()

    topology = open_ro(TOPOLOGY)
    runs = {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs")
    }
    selections = {
        "holes": _diverse_selection(_candidate_rows("holes"), SAMPLES_PER_FAMILY),
        "islands": _diverse_selection(_candidate_rows("islands"), SAMPLES_PER_FAMILY),
        "cross_instance_overlap": _duplicate_selection(runs, SAMPLES_PER_FAMILY),
    }

    # Decode each required source frame once.
    required: dict[Path, list[int]] = collections.defaultdict(list)
    for family, rows in selections.items():
        for row in rows:
            run = runs[str(row["run_key"])]
            required[Path(str(run["input_video"]))].append(int(row["frame"]))
    frames: dict[tuple[Path, int], np.ndarray] = {}
    for video, indices in required.items():
        for frame, image in read_frames(video, indices).items():
            frames[(video, frame)] = image

    manifest: dict[str, object] = {
        "schema_version": 1,
        "privacy": "Local-only source-frame review; no network access.",
        "families": {},
    }
    folders = {
        "holes": "01_holes",
        "islands": "02_islands",
        "cross_instance_overlap": "03_cross_instance_overlap",
    }

    for family, rows in selections.items():
        family_panels: list[np.ndarray] = []
        family_manifest: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            run_key = str(row["run_key"])
            frame = int(row["frame"])
            run = runs[run_key]
            video = Path(str(run["input_video"]))
            image = frames[(video, frame)]

            if family in {"holes", "islands"}:
                detection_id = int(row["detection_id"])
                track_id = str(row["final_track_id"])
                raw_groups = _load_raw(topology, run, run_key, detection_id)
                final_groups = final_components(run_key, track_id, frame)
                raw_rendered = (
                    draw_topology_mask(image, raw_groups, stage="raw")
                    if family == "holes"
                    else draw_groups(image, raw_groups, label_prefix="raw_component")
                )
                final_rendered = draw_groups(image, final_groups, label_prefix="final_slot")
                diff_rendered, metrics = draw_difference(image, raw_groups, final_groups)
                common = all_points([raw_groups, final_groups])
                if family == "holes":
                    fill = _hole_fill_fraction(image.shape[:2], raw_groups, final_groups)
                    metrics["raw_hole_count"] = int(row["hole_count"])
                    metrics["hole_filled_by_final_fraction"] = fill
                    subtitle = f"holes={int(row['hole_count'])} largest_ratio={float(row['severity']):.3f}"
                    diagnosis = f"final fill inside raw holes={fill:.1%}"
                else:
                    metrics["raw_foreground_components"] = int(row["foreground_component_count"])
                    metrics["final_components"] = len(final_groups)
                    subtitle = (
                        f"raw_components={int(row['foreground_component_count'])} "
                        f"second/largest={float(row['severity']):.3f}"
                    )
                    diagnosis = f"final_slots={len(final_groups)} area_ratio={metrics['right_to_left_area']:.3f}"
                panels = [
                    crop_panel(raw_rendered, common, [
                        f"{index:02d} AI RAW MASK", f"frame={frame} detection={detection_id}", subtitle,
                    ]),
                    crop_panel(final_rendered, common, [
                        f"{index:02d} FINAL MASK", f"track={track_id} frame={frame}", diagnosis,
                    ]),
                    crop_panel(diff_rendered, common, [
                        f"{index:02d} RAW vs FINAL", f"IoU={metrics['iou']:.3f}",
                        "green=shared / red=raw / blue=final",
                    ]),
                ]
                record = {
                    "sample": index,
                    "run_key": run_key,
                    "video": str(video),
                    "frame": frame,
                    "detection_id": detection_id,
                    "final_track_id": track_id,
                    "score": float(row.get("score") or 0.0),
                    **metrics,
                }
            else:
                detection_ids = list(map(int, row["detection_ids"]))
                track_ids = list(map(str, row["track_ids"]))
                raw_first = _load_raw(topology, run, run_key, detection_ids[0])
                raw_second = _load_raw(topology, run, run_key, detection_ids[1])
                final_first = final_components(run_key, track_ids[0], frame)
                final_second = final_components(run_key, track_ids[1], frame)
                raw_rendered = draw_two_masks(image, raw_first, raw_second)
                final_rendered = draw_two_masks(image, final_first, final_second)
                diagnostic, final_metrics = draw_pair_difference(
                    image, final_first, final_second,
                    first_name=f"track{track_ids[0]}", second_name=f"track{track_ids[1]}",
                )
                common = all_points([raw_first, raw_second, final_first, final_second])
                panels = [
                    crop_panel(raw_rendered, common, [
                        f"{index:02d} AI RAW: TWO INSTANCES", f"frame={frame} det={detection_ids}",
                        f"raw containment={float(row['smaller_containment']):.1%}",
                    ]),
                    crop_panel(final_rendered, common, [
                        f"{index:02d} FINAL: TWO TRACKS", f"tracks={track_ids}",
                        "Both instances remain after NMS/tracking",
                    ]),
                    crop_panel(diagnostic, common, [
                        f"{index:02d} FINAL OVERLAP", f"IoU={final_metrics['iou']:.3f}",
                        f"smaller containment={final_metrics['smaller_containment']:.1%}",
                    ]),
                ]
                record = {
                    "sample": index,
                    "run_key": run_key,
                    "video": str(video),
                    "frame": frame,
                    "detection_ids": detection_ids,
                    "final_track_ids": track_ids,
                    "raw_mask_iou": float(row["mask_iou"]),
                    "raw_smaller_containment": float(row["smaller_containment"]),
                    "final_mask_iou": final_metrics["iou"],
                    "final_smaller_containment": final_metrics["smaller_containment"],
                }

            sample_panel = contact_sheet(panels, columns=3)
            filename = f"sample_{index:02d}_{run_key}_f{frame}.jpg"
            path = OUTPUT / folders[family] / filename
            cv2.imwrite(str(path), sample_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
            record["overlay"] = str(path)
            family_manifest.append(record)
            family_panels.append(sample_panel)

        # Two pages per family make each example legible at normal zoom.
        page_paths = []
        for page_index, offset in enumerate(range(0, len(family_panels), 5), start=1):
            page = contact_sheet(family_panels[offset : offset + 5], columns=1)
            page_path = OUTPUT / folders[family] / f"contact_sheet_{page_index}.jpg"
            cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 94])
            page_paths.append(str(page_path))
        manifest["families"][family] = {
            "sample_count": len(family_manifest),
            "contact_sheets": page_paths,
            "samples": family_manifest,
        }

    topology.close()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Three core topology failure families\n\n"
        "Actual source frames are used locally. Each sample has three panels: "
        "AI raw mask, final postprocessed mask, and a diagnostic comparison.\n\n"
        "- `01_holes`: raw background holes and their final export behavior\n"
        "- `02_islands`: disconnected foreground islands in one AI instance\n"
        "- `03_cross_instance_overlap`: overlapping/contained separate instances retained as separate final tracks\n\n"
        "Colors in raw/final difference panels: green=shared, red=raw-only, blue=final-only.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: len(value) for key, value in selections.items()}, indent=2))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
