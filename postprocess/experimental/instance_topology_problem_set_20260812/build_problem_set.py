#!/usr/bin/env python3
"""Build a mask-only review set for duplicate and island failure modes.

No video is opened.  Every PNG is rendered from SQLite polygon coordinates on
a synthetic dark canvas so it is safe for the agent and the user to inspect.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = ROOT / "output/instance_mask_topology_20260806"
DEFAULT_OUTPUT = ROOT / "output/instance_topology_problem_set_20260812"
FPS = 30000.0 / 1001.0


@dataclass
class Case:
    case_id: str
    category: str
    run_key: str
    model: str
    frame: int
    timestamp_seconds: float
    playback_clock: str
    timecode_30base: str
    detection_ids: list[int]
    scores: list[float]
    component_counts: list[int]
    hole_counts: list[int]
    secondary_ratios: list[float]
    disposition: str | None
    final_track_id: str | None
    exact_keyframe: bool | None
    mask_iou: float | None = None
    smaller_mask_containment: float | None = None
    bbox_iou: float | None = None
    current_score_gate_survivors: int | None = None
    review_question: str = ""
    png: str = ""


def _tc(frame: int) -> str:
    # Editing UIs normally display this 29.97 stream with a nominal 30-frame
    # counter.  Keep the exact frame number in every artifact as the authority.
    value = int(frame)
    ff = value % 30
    seconds = value // 30
    ss = seconds % 60
    minutes = seconds // 60
    mm = minutes % 60
    hh = minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _clock(frame: int) -> str:
    total = float(frame) / FPS
    hh = int(total // 3600)
    total -= hh * 3600
    mm = int(total // 60)
    total -= mm * 60
    return f"{hh:02d}:{mm:02d}:{total:06.3f}"


def _open_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _run_metadata(topology: sqlite3.Connection) -> dict[str, dict[str, object]]:
    return {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs ORDER BY run_key")
    }


def _contours(
    inference: sqlite3.Connection,
    topology: sqlite3.Connection,
    run_key: str,
    detection_id: int,
) -> list[tuple[int, str, float, np.ndarray]]:
    role_rows = {
        int(row["polygon_index"]): (str(row["role"]), float(row["absolute_area"]))
        for row in topology.execute(
            """SELECT polygon_index,role,absolute_area FROM contour_topology
               WHERE run_key=? AND detection_id=? ORDER BY polygon_index""",
            (run_key, detection_id),
        )
    }
    grouped: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in inference.execute(
        """SELECT p.polygon_index,pt.x,pt.y
           FROM segmentation_polygons p
           JOIN segmentation_points pt ON pt.polygon_id=p.id
           WHERE p.detection_id=? ORDER BY p.polygon_index,pt.point_index""",
        (detection_id,),
    ):
        grouped[int(row["polygon_index"])].append((float(row["x"]), float(row["y"])))
    output = []
    for index, points in sorted(grouped.items()):
        role, area = role_rows.get(index, ("foreground", 0.0))
        output.append((index, role, area, np.asarray(points, dtype=np.float32)))
    return output


def _mask_on_roi(
    contours: list[tuple[int, str, float, np.ndarray]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    foreground = [
        np.rint(points - np.asarray([x0, y0])).astype(np.int32)
        for _index, role, _area, points in contours
        if role == "foreground" and len(points) >= 3
    ]
    holes = [
        np.rint(points - np.asarray([x0, y0])).astype(np.int32)
        for _index, role, _area, points in contours
        if role == "hole" and len(points) >= 3
    ]
    if foreground:
        cv2.fillPoly(mask, foreground, 1)
    if holes:
        cv2.fillPoly(mask, holes, 0)
    return mask


def _bounds(all_contours: list[list[tuple[int, str, float, np.ndarray]]]) -> tuple[int, int, int, int]:
    points = np.concatenate(
        [entry[3] for contours in all_contours for entry in contours if len(entry[3])],
        axis=0,
    )
    x0, y0 = np.floor(np.min(points, axis=0)).astype(int)
    x1, y1 = np.ceil(np.max(points, axis=0)).astype(int)
    margin = max(12, int(round(0.15 * max(x1 - x0, y1 - y0, 1))))
    return max(0, x0 - margin), max(0, y0 - margin), x1 + margin + 1, y1 + margin + 1


def _put(image: np.ndarray, text: str, y: int, scale: float = 0.54) -> None:
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 240, 245), 1, cv2.LINE_AA)


def _render_case(
    path: Path,
    case: Case,
    contour_groups: list[list[tuple[int, str, float, np.ndarray]]],
) -> None:
    width, height = 1100, 700
    header = 112
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    bounds = _bounds(contour_groups)
    x0, y0, x1, y1 = bounds
    roi_w, roi_h = max(1, x1 - x0), max(1, y1 - y0)
    scale = min((width - 70) / roi_w, (height - header - 45) / roi_h)
    offset = np.asarray(
        [(width - roi_w * scale) * 0.5, header + (height - header - roi_h * scale) * 0.5],
        dtype=np.float32,
    )
    palettes = [
        [(230, 85, 60), (80, 210, 255), (130, 230, 90), (225, 80, 210)],
        [(60, 90, 240), (60, 220, 180), (200, 210, 70), (200, 90, 220)],
    ]
    if len(contour_groups) == 2:
        masks = [_mask_on_roi(contours, bounds) for contours in contour_groups]
        overlap = (masks[0] & masks[1]).astype(bool)
        colors = [(235, 105, 55), (70, 100, 240)]
        roi = np.full((roi_h, roi_w, 3), 20, dtype=np.uint8)
        for mask, color in zip(masks, colors, strict=True):
            roi[mask.astype(bool)] = np.maximum(roi[mask.astype(bool)], np.asarray(color, np.uint8))
        roi[overlap] = (235, 235, 235)
        resized = cv2.resize(roi, (round(roi_w * scale), round(roi_h * scale)), interpolation=cv2.INTER_NEAREST)
        ox, oy = np.rint(offset).astype(int)
        image[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    else:
        for group_index, contours in enumerate(contour_groups):
            fg_index = 0
            for _polygon_index, role, _area, points in contours:
                transformed = (points - np.asarray([x0, y0])) * scale + offset
                polygon = np.rint(transformed).astype(np.int32).reshape((-1, 1, 2))
                if role == "foreground":
                    color = palettes[group_index % len(palettes)][fg_index % 4]
                    overlay = image.copy()
                    cv2.fillPoly(overlay, [polygon], color)
                    cv2.addWeighted(overlay, 0.70, image, 0.30, 0.0, dst=image)
                    cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
                    fg_index += 1
                else:
                    cv2.fillPoly(image, [polygon], (20, 20, 20))
                    cv2.polylines(image, [polygon], True, (255, 220, 60), 4, cv2.LINE_AA)
    _put(image, f"{case.case_id}  {case.category}  {case.model}", 28, 0.58)
    _put(image, f"frame={case.frame}  clock={case.playback_clock}  TC30={case.timecode_30base}  detections={case.detection_ids}", 54)
    _put(image, f"components={case.component_counts} holes={case.hole_counts} ratios={[round(x,5) for x in case.secondary_ratios]}", 80)
    if case.mask_iou is not None:
        _put(image, f"mask IoU={case.mask_iou:.4f}  smaller containment={case.smaller_mask_containment:.4f}  bbox IoU={case.bbox_iou:.4f}", 106)
    else:
        _put(image, "colors=independent foreground components; yellow outline=hole", 106)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def _bbox_iou(a: sqlite3.Row, b: sqlite3.Row) -> float:
    ix = max(0.0, min(float(a["x2"]), float(b["x2"])) - max(float(a["x1"]), float(b["x1"])))
    iy = max(0.0, min(float(a["y2"]), float(b["y2"])) - max(float(a["y1"]), float(b["y1"])))
    intersection = ix * iy
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _duplicate_pairs(
    inference: sqlite3.Connection,
    topology: sqlite3.Connection,
    run_key: str,
) -> list[tuple[float, float, float, sqlite3.Row, sqlite3.Row]]:
    rows = list(
        inference.execute(
            """SELECT f.frame_index AS frame,d.id,d.score,d.x1,d.y1,d.x2,d.y2
               FROM detections d JOIN frames f ON f.id=d.frame_id
               JOIN segmentations s ON s.detection_id=d.id
               ORDER BY f.frame_index,d.id"""
        )
    )
    by_frame: dict[int, list[sqlite3.Row]] = collections.defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    output = []
    contour_cache: dict[int, list[tuple[int, str, float, np.ndarray]]] = {}
    for frame_rows in by_frame.values():
        if len(frame_rows) < 2:
            continue
        for left_index, left in enumerate(frame_rows):
            for right in frame_rows[left_index + 1:]:
                bbox_iou = _bbox_iou(left, right)
                if bbox_iou < 0.03:
                    continue
                lid, rid = int(left["id"]), int(right["id"])
                contour_cache.setdefault(lid, _contours(inference, topology, run_key, lid))
                contour_cache.setdefault(rid, _contours(inference, topology, run_key, rid))
                bounds = _bounds([contour_cache[lid], contour_cache[rid]])
                lm = _mask_on_roi(contour_cache[lid], bounds)
                rm = _mask_on_roi(contour_cache[rid], bounds)
                intersection = int(np.count_nonzero(lm & rm))
                union = int(np.count_nonzero(lm | rm))
                smaller = min(int(lm.sum()), int(rm.sum()))
                mask_iou = intersection / union if union else 0.0
                containment = intersection / smaller if smaller else 0.0
                if mask_iou >= 0.20 or containment >= 0.80:
                    output.append((containment, mask_iou, bbox_iou, left, right))
    output.sort(key=lambda value: (-value[0], -value[1], int(value[3]["frame"])))
    return output


def build(args: argparse.Namespace) -> dict[str, object]:
    experiment = args.experiment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology = _open_ro(experiment / "topology.sqlite")
    trace = _open_ro(experiment / "postprocess_trace.sqlite")
    metadata = _run_metadata(topology)
    inference_connections: dict[str, sqlite3.Connection] = {}

    def inference(run_key: str) -> sqlite3.Connection:
        if run_key not in inference_connections:
            inference_connections[run_key] = _open_ro(Path(str(metadata[run_key]["inference_sqlite"])))
        return inference_connections[run_key]

    def outcome(run_key: str, detection_id: int) -> tuple[str | None, str | None, bool | None]:
        row = trace.execute(
            """SELECT disposition,final_track_id,exact_keyframe
               FROM detection_outcomes WHERE run_key=? AND detection_id=?""",
            (run_key, detection_id),
        ).fetchone()
        return (
            (None, None, None)
            if row is None
            else (str(row["disposition"]), None if row["final_track_id"] is None else str(row["final_track_id"]), bool(row["exact_keyframe"]))
        )

    cases: list[Case] = []
    rendered: dict[str, list[list[tuple[int, str, float, np.ndarray]]]] = {}
    selected: set[tuple[str, int]] = set()

    def add_topology(category: str, rows: list[sqlite3.Row], question: str) -> None:
        for row in rows:
            run_key = str(row["run_key"])
            did = int(row["detection_id"])
            if (run_key, did) in selected:
                continue
            selected.add((run_key, did))
            disp, track, exact = outcome(run_key, did)
            case_id = f"Q{len(cases)+1:02d}"
            case = Case(
                case_id=case_id,
                category=category,
                run_key=run_key,
                model=str(metadata[run_key]["model_key"]),
                frame=int(row["frame"]),
                timestamp_seconds=float(row["frame"]) / FPS,
                playback_clock=_clock(int(row["frame"])),
                timecode_30base=_tc(int(row["frame"])),
                detection_ids=[did],
                scores=[float(row["score"] or 0.0)],
                component_counts=[int(row["foreground_component_count"])],
                hole_counts=[int(row["hole_count"])],
                secondary_ratios=[float(row["second_to_largest_ratio"])],
                disposition=disp,
                final_track_id=track,
                exact_keyframe=exact,
                review_question=question,
            )
            cases.append(case)
            rendered[case_id] = [_contours(inference(run_key), topology, run_key, did)]

    # V3-lite KPI: visible and tiny islands, explicitly separated.
    rows = list(topology.execute(
        """SELECT * FROM mask_topology WHERE run_key='v3lite__kpi_2025_12'
           AND foreground_component_count>1 ORDER BY second_to_largest_ratio DESC,frame LIMIT 8"""
    ))
    add_topology("intra_instance_large_island", rows, "Should the secondary foreground be removed, merged, or retained?")
    rows = list(topology.execute(
        """SELECT * FROM mask_topology WHERE run_key='v3lite__kpi_2025_12'
           AND foreground_component_count>1 AND second_to_largest_ratio<0.001
           ORDER BY frame LIMIT 4"""
    ))
    add_topology("intra_instance_tiny_island", rows, "Is this tiny component always safe to discard?")
    # Severe V3-lite examples from other videos ensure a KPI-only rule does
    # not hide the genuinely large secondary components seen in the matrix.
    rows = list(topology.execute(
        """SELECT m.* FROM mask_topology m JOIN audit_runs a USING(run_key)
           WHERE a.model_key='v3lite'
             AND a.video_slug!='heyzo_3545_30_45_duplicate'
             AND m.foreground_component_count>1
           ORDER BY m.second_to_largest_ratio DESC,m.frame LIMIT 6"""
    ))
    add_topology("severe_v3lite_island", rows, "Is the secondary component a model artifact, or can it represent a legitimate disconnected target part?")
    # Long-lived islands need a different decision from one-frame speckles.
    persistent_rows = []
    for run in trace.execute(
        """SELECT * FROM multi_component_runs
           WHERE run_key='v3lite__kpi_2025_12'
           ORDER BY frame_count DESC,maximum_second_ratio DESC LIMIT 6"""
    ):
        candidate_ids = [
            int(row[0])
            for row in trace.execute(
                """SELECT detection_id FROM detection_outcomes
                   WHERE run_key=? AND final_track_id=?
                     AND frame BETWEEN ? AND ?""",
                (run["run_key"], run["final_track_id"], run["start_frame"], run["end_frame"]),
            )
        ]
        if not candidate_ids:
            continue
        placeholders = ",".join("?" for _ in candidate_ids)
        row = topology.execute(
            f"""SELECT * FROM mask_topology
                WHERE run_key=? AND detection_id IN ({placeholders})
                  AND foreground_component_count>1
                ORDER BY second_to_largest_ratio DESC LIMIT 1""",
            (run["run_key"], *candidate_ids),
        ).fetchone()
        if row is not None:
            persistent_rows.append(row)
    add_topology("persistent_v3lite_island", persistent_rows, "This island persists across frames: remove it, preserve it, or require stronger spatial evidence?")
    # Rare V3 islands across the broader matrix, to avoid overfitting cleanup to lite.
    rows = list(topology.execute(
        """SELECT m.* FROM mask_topology m JOIN audit_runs a USING(run_key)
           WHERE a.model_key='v3' AND m.foreground_component_count>1
           ORDER BY m.second_to_largest_ratio DESC,m.frame LIMIT 5"""
    ))
    add_topology("rare_v3_island", rows, "Would the same cleanup rule be safe for the larger V3 model?")
    # Holes must not be confused with disconnected foreground islands.
    rows = list(topology.execute(
        """SELECT * FROM mask_topology WHERE run_key='v3__kpi_2025_12'
           AND hole_count>0 ORDER BY largest_hole_to_outer_ratio DESC"""
    ))
    add_topology("hole_not_island", rows, "This yellow contour is a hole: should it remain a hole rather than become a second mask slot?")
    rows = list(topology.execute(
        """SELECT * FROM mask_topology WHERE run_key='v3lite__kpi_2025_12'
           AND hole_count>0 ORDER BY largest_hole_to_outer_ratio DESC LIMIT 3"""
    ))
    add_topology("hole_not_island", rows, "This yellow contour is a hole: should it remain a hole rather than become a second mask slot?")

    # Cross-instance duplicates are a different operation from island cleanup.
    for run_key in ("v3__kpi_2025_12", "v3lite__kpi_2025_12"):
        pairs = _duplicate_pairs(inference(run_key), topology, run_key)
        # Always retain the user's reported neighbourhood for V3, then add the
        # strongest distinct examples in the same run.
        chosen = []
        if run_key == "v3__kpi_2025_12":
            chosen.extend(pair for pair in pairs if int(pair[3]["frame"]) == 19251)
        for pair in pairs:
            if pair in chosen:
                continue
            if len(chosen) >= 4:
                break
            chosen.append(pair)
        for containment, mask_iou, bbox_iou, left, right in chosen[:4]:
            ids = [int(left["id"]), int(right["id"])]
            info = [
                topology.execute(
                    "SELECT * FROM mask_topology WHERE run_key=? AND detection_id=?",
                    (run_key, did),
                ).fetchone()
                for did in ids
            ]
            outcomes = [outcome(run_key, did) for did in ids]
            case_id = f"Q{len(cases)+1:02d}"
            score_survivors = sum(float(row["score"]) >= 0.6 for row in (left, right))
            case = Case(
                case_id=case_id,
                category="cross_instance_duplicate",
                run_key=run_key,
                model=str(metadata[run_key]["model_key"]),
                frame=int(left["frame"]),
                timestamp_seconds=float(left["frame"]) / FPS,
                playback_clock=_clock(int(left["frame"])),
                timecode_30base=_tc(int(left["frame"])),
                detection_ids=ids,
                scores=[float(left["score"]), float(right["score"])],
                component_counts=[int(row["foreground_component_count"]) for row in info],
                hole_counts=[int(row["hole_count"]) for row in info],
                secondary_ratios=[float(row["second_to_largest_ratio"]) for row in info],
                disposition=" / ".join(str(value[0]) for value in outcomes),
                final_track_id=" / ".join(str(value[1]) for value in outcomes),
                exact_keyframe=any(bool(value[2]) for value in outcomes),
                mask_iou=float(mask_iou),
                smaller_mask_containment=float(containment),
                bbox_iou=float(bbox_iou),
                current_score_gate_survivors=score_survivors,
                review_question="Are these two detections the same physical instance, and should only the higher-score one survive?",
            )
            cases.append(case)
            rendered[case_id] = [
                _contours(inference(run_key), topology, run_key, did) for did in ids
            ]

    for case in cases:
        filename = f"{case.case_id}_{case.category}_{case.model}_f{case.frame}.png"
        case.png = filename
        _render_case(output / filename, case, rendered[case.case_id])

    manifest = {
        "schema_version": 1,
        "privacy": "mask coordinates only; no video was decoded or opened",
        "reference_fps": FPS,
        "authoritative_locator": "run_key + exact zero-based frame + detection_id",
        "cases": [asdict(case) for case in cases],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = [asdict(case) for case in cases]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for connection in inference_connections.values():
        connection.close()
    topology.close()
    trace.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args)
    counts = collections.Counter(case["category"] for case in manifest["cases"])
    print(json.dumps({"cases": len(manifest["cases"]), "categories": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
