#!/usr/bin/env python3
"""Replay and audit the production NMS over diverse archived raw masks."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPOLOGY = ROOT / "output/instance_mask_topology_20260806/topology.sqlite"
DEFAULT_OUTPUT = ROOT / "output/nms_ablation_20260813"


@dataclass
class Detection:
    detection_id: int
    frame: int
    detection_index: int
    score: float
    class_name: str
    bbox: tuple[float, float, float, float]
    bbox_area: float
    naive_mask_area: float
    contour_bboxes: list[tuple[float, float, float, float]]

    @property
    def size_ref(self) -> float:
        return min(self.bbox_area, self.naive_mask_area) if self.naive_mask_area > 0 else self.bbox_area


@dataclass
class Suppression:
    run_key: str
    frame: int
    suppressor: Detection
    suppressed: Detection
    reason: str
    threshold: float
    contain_limit: float
    bbox_iou: float
    bbox_area_ratio: float


def open_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def bbox_area(box: tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    if intersection <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_contains(a: tuple[float, ...], b: tuple[float, ...], margin: float = 2.0) -> bool:
    return (
        a[0] - margin <= b[0]
        and a[1] - margin <= b[1]
        and a[2] + margin >= b[2]
        and a[3] + margin >= b[3]
    )


def contour_inside_bbox(
    contour_boxes: Iterable[tuple[float, ...]], box: tuple[float, ...], margin: float = 2.0
) -> bool:
    return any(bbox_contains(box, contour, margin) for contour in contour_boxes)


def contained_pair(first: Detection, second: Detection) -> bool:
    return (
        bbox_contains(first.bbox, second.bbox)
        or bbox_contains(second.bbox, first.bbox)
        or contour_inside_bbox(second.contour_bboxes, first.bbox)
        or contour_inside_bbox(first.contour_bboxes, second.bbox)
    )


def thresholds(area: float) -> tuple[float, float]:
    if area <= 2000.0:
        return 0.05, 5.0
    if area <= 5000.0:
        return 0.10, 5.0
    return 0.20, 8.0


def replay(
    run_key: str,
    detections: list[Detection],
    *,
    containment: bool,
    class_aware: bool,
    record_events: bool = False,
) -> tuple[list[int], list[Suppression]]:
    order = sorted(range(len(detections)), key=lambda i: (-detections[i].score, i))
    suppressed: set[int] = set()
    retained: list[int] = []
    events: list[Suppression] = []
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        retained.append(index)
        first = detections[index]
        for other in order[position + 1 :]:
            if other in suppressed:
                continue
            second = detections[other]
            if class_aware and first.class_name != second.class_name:
                continue
            threshold, contain_limit = thresholds(min(first.size_ref, second.size_ref))
            area_min = min(first.bbox_area, second.bbox_area)
            area_max = max(first.bbox_area, second.bbox_area)
            ratio = area_max / area_min if area_min > 0 else math.inf
            overlap = bbox_iou(first.bbox, second.bbox)
            is_contained = containment and contained_pair(first, second)
            reason = ""
            if is_contained and area_min > 0 and ratio <= contain_limit:
                reason = "containment"
            elif overlap >= threshold:
                reason = "bbox_iou"
            if not reason:
                continue
            suppressed.add(other)
            if record_events:
                events.append(
                    Suppression(
                        run_key=run_key,
                        frame=first.frame,
                        suppressor=first,
                        suppressed=second,
                        reason=reason,
                        threshold=threshold,
                        contain_limit=contain_limit,
                        bbox_iou=overlap,
                        bbox_area_ratio=ratio,
                    )
                )
    return retained, events


def frame_groups(rows: Iterable[sqlite3.Row]) -> Iterator[tuple[int, list[sqlite3.Row]]]:
    frame = None
    group: list[sqlite3.Row] = []
    for row in rows:
        value = int(row["frame"])
        if frame is not None and value != frame:
            yield frame, group
            group = []
        frame = value
        group.append(row)
    if frame is not None:
        yield frame, group


def load_contour_metadata(
    connection: sqlite3.Connection, run_key: str
) -> tuple[dict[int, list[tuple[float, ...]]], dict[int, float]]:
    result: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    areas: dict[int, float] = defaultdict(float)
    for row in connection.execute(
        """SELECT detection_id,x1,y1,x2,y2,absolute_area FROM contour_topology
           WHERE run_key=? ORDER BY detection_id,polygon_index""",
        (run_key,),
    ):
        detection_id = int(row["detection_id"])
        result[detection_id].append(
            (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        )
        areas[detection_id] += float(row["absolute_area"])
    return result, dict(areas)


def make_detection(
    row: sqlite3.Row,
    contour_boxes: dict[int, list[tuple[float, ...]]],
    naive_areas: dict[int, float] | None = None,
) -> Detection:
    box = tuple(float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    return Detection(
        detection_id=int(row["detection_id"]),
        frame=int(row["frame"]),
        detection_index=int(row["detection_index"] or 0),
        score=float(row["score"] or 0.0),
        class_name=str(row["class_name"] or "unknown"),
        bbox=box,
        bbox_area=bbox_area(box),
        # Production sums absolute polygon areas, including holes.  The topology
        # aggregate is therefore intentionally not the net foreground area.
        naive_mask_area=float((naive_areas or {}).get(int(row["detection_id"]), row["net_foreground_area"])),
        contour_bboxes=contour_boxes.get(int(row["detection_id"]), []),
    )


def create_event_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE events(
          event_id INTEGER PRIMARY KEY,
          run_key TEXT NOT NULL, frame INTEGER NOT NULL,
          suppressor_id INTEGER NOT NULL, suppressed_id INTEGER NOT NULL,
          suppressor_class TEXT NOT NULL, suppressed_class TEXT NOT NULL,
          suppressor_score REAL NOT NULL, suppressed_score REAL NOT NULL,
          reason TEXT NOT NULL, adaptive_threshold REAL NOT NULL,
          contain_limit REAL NOT NULL, bbox_iou REAL NOT NULL,
          bbox_area_ratio REAL NOT NULL,
          mask_iou REAL, smaller_mask_containment REAL,
          suppressor_coverage REAL, suppressed_coverage REAL,
          suppressor_mask_area INTEGER, suppressed_mask_area INTEGER,
          mask_iou_would_suppress INTEGER,
          audit_label TEXT, audit_reason TEXT
        );
        CREATE INDEX idx_events_run_frame ON events(run_key,frame);
        CREATE INDEX idx_events_label ON events(audit_label);
        """
    )
    return connection


def insert_events(connection: sqlite3.Connection, events: list[Suppression]) -> None:
    connection.executemany(
        """INSERT INTO events(
          run_key,frame,suppressor_id,suppressed_id,suppressor_class,suppressed_class,
          suppressor_score,suppressed_score,reason,adaptive_threshold,contain_limit,
          bbox_iou,bbox_area_ratio)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                event.run_key,
                event.frame,
                event.suppressor.detection_id,
                event.suppressed.detection_id,
                event.suppressor.class_name,
                event.suppressed.class_name,
                event.suppressor.score,
                event.suppressed.score,
                event.reason,
                event.threshold,
                event.contain_limit,
                event.bbox_iou,
                event.bbox_area_ratio,
            )
            for event in events
        ],
    )


def metadata_phase(topology_path: Path, event_db: Path) -> dict[str, object]:
    topology = open_ro(topology_path)
    output = create_event_db(event_db)
    runs = list(topology.execute("SELECT * FROM audit_runs ORDER BY model_key,run_key"))
    per_run: list[dict[str, object]] = []
    totals = Counter()
    started = time.perf_counter()
    for run_index, run in enumerate(runs, 1):
        run_key = str(run["run_key"])
        contour_boxes, naive_areas = load_contour_metadata(topology, run_key)
        rows = topology.execute(
            """SELECT * FROM mask_topology WHERE run_key=?
               ORDER BY frame,detection_index,detection_id""",
            (run_key,),
        )
        counters = Counter()
        run_events: list[Suppression] = []
        for _, frame_rows in frame_groups(rows):
            detections = [make_detection(row, contour_boxes, naive_areas) for row in frame_rows]
            count = len(detections)
            counters["raw"] += count
            counters["frames"] += 1
            if count < 2:
                counters["current"] += count
                counters["bbox_only"] += count
                counters["class_aware"] += count
                continue
            current, events = replay(
                run_key, detections, containment=True, class_aware=False, record_events=True
            )
            bbox_only, _ = replay(
                run_key, detections, containment=False, class_aware=False
            )
            class_aware, _ = replay(
                run_key, detections, containment=True, class_aware=True
            )
            counters["multi_frames"] += 1
            counters["current"] += len(current)
            counters["bbox_only"] += len(bbox_only)
            counters["class_aware"] += len(class_aware)
            run_events.extend(events)
            if len(run_events) >= 10000:
                insert_events(output, run_events)
                output.commit()
                run_events.clear()
        if run_events:
            insert_events(output, run_events)
            output.commit()
        counters["current_suppressed"] = counters["raw"] - counters["current"]
        counters["bbox_only_suppressed"] = counters["raw"] - counters["bbox_only"]
        counters["class_aware_suppressed"] = counters["raw"] - counters["class_aware"]
        item = {
            "run_key": run_key,
            "model_key": str(run["model_key"]),
            "video_slug": str(run["video_slug"]),
            **dict(counters),
        }
        per_run.append(item)
        totals.update(counters)
        print(
            f"[metadata {run_index:02d}/{len(runs)}] {run_key}: "
            f"raw={counters['raw']} current_drop={counters['current_suppressed']} "
            f"bbox_drop={counters['bbox_only_suppressed']} class_drop={counters['class_aware_suppressed']}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    output.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    output.close()
    topology.close()
    return {"elapsed_sec": elapsed, "runs": per_run, "totals": dict(totals)}


def load_polygon_groups(
    inference_path: Path,
    topology_path: Path,
    run_key: str,
    detection_ids: list[int],
) -> dict[int, list[tuple[str, int, np.ndarray]]]:
    """Load only requested polygons without modifying the source inference DB."""
    if not detection_ids:
        return {}
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("ATTACH DATABASE ? AS src", (str(inference_path),))
    connection.execute("ATTACH DATABASE ? AS topo", (str(topology_path),))
    connection.execute("CREATE TEMP TABLE wanted(id INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO wanted(id) VALUES(?)", ((value,) for value in detection_ids))
    rows = connection.execute(
        """SELECT p.detection_id,p.polygon_index,t.role,t.nesting_depth,
                  s.point_index,s.x,s.y
           FROM wanted w
           JOIN src.segmentation_polygons p ON p.detection_id=w.id
           JOIN src.segmentation_points s ON s.polygon_id=p.id
           JOIN topo.contour_topology t
             ON t.run_key=? AND t.detection_id=p.detection_id
            AND t.polygon_index=p.polygon_index
           ORDER BY p.detection_id,p.polygon_index,s.point_index""",
        (run_key,),
    )
    grouped: dict[int, list[tuple[str, int, np.ndarray]]] = defaultdict(list)
    current: tuple[int, int] | None = None
    role = "foreground"
    depth = 0
    points: list[tuple[float, float]] = []
    for row in rows:
        key = (int(row["detection_id"]), int(row["polygon_index"]))
        if current is not None and key != current:
            grouped[current[0]].append((role, depth, np.asarray(points, np.float32)))
            points = []
        current = key
        role = str(row["role"])
        depth = int(row["nesting_depth"])
        points.append((float(row["x"]), float(row["y"])))
    if current is not None:
        grouped[current[0]].append((role, depth, np.asarray(points, np.float32)))
    connection.close()
    return dict(grouped)


def pair_mask_metrics(
    first: list[tuple[str, int, np.ndarray]], second: list[tuple[str, int, np.ndarray]]
) -> tuple[float, float, float, float, int, int]:
    points = [polygon for _, _, polygon in first + second if len(polygon)]
    if not points:
        return 0.0, 0.0, 0.0, 0.0, 0, 0
    all_points = np.concatenate(points, axis=0)
    x1, y1 = np.floor(all_points.min(axis=0)).astype(int) - 2
    x2, y2 = np.ceil(all_points.max(axis=0)).astype(int) + 2
    width, height = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)

    def raster(groups: list[tuple[str, int, np.ndarray]]) -> np.ndarray:
        mask = np.zeros((height, width), np.uint8)
        for role, depth, polygon in sorted(groups, key=lambda item: item[1]):
            if len(polygon) < 3:
                continue
            contour = np.rint(polygon - (x1, y1)).astype(np.int32)
            cv2.fillPoly(mask, [contour], 1 if role == "foreground" else 0)
        return mask

    a, b = raster(first), raster(second)
    area_a, area_b = int(a.sum()), int(b.sum())
    intersection = int(np.logical_and(a, b).sum())
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    smaller = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    cover_a = intersection / area_a if area_a else 0.0
    cover_b = intersection / area_b if area_b else 0.0
    return iou, smaller, cover_a, cover_b, area_a, area_b


def audit_label(
    mask_iou: float,
    smaller_containment: float,
    area_ratio: float,
    same_class: bool,
) -> tuple[str, str]:
    # Conservative labels: the middle region is intentionally left ambiguous.
    # The raw inference schema calls every detection `foreground`; therefore
    # same_class is not evidence that two masks represent the same semantic
    # instance.  Only strong actual-mask overlap is auto-labelled beneficial.
    if mask_iou >= 0.70:
        return "likely_beneficial", "near_duplicate_masks"
    if mask_iou <= 0.25 and (not same_class or area_ratio >= 2.5):
        return "likely_harmful", "merged_or_distinct_masks"
    return "ambiguous", "requires_visual_or_gt_review"


def exact_mask_phase(
    topology_path: Path, event_db: Path, run_rows: list[dict[str, object]]
) -> dict[str, object]:
    topology = open_ro(topology_path)
    event_connection = sqlite3.connect(event_db)
    event_connection.row_factory = sqlite3.Row
    run_meta = {str(row["run_key"]): row for row in topology.execute("SELECT * FROM audit_runs")}
    counts = Counter()
    per_run: dict[str, Counter] = defaultdict(Counter)
    started = time.perf_counter()
    for index, run in enumerate(run_rows, 1):
        run_key = str(run["run_key"])
        events = list(event_connection.execute("SELECT * FROM events WHERE run_key=?", (run_key,)))
        ids = sorted({int(row[key]) for row in events for key in ("suppressor_id", "suppressed_id")})
        groups = load_polygon_groups(
            Path(str(run_meta[run_key]["inference_sqlite"])), topology_path, run_key, ids
        )
        updates = []
        for row in events:
            first = groups.get(int(row["suppressor_id"]), [])
            second = groups.get(int(row["suppressed_id"]), [])
            iou, smaller, cover_a, cover_b, area_a, area_b = pair_mask_metrics(first, second)
            label, reason = audit_label(
                iou,
                smaller,
                float(row["bbox_area_ratio"]),
                str(row["suppressor_class"]) == str(row["suppressed_class"]),
            )
            would = int(iou >= float(row["adaptive_threshold"]))
            updates.append(
                (
                    iou,
                    smaller,
                    cover_a,
                    cover_b,
                    area_a,
                    area_b,
                    would,
                    label,
                    reason,
                    int(row["event_id"]),
                )
            )
            counts[label] += 1
            counts["mask_iou_would_suppress" if would else "mask_iou_would_keep"] += 1
            per_run[run_key][label] += 1
        event_connection.executemany(
            """UPDATE events SET mask_iou=?,smaller_mask_containment=?,
               suppressor_coverage=?,suppressed_coverage=?,suppressor_mask_area=?,
               suppressed_mask_area=?,mask_iou_would_suppress=?,audit_label=?,audit_reason=?
               WHERE event_id=?""",
            updates,
        )
        event_connection.commit()
        print(
            f"[masks {index:02d}/{len(run_rows)}] {run_key}: events={len(events)} "
            f"beneficial={per_run[run_key]['likely_beneficial']} "
            f"harmful={per_run[run_key]['likely_harmful']}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    event_connection.close()
    topology.close()
    return {
        "elapsed_sec": elapsed,
        "counts": dict(counts),
        "per_run": {key: dict(value) for key, value in per_run.items()},
    }


PALETTE = [
    (40, 70, 255), (255, 120, 30), (40, 220, 90), (220, 70, 220),
    (20, 210, 245), (230, 190, 40), (80, 150, 255), (190, 100, 255),
]


def fill_groups(shape: tuple[int, int], groups: list[tuple[str, int, np.ndarray]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for role, depth, polygon in sorted(groups, key=lambda item: item[1]):
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1 if role == "foreground" else 0)
    return mask


def put(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.54) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def panel(
    image: np.ndarray,
    title: str,
    ids: list[int],
    groups: dict[int, list[tuple[str, int, np.ndarray]]],
    labels: dict[int, str],
    colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    for detection_id in ids:
        mask = fill_groups(image.shape[:2], groups.get(detection_id, [])).astype(bool)
        overlay[mask] = colors[detection_id]
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)
    for detection_id in ids:
        for role, _, polygon in groups.get(detection_id, []):
            if len(polygon) >= 3:
                cv2.polylines(
                    canvas, [np.rint(polygon).astype(np.int32)], True,
                    colors[detection_id], 3 if role == "foreground" else 2, cv2.LINE_AA,
                )
    header = np.full((92, canvas.shape[1], 3), 15, np.uint8)
    put(header, title, (14, 28), 0.62)
    put(header, " | ".join(labels[value] for value in ids)[:190], (14, 62), 0.43)
    return np.vstack([header, canvas])


def seek_frame(video: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"failed to decode {video} frame {frame}")
    return image


def selected_events(connection: sqlite3.Connection, label: str, limit: int) -> list[sqlite3.Row]:
    # Diversity first: one high-confidence example per run, then fill remaining slots.
    rows = list(
        connection.execute(
            """SELECT *,abs(suppressor_score-suppressed_score) AS score_gap
               FROM events WHERE audit_label=?
               ORDER BY run_key,
                 CASE WHEN ?='likely_beneficial' THEN mask_iou ELSE -mask_iou END DESC,
                 score_gap ASC""",
            (label, label),
        )
    )
    chosen: list[sqlite3.Row] = []
    seen_runs: set[str] = set()
    seen_frames: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["run_key"]), int(row["frame"]))
        if str(row["run_key"]) not in seen_runs and key not in seen_frames:
            chosen.append(row)
            seen_runs.add(str(row["run_key"]))
            seen_frames.add(key)
            if len(chosen) >= limit:
                return chosen
    for row in rows:
        key = (str(row["run_key"]), int(row["frame"]))
        if key not in seen_frames:
            chosen.append(row)
            seen_frames.add(key)
            if len(chosen) >= limit:
                break
    return chosen


def render_phase(
    topology_path: Path, event_db: Path, output_dir: Path, samples_per_label: int
) -> list[dict[str, object]]:
    topology = open_ro(topology_path)
    events = sqlite3.connect(event_db)
    events.row_factory = sqlite3.Row
    run_meta = {str(row["run_key"]): row for row in topology.execute("SELECT * FROM audit_runs")}
    manifest: list[dict[str, object]] = []
    for audit_label in ("likely_beneficial", "likely_harmful", "ambiguous"):
        folder = output_dir / "overlays" / audit_label
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.jpg"):
            stale.unlink()
        for number, event in enumerate(selected_events(events, audit_label, samples_per_label), 1):
            run_key, frame = str(event["run_key"]), int(event["frame"])
            frame_rows = list(
                topology.execute(
                    "SELECT * FROM mask_topology WHERE run_key=? AND frame=? ORDER BY detection_index,detection_id",
                    (run_key, frame),
                )
            )
            contour_boxes, naive_areas = load_contour_boxes_for_ids(
                topology, run_key, [int(row["detection_id"]) for row in frame_rows]
            )
            detections = [make_detection(row, contour_boxes, naive_areas) for row in frame_rows]
            retained, _ = replay(run_key, detections, containment=True, class_aware=False)
            ids = [item.detection_id for item in detections]
            groups = load_polygon_groups(
                Path(str(run_meta[run_key]["inference_sqlite"])), topology_path, run_key, ids
            )
            image = seek_frame(Path(str(run_meta[run_key]["input_video"])), frame)
            labels = {
                item.detection_id: f"D{item.detection_id}:{item.class_name}:{item.score:.3f}"
                for item in detections
            }
            colors = {value: PALETTE[index % len(PALETTE)] for index, value in enumerate(ids)}
            current_ids = [detections[index].detection_id for index in retained]
            pair_ids = [int(event["suppressor_id"]), int(event["suppressed_id"])]
            no_nms = panel(image, "NO NMS: all AI raw instances", ids, groups, labels, colors)
            current = panel(image, "CURRENT NMS: retained instances", current_ids, groups, labels, colors)
            comparison = np.concatenate([no_nms, current], axis=1)
            path = folder / f"sample_{number:02d}_{run_key}_f{frame}.jpg"
            cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
            manifest.append(
                {
                    "audit_label": audit_label,
                    "run_key": run_key,
                    "model_key": str(run_meta[run_key]["model_key"]),
                    "video_slug": str(run_meta[run_key]["video_slug"]),
                    "frame": frame,
                    "suppressor_id": pair_ids[0],
                    "suppressed_id": pair_ids[1],
                    "reason": str(event["reason"]),
                    "bbox_iou": float(event["bbox_iou"]),
                    "mask_iou": float(event["mask_iou"]),
                    "smaller_mask_containment": float(event["smaller_mask_containment"]),
                    "path": str(path),
                }
            )
    events.close()
    topology.close()
    return manifest


def load_contour_boxes_for_ids(
    connection: sqlite3.Connection, run_key: str, detection_ids: list[int]
) -> tuple[dict[int, list[tuple[float, ...]]], dict[int, float]]:
    if not detection_ids:
        return {}, {}
    placeholders = ",".join("?" for _ in detection_ids)
    result: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    areas: dict[int, float] = defaultdict(float)
    for row in connection.execute(
        f"""SELECT detection_id,x1,y1,x2,y2,absolute_area FROM contour_topology
             WHERE run_key=? AND detection_id IN ({placeholders})
             ORDER BY detection_id,polygon_index""",
        (run_key, *detection_ids),
    ):
        detection_id = int(row["detection_id"])
        result[detection_id].append(
            (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        )
        areas[detection_id] += float(row["absolute_area"])
    return result, dict(areas)


def summarize_events(event_db: Path) -> dict[str, object]:
    connection = sqlite3.connect(event_db)
    connection.row_factory = sqlite3.Row
    result: dict[str, object] = {}
    result["by_label"] = {
        str(row["audit_label"]): int(row["n"])
        for row in connection.execute(
            "SELECT audit_label,count(*) n FROM events GROUP BY audit_label"
        )
    }
    result["by_reason"] = {
        str(row["reason"]): int(row["n"])
        for row in connection.execute("SELECT reason,count(*) n FROM events GROUP BY reason")
    }
    result["by_label_reason"] = [
        dict(row)
        for row in connection.execute(
            """SELECT audit_label,reason,count(*) n,avg(bbox_iou) avg_bbox_iou,
                      avg(mask_iou) avg_mask_iou,avg(bbox_area_ratio) avg_bbox_area_ratio
               FROM events GROUP BY audit_label,reason ORDER BY audit_label,reason"""
        )
    ]
    result["cross_class"] = [
        dict(row)
        for row in connection.execute(
            """SELECT audit_label,count(*) n FROM events
               WHERE suppressor_class<>suppressed_class GROUP BY audit_label"""
        )
    ]
    connection.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--samples-per-label", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    event_db = args.output / "nms_events.sqlite"
    metadata = metadata_phase(args.topology, event_db)
    summary: dict[str, object] = {"metadata": metadata}
    (args.output / "metadata_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.metadata_only:
        exact = exact_mask_phase(args.topology, event_db, list(metadata["runs"]))
        manifest = render_phase(
            args.topology, event_db, args.output, args.samples_per_label
        )
        summary.update(
            {"exact_masks": exact, "events": summarize_events(event_db), "overlay_manifest": manifest}
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
