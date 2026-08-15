#!/usr/bin/env python3
"""Render deterministic two-panel human-review galleries for NMS candidate v2.

Only local OpenCV decoding is used.  Source frames are never uploaded and the
script has no network path.  Each image has exactly two result panels:

* left: legacy Production NMS;
* right: component-aware candidate v2.

Retained masks are filled.  Input instances suppressed by an arm are retained
as dashed outlines so a reviewer can still see which same-colour ID vanished.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import iter_detection_records  # noqa: E402
from nms.adaptive import (  # noqa: E402
    AdaptiveNms,
    _bbox_area,
    _bbox_iou,
    _contained_pair,
    _mask_area,
)
from nms.component_aware import (  # noqa: E402
    _overlap_slices,
    _raster_mask,
    exact_mask_iou,
)
from nms.components import fill_holes_and_remove_tiny_islands  # noqa: E402
from experimental.nms_ablation_20260813.render_conservative_island_policy_audit import (  # noqa: E402,E501
    seek_frame,
)


DEFAULT_ABLATION = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_ablation_20260813"
)
DEFAULT_TOPOLOGY = (
    REPOSITORY_ROOT / "output/instance_mask_topology_20260806/topology.sqlite"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_review_gallery_20260813"
)
ARM_LEGACY = "legacy"
ARM_CANDIDATE = "component_candidate_v2"
CATEGORY_ORDER = (
    "01_legacy_only_low_mask_iou",
    "02_both_suppress_high_iou",
    "03_candidate_only_suppressed",
    "04_hole_fill",
    "05_tiny_island_at_most_1pct",
    "06_redundant_island_80_50",
    "07_cross_class_or_three_detection_chain",
)
CLASS_NAMES = {
    "女性器": "female",
    "男性器": "male",
    "結合部分": "junction",
}


def _open_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _detection_id(detection: dict[str, Any], frame: int, index: int) -> int | str:
    value = detection.get("source_detection_id")
    if value is None:
        return f"synthetic:{frame}:{index}"
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _ids(detections: list[dict[str, Any]], frame: int) -> list[int | str]:
    values = [
        _detection_id(detection, frame, index)
        for index, detection in enumerate(detections)
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"frame={frame}: duplicate source_detection_id")
    return values


def _legacy_events(
    detections: list[dict[str, Any]], frame: int
) -> list[dict[str, Any]]:
    """Replay exact legacy suppression while exposing its direct pairs."""
    policy = AdaptiveNms()
    ids = _ids(detections, frame)
    bboxes = [
        tuple(map(float, detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])))
        for detection in detections
    ]
    bbox_areas = [_bbox_area(detection) for detection in detections]
    mask_areas = [_mask_area(detection) for detection in detections]
    size_refs = [
        min(bbox_area, mask_area) if mask_area > 0.0 else bbox_area
        for bbox_area, mask_area in zip(bbox_areas, mask_areas, strict=True)
    ]
    rasters = [_raster_mask(detection) for detection in detections]
    order = sorted(
        range(len(detections)),
        key=lambda index: (-float(detections[index].get("score") or 0.0), index),
    )
    suppressed: set[int] = set()
    events: list[dict[str, Any]] = []
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        for other in order[position + 1 :]:
            if other in suppressed:
                continue
            threshold, contain_limit = policy.thresholds_for_area(
                min(size_refs[index], size_refs[other])
            )
            area_min = min(bbox_areas[index], bbox_areas[other])
            area_max = max(bbox_areas[index], bbox_areas[other])
            contained = _contained_pair(
                detections[index],
                detections[other],
                bboxes[index],
                bboxes[other],
                policy.contain_margin,
            )
            bbox_iou = _bbox_iou(bboxes[index], bboxes[other])
            reason = None
            if contained and area_min > 0.0 and area_max / area_min <= contain_limit:
                reason = "legacy_containment"
            elif bbox_iou >= threshold:
                reason = "legacy_bbox_iou"
            if reason is None:
                continue
            suppressed.add(other)
            events.append(
                {
                    "winner_id": ids[index],
                    "loser_id": ids[other],
                    "winner_class": str(detections[index].get("class_name", "")),
                    "loser_class": str(detections[other].get("class_name", "")),
                    "winner_score": float(detections[index].get("score") or 0.0),
                    "loser_score": float(detections[other].get("score") or 0.0),
                    "reason": reason,
                    "bbox_iou": float(bbox_iou),
                    "mask_iou": float(exact_mask_iou(rasters[index], rasters[other])),
                }
            )
    return events


def _candidate_events(
    detections: list[dict[str, Any]], frame: int
) -> list[dict[str, Any]]:
    """Replay the candidate's whole-instance suppression pairs."""
    preprocessed, _ = fill_holes_and_remove_tiny_islands(
        detections,
        fill_all_holes=True,
        unconditional_owner_ratio_max=0.01,
    )
    ids = _ids(preprocessed, frame)
    rasters = [_raster_mask(detection) for detection in preprocessed]
    order = sorted(
        range(len(preprocessed)),
        key=lambda index: (
            -float(preprocessed[index].get("score") or 0.0),
            index,
        ),
    )
    suppressed: set[int] = set()
    events: list[dict[str, Any]] = []
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        first = rasters[index]
        if first is None or first.area <= 0:
            continue
        for other in order[position + 1 :]:
            if other in suppressed:
                continue
            second = rasters[other]
            if second is None or second.area <= 0:
                continue
            if _overlap_slices(first, second) is None:
                continue
            mask_iou = exact_mask_iou(first, second)
            if mask_iou < 0.70:
                continue
            suppressed.add(other)
            events.append(
                {
                    "winner_id": ids[index],
                    "loser_id": ids[other],
                    "winner_class": str(preprocessed[index].get("class_name", "")),
                    "loser_class": str(preprocessed[other].get("class_name", "")),
                    "winner_score": float(preprocessed[index].get("score") or 0.0),
                    "loser_score": float(preprocessed[other].get("score") or 0.0),
                    "reason": "candidate_mask_iou_at_least_0.70",
                    "mask_iou": float(mask_iou),
                }
            )
    return events


def _component_events(path: Path) -> dict[int, dict[str, int]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["frame_index"]): {
                key: int(row[key])
                for key in (
                    "holes_filled",
                    "tiny_islands_removed",
                    "redundant_islands_removed",
                )
            }
            for row in csv.DictReader(handle)
        }


def _decision_frames(
    run_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, int]]]:
    component = _component_events(run_dir / "component_events.csv")
    relevant: dict[int, dict[str, Any]] = {}
    with gzip.open(run_dir / "retained_ids.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frame = int(row["frame_index"])
            input_ids = set(row["input_ids"])
            legacy = set(row["retained_ids"][ARM_LEGACY])
            candidate = set(row["retained_ids"][ARM_CANDIDATE])
            old_only_suppressed = sorted(candidate - legacy, key=str)
            candidate_only_suppressed = sorted(legacy - candidate, key=str)
            both_suppressed = sorted(input_ids - legacy - candidate, key=str)
            if (
                old_only_suppressed
                or candidate_only_suppressed
                or both_suppressed
                or frame in component
            ):
                relevant[frame] = {
                    "input_ids": sorted(input_ids, key=str),
                    "legacy_ids": sorted(legacy, key=str),
                    "candidate_ids": sorted(candidate, key=str),
                    "legacy_only_suppressed_ids": old_only_suppressed,
                    "candidate_only_suppressed_ids": candidate_only_suppressed,
                    "both_suppressed_ids": both_suppressed,
                }
    return relevant, component


def _candidate_row(
    *,
    category: str,
    run_key: str,
    frame: int,
    decision: dict[str, Any],
    reason: str,
    event: dict[str, Any] | None = None,
    component: dict[str, int] | None = None,
    event_count: int = 0,
) -> dict[str, Any]:
    return {
        "category": category,
        "run_key": run_key,
        "frame": int(frame),
        "reason": reason,
        "input_count": len(decision["input_ids"]),
        "input_ids": decision["input_ids"],
        "legacy_ids": decision["legacy_ids"],
        "candidate_ids": decision["candidate_ids"],
        "legacy_only_suppressed_ids": decision["legacy_only_suppressed_ids"],
        "candidate_only_suppressed_ids": decision[
            "candidate_only_suppressed_ids"
        ],
        "both_suppressed_ids": decision["both_suppressed_ids"],
        "event": event,
        "component": component,
        "event_count": event_count,
    }


def _scan_candidates(
    ablation_root: Path,
    run_key: str,
    scored_jsonl: Path,
    decisions: dict[int, dict[str, Any]],
    component: dict[int, dict[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_detection_records(scored_jsonl):
        frame = int(record["frame_index"])
        decision = decisions.get(frame)
        if decision is None:
            continue
        detections = list(record["detections"])
        legacy_events = _legacy_events(detections, frame)
        candidate_events = _candidate_events(detections, frame)

        for event in legacy_events:
            if event["loser_id"] in decision["legacy_only_suppressed_ids"]:
                result["01_legacy_only_low_mask_iou"].append(
                    _candidate_row(
                        category="01_legacy_only_low_mask_iou",
                        run_key=run_key,
                        frame=frame,
                        decision=decision,
                        reason=(
                            f"legacy suppresses D{event['loser_id']} by "
                            f"{event['reason']} although Mask-IoU="
                            f"{event['mask_iou']:.4f}; candidate keeps it"
                        ),
                        event=event,
                        event_count=len(legacy_events) + len(candidate_events),
                    )
                )
        for event in candidate_events:
            if event["loser_id"] in decision["candidate_only_suppressed_ids"]:
                result["03_candidate_only_suppressed"].append(
                    _candidate_row(
                        category="03_candidate_only_suppressed",
                        run_key=run_key,
                        frame=frame,
                        decision=decision,
                        reason=(
                            f"candidate alone suppresses D{event['loser_id']} "
                            f"under Mask-IoU={event['mask_iou']:.4f}"
                        ),
                        event=event,
                        event_count=len(legacy_events) + len(candidate_events),
                    )
                )
            if event["loser_id"] in decision["both_suppressed_ids"]:
                result["02_both_suppress_high_iou"].append(
                    _candidate_row(
                        category="02_both_suppress_high_iou",
                        run_key=run_key,
                        frame=frame,
                        decision=decision,
                        reason=(
                            f"both suppress D{event['loser_id']}; candidate "
                            f"pair Mask-IoU={event['mask_iou']:.4f}"
                        ),
                        event=event,
                        event_count=len(legacy_events) + len(candidate_events),
                    )
                )

        flags = component.get(frame, {})
        for category, key, label in (
            ("04_hole_fill", "holes_filled", "candidate fills all mask holes"),
            (
                "05_tiny_island_at_most_1pct",
                "tiny_islands_removed",
                "candidate removes owner-relative island <=1%",
            ),
            (
                "06_redundant_island_80_50",
                "redundant_islands_removed",
                "candidate removes survivor island under 80% coverage / 50% area rule",
            ),
        ):
            if int(flags.get(key, 0)):
                result[category].append(
                    _candidate_row(
                        category=category,
                        run_key=run_key,
                        frame=frame,
                        decision=decision,
                        reason=f"{label}; changed_components={flags[key]}",
                        component=flags,
                        event_count=len(legacy_events) + len(candidate_events),
                    )
                )

        all_events = legacy_events + candidate_events
        cross_class = [
            event
            for event in all_events
            if event["winner_class"] != event["loser_class"]
        ]
        is_chain = len(detections) >= 3 and len(all_events) >= 2
        if cross_class or is_chain:
            selected_event = cross_class[0] if cross_class else all_events[0]
            kind = "cross-class" if cross_class else "three-detection suppression chain"
            result["07_cross_class_or_three_detection_chain"].append(
                _candidate_row(
                    category="07_cross_class_or_three_detection_chain",
                    run_key=run_key,
                    frame=frame,
                    decision=decision,
                    reason=(
                        f"{kind}; input={len(detections)} direct_events={len(all_events)}; "
                        f"D{selected_event['winner_id']} -> D{selected_event['loser_id']}"
                    ),
                    event=selected_event,
                    event_count=len(all_events),
                )
            )
    return result


def _metric(row: dict[str, Any], category: str) -> tuple[Any, ...]:
    event = row.get("event") or {}
    if category == "01_legacy_only_low_mask_iou":
        return (float(event.get("mask_iou", 1.0)), row["run_key"], row["frame"])
    if category == "02_both_suppress_high_iou":
        return (-float(event.get("mask_iou", 0.0)), row["run_key"], row["frame"])
    if category == "03_candidate_only_suppressed":
        return (row["run_key"], row["frame"], str(event.get("loser_id", "")))
    if category == "07_cross_class_or_three_detection_chain":
        cross = event.get("winner_class") != event.get("loser_class")
        return (
            0 if cross else 1,
            -int(row.get("input_count", 0)),
            -int(row.get("event_count", 0)),
            row["run_key"],
            row["frame"],
        )
    flags = row.get("component") or {}
    key = {
        "04_hole_fill": "holes_filled",
        "05_tiny_island_at_most_1pct": "tiny_islands_removed",
        "06_redundant_island_80_50": "redundant_islands_removed",
    }.get(category, "")
    return (-int(flags.get(key, 0)), row["run_key"], row["frame"])


def _select_diverse(
    rows: list[dict[str, Any]], category: str, limit: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _metric(row, category))
    selected: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, int]] = set()
    seen_runs: set[str] = set()
    for row in ordered:
        key = (str(row["run_key"]), int(row["frame"]))
        if key in seen_frames or str(row["run_key"]) in seen_runs:
            continue
        selected.append(row)
        seen_frames.add(key)
        seen_runs.add(str(row["run_key"]))
        if len(selected) >= limit:
            return selected
    for row in ordered:
        key = (str(row["run_key"]), int(row["frame"]))
        if key in seen_frames:
            continue
        selected.append(row)
        seen_frames.add(key)
        if len(selected) >= limit:
            break
    return selected


def _load_selected_records(
    run_dir: Path,
    scored_jsonl: Path,
    wanted: set[int],
) -> dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    paths = (
        scored_jsonl,
        run_dir / "arm_outputs/legacy.jsonl",
        run_dir / "arm_outputs/component_candidate_v2.jsonl",
    )
    result: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    iterators = [iter_detection_records(path) for path in paths]
    for rows in itertools.zip_longest(*iterators):
        if any(row is None for row in rows):
            raise RuntimeError(f"arm JSONL length mismatch: {run_dir}")
        frames = {int(row["frame_index"]) for row in rows if row is not None}
        if len(frames) != 1:
            raise RuntimeError(f"arm JSONL frame mismatch: {run_dir}: {frames}")
        frame = next(iter(frames))
        if frame in wanted:
            result[frame] = rows  # type: ignore[assignment]
        if len(result) == len(wanted):
            break
    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"selected frames absent from arm JSONL: {run_dir}: {missing}")
    return result


def _color(detection_id: int | str) -> tuple[int, int, int]:
    digest = hashlib.sha1(str(detection_id).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 180
    hsv = np.uint8([[[hue, 205, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def _put(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    safe = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(
        image, safe, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        image, safe, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _draw_dashed_polygon(
    image: np.ndarray,
    polygon: Iterable[Iterable[float]],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash: float = 9.0,
) -> None:
    points = np.asarray(list(polygon), dtype=np.float64)
    if len(points) < 2:
        return
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
        length = float(np.linalg.norm(end - start))
        if length <= 1e-9:
            continue
        pieces = max(1, int(math.ceil(length / dash)))
        for piece in range(0, pieces, 2):
            first = start + (end - start) * (piece / pieces)
            second = start + (end - start) * (min(piece + 1, pieces) / pieces)
            cv2.line(
                image,
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(second).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )


def _paint_raster(
    overlay: np.ndarray,
    raster: Any,
    color: tuple[int, int, int],
) -> None:
    if raster is None or raster.area <= 0:
        return
    height, width = overlay.shape[:2]
    left = max(0, int(raster.left))
    top = max(0, int(raster.top))
    right = min(width - 1, int(raster.right))
    bottom = min(height - 1, int(raster.bottom))
    if right < left or bottom < top:
        return
    source = raster.mask[
        top - raster.top : bottom - raster.top + 1,
        left - raster.left : right - raster.left + 1,
    ]
    target = overlay[top : bottom + 1, left : right + 1]
    target[source != 0] = color


def _polygon_signature(detection: dict[str, Any]) -> str:
    return json.dumps(
        detection.get("polygons") or [], separators=(",", ":"), ensure_ascii=False
    )


def _render_panel(
    image: np.ndarray,
    input_record: dict[str, Any],
    output_record: dict[str, Any],
    title: str,
) -> np.ndarray:
    frame = int(input_record["frame_index"])
    inputs = list(input_record["detections"])
    outputs = list(output_record["detections"])
    input_map = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(inputs)
    }
    output_map = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(outputs)
    }
    canvas = image.copy()
    overlay = image.copy()
    for detection_id, detection in output_map.items():
        _paint_raster(overlay, _raster_mask(detection), _color(detection_id))
    cv2.addWeighted(overlay, 0.46, canvas, 0.54, 0.0, dst=canvas)
    for detection_id, detection in output_map.items():
        color = _color(detection_id)
        for polygon in detection.get("polygons") or []:
            points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
            if len(points) >= 3:
                cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
        bbox = detection.get("bbox_xyxy") or [8, 30, 0, 0]
        _put(
            canvas,
            f"D{detection_id} {float(detection.get('score') or 0.0):.3f}",
            (max(6, int(float(bbox[0]))), max(22, int(float(bbox[1])) - 6)),
            scale=0.50,
            color=color,
        )
    for detection_id, detection in input_map.items():
        output = output_map.get(detection_id)
        changed = output is not None and _polygon_signature(detection) != _polygon_signature(output)
        if output is None or changed:
            color = _color(detection_id)
            for polygon in detection.get("polygons") or []:
                _draw_dashed_polygon(canvas, polygon, color)
    bar = np.full((48, canvas.shape[1], 3), 16, np.uint8)
    _put(bar, title, (14, 31), scale=0.66)
    return np.vstack([bar, canvas])


def _header(
    width: int,
    row: dict[str, Any],
    input_record: dict[str, Any],
) -> np.ndarray:
    frame = int(input_record["frame_index"])
    legacy = set(row["legacy_ids"])
    candidate = set(row["candidate_ids"])
    entries: list[str] = []
    for index, detection in enumerate(input_record["detections"]):
        detection_id = _detection_id(detection, frame, index)
        class_name = CLASS_NAMES.get(
            str(detection.get("class_name", "")),
            str(detection.get("class_name", "unknown")),
        )
        entries.append(
            f"D{detection_id}:{float(detection.get('score') or 0.0):.3f}:"
            f"{class_name}:L{'K' if detection_id in legacy else 'X'}:"
            f"C{'K' if detection_id in candidate else 'X'}"
        )
    chunks: list[str] = []
    current = ""
    for entry in entries:
        proposal = entry if not current else f"{current} | {entry}"
        if len(proposal) > 145 and current:
            chunks.append(current)
            current = entry
        else:
            current = proposal
    if current:
        chunks.append(current)
    lines = [
        f"run={row['run_key']} frame={row['frame']} category={row['category']}",
        f"reason={row['reason']}",
        *chunks,
        "solid+fill=retained output; dashed=input mask suppressed or topology-changed",
    ]
    height = 30 + 25 * len(lines)
    header = np.full((height, width, 3), 12, np.uint8)
    for index, line in enumerate(lines):
        _put(header, line, (14, 25 + 25 * index), scale=0.47)
    return header


def _render_one(
    row: dict[str, Any],
    records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    video: Path,
    path: Path,
    panel_width: int,
) -> dict[str, Any]:
    input_record, legacy_record, candidate_record = records
    image = seek_frame(video, int(row["frame"]))
    left = _render_panel(image, input_record, legacy_record, "LEFT: legacy Production NMS")
    right = _render_panel(image, input_record, candidate_record, "RIGHT: component candidate v2")
    panels = np.concatenate([left, right], axis=1)
    header = _header(panels.shape[1], row, input_record)
    combined = np.vstack([header, panels])
    if panel_width > 0 and image.shape[1] > panel_width:
        target_width = panel_width * 2
        target_height = max(1, int(round(combined.shape[0] * target_width / combined.shape[1])))
        combined = cv2.resize(
            combined, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {path}")

    frame = int(row["frame"])
    input_map = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(input_record["detections"])
    }
    legacy_map = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(legacy_record["detections"])
    }
    candidate_map = {
        _detection_id(detection, frame, index): detection
        for index, detection in enumerate(candidate_record["detections"])
    }
    changed_ids = [
        detection_id
        for detection_id in sorted(set(legacy_map) & set(candidate_map), key=str)
        if _polygon_signature(legacy_map[detection_id])
        != _polygon_signature(candidate_map[detection_id])
    ]
    score_by_id = {
        str(detection_id): float(detection.get("score") or 0.0)
        for detection_id, detection in input_map.items()
    }
    class_by_id = {
        str(detection_id): str(detection.get("class_name", ""))
        for detection_id, detection in input_map.items()
    }
    return {
        **row,
        "scores": score_by_id,
        "classes": class_by_id,
        "topology_changed_ids": changed_ids,
        "image": str(path),
        "image_width": int(combined.shape[1]),
        "image_height": int(combined.shape[0]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "category",
        "sample_index",
        "run_key",
        "frame",
        "reason",
        "input_count",
        "input_ids",
        "legacy_ids",
        "candidate_ids",
        "legacy_only_suppressed_ids",
        "candidate_only_suppressed_ids",
        "both_suppressed_ids",
        "topology_changed_ids",
        "event",
        "component",
        "scores",
        "classes",
        "image",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in (
                "input_ids",
                "legacy_ids",
                "candidate_ids",
                "legacy_only_suppressed_ids",
                "candidate_only_suppressed_ids",
                "both_suppressed_ids",
                "topology_changed_ids",
                "event",
                "component",
                "scores",
                "classes",
            ):
                encoded[key] = json.dumps(
                    encoded.get(key), ensure_ascii=False, separators=(",", ":")
                )
            writer.writerow(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--topology-sqlite", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-category", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=1280)
    args = parser.parse_args()

    ablation_root = args.ablation_root.expanduser().resolve()
    topology_path = args.topology_sqlite.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing gallery: {output}")
    if not ablation_root.is_dir() or not topology_path.is_file():
        raise FileNotFoundError((ablation_root, topology_path))
    if not 1 <= args.samples_per_category <= 12:
        raise ValueError("samples-per-category must be in [1, 12]")

    topology = _open_ro(topology_path)
    run_metadata = {
        str(row["run_key"]): {
            "video": Path(str(row["input_video"])),
            "video_slug": str(row["video_slug"]),
        }
        for row in topology.execute(
            "SELECT run_key,input_video,video_slug FROM audit_runs WHERE model_key='v3'"
        )
    }
    topology.close()

    all_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_sources: dict[str, Path] = {}
    run_dirs: dict[str, Path] = {}
    for run_dir in sorted((ablation_root / "runs").glob("v3__*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file() or run_dir.name not in run_metadata:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scored = Path(str(summary["input"]["jsonl"])).resolve()
        decisions, components = _decision_frames(run_dir)
        per_run = _scan_candidates(
            ablation_root,
            run_dir.name,
            scored,
            decisions,
            components,
        )
        for category, rows in per_run.items():
            all_candidates[category].extend(rows)
        run_sources[run_dir.name] = scored
        run_dirs[run_dir.name] = run_dir

    selected: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORY_ORDER:
        limit = 2 if category == "03_candidate_only_suppressed" else int(
            args.samples_per_category
        )
        selected[category] = _select_diverse(
            all_candidates.get(category, []), category, limit
        )

    selected_by_run: dict[str, set[int]] = defaultdict(set)
    for rows in selected.values():
        for row in rows:
            selected_by_run[str(row["run_key"])].add(int(row["frame"]))
    records_by_run = {
        run_key: _load_selected_records(
            run_dirs[run_key], run_sources[run_key], frames
        )
        for run_key, frames in selected_by_run.items()
    }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    manifest: list[dict[str, Any]] = []
    try:
        for category in CATEGORY_ORDER:
            folder = staging / category
            folder.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(selected[category], 1):
                run_key = str(row["run_key"])
                frame = int(row["frame"])
                filename = (
                    f"sample_{index:02d}_{run_metadata[run_key]['video_slug']}_"
                    f"f{frame}.jpg"
                )
                rendered = _render_one(
                    row,
                    records_by_run[run_key][frame],
                    run_metadata[run_key]["video"],
                    folder / filename,
                    int(args.panel_width),
                )
                rendered["sample_index"] = index
                rendered["image"] = str(
                    output / category / filename
                )
                manifest.append(rendered)

        summary = {
            "schema_version": 1,
            "privacy": (
                "Video frames were decoded and written only by local OpenCV. "
                "No image was uploaded or opened through an AI image tool."
            ),
            "source_ablation": str(ablation_root),
            "topology_sqlite": str(topology_path),
            "panels": {
                "left": "legacy Production NMS",
                "right": "component candidate v2",
            },
            "legend": {
                "same_id_same_color": True,
                "solid_fill": "retained output mask",
                "dashed": "input mask suppressed by this arm or geometry changed by topology cleanup",
                "L_K_C_K": "legacy keep / candidate keep flags in header",
            },
            "available_candidates": {
                category: len(all_candidates.get(category, []))
                for category in CATEGORY_ORDER
            },
            "selected_counts": {
                category: len(selected[category]) for category in CATEGORY_ORDER
            },
            "images": len(manifest),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(staging / "manifest.csv", manifest)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        readme_lines = [
            "# NMS component candidate v2 review gallery",
            "",
            "All images are exactly two panels: left is legacy Production NMS; right is component candidate v2.",
            "The same source detection ID always has the same colour in both panels.",
            "Solid fill is the retained output. Dashed contours are input masks suppressed by that arm, or the raw boundary of a topology-modified retained mask.",
            "",
            "Frames were decoded locally with OpenCV only. They were not uploaded or opened with an AI image-view tool.",
            "",
            "## Categories",
            "",
        ]
        for category in CATEGORY_ORDER:
            readme_lines.append(
                f"- `{category}`: selected {len(selected[category])} / available {len(all_candidates.get(category, []))}"
            )
        readme_lines.extend(
            [
                "",
                "See `manifest.csv` for spreadsheet review and `manifest.json` for exact IDs, scores, classes, suppression metrics, and reasons.",
                "",
            ]
        )
        (staging / "README.md").write_text(
            "\n".join(readme_lines), encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(output)
        os.replace(staging, output)
    except BaseException:
        # Keep a failed staging directory for diagnosis; never replace output.
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
