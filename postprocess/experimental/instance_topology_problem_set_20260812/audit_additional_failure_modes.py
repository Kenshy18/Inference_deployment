#!/usr/bin/env python3
"""Audit mask-topology failure modes beyond islands, holes and duplicates.

The audit reads only SQLite geometry.  It never opens a video or image.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = ROOT / "output/instance_mask_topology_20260806"
DEFAULT_OUTPUT = ROOT / "output/instance_topology_problem_set_20260812/additional_audit.json"


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _load_contours(
    connection: sqlite3.Connection, detection_ids: set[int]
) -> dict[int, dict[int, np.ndarray]]:
    if not detection_ids:
        return {}
    output: dict[int, dict[int, list[tuple[float, float]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    ids = sorted(detection_ids)
    for start in range(0, len(ids), 700):
        chunk = ids[start : start + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"""SELECT p.detection_id,p.polygon_index,pt.x,pt.y
                FROM segmentation_polygons p
                JOIN segmentation_points pt ON pt.polygon_id=p.id
                WHERE p.detection_id IN ({placeholders})
                ORDER BY p.detection_id,p.polygon_index,pt.point_index""",
            chunk,
        ):
            output[int(row["detection_id"])][int(row["polygon_index"])].append(
                (float(row["x"]), float(row["y"]))
            )
    return {
        detection_id: {
            polygon_index: np.asarray(points, dtype=np.float64)
            for polygon_index, points in polygons.items()
        }
        for detection_id, polygons in output.items()
    }


def _safe_polygon(points: np.ndarray) -> Polygon | None:
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    try:
        polygon = Polygon(points)
    except Exception:
        return None
    return polygon


def _raw_geometry_audit(
    experiment: Path,
    topology: sqlite3.Connection,
) -> dict[str, object]:
    audit_runs = list(topology.execute("SELECT * FROM audit_runs ORDER BY run_key"))
    aggregate = collections.Counter()
    examples: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    per_run = []
    for run in audit_runs:
        run_key = str(run["run_key"])
        candidate_rows = list(
            topology.execute(
                """SELECT * FROM mask_topology
                   WHERE run_key=? AND (contour_count>1 OR foreground_component_count>1 OR hole_count>0)
                   ORDER BY detection_id""",
                (run_key,),
            )
        )
        candidate_ids = {int(row["detection_id"]) for row in candidate_rows}
        # Representative single-contour masks catch general invalid polygons
        # without loading the entire 892k-detection corpus into memory.
        sample_ids = {
            int(row[0])
            for row in topology.execute(
                """SELECT detection_id FROM mask_topology
                   WHERE run_key=? AND contour_count=1
                   ORDER BY detection_id LIMIT 1000""",
                (run_key,),
            )
        }
        inference = _open(Path(str(run["inference_sqlite"])))
        video_row = inference.execute(
            "SELECT width,height FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
        video_width = float(video_row[0]) if video_row is not None else math.inf
        video_height = float(video_row[1]) if video_row is not None else math.inf
        geometries = _load_contours(inference, candidate_ids | sample_ids)
        roles = {
            (int(row["detection_id"]), int(row["polygon_index"])): (
                str(row["role"]),
                int(row["nesting_depth"]),
                None if row["parent_polygon_index"] is None else int(row["parent_polygon_index"]),
            )
            for row in topology.execute(
                """SELECT detection_id,polygon_index,role,nesting_depth,parent_polygon_index
                   FROM contour_topology WHERE run_key=?""",
                (run_key,),
            )
        }
        local = collections.Counter()
        for detection_id, contour_map in geometries.items():
            local["audited_detections"] += 1
            local["audited_contours"] += len(contour_map)
            polygons: dict[int, Polygon] = {}
            for polygon_index, points in contour_map.items():
                if len(points) and (
                    np.any(points[:, 0] < 0.0)
                    or np.any(points[:, 1] < 0.0)
                    or np.any(points[:, 0] > video_width)
                    or np.any(points[:, 1] > video_height)
                ):
                    local["raw_contours_outside_frame_bounds"] += 1
                _role, nesting_depth, _parent = roles.get(
                    (detection_id, polygon_index), ("foreground", 0, None)
                )
                if nesting_depth >= 2:
                    local["nested_depth_ge_2_contours"] += 1
                polygon = _safe_polygon(points)
                if polygon is None or polygon.area <= 0.0:
                    local["degenerate_or_nonfinite_contours"] += 1
                    if len(examples["degenerate_or_nonfinite_contours"]) < 20:
                        examples["degenerate_or_nonfinite_contours"].append(
                            {"run_key": run_key, "detection_id": detection_id, "polygon_index": polygon_index}
                        )
                    continue
                if not polygon.is_valid:
                    local["invalid_or_self_intersecting_contours"] += 1
                    if detection_id in candidate_ids:
                        local["invalid_contours_in_topology_candidates"] += 1
                    else:
                        local["invalid_contours_in_single_contour_sample"] += 1
                    if len(examples["invalid_or_self_intersecting_contours"]) < 20:
                        examples["invalid_or_self_intersecting_contours"].append(
                            {
                                "run_key": run_key,
                                "detection_id": detection_id,
                                "polygon_index": polygon_index,
                                "reason": explain_validity(polygon),
                            }
                        )
                    # Pair-wise intersections on invalid geometry are not
                    # meaningful and can raise a GEOS exception.
                    continue
                polygons[polygon_index] = polygon
            indices = sorted(polygons)
            for pos, left_index in enumerate(indices):
                left = polygons[left_index]
                left_role, left_depth, left_parent = roles.get(
                    (detection_id, left_index), ("foreground", 0, None)
                )
                for right_index in indices[pos + 1 :]:
                    right = polygons[right_index]
                    right_role, right_depth, right_parent = roles.get(
                        (detection_id, right_index), ("foreground", 0, None)
                    )
                    if left_role != "foreground" or right_role != "foreground":
                        continue
                    intersection = left.intersection(right).area
                    if intersection <= 1e-6:
                        continue
                    minimum = min(left.area, right.area)
                    union = left.union(right).area
                    containment = intersection / minimum if minimum else 0.0
                    iou = intersection / union if union else 0.0
                    # An even-depth foreground inside an odd-depth hole is a
                    # compound island/hole hierarchy, not a duplicate contour.
                    hierarchical = (
                        left_parent == right_index
                        or right_parent == left_index
                        or left_depth != right_depth
                    )
                    if hierarchical:
                        key = "same_instance_compound_nested_foregrounds"
                    elif containment >= 0.95:
                        key = "same_instance_nested_or_duplicate_foregrounds"
                    else:
                        key = "same_instance_overlapping_foregrounds"
                    local[key] += 1
                    if len(examples[key]) < 20:
                        examples[key].append(
                            {
                                "run_key": run_key,
                                "detection_id": detection_id,
                                "left_polygon_index": left_index,
                                "right_polygon_index": right_index,
                                "containment": containment,
                                "iou": iou,
                            }
                        )
        inference.close()
        aggregate.update(local)
        per_run.append(
            {
                "run_key": run_key,
                "model": str(run["model_key"]),
                "candidate_detections": len(candidate_ids),
                "single_contour_sample": len(sample_ids),
                **dict(local),
            }
        )
    return {
        "scope": "all multi-contour detections plus first 1000 single-contour detections per run",
        "counts": dict(aggregate),
        "examples": dict(examples),
        "per_run": per_run,
    }


def _result_paths(experiment: Path) -> list[tuple[str, Path]]:
    output = []
    for path in sorted(experiment.glob("postprocess_trace.*.summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = Path(str(data["result_sqlite"]))
        if result.is_file():
            output.append((str(data["run_key"]), result))
    current_kpi = ROOT / "data/12月KPI動画.sqlite"
    if current_kpi.is_file():
        output.append(("current__kpi_2025_12", current_kpi))
    return output


def _final_sqlite_audit(experiment: Path) -> dict[str, object]:
    totals = collections.Counter()
    per_run = []
    examples: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for run_key, result in _result_paths(experiment):
        connection = _open(result)
        local = collections.Counter()
        local["polygon_rings_exterior"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM keyframe_polygon_rings WHERE ring_role='exterior'"
            ).fetchone()[0]
        )
        local["polygon_rings_hole"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM keyframe_polygon_rings WHERE ring_role='hole'"
            ).fetchone()[0]
        )
        local["genital_segments"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM mask_track_segments s
                   JOIN tracks t ON t.track_id=s.track_id WHERE t.domain='genital'"""
            ).fetchone()[0]
        )
        local["genital_keyframes"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM mask_keyframes k
                   JOIN mask_track_segments s ON s.id=k.segment_id
                   JOIN tracks t ON t.track_id=s.track_id WHERE t.domain='genital'"""
            ).fetchone()[0]
        )
        local["genital_polygon_components"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM keyframe_components c
                   JOIN mask_keyframes k ON k.id=c.keyframe_id
                   JOIN mask_track_segments s ON s.id=k.segment_id
                   JOIN tracks t ON t.track_id=s.track_id
                   WHERE t.domain='genital' AND c.geometry_type='polygon'"""
            ).fetchone()[0]
        )
        video_row = connection.execute(
            "SELECT width,height FROM videos ORDER BY id LIMIT 1"
        ).fetchone()
        video_width = float(video_row[0]) if video_row is not None else math.inf
        video_height = float(video_row[1]) if video_row is not None else math.inf

        # Validate every exterior ring, not only multi-component keyframes.
        current_ring: int | None = None
        current_meta: dict[str, object] | None = None
        current_points: list[tuple[float, float]] = []
        previous_ring_by_slot: dict[tuple[int, str, int], tuple[int, np.ndarray, int]] = {}

        def finish_ring() -> None:
            nonlocal current_ring, current_meta, current_points
            if current_ring is None or current_meta is None:
                return
            polygon = _safe_polygon(np.asarray(current_points, dtype=np.float64))
            points_array = np.asarray(current_points, dtype=np.float64)
            if len(points_array) and (
                np.any(points_array[:, 0] < 0.0)
                or np.any(points_array[:, 1] < 0.0)
                or np.any(points_array[:, 0] > video_width)
                or np.any(points_array[:, 1] > video_height)
            ):
                local["final_exterior_rings_outside_frame_bounds"] += 1
            if polygon is None or polygon.area <= 0:
                local["final_degenerate_exterior_rings"] += 1
            elif not polygon.is_valid:
                local["final_invalid_or_self_intersecting_exterior_rings"] += 1
                key_name = "final_invalid_or_self_intersecting_exterior_rings"
                if len(examples[key_name]) < 20:
                    examples[key_name].append(
                        {
                            "run_key": run_key,
                            **current_meta,
                            "reason": explain_validity(polygon),
                        }
                    )
            else:
                signed_twice_area = float(
                    np.sum(
                        points_array[:, 0] * np.roll(points_array[:, 1], -1)
                        - np.roll(points_array[:, 0], -1) * points_array[:, 1]
                    )
                )
                orientation = 1 if signed_twice_area > 0 else -1
                local[
                    "final_exterior_rings_ccw"
                    if orientation > 0
                    else "final_exterior_rings_cw"
                ] += 1
                slot_key = (
                    int(current_meta["segment_id"]),
                    str(current_meta["track_id"]),
                    int(current_meta["slot"]),
                )
                previous = previous_ring_by_slot.get(slot_key)
                if previous is not None:
                    previous_frame, previous_points, previous_orientation = previous
                    if previous_orientation != orientation:
                        local["final_consecutive_winding_flips"] += 1
                        if len(examples["final_consecutive_winding_flips"]) < 20:
                            examples["final_consecutive_winding_flips"].append(
                                {
                                    "run_key": run_key,
                                    "track_id": slot_key[1],
                                    "slot": slot_key[2],
                                    "previous_frame": previous_frame,
                                    "frame": int(current_meta["frame"]),
                                }
                            )
                    if len(previous_points) == len(points_array) and len(points_array) >= 3:
                        left = previous_points - np.mean(previous_points, axis=0)
                        right = points_array - np.mean(points_array, axis=0)
                        costs = np.asarray(
                            [np.mean(np.sum((left - np.roll(right, shift, axis=0)) ** 2, axis=1))
                             for shift in range(len(right))]
                        )
                        best_shift = int(np.argmin(costs))
                        if best_shift != 0 and costs[best_shift] + 1e-6 < 0.35 * costs[0]:
                            local["probable_consecutive_vertex_phase_jumps"] += 1
                            if len(examples["probable_consecutive_vertex_phase_jumps"]) < 20:
                                examples["probable_consecutive_vertex_phase_jumps"].append(
                                    {
                                        "run_key": run_key,
                                        "track_id": slot_key[1],
                                        "slot": slot_key[2],
                                        "previous_frame": previous_frame,
                                        "frame": int(current_meta["frame"]),
                                        "best_cyclic_shift": best_shift,
                                        "cost_ratio": float(costs[best_shift] / max(costs[0], 1e-12)),
                                    }
                                )
                previous_ring_by_slot[slot_key] = (
                    int(current_meta["frame"]), points_array, orientation
                )
            current_ring = None
            current_meta = None
            current_points = []

        for row in connection.execute(
            """SELECT r.id ring_id,k.frame,s.id segment_id,s.track_id,c.slot_index,p.x,p.y
               FROM keyframe_polygon_rings r
               JOIN keyframe_components c ON c.id=r.component_id
               JOIN mask_keyframes k ON k.id=c.keyframe_id
               JOIN mask_track_segments s ON s.id=k.segment_id
               JOIN tracks t ON t.track_id=s.track_id
               JOIN keyframe_polygon_points p ON p.ring_id=r.id
               WHERE t.domain='genital' AND r.ring_role='exterior'
               ORDER BY r.id,p.point_index"""
        ):
            ring_id = int(row["ring_id"])
            if current_ring != ring_id:
                finish_ring()
                current_ring = ring_id
                current_meta = {
                    "segment_id": int(row["segment_id"]),
                    "track_id": str(row["track_id"]),
                    "frame": int(row["frame"]),
                    "slot": int(row["slot_index"]),
                }
            current_points.append((float(row["x"]), float(row["y"])))
        finish_ring()
        varying = list(
            connection.execute(
                """WITH counts AS (
                       SELECT s.id segment_id,s.track_id,s.component_count,k.id keyframe_id,
                              COUNT(c.id) actual_count
                       FROM mask_track_segments s
                       JOIN tracks t ON t.track_id=s.track_id
                       JOIN mask_keyframes k ON k.segment_id=s.id
                       LEFT JOIN keyframe_components c ON c.keyframe_id=k.id
                       WHERE t.domain='genital'
                       GROUP BY s.id,k.id
                   )
                   SELECT segment_id,track_id,component_count,
                          MIN(actual_count) minimum_count,MAX(actual_count) maximum_count,
                          SUM(actual_count<component_count) incomplete_keyframes,
                          COUNT(*) keyframes
                   FROM counts GROUP BY segment_id
                   HAVING MIN(actual_count)!=MAX(actual_count)
                       OR MIN(actual_count)<component_count"""
            )
        )
        local["segments_with_variable_or_missing_component_slots"] = len(varying)
        local["keyframes_missing_declared_component_slots"] = sum(
            int(row["incomplete_keyframes"]) for row in varying
        )
        if varying:
            examples["segments_with_variable_or_missing_component_slots"].extend(
                {"run_key": run_key, **dict(row)} for row in varying[:5]
            )

        # Audit all multi-component final keyframes for overlap/containment and
        # slot-order flips.  Single-component validity is sampled separately.
        multi_keys = list(
            connection.execute(
                """SELECT k.id keyframe_id,k.frame,s.track_id
                   FROM mask_keyframes k
                   JOIN mask_track_segments s ON s.id=k.segment_id
                   JOIN tracks t ON t.track_id=s.track_id
                   JOIN keyframe_components c ON c.keyframe_id=k.id
                   WHERE t.domain='genital' AND c.geometry_type='polygon'
                   GROUP BY k.id HAVING COUNT(c.id)>1 ORDER BY s.track_id,k.frame"""
            )
        )
        previous_by_track: dict[str, tuple[int, dict[int, np.ndarray]]] = {}
        for key in multi_keys:
            rows = connection.execute(
                """SELECT c.slot_index,p.x,p.y
                   FROM keyframe_components c
                   JOIN keyframe_polygon_rings r ON r.component_id=c.id
                   JOIN keyframe_polygon_points p ON p.ring_id=r.id
                   WHERE c.keyframe_id=? AND r.ring_role='exterior'
                   ORDER BY c.slot_index,r.ring_index,p.point_index""",
                (key["keyframe_id"],),
            )
            grouped: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
            for row in rows:
                grouped[int(row["slot_index"])].append((float(row["x"]), float(row["y"])))
            slots = {slot: np.asarray(points, dtype=np.float64) for slot, points in grouped.items()}
            polygons = {slot: _safe_polygon(points) for slot, points in slots.items()}
            valid: dict[int, Polygon] = {}
            for slot, polygon in polygons.items():
                if polygon is None or polygon.area <= 0 or not polygon.is_valid:
                    continue
                valid[slot] = polygon
            for left_pos, left_slot in enumerate(sorted(valid)):
                for right_slot in sorted(valid)[left_pos + 1 :]:
                    left, right = valid[left_slot], valid[right_slot]
                    intersection = left.intersection(right).area
                    if intersection <= 1e-6:
                        continue
                    containment = intersection / min(left.area, right.area)
                    local["final_overlapping_component_pairs"] += 1
                    if containment >= 0.95:
                        overlap_kind = "nested"
                        local["final_nested_component_pairs"] += 1
                    else:
                        overlap_kind = "partial"
                        local["final_partially_overlapping_component_pairs"] += 1
                    example_key = f"final_component_overlap_{overlap_kind}"
                    if len(examples[example_key]) < 20:
                        examples[example_key].append(
                            {
                                "run_key": run_key,
                                "track_id": str(key["track_id"]),
                                "frame": int(key["frame"]),
                                "slots": [left_slot, right_slot],
                                "containment": containment,
                            }
                        )
            track_id = str(key["track_id"])
            if len(slots) == 2 and track_id in previous_by_track:
                previous_frame, previous = previous_by_track[track_id]
                if len(previous) == 2:
                    old = [np.mean(previous[index], axis=0) for index in sorted(previous)]
                    new = [np.mean(slots[index], axis=0) for index in sorted(slots)]
                    same = np.linalg.norm(old[0] - new[0]) + np.linalg.norm(old[1] - new[1])
                    swapped = np.linalg.norm(old[0] - new[1]) + np.linalg.norm(old[1] - new[0])
                    if swapped + 1e-6 < 0.70 * same:
                        local["probable_component_slot_swaps"] += 1
                        if len(examples["probable_component_slot_swaps"]) < 20:
                            examples["probable_component_slot_swaps"].append(
                                {
                                    "run_key": run_key,
                                    "track_id": track_id,
                                    "previous_frame": previous_frame,
                                    "frame": int(key["frame"]),
                                    "same_slot_motion": float(same),
                                    "swapped_slot_motion": float(swapped),
                                }
                            )
            previous_by_track[track_id] = (int(key["frame"]), slots)
        totals.update(local)
        per_run.append({"run_key": run_key, "result_sqlite": str(result), **dict(local)})
        connection.close()
    return {"counts": dict(totals), "examples": dict(examples), "per_run": per_run}


def _retained_duplicate_audit(
    experiment: Path,
    topology: sqlite3.Connection,
) -> dict[str, object]:
    trace = _open(experiment / "postprocess_trace.sqlite")
    run_meta = {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs")
    }
    totals = collections.Counter()
    examples = []
    per_run = []
    for run_key in sorted(
        str(row[0]) for row in trace.execute("SELECT DISTINCT run_key FROM detection_outcomes")
    ):
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
        candidate_ids = {
            int(row["detection_id"])
            for rows in by_frame.values() if len(rows) > 1
            for row in rows
        }
        inference = _open(Path(str(run_meta[run_key]["inference_sqlite"])))
        contours = _load_contours(inference, candidate_ids)
        info = {
            int(row["id"]): row
            for start in range(0, len(candidate_ids), 700)
            for row in inference.execute(
                f"""SELECT id,score,x1,y1,x2,y2 FROM detections
                    WHERE id IN ({','.join('?' for _ in sorted(candidate_ids)[start:start+700])})""",
                sorted(candidate_ids)[start:start+700],
            )
        }
        local = collections.Counter()
        for frame, rows in by_frame.items():
            if len(rows) < 2:
                continue
            for left_pos, left_row in enumerate(rows):
                for right_row in rows[left_pos + 1 :]:
                    left_id, right_id = int(left_row["detection_id"]), int(right_row["detection_id"])
                    left_info, right_info = info.get(left_id), info.get(right_id)
                    if left_info is None or right_info is None:
                        continue
                    ix = max(0.0, min(left_info["x2"], right_info["x2"]) - max(left_info["x1"], right_info["x1"]))
                    iy = max(0.0, min(left_info["y2"], right_info["y2"]) - max(left_info["y1"], right_info["y1"]))
                    if ix * iy <= 0:
                        continue
                    left_polys = [
                        _safe_polygon(points) for points in contours.get(left_id, {}).values()
                    ]
                    right_polys = [
                        _safe_polygon(points) for points in contours.get(right_id, {}).values()
                    ]
                    left_polys = [p for p in left_polys if p is not None and p.is_valid]
                    right_polys = [p for p in right_polys if p is not None and p.is_valid]
                    if not left_polys or not right_polys:
                        continue
                    left_shape = left_polys[0]
                    for value in left_polys[1:]:
                        left_shape = left_shape.union(value)
                    right_shape = right_polys[0]
                    for value in right_polys[1:]:
                        right_shape = right_shape.union(value)
                    intersection = left_shape.intersection(right_shape).area
                    if intersection <= 0:
                        continue
                    union = left_shape.union(right_shape).area
                    containment = intersection / min(left_shape.area, right_shape.area)
                    iou = intersection / union
                    if containment >= 0.80 or iou >= 0.20:
                        local["potential_duplicate_pairs_retained_after_nms"] += 1
                        if str(left_row["final_track_id"]) != str(right_row["final_track_id"]):
                            local["potential_duplicate_pairs_on_different_final_tracks"] += 1
                        if len(examples) < 50:
                            examples.append(
                                {
                                    "run_key": run_key,
                                    "frame": frame,
                                    "detection_ids": [left_id, right_id],
                                    "track_ids": [left_row["final_track_id"], right_row["final_track_id"]],
                                    "scores": [float(left_info["score"]), float(right_info["score"])],
                                    "mask_iou": iou,
                                    "smaller_containment": containment,
                                }
                            )
        inference.close()
        totals.update(local)
        per_run.append({"run_key": run_key, **dict(local)})
    trace.close()
    return {"counts": dict(totals), "examples": examples, "per_run": per_run}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    topology = _open(experiment / "topology.sqlite")
    payload = {
        "schema_version": 1,
        "privacy": "SQLite geometry only; no video or image was opened",
        "raw_geometry": _raw_geometry_audit(experiment, topology),
        "final_sqlite": _final_sqlite_audit(experiment),
        "retained_duplicates": _retained_duplicate_audit(experiment, topology),
    }
    topology.close()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "raw_geometry": payload["raw_geometry"]["counts"],
            "final_sqlite": payload["final_sqlite"]["counts"],
            "retained_duplicates": payload["retained_duplicates"]["counts"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
