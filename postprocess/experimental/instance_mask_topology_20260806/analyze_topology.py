#!/usr/bin/env python3
"""Classify flat segmentation contours as foreground islands or holes."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_runs(
  run_key TEXT PRIMARY KEY,
  model_key TEXT NOT NULL,
  segmentation_model TEXT NOT NULL,
  video_slug TEXT NOT NULL,
  input_video TEXT NOT NULL,
  inference_sqlite TEXT NOT NULL,
  analyzed_at_unix REAL NOT NULL,
  frame_count INTEGER NOT NULL,
  detection_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS mask_topology(
  run_key TEXT NOT NULL,
  detection_id INTEGER NOT NULL,
  frame INTEGER NOT NULL,
  detection_index INTEGER,
  score REAL,
  class_name TEXT,
  contour_count INTEGER NOT NULL,
  foreground_component_count INTEGER NOT NULL,
  hole_count INTEGER NOT NULL,
  largest_component_area REAL NOT NULL,
  second_component_area REAL NOT NULL,
  second_to_largest_ratio REAL NOT NULL,
  net_foreground_area REAL NOT NULL,
  largest_hole_area REAL NOT NULL,
  largest_hole_to_outer_ratio REAL NOT NULL,
  bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
  PRIMARY KEY(run_key, detection_id)
);
CREATE TABLE IF NOT EXISTS contour_topology(
  run_key TEXT NOT NULL,
  detection_id INTEGER NOT NULL,
  polygon_index INTEGER NOT NULL,
  nesting_depth INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('foreground', 'hole')),
  signed_area REAL NOT NULL,
  absolute_area REAL NOT NULL,
  parent_polygon_index INTEGER,
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  point_count INTEGER NOT NULL,
  PRIMARY KEY(run_key, detection_id, polygon_index)
);
CREATE INDEX IF NOT EXISTS idx_topology_multi
  ON mask_topology(run_key, foreground_component_count, second_to_largest_ratio);
CREATE INDEX IF NOT EXISTS idx_topology_frame ON mask_topology(run_key, frame);
"""


def _polygon_relation(polygons: list[np.ndarray]) -> tuple[list[int], list[int | None]]:
    """Return spatial nesting depth and the smallest containing parent."""
    areas = [abs(float(cv2.contourArea(poly))) for poly in polygons]
    parents: list[int | None] = [None] * len(polygons)
    for child_index, child in enumerate(polygons):
        # A contour point is preferable to a centroid for strongly concave
        # contours.  Allowing a boundary match handles quantized touching.
        point = tuple(float(value) for value in child[0])
        candidates: list[int] = []
        for parent_index, parent in enumerate(polygons):
            if parent_index == child_index or areas[parent_index] <= areas[child_index]:
                continue
            if cv2.pointPolygonTest(parent, point, False) >= 0:
                candidates.append(parent_index)
        if candidates:
            parents[child_index] = min(candidates, key=lambda index: areas[index])
    depths: list[int] = []
    for index in range(len(polygons)):
        depth = 0
        seen = {index}
        parent = parents[index]
        while parent is not None and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    return depths, parents


def _flush_detection(
    destination: sqlite3.Connection,
    run_key: str,
    metadata: tuple[object, ...],
    polygon_rows: list[tuple[int, np.ndarray]],
) -> None:
    detection_id, frame, detection_index, score, class_name, x1, y1, x2, y2 = metadata
    polygons = [points for _, points in polygon_rows]
    depths, parents = _polygon_relation(polygons)
    areas = [abs(float(cv2.contourArea(poly))) for poly in polygons]
    signed = [float(cv2.contourArea(poly, oriented=True)) for poly in polygons]
    foreground_areas = sorted(
        (area for area, depth in zip(areas, depths) if depth % 2 == 0), reverse=True
    )
    hole_areas = sorted(
        (area for area, depth in zip(areas, depths) if depth % 2 == 1), reverse=True
    )
    largest = foreground_areas[0] if foreground_areas else 0.0
    second = foreground_areas[1] if len(foreground_areas) > 1 else 0.0
    largest_hole = hole_areas[0] if hole_areas else 0.0
    net_area = sum(
        area if depth % 2 == 0 else -area for area, depth in zip(areas, depths)
    )
    destination.execute(
        """
        INSERT OR REPLACE INTO mask_topology(
          run_key, detection_id, frame, detection_index, score, class_name,
          contour_count, foreground_component_count, hole_count,
          largest_component_area, second_component_area,
          second_to_largest_ratio, net_foreground_area, largest_hole_area,
          largest_hole_to_outer_ratio, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        ) VALUES(
          ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            run_key, int(detection_id), int(frame), detection_index, score, class_name,
            len(polygons), len(foreground_areas), len(hole_areas), largest, second,
            second / largest if largest > 0 else 0.0, net_area, largest_hole,
            largest_hole / largest if largest > 0 else 0.0,
            x1, y1, x2, y2,
        ),
    )
    for local_index, ((polygon_index, points), depth, parent, area, oriented) in enumerate(
        zip(polygon_rows, depths, parents, areas, signed, strict=True)
    ):
        px, py, pw, ph = cv2.boundingRect(points)
        destination.execute(
            """INSERT OR REPLACE INTO contour_topology(
                 run_key, detection_id, polygon_index, nesting_depth, role,
                 signed_area, absolute_area, parent_polygon_index,
                 x1, y1, x2, y2, point_count
               ) VALUES(
                 ?,?,?,?,?,?,?,?,?,?,?,?,?
               )""",
            (
                run_key, int(detection_id), int(polygon_index), int(depth),
                "foreground" if depth % 2 == 0 else "hole", oriented, area,
                None if parent is None else int(polygon_rows[parent][0]),
                float(px), float(py), float(px + pw), float(py + ph), len(points),
            ),
        )


def analyze_one(destination: sqlite3.Connection, item: dict[str, object]) -> dict[str, object]:
    source_path = Path(str(item["inference_sqlite"]))
    if not source_path.is_file():
        manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                published = payload.get("artifacts", {}).get("result_sqlite")
                if published:
                    source_path = Path(str(published))
            except (OSError, ValueError, TypeError):
                pass
    if not source_path.is_file():
        return {"run_key": item["run_key"], "status": "missing"}
    started = time.monotonic()
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    source.execute("PRAGMA query_only=ON")
    frame_count = int(source.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
    detection_count = int(source.execute("SELECT COUNT(*) FROM segmentations").fetchone()[0])
    existing = destination.execute(
        """SELECT inference_sqlite, frame_count, detection_count
           FROM audit_runs WHERE run_key=?""",
        (item["run_key"],),
    ).fetchone()
    if existing is not None and (
        Path(str(existing[0])).resolve() == source_path.resolve()
        and int(existing[1]) == frame_count
        and int(existing[2]) == detection_count
    ):
        summary = destination.execute(
            """
            SELECT COUNT(*), SUM(foreground_component_count>1),
                   SUM(hole_count>0), MAX(foreground_component_count),
                   MAX(hole_count)
            FROM mask_topology WHERE run_key=?
            """,
            (item["run_key"],),
        ).fetchone()
        source.close()
        return {
            "run_key": item["run_key"], "status": "reused",
            "frames": frame_count, "detections": detection_count,
            "multi_foreground_detections": int(summary[1] or 0),
            "detections_with_holes": int(summary[2] or 0),
            "max_foreground_components": int(summary[3] or 0),
            "max_holes": int(summary[4] or 0),
            "elapsed_seconds": time.monotonic()-started,
        }
    destination.execute("DELETE FROM contour_topology WHERE run_key=?", (item["run_key"],))
    destination.execute("DELETE FROM mask_topology WHERE run_key=?", (item["run_key"],))
    query = """
      SELECT d.id, f.frame_index, d.group_id, d.score, d.class_name,
             d.x1, d.y1, d.x2, d.y2,
             p.polygon_index, pt.point_index, pt.x, pt.y
      FROM detections d
      JOIN frames f ON f.id=d.frame_id
      JOIN segmentation_polygons p ON p.detection_id=d.id
      JOIN segmentation_points pt ON pt.polygon_id=p.id
      ORDER BY d.id, p.polygon_index, pt.point_index
    """
    current_id: int | None = None
    metadata: tuple[object, ...] | None = None
    current_polygon: int | None = None
    points: list[tuple[float, float]] = []
    polygons: list[tuple[int, np.ndarray]] = []
    processed = 0
    for row in source.execute(query):
        did = int(row[0]); polygon_index = int(row[9])
        if current_id is not None and did != current_id:
            if current_polygon is not None and points:
                polygons.append((current_polygon, np.asarray(points, dtype=np.float32)))
            assert metadata is not None
            _flush_detection(destination, str(item["run_key"]), metadata, polygons)
            processed += 1
            if processed % 10_000 == 0:
                destination.commit()
                print(f'[topology] {item["run_key"]}: {processed}/{detection_count}', flush=True)
            polygons=[]; points=[]; current_polygon=None
        if current_id != did:
            current_id=did
            metadata=tuple(row[:9])
        if current_polygon is not None and polygon_index != current_polygon:
            polygons.append((current_polygon, np.asarray(points, dtype=np.float32)))
            points=[]
        current_polygon=polygon_index
        points.append((float(row[11]), float(row[12])))
    if current_id is not None and metadata is not None:
        if current_polygon is not None and points:
            polygons.append((current_polygon, np.asarray(points, dtype=np.float32)))
        _flush_detection(destination, str(item["run_key"]), metadata, polygons)
        processed += 1
    destination.execute(
        """INSERT OR REPLACE INTO audit_runs VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            item["run_key"], item["model_key"], item["segmentation_model"],
            item["video_slug"], item["input_video"], str(source_path.resolve()),
            time.time(), frame_count, detection_count,
        ),
    )
    destination.commit()
    summary = destination.execute(
        """
        SELECT COUNT(*),
               SUM(foreground_component_count>1),
               SUM(hole_count>0),
               MAX(foreground_component_count),
               MAX(hole_count)
        FROM mask_topology WHERE run_key=?
        """,
        (item["run_key"],),
    ).fetchone()
    source.close()
    return {
        "run_key": item["run_key"], "status": "complete",
        "frames": frame_count, "detections": detection_count,
        "multi_foreground_detections": int(summary[1] or 0),
        "detections_with_holes": int(summary[2] or 0),
        "max_foreground_components": int(summary[3] or 0),
        "max_holes": int(summary[4] or 0),
        "elapsed_seconds": time.monotonic()-started,
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    matrix=json.loads(args.matrix.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination=sqlite3.connect(args.output)
    destination.executescript(SCHEMA)
    summaries=[]
    for item in matrix:
        result=analyze_one(destination,item)
        summaries.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    destination.close()
    summary_path=args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0 if all(
        row['status'] in {'complete', 'reused', 'missing'} for row in summaries
    ) else 1


if __name__=='__main__':
    raise SystemExit(main())
