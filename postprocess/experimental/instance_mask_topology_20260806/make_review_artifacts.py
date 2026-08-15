#!/usr/bin/env python3
"""Create local, human-reviewable raw/final topology comparison stills.

This program never transmits pixels.  It decodes selected frames locally and
writes PNG files plus a JSON manifest.  The agent must not inspect those PNGs;
they are review artifacts for the user.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _put(image: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2,
        cv2.LINE_AA,
    )


def _blend_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int = 3,
) -> None:
    if len(points) < 3:
        return
    polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.polylines(image, [polygon], True, color, thickness, cv2.LINE_AA)


def _raw_polygons(
    connection: sqlite3.Connection, detection_id: int
) -> list[tuple[int, str, int, np.ndarray]]:
    rows = connection.execute(
        """
        SELECT p.polygon_index, c.role, c.nesting_depth, pt.x, pt.y
        FROM raw.segmentation_polygons p
        JOIN raw.segmentation_points pt ON pt.polygon_id=p.id
        JOIN topo.contour_topology c
          ON c.run_key=? AND c.detection_id=p.detection_id
         AND c.polygon_index=p.polygon_index
        WHERE p.detection_id=?
        ORDER BY p.polygon_index, pt.point_index
        """,
        (RUN_KEY, detection_id),
    )
    grouped: dict[tuple[int, str, int], list[tuple[float, float]]] = {}
    for polygon_index, role, depth, x, y in rows:
        grouped.setdefault((int(polygon_index), str(role), int(depth)), []).append(
            (float(x), float(y))
        )
    return [
        (index, role, depth, np.asarray(points, dtype=np.float32))
        for (index, role, depth), points in grouped.items()
    ]


def _final_polygons(
    connection: sqlite3.Connection, track_id: str, frame: int
) -> list[tuple[int, str, np.ndarray]]:
    rows = connection.execute(
        """
        SELECT c.slot_index, r.ring_role, p.x, p.y
        FROM final.mask_track_segments s
        JOIN final.mask_keyframes k ON k.segment_id=s.id
        JOIN final.keyframe_components c ON c.keyframe_id=k.id
        JOIN final.keyframe_polygon_rings r ON r.component_id=c.id
        JOIN final.keyframe_polygon_points p ON p.ring_id=r.id
        WHERE s.track_id=? AND k.frame=?
        ORDER BY c.slot_index, r.ring_index, p.point_index
        """,
        (track_id, frame),
    )
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for slot, role, x, y in rows:
        grouped.setdefault((int(slot), str(role)), []).append((float(x), float(y)))
    return [
        (slot, role, np.asarray(points, dtype=np.float32))
        for (slot, role), points in grouped.items()
    ]


def _select(connection: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[int] = set()

    def add(kind: str, query: str, params: tuple[object, ...], count: int) -> None:
        for row in connection.execute(query, params):
            did = int(row[0])
            if did in seen:
                continue
            seen.add(did)
            selected.append(
                {
                    "kind": kind,
                    "detection_id": did,
                    "frame": int(row[1]),
                    "track_id": None if row[2] is None else str(row[2]),
                    "foreground_components": int(row[3]),
                    "holes": int(row[4]),
                    "second_ratio": float(row[5]),
                    "exact_keyframe": bool(row[6]),
                    "final_components": None if row[7] is None else int(row[7]),
                }
            )
            if sum(1 for item in selected if item["kind"] == kind) >= count:
                break

    base = """
      SELECT detection_id, frame, final_track_id,
             foreground_component_count, hole_count, second_to_largest_ratio,
             exact_keyframe, keyframe_component_count
      FROM audit.detection_outcomes
      WHERE run_key=? AND disposition='retained'
    """
    add(
        "large_secondary",
        base + " AND foreground_component_count>1 AND exact_keyframe=1"
        " ORDER BY second_to_largest_ratio DESC",
        (RUN_KEY,),
        min(10, limit),
    )
    add(
        "tiny_secondary",
        base + " AND foreground_component_count>1"
        " AND second_to_largest_ratio<0.001 ORDER BY RANDOM()",
        (RUN_KEY,),
        min(6, limit),
    )
    add(
        "hole_only",
        base + " AND foreground_component_count=1 AND hole_count>0"
        " AND exact_keyframe=1 AND COALESCE(keyframe_component_count,0)>1"
        " ORDER BY hole_count DESC, frame",
        (RUN_KEY,),
        min(8, limit),
    )
    return selected[:limit]


def _decode_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index}")
    return frame


def _resize_panel(image: np.ndarray, width: int = 960) -> np.ndarray:
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(
        image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--inference-sqlite", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--postprocess-audit", type=Path, required=True)
    parser.add_argument("--result-sqlite", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    global RUN_KEY
    RUN_KEY = args.run_key
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(":memory:")
    connection.execute(f"ATTACH DATABASE '{_q(args.inference_sqlite)}' AS raw")
    connection.execute(f"ATTACH DATABASE '{_q(args.topology)}' AS topo")
    connection.execute(f"ATTACH DATABASE '{_q(args.postprocess_audit)}' AS audit")
    connection.execute(f"ATTACH DATABASE '{_q(args.result_sqlite)}' AS final")
    selected = _select(connection, args.limit)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    manifest: list[dict[str, object]] = []
    raw_colors = [(255, 80, 40), (255, 0, 255), (0, 180, 255), (40, 255, 80)]
    final_colors = [(50, 220, 50), (0, 180, 255), (255, 120, 0), (220, 0, 220)]
    for order, item in enumerate(selected, start=1):
        frame = int(item["frame"])
        source = _decode_frame(capture, frame)
        raw_panel = source.copy()
        final_panel = source.copy()
        raw_polygons = _raw_polygons(connection, int(item["detection_id"]))
        foreground_index = 0
        for _index, role, _depth, points in raw_polygons:
            if role == "hole":
                _blend_polygon(raw_panel, points, (255, 255, 0), 0.12, 4)
            else:
                color = raw_colors[min(foreground_index, len(raw_colors) - 1)]
                _blend_polygon(raw_panel, points, color, 0.36, 3)
                foreground_index += 1
        track_id = item["track_id"]
        final_polygons = (
            _final_polygons(connection, str(track_id), frame)
            if track_id is not None and item["exact_keyframe"]
            else []
        )
        for slot, role, points in final_polygons:
            color = final_colors[min(slot, len(final_colors) - 1)]
            _blend_polygon(final_panel, points, color, 0.36, 3)
        _put(
            raw_panel,
            f"RAW frame={frame} detection={item['detection_id']} "
            f"fg={item['foreground_components']} holes={item['holes']} "
            f"ratio={item['second_ratio']:.4f}",
            34,
            (255, 255, 255),
        )
        _put(
            final_panel,
            f"FINAL track={track_id} exact_key={int(bool(item['exact_keyframe']))} "
            f"slots={item['final_components']} rings={len(final_polygons)}",
            34,
            (255, 255, 255),
        )
        left = _resize_panel(raw_panel)
        right = _resize_panel(final_panel)
        if left.shape[0] != right.shape[0]:
            height = min(left.shape[0], right.shape[0])
            left = left[:height]
            right = right[:height]
        comparison = np.concatenate([left, right], axis=1)
        filename = (
            f"{order:03d}_{item['kind']}_f{frame}_d{item['detection_id']}.png"
        )
        target = args.output_dir / filename
        if not cv2.imwrite(str(target), comparison):
            raise RuntimeError(f"could not write {target}")
        record = dict(item)
        record["output"] = str(target.resolve())
        record["raw_polygon_roles"] = [role for _, role, _, _ in raw_polygons]
        record["final_ring_roles"] = [role for _, role, _ in final_polygons]
        manifest.append(record)
    capture.release()
    connection.close()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifest)} local review stills to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
