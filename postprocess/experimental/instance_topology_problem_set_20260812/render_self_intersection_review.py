#!/usr/bin/env python3
"""Render local-only review sheets for clean-source self intersections."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "output/instance_topology_actual_frame_overlays_20260812/self_intersection_audit.json"
POSTPROCESS = ROOT / "output/instance_mask_topology_20260806/postprocess"
OUTPUT = ROOT / "output/instance_topology_actual_frame_overlays_20260812/self_intersection_clear_cases"


def put(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def read_frame(video: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"cannot decode {video} frame {frame}")
    return image


def intersections(points: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    count = len(points)
    found: list[tuple[int, int, np.ndarray]] = []
    for left in range(count):
        a0, a1 = points[left], points[(left + 1) % count]
        first = LineString([a0, a1])
        for right in range(left + 1, count):
            if right == left or right == (left + 1) % count or (right + 1) % count == left:
                continue
            b0, b1 = points[right], points[(right + 1) % count]
            cross = first.intersection(LineString([b0, b1]))
            if not cross.is_empty and cross.geom_type == "Point":
                found.append((left, right, np.asarray([cross.x, cross.y], np.float32)))
    return found


def crop_bounds(groups: list[np.ndarray], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    points = np.concatenate(groups, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    margin = max(45.0, float(max(span)) * 0.32)
    height, width = shape
    return (
        max(0, int(np.floor(lo[0] - margin))),
        max(0, int(np.floor(lo[1] - margin))),
        min(width, int(np.ceil(hi[0] + margin + 1))),
        min(height, int(np.ceil(hi[1] + margin + 1))),
    )


def panel(image: np.ndarray, bounds: tuple[int, int, int, int], title: list[str]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1]
    target = np.full((720, 620, 3), 18, np.uint8)
    available_h = 630
    scale = min(620 / max(crop.shape[1], 1), available_h / max(crop.shape[0], 1))
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    ox = (620 - resized.shape[1]) // 2
    oy = 90 + (available_h - resized.shape[0]) // 2
    target[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    for index, line in enumerate(title[:3]):
        put(target, line, (12, 24 + 27 * index), 0.53)
    return target


def schematic(raw: np.ndarray, final: np.ndarray, crosses: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    canvas = np.full((720, 620, 3), 245, np.uint8)
    all_points = np.concatenate([raw, final], axis=0)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = min(530.0 / span[0], 560.0 / span[1])
    origin = np.asarray([45.0, 115.0]) + 0.5 * (np.asarray([530.0, 560.0]) - span * scale)
    transform = lambda pts: np.rint((pts - lo) * scale + origin).astype(np.int32)
    raw_px, final_px = transform(raw), transform(final)
    cv2.polylines(canvas, [raw_px], True, (210, 150, 20), 3, cv2.LINE_AA)
    for idx in range(len(final_px)):
        a, b = final_px[idx], final_px[(idx + 1) % len(final_px)]
        cv2.line(canvas, tuple(a), tuple(b), (190, 40, 190), 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(a), 5, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (int(a[0] + 5), int(a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    for left, right, point in crosses:
        for edge, color in ((left, (0, 0, 255)), (right, (0, 150, 255))):
            cv2.line(canvas, tuple(final_px[edge]), tuple(final_px[(edge + 1) % len(final_px)]), color, 8, cv2.LINE_AA)
        cross_px = transform(point.reshape(1, 2))[0]
        cv2.drawMarker(canvas, tuple(cross_px), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 28, 5, cv2.LINE_AA)
    cv2.putText(canvas, "GEOMETRY ONLY", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "cyan=raw  magenta=final", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red/yellow=crossing edges", (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [row for row in json.loads(AUDIT.read_text())["cases"] if row.get("source_valid") is True]
    manifest: list[dict[str, object]] = []
    for index, case in enumerate(cases, 1):
        pipeline = POSTPROCESS / str(case["pipeline"])
        pipeline_manifest = json.loads((pipeline / "pipeline_manifest.json").read_text())
        video = Path(pipeline_manifest["artifacts"]["input_video"])
        if not video.is_absolute():
            video = ROOT / video
        frame = int(case["frame"])
        track = str(case["track_id"])
        slot = int(case["slot"])
        with sqlite3.connect(pipeline / "05_polygon_optimization/endpoint_extended.sqlite") as connection:
            raw_json = connection.execute("SELECT polygons FROM masks WHERE frame=? AND track_id=?", (frame, track)).fetchone()[0]
        raw_groups = [np.asarray(points, np.float32) for points in json.loads(raw_json)]
        keyframes = json.loads((pipeline / "05_polygon_optimization/vendor_output/opt/final_keyframes.json").read_text())
        row = next(item for item in keyframes if int(item["frame"]) == frame and str(item["track_id"]) == track and int(item.get("run_id", -1)) == int(case["run_id"]))
        final_groups = [np.asarray(points, np.float32) for points in row["polygons"]]
        raw = raw_groups[min(slot, len(raw_groups) - 1)]
        final = final_groups[slot]
        crosses = intersections(final)
        source = read_frame(video, frame)
        bounds = crop_bounds([raw, final], source.shape[:2])

        raw_image = source.copy()
        shade = raw_image.copy()
        for component_index, points in enumerate(raw_groups):
            contour = np.rint(points).astype(np.int32)
            color = (255, 210, 30) if component_index == slot else (130, 130, 130)
            cv2.fillPoly(shade, [contour], color)
            cv2.polylines(raw_image, [contour], True, color, 4 if component_index == slot else 2, cv2.LINE_AA)
        cv2.addWeighted(shade, 0.35, raw_image, 0.65, 0, raw_image)
        cv2.polylines(raw_image, [np.rint(raw).astype(np.int32)], True, (255, 220, 20), 5, cv2.LINE_AA)

        final_image = source.copy()
        cv2.polylines(final_image, [np.rint(raw).astype(np.int32)], True, (255, 220, 20), 3, cv2.LINE_AA)
        final_px = np.rint(final).astype(np.int32)
        cv2.polylines(final_image, [final_px], True, (220, 40, 220), 5, cv2.LINE_AA)
        for left, right, point in crosses:
            cv2.line(final_image, tuple(final_px[left]), tuple(final_px[(left + 1) % len(final_px)]), (0, 0, 255), 9, cv2.LINE_AA)
            cv2.line(final_image, tuple(final_px[right]), tuple(final_px[(right + 1) % len(final_px)]), (0, 180, 255), 9, cv2.LINE_AA)
            cv2.drawMarker(final_image, tuple(np.rint(point).astype(int)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 34, 6, cv2.LINE_AA)

        sheet = np.hstack([
            panel(raw_image, bounds, ["A  SOURCE POLYGON", "cyan = valid source", f"points={len(raw)} area={Polygon(raw).area:.1f}"]),
            panel(final_image, bounds, ["B  FINAL KEYFRAME", "magenta = invalid final", f"points={len(final)} area={Polygon(final).area:.1f}"]),
            schematic(raw, final, crosses),
        ])
        header = np.full((80, sheet.shape[1], 3), 12, np.uint8)
        put(header, f"CASE {index}/6  {case['pipeline']}  frame={frame} track={track} slot={slot}", (18, 30), 0.68)
        put(header, f"crossings={len(crosses)}  final reason={case['reason']}", (18, 62), 0.54)
        sheet = np.vstack([header, sheet])
        path = OUTPUT / f"case_{index:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', str(case['pipeline']))}_f{frame}_t{track}_s{slot}.png"
        cv2.imwrite(str(path), sheet)
        manifest.append({**case, "video": str(video), "image": str(path), "crossing_count": len(crosses)})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"case_count": len(manifest), "output": str(OUTPUT), "images": [row["image"] for row in manifest]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
