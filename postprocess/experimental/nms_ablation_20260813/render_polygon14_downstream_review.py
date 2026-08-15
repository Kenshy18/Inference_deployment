#!/usr/bin/env python3
"""Render the bounded KPI polygon14 downstream failure sequence locally."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output/nms_component_candidate_v2_fixed_downstream_kpi_corrected_20260813"
VIDEO = ROOT / "data/新しいフォルダー/12月KPI動画.mp4"
OUTPUT = ROOT / "output/nms_polygon14_downstream_review_f4285_4289_20260813"
FRAMES = tuple(range(4285, 4290))
LABEL = "女性器"
PROFILE = Path("polygon14/interval_6/polygon14_keyframe_v1")


def open_ro(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def parse_polygons(value: str) -> list[np.ndarray]:
    return [np.asarray(item, dtype=np.float32) for item in json.loads(value)]


def load_masks(path: Path, track_id: str) -> dict[int, list[np.ndarray]]:
    with open_ro(path) as db:
        rows = db.execute(
            "SELECT frame,polygons FROM masks WHERE track_id=? "
            "AND frame BETWEEN ? AND ? ORDER BY frame",
            (track_id, FRAMES[0], FRAMES[-1]),
        )
        return {int(row["frame"]): parse_polygons(row["polygons"]) for row in rows}


def load_raw(path: Path) -> dict[int, dict[str, Any]]:
    with open_ro(path) as db:
        rows = db.execute(
            """SELECT frame,raw_track_id,source_detection_id,score,raw_label,
                      final_label,bbox_xyxy_json,polygons
               FROM raw_tracked_masks
               WHERE final_track_id='23' AND frame BETWEEN ? AND ?
               ORDER BY frame""",
            (FRAMES[0], FRAMES[-1]),
        )
        return {
            int(row["frame"]): {
                **dict(row),
                "polygons": parse_polygons(row["polygons"]),
                "bbox_xyxy": json.loads(row["bbox_xyxy_json"]),
            }
            for row in rows
        }


def load_exact(arm: str) -> dict[int, dict[str, Any]]:
    path = RUN / arm / PROFILE / LABEL / "runtime/exact/keyframe_exact_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["frame"]): {
                key: (int(row[key]) if key in {"frame", "run_id", "has_keyframe"}
                      else str(row[key]) if key == "track_id" else float(row[key]))
                for key in row
            }
            for row in csv.DictReader(handle)
            if int(row["frame"]) in FRAMES
        }


def keyframes(arm: str, track_id: str) -> list[dict[str, Any]]:
    path = RUN / arm / PROFILE / LABEL / "runtime/opt/final_keyframes.json"
    return [
        {key: row[key] for key in ("track_id", "run_id", "frame", "candidate_id")}
        for row in json.loads(path.read_text(encoding="utf-8"))
        if str(row["track_id"]) == track_id and 4275 <= int(row["frame"]) <= 4310
    ]


def metrics(gt: list[np.ndarray], pred: list[np.ndarray]) -> dict[str, float]:
    polys = [np.asarray(poly, np.float32) for poly in gt + pred if len(poly) >= 3]
    points = np.concatenate(polys)
    low = np.floor(points.min(axis=0)).astype(np.int32)
    high = np.ceil(points.max(axis=0)).astype(np.int32)
    shape = (int(high[1] - low[1] + 1), int(high[0] - low[0] + 1))

    def raster(items: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros(shape, np.uint8)
        for polygon in items:
            cv2.fillPoly(mask, [np.round(polygon - low).astype(np.int32)], 1)
        return mask

    first, second = raster(gt), raster(pred)
    gt_area = int(first.sum())
    pred_area = int(second.sum())
    intersection = int((first & second).sum())
    union = gt_area + pred_area - intersection
    return {
        "gt_area": gt_area,
        "pred_area": pred_area,
        "intersection": intersection,
        "union": union,
        "recall": intersection / gt_area if gt_area else 1.0,
        "precision": intersection / pred_area if pred_area else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def put(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def draw_polygons(
    image: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    fill_alpha: float = 0.0,
    dashed: bool = False,
) -> None:
    points = [np.round(poly).astype(np.int32) for poly in polygons if len(poly) >= 3]
    if fill_alpha and points:
        overlay = image.copy()
        cv2.fillPoly(overlay, points, color)
        cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0, image)
    for polygon in points:
        if not dashed:
            cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
            continue
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            distance = float(np.linalg.norm(end - start))
            pieces = max(1, int(np.ceil(distance / 9.0)))
            for index in range(0, pieces, 2):
                a = start + (end - start) * (index / pieces)
                b = start + (end - start) * (min(index + 1, pieces) / pieces)
                cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 3, cv2.LINE_AA)


def crop_panel(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    title: str,
    detail: str,
    reference: list[np.ndarray],
    result: list[np.ndarray],
    color: tuple[int, int, int],
    raw: list[np.ndarray] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    if result:
        draw_polygons(canvas, result, color, fill_alpha=0.38)
    if reference and result is not reference:
        draw_polygons(canvas, reference, (70, 235, 90))
    if raw is not None:
        draw_polygons(canvas, raw, (30, 225, 245), dashed=True)
    x1, y1, x2, y2 = roi
    crop = canvas[y1:y2, x1:x2]
    crop = cv2.resize(crop, (720, 540), interpolation=cv2.INTER_CUBIC)
    bar = np.full((68, 720, 3), 14, np.uint8)
    put(bar, title, (12, 26), 0.60)
    put(bar, detail, (12, 54), 0.46)
    return np.vstack([bar, crop])


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    paths = {
        "candidate_reference": RUN / "component_mask_v2/polygon_preparation/00_女性器/polygon_preparation/endpoint_extended.sqlite",
        "candidate_tracked": RUN / "component_mask_v2/tracked.sqlite",
        "legacy_final": RUN / "legacy_production" / PROFILE / LABEL / "runtime/pred/predictions.sqlite",
        "candidate_final": RUN / "component_mask_v2" / PROFILE / LABEL / "runtime/pred/predictions.sqlite",
    }
    for path in [VIDEO, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = load_masks(paths["candidate_reference"], "23")
    legacy = load_masks(paths["legacy_final"], "21")
    candidate = load_masks(paths["candidate_final"], "23")
    raw = load_raw(paths["candidate_tracked"])
    if any(set(source) != set(FRAMES) for source in (reference, legacy, candidate, raw)):
        raise RuntimeError("one or more artifacts do not cover every requested frame")

    official_legacy = load_exact("legacy_production")
    official_candidate = load_exact("component_mask_v2")
    legacy_keys = keyframes("legacy_production", "21")
    candidate_keys = keyframes("component_mask_v2", "23")
    all_points = np.concatenate(
        [poly for frame in FRAMES for source in (reference, legacy, candidate)
         for poly in source[frame]]
    )
    center = np.mean([all_points.min(axis=0), all_points.max(axis=0)], axis=0)
    roi = (
        max(0, int(round(center[0] - 240))),
        max(0, int(round(center[1] - 180))),
        min(1920, int(round(center[0] + 240))),
        min(1080, int(round(center[1] + 180))),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.building-", dir=OUTPUT.parent))
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, FRAMES[0])
    manifest_frames: list[dict[str, Any]] = []
    rendered_rows: list[np.ndarray] = []
    try:
        for frame_index in FRAMES:
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"failed to decode frame {frame_index}")
            reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if reported != frame_index + 1:
                raise RuntimeError(f"decode position mismatch: {reported} after {frame_index}")
            legacy_metric = metrics(reference[frame_index], legacy[frame_index])
            candidate_metric = metrics(reference[frame_index], candidate[frame_index])
            raw_metric = metrics(reference[frame_index], raw[frame_index]["polygons"])
            for key in ("recall", "precision", "iou"):
                if abs(candidate_metric[key] - official_candidate[frame_index][key]) > 1e-12:
                    raise RuntimeError(f"candidate metric mismatch frame={frame_index} key={key}")
                if abs(legacy_metric[key] - official_legacy[frame_index][key]) > 1e-12:
                    raise RuntimeError(f"legacy metric mismatch frame={frame_index} key={key}")
            source = raw[frame_index]
            left = crop_panel(
                image, roi, "LEFT: candidate tracked reference / AI mask",
                f"T23 rawT{source['raw_track_id']} D{source['source_detection_id']} score={source['score']:.3f}",
                reference[frame_index], reference[frame_index], (70, 235, 90), source["polygons"],
            )
            middle = crop_panel(
                image, roi, "MIDDLE: legacy polygon14 final (T21)",
                f"R={legacy_metric['recall']:.3f} P={legacy_metric['precision']:.3f} IoU={legacy_metric['iou']:.3f} key={official_legacy[frame_index]['has_keyframe']}",
                reference[frame_index], legacy[frame_index], (245, 135, 45),
            )
            right = crop_panel(
                image, roi, "RIGHT: candidate polygon14 final (T23)",
                f"R={candidate_metric['recall']:.3f} P={candidate_metric['precision']:.3f} IoU={candidate_metric['iou']:.3f} key={official_candidate[frame_index]['has_keyframe']}",
                reference[frame_index], candidate[frame_index], (210, 70, 230),
            )
            row = np.concatenate([left, middle, right], axis=1)
            top = np.full((52, row.shape[1], 3), 10, np.uint8)
            put(top, f"frame={frame_index} fixed ROI={roi} green outline=candidate tracked reference", (12, 34), 0.62)
            row = np.vstack([top, row])
            filename = f"frame_{frame_index:06d}_three_panel.jpg"
            if not cv2.imwrite(str(staging / filename), row, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write {filename}")
            rendered_rows.append(row)
            manifest_frames.append({
                "frame": frame_index,
                "image": str(OUTPUT / filename),
                "candidate_reference": {
                    "track_id": "23", "raw_track_id": str(source["raw_track_id"]),
                    "source_detection_id": int(source["source_detection_id"]),
                    "score": float(source["score"]), "raw_label": source["raw_label"],
                    "final_label": source["final_label"], "bbox_xyxy": source["bbox_xyxy"],
                    "raw_detection_vs_reference": raw_metric,
                },
                "legacy_final": {"track_id": "21", **legacy_metric,
                    "has_keyframe": bool(official_legacy[frame_index]["has_keyframe"])},
                "candidate_final": {"track_id": "23", **candidate_metric,
                    "has_keyframe": bool(official_candidate[frame_index]["has_keyframe"])},
            })
        contact = np.vstack(rendered_rows)
        contact_name = "contact_sheet_frames_4285_4289.jpg"
        if not cv2.imwrite(str(staging / contact_name), contact, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError("failed to write contact sheet")
        manifest = {
            "schema_version": 1,
            "privacy": "local OpenCV decode/render only; no image was uploaded or AI-viewed",
            "video": str(VIDEO), "frames": manifest_frames, "fixed_roi_xyxy": roi,
            "panels": ["candidate tracked reference plus dashed raw detection", "legacy polygon14 final", "candidate polygon14 final"],
            "colors_bgr": {"reference": [70,235,90], "raw_detection_dashed": [30,225,245], "legacy": [245,135,45], "candidate": [210,70,230]},
            "metrics_reference": "candidate polygon14 stage input tracked mask; values reproduced exactly from each arm exact CSV because arm references are polygon-identical here",
            "legacy_keyframes_nearby": legacy_keys,
            "candidate_keyframes_nearby": candidate_keys,
            "artifacts": {key: str(value) for key, value in paths.items()},
            "contact_sheet": str(OUTPUT / contact_name),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksums = {}
        for path in sorted(staging.glob("*.jpg")):
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.size == 0:
                raise RuntimeError(f"structural decode failed: {path}")
        (staging / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, OUTPUT)
    finally:
        capture.release()
    print(json.dumps({"output": str(OUTPUT), "images": 6, "frames": list(FRAMES), "roi": roi}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
