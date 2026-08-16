#!/usr/bin/env python3
"""Compare dense legacy/v3 polygon14 masks and render the largest deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
HERE = Path(__file__).resolve().parent
for value in (POSTPROCESS, HERE):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_four_arm_v3 as base  # noqa: E402
from contracts.detections import iter_detection_records  # noqa: E402
from render_component_candidate_v2_review_gallery import (  # noqa: E402
    _put,
    _render_panel,
    seek_frame,
)


LABELS = ("女性器", "男性器", "結合部分")


def _bbox(polygons: list[list[list[float]]]) -> list[float]:
    points = [point for polygon in polygons for point in polygon]
    return [
        min(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[0]) for point in points),
        max(float(point[1]) for point in points),
    ]


def _load_predictions(root: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for label in LABELS:
        path = root / label / "runtime/pred/predictions.sqlite"
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            for frame, track_id, payload in db.execute(
                "SELECT frame,track_id,polygons FROM masks ORDER BY frame,track_id"
            ):
                polygons = json.loads(str(payload))
                box = _bbox(polygons)
                result[int(frame)].append(
                    {
                        "source_detection_id": f"{label}:{track_id}",
                        "class_name": label,
                        "label": label,
                        "score": 1.0,
                        "bbox_xyxy": box,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "polygons": polygons,
                        "segmentation": polygons,
                    }
                )
    return result


def _record(frame: int, detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {"frame_index": frame, "detections": detections}


def _resize(panel: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(round(panel.shape[0] * width / panel.shape[1])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-phase2", type=Path, required=True)
    parser.add_argument("--v3-phase2", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-top", type=int, default=60)
    parser.add_argument("--panel-width", type=int, default=960)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    legacy = _load_predictions(args.legacy_phase2.resolve())
    v3 = _load_predictions(args.v3_phase2.resolve())
    raw: dict[int, dict[str, Any]] = {
        int(value["frame_index"]): value
        for value in iter_detection_records(args.scored_jsonl.resolve())
    }
    frames = sorted(set(legacy) | set(v3))
    metrics: list[dict[str, Any]] = []
    for frame in frames:
        legacy_union = base._union_raster(legacy.get(frame, []), max_pixels=32_000_000)
        v3_union = base._union_raster(v3.get(frame, []), max_pixels=32_000_000)
        if legacy_union is None or v3_union is None:
            continue
        comparison = base._union_metrics(legacy_union, v3_union)
        metrics.append(
            {
                "frame_index": frame,
                "union_iou": comparison["union_iou"],
                "legacy_coverage_by_v3": comparison["union_recall"],
                "legacy_area": comparison["input_union_area"],
                "v3_area": comparison["output_union_area"],
                "area_delta": int(comparison["output_union_area"])
                - int(comparison["input_union_area"]),
                "legacy_instances": len(legacy.get(frame, [])),
                "v3_instances": len(v3.get(frame, [])),
            }
        )
    changed = [row for row in metrics if float(row["union_iou"]) < 1.0 - 1e-12]
    selected = sorted(
        changed,
        key=lambda row: (float(row["union_iou"]), -abs(int(row["area_delta"]))),
    )[: max(0, args.render_top)]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    images = staging / "largest_final_differences"
    images.mkdir()
    for row in selected:
        frame = int(row["frame_index"])
        image = seek_frame(args.video.resolve(), frame)
        raw_record = raw.get(frame, _record(frame, []))
        legacy_record = _record(frame, legacy.get(frame, []))
        v3_record = _record(frame, v3.get(frame, []))
        panels = [
            _render_panel(image, raw_record, raw_record, "RAW AI MASKS"),
            _render_panel(
                image, legacy_record, legacy_record, "LEGACY FINAL polygon14"
            ),
            _render_panel(image, v3_record, v3_record, "VIRTUAL v3 FINAL polygon14"),
        ]
        panels = [_resize(panel, args.panel_width) for panel in panels]
        body = np.concatenate(panels, axis=1)
        header = np.full((86, body.shape[1], 3), 12, np.uint8)
        _put(
            header,
            f"frame={frame} final union IoU={float(row['union_iou']):.6f}",
            (14, 28),
            scale=0.60,
        )
        _put(
            header,
            f"legacy area={row['legacy_area']} v3 area={row['v3_area']} delta={row['area_delta']} "
            f"instances={row['legacy_instances']}->{row['v3_instances']}",
            (14, 62),
            scale=0.55,
        )
        rendered = np.vstack([header, body])
        path = images / f"frame_{frame:06d}_iou_{float(row['union_iou']):.6f}.jpg"
        if not cv2.imwrite(str(path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(path)
        decoded = cv2.imread(str(path))
        if decoded is None or decoded.shape != rendered.shape:
            raise RuntimeError(f"JPEG validation failed: {path}")
        row["image"] = str(output / "largest_final_differences" / path.name)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with (staging / "all_frame_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "frame_index",
            "union_iou",
            "legacy_coverage_by_v3",
            "legacy_area",
            "v3_area",
            "area_delta",
            "legacy_instances",
            "v3_instances",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    payload = {
        "privacy": "Local SQLite/OpenCV only; no network or image-view tool.",
        "evaluated_frames": len(metrics),
        "changed_frames": len(changed),
        "identical_frames": len(metrics) - len(changed),
        "union_iou_min": min(float(row["union_iou"]) for row in metrics),
        "union_iou_mean": float(np.mean([float(row["union_iou"]) for row in metrics])),
        "rendered": len(selected),
        "selected": selected,
    }
    (staging / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(staging, output)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "evaluated_frames",
                    "changed_frames",
                    "identical_frames",
                    "union_iou_min",
                    "union_iou_mean",
                    "rendered",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
