#!/usr/bin/env python3
"""Render the largest KPI final-mask changes through six controlled stages.

All frames are decoded and rendered locally.  The script never opens an image
through an AI viewer and has no network path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
import sys

if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from contracts.detections import iter_detection_records  # noqa: E402
from nms.adaptive import DEFAULT_NMS  # noqa: E402
from nms.component_aware import _raster_mask  # noqa: E402
from nms.components import fill_holes_and_remove_tiny_islands  # noqa: E402


BASE = ROOT / "output/nms_component_candidate_v2_fixed_downstream_kpi_corrected_20260813"
TOPOLOGY_BASE = ROOT / "output/nms_topology_legacy_fixed_downstream_kpi_20260813"
VIDEO = ROOT / "data/新しいフォルダー/12月KPI動画.mp4"
SCORED = ROOT / "output/nms_component_candidate_v2_ablation_20260813/inputs/v3__kpi_2025_12/scored.jsonl"
OUTPUT = ROOT / "output/nms_kpi_six_stage_impact_review_20260813"
LABELS = ("女性器", "男性器", "結合部分")
PROFILE = Path("polygon14/interval_6/polygon14_keyframe_v1")
PALETTE = {
    "女性器": (216, 85, 225),
    "男性器": (225, 190, 45),
    "結合部分": (45, 165, 245),
    "unknown": (190, 190, 190),
}
SHORT_LABEL = {"女性器": "F", "男性器": "M", "結合部分": "J"}


@dataclass(frozen=True)
class MaskItem:
    identity: str
    label: str
    polygons: list[np.ndarray]


def _open_ro(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _parse(value: str) -> list[np.ndarray]:
    return [np.asarray(polygon, dtype=np.float32) for polygon in json.loads(value)]


def _prediction_path(root: Path, label: str) -> Path:
    return root / PROFILE / label / "runtime/pred/predictions.sqlite"


def _load_final(root: Path) -> dict[int, list[MaskItem]]:
    result: dict[int, list[MaskItem]] = defaultdict(list)
    for label in LABELS:
        path = _prediction_path(root, label)
        if not path.is_file():
            raise FileNotFoundError(path)
        with _open_ro(path) as db:
            for row in db.execute("SELECT frame,track_id,polygons FROM masks ORDER BY frame"):
                result[int(row["frame"])].append(
                    MaskItem(
                        identity=f"T{row['track_id']}",
                        label=label,
                        polygons=_parse(str(row["polygons"])),
                    )
                )
    return dict(result)


def _points(items: list[MaskItem]) -> list[np.ndarray]:
    return [polygon for item in items for polygon in item.polygons if len(polygon) >= 3]


def _union_mask(items: list[MaskItem]) -> tuple[np.ndarray, tuple[int, int]] | None:
    polygons = _points(items)
    if not polygons:
        return None
    all_points = np.concatenate(polygons)
    low = np.floor(all_points.min(axis=0)).astype(np.int32) - 2
    high = np.ceil(all_points.max(axis=0)).astype(np.int32) + 2
    shape = (int(high[1] - low[1] + 1), int(high[0] - low[0] + 1))
    if shape[0] <= 0 or shape[1] <= 0:
        return None
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.rint(polygon - low).astype(np.int32)], 1)
    return mask, (int(low[0]), int(low[1]))


def _pair_metrics(first: list[MaskItem], second: list[MaskItem]) -> dict[str, float]:
    polygons = _points(first) + _points(second)
    if not polygons:
        return {"iou": 1.0, "first_area": 0.0, "second_area": 0.0, "symmetric_difference": 0.0}
    all_points = np.concatenate(polygons)
    low = np.floor(all_points.min(axis=0)).astype(np.int32) - 2
    high = np.ceil(all_points.max(axis=0)).astype(np.int32) + 2
    shape = (int(high[1] - low[1] + 1), int(high[0] - low[0] + 1))

    def raster(items: list[MaskItem]) -> np.ndarray:
        output = np.zeros(shape, np.uint8)
        for polygon in _points(items):
            cv2.fillPoly(output, [np.rint(polygon - low).astype(np.int32)], 1)
        return output

    left, right = raster(first), raster(second)
    first_area = int(left.sum())
    second_area = int(right.sum())
    intersection = int(np.count_nonzero(left & right))
    union = first_area + second_area - intersection
    symmetric = int(np.count_nonzero(left ^ right))
    return {
        "iou": intersection / union if union else 1.0,
        "first_area": float(first_area),
        "second_area": float(second_area),
        "symmetric_difference": float(symmetric),
    }


def _class_items(items: list[MaskItem], label: str) -> list[MaskItem]:
    return [item for item in items if item.label == label]


def _impact_rows(
    legacy: dict[int, list[MaskItem]],
    topology: dict[int, list[MaskItem]],
    candidate: dict[int, list[MaskItem]],
) -> list[dict[str, Any]]:
    frames = sorted(set(legacy) | set(topology) | set(candidate))
    rows: list[dict[str, Any]] = []
    for frame in frames:
        old = legacy.get(frame, [])
        topo = topology.get(frame, [])
        new = candidate.get(frame, [])
        old_new = _pair_metrics(old, new)
        old_topology = _pair_metrics(old, topo)
        topology_new = _pair_metrics(topo, new)
        class_iou = {
            label: _pair_metrics(_class_items(old, label), _class_items(new, label))["iou"]
            for label in LABELS
        }
        old_area = old_new["first_area"]
        new_area = old_new["second_area"]
        rows.append(
            {
                "frame": frame,
                "legacy_vs_latest_iou": old_new["iou"],
                "legacy_vs_topology_iou": old_topology["iou"],
                "topology_vs_latest_iou": topology_new["iou"],
                "legacy_area": old_area,
                "topology_area": old_topology["second_area"],
                "latest_area": new_area,
                "latest_to_legacy_area_ratio": (
                    new_area / old_area if old_area > 0 else float("inf") if new_area > 0 else 1.0
                ),
                "legacy_to_latest_area_ratio": (
                    old_area / new_area if new_area > 0 else float("inf") if old_area > 0 else 1.0
                ),
                "symmetric_difference": old_new["symmetric_difference"],
                **{f"iou_{SHORT_LABEL[label]}": class_iou[label] for label in LABELS},
            }
        )
    return rows


def _select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: dict[int, set[str]] = defaultdict(set)

    def take(reason: str, ordered: list[dict[str, Any]], count: int) -> None:
        for row in ordered[:count]:
            selected[int(row["frame"])].add(reason)

    positive = [row for row in rows if row["legacy_area"] or row["latest_area"]]
    take(
        "largest_total_shape_change",
        sorted(positive, key=lambda row: (row["legacy_vs_latest_iou"], -row["symmetric_difference"])),
        20,
    )
    take(
        "largest_hole_island_only_change",
        sorted(positive, key=lambda row: (row["legacy_vs_topology_iou"], -row["symmetric_difference"])),
        12,
    )
    take(
        "largest_nms_change_after_same_topology",
        sorted(positive, key=lambda row: (row["topology_vs_latest_iou"], -row["symmetric_difference"])),
        16,
    )
    finite_expand = [row for row in positive if np.isfinite(row["latest_to_legacy_area_ratio"])]
    take(
        "largest_latest_expansion",
        sorted(finite_expand, key=lambda row: (-row["latest_to_legacy_area_ratio"], -row["latest_area"])),
        8,
    )
    finite_shrink = [row for row in positive if np.isfinite(row["legacy_to_latest_area_ratio"])]
    take(
        "largest_latest_contraction",
        sorted(finite_shrink, key=lambda row: (-row["legacy_to_latest_area_ratio"], -row["legacy_area"])),
        8,
    )
    for label in ("F", "M", "J"):
        take(
            f"largest_class_{label}_change",
            sorted(positive, key=lambda row, key=f"iou_{label}": (row[key], -row["symmetric_difference"])),
            4,
        )

    by_frame = {int(row["frame"]): dict(row) for row in rows}
    # The quotas above sum to at most 76 distinct frames.  Fill any remaining
    # review slots with the largest overall changes without sacrificing the
    # topology-only and NMS-only strata.
    if len(selected) < limit:
        for row in sorted(
            positive,
            key=lambda value: (
                value["legacy_vs_latest_iou"],
                -value["symmetric_difference"],
            ),
        ):
            frame = int(row["frame"])
            if frame in selected:
                continue
            selected[frame].add("overall_fill")
            if len(selected) >= limit:
                break
    ordered_frames = sorted(
        selected,
        key=lambda frame: (
            by_frame[frame]["legacy_vs_latest_iou"],
            -by_frame[frame]["symmetric_difference"],
            frame,
        ),
    )
    output: list[dict[str, Any]] = []
    for rank, frame in enumerate(ordered_frames, 1):
        row = by_frame[frame]
        row["rank"] = rank
        row["selection_reasons"] = sorted(selected[frame])
        output.append(row)
    return output


def _load_raw_records(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in iter_detection_records(path):
        frame = int(record["frame_index"])
        if frame in wanted:
            result[frame] = record
        if len(result) == len(wanted):
            break
    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"missing raw records: {missing[:20]}")
    return result


def _detection_items(detections: list[dict[str, Any]]) -> list[MaskItem]:
    output: list[MaskItem] = []
    for index, detection in enumerate(detections):
        identity = detection.get("source_detection_id")
        output.append(
            MaskItem(
                identity=f"D{identity if identity is not None else index}",
                label=str(detection.get("class_name") or "unknown"),
                polygons=[
                    np.asarray(polygon, dtype=np.float32)
                    for polygon in detection.get("polygons") or []
                ],
            )
        )
    return output


def _legacy_direct(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return DEFAULT_NMS.apply(copy.deepcopy(detections))


def _topology_legacy_direct(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned, _ = fill_holes_and_remove_tiny_islands(
        copy.deepcopy(detections),
        fill_all_holes=True,
        unconditional_owner_ratio_max=0.01,
    )
    return DEFAULT_NMS.apply(cleaned)


def _put(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.46) -> None:
    safe = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(image, safe, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, safe, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def _draw_items(image: np.ndarray, items: list[MaskItem]) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    for item in items:
        color = PALETTE.get(item.label, PALETTE["unknown"])
        polygons = [np.rint(polygon).astype(np.int32) for polygon in item.polygons if len(polygon) >= 3]
        if polygons:
            cv2.fillPoly(overlay, polygons, color)
    cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0.0, canvas)
    for item in items:
        color = PALETTE.get(item.label, PALETTE["unknown"])
        polygons = [np.rint(polygon).astype(np.int32) for polygon in item.polygons if len(polygon) >= 3]
        for polygon in polygons:
            cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_AA)
        if polygons:
            point = np.concatenate(polygons).min(axis=0)
            _put(
                canvas,
                f"{item.identity}:{SHORT_LABEL.get(item.label, '?')}",
                (max(4, int(point[0])), max(18, int(point[1]) - 5)),
                0.42,
            )
    return canvas


def _roi(items_by_stage: list[list[MaskItem]], width: int, height: int) -> tuple[int, int, int, int]:
    polygons = [polygon for items in items_by_stage for polygon in _points(items)]
    if not polygons:
        return (0, 0, width, height)
    points = np.concatenate(polygons)
    low = points.min(axis=0)
    high = points.max(axis=0)
    margin = max(48.0, 0.18 * float(max(high - low)))
    x1, y1 = np.floor(low - margin).astype(int)
    x2, y2 = np.ceil(high + margin).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 320:
        center = (x1 + x2) // 2
        x1, x2 = max(0, center - 160), min(width, center + 160)
    if y2 - y1 < 240:
        center = (y1 + y2) // 2
        y1, y2 = max(0, center - 120), min(height, center + 120)
    return x1, y1, x2, y2


def _panel(
    image: np.ndarray,
    items: list[MaskItem],
    title: str,
    detail: str,
    *,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    canvas = _draw_items(image, items)
    if roi is not None:
        x1, y1, x2, y2 = roi
        canvas = canvas[y1:y2, x1:x2]
        size = (640, 480)
    else:
        size = (640, 360)
    canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
    header = np.full((74, size[0], 3), 14, np.uint8)
    _put(header, title, (10, 27), 0.52)
    _put(header, detail, (10, 57), 0.40)
    return np.vstack([header, canvas])


def _render(
    image: np.ndarray,
    stages: list[tuple[str, list[MaskItem]]],
    impact: dict[str, Any],
    *,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    details = [
        f"detections={len(stages[0][1])}",
        f"kept={len(stages[1][1])}",
        f"kept={len(stages[2][1])}",
        f"area={impact['legacy_area']:.0f}",
        f"area={impact['topology_area']:.0f}",
        f"area={impact['latest_area']:.0f}",
    ]
    panels = [
        _panel(image, items, title, detail, roi=roi)
        for (title, items), detail in zip(stages, details, strict=True)
    ]
    body = np.concatenate(panels, axis=1)
    top = np.full((84, body.shape[1], 3), 10, np.uint8)
    reasons = ",".join(impact["selection_reasons"])
    _put(
        top,
        f"frame={impact['frame']} rank={impact['rank']} old/latest IoU={impact['legacy_vs_latest_iou']:.4f} old/topology={impact['legacy_vs_topology_iou']:.4f} topology/latest={impact['topology_vs_latest_iou']:.4f}",
        (12, 30),
        0.50,
    )
    _put(top, f"selected_by={reasons}", (12, 64), 0.41)
    return np.vstack([top, body])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 1 <= args.limit <= 120:
        raise ValueError("--limit must be in [1,120]")

    legacy_root = BASE / "legacy_production"
    candidate_root = BASE / "component_mask_v2"
    topology_root = TOPOLOGY_BASE / "topology_legacy_nms"
    for path in (VIDEO, SCORED, legacy_root, candidate_root, topology_root):
        if not path.exists():
            raise FileNotFoundError(path)

    legacy = _load_final(legacy_root)
    topology = _load_final(topology_root)
    candidate = _load_final(candidate_root)
    all_impacts = _impact_rows(legacy, topology, candidate)
    selected = _select(all_impacts, int(args.limit))
    wanted = {int(row["frame"]) for row in selected}
    raw_records = _load_raw_records(SCORED, wanted)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    (staging / "01_full_frame").mkdir(parents=True)
    (staging / "02_roi_zoom").mkdir(parents=True)
    capture = cv2.VideoCapture(str(VIDEO))
    manifest: list[dict[str, Any]] = []
    try:
        for impact in selected:
            frame = int(impact["frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"failed to decode frame {frame}")
            raw_detections = list(raw_records[frame]["detections"])
            stages = [
                ("1 RAW AI masks (score>=0.30)", _detection_items(raw_detections)),
                ("2 legacy NMS direct", _detection_items(_legacy_direct(raw_detections))),
                ("3 holes/tiny + legacy NMS direct", _detection_items(_topology_legacy_direct(raw_detections))),
                ("4 legacy NMS + fixed polygon14/DP", legacy.get(frame, [])),
                ("5 holes/tiny + legacy NMS + same DP", topology.get(frame, [])),
                ("6 latest topology/NMS + same DP", candidate.get(frame, [])),
            ]
            roi = _roi([items for _, items in stages], image.shape[1], image.shape[0])
            full = _render(image, stages, impact, roi=None)
            zoom = _render(image, stages, impact, roi=roi)
            name = f"rank_{impact['rank']:03d}_frame_{frame:06d}.jpg"
            full_path = staging / "01_full_frame" / name
            zoom_path = staging / "02_roi_zoom" / name
            for path, rendered in ((full_path, full), (zoom_path, zoom)):
                if not cv2.imwrite(str(path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to write {path}")
            manifest.append(
                {
                    **impact,
                    "roi_xyxy": [int(value) for value in roi],
                    "raw_detection_ids": [
                        detection.get("source_detection_id") for detection in raw_detections
                    ],
                    "legacy_nms_ids": [
                        detection.get("source_detection_id")
                        for detection in _legacy_direct(raw_detections)
                    ],
                    "topology_legacy_nms_ids": [
                        detection.get("source_detection_id")
                        for detection in _topology_legacy_direct(raw_detections)
                    ],
                    "full_frame_image": str(output / "01_full_frame" / name),
                    "roi_zoom_image": str(output / "02_roi_zoom" / name),
                }
            )
    finally:
        capture.release()

    fields = list(manifest[0])
    with (staging / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            encoded = dict(row)
            for key in ("selection_reasons", "roi_xyxy", "raw_detection_ids", "legacy_nms_ids", "topology_legacy_nms_ids"):
                encoded[key] = json.dumps(encoded[key], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(encoded)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# KPI six-stage impact review

大きな最終マスク差を、同一フレーム・同一後段条件で6段階表示します。

1. AI生マスク（score >= 0.30）
2. 旧Production NMS直後
3. 全穴埋め＋本体比1%以下の島削除＋旧NMS直後
4. 旧NMS＋Production候補polygon14/最小Recall DP/pair-vote後
5. 穴・1%島＋旧NMS＋同じ後段最適化後
6. 最新Mask-IoU NMS＋穴・島処理＋同じ後段最適化後

`01_full_frame` は全画面、`02_roi_zoom` は全6段階のマスクを包含する同一ROIです。
色はクラス固定（F=女性器、M=男性器、J=結合部分）、Dは生検出ID、Tは最終track IDです。

画像はローカルOpenCVのみで生成し、AI画像閲覧やアップロードは行っていません。
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    hashes: dict[str, str] = {}
    failures: list[str] = []
    for path in sorted(staging.rglob("*.jpg")):
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            failures.append(str(path))
        hashes[str(path.relative_to(staging))] = hashlib.sha256(path.read_bytes()).hexdigest()
    qa = {
        "selected_frames": len(selected),
        "expected_jpegs": 2 * len(selected),
        "decoded_jpegs": len(hashes),
        "decode_failures": failures,
        "passed": len(hashes) == 2 * len(selected) and not failures,
        "privacy": "local OpenCV only; no AI image viewer",
    }
    (staging / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    (staging / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    if not qa["passed"]:
        raise RuntimeError(f"QA failed: {qa}")
    os.replace(staging, output)
    print(json.dumps({"output": str(output), **qa}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
