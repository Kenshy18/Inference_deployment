#!/usr/bin/env python3
"""Render every historical exact-Recall failure before and after 14-point repair.

Video pixels are decoded locally only.  The renderer deliberately uses the
pre-DP spatial anchors and asserts that they are coordinate-identical to the
recorded failing keyframes before producing any review artifact.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from experimental.production_candidate_polygon14.config import CANDIDATE
from experimental.production_candidate_polygon14.integration import (
    _repair_sequence_exact_recall,
)
from experimental.production_candidate_polygon14.render_vertex_fallback_review import (
    ROOT,
    _ExactAdapter,
    _bounds,
    _load_track_rows,
    _runtime_module,
    _transform,
)
from experimental.production_candidate_polygon14.spatial import build_spatial_track


DEFAULT_BASELINE = (
    ROOT / "output/production_candidate_20260814_v3_exact_vs_default_i2_i5_20260814"
)
DEFAULT_OUTPUT = ROOT / "output/polygon14_exact_recall_repair_video_review_20260815"
_CYAN = (255, 220, 30)
_RED = (70, 70, 255)
_GREEN = (70, 230, 70)
_YELLOW = (0, 240, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contact-samples", type=int, default=16)
    return parser.parse_args()


def _put_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _full_context(frame_image: np.ndarray, bounds, size: int = 500) -> np.ndarray:
    height, width = frame_image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = cv2.resize(
        frame_image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.full((size, size, 3), 20, dtype=np.uint8)
    ox = (size - resized.shape[1]) // 2
    oy = (size - resized.shape[0]) // 2
    output[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    x1, y1, x2, y2 = bounds
    p1 = (round(x1 * scale) + ox, round(y1 * scale) + oy)
    p2 = (round(x2 * scale) + ox, round(y2 * scale) + oy)
    cv2.rectangle(output, p1, p2, _YELLOW, 2, cv2.LINE_AA)
    _put_title(output, "Original frame / yellow ROI")
    return output


def _warped_frame(frame_image: np.ndarray, bounds, size: int = 500) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    scale = min((size - 40) / max(x2 - x1, 1.0), (size - 40) / max(y2 - y1, 1.0))
    offset_x = (size - (x2 - x1) * scale) / 2.0 - x1 * scale
    offset_y = (size - (y2 - y1) * scale) / 2.0 - y1 * scale
    return cv2.warpAffine(
        frame_image,
        np.asarray([[scale, 0.0, offset_x], [0.0, scale, offset_y]], np.float32),
        (size, size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(20, 20, 20),
    )


def _overlay_panel(
    frame_image: np.ndarray,
    reference,
    polygon,
    bounds,
    *,
    title: str,
    color: tuple[int, int, int],
    metrics: dict[str, float] | None,
    raw_only: bool = False,
    size: int = 500,
) -> np.ndarray:
    image = _warped_frame(frame_image, bounds, size)
    raw = [_transform(value, bounds, size) for value in reference]
    predicted = [] if polygon is None else [
        _transform(value, bounds, size) for value in polygon
    ]
    raw_mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(raw_mask, raw, 255)
    layer = image.copy()
    if raw_only:
        cv2.fillPoly(layer, raw, _CYAN)
        image = cv2.addWeighted(layer, 0.34, image, 0.66, 0.0)
    else:
        predicted_mask = np.zeros((size, size), np.uint8)
        cv2.fillPoly(predicted_mask, predicted, 255)
        cv2.fillPoly(layer, predicted, color)
        image = cv2.addWeighted(layer, 0.18, image, 0.82, 0.0)
        missing = cv2.bitwise_and(raw_mask, cv2.bitwise_not(predicted_mask))
        image[missing > 0] = (
            image[missing > 0].astype(np.float32) * 0.25
            + np.asarray(_YELLOW, np.float32) * 0.75
        ).astype(np.uint8)
    for contour in raw:
        cv2.polylines(image, [contour], True, _CYAN, 2, cv2.LINE_AA)
    for contour in predicted:
        cv2.polylines(image, [contour], True, color, 3, cv2.LINE_AA)
        for x, y in contour:
            cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    _put_title(image, title)
    if metrics is not None:
        cv2.rectangle(image, (0, size - 38), (size - 1, size - 1), (0, 0, 0), -1)
        text = f"Recall {metrics['recall']:.5f}  IoU {metrics['iou']:.5f}"
        cv2.putText(
            image,
            text,
            (14, size - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def _decode_selected_frames(video_path: str, frames: set[int]):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    ordered = sorted(frames)
    capture.set(cv2.CAP_PROP_POS_FRAMES, ordered[0])
    target_index = 0
    current = ordered[0]
    try:
        while target_index < len(ordered):
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"could not decode {video_path} frame {current}")
            target = ordered[target_index]
            if current == target:
                yield target, image
                target_index += 1
            current += 1
    finally:
        capture.release()


def main() -> int:
    args = parse_args()
    baseline = args.baseline_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True)

    violations = [
        row
        for row in csv.DictReader(
            (baseline / "recall_violations.csv").open(encoding="utf-8")
        )
        if row["mode"] == "cpu_exact" and row["target_interval"] == "2"
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    for row in violations:
        key = (
            row["run_id"],
            row["label"],
            row["track_id"],
            int(row["run_segment_id"]),
        )
        groups.setdefault(key, []).append(row)

    module = _runtime_module()
    rendered: dict[tuple[str, int], np.ndarray] = {}
    records: list[dict[str, object]] = []
    group_audit = []
    for (run_name, label, track_id, reported_run_id), source_rows in groups.items():
        run_root = baseline / "runs" / run_name
        source_sqlite = next(
            (run_root / "shared/06_polygon_preparation/classes").glob(
                f"*_{label}/endpoint_extended.sqlite"
            )
        )
        track_rows = _load_track_rows(module, source_sqlite, track_id)
        streams, _stats = module.build_track_streams(
            track_rows,
            anchors_per_contour=14,
            predictor=None,
            adaptive_anchor_counts=False,
            min_anchors_per_contour=14,
            gapfill_enabled=True,
            gapfill_max_gap=15,
            max_tracks=0,
            max_run_frames=30000,
            run_overlap_frames=900,
        )
        target_frames = {int(row["frame"]) for row in source_rows}
        run = next(
            value
            for value in streams
            if target_frames.intersection(map(int, value.frame_numbers))
        )
        frame_to_index = {int(value): index for index, value in enumerate(run.frame_numbers)}
        old, spatial_stats = build_spatial_track(
            run.gt_polygons, replace(CANDIDATE, vertices_per_component=14)
        )
        evaluator = _ExactAdapter(module, run.gt_polygons)
        repaired, repaired_count, maximum_scale = _repair_sequence_exact_recall(
            run,
            evaluator,
            old,
            recall_floor=float(CANDIDATE.spatial_recall_floor),
        )
        keyframe_path = (
            run_root
            / "cpu_exact/interval_2/polygon/interval_2/polygon14_keyframe_v1"
            / label
            / "runtime/opt/final_keyframes.json"
        )
        saved_keys = {
            int(value["frame"]): value
            for value in json.loads(keyframe_path.read_text(encoding="utf-8"))
        }
        shared = json.loads(
            (run_root / "shared/shared_manifest.json").read_text(encoding="utf-8")
        )
        video_path = str(shared["run"]["video"])
        row_by_frame = {int(value["frame"]): value for value in source_rows}
        exact_matches = 0
        repaired_passes = 0
        for frame, frame_image in _decode_selected_frames(video_path, target_frames):
            index = frame_to_index[frame]
            reference = run.gt_polygons[index]
            before = [np.asarray(value) for value in old[index]]
            after = [np.asarray(value) for value in repaired[index]]
            saved = np.asarray(saved_keys[frame]["polygons"], dtype=np.float32)
            if not np.array_equal(np.asarray(before, np.float32), saved):
                raise RuntimeError(f"pre-DP polygon mismatch at {run_name}:{frame}")
            exact_matches += 1
            values_before = evaluator.exact_frame_metrics(
                index, old[index].reshape(-1), spatial_stats.components, 14
            )
            values_after = evaluator.exact_frame_metrics(
                index, repaired[index].reshape(-1), spatial_stats.components, 14
            )
            metric_before = {"recall": values_before[4], "iou": values_before[6]}
            metric_after = {"recall": values_after[4], "iou": values_after[6]}
            source_row = row_by_frame[frame]
            if abs(metric_before["recall"] - float(source_row["recall"])) > 1e-12:
                raise RuntimeError(f"historical Recall mismatch at {run_name}:{frame}")
            if abs(metric_before["iou"] - float(source_row["iou"])) > 1e-12:
                raise RuntimeError(f"historical IoU mismatch at {run_name}:{frame}")
            repaired_passes += int(metric_after["recall"] + 1e-12 >= 0.97)
            bounds = _bounds([reference, before, after])
            image = np.concatenate(
                [
                    _full_context(frame_image, bounds),
                    _overlay_panel(
                        frame_image,
                        reference,
                        None,
                        bounds,
                        title="AI source mask (cyan)",
                        color=_CYAN,
                        metrics=None,
                        raw_only=True,
                    ),
                    _overlay_panel(
                        frame_image,
                        reference,
                        before,
                        bounds,
                        title="Before DP: original 14pt (red)",
                        color=_RED,
                        metrics=metric_before,
                    ),
                    _overlay_panel(
                        frame_image,
                        reference,
                        after,
                        bounds,
                        title="Before DP: repaired 14pt (green)",
                        color=_GREEN,
                        metrics=metric_after,
                    ),
                ],
                axis=1,
            )
            filename = (
                f"{run_name}_{label}_track{track_id}_run{reported_run_id}_"
                f"f{frame:06d}.jpg"
            )
            destination = frames_root / filename
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write {destination}")
            rendered[(run_name, frame)] = image
            records.append(
                {
                    "file": f"frames/{filename}",
                    "run": run_name,
                    "label": label,
                    "track_id": track_id,
                    "run_segment_id": reported_run_id,
                    "frame": frame,
                    "before_recall": float(metric_before["recall"]),
                    "before_iou": float(metric_before["iou"]),
                    "after_recall": float(metric_after["recall"]),
                    "after_iou": float(metric_after["iou"]),
                    "vertex_count_before": 14,
                    "vertex_count_after": 14,
                }
            )
        group_audit.append(
            {
                "run": run_name,
                "track_id": track_id,
                "run_segment_id": reported_run_id,
                "historical_violations": len(source_rows),
                "pre_dp_coordinate_matches": exact_matches,
                "repaired_recall_passes": repaired_passes,
                "repaired_frames_in_whole_run": int(repaired_count),
                "maximum_repair_scale": float(maximum_scale),
            }
        )

    records.sort(key=lambda value: (str(value["run"]), int(value["frame"])))
    selected: list[dict[str, object]] = []
    for run_name in sorted({str(value["run"]) for value in records}):
        candidates = sorted(
            (value for value in records if value["run"] == run_name),
            key=lambda value: float(value["before_recall"]),
        )
        selected.extend(candidates[: min(4, len(candidates))])
    for value in sorted(records, key=lambda row: float(row["before_recall"])):
        if value in selected:
            continue
        selected.append(value)
        if len(selected) >= int(args.contact_samples):
            break
    selected = selected[: int(args.contact_samples)]
    contact = np.concatenate(
        [rendered[(str(value["run"]), int(value["frame"]))] for value in selected],
        axis=0,
    )
    contact_path = output_root / "00_contact_sheet_worst_recall.jpg"
    if not cv2.imwrite(str(contact_path), contact, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {contact_path}")

    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit = {
        "privacy": "Video frames decoded locally only; nothing was uploaded.",
        "comparison": [
            "original full frame",
            "AI source mask",
            "pre-DP original 14-point polygon",
            "pre-DP exact-Recall-repaired 14-point polygon",
        ],
        "colors": {
            "AI_source_outline": "cyan",
            "original_14_point": "red",
            "repaired_14_point": "green",
            "AI_pixels_missed_by_polygon": "yellow",
        },
        "source_violation_rows": len(violations),
        "pre_dp_coordinate_matches": sum(
            int(value["pre_dp_coordinate_matches"]) for value in group_audit
        ),
        "repaired_recall_passes": sum(
            int(value["repaired_recall_passes"]) for value in group_audit
        ),
        "groups": group_audit,
        "contact_sheet_samples": [value["file"] for value in selected],
        "frames": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "README.txt").write_text(
        "Left to right: original full frame; AI source mask; original pre-DP "
        "14-point polygon; exact-Recall-repaired pre-DP 14-point polygon.\n"
        "Cyan=AI source, red=original 14-point, green=repaired 14-point, "
        "yellow=AI pixels missed by the displayed polygon.\n"
        "All 130 original CPU-exact interval-2 Recall failures are in frames/.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "frames": len(records),
                "coordinate_matches": audit["pre_dp_coordinate_matches"],
                "repaired_passes": audit["repaired_recall_passes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
