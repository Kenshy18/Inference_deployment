#!/usr/bin/env python3
"""Render low-IoU polygon comparisons without opening video pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-sqlite", type=Path, required=True)
    parser.add_argument("--candidate-sqlite", type=Path, required=True)
    parser.add_argument("--control-sqlite", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    return parser


def _read(path: Path) -> dict[int, np.ndarray]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT frame, polygons FROM masks ORDER BY frame"
        ).fetchall()
    finally:
        connection.close()
    result: dict[int, np.ndarray] = {}
    for frame, payload in rows:
        polygons = json.loads(str(payload))
        if len(polygons) != 1:
            raise ValueError(f"frame {frame}: expected one component")
        result[int(frame)] = np.asarray(polygons[0], dtype=np.float64)
    return result


def _masks(
    reference: np.ndarray,
    candidates: list[np.ndarray],
    padding: int = 8,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    all_points = np.concatenate([reference, *candidates], axis=0)
    origin = np.floor(np.min(all_points, axis=0)).astype(np.int32) - padding
    maximum = np.ceil(np.max(all_points, axis=0)).astype(np.int32) + padding
    width = max(1, int(maximum[0] - origin[0] + 1))
    height = max(1, int(maximum[1] - origin[1] + 1))

    def raster(points: np.ndarray) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted = np.rint(points - origin[None, :]).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
        return mask

    return raster(reference), [raster(value) for value in candidates], origin


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(reference & candidate))
    gt_area = int(np.count_nonzero(reference))
    pred_area = int(np.count_nonzero(candidate))
    union = gt_area + pred_area - intersection
    return {
        "iou": float(intersection / union) if union else 1.0,
        "recall": float(intersection / gt_area) if gt_area else 1.0,
        "precision": float(intersection / pred_area) if pred_area else 1.0,
        "area_ratio": float(pred_area / gt_area) if gt_area else 1.0,
        "gt_area": gt_area,
        "pred_area": pred_area,
    }


def _diagnostic(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    image = np.full((*reference.shape, 3), 20, dtype=np.uint8)
    overlap = (reference != 0) & (candidate != 0)
    missed = (reference != 0) & (candidate == 0)
    excess = (reference == 0) & (candidate != 0)
    image[overlap] = (70, 190, 80)   # green
    image[missed] = (235, 120, 35)   # blue (BGR)
    image[excess] = (45, 45, 235)    # red
    return image


def _fit_panel(image: np.ndarray, width: int = 520, height: int = 360) -> np.ndarray:
    scale = min((width - 20) / image.shape[1], (height - 70) / image.shape[0])
    scale = max(scale, 0.1)
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))),
         max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = np.full((height, width, 3), 12, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = 58 + (height - 58 - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _label(
    panel: np.ndarray,
    title: str,
    metrics: dict[str, float],
) -> None:
    cv2.putText(
        panel,
        title,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"IoU {metrics['iou']:.4f}  Recall {metrics['recall']:.4f}  "
            f"Precision {metrics['precision']:.4f}  Area x{metrics['area_ratio']:.3f}"
        ),
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = _parser().parse_args()
    reference = _read(args.reference_sqlite)
    candidate = _read(args.candidate_sqlite)
    control = _read(args.control_sqlite) if args.control_sqlite else None
    common = sorted(set(reference) & set(candidate))
    if control is not None:
        common = sorted(set(common) & set(control))
    if not common:
        raise ValueError("the SQLite inputs contain no common frames")

    rows = []
    for frame in common:
        candidates = [candidate[frame]]
        if control is not None:
            candidates.append(control[frame])
        gt_mask, masks, _origin = _masks(reference[frame], candidates)
        row = {
            "frame": int(frame),
            "candidate": _metrics(gt_mask, masks[0]),
        }
        if control is not None:
            row["control"] = _metrics(gt_mask, masks[1])
        rows.append(row)
    rows.sort(key=lambda value: (value["candidate"]["iou"], value["frame"]))
    selected = rows[: max(1, int(args.count))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    for rank, row in enumerate(selected, start=1):
        frame = int(row["frame"])
        polygons = [candidate[frame]]
        if control is not None:
            polygons.append(control[frame])
        gt_mask, masks, _origin = _masks(reference[frame], polygons)
        panels = []
        candidate_panel = _fit_panel(_diagnostic(gt_mask, masks[0]))
        _label(candidate_panel, f"rank {rank:02d} frame {frame} | greedy terminal", row["candidate"])
        panels.append(candidate_panel)
        if control is not None:
            control_panel = _fit_panel(_diagnostic(gt_mask, masks[1]))
            _label(control_panel, f"rank {rank:02d} frame {frame} | recommended", row["control"])
            panels.append(control_panel)
        image = np.concatenate(panels, axis=1)
        cv2.putText(
            image,
            "green=overlap  blue=raw only  red=approximation only",
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        output = args.output_dir / f"rank_{rank:02d}_frame_{frame}.png"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"failed to write {output}")
        rendered.append(image)

    thumb_width, thumb_height = 520, 180
    thumbs = [
        cv2.resize(value, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        for value in rendered
    ]
    columns = 2
    rows_count = (len(thumbs) + columns - 1) // columns
    sheet = np.full((rows_count * thumb_height, columns * thumb_width, 3), 8, np.uint8)
    for index, thumb in enumerate(thumbs):
        y = (index // columns) * thumb_height
        x = (index % columns) * thumb_width
        sheet[y : y + thumb_height, x : x + thumb_width] = thumb
    cv2.imwrite(str(args.output_dir / "contact_sheet.png"), sheet)
    report = {
        "privacy": "SQLite polygon geometry only; no video pixels opened",
        "reference_sqlite": str(args.reference_sqlite.resolve()),
        "candidate_sqlite": str(args.candidate_sqlite.resolve()),
        "control_sqlite": str(args.control_sqlite.resolve()) if args.control_sqlite else None,
        "evaluated_frames": len(common),
        "selected": selected,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
