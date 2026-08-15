#!/usr/bin/env python3
"""Render mask-only 14-vs-20 spatial fallback review images."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from experimental.production_candidate_polygon14.config import CANDIDATE
from experimental.production_candidate_polygon14.integration import (
    _repair_sequence_exact_recall,
)
from experimental.production_candidate_polygon14.spatial import build_spatial_track


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = (
    ROOT / "output/production_adaptive_vertices_14_20_known_failures_v2_20260815",
    ROOT / "output/production_adaptive_vertices_14_20_remaining_v3_20260815",
)
DEFAULT_OUTPUT = ROOT / "output/polygon_vertex_fallback_14_vs_20_review_20260815"


def _runtime_module():
    import sys

    runtime_dir = ROOT / "postprocess/experimental/0809"
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    import phase2_runtime

    return phase2_runtime._load_production_runtime()._build_embedded_polygon_v22_module()


class _ExactAdapter:
    def __init__(self, module, references: list[list[np.ndarray]]) -> None:
        self.module = module
        self.references = references

    def exact_frame_metrics(
        self,
        frame_index: int,
        vector: np.ndarray,
        components: int,
        vertices: int,
    ) -> tuple[float, ...]:
        polygons = self.module.split_vector_to_polygons(
            np.asarray(vector, dtype=np.float32), int(components), int(vertices)
        )
        value = self.module.compute_exact_metrics_from_polygons(
            self.references[int(frame_index)], polygons
        )
        return (
            float(value.get("gt_area", 0.0)),
            float(value.get("pred_area", 0.0)),
            float(value.get("intersection", 0.0)),
            float(value.get("union", 0.0)),
            float(value["recall"]),
            float(value["precision"]),
            float(value["iou"]),
        )


def _load_track_rows(module, sqlite_path: Path, track_id: str):
    with sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT frame,track_id,polygons FROM masks "
            "WHERE track_id=? ORDER BY frame",
            (str(track_id),),
        ).fetchall()
    return [
        module.TrackRow(
            frame=int(frame),
            track_id=str(value),
            polygons=module.parse_polygons(str(polygons)),
        )
        for frame, value, polygons in rows
    ]


def _metrics(module, reference, polygons) -> dict[str, float]:
    value = module.compute_exact_metrics_from_polygons(reference, polygons)
    return {name: float(value[name]) for name in ("recall", "precision", "iou")}


def _bounds(polygons: list[list[np.ndarray]]) -> tuple[float, float, float, float]:
    points = np.concatenate(
        [np.asarray(p, dtype=np.float32) for group in polygons for p in group], axis=0
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    margin = max(float(np.max(maximum - minimum)) * 0.12, 8.0)
    return (
        float(minimum[0] - margin),
        float(minimum[1] - margin),
        float(maximum[0] + margin),
        float(maximum[1] + margin),
    )


def _transform(polygon: np.ndarray, bounds, size: int) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    scale = min((size - 40) / max(x2 - x1, 1.0), (size - 40) / max(y2 - y1, 1.0))
    offset = np.asarray(
        [(size - (x2 - x1) * scale) / 2.0, (size - (y2 - y1) * scale) / 2.0]
    )
    value = (np.asarray(polygon) - np.asarray([x1, y1])) * scale + offset
    return np.round(value).astype(np.int32)


def _panel(
    reference,
    approximation,
    bounds,
    *,
    title: str,
    color: tuple[int, int, int],
    metrics: dict[str, float] | None,
    size: int = 500,
) -> np.ndarray:
    image = np.full((size, size, 3), 20, dtype=np.uint8)
    raw = [_transform(poly, bounds, size) for poly in reference]
    cv2.fillPoly(image, raw, (75, 75, 75))
    for contour in raw:
        cv2.polylines(image, [contour], True, (210, 210, 210), 2, cv2.LINE_AA)
    if approximation is not None:
        for polygon in approximation:
            contour = _transform(polygon, bounds, size)
            cv2.polylines(image, [contour], True, color, 3, cv2.LINE_AA)
            for x, y in contour:
                cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    cv2.putText(image, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if metrics is not None:
        text = (
            f"Recall {metrics['recall']:.4f}  IoU {metrics['iou']:.4f}  "
            f"Precision {metrics['precision']:.4f}"
        )
        cv2.putText(image, text, (14, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
    return image


def _discover_groups(roots: tuple[Path, ...]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for root in roots:
        for path in root.glob("runs/*/cpu_exact/interval_2/polygon/interval_2/**/final_keyframes.json"):
            label = path.parents[2].name
            run_root = path.parents[8]
            rows = json.loads(path.read_text(encoding="utf-8"))
            groups: dict[tuple[str, int], list[int]] = {}
            for row in rows:
                polygons = row.get("polygons") or []
                if not polygons or any(len(polygon) != 20 for polygon in polygons):
                    continue
                key = (str(row["track_id"]), int(row["run_id"]))
                groups.setdefault(key, []).append(int(row["frame"]))
            for (track_id, run_id), frames in groups.items():
                output.append(
                    {
                        "run": run_root.name,
                        "run_root": run_root,
                        "label": label,
                        "track_id": track_id,
                        "run_id": run_id,
                        "keyframes": sorted(frames),
                        "span": max(frames) - min(frames) + 1,
                    }
                )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    module = _runtime_module()
    inventory = _discover_groups(DEFAULT_INPUTS)
    # Prefer short runs for a quick diagnostic, while spreading samples across
    # source videos and labels instead of over-representing one long track.
    inventory.sort(key=lambda value: (int(value["span"]), str(value["run"])))
    selected = []
    per_stratum: dict[tuple[str, str], int] = {}
    for value in inventory:
        stratum = (str(value["run"]), str(value["label"]))
        if per_stratum.get(stratum, 0) >= 2:
            continue
        selected.append(value)
        per_stratum[stratum] = per_stratum.get(stratum, 0) + 1
        if len(selected) >= int(args.samples):
            break

    manifest = []
    rendered = []
    cache: dict[tuple[Path, str], list] = {}
    for sample_index, item in enumerate(selected, start=1):
        run_root = Path(item["run_root"])
        label = str(item["label"])
        source_sqlite = next(
            (run_root / "shared/06_polygon_preparation/classes").glob(
                f"*_{label}/endpoint_extended.sqlite"
            )
        )
        cache_key = (source_sqlite, str(item["track_id"]))
        if cache_key not in cache:
            rows = _load_track_rows(module, source_sqlite, str(item["track_id"]))
            cache[cache_key], _stats = module.build_track_streams(
                rows,
                anchors_per_contour=20,
                predictor=None,
                adaptive_anchor_counts=False,
                min_anchors_per_contour=14,
                gapfill_enabled=True,
                gapfill_max_gap=15,
                max_tracks=0,
                max_run_frames=30000,
                run_overlap_frames=900,
            )
        keyframes = set(int(value) for value in item["keyframes"])
        run = next(
            value
            for value in cache[cache_key]
            if keyframes.intersection(int(frame) for frame in value.frame_numbers)
        )
        candidates = {}
        for vertices in (14, 20):
            config = replace(CANDIDATE, vertices_per_component=vertices)
            anchors, _stats = build_spatial_track(run.gt_polygons, config)
            evaluator = _ExactAdapter(module, run.gt_polygons)
            anchors, _repaired, _scale = _repair_sequence_exact_recall(
                run,
                evaluator,
                anchors,
                recall_floor=CANDIDATE.spatial_recall_floor,
            )
            candidates[vertices] = anchors

        best = None
        for local_index, frame in enumerate(run.frame_numbers):
            if int(frame) not in keyframes:
                continue
            reference = run.gt_polygons[local_index]
            poly14 = [np.asarray(value) for value in candidates[14][local_index]]
            poly20 = [np.asarray(value) for value in candidates[20][local_index]]
            metric14 = _metrics(module, reference, poly14)
            metric20 = _metrics(module, reference, poly20)
            score = (
                metric20["iou"] - metric14["iou"],
                metric20["recall"] - metric14["recall"],
            )
            if best is None or score > best[0]:
                best = (score, int(frame), reference, poly14, poly20, metric14, metric20)
        if best is None:
            continue
        _score, frame, reference, poly14, poly20, metric14, metric20 = best
        bounds = _bounds([reference, poly14, poly20])
        panels = [
            _panel(
                reference,
                None,
                bounds,
                title="AI source mask",
                color=(255, 255, 255),
                metrics=None,
            ),
            _panel(
                reference,
                poly14,
                bounds,
                title="14 vertices (counterfactual)",
                color=(80, 80, 255),
                metrics=metric14,
            ),
            _panel(
                reference,
                poly20,
                bounds,
                title="20 vertices (selected fallback)",
                color=(80, 230, 80),
                metrics=metric20,
            ),
        ]
        image = np.concatenate(panels, axis=1)
        filename = (
            f"sample_{sample_index:02d}_{item['run']}_{label}_"
            f"track{item['track_id']}_run{item['run_id']}_f{frame:06d}.jpg"
        )
        destination = output_root / filename
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write {destination}")
        rendered.append(image)
        manifest.append(
            {
                "file": filename,
                "run": item["run"],
                "label": label,
                "track_id": item["track_id"],
                "run_id": item["run_id"],
                "frame": frame,
                "metrics_14": metric14,
                "metrics_20": metric20,
                "iou_improvement": metric20["iou"] - metric14["iou"],
                "recall_improvement": metric20["recall"] - metric14["recall"],
            }
        )

    if rendered:
        contact = np.concatenate(rendered, axis=0)
        cv2.imwrite(
            str(output_root / "00_contact_sheet.jpg"),
            contact,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "privacy": "Mask geometry only; no video pixels were opened.",
                "description": (
                    "Frames from continuous runs selected at 20 vertices. "
                    "Panels isolate the pre-temporal spatial approximation."
                ),
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_root), "samples": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
