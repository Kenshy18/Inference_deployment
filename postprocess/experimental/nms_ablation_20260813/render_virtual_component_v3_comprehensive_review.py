#!/usr/bin/env python3
"""Render every legacy-vs-virtual-component impact frame locally."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
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

from contracts.detections import iter_detection_records  # noqa: E402
from nms.components import fill_holes_and_remove_tiny_islands  # noqa: E402
from render_component_candidate_v2_review_gallery import (  # noqa: E402
    _detection_id,
    _put,
    _render_panel,
    seek_frame,
)


def _ids(record: dict[str, Any]) -> set[int | str]:
    frame = int(record["frame_index"])
    return {
        _detection_id(value, frame, index)
        for index, value in enumerate(record["detections"])
    }


def _resize(panel: np.ndarray, width: int) -> np.ndarray:
    if panel.shape[1] == width:
        return panel
    height = max(1, int(round(panel.shape[0] * width / panel.shape[1])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def _header(
    width: int,
    run_key: str,
    frame: int,
    categories: list[str],
    records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> np.ndarray:
    raw, legacy, candidate = records
    legacy_ids, candidate_ids = _ids(legacy), _ids(candidate)
    lines = [
        f"run={run_key} frame={frame} categories={','.join(categories)}",
    ]
    for index, value in enumerate(raw["detections"]):
        source_id = _detection_id(value, frame, index)
        label = str(value.get("class_name") or value.get("label") or "unknown")
        lines.append(
            f"D{source_id} score={float(value.get('score') or 0.0):.3f} "
            f"class={label} legacy={'K' if source_id in legacy_ids else 'X'} "
            f"candidate={'K' if source_id in candidate_ids else 'X'}"
        )
    lines.append(
        "same D-ID=same colour; solid+fill=retained; dashed=raw suppressed or geometry-changed"
    )
    image = np.full((22 + 28 * len(lines), width, 3), 12, np.uint8)
    for index, line in enumerate(lines):
        _put(image, line, (14, 24 + 28 * index), scale=0.50)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--topology-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--candidate-arm", default="virtual_component_v3")
    parser.add_argument("--candidate-label", default="VIRTUAL COMPONENT v3")
    args = parser.parse_args()
    root = args.ablation_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with sqlite3.connect(
        f"file:{args.topology_sqlite.resolve()}?mode=ro", uri=True
    ) as connection:
        videos = {
            str(run_key): Path(str(video)).resolve()
            for run_key, video in connection.execute(
                "SELECT run_key,input_video FROM audit_runs WHERE model_key='v3'"
            )
        }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    images = staging / "frames"
    images.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    category_counts: dict[str, int] = defaultdict(int)
    try:
        for run_dir in sorted((root / "runs").iterdir()):
            if not run_dir.is_dir():
                continue
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            raw_path = Path(str(summary["source_jsonl"]))
            legacy_path = run_dir / "arm_outputs/legacy.jsonl"
            candidate_path = run_dir / "arm_outputs" / f"{args.candidate_arm}.jsonl"
            trace_frames: set[int] = set()
            trace_path = run_dir / f"{args.candidate_arm}_trace.jsonl.gz"
            if not trace_path.exists() and args.candidate_arm == "virtual_component_v3":
                trace_path = run_dir / "virtual_component_trace.jsonl.gz"
            with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    value = json.loads(line)
                    if value.get("reason") in {
                        "island_subordinate_to_main",
                        "island_island_legacy_nms",
                        "island_island_mask_nms",
                    }:
                        trace_frames.add(int(value["frame_index"]))
            iterators = [
                iter_detection_records(path)
                for path in (raw_path, legacy_path, candidate_path)
            ]
            for records in itertools.zip_longest(*iterators):
                if any(value is None for value in records):
                    raise RuntimeError(f"arm length mismatch: {run_dir}")
                raw, legacy, candidate = records  # type: ignore[misc]
                frame = int(raw["frame_index"])
                categories: list[str] = []
                legacy_ids, candidate_ids = _ids(legacy), _ids(candidate)
                if candidate_ids - legacy_ids:
                    categories.append("01_retained_by_candidate_only")
                if legacy_ids - candidate_ids:
                    categories.append("02_suppressed_by_candidate_only")
                _cleaned, topology = fill_holes_and_remove_tiny_islands(
                    list(raw["detections"]),
                    fill_all_holes=True,
                    unconditional_owner_ratio_max=0.01,
                )
                if topology.holes_filled:
                    categories.append("03_holes_filled")
                if topology.tiny_islands_removed:
                    categories.append("04_tiny_islands_removed")
                if frame in trace_frames:
                    categories.append("05_component_pair_island_removed")
                if run_dir.name == "v3__kpi_2025_12" and 4275 <= frame <= 4280:
                    categories.append("06_kpi_nested_regression")
                if not categories:
                    continue
                video = videos[run_dir.name]
                image = seek_frame(video, frame)
                panels = [
                    _render_panel(image, raw, raw, "RAW AI MASKS (pre-NMS)"),
                    _render_panel(image, raw, legacy, "LEGACY PRODUCTION"),
                    _render_panel(image, raw, candidate, args.candidate_label),
                ]
                panels = [_resize(panel, args.panel_width) for panel in panels]
                body = np.concatenate(panels, axis=1)
                result = np.vstack(
                    [
                        _header(
                            body.shape[1], run_dir.name, frame, categories, records
                        ),
                        body,
                    ]
                )
                filename = f"{run_dir.name}_f{frame:06d}.jpg"
                path = images / filename
                if not cv2.imwrite(str(path), result, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(path)
                decoded = cv2.imread(str(path))
                if decoded is None or decoded.shape != result.shape:
                    raise RuntimeError(f"JPEG structural validation failed: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                row = {
                    "run_key": run_dir.name,
                    "frame_index": frame,
                    "categories": categories,
                    "raw_ids": sorted(_ids(raw), key=str),
                    "legacy_ids": sorted(legacy_ids, key=str),
                    "candidate_ids": sorted(candidate_ids, key=str),
                    "holes_filled": topology.holes_filled,
                    "tiny_islands_removed": topology.tiny_islands_removed,
                    "image": str(output / "frames" / filename),
                    "sha256": digest,
                }
                manifest.append(row)
                for category in categories:
                    category_counts[category] += 1

        with (staging / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = [
                "run_key",
                "frame_index",
                "categories",
                "raw_ids",
                "legacy_ids",
                "candidate_ids",
                "holes_filled",
                "tiny_islands_removed",
                "image",
                "sha256",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in manifest:
                writer.writerow(
                    {
                        **row,
                        "categories": "|".join(row["categories"]),
                        "raw_ids": json.dumps(row["raw_ids"], ensure_ascii=False),
                        "legacy_ids": json.dumps(row["legacy_ids"], ensure_ascii=False),
                        "candidate_ids": json.dumps(
                            row["candidate_ids"], ensure_ascii=False
                        ),
                    }
                )
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "privacy": "Local OpenCV only; no network or image-view tool.",
                    "panel_order": [
                        "raw AI masks",
                        "legacy Production",
                        args.candidate_label,
                    ],
                    "images": len(manifest),
                    "category_counts": dict(category_counts),
                    "rows": manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(manifest),
                "category_counts": category_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
