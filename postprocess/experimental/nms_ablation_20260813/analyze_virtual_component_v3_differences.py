#!/usr/bin/env python3
"""Explain every retained-ID difference between legacy and virtual v3 NMS."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
HERE = Path(__file__).resolve().parent
for value in (POSTPROCESS, HERE):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_four_arm_v3 as base  # noqa: E402
from contracts.detections import iter_detection_records  # noqa: E402
from nms import AdaptiveNms  # noqa: E402
from nms.component_aware import _raster_mask  # noqa: E402
from nms.components import _geometry  # noqa: E402


def _id(value: dict[str, Any], frame: int, index: int) -> int | str:
    raw = value.get("source_detection_id")
    if raw is None:
        return f"synthetic:{frame}:{index}"
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def _by_id(values: list[dict[str, Any]], frame: int) -> dict[int | str, dict[str, Any]]:
    return {_id(value, frame, index): value for index, value in enumerate(values)}


def _legacy_trace(
    detections: list[dict[str, Any]], policy: AdaptiveNms
) -> dict[int, tuple[int, str]]:
    order = sorted(
        range(len(detections)),
        key=lambda index: (-float(detections[index].get("score") or 0.0), index),
    )
    suppressed: set[int] = set()
    result: dict[int, tuple[int, str]] = {}
    for position, winner in enumerate(order):
        if winner in suppressed:
            continue
        for loser in order[position + 1 :]:
            if loser in suppressed:
                continue
            reason = policy.pair_suppression_reason(
                detections[winner], detections[loser]
            )
            if reason is not None:
                suppressed.add(loser)
                result[loser] = (winner, reason)
    return result


def _coverage(value: dict[str, Any], union: base.UnionRaster | None) -> float | None:
    if union is None:
        return None
    return base._coverage_by_union(value, union)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p50": None,
            "mean": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-union-pixels", type=int, default=32_000_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = AdaptiveNms()
    detail: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    changed_frames = 0
    exact_4275_4280: list[dict[str, Any]] = []

    for run_dir in sorted((args.input_root / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        arms = run_dir / "arm_outputs"
        sources = [
            arms / "legacy.jsonl",
            arms / "virtual_component_v3.jsonl",
        ]
        if not all(path.is_file() for path in sources):
            continue
        source_jsonl = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )["source_jsonl"]
        raw_iter = iter_detection_records(Path(source_jsonl))
        legacy_iter = iter_detection_records(sources[0])
        v3_iter = iter_detection_records(sources[1])
        trace_by_frame: dict[int, list[dict[str, Any]]] = {}
        trace_path = run_dir / "virtual_component_trace.jsonl.gz"
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                trace_by_frame.setdefault(int(row["frame_index"]), []).append(row)
                trace_rows.append({"run_key": run_dir.name, **row})

        for raw, legacy, v3 in zip(raw_iter, legacy_iter, v3_iter, strict=True):
            frame = int(raw["frame_index"])
            if frame != int(legacy["frame_index"]) or frame != int(v3["frame_index"]):
                raise ValueError(f"frame misalignment in {run_dir.name}")
            raw_values = list(raw["detections"])
            legacy_values = list(legacy["detections"])
            v3_values = list(v3["detections"])
            raw_by_id = _by_id(raw_values, frame)
            legacy_by_id = _by_id(legacy_values, frame)
            v3_by_id = _by_id(v3_values, frame)
            legacy_ids, v3_ids = set(legacy_by_id), set(v3_by_id)
            if legacy_ids != v3_ids:
                changed_frames += 1
            legacy_union = base._union_raster(
                legacy_values, max_pixels=args.max_union_pixels
            )
            v3_union = base._union_raster(v3_values, max_pixels=args.max_union_pixels)
            legacy_suppression = _legacy_trace(raw_values, policy)
            raw_indices = {
                _id(value, frame, index): index
                for index, value in enumerate(raw_values)
            }

            for source_id in sorted(v3_ids - legacy_ids, key=str):
                value = raw_by_id[source_id]
                loser_index = raw_indices[source_id]
                winner_info = legacy_suppression.get(loser_index)
                winner_id: int | str | None = None
                reason: str | None = None
                if winner_info is not None:
                    winner_index, reason = winner_info
                    winner_id = _id(raw_values[winner_index], frame, winner_index)
                geometry = _geometry(value)
                raster = _raster_mask(value)
                detail.append(
                    {
                        "run_key": run_dir.name,
                        "frame_index": frame,
                        "direction": "retained_by_v3_only",
                        "source_detection_id": source_id,
                        "class_name": value.get("class_name") or value.get("label"),
                        "score": float(value.get("score") or 0.0),
                        "foreground_components": len(geometry.foreground)
                        if geometry
                        else 0,
                        "raster_area": raster.area if raster else 0,
                        "other_arm_union_coverage": _coverage(value, legacy_union),
                        "suppressor_source_detection_id": winner_id,
                        "suppression_reason": reason,
                    }
                )
            for source_id in sorted(legacy_ids - v3_ids, key=str):
                value = raw_by_id[source_id]
                raster = _raster_mask(value)
                events = [
                    row
                    for row in trace_by_frame.get(frame, [])
                    if row.get("loser_source_detection_id") == source_id
                ]
                detail.append(
                    {
                        "run_key": run_dir.name,
                        "frame_index": frame,
                        "direction": "suppressed_by_v3_only",
                        "source_detection_id": source_id,
                        "class_name": value.get("class_name") or value.get("label"),
                        "score": float(value.get("score") or 0.0),
                        "foreground_components": len(_geometry(value).foreground)
                        if _geometry(value)
                        else 0,
                        "raster_area": raster.area if raster else 0,
                        "other_arm_union_coverage": _coverage(value, v3_union),
                        "suppressor_source_detection_id": (
                            events[0].get("winner_source_detection_id")
                            if events
                            else None
                        ),
                        "suppression_reason": events[0].get("reason")
                        if events
                        else None,
                    }
                )

            if run_dir.name == "v3__kpi_2025_12" and 4275 <= frame <= 4280:
                exact_4275_4280.append(
                    {
                        "frame_index": frame,
                        "raw_ids": sorted(raw_by_id, key=str),
                        "legacy_ids": sorted(legacy_by_id, key=str),
                        "virtual_v3_ids": sorted(v3_by_id, key=str),
                        "identical_retention": legacy_ids == v3_ids,
                        "v3_trace": trace_by_frame.get(frame, []),
                    }
                )

    fields = [
        "run_key",
        "frame_index",
        "direction",
        "source_detection_id",
        "class_name",
        "score",
        "foreground_components",
        "raster_area",
        "other_arm_union_coverage",
        "suppressor_source_detection_id",
        "suppression_reason",
    ]
    with (output / "changed_detections.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail)
    with (output / "virtual_component_trace.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        trace_fields = sorted({key for row in trace_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)

    by_direction = Counter(row["direction"] for row in detail)
    by_class = Counter((row["direction"], row["class_name"]) for row in detail)
    by_reason = Counter((row["direction"], row["suppression_reason"]) for row in detail)
    coverages = {
        direction: _quantiles(
            [
                float(row["other_arm_union_coverage"])
                for row in detail
                if row["direction"] == direction
                and row["other_arm_union_coverage"] is not None
            ]
        )
        for direction in by_direction
    }
    summary = {
        "changed_frames": changed_frames,
        "changed_detections": len(detail),
        "by_direction": dict(by_direction),
        "by_class": {f"{key[0]}::{key[1]}": value for key, value in by_class.items()},
        "by_reason": {f"{key[0]}::{key[1]}": value for key, value in by_reason.items()},
        "other_arm_union_coverage": coverages,
        "virtual_trace_reasons": dict(Counter(row["reason"] for row in trace_rows)),
        "kpi_frames_4275_4280": exact_4275_4280,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
