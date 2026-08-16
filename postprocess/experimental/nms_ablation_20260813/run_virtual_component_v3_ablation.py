#!/usr/bin/env python3
"""All-V3 comparison of legacy, v2 mask-IoU, and virtual-component v3 NMS."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS = ROOT / "postprocess"
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_four_arm_v3 as base  # noqa: E402
from contracts.detections import dumps_json_line, iter_detection_records  # noqa: E402
from nms import (  # noqa: E402
    AdaptiveNms,
    ComponentAwareMaskNms,
    VirtualComponentMaskNms,
    VirtualComponentNms,
)


ARMS = (
    "legacy",
    "component_candidate_v2",
    "virtual_component_v3",
    "virtual_component_mask_v4",
)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _id(value: dict[str, Any], frame: int, index: int) -> int | str:
    raw = value.get("source_detection_id")
    if raw is None:
        return f"synthetic:{frame}:{index}"
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def _ids(values: list[dict[str, Any]], frame: int) -> list[int | str]:
    result = [_id(value, frame, index) for index, value in enumerate(values)]
    if len(result) != len(set(result)):
        raise ValueError(f"frame {frame}: duplicate source_detection_id")
    return result


def _geometry_changed(
    before: list[dict[str, Any]], after: list[dict[str, Any]], frame: int
) -> int:
    before_by_id = {
        _id(value, frame, index): value for index, value in enumerate(before)
    }
    changed = 0
    for index, value in enumerate(after):
        source_id = _id(value, frame, index)
        original = before_by_id.get(source_id)
        if original is None:
            continue
        if (original.get("polygons") or []) != (value.get("polygons") or []):
            changed += 1
    return changed


def _apply(
    detections: list[dict[str, Any]],
    legacy: AdaptiveNms,
    candidate_v2: ComponentAwareMaskNms,
    candidate_v3: VirtualComponentNms,
    candidate_v4: VirtualComponentMaskNms,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, int],
    dict[str, list[dict[str, Any]]],
]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    timings: dict[str, int] = {}

    started = time.perf_counter_ns()
    outputs["legacy"] = legacy.apply(detections)
    timings["legacy"] = time.perf_counter_ns() - started
    diagnostics["legacy"] = {"nms_suppressed": len(detections) - len(outputs["legacy"])}

    started = time.perf_counter_ns()
    outputs["component_candidate_v2"], v2_stats = candidate_v2.apply_with_diagnostics(
        detections
    )
    timings["component_candidate_v2"] = time.perf_counter_ns() - started
    diagnostics["component_candidate_v2"] = v2_stats.as_dict()

    started = time.perf_counter_ns()
    outputs["virtual_component_v3"], v3_stats, trace = candidate_v3.apply_with_trace(
        detections
    )
    timings["virtual_component_v3"] = time.perf_counter_ns() - started
    diagnostics["virtual_component_v3"] = v3_stats.as_dict()

    started = time.perf_counter_ns()
    (
        outputs["virtual_component_mask_v4"],
        v4_stats,
        trace_v4,
    ) = candidate_v4.apply_with_trace(detections)
    timings["virtual_component_mask_v4"] = time.perf_counter_ns() - started
    diagnostics["virtual_component_mask_v4"] = v4_stats.as_dict()
    return (
        outputs,
        diagnostics,
        timings,
        {
            "virtual_component_v3": trace,
            "virtual_component_mask_v4": trace_v4,
        },
    )


def run_one(
    run: base.RunInput,
    lineage: base.InputLineage,
    output_root: Path,
    *,
    max_union_pixels: int,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / run.run_key
    arm_dir = run_dir / "arm_outputs"
    arm_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        arm: (arm_dir / f"{arm}.jsonl").open("w", encoding="utf-8") for arm in ARMS
    }
    decisions_path = run_dir / "decisions.jsonl.gz"
    trace_paths = {
        arm: run_dir / f"{arm}_trace.jsonl.gz"
        for arm in ("virtual_component_v3", "virtual_component_mask_v4")
    }
    legacy = AdaptiveNms()
    candidate_v2 = ComponentAwareMaskNms()
    candidate_v3 = VirtualComponentNms()
    candidate_v4 = VirtualComponentMaskNms()
    totals = {
        arm: {
            "frames": 0,
            "input_detections": 0,
            "retained_detections": 0,
            "elapsed_ns": 0,
            "geometry_changed": 0,
            "diagnostics": Counter(),
            "union_recall": [],
            "union_iou": [],
            "removed_area_rate": [],
            "added_area_rate": [],
        }
        for arm in ARMS
    }
    pairwise = {
        (left, right): Counter()
        for i, left in enumerate(ARMS)
        for right in ARMS[i + 1 :]
    }
    wall_started = time.perf_counter()
    trace_handles = {
        arm: gzip.open(path, "wt", encoding="utf-8")
        for arm, path in trace_paths.items()
    }
    with gzip.open(decisions_path, "wt", encoding="utf-8") as decisions:
        try:
            for record in iter_detection_records(lineage.jsonl):
                frame = int(record["frame_index"])
                detections = list(record["detections"])
                outputs, diagnostics, timings, trace_by_arm = _apply(
                    detections, legacy, candidate_v2, candidate_v3, candidate_v4
                )
                input_ids = _ids(detections, frame)
                arm_ids = {arm: _ids(outputs[arm], frame) for arm in ARMS}
                decisions.write(
                    json.dumps(
                        {
                            "frame_index": frame,
                            "input_ids": input_ids,
                            "retained_ids": arm_ids,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for arm, events in trace_by_arm.items():
                    for event in events:
                        trace_handles[arm].write(
                            json.dumps(
                                {"frame_index": frame, **event},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                raw_union = base._union_raster(detections, max_pixels=max_union_pixels)
                for arm in ARMS:
                    values = totals[arm]
                    values["frames"] += 1
                    values["input_detections"] += len(detections)
                    values["retained_detections"] += len(outputs[arm])
                    values["elapsed_ns"] += timings[arm]
                    values["geometry_changed"] += _geometry_changed(
                        detections, outputs[arm], frame
                    )
                    values["diagnostics"].update(diagnostics[arm])
                    handles[arm].write(
                        dumps_json_line({**record, "detections": outputs[arm]}).decode(
                            "utf-8"
                        )
                    )
                    if raw_union is None:
                        continue
                    output_union = base._union_raster(
                        outputs[arm], max_pixels=max_union_pixels
                    )
                    if output_union is None:
                        continue
                    metrics = base._union_metrics(raw_union, output_union)
                    values["union_recall"].append(float(metrics["union_recall"]))
                    values["union_iou"].append(float(metrics["union_iou"]))
                    values["removed_area_rate"].append(
                        float(metrics["removed_area_rate"])
                    )
                    values["added_area_rate"].append(float(metrics["added_area_rate"]))
                for (left, right), counter in pairwise.items():
                    left_ids = set(arm_ids[left])
                    right_ids = set(arm_ids[right])
                    if left_ids != right_ids:
                        counter["changed_frames"] += 1
                    counter["left_only_ids"] += len(left_ids - right_ids)
                    counter["right_only_ids"] += len(right_ids - left_ids)
        finally:
            for handle in handles.values():
                handle.close()
            for handle in trace_handles.values():
                handle.close()

    arm_summary: dict[str, dict[str, Any]] = {}
    for arm, values in totals.items():
        seconds = float(values["elapsed_ns"]) / 1e9
        arm_summary[arm] = {
            "frames": int(values["frames"]),
            "input_detections": int(values["input_detections"]),
            "retained_detections": int(values["retained_detections"]),
            "suppressed_detections": int(values["input_detections"])
            - int(values["retained_detections"]),
            "geometry_changed": int(values["geometry_changed"]),
            "elapsed_seconds": seconds,
            "frames_per_second": int(values["frames"]) / seconds if seconds else None,
            "diagnostics": dict(values["diagnostics"]),
            "union_recall": _quantiles(values["union_recall"]),
            "union_iou": _quantiles(values["union_iou"]),
            "removed_area_rate": _quantiles(values["removed_area_rate"]),
            "added_area_rate": _quantiles(values["added_area_rate"]),
        }
    summary = {
        "run_key": run.run_key,
        "video_slug": run.video_slug,
        "source_jsonl": str(lineage.jsonl.resolve()),
        "source_kind": lineage.source_kind,
        "score_min": lineage.score_min,
        "wall_seconds": time.perf_counter() - wall_started,
        "arms": arm_summary,
        "pairwise": {
            f"{left}__vs__{right}": dict(counter)
            for (left, right), counter in pairwise.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [summary["arms"][arm] for summary in summaries]
        arms[arm] = {
            "frames": sum(int(row["frames"]) for row in rows),
            "input_detections": sum(int(row["input_detections"]) for row in rows),
            "retained_detections": sum(int(row["retained_detections"]) for row in rows),
            "suppressed_detections": sum(
                int(row["suppressed_detections"]) for row in rows
            ),
            "geometry_changed": sum(int(row["geometry_changed"]) for row in rows),
            "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "diagnostics": dict(
                sum((Counter(row["diagnostics"]) for row in rows), Counter())
            ),
            # Global distributions are recalculated in the companion notebook
            # from per-run summaries/decision traces when needed.
            "run_weighted_union_recall_min": min(
                float(row["union_recall"]["min"]) for row in rows
            ),
            "run_weighted_union_iou_mean": sum(
                float(row["union_iou"]["mean"]) * int(row["union_iou"]["count"])
                for row in rows
            )
            / sum(int(row["union_iou"]["count"]) for row in rows),
        }
        seconds = float(arms[arm]["elapsed_seconds"])
        arms[arm]["frames_per_second"] = (
            int(arms[arm]["frames"]) / seconds if seconds else None
        )
    pairwise: dict[str, Counter[str]] = {}
    for summary in summaries:
        for key, values in summary["pairwise"].items():
            pairwise.setdefault(key, Counter()).update(values)
    return {
        "runs": len(summaries),
        "arms": arms,
        "pairwise": {key: dict(value) for key, value in pairwise.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, default=base.DEFAULT_TOPOLOGY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--score-min", type=float, default=0.30)
    parser.add_argument(
        "--input-mode", choices=("prefer-scored", "normalize"), default="prefer-scored"
    )
    parser.add_argument("--max-union-pixels", type=int, default=32_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for run in base._read_runs(args.topology):
        lineage = base._prepare_input(
            run,
            topology=args.topology,
            output_root=output_root,
            input_mode=args.input_mode,
            score_min=args.score_min,
            force=args.force,
        )
        print(f"[{run.run_key}] {run.frame_count} frames", flush=True)
        summaries.append(
            run_one(
                run,
                lineage,
                output_root,
                max_union_pixels=args.max_union_pixels,
            )
        )
    aggregate = _aggregate(summaries)
    payload = {"aggregate": aggregate, "runs": summaries}
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "frames",
                "input_detections",
                "retained_detections",
                "suppressed_detections",
                "geometry_changed",
                "elapsed_seconds",
                "frames_per_second",
                "union_recall_min",
                "union_iou_mean",
            ],
        )
        writer.writeheader()
        for arm, row in aggregate["arms"].items():
            writer.writerow(
                {
                    "arm": arm,
                    "frames": row["frames"],
                    "input_detections": row["input_detections"],
                    "retained_detections": row["retained_detections"],
                    "suppressed_detections": row["suppressed_detections"],
                    "geometry_changed": row["geometry_changed"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "frames_per_second": row["frames_per_second"],
                    "union_recall_min": row["run_weighted_union_recall_min"],
                    "union_iou_mean": row["run_weighted_union_iou_mean"],
                }
            )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
