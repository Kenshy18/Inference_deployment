#!/usr/bin/env python3
"""Reproducible four-arm V3 topology/NMS ablation.

The runner deliberately lives under ``experimental``.  It does not change the
default Production pipeline and compares only the topology/NMS front-end:

``legacy``
    Current adaptive bbox-IoU/containment NMS.
``topology_then_legacy``
    Fill every hole and remove owner-relative islands <=1%, then legacy NMS.
``mask_iou_only``
    Score-ordered full-instance mask-IoU NMS at 0.70, without topology cleanup.
``component_candidate_v2``
    Holes + <=1% cleanup, mask-IoU NMS, then survivor-only 80%/50% island
    redundancy cleanup.

Existing scored JSONL is preferred when its pipeline manifest identifies the
same raw SQLite.  Missing inputs are reproducibly normalized from the raw V3
SQLite and filtered with the recorded/default score threshold (0.30).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import (  # noqa: E402
    dumps_json_line,
    iter_detection_records,
)
from nms import (  # noqa: E402
    AdaptiveNms,
    ComponentAwareMaskNms,
    fill_holes_and_remove_tiny_islands,
)
from nms.component_aware import (  # noqa: E402
    _overlap_slices,
    _raster_mask,
    exact_mask_iou,
)
from preprocessing.raw_sqlite import normalize_raw_detection_sqlite  # noqa: E402
from preprocessing.score_policy import (  # noqa: E402
    ScorePolicy,
    apply_score_policy_jsonl,
)


DEFAULT_TOPOLOGY = (
    REPOSITORY_ROOT / "output/instance_mask_topology_20260806/topology.sqlite"
)
ARM_NAMES = (
    "legacy",
    "topology_then_legacy",
    "mask_iou_only",
    "component_candidate_v2",
)


@dataclass(frozen=True)
class RunInput:
    run_key: str
    video_slug: str
    raw_sqlite: Path
    input_video: Path
    frame_count: int
    detection_count: int


@dataclass(frozen=True)
class InputLineage:
    run_key: str
    jsonl: Path
    source_kind: str
    raw_sqlite: Path
    source_manifest: Path | None
    score_min: float
    normalization_stats: dict[str, Any] | None = None
    score_stats: dict[str, Any] | None = None


@dataclass
class ArmAccumulator:
    frames: int = 0
    input_detections: int = 0
    retained_detections: int = 0
    elapsed_ns: int = 0
    diagnostics: Counter[str] = field(default_factory=Counter)
    union_recalls: list[float] = field(default_factory=list)
    union_ious: list[float] = field(default_factory=list)
    removed_area_rates: list[float] = field(default_factory=list)
    added_area_rates: list[float] = field(default_factory=list)
    suppressed_coverages: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class UnionRaster:
    mask: np.ndarray
    left: int
    top: int
    area: int

    @property
    def right(self) -> int:
        return self.left + int(self.mask.shape[1]) - 1

    @property
    def bottom(self) -> int:
        return self.top + int(self.mask.shape[0]) - 1


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_runs(topology: Path) -> list[RunInput]:
    uri = f"file:{topology.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(
            connection.execute(
                """SELECT run_key,video_slug,inference_sqlite,input_video,
                          frame_count,detection_count
                   FROM audit_runs WHERE model_key='v3' ORDER BY run_key"""
            )
        )
    return [
        RunInput(
            run_key=str(row["run_key"]),
            video_slug=str(row["video_slug"]),
            raw_sqlite=Path(str(row["inference_sqlite"])),
            input_video=Path(str(row["input_video"])),
            frame_count=int(row["frame_count"]),
            detection_count=int(row["detection_count"]),
        )
        for row in rows
    ]


def _manifest_score_min(manifest: dict[str, Any], fallback: float) -> float:
    for stage in manifest.get("stages") or []:
        if stage.get("id") != "score_policy":
            continue
        try:
            return float((stage.get("metadata") or {})["default_min"])
        except (KeyError, TypeError, ValueError):
            break
    return fallback


def _discover_scored_inputs(topology: Path, score_min: float) -> dict[Path, tuple[Path, Path, float]]:
    """Map canonical raw SQLite paths to (scored JSONL, manifest, threshold)."""
    result: dict[Path, tuple[Path, Path, float]] = {}
    search_root = topology.parent / "postprocess" / "v3"
    for manifest_path in sorted(search_root.glob("*/pipeline_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest["artifacts"]
            raw = Path(str(artifacts["input_raw_sqlite"])).resolve()
            scored = Path(str(artifacts["scored_jsonl"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if scored.is_file():
            result[raw] = (
                scored.resolve(),
                manifest_path.resolve(),
                _manifest_score_min(manifest, score_min),
            )
    return result


def _prepare_input(
    run: RunInput,
    *,
    topology: Path,
    output_root: Path,
    input_mode: str,
    score_min: float,
    force: bool,
) -> InputLineage:
    discovered = _discover_scored_inputs(topology, score_min)
    existing = discovered.get(run.raw_sqlite.resolve())
    if input_mode == "prefer-scored" and existing is not None:
        scored, manifest, recorded_score_min = existing
        return InputLineage(
            run_key=run.run_key,
            jsonl=scored,
            source_kind="authoritative_existing_scored_jsonl",
            raw_sqlite=run.raw_sqlite,
            source_manifest=manifest,
            score_min=recorded_score_min,
        )

    cache = output_root / "inputs" / run.run_key
    normalized = cache / "normalized.jsonl"
    scored = cache / "scored.jsonl"
    metadata = cache / "input_metadata.json"
    if scored.is_file() and metadata.is_file() and not force:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            Path(str(saved.get("raw_sqlite", ""))).resolve()
            == run.raw_sqlite.resolve()
            and float(saved.get("score_min", -1.0)) == score_min
        ):
            return InputLineage(
                run_key=run.run_key,
                jsonl=scored,
                source_kind="cached_raw_sqlite_normalization",
                raw_sqlite=run.raw_sqlite,
                source_manifest=None,
                score_min=score_min,
                normalization_stats=saved.get("normalization_stats"),
                score_stats=saved.get("score_stats"),
            )

    cache.mkdir(parents=True, exist_ok=True)
    normalization_stats = normalize_raw_detection_sqlite(run.raw_sqlite, normalized)
    score_stats = apply_score_policy_jsonl(
        normalized,
        scored,
        policy=ScorePolicy(default_min=score_min),
    )
    payload = {
        "raw_sqlite": str(run.raw_sqlite.resolve()),
        "score_min": score_min,
        "normalization_stats": normalization_stats,
        "score_stats": score_stats,
    }
    metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return InputLineage(
        run_key=run.run_key,
        jsonl=scored,
        source_kind="raw_sqlite_normalization",
        raw_sqlite=run.raw_sqlite,
        source_manifest=None,
        score_min=score_min,
        normalization_stats=normalization_stats,
        score_stats=score_stats,
    )


def _mask_iou_only(
    detections: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rasters = [_raster_mask(detection) for detection in detections]
    order = sorted(
        range(len(detections)),
        key=lambda index: (-float(detections[index].get("score") or 0.0), index),
    )
    suppressed: set[int] = set()
    retained: list[int] = []
    broad_pairs = exact_pairs = 0
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        retained.append(index)
        first = rasters[index]
        if first is None or first.area <= 0:
            continue
        for other in order[position + 1 :]:
            if other in suppressed:
                continue
            second = rasters[other]
            if second is None or second.area <= 0:
                continue
            if _overlap_slices(first, second) is None:
                continue
            broad_pairs += 1
            exact_pairs += 1
            if exact_mask_iou(first, second) >= threshold:
                suppressed.add(other)
    return [detections[index] for index in retained], {
        "bbox_overlap_pairs": broad_pairs,
        "mask_iou_pairs": exact_pairs,
        "nms_suppressed": len(suppressed),
    }


def _apply_arms(
    detections: list[dict[str, Any]],
    *,
    legacy: AdaptiveNms,
    candidate: ComponentAwareMaskNms,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]], dict[str, int]]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    timings: dict[str, int] = {}

    started = time.perf_counter_ns()
    outputs["legacy"] = legacy.apply(detections)
    timings["legacy"] = time.perf_counter_ns() - started
    diagnostics["legacy"] = {
        "nms_suppressed": len(detections) - len(outputs["legacy"])
    }

    started = time.perf_counter_ns()
    topology, cleanup = fill_holes_and_remove_tiny_islands(
        detections,
        fill_all_holes=True,
        unconditional_owner_ratio_max=0.01,
    )
    outputs["topology_then_legacy"] = legacy.apply(topology)
    timings["topology_then_legacy"] = time.perf_counter_ns() - started
    diagnostics["topology_then_legacy"] = {
        **cleanup.as_dict(),
        "nms_suppressed": len(topology) - len(outputs["topology_then_legacy"]),
    }

    started = time.perf_counter_ns()
    outputs["mask_iou_only"], diagnostics["mask_iou_only"] = _mask_iou_only(
        detections, candidate.mask_iou_threshold
    )
    timings["mask_iou_only"] = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    outputs["component_candidate_v2"], candidate_diagnostics = (
        candidate.apply_with_diagnostics(detections)
    )
    timings["component_candidate_v2"] = time.perf_counter_ns() - started
    diagnostics["component_candidate_v2"] = candidate_diagnostics.as_dict()
    return outputs, diagnostics, timings


def _detection_id(
    detection: dict[str, Any], *, frame_index: int, fallback_index: int
) -> int | str:
    value = detection.get("source_detection_id")
    if value is None:
        return f"synthetic:{frame_index}:{fallback_index}"
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _ids(detections: list[dict[str, Any]], frame_index: int) -> list[int | str]:
    values = [
        _detection_id(value, frame_index=frame_index, fallback_index=index)
        for index, value in enumerate(detections)
    ]
    if len(set(values)) != len(values):
        raise ValueError(f"frame {frame_index}: duplicate source detection IDs")
    return values


def _union_raster(
    detections: list[dict[str, Any]], *, max_pixels: int
) -> UnionRaster | None:
    rasters = [raster for detection in detections if (raster := _raster_mask(detection))]
    rasters = [raster for raster in rasters if raster.area > 0]
    if not rasters:
        return UnionRaster(np.zeros((1, 1), np.uint8), 0, 0, 0)
    left = min(raster.left for raster in rasters)
    top = min(raster.top for raster in rasters)
    right = max(raster.right for raster in rasters)
    bottom = max(raster.bottom for raster in rasters)
    width, height = right - left + 1, bottom - top + 1
    if width <= 0 or height <= 0 or width * height > max_pixels:
        return None
    union = np.zeros((height, width), np.uint8)
    for raster in rasters:
        y = raster.top - top
        x = raster.left - left
        target = union[y : y + raster.mask.shape[0], x : x + raster.mask.shape[1]]
        np.maximum(target, raster.mask, out=target)
    return UnionRaster(union, left, top, int(np.count_nonzero(union)))


def _intersection_area(first: UnionRaster, second: UnionRaster) -> int:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right < left or bottom < top:
        return 0
    first_part = first.mask[
        top - first.top : bottom - first.top + 1,
        left - first.left : right - first.left + 1,
    ]
    second_part = second.mask[
        top - second.top : bottom - second.top + 1,
        left - second.left : right - second.left + 1,
    ]
    return int(np.count_nonzero((first_part != 0) & (second_part != 0)))


def _union_metrics(reference: UnionRaster, output: UnionRaster) -> dict[str, float | int]:
    intersection = _intersection_area(reference, output)
    union_area = reference.area + output.area - intersection
    denominator = max(reference.area, 1)
    return {
        "input_union_area": reference.area,
        "output_union_area": output.area,
        "intersection_area": intersection,
        "union_recall": intersection / denominator,
        "union_iou": intersection / union_area if union_area else 1.0,
        "removed_area_rate": (reference.area - intersection) / denominator,
        "added_area_rate": (output.area - intersection) / denominator,
    }


def _coverage_by_union(detection: dict[str, Any], union: UnionRaster) -> float | None:
    raster = _raster_mask(detection)
    if raster is None or raster.area <= 0:
        return None
    wrapped = UnionRaster(raster.mask, raster.left, raster.top, raster.area)
    return _intersection_area(wrapped, union) / raster.area


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _summary_for_arm(name: str, value: ArmAccumulator) -> dict[str, Any]:
    elapsed = value.elapsed_ns / 1_000_000_000.0
    return {
        "arm": name,
        "frames": value.frames,
        "input_detections": value.input_detections,
        "retained_detections": value.retained_detections,
        "suppressed_detections": value.input_detections - value.retained_detections,
        "retention_rate": (
            value.retained_detections / value.input_detections
            if value.input_detections
            else 1.0
        ),
        "elapsed_seconds": elapsed,
        "frames_per_second": value.frames / elapsed if elapsed else None,
        "detections_per_second": value.input_detections / elapsed if elapsed else None,
        **dict(value.diagnostics),
        "safety_frames": len(value.union_recalls),
        "union_recall_min": min(value.union_recalls) if value.union_recalls else None,
        "union_recall_p01": _quantile(value.union_recalls, 0.01),
        "union_recall_p05": _quantile(value.union_recalls, 0.05),
        "union_recall_mean": (
            float(np.mean(value.union_recalls)) if value.union_recalls else None
        ),
        "union_iou_p05": _quantile(value.union_ious, 0.05),
        "union_iou_mean": float(np.mean(value.union_ious)) if value.union_ious else None,
        "removed_area_rate_p95": _quantile(value.removed_area_rates, 0.95),
        "added_area_rate_p95": _quantile(value.added_area_rates, 0.95),
        "suppressed_mask_coverage_min": (
            min(value.suppressed_coverages) if value.suppressed_coverages else None
        ),
        "suppressed_mask_coverage_p05": _quantile(value.suppressed_coverages, 0.05),
        "suppressed_mask_coverage_mean": (
            float(np.mean(value.suppressed_coverages))
            if value.suppressed_coverages
            else None
        ),
    }


def _open_optional_outputs(
    run_dir: Path, *, write_arm_jsonl: bool
) -> dict[str, TextIO]:
    if not write_arm_jsonl:
        return {}
    folder = run_dir / "arm_outputs"
    folder.mkdir(parents=True, exist_ok=True)
    return {
        arm: (folder / f"{arm}.jsonl").open("w", encoding="utf-8")
        for arm in ARM_NAMES
    }


def _run_one(
    run: RunInput,
    lineage: InputLineage,
    *,
    output_root: Path,
    start_frame: int,
    max_frames: int | None,
    max_union_pixels: int,
    safety_mode: str,
    write_arm_jsonl: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / run.run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy = AdaptiveNms()
    candidate = ComponentAwareMaskNms(
        mask_iou_threshold=0.70,
        fill_all_holes=True,
        unconditional_owner_ratio_max=0.01,
        island_other_coverage_min=0.80,
        island_to_other_area_max=0.50,
    )
    accumulators = {name: ArmAccumulator() for name in ARM_NAMES}
    pairwise: dict[tuple[str, str], Counter[str]] = {
        (left, right): Counter()
        for left_index, left in enumerate(ARM_NAMES)
        for right in ARM_NAMES[left_index + 1 :]
    }
    safety_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    total_read = 0
    ids_path = run_dir / "retained_ids.jsonl.gz"
    arm_handles = _open_optional_outputs(run_dir, write_arm_jsonl=write_arm_jsonl)
    wall_started = time.perf_counter()
    with gzip.open(ids_path, "wt", encoding="utf-8") as decisions:
        try:
            for record in iter_detection_records(lineage.jsonl):
                frame_index = int(record["frame_index"])
                if frame_index < start_frame:
                    continue
                if max_frames is not None and total_read >= max_frames:
                    break
                total_read += 1
                detections = list(record["detections"])
                outputs, diagnostics, timings = _apply_arms(
                    detections, legacy=legacy, candidate=candidate
                )
                input_ids = _ids(detections, frame_index)
                arm_ids = {name: _ids(outputs[name], frame_index) for name in ARM_NAMES}
                decisions.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "input_ids": input_ids,
                            "retained_ids": arm_ids,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for name in ARM_NAMES:
                    accumulator = accumulators[name]
                    accumulator.frames += 1
                    accumulator.input_detections += len(detections)
                    accumulator.retained_detections += len(outputs[name])
                    accumulator.elapsed_ns += timings[name]
                    accumulator.diagnostics.update(diagnostics[name])
                    if name in arm_handles:
                        transformed = dict(record)
                        transformed["detections"] = outputs[name]
                        arm_handles[name].write(
                            dumps_json_line(transformed).decode("utf-8")
                        )

                for (left, right), counts in pairwise.items():
                    left_set, right_set = set(arm_ids[left]), set(arm_ids[right])
                    only_left, only_right = left_set - right_set, right_set - left_set
                    counts["frames"] += 1
                    counts["differing_frames"] += int(bool(only_left or only_right))
                    counts["only_left_ids"] += len(only_left)
                    counts["only_right_ids"] += len(only_right)

                full_diag = diagnostics["component_candidate_v2"]
                topology_changed = any(
                    int(full_diag.get(key, 0))
                    for key in (
                        "holes_filled",
                        "tiny_islands_removed",
                        "redundant_islands_removed",
                    )
                )
                ids_differ = len({tuple(arm_ids[name]) for name in ARM_NAMES}) > 1
                should_measure = safety_mode == "all" or topology_changed or ids_differ
                if should_measure:
                    reference = _union_raster(
                        detections, max_pixels=max_union_pixels
                    )
                    if reference is not None:
                        input_by_id = dict(zip(input_ids, detections, strict=True))
                        for name in ARM_NAMES:
                            output_union = _union_raster(
                                outputs[name], max_pixels=max_union_pixels
                            )
                            if output_union is None:
                                accumulators[name].diagnostics[
                                    "safety_frames_skipped_large_roi"
                                ] += 1
                                continue
                            metrics = _union_metrics(reference, output_union)
                            accumulator = accumulators[name]
                            accumulator.union_recalls.append(float(metrics["union_recall"]))
                            accumulator.union_ious.append(float(metrics["union_iou"]))
                            accumulator.removed_area_rates.append(
                                float(metrics["removed_area_rate"])
                            )
                            accumulator.added_area_rates.append(
                                float(metrics["added_area_rate"])
                            )
                            removed_ids = set(input_ids) - set(arm_ids[name])
                            coverages = [
                                value
                                for detection_id in removed_ids
                                if (
                                    value := _coverage_by_union(
                                        input_by_id[detection_id], output_union
                                    )
                                )
                                is not None
                            ]
                            accumulator.suppressed_coverages.extend(coverages)
                            safety_rows.append(
                                {
                                    "run_key": run.run_key,
                                    "frame_index": frame_index,
                                    "arm": name,
                                    **metrics,
                                    "suppressed_count": len(removed_ids),
                                    "suppressed_coverage_min": (
                                        min(coverages) if coverages else None
                                    ),
                                }
                            )
                    else:
                        for name in ARM_NAMES:
                            accumulators[name].diagnostics[
                                "safety_frames_skipped_large_roi"
                            ] += 1

                if topology_changed:
                    component_rows.append(
                        {
                            "run_key": run.run_key,
                            "frame_index": frame_index,
                            "input_detections": len(detections),
                            **{
                                key: int(full_diag.get(key, 0))
                                for key in (
                                    "holes_filled",
                                    "tiny_islands_removed",
                                    "redundant_islands_removed",
                                    "nms_suppressed",
                                )
                            },
                            **{
                                f"retained_{name}": len(outputs[name])
                                for name in ARM_NAMES
                            },
                        }
                    )

                if total_read % 10000 == 0:
                    print(
                        f"[{run.run_key}] frames={total_read:,} "
                        f"detections={sum(a.input_detections for a in accumulators.values()) // 4:,}",
                        flush=True,
                    )
        finally:
            for handle in arm_handles.values():
                handle.close()

    wall_elapsed = time.perf_counter() - wall_started
    arm_summaries = [
        _summary_for_arm(name, accumulators[name]) for name in ARM_NAMES
    ]
    pairwise_rows = [
        {"run_key": run.run_key, "left_arm": left, "right_arm": right, **dict(counts)}
        for (left, right), counts in pairwise.items()
    ]
    payload = {
        "run_key": run.run_key,
        "video_slug": run.video_slug,
        "input": {
            **asdict(lineage),
            "jsonl": str(lineage.jsonl),
            "raw_sqlite": str(lineage.raw_sqlite),
            "source_manifest": (
                str(lineage.source_manifest) if lineage.source_manifest else None
            ),
        },
        "frames_processed": total_read,
        "wall_elapsed_seconds": wall_elapsed,
        "arms": arm_summaries,
        "pairwise": pairwise_rows,
        "safety_rows": len(safety_rows),
        "component_event_frames": len(component_rows),
        "retained_ids_jsonl_gz": str(ids_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(run_dir / "safety_frames.csv", safety_rows)
    _write_csv(run_dir / "component_events.csv", component_rows)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summaries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_key": payload["run_key"],
            "video_slug": payload["video_slug"],
            "source_kind": payload["input"]["source_kind"],
            **arm,
        }
        for payload in payloads
        for arm in payload["arms"]
    ]


def _aggregate_arm_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, Counter[str]] = {name: Counter() for name in ARM_NAMES}
    for row in rows:
        arm = str(row["arm"])
        for key in (
            "frames",
            "input_detections",
            "retained_detections",
            "suppressed_detections",
            "elapsed_seconds",
            "holes_filled",
            "tiny_islands_removed",
            "redundant_islands_removed",
            "nms_suppressed",
            "bbox_overlap_pairs",
            "mask_iou_pairs",
            "safety_frames",
            "safety_frames_skipped_large_roi",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)):
                totals[arm][key] += value
    result: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        values = totals[arm]
        elapsed = float(values["elapsed_seconds"])
        result.append(
            {
                "arm": arm,
                **dict(values),
                "retention_rate": (
                    values["retained_detections"] / values["input_detections"]
                    if values["input_detections"]
                    else 1.0
                ),
                "frames_per_second": values["frames"] / elapsed if elapsed else None,
                "detections_per_second": (
                    values["input_detections"] / elapsed if elapsed else None
                ),
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-sqlite", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument(
        "--input-mode", choices=("prefer-scored", "normalize"), default="prefer-scored"
    )
    parser.add_argument("--score-min", type=float, default=0.30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-union-pixels", type=int, default=16_777_216)
    parser.add_argument("--safety-mode", choices=("changed", "all"), default="changed")
    parser.add_argument("--write-arm-jsonl", action="store_true")
    parser.add_argument("--force-input-rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    topology = args.topology_sqlite.resolve()
    runs = _read_runs(topology)
    if args.run_key:
        requested = set(args.run_key)
        runs = [run for run in runs if run.run_key in requested]
        missing = sorted(requested - {run.run_key for run in runs})
        if missing:
            raise ValueError(f"unknown V3 run keys: {', '.join(missing)}")
    if args.list_runs:
        for run in runs:
            print(
                f"{run.run_key}\tframes={run.frame_count}\t"
                f"detections={run.detection_count}\t{run.raw_sqlite}"
            )
        return 0
    if not runs:
        raise ValueError("no V3 runs selected")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        discovered = _discover_scored_inputs(topology, args.score_min)
        print(
            json.dumps(
                [
                    {
                        "run_key": run.run_key,
                        "raw_sqlite": str(run.raw_sqlite),
                        "planned_source_kind": (
                            "authoritative_existing_scored_jsonl"
                            if args.input_mode == "prefer-scored"
                            and run.raw_sqlite.resolve() in discovered
                            else "raw_sqlite_normalization"
                        ),
                        "planned_jsonl": (
                            str(discovered[run.raw_sqlite.resolve()][0])
                            if args.input_mode == "prefer-scored"
                            and run.raw_sqlite.resolve() in discovered
                            else str(output_root / "inputs" / run.run_key / "scored.jsonl")
                        ),
                    }
                    for run in runs
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "git_revision": _git_revision(),
        "topology_sqlite": str(topology),
        "run_keys": [run.run_key for run in runs],
        "input_mode": args.input_mode,
        "score_min": args.score_min,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "max_union_pixels": args.max_union_pixels,
        "safety_mode": args.safety_mode,
        "write_arm_jsonl": args.write_arm_jsonl,
        "arms": {
            "legacy": {"policy": "AdaptiveNms defaults"},
            "topology_then_legacy": {
                "holes": "fill_all",
                "tiny_island_owner_ratio_max": 0.01,
                "policy": "AdaptiveNms defaults",
            },
            "mask_iou_only": {"mask_iou_threshold": 0.70},
            "component_candidate_v2": {
                "holes": "fill_all",
                "tiny_island_owner_ratio_max": 0.01,
                "mask_iou_threshold": 0.70,
                "survivor_island_coverage_min": 0.80,
                "island_to_other_main_area_max": 0.50,
            },
        },
    }
    config["config_sha256"] = _json_hash(config)
    (output_root / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lineages = [
        _prepare_input(
            run,
            topology=topology,
            output_root=output_root,
            input_mode=args.input_mode,
            score_min=args.score_min,
            force=args.force_input_rebuild,
        )
        for run in runs
    ]
    payloads: list[dict[str, Any]] = []
    for index, (run, lineage) in enumerate(zip(runs, lineages, strict=True), 1):
        summary_path = output_root / "runs" / run.run_key / "summary.json"
        if args.resume and summary_path.is_file():
            print(f"[{index}/{len(runs)}] resume {run.run_key}", flush=True)
            payloads.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        print(
            f"[{index}/{len(runs)}] {run.run_key} source={lineage.source_kind}",
            flush=True,
        )
        payloads.append(
            _run_one(
                run,
                lineage,
                output_root=output_root,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                max_union_pixels=args.max_union_pixels,
                safety_mode=args.safety_mode,
                write_arm_jsonl=args.write_arm_jsonl,
            )
        )

    run_rows = _flatten_summaries(payloads)
    pairwise_rows = [row for payload in payloads for row in payload["pairwise"]]
    aggregate = _aggregate_arm_summaries(run_rows)
    final = {
        "schema_version": 1,
        "config": config,
        "runs": payloads,
        "aggregate_arms": aggregate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_root / "run_arm_summary.csv", run_rows)
    _write_csv(output_root / "aggregate_arm_summary.csv", aggregate)
    _write_csv(output_root / "pairwise_deltas.csv", pairwise_rows)
    print(f"saved: {output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
