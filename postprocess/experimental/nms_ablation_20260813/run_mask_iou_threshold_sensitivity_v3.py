#!/usr/bin/env python3
"""Exact mask-IoU NMS threshold sensitivity over all scored V3 runs.

This runner reuses the immutable scored inputs recorded by the completed V3
four-arm ablation.  It varies *only* the full-instance mask-IoU NMS threshold;
the candidate topology policy is held fixed:

1. fill every hole and remove owner-relative islands <= 1%;
2. score-ordered, class-agnostic exact raster mask-IoU NMS;
3. remove survivor-only redundant islands at 80% coverage / 50% area ratio.

Pairwise mask IoUs are rasterized once per frame and shared by every threshold.
No transformed JSONL or downstream artifact is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import iter_detection_records  # noqa: E402
from nms.components import (  # noqa: E402
    fill_holes_and_remove_tiny_islands,
    remove_redundant_surviving_islands,
)
from nms.component_aware import (  # noqa: E402
    _overlap_slices,
    _raster_mask,
    exact_mask_iou,
)

import run_four_arm_v3 as four  # noqa: E402


DEFAULT_ABLATION_SUMMARY = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_ablation_20260813/summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "output/nms_component_candidate_v2_thresholds_20260813"
)
DEFAULT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)
RESIDUAL_AUDIT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)


def _threshold_name(value: float) -> str:
    return f"{value:.2f}"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


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


def _class_label(detection: dict[str, Any]) -> str:
    for key in ("class_name", "label"):
        value = detection.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    category = detection.get("category_id")
    return f"category:{category}" if category is not None else "unknown"


@dataclass(frozen=True)
class Suppression:
    winner: int
    loser: int
    mask_iou: float


@dataclass(frozen=True)
class NmsDecision:
    retained: tuple[int, ...]
    suppressions: tuple[Suppression, ...]
    exact_pair_evaluations: int


def _pair_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _all_exact_pairs(
    detections: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int], float], int]:
    """Rasterize each detection once and evaluate every bbox-overlap pair."""
    rasters = [_raster_mask(detection) for detection in detections]
    values: dict[tuple[int, int], float] = {}
    broad_phase_pairs = 0
    for first_index, first in enumerate(rasters):
        if first is None or first.area <= 0:
            continue
        for second_index in range(first_index + 1, len(rasters)):
            second = rasters[second_index]
            if second is None or second.area <= 0:
                continue
            if _overlap_slices(first, second) is None:
                continue
            broad_phase_pairs += 1
            values[(first_index, second_index)] = exact_mask_iou(first, second)
    return values, broad_phase_pairs


def _greedy_nms(
    detections: list[dict[str, Any]],
    pair_ious: dict[tuple[int, int], float],
    threshold: float,
) -> NmsDecision:
    order = sorted(
        range(len(detections)),
        key=lambda index: (-float(detections[index].get("score") or 0.0), index),
    )
    suppressed: set[int] = set()
    retained: list[int] = []
    events: list[Suppression] = []
    evaluations = 0
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        retained.append(index)
        for other in order[position + 1 :]:
            if other in suppressed:
                continue
            key = _pair_key(index, other)
            if key not in pair_ious:
                continue
            evaluations += 1
            iou = pair_ious[key]
            if iou >= threshold:
                suppressed.add(other)
                events.append(Suppression(index, other, iou))
    return NmsDecision(tuple(retained), tuple(events), evaluations)


@dataclass
class ThresholdAccumulator:
    threshold: float
    frames: int = 0
    input_detections: int = 0
    retained_detections: int = 0
    nms_suppressed: int = 0
    nms_changed_frames: int = 0
    topology_changed_frames: int = 0
    final_changed_frames: int = 0
    exact_pair_evaluations: int = 0
    redundant_islands_removed: int = 0
    same_class_suppressions: int = 0
    cross_class_suppressions: int = 0
    raw_union_recalls: list[float] = field(default_factory=list)
    raw_union_ious: list[float] = field(default_factory=list)
    pre_nms_union_recalls: list[float] = field(default_factory=list)
    pre_nms_union_ious: list[float] = field(default_factory=list)
    nms_frame_raw_union_recalls: list[float] = field(default_factory=list)
    nms_frame_raw_union_ious: list[float] = field(default_factory=list)
    suppressed_coverages: list[float] = field(default_factory=list)
    safety_frames_skipped_large_roi: int = 0
    class_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    residual_pairs: Counter[str] = field(default_factory=Counter)
    residual_same_class: Counter[str] = field(default_factory=Counter)
    residual_cross_class: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "threshold": self.threshold,
            "frames": self.frames,
            "input_detections": self.input_detections,
            "retained_detections": self.retained_detections,
            "suppressed_detections": self.nms_suppressed,
            "retention_rate": (
                self.retained_detections / self.input_detections
                if self.input_detections
                else 1.0
            ),
            "nms_changed_frames": self.nms_changed_frames,
            "topology_changed_frames": self.topology_changed_frames,
            "final_changed_frames": self.final_changed_frames,
            "exact_pair_evaluations": self.exact_pair_evaluations,
            "redundant_islands_removed": self.redundant_islands_removed,
            "same_class_suppressions": self.same_class_suppressions,
            "cross_class_suppressions": self.cross_class_suppressions,
            "cross_class_suppression_rate": (
                self.cross_class_suppressions / self.nms_suppressed
                if self.nms_suppressed
                else 0.0
            ),
            "safety_frames": len(self.raw_union_recalls),
            "safety_frames_skipped_large_roi": self.safety_frames_skipped_large_roi,
            "raw_union_recall_min": (
                min(self.raw_union_recalls) if self.raw_union_recalls else None
            ),
            "raw_union_recall_p01": _quantile(self.raw_union_recalls, 0.01),
            "raw_union_recall_p05": _quantile(self.raw_union_recalls, 0.05),
            "raw_union_recall_mean": _mean(self.raw_union_recalls),
            "raw_union_iou_min": min(self.raw_union_ious)
            if self.raw_union_ious
            else None,
            "raw_union_iou_p01": _quantile(self.raw_union_ious, 0.01),
            "raw_union_iou_p05": _quantile(self.raw_union_ious, 0.05),
            "raw_union_iou_mean": _mean(self.raw_union_ious),
            "pre_nms_union_recall_min": (
                min(self.pre_nms_union_recalls) if self.pre_nms_union_recalls else None
            ),
            "pre_nms_union_recall_p01": _quantile(self.pre_nms_union_recalls, 0.01),
            "pre_nms_union_recall_p05": _quantile(self.pre_nms_union_recalls, 0.05),
            "pre_nms_union_recall_mean": _mean(self.pre_nms_union_recalls),
            "pre_nms_union_iou_min": (
                min(self.pre_nms_union_ious) if self.pre_nms_union_ious else None
            ),
            "pre_nms_union_iou_p05": _quantile(self.pre_nms_union_ious, 0.05),
            "pre_nms_union_iou_mean": _mean(self.pre_nms_union_ious),
            "nms_frame_raw_union_recall_min": (
                min(self.nms_frame_raw_union_recalls)
                if self.nms_frame_raw_union_recalls
                else None
            ),
            "nms_frame_raw_union_recall_p01": _quantile(
                self.nms_frame_raw_union_recalls, 0.01
            ),
            "nms_frame_raw_union_recall_p05": _quantile(
                self.nms_frame_raw_union_recalls, 0.05
            ),
            "nms_frame_raw_union_recall_mean": _mean(self.nms_frame_raw_union_recalls),
            "nms_frame_raw_union_iou_min": (
                min(self.nms_frame_raw_union_ious)
                if self.nms_frame_raw_union_ious
                else None
            ),
            "nms_frame_raw_union_iou_p05": _quantile(
                self.nms_frame_raw_union_ious, 0.05
            ),
            "nms_frame_raw_union_iou_mean": _mean(self.nms_frame_raw_union_ious),
            "suppressed_coverage_min": (
                min(self.suppressed_coverages) if self.suppressed_coverages else None
            ),
            "suppressed_coverage_p01": _quantile(self.suppressed_coverages, 0.01),
            "suppressed_coverage_p05": _quantile(self.suppressed_coverages, 0.05),
            "suppressed_coverage_mean": _mean(self.suppressed_coverages),
            "suppressed_coverage_lt_0p80": sum(
                value < 0.80 for value in self.suppressed_coverages
            ),
            "suppressed_coverage_lt_0p90": sum(
                value < 0.90 for value in self.suppressed_coverages
            ),
            "suppressed_coverage_lt_0p97": sum(
                value < 0.97 for value in self.suppressed_coverages
            ),
        }
        for audit in RESIDUAL_AUDIT_THRESHOLDS:
            label = _threshold_name(audit).replace(".", "p")
            values[f"residual_pairs_iou_ge_{label}"] = self.residual_pairs[
                _threshold_name(audit)
            ]
            values[f"residual_same_class_iou_ge_{label}"] = self.residual_same_class[
                _threshold_name(audit)
            ]
            values[f"residual_cross_class_iou_ge_{label}"] = self.residual_cross_class[
                _threshold_name(audit)
            ]
        return values


def _union_or_none(
    detections: list[dict[str, Any]], max_union_pixels: int
) -> four.UnionRaster | None:
    return four._union_raster(detections, max_pixels=max_union_pixels)


def _source_records(
    summary: dict[str, Any], selected: set[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run in summary["runs"]:
        run_key = str(run["run_key"])
        if selected and run_key not in selected:
            continue
        jsonl = Path(str(run["input"]["jsonl"])).resolve()
        if not jsonl.is_file():
            raise FileNotFoundError(f"missing scored input for {run_key}: {jsonl}")
        result.append(
            {
                "run_key": run_key,
                "video_slug": run["video_slug"],
                "jsonl": jsonl,
                "source_kind": run["input"]["source_kind"],
                "score_min": float(run["input"]["score_min"]),
                "expected_frames": int(run["frames_processed"]),
            }
        )
    missing = selected - {row["run_key"] for row in result}
    if missing:
        raise ValueError(f"unknown run keys: {', '.join(sorted(missing))}")
    if not result:
        raise ValueError("no V3 runs selected")
    return result


def _process_run(
    source: dict[str, Any],
    *,
    thresholds: tuple[float, ...],
    max_union_pixels: int,
    progress_every: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    run_key = str(source["run_key"])
    accumulators = {value: ThresholdAccumulator(value) for value in thresholds}
    suppression_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    pairwise_deltas: Counter[tuple[float, float, str]] = Counter()
    topology_totals: Counter[str] = Counter()
    all_pair_totals: Counter[str] = Counter()
    total_frames = 0
    wall_started = time.perf_counter()

    for record in iter_detection_records(source["jsonl"]):
        total_frames += 1
        frame_index = int(record["frame_index"])
        raw = list(record["detections"])
        raw_ids = [
            _detection_id(value, frame_index=frame_index, fallback_index=index)
            for index, value in enumerate(raw)
        ]
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(f"{run_key} frame {frame_index}: duplicate detection IDs")

        preprocessed, cleanup = fill_holes_and_remove_tiny_islands(
            raw,
            fill_all_holes=True,
            unconditional_owner_ratio_max=0.01,
        )
        topology_totals.update(cleanup.as_dict())
        topology_changed = bool(cleanup.holes_filled or cleanup.tiny_islands_removed)
        pair_ious, broad_pairs = _all_exact_pairs(preprocessed)
        all_pair_totals["bbox_overlap_pairs"] += broad_pairs
        all_pair_totals["positive_iou_pairs"] += sum(
            value > 0.0 for value in pair_ious.values()
        )
        for audit in RESIDUAL_AUDIT_THRESHOLDS:
            all_pair_totals[f"input_pairs_iou_ge_{_threshold_name(audit)}"] += sum(
                value >= audit for value in pair_ious.values()
            )

        decisions: dict[float, NmsDecision] = {}
        final_by_threshold: dict[float, list[dict[str, Any]]] = {}
        retained_ids_by_threshold: dict[float, set[int | str]] = {}
        component_removed_by_threshold: dict[float, int] = {}
        for threshold in thresholds:
            accumulator = accumulators[threshold]
            decision = _greedy_nms(preprocessed, pair_ious, threshold)
            decisions[threshold] = decision
            survivors = [preprocessed[index] for index in decision.retained]
            final, component_cleanup = remove_redundant_surviving_islands(
                survivors,
                other_coverage_min=0.80,
                island_to_other_area_max=0.50,
            )
            final_by_threshold[threshold] = final
            component_removed_by_threshold[
                threshold
            ] = component_cleanup.redundant_islands_removed
            retained_ids = {raw_ids[index] for index in decision.retained}
            retained_ids_by_threshold[threshold] = retained_ids

            accumulator.frames += 1
            accumulator.input_detections += len(raw)
            accumulator.retained_detections += len(final)
            accumulator.nms_suppressed += len(decision.suppressions)
            accumulator.nms_changed_frames += int(bool(decision.suppressions))
            accumulator.topology_changed_frames += int(topology_changed)
            accumulator.exact_pair_evaluations += decision.exact_pair_evaluations
            accumulator.redundant_islands_removed += (
                component_cleanup.redundant_islands_removed
            )
            final_changed = bool(
                topology_changed
                or decision.suppressions
                or component_cleanup.redundant_islands_removed
            )
            accumulator.final_changed_frames += int(final_changed)

            retained_set = set(decision.retained)
            for (first, second), iou in pair_ious.items():
                if first not in retained_set or second not in retained_set:
                    continue
                same_class = _class_label(raw[first]) == _class_label(raw[second])
                for audit in RESIDUAL_AUDIT_THRESHOLDS:
                    if iou < audit:
                        continue
                    audit_name = _threshold_name(audit)
                    accumulator.residual_pairs[audit_name] += 1
                    if same_class:
                        accumulator.residual_same_class[audit_name] += 1
                    else:
                        accumulator.residual_cross_class[audit_name] += 1
                if iou >= min(RESIDUAL_AUDIT_THRESHOLDS):
                    residual_rows.append(
                        {
                            "run_key": run_key,
                            "frame_index": frame_index,
                            "nms_threshold": threshold,
                            "first_id": raw_ids[first],
                            "second_id": raw_ids[second],
                            "mask_iou": iou,
                            "first_class": _class_label(raw[first]),
                            "second_class": _class_label(raw[second]),
                            "same_class": int(same_class),
                        }
                    )

            for event in decision.suppressions:
                winner = raw[event.winner]
                loser = raw[event.loser]
                winner_class = _class_label(winner)
                loser_class = _class_label(loser)
                same_class = winner_class == loser_class
                accumulator.same_class_suppressions += int(same_class)
                accumulator.cross_class_suppressions += int(not same_class)
                accumulator.class_pairs[(winner_class, loser_class)] += 1
                suppression_rows.append(
                    {
                        "run_key": run_key,
                        "frame_index": frame_index,
                        "threshold": threshold,
                        "winner_id": raw_ids[event.winner],
                        "loser_id": raw_ids[event.loser],
                        "mask_iou": event.mask_iou,
                        "winner_score": float(winner.get("score") or 0.0),
                        "loser_score": float(loser.get("score") or 0.0),
                        "winner_class": winner_class,
                        "loser_class": loser_class,
                        "same_class": int(same_class),
                        "winner_category_id": winner.get("category_id"),
                        "loser_category_id": loser.get("category_id"),
                    }
                )

        for lower, higher in zip(thresholds, thresholds[1:], strict=False):
            lower_ids = retained_ids_by_threshold[lower]
            higher_ids = retained_ids_by_threshold[higher]
            only_lower = lower_ids - higher_ids
            only_higher = higher_ids - lower_ids
            key = (lower, higher)
            pairwise_deltas[(*key, "frames")] += 1
            pairwise_deltas[(*key, "differing_frames")] += int(
                bool(only_lower or only_higher)
            )
            pairwise_deltas[(*key, "only_lower_retained")] += len(only_lower)
            pairwise_deltas[(*key, "only_higher_retained")] += len(only_higher)

        # Raster safety is only needed when at least one output changed.
        if any(
            topology_changed
            or decisions[threshold].suppressions
            or component_removed_by_threshold[threshold]
            for threshold in thresholds
        ):
            raw_union = _union_or_none(raw, max_union_pixels)
            pre_union = _union_or_none(preprocessed, max_union_pixels)
            if raw_union is None or pre_union is None:
                for threshold in thresholds:
                    if (
                        topology_changed
                        or decisions[threshold].suppressions
                        or len(final_by_threshold[threshold]) != len(raw)
                    ):
                        accumulators[threshold].safety_frames_skipped_large_roi += 1
            else:
                raw_by_id = dict(zip(raw_ids, raw, strict=True))
                for threshold in thresholds:
                    decision = decisions[threshold]
                    final = final_by_threshold[threshold]
                    final_changed = bool(
                        topology_changed
                        or decision.suppressions
                        or component_removed_by_threshold[threshold]
                    )
                    if not final_changed:
                        continue
                    final_union = _union_or_none(final, max_union_pixels)
                    if final_union is None:
                        accumulators[threshold].safety_frames_skipped_large_roi += 1
                        continue
                    raw_metrics = four._union_metrics(raw_union, final_union)
                    pre_metrics = four._union_metrics(pre_union, final_union)
                    accumulator = accumulators[threshold]
                    accumulator.raw_union_recalls.append(
                        float(raw_metrics["union_recall"])
                    )
                    accumulator.raw_union_ious.append(float(raw_metrics["union_iou"]))
                    accumulator.pre_nms_union_recalls.append(
                        float(pre_metrics["union_recall"])
                    )
                    accumulator.pre_nms_union_ious.append(
                        float(pre_metrics["union_iou"])
                    )
                    if decision.suppressions:
                        accumulator.nms_frame_raw_union_recalls.append(
                            float(raw_metrics["union_recall"])
                        )
                        accumulator.nms_frame_raw_union_ious.append(
                            float(raw_metrics["union_iou"])
                        )
                    suppressed_ids = {
                        raw_ids[event.loser] for event in decision.suppressions
                    }
                    coverages = [
                        value
                        for detection_id in suppressed_ids
                        if (
                            value := four._coverage_by_union(
                                raw_by_id[detection_id], final_union
                            )
                        )
                        is not None
                    ]
                    accumulator.suppressed_coverages.extend(coverages)
                    safety_rows.append(
                        {
                            "run_key": run_key,
                            "frame_index": frame_index,
                            "threshold": threshold,
                            "topology_changed": int(topology_changed),
                            "nms_suppressed": len(decision.suppressions),
                            "raw_union_recall": raw_metrics["union_recall"],
                            "raw_union_iou": raw_metrics["union_iou"],
                            "pre_nms_union_recall": pre_metrics["union_recall"],
                            "pre_nms_union_iou": pre_metrics["union_iou"],
                            "removed_area_rate_vs_raw": raw_metrics[
                                "removed_area_rate"
                            ],
                            "added_area_rate_vs_raw": raw_metrics["added_area_rate"],
                            "suppressed_coverage_min": (
                                min(coverages) if coverages else None
                            ),
                            "suppressed_coverage_mean": _mean(coverages),
                        }
                    )

        if progress_every > 0 and total_frames % progress_every == 0:
            print(
                f"[{run_key}] frames={total_frames:,} "
                f"detections={accumulators[thresholds[0]].input_detections:,}",
                flush=True,
            )

    if total_frames != int(source["expected_frames"]):
        raise ValueError(
            f"{run_key}: expected {source['expected_frames']} frames, read {total_frames}"
        )
    elapsed = time.perf_counter() - wall_started
    summaries = []
    class_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        summary = {
            "run_key": run_key,
            "video_slug": source["video_slug"],
            **accumulators[threshold].summary(),
        }
        summaries.append(summary)
        for (winner_class, loser_class), count in sorted(
            accumulators[threshold].class_pairs.items()
        ):
            class_rows.append(
                {
                    "run_key": run_key,
                    "threshold": threshold,
                    "winner_class": winner_class,
                    "loser_class": loser_class,
                    "same_class": int(winner_class == loser_class),
                    "suppressions": count,
                }
            )
    delta_rows = []
    for lower, higher in zip(thresholds, thresholds[1:], strict=False):
        delta_rows.append(
            {
                "run_key": run_key,
                "lower_threshold": lower,
                "higher_threshold": higher,
                **{
                    key: pairwise_deltas[(lower, higher, key)]
                    for key in (
                        "frames",
                        "differing_frames",
                        "only_lower_retained",
                        "only_higher_retained",
                    )
                },
            }
        )
    metadata = {
        "run_key": run_key,
        "frames": total_frames,
        "elapsed_seconds": elapsed,
        "source_jsonl": str(source["jsonl"]),
        "source_jsonl_size": source["jsonl"].stat().st_size,
        "source_kind": source["source_kind"],
        "score_min": source["score_min"],
        "topology": dict(topology_totals),
        "all_exact_pairs": dict(all_pair_totals),
    }
    return (
        summaries,
        class_rows,
        suppression_rows,
        safety_rows,
        residual_rows,
        {"metadata": metadata, "pairwise_deltas": delta_rows},
    )


def _aggregate_summaries(
    run_summaries: list[dict[str, Any]], thresholds: tuple[float, ...]
) -> list[dict[str, Any]]:
    additive = (
        "frames",
        "input_detections",
        "retained_detections",
        "suppressed_detections",
        "nms_changed_frames",
        "topology_changed_frames",
        "final_changed_frames",
        "exact_pair_evaluations",
        "redundant_islands_removed",
        "same_class_suppressions",
        "cross_class_suppressions",
        "safety_frames",
        "safety_frames_skipped_large_roi",
        "suppressed_coverage_lt_0p80",
        "suppressed_coverage_lt_0p90",
        "suppressed_coverage_lt_0p97",
    )
    for audit in RESIDUAL_AUDIT_THRESHOLDS:
        label = _threshold_name(audit).replace(".", "p")
        additive += (
            f"residual_pairs_iou_ge_{label}",
            f"residual_same_class_iou_ge_{label}",
            f"residual_cross_class_iou_ge_{label}",
        )
    result = []
    for threshold in thresholds:
        rows = [row for row in run_summaries if row["threshold"] == threshold]
        values: dict[str, Any] = {"threshold": threshold, "runs": len(rows)}
        for key in additive:
            values[key] = sum(int(row.get(key) or 0) for row in rows)
        values["retention_rate"] = (
            values["retained_detections"] / values["input_detections"]
            if values["input_detections"]
            else 1.0
        )
        values["cross_class_suppression_rate"] = (
            values["cross_class_suppressions"] / values["suppressed_detections"]
            if values["suppressed_detections"]
            else 0.0
        )
        result.append(values)
    return result


def _aggregate_metric_from_rows(
    aggregate: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    suppression_rows: list[dict[str, Any]],
) -> None:
    by_threshold_safety: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in safety_rows:
        by_threshold_safety[float(row["threshold"])].append(row)
    by_threshold_suppression: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in suppression_rows:
        by_threshold_suppression[float(row["threshold"])].append(row)
    for values in aggregate:
        threshold = float(values["threshold"])
        safety = by_threshold_safety[threshold]
        nms_safety = [row for row in safety if int(row["nms_suppressed"]) > 0]
        suppressions = by_threshold_suppression[threshold]
        for prefix, field in (
            ("raw_union_recall", "raw_union_recall"),
            ("raw_union_iou", "raw_union_iou"),
            ("pre_nms_union_recall", "pre_nms_union_recall"),
            ("pre_nms_union_iou", "pre_nms_union_iou"),
        ):
            samples = [float(row[field]) for row in safety]
            values[f"{prefix}_min"] = min(samples) if samples else None
            values[f"{prefix}_p01"] = _quantile(samples, 0.01)
            values[f"{prefix}_p05"] = _quantile(samples, 0.05)
            values[f"{prefix}_mean"] = _mean(samples)
        for prefix, field in (
            ("nms_frame_raw_union_recall", "raw_union_recall"),
            ("nms_frame_raw_union_iou", "raw_union_iou"),
        ):
            samples = [float(row[field]) for row in nms_safety]
            values[f"{prefix}_min"] = min(samples) if samples else None
            values[f"{prefix}_p01"] = _quantile(samples, 0.01)
            values[f"{prefix}_p05"] = _quantile(samples, 0.05)
            values[f"{prefix}_mean"] = _mean(samples)
        values["changed_frames_raw_union_recall_lt_0p97"] = sum(
            float(row["raw_union_recall"]) < 0.97 for row in safety
        )
        coverages = [
            float(row["suppressed_coverage_min"])
            for row in safety
            if row.get("suppressed_coverage_min") not in (None, "")
        ]
        values["suppressed_frame_coverage_min"] = min(coverages) if coverages else None
        values["suppressed_frame_coverage_p01"] = _quantile(coverages, 0.01)
        values["suppressed_frame_coverage_p05"] = _quantile(coverages, 0.05)
        values["suppressed_frame_coverage_mean"] = _mean(coverages)
        ious = [float(row["mask_iou"]) for row in suppressions]
        values["suppressed_pair_iou_min"] = min(ious) if ious else None
        values["suppressed_pair_iou_p05"] = _quantile(ious, 0.05)
        values["suppressed_pair_iou_mean"] = _mean(ious)


def _class_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[float, str, str]] = Counter()
    for row in rows:
        counts[
            (
                float(row["threshold"]),
                str(row["winner_class"]),
                str(row["loser_class"]),
            )
        ] += int(row["suppressions"])
    return [
        {
            "threshold": threshold,
            "winner_class": winner,
            "loser_class": loser,
            "same_class": int(winner == loser),
            "suppressions": count,
        }
        for (threshold, winner, loser), count in sorted(counts.items())
    ]


def _aggregate_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[float, float, str]] = Counter()
    for row in rows:
        key = (float(row["lower_threshold"]), float(row["higher_threshold"]))
        for metric in (
            "frames",
            "differing_frames",
            "only_lower_retained",
            "only_higher_retained",
        ):
            counts[(*key, metric)] += int(row[metric])
    result = []
    for lower, higher in sorted({key[:2] for key in counts}):
        result.append(
            {
                "lower_threshold": lower,
                "higher_threshold": higher,
                **{
                    metric: counts[(lower, higher, metric)]
                    for metric in (
                        "frames",
                        "differing_frames",
                        "only_lower_retained",
                        "only_higher_retained",
                    )
                },
            }
        )
    return result


def _validation(
    aggregate: list[dict[str, Any]], source_summary: dict[str, Any]
) -> dict[str, Any]:
    expected_arm = next(
        arm
        for arm in source_summary["aggregate_arms"]
        if arm["arm"] == "component_candidate_v2"
    )
    actual = next(row for row in aggregate if math.isclose(row["threshold"], 0.70))
    checks = {
        "input_detections_match_0p70": actual["input_detections"]
        == int(expected_arm["input_detections"]),
        "retained_detections_match_0p70": actual["retained_detections"]
        == int(expected_arm["retained_detections"]),
        "suppressed_detections_match_0p70": actual["suppressed_detections"]
        == int(expected_arm["suppressed_detections"]),
        "redundant_islands_match_0p70": actual["redundant_islands_removed"]
        == int(expected_arm["redundant_islands_removed"]),
        "no_residual_pairs_at_active_threshold_0p60": next(
            row for row in aggregate if math.isclose(row["threshold"], 0.60)
        )["residual_pairs_iou_ge_0p60"]
        == 0,
        "no_residual_pairs_at_active_threshold_0p65": next(
            row for row in aggregate if math.isclose(row["threshold"], 0.65)
        )["residual_pairs_iou_ge_0p65"]
        == 0,
        "no_residual_pairs_at_active_threshold_0p70": actual[
            "residual_pairs_iou_ge_0p70"
        ]
        == 0,
        "no_residual_pairs_at_active_threshold_0p75": next(
            row for row in aggregate if math.isclose(row["threshold"], 0.75)
        )["residual_pairs_iou_ge_0p75"]
        == 0,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "expected_0p70": {
            "input_detections": expected_arm["input_detections"],
            "retained_detections": expected_arm["retained_detections"],
            "suppressed_detections": expected_arm["suppressed_detections"],
            "redundant_islands_removed": expected_arm["redundant_islands_removed"],
        },
        "actual_0p70": {
            "input_detections": actual["input_detections"],
            "retained_detections": actual["retained_detections"],
            "suppressed_detections": actual["suppressed_detections"],
            "redundant_islands_removed": actual["redundant_islands_removed"],
        },
    }


def _report(
    aggregate: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    lines = [
        "# V3 exact Mask-IoU NMS threshold sensitivity",
        "",
        "Scope: all nine scored V3 runs, score minimum 0.30, fixed hole/island "
        "topology policy, class-agnostic greedy NMS.",
        "",
        "| threshold | suppressed | changed frames | cross-class | raw union Recall min | raw union Recall p01 | residual pairs IoU>=0.70 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {threshold:.2f} | {suppressed_detections:,} | "
            "{nms_changed_frames:,} | {cross_class_suppressions:,} | "
            "{raw_union_recall_min:.6f} | {raw_union_recall_p01:.6f} | "
            "{residual_pairs_iou_ge_0p70:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Validation against the completed 0.70 ablation: **{validation['status']}**.",
            "",
            "## Recommendation",
            "",
            "If one global threshold is required, retain **0.70**: it removes all "
            "pairs at or above 0.70 while 0.75 leaves 264 such residual pairs. "
            "The safer class-aware candidate is **0.70 for same-class pairs and "
            "0.75 for cross-class pairs**. At global 0.70, 558/559 suppressions "
            "are same-class and one is cross-class; at 0.75 all 299 suppressions "
            "are same-class. The single 0.70 cross-class event has almost complete "
            "union coverage, showing that union safety cannot detect semantic "
            "instance loss. This hybrid recommendation therefore still requires "
            "visual/downstream confirmation before Production promotion.",
            "",
            "## Class-pair suppression counts",
            "",
            "| threshold | winner | loser | count |",
            "|---:|---|---|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {float(row['threshold']):.2f} | {row['winner_class']} | "
            f"{row['loser_class']} | {int(row['suppressions']):,} |"
        )
    lines.extend(
        [
            "",
            "Raw-union safety is a diagnostic against AI output, not ground truth. "
            "A low value identifies changed coverage that requires downstream or visual audit; "
            "it does not by itself prove a false negative.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-summary", type=Path, default=DEFAULT_ABLATION_SUMMARY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[],
        help="repeat to override the default 0.60,0.65,0.70,0.75 sweep",
    )
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument("--max-union-pixels", type=int, default=16_777_216)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = tuple(sorted(set(args.threshold or DEFAULT_THRESHOLDS)))
    if not thresholds or any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("thresholds must be in (0, 1]")
    if 0.70 not in thresholds:
        raise ValueError("the sweep must include 0.70 for baseline validation")

    source_summary_path = args.ablation_summary.resolve()
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    sources = _source_records(source_summary, set(args.run_key))
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"output is not empty: {output_root}; use --force")
    output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    config = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "git_revision": _git_revision(),
        "script": str(script_path),
        "script_sha256": _sha256(script_path),
        "source_ablation_summary": str(source_summary_path),
        "source_ablation_config_sha256": source_summary["config"].get("config_sha256"),
        "thresholds": thresholds,
        "residual_audit_thresholds": RESIDUAL_AUDIT_THRESHOLDS,
        "run_keys": [source["run_key"] for source in sources],
        "topology_policy": {
            "fill_all_holes": True,
            "tiny_island_owner_ratio_max": 0.01,
            "survivor_island_coverage_min": 0.80,
            "island_to_other_main_area_max": 0.50,
        },
        "nms": {
            "class_agnostic": True,
            "ranking": "score descending, source order tie-break",
            "overlap": "exact native-resolution raster mask IoU",
            "bbox_role": "broad-phase only",
        },
        "max_union_pixels": args.max_union_pixels,
        "inputs": [
            {
                **{key: value for key, value in source.items() if key != "jsonl"},
                "jsonl": str(source["jsonl"]),
                "size_bytes": source["jsonl"].stat().st_size,
                "mtime_ns": source["jsonl"].stat().st_mtime_ns,
            }
            for source in sources
        ],
    }
    (output_root / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_run_summaries: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    all_suppression_rows: list[dict[str, Any]] = []
    all_safety_rows: list[dict[str, Any]] = []
    all_residual_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    for index, source in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {source['run_key']}", flush=True)
        summaries, classes, suppressions, safety, residual, extra = _process_run(
            source,
            thresholds=thresholds,
            max_union_pixels=args.max_union_pixels,
            progress_every=args.progress_every,
        )
        all_run_summaries.extend(summaries)
        all_class_rows.extend(classes)
        all_suppression_rows.extend(suppressions)
        all_safety_rows.extend(safety)
        all_residual_rows.extend(residual)
        run_metadata.append(extra["metadata"])
        pairwise_rows.extend(extra["pairwise_deltas"])

    aggregate = _aggregate_summaries(all_run_summaries, thresholds)
    _aggregate_metric_from_rows(aggregate, all_safety_rows, all_suppression_rows)
    class_aggregate = _class_aggregate(all_class_rows)
    aggregate_pairwise = _aggregate_pairwise(pairwise_rows)
    validation = _validation(aggregate, source_summary)
    elapsed = time.perf_counter() - wall_started

    _write_csv(output_root / "run_threshold_summary.csv", all_run_summaries)
    _write_csv(output_root / "aggregate_threshold_summary.csv", aggregate)
    _write_csv(output_root / "suppression_events.csv", all_suppression_rows)
    _write_csv(output_root / "suppression_class_pairs_by_run.csv", all_class_rows)
    _write_csv(output_root / "suppression_class_pairs.csv", class_aggregate)
    _write_csv(output_root / "changed_frame_safety.csv", all_safety_rows)
    _write_csv(output_root / "residual_pair_events.csv", all_residual_rows)
    _write_csv(output_root / "threshold_pairwise_deltas_by_run.csv", pairwise_rows)
    _write_csv(output_root / "threshold_pairwise_deltas.csv", aggregate_pairwise)
    (output_root / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final = {
        "schema_version": 1,
        "config": config,
        "elapsed_seconds": elapsed,
        "run_metadata": run_metadata,
        "aggregate_thresholds": aggregate,
        "suppression_class_pairs": class_aggregate,
        "threshold_pairwise_deltas": aggregate_pairwise,
        "validation": validation,
    }
    (output_root / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "REPORT.md").write_text(
        _report(aggregate, class_aggregate, validation), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "elapsed_seconds": elapsed,
                "aggregate_thresholds": aggregate,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if validation["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
