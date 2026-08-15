#!/usr/bin/env python3
"""Attribute tracking changes to NMS revival, topology, and short tracks.

The four-arm NMS experiment deliberately separates the interventions:

* ``legacy -> mask_iou_only`` changes only the suppression decision and keeps
  the original detection geometry;
* ``mask_iou_only -> component_candidate_v2`` keeps the retained source IDs
  fixed and changes only hole/island geometry;
* tracking's fixed length <= 10 filter identifies the subset that is removed
  as short-lived (it does not prove that those detections are false positives).

This script reruns the canonical deterministic tracker in a temporary folder,
joins every row by ``(frame, source_detection_id)``, writes small audit files,
and never retains the large tracked SQLite intermediates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import CutList, write_cut_list  # noqa: E402
from nms.component_aware import _raster_mask, exact_mask_iou  # noqa: E402
from tracking.association import (  # noqa: E402
    AssociationConfig,
    TrackState,
    detection_features,
    match_score,
)
from tracking.builder import build_tracked_sqlite  # noqa: E402
from tracking.records import prepare_detection  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "nms_component_candidate_v2_ablation_20260813"
    / "runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "nms_component_candidate_v2_tracking_causality_20260813"
)
ARMS = ("legacy", "mask_iou_only", "component_candidate_v2")
SHORT_LIMIT = 10
Key = tuple[int, int]


@dataclass(frozen=True)
class Row:
    frame: int
    source_detection_id: int
    raw_track_id: str
    removed: bool
    track_length: int
    final_track_id: str | None
    raw_label: str
    polygons_json: str
    bbox_xyxy_json: str
    score: float | None

    @property
    def key(self) -> Key:
        return self.frame, self.source_detection_id

    @property
    def geometry_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.polygons_json.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.bbox_xyxy_json.encode("utf-8"))
        return digest.hexdigest()

    def detection(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "source_detection_id": self.source_detection_id,
            "polygons": json.loads(self.polygons_json),
            "bbox_xyxy": json.loads(self.bbox_xyxy_json),
        }
        if self.score is not None:
            value["score"] = self.score
        return prepare_detection(value)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(materialized),
        "min": min(materialized),
        "p50": _quantile(materialized, 0.5),
        "p90": _quantile(materialized, 0.9),
        "max": max(materialized),
        "mean": sum(materialized) / len(materialized),
    }


def _read_rows(path: Path) -> dict[Key, Row]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        raw = connection.execute(
            """
            SELECT frame, source_detection_id, raw_track_id,
                   removed_by_short_track, raw_track_length, final_track_id,
                   COALESCE(raw_label, ''), COALESCE(polygons, '[]'),
                   COALESCE(bbox_xyxy_json, '[]'), score
            FROM raw_tracked_masks
            ORDER BY frame, source_detection_id
            """
        ).fetchall()
    rows: dict[Key, Row] = {}
    for value in raw:
        if value[1] is None:
            raise ValueError(f"{path}: source_detection_id is required for causal audit")
        row = Row(
            frame=int(value[0]),
            source_detection_id=int(value[1]),
            raw_track_id=str(value[2]),
            removed=bool(value[3]),
            track_length=int(value[4]),
            final_track_id=None if value[5] is None else str(value[5]),
            raw_label=str(value[6]),
            polygons_json=str(value[7]),
            bbox_xyxy_json=str(value[8]),
            score=None if value[9] is None else float(value[9]),
        )
        if row.key in rows:
            raise ValueError(f"{path}: duplicate source key {row.key}")
        rows[row.key] = row
    return rows


def _tracks(rows: dict[Key, Row]) -> dict[str, list[Row]]:
    result: dict[str, list[Row]] = defaultdict(list)
    for row in rows.values():
        result[row.raw_track_id].append(row)
    for values in result.values():
        values.sort(key=lambda row: (row.frame, row.source_detection_id))
    return result


def _marker_track_audit(rows: dict[Key, Row], marker: set[Key]) -> dict[str, object]:
    tracks = _tracks(rows)
    touched: list[list[Row]] = []
    marker_only: list[list[Row]] = []
    mixed: list[list[Row]] = []
    for track in tracks.values():
        marked = sum(row.key in marker for row in track)
        if not marked:
            continue
        touched.append(track)
        (marker_only if marked == len(track) else mixed).append(track)

    def removed(track: list[Row]) -> bool:
        return track[0].removed

    marker_rows = [row for row in rows.values() if row.key in marker]
    marked_kept = [row for row in marker_rows if not row.removed]
    marked_removed = [row for row in marker_rows if row.removed]
    strong_short_noise = [track for track in marker_only if removed(track)]
    persistent_independent = [track for track in marker_only if not removed(track)]
    mixed_kept = [track for track in mixed if not removed(track)]
    mixed_removed = [track for track in mixed if removed(track)]
    return {
        "marker_rows": len(marker_rows),
        "marker_rows_after_short_prune": len(marked_kept),
        "marker_rows_removed_by_short_prune": len(marked_removed),
        "marker_row_survival_rate": len(marked_kept) / len(marker_rows) if marker_rows else 0.0,
        "tracks_touched": len(touched),
        "marker_only_tracks": len(marker_only),
        "mixed_tracks": len(mixed),
        "strong_short_noise_tracks": len(strong_short_noise),
        "strong_short_noise_rows": sum(len(track) for track in strong_short_noise),
        "persistent_independent_tracks": len(persistent_independent),
        "persistent_independent_rows": sum(len(track) for track in persistent_independent),
        "mixed_kept_tracks": len(mixed_kept),
        "mixed_kept_marker_rows": sum(
            sum(row.key in marker for row in track) for track in mixed_kept
        ),
        "mixed_removed_tracks": len(mixed_removed),
        "mixed_removed_marker_rows": sum(
            sum(row.key in marker for row in track) for track in mixed_removed
        ),
        "touched_track_length": _distribution(len(track) for track in touched),
        "strong_short_track_length": _distribution(
            len(track) for track in strong_short_noise
        ),
    }


def _gap_audit(rows: dict[Key, Row], marker: set[Key]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for track in _tracks(rows).values():
        track_has_marker = any(row.key in marker for row in track)
        for previous, current in zip(track, track[1:]):
            difference = current.frame - previous.frame
            if difference <= 1:
                continue
            missing = difference - 1
            totals["gap_events"] += 1
            totals["missing_frames"] += missing
            if track_has_marker:
                totals["gap_events_on_marker_tracks"] += 1
                totals["missing_frames_on_marker_tracks"] += missing
            else:
                totals["gap_events_on_unmarked_tracks"] += 1
                totals["missing_frames_on_unmarked_tracks"] += missing
            if previous.key in marker or current.key in marker:
                totals["gap_events_adjacent_to_marker"] += 1
                totals["missing_frames_adjacent_to_marker"] += missing
    return dict(totals)


def _partition_audit(
    left: dict[Key, Row], right: dict[Key, Row], keys: set[Key]
) -> dict[str, object]:
    contingency: Counter[tuple[str, str]] = Counter(
        (left[key].raw_track_id, right[key].raw_track_id) for key in keys
    )
    left_to_right: dict[str, Counter[str]] = defaultdict(Counter)
    right_to_left: dict[str, Counter[str]] = defaultdict(Counter)
    for (left_track, right_track), count in contingency.items():
        left_to_right[left_track][right_track] += count
        right_to_left[right_track][left_track] += count
    split_tracks = {key: value for key, value in left_to_right.items() if len(value) > 1}
    merged_tracks = {key: value for key, value in right_to_left.items() if len(value) > 1}
    split_mismatch_rows = sum(sum(value.values()) - max(value.values()) for value in left_to_right.values())
    merge_mismatch_rows = sum(sum(value.values()) - max(value.values()) for value in right_to_left.values())
    prune_outcome_changed = sum(left[key].removed != right[key].removed for key in keys)
    left_kept_right_removed = sum(
        not left[key].removed and right[key].removed for key in keys
    )
    left_removed_right_kept = sum(
        left[key].removed and not right[key].removed for key in keys
    )
    return {
        "common_rows": len(keys),
        "left_tracks": len(left_to_right),
        "right_tracks": len(right_to_left),
        "left_tracks_split_across_right": len(split_tracks),
        "right_tracks_merging_left": len(merged_tracks),
        "split_mismatch_rows": split_mismatch_rows,
        "merge_mismatch_rows": merge_mismatch_rows,
        "prune_outcome_changed_rows": prune_outcome_changed,
        "left_kept_right_removed_rows": left_kept_right_removed,
        "left_removed_right_kept_rows": left_removed_right_kept,
        "split_examples": {
            key: dict(value.most_common())
            for key, value in list(sorted(split_tracks.items(), key=lambda item: int(item[0])))[:10]
        },
        "merge_examples": {
            key: dict(value.most_common())
            for key, value in list(sorted(merged_tracks.items(), key=lambda item: int(item[0])))[:10]
        },
    }


def _class_counts(rows: dict[Key, Row], keys: set[Key]) -> dict[str, int]:
    return dict(sorted(Counter(rows[key].raw_label for key in keys).items()))


def _mechanical_3549_evidence(
    mask_rows: dict[Key, Row], candidate_rows: dict[Key, Row]
) -> dict[str, object]:
    frame = 75789
    changed_id = 45783
    competitor_id = 45784
    before = mask_rows[(frame, changed_id)]
    after = candidate_rows[(frame, changed_id)]
    before_raster = _raster_mask(before.detection())
    after_raster = _raster_mask(after.detection())
    if before_raster is None or after_raster is None:
        raise RuntimeError("mechanical evidence mask rasterization failed")
    result: dict[str, object] = {
        "frame": frame,
        "changed_source_detection_id": changed_id,
        "competitor_source_detection_id": competitor_id,
        "geometry": {
            "before_bbox_xyxy": json.loads(before.bbox_xyxy_json),
            "after_bbox_xyxy": json.loads(after.bbox_xyxy_json),
            "before_polygons": len(json.loads(before.polygons_json)),
            "after_polygons": len(json.loads(after.polygons_json)),
            "before_raster_area": before_raster.area,
            "after_raster_area": after_raster.area,
            "removed_area": before_raster.area - after_raster.area,
            "retained_area_fraction": after_raster.area / before_raster.area,
            "before_after_mask_iou": exact_mask_iou(before_raster, after_raster),
        },
        "arms": {},
    }
    for name, rows in (
        ("mask_iou_only", mask_rows),
        ("component_candidate_v2", candidate_rows),
    ):
        current = [rows[(frame, changed_id)], rows[(frame, competitor_id)]]
        current_track_ids = {row.raw_track_id for row in current}
        track_rows = _tracks(rows)
        scores: dict[str, dict[str, float | None]] = {}
        prior: dict[str, dict[str, int]] = {}
        for track_id in sorted(current_track_ids, key=int):
            previous = max(
                (row for row in track_rows[track_id] if row.frame < frame),
                key=lambda row: row.frame,
            )
            state = TrackState(
                track_id=int(track_id),
                scene_id=0,
                last_frame=previous.frame,
                features=detection_features(previous.detection()),
            )
            prior[track_id] = {
                "frame": previous.frame,
                "source_detection_id": previous.source_detection_id,
            }
            scores[track_id] = {
                str(row.source_detection_id): match_score(
                    state,
                    detection_features(row.detection()),
                    frame,
                    AssociationConfig(),
                )
                for row in current
            }
        result["arms"][name] = {
            "prior_track_state": prior,
            "greedy_match_scores": scores,
            "assignments": {
                str(row.source_detection_id): row.raw_track_id for row in current
            },
        }
    return result


def _run_tracking(source: Path, output: Path, cuts: Path) -> dict[str, object]:
    return build_tracked_sqlite(
        source,
        output,
        cuts,
        remove_short_tracks_max_frames=SHORT_LIMIT,
        association_config=AssociationConfig(),
    )


def _csv_rows(run_results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in run_results:
        revival = result["nms_revival"]
        geometry = result["geometry_effect"]
        rows.append(
            {
                "run_key": result["run_key"],
                "revived_rows": revival["marker_rows"],
                "newly_suppressed_rows": result["new_mask_suppression"]["marker_rows"],
                "revived_rows_after_short_prune": revival["marker_rows_after_short_prune"],
                "revived_rows_removed_by_short_prune": revival["marker_rows_removed_by_short_prune"],
                "revived_row_survival_rate": revival["marker_row_survival_rate"],
                "revived_only_short_tracks": revival["strong_short_noise_tracks"],
                "revived_only_persistent_tracks": revival["persistent_independent_tracks"],
                "mixed_kept_tracks": revival["mixed_kept_tracks"],
                "legacy_gap_events": result["legacy_gaps"].get("gap_events", 0),
                "mask_gap_events": result["mask_gaps"].get("gap_events", 0),
                "geometry_changed_rows": geometry["changed_geometry_rows"],
                "geometry_partition_split_rows": geometry["partition"]["split_mismatch_rows"],
                "geometry_partition_merge_rows": geometry["partition"]["merge_mismatch_rows"],
                "geometry_prune_outcome_changed_rows": geometry["partition"]["prune_outcome_changed_rows"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sum(results: list[dict[str, object]], section: str, field: str) -> int:
    return sum(int(result[section][field]) for result in results)


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    revival_rows = _sum(results, "nms_revival", "marker_rows")
    revival_survived = _sum(results, "nms_revival", "marker_rows_after_short_prune")
    common_legacy_kept_mask_removed = sum(
        int(result["nms_partition"]["left_kept_right_removed_rows"])
        for result in results
    )
    common_legacy_removed_mask_kept = sum(
        int(result["nms_partition"]["left_removed_right_kept_rows"])
        for result in results
    )
    newly_suppressed_legacy_kept = _sum(
        results, "new_mask_suppression", "marker_rows_after_short_prune"
    )
    exact_after_prune_delta = (
        revival_survived
        + common_legacy_removed_mask_kept
        - newly_suppressed_legacy_kept
        - common_legacy_kept_mask_removed
    )
    return {
        "runs": len(results),
        "nms_revival": {
            "revived_rows": revival_rows,
            "revived_rows_after_short_prune": revival_survived,
            "revived_rows_removed_by_short_prune": _sum(results, "nms_revival", "marker_rows_removed_by_short_prune"),
            "revived_row_survival_rate": revival_survived / revival_rows if revival_rows else 0.0,
            "tracks_touched": _sum(results, "nms_revival", "tracks_touched"),
            "revived_only_tracks": _sum(results, "nms_revival", "marker_only_tracks"),
            "mixed_tracks": _sum(results, "nms_revival", "mixed_tracks"),
            "strong_short_noise_tracks": _sum(results, "nms_revival", "strong_short_noise_tracks"),
            "strong_short_noise_rows": _sum(results, "nms_revival", "strong_short_noise_rows"),
            "persistent_independent_tracks": _sum(results, "nms_revival", "persistent_independent_tracks"),
            "persistent_independent_rows": _sum(results, "nms_revival", "persistent_independent_rows"),
            "mixed_kept_tracks": _sum(results, "nms_revival", "mixed_kept_tracks"),
            "mixed_kept_marker_rows": _sum(results, "nms_revival", "mixed_kept_marker_rows"),
        },
        "new_mask_suppression": {
            "rows": _sum(results, "new_mask_suppression", "marker_rows"),
            "rows_that_had_survived_legacy_short_prune": _sum(
                results, "new_mask_suppression", "marker_rows_after_short_prune"
            ),
            "rows_already_removed_by_legacy_short_prune": _sum(
                results, "new_mask_suppression", "marker_rows_removed_by_short_prune"
            ),
            "class_counts": dict(
                sorted(
                    sum(
                        (
                            Counter(result["newly_suppressed_class_counts"])
                            for result in results
                        ),
                        Counter(),
                    ).items()
                )
            ),
        },
        "gaps": {
            "legacy_gap_events": sum(int(result["legacy_gaps"].get("gap_events", 0)) for result in results),
            "mask_gap_events": sum(int(result["mask_gaps"].get("gap_events", 0)) for result in results),
            "mask_gap_events_on_revived_tracks": sum(int(result["mask_gaps"].get("gap_events_on_marker_tracks", 0)) for result in results),
            "mask_gap_events_on_unmarked_tracks": sum(int(result["mask_gaps"].get("gap_events_on_unmarked_tracks", 0)) for result in results),
            "mask_gap_events_adjacent_to_revived_row": sum(int(result["mask_gaps"].get("gap_events_adjacent_to_marker", 0)) for result in results),
        },
        "nms_tracking_partition": {
            "split_mismatch_rows": sum(
                int(result["nms_partition"]["split_mismatch_rows"])
                for result in results
            ),
            "merge_mismatch_rows": sum(
                int(result["nms_partition"]["merge_mismatch_rows"])
                for result in results
            ),
            "common_rows_legacy_kept_mask_removed": common_legacy_kept_mask_removed,
            "common_rows_legacy_removed_mask_kept": common_legacy_removed_mask_kept,
            "exact_rows_after_prune_delta": exact_after_prune_delta,
            "delta_equation": (
                "revived_kept + common_legacy_removed_mask_kept "
                "- newly_suppressed_legacy_kept "
                "- common_legacy_kept_mask_removed"
            ),
        },
        "geometry": {
            "changed_geometry_rows": _sum(results, "geometry_effect", "changed_geometry_rows"),
            "runs_with_geometry_tracking_partition_change": [
                result["run_key"]
                for result in results
                if result["geometry_effect"]["partition"]["split_mismatch_rows"]
                or result["geometry_effect"]["partition"]["merge_mismatch_rows"]
            ],
            "split_mismatch_rows": sum(int(result["geometry_effect"]["partition"]["split_mismatch_rows"]) for result in results),
            "merge_mismatch_rows": sum(int(result["geometry_effect"]["partition"]["merge_mismatch_rows"]) for result in results),
            "prune_outcome_changed_rows": sum(int(result["geometry_effect"]["partition"]["prune_outcome_changed_rows"]) for result in results),
        },
        "class_counts_revived": dict(
            sorted(
                sum(
                    (Counter(result["revived_class_counts"]) for result in results),
                    Counter(),
                ).items()
            )
        ),
    }


def _report(aggregate: dict[str, object], results: list[dict[str, object]]) -> str:
    revival = aggregate["nms_revival"]
    new_suppression = aggregate["new_mask_suppression"]
    gaps = aggregate["gaps"]
    geometry = aggregate["geometry"]
    partition = aggregate["nms_tracking_partition"]
    lines = [
        "# NMS candidate v2: tracking-causality audit",
        "",
        "## Attribution design",
        "",
        "- `legacy -> mask_iou_only`: suppression policy only; original mask geometry is unchanged.",
        "- `mask_iou_only -> component_candidate_v2`: identical retained source IDs; only hole/island geometry changes.",
        "- `length <= 10`: the fixed short-track filter. A removed track is evidence of short lifetime, not proof of a false positive.",
        "- No video frames or identity ground truth were used.",
        "",
        "## Aggregate NMS revival",
        "",
        f"- Rows restored by mask-IoU NMS: **{revival['revived_rows']:,}**.",
        f"- Rows retained by legacy but newly suppressed by mask-IoU NMS: **{new_suppression['rows']:,}** (net row change {revival['revived_rows'] - new_suppression['rows']:+,}).",
        f"- Restored rows surviving short-track removal: **{revival['revived_rows_after_short_prune']:,} ({revival['revived_row_survival_rate']:.1%})**.",
        f"- Restored rows removed with short tracks: **{revival['revived_rows_removed_by_short_prune']:,}**.",
        f"- Revived-only short tracks (strongest noise-like subset): **{revival['strong_short_noise_tracks']:,} tracks / {revival['strong_short_noise_rows']:,} rows**.",
        f"- Revived-only persistent tracks: **{revival['persistent_independent_tracks']:,} tracks / {revival['persistent_independent_rows']:,} rows**.",
        f"- Mixed kept tracks containing both legacy and revived rows: **{revival['mixed_kept_tracks']:,}**.",
        f"- Exact post-prune net change: **{revival['revived_rows_after_short_prune']:,} revived kept + {partition['common_rows_legacy_removed_mask_kept']:,} common rows promoted − {new_suppression['rows_that_had_survived_legacy_short_prune']:,} newly suppressed legacy-kept − {partition['common_rows_legacy_kept_mask_removed']:,} common rows demoted = {partition['exact_rows_after_prune_delta']:+,} rows**.",
        f"- Common-detection partition changed substantially: split mismatch **{partition['split_mismatch_rows']:,}** rows; merge mismatch **{partition['merge_mismatch_rows']:,}** rows. These are label-invariant partition diagnostics, not raw track-ID renumbering.",
        "",
        "## Gap attribution",
        "",
        f"- Legacy gap events: **{gaps['legacy_gap_events']:,}**; mask-IoU-only: **{gaps['mask_gap_events']:,}**.",
        f"- Mask-IoU gaps on tracks containing at least one revived row: **{gaps['mask_gap_events_on_revived_tracks']:,}**.",
        f"- Mask-IoU gaps on tracks with no revived rows: **{gaps['mask_gap_events_on_unmarked_tracks']:,}**.",
        f"- Gap events directly adjacent to a revived row: **{gaps['mask_gap_events_adjacent_to_revived_row']:,}**.",
        "",
        "These are associations, not an additive causal decomposition of the gap delta: revived detections can change greedy assignment of common detections elsewhere in the same track.",
        "",
        "## Hole/island geometry effect",
        "",
        f"- Geometry changed on **{geometry['changed_geometry_rows']:,} retained detections** while source-ID sets stayed identical.",
        f"- Tracking partition changed in: **{', '.join(geometry['runs_with_geometry_tracking_partition_change']) or 'none'}**.",
        f"- Partition diagnostics: split mismatch {geometry['split_mismatch_rows']:,} rows, merge mismatch {geometry['merge_mismatch_rows']:,} rows, short-prune outcome changed {geometry['prune_outcome_changed_rows']:,} rows.",
        "",
        "## Per-run causal counts",
        "",
        "| run | revived | newly suppressed | survived | removed short | revived-only short tracks | persistent revived-only tracks | geometry changed | split mismatch | merge mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        revival = result["nms_revival"]
        geometry = result["geometry_effect"]
        lines.append(
            f"| {result['run_key']} | {revival['marker_rows']:,} | "
            f"{result['new_mask_suppression']['marker_rows']:,} | "
            f"{revival['marker_rows_after_short_prune']:,} | "
            f"{revival['marker_rows_removed_by_short_prune']:,} | "
            f"{revival['strong_short_noise_tracks']:,} | "
            f"{revival['persistent_independent_tracks']:,} | "
            f"{geometry['changed_geometry_rows']:,} | "
            f"{geometry['partition']['split_mismatch_rows']:,} | "
            f"{geometry['partition']['merge_mismatch_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "## Production decision",
            "",
            "1. Keep mask-IoU NMS as the candidate: the restored rows are not predominantly explainable as short-lived noise, but this remains a recall/fragmentation trade-off until sampled identity GT is available.",
            "2. Keep unconditional hole fill and <=1% owner-relative island cleanup: their aggregate tracking effect is small; do not infer visual correctness from tracking alone.",
            "3. Do not silently ship survivor-island geometry cleanup coupled directly to greedy tracking features. The HEYZO-3549 example proves that removing only 46 pixels can reverse the score ordering and merge two long raw tracks. Either validate that case visually, or make tracking association use pre-cleanup geometry while final masks use cleaned geometry. This can remain an internal artifact and does not require a final SQLite schema change.",
            "4. Do not tighten NMS or component thresholds based only on these proxies. No GT establishes whether the extra persistent tracks are distinct objects or duplicate detections.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.include)
    run_dirs = [
        path
        for path in sorted(input_root.iterdir())
        if path.is_dir() and (not selected or path.name in selected)
    ]
    if not run_dirs:
        raise RuntimeError("no NMS ablation runs selected")
    config = {
        "schema_version": 1,
        "input_root": str(input_root),
        "run_keys": [path.name for path in run_dirs],
        "arms": list(ARMS),
        "empty_cuts": True,
        "association_config": asdict(AssociationConfig()),
        "remove_short_tracks_max_frames": SHORT_LIMIT,
    }
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nms-tracking-causality-") as folder:
        temporary = Path(folder)
        cuts = write_cut_list(
            temporary / "cuts.json",
            CutList((), "disabled_empty_ablation", 0.0),
        )
        for index, run_dir in enumerate(run_dirs, 1):
            arm_paths = {
                arm: run_dir / "arm_outputs" / f"{arm}.jsonl" for arm in ARMS
            }
            for arm, path in arm_paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"{run_dir.name}/{arm}: {path}")
            sqlite_paths = {
                arm: temporary / f"{run_dir.name}__{arm}.sqlite" for arm in ARMS
            }
            tracking_results = {
                arm: _run_tracking(arm_paths[arm], sqlite_paths[arm], cuts)
                for arm in ARMS
            }
            rows = {arm: _read_rows(sqlite_paths[arm]) for arm in ARMS}
            legacy_keys = set(rows["legacy"])
            mask_keys = set(rows["mask_iou_only"])
            candidate_keys = set(rows["component_candidate_v2"])
            if mask_keys != candidate_keys:
                raise RuntimeError(f"{run_dir.name}: candidate source IDs differ from mask-IoU-only")
            revived = mask_keys - legacy_keys
            newly_suppressed = legacy_keys - mask_keys
            geometry_changed = {
                key
                for key in mask_keys
                if rows["mask_iou_only"][key].geometry_sha256
                != rows["component_candidate_v2"][key].geometry_sha256
            }
            result: dict[str, object] = {
                "run_key": run_dir.name,
                "tracking_results": tracking_results,
                "legacy_rows": len(legacy_keys),
                "mask_rows": len(mask_keys),
                "nms_revival": _marker_track_audit(rows["mask_iou_only"], revived),
                "revived_class_counts": _class_counts(rows["mask_iou_only"], revived),
                "new_mask_suppression": _marker_track_audit(
                    rows["legacy"], newly_suppressed
                ),
                "newly_suppressed_class_counts": _class_counts(
                    rows["legacy"], newly_suppressed
                ),
                "newly_suppressed_rows": [
                    {
                        "frame": rows["legacy"][key].frame,
                        "source_detection_id": rows["legacy"][key].source_detection_id,
                        "label": rows["legacy"][key].raw_label,
                        "legacy_raw_track_id": rows["legacy"][key].raw_track_id,
                        "legacy_removed_by_short_track": rows["legacy"][key].removed,
                        "legacy_track_length": rows["legacy"][key].track_length,
                    }
                    for key in sorted(newly_suppressed)
                ],
                "legacy_gaps": _gap_audit(rows["legacy"], set()),
                "mask_gaps": _gap_audit(rows["mask_iou_only"], revived),
                "nms_partition": _partition_audit(
                    rows["legacy"], rows["mask_iou_only"], legacy_keys & mask_keys
                ),
                "geometry_effect": {
                    "changed_geometry_rows": len(geometry_changed),
                    "changed_frames": len({key[0] for key in geometry_changed}),
                    "changed_class_counts": _class_counts(rows["component_candidate_v2"], geometry_changed),
                    "partition": _partition_audit(rows["mask_iou_only"], rows["component_candidate_v2"], mask_keys),
                    "changed_geometry_track_audit": _marker_track_audit(rows["component_candidate_v2"], geometry_changed),
                },
            }
            if run_dir.name == "v3__heyzo_3549_full":
                result["mechanical_evidence_frame_75789"] = _mechanical_3549_evidence(
                    rows["mask_iou_only"], rows["component_candidate_v2"]
                )
            results.append(result)
            print(
                f"[{index}/{len(run_dirs)}] {run_dir.name} revived={len(revived)} "
                f"newly_suppressed={len(newly_suppressed)} geometry={len(geometry_changed)}",
                flush=True,
            )

    aggregate = _aggregate(results)
    payload = {
        "config": config,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
        "runs": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = _csv_rows(results)
    _write_csv(output_root / "per_run.csv", rows)
    (output_root / "REPORT.md").write_text(
        _report(aggregate, results), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
