#!/usr/bin/env python3
"""Run one fixed tracking/pruning configuration over every NMS ablation arm.

This experiment intentionally varies only the JSONL produced by the four NMS
arms.  Cut detection is disabled with one shared empty cut-list, tracking uses
the canonical default ``AssociationConfig``, and tracks with at most 10
observations are removed.  Large tracked SQLite files are audited and deleted
by default; pass ``--keep-sqlite`` when the intermediate databases are needed.

The script is restartable.  A completed per-arm ``summary.json`` is reused
only when its input SHA-256 and experiment configuration SHA-256 still match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POSTPROCESS_ROOT = REPOSITORY_ROOT / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from contracts.detections import CutList, write_cut_list  # noqa: E402
from tracking.association import AssociationConfig  # noqa: E402
from tracking.builder import build_tracked_sqlite  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPOSITORY_ROOT / "output" / "nms_component_candidate_v2_ablation_20260813" / "runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "output" / "nms_component_candidate_v2_tracking_20260813"
)
ARM_ORDER = (
    "legacy",
    "topology_then_legacy",
    "mask_iou_only",
    "component_candidate_v2",
    "virtual_component_v3",
    "virtual_component_mask_v4",
)
DEFAULT_ARMS = ARM_ORDER[:-1]
FRAME_RE = re.compile(rb'"frame_index"\s*:\s*(\d+)')
EXPERIMENT_SCHEMA_VERSION = 1


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_jsonl(path: Path) -> dict[str, object]:
    """Hash and count canonical frame lines without decoding polygon payloads."""

    digest = hashlib.sha256()
    frames = 0
    first_frame: int | None = None
    last_frame: int | None = None
    monotonic = True
    previous: int | None = None
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if not line.strip():
                continue
            match = FRAME_RE.search(line[:256])
            if match is None:
                raise ValueError(
                    f"{path}:{line_number}: frame_index is not near line start"
                )
            frame = int(match.group(1))
            if previous is not None and frame <= previous:
                monotonic = False
            if first_frame is None:
                first_frame = frame
            last_frame = frame
            previous = frame
            frames += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "frame_records": frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "strictly_increasing_frames": monotonic,
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(materialized),
        "min": min(materialized),
        "p01": _percentile(materialized, 0.01),
        "p05": _percentile(materialized, 0.05),
        "p25": _percentile(materialized, 0.25),
        "p50": _percentile(materialized, 0.50),
        "p75": _percentile(materialized, 0.75),
        "p90": _percentile(materialized, 0.90),
        "p95": _percentile(materialized, 0.95),
        "p99": _percentile(materialized, 0.99),
        "max": max(materialized),
        "mean": sum(materialized) / len(materialized),
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(int(value) for value in values).items())
    }


def _schema_inventory(connection: sqlite3.Connection) -> dict[str, object]:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, COALESCE(sql, '')
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    canonical = [list(row) for row in rows]
    table_info: dict[str, list[list[object]]] = {}
    table_counts: dict[str, int] = {}
    for kind, name, _table_name, _sql in rows:
        if kind != "table":
            continue
        escaped = str(name).replace('"', '""')
        table_info[str(name)] = [
            list(row)
            for row in connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
        ]
        table_counts[str(name)] = int(
            connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
        )
    signature = {"sqlite_master": canonical, "table_info": table_info}
    return {
        "sha256": hashlib.sha256(_json_bytes(signature)).hexdigest(),
        "objects": canonical,
        "table_info": table_info,
        "table_counts": table_counts,
    }


def _assignment_hash(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute(
        """
        SELECT frame, raw_track_id, raw_detection_index,
               COALESCE(source_detection_id, ''),
               COALESCE(final_track_id, ''), removed_by_short_track,
               raw_track_length, COALESCE(raw_label, ''),
               COALESCE(final_label, ''), COALESCE(scene_id, -1)
        FROM raw_tracked_masks
        ORDER BY CAST(raw_track_id AS INTEGER), frame, raw_detection_index
        """
    )
    for row in cursor:
        digest.update(_json_bytes(list(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _class_counts(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT COALESCE(raw_label, '') AS label,
               COUNT(*) AS rows_before_prune,
               COUNT(DISTINCT raw_track_id) AS raw_tracks,
               SUM(CASE WHEN removed_by_short_track=0 THEN 1 ELSE 0 END)
                   AS rows_after_prune,
               COUNT(DISTINCT CASE WHEN removed_by_short_track=0
                                   THEN final_track_id END) AS kept_tracks,
               COUNT(DISTINCT CASE WHEN removed_by_short_track=1
                                   THEN raw_track_id END) AS removed_tracks
        FROM raw_tracked_masks
        GROUP BY COALESCE(raw_label, '')
        ORDER BY label
        """
    ).fetchall()
    return [
        {
            "label": str(row[0]),
            "rows_before_prune": int(row[1]),
            "raw_tracks": int(row[2]),
            "rows_after_prune": int(row[3]),
            "kept_tracks": int(row[4]),
            "removed_tracks": int(row[5]),
        }
        for row in rows
    ]


def _track_metrics(
    connection: sqlite3.Connection, frame_records: int
) -> dict[str, object]:
    track_rows = connection.execute(
        """
        SELECT raw_track_id, removed_by_short_track, raw_track_length,
               COALESCE(raw_label, ''), COALESCE(final_label, ''),
               COALESCE(scene_id, -1)
        FROM raw_tracks
        ORDER BY CAST(raw_track_id AS INTEGER)
        """
    ).fetchall()
    resolution = {
        str(row[0]): {
            "removed": bool(row[1]),
            "length": int(row[2]),
            "raw_label": str(row[3]),
            "final_label": str(row[4]),
            "scene_id": int(row[5]),
        }
        for row in track_rows
    }

    summaries: dict[str, dict[str, object]] = {}
    active_id: str | None = None
    first = previous = last = observations = gap_events = missing = max_gap = 0

    def finish() -> None:
        nonlocal active_id, first, previous, last, observations
        nonlocal gap_events, missing, max_gap
        if active_id is None:
            return
        metadata = resolution[active_id]
        span = last - first + 1
        summaries[active_id] = {
            **metadata,
            "observations": observations,
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "density": observations / span if span else 0.0,
            "gap_events": gap_events,
            "missing_frames": missing,
            "max_internal_gap": max_gap,
        }

    cursor = connection.execute(
        """
        SELECT raw_track_id, frame
        FROM raw_tracked_masks
        ORDER BY CAST(raw_track_id AS INTEGER), frame, raw_detection_index
        """
    )
    for raw_track_id_value, frame_value in cursor:
        raw_track_id = str(raw_track_id_value)
        frame = int(frame_value)
        if raw_track_id != active_id:
            finish()
            active_id = raw_track_id
            first = previous = last = frame
            observations = 1
            gap_events = missing = max_gap = 0
            continue
        difference = frame - previous
        if difference > 1:
            gap_events += 1
            missing += difference - 1
            max_gap = max(max_gap, difference - 1)
        previous = last = frame
        observations += 1
    finish()

    if set(summaries) != set(resolution):
        raise RuntimeError("raw_tracks and raw_tracked_masks track IDs disagree")
    for track_id, summary in summaries.items():
        if int(summary["observations"]) != int(summary["length"]):
            raise RuntimeError(
                f"track {track_id}: stored and observed lengths disagree"
            )

    def group_metrics(items: list[dict[str, object]]) -> dict[str, object]:
        lengths = [int(item["observations"]) for item in items]
        spans = [int(item["span_frames"]) for item in items]
        densities = [float(item["density"]) for item in items]
        gap_counts = [int(item["gap_events"]) for item in items]
        missing_counts = [int(item["missing_frames"]) for item in items]
        max_gaps = [int(item["max_internal_gap"]) for item in items]
        total_observations = sum(lengths)
        return {
            "tracks": len(items),
            "observations": total_observations,
            "tracks_per_1000_frames": (
                len(items) * 1000.0 / frame_records if frame_records else None
            ),
            "tracks_per_1000_observations": (
                len(items) * 1000.0 / total_observations if total_observations else None
            ),
            "observation_length": _distribution(lengths),
            "span_frames": _distribution(spans),
            "observation_density": _distribution(densities),
            "length_histogram": _histogram(lengths),
            "span_histogram": _histogram(spans),
            "gap_events": sum(gap_counts),
            "internal_missing_frames": sum(missing_counts),
            "tracks_with_gaps": sum(value > 0 for value in gap_counts),
            "max_internal_gap": max(max_gaps, default=0),
            "fragmentation_proxy": {
                "definition": "new raw tracks per 1000 retained detection rows",
                "value": (
                    len(items) * 1000.0 / total_observations
                    if total_observations
                    else None
                ),
            },
        }

    all_items = list(summaries.values())
    kept_items = [item for item in all_items if not bool(item["removed"])]
    removed_items = [item for item in all_items if bool(item["removed"])]
    return {
        "all": group_metrics(all_items),
        "kept": group_metrics(kept_items),
        "removed": group_metrics(removed_items),
        "removed_track_fraction": (
            len(removed_items) / len(all_items) if all_items else 0.0
        ),
        "removed_row_fraction": (
            sum(int(item["observations"]) for item in removed_items)
            / sum(int(item["observations"]) for item in all_items)
            if all_items
            else 0.0
        ),
    }


def _audit_sqlite(path: Path, input_scan: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        schema = _schema_inventory(connection)
        tracking = _track_metrics(connection, int(input_scan["frame_records"]))
        classes = _class_counts(connection)
        assignment_sha256 = _assignment_hash(connection)
    return {
        "integrity_check": integrity,
        "quick_check": quick_check,
        "schema": schema,
        "tracking": tracking,
        "class_counts": classes,
        "assignment_sha256": assignment_sha256,
        "sqlite_sha256": _sha256_file(path),
        "sqlite_bytes": path.stat().st_size,
        "audit_elapsed_seconds": time.perf_counter() - started,
    }


def _config_payload(remove_short: int) -> dict[str, object]:
    return {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "cut_list": {
            "schema_version": 1,
            "frames": [],
            "method": "disabled_empty_ablation",
            "elapsed_seconds": 0.0,
        },
        "association_config": asdict(AssociationConfig()),
        "remove_short_tracks_max_frames": remove_short,
    }


def _run_one(task: dict[str, object]) -> dict[str, object]:
    run_key = str(task["run_key"])
    arm = str(task["arm"])
    input_jsonl = Path(str(task["input_jsonl"]))
    output_dir = Path(str(task["output_dir"]))
    cuts_path = Path(str(task["cuts_path"]))
    remove_short = int(task["remove_short"])
    keep_sqlite = bool(task["keep_sqlite"])
    force = bool(task["force"])
    config_sha256 = str(task["config_sha256"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    input_scan = _scan_jsonl(input_jsonl)
    if summary_path.is_file() and not force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("config_sha256") == config_sha256
            and existing.get("input", {}).get("sha256") == input_scan["sha256"]
        ):
            return existing

    sqlite_path = output_dir / "tracked.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{sqlite_path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()

    started = time.perf_counter()
    tracking_result = build_tracked_sqlite(
        input_jsonl,
        sqlite_path,
        cuts_path,
        remove_short_tracks_max_frames=remove_short,
        association_config=AssociationConfig(),
    )
    audit = _audit_sqlite(sqlite_path, input_scan)
    if audit["integrity_check"] != "ok" or audit["quick_check"] != "ok":
        raise RuntimeError(f"{run_key}/{arm}: SQLite integrity validation failed")

    result = {
        "status": "complete",
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_key": run_key,
        "arm": arm,
        "config_sha256": config_sha256,
        "input_jsonl": str(input_jsonl.resolve()),
        "input": input_scan,
        "tracking_result": tracking_result,
        "audit": audit,
        "total_elapsed_seconds": time.perf_counter() - started,
        "tracked_sqlite_retained": keep_sqlite,
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, summary_path)
    if not keep_sqlite:
        sqlite_path.unlink()
    return result


def _discover_tasks(
    input_root: Path,
    output_root: Path,
    includes: set[str],
    arms: tuple[str, ...],
    cuts_path: Path,
    remove_short: int,
    keep_sqlite: bool,
    force: bool,
    config_sha256: str,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for run_dir in sorted(input_root.iterdir()):
        if not run_dir.is_dir() or (includes and run_dir.name not in includes):
            continue
        arm_dir = run_dir / "arm_outputs"
        for arm in arms:
            source = arm_dir / f"{arm}.jsonl"
            if not source.is_file():
                raise FileNotFoundError(f"missing ablation arm: {source}")
            tasks.append(
                {
                    "run_key": run_dir.name,
                    "arm": arm,
                    "input_jsonl": str(source),
                    "output_dir": str(output_root / "runs" / run_dir.name / arm),
                    "cuts_path": str(cuts_path),
                    "remove_short": remove_short,
                    "keep_sqlite": keep_sqlite,
                    "force": force,
                    "config_sha256": config_sha256,
                }
            )
    if not tasks:
        raise RuntimeError(f"no arm JSONLs found under {input_root}")
    return tasks


def _sum_class_counts(results: list[dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for row in result["audit"]["class_counts"]:  # type: ignore[index]
            label = str(row["label"])
            for key in (
                "rows_before_prune",
                "raw_tracks",
                "rows_after_prune",
                "kept_tracks",
                "removed_tracks",
            ):
                totals[label][key] += int(row[key])
    return [{"label": label, **dict(totals[label])} for label in sorted(totals)]


def _merged_histogram(
    tracking_metrics: list[dict[str, object]],
    group: str,
    field: str,
) -> dict[str, int]:
    merged: Counter[int] = Counter()
    for metric in tracking_metrics:
        histogram = metric[group][field]  # type: ignore[index]
        for value, count in histogram.items():
            merged[int(value)] += int(count)
    return {str(value): count for value, count in sorted(merged.items())}


def _distribution_from_histogram(histogram: dict[str, int]) -> dict[str, object]:
    values = [
        float(value) for value, count in histogram.items() for _ in range(int(count))
    ]
    return _distribution(values)


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        by_arm[str(result["arm"])].append(result)
    arms: dict[str, object] = {}
    for arm in ARM_ORDER:
        members = by_arm.get(arm, [])
        if not members:
            continue
        fields = (
            "rows_before_prune",
            "rows_after_prune",
            "removed_short_tracks",
            "removed_rows",
            "tracks_after_prune",
            "raw_tracked_rows",
            "raw_tracks",
        )
        totals = {
            field: sum(int(member["tracking_result"][field]) for member in members)  # type: ignore[index]
            for field in fields
        }
        raw_tracking = [member["audit"]["tracking"] for member in members]  # type: ignore[index]
        all_observations = totals["rows_before_prune"]
        kept_observations = totals["rows_after_prune"]
        lifetime: dict[str, object] = {}
        for group in ("all", "kept", "removed"):
            length_histogram = _merged_histogram(
                raw_tracking, group, "length_histogram"
            )
            span_histogram = _merged_histogram(raw_tracking, group, "span_histogram")
            lifetime[group] = {
                "observation_length": _distribution_from_histogram(length_histogram),
                "span_frames": _distribution_from_histogram(span_histogram),
                "length_histogram": length_histogram,
                "span_histogram": span_histogram,
                "tracks_with_gaps": sum(
                    int(metric[group]["tracks_with_gaps"])  # type: ignore[index]
                    for metric in raw_tracking
                ),
                "gap_events": sum(
                    int(metric[group]["gap_events"])  # type: ignore[index]
                    for metric in raw_tracking
                ),
                "internal_missing_frames": sum(
                    int(metric[group]["internal_missing_frames"])  # type: ignore[index]
                    for metric in raw_tracking
                ),
                "max_internal_gap": max(
                    (
                        int(metric[group]["max_internal_gap"])  # type: ignore[index]
                        for metric in raw_tracking
                    ),
                    default=0,
                ),
            }
        arms[arm] = {
            "runs": len(members),
            **totals,
            "removed_track_fraction": (
                totals["removed_short_tracks"] / totals["raw_tracks"]
                if totals["raw_tracks"]
                else 0.0
            ),
            "removed_row_fraction": (
                totals["removed_rows"] / all_observations if all_observations else 0.0
            ),
            "raw_tracks_per_1000_observations": (
                totals["raw_tracks"] * 1000.0 / all_observations
                if all_observations
                else None
            ),
            "kept_tracks_per_1000_observations": (
                totals["tracks_after_prune"] * 1000.0 / kept_observations
                if kept_observations
                else None
            ),
            "raw_gap_events": sum(
                int(metric["all"]["gap_events"]) for metric in raw_tracking
            ),
            "raw_internal_missing_frames": sum(
                int(metric["all"]["internal_missing_frames"]) for metric in raw_tracking
            ),
            "kept_gap_events": sum(
                int(metric["kept"]["gap_events"]) for metric in raw_tracking
            ),
            "tracking_elapsed_seconds": sum(
                float(member["tracking_result"]["elapsed_sec"])  # type: ignore[index]
                for member in members
            ),
            "audit_elapsed_seconds": sum(
                float(member["audit"]["audit_elapsed_seconds"])  # type: ignore[index]
                for member in members
            ),
            "lifetime": lifetime,
            "class_counts": _sum_class_counts(members),
            "schema_sha256_values": sorted(
                {str(member["audit"]["schema"]["sha256"]) for member in members}  # type: ignore[index]
            ),
            "integrity_all_ok": all(
                member["audit"]["integrity_check"] == "ok"  # type: ignore[index]
                and member["audit"]["quick_check"] == "ok"  # type: ignore[index]
                for member in members
            ),
        }

    legacy = arms.get("legacy")
    comparisons: dict[str, object] = {}
    if isinstance(legacy, dict):
        for arm, summary in arms.items():
            if arm == "legacy" or not isinstance(summary, dict):
                continue
            comparisons[f"{arm}_vs_legacy"] = {
                field: int(summary[field]) - int(legacy[field])
                for field in (
                    "rows_before_prune",
                    "rows_after_prune",
                    "removed_short_tracks",
                    "removed_rows",
                    "tracks_after_prune",
                    "raw_tracks",
                    "raw_gap_events",
                    "kept_gap_events",
                )
            }

    schema_hashes = sorted(
        {
            str(result["audit"]["schema"]["sha256"])  # type: ignore[index]
            for result in results
        }
    )
    by_run_arm = {
        (str(result["run_key"]), str(result["arm"])): result for result in results
    }
    pairwise_assignment_identity: dict[str, object] = {}
    for left, right in (
        ("legacy", "topology_then_legacy"),
        ("legacy", "mask_iou_only"),
        ("mask_iou_only", "component_candidate_v2"),
        ("legacy", "virtual_component_v3"),
        ("component_candidate_v2", "virtual_component_v3"),
        ("legacy", "virtual_component_mask_v4"),
        ("virtual_component_v3", "virtual_component_mask_v4"),
    ):
        matching_runs: list[str] = []
        differing_runs: list[str] = []
        for run_key in sorted({key[0] for key in by_run_arm}):
            left_result = by_run_arm.get((run_key, left))
            right_result = by_run_arm.get((run_key, right))
            if left_result is None or right_result is None:
                continue
            identical = (
                left_result["audit"]["assignment_sha256"]  # type: ignore[index]
                == right_result["audit"]["assignment_sha256"]  # type: ignore[index]
            )
            (matching_runs if identical else differing_runs).append(run_key)
        pairwise_assignment_identity[f"{left}_vs_{right}"] = {
            "identical_runs": matching_runs,
            "differing_runs": differing_runs,
            "identical_count": len(matching_runs),
            "differing_count": len(differing_runs),
        }
    return {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "completed_tasks": len(results),
        "run_keys": sorted({str(result["run_key"]) for result in results}),
        "arms": arms,
        "comparisons": comparisons,
        "pairwise_assignment_identity": pairwise_assignment_identity,
        "schema_sha256_values": schema_hashes,
        "schema_is_identical_across_all_outputs": len(schema_hashes) == 1,
        "integrity_all_ok": all(
            result["audit"]["integrity_check"] == "ok"  # type: ignore[index]
            and result["audit"]["quick_check"] == "ok"  # type: ignore[index]
            for result in results
        ),
    }


def _write_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "run_key",
        "arm",
        "frame_records",
        "rows_before_prune",
        "rows_after_prune",
        "raw_tracks",
        "tracks_after_prune",
        "removed_short_tracks",
        "removed_rows",
        "removed_track_fraction",
        "removed_row_fraction",
        "raw_gap_events",
        "raw_internal_missing_frames",
        "kept_gap_events",
        "raw_tracks_per_1000_observations",
        "tracking_elapsed_seconds",
        "audit_elapsed_seconds",
        "schema_sha256",
        "assignment_sha256",
        "sqlite_sha256",
        "integrity_check",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(
            results,
            key=lambda item: (str(item["run_key"]), ARM_ORDER.index(str(item["arm"]))),
        ):
            tracking_result = result["tracking_result"]
            audit = result["audit"]
            track_metrics = audit["tracking"]
            writer.writerow(
                {
                    "run_key": result["run_key"],
                    "arm": result["arm"],
                    "frame_records": result["input"]["frame_records"],
                    "rows_before_prune": tracking_result["rows_before_prune"],
                    "rows_after_prune": tracking_result["rows_after_prune"],
                    "raw_tracks": tracking_result["raw_tracks"],
                    "tracks_after_prune": tracking_result["tracks_after_prune"],
                    "removed_short_tracks": tracking_result["removed_short_tracks"],
                    "removed_rows": tracking_result["removed_rows"],
                    "removed_track_fraction": track_metrics["removed_track_fraction"],
                    "removed_row_fraction": track_metrics["removed_row_fraction"],
                    "raw_gap_events": track_metrics["all"]["gap_events"],
                    "raw_internal_missing_frames": track_metrics["all"][
                        "internal_missing_frames"
                    ],
                    "kept_gap_events": track_metrics["kept"]["gap_events"],
                    "raw_tracks_per_1000_observations": track_metrics["all"][
                        "tracks_per_1000_observations"
                    ],
                    "tracking_elapsed_seconds": tracking_result["elapsed_sec"],
                    "audit_elapsed_seconds": audit["audit_elapsed_seconds"],
                    "schema_sha256": audit["schema"]["sha256"],
                    "assignment_sha256": audit["assignment_sha256"],
                    "sqlite_sha256": audit["sqlite_sha256"],
                    "integrity_check": audit["integrity_check"],
                }
            )


def _format_delta(value: object) -> str:
    number = int(value)
    return f"{number:+,d}"


def _write_report(
    path: Path, aggregate: dict[str, object], config: dict[str, object]
) -> None:
    arms = aggregate["arms"]
    comparisons = aggregate["comparisons"]
    lines = [
        "# Fixed-tracking ablation after four NMS arms",
        "",
        "## Fixed conditions",
        "",
        "- Cuts: disabled with one empty cut list.",
        "- Tracking: canonical `AssociationConfig()` defaults.",
        f"- Short-track removal: length <= {config['remove_short_tracks_max_frames']} observations.",
        "- Only the four NMS/topology arm JSONLs differ.",
        "- Fragmentation proxy: newly created raw tracks per 1,000 retained detection rows; this is not identity-ground-truth fragmentation.",
        "",
        "## Aggregate counts",
        "",
        "| arm | rows before | rows after | raw tracks | kept tracks | short tracks removed | removed rows | raw gap events | tracking sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        summary = arms.get(arm)
        if not isinstance(summary, dict):
            continue
        lines.append(
            f"| {arm} | {summary['rows_before_prune']:,} | {summary['rows_after_prune']:,} | "
            f"{summary['raw_tracks']:,} | {summary['tracks_after_prune']:,} | "
            f"{summary['removed_short_tracks']:,} | {summary['removed_rows']:,} | "
            f"{summary['raw_gap_events']:,} | {summary['tracking_elapsed_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Track lifetime and continuity",
            "",
            "| arm | raw length p50 | raw length p90 | kept length p50 | kept length p90 | tracks with gaps | missing frames inside tracks |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARM_ORDER:
        summary = arms.get(arm)
        if not isinstance(summary, dict):
            continue
        lifetime = summary["lifetime"]
        lines.append(
            f"| {arm} | {lifetime['all']['observation_length']['p50']:.3f} | "
            f"{lifetime['all']['observation_length']['p90']:.3f} | "
            f"{lifetime['kept']['observation_length']['p50']:.3f} | "
            f"{lifetime['kept']['observation_length']['p90']:.3f} | "
            f"{lifetime['all']['tracks_with_gaps']:,} | "
            f"{lifetime['all']['internal_missing_frames']:,} |"
        )
    lines.extend(
        [
            "",
            "## Change versus legacy",
            "",
            "| arm | retained rows before | retained rows after | raw tracks | kept tracks | removed short tracks | gap events |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, delta in comparisons.items():
        arm = key.removesuffix("_vs_legacy")
        lines.append(
            f"| {arm} | {_format_delta(delta['rows_before_prune'])} | "
            f"{_format_delta(delta['rows_after_prune'])} | {_format_delta(delta['raw_tracks'])} | "
            f"{_format_delta(delta['tracks_after_prune'])} | "
            f"{_format_delta(delta['removed_short_tracks'])} | {_format_delta(delta['raw_gap_events'])} |"
        )
    lines.extend(
        [
            "",
            "## Assignment-hash equality by run",
            "",
            "| comparison | identical runs | differing runs |",
            "|---|---:|---:|",
        ]
    )
    for comparison, identity in aggregate["pairwise_assignment_identity"].items():
        lines.append(
            f"| {comparison} | {identity['identical_count']} | {identity['differing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Completed tasks: {aggregate['completed_tasks']}",
            f"- SQLite integrity/quick checks all pass: {aggregate['integrity_all_ok']}",
            f"- One identical schema hash across all outputs: {aggregate['schema_is_identical_across_all_outputs']}",
            f"- Schema SHA-256: `{', '.join(aggregate['schema_sha256_values'])}`",
            "- Per-arm input, assignment, schema, and complete SQLite SHA-256 hashes are in `runs/*/*/summary.json` and `tracking_rows.csv`.",
            "",
            "## Interpretation boundary",
            "",
            "These measurements isolate NMS/topology effects on deterministic association and short-track pruning. They do not supply object identity ground truth, so raw-track creation and gaps are fragmentation proxies rather than direct identity-error measurements.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--include", action="append", default=[], help="run key to include; repeatable"
    )
    parser.add_argument(
        "--arms", nargs="+", choices=ARM_ORDER, default=list(DEFAULT_ARMS)
    )
    parser.add_argument("--remove-short", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "parallel arm builds; the default is deliberately 1 so per-arm "
            "tracking timings are not biased by I/O or CPU contention"
        ),
    )
    parser.add_argument("--keep-sqlite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.remove_short < 0:
        raise ValueError("--remove-short must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    output_root.mkdir(parents=True, exist_ok=True)

    config = _config_payload(args.remove_short)
    config_sha256 = hashlib.sha256(_json_bytes(config)).hexdigest()
    config_document = {**config, "config_sha256": config_sha256}
    (output_root / "config.json").write_text(
        json.dumps(config_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cuts_path = write_cut_list(
        output_root / "cuts_empty.json",
        CutList(frames=(), method="disabled_empty_ablation", elapsed_seconds=0.0),
    )
    tasks = _discover_tasks(
        input_root,
        output_root,
        set(args.include),
        tuple(args.arms),
        cuts_path,
        args.remove_short,
        args.keep_sqlite,
        args.force,
        config_sha256,
    )

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(tasks)}] {task['run_key']}/{task['arm']} "
                f"rows={result['tracking_result']['rows_before_prune']} "
                f"tracks={result['tracking_result']['raw_tracks']} "
                f"elapsed={result['total_elapsed_seconds']:.3f}s",
                flush=True,
            )

    results.sort(
        key=lambda item: (str(item["run_key"]), ARM_ORDER.index(str(item["arm"])))
    )
    aggregate = _aggregate(results)
    aggregate["config_sha256"] = config_sha256
    aggregate["wall_elapsed_seconds"] = time.perf_counter() - started
    aggregate["tracked_sqlites_retained"] = bool(args.keep_sqlite)
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_root / "tracking_rows.csv", results)
    _write_report(output_root / "REPORT.md", aggregate, config)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
