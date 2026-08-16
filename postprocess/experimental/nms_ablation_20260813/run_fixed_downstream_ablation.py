#!/usr/bin/env python3
"""Run an NMS-only A/B with one frozen downstream implementation.

The input is the same canonical ``scored.jsonl`` and cut list for every arm.
Only the NMS/topology policy changes.  Tracking and the experimental
``polygon14_keyframe_v1`` optimizer are then invoked with identical settings:
minimum Recall 0.97, target interval 6, and two per-key pair-vote sweeps.

This harness is intentionally isolated from the Production pipeline registry.
It writes new artifacts below ``--output-root`` and never mutates an input
SQLite or the software-facing SQLite schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POSTPROCESS = ROOT / "postprocess"
if str(POSTPROCESS) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS))

from contracts.detections import CutList, transform_detection_jsonl
from nms.adaptive import AdaptiveNms
from nms.component_aware import ComponentAwareMaskNms
from nms.component_virtual import VirtualComponentMaskNms, VirtualComponentNms
from nms.components import fill_holes_and_remove_tiny_islands
from tracking.builder import build_tracked_sqlite
from production.polygon.input_geometry import (
    apply_border_expansion,
    apply_endpoint_extension,
)


POLICIES = (
    "legacy_production",
    "topology_legacy_nms",
    "mask_nms_only",
    "component_mask_v2",
    "virtual_component_v3",
    "virtual_component_mask_v4",
)
LABELS = ("女性器", "男性器", "結合部分")
PROFILE = "polygon14_keyframe_v1"
RECALL_FLOOR = 0.97
TARGET_INTERVAL = 6
PAIR_VOTE_SWEEPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy and component-aware NMS while freezing tracking "
            "and polygon14/min-Recall/DP/pair-vote downstream processing."
        )
    )
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--cuts-json", type=Path, required=True)
    parser.add_argument(
        "--input-video",
        type=Path,
        help=(
            "local source video; only frame-count metadata is read for the "
            "frozen endpoint-extension safeguard"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help=f"comma-separated subset of {POLICIES}",
    )
    parser.add_argument("--source-name", default="v3_scored_masks")
    parser.add_argument("--remove-short-tracks-max-frames", type=int, default=10)
    parser.add_argument("--label-workers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pair-vote-threads", type=int, default=2)
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="limit tracks per class in the downstream smoke run; 0 means all",
    )
    parser.add_argument(
        "--skip-polygon",
        action="store_true",
        help="run only NMS and tracking (useful for a fast harness smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow existing arm artifacts to be replaced/recomputed",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_fingerprint(path: Path) -> dict[str, object]:
    """Return a stable structural fingerprint without reading data rows."""
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        objects = [
            {"type": str(kind), "name": str(name), "sql": str(sql or "")}
            for kind, name, sql in db.execute(
                """
                SELECT type, name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        ]
        integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_errors = len(db.execute("PRAGMA foreign_key_check").fetchall())
    canonical = json.dumps(
        objects, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "objects": objects,
        "integrity_check": integrity,
        "foreign_key_errors": foreign_key_errors,
    }


def _policy(
    name: str,
) -> AdaptiveNms | ComponentAwareMaskNms | VirtualComponentNms:
    if name == "legacy_production":
        # Exact current defaults: adaptive bbox IoU plus containment, with no
        # hole or connected-component cleanup.
        return AdaptiveNms()
    if name == "component_mask_v2":
        # Frozen 2026-08-13 candidate thresholds.
        return ComponentAwareMaskNms(
            mask_iou_threshold=0.70,
            fill_all_holes=True,
            unconditional_owner_ratio_max=0.01,
            island_other_coverage_min=0.80,
            island_to_other_area_max=0.50,
        )
    if name == "virtual_component_v3":
        return VirtualComponentNms(
            fill_all_holes=True,
            unconditional_owner_ratio_max=0.01,
            island_other_coverage_min=0.80,
            island_to_other_area_max=0.50,
        )
    if name == "virtual_component_mask_v4":
        return VirtualComponentMaskNms(
            fill_all_holes=True,
            unconditional_owner_ratio_max=0.01,
            island_other_coverage_min=0.80,
            island_to_other_area_max=0.50,
        )
    if name == "mask_nms_only":
        return ComponentAwareMaskNms(
            name="mask_nms_only",
            mask_iou_threshold=0.70,
            fill_all_holes=False,
            unconditional_owner_ratio_max=-1.0,
            island_other_coverage_min=2.0,
            island_to_other_area_max=-1.0,
        )
    if name == "topology_legacy_nms":
        # The caller applies topology cleanup before the unchanged legacy NMS.
        return AdaptiveNms(name="topology_legacy_nms")
    raise ValueError(f"unsupported policy: {name}")


def _run_nms(
    scored_jsonl: Path,
    output_jsonl: Path,
    policy_name: str,
) -> dict[str, object]:
    implementation = _policy(policy_name)
    diagnostics: Counter[str] = Counter()

    def transform(record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        detections = list(record["detections"])
        if policy_name == "topology_legacy_nms":
            preprocessed, topology = fill_holes_and_remove_tiny_islands(
                detections,
                fill_all_holes=True,
                unconditional_owner_ratio_max=0.01,
            )
            diagnostics.update(topology.as_dict())
            retained = implementation.apply(preprocessed)
        elif isinstance(implementation, (ComponentAwareMaskNms, VirtualComponentNms)):
            retained, frame_diagnostics = implementation.apply_with_diagnostics(
                detections
            )
            diagnostics.update(frame_diagnostics.as_dict())
        else:
            retained = implementation.apply(detections)
        result["detections"] = retained
        return result

    started = time.perf_counter()
    stream = transform_detection_jsonl(
        scored_jsonl,
        output_jsonl,
        transform,
    )
    elapsed = time.perf_counter() - started
    return {
        **stream,
        "policy": policy_name,
        "implementation": implementation.name,
        "elapsed_seconds": elapsed,
        "frames_per_second": float(stream["frames"]) / max(elapsed, 1e-9),
        "diagnostics": dict(sorted(diagnostics.items())),
    }


def _source_dimensions(scored_jsonl: Path) -> tuple[int, int]:
    from contracts.detections import iter_detection_records

    first = next(iter_detection_records(scored_jsonl), None)
    if first is None:
        raise ValueError(f"empty scored JSONL: {scored_jsonl}")
    width = int(first.get("width") or 0)
    height = int(first.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"scored JSONL needs positive width/height: {scored_jsonl}")
    return width, height


def _split_tracked_sqlite_by_label(
    tracked_sqlite: Path,
    output_sqlite: Path,
    label: str,
) -> dict[str, object]:
    """Create a schema-identical, label-specific Phase-2 input database.

    Phase 2 uses ``MASK_PIPELINE_PHASE2_LABEL`` to select candidate roles; it
    does *not* filter rows.  Consequently each label must receive a physically
    separate SQLite.  Copying then deleting rows preserves every table/index
    in the tracked schema while keeping the related raw provenance consistent.
    """
    if label not in LABELS:
        raise ValueError(f"unsupported label: {label}")
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tracked_sqlite, output_sqlite)
    with sqlite3.connect(output_sqlite) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM masks WHERE label <> ?", (label,))
        db.execute("DELETE FROM tracks WHERE label <> ?", (label,))
        db.execute(
            "DELETE FROM raw_tracked_masks WHERE final_label IS NULL OR final_label <> ?",
            (label,),
        )
        db.execute(
            "DELETE FROM raw_tracks WHERE final_label IS NULL OR final_label <> ?",
            (label,),
        )
        db.commit()
        db.execute("VACUUM")
        counts = {
            table: int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in ("masks", "tracks", "raw_tracked_masks", "raw_tracks")
        }
        distinct = {
            "masks": [row[0] for row in db.execute("SELECT DISTINCT label FROM masks")],
            "tracks": [
                row[0] for row in db.execute("SELECT DISTINCT label FROM tracks")
            ],
            "raw_tracked_masks": [
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT final_label FROM raw_tracked_masks"
                )
            ],
            "raw_tracks": [
                row[0]
                for row in db.execute("SELECT DISTINCT final_label FROM raw_tracks")
            ],
        }
        integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
    if counts["masks"] <= 0 or counts["tracks"] <= 0:
        raise RuntimeError(f"empty label split for {label}: {counts}")
    if any(values != [label] for values in distinct.values()):
        raise RuntimeError(f"label split contamination for {label}: {distinct}")
    return {
        "label": label,
        "output": str(output_sqlite),
        "counts": counts,
        "distinct_labels": distinct,
        "integrity_check": integrity,
        "schema": _schema_fingerprint(output_sqlite),
    }


def _prepare_polygon_input(
    tracked_sqlite: Path,
    arm_root: Path,
    *,
    width: int,
    height: int,
    input_video: Path | None,
) -> tuple[Path, dict[str, object]]:
    preparation = arm_root / "polygon_preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    border_path = preparation / "border_expanded.sqlite"
    endpoint_path = preparation / "endpoint_extended.sqlite"
    _, border = apply_border_expansion(
        tracked_sqlite,
        border_path,
        width=width,
        height=height,
    )
    _, endpoint = apply_endpoint_extension(
        border_path,
        endpoint_path,
        video=input_video,
        extend_frames=5,
        motion_frames=10,
        max_speed_px=1000.0,
    )
    return endpoint_path, {
        "input": str(tracked_sqlite),
        "border_output": str(border_path),
        "endpoint_output": str(endpoint_path),
        "border": border,
        "endpoint": endpoint,
        "width": width,
        "height": height,
        "video_metadata_source": None if input_video is None else str(input_video),
    }


def _write_phase2_source_shim(root: Path, prepared_by_label: dict[str, Path]) -> Path:
    """Build the minimal read-only source manifest expected by run_phase2.

    ``production_candidate_polygon14.run`` discovers one prepared polygon
    input per class through the historical classwise manifest.  Phase 2 does
    not filter rows by label, so the three groups must point to three distinct
    label-specific SQLite files.
    """
    work = root / "interval_10/production_raw/work/04_classwise_postprocess"
    work.mkdir(parents=True, exist_ok=True)
    groups = []
    for index, label in enumerate(LABELS):
        prepared = prepared_by_label[label].resolve()
        group_root = work / "groups" / f"{index:02d}_{label}"
        group_root.mkdir(parents=True, exist_ok=True)
        pipeline_manifest = group_root / "pipeline_manifest.json"
        pipeline_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stages": [
                        {
                            "id": "polygon_optimization",
                            "metadata": {"optimizer": {"input_sqlite": str(prepared)}},
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        groups.append(
            {
                "id": f"{index:02d}_{label}",
                "labels": [label],
                "pipeline_manifest": str(pipeline_manifest.resolve()),
            }
        )
    classwise_manifest = work / "classwise_manifest.json"
    classwise_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _run_polygon14(
    source_root: Path,
    output_root: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "experimental.production_candidate_polygon14.run",
        "--source-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--intervals",
        str(TARGET_INTERVAL),
        "--labels",
        ",".join(LABELS),
        "--label-workers",
        str(max(1, int(args.label_workers))),
        "--num-workers",
        str(max(1, int(args.num_workers))),
        "--pair-vote-threads",
        str(max(1, int(args.pair_vote_threads))),
        "--max-tracks",
        str(max(0, int(args.max_tracks))),
    ]
    if args.force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(POSTPROCESS), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"polygon14 downstream failed with exit {process.returncode}; "
            f"see {log_path}"
        )
    manifest_path = output_root / "production_candidate_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_schemas: dict[str, dict[str, object]] = {}
    interval_root = output_root / f"interval_{TARGET_INTERVAL}" / PROFILE
    for label in LABELS:
        prediction = interval_root / label / "runtime/pred/predictions.sqlite"
        prediction_schemas[label] = _schema_fingerprint(prediction)
    return {
        "command": command,
        "elapsed_seconds": elapsed,
        "manifest": str(manifest_path),
        "manifest_payload": payload,
        "prediction_schemas": prediction_schemas,
        "log": str(log_path),
    }


def _assert_identical_schemas(
    runs: dict[str, dict[str, object]], key: str
) -> str | None:
    fingerprints = {
        name: str(run[key]["sha256"]) for name, run in runs.items() if key in run
    }
    if not fingerprints:
        return None
    unique = set(fingerprints.values())
    if len(unique) != 1:
        raise RuntimeError(f"schema mismatch across policies for {key}: {fingerprints}")
    return next(iter(unique))


def main() -> int:
    args = parse_args()
    scored_jsonl = args.scored_jsonl.expanduser().resolve()
    cuts_json = args.cuts_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not scored_jsonl.is_file():
        raise FileNotFoundError(scored_jsonl)
    if not cuts_json.is_file():
        raise FileNotFoundError(cuts_json)
    input_video = (
        None if args.input_video is None else args.input_video.expanduser().resolve()
    )
    if input_video is not None and not input_video.is_file():
        raise FileNotFoundError(input_video)
    CutList.read(cuts_json)  # validate before any expensive work
    width, height = _source_dimensions(scored_jsonl)
    if args.remove_short_tracks_max_frames < 0:
        raise ValueError("remove-short-tracks-max-frames must be >= 0")
    selected = [value.strip() for value in args.policies.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(POLICIES))
    if not selected or unknown:
        raise ValueError(
            f"policies must be a non-empty subset of {POLICIES}: {unknown}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("policies must not contain duplicates")

    output_root.mkdir(parents=True, exist_ok=True)
    input_contract = {
        "source_name": str(args.source_name),
        "scored_jsonl": str(scored_jsonl),
        "scored_jsonl_sha256": _sha256(scored_jsonl),
        "cuts_json": str(cuts_json),
        "cuts_json_sha256": _sha256(cuts_json),
    }
    runs: dict[str, dict[str, object]] = {}
    for policy_name in selected:
        arm_root = output_root / policy_name
        arm_manifest = arm_root / "arm_manifest.json"
        if arm_manifest.exists() and not args.force:
            raise FileExistsError(
                f"arm already exists: {arm_manifest}; use --force or a new output root"
            )
        arm_root.mkdir(parents=True, exist_ok=True)
        nms_jsonl = arm_root / "nms.jsonl"
        tracked_sqlite = arm_root / "tracked.sqlite"
        nms = _run_nms(scored_jsonl, nms_jsonl, policy_name)
        tracking = build_tracked_sqlite(
            nms_jsonl,
            tracked_sqlite,
            cuts_json,
            remove_short_tracks_max_frames=int(args.remove_short_tracks_max_frames),
        )
        run: dict[str, object] = {
            "policy": policy_name,
            "nms_jsonl": str(nms_jsonl),
            "nms": nms,
            "tracked_sqlite": str(tracked_sqlite),
            "tracking": tracking,
            "tracked_schema": _schema_fingerprint(tracked_sqlite),
        }
        if not args.skip_polygon:
            split_root = arm_root / "classwise_tracked"
            prepared_by_label: dict[str, Path] = {}
            classwise: dict[str, object] = {}
            for index, label in enumerate(LABELS):
                class_root = arm_root / "polygon_preparation" / f"{index:02d}_{label}"
                split_sqlite = split_root / f"{index:02d}_{label}.sqlite"
                split = _split_tracked_sqlite_by_label(
                    tracked_sqlite, split_sqlite, label
                )
                prepared_sqlite, preparation = _prepare_polygon_input(
                    split_sqlite,
                    class_root,
                    width=width,
                    height=height,
                    input_video=input_video,
                )
                prepared_by_label[label] = prepared_sqlite
                classwise[label] = {
                    "split": split,
                    "preparation": preparation,
                    "prepared_schema": _schema_fingerprint(prepared_sqlite),
                }
            run["polygon_preparation"] = {
                "mode": "label_specific",
                "classes": classwise,
            }
            source_root = _write_phase2_source_shim(
                arm_root / "phase2_source", prepared_by_label
            )
            run["polygon14"] = _run_polygon14(
                source_root,
                arm_root / "polygon14",
                arm_root / "polygon14.log",
                args,
            )
        arm_manifest.write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runs[policy_name] = run

    tracked_schema_sha256 = _assert_identical_schemas(runs, "tracked_schema")
    prediction_schema_comparison: dict[str, object] = {}
    if not args.skip_polygon:
        for label in LABELS:
            fingerprints = {
                policy: str(run["polygon14"]["prediction_schemas"][label]["sha256"])
                for policy, run in runs.items()
            }
            if len(set(fingerprints.values())) != 1:
                raise RuntimeError(
                    f"prediction schema mismatch for {label}: {fingerprints}"
                )
            prediction_schema_comparison[label] = {
                "identical": True,
                "sha256": next(iter(fingerprints.values())),
            }

    manifest = {
        "schema_version": 1,
        "experimental": True,
        "privacy": "Mask geometry only; this harness does not open video pixels.",
        "input": input_contract,
        "controlled_variable": "NMS, hole fill, and island handling only",
        "fixed_downstream": {
            "tracking_remove_short_tracks_max_frames": int(
                args.remove_short_tracks_max_frames
            ),
            "polygon_profile": PROFILE,
            "polygon_preparation": (
                "border expansion then endpoint extension (5 frames, "
                "10-frame motion fit)"
            ),
            "minimum_recall": RECALL_FLOOR,
            "target_interval": TARGET_INTERVAL,
            "pair_vote": "per-key IoU under exact Recall",
            "pair_vote_sweeps": PAIR_VOTE_SWEEPS,
            "software_facing_sqlite_schema_changed": False,
        },
        "policies": selected,
        "runs": runs,
        "schema_validation": {
            "tracked_sqlite_identical": True,
            "tracked_sqlite_sha256": tracked_schema_sha256,
            "prediction_sqlite": prediction_schema_comparison,
        },
        "limitations": [
            "The source model family is recorded from --source-name but cannot be proven from canonical scored JSONL alone.",
            "The harness emits tracked and per-class optimizer SQLite artifacts; it does not build the final unified software handoff SQLite.",
            "Post-NMS quality is evaluated against each arm's own tracked source masks, so cross-arm IoU is not a ground-truth accuracy comparison.",
        ],
    }
    manifest_path = output_root / "ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "policies": selected}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
