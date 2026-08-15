#!/usr/bin/env python3
"""Compare two completed fixed-downstream NMS arms.

This is a read-only evaluator.  It never opens source video pixels and it does
not rewrite either arm.  Geometry quality is read from the exact per-frame
CSV emitted by ``polygon14_keyframe_v1``.  SQLite validation is structural:
the schema DDL is hashed and ``integrity_check`` / ``foreign_key_check`` are
run against every tracked and prediction database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable


LABELS = ("女性器", "男性器", "結合部分")
PROFILE_REL = Path("polygon14/interval_6/polygon14_keyframe_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-arm", type=Path, required=True)
    parser.add_argument("--candidate-arm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exact_csv(path: Path, recall_floor: float = 0.97) -> dict[str, Any]:
    recalls: list[float] = []
    ious: list[float] = []
    keyframes = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recalls.append(float(row["recall"]))
            ious.append(float(row["iou"]))
            keyframes += int(row["has_keyframe"])
    # Runtime metrics use a tiny numerical tolerance around the hard floor.
    violations = sum(value < recall_floor - 1e-9 for value in recalls)
    return {
        "rows": len(ious),
        "keyframe_rows": keyframes,
        "recall_min": min(recalls),
        "recall_violations_below_0p97": violations,
        "iou_mean": sum(ious) / len(ious),
        "iou_q01": _quantile(ious, 0.01),
        "iou_q05": _quantile(ious, 0.05),
        "iou_min": min(ious),
    }


def _schema(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as db:
        objects = [
            {"type": kind, "name": name, "sql": sql or ""}
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
        "path": str(resolved),
        "schema_sha256": hashlib.sha256(canonical).hexdigest(),
        "integrity_check": integrity,
        "foreign_key_errors": foreign_key_errors,
        "object_count": len(objects),
    }


def _sum_dicts(items: Iterable[dict[str, Any]], key: str) -> int | float:
    return sum(item.get(key, 0) for item in items)


def _arm(name: str, root: Path) -> dict[str, Any]:
    root = root.resolve()
    arm_manifest = _load(root / "arm_manifest.json")
    candidate_manifest = _load(root / "polygon14/production_candidate_manifest.json")
    phase2 = _load(root / "polygon14/interval_6/phase2_matrix.json")
    classes: dict[str, dict[str, Any]] = {}
    aggregate_recalls: list[float] = []
    aggregate_ious: list[float] = []

    for label in LABELS:
        label_root = root / PROFILE_REL / label
        metrics = _load(label_root / "metrics.json")
        audit = _load(label_root / "runtime/phase2_audit.json")
        exact_path = label_root / "runtime/exact/keyframe_exact_metrics.csv"
        exact = _exact_csv(exact_path)
        with exact_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                aggregate_recalls.append(float(row["recall"]))
                aggregate_ious.append(float(row["iou"]))
        topology = audit["topology_guard"]
        classes[label] = {
            "processing": {
                "wall_seconds": metrics["wall_seconds"],
                "optimizer_seconds": metrics["optimizer_seconds"],
                "pair_vote_seconds": metrics["pair_vote_seconds"],
            },
            "keyframes": metrics["keyframes"],
            "actual_mean_interval": metrics["actual_mean_interval"],
            "exact": exact,
            "runtime_reported_exact": {
                "recall_min": metrics["recall_min"],
                "recall_violations": metrics["recall_violations"],
                "iou_mean": metrics["iou_mean"],
                "iou_q01": metrics["iou_q01"],
                "iou_q05": metrics["iou_q05"],
                "iou_min": metrics["iou_min"],
            },
            "pair_vote": {
                "enabled": audit["pair_vote_acceleration"]["enabled"],
                # A rejected complete path is the explicit DP-shape fallback.
                "dp_shape_fallbacks": topology["pair_vote_paths_rejected"],
                "paths_checked": topology["pair_vote_paths_checked"],
                "local_trials_checked": topology["pair_vote_local_trials_checked"],
                "local_trials_rejected": topology[
                    "pair_vote_local_trials_rejected"
                ],
            },
            "topology_gate": {
                "dp_edges_checked": topology["dp_selected_edges_checked"],
                "dp_invalid_edges": topology["dp_invalid_edges"],
                "dp_inserted_keys": topology["dp_inserted_keys"],
                "pair_vote_paths_checked": topology["pair_vote_paths_checked"],
                "pair_vote_paths_rejected": topology[
                    "pair_vote_paths_rejected"
                ],
                "pair_vote_trials_checked": topology[
                    "pair_vote_local_trials_checked"
                ],
                "pair_vote_trials_rejected": topology[
                    "pair_vote_local_trials_rejected"
                ],
                "implementation_contract_satisfied": audit[
                    "implementation_contract_satisfied"
                ],
            },
            "prediction_sqlite": _schema(
                label_root / "runtime/pred/predictions.sqlite"
            ),
        }

    row = phase2["completed_profiles"][0]
    topology_rows = [value["topology_gate"] for value in classes.values()]
    pair_rows = [value["pair_vote"] for value in classes.values()]
    polygon_run = candidate_manifest["runs"][0]
    aggregate = {
        "processing": {
            "nms_seconds": arm_manifest["nms"]["elapsed_seconds"],
            "tracking_seconds": arm_manifest["tracking"]["elapsed_sec"],
            "polygon_subprocess_seconds": arm_manifest["polygon14"][
                "elapsed_seconds"
            ],
            "polygon_profile_parallel_wall_seconds": row["profile_wall_seconds"],
            "sum_class_wall_seconds": _sum_dicts(
                (value["processing"] for value in classes.values()), "wall_seconds"
            ),
            "candidate_manifest_run_wall_seconds": polygon_run["wall_seconds"],
        },
        "keyframes": sum(value["keyframes"] for value in classes.values()),
        "actual_mean_interval": row["actual_mean_interval"],
        "exact": {
            "rows": len(aggregate_ious),
            "recall_min": min(aggregate_recalls),
            "recall_violations_below_0p97": sum(
                value < 0.97 - 1e-9 for value in aggregate_recalls
            ),
            "iou_mean": sum(aggregate_ious) / len(aggregate_ious),
            "iou_q01": _quantile(aggregate_ious, 0.01),
            "iou_q05": _quantile(aggregate_ious, 0.05),
            "iou_min": min(aggregate_ious),
        },
        "pair_vote": {
            "dp_shape_fallbacks": _sum_dicts(pair_rows, "dp_shape_fallbacks"),
            "paths_checked": _sum_dicts(pair_rows, "paths_checked"),
            "local_trials_checked": _sum_dicts(pair_rows, "local_trials_checked"),
            "local_trials_rejected": _sum_dicts(
                pair_rows, "local_trials_rejected"
            ),
        },
        "topology_gate": {
            "dp_edges_checked": _sum_dicts(topology_rows, "dp_edges_checked"),
            "dp_invalid_edges": _sum_dicts(topology_rows, "dp_invalid_edges"),
            "dp_inserted_keys": _sum_dicts(topology_rows, "dp_inserted_keys"),
            "pair_vote_paths_rejected": _sum_dicts(
                topology_rows, "pair_vote_paths_rejected"
            ),
            "pair_vote_trials_rejected": _sum_dicts(
                topology_rows, "pair_vote_trials_rejected"
            ),
            "all_contracts_satisfied": all(
                value["implementation_contract_satisfied"] for value in topology_rows
            ),
        },
    }
    preparation = arm_manifest["polygon_preparation"]
    if preparation.get("mode") != "label_specific":
        raise RuntimeError(
            f"invalid combined Phase-2 source in {root}: "
            f"polygon_preparation.mode={preparation.get('mode')!r}"
        )
    split_counts = {
        label: int(preparation["classes"][label]["split"]["counts"]["masks"])
        for label in LABELS
    }
    input_validation = {
        "split_mask_rows": split_counts,
        "split_sum": sum(split_counts.values()),
        "combined_rows_after_prune": int(arm_manifest["tracking"]["rows_after_prune"]),
        "split_sum_matches_combined": (
            sum(split_counts.values())
            == int(arm_manifest["tracking"]["rows_after_prune"])
        ),
        "three_distinct_input_paths": len(
            {
                preparation["classes"][label]["split"]["output"]
                for label in LABELS
            }
        )
        == len(LABELS),
        "each_split_contains_only_its_label": all(
            all(
                labels == [label]
                for labels in preparation["classes"][label]["split"][
                    "distinct_labels"
                ].values()
            )
            for label in LABELS
        ),
        "split_schema_matches_combined": all(
            preparation["classes"][label]["split"]["schema"]["sha256"]
            == arm_manifest["tracked_schema"]["sha256"]
            for label in LABELS
        ),
    }
    if not all(
        input_validation[key]
        for key in (
            "split_sum_matches_combined",
            "three_distinct_input_paths",
            "each_split_contains_only_its_label",
            "split_schema_matches_combined",
        )
    ):
        raise RuntimeError(f"invalid classwise Phase-2 inputs: {input_validation}")
    return {
        "name": name,
        "root": str(root),
        "nms": arm_manifest["nms"],
        "tracking": arm_manifest["tracking"],
        "classes": classes,
        "aggregate": aggregate,
        "classwise_input_validation": input_validation,
        "tracked_sqlite": _schema(root / "tracked.sqlite"),
    }


def _delta(candidate: float, legacy: float) -> dict[str, float]:
    return {
        "absolute": candidate - legacy,
        "relative_percent": (candidate / legacy - 1.0) * 100.0 if legacy else math.nan,
    }


def _write_csv(path: Path, arms: dict[str, dict[str, Any]]) -> None:
    fields = [
        "arm",
        "scope",
        "wall_seconds",
        "keyframes",
        "effective_interval",
        "recall_min",
        "recall_violations",
        "iou_mean",
        "iou_q01",
        "iou_q05",
        "iou_min",
        "pair_vote_fallbacks",
        "topology_dp_invalid_edges",
        "topology_dp_inserted_keys",
        "topology_pair_vote_paths_rejected",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm_name, arm in arms.items():
            scopes = [(label, value) for label, value in arm["classes"].items()]
            scopes.append(("ALL", arm["aggregate"]))
            for scope, value in scopes:
                processing = value["processing"]
                exact = value["exact"]
                pair = value["pair_vote"]
                topology = value["topology_gate"]
                writer.writerow(
                    {
                        "arm": arm_name,
                        "scope": scope,
                        "wall_seconds": processing.get(
                            "wall_seconds",
                            processing.get("polygon_profile_parallel_wall_seconds"),
                        ),
                        "keyframes": value["keyframes"],
                        "effective_interval": value["actual_mean_interval"],
                        "recall_min": exact["recall_min"],
                        "recall_violations": exact[
                            "recall_violations_below_0p97"
                        ],
                        "iou_mean": exact["iou_mean"],
                        "iou_q01": exact["iou_q01"],
                        "iou_q05": exact["iou_q05"],
                        "iou_min": exact["iou_min"],
                        "pair_vote_fallbacks": pair["dp_shape_fallbacks"],
                        "topology_dp_invalid_edges": topology["dp_invalid_edges"],
                        "topology_dp_inserted_keys": topology["dp_inserted_keys"],
                        "topology_pair_vote_paths_rejected": topology[
                            "pair_vote_paths_rejected"
                        ],
                    }
                )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    for arm in (args.legacy_arm, args.candidate_arm):
        if not (arm / "arm_manifest.json").is_file():
            raise FileNotFoundError(f"incomplete arm: {arm}")
    output.mkdir(parents=True)
    arms = {
        "legacy_production": _arm("legacy_production", args.legacy_arm),
        "component_mask_v2": _arm("component_mask_v2", args.candidate_arm),
    }
    legacy = arms["legacy_production"]["aggregate"]
    candidate = arms["component_mask_v2"]["aggregate"]
    schema = {
        "tracked_equal": (
            arms["legacy_production"]["tracked_sqlite"]["schema_sha256"]
            == arms["component_mask_v2"]["tracked_sqlite"]["schema_sha256"]
        ),
        "prediction_equal_by_label": {
            label: (
                arms["legacy_production"]["classes"][label]["prediction_sqlite"][
                    "schema_sha256"
                ]
                == arms["component_mask_v2"]["classes"][label][
                    "prediction_sqlite"
                ]["schema_sha256"]
            )
            for label in LABELS
        },
        "all_integrity_ok": all(
            database["integrity_check"] == "ok"
            and database["foreign_key_errors"] == 0
            for arm in arms.values()
            for database in [
                arm["tracked_sqlite"],
                *(value["prediction_sqlite"] for value in arm["classes"].values()),
            ]
        ),
    }
    comparison = {
        "schema_version": 1,
        "privacy": (
            "Metadata/mask-geometry evaluation only. Source video frames were not "
            "decoded, viewed, or transmitted."
        ),
        "controlled_comparison": (
            "NMS/topology policy differs; tracking and polygon14_keyframe_v1 "
            "settings are frozen (Recall 0.97, target interval 6, pair-vote 2 sweeps)."
        ),
        "arms": arms,
        "delta_component_minus_legacy": {
            "polygon_parallel_wall_seconds": _delta(
                candidate["processing"]["polygon_profile_parallel_wall_seconds"],
                legacy["processing"]["polygon_profile_parallel_wall_seconds"],
            ),
            "keyframes": _delta(candidate["keyframes"], legacy["keyframes"]),
            "effective_interval": _delta(
                candidate["actual_mean_interval"], legacy["actual_mean_interval"]
            ),
            "recall_min": _delta(
                candidate["exact"]["recall_min"], legacy["exact"]["recall_min"]
            ),
            "iou_mean": _delta(
                candidate["exact"]["iou_mean"], legacy["exact"]["iou_mean"]
            ),
            "iou_q01": _delta(
                candidate["exact"]["iou_q01"], legacy["exact"]["iou_q01"]
            ),
            "iou_q05": _delta(
                candidate["exact"]["iou_q05"], legacy["exact"]["iou_q05"]
            ),
            "iou_min": _delta(
                candidate["exact"]["iou_min"], legacy["exact"]["iou_min"]
            ),
        },
        "schema_validation": schema,
        "interpretation_limits": [
            "Exact IoU/Recall are measured against each arm's own post-NMS tracked masks; they measure downstream fidelity, not semantic detector accuracy.",
            "The polygon optimizer runs three classes concurrently, so sum_class_wall_seconds is compute work and profile_parallel_wall_seconds is elapsed wall time.",
            "The harness emits intermediate tracked/per-class prediction SQLite files, not the final unified software handoff SQLite.",
            "Polygon preparation was not separately timed by the original harness and is excluded from stage timing fields.",
        ],
    }
    json_path = output / "comparison.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "comparison.csv", arms)
    readme = output / "README.md"
    lines = [
        "# KPI fixed-downstream comparison",
        "",
        "The experiment changes only NMS/topology cleanup. Both arms use the "
        "same tracking and `polygon14_keyframe_v1` contract (minimum Recall "
        "0.97, target interval 6, pair-vote 2 sweeps). No source video frame "
        "was decoded or viewed by this evaluator.",
        "",
        "| arm | scope | wall s | keys | interval | Recall min | violations | IoU mean | q01 | q05 | IoU min | pair-vote fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, arm in arms.items():
        scopes = [(label, value) for label, value in arm["classes"].items()]
        scopes.append(("ALL", arm["aggregate"]))
        for scope, value in scopes:
            processing = value["processing"]
            exact = value["exact"]
            wall = processing.get(
                "wall_seconds",
                processing.get("polygon_profile_parallel_wall_seconds"),
            )
            lines.append(
                f"| {arm_name} | {scope} | {wall:.3f} | {value['keyframes']} "
                f"| {value['actual_mean_interval']:.6f} "
                f"| {exact['recall_min']:.9f} "
                f"| {exact['recall_violations_below_0p97']} "
                f"| {exact['iou_mean']:.9f} | {exact['iou_q01']:.9f} "
                f"| {exact['iou_q05']:.9f} | {exact['iou_min']:.9f} "
                f"| {value['pair_vote']['dp_shape_fallbacks']} |"
            )
    lines.extend(
        [
            "",
            f"- Tracked schema equal: `{schema['tracked_equal']}`",
            f"- Prediction schemas equal by label: `{schema['prediction_equal_by_label']}`",
            f"- All SQLite integrity/FK checks pass: `{schema['all_integrity_ok']}`",
            "- `comparison.json` contains complete timings, topology-gate counters, schema hashes, and limitations.",
            "- Exact IoU/Recall use each arm's own post-NMS source; this is a downstream-fidelity check, not semantic GT accuracy.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparison": str(json_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
