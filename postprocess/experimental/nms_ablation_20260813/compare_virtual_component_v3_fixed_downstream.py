#!/usr/bin/env python3
"""Compare legacy, mask-IoU v2, and virtual-component v3 frozen downstream."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_fixed_downstream_kpi as fixed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-arm", type=Path, required=True)
    parser.add_argument("--v2-arm", type=Path, required=True)
    parser.add_argument("--v3-arm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    roots = {
        "legacy_production": args.legacy_arm,
        "component_mask_v2": args.v2_arm,
        "virtual_component_v3": args.v3_arm,
    }
    arms = {name: fixed._arm(name, root) for name, root in roots.items()}
    software = {
        name: fixed._schema(root / "software_sqlite/12月KPI動画.sqlite")
        for name, root in roots.items()
    }
    legacy = arms["legacy_production"]["aggregate"]
    deltas = {}
    for name in ("component_mask_v2", "virtual_component_v3"):
        value = arms[name]["aggregate"]
        deltas[f"{name}_minus_legacy"] = {
            key: float(value["exact"][key]) - float(legacy["exact"][key])
            for key in ("recall_min", "iou_mean", "iou_q01", "iou_q05", "iou_min")
        } | {
            "keyframes": int(value["keyframes"]) - int(legacy["keyframes"]),
            "effective_interval": float(value["actual_mean_interval"])
            - float(legacy["actual_mean_interval"]),
            "tracked_rows_after_prune": int(arms[name]["tracking"]["rows_after_prune"])
            - int(arms["legacy_production"]["tracking"]["rows_after_prune"]),
            "tracks_after_prune": int(arms[name]["tracking"]["tracks_after_prune"])
            - int(arms["legacy_production"]["tracking"]["tracks_after_prune"]),
        }
    schema_hashes = {value["schema_sha256"] for value in software.values()}
    payload = {
        "schema_version": 1,
        "privacy": "Mask/SQLite geometry only; video frames were not decoded or viewed.",
        "controlled_comparison": (
            "Only NMS/hole/island policy differs. Tracking, border/endpoint preparation, "
            "polygon14, minimum Recall 0.97, target interval 6, and pair-vote 2 sweeps are fixed."
        ),
        "arms": arms,
        "deltas": deltas,
        "software_sqlite": software,
        "validation": {
            "software_schema_identical": len(schema_hashes) == 1,
            "software_integrity_all_ok": all(
                value["integrity_check"] == "ok" and value["foreign_key_errors"] == 0
                for value in software.values()
            ),
            "tracked_schema_identical": len(
                {value["tracked_sqlite"]["schema_sha256"] for value in arms.values()}
            )
            == 1,
            "prediction_schema_identical": all(
                len(
                    {
                        arm["classes"][label]["prediction_sqlite"]["schema_sha256"]
                        for arm in arms.values()
                    }
                )
                == 1
                for label in fixed.LABELS
            ),
            "recall_violations_all_zero": all(
                arm["aggregate"]["exact"]["recall_violations_below_0p97"] == 0
                for arm in arms.values()
            ),
            "topology_contracts_all_pass": all(
                arm["aggregate"]["topology_gate"]["all_contracts_satisfied"]
                for arm in arms.values()
            ),
        },
        "limitations": [
            "IoU and Recall use each arm's own tracked reference and measure downstream fidelity, not semantic ground truth.",
            "Natural V3 data contained no island-island suppression event after <=1% cleanup; that branch is unit-tested but not empirically exercised here.",
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "arm",
        "scope",
        "rows",
        "keyframes",
        "effective_interval",
        "recall_min",
        "recall_violations",
        "iou_mean",
        "iou_q01",
        "iou_q05",
        "iou_min",
        "wall_seconds",
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, arm in arms.items():
            for scope, value in [*arm["classes"].items(), ("ALL", arm["aggregate"])]:
                exact = value["exact"]
                processing = value["processing"]
                writer.writerow(
                    {
                        "arm": name,
                        "scope": scope,
                        "rows": exact["rows"],
                        "keyframes": value["keyframes"],
                        "effective_interval": value["actual_mean_interval"],
                        "recall_min": exact["recall_min"],
                        "recall_violations": exact["recall_violations_below_0p97"],
                        "iou_mean": exact["iou_mean"],
                        "iou_q01": exact["iou_q01"],
                        "iou_q05": exact["iou_q05"],
                        "iou_min": exact["iou_min"],
                        "wall_seconds": processing.get(
                            "wall_seconds",
                            processing.get("polygon_profile_parallel_wall_seconds"),
                        ),
                    }
                )
    print(
        json.dumps({"comparison": str(output / "comparison.json")}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
