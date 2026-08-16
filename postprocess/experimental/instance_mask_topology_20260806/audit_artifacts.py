#!/usr/bin/env python3
"""Audit completeness, keys, schema and topology coverage for matrix outputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _published(item: dict[str, object]) -> tuple[Path, dict[str, object] | None]:
    manifest_path = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        return Path(str(item["inference_sqlite"])), None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = manifest.get("artifacts", {}).get("result_sqlite", item["inference_sqlite"])
    return Path(str(result)), manifest


def audit(matrix_path: Path, topology_path: Path) -> dict[str, object]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    topology = sqlite3.connect(f"file:{topology_path.resolve()}?mode=ro", uri=True)
    outputs: list[dict[str, object]] = []
    signatures: set[str] = set()
    for item in matrix:
        run_key = str(item["run_key"])
        path, manifest = _published(item)
        record: dict[str, object] = {
            "run_key": run_key,
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "manifest_complete": bool(manifest and manifest.get("status") == "complete"),
            "errors": [],
        }
        errors: list[str] = record["errors"]  # type: ignore[assignment]
        if not path.is_file():
            errors.append("missing_result_sqlite")
            outputs.append(record)
            continue
        if not record["manifest_complete"]:
            errors.append("manifest_not_complete")
        if manifest:
            signature = manifest.get("validation", {}).get("result_sqlite", {}).get("schema_signature")
            if signature:
                signatures.add(str(signature))
                record["schema_signature"] = str(signature)
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        record["quick_check"] = quick
        if quick != "ok":
            errors.append(f"quick_check:{quick}")
        frames, minimum, maximum, distinct_frames = connection.execute(
            "SELECT COUNT(*),MIN(frame_index),MAX(frame_index),COUNT(DISTINCT frame_index) FROM frames"
        ).fetchone()
        detections = int(connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0])
        segmentations = int(connection.execute("SELECT COUNT(*) FROM segmentations").fetchone()[0])
        polygons = int(connection.execute("SELECT COUNT(*) FROM segmentation_polygons").fetchone()[0])
        points = int(connection.execute("SELECT COUNT(*) FROM segmentation_points").fetchone()[0])
        record.update(
            {
                "frames": int(frames),
                "first_frame": None if minimum is None else int(minimum),
                "last_frame": None if maximum is None else int(maximum),
                "detections": detections,
                "segmentations": segmentations,
                "polygons": polygons,
                "points": points,
                "size_bytes": path.stat().st_size,
            }
        )
        if frames and (int(minimum) != 0 or int(maximum) != int(frames) - 1 or int(distinct_frames) != int(frames)):
            errors.append("non_contiguous_frames")
        if detections != segmentations:
            errors.append("detections_segmentations_count_mismatch")
        missing_segmentation = int(
            connection.execute(
                """SELECT COUNT(*) FROM detections d
                   LEFT JOIN segmentations s ON s.detection_id=d.id
                   WHERE s.detection_id IS NULL"""
            ).fetchone()[0]
        )
        empty_polygons = int(
            connection.execute(
                """SELECT COUNT(*) FROM segmentation_polygons p
                   LEFT JOIN segmentation_points pt ON pt.polygon_id=p.id
                   WHERE pt.polygon_id IS NULL"""
            ).fetchone()[0]
        )
        if missing_segmentation:
            errors.append(f"detections_without_segmentation:{missing_segmentation}")
        if empty_polygons:
            errors.append(f"polygons_without_points:{empty_polygons}")
        audit_row = topology.execute(
            "SELECT frame_count,detection_count FROM audit_runs WHERE run_key=?",
            (run_key,),
        ).fetchone()
        if audit_row is None:
            errors.append("missing_topology_audit")
        else:
            if tuple(map(int, audit_row)) != (int(frames), detections):
                errors.append("topology_audit_count_mismatch")
            topology_masks = int(
                topology.execute(
                    "SELECT COUNT(*) FROM mask_topology WHERE run_key=?", (run_key,)
                ).fetchone()[0]
            )
            topology_contours = int(
                topology.execute(
                    "SELECT COUNT(*) FROM contour_topology WHERE run_key=?", (run_key,)
                ).fetchone()[0]
            )
            record["topology_masks"] = topology_masks
            record["topology_contours"] = topology_contours
            if topology_masks != detections:
                errors.append("topology_mask_coverage_mismatch")
            if topology_contours != polygons:
                errors.append("topology_contour_coverage_mismatch")
            invalid_topology = int(
                topology.execute(
                    """SELECT COUNT(*) FROM mask_topology
                       WHERE run_key=? AND (
                         foreground_component_count<1 OR hole_count<0 OR
                         second_to_largest_ratio<0 OR second_to_largest_ratio>1 OR
                         net_foreground_area<=0
                       )""",
                    (run_key,),
                ).fetchone()[0]
            )
            if invalid_topology:
                errors.append(f"invalid_topology_rows:{invalid_topology}")
        connection.close()
        outputs.append(record)
    topology.close()
    return {
        "matrix_runs": len(matrix),
        "complete_outputs": sum(not row["errors"] for row in outputs),
        "schema_signatures": sorted(signatures),
        "schema_signature_consistent": len(signatures) == 1,
        "outputs": outputs,
        "success": all(not row["errors"] for row in outputs) and len(signatures) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.matrix, args.topology)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("matrix_runs", "complete_outputs", "schema_signatures", "success")}, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
