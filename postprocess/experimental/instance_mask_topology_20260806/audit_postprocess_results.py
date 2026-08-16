#!/usr/bin/env python3
"""Validate the representative stable result SQLite files used in the audit."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def audit(summary_paths: list[Path]) -> dict[str, object]:
    outputs: list[dict[str, object]] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = Path(str(summary["result_sqlite"]))
        manifest_path = Path(str(summary["postprocess_manifest"]))
        errors: list[str] = []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            errors.append("postprocess_manifest_not_complete")
        connection = sqlite3.connect(f"file:{result.resolve()}?mode=ro", uri=True)
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            errors.append(f"quick_check:{quick_check}")
        foreign_key_errors = list(connection.execute("PRAGMA foreign_key_check"))
        if foreign_key_errors:
            errors.append(f"foreign_key_errors:{len(foreign_key_errors)}")
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "tracks",
                "mask_track_segments",
                "mask_keyframes",
                "keyframe_components",
                "keyframe_polygon_rings",
                "keyframe_polygon_points",
                "tracking_assignments",
            )
        }
        orphan_keyframes = int(
            connection.execute(
                """SELECT COUNT(*) FROM mask_keyframes k
                   LEFT JOIN mask_track_segments s ON s.id=k.segment_id
                   WHERE s.id IS NULL"""
            ).fetchone()[0]
        )
        orphan_components = int(
            connection.execute(
                """SELECT COUNT(*) FROM keyframe_components c
                   LEFT JOIN mask_keyframes k ON k.id=c.keyframe_id
                   WHERE k.id IS NULL"""
            ).fetchone()[0]
        )
        slot_overflow = int(
            connection.execute(
                """SELECT COUNT(*) FROM keyframe_components c
                   JOIN mask_keyframes k ON k.id=c.keyframe_id
                   JOIN mask_track_segments s ON s.id=k.segment_id
                   WHERE c.slot_index>=s.component_count"""
            ).fetchone()[0]
        )
        if orphan_keyframes:
            errors.append(f"orphan_keyframes:{orphan_keyframes}")
        if orphan_components:
            errors.append(f"orphan_components:{orphan_components}")
        if slot_overflow:
            errors.append(f"slot_index_overflow:{slot_overflow}")
        ring_roles = dict(
            (str(role), int(count))
            for role, count in connection.execute(
                "SELECT ring_role,COUNT(*) FROM keyframe_polygon_rings GROUP BY ring_role"
            )
        )
        connection.close()
        outputs.append(
            {
                "run_key": summary["run_key"],
                "result_sqlite": str(result.resolve()),
                "size_bytes": result.stat().st_size,
                "manifest_complete": bool(manifest.get("complete")),
                "quick_check": quick_check,
                "foreign_key_errors": len(foreign_key_errors),
                "counts": counts,
                "ring_roles": ring_roles,
                "errors": errors,
            }
        )
    return {
        "outputs": outputs,
        "success": all(not item["errors"] for item in outputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
