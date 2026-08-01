#!/usr/bin/env python3
"""Validate real GUI matrix outputs beyond process exit status.

The GUI harness proves that Electron can drive the workflow.  This companion
checks the downstream contract, SQLite integrity/domain separation, and video
packet continuity so a green GUI job cannot hide an empty or overlong overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
FFPROBE = os.environ.get(
    "MASK_STUDIO_FFPROBE",
    str(Path.home() / ".local/share/video-mask-runtime/tools/ffmpeg/bin/ffprobe"),
)
sys.path.insert(0, str(REPOSITORY))

from orchestration.contracts import (  # noqa: E402
    PUBLIC_RESULT_SCHEMA_SIGNATURE,
    public_result_schema_signature,
    validate_result_sqlite,
)


def ffprobe(path: Path, *, packets: bool = False) -> dict[str, Any]:
    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration,nb_read_packets",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    if packets:
        packet_result = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts,dts",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        packet_timestamps = []
        for line in packet_result.stdout.splitlines():
            values = line.split(",")
            if len(values) != 2 or "N/A" in values:
                continue
            packet_timestamps.append((int(values[0]), int(values[1])))
        pts = [value[0] for value in packet_timestamps]
        dts = [value[1] for value in packet_timestamps]
        stream["pts_count"] = len(pts)
        stream["pts_non_increasing"] = sum(
            right <= left for left, right in zip(pts, pts[1:])
        )
        stream["dts_non_increasing"] = sum(
            right <= left for left, right in zip(dts, dts[1:])
        )
        deltas = [right - left for left, right in zip(pts, pts[1:])]
        if deltas:
            stream["pts_delta_min"] = min(deltas)
            stream["pts_delta_max"] = max(deltas)
    return stream


def scalar(connection: sqlite3.Connection, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0])


def validate_sqlite(path: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
        roles = {
            str(role): str(model)
            for role, model in connection.execute(
                "SELECT role, model_id FROM model_executions"
            )
        }
        require_segmentation = "instance_segmentation" in roles
        require_faces = "face_detection" in roles
        contract = validate_result_sqlite(
            path,
            require_segmentation=require_segmentation,
            require_faces=require_faces,
            expected_face_model=roles.get("face_detection"),
        )
        signature = public_result_schema_signature(connection)
        counts = {
            table: scalar(connection, f'SELECT COUNT(*) FROM "{table}"')
            for table in [
                "frames",
                "detections",
                "segmentations",
                "face_observations",
                "face_keypoints",
                "face_masks",
                "tracking_assignments",
                "face_tracks",
                "face_tracking_assignments",
                "cuts",
                "mask_track_segments",
                "mask_keyframes",
                "keyframe_components",
                "keyframe_ellipses",
                "keyframe_rectangles",
                "keyframe_polygon_points",
            ]
        }
        counts["face_detection_rows"] = scalar(
            connection,
            """
            SELECT COUNT(*) FROM detections AS d
            JOIN model_executions AS m ON m.id=d.model_execution_id
            WHERE m.role='face_detection'
            """,
        )
        bad_segmentation_domain = scalar(
            connection,
            """
            SELECT COUNT(*) FROM segmentations AS s
            JOIN detections AS d ON d.id=s.detection_id
            JOIN model_executions AS m ON m.id=d.model_execution_id
            WHERE m.role != 'instance_segmentation'
            """,
        )
        bad_face_domain = scalar(
            connection,
            """
            SELECT COUNT(*) FROM face_observations AS f
            JOIN detections AS d ON d.id=f.anchor_detection_id
            JOIN model_executions AS m ON m.id=d.model_execution_id
            WHERE m.role != 'face_detection'
            """,
        )
        bad_scores = scalar(
            connection,
            "SELECT COUNT(*) FROM detections WHERE score < 0 OR score > 1",
        )
        frame_bounds = connection.execute(
            "SELECT COUNT(*), MIN(frame_index), MAX(frame_index) FROM frames"
        ).fetchone()
        bad_cuts = scalar(
            connection,
            """
            SELECT COUNT(*) FROM cuts
            WHERE frame < (SELECT MIN(frame_index) FROM frames)
               OR frame > (SELECT MAX(frame_index) FROM frames)
            """,
        )
        capabilities = {
            str(name): {"available": bool(available), "rows": int(rows)}
            for name, available, rows in connection.execute(
                "SELECT name, available, row_count FROM result_capabilities"
            )
        }
    if integrity != "ok":
        issues.append(f"integrity_check={integrity}")
    if foreign_keys:
        issues.append(f"foreign_key_check={len(foreign_keys)} violations")
    if signature != PUBLIC_RESULT_SCHEMA_SIGNATURE:
        issues.append(f"schema_signature={signature}")
    if bad_segmentation_domain:
        issues.append(f"{bad_segmentation_domain} segmentation rows use a face model")
    if bad_face_domain:
        issues.append(f"{bad_face_domain} face rows use a segmentation model")
    if bad_scores:
        issues.append(f"{bad_scores} detection scores are outside [0,1]")
    if bad_cuts:
        issues.append(f"{bad_cuts} cuts are outside processed frames")
    return (
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "roles": roles,
            "contract": contract,
            "schema_signature": signature,
            "frame_bounds": list(frame_bounds),
            "counts": counts,
            "capabilities": capabilities,
        },
        issues,
    )


def validate_output(output: Path) -> dict[str, Any]:
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = Path(manifest["artifacts"]["result_sqlite"])
    sqlite_report, issues = validate_sqlite(result_path)
    expected_frames = int(sqlite_report["counts"]["frames"])
    overlays = []
    for video in sorted((output / "03_overlay").glob("*.mp4")):
        overlay_manifest_path = video.with_suffix(".json")
        overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
        summary = overlay_manifest.get("summary", overlay_manifest)
        probe = ffprobe(video, packets=True)
        packet_count = int(probe.get("nb_read_packets") or 0)
        manifest_frames = int(
            summary.get("frames") or summary.get("frames_written") or 0
        )
        workers = summary.get("workers_detail") or []
        if workers:
            mask_rows = sum(
                int(worker.get("renderer_summary", {}).get("mask_rows_drawn") or 0)
                for worker in workers
            )
            face_rows = sum(
                int(worker.get("renderer_summary", {}).get("face_rows_drawn") or 0)
                for worker in workers
            )
        else:
            mask_rows = int(summary.get("masks_drawn") or 0)
            face_rows = int(summary.get("faces_drawn") or 0)
        preset = video.stem
        if packet_count != expected_frames:
            issues.append(
                f"{video.name}: packets={packet_count}, SQLite frames={expected_frames}"
            )
        if manifest_frames and manifest_frames != packet_count:
            issues.append(
                f"{video.name}: manifest frames={manifest_frames}, packets={packet_count}"
            )
        if int(probe.get("dts_non_increasing") or 0):
            issues.append(
                f"{video.name}: {probe['dts_non_increasing']} non-increasing packet DTS"
            )
        if "face" in preset or "combined" in preset:
            face_rows_available = sqlite_report["counts"]["face_detection_rows"] > 0
            if face_rows_available and face_rows == 0:
                issues.append(f"{video.name}: face data exists but no face rows were drawn")
        if "genital" in preset or "combined" in preset:
            masks_available = sqlite_report["counts"]["mask_keyframes"] > 0
            if masks_available and mask_rows == 0:
                issues.append(f"{video.name}: mask data exists but no mask rows were drawn")
        overlays.append(
            {
                "path": str(video),
                "size_bytes": video.stat().st_size,
                "probe": probe,
                "manifest_frames": manifest_frames,
                "aggregate_fps": summary.get("aggregate_fps")
                or (
                    manifest_frames / float(summary["elapsed_seconds"])
                    if summary.get("elapsed_seconds")
                    else None
                ),
                "mask_rows_drawn": mask_rows,
                "face_rows_drawn": face_rows,
            }
        )
    return {
        "output": str(output),
        "manifest_status": manifest.get("status"),
        "stage_seconds": {
            stage["name"]: stage.get("elapsed_seconds")
            for stage in manifest.get("stages", [])
        },
        "sqlite": sqlite_report,
        "overlays": overlays,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    outputs: list[Path] = []
    for case in source["cases"]:
        for item in case.get("queue", []):
            if item.get("status") == "done" and item.get("outputDir"):
                outputs.append(Path(item["outputDir"]))
    report = {
        "source_report": str(args.report),
        "outputs": [validate_output(output) for output in outputs],
    }
    report["issue_count"] = sum(len(output["issues"]) for output in report["outputs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
