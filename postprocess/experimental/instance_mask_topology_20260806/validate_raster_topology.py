#!/usr/bin/env python3
"""Cross-check contour nesting counts by rasterizing stored polygons locally."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _counts(polygons: list[tuple[int, np.ndarray]]) -> tuple[int, int]:
    if not polygons:
        return 0, 0
    all_points = np.concatenate([points for _, points in polygons], axis=0)
    x0 = int(np.floor(all_points[:, 0].min())) - 3
    y0 = int(np.floor(all_points[:, 1].min())) - 3
    x1 = int(np.ceil(all_points[:, 0].max())) + 3
    y1 = int(np.ceil(all_points[:, 1].max())) + 3
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    # Parents are painted before children, restoring even/odd nesting parity.
    for depth, points in sorted(polygons, key=lambda item: item[0]):
        shifted = np.rint(points - np.asarray([x0, y0])).astype(np.int32)
        cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 1 if depth % 2 == 0 else 0)
    foreground = int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)
    zero = (mask == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(zero, connectivity=8)
    border_labels = set(int(v) for v in np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    )))
    holes = sum(label not in border_labels for label in range(1, count))
    return foreground, holes


def validate(
    topology: Path,
    matrix_path: Path,
    output: Path,
    selected_run_keys: set[str] | None = None,
) -> dict[str, object]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_by_key = {str(item["run_key"]): item for item in matrix}
    destination = sqlite3.connect(f"file:{topology.resolve()}?mode=ro", uri=True)
    run_keys = [str(row[0]) for row in destination.execute("SELECT run_key FROM audit_runs")]
    if selected_run_keys:
        run_keys = [run_key for run_key in run_keys if run_key in selected_run_keys]
    summaries: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for run_key in run_keys:
        item = matrix_by_key[run_key]
        raw = Path(str(item["inference_sqlite"]))
        manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            raw = Path(str(payload.get("artifacts", {}).get("result_sqlite", raw)))
        connection = sqlite3.connect(":memory:")
        connection.execute(f"ATTACH DATABASE '{_q(topology)}' AS topo")
        connection.execute(f"ATTACH DATABASE '{_q(raw)}' AS raw")
        query = """
          SELECT m.detection_id, m.foreground_component_count, m.hole_count,
                 c.nesting_depth, p.polygon_index, pt.x, pt.y
          FROM topo.mask_topology m
          JOIN topo.contour_topology c
            ON c.run_key=m.run_key AND c.detection_id=m.detection_id
          JOIN raw.segmentation_polygons p
            ON p.detection_id=m.detection_id AND p.polygon_index=c.polygon_index
          JOIN raw.segmentation_points pt ON pt.polygon_id=p.id
          WHERE m.run_key=? AND m.contour_count>1
          ORDER BY m.detection_id, c.polygon_index, pt.point_index
        """
        current_id: int | None = None
        expected = (0, 0)
        current_polygon: int | None = None
        depth = 0
        points: list[tuple[float, float]] = []
        polygons: list[tuple[int, np.ndarray]] = []
        checked = 0
        mismatch_count = 0
        foreground_mismatch_count = 0
        hole_mismatch_count = 0

        def flush_polygon() -> None:
            nonlocal points
            if points:
                polygons.append((depth, np.asarray(points, dtype=np.float32)))
                points = []

        def flush_detection() -> None:
            nonlocal checked, mismatch_count, foreground_mismatch_count
            nonlocal hole_mismatch_count, polygons
            if current_id is None:
                return
            flush_polygon()
            actual = _counts(polygons)
            checked += 1
            if actual != expected:
                mismatch_count += 1
                foreground_mismatch_count += int(actual[0] != expected[0])
                hole_mismatch_count += int(actual[1] != expected[1])
                if len(mismatches) < 500:
                    mismatches.append(
                        {
                            "run_key": run_key,
                            "detection_id": current_id,
                            "expected_foreground": expected[0],
                            "expected_holes": expected[1],
                            "raster_foreground": actual[0],
                            "raster_holes": actual[1],
                        }
                    )
            polygons = []

        for did, expected_fg, expected_holes, row_depth, polygon_index, x, y in connection.execute(query, (run_key,)):
            did = int(did); polygon_index = int(polygon_index)
            if current_id is not None and did != current_id:
                flush_detection()
                current_polygon = None
            if current_id != did:
                current_id = did
                expected = (int(expected_fg), int(expected_holes))
            if current_polygon is not None and polygon_index != current_polygon:
                flush_polygon()
            current_polygon = polygon_index
            depth = int(row_depth)
            points.append((float(x), float(y)))
        flush_detection()
        connection.close()
        summaries.append(
            {
                "run_key": run_key,
                "checked_multi_contour_detections": checked,
                "mismatches": mismatch_count,
                "foreground_count_mismatches": foreground_mismatch_count,
                "hole_count_mismatches": hole_mismatch_count,
                "agreement_rate": 1.0 - mismatch_count / checked if checked else 1.0,
                "foreground_agreement_rate": (
                    1.0 - foreground_mismatch_count / checked if checked else 1.0
                ),
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    destination.close()
    payload = {"runs": summaries, "mismatch_examples": mismatches}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-key", action="append", default=[])
    args = parser.parse_args()
    payload = validate(
        args.topology,
        args.matrix,
        args.output,
        set(args.run_key) if args.run_key else None,
    )
    mismatches = sum(int(item["mismatches"]) for item in payload["runs"])
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
