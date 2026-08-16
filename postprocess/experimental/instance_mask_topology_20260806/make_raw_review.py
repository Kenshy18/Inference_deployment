#!/usr/bin/env python3
"""Generate local raw-mask topology stills across every completed matrix run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _put(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)


def _draw(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)


def _published(item: dict[str, object]) -> Path:
    default = Path(str(item["inference_sqlite"]))
    manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
    if not manifest.is_file():
        return default
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return Path(str(payload.get("artifacts", {}).get("result_sqlite", default)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-run", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = {str(item["run_key"]): item for item in json.loads(args.matrix.read_text(encoding="utf-8"))}
    topology = sqlite3.connect(f"file:{args.topology.resolve()}?mode=ro", uri=True)
    records: list[dict[str, object]] = []
    for run_key, in topology.execute("SELECT run_key FROM audit_runs ORDER BY run_key"):
        item = matrix[str(run_key)]
        source_sqlite = _published(item)
        if not source_sqlite.is_file():
            continue
        candidates: list[tuple[str, tuple[object, ...]]] = []
        large = topology.execute(
            """SELECT detection_id,frame,foreground_component_count,hole_count,
                      second_to_largest_ratio
               FROM mask_topology WHERE run_key=? AND foreground_component_count>1
               ORDER BY second_to_largest_ratio DESC,detection_id LIMIT ?""",
            (run_key, max(1, args.per_run - 2)),
        ).fetchall()
        candidates.extend(("large_secondary", row) for row in large)
        tiny = topology.execute(
            """SELECT detection_id,frame,foreground_component_count,hole_count,
                      second_to_largest_ratio
               FROM mask_topology WHERE run_key=? AND foreground_component_count>1
                 AND second_to_largest_ratio<.001
               ORDER BY detection_id LIMIT 1""",
            (run_key,),
        ).fetchall()
        candidates.extend(("tiny_secondary", row) for row in tiny)
        hole = topology.execute(
            """SELECT detection_id,frame,foreground_component_count,hole_count,
                      second_to_largest_ratio
               FROM mask_topology WHERE run_key=? AND hole_count>0
               ORDER BY largest_hole_to_outer_ratio DESC,detection_id LIMIT 1""",
            (run_key,),
        ).fetchall()
        candidates.extend(("hole", row) for row in hole)
        if not candidates:
            continue
        local = sqlite3.connect(":memory:")
        local.execute(f"ATTACH DATABASE '{_q(source_sqlite)}' AS raw")
        local.execute(f"ATTACH DATABASE '{_q(args.topology)}' AS topo")
        capture = cv2.VideoCapture(str(item["input_video"]))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {item['input_video']}")
        run_dir = args.output_dir / str(run_key)
        run_dir.mkdir(parents=True, exist_ok=True)
        for ordinal, (kind, row) in enumerate(candidates[: args.per_run], start=1):
            did, frame, fg_count, hole_count, ratio = row
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"could not decode {run_key} frame {frame}")
            points = local.execute(
                """SELECT c.polygon_index,c.role,pt.x,pt.y
                   FROM topo.contour_topology c
                   JOIN raw.segmentation_polygons p
                     ON p.detection_id=c.detection_id
                    AND p.polygon_index=c.polygon_index
                   JOIN raw.segmentation_points pt ON pt.polygon_id=p.id
                   WHERE c.run_key=? AND c.detection_id=?
                   ORDER BY c.polygon_index,pt.point_index""",
                (run_key, int(did)),
            ).fetchall()
            grouped: dict[tuple[int, str], list[tuple[float, float]]] = {}
            for polygon_index, role, x, y in points:
                grouped.setdefault((int(polygon_index), str(role)), []).append((float(x), float(y)))
            foreground_index = 0
            for (_polygon_index, role), polygon_points in grouped.items():
                if role == "hole":
                    color, alpha = (255, 255, 0), .12
                else:
                    palette = [(255, 70, 20), (255, 0, 255), (0, 180, 255), (40, 255, 80)]
                    color, alpha = palette[min(foreground_index, len(palette) - 1)], .38
                    foreground_index += 1
                _draw(image, np.asarray(polygon_points, dtype=np.float32), color, alpha)
            _put(image, f"{run_key} frame={frame} det={did} fg={fg_count} holes={hole_count} ratio={float(ratio):.4f}")
            if image.shape[1] > 1280:
                scale = 1280 / image.shape[1]
                image = cv2.resize(image, (1280, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            target = run_dir / f"{ordinal:02d}_{kind}_f{frame}_d{did}.png"
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"could not write {target}")
            records.append(
                {
                    "run_key": run_key,
                    "kind": kind,
                    "frame": int(frame),
                    "detection_id": int(did),
                    "foreground_components": int(fg_count),
                    "holes": int(hole_count),
                    "second_ratio": float(ratio),
                    "output": str(target.resolve()),
                }
            )
        capture.release()
        local.close()
    topology.close()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} local raw review stills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
