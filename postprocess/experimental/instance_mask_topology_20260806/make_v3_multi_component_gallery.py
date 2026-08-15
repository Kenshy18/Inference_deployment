#!/usr/bin/env python3
"""Create a local-only gallery for every V3 multi-foreground detection.

The script decodes source frames locally and never sends image data outside the
machine.  It renders the largest foreground component, additional disconnected
foreground components, holes, and the detector bounding box with distinct
visual styles.  A self-contained HTML index and machine-readable manifests are
written beside the PNG files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PALETTE_BGR = [
    (180, 70, 255),   # main: pink
    (0, 145, 255),    # secondary 1: orange
    (0, 220, 255),    # secondary 2: yellow
    (80, 230, 90),    # secondary 3: green
    (255, 120, 40),   # secondary 4+: blue
]
HOLE_BGR = (255, 255, 0)
BOX_BGR = (245, 245, 245)


def _published(item: dict[str, object]) -> Path:
    default = Path(str(item["inference_sqlite"]))
    manifest = Path(str(item["output_root"])) / "logs" / "run_manifest.json"
    if not manifest.is_file():
        return default
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return Path(str(payload.get("artifacts", {}).get("result_sqlite", default)))


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.62,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _severity(second_ratio: float) -> str:
    if second_ratio >= .20:
        return "critical_20pct_plus"
    if second_ratio >= .05:
        return "large_5_20pct"
    if second_ratio >= .01:
        return "medium_1_5pct"
    if second_ratio >= .001:
        return "small_0_1_1pct"
    return "tiny_below_0_1pct"


def _timestamp(frame: int, fps: float) -> str:
    seconds = frame / fps if fps > 0 else 0.0
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def _render(
    image: np.ndarray,
    contours: list[dict[str, object]],
    metadata: dict[str, object],
) -> np.ndarray:
    height, width = image.shape[:2]
    foreground = [item for item in contours if item["role"] == "foreground"]
    foreground.sort(key=lambda item: float(item["area"]), reverse=True)
    rank = {int(item["polygon_index"]): index for index, item in enumerate(foreground)}

    component_map = np.zeros((height, width), dtype=np.uint8)
    for item in sorted(contours, key=lambda row: (int(row["depth"]), int(row["polygon_index"]))):
        polygon = np.rint(item["points"]).astype(np.int32).reshape((-1, 1, 2))
        if item["role"] == "hole":
            cv2.fillPoly(component_map, [polygon], 0)
        else:
            cv2.fillPoly(component_map, [polygon], rank[int(item["polygon_index"])] + 1)

    rendered = image.copy()
    alpha = .46
    for index in range(len(foreground)):
        mask = component_map == index + 1
        if not np.any(mask):
            continue
        color = np.asarray(PALETTE_BGR[min(index, len(PALETTE_BGR) - 1)], dtype=np.float32)
        rendered[mask] = np.clip(
            rendered[mask].astype(np.float32) * (1.0 - alpha) + color * alpha,
            0,
            255,
        ).astype(np.uint8)

    for item in contours:
        polygon = np.rint(item["points"]).astype(np.int32).reshape((-1, 1, 2))
        if item["role"] == "hole":
            color = HOLE_BGR
        else:
            color = PALETTE_BGR[min(rank[int(item["polygon_index"])], len(PALETTE_BGR) - 1)]
        cv2.polylines(rendered, [polygon], True, color, 3, cv2.LINE_AA)

    for index, item in enumerate(foreground):
        points = np.asarray(item["points"], dtype=np.float32)
        center = tuple(np.rint(points.mean(axis=0)).astype(int))
        label = "MAIN" if index == 0 else f"SUB {index}"
        ratio = float(item["area"]) / max(float(foreground[0]["area"]), 1e-9)
        _text(rendered, f"{label} {ratio:.3f}x", center, .56, PALETTE_BGR[min(index, len(PALETTE_BGR) - 1)])

    for item in contours:
        if item["role"] != "hole":
            continue
        points = np.asarray(item["points"], dtype=np.float32)
        center = tuple(np.rint(points.mean(axis=0)).astype(int))
        _text(rendered, "HOLE", center, .50, HOLE_BGR)

    x1, y1, x2, y2 = (int(round(float(metadata[key]))) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    cv2.rectangle(rendered, (x1, y1), (x2, y2), BOX_BGR, 2, cv2.LINE_AA)

    title = (
        f"V3 | {metadata['run_key']} | frame={metadata['frame']} ({metadata['timestamp']}) "
        f"| det={metadata['detection_id']} score={float(metadata['score']):.3f} "
        f"| FG={metadata['foreground_components']} holes={metadata['holes']} "
        f"| second/main={float(metadata['second_ratio']):.4f}"
    )
    _text(rendered, title, (18, 34), .58)
    _text(rendered, "MAIN=pink  SUB=orange/yellow/green  HOLE=cyan outline  detector box=white", (18, 64), .50)
    return rendered


def _write_gallery(output_dir: Path, records: list[dict[str, object]]) -> None:
    output_dir = output_dir.resolve()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["run_key"])].append(record)
    sections: list[str] = []
    for run_key in sorted(grouped):
        cards: list[str] = []
        for record in sorted(grouped[run_key], key=lambda row: (-float(row["second_ratio"]), int(row["frame"]))):
            rel = Path(str(record["output"])).relative_to(output_dir).as_posix()
            cards.append(
                "<article class='card' "
                f"data-severity='{html.escape(str(record['severity']))}'>"
                f"<a href='{html.escape(rel)}' target='_blank'><img loading='lazy' src='{html.escape(rel)}'></a>"
                "<div class='meta'>"
                f"<strong>{html.escape(str(record['timestamp']))}</strong> &nbsp; frame {record['frame']}"
                f"<br>det {record['detection_id']} · score {float(record['score']):.3f}"
                f" · FG {record['foreground_components']} · holes {record['holes']}"
                f"<br>副成分/主成分 <b>{float(record['second_ratio']):.4f}</b>"
                f" · {html.escape(str(record['severity']))}"
                "</div></article>"
            )
        sections.append(
            f"<section><h2>{html.escape(run_key)} <span>{len(grouped[run_key])}件</span></h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    document = f"""<!doctype html>
<html lang='ja'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>V3 複数連結成分ローカル確認</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background:#0b1018; color:#e8edf6; }}
body {{ margin:0; padding:28px; }} h1 {{ margin:0 0 8px; }} .lead {{ color:#aeb9cc; margin-bottom:24px; }}
h2 {{ margin-top:34px; border-bottom:1px solid #2a3445; padding-bottom:8px; }} h2 span {{ color:#90a0b8; font-size:.7em; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }}
.card {{ background:#141c29; border:1px solid #273247; border-radius:12px; overflow:hidden; box-shadow:0 8px 24px #0005; }}
.card img {{ width:100%; display:block; aspect-ratio:16/9; object-fit:contain; background:#05070b; }}
.meta {{ padding:11px 13px 14px; color:#c7d1e1; line-height:1.5; font-size:14px; }} .meta b {{ color:#ffb64b; }}
a {{ color:inherit; }}
</style></head><body>
<h1>V3 複数連結成分ローカル確認</h1>
<p class='lead'>全{len(records)}件。主成分はピンク、副成分はオレンジ等、穴はシアン輪郭、検出boxは白。画像クリックで原寸表示。</p>
{''.join(sections)}
</body></html>"""
    (output_dir / "gallery.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=1600)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = {str(item["run_key"]): item for item in json.loads(args.matrix.read_text(encoding="utf-8"))}
    topology = sqlite3.connect(f"file:{args.topology.resolve()}?mode=ro", uri=True)
    rows = topology.execute(
        """SELECT run_key,detection_id,frame,score,foreground_component_count,hole_count,
                  second_to_largest_ratio,bbox_x1,bbox_y1,bbox_x2,bbox_y2
           FROM mask_topology
           WHERE run_key GLOB 'v3__*' AND foreground_component_count>1
           ORDER BY run_key,frame,detection_id"""
    ).fetchall()
    by_run: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for row in rows:
        by_run[str(row[0])].append(row)

    records: list[dict[str, object]] = []
    for run_key in sorted(by_run):
        item = matrix[run_key]
        source_sqlite = _published(item)
        local = sqlite3.connect(":memory:")
        local.execute(f"ATTACH DATABASE '{_q(source_sqlite)}' AS raw")
        local.execute(f"ATTACH DATABASE '{_q(args.topology)}' AS topo")
        capture = cv2.VideoCapture(str(item["input_video"]))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {item['input_video']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        run_dir = args.output_dir / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        for row in by_run[run_key]:
            (
                _run_key, detection_id, frame, score, fg_count, hole_count, second_ratio,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            ) = row
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"could not decode {run_key} frame {frame}")
            contour_rows = local.execute(
                """SELECT c.polygon_index,c.nesting_depth,c.role,c.absolute_area,pt.x,pt.y
                   FROM topo.contour_topology c
                   JOIN raw.segmentation_polygons p
                     ON p.detection_id=c.detection_id AND p.polygon_index=c.polygon_index
                   JOIN raw.segmentation_points pt ON pt.polygon_id=p.id
                   WHERE c.run_key=? AND c.detection_id=?
                   ORDER BY c.polygon_index,pt.point_index""",
                (run_key, int(detection_id)),
            ).fetchall()
            grouped: dict[int, dict[str, object]] = {}
            for polygon_index, depth, role, area, x, y in contour_rows:
                entry = grouped.setdefault(
                    int(polygon_index),
                    {
                        "polygon_index": int(polygon_index),
                        "depth": int(depth),
                        "role": str(role),
                        "area": float(area),
                        "points": [],
                    },
                )
                entry["points"].append((float(x), float(y)))
            contours = list(grouped.values())
            for contour in contours:
                contour["points"] = np.asarray(contour["points"], dtype=np.float32)
            timestamp = _timestamp(int(frame), fps)
            metadata = {
                "run_key": run_key,
                "detection_id": int(detection_id),
                "frame": int(frame),
                "timestamp": timestamp,
                "score": float(score),
                "foreground_components": int(fg_count),
                "holes": int(hole_count),
                "second_ratio": float(second_ratio),
                "bbox_x1": float(bbox_x1), "bbox_y1": float(bbox_y1),
                "bbox_x2": float(bbox_x2), "bbox_y2": float(bbox_y2),
            }
            rendered = _render(image, contours, metadata)
            if args.max_width > 0 and rendered.shape[1] > args.max_width:
                scale = args.max_width / rendered.shape[1]
                rendered = cv2.resize(
                    rendered,
                    (args.max_width, round(rendered.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            target = run_dir / f"f{int(frame):06d}_d{int(detection_id):07d}_r{float(second_ratio):.4f}.png"
            if not cv2.imwrite(str(target), rendered):
                raise RuntimeError(f"could not write {target}")
            records.append(
                {
                    **metadata,
                    "severity": _severity(float(second_ratio)),
                    "source_video": str(item["input_video"]),
                    "source_sqlite": str(source_sqlite.resolve()),
                    "output": str(target.resolve()),
                }
            )
        capture.release()
        local.close()
        print(f"{run_key}: {len(by_run[run_key])} overlays")
    topology.close()

    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    _write_gallery(args.output_dir, records)
    (args.output_dir / "README.txt").write_text(
        "V3 multi-component local review\n"
        "Open gallery.html in a browser.\n"
        "Pink=largest foreground, orange/yellow/green=additional disconnected foreground, "
        "cyan outline=hole, white=detector box.\n"
        "All source decoding and rendering was performed locally.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} overlays to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
