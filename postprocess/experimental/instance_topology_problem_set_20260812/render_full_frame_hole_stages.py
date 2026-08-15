#!/usr/bin/env python3
"""Render full-frame raw, pre-DP polygon and post-DP masks for hole cases.

All source-video access is local-only.  The pre-DP polygon is the optimizer
input contour resampled to the exact adaptive vertex count used for that run.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from render_actual_frame_overlays import (
    COLORS,
    draw_groups,
    draw_topology_mask,
    mask_from_groups,
    open_ro,
    put,
    raw_contours,
    read_frames,
)


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "output/instance_mask_topology_20260806"
TOPOLOGY = EXPERIMENT / "topology.sqlite"
OUTPUT = (
    ROOT
    / "output/instance_topology_three_core_families_20260812/01_holes/full_frame_stage_review"
)

CASES = [
    {
        "sample": 2,
        "run_key": "v3__heyzo_3549_full",
        "frame": 78614,
        "detection_id": 48603,
        "track_id": "133",
    },
    {
        "sample": 3,
        "run_key": "v3__heyzo_3554_full",
        "frame": 44679,
        "detection_id": 32237,
        "track_id": "189",
    },
    {
        "sample": 5,
        "run_key": "v3__white_2025_03_0210",
        "frame": 35065,
        "detection_id": 55311,
        "track_id": "304",
    },
]


def summary(run_key: str) -> dict[str, object]:
    return json.loads(
        (EXPERIMENT / f"postprocess_trace.{run_key}.summary.json").read_text(encoding="utf-8")
    )


def stage_root(run_key: str) -> Path:
    return Path(str(summary(run_key)["postprocess_manifest"])).parent / "05_polygon_optimization"


def sqlite_groups(path: Path, track_id: str, frame: int) -> list[dict[str, object]]:
    connection = open_ro(path)
    row = connection.execute(
        "SELECT polygons FROM masks WHERE track_id=? AND frame=?",
        (str(track_id), int(frame)),
    ).fetchone()
    connection.close()
    if row is None or not row["polygons"]:
        return []
    return [
        {"slot": index, "role": "exterior", "points": np.asarray(points, np.float32)}
        for index, points in enumerate(json.loads(str(row["polygons"])))
        if len(points) >= 3
    ]


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points.astype(np.float32)
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, int(count), endpoint=False)
    output = np.empty((int(count), 2), np.float64)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] <= target:
            segment += 1
        span = max(float(cumulative[segment + 1] - cumulative[segment]), 1e-9)
        alpha = (target - cumulative[segment]) / span
        output[index] = closed[segment] * (1.0 - alpha) + closed[segment + 1] * alpha
    return output.astype(np.float32)


def run_id_at_frame(root: Path, track_id: str, frame: int) -> int:
    data = json.loads((root / "vendor_output/opt/interpolated_union.json").read_text())
    for row in data:
        if str(row["track_id"]) == str(track_id) and int(row["frame"]) == int(frame):
            return int(row["run_id"])
    raise KeyError(f"run_id not found: track={track_id} frame={frame}")


def anchor_count(root: Path, track_id: str, run_id: int) -> int:
    with (root / "vendor_output/opt/stream_segments.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["track_id"]) == str(track_id) and int(row["run_id"]) == int(run_id):
                return int(row["anchors_per_contour"])
    raise KeyError(f"anchor count not found: track={track_id} run={run_id}")


def add_header(image: np.ndarray, lines: list[str]) -> np.ndarray:
    header = 104
    output = np.full((image.shape[0] + header, image.shape[1], 3), 14, np.uint8)
    output[header:] = image
    for index, line in enumerate(lines[:3]):
        put(output, line, (18, 28 + 30 * index), 0.62 if index == 0 else 0.54)
    return output


def draw_stage(
    image: np.ndarray,
    groups: list[dict[str, object]],
    title: str,
    subtitle: str,
    detail: str,
    *,
    raw_topology: bool = False,
) -> np.ndarray:
    rendered = (
        draw_topology_mask(image, groups, stage="raw")
        if raw_topology
        else draw_groups(image, groups, alpha=0.42, label_prefix="component")
    )
    return add_header(rendered, [title, subtitle, detail])


def metrics(shape: tuple[int, int], left: list[dict[str, object]], right: list[dict[str, object]]) -> dict[str, float]:
    a = mask_from_groups(shape, left).astype(bool)
    b = mask_from_groups(shape, right).astype(bool)
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return {
        "iou": intersection / max(union, 1),
        "left_area": int(np.count_nonzero(a)),
        "right_area": int(np.count_nonzero(b)),
        "right_to_left_area": int(np.count_nonzero(b)) / max(int(np.count_nonzero(a)), 1),
    }


def hole_fill_fraction(
    shape: tuple[int, int],
    raw: list[dict[str, object]],
    stage: list[dict[str, object]],
) -> float:
    holes = [group for group in raw if group["role"] == "hole"]
    hole_mask = np.zeros(shape, np.uint8)
    if holes:
        cv2.fillPoly(
            hole_mask,
            [np.rint(group["points"]).astype(np.int32) for group in holes],
            1,
        )
    stage_mask = mask_from_groups(shape, stage).astype(bool)
    return int(np.count_nonzero(stage_mask & (hole_mask > 0))) / max(
        int(np.count_nonzero(hole_mask)), 1
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("*.png"):
        stale.unlink()

    topology = open_ro(TOPOLOGY)
    run_meta = {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs")
    }
    manifest: list[dict[str, object]] = []

    for case in CASES:
        sample = int(case["sample"])
        run_key = str(case["run_key"])
        frame = int(case["frame"])
        detection_id = int(case["detection_id"])
        track_id = str(case["track_id"])
        root = stage_root(run_key)
        video = Path(str(run_meta[run_key]["input_video"]))
        image = read_frames(video, [frame])[frame]

        inference = open_ro(Path(str(run_meta[run_key]["inference_sqlite"])))
        raw = raw_contours(inference, topology, run_key, detection_id)
        inference.close()

        optimizer_input = sqlite_groups(root / "endpoint_extended.sqlite", track_id, frame)
        run_id = run_id_at_frame(root, track_id, frame)
        anchors = anchor_count(root, track_id, run_id)
        pre_dp = [
            {**group, "points": resample_closed(group["points"], anchors)}
            for group in optimizer_input
        ]
        post_dp = sqlite_groups(root / "predictions.sqlite", track_id, frame)

        raw_holes = sum(group["role"] == "hole" for group in raw)
        raw_view = draw_stage(
            image,
            raw,
            "1/3 AI RAW BINARY MASK",
            f"run={run_key} frame={frame} detection={detection_id}",
            f"foreground={sum(g['role']=='foreground' for g in raw)} holes={raw_holes}; yellow=hole",
            raw_topology=True,
        )
        pre_view = draw_stage(
            image,
            pre_dp,
            "2/3 INITIAL POLYGON (BEFORE DP)",
            f"track={track_id} run_id={run_id} vertices/contour={anchors}",
            f"components={len(pre_dp)}; optimizer input has no hole-role field",
        )
        post_view = draw_stage(
            image,
            post_dp,
            "3/3 FINAL MASK (AFTER DP + INTERPOLATION)",
            f"track={track_id} frame={frame}",
            f"components={len(post_dp)}; final dense prediction",
        )

        stem = f"sample_{sample:02d}_{run_key}_f{frame}"
        paths = {
            "raw": OUTPUT / f"{stem}_01_ai_raw_full.png",
            "pre_dp": OUTPUT / f"{stem}_02_polygon_pre_dp_full.png",
            "post_dp": OUTPUT / f"{stem}_03_dp_final_full.png",
            "comparison": OUTPUT / f"{stem}_comparison_full.png",
        }
        cv2.imwrite(str(paths["raw"]), raw_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(paths["pre_dp"]), pre_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(paths["post_dp"]), post_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        comparison = np.concatenate([raw_view, pre_view, post_view], axis=1)
        cv2.imwrite(str(paths["comparison"]), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        raw_pre = metrics(image.shape[:2], raw, pre_dp)
        raw_post = metrics(image.shape[:2], raw, post_dp)
        pre_post = metrics(image.shape[:2], pre_dp, post_dp)
        manifest.append({
            **case,
            "video": str(video),
            "run_id": run_id,
            "anchors_per_contour": anchors,
            "raw_foreground_count": sum(group["role"] == "foreground" for group in raw),
            "raw_hole_count": raw_holes,
            "pre_dp_component_count": len(pre_dp),
            "post_dp_component_count": len(post_dp),
            "raw_to_pre_dp": raw_pre,
            "raw_to_post_dp": raw_post,
            "pre_dp_to_post_dp": pre_post,
            "pre_dp_fill_fraction_inside_raw_holes": hole_fill_fraction(image.shape[:2], raw, pre_dp),
            "post_dp_fill_fraction_inside_raw_holes": hole_fill_fraction(image.shape[:2], raw, post_dp),
            "paths": {key: str(value) for key, value in paths.items()},
        })

    topology.close()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# 穴の重大3例・フルフレーム段階比較\n\n"
        "各例について、元動画の全フレームを切り抜かずに次の4画像を保存しています。\n\n"
        "1. `01_ai_raw_full`: AI生マスク。黄色輪郭は背景としての穴。\n"
        "2. `02_polygon_pre_dp_full`: DP入力直前の初期ポリゴン。実際のadaptive頂点数で再サンプリング。\n"
        "3. `03_dp_final_full`: DP、pair-vote、キーフレーム間補完後の最終dense mask。\n"
        "4. `comparison_full`: 上記3段階の横並び。\n\n"
        "重要: `masks.polygons` はring roleを保持しないため、pre-DPでは入力された全輪郭がexterior componentとして扱われます。"
        "各段階のIoU、面積比、元の穴内部を埋めた割合はmanifest.jsonに保存しています。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
