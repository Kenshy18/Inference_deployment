#!/usr/bin/env python3
"""Render every instance in three full-frame processing stages, color-stable."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from render_actual_frame_overlays import mask_from_groups, open_ro, put, raw_contours, read_frames
from render_full_frame_hole_stages import (
    CASES,
    EXPERIMENT,
    ROOT,
    TOPOLOGY,
    anchor_count,
    resample_closed,
    run_id_at_frame,
    sqlite_groups,
    stage_root,
)


TRACE = EXPERIMENT / "postprocess_trace.sqlite"
OUTPUT = (
    ROOT
    / "output/instance_topology_three_core_families_20260812/01_holes/full_frame_all_instances"
)

# BGR; ordered for strong separation on video frames.
PALETTE = [
    (40, 70, 255),    # red
    (255, 120, 30),   # blue
    (40, 220, 90),    # green
    (220, 70, 220),   # magenta
    (20, 210, 245),   # yellow
    (230, 190, 40),   # cyan
    (80, 150, 255),   # orange
    (190, 100, 255),  # pink
    (180, 220, 70),   # lime
    (255, 90, 150),   # violet-blue
]


def _all_sqlite_instances(path: Path, frame: int) -> dict[str, list[dict[str, object]]]:
    connection = open_ro(path)
    track_ids = [
        str(row["track_id"])
        for row in connection.execute(
            "SELECT track_id FROM masks WHERE frame=? ORDER BY CAST(track_id AS INTEGER),track_id",
            (int(frame),),
        )
    ]
    connection.close()
    return {track_id: sqlite_groups(path, track_id, frame) for track_id in track_ids}


def _centroid(groups: list[dict[str, object]]) -> tuple[int, int]:
    arrays = [np.asarray(group["points"]) for group in groups if len(group["points"])]
    if not arrays:
        return 10, 10
    center = np.mean(np.concatenate(arrays, axis=0), axis=0)
    return int(center[0]), int(center[1])


def _draw_instances(
    image: np.ndarray,
    instances: dict[str, dict[str, object]],
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    for identity, item in instances.items():
        groups = list(item["groups"])
        mask = mask_from_groups(image.shape[:2], groups).astype(bool)
        overlay[mask] = colors[identity]
    cv2.addWeighted(overlay, 0.46, output, 0.54, 0.0, dst=output)

    for identity, item in instances.items():
        groups = list(item["groups"])
        color = colors[identity]
        for group in groups:
            contour = np.rint(group["points"]).astype(np.int32)
            if len(contour) < 3:
                continue
            cv2.polylines(output, [contour], True, color, 4, cv2.LINE_AA)
        x, y = _centroid(groups)
        label = str(item["label"])
        # Solid identity badge makes labels readable even on overlapping masks.
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        x = int(np.clip(x, 4, max(4, output.shape[1] - width - 10)))
        y = int(np.clip(y, height + 8, max(height + 8, output.shape[0] - 8)))
        cv2.rectangle(output, (x - 4, y - height - 7), (x + width + 6, y + 6), color, -1)
        cv2.putText(
            output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (15, 15, 15), 2, cv2.LINE_AA,
        )
    return output


def _header(
    image: np.ndarray,
    title: str,
    detail: str,
    visible: list[str],
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    header_height = 138
    output = np.full((image.shape[0] + header_height, image.shape[1], 3), 14, np.uint8)
    output[header_height:] = image
    put(output, title, (18, 30), 0.66)
    put(output, detail, (18, 62), 0.52)
    x, y = 18, 104
    for identity in visible:
        label = identity
        (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        if x + width + 48 > output.shape[1]:
            break
        cv2.rectangle(output, (x, y - 17), (x + 24, y + 7), colors[identity], -1)
        put(output, label, (x + 31, y + 3), 0.48)
        x += width + 66
    return output


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("*.png"):
        stale.unlink()

    topology = open_ro(TOPOLOGY)
    trace = open_ro(TRACE)
    run_meta = {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs")
    }
    manifest: list[dict[str, object]] = []

    for case in CASES:
        sample = int(case["sample"])
        run_key = str(case["run_key"])
        frame = int(case["frame"])
        root = stage_root(run_key)
        video = Path(str(run_meta[run_key]["input_video"]))
        image = read_frames(video, [frame])[frame]

        inference = open_ro(Path(str(run_meta[run_key]["inference_sqlite"])))
        raw_instances: dict[str, dict[str, object]] = {}
        raw_rows = list(
            trace.execute(
                """SELECT detection_id,score,disposition,final_track_id
                   FROM detection_outcomes WHERE run_key=? AND frame=? ORDER BY detection_id""",
                (run_key, frame),
            )
        )
        for row in raw_rows:
            retained = row["final_track_id"] is not None and str(row["disposition"]) == "retained"
            identity = f"T{row['final_track_id']}" if retained else f"D{row['detection_id']}"
            status = "" if retained else ":raw-only"
            raw_instances[identity] = {
                "groups": raw_contours(
                    inference, topology, run_key, int(row["detection_id"])
                ),
                "label": f"{identity}{status}",
                "detection_id": int(row["detection_id"]),
                "score": float(row["score"] or 0.0),
                "disposition": str(row["disposition"]),
            }
        inference.close()

        pre_rows = _all_sqlite_instances(root / "endpoint_extended.sqlite", frame)
        pre_instances: dict[str, dict[str, object]] = {}
        pre_details: dict[str, dict[str, int]] = {}
        for track_id, groups in pre_rows.items():
            run_id = run_id_at_frame(root, track_id, frame)
            anchors = anchor_count(root, track_id, run_id)
            identity = f"T{track_id}"
            pre_instances[identity] = {
                "groups": [
                    {**group, "points": resample_closed(group["points"], anchors)}
                    for group in groups
                ],
                "label": identity,
            }
            pre_details[identity] = {"run_id": run_id, "anchors_per_contour": anchors}

        post_rows = _all_sqlite_instances(root / "predictions.sqlite", frame)
        post_instances = {
            f"T{track_id}": {"groups": groups, "label": f"T{track_id}"}
            for track_id, groups in post_rows.items()
        }

        # Stable identity-to-color mapping across all three panels.
        identities = sorted(
            set(raw_instances) | set(pre_instances) | set(post_instances),
            key=lambda value: (value[0] != "T", int(value[1:])),
        )
        colors = {identity: PALETTE[index % len(PALETTE)] for index, identity in enumerate(identities)}

        raw_view = _header(
            _draw_instances(image, raw_instances, colors),
            "1/3 ALL AI RAW INSTANCES",
            f"run={run_key} frame={frame}; raw-only means removed before tracking",
            list(raw_instances),
            colors,
        )
        pre_view = _header(
            _draw_instances(image, pre_instances, colors),
            "2/3 ALL INITIAL POLYGONS (BEFORE DP)",
            "Same T-number and color = same tracked instance",
            list(pre_instances),
            colors,
        )
        post_view = _header(
            _draw_instances(image, post_instances, colors),
            "3/3 ALL FINAL MASKS (AFTER DP + INTERPOLATION)",
            "A T-number appearing only here is a gap-filled/interpolated track",
            list(post_instances),
            colors,
        )

        stem = f"sample_{sample:02d}_{run_key}_f{frame}"
        paths = {
            "raw_all": OUTPUT / f"{stem}_01_ai_raw_all_full.png",
            "pre_dp_all": OUTPUT / f"{stem}_02_polygon_pre_dp_all_full.png",
            "post_dp_all": OUTPUT / f"{stem}_03_dp_final_all_full.png",
            "comparison_all": OUTPUT / f"{stem}_comparison_all_full.png",
        }
        cv2.imwrite(str(paths["raw_all"]), raw_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(paths["pre_dp_all"]), pre_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(paths["post_dp_all"]), post_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(
            str(paths["comparison_all"]),
            np.concatenate([raw_view, pre_view, post_view], axis=1),
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        manifest.append({
            **case,
            "video": str(video),
            "identity_colors_bgr": {key: list(value) for key, value in colors.items()},
            "raw_instances": {
                key: {
                    "detection_id": value["detection_id"],
                    "score": value["score"],
                    "disposition": value["disposition"],
                }
                for key, value in raw_instances.items()
            },
            "pre_dp_instances": list(pre_instances),
            "pre_dp_details": pre_details,
            "post_dp_instances": list(post_instances),
            "paths": {key: str(value) for key, value in paths.items()},
        })

    topology.close()
    trace.close()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# 全インスタンス・フルフレーム3段階比較\n\n"
        "同じT番号は同じトラックで、3段階を通じて同じ色です。"
        "D番号はAI生出力には存在するもののNMS等で追跡へ渡らなかったraw-only検出です。"
        "DP後にだけ現れるT番号はgap-fillまたは補間されたトラックです。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
