#!/usr/bin/env python3
"""Render one actual-video overlay artifact for each audited failure family.

This script intentionally opens local source videos.  It never performs any
network access.  The resulting review artifacts are therefore sensitive and
must remain local.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.validation import explain_validity


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "output/instance_mask_topology_20260806"
OUTPUT = ROOT / "output/instance_topology_actual_frame_overlays_20260812"
TOPOLOGY = EXPERIMENT / "topology.sqlite"
CURRENT_KPI = ROOT / "data/12月KPI動画.sqlite"

COLORS = [
    (80, 80, 255),   # red
    (255, 170, 40),  # blue
    (70, 220, 110),  # green
    (220, 80, 220),  # magenta
    (40, 220, 240),  # yellow
]


def open_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def put(image: np.ndarray, text: str, position: tuple[int, int], scale: float = 0.58) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def read_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    output: dict[int, np.ndarray] = {}
    for frame_index in sorted(set(frames)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"failed to decode {video} frame {frame_index}")
        output[frame_index] = image
    capture.release()
    return output


def raw_contours(
    inference: sqlite3.Connection,
    topology: sqlite3.Connection,
    run_key: str,
    detection_id: int,
) -> list[dict[str, object]]:
    roles = {
        int(row["polygon_index"]): str(row["role"])
        for row in topology.execute(
            """SELECT polygon_index,role FROM contour_topology
               WHERE run_key=? AND detection_id=?""",
            (run_key, detection_id),
        )
    }
    grouped: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in inference.execute(
        """SELECT p.polygon_index,pt.x,pt.y
           FROM segmentation_polygons p
           JOIN segmentation_points pt ON pt.polygon_id=p.id
           WHERE p.detection_id=? ORDER BY p.polygon_index,pt.point_index""",
        (detection_id,),
    ):
        grouped[int(row["polygon_index"])].append((float(row["x"]), float(row["y"])))
    return [
        {
            "slot": polygon_index,
            "role": roles.get(polygon_index, "foreground"),
            "points": np.asarray(points, dtype=np.float32),
        }
        for polygon_index, points in sorted(grouped.items())
    ]


def final_components(
    database: Path,
    track_id: str,
    frame: int,
) -> list[dict[str, object]]:
    connection = open_ro(database)
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = collections.defaultdict(list)
    for row in connection.execute(
        """SELECT c.slot_index,r.ring_role,p.x,p.y
           FROM mask_keyframes k
           JOIN mask_track_segments s ON s.id=k.segment_id
           JOIN keyframe_components c ON c.keyframe_id=k.id
           JOIN keyframe_polygon_rings r ON r.component_id=c.id
           JOIN keyframe_polygon_points p ON p.ring_id=r.id
           WHERE s.track_id=? AND k.frame=?
           ORDER BY c.slot_index,r.ring_index,p.point_index""",
        (track_id, int(frame)),
    ):
        grouped[(int(row["slot_index"]), str(row["ring_role"]))].append(
            (float(row["x"]), float(row["y"]))
        )
    connection.close()
    return [
        {"slot": slot, "role": role, "points": np.asarray(points, dtype=np.float32)}
        for (slot, role), points in sorted(grouped.items())
    ]


def mask_from_groups(shape: tuple[int, int], groups: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    foreground = [
        np.rint(group["points"]).astype(np.int32)
        for group in groups
        if group["role"] == "foreground" or group["role"] == "exterior"
    ]
    holes = [
        np.rint(group["points"]).astype(np.int32)
        for group in groups
        if group["role"] == "hole"
    ]
    if foreground:
        cv2.fillPoly(mask, foreground, 1)
    if holes:
        cv2.fillPoly(mask, holes, 0)
    return mask


def draw_groups(
    image: np.ndarray,
    groups: list[dict[str, object]],
    *,
    alpha: float = 0.38,
    label_prefix: str = "slot",
    invalid_slot: int | None = None,
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    for group in groups:
        points = np.rint(group["points"]).astype(np.int32)
        if len(points) < 3:
            continue
        slot = int(group["slot"])
        role = str(group["role"])
        color = COLORS[slot % len(COLORS)]
        if role == "hole":
            cv2.polylines(overlay, [points], True, (0, 255, 255), 5, cv2.LINE_AA)
        else:
            cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0, dst=output)
    for group in groups:
        points = np.rint(group["points"]).astype(np.int32)
        if len(points) < 3:
            continue
        slot = int(group["slot"])
        role = str(group["role"])
        color = (0, 0, 255) if slot == invalid_slot else COLORS[slot % len(COLORS)]
        thickness = 5 if slot == invalid_slot else 3
        cv2.polylines(output, [points], True, color if role != "hole" else (0, 255, 255), thickness, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        put(output, f"{label_prefix}{slot}:{role}", (int(center[0]), int(center[1])), 0.50)
    return output


def draw_topology_mask(
    image: np.ndarray,
    groups: list[dict[str, object]],
    *,
    stage: str,
) -> np.ndarray:
    """Draw foreground-minus-holes as a true binary mask."""
    output = image.copy()
    binary = mask_from_groups(image.shape[:2], groups).astype(bool)
    overlay = output.copy()
    overlay[binary] = (90, 70, 255)
    cv2.addWeighted(overlay, 0.38, output, 0.62, 0.0, dst=output)
    for group in groups:
        points = np.rint(group["points"]).astype(np.int32)
        if len(points) < 3:
            continue
        role = str(group["role"])
        color = (0, 255, 255) if role == "hole" else (80, 80, 255)
        cv2.polylines(output, [points], True, color, 4, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        put(output, f"{stage}:{role}", (int(center[0]), int(center[1])), 0.46)
    return output


def draw_two_masks(
    image: np.ndarray,
    first: list[dict[str, object]],
    second: list[dict[str, object]],
) -> np.ndarray:
    output = image.copy()
    left = mask_from_groups(image.shape[:2], first).astype(bool)
    right = mask_from_groups(image.shape[:2], second).astype(bool)
    overlay = output.copy()
    overlay[left] = (60, 90, 255)
    overlay[right] = (255, 130, 50)
    overlay[left & right] = (245, 245, 245)
    cv2.addWeighted(overlay, 0.45, output, 0.55, 0.0, dst=output)
    for groups, color, name in ((first, (60, 90, 255), "detA"), (second, (255, 130, 50), "detB")):
        points = np.concatenate([group["points"] for group in groups if len(group["points"])], axis=0)
        for group in groups:
            contour = np.rint(group["points"]).astype(np.int32)
            if len(contour) >= 3:
                cv2.polylines(output, [contour], True, color, 3, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        put(output, name, (int(center[0]), int(center[1])), 0.55)
    return output


def all_points(group_sets: list[list[dict[str, object]]]) -> np.ndarray:
    arrays = [group["points"] for groups in group_sets for group in groups if len(group["points"])]
    return np.concatenate(arrays, axis=0) if arrays else np.asarray([[0, 0], [1, 1]], np.float32)


def crop_panel(image: np.ndarray, points: np.ndarray, title: list[str]) -> np.ndarray:
    height, width = image.shape[:2]
    minimum = np.floor(points.min(axis=0)).astype(int)
    maximum = np.ceil(points.max(axis=0)).astype(int)
    span = np.maximum(maximum - minimum, 1)
    margin = max(40, int(round(0.60 * max(span))))
    x0 = max(0, int(minimum[0]) - margin)
    y0 = max(0, int(minimum[1]) - margin)
    x1 = min(width, int(maximum[0]) + margin + 1)
    y1 = min(height, int(maximum[1]) + margin + 1)
    crop = image[y0:y1, x0:x1].copy()
    target_w, target_h = 720, 480
    scale = min(target_w / max(crop.shape[1], 1), target_h / max(crop.shape[0], 1))
    resized = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    panel = np.full((target_h + 86, target_w, 3), 16, np.uint8)
    ox = (target_w - resized.shape[1]) // 2
    oy = 86 + (target_h - resized.shape[0]) // 2
    panel[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    for index, line in enumerate(title[:3]):
        put(panel, line, (14, 24 + index * 25), 0.50)
    return panel


def contact_sheet(panels: list[np.ndarray], columns: int = 3) -> np.ndarray:
    rows = (len(panels) + columns - 1) // columns
    h = max(panel.shape[0] for panel in panels)
    w = max(panel.shape[1] for panel in panels)
    sheet = np.full((rows * h, columns * w, 3), 12, np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * h : row * h + panel.shape[0], column * w : column * w + panel.shape[1]] = panel
    return sheet


def polygon_diagram(
    points: np.ndarray,
    crossing: tuple[int, int, tuple[float, float]],
    title: list[str],
) -> np.ndarray:
    panel = np.full((566, 720, 3), 12, np.uint8)
    for index, line in enumerate(title[:3]):
        put(panel, line, (14, 24 + index * 25), 0.50)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = min(600.0 / span[0], 410.0 / span[1])
    mapped = (points - minimum) * scale
    mapped[:, 0] += (720.0 - span[0] * scale) * 0.5
    mapped[:, 1] += 112.0 + (410.0 - span[1] * scale) * 0.5
    mapped_i = np.rint(mapped).astype(np.int32)
    cv2.polylines(panel, [mapped_i], True, (90, 90, 255), 4, cv2.LINE_AA)
    left, right, cross_xy = crossing
    for segment, color in ((left, (0, 255, 255)), (right, (255, 255, 0))):
        cv2.line(
            panel,
            tuple(mapped_i[segment]),
            tuple(mapped_i[(segment + 1) % len(mapped_i)]),
            color,
            10,
            cv2.LINE_AA,
        )
    for index, point in enumerate(mapped_i):
        cv2.circle(panel, tuple(point), 5, (255, 255, 255), -1, cv2.LINE_AA)
        put(panel, str(index), (int(point[0]) + 7, int(point[1]) - 4), 0.32)
    cross_mapped = (np.asarray(cross_xy) - minimum) * scale
    cross_mapped[0] += (720.0 - span[0] * scale) * 0.5
    cross_mapped[1] += 112.0 + (410.0 - span[1] * scale) * 0.5
    cv2.circle(panel, tuple(np.rint(cross_mapped).astype(int)), 12, (60, 255, 60), 4, cv2.LINE_AA)
    return panel


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def crossing_segments(points: np.ndarray) -> tuple[int, int, tuple[float, float]] | None:
    count = len(points)
    for left in range(count):
        left_line = LineString([points[left], points[(left + 1) % count]])
        for right in range(left + 1, count):
            if right in (left, (left + 1) % count) or left in (right, (right + 1) % count):
                continue
            if left == 0 and right == count - 1:
                continue
            intersection = left_line.intersection(
                LineString([points[right], points[(right + 1) % count]])
            )
            if not intersection.is_empty and intersection.geom_type == "Point":
                return left, right, (float(intersection.x), float(intersection.y))
    return None


def result_path(run_key: str) -> Path:
    data = json.loads((EXPERIMENT / f"postprocess_trace.{run_key}.summary.json").read_text())
    return Path(str(data["result_sqlite"]))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    topology = open_ro(TOPOLOGY)
    runs = {
        str(row["run_key"]): dict(row)
        for row in topology.execute("SELECT * FROM audit_runs")
    }
    manifest: list[dict[str, object]] = []

    def raw(run_key: str, detection_id: int) -> list[dict[str, object]]:
        connection = open_ro(Path(str(runs[run_key]["inference_sqlite"])))
        groups = raw_contours(connection, topology, run_key, detection_id)
        connection.close()
        return groups

    # 1. Disconnected island in one inference instance, raw versus final.
    run_key, frame, detection_id, track_id = "v3lite__kpi_2025_12", 22124, 22530, "102"
    groups = raw(run_key, detection_id)
    final_groups = final_components(result_path(run_key), track_id, frame)
    image = read_frames(Path(str(runs[run_key]["input_video"])), [frame])[frame]
    raw_rendered = draw_groups(image, groups, label_prefix="raw_component")
    final_rendered = draw_groups(image, final_groups, label_prefix="final_slot")
    common = all_points([groups, final_groups])
    panel = contact_sheet([
        crop_panel(raw_rendered, common, [
            "01A AI raw mask: disconnected island",
            f"frame={frame} detection={detection_id}",
            "Two disjoint raw foreground components; secondary=3.95%",
        ]),
        crop_panel(final_rendered, common, [
            "01B Final polygon keyframe",
            f"track={track_id} components={len(final_groups)}",
            "The disconnected component survives into final polygon slots",
        ]),
    ], columns=2)
    path = OUTPUT / "01_disconnected_island.png"
    save(path, panel)
    manifest.append({"family": 1, "path": str(path), "run_key": run_key, "frame": frame})

    # 2. A true hole exported as another exterior component.
    run_key, frame, detection_id = "v3__kpi_2025_12", 15338, 15679
    raw_groups = raw(run_key, detection_id)
    exported = final_components(CURRENT_KPI, "63", frame)
    image = read_frames(Path(str(runs[run_key]["input_video"])), [frame])[frame]
    raw_rendered = draw_topology_mask(image, raw_groups, stage="raw")
    final_rendered = draw_groups(image, exported, label_prefix="final_slot")
    common = all_points([raw_groups, exported])
    panel = contact_sheet([
        crop_panel(raw_rendered, common, [
            "02A AI raw binary mask: one foreground with a hole",
            f"frame={frame} detection={detection_id}",
            "Yellow=background hole; the binary mask correctly excludes it",
        ]),
        crop_panel(final_rendered, common, [
            "02B Final SQLite polygon keyframe",
            "final_track=63; both rings were written as exterior slots",
            "The raw hole becomes a second foreground mask",
        ]),
    ], columns=2)
    path = OUTPUT / "02_hole_exported_as_mask.png"
    save(path, panel)
    manifest.append({"family": 2, "path": str(path), "run_key": run_key, "frame": frame})

    # 3. Two retained inference instances overlap/contain one another, raw
    # versus their two final tracks.
    run_key, frame, ids = "v3__heyzo_3545_full", 13813, [7216, 7217]
    first, second = raw(run_key, ids[0]), raw(run_key, ids[1])
    final_first = final_components(result_path(run_key), "9", frame)
    final_second = final_components(result_path(run_key), "10", frame)
    image = read_frames(Path(str(runs[run_key]["input_video"])), [frame])[frame]
    raw_rendered = draw_two_masks(image, first, second)
    final_rendered = draw_two_masks(image, final_first, final_second)
    common = all_points([first, second, final_first, final_second])
    panel = contact_sheet([
        crop_panel(raw_rendered, common, [
            "03A AI raw masks: cross-instance containment",
            f"frame={frame} detections={ids}",
            "Red/blue=detections; white=overlap; smaller containment=96.39%",
        ]),
        crop_panel(final_rendered, common, [
            "03B Final polygon keyframes",
            "Both detections survived as final tracks 9 and 10",
            "The inter-instance duplication remains after NMS/tracking",
        ]),
    ], columns=2)
    path = OUTPUT / "03_cross_instance_overlap.png"
    save(path, panel)
    manifest.append({"family": 3, "path": str(path), "run_key": run_key, "frame": frame})

    # 4. Components within one final track partially overlap.
    run_key, frame, track_id = "v3__heyzo_3560_full", 82042, "179"
    raw_groups = raw(run_key, 81062)
    groups = final_components(result_path(run_key), track_id, frame)
    image = read_frames(Path(str(runs[run_key]["input_video"])), [frame])[frame]
    raw_rendered = draw_topology_mask(image, raw_groups, stage="raw")
    rendered = draw_groups(image, groups, label_prefix="final_slot")
    masks = []
    for group in groups:
        masks.append(mask_from_groups(image.shape[:2], [group]).astype(bool))
    if len(masks) >= 2:
        rendered[masks[0] & masks[1]] = (245, 245, 245)
    common = all_points([raw_groups, groups])
    panel = contact_sheet([
        crop_panel(raw_rendered, common, [
            "04A AI raw mask: foreground with a hole",
            "frame=82042 detection=81062",
            "This is not two overlapping raw foreground components",
        ]),
        crop_panel(rendered, common, [
            "04B Final polygons: partial overlap",
            f"track={track_id}; white=slot0/slot1 overlap",
            "Hole mis-export + later geometry processing created this overlap",
        ]),
    ], columns=2)
    path = OUTPUT / "04_same_instance_partial_overlap.png"
    save(path, panel)
    manifest.append({"family": 4, "path": str(path), "run_key": run_key, "frame": frame})

    # 5. Invalid/self-intersecting final polygon.  Use a case where the raw
    # contour is valid and the final postprocessed polygon becomes invalid,
    # then show both stages and a magnified intersection.
    run_key, frame, track_id, detection_id, invalid_slot = (
        "v3__heyzo_3560_full", 57233, "100", 46784, 0
    )
    raw_groups = raw(run_key, detection_id)
    groups = final_components(result_path(run_key), track_id, frame)
    image = read_frames(Path(str(runs[run_key]["input_video"])), [frame])[frame]
    # The final polygon extends slightly outside the right frame edge.  Retain
    # that area in a black padded canvas so the failure is not clipped away.
    padded = np.zeros((image.shape[0], image.shape[1] + 100, 3), dtype=np.uint8)
    padded[:, : image.shape[1]] = image
    raw_rendered = draw_groups(padded, raw_groups, label_prefix="raw")
    rendered = padded.copy()
    invalid = next(group for group in groups if int(group["slot"]) == invalid_slot)
    polygon = Polygon(invalid["points"])
    reason = explain_validity(polygon)
    points = np.rint(invalid["points"]).astype(np.int32)
    overlay = rendered.copy()
    cv2.fillPoly(overlay, [points], (0, 0, 255))
    cv2.addWeighted(overlay, 0.18, rendered, 0.82, 0.0, dst=rendered)
    cv2.polylines(rendered, [points], True, (0, 0, 255), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(rendered, tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
    crossing = crossing_segments(np.asarray(invalid["points"], dtype=np.float64))
    if crossing is None:
        raise RuntimeError("expected a self-intersection but no crossing pair was found")
    left_segment, right_segment, cross_xy = crossing
    cv2.line(
        rendered,
        tuple(points[left_segment]),
        tuple(points[(left_segment + 1) % len(points)]),
        (0, 255, 255),
        7,
        cv2.LINE_AA,
    )
    cv2.line(
        rendered,
        tuple(points[right_segment]),
        tuple(points[(right_segment + 1) % len(points)]),
        (255, 255, 0),
        7,
        cv2.LINE_AA,
    )
    cross = tuple(np.rint(cross_xy).astype(int))
    cv2.circle(rendered, cross, 7, (255, 255, 255), 2, cv2.LINE_AA)
    raw_panel = crop_panel(raw_rendered, all_points([raw_groups]), [
        "05A AI raw binary mask / raw contour",
        f"frame={frame} detection={detection_id} area=3732px2",
        "Raw contour is valid (red fill/outline)",
    ])
    final_panel = crop_panel(rendered, all_points([groups]), [
        "05B Final postprocessed polygon",
        f"track={track_id} slot={invalid_slot} area=366.5px2",
        reason,
    ])
    zoom_panel = polygon_diagram(
        np.asarray(invalid["points"], dtype=np.float64),
        crossing,
        [
            "05C Geometry-only magnification of final polygon",
            "Yellow/cyan edges cross at green circle; white dots=vertices",
            "This crossing is absent from the raw mask",
        ],
    )
    panel = contact_sheet([raw_panel, final_panel, zoom_panel], columns=3)
    path = OUTPUT / "05_self_intersecting_polygon.png"
    save(path, panel)
    manifest.append({
        "family": 5,
        "path": str(path),
        "run_key": run_key,
        "frame": frame,
        "detection_id": detection_id,
        "final_track_id": track_id,
        "raw_geometry_valid": True,
        "final_geometry_valid": False,
    })

    # 6. Temporal component-count flicker and unstable area-ranked slots.
    run_key, track_id = "v3lite__heyzo_3545_30_45_duplicate", "30"
    frame_detection_ids = {
        8644: 12765,
        8645: 12767,
        8646: 12769,
        8680: 12841,
        8681: 12843,
        8682: 12845,
    }
    frames = list(frame_detection_ids)
    video_frames = read_frames(Path(str(runs[run_key]["input_video"])), frames)
    groups_by_frame = {frame: final_components(result_path(run_key), track_id, frame) for frame in frames}
    raw_by_frame = {
        frame: raw(run_key, detection_id)
        for frame, detection_id in frame_detection_ids.items()
    }
    common_points = all_points(list(groups_by_frame.values()) + list(raw_by_frame.values()))
    panels = []
    for frame in frames:
        raw_groups = raw_by_frame[frame]
        groups = groups_by_frame[frame]
        raw_rendered = draw_topology_mask(video_frames[frame], raw_groups, stage="raw")
        final_rendered = draw_groups(video_frames[frame], groups, label_prefix="final_slot")
        panels.append(crop_panel(raw_rendered, common_points, [
            "06A Raw topology",
            f"frame={frame} foreground={sum(g['role']=='foreground' for g in raw_groups)} holes={sum(g['role']=='hole' for g in raw_groups)}",
            "Intermittent raw hole is shown in yellow",
        ]))
        panels.append(crop_panel(final_rendered, common_points, [
            "06B Final polygon slots",
            f"frame={frame} track={track_id} components={len(groups)}",
            "The intermittent hole becomes an appearing/disappearing slot1",
        ]))
    sheet = contact_sheet(panels, columns=2)
    path = OUTPUT / "06_temporal_component_slot_instability.png"
    save(path, sheet)
    manifest.append({"family": 6, "path": str(path), "run_key": run_key, "frames": frames})

    topology.close()
    payload = {
        "schema_version": 1,
        "privacy": "Contains locally decoded source-video frames; do not upload",
        "artifacts": manifest,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Actual-frame topology overlays\n\n"
        "ローカル動画フレームを使用した機密性のある確認用成果物です。外部へアップロードしないでください。\n\n"
        "1. `01_disconnected_island.png` — 同一AIインスタンス内の離れた島\n"
        "2. `02_hole_exported_as_mask.png` — 穴の別mask誤export\n"
        "3. `03_cross_instance_overlap.png` — 別インスタンス間の重複・内包\n"
        "4. `04_same_instance_partial_overlap.png` — raw holeの誤export後に生じた同一track内componentの部分重複\n"
        "5. `05_self_intersecting_polygon.png` — validなraw maskから後処理後に生じた自己交差polygon\n"
        "6. `06_temporal_component_slot_instability.png` — intermittent raw holeがslot 1の点滅へ変換される過程\n\n"
        "各画像は左側にAI生マスク、右側に最終polygonを配置しています。5番のみ3枚目にpolygon幾何の拡大図があります。\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
