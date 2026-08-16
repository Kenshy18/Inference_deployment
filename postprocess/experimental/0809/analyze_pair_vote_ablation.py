#!/usr/bin/env python3
"""Compare one Phase-2 path before/after Production post-DP pair-vote.

The comparison is intentionally path-locked: keyframe identities must match.
It reports geometric movement at selected keys and dense exact raster metrics.
No video pixels are read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


LABELS = ("女性器", "男性器", "結合部分")


def _quantile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, q)) if array.size else float(default)


def _area(polygons: list[list[list[float]]]) -> float:
    total = 0.0
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if len(points) < 3:
            continue
        x = points[:, 0]
        y = points[:, 1]
        total += 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return float(total)


def _geometry(polygons: list[list[list[float]]]):
    geometries = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if len(points) < 3:
            continue
        geometry = Polygon(points)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if not geometry.is_empty:
            geometries.append(geometry)
    return unary_union(geometries) if geometries else Polygon()


def _polygons_valid(polygons: list[list[list[float]]]) -> bool:
    return all(
        len(polygon) >= 3 and Polygon(np.asarray(polygon, dtype=np.float64)).is_valid
        for polygon in polygons
    )


def _flatten_matching(
    before: list[list[list[float]]], after: list[list[list[float]]]
) -> tuple[np.ndarray, np.ndarray]:
    if len(before) != len(after):
        raise ValueError("pair-vote changed polygon component count")
    left = []
    right = []
    for before_polygon, after_polygon in zip(before, after):
        before_points = np.asarray(before_polygon, dtype=np.float64)
        after_points = np.asarray(after_polygon, dtype=np.float64)
        if before_points.shape != after_points.shape:
            raise ValueError("pair-vote changed polygon vertex contract")
        left.append(before_points)
        right.append(after_points)
    if not left:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.concatenate(left, axis=0), np.concatenate(right, axis=0)


def _key_id(row: dict[str, object]) -> tuple[str, int, int]:
    return str(row["track_id"]), int(row["run_id"]), int(row["frame"])


def _load_keys(path: Path) -> dict[tuple[str, int, int], dict[str, object]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    result = {_key_id(row): row for row in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate keyframe identity in {path}")
    return result


def _load_metrics(path: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (str(row["track_id"]), int(row["run_id"]), int(row["frame"]))
            result[identity] = {
                "has_keyframe": float(row["has_keyframe"]),
                "gt_area": float(row["gt_area"]),
                "pred_area": float(row["pred_area"]),
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "iou": float(row["iou"]),
            }
    return result


def _metric_summary(
    before: dict[tuple[str, int, int], dict[str, float]],
    after: dict[tuple[str, int, int], dict[str, float]],
    *,
    recall_floor: float,
    keyframes_only: bool,
) -> dict[str, object]:
    if set(before) != set(after):
        raise ValueError("dense exact metric identities differ")
    identities = [
        identity
        for identity in before
        if not keyframes_only or bool(before[identity]["has_keyframe"])
    ]
    before_iou = np.asarray([before[key]["iou"] for key in identities])
    after_iou = np.asarray([after[key]["iou"] for key in identities])
    before_recall = np.asarray([before[key]["recall"] for key in identities])
    after_recall = np.asarray([after[key]["recall"] for key in identities])
    before_area = np.asarray(
        [before[key]["pred_area"] / max(before[key]["gt_area"], 1.0) for key in identities]
    )
    after_area = np.asarray(
        [after[key]["pred_area"] / max(after[key]["gt_area"], 1.0) for key in identities]
    )
    delta = after_iou - before_iou
    return {
        "rows": int(len(identities)),
        "iou_before": {
            "mean": float(np.mean(before_iou)),
            "min": float(np.min(before_iou)),
            "q01": float(np.quantile(before_iou, 0.01)),
            "q05": float(np.quantile(before_iou, 0.05)),
        },
        "iou_after": {
            "mean": float(np.mean(after_iou)),
            "min": float(np.min(after_iou)),
            "q01": float(np.quantile(after_iou, 0.01)),
            "q05": float(np.quantile(after_iou, 0.05)),
        },
        "iou_delta_mean": float(np.mean(delta)),
        "iou_improved_rows": int(np.count_nonzero(delta > 1e-6)),
        "iou_degraded_rows": int(np.count_nonzero(delta < -1e-6)),
        "iou_unchanged_rows": int(np.count_nonzero(np.abs(delta) <= 1e-6)),
        "iou_degraded_001_rows": int(np.count_nonzero(delta < -0.01)),
        "iou_degraded_005_rows": int(np.count_nonzero(delta < -0.05)),
        "recall_before_min": float(np.min(before_recall)),
        "recall_after_min": float(np.min(after_recall)),
        "recall_before_violations": int(np.count_nonzero(before_recall + 1e-12 < recall_floor)),
        "recall_after_violations": int(np.count_nonzero(after_recall + 1e-12 < recall_floor)),
        "area_ratio_before": {
            "mean": float(np.mean(before_area)),
            "q95": float(np.quantile(before_area, 0.95)),
            "q99": float(np.quantile(before_area, 0.99)),
            "max": float(np.max(before_area)),
        },
        "area_ratio_after": {
            "mean": float(np.mean(after_area)),
            "q95": float(np.quantile(after_area, 0.95)),
            "q99": float(np.quantile(after_area, 0.99)),
            "max": float(np.max(after_area)),
        },
    }


def _compare_label(
    baseline_root: Path,
    vote_root: Path,
    label: str,
    recall_floor: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    before_keys = _load_keys(baseline_root / label / "runtime/opt/final_keyframes.json")
    after_keys = _load_keys(vote_root / label / "runtime/opt/final_keyframes.json")
    if set(before_keys) != set(after_keys):
        missing = sorted(set(before_keys) - set(after_keys))[:5]
        added = sorted(set(after_keys) - set(before_keys))[:5]
        raise ValueError(f"pair-vote changed key path for {label}: missing={missing}, added={added}")

    key_rows = []
    all_vertex_moves = []
    for identity in sorted(before_keys):
        before_polygons = before_keys[identity]["polygons"]
        after_polygons = after_keys[identity]["polygons"]
        before_points, after_points = _flatten_matching(before_polygons, after_polygons)
        distances = np.linalg.norm(after_points - before_points, axis=1)
        all_vertex_moves.extend(distances.tolist())
        before_area = _area(before_polygons)
        after_area = _area(after_polygons)
        radius = math.sqrt(max(before_area, 1e-9) / math.pi)
        centroid_shift = float(
            np.linalg.norm(np.mean(after_points, axis=0) - np.mean(before_points, axis=0))
        ) if len(before_points) else 0.0
        before_geometry = _geometry(before_polygons)
        after_geometry = _geometry(after_polygons)
        union_area = float(before_geometry.union(after_geometry).area)
        shape_iou = (
            float(before_geometry.intersection(after_geometry).area) / union_area
            if union_area > 0.0 else 1.0
        )
        key_rows.append(
            {
                "label": label,
                "track_id": identity[0],
                "run_id": identity[1],
                "frame": identity[2],
                "vertex_move_mean_px": float(np.mean(distances)) if len(distances) else 0.0,
                "vertex_count": int(len(distances)),
                "vertex_move_sum_px": float(np.sum(distances)),
                "vertex_move_q95_px": float(np.quantile(distances, 0.95)) if len(distances) else 0.0,
                "vertex_move_max_px": float(np.max(distances)) if len(distances) else 0.0,
                "vertex_move_mean_over_radius": float(np.mean(distances) / radius) if len(distances) else 0.0,
                "centroid_shift_px": centroid_shift,
                "centroid_shift_over_radius": centroid_shift / radius,
                "area_after_over_before": after_area / max(before_area, 1e-9),
                "shape_iou_before_after": shape_iou,
                "before_valid": _polygons_valid(before_polygons),
                "after_valid": _polygons_valid(after_polygons),
            }
        )

    before_metrics = _load_metrics(baseline_root / label / "runtime/exact/keyframe_exact_metrics.csv")
    after_metrics = _load_metrics(vote_root / label / "runtime/exact/keyframe_exact_metrics.csv")
    changed = [row for row in key_rows if row["vertex_move_max_px"] > 1e-6]
    area_ratios = [row["area_after_over_before"] for row in key_rows]
    movement = [row["vertex_move_mean_px"] for row in key_rows]
    movement_relative = [row["vertex_move_mean_over_radius"] for row in key_rows]
    centroid_relative = [row["centroid_shift_over_radius"] for row in key_rows]
    shape_ious = [row["shape_iou_before_after"] for row in key_rows]
    baseline_metrics = json.loads((baseline_root / label / "metrics.json").read_text(encoding="utf-8"))
    vote_metrics = json.loads((vote_root / label / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "label": label,
        "keyframe_path_identical": True,
        "keyframes": len(key_rows),
        "changed_keyframes": len(changed),
        "changed_keyframe_rate": len(changed) / max(len(key_rows), 1),
        "vertex_move_px": {
            "mean": float(np.mean(all_vertex_moves)),
            "median": _quantile(all_vertex_moves, 0.5),
            "q95": _quantile(all_vertex_moves, 0.95),
            "q99": _quantile(all_vertex_moves, 0.99),
            "max": max(all_vertex_moves, default=0.0),
        },
        "key_mean_vertex_move_px": {
            "median": _quantile(movement, 0.5),
            "q95": _quantile(movement, 0.95),
            "max": max(movement, default=0.0),
        },
        "key_mean_vertex_move_over_radius": {
            "median": _quantile(movement_relative, 0.5),
            "q95": _quantile(movement_relative, 0.95),
            "max": max(movement_relative, default=0.0),
        },
        "centroid_shift_over_radius": {
            "median": _quantile(centroid_relative, 0.5),
            "q95": _quantile(centroid_relative, 0.95),
            "max": max(centroid_relative, default=0.0),
        },
        "area_after_over_before": {
            "median": _quantile(area_ratios, 0.5, 1.0),
            "q05": _quantile(area_ratios, 0.05, 1.0),
            "q95": _quantile(area_ratios, 0.95, 1.0),
            "min": min(area_ratios, default=1.0),
            "max": max(area_ratios, default=1.0),
        },
        "shape_iou_before_after": {
            "median": _quantile(shape_ious, 0.5, 1.0),
            "q05": _quantile(shape_ious, 0.05, 1.0),
            "min": min(shape_ious, default=1.0),
        },
        "invalid_polygons_before": sum(not row["before_valid"] for row in key_rows),
        "invalid_polygons_after": sum(not row["after_valid"] for row in key_rows),
        "dense": _metric_summary(before_metrics, after_metrics, recall_floor=recall_floor, keyframes_only=False),
        "at_keyframes": _metric_summary(before_metrics, after_metrics, recall_floor=recall_floor, keyframes_only=True),
        "runtime": {
            "baseline_wall_seconds": float(baseline_metrics["wall_seconds"]),
            "pair_vote_wall_seconds": float(vote_metrics["wall_seconds"]),
            "pair_vote_stage_seconds": float(vote_metrics["pair_vote_seconds"]),
        },
    }
    return summary, key_rows


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# best_v4 pair-vote ablation",
        "",
        "Pair-vote is the only semantic difference. Keyframe paths are required to match.",
        "",
        "| class | keys | moved | vertex mean px | vertex q95 px | dense IoU delta | Recall violations after | pair-vote seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["classes"]:
        lines.append(
            f"| {row['label']} | {row['keyframes']} | {row['changed_keyframes']} "
            f"| {row['vertex_move_px']['mean']:.4f} | {row['vertex_move_px']['q95']:.4f} "
            f"| {row['dense']['iou_delta_mean']:+.6f} "
            f"| {row['dense']['recall_after_violations']} "
            f"| {row['runtime']['pair_vote_stage_seconds']:.4f} |"
        )
    overall = report["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- keyframes: {overall['keyframes']}",
            f"- changed keyframes: {overall['changed_keyframes']} ({overall['changed_keyframe_rate']:.2%})",
            f"- vertex movement mean / key-mean q95 / maximum: {overall['vertex_move_px']['mean']:.4f} / {overall['vertex_move_px']['key_mean_q95']:.4f} / {overall['vertex_move_px']['max']:.4f} px",
            f"- key shape IoU before/after median/q05/min: {overall['shape_iou_before_after']['median']:.6f} / {overall['shape_iou_before_after']['q05']:.6f} / {overall['shape_iou_before_after']['min']:.6f}",
            f"- dense mean IoU delta: {overall['dense']['iou_delta_mean']:+.6f}",
            f"- dense Recall violations after: {overall['dense']['recall_after_violations']}",
            f"- area after/before q05/median/q95: {overall['area_after_over_before']['q05']:.6f} / {overall['area_after_over_before']['median']:.6f} / {overall['area_after_over_before']['q95']:.6f}",
            "",
            "## Largest key movements",
            "",
            "| class | track | run | frame | mean px | max px | centroid/radius | area ratio | shape IoU |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["largest_key_movements"][:20]:
        lines.append(
            f"| {row['label']} | {row['track_id']} | {row['run_id']} | {row['frame']} "
            f"| {row['vertex_move_mean_px']:.3f} | {row['vertex_move_max_px']:.3f} "
            f"| {row['centroid_shift_over_radius']:.4f} | {row['area_after_over_before']:.4f} "
            f"| {row['shape_iou_before_after']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--pair-vote-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--recall-floor", type=float, default=0.97)
    parser.add_argument("--labels", default=",".join(LABELS))
    args = parser.parse_args()
    labels = tuple(value.strip() for value in args.labels.split(",") if value.strip())
    if not labels or any(label not in LABELS for label in labels):
        raise ValueError(f"labels must be selected from {LABELS}")
    summaries = []
    all_key_rows = []
    for label in labels:
        summary, key_rows = _compare_label(
            args.baseline_root, args.pair_vote_root, label, args.recall_floor
        )
        summaries.append(summary)
        all_key_rows.extend(key_rows)

    # Pool exact rows rather than averaging class quantiles.
    before_metrics = {}
    after_metrics = {}
    for label in labels:
        for identity, values in _load_metrics(
            args.baseline_root / label / "runtime/exact/keyframe_exact_metrics.csv"
        ).items():
            before_metrics[(label, *identity)] = values
        for identity, values in _load_metrics(
            args.pair_vote_root / label / "runtime/exact/keyframe_exact_metrics.csv"
        ).items():
            after_metrics[(label, *identity)] = values
    key_mean_vertex_moves = [float(row["vertex_move_mean_px"]) for row in all_key_rows]
    vertex_count = sum(int(row["vertex_count"]) for row in all_key_rows)
    vertex_move_sum = sum(float(row["vertex_move_sum_px"]) for row in all_key_rows)
    area_ratios = [float(row["area_after_over_before"]) for row in all_key_rows]
    shape_ious = [float(row["shape_iou_before_after"]) for row in all_key_rows]
    overall = {
        "keyframes": len(all_key_rows),
        "changed_keyframes": sum(float(row["vertex_move_max_px"]) > 1e-6 for row in all_key_rows),
        "vertex_move_px": {
            "mean": vertex_move_sum / max(vertex_count, 1),
            "key_mean_q95": _quantile(key_mean_vertex_moves, 0.95),
            "max": max(float(row["vertex_move_max_px"]) for row in all_key_rows),
        },
        "area_after_over_before": {
            "q05": _quantile(area_ratios, 0.05, 1.0),
            "median": _quantile(area_ratios, 0.5, 1.0),
            "q95": _quantile(area_ratios, 0.95, 1.0),
            "min": min(area_ratios, default=1.0),
            "max": max(area_ratios, default=1.0),
        },
        "shape_iou_before_after": {
            "median": _quantile(shape_ious, 0.5, 1.0),
            "q05": _quantile(shape_ious, 0.05, 1.0),
            "min": min(shape_ious, default=1.0),
        },
        "dense": _metric_summary(before_metrics, after_metrics, recall_floor=args.recall_floor, keyframes_only=False),
        "at_keyframes": _metric_summary(before_metrics, after_metrics, recall_floor=args.recall_floor, keyframes_only=True),
        "invalid_polygons_before": sum(not row["before_valid"] for row in all_key_rows),
        "invalid_polygons_after": sum(not row["after_valid"] for row in all_key_rows),
    }
    overall["changed_keyframe_rate"] = overall["changed_keyframes"] / max(overall["keyframes"], 1)
    report = {
        "schema_version": 1,
        "privacy": "SQLite polygon geometry and exact metric CSV only; no video frame was opened.",
        "baseline": str(args.baseline_root),
        "pair_vote": str(args.pair_vote_root),
        "recall_floor": args.recall_floor,
        "classes": summaries,
        "overall": overall,
        "largest_key_movements": sorted(
            all_key_rows,
            key=lambda row: (float(row["vertex_move_mean_px"]), float(row["vertex_move_max_px"])),
            reverse=True,
        )[:50],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output_json), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
