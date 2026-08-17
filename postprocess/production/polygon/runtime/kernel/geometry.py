"""Polygon geometry, temporal contour alignment, and exact raster metrics."""

from __future__ import annotations

import itertools
import json
import math

import cv2
import numpy as np

from .types import SimilarityTransform, TrackRow


def normalize_closed_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= 1:
        return pts.copy()
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts.astype(np.float32, copy=True)


def parse_polygons(polygons_json: str) -> list[np.ndarray]:
    polygons = json.loads(str(polygons_json))
    out: list[np.ndarray] = []
    for poly in polygons:
        arr = normalize_closed_points(np.asarray(poly, dtype=np.float32).reshape(-1, 2))
        if len(arr) >= 3:
            out.append(arr)
    return out


def signed_area(poly: np.ndarray) -> float:
    pts = normalize_closed_points(poly)
    if len(pts) < 3:
        return 0.0
    xs = pts[:, 0]
    ys = pts[:, 1]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def polygon_area(poly: np.ndarray) -> float:
    return abs(signed_area(poly))


def orient_ccw(poly: np.ndarray) -> np.ndarray:
    pts = normalize_closed_points(poly)
    if len(pts) < 3:
        return pts
    if signed_area(pts) < 0.0:
        return pts[::-1].copy()
    return pts


def contour_centroid(poly: np.ndarray) -> np.ndarray:
    pts = normalize_closed_points(poly)
    if len(pts) == 0:
        return np.zeros((2,), dtype=np.float32)
    return np.mean(pts, axis=0).astype(np.float32)


def sort_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
    normalized = [
        orient_ccw(poly) for poly in polygons if len(normalize_closed_points(poly)) >= 3
    ]
    normalized.sort(key=lambda poly: (polygon_area(poly), len(poly)), reverse=True)
    return normalized


def cyclic_shift_points(poly: np.ndarray, shift: int) -> np.ndarray:
    pts = normalize_closed_points(poly)
    if len(pts) == 0:
        return pts
    return np.roll(pts, -int(shift), axis=0)


def align_polygon_phase(reference: np.ndarray | None, poly: np.ndarray) -> np.ndarray:
    candidate = orient_ccw(poly)
    if reference is None or len(reference) != len(candidate):
        return candidate
    best = candidate
    best_score = float("inf")
    for variant in (candidate, candidate[::-1].copy()):
        for shift in range(len(variant)):
            rolled = cyclic_shift_points(variant, shift)
            score = float(np.mean(np.sum((rolled - reference) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best = rolled
    return best


def resample_closed_contour(poly: np.ndarray, n_points: int) -> np.ndarray:
    pts = normalize_closed_points(poly)
    n_points = max(3, int(n_points))
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n_points, axis=0).astype(np.float32)
    nxt = np.roll(pts, -1, axis=0)
    seg_lens = np.linalg.norm(nxt - pts, axis=1)
    total = float(seg_lens.sum())
    if total <= 1e-6:
        return np.repeat(pts[:1], n_points, axis=0).astype(np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    sample_pos = np.linspace(0.0, total, n_points, endpoint=False, dtype=np.float64)
    out = np.zeros((n_points, 2), dtype=np.float32)
    for idx, dist in enumerate(sample_pos):
        seg_idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(pts) - 1))
        seg_start = cumulative[seg_idx]
        seg_len = max(float(seg_lens[seg_idx]), 1e-6)
        alpha = float((dist - seg_start) / seg_len)
        out[idx] = ((1.0 - alpha) * pts[seg_idx] + alpha * nxt[seg_idx]).astype(
            np.float32
        )
    return out


def align_contour_slots(
    prev: list[np.ndarray] | None, current: list[np.ndarray]
) -> list[np.ndarray]:
    current_sorted = sort_polygons(current)
    if prev is None or len(prev) != len(current_sorted):
        return current_sorted
    count = len(current_sorted)
    best_perm = list(range(count))
    best_cost = float("inf")
    prev_centroids = [contour_centroid(poly) for poly in prev]
    prev_areas = [polygon_area(poly) for poly in prev]
    curr_centroids = [contour_centroid(poly) for poly in current_sorted]
    curr_areas = [polygon_area(poly) for poly in current_sorted]
    for perm in itertools.permutations(range(count)):
        cost = 0.0
        for idx, src in enumerate(perm):
            center_term = float(
                np.linalg.norm(prev_centroids[idx] - curr_centroids[src])
            )
            area_term = abs(
                math.log(max(curr_areas[src], 1e-6) / max(prev_areas[idx], 1e-6))
            )
            cost += center_term + 8.0 * area_term
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)
    return [current_sorted[idx] for idx in best_perm]


def build_local_mask_from_polygons(polygons: list[np.ndarray]) -> np.ndarray:
    valid_polys = [
        np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        for poly in polygons
        if len(poly) >= 3
    ]
    if not valid_polys:
        return np.zeros((1, 1), dtype=np.uint8)
    all_pts = np.concatenate(valid_polys, axis=0)
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    shift_xy = min_xy.astype(np.float32)
    shape = (int(max_xy[1] - min_xy[1] + 1), int(max_xy[0] - min_xy[0] + 1))
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in valid_polys:
        pts_i32 = np.round(poly - shift_xy[None, :]).astype(np.int32)
        if len(pts_i32) >= 3:
            cv2.fillPoly(mask, [pts_i32], 1)
    return mask


def interpolate_gapfill_polygons(
    left_slots: list[np.ndarray],
    right_slots: list[np.ndarray],
    *,
    step: int,
    gap: int,
    temp_points: int,
) -> list[np.ndarray]:
    alpha = float(step) / float(gap + 1)
    out: list[np.ndarray] = []
    for left_poly, right_poly in zip(left_slots, right_slots, strict=False):
        left_anchor = resample_closed_contour(orient_ccw(left_poly), int(temp_points))
        right_anchor = resample_closed_contour(orient_ccw(right_poly), int(temp_points))
        right_anchor = align_polygon_phase(left_anchor, right_anchor)
        interp = ((1.0 - alpha) * left_anchor + alpha * right_anchor).astype(np.float32)
        out.append(interp)
    return out


def build_track_segments_with_gapfill(
    rows: list[TrackRow],
    *,
    max_gap: int,
    temp_points: int,
) -> tuple[list[list[TrackRow]], dict[str, int]]:
    by_track: dict[str, list[TrackRow]] = {}
    for row in rows:
        by_track.setdefault(row.track_id, []).append(row)
    segments: list[list[TrackRow]] = []
    stats = {
        "source_tracks": int(len(by_track)),
        "source_rows": int(len(rows)),
        "gapfill_inserted_frames": 0,
        "gapfill_events": 0,
        "hard_split_events": 0,
    }
    for track_id, track_rows in sorted(by_track.items(), key=lambda item: int(item[0])):
        track_rows.sort(key=lambda row: row.frame)
        current_segment: list[TrackRow] = []
        prev: TrackRow | None = None
        for row in track_rows:
            current_slots = sort_polygons(row.polygons)
            if prev is None:
                current_segment = [
                    TrackRow(
                        frame=int(row.frame),
                        track_id=str(row.track_id),
                        polygons=[
                            np.asarray(poly, dtype=np.float32) for poly in current_slots
                        ],
                        is_gapfill=bool(row.is_gapfill),
                    )
                ]
                prev = current_segment[-1]
                continue

            prev_slots = sort_polygons(prev.polygons)
            same_contour_count = len(prev_slots) == len(current_slots)
            gap = int(row.frame) - int(prev.frame) - 1
            if same_contour_count:
                current_slots = align_contour_slots(prev_slots, current_slots)

            if gap <= 0 and same_contour_count:
                current_segment.append(
                    TrackRow(
                        frame=int(row.frame),
                        track_id=str(row.track_id),
                        polygons=[
                            np.asarray(poly, dtype=np.float32) for poly in current_slots
                        ],
                        is_gapfill=bool(row.is_gapfill),
                    )
                )
                prev = current_segment[-1]
                continue

            can_gapfill = same_contour_count and gap > 0 and gap <= int(max_gap)
            if can_gapfill:
                for step in range(1, gap + 1):
                    interp_polys = interpolate_gapfill_polygons(
                        prev_slots,
                        current_slots,
                        step=step,
                        gap=gap,
                        temp_points=int(temp_points),
                    )
                    current_segment.append(
                        TrackRow(
                            frame=int(prev.frame) + step,
                            track_id=str(track_id),
                            polygons=[
                                np.asarray(poly, dtype=np.float32)
                                for poly in interp_polys
                            ],
                            is_gapfill=True,
                        )
                    )
                stats["gapfill_events"] += 1
                stats["gapfill_inserted_frames"] += int(gap)
                current_segment.append(
                    TrackRow(
                        frame=int(row.frame),
                        track_id=str(row.track_id),
                        polygons=[
                            np.asarray(poly, dtype=np.float32) for poly in current_slots
                        ],
                        is_gapfill=bool(row.is_gapfill),
                    )
                )
                prev = current_segment[-1]
                continue

            if current_segment:
                segments.append(current_segment)
            stats["hard_split_events"] += 1
            current_segment = [
                TrackRow(
                    frame=int(row.frame),
                    track_id=str(row.track_id),
                    polygons=[
                        np.asarray(poly, dtype=np.float32)
                        for poly in sort_polygons(row.polygons)
                    ],
                    is_gapfill=bool(row.is_gapfill),
                )
            ]
            prev = current_segment[-1]

        if current_segment:
            segments.append(current_segment)
    stats["segment_count"] = int(len(segments))
    return segments, stats


def rasterize_mask_from_polygons(
    polygons: list[np.ndarray],
    shape: tuple[int, int],
    shift_xy: np.ndarray,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    shift = np.asarray(shift_xy, dtype=np.float32)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        pts_i32 = np.round(pts - shift[None, :]).astype(np.int32)
        cv2.fillPoly(mask, [pts_i32], 1)
    return mask


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def apply_similarity_transform(
    points: np.ndarray, transform: SimilarityTransform
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    rot = _rotation_matrix(float(transform.angle_rad))
    out = float(transform.scale) * (pts @ rot.T) + np.asarray(
        transform.translation, dtype=np.float64
    )
    return out.astype(np.float32)


def estimate_similarity_transform(
    src: np.ndarray, dst: np.ndarray
) -> SimilarityTransform:
    src_pts = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst_pts = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return SimilarityTransform(
            scale=1.0, angle_rad=0.0, translation=np.zeros((2,), dtype=np.float64)
        )
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    src_var = float(np.sum(src_centered**2) / max(len(src_pts), 1))
    if len(src_pts) < 2 or src_var <= 1e-9:
        return SimilarityTransform(
            scale=1.0,
            angle_rad=0.0,
            translation=(dst_mean - src_mean).astype(np.float64),
        )
    cov = (dst_centered.T @ src_centered) / float(len(src_pts))
    u, singular_vals, vt = np.linalg.svd(cov)
    sign_fix = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign_fix[-1, -1] = -1.0
    rot = u @ sign_fix @ vt
    scale = float(np.trace(np.diag(singular_vals) @ sign_fix) / max(src_var, 1e-9))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    translation = dst_mean - scale * (rot @ src_mean)
    angle_rad = float(math.atan2(rot[1, 0], rot[0, 0]))
    return SimilarityTransform(
        scale=scale, angle_rad=angle_rad, translation=translation.astype(np.float64)
    )


def similarity_residuals(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, SimilarityTransform]:
    transform = estimate_similarity_transform(src, dst)
    aligned_src = apply_similarity_transform(src, transform)
    residual = np.asarray(dst, dtype=np.float64) - np.asarray(
        aligned_src, dtype=np.float64
    )
    return residual.astype(np.float32), transform


def compute_exact_metrics_from_polygons(
    gt_polys: list[np.ndarray], pred_polys: list[np.ndarray]
) -> dict[str, float]:
    if not gt_polys and not pred_polys:
        return {
            "gt_area": 0.0,
            "pred_area": 0.0,
            "intersection": 0.0,
            "union": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "iou": 1.0,
        }
    all_polys = [
        np.asarray(poly, dtype=np.float32)
        for poly in gt_polys + pred_polys
        if len(poly) >= 3
    ]
    if not all_polys:
        return {
            "gt_area": 0.0,
            "pred_area": 0.0,
            "intersection": 0.0,
            "union": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "iou": 1.0,
        }
    all_pts = np.concatenate(all_polys, axis=0)
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    shift_xy = min_xy.astype(np.float32)
    shape = (int(max_xy[1] - min_xy[1] + 1), int(max_xy[0] - min_xy[0] + 1))
    gt_mask = rasterize_mask_from_polygons(gt_polys, shape, shift_xy)
    pred_mask = rasterize_mask_from_polygons(pred_polys, shape, shift_xy)
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    intersection = int((gt_mask & pred_mask).sum())
    union = int(gt_area + pred_area - intersection)
    recall = intersection / gt_area if gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def compute_weighted_error(metrics: dict[str, float]) -> int:
    fn_pixels = int(round(float(metrics["gt_area"]) - float(metrics["intersection"])))
    fp_pixels = int(round(float(metrics["pred_area"]) - float(metrics["intersection"])))
    return int(2 * fn_pixels + fp_pixels)


__all__ = (
    "align_contour_slots",
    "align_polygon_phase",
    "apply_similarity_transform",
    "build_local_mask_from_polygons",
    "build_track_segments_with_gapfill",
    "compute_exact_metrics_from_polygons",
    "compute_weighted_error",
    "contour_centroid",
    "cyclic_shift_points",
    "estimate_similarity_transform",
    "interpolate_gapfill_polygons",
    "normalize_closed_points",
    "orient_ccw",
    "parse_polygons",
    "polygon_area",
    "rasterize_mask_from_polygons",
    "resample_closed_contour",
    "signed_area",
    "similarity_residuals",
    "sort_polygons",
)
