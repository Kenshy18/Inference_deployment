"""Numerical kernel for track-first adaptive polygon optimization.

The kernel performs:

- dense polygon input
- short-gap polygon gapfill inside each track
- track-first AI point-count prediction after gapfill
- track-segment anchor-count fixing via p90 + 1
- contour resampling / phase alignment per track segment
- raw-only per-frame shape state
- candidate-frame pooling via saliency + surrogate path
- penalty shortest-path DP with exact recall budget
- interval endpoint vote refinement (pair-vote)
- exact recall repair
- exact mask evaluation

The promoted implementation intentionally excludes:

- multi-candidate per-frame shape proposals
- interval-synthesized endpoint candidates
- soft-raster fitting
- joint keyframe gradient refinement
- local search
- polish passes
- exact-K main solver
- proxy-fast recall mode

The module is private numerical infrastructure. Pipeline orchestration,
configuration, candidate construction, topology guards, pair-vote policy and
artifact publication are owned by responsibility-specific Production modules.
"""

import argparse
import bisect
import concurrent.futures
import csv
import itertools
import json
import math
import multiprocessing
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

# Stable numerical defaults for the raw-only kernel.
DEFAULT_ANCHORS_PER_CONTOUR = 48
DEFAULT_RECALL_MIN = 0.97
DEFAULT_MAX_GAP = 30
DEFAULT_DP_EVAL_SCALE = 1.0
DEFAULT_DP_EVAL_PAD = 8
DEFAULT_SURROGATE_POOL_FACTOR = 2.0
DEFAULT_SURROGATE_PEAK_FACTOR = 1.2
DEFAULT_SURROGATE_NEIGHBOR_RADIUS = 1
DEFAULT_SURROGATE_SHAPE_WEIGHT = 0.15
DEFAULT_SALIENCY_SHAPE_ETA = 0.5
DEFAULT_SALIENCY_AREA_ETA = 0.45
DEFAULT_PENALTY_BINARY_STEPS = 12
DEFAULT_PENALTY_MAX = 1024.0
DEFAULT_RECALL_BUDGET_BINARY_STEPS = 8
DEFAULT_RECALL_BUDGET_MAX_MU = 64.0
DEFAULT_PATH_RECALL_VIOLATION_WEIGHT = 64.0
DEFAULT_SHAPE_SWITCH_WEIGHT = 2.0
DEFAULT_SHAPE_DISTANCE_WEIGHT = 0.4
DEFAULT_SHAPE_UPDATE_THRESHOLD_RATIO = 0.09
DEFAULT_SHAPE_PENALTY_ADAPT_GAIN = 1.25
DEFAULT_SHAPE_DISTANCE_RELIEF = 1.10
DEFAULT_SHAPE_SWITCH_RELIEF = 0.45
DEFAULT_SHAPE_DISTANCE_MIN_SCALE = 0.10
DEFAULT_SHAPE_SWITCH_MIN_SCALE = 0.45
DEFAULT_DYNAMIC_MAX_GAP_FACTOR = 4.0
DEFAULT_INTERVAL_IOU_WEIGHT = 1.0
DEFAULT_EXACT_RECALL_REPAIR_MAX_PASSES = 4
DEFAULT_EXACT_RECALL_REPAIR_TOPK = 3
DEFAULT_EXACT_RECALL_REPAIR_SCALE_DELTAS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
DEFAULT_ADAPTIVE_ANCHOR_COUNTS = False
DEFAULT_ADAPTIVE_POINT_QUANTILE = 0.95
DEFAULT_ADAPTIVE_POINT_OFFSET = 2
DEFAULT_MIN_ANCHORS_PER_CONTOUR = 4
DEFAULT_PREDICTOR_BATCH_SIZE = 256
DEFAULT_PREDICTOR_DEVICE = "cuda"
DEFAULT_GAPFILL_ENABLED = True
DEFAULT_GAPFILL_MAX_GAP = 30
DEFAULT_GAPFILL_TEMP_POINTS = 128
DEFAULT_MAX_RUN_FRAMES = 30000
DEFAULT_RUN_OVERLAP_FRAMES = 900
DEFAULT_POINT_PREDICTOR_MODEL_DIR = (
    ROOT
    / "experiments"
    / "linear_polygon_bezier_workspace_20260410"
    / "output"
    / "mask_point_predictor_wide96_20260411"
)

FEATURE_NAMES = [
    "area",
    "perimeter",
    "bbox_w",
    "bbox_h",
    "area_ratio",
    "compactness",
    "aspect_ratio",
    "extent",
    "solidity",
    "components",
    "holes",
    "eccentricity",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production polygon optimizer kernel with gapfill-first track-level anchor counts. "
            "Input is a SQLite file with masks(frame, track_id, polygons). "
            "Each polygons cell is a JSON array of polygons, where each polygon is [[x, y], ...]."
        )
    )
    parser.add_argument("--input-sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=1.0 / 9.0)
    parser.add_argument(
        "--anchors-per-contour",
        type=int,
        default=DEFAULT_ANCHORS_PER_CONTOUR,
        help="Fallback or maximum anchors per contour; runs select up to this cap.",
    )
    parser.add_argument(
        "--adaptive-anchor-counts",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
        help="Predict per-frame polygon counts and fix a run-wise anchor count with p90 + offset.",
    )
    parser.add_argument(
        "--point-predictor-model-dir",
        type=Path,
        default=DEFAULT_POINT_PREDICTOR_MODEL_DIR,
    )
    parser.add_argument(
        "--predictor-device", type=str, default=DEFAULT_PREDICTOR_DEVICE
    )
    parser.add_argument(
        "--predictor-batch-size", type=int, default=DEFAULT_PREDICTOR_BATCH_SIZE
    )
    parser.add_argument(
        "--adaptive-point-quantile", type=float, default=DEFAULT_ADAPTIVE_POINT_QUANTILE
    )
    parser.add_argument(
        "--adaptive-point-offset", type=int, default=DEFAULT_ADAPTIVE_POINT_OFFSET
    )
    parser.add_argument(
        "--min-anchors-per-contour", type=int, default=DEFAULT_MIN_ANCHORS_PER_CONTOUR
    )
    parser.add_argument(
        "--gapfill-enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GAPFILL_ENABLED,
    )
    parser.add_argument("--gapfill-max-gap", type=int, default=DEFAULT_GAPFILL_MAX_GAP)
    parser.add_argument(
        "--gapfill-temp-points", type=int, default=DEFAULT_GAPFILL_TEMP_POINTS
    )
    parser.add_argument("--max-run-frames", type=int, default=DEFAULT_MAX_RUN_FRAMES)
    parser.add_argument(
        "--run-overlap-frames", type=int, default=DEFAULT_RUN_OVERLAP_FRAMES
    )
    parser.add_argument("--recall-min", type=float, default=DEFAULT_RECALL_MIN)
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument("--max-tracks", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--stream-sqlite-rows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--write-pred-sqlite", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def apply_fixed_practical_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.speed_profile = "production_gapfill_track_anchor_count"
    args.solver_mode = "penalty"
    args.recall_constraint_mode = "exact_dp"
    args.proxy_recall_penalty_weight = 0.0
    args.surrogate_pool_factor = DEFAULT_SURROGATE_POOL_FACTOR
    args.surrogate_peak_factor = DEFAULT_SURROGATE_PEAK_FACTOR
    args.surrogate_neighbor_radius = DEFAULT_SURROGATE_NEIGHBOR_RADIUS
    args.surrogate_shape_weight = DEFAULT_SURROGATE_SHAPE_WEIGHT
    args.saliency_shape_eta = DEFAULT_SALIENCY_SHAPE_ETA
    args.saliency_area_eta = DEFAULT_SALIENCY_AREA_ETA
    args.interval_iou_weight = DEFAULT_INTERVAL_IOU_WEIGHT
    args.shape_switch_weight = DEFAULT_SHAPE_SWITCH_WEIGHT
    args.shape_distance_weight = DEFAULT_SHAPE_DISTANCE_WEIGHT
    args.shape_update_threshold_ratio = DEFAULT_SHAPE_UPDATE_THRESHOLD_RATIO
    args.shape_penalty_adapt_gain = DEFAULT_SHAPE_PENALTY_ADAPT_GAIN
    args.shape_distance_relief = DEFAULT_SHAPE_DISTANCE_RELIEF
    args.shape_switch_relief = DEFAULT_SHAPE_SWITCH_RELIEF
    args.shape_distance_min_scale = DEFAULT_SHAPE_DISTANCE_MIN_SCALE
    args.shape_switch_min_scale = DEFAULT_SHAPE_SWITCH_MIN_SCALE
    args.dynamic_max_gap_factor = DEFAULT_DYNAMIC_MAX_GAP_FACTOR
    args.dp_eval_scale = DEFAULT_DP_EVAL_SCALE
    args.dp_eval_pad = DEFAULT_DP_EVAL_PAD
    args.penalty_binary_steps = DEFAULT_PENALTY_BINARY_STEPS
    args.penalty_max = DEFAULT_PENALTY_MAX
    args.recall_budget_binary_steps = DEFAULT_RECALL_BUDGET_BINARY_STEPS
    args.recall_budget_max_mu = DEFAULT_RECALL_BUDGET_MAX_MU
    args.path_recall_violation_weight = DEFAULT_PATH_RECALL_VIOLATION_WEIGHT
    args.pair_vote_refine_enabled = True
    args.exact_recall_repair_enabled = True
    args.exact_recall_repair_max_passes = DEFAULT_EXACT_RECALL_REPAIR_MAX_PASSES
    args.exact_recall_repair_topk = DEFAULT_EXACT_RECALL_REPAIR_TOPK
    args.exact_recall_repair_scale_deltas = ",".join(
        str(v) for v in DEFAULT_EXACT_RECALL_REPAIR_SCALE_DELTAS
    )
    return args


@dataclass
class TrackRow:
    frame: int
    track_id: str
    polygons: list[np.ndarray]
    is_gapfill: bool = False


@dataclass
class SimilarityTransform:
    scale: float
    angle_rad: float
    translation: np.ndarray


@dataclass
class InstanceRun:
    stream_id: str
    track_id: str
    run_id: int
    frame_numbers: np.ndarray
    gt_polygons: list[list[np.ndarray]]
    anchors: np.ndarray
    contour_count: int
    anchors_per_contour: int
    scale: float
    gapfilled_flags: np.ndarray | None = None
    predicted_total_points: np.ndarray | None = None
    run_target_total_points: int = 0
    emit_start_idx: int = 0
    emit_end_idx: int = -1
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_process_start: int = 0
    chunk_process_end: int = -1
    chunked_from_long_run: bool = False


@dataclass
class ShapeCandidate:
    label: str
    vector: np.ndarray
    polygons: list[np.ndarray]
    frame_loss: float
    objective: float
    recall_budget: float = 0.0
    area: float = 0.0
    center: np.ndarray | None = None
    radii: np.ndarray | None = None
    mean_radius: float = 0.0


@dataclass
class IntervalCost:
    cost: float
    shape_distance: float
    shape_update: float
    frames_covered: int
    frame_loss_mean: float = 0.0
    shape_distance_scale: float = 1.0
    shape_switch_scale: float = 1.0
    recall_budget: float = 0.0


@dataclass
class FrameEvalContext:
    gt_mask: np.ndarray
    gt_area: int
    shift_xy: np.ndarray
    shape_hw: tuple[int, int]
    scale_factor: float
    gt_center: np.ndarray
    gt_radii: np.ndarray
    gt_mean_radius: float
    gt_polygon_area: float
    scratch_pred_mask: np.ndarray | None = None
    scratch_intersection_mask: np.ndarray | None = None


def compute_mask_descriptors(mask: np.ndarray) -> dict[str, float | int]:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area = float(binary.sum())
    h, w = binary.shape[:2]
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours or area <= 0.0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "area_ratio": 0.0,
            "compactness": 0.0,
            "aspect_ratio": 1.0,
            "extent": 0.0,
            "solidity": 0.0,
            "components": 0,
            "holes": 0,
            "eccentricity": 0.0,
        }
    outer = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(outer, True))
    x, y, bw, bh = cv2.boundingRect(outer)
    bbox_area = float(max(bw * bh, 1))
    hull = cv2.convexHull(outer)
    hull_area = float(max(cv2.contourArea(hull), 1.0))
    compactness = float((perimeter * perimeter) / max(4.0 * math.pi * area, 1e-6))
    ys, xs = np.nonzero(binary)
    if len(xs) >= 2:
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-6))[::-1]
        eccentricity = float(np.sqrt(max(0.0, 1.0 - float(eigvals[1] / eigvals[0]))))
    else:
        eccentricity = 0.0
    component_count = 0
    hole_count = 0
    if hierarchy is not None:
        for node in hierarchy[0]:
            parent = int(node[3])
            if parent < 0:
                component_count += 1
            else:
                hole_count += 1
    return {
        "area": area,
        "perimeter": perimeter,
        "bbox_w": float(bw),
        "bbox_h": float(bh),
        "area_ratio": float(area / max(h * w, 1)),
        "compactness": compactness,
        "aspect_ratio": float(max(bw, 1) / max(bh, 1)),
        "extent": float(area / bbox_area),
        "solidity": float(area / hull_area),
        "components": int(component_count),
        "holes": int(hole_count),
        "eccentricity": eccentricity,
    }


def resize_mask_with_padding(mask: np.ndarray, image_size: int) -> np.ndarray:
    height, width = mask.shape[:2]
    scale = float(image_size) / float(max(height, width, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    offset_y = (image_size - new_h) // 2
    offset_x = (image_size - new_w) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def build_feature_vector(
    descriptors: dict[str, float | int], means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    values = np.asarray(
        [
            math.log1p(float(descriptors["area"])),
            math.log1p(float(descriptors["perimeter"])),
            math.log1p(float(descriptors["bbox_w"])),
            math.log1p(float(descriptors["bbox_h"])),
            float(descriptors["area_ratio"]),
            float(descriptors["compactness"]),
            math.log1p(float(descriptors["aspect_ratio"])),
            float(descriptors["extent"]),
            float(descriptors["solidity"]),
            float(descriptors["components"]),
            float(descriptors["holes"]),
            float(descriptors["eccentricity"]),
        ],
        dtype=np.float32,
    )
    return (values - means) / stds


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyMaskPointNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        use_feature_branch: bool,
        width_mult: float = 1.0,
        feature_hidden_dim: int = 32,
        head_hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.use_feature_branch = bool(use_feature_branch)

        def ch(value: int) -> int:
            return max(8, int(round(float(value) * float(width_mult))))

        stem_ch = ch(16)
        c1 = ch(24)
        c2 = ch(32)
        c3 = ch(48)
        c4 = ch(64)
        self.stem = ConvBNAct(1, stem_ch, stride=2)
        self.encoder = nn.Sequential(
            ConvBNAct(stem_ch, c1, stride=2),
            ConvBNAct(c1, c2, stride=2),
            ConvBNAct(c2, c3, stride=2),
            ConvBNAct(c3, c4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_head = nn.Sequential(
            nn.Linear(c4, head_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        if self.use_feature_branch:
            self.feature_head = nn.Sequential(
                nn.Linear(feature_dim, feature_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(feature_hidden_dim, feature_hidden_dim),
                nn.SiLU(inplace=True),
            )
            fusion_dim = head_hidden_dim + feature_hidden_dim
        else:
            self.feature_head = None
            fusion_dim = head_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.image_head(x)
        if self.use_feature_branch:
            assert self.feature_head is not None
            x = torch.cat([x, self.feature_head(features)], dim=1)
        return self.classifier(x)


class LearnedPointPredictor:
    def __init__(self, model_dir: Path, device_name: str) -> None:
        ckpt_path = Path(model_dir) / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run_config = checkpoint["run_config"]
        self.label_min = int(run_config["label_min"])
        self.label_max = int(run_config["label_max"])
        self.image_size = int(run_config["image_size"])
        self.use_feature_branch = bool(run_config["feature_branch"])
        self.feature_means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
        self.feature_stds = np.asarray(checkpoint["feature_stds"], dtype=np.float32)
        model = TinyMaskPointNet(
            feature_dim=len(FEATURE_NAMES),
            num_classes=self.label_max - self.label_min + 1,
            use_feature_branch=self.use_feature_branch,
            width_mult=float(run_config.get("width_mult", 1.0)),
            feature_hidden_dim=int(run_config.get("feature_hidden_dim", 32)),
            head_hidden_dim=int(run_config.get("head_hidden_dim", 64)),
            dropout=float(run_config.get("dropout", 0.10)),
        )
        model.load_state_dict(checkpoint["model"])
        if str(device_name).startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(str(device_name))
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device).eval()

    def predict_total_points_batch(
        self,
        masks: list[np.ndarray],
        descriptors_list: list[dict[str, float | int]],
        batch_size: int,
    ) -> list[int]:
        outputs: list[int] = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(masks), batch_size):
            end = min(start + batch_size, len(masks))
            image_list: list[np.ndarray] = []
            feature_list: list[np.ndarray] = []
            for mask, descriptors in zip(
                masks[start:end], descriptors_list[start:end], strict=False
            ):
                resized = resize_mask_with_padding(
                    (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255,
                    self.image_size,
                )
                image_list.append((resized.astype(np.float32) / 255.0)[None, :, :])
                if self.use_feature_branch:
                    feature_list.append(
                        build_feature_vector(
                            descriptors, self.feature_means, self.feature_stds
                        )
                    )
                else:
                    feature_list.append(
                        np.zeros((len(FEATURE_NAMES),), dtype=np.float32)
                    )
            images = torch.from_numpy(np.asarray(image_list, dtype=np.float32)).to(
                self.device
            )
            features = torch.from_numpy(np.asarray(feature_list, dtype=np.float32)).to(
                self.device
            )
            with torch.no_grad():
                logits = self.model(images, features)
                pred_indices = (
                    logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
                )
            outputs.extend(int(idx + self.label_min) for idx in pred_indices.tolist())
        return outputs


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


def load_rows(sqlite_path: Path) -> list[TrackRow]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        raw_rows = conn.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
        ).fetchall()
    finally:
        conn.close()
    rows: list[TrackRow] = []
    for frame, track_id, polygons_json in raw_rows:
        rows.append(
            TrackRow(
                frame=int(frame),
                track_id=str(track_id),
                polygons=parse_polygons(str(polygons_json)),
            )
        )
    return rows


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


def write_csv(
    rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_rows_to_pred_sqlite(
    union_rows: list[dict[str, object]], output_sqlite: Path
) -> None:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists():
        output_sqlite.unlink()
    conn = sqlite3.connect(str(output_sqlite))
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)")
        for row in union_rows:
            cur.execute(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    int(row["frame"]),
                    str(row["track_id"]),
                    json.dumps(row["polygons"], ensure_ascii=False),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def aggregate_exact_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    gt_area = sum(float(row["gt_area"]) for row in rows)
    pred_area = sum(float(row["pred_area"]) for row in rows)
    intersection = sum(float(row["intersection"]) for row in rows)
    union = sum(float(row["union"]) for row in rows)
    weighted_error = sum(float(row["weighted_error"]) for row in rows)
    mean_recall = (
        float(
            np.mean(
                np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
            )
        )
        if rows
        else 1.0
    )
    mean_precision = (
        float(
            np.mean(
                np.asarray([float(row["precision"]) for row in rows], dtype=np.float64)
            )
        )
        if rows
        else 1.0
    )
    mean_iou = (
        float(
            np.mean(np.asarray([float(row["iou"]) for row in rows], dtype=np.float64))
        )
        if rows
        else 1.0
    )
    return {
        "row_count": float(len(rows)),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
        "global_precision": float(intersection / pred_area) if pred_area > 0 else 1.0,
        "global_iou": float(intersection / union) if union > 0 else 1.0,
        "mean_recall": float(mean_recall),
        "mean_precision": float(mean_precision),
        "mean_iou": float(mean_iou),
        "weighted_error_total": float(weighted_error),
        "weighted_error_mean": float(weighted_error / max(len(rows), 1)),
    }


def evaluate_union_exact(
    union_rows: list[dict[str, object]], tracked_sqlite: Path, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_lookup = {(int(row["frame"]), str(row["track_id"])): row for row in union_rows}
    result_rows: list[dict[str, object]] = []
    conn = sqlite3.connect(str(tracked_sqlite))
    try:
        cur = conn.cursor()
        for frame, track_id, polygons_json in cur.execute(
            "SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)"
        ):
            key = (int(frame), str(track_id))
            pred = pred_lookup.get(key)
            if pred is None:
                continue
            gt_polys = parse_polygons(str(polygons_json))
            pred_polys = [
                np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                for poly in pred["polygons"]
            ]
            metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
            weighted_error = float(compute_weighted_error(metrics))
            result_rows.append(
                {
                    "frame": int(frame),
                    "track_id": str(track_id),
                    "run_id": int(pred.get("run_id", -1)),
                    "has_keyframe": int(pred.get("has_keyframe", 0)),
                    "gt_area": float(metrics["gt_area"]),
                    "pred_area": float(metrics["pred_area"]),
                    "intersection": float(metrics["intersection"]),
                    "union": float(metrics["union"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "iou": float(metrics["iou"]),
                    "weighted_error": weighted_error,
                }
            )
    finally:
        conn.close()
    metrics_csv = output_dir / "keyframe_exact_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "track_id",
                "run_id",
                "has_keyframe",
                "gt_area",
                "pred_area",
                "intersection",
                "union",
                "recall",
                "precision",
                "iou",
                "weighted_error",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                result_rows,
                key=lambda row: (int(row["frame"]), int(str(row["track_id"]))),
            )
        )
    summary = {
        "input_tracked_sqlite": str(tracked_sqlite),
        "optimized": aggregate_exact_rows(result_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_float_list(text: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        values = list(default)
    return sorted(set(float(v) for v in values))


def split_long_track_segments(
    segments: list[list[TrackRow]],
    max_run_frames: int,
    run_overlap_frames: int,
) -> tuple[list[list[TrackRow]], dict[int, dict[str, int]], dict[str, int]]:
    source_lengths = [int(len(segment)) for segment in segments]
    max_source_segment_frames = int(max(source_lengths, default=0))
    max_frames = int(max_run_frames)
    requested_overlap = max(0, int(run_overlap_frames))
    disabled = max_frames <= 0
    effective_overlap = (
        0 if disabled else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
    )
    emit_stride = 0 if disabled else int(max(1, max_frames - 2 * effective_overlap))

    def make_stats(
        *,
        processed_segment_count: int,
        long_segment_count: int,
        chunked_source_segment_count: int,
        chunk_output_segment_count: int,
        max_processed_segment_frames: int,
        overlap_added_rows: int,
    ) -> dict[str, int]:
        return {
            "max_run_frames": int(max_frames),
            "run_overlap_frames": int(effective_overlap),
            "source_segment_count": int(len(segments)),
            "processed_segment_count": int(processed_segment_count),
            "long_segment_count": int(long_segment_count),
            "chunked_source_segment_count": int(chunked_source_segment_count),
            "chunk_output_segment_count": int(chunk_output_segment_count),
            "max_source_segment_frames": int(max_source_segment_frames),
            "max_processed_segment_frames": int(max_processed_segment_frames),
            "emit_stride_frames": int(emit_stride),
            "overlap_added_rows": int(overlap_added_rows),
        }

    if disabled or max_source_segment_frames <= max_frames:
        return (
            segments,
            {},
            make_stats(
                processed_segment_count=len(segments),
                long_segment_count=sum(
                    1
                    for length in source_lengths
                    if max_frames > 0 and length > max_frames
                ),
                chunked_source_segment_count=0,
                chunk_output_segment_count=0,
                max_processed_segment_frames=max_source_segment_frames,
                overlap_added_rows=0,
            ),
        )

    split_segments: list[list[TrackRow]] = []
    segment_meta: dict[int, dict[str, int]] = {}
    chunked_source_segment_count = 0
    chunk_output_segment_count = 0
    overlap_added_rows = 0
    max_processed_segment_frames = 0

    for source_run_id, segment in enumerate(segments):
        length = int(len(segment))
        if length <= max_frames:
            split_segments.append(segment)
            max_processed_segment_frames = max(max_processed_segment_frames, length)
            continue

        chunk_ranges: list[tuple[int, int, int, int]] = []
        for emit_start in range(0, length, emit_stride):
            emit_end = int(min(length, emit_start + emit_stride))
            if emit_start >= emit_end:
                continue
            process_start = int(max(0, emit_start - effective_overlap))
            process_end = int(min(length, emit_end + effective_overlap))
            chunk_ranges.append((process_start, process_end, int(emit_start), emit_end))

        chunk_count = int(len(chunk_ranges))
        if chunk_count <= 1:
            split_segments.append(segment)
            max_processed_segment_frames = max(max_processed_segment_frames, length)
            continue

        chunked_source_segment_count += 1
        chunk_output_segment_count += chunk_count
        for chunk_index, (
            process_start,
            process_end,
            emit_start,
            emit_end,
        ) in enumerate(chunk_ranges):
            chunk_rows = list(segment[process_start:process_end])
            split_segments.append(chunk_rows)
            segment_meta[id(chunk_rows)] = {
                "source_run_id": int(source_run_id),
                "chunk_index": int(chunk_index),
                "chunk_count": int(chunk_count),
                "process_start": int(process_start),
                "process_end": int(process_end),
                "emit_start": int(emit_start - process_start),
                "emit_end": int(emit_end - process_start),
            }
            processed_len = int(process_end - process_start)
            emitted_len = int(emit_end - emit_start)
            max_processed_segment_frames = max(
                max_processed_segment_frames, processed_len
            )
            overlap_added_rows += max(0, processed_len - emitted_len)

    return (
        split_segments,
        segment_meta,
        make_stats(
            processed_segment_count=len(split_segments),
            long_segment_count=sum(
                1 for length in source_lengths if length > max_frames
            ),
            chunked_source_segment_count=chunked_source_segment_count,
            chunk_output_segment_count=chunk_output_segment_count,
            max_processed_segment_frames=max_processed_segment_frames,
            overlap_added_rows=overlap_added_rows,
        ),
    )


def build_track_streams(
    rows: list[TrackRow],
    anchors_per_contour: int,
    predictor: LearnedPointPredictor | None = None,
    predictor_batch_size: int = DEFAULT_PREDICTOR_BATCH_SIZE,
    adaptive_anchor_counts: bool = DEFAULT_ADAPTIVE_ANCHOR_COUNTS,
    adaptive_point_quantile: float = DEFAULT_ADAPTIVE_POINT_QUANTILE,
    adaptive_point_offset: int = DEFAULT_ADAPTIVE_POINT_OFFSET,
    min_anchors_per_contour: int = DEFAULT_MIN_ANCHORS_PER_CONTOUR,
    gapfill_enabled: bool = DEFAULT_GAPFILL_ENABLED,
    gapfill_max_gap: int = DEFAULT_GAPFILL_MAX_GAP,
    gapfill_temp_points: int = DEFAULT_GAPFILL_TEMP_POINTS,
    max_tracks: int = -1,
    max_run_frames: int = DEFAULT_MAX_RUN_FRAMES,
    run_overlap_frames: int = DEFAULT_RUN_OVERLAP_FRAMES,
) -> tuple[list[InstanceRun], dict[str, int]]:
    if max_tracks > 0:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.track_id] = counts.get(row.track_id, 0) + 1
        allowed_tracks = [
            track_id
            for track_id, _count in sorted(
                counts.items(), key=lambda item: (-item[1], int(item[0]))
            )
        ][: int(max_tracks)]
        allowed = set(allowed_tracks)
        rows = [row for row in rows if row.track_id in allowed]

    if bool(gapfill_enabled):
        segments, segmentation_stats = build_track_segments_with_gapfill(
            rows,
            max_gap=int(gapfill_max_gap),
            temp_points=int(gapfill_temp_points),
        )
    else:
        segments = []
        current: list[TrackRow] = []
        prev: TrackRow | None = None
        for row in rows:
            split = (
                prev is None
                or row.track_id != prev.track_id
                or row.frame != prev.frame + 1
                or len(row.polygons) != len(prev.polygons)
            )
            if split:
                if current:
                    segments.append(current)
                current = [
                    TrackRow(
                        frame=row.frame,
                        track_id=row.track_id,
                        polygons=row.polygons,
                        is_gapfill=row.is_gapfill,
                    )
                ]
            else:
                current.append(
                    TrackRow(
                        frame=row.frame,
                        track_id=row.track_id,
                        polygons=row.polygons,
                        is_gapfill=row.is_gapfill,
                    )
                )
            prev = row
        if current:
            segments.append(current)
        segmentation_stats = {
            "source_tracks": int(len({row.track_id for row in rows})),
            "source_rows": int(len(rows)),
            "gapfill_inserted_frames": 0,
            "gapfill_events": 0,
            "hard_split_events": 0,
            "segment_count": int(len(segments)),
        }

    segments, segment_meta, split_stats = split_long_track_segments(
        segments,
        max_run_frames=int(max_run_frames),
        run_overlap_frames=int(run_overlap_frames),
    )
    segmentation_stats.update(split_stats)

    streams: list[InstanceRun] = []
    for run_id, run_rows in enumerate(segments):
        meta = segment_meta.get(
            id(run_rows),
            {
                "source_run_id": int(run_id),
                "chunk_index": 0,
                "chunk_count": 1,
                "process_start": 0,
                "process_end": int(len(run_rows)),
                "emit_start": 0,
                "emit_end": int(len(run_rows)),
            },
        )
        source_run_id = int(meta["source_run_id"])
        chunk_index = int(meta["chunk_index"])
        chunk_count = int(meta["chunk_count"])
        chunk_suffix = (
            f":chunk{chunk_index + 1}of{chunk_count}" if chunk_count > 1 else ""
        )
        aligned_rows: list[list[np.ndarray]] = []
        gapfilled_flags: list[bool] = []
        prev_slots: list[np.ndarray] | None = None
        for row in run_rows:
            slots = align_contour_slots(prev_slots, row.polygons)
            aligned_rows.append(slots)
            gapfilled_flags.append(bool(row.is_gapfill))
            prev_slots = slots
        contour_count = len(aligned_rows[0]) if aligned_rows else 0
        if contour_count <= 0:
            continue

        predicted_total_points: np.ndarray | None = None
        run_anchor_count = int(anchors_per_contour)
        run_target_total_points = int(contour_count * run_anchor_count)
        if bool(adaptive_anchor_counts) and predictor is not None:
            masks = [build_local_mask_from_polygons(slots) for slots in aligned_rows]
            descriptors_list = [compute_mask_descriptors(mask) for mask in masks]
            predicted_totals = predictor.predict_total_points_batch(
                masks,
                descriptors_list,
                batch_size=int(predictor_batch_size),
            )
            predicted_total_points = np.asarray(predicted_totals, dtype=np.int32)
            quantile_total = int(
                math.ceil(
                    float(
                        np.quantile(
                            predicted_total_points.astype(np.float64),
                            float(adaptive_point_quantile),
                        )
                    )
                )
            )
            run_target_total_points = int(
                max(
                    contour_count * int(min_anchors_per_contour),
                    quantile_total + int(adaptive_point_offset),
                )
            )
            run_anchor_count = int(
                math.ceil(run_target_total_points / max(contour_count, 1))
            )
            run_anchor_count = int(
                np.clip(
                    run_anchor_count,
                    int(min_anchors_per_contour),
                    int(anchors_per_contour),
                )
            )
            run_target_total_points = int(run_anchor_count * contour_count)

        frame_anchor_stack: list[np.ndarray] = []
        frame_polygons: list[list[np.ndarray]] = []
        frame_areas: list[float] = []
        prev_anchors_by_slot: list[np.ndarray | None] = [None] * contour_count
        for slots in aligned_rows:
            contour_anchors: list[np.ndarray] = []
            contour_polygons: list[np.ndarray] = []
            area_sum = 0.0
            for slot_id in range(contour_count):
                poly = np.asarray(orient_ccw(slots[slot_id]), dtype=np.float32)
                anchor = resample_closed_contour(poly, int(run_anchor_count))
                anchor = align_polygon_phase(prev_anchors_by_slot[slot_id], anchor)
                contour_anchors.append(np.asarray(anchor, dtype=np.float32))
                contour_polygons.append(np.asarray(poly, dtype=np.float32))
                area_sum += float(polygon_area(poly))
                prev_anchors_by_slot[slot_id] = np.asarray(anchor, dtype=np.float32)
            frame_anchor_stack.append(np.asarray(contour_anchors, dtype=np.float32))
            frame_polygons.append(contour_polygons)
            frame_areas.append(area_sum)
        scale = float(
            max(
                math.sqrt(
                    max(
                        float(np.median(np.asarray(frame_areas, dtype=np.float64))), 1.0
                    )
                ),
                1.0,
            )
        )
        streams.append(
            InstanceRun(
                stream_id=f"{run_rows[0].track_id}:run{source_run_id}{chunk_suffix}:instance",
                track_id=run_rows[0].track_id,
                run_id=source_run_id,
                frame_numbers=np.asarray(
                    [row.frame for row in run_rows], dtype=np.int32
                ),
                gt_polygons=frame_polygons,
                anchors=np.asarray(frame_anchor_stack, dtype=np.float32),
                contour_count=contour_count,
                anchors_per_contour=int(run_anchor_count),
                scale=scale,
                gapfilled_flags=np.asarray(gapfilled_flags, dtype=np.uint8),
                predicted_total_points=predicted_total_points,
                run_target_total_points=int(run_target_total_points),
                emit_start_idx=int(meta["emit_start"]),
                emit_end_idx=int(meta["emit_end"]),
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_process_start=int(meta["process_start"]),
                chunk_process_end=int(meta["process_end"]),
                chunked_from_long_run=bool(chunk_count > 1),
            )
        )
    segmentation_stats["effective_stream_count"] = int(len(streams))
    return streams, segmentation_stats


def sqlite_allowed_track_ids(sqlite_path: Path, max_tracks: int) -> list[str] | None:
    if int(max_tracks) <= 0:
        return None
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id, count(*) AS n
            FROM masks
            GROUP BY track_id
            ORDER BY n DESC, CAST(track_id AS INTEGER)
            LIMIT ?
            """,
            (int(max_tracks),),
        ).fetchall()
    finally:
        conn.close()
    return [str(track_id) for track_id, _count in rows]


def sqlite_mask_stats_for_tracks(
    sqlite_path: Path, allowed_track_ids: list[str] | None
) -> dict[str, int]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if allowed_track_ids is None:
            row = conn.execute(
                "SELECT count(*), count(DISTINCT track_id) FROM masks"
            ).fetchone()
        elif not allowed_track_ids:
            row = (0, 0)
        else:
            placeholders = ",".join("?" for _ in allowed_track_ids)
            row = conn.execute(
                f"SELECT count(*), count(DISTINCT track_id) FROM masks WHERE track_id IN ({placeholders})",
                tuple(str(track_id) for track_id in allowed_track_ids),
            ).fetchone()
    finally:
        conn.close()
    return {"source_rows": int(row[0] or 0), "source_tracks": int(row[1] or 0)}


def iter_sqlite_track_rows(sqlite_path: Path, allowed_track_ids: list[str] | None):
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if allowed_track_ids is None:
            rows_iter = conn.execute(
                "SELECT frame, track_id, polygons FROM masks ORDER BY CAST(track_id AS INTEGER), frame"
            )
        elif not allowed_track_ids:
            rows_iter = iter(())
        else:
            placeholders = ",".join("?" for _ in allowed_track_ids)
            rows_iter = conn.execute(
                f"SELECT frame, track_id, polygons FROM masks WHERE track_id IN ({placeholders}) ORDER BY CAST(track_id AS INTEGER), frame",
                tuple(str(track_id) for track_id in allowed_track_ids),
            )
        for frame, track_id, polygons_json in rows_iter:
            yield TrackRow(
                frame=int(frame),
                track_id=str(track_id),
                polygons=parse_polygons(str(polygons_json)),
            )
    finally:
        conn.close()


def iter_track_streams_from_sqlite(
    sqlite_path: Path,
    *,
    anchors_per_contour: int,
    predictor: LearnedPointPredictor | None,
    predictor_batch_size: int,
    adaptive_anchor_counts: bool,
    adaptive_point_quantile: float,
    adaptive_point_offset: int,
    min_anchors_per_contour: int,
    gapfill_enabled: bool,
    gapfill_max_gap: int,
    gapfill_temp_points: int,
    max_tracks: int,
    max_run_frames: int,
    run_overlap_frames: int,
    segmentation_stats: dict[str, int],
):
    allowed_track_ids = sqlite_allowed_track_ids(sqlite_path, int(max_tracks))
    source_stats = sqlite_mask_stats_for_tracks(sqlite_path, allowed_track_ids)
    max_frames = int(max_run_frames)
    requested_overlap = max(0, int(run_overlap_frames))
    effective_overlap = (
        0
        if max_frames <= 0
        else int(min(requested_overlap, max(0, (max_frames - 1) // 2)))
    )
    emit_stride = (
        0 if max_frames <= 0 else int(max(1, max_frames - 2 * effective_overlap))
    )
    segmentation_stats.clear()
    segmentation_stats.update(
        {
            "source_tracks": int(source_stats["source_tracks"]),
            "source_rows": int(source_stats["source_rows"]),
            "gapfill_inserted_frames": 0,
            "gapfill_events": 0,
            "hard_split_events": 0,
            "segment_count": 0,
            "max_run_frames": int(max_frames),
            "run_overlap_frames": int(effective_overlap),
            "source_segment_count": 0,
            "processed_segment_count": 0,
            "long_segment_count": 0,
            "chunked_source_segment_count": 0,
            "chunk_output_segment_count": 0,
            "max_source_segment_frames": 0,
            "max_processed_segment_frames": 0,
            "emit_stride_frames": int(emit_stride),
            "overlap_added_rows": 0,
            "effective_stream_count": 0,
        }
    )

    buffer: list[TrackRow] = []
    buffer_start_idx = 0
    segment_len = 0
    next_emit_start = 0
    source_run_id = 0
    chunk_index = 0
    current_track_id: str | None = None
    prev: TrackRow | None = None

    def build_runs_for_chunk(
        chunk_rows: list[TrackRow],
        *,
        emit_start: int,
        emit_end: int,
        process_start: int,
        process_end: int,
        chunk_idx: int,
        chunked: bool,
    ) -> list[InstanceRun]:
        runs, _ignored_stats = build_track_streams(
            chunk_rows,
            anchors_per_contour=int(anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(predictor_batch_size),
            adaptive_anchor_counts=bool(adaptive_anchor_counts),
            adaptive_point_quantile=float(adaptive_point_quantile),
            adaptive_point_offset=int(adaptive_point_offset),
            min_anchors_per_contour=int(min_anchors_per_contour),
            gapfill_enabled=False,
            gapfill_max_gap=int(gapfill_max_gap),
            gapfill_temp_points=int(gapfill_temp_points),
            max_tracks=-1,
            max_run_frames=0,
            run_overlap_frames=0,
            _release_predictor_after_build=False,
        )
        out: list[InstanceRun] = []
        for sub_idx, run in enumerate(runs):
            suffix = f":chunk{chunk_idx + 1}" if bool(chunked) else ""
            extra = f":part{sub_idx + 1}" if len(runs) > 1 else ""
            run.run_id = int(source_run_id)
            run.stream_id = f"{run.track_id}:run{source_run_id}{suffix}{extra}:instance"
            run.emit_start_idx = int(emit_start)
            run.emit_end_idx = int(emit_end)
            run.chunk_index = int(chunk_idx)
            run.chunk_count = -1 if bool(chunked) else 1
            run.chunk_process_start = int(process_start)
            run.chunk_process_end = int(process_end)
            run.chunked_from_long_run = bool(chunked)
            out.append(run)
        return out

    def emit_chunk(
        process_start: int,
        process_end: int,
        emit_start: int,
        emit_end: int,
        *,
        final: bool,
    ) -> list[InstanceRun]:
        nonlocal buffer, buffer_start_idx, next_emit_start, chunk_index
        start_offset = int(process_start - buffer_start_idx)
        end_offset = int(process_end - buffer_start_idx)
        chunk_rows = list(buffer[start_offset:end_offset])
        chunked = bool(segment_len > max_frames and max_frames > 0)
        emitted_len = int(emit_end - emit_start)
        processed_len = int(process_end - process_start)
        segmentation_stats["processed_segment_count"] += 1
        segmentation_stats["max_processed_segment_frames"] = int(
            max(segmentation_stats["max_processed_segment_frames"], processed_len)
        )
        if chunked:
            segmentation_stats["chunk_output_segment_count"] += 1
            segmentation_stats["overlap_added_rows"] += int(
                max(0, processed_len - emitted_len)
            )
        runs = build_runs_for_chunk(
            chunk_rows,
            emit_start=int(emit_start - process_start),
            emit_end=int(emit_end - process_start),
            process_start=int(process_start),
            process_end=int(process_end),
            chunk_idx=int(chunk_index),
            chunked=chunked,
        )
        segmentation_stats["effective_stream_count"] += int(len(runs))
        chunk_index += 1
        next_emit_start = int(emit_end)
        if not final:
            keep_from = int(max(0, next_emit_start - effective_overlap))
            drop_count = int(keep_from - buffer_start_idx)
            if drop_count > 0:
                buffer = buffer[drop_count:]
                buffer_start_idx = keep_from
        return runs

    def emit_ready_chunks(final: bool) -> list[InstanceRun]:
        out: list[InstanceRun] = []
        if segment_len <= 0:
            return out
        if max_frames <= 0:
            if final and next_emit_start < segment_len:
                out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
            return out
        if segment_len <= max_frames:
            if final and next_emit_start < segment_len:
                out.extend(emit_chunk(0, segment_len, 0, segment_len, final=True))
            return out
        while next_emit_start < segment_len:
            emit_start = int(next_emit_start)
            emit_end = int(min(segment_len, emit_start + emit_stride))
            process_start = int(max(0, emit_start - effective_overlap))
            desired_process_end = int(emit_end + effective_overlap)
            if not final and desired_process_end > segment_len:
                break
            process_end = int(min(segment_len, desired_process_end))
            if not final and emit_end >= segment_len:
                break
            out.extend(
                emit_chunk(
                    process_start, process_end, emit_start, emit_end, final=final
                )
            )
            if final:
                continue
        return out

    def flush_segment() -> list[InstanceRun]:
        nonlocal buffer, buffer_start_idx, segment_len, next_emit_start, source_run_id, chunk_index, prev
        if segment_len <= 0:
            return []
        segmentation_stats["segment_count"] += 1
        segmentation_stats["source_segment_count"] += 1
        segmentation_stats["max_source_segment_frames"] = int(
            max(segmentation_stats["max_source_segment_frames"], segment_len)
        )
        if max_frames > 0 and segment_len > max_frames:
            segmentation_stats["long_segment_count"] += 1
            segmentation_stats["chunked_source_segment_count"] += 1
        runs = emit_ready_chunks(final=True)
        source_run_id += 1
        buffer = []
        buffer_start_idx = 0
        segment_len = 0
        next_emit_start = 0
        chunk_index = 0
        prev = None
        return runs

    def append_segment_row(row: TrackRow) -> list[InstanceRun]:
        nonlocal segment_len
        buffer.append(row)
        segment_len += 1
        return emit_ready_chunks(final=False)

    for row in iter_sqlite_track_rows(sqlite_path, allowed_track_ids):
        if current_track_id is not None and str(row.track_id) != current_track_id:
            for run in flush_segment():
                yield run
        if current_track_id != str(row.track_id):
            current_track_id = str(row.track_id)
            prev = None

        current_slots = sort_polygons(row.polygons)
        if prev is None:
            first = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(first):
                yield run
            prev = first
            continue

        prev_slots = sort_polygons(prev.polygons)
        same_contour_count = len(prev_slots) == len(current_slots)
        gap = int(row.frame) - int(prev.frame) - 1
        if same_contour_count:
            current_slots = align_contour_slots(prev_slots, current_slots)

        if (not bool(gapfill_enabled)) and gap > 0:
            for run in flush_segment():
                yield run
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[
                    np.asarray(poly, dtype=np.float32)
                    for poly in sort_polygons(row.polygons)
                ],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        if gap <= 0 and same_contour_count:
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        can_gapfill = (
            bool(gapfill_enabled)
            and same_contour_count
            and gap > 0
            and gap <= int(gapfill_max_gap)
        )
        if can_gapfill:
            for step in range(1, gap + 1):
                interp_polys = interpolate_gapfill_polygons(
                    prev_slots,
                    current_slots,
                    step=step,
                    gap=gap,
                    temp_points=int(gapfill_temp_points),
                )
                gap_row = TrackRow(
                    frame=int(prev.frame) + step,
                    track_id=str(row.track_id),
                    polygons=[
                        np.asarray(poly, dtype=np.float32) for poly in interp_polys
                    ],
                    is_gapfill=True,
                )
                for run in append_segment_row(gap_row):
                    yield run
            segmentation_stats["gapfill_events"] += 1
            segmentation_stats["gapfill_inserted_frames"] += int(gap)
            current = TrackRow(
                frame=int(row.frame),
                track_id=str(row.track_id),
                polygons=[np.asarray(poly, dtype=np.float32) for poly in current_slots],
                is_gapfill=bool(row.is_gapfill),
            )
            for run in append_segment_row(current):
                yield run
            prev = current
            continue

        for run in flush_segment():
            yield run
        segmentation_stats["hard_split_events"] += 1
        current = TrackRow(
            frame=int(row.frame),
            track_id=str(row.track_id),
            polygons=[
                np.asarray(poly, dtype=np.float32)
                for poly in sort_polygons(row.polygons)
            ],
            is_gapfill=bool(row.is_gapfill),
        )
        for run in append_segment_row(current):
            yield run
        prev = current

    for run in flush_segment():
        yield run


def flatten_contours(contours: np.ndarray) -> np.ndarray:
    return np.asarray(contours, dtype=np.float32).reshape(-1, 2)


def split_vector_to_polygons(
    vector: np.ndarray, contour_count: int, anchors_per_contour: int
) -> list[np.ndarray]:
    vec = np.asarray(vector, dtype=np.float32).reshape(
        contour_count, anchors_per_contour, 2
    )
    return [np.asarray(vec[idx], dtype=np.float32) for idx in range(contour_count)]


def vector_proxy_stats(
    vector: np.ndarray, contour_count: int, anchors_per_contour: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    arr = np.asarray(vector, dtype=np.float32).reshape(
        contour_count, anchors_per_contour, 2
    )
    pts = arr.reshape(-1, 2)
    if pts.size <= 0:
        return (
            0.0,
            np.zeros((2,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            1.0,
        )
    center = np.mean(pts, axis=0).astype(np.float32)
    radii = np.linalg.norm(pts - center[None, :], axis=1).astype(np.float32)
    mean_radius = float(max(np.mean(radii, dtype=np.float64), 1e-6))
    x = arr[..., 0].astype(np.float64, copy=False)
    y = arr[..., 1].astype(np.float64, copy=False)
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    area = float(0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1)).sum())
    return area, center, radii, mean_radius


def scale_vector_about_centroid(vector: np.ndarray, scale_mul: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim == 3:
        out = np.zeros_like(arr, dtype=np.float64)
        for idx in range(arr.shape[0]):
            center = np.asarray(np.mean(arr[idx], axis=0), dtype=np.float64)
            out[idx] = center + float(scale_mul) * (arr[idx] - center)
        return out.astype(np.float32)
    pts = arr.reshape(-1, 2)
    center = np.asarray(np.mean(pts, axis=0), dtype=np.float64)
    return (center + float(scale_mul) * (pts - center)).astype(np.float32)


def rasterize_mask_with_context(
    polygons: list[np.ndarray],
    context: FrameEvalContext,
    out_mask: np.ndarray | None = None,
) -> np.ndarray:
    if out_mask is None:
        mask = np.zeros(context.shape_hw, dtype=np.uint8)
    else:
        mask = np.asarray(out_mask, dtype=np.uint8)
        mask.fill(0)
    pts_list: list[np.ndarray] = []
    for poly in polygons:
        pts = (np.asarray(poly, dtype=np.float32) - context.shift_xy[None, :]) * float(
            context.scale_factor
        )
        pts = np.round(pts).astype(np.int32)
        if len(pts) >= 3:
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)
    return mask


def rasterize_interpolated_mask_with_context(
    start_polygons: list[np.ndarray],
    end_polygons: list[np.ndarray],
    alpha: float,
    context: FrameEvalContext,
    out_mask: np.ndarray | None = None,
) -> np.ndarray:
    if out_mask is None:
        mask = np.zeros(context.shape_hw, dtype=np.uint8)
    else:
        mask = np.asarray(out_mask, dtype=np.uint8)
        mask.fill(0)
    pts_list: list[np.ndarray] = []
    alpha32 = np.float32(alpha)
    beta32 = np.float32(1.0) - alpha32
    for start_poly, end_poly in zip(start_polygons, end_polygons):
        start_pts = np.asarray(start_poly, dtype=np.float32)
        end_pts = np.asarray(end_poly, dtype=np.float32)
        pts = (
            beta32 * start_pts + alpha32 * end_pts - context.shift_xy[None, :]
        ) * float(context.scale_factor)
        pts = np.round(pts).astype(np.int32)
        if len(pts) >= 3:
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)
    return mask


def build_frame_eval_contexts(
    run: InstanceRun, args: argparse.Namespace
) -> list[FrameEvalContext]:
    contexts: list[FrameEvalContext] = []
    scale_factor = float(np.clip(float(args.dp_eval_scale), 0.1, 1.0))
    pad = int(max(0, int(args.dp_eval_pad)))
    for frame_idx in range(len(run.frame_numbers)):
        raw_vector = flatten_contours(run.anchors[frame_idx])
        gt_polygon_area, gt_center, gt_radii, gt_mean_radius = vector_proxy_stats(
            raw_vector, run.contour_count, run.anchors_per_contour
        )
        raw_polys = split_vector_to_polygons(
            flatten_contours(run.anchors[frame_idx]),
            run.contour_count,
            run.anchors_per_contour,
        )
        all_polys = [
            np.asarray(poly, dtype=np.float32)
            for poly in run.gt_polygons[frame_idx] + raw_polys
            if len(poly) >= 3
        ]
        if all_polys:
            all_pts = np.concatenate(all_polys, axis=0)
            min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32) - pad
            max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32) + pad
        else:
            min_xy = np.asarray([0, 0], dtype=np.int32)
            max_xy = np.asarray([4, 4], dtype=np.int32)
        shift_xy = min_xy.astype(np.float32)
        width = int(max_xy[0] - min_xy[0] + 1)
        height = int(max_xy[1] - min_xy[1] + 1)
        shape_hw = (
            max(1, int(math.ceil(height * scale_factor))),
            max(1, int(math.ceil(width * scale_factor))),
        )
        context = FrameEvalContext(
            gt_mask=np.zeros(shape_hw, dtype=np.uint8),
            gt_area=0,
            shift_xy=shift_xy,
            shape_hw=shape_hw,
            scale_factor=scale_factor,
            gt_center=np.asarray(gt_center, dtype=np.float32),
            gt_radii=np.asarray(gt_radii, dtype=np.float32),
            gt_mean_radius=float(gt_mean_radius),
            gt_polygon_area=float(gt_polygon_area),
        )
        gt_mask = rasterize_mask_with_context(run.gt_polygons[frame_idx], context)
        contexts.append(
            FrameEvalContext(
                gt_mask=gt_mask,
                gt_area=int(gt_mask.sum()),
                shift_xy=shift_xy,
                shape_hw=shape_hw,
                scale_factor=scale_factor,
                gt_center=np.asarray(gt_center, dtype=np.float32),
                gt_radii=np.asarray(gt_radii, dtype=np.float32),
                gt_mean_radius=float(gt_mean_radius),
                gt_polygon_area=float(gt_polygon_area),
                scratch_pred_mask=np.zeros(shape_hw, dtype=np.uint8),
                scratch_intersection_mask=np.zeros(shape_hw, dtype=np.uint8),
            )
        )
    return contexts


def compute_cached_metrics_from_polygons(
    gt_context: FrameEvalContext, pred_polys: list[np.ndarray]
) -> dict[str, float]:
    pred_mask = rasterize_mask_with_context(
        pred_polys,
        gt_context,
        out_mask=gt_context.scratch_pred_mask,
    )
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection_mask = gt_context.scratch_intersection_mask
    if intersection_mask is None:
        intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
    cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    union = int(gt_context.gt_area + pred_area - intersection)
    recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_context.gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def compute_cached_metrics_from_interpolated_polygons(
    gt_context: FrameEvalContext,
    start_polys: list[np.ndarray],
    end_polys: list[np.ndarray],
    alpha: float,
) -> dict[str, float]:
    pred_mask = rasterize_interpolated_mask_with_context(
        start_polys,
        end_polys,
        alpha,
        gt_context,
        out_mask=gt_context.scratch_pred_mask,
    )
    pred_area = int(cv2.countNonZero(pred_mask))
    intersection_mask = gt_context.scratch_intersection_mask
    if intersection_mask is None:
        intersection_mask = np.zeros(gt_context.shape_hw, dtype=np.uint8)
    cv2.bitwise_and(gt_context.gt_mask, pred_mask, dst=intersection_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    union = int(gt_context.gt_area + pred_area - intersection)
    recall = intersection / gt_context.gt_area if gt_context.gt_area > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return {
        "gt_area": float(gt_context.gt_area),
        "pred_area": float(pred_area),
        "intersection": float(intersection),
        "union": float(union),
        "recall": float(recall),
        "precision": float(precision),
        "iou": float(iou),
    }


def evaluate_frame_vector_loss_budget(
    run: InstanceRun,
    frame_idx: int,
    vector: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, float]:
    pred_polys = split_vector_to_polygons(
        vector, run.contour_count, run.anchors_per_contour
    )
    if eval_contexts is not None:
        metrics = compute_cached_metrics_from_polygons(
            eval_contexts[int(frame_idx)], pred_polys
        )
    else:
        metrics = compute_exact_metrics_from_polygons(
            run.gt_polygons[int(frame_idx)], pred_polys
        )
    return float(frame_accuracy_loss(metrics, args)), float(
        recall_budget_from_metrics(metrics)
    )


def recall_budget_from_metrics(metrics: dict[str, float]) -> float:
    return max(0.0, 1.0 - float(metrics["recall"]))


def recall_budget_limit(frame_count: int, args: argparse.Namespace) -> float:
    recall_min = float(np.clip(float(args.recall_min), 0.0, 1.0))
    return float(max(frame_count, 0)) * max(0.0, 1.0 - recall_min)


def recall_violation(
    total_budget: float, frame_count: int, args: argparse.Namespace
) -> float:
    return max(float(total_budget) - float(recall_budget_limit(frame_count, args)), 0.0)


def frame_accuracy_loss(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return float(args.interval_iou_weight) * (1.0 - float(metrics["iou"]))


def adaptive_shape_penalty_scales(
    frame_loss_mean: float, args: argparse.Namespace
) -> tuple[float, float]:
    mean_loss = max(float(frame_loss_mean), 0.0)
    gain = max(float(args.shape_penalty_adapt_gain), 0.0)
    if gain <= 0.0:
        return 1.0, 1.0
    base = 1.0 + gain * mean_loss
    distance_scale = 1.0 / max(
        base ** max(float(args.shape_distance_relief), 0.0), 1e-6
    )
    switch_scale = 1.0 / max(base ** max(float(args.shape_switch_relief), 0.0), 1e-6)
    distance_scale = max(float(args.shape_distance_min_scale), float(distance_scale))
    switch_scale = max(float(args.shape_switch_min_scale), float(switch_scale))
    return float(distance_scale), float(switch_scale)


def build_frame_candidates(
    run: InstanceRun,
    _contexts: list[object],
    eval_contexts: list[FrameEvalContext],
    args: argparse.Namespace,
) -> list[list[ShapeCandidate]]:
    candidates_by_frame: list[list[ShapeCandidate]] = []
    for idx in range(len(run.frame_numbers)):
        raw_vector = flatten_contours(run.anchors[idx])
        raw_metrics = compute_cached_metrics_from_polygons(
            eval_contexts[idx],
            split_vector_to_polygons(
                raw_vector, run.contour_count, run.anchors_per_contour
            ),
        )
        raw_frame_loss = frame_accuracy_loss(raw_metrics, args)
        raw_area, raw_center, raw_radii, raw_mean_radius = vector_proxy_stats(
            raw_vector, run.contour_count, run.anchors_per_contour
        )
        raw_candidate = ShapeCandidate(
            label="raw",
            vector=np.asarray(raw_vector, dtype=np.float32),
            polygons=split_vector_to_polygons(
                raw_vector, run.contour_count, run.anchors_per_contour
            ),
            frame_loss=float(raw_frame_loss),
            objective=float(raw_frame_loss),
            recall_budget=float(recall_budget_from_metrics(raw_metrics)),
            area=float(raw_area),
            center=np.asarray(raw_center, dtype=np.float32),
            radii=np.asarray(raw_radii, dtype=np.float32),
            mean_radius=float(raw_mean_radius),
        )
        candidates_by_frame.append([raw_candidate])
    return candidates_by_frame


def shape_distance(vector_a: np.ndarray, vector_b: np.ndarray, scale: float) -> float:
    residual, _ = similarity_residuals(
        np.asarray(vector_a, dtype=np.float32), np.asarray(vector_b, dtype=np.float32)
    )
    norms = np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=1)
    return float(np.mean(norms) / max(float(scale), 1.0))


def compute_saliency_scores(
    run: InstanceRun,
    fit_vectors: list[np.ndarray],
    area_series: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    length = len(fit_vectors)
    scores = np.zeros((length,), dtype=np.float64)
    area_scale = max(float(np.mean(np.asarray(area_series, dtype=np.float64))), 1.0)
    for idx in range(1, length - 1):
        prev_vec = np.asarray(fit_vectors[idx - 1], dtype=np.float64)
        cur_vec = np.asarray(fit_vectors[idx], dtype=np.float64)
        next_vec = np.asarray(fit_vectors[idx + 1], dtype=np.float64)
        second = float(
            np.linalg.norm(next_vec - 2.0 * cur_vec + prev_vec)
            / max(float(run.scale), 1.0)
        )
        jump = shape_distance(fit_vectors[idx - 1], fit_vectors[idx + 1], run.scale)
        area_peak = (
            max(
                float(area_series[idx])
                - 0.5 * float(area_series[idx - 1] + area_series[idx + 1]),
                0.0,
            )
            / area_scale
        )
        area_swing = (
            abs(float(area_series[idx + 1]) - float(area_series[idx - 1])) / area_scale
        )
        scores[idx] = (
            second
            + float(args.saliency_shape_eta) * jump
            + float(args.saliency_area_eta) * (area_peak + 0.5 * area_swing)
        )
    if length > 0:
        scores[0] = float(scores[1] if length > 1 else 0.0)
        scores[-1] = float(scores[-2] if length > 1 else 0.0)
    return scores


def compute_surrogate_prefix(
    vectors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(
        [np.asarray(vector, dtype=np.float64).reshape(-1) for vector in vectors],
        dtype=np.float64,
    )
    times = np.arange(q.shape[0], dtype=np.float64)[:, None]
    prefix_q = np.concatenate(
        [np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q, axis=0)], axis=0
    )
    prefix_tq = np.concatenate(
        [np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q * times, axis=0)],
        axis=0,
    )
    prefix_q2 = np.concatenate(
        [np.zeros((1,), dtype=np.float64), np.cumsum(np.sum(q * q, axis=1), axis=0)],
        axis=0,
    )
    return prefix_q, prefix_tq, prefix_q2


def surrogate_interval_cost(
    u: int,
    v: int,
    prefix_q: np.ndarray,
    prefix_tq: np.ndarray,
    prefix_q2: np.ndarray,
    vector_dim: int,
    contour_count: int,
    anchors_per_contour: int,
    scale: float,
    args: argparse.Namespace,
) -> float:
    cost, _start_vec, _end_vec = surrogate_interval_solution(
        u,
        v,
        prefix_q,
        prefix_tq,
        prefix_q2,
        vector_dim,
        contour_count,
        anchors_per_contour,
        scale,
        args,
    )
    return float(cost)


def surrogate_interval_solution(
    u: int,
    v: int,
    prefix_q: np.ndarray,
    prefix_tq: np.ndarray,
    prefix_q2: np.ndarray,
    vector_dim: int,
    contour_count: int,
    anchors_per_contour: int,
    scale: float,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray, np.ndarray]:
    if v <= u:
        zero = np.zeros((vector_dim // 2, 2), dtype=np.float32)
        return 0.0, zero, zero
    h = int(v - u)
    s0 = prefix_q[v + 1] - prefix_q[u]
    s1 = prefix_tq[v + 1] - prefix_tq[u]
    s2 = float(prefix_q2[v + 1] - prefix_q2[u])
    a = float((h + 1) * (2 * h + 1) / (6.0 * h))
    b = float((h + 1) * (h - 1) / (6.0 * h))
    c = float(a)
    gu = (float(v) * s0 - s1) / float(h)
    gv = (s1 - float(u) * s0) / float(h)
    det = max(a * c - b * b, 1e-9)
    avec = (c * gu - b * gv) / det
    bvec = (-b * gu + a * gv) / det
    quad = (
        a * float(np.dot(avec, avec))
        + 2.0 * b * float(np.dot(avec, bvec))
        + c * float(np.dot(bvec, bvec))
    )
    cross = 2.0 * float(np.dot(gu, avec) + np.dot(gv, bvec))
    sse = max(s2 - cross + quad, 0.0)
    start_vec = np.asarray(avec, dtype=np.float32).reshape(vector_dim // 2, 2)
    end_vec = np.asarray(bvec, dtype=np.float32).reshape(vector_dim // 2, 2)
    d = shape_distance(start_vec, end_vec, scale)
    return (
        float(
            sse / max(float(scale) ** 2, 1.0) + float(args.surrogate_shape_weight) * d
        ),
        start_vec,
        end_vec,
    )


def exact_k_dp(cost_fn, nodes: list[int], target_count: int, max_gap: int) -> list[int]:
    node_count = len(nodes)
    target_count = max(2, min(int(target_count), node_count))
    dp = np.full((target_count, node_count), np.inf, dtype=np.float64)
    back = np.full((target_count, node_count), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for used in range(1, target_count):
        for end_pos in range(used, node_count):
            end_node = int(nodes[end_pos])
            min_prev_pos = max(
                used - 1,
                int(bisect.bisect_left(nodes, end_node - int(max_gap), 0, end_pos)),
            )
            best_cost = float("inf")
            best_prev = -1
            for prev_pos in range(min_prev_pos, end_pos):
                prev_node = int(nodes[prev_pos])
                prev_cost = float(dp[used - 1, prev_pos])
                if not np.isfinite(prev_cost):
                    continue
                cand = prev_cost + float(cost_fn(prev_node, end_node))
                if cand < best_cost:
                    best_cost = cand
                    best_prev = int(prev_pos)
            dp[used, end_pos] = best_cost
            back[used, end_pos] = best_prev
    path = [node_count - 1]
    cur_pos = node_count - 1
    cur_used = target_count - 1
    while cur_used > 0:
        cur_pos = int(back[cur_used, cur_pos])
        if cur_pos < 0:
            return [int(nodes[0]), int(nodes[-1])]
        path.append(cur_pos)
        cur_used -= 1
    path.reverse()
    return [int(nodes[pos]) for pos in path]


def build_candidate_frame_pool(
    run: InstanceRun,
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], np.ndarray]:
    raw_vectors = [
        frame_candidates[0].vector for frame_candidates in candidates_by_frame
    ]
    area_series = np.asarray(
        [float(frame_candidates[0].area) for frame_candidates in candidates_by_frame],
        dtype=np.float64,
    )
    scores = compute_saliency_scores(run, raw_vectors, area_series, args)
    length = len(run.frame_numbers)
    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    prefix_q, prefix_tq, prefix_q2 = compute_surrogate_prefix(raw_vectors)
    vector_dim = (
        int(np.asarray(raw_vectors[0], dtype=np.float32).size) if raw_vectors else 0
    )

    surrogate_cost_cache: dict[tuple[int, int], float] = {}

    def surrogate_cost(u: int, v: int) -> float:
        key = (int(u), int(v))
        cached = surrogate_cost_cache.get(key)
        if cached is not None:
            return float(cached)
        cost = surrogate_interval_cost(
            int(u),
            int(v),
            prefix_q,
            prefix_tq,
            prefix_q2,
            vector_dim,
            run.contour_count,
            run.anchors_per_contour,
            run.scale,
            args,
        )
        surrogate_cost_cache[key] = float(cost)
        return float(cost)

    all_nodes = list(range(length))
    surrogate_path = exact_k_dp(
        surrogate_cost, all_nodes, int(target_count), dynamic_max_gap
    )
    pool_target = min(
        length,
        max(
            int(round(float(args.surrogate_pool_factor) * float(target_count))),
            int(math.ceil(math.sqrt(max(length, 1)))),
            int(target_count) + 2,
        ),
    )
    peak_target = min(
        length,
        max(0, int(round(float(args.surrogate_peak_factor) * float(target_count)))),
    )
    peak_ids = [int(idx) for idx in np.argsort(-scores)[:peak_target].tolist()]
    grid = list(range(0, length, max(1, target_interval)))
    if grid[-1] != length - 1:
        grid.append(length - 1)
    pool = {0, length - 1}
    for frame_idx in surrogate_path:
        for delta in range(
            -int(args.surrogate_neighbor_radius),
            int(args.surrogate_neighbor_radius) + 1,
        ):
            cand = int(frame_idx) + int(delta)
            if 0 <= cand < length:
                pool.add(int(cand))
    for frame_idx in peak_ids:
        pool.add(int(frame_idx))
    for frame_idx in grid:
        pool.add(int(frame_idx))
    if len(pool) < int(target_count):
        for frame_idx in np.argsort(-scores).tolist():
            pool.add(int(frame_idx))
            if len(pool) >= int(target_count):
                break
    if len(pool) < pool_target:
        for frame_idx in np.argsort(-scores).tolist():
            pool.add(int(frame_idx))
            if len(pool) >= pool_target:
                break
    return (
        sorted(int(frame_idx) for frame_idx in pool),
        [int(frame_idx) for frame_idx in surrogate_path],
        scores,
    )


def build_ring_second_difference_rtr(
    contour_count: int, anchors_per_contour: int
) -> np.ndarray:
    point_count = int(contour_count) * int(anchors_per_contour)
    dim = int(point_count * 2)
    rows: list[np.ndarray] = []
    for contour_idx in range(int(contour_count)):
        base = contour_idx * int(anchors_per_contour)
        for anchor_idx in range(int(anchors_per_contour)):
            prev_idx = base + ((anchor_idx - 1) % int(anchors_per_contour))
            cur_idx = base + anchor_idx
            next_idx = base + ((anchor_idx + 1) % int(anchors_per_contour))
            for axis in range(2):
                row = np.zeros((dim,), dtype=np.float64)
                row[2 * prev_idx + axis] = 1.0
                row[2 * cur_idx + axis] = -2.0
                row[2 * next_idx + axis] = 1.0
                rows.append(row)
    if not rows:
        return np.zeros((dim, dim), dtype=np.float64)
    mat = np.asarray(rows, dtype=np.float64)
    return mat.T @ mat


def build_interpolation_weights(
    frame_count: int, chosen_frames: list[int]
) -> np.ndarray:
    key_count = int(len(chosen_frames))
    weights = np.zeros((int(frame_count), key_count), dtype=np.float64)
    chosen = [int(v) for v in chosen_frames]
    if key_count <= 0:
        return weights
    for frame_idx in range(int(frame_count)):
        if frame_idx <= chosen[0]:
            weights[frame_idx, 0] = 1.0
            continue
        if frame_idx >= chosen[-1]:
            weights[frame_idx, -1] = 1.0
            continue
        right_pos = next(
            pos for pos, keyframe in enumerate(chosen) if keyframe >= frame_idx
        )
        left_pos = max(0, right_pos - 1)
        left_frame = int(chosen[left_pos])
        right_frame = int(chosen[right_pos])
        if frame_idx == right_frame or right_frame <= left_frame:
            weights[frame_idx, right_pos] = 1.0
        else:
            alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
            weights[frame_idx, left_pos] = 1.0 - alpha
            weights[frame_idx, right_pos] = alpha
    return weights


def pair_vote_refine_keyframe_vectors(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if (
        not bool(getattr(args, "pair_vote_refine_enabled", True))
        or len(chosen_frames) <= 1
    ):
        return np.asarray(keyframe_vectors, dtype=np.float32)
    frame_count = int(len(run.frame_numbers))
    targets = np.asarray(
        [flatten_contours(run.anchors[idx]).reshape(-1) for idx in range(frame_count)],
        dtype=np.float64,
    )
    init = np.asarray(keyframe_vectors, dtype=np.float64).reshape(
        len(chosen_frames), -1
    )
    proposals: list[list[tuple[np.ndarray, float]]] = [[] for _ in chosen_frames]
    eye2 = np.eye(2, dtype=np.float64)
    for left_pos in range(len(chosen_frames) - 1):
        right_pos = left_pos + 1
        u = int(chosen_frames[left_pos])
        v = int(chosen_frames[right_pos])
        span = max(v - u, 1)
        rows = []
        local_targets = []
        for frame_idx in range(u, v + 1):
            beta = float(v - frame_idx) / float(span)
            gamma = float(frame_idx - u) / float(span)
            rows.append([beta, gamma])
            local_targets.append(targets[frame_idx])
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(local_targets, dtype=np.float64)
        gram = x.T @ x
        rhs = x.T @ y
        ab = np.linalg.solve(gram + 1e-8 * eye2, rhs)
        interval_weight = float(v - u + 1)
        proposals[left_pos].append(
            (np.asarray(ab[0], dtype=np.float32), interval_weight)
        )
        proposals[right_pos].append(
            (np.asarray(ab[1], dtype=np.float32), interval_weight)
        )
    out = init.copy()
    for idx, items in enumerate(proposals):
        if not items:
            continue
        total_w = float(sum(weight for _vec, weight in items))
        voted = sum(
            np.asarray(vec, dtype=np.float64) * float(weight) for vec, weight in items
        ) / max(total_w, 1e-8)
        out[idx] = voted
    return np.asarray(out.reshape(np.asarray(keyframe_vectors).shape), dtype=np.float32)


def interpolate_vectors(
    start_vec: np.ndarray, end_vec: np.ndarray, alpha: float
) -> np.ndarray:
    return (
        (1.0 - float(alpha)) * np.asarray(start_vec, dtype=np.float32)
        + float(alpha) * np.asarray(end_vec, dtype=np.float32)
    ).astype(np.float32)


def interpolate_polygons(
    start_polys: list[np.ndarray], end_polys: list[np.ndarray], alpha: float
) -> list[np.ndarray]:
    alpha32 = np.float32(alpha)
    beta32 = np.float32(1.0) - alpha32
    out: list[np.ndarray] = []
    for start_poly, end_poly in zip(start_polys, end_polys):
        start_pts = np.asarray(start_poly, dtype=np.float32)
        end_pts = np.asarray(end_poly, dtype=np.float32)
        out.append((beta32 * start_pts + alpha32 * end_pts).astype(np.float32))
    return out


def assign_candidate_ids_to_keyframes(
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    candidates_by_frame: list[list[ShapeCandidate]],
) -> list[int]:
    candidate_ids: list[int] = []
    for frame_idx, vector in zip(chosen_frames, keyframe_vectors):
        frame_candidates = candidates_by_frame[int(frame_idx)]
        best_cand = 0
        best_dist = float("inf")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        for cand_id, candidate in enumerate(frame_candidates):
            cand_vec = np.asarray(candidate.vector, dtype=np.float32).reshape(-1)
            dist = float(np.mean(np.square(vec - cand_vec)))
            if dist < best_dist:
                best_dist = dist
                best_cand = int(cand_id)
        candidate_ids.append(int(best_cand))
    return candidate_ids


def interval_cost_from_vectors(
    run: InstanceRun,
    start_idx: int,
    start_vec: np.ndarray,
    end_idx: int,
    end_vec: np.ndarray,
    args: argparse.Namespace,
    *,
    include_start: bool,
    eval_contexts: list[FrameEvalContext] | None = None,
    start_candidate: ShapeCandidate | None = None,
    end_candidate: ShapeCandidate | None = None,
) -> IntervalCost:
    if end_idx < start_idx:
        return IntervalCost(
            cost=float("inf"),
            shape_distance=float("inf"),
            shape_update=1.0,
            frames_covered=0,
        )
    start_polys = (
        start_candidate.polygons
        if start_candidate is not None
        else split_vector_to_polygons(
            start_vec, run.contour_count, run.anchors_per_contour
        )
    )
    end_polys = (
        end_candidate.polygons
        if end_candidate is not None
        else split_vector_to_polygons(
            end_vec, run.contour_count, run.anchors_per_contour
        )
    )
    dist = shape_distance(start_vec, end_vec, run.scale)
    update = 1.0 if dist > float(args.shape_update_threshold_ratio) else 0.0
    total = 0.0
    frames_covered = 0
    frame_loss_total = 0.0
    recall_budget_total = 0.0
    start_frame = int(start_idx if include_start else start_idx + 1)
    for frame_idx in range(start_frame, int(end_idx) + 1):
        if frame_idx == start_idx:
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_polygons(
                    eval_contexts[frame_idx], start_polys
                )
            else:
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], start_polys
                )
        elif frame_idx == end_idx:
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_polygons(
                    eval_contexts[frame_idx], end_polys
                )
            else:
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], end_polys
                )
        else:
            alpha = float((frame_idx - start_idx) / max(end_idx - start_idx, 1))
            if eval_contexts is not None:
                metrics = compute_cached_metrics_from_interpolated_polygons(
                    eval_contexts[frame_idx],
                    start_polys,
                    end_polys,
                    alpha,
                )
            else:
                pred_polys = interpolate_polygons(start_polys, end_polys, alpha)
                metrics = compute_exact_metrics_from_polygons(
                    run.gt_polygons[frame_idx], pred_polys
                )
        frame_loss = float(frame_accuracy_loss(metrics, args))
        recall_budget = float(recall_budget_from_metrics(metrics))
        total += float(frame_loss)
        frame_loss_total += float(frame_loss)
        recall_budget_total += float(recall_budget)
        frames_covered += 1
    frame_loss_mean = float(frame_loss_total / max(frames_covered, 1))
    dist_scale, switch_scale = adaptive_shape_penalty_scales(frame_loss_mean, args)
    total += float(args.shape_switch_weight) * float(switch_scale) * float(update)
    total += float(args.shape_distance_weight) * float(dist_scale) * float(dist)
    return IntervalCost(
        cost=float(total),
        shape_distance=float(dist),
        shape_update=float(update),
        frames_covered=int(frames_covered),
        frame_loss_mean=float(frame_loss_mean),
        shape_distance_scale=float(dist_scale),
        shape_switch_scale=float(switch_scale),
        recall_budget=float(recall_budget_total),
    )


def run_multistate_penalty_path(
    run: InstanceRun,
    candidate_frames: list[int],
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[
    list[int],
    list[int],
    dict[str, int],
    dict[tuple[int, int, int, int, int], IntervalCost],
    float,
]:
    if all(len(frame_candidates) == 1 for frame_candidates in candidates_by_frame):
        return run_single_state_penalty_path(
            run,
            candidate_frames,
            candidates_by_frame,
            target_count,
            args,
            eval_contexts=eval_contexts,
        )

    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    state_frames: list[int] = []
    state_candidate_ids: list[int] = []
    node_offsets: list[tuple[int, int]] = []
    cursor = 0
    for frame_idx in candidate_frames:
        start = cursor
        for cand_id in range(len(candidates_by_frame[int(frame_idx)])):
            state_frames.append(int(frame_idx))
            state_candidate_ids.append(int(cand_id))
            cursor += 1
        node_offsets.append((start, cursor))
    state_count = cursor
    cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
    edge_array_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    counters = {"interval_evals": 0, "interval_frames": 0}
    use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
    recall_penalty_weight = float(args.proxy_recall_penalty_weight)
    predecessor_nodes: list[list[int]] = []
    for node_pos, end_frame in enumerate(candidate_frames):
        valid_prev: list[int] = []
        end_frame_i = int(end_frame)
        for prev_node_pos in range(node_pos):
            if end_frame_i - int(candidate_frames[prev_node_pos]) <= int(
                dynamic_max_gap
            ):
                valid_prev.append(int(prev_node_pos))
        predecessor_nodes.append(valid_prev)

    def get_cost(
        start_frame: int,
        start_cand: int,
        end_frame: int,
        end_cand: int,
        include_start: bool,
    ) -> IntervalCost:
        key = (
            int(start_frame),
            int(start_cand),
            int(end_frame),
            int(end_cand),
            1 if include_start else 0,
        )
        info = cost_cache.get(key)
        if info is None:
            full_key = (
                int(start_frame),
                int(start_cand),
                int(end_frame),
                int(end_cand),
                1,
            )
            full_info = cost_cache.get(full_key)
            if full_info is None:
                start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                end_candidate = candidates_by_frame[int(end_frame)][int(end_cand)]
                full_info = interval_cost_from_vectors(
                    run,
                    int(start_frame),
                    start_candidate.vector,
                    int(end_frame),
                    end_candidate.vector,
                    args,
                    include_start=True,
                    eval_contexts=eval_contexts,
                    start_candidate=start_candidate,
                    end_candidate=end_candidate,
                )
                cost_cache[full_key] = full_info
                counters["interval_evals"] += 1
                counters["interval_frames"] += int(full_info.frames_covered)
            if include_start:
                info = full_info
            else:
                start_candidate = candidates_by_frame[int(start_frame)][int(start_cand)]
                info = IntervalCost(
                    cost=float(full_info.cost - float(start_candidate.frame_loss)),
                    shape_distance=float(full_info.shape_distance),
                    shape_update=float(full_info.shape_update),
                    frames_covered=max(int(full_info.frames_covered) - 1, 0),
                    frame_loss_mean=float(full_info.frame_loss_mean),
                    shape_distance_scale=float(full_info.shape_distance_scale),
                    shape_switch_scale=float(full_info.shape_switch_scale),
                    recall_budget=max(
                        float(full_info.recall_budget)
                        - float(start_candidate.recall_budget),
                        0.0,
                    ),
                )
                cost_cache[key] = info
        return info

    def get_edge_arrays(
        prev_node_pos: int, node_pos: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(prev_node_pos), int(node_pos))
        cached = edge_array_cache.get(key)
        if cached is not None:
            return cached
        start_frame = int(candidate_frames[prev_node_pos])
        end_frame = int(candidate_frames[node_pos])
        src_start, src_end = node_offsets[prev_node_pos]
        dst_start, dst_end = node_offsets[node_pos]
        src_count = int(src_end - src_start)
        dst_count = int(dst_end - dst_start)
        cost_arr = np.empty((src_count, dst_count), dtype=np.float64)
        budget_arr = np.empty((src_count, dst_count), dtype=np.float64)
        for src_local, src_state in enumerate(range(src_start, src_end)):
            start_cand = int(state_candidate_ids[src_state])
            for dst_local, dst_state in enumerate(range(dst_start, dst_end)):
                end_cand = int(state_candidate_ids[dst_state])
                info = get_cost(
                    start_frame, start_cand, end_frame, end_cand, include_start=False
                )
                cost_arr[src_local, dst_local] = float(info.cost)
                budget_arr[src_local, dst_local] = float(info.recall_budget)
        edge_array_cache[key] = (cost_arr, budget_arr)
        return cost_arr, budget_arr

    def decode(
        lambda_penalty: float, recall_mu: float
    ) -> tuple[list[int], list[int], float, float]:
        dp = np.full((state_count,), np.inf, dtype=np.float64)
        back = np.full((state_count,), -1, dtype=np.int32)
        raw_cost = np.full((state_count,), np.inf, dtype=np.float64)
        raw_budget = np.full((state_count,), np.inf, dtype=np.float64)
        first_start, first_end = node_offsets[0]
        for state_idx in range(first_start, first_end):
            cand_id = int(state_candidate_ids[state_idx])
            frame_loss = float(candidates_by_frame[0][cand_id].frame_loss)
            frame_budget = float(candidates_by_frame[0][cand_id].recall_budget)
            penalty = (
                float(recall_mu) * frame_budget
                if use_exact_recall_dp
                else recall_penalty_weight * frame_budget
            )
            dp[state_idx] = frame_loss + penalty + float(lambda_penalty)
            raw_cost[state_idx] = frame_loss
            raw_budget[state_idx] = frame_budget
        for node_pos in range(1, len(candidate_frames)):
            dst_start, dst_end = node_offsets[node_pos]
            prev_entries = []
            for prev_node_pos in predecessor_nodes[node_pos]:
                src_start, src_end = node_offsets[prev_node_pos]
                edge_costs, edge_budgets = get_edge_arrays(prev_node_pos, node_pos)
                prev_entries.append((src_start, src_end, edge_costs, edge_budgets))
            for dst_state in range(dst_start, dst_end):
                dst_local = int(dst_state - dst_start)
                best_cost = float("inf")
                best_raw = float("inf")
                best_budget = float("inf")
                best_prev = -1
                for src_start, src_end, edge_costs, edge_budgets in prev_entries:
                    for src_state in range(src_start, src_end):
                        prev_cost = float(dp[src_state])
                        if not np.isfinite(prev_cost):
                            continue
                        src_local = int(src_state - src_start)
                        edge_cost = float(edge_costs[src_local, dst_local])
                        edge_budget = float(edge_budgets[src_local, dst_local])
                        penalty = (
                            float(recall_mu) * edge_budget
                            if use_exact_recall_dp
                            else recall_penalty_weight * edge_budget
                        )
                        cand_cost = (
                            prev_cost + edge_cost + penalty + float(lambda_penalty)
                        )
                        cand_raw = float(raw_cost[src_state]) + edge_cost
                        cand_budget = float(raw_budget[src_state]) + edge_budget
                        if cand_cost < best_cost or (
                            abs(cand_cost - best_cost) <= 1e-9
                            and (
                                cand_budget < best_budget
                                or (
                                    abs(cand_budget - best_budget) <= 1e-9
                                    and cand_raw < best_raw
                                )
                            )
                        ):
                            best_cost = float(cand_cost)
                            best_raw = float(cand_raw)
                            best_budget = float(cand_budget)
                            best_prev = int(src_state)
                dp[dst_state] = best_cost
                raw_cost[dst_state] = best_raw
                raw_budget[dst_state] = best_budget
                back[dst_state] = int(best_prev)
        last_start, last_end = node_offsets[-1]
        best_state = -1
        best_cost = float("inf")
        best_raw = float("inf")
        best_budget = float("inf")
        for state_idx in range(last_start, last_end):
            cost = float(dp[state_idx])
            raw = float(raw_cost[state_idx])
            budget = float(raw_budget[state_idx])
            if cost < best_cost or (
                abs(cost - best_cost) <= 1e-9
                and (
                    budget < best_budget
                    or (abs(budget - best_budget) <= 1e-9 and raw < best_raw)
                )
            ):
                best_cost = cost
                best_raw = raw
                best_budget = budget
                best_state = int(state_idx)
        if best_state < 0:
            raise RuntimeError("failed to decode penalized multistate path")
        chosen_frames: list[int] = []
        chosen_candidate_ids: list[int] = []
        cur_state = best_state
        while cur_state >= 0:
            chosen_frames.append(int(state_frames[cur_state]))
            chosen_candidate_ids.append(int(state_candidate_ids[cur_state]))
            cur_state = int(back[cur_state])
        chosen_frames.reverse()
        chosen_candidate_ids.reverse()
        return chosen_frames, chosen_candidate_ids, best_raw, best_budget

    def decode_for_recall_mu(
        recall_mu: float,
    ) -> tuple[list[int], list[int], float, float, float]:
        best: tuple[list[int], list[int], float, float] | None = None
        lo = 0.0
        hi = float(args.penalty_max)
        for _ in range(max(1, int(args.penalty_binary_steps))):
            mid = 0.5 * (lo + hi)
            cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
            candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
            if best is None:
                best = candidate
            else:
                cur_frames, cur_ids, cur_raw, cur_budget = best
                cand_gap = abs(len(cand_frames) - int(target_count))
                cur_gap = abs(len(cur_frames) - int(target_count))
                if cand_gap < cur_gap or (
                    cand_gap == cur_gap
                    and (
                        len(cand_frames) < len(cur_frames)
                        or (
                            len(cand_frames) == len(cur_frames)
                            and (
                                cand_budget < cur_budget
                                or (
                                    abs(cand_budget - cur_budget) <= 1e-9
                                    and cand_raw < cur_raw
                                )
                            )
                        )
                    )
                ):
                    best = candidate
            if len(cand_frames) > int(target_count):
                lo = mid
            else:
                hi = mid
        assert best is not None
        best_frames, best_ids, best_raw, best_budget = best
        return best_frames, best_ids, best_raw, best_budget, float(hi)

    if use_exact_recall_dp:
        best_result: tuple[list[int], list[int], float, float, float] | None = None
        recall_lo = 0.0
        recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
        for _ in range(max(1, int(args.recall_budget_binary_steps))):
            recall_mid = 0.5 * (recall_lo + recall_hi)
            (
                cand_frames,
                cand_ids,
                cand_raw,
                cand_budget,
                cand_lambda,
            ) = decode_for_recall_mu(recall_mid)
            cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
            if best_result is None:
                best_result = (
                    cand_frames,
                    cand_ids,
                    cand_raw,
                    cand_budget,
                    cand_lambda,
                )
            else:
                _bf, _bi, best_raw, best_budget, best_lambda = best_result
                best_violation = recall_violation(
                    best_budget, len(run.frame_numbers), args
                )
                if cand_violation < best_violation - 1e-12 or (
                    abs(cand_violation - best_violation) <= 1e-12
                    and (
                        cand_raw < best_raw
                        or (
                            abs(cand_raw - best_raw) <= 1e-9
                            and cand_lambda < best_lambda
                        )
                    )
                ):
                    best_result = (
                        cand_frames,
                        cand_ids,
                        cand_raw,
                        cand_budget,
                        cand_lambda,
                    )
            if cand_violation > 0.0:
                recall_lo = recall_mid
            else:
                recall_hi = recall_mid
        assert best_result is not None
        best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
    else:
        (
            best_frames,
            best_ids,
            _best_raw,
            _best_budget,
            best_lambda,
        ) = decode_for_recall_mu(0.0)
    return best_frames, best_ids, counters, cost_cache, float(best_lambda)


def run_single_state_penalty_path(
    run: InstanceRun,
    candidate_frames: list[int],
    candidates_by_frame: list[list[ShapeCandidate]],
    target_count: int,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[
    list[int],
    list[int],
    dict[str, int],
    dict[tuple[int, int, int, int, int], IntervalCost],
    float,
]:
    target_interval = max(1, int(round(1.0 / max(float(args.target_ratio), 1e-6))))
    dynamic_max_gap = max(
        int(args.max_gap),
        int(math.ceil(float(args.dynamic_max_gap_factor) * float(target_interval))),
    )
    node_count = int(len(candidate_frames))
    cost_cache: dict[tuple[int, int, int, int, int], IntervalCost] = {}
    edge_cache: dict[tuple[int, int], IntervalCost] = {}
    counters = {"interval_evals": 0, "interval_frames": 0}
    use_exact_recall_dp = str(args.recall_constraint_mode) == "exact_dp"
    recall_penalty_weight = float(args.proxy_recall_penalty_weight)

    predecessor_nodes: list[list[int]] = []
    for node_pos, end_frame in enumerate(candidate_frames):
        end_frame_i = int(end_frame)
        min_prev_pos = int(
            bisect.bisect_left(
                candidate_frames, end_frame_i - int(dynamic_max_gap), 0, node_pos
            )
        )
        predecessor_nodes.append(list(range(min_prev_pos, node_pos)))

    def get_edge_info(prev_node_pos: int, node_pos: int) -> IntervalCost:
        key = (int(prev_node_pos), int(node_pos))
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        start_frame = int(candidate_frames[prev_node_pos])
        end_frame = int(candidate_frames[node_pos])
        start_candidate = candidates_by_frame[start_frame][0]
        end_candidate = candidates_by_frame[end_frame][0]
        info = interval_cost_from_vectors(
            run,
            start_frame,
            start_candidate.vector,
            end_frame,
            end_candidate.vector,
            args,
            include_start=False,
            eval_contexts=eval_contexts,
            start_candidate=start_candidate,
            end_candidate=end_candidate,
        )
        edge_cache[key] = info
        cost_cache[(start_frame, 0, end_frame, 0, 0)] = info
        counters["interval_evals"] += 1
        counters["interval_frames"] += int(info.frames_covered)
        return info

    def decode(
        lambda_penalty: float, recall_mu: float
    ) -> tuple[list[int], list[int], float, float]:
        dp = np.full((node_count,), np.inf, dtype=np.float64)
        back = np.full((node_count,), -1, dtype=np.int32)
        raw_cost = np.full((node_count,), np.inf, dtype=np.float64)
        raw_budget = np.full((node_count,), np.inf, dtype=np.float64)

        first_candidate = candidates_by_frame[int(candidate_frames[0])][0]
        first_budget = float(first_candidate.recall_budget)
        first_penalty = (
            float(recall_mu) * first_budget
            if use_exact_recall_dp
            else recall_penalty_weight * first_budget
        )
        dp[0] = (
            float(first_candidate.frame_loss) + first_penalty + float(lambda_penalty)
        )
        raw_cost[0] = float(first_candidate.frame_loss)
        raw_budget[0] = float(first_budget)

        for node_pos in range(1, node_count):
            best_cost = float("inf")
            best_raw = float("inf")
            best_budget = float("inf")
            best_prev = -1
            for prev_node_pos in predecessor_nodes[node_pos]:
                prev_cost = float(dp[prev_node_pos])
                if not np.isfinite(prev_cost):
                    continue
                info = get_edge_info(prev_node_pos, node_pos)
                edge_budget = float(info.recall_budget)
                penalty = (
                    float(recall_mu) * edge_budget
                    if use_exact_recall_dp
                    else recall_penalty_weight * edge_budget
                )
                cand_cost = (
                    prev_cost + float(info.cost) + penalty + float(lambda_penalty)
                )
                cand_raw = float(raw_cost[prev_node_pos]) + float(info.cost)
                cand_budget = float(raw_budget[prev_node_pos]) + edge_budget
                if cand_cost < best_cost or (
                    abs(cand_cost - best_cost) <= 1e-9
                    and (
                        cand_budget < best_budget
                        or (
                            abs(cand_budget - best_budget) <= 1e-9
                            and cand_raw < best_raw
                        )
                    )
                ):
                    best_cost = float(cand_cost)
                    best_raw = float(cand_raw)
                    best_budget = float(cand_budget)
                    best_prev = int(prev_node_pos)
            dp[node_pos] = best_cost
            raw_cost[node_pos] = best_raw
            raw_budget[node_pos] = best_budget
            back[node_pos] = int(best_prev)

        last_pos = int(node_count - 1)
        if last_pos < 0 or not np.isfinite(dp[last_pos]):
            raise RuntimeError("failed to decode single-state penalized path")

        chosen_frames: list[int] = []
        cur_pos = last_pos
        while cur_pos >= 0:
            chosen_frames.append(int(candidate_frames[cur_pos]))
            cur_pos = int(back[cur_pos])
        chosen_frames.reverse()
        chosen_candidate_ids = [0] * len(chosen_frames)
        return (
            chosen_frames,
            chosen_candidate_ids,
            float(raw_cost[last_pos]),
            float(raw_budget[last_pos]),
        )

    def decode_for_recall_mu(
        recall_mu: float,
    ) -> tuple[list[int], list[int], float, float, float]:
        best: tuple[list[int], list[int], float, float] | None = None
        lo = 0.0
        hi = float(args.penalty_max)
        for _ in range(max(1, int(args.penalty_binary_steps))):
            mid = 0.5 * (lo + hi)
            cand_frames, cand_ids, cand_raw, cand_budget = decode(mid, recall_mu)
            candidate = (cand_frames, cand_ids, cand_raw, cand_budget)
            if best is None:
                best = candidate
            else:
                cur_frames, _cur_ids, cur_raw, cur_budget = best
                cand_gap = abs(len(cand_frames) - int(target_count))
                cur_gap = abs(len(cur_frames) - int(target_count))
                if cand_gap < cur_gap or (
                    cand_gap == cur_gap
                    and (
                        len(cand_frames) < len(cur_frames)
                        or (
                            len(cand_frames) == len(cur_frames)
                            and (
                                cand_budget < cur_budget
                                or (
                                    abs(cand_budget - cur_budget) <= 1e-9
                                    and cand_raw < cur_raw
                                )
                            )
                        )
                    )
                ):
                    best = candidate
            if len(cand_frames) > int(target_count):
                lo = mid
            else:
                hi = mid
        assert best is not None
        best_frames, best_ids, best_raw, best_budget = best
        return best_frames, best_ids, best_raw, best_budget, float(hi)

    if use_exact_recall_dp:
        best_result: tuple[list[int], list[int], float, float, float] | None = None
        recall_lo = 0.0
        recall_hi = float(max(args.recall_budget_max_mu, 1e-6))
        for _ in range(max(1, int(args.recall_budget_binary_steps))):
            recall_mid = 0.5 * (recall_lo + recall_hi)
            (
                cand_frames,
                cand_ids,
                cand_raw,
                cand_budget,
                cand_lambda,
            ) = decode_for_recall_mu(recall_mid)
            cand_violation = recall_violation(cand_budget, len(run.frame_numbers), args)
            if best_result is None:
                best_result = (
                    cand_frames,
                    cand_ids,
                    cand_raw,
                    cand_budget,
                    cand_lambda,
                )
            else:
                _bf, _bi, best_raw, best_budget, best_lambda = best_result
                best_violation = recall_violation(
                    best_budget, len(run.frame_numbers), args
                )
                if cand_violation < best_violation - 1e-12 or (
                    abs(cand_violation - best_violation) <= 1e-12
                    and (
                        cand_raw < best_raw
                        or (
                            abs(cand_raw - best_raw) <= 1e-9
                            and cand_lambda < best_lambda
                        )
                    )
                ):
                    best_result = (
                        cand_frames,
                        cand_ids,
                        cand_raw,
                        cand_budget,
                        cand_lambda,
                    )
            if cand_violation > 0.0:
                recall_lo = recall_mid
            else:
                recall_hi = recall_mid
        assert best_result is not None
        best_frames, best_ids, _best_raw, _best_budget, best_lambda = best_result
    else:
        (
            best_frames,
            best_ids,
            _best_raw,
            _best_budget,
            best_lambda,
        ) = decode_for_recall_mu(0.0)
    return best_frames, best_ids, counters, cost_cache, float(best_lambda)


def evaluate_keyframe_path(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, list[IntervalCost], float]:
    (
        total,
        _start_loss,
        interval_infos,
        total_recall_budget,
        _start_budget,
    ) = evaluate_keyframe_path_parts(
        run,
        chosen_frames,
        keyframe_vectors,
        args,
        eval_contexts=eval_contexts,
    )
    return float(total), interval_infos, float(total_recall_budget)


def evaluate_keyframe_path_parts(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    args: argparse.Namespace,
    eval_contexts: list[FrameEvalContext] | None = None,
) -> tuple[float, float, list[IntervalCost], float, float]:
    total = 0.0
    interval_infos: list[IntervalCost] = []
    if len(chosen_frames) <= 0:
        return float("inf"), float("inf"), interval_infos, float("inf"), float("inf")
    start_vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
    start_loss, start_budget = evaluate_frame_vector_loss_budget(
        run, int(chosen_frames[0]), start_vec, args, eval_contexts=eval_contexts
    )
    total_recall_budget = float(start_budget)
    total += float(start_loss)
    for left_idx, right_idx, left_vec, right_vec in zip(
        chosen_frames[:-1],
        chosen_frames[1:],
        keyframe_vectors[:-1],
        keyframe_vectors[1:],
    ):
        info = interval_cost_from_vectors(
            run,
            int(left_idx),
            np.asarray(left_vec, dtype=np.float32),
            int(right_idx),
            np.asarray(right_vec, dtype=np.float32),
            args,
            include_start=False,
            eval_contexts=eval_contexts,
        )
        interval_infos.append(info)
        total += float(info.cost)
        total_recall_budget += float(info.recall_budget)
    return (
        float(total),
        float(start_loss),
        interval_infos,
        float(total_recall_budget),
        float(start_budget),
    )


def exact_interpolated_metrics(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
) -> tuple[list[dict[str, float]], float, float, float, float, float]:
    metrics_rows: list[dict[str, float]] = []
    total_iou_loss = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_gt_area = 0.0
    total_intersection = 0.0
    chosen_frames_arr = [int(v) for v in chosen_frames]
    interval_pos = 0
    for frame_idx in range(len(run.frame_numbers)):
        if frame_idx <= chosen_frames_arr[0]:
            vec = np.asarray(keyframe_vectors[0], dtype=np.float32)
        elif frame_idx >= chosen_frames_arr[-1]:
            vec = np.asarray(keyframe_vectors[-1], dtype=np.float32)
        else:
            while interval_pos + 1 < len(chosen_frames_arr) and frame_idx > int(
                chosen_frames_arr[interval_pos + 1]
            ):
                interval_pos += 1
            right_pos = int(interval_pos + 1)
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen_frames_arr[left_pos])
            right_frame = int(chosen_frames_arr[right_pos])
            if frame_idx == right_frame:
                vec = np.asarray(keyframe_vectors[right_pos], dtype=np.float32)
            else:
                alpha = float(
                    (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                )
                vec = interpolate_vectors(
                    keyframe_vectors[left_pos], keyframe_vectors[right_pos], alpha
                )
        pred_polys = split_vector_to_polygons(
            vec, run.contour_count, run.anchors_per_contour
        )
        metrics = compute_exact_metrics_from_polygons(
            run.gt_polygons[frame_idx], pred_polys
        )
        metrics_rows.append(metrics)
        total_iou_loss += 1.0 - float(metrics["iou"])
        total_recall += float(metrics["recall"])
        total_precision += float(metrics["precision"])
        total_gt_area += float(metrics["gt_area"])
        total_intersection += float(metrics["intersection"])
    mean_iou = float(1.0 - total_iou_loss / max(len(metrics_rows), 1))
    mean_recall = float(total_recall / max(len(metrics_rows), 1))
    mean_precision = float(total_precision / max(len(metrics_rows), 1))
    global_recall = (
        float(total_intersection / total_gt_area) if total_gt_area > 0 else 1.0
    )
    return (
        metrics_rows,
        float(total_iou_loss),
        float(mean_iou),
        float(mean_recall),
        float(mean_precision),
        float(global_recall),
    )


def exact_recall_solution_key(
    total_iou_loss: float, mean_recall: float, args: argparse.Namespace
) -> tuple[float, float, float]:
    violation = max(float(args.recall_min) - float(mean_recall), 0.0)
    return float(violation), float(total_iou_loss), float(-mean_recall)


def repair_keyframe_vectors_for_exact_recall(
    run: InstanceRun,
    chosen_frames: list[int],
    keyframe_vectors: np.ndarray,
    candidates_by_frame: list[list[ShapeCandidate]],
    args: argparse.Namespace,
) -> np.ndarray:
    if not bool(args.exact_recall_repair_enabled) or len(chosen_frames) <= 0:
        return np.asarray(keyframe_vectors, dtype=np.float32)
    current = np.asarray(keyframe_vectors, dtype=np.float32).copy()
    scale_deltas = parse_float_list(
        str(args.exact_recall_repair_scale_deltas), [0.01, 0.02, 0.04, 0.06, 0.08]
    )
    (
        metrics_rows,
        current_iou_loss,
        _current_mean_iou,
        current_mean_recall,
        _current_mean_precision,
        _current_global_recall,
    ) = exact_interpolated_metrics(run, chosen_frames, current)
    best_key = exact_recall_solution_key(current_iou_loss, current_mean_recall, args)
    if best_key[0] <= 0.0:
        return current

    for _pass in range(max(1, int(args.exact_recall_repair_max_passes))):
        frame_deficits = np.asarray(
            [
                float(row["gt_area"])
                * max(float(args.recall_min) - float(row["recall"]), 0.0)
                for row in metrics_rows
            ],
            dtype=np.float64,
        )
        if float(np.mean(frame_deficits)) <= 0.0 and best_key[0] <= 0.0:
            break
        key_scores = np.zeros((len(chosen_frames),), dtype=np.float64)
        for frame_idx, deficit in enumerate(frame_deficits.tolist()):
            if deficit <= 0.0:
                continue
            if frame_idx <= int(chosen_frames[0]):
                key_scores[0] += float(deficit)
                continue
            if frame_idx >= int(chosen_frames[-1]):
                key_scores[-1] += float(deficit)
                continue
            right_pos = next(
                pos
                for pos, keyframe in enumerate(chosen_frames)
                if keyframe >= frame_idx
            )
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen_frames[left_pos])
            right_frame = int(chosen_frames[right_pos])
            alpha = float((frame_idx - left_frame) / max(right_frame - left_frame, 1))
            key_scores[left_pos] += (1.0 - alpha) * float(deficit)
            key_scores[right_pos] += alpha * float(deficit)
        key_order = [
            int(idx)
            for idx in np.argsort(-key_scores)[
                : max(1, int(args.exact_recall_repair_topk))
            ].tolist()
        ]
        improved = False

        trial_vectors: list[np.ndarray] = []
        for delta in scale_deltas:
            scaled_all = np.asarray(current, dtype=np.float32).copy()
            for key_idx in range(len(chosen_frames)):
                scaled_all[key_idx] = scale_vector_about_centroid(
                    scaled_all[key_idx], 1.0 + float(delta)
                )
            trial_vectors.append(scaled_all)
        for delta in scale_deltas:
            scaled = np.asarray(current, dtype=np.float32).copy()
            for key_idx in key_order:
                scaled[key_idx] = scale_vector_about_centroid(
                    scaled[key_idx], 1.0 + float(delta)
                )
            trial_vectors.append(scaled)

        for key_idx in key_order:
            frame_idx = int(chosen_frames[key_idx])
            current_area, _center, _radii, _mean_radius = vector_proxy_stats(
                current[key_idx], run.contour_count, run.anchors_per_contour
            )
            for candidate in candidates_by_frame[frame_idx]:
                if float(candidate.area) <= float(current_area) + 1e-3:
                    continue
                upgraded = np.asarray(current, dtype=np.float32).copy()
                upgraded[key_idx] = np.asarray(candidate.vector, dtype=np.float32)
                trial_vectors.append(upgraded)
            for delta in scale_deltas:
                upgraded = np.asarray(current, dtype=np.float32).copy()
                upgraded[key_idx] = scale_vector_about_centroid(
                    upgraded[key_idx], 1.0 + float(delta)
                )
                trial_vectors.append(upgraded)

        seen: list[np.ndarray] = []
        for trial in trial_vectors:
            if any(np.allclose(trial, existing, atol=1e-4) for existing in seen):
                continue
            seen.append(np.asarray(trial, dtype=np.float32))
            (
                trial_metrics,
                trial_iou_loss,
                _trial_mean_iou,
                trial_mean_recall,
                _trial_mean_precision,
                _trial_global_recall,
            ) = exact_interpolated_metrics(run, chosen_frames, trial)
            trial_key = exact_recall_solution_key(
                trial_iou_loss, trial_mean_recall, args
            )
            if trial_key < best_key:
                current = np.asarray(trial, dtype=np.float32)
                metrics_rows = trial_metrics
                best_key = trial_key
                improved = True
        if not improved:
            break
        if best_key[0] <= 0.0:
            break
    return np.asarray(current, dtype=np.float32)


class LazyInterpolatedRun:
    def __init__(
        self, run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray
    ):
        self.run = run
        self.chosen_frames = [int(v) for v in chosen_frames]
        self.keyframe_vectors = np.asarray(keyframe_vectors, dtype=np.float32)
        self.length = int(len(run.frame_numbers))

    def __len__(self) -> int:
        return int(self.length)

    def _polygons_at(self, frame_idx: int) -> list[np.ndarray]:
        idx = int(frame_idx)
        if idx < 0:
            idx += int(self.length)
        if idx < 0 or idx >= int(self.length):
            raise IndexError(frame_idx)
        chosen = self.chosen_frames
        if idx <= chosen[0]:
            vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
        elif idx >= chosen[-1]:
            vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
        else:
            right_pos = int(
                np.searchsorted(
                    np.asarray(chosen, dtype=np.int32), int(idx), side="left"
                )
            )
            left_pos = max(0, right_pos - 1)
            left_frame = int(chosen[left_pos])
            right_frame = int(chosen[right_pos])
            if idx == right_frame:
                vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
            else:
                alpha = float((idx - left_frame) / max(right_frame - left_frame, 1))
                vec = interpolate_vectors(
                    self.keyframe_vectors[left_pos],
                    self.keyframe_vectors[right_pos],
                    alpha,
                )
        return split_vector_to_polygons(
            vec, self.run.contour_count, self.run.anchors_per_contour
        )

    def __getitem__(self, frame_idx):
        if isinstance(frame_idx, slice):
            return [
                self._polygons_at(idx)
                for idx in range(*frame_idx.indices(int(self.length)))
            ]
        return self._polygons_at(int(frame_idx))

    def __iter__(self):
        if self.length <= 0:
            return
        chosen = self.chosen_frames
        interval_pos = 0
        for frame_idx in range(int(self.length)):
            if frame_idx <= chosen[0]:
                vec = np.asarray(self.keyframe_vectors[0], dtype=np.float32)
            elif frame_idx >= chosen[-1]:
                vec = np.asarray(self.keyframe_vectors[-1], dtype=np.float32)
            else:
                while interval_pos + 1 < len(chosen) and frame_idx > int(
                    chosen[interval_pos + 1]
                ):
                    interval_pos += 1
                right_pos = int(interval_pos + 1)
                left_pos = max(0, right_pos - 1)
                left_frame = int(chosen[left_pos])
                right_frame = int(chosen[right_pos])
                if frame_idx == right_frame:
                    vec = np.asarray(self.keyframe_vectors[right_pos], dtype=np.float32)
                else:
                    alpha = float(
                        (frame_idx - left_frame) / max(right_frame - left_frame, 1)
                    )
                    vec = interpolate_vectors(
                        self.keyframe_vectors[left_pos],
                        self.keyframe_vectors[right_pos],
                        alpha,
                    )
            yield split_vector_to_polygons(
                vec, self.run.contour_count, self.run.anchors_per_contour
            )


def interpolate_run(
    run: InstanceRun, chosen_frames: list[int], keyframe_vectors: np.ndarray
):
    length = len(run.frame_numbers)
    if length <= 0:
        return []
    return LazyInterpolatedRun(run, chosen_frames, keyframe_vectors)


class LazyUnionRows:
    def __init__(self, run: InstanceRun, interp_polygons, chosen_frames: list[int]):
        self.run = run
        self.interp_polygons = interp_polygons
        self.chosen_set = {int(v) for v in chosen_frames}
        length = int(len(run.frame_numbers))
        self.emit_start = int(max(0, min(length, int(run.emit_start_idx))))
        emit_end = length if int(run.emit_end_idx) < 0 else int(run.emit_end_idx)
        self.emit_end = int(max(self.emit_start, min(length, emit_end)))

    def __len__(self) -> int:
        return int(self.emit_end - self.emit_start)

    def __iter__(self):
        for local_idx, (frame, polygons) in enumerate(
            zip(self.run.frame_numbers.tolist(), self.interp_polygons)
        ):
            if local_idx < self.emit_start or local_idx >= self.emit_end:
                continue
            yield {
                "frame": int(frame),
                "track_id": str(self.run.track_id),
                "run_id": int(self.run.run_id),
                "polygons": [
                    np.asarray(poly, dtype=np.float32).tolist() for poly in polygons
                ],
                "has_keyframe": 1 if local_idx in self.chosen_set else 0,
                "is_gapfill": int(self.run.gapfilled_flags[local_idx])
                if self.run.gapfilled_flags is not None
                and local_idx < len(self.run.gapfilled_flags)
                else 0,
            }


def process_single_run(run: InstanceRun, args: argparse.Namespace) -> dict[str, object]:
    length = len(run.frame_numbers)
    if length <= 0:
        return {
            "union_rows": [],
            "final_keyframes": [],
            "stream_row": None,
            "interval_eval_count": 0,
            "interval_eval_frames": 0,
            "candidate_frame_count": 0,
        }
    target_count = max(2, min(length, int(round(length * float(args.target_ratio)))))
    stage_times: dict[str, float] = {}

    stage_t0 = time.perf_counter()
    eval_contexts = build_frame_eval_contexts(run, args)
    stage_times["build_eval_contexts_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    candidates_by_frame = build_frame_candidates(run, [], eval_contexts, args)
    stage_times["build_candidates_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    candidate_frames, surrogate_frames, saliency_scores = build_candidate_frame_pool(
        run, candidates_by_frame, target_count, args
    )
    stage_times["build_candidate_pool_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    (
        chosen_frames,
        chosen_candidate_ids,
        counters,
        _cache,
        _best_lambda,
    ) = run_multistate_penalty_path(
        run,
        candidate_frames,
        candidates_by_frame,
        target_count,
        args,
        eval_contexts=eval_contexts,
    )
    stage_times["solve_dp_seconds"] = float(time.perf_counter() - stage_t0)

    keyframe_vectors = np.asarray(
        [
            np.asarray(
                candidates_by_frame[int(frame_idx)][int(cand_id)].vector,
                dtype=np.float32,
            )
            for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)
        ],
        dtype=np.float32,
    )

    stage_t0 = time.perf_counter()
    keyframe_vectors = pair_vote_refine_keyframe_vectors(
        run, chosen_frames, keyframe_vectors, args
    )
    stage_times["pair_vote_refine_seconds"] = float(time.perf_counter() - stage_t0)

    stage_t0 = time.perf_counter()
    keyframe_vectors = repair_keyframe_vectors_for_exact_recall(
        run, chosen_frames, keyframe_vectors, candidates_by_frame, args
    )
    stage_times["exact_recall_repair_seconds"] = float(time.perf_counter() - stage_t0)

    chosen_candidate_ids = assign_candidate_ids_to_keyframes(
        chosen_frames, keyframe_vectors, candidates_by_frame
    )

    stage_t0 = time.perf_counter()
    objective, interval_infos, total_recall_budget = evaluate_keyframe_path(
        run, chosen_frames, keyframe_vectors, args, eval_contexts=eval_contexts
    )
    interp_polygons = interpolate_run(run, chosen_frames, keyframe_vectors)
    stage_times["final_eval_seconds"] = float(time.perf_counter() - stage_t0)

    chosen_frame_to_candidate = {
        int(frame_idx): int(cand_id)
        for frame_idx, cand_id in zip(chosen_frames, chosen_candidate_ids)
    }
    union_rows = LazyUnionRows(run, interp_polygons, chosen_frames)
    emit_start = int(union_rows.emit_start)
    emit_end = int(union_rows.emit_end)
    emit_frame_count = int(max(0, emit_end - emit_start))

    final_keyframes: list[dict[str, object]] = []
    for keyframe_pos, frame_idx in enumerate(chosen_frames):
        if int(frame_idx) < emit_start or int(frame_idx) >= emit_end:
            continue
        final_keyframes.append(
            {
                "track_id": str(run.track_id),
                "run_id": int(run.run_id),
                "frame": int(run.frame_numbers[frame_idx]),
                "candidate_id": int(chosen_frame_to_candidate.get(int(frame_idx), -1)),
                "polygons": [
                    np.asarray(poly, dtype=np.float32).tolist()
                    for poly in split_vector_to_polygons(
                        keyframe_vectors[keyframe_pos],
                        run.contour_count,
                        run.anchors_per_contour,
                    )
                ],
            }
        )

    shape_distance_total = (
        float(
            np.sum(
                np.asarray(
                    [info.shape_distance for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    shape_update_count = (
        float(
            np.sum(
                np.asarray(
                    [info.shape_update for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_distance = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_distance for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_update = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_update for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_interval_frame_loss = (
        float(
            np.mean(
                np.asarray(
                    [info.frame_loss_mean for info in interval_infos], dtype=np.float64
                )
            )
        )
        if interval_infos
        else 0.0
    )
    mean_shape_distance_scale = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_distance_scale for info in interval_infos],
                    dtype=np.float64,
                )
            )
        )
        if interval_infos
        else 1.0
    )
    mean_shape_switch_scale = (
        float(
            np.mean(
                np.asarray(
                    [info.shape_switch_scale for info in interval_infos],
                    dtype=np.float64,
                )
            )
        )
        if interval_infos
        else 1.0
    )
    mean_recall_budget = float(total_recall_budget / max(length, 1))
    achieved_recall_floor = float(max(0.0, 1.0 - mean_recall_budget))
    recall_budget_violation = float(recall_violation(total_recall_budget, length, args))
    keyframe_rate = float(len(chosen_frames) / max(length, 1))
    gapfilled_frame_count = (
        int(np.sum(run.gapfilled_flags.astype(np.int32)))
        if run.gapfilled_flags is not None
        else 0
    )
    shape_update_rate = float(shape_update_count / max(length, 1))
    shape_distance_rate = float(shape_distance_total / max(length, 1))
    mean_state_count = (
        float(
            np.mean(
                np.asarray(
                    [len(frame_candidates) for frame_candidates in candidates_by_frame],
                    dtype=np.float64,
                )
            )
        )
        if candidates_by_frame
        else 0.0
    )
    stream_row = {
        "stream_id": str(run.stream_id),
        "track_id": str(run.track_id),
        "run_id": int(run.run_id),
        "frame_count": int(length),
        "emit_frame_count": int(emit_frame_count),
        "chunk_index": int(run.chunk_index),
        "chunk_count": int(run.chunk_count),
        "chunk_process_start": int(run.chunk_process_start),
        "chunk_process_end": int(run.chunk_process_end),
        "gapfilled_frame_count": int(gapfilled_frame_count),
        "contour_count": int(run.contour_count),
        "anchors_per_contour": int(run.anchors_per_contour),
        "run_target_total_points": int(run.run_target_total_points),
        "predicted_total_points_p90": float(
            np.quantile(run.predicted_total_points.astype(np.float64), 0.90)
        )
        if run.predicted_total_points is not None
        and len(run.predicted_total_points) > 0
        else 0.0,
        "predicted_total_points_mean": float(
            np.mean(run.predicted_total_points.astype(np.float64))
        )
        if run.predicted_total_points is not None
        and len(run.predicted_total_points) > 0
        else 0.0,
        "candidate_frame_count": int(len(candidate_frames)),
        "mean_state_count": float(mean_state_count),
        "surrogate_frame_count": int(len(surrogate_frames)),
        "target_keyframes": int(target_count),
        "chosen_keyframes": int(len(chosen_frames)),
        "achieved_ratio": float(keyframe_rate),
        "keyframe_rate": float(keyframe_rate),
        "objective": float(objective),
        "mean_interval_frame_loss": float(mean_interval_frame_loss),
        "mean_shape_distance": float(mean_shape_distance),
        "mean_shape_update": float(mean_shape_update),
        "shape_distance_total": float(shape_distance_total),
        "shape_distance_rate": float(shape_distance_rate),
        "shape_update_count": float(shape_update_count),
        "shape_update_rate": float(shape_update_rate),
        "mean_shape_distance_scale": float(mean_shape_distance_scale),
        "mean_shape_switch_scale": float(mean_shape_switch_scale),
        "mean_recall_budget": float(mean_recall_budget),
        "achieved_recall_floor": float(achieved_recall_floor),
        "recall_budget_violation": float(recall_budget_violation),
        "interval_eval_count": int(counters["interval_evals"]),
        "interval_eval_frames": int(counters["interval_frames"]),
        "max_saliency": float(np.max(saliency_scores))
        if saliency_scores.size > 0
        else 0.0,
        **{key: float(val) for key, val in stage_times.items()},
    }
    return {
        "union_rows": union_rows,
        "final_keyframes": final_keyframes,
        "stream_row": stream_row,
        "interval_eval_count": int(counters["interval_evals"]),
        "interval_eval_frames": int(counters["interval_frames"]),
        "candidate_frame_count": int(len(candidate_frames)),
        "stage_times": {key: float(val) for key, val in stage_times.items()},
    }


def write_compact_json_array(output_path: Path, rows) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for row in rows:
            if first:
                first = False
            else:
                f.write(",")
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        f.write("]")


class SqliteUnionRowStore:
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        if self.store_path.exists():
            self.store_path.unlink()
        self.conn = sqlite3.connect(str(self.store_path))
        self.conn.execute(
            "CREATE TABLE union_rows (frame INTEGER NOT NULL, track_id TEXT NOT NULL, track_sort INTEGER NOT NULL, row_json TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX idx_union_rows_order ON union_rows(frame, track_sort)"
        )
        self.row_count = 0

    def add_rows(self, rows) -> int:
        inserted = 0

        def iter_records():
            nonlocal inserted
            for row in rows:
                inserted += 1
                track_id = str(row["track_id"])
                yield (
                    int(row["frame"]),
                    track_id,
                    int(track_id),
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                )

        self.conn.executemany(
            "INSERT INTO union_rows(frame, track_id, track_sort, row_json) VALUES (?, ?, ?, ?)",
            iter_records(),
        )
        self.row_count += int(inserted)
        return int(inserted)

    def commit(self) -> None:
        self.conn.commit()

    def iter_rows_sorted(self):
        self.commit()
        for (row_json,) in self.conn.execute(
            "SELECT row_json FROM union_rows ORDER BY frame, track_sort"
        ):
            yield json.loads(str(row_json))

    def write_union_json(self, output_path: Path) -> None:
        write_compact_json_array(output_path, self.iter_rows_sorted())

    def write_pred_sqlite(self, output_sqlite: Path) -> None:
        output_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if output_sqlite.exists():
            output_sqlite.unlink()
        self.commit()
        out_conn = sqlite3.connect(str(output_sqlite))
        try:
            cur = out_conn.cursor()
            cur.execute(
                "CREATE TABLE masks (frame INTEGER, track_id TEXT, polygons TEXT)"
            )
            cur.executemany(
                "INSERT INTO masks(frame, track_id, polygons) VALUES (?, ?, ?)",
                (
                    (
                        int(row["frame"]),
                        str(row["track_id"]),
                        json.dumps(row["polygons"], ensure_ascii=False),
                    )
                    for row in self.iter_rows_sorted()
                ),
            )
            out_conn.commit()
        finally:
            out_conn.close()

    def evaluate_exact(
        self, tracked_sqlite: Path, output_dir: Path
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.commit()
        metrics_csv = output_dir / "keyframe_exact_metrics.csv"
        attached = False
        totals = {
            "row_count": 0.0,
            "gt_area": 0.0,
            "pred_area": 0.0,
            "intersection": 0.0,
            "union": 0.0,
            "weighted_error_total": 0.0,
            "recall_sum": 0.0,
            "precision_sum": 0.0,
            "iou_sum": 0.0,
        }
        try:
            self.conn.execute(
                "ATTACH DATABASE ? AS tracked_eval", (str(tracked_sqlite),)
            )
            attached = True
            rows_iter = self.conn.execute(
                """
                SELECT m.frame, m.track_id, m.polygons, u.row_json
                FROM tracked_eval.masks AS m
                JOIN union_rows AS u
                  ON u.frame = m.frame AND u.track_id = CAST(m.track_id AS TEXT)
                ORDER BY m.frame, CAST(m.track_id AS INTEGER)
                """
            )
            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame",
                        "track_id",
                        "run_id",
                        "has_keyframe",
                        "gt_area",
                        "pred_area",
                        "intersection",
                        "union",
                        "recall",
                        "precision",
                        "iou",
                        "weighted_error",
                    ],
                )
                writer.writeheader()
                for frame, track_id, polygons_json, row_json in rows_iter:
                    pred = json.loads(str(row_json))
                    gt_polys = parse_polygons(str(polygons_json))
                    pred_polys = [
                        np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                        for poly in pred["polygons"]
                    ]
                    metrics = compute_exact_metrics_from_polygons(gt_polys, pred_polys)
                    weighted_error = float(compute_weighted_error(metrics))
                    result_row = {
                        "frame": int(frame),
                        "track_id": str(track_id),
                        "run_id": int(pred.get("run_id", -1)),
                        "has_keyframe": int(pred.get("has_keyframe", 0)),
                        "gt_area": float(metrics["gt_area"]),
                        "pred_area": float(metrics["pred_area"]),
                        "intersection": float(metrics["intersection"]),
                        "union": float(metrics["union"]),
                        "recall": float(metrics["recall"]),
                        "precision": float(metrics["precision"]),
                        "iou": float(metrics["iou"]),
                        "weighted_error": weighted_error,
                    }
                    writer.writerow(result_row)
                    totals["row_count"] += 1.0
                    totals["gt_area"] += float(result_row["gt_area"])
                    totals["pred_area"] += float(result_row["pred_area"])
                    totals["intersection"] += float(result_row["intersection"])
                    totals["union"] += float(result_row["union"])
                    totals["weighted_error_total"] += weighted_error
                    totals["recall_sum"] += float(result_row["recall"])
                    totals["precision_sum"] += float(result_row["precision"])
                    totals["iou_sum"] += float(result_row["iou"])
        finally:
            if attached:
                self.conn.execute("DETACH DATABASE tracked_eval")
        row_count = float(totals["row_count"])
        gt_area = float(totals["gt_area"])
        pred_area = float(totals["pred_area"])
        intersection = float(totals["intersection"])
        union = float(totals["union"])
        weighted_error = float(totals["weighted_error_total"])
        optimized = {
            "row_count": row_count,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "intersection": intersection,
            "union": union,
            "global_recall": float(intersection / gt_area) if gt_area > 0 else 1.0,
            "global_precision": float(intersection / pred_area)
            if pred_area > 0
            else 1.0,
            "global_iou": float(intersection / union) if union > 0 else 1.0,
            "mean_recall": float(totals["recall_sum"] / max(row_count, 1.0)),
            "mean_precision": float(totals["precision_sum"] / max(row_count, 1.0)),
            "mean_iou": float(totals["iou_sum"] / max(row_count, 1.0)),
            "weighted_error_total": weighted_error,
            "weighted_error_mean": float(weighted_error / max(row_count, 1.0)),
        }
        summary = {
            "input_tracked_sqlite": str(tracked_sqlite),
            "optimized": optimized,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def close(self, unlink: bool = False) -> None:
        self.conn.close()
        if bool(unlink):
            try:
                self.store_path.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    args = apply_fixed_practical_defaults(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = output_dir / "opt"
    exact_dir = output_dir / "exact"
    pred_dir = output_dir / "pred"
    pred_sqlite = pred_dir / "predictions.sqlite"
    t0 = time.perf_counter()

    predictor: LearnedPointPredictor | None = None
    if bool(args.adaptive_anchor_counts):
        predictor = LearnedPointPredictor(
            Path(args.point_predictor_model_dir), str(args.predictor_device)
        )
    effective_workers = max(1, int(args.num_workers))
    streaming_rows = bool(args.stream_sqlite_rows) and effective_workers == 1
    runs: list[InstanceRun] = []
    segmentation_stats: dict[str, int] = {}
    if not streaming_rows:
        rows = load_rows(args.input_sqlite)
        runs, segmentation_stats = build_track_streams(
            rows,
            anchors_per_contour=int(args.anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(args.predictor_batch_size),
            adaptive_anchor_counts=bool(args.adaptive_anchor_counts),
            adaptive_point_quantile=float(args.adaptive_point_quantile),
            adaptive_point_offset=int(args.adaptive_point_offset),
            min_anchors_per_contour=int(args.min_anchors_per_contour),
            gapfill_enabled=bool(args.gapfill_enabled),
            gapfill_max_gap=int(args.gapfill_max_gap),
            gapfill_temp_points=int(args.gapfill_temp_points),
            max_tracks=int(args.max_tracks),
            max_run_frames=int(args.max_run_frames),
            run_overlap_frames=int(args.run_overlap_frames),
        )

    run_count = int(len(runs))
    union_rows_all: list[dict[str, object]] = []
    union_rows: list[dict[str, object]] = []
    union_store: SqliteUnionRowStore | None = None
    union_row_count = 0
    final_keyframes: list[dict[str, object]] = []
    stream_rows: list[dict[str, object]] = []
    total_interval_evals = 0
    total_interval_frames = 0
    total_candidate_frames = 0
    total_stage_times: dict[str, float] = {}

    def collect_result(result: dict[str, object]) -> None:
        nonlocal total_interval_evals, total_interval_frames, total_candidate_frames
        final_keyframes.extend(result["final_keyframes"])
        stream_row = result["stream_row"]
        if stream_row is not None:
            stream_rows.append(stream_row)
        total_interval_evals += int(result["interval_eval_count"])
        total_interval_frames += int(result["interval_eval_frames"])
        total_candidate_frames += int(result["candidate_frame_count"])
        for key, value in result.get("stage_times", {}).items():
            total_stage_times[str(key)] = float(
                total_stage_times.get(str(key), 0.0) + float(value)
            )

    if streaming_rows:
        union_store = SqliteUnionRowStore(output_dir / ".polygon_union_rows.tmp.sqlite")
        for run in iter_track_streams_from_sqlite(
            args.input_sqlite,
            anchors_per_contour=int(args.anchors_per_contour),
            predictor=predictor,
            predictor_batch_size=int(args.predictor_batch_size),
            adaptive_anchor_counts=bool(args.adaptive_anchor_counts),
            adaptive_point_quantile=float(args.adaptive_point_quantile),
            adaptive_point_offset=int(args.adaptive_point_offset),
            min_anchors_per_contour=int(args.min_anchors_per_contour),
            gapfill_enabled=bool(args.gapfill_enabled),
            gapfill_max_gap=int(args.gapfill_max_gap),
            gapfill_temp_points=int(args.gapfill_temp_points),
            max_tracks=int(args.max_tracks),
            max_run_frames=int(args.max_run_frames),
            run_overlap_frames=int(args.run_overlap_frames),
            segmentation_stats=segmentation_stats,
        ):
            run_count += 1
            result = process_single_run(run, args)
            union_store.add_rows(result["union_rows"])
            collect_result(result)
            run = None
            result = None
            __import__("gc").collect()
        if predictor is not None:
            try:
                predictor.model.to("cpu")
                torch_mod = __import__("torch")
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.synchronize()
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass
        union_store.commit()
        union_row_count = int(union_store.row_count)
    elif effective_workers == 1 or len(runs) <= 1:
        union_store = SqliteUnionRowStore(output_dir / ".polygon_union_rows.tmp.sqlite")
        for run_idx, run in enumerate(runs):
            result = process_single_run(run, args)
            union_store.add_rows(result["union_rows"])
            collect_result(result)
            runs[run_idx] = None
            result = None
            __import__("gc").collect()
        union_store.commit()
        union_row_count = int(union_store.row_count)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=mp_ctx
        ) as executor:
            results = list(executor.map(process_single_run, runs, [args] * len(runs)))
        for result in results:
            union_rows_all.extend(result["union_rows"])
            collect_result(result)
        union_rows = sorted(
            union_rows_all,
            key=lambda row: (int(row["frame"]), int(str(row["track_id"]))),
        )
        union_row_count = int(len(union_rows))
    chunk_counts_by_run: dict[tuple[str, int], int] = {}
    for row in stream_rows:
        if int(row.get("chunk_count", 1)) < 0:
            key = (str(row["track_id"]), int(row["run_id"]))
            chunk_counts_by_run[key] = max(
                int(chunk_counts_by_run.get(key, 0)), int(row["chunk_index"]) + 1
            )
    for row in stream_rows:
        if int(row.get("chunk_count", 1)) < 0:
            key = (str(row["track_id"]), int(row["run_id"]))
            chunk_count = int(chunk_counts_by_run.get(key, int(row["chunk_index"]) + 1))
            chunk_index = int(row["chunk_index"])
            row["chunk_count"] = int(chunk_count)
            row["stream_id"] = str(row["stream_id"]).replace(
                f":chunk{chunk_index + 1}:instance",
                f":chunk{chunk_index + 1}of{chunk_count}:instance",
            )
    opt_dir.mkdir(parents=True, exist_ok=True)
    if union_store is not None:
        union_store.write_union_json(opt_dir / "interpolated_union.json")
    else:
        (opt_dir / "interpolated_union.json").write_text(
            json.dumps(union_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_compact_json_array(opt_dir / "final_keyframes.json", final_keyframes)
    write_csv(
        stream_rows,
        opt_dir / "stream_segments.csv",
        [
            "stream_id",
            "track_id",
            "run_id",
            "frame_count",
            "gapfilled_frame_count",
            "contour_count",
            "anchors_per_contour",
            "run_target_total_points",
            "predicted_total_points_p90",
            "predicted_total_points_mean",
            "candidate_frame_count",
            "mean_state_count",
            "surrogate_frame_count",
            "target_keyframes",
            "chosen_keyframes",
            "achieved_ratio",
            "keyframe_rate",
            "objective",
            "mean_interval_frame_loss",
            "mean_shape_distance",
            "mean_shape_update",
            "shape_distance_total",
            "shape_distance_rate",
            "shape_update_count",
            "shape_update_rate",
            "mean_shape_distance_scale",
            "mean_shape_switch_scale",
            "mean_recall_budget",
            "achieved_recall_floor",
            "recall_budget_violation",
            "interval_eval_count",
            "interval_eval_frames",
            "max_saliency",
            "build_eval_contexts_seconds",
            "build_candidates_seconds",
            "build_candidate_pool_seconds",
            "solve_dp_seconds",
            "pair_vote_refine_seconds",
            "exact_recall_repair_seconds",
            "final_eval_seconds",
            "emit_frame_count",
            "chunk_index",
            "chunk_count",
            "chunk_process_start",
            "chunk_process_end",
        ],
    )

    optimizer_seconds = float(time.perf_counter() - t0)
    optimizer_summary = {
        "description": "Production polygon optimizer kernel with gapfill-first track-level anchor counts and pair-vote keyframe-shape refinement.",
        "input_sqlite": str(args.input_sqlite),
        "output_dir": str(opt_dir),
        "run_count": int(run_count),
        "gapfill_enabled": bool(args.gapfill_enabled),
        "gapfill_max_gap": int(args.gapfill_max_gap),
        "gapfill_temp_points": int(args.gapfill_temp_points),
        "max_run_frames": int(args.max_run_frames),
        "run_overlap_frames": int(args.run_overlap_frames),
        "num_workers": int(effective_workers),
        "stream_sqlite_rows": bool(streaming_rows),
        "row_count": int(union_row_count),
        "target_ratio": float(args.target_ratio),
        "anchors_per_contour_cap": int(args.anchors_per_contour),
        "adaptive_anchor_counts": bool(args.adaptive_anchor_counts),
        "point_predictor_model_dir": str(args.point_predictor_model_dir)
        if bool(args.adaptive_anchor_counts)
        else None,
        "predictor_device": str(args.predictor_device),
        "predictor_batch_size": int(args.predictor_batch_size),
        "adaptive_point_quantile": float(args.adaptive_point_quantile),
        "adaptive_point_offset": int(args.adaptive_point_offset),
        "min_anchors_per_contour": int(args.min_anchors_per_contour),
        "solver_mode": str(args.solver_mode),
        "recall_constraint_mode": str(args.recall_constraint_mode),
        "recall_min": float(args.recall_min),
        "pair_vote_refine_enabled": bool(args.pair_vote_refine_enabled),
        "surrogate_pool_factor": float(args.surrogate_pool_factor),
        "surrogate_peak_factor": float(args.surrogate_peak_factor),
        "surrogate_neighbor_radius": int(args.surrogate_neighbor_radius),
        "surrogate_shape_weight": float(args.surrogate_shape_weight),
        "saliency_shape_eta": float(args.saliency_shape_eta),
        "saliency_area_eta": float(args.saliency_area_eta),
        "shape_switch_weight": float(args.shape_switch_weight),
        "shape_distance_weight": float(args.shape_distance_weight),
        "shape_update_threshold_ratio": float(args.shape_update_threshold_ratio),
        "penalty_binary_steps": int(args.penalty_binary_steps),
        "recall_budget_binary_steps": int(args.recall_budget_binary_steps),
        "dp_eval_scale": float(args.dp_eval_scale),
        "dp_eval_pad": int(args.dp_eval_pad),
        "segmentation_stats": {
            key: int(val) for key, val in segmentation_stats.items()
        },
        "mean_run_anchors_per_contour": float(
            np.mean(
                np.asarray(
                    [row["anchors_per_contour"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "median_run_anchors_per_contour": float(
            np.median(
                np.asarray(
                    [row["anchors_per_contour"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "max_run_anchors_per_contour": int(
            max(int(row["anchors_per_contour"]) for row in stream_rows)
        )
        if stream_rows
        else 0,
        "mean_gapfilled_frame_count": float(
            np.mean(
                np.asarray(
                    [row["gapfilled_frame_count"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_achieved_ratio": float(
            np.mean(
                np.asarray(
                    [row["achieved_ratio"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_keyframe_rate": float(
            np.mean(
                np.asarray(
                    [row["keyframe_rate"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_state_count": float(
            np.mean(
                np.asarray(
                    [row["mean_state_count"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_interval_frame_loss": float(
            np.mean(
                np.asarray(
                    [row["mean_interval_frame_loss"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_distance": float(
            np.mean(
                np.asarray(
                    [row["mean_shape_distance"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_update": float(
            np.mean(
                np.asarray(
                    [row["mean_shape_update"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_distance_rate": float(
            np.mean(
                np.asarray(
                    [row["shape_distance_rate"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_shape_update_rate": float(
            np.mean(
                np.asarray(
                    [row["shape_update_rate"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_recall_budget": float(
            np.mean(
                np.asarray(
                    [row["mean_recall_budget"] for row in stream_rows], dtype=np.float64
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_achieved_recall_floor": float(
            np.mean(
                np.asarray(
                    [row["achieved_recall_floor"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 1.0,
        "mean_recall_budget_violation": float(
            np.mean(
                np.asarray(
                    [row["recall_budget_violation"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "mean_candidate_frame_count": float(
            np.mean(
                np.asarray(
                    [row["candidate_frame_count"] for row in stream_rows],
                    dtype=np.float64,
                )
            )
        )
        if stream_rows
        else 0.0,
        "interval_eval_count": int(total_interval_evals),
        "interval_eval_frames": int(total_interval_frames),
        "candidate_frame_count_total": int(total_candidate_frames),
        "optimizer_seconds": float(optimizer_seconds),
        "stage_seconds_total": {
            key: float(val) for key, val in sorted(total_stage_times.items())
        },
        "stage_seconds_mean_per_run": {
            key: float(val / max(run_count, 1))
            for key, val in sorted(total_stage_times.items())
        },
        "artifacts": {
            "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
            "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
            "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
        },
    }
    (opt_dir / "summary.json").write_text(
        json.dumps(optimizer_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exact_summary: dict[str, object] | None = None
    if union_store is not None:
        try:
            if bool(args.evaluate_exact):
                exact_summary = union_store.evaluate_exact(args.input_sqlite, exact_dir)
            if bool(args.write_pred_sqlite):
                union_store.write_pred_sqlite(pred_sqlite)
        finally:
            union_store.close(unlink=True)
    else:
        if bool(args.evaluate_exact):
            exact_summary = evaluate_union_exact(
                union_rows, args.input_sqlite, exact_dir
            )
        if bool(args.write_pred_sqlite):
            union_rows_to_pred_sqlite(union_rows, pred_sqlite)

    summary = {
        "description": "Production polygon optimizer with gapfill-first track-level anchor counts.",
        "input_sqlite": str(args.input_sqlite),
        "output_dir": str(output_dir),
        "optimizer_summary": optimizer_summary,
        "exact_summary": exact_summary,
        "artifacts": {
            "interpolated_union_json": str(opt_dir / "interpolated_union.json"),
            "final_keyframes_json": str(opt_dir / "final_keyframes.json"),
            "stream_segments_csv": str(opt_dir / "stream_segments.csv"),
            "optimizer_summary_json": str(opt_dir / "summary.json"),
            "exact_summary_json": None
            if exact_summary is None
            else str(exact_dir / "summary.json"),
            "pred_sqlite": str(pred_sqlite) if bool(args.write_pred_sqlite) else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
