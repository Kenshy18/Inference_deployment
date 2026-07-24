from __future__ import annotations

import sys
import types
from pathlib import Path

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parent


def _register_inline_module(
    module_name: str, export_map: dict[str, str]
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(SELF_PATH)
    for public_name, global_name in export_map.items():
        value = globals()[global_name]
        setattr(module, public_name, value)
        setattr(module, global_name, value)
    sys.modules[module_name] = module
    return module


import math
import cv2
import numpy as np
import torch
from torch import nn
from .runtime_fst import fst

k2v5_ELLIPSE_CENTER_MIN = -0.75
k2v5_ELLIPSE_CENTER_MAX = 1.75
k2v5_LOG_AXIS_MIN = -9.0
k2v5_LOG_AXIS_MAX = 0.75
k2v5__COORD_GRID_CACHE: dict[int, np.ndarray] = {}
k2v5__BORDER_GRID_CACHE: dict[int, np.ndarray] = {}


def k2v5_square_pad_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    padded = np.zeros((side, side), dtype=np.uint8)
    padded[pad_top : pad_top + height, pad_left : pad_left + width] = mask
    return (padded, (pad_left, pad_top), side)


def k2v5_build_signed_distance_channel(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside - outside
    scale = float(max(mask.shape))
    return (signed / max(scale, 1.0)).astype(np.float32)


def k2v5_build_edge_channel(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return edge.astype(np.float32)


def k2v5_get_coord_grid(image_size: int) -> np.ndarray:
    cached = k2v5__COORD_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    stacked = np.stack([xx, yy], axis=0)
    k2v5__COORD_GRID_CACHE[image_size] = stacked
    return stacked


def k2v5_get_border_distance_grid(image_size: int) -> np.ndarray:
    cached = k2v5__BORDER_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    border = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy]).astype(np.float32)
    maxv = float(border.max()) if border.size else 1.0
    if maxv > 0.0:
        border /= maxv
    border = border[None, ...]
    k2v5__BORDER_GRID_CACHE[image_size] = border
    return border


def k2v5_build_touch_flag_planes(
    image_size: int, touch_flags: np.ndarray
) -> np.ndarray:
    planes = np.broadcast_to(
        touch_flags.astype(np.float32)[:, None, None], (4, image_size, image_size)
    )
    return np.asarray(planes, dtype=np.float32)


def k2v5_edge_touch_vector_from_row(
    row: dict[str, object], gt_mask: np.ndarray | None = None
) -> np.ndarray:
    meta = row.get("source_metadata")
    if isinstance(meta, dict):
        sides = meta.get("edge_sides")
        if isinstance(sides, dict):
            return np.asarray(
                [
                    float(bool(sides.get("left", False))),
                    float(bool(sides.get("right", False))),
                    float(bool(sides.get("top", False))),
                    float(bool(sides.get("bottom", False))),
                ],
                dtype=np.float32,
            )
    if gt_mask is None:
        return np.zeros((4,), dtype=np.float32)
    touches = fst.detect_edge_touches(gt_mask.astype(np.uint8))
    return np.asarray(
        [
            float(bool(touches.get("left", False))),
            float(bool(touches.get("right", False))),
            float(bool(touches.get("top", False))),
            float(bool(touches.get("bottom", False))),
        ],
        dtype=np.float32,
    )


def k2v5_build_input_image(
    mask_resized: np.ndarray,
    signed_resized: np.ndarray,
    edge_resized: np.ndarray,
    touch_flags: np.ndarray,
    image_size: int,
) -> np.ndarray:
    coords = k2v5_get_coord_grid(image_size)
    border = k2v5_get_border_distance_grid(image_size)
    touch_planes = k2v5_build_touch_flag_planes(image_size, touch_flags)
    return np.concatenate(
        [
            mask_resized[None, ...].astype(np.float32, copy=False),
            signed_resized[None, ...].astype(np.float32, copy=False),
            edge_resized[None, ...].astype(np.float32, copy=False),
            coords.astype(np.float32, copy=False),
            border.astype(np.float32, copy=False),
            touch_planes.astype(np.float32, copy=False),
        ],
        axis=0,
    )


def k2v5_states_to_abs_ellipses_from_payload(
    states: np.ndarray,
    payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]],
) -> list[tuple[float, float, float, float, float]]:
    if states.ndim == 1:
        states = states.reshape(2, 6)
    (height, width), origin, _ = payload
    side = max(height, width)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    absolute: list[tuple[float, float, float, float, float]] = []
    for state in states:
        cx_n, cy_n, loga, logb, cos2, sin2 = [float(v) for v in state]
        cx_n = min(max(cx_n, k2v5_ELLIPSE_CENTER_MIN), k2v5_ELLIPSE_CENTER_MAX)
        cy_n = min(max(cy_n, k2v5_ELLIPSE_CENTER_MIN), k2v5_ELLIPSE_CENTER_MAX)
        loga = min(max(loga, k2v5_LOG_AXIS_MIN), k2v5_LOG_AXIS_MAX)
        logb = min(max(logb, k2v5_LOG_AXIS_MIN), k2v5_LOG_AXIS_MAX)
        norm = math.hypot(cos2, sin2)
        if not math.isfinite(norm) or norm < 1e-06:
            cos2, sin2 = (1.0, 0.0)
        else:
            cos2 /= norm
            sin2 /= norm
        cx = cx_n * side - pad_left
        cy = cy_n * side - pad_top
        a = math.exp(loga) * side
        b = math.exp(logb) * side
        angle = math.degrees(0.5 * math.atan2(sin2, cos2))
        absolute.append(
            fst.normalize_ellipse((cx + origin[0], cy + origin[1], a, b, angle))
        )
    return absolute


class k2v5_ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class k2v5_SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class k2v5_ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            k2v5_ConvBNAct(channels, hidden, 1),
            k2v5_ConvBNAct(hidden, hidden, 3, groups=hidden),
            k2v5_SqueezeExcite(hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class k2v5_SlotDecoderBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_q3 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q1(queries)
        c = self.norm_ctx(context)
        queries = queries + self.cross_attn(q, c, c, need_weights=False)[0]
        q = self.norm_q2(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.mlp(self.norm_q3(queries))
        return queries


def k2v5_render_soft_slots_from_spd(
    centers: torch.Tensor, chol_params: torch.Tensor, image_size: int, sharpness: float
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_slots, _ = centers.shape
    grid = torch.linspace(
        0.0, 1.0, image_size, device=centers.device, dtype=centers.dtype
    )
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx.view(1, 1, image_size, image_size)
    yy = yy.view(1, 1, image_size, image_size)
    l11 = (
        torch.nn.functional.softplus(chol_params[..., 0]).view(batch, n_slots, 1, 1)
        + 0.0001
    )
    l21 = chol_params[..., 1].view(batch, n_slots, 1, 1)
    l22 = (
        torch.nn.functional.softplus(chol_params[..., 2]).view(batch, n_slots, 1, 1)
        + 0.0001
    )
    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22
    dx = xx - centers[..., 0].view(batch, n_slots, 1, 1)
    dy = yy - centers[..., 1].view(batch, n_slots, 1, 1)
    quad = a11 * dx * dx + 2.0 * a12 * dx * dy + a22 * dy * dy
    q = 1.0 - quad
    slot_masks = torch.sigmoid(q * sharpness)
    union = 1.0 - torch.prod(1.0 - slot_masks, dim=1)
    return (slot_masks, union)


def k2v5_spd_to_normalized_states(
    centers: torch.Tensor, chol_params: torch.Tensor
) -> torch.Tensor:
    l11 = torch.nn.functional.softplus(chol_params[..., 0]) + 0.0001
    l21 = chol_params[..., 1]
    l22 = torch.nn.functional.softplus(chol_params[..., 2]) + 0.0001
    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22
    trace = a11 + a22
    disc = torch.sqrt(((a11 - a22) ** 2 + 4.0 * a12 * a12).clamp_min(1e-10))
    lam_min = ((trace - disc) * 0.5).clamp_min(1e-08)
    lam_max = ((trace + disc) * 0.5).clamp_min(1e-08)
    major = torch.rsqrt(lam_min).clamp_min(0.0001)
    minor = torch.rsqrt(lam_max).clamp_min(0.0001)
    det = (a11 * a22 - a12 * a12).clamp_min(1e-10)
    cov_xx = a22 / det
    cov_xy = -a12 / det
    cov_yy = a11 / det
    denom = torch.sqrt((cov_xx - cov_yy) ** 2 + (2.0 * cov_xy) ** 2).clamp_min(1e-08)
    cos2 = (cov_xx - cov_yy) / denom
    sin2 = 2.0 * cov_xy / denom
    return torch.stack(
        [
            centers[..., 0],
            centers[..., 1],
            torch.log(major),
            torch.log(minor),
            cos2,
            sin2,
        ],
        dim=-1,
    )


class k2v5_K2SlotSetSPDNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_width: int,
        slot_dim: int,
        decoder_layers: int,
        num_heads: int,
        sharpness: float,
    ) -> None:
        super().__init__()
        c1 = int(base_width)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 12
        self.sharpness = float(sharpness)
        self.stem = nn.Sequential(
            k2v5_ConvBNAct(in_channels, c1, 3, stride=1),
            k2v5_ResidualBlock(c1),
            k2v5_ResidualBlock(c1),
        )
        self.stage2 = nn.Sequential(
            k2v5_ConvBNAct(c1, c2, 3, stride=2),
            k2v5_ResidualBlock(c2),
            k2v5_ResidualBlock(c2),
        )
        self.stage3 = nn.Sequential(
            k2v5_ConvBNAct(c2, c3, 3, stride=2),
            k2v5_ResidualBlock(c3),
            k2v5_ResidualBlock(c3),
        )
        self.stage4 = nn.Sequential(
            k2v5_ConvBNAct(c3, c4, 3, stride=2),
            k2v5_ResidualBlock(c4),
            k2v5_ResidualBlock(c4),
            k2v5_ResidualBlock(c4),
        )
        self.stage5 = nn.Sequential(
            k2v5_ConvBNAct(c4, c5, 3, stride=2),
            k2v5_ResidualBlock(c5),
            k2v5_ResidualBlock(c5),
            k2v5_ResidualBlock(c5),
        )
        self.lat5 = nn.Conv2d(c5, slot_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, slot_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, slot_dim, kernel_size=1)
        self.fpn4 = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.fpn3 = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.context_proj = k2v5_ConvBNAct(slot_dim, slot_dim, 3)
        self.slot_queries = nn.Parameter(torch.randn(2, slot_dim) * 0.02)
        self.decoder = nn.ModuleList(
            [
                k2v5_SlotDecoderBlock(slot_dim, num_heads=num_heads, mlp_ratio=4)
                for _ in range(decoder_layers)
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(slot_dim, slot_dim, 1),
            nn.SiLU(inplace=True),
        )
        self.slot_refine = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim)
        )
        self.center_head = nn.Linear(slot_dim, 2)
        self.chol_head = nn.Linear(slot_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        p5 = self.lat5(s5)
        p4 = self.fpn4(
            self.lat4(s4)
            + torch.nn.functional.interpolate(
                p5, size=s4.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        p3 = self.fpn3(
            self.lat3(s3)
            + torch.nn.functional.interpolate(
                p4, size=s3.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        context_map = self.context_proj(p4)
        context_tokens = context_map.flatten(2).transpose(1, 2)
        queries = self.slot_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        global_feat = self.global_pool(
            torch.nn.functional.interpolate(
                p3, size=context_map.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        global_feat = global_feat.flatten(1).unsqueeze(1).expand(-1, 2, -1)
        queries = queries + self.slot_refine(torch.cat([queries, global_feat], dim=-1))
        for block in self.decoder:
            queries = block(queries, context_tokens)
        centers = self.center_head(queries)
        chol_params = self.chol_head(queries)
        slot_masks, union_mask = k2v5_render_soft_slots_from_spd(
            centers, chol_params, image_size=x.shape[-1], sharpness=self.sharpness
        )
        states = k2v5_spd_to_normalized_states(centers, chol_params)
        return {
            "states": states.reshape(x.shape[0], 12),
            "centers": centers,
            "chol_params": chol_params,
            "slot_masks": slot_masks,
            "union_mask": union_mask,
        }


def k2v5_pair_cost(
    pred_states: torch.Tensor, target_states: torch.Tensor
) -> torch.Tensor:
    weights = torch.tensor(
        [2.0, 2.0, 1.5, 1.5, 0.5, 0.5],
        device=pred_states.device,
        dtype=pred_states.dtype,
    )
    diff = torch.nn.functional.smooth_l1_loss(
        pred_states, target_states, reduction="none"
    )
    return (diff * weights.view(1, 1, 6)).mean(dim=(1, 2))


def k2v5_states_to_geometry_tensors(
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = states[..., :2]
    log_major = states[..., 2]
    log_minor = states[..., 3]
    cos2 = states[..., 4].clamp(-1.0, 1.0)
    sin2 = states[..., 5].clamp(-1.0, 1.0)
    cos_theta = torch.sqrt(((1.0 + cos2) * 0.5).clamp_min(1e-08))
    sin_theta = torch.sign(sin2) * torch.sqrt(((1.0 - cos2) * 0.5).clamp_min(1e-08))
    major2 = torch.exp(2.0 * log_major).clamp_min(1e-08)
    minor2 = torch.exp(2.0 * log_minor).clamp_min(1e-08)
    c2 = cos_theta * cos_theta
    s2 = sin_theta * sin_theta
    cs = cos_theta * sin_theta
    cov_xx = c2 * major2 + s2 * minor2
    cov_xy = cs * (major2 - minor2)
    cov_yy = s2 * major2 + c2 * minor2
    cov = torch.stack([cov_xx, cov_xy, cov_yy], dim=-1)
    return (centers, cov)


k2v5 = _register_inline_module(
    "standalone_runtime_k2v5",
    {
        "ELLIPSE_CENTER_MIN": "k2v5_ELLIPSE_CENTER_MIN",
        "ELLIPSE_CENTER_MAX": "k2v5_ELLIPSE_CENTER_MAX",
        "LOG_AXIS_MIN": "k2v5_LOG_AXIS_MIN",
        "LOG_AXIS_MAX": "k2v5_LOG_AXIS_MAX",
        "_COORD_GRID_CACHE": "k2v5__COORD_GRID_CACHE",
        "_BORDER_GRID_CACHE": "k2v5__BORDER_GRID_CACHE",
        "square_pad_mask": "k2v5_square_pad_mask",
        "build_signed_distance_channel": "k2v5_build_signed_distance_channel",
        "build_edge_channel": "k2v5_build_edge_channel",
        "get_coord_grid": "k2v5_get_coord_grid",
        "get_border_distance_grid": "k2v5_get_border_distance_grid",
        "build_touch_flag_planes": "k2v5_build_touch_flag_planes",
        "edge_touch_vector_from_row": "k2v5_edge_touch_vector_from_row",
        "build_input_image": "k2v5_build_input_image",
        "states_to_abs_ellipses_from_payload": "k2v5_states_to_abs_ellipses_from_payload",
        "ConvBNAct": "k2v5_ConvBNAct",
        "SqueezeExcite": "k2v5_SqueezeExcite",
        "ResidualBlock": "k2v5_ResidualBlock",
        "SlotDecoderBlock": "k2v5_SlotDecoderBlock",
        "render_soft_slots_from_spd": "k2v5_render_soft_slots_from_spd",
        "spd_to_normalized_states": "k2v5_spd_to_normalized_states",
        "K2SlotSetSPDNet": "k2v5_K2SlotSetSPDNet",
        "pair_cost": "k2v5_pair_cost",
        "states_to_geometry_tensors": "k2v5_states_to_geometry_tensors",
    },
)
