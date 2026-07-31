"""Minimal inference-only heads used by the delivered backbone classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


META_DIM = 5


class SpatialConvClassifier(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.use_meta = bool(cfg.get("use_meta", True))
        self.channels = int(cfg["pooler_channels"])
        self.size = int(cfg["pooler_size"])
        conv_channels = [int(value) for value in cfg.get("conv_channels", (96, 64))]
        kernels = [int(value) for value in cfg.get("conv_kernels", (1, 3))]
        if len(kernels) == 1:
            kernels *= len(conv_channels)
        if len(kernels) != len(conv_channels):
            raise ValueError("conv_kernels must match conv_channels")
        layers: list[nn.Module] = []
        input_channels = self.channels
        for output_channels, kernel in zip(conv_channels, kernels):
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),
                )
            )
            dropout = float(cfg.get("conv_dropout", 0.0))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            input_channels = output_channels
        self.conv = nn.Sequential(*layers)
        head_input = input_channels * self.size**2 + (META_DIM if self.use_meta else 0)
        hidden = int(cfg.get("head_hidden_dim", 512))
        count = max(1, int(cfg.get("head_num_layers", 2)))
        head: list[nn.Module] = []
        current = head_input
        for _ in range(count - 1):
            head.extend(
                (
                    nn.Linear(current, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(cfg.get("head_dropout", 0.2))),
                )
            )
            current = hidden
        head.append(nn.Linear(current, int(cfg.get("num_classes", 3))))
        self.head = nn.Sequential(*head)

    def forward(self, roi: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        value = _reshape_roi(roi, self.channels, self.size)
        features = self.conv(value).flatten(1)
        if self.use_meta:
            features = torch.cat((features, metadata), dim=1)
        return self.head(features)


class SpatialGapClassifier(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.use_meta = bool(cfg.get("use_meta", True))
        self.channels = int(cfg["pooler_channels"])
        self.size = int(cfg["pooler_size"])
        stem = int(cfg.get("gap_stem_channels", 64))
        middle = int(cfg.get("gap_mid_channels", 64))
        kernel = int(cfg.get("gap_dw_kernel", 3))
        self.stem = nn.Sequential(
            nn.Conv2d(self.channels, stem, 1, bias=False),
            nn.BatchNorm2d(stem),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                stem,
                stem,
                kernel,
                padding=kernel // 2,
                groups=stem,
                bias=False,
            ),
            nn.BatchNorm2d(stem),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(stem, middle, 1, bias=False),
            nn.BatchNorm2d(middle),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        dropout = float(cfg.get("gap_dropout", 0.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(
            middle + (META_DIM if self.use_meta else 0),
            int(cfg.get("num_classes", 3)),
        )

    def forward(self, roi: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        value = _reshape_roi(roi, self.channels, self.size)
        features = self.dropout(
            self.pool(self.proj(self.depthwise(self.stem(value)))).flatten(1)
        )
        if self.use_meta:
            features = torch.cat((features, metadata), dim=1)
        return self.head(features)


class SpatialStatsClassifier(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.use_meta = bool(cfg.get("use_meta", True))
        self.channels = int(cfg["pooler_channels"])
        self.size = int(cfg["pooler_size"])
        hidden = int(cfg.get("stats_hidden_dim", 512))
        dropout = float(cfg.get("stats_dropout", 0.15))
        stats = self.channels * 3
        self.stats_norm = nn.LayerNorm(stats)
        self.project = nn.Sequential(
            nn.Linear(stats, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(
            hidden + (META_DIM if self.use_meta else 0),
            int(cfg.get("num_classes", 3)),
        )

    def forward(self, roi: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        value = _reshape_roi(roi, self.channels, self.size)
        mean = value.mean(dim=(2, 3))
        maximum = value.amax(dim=(2, 3))
        std = (
            value.float()
            .var(dim=(2, 3), unbiased=False)
            .add_(1e-6)
            .sqrt_()
            .to(value.dtype)
        )
        features = self.project(
            self.stats_norm(torch.cat((mean, maximum, std), dim=1))
        )
        features = features + self.residual(features)
        if self.use_meta:
            features = torch.cat((features, metadata), dim=1)
        return self.head(features)


def _reshape_roi(value: torch.Tensor, channels: int, size: int) -> torch.Tensor:
    if value.ndim == 2:
        expected = channels * size**2
        if int(value.shape[1]) != expected:
            raise ValueError(f"ROI feature width must be {expected}")
        return value.contiguous().view(-1, channels, size, size)
    if value.ndim != 4 or tuple(value.shape[1:]) != (channels, size, size):
        raise ValueError(
            f"ROI feature must be [N,{channels},{size},{size}], got {tuple(value.shape)}"
        )
    return value


def build_model(config: Mapping[str, object]) -> nn.Module:
    model_type = str(config.get("model_type", "")).strip().lower()
    expected = int(config["pooler_channels"]) * int(config["pooler_size"]) ** 2
    if int(config.get("input_dim", 0)) != expected:
        raise ValueError("classifier input_dim does not equal C*H*W")
    constructors = {
        "spatial_conv": SpatialConvClassifier,
        "spatial_gap": SpatialGapClassifier,
        "spatial_stats": SpatialStatsClassifier,
    }
    try:
        return constructors[model_type](config)
    except KeyError as exc:
        raise ValueError(f"unsupported backbone classifier type: {model_type!r}") from exc


__all__ = ["build_model"]
