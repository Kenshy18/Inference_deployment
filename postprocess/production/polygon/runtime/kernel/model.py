"""Mask descriptors and learned point-count prediction."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

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


__all__ = (
    "FEATURE_NAMES",
    "ConvBNAct",
    "LearnedPointPredictor",
    "TinyMaskPointNet",
    "build_feature_vector",
    "compute_mask_descriptors",
    "resize_mask_with_padding",
)
