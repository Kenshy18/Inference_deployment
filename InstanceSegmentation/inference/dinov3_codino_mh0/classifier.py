"""MH0-specific three-class classifier contract.

The detector is intentionally one-class (``foreground``).  This classifier
maps each 192x14x14 mask ROI feature to the application classes while keeping
the detector label and classifier label separate in the shared SQLite schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

try:
    from dinov3_codino.classifier import (
        RoiSpatialGapClassifier,
        contract_from_checkpoint as base_contract_from_checkpoint,
    )
except ImportError:
    from ..dinov3_codino.classifier import (
        RoiSpatialGapClassifier,
        contract_from_checkpoint as base_contract_from_checkpoint,
    )


POOLER_CHANNELS = 192
POOLER_SIZE = 14
INPUT_DIM = POOLER_CHANNELS * POOLER_SIZE * POOLER_SIZE
CLASS_NAMES = ("女性器", "男性器", "結合部分")
CLASS_IDS = (1, 2, 3)


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject a classifier trained for another detector feature space."""

    contract = base_contract_from_checkpoint(checkpoint)
    cfg = checkpoint.get("model_cfg") or {}
    channels = int(cfg.get("pooler_channels", 0))
    size = int(cfg.get("pooler_size", 0))
    if (
        contract.input_dim != INPUT_DIM
        or channels != POOLER_CHANNELS
        or size != POOLER_SIZE
    ):
        raise ValueError(
            "MH0 classifier must consume "
            f"[N,{POOLER_CHANNELS},{POOLER_SIZE},{POOLER_SIZE}] features; "
            f"checkpoint declares input_dim={contract.input_dim}, "
            f"pooler_channels={channels}, pooler_size={size}"
        )
    class_names = tuple(str(value) for value in checkpoint.get("class_names") or ())
    class_ids = tuple(int(value) for value in checkpoint.get("class_ids") or ())
    if class_names != CLASS_NAMES or class_ids != CLASS_IDS:
        raise ValueError(
            "MH0 classifier classes must be "
            f"names={CLASS_NAMES!r}, ids={CLASS_IDS!r}; "
            f"got names={class_names!r}, ids={class_ids!r}"
        )


def classifier_from_checkpoint(
    checkpoint_path: Path,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("MH0 classifier checkpoint root must be a mapping")
    validate_checkpoint(checkpoint)
    cfg = checkpoint["model_cfg"]
    model = RoiSpatialGapClassifier(
        input_dim=INPUT_DIM,
        num_classes=len(CLASS_NAMES),
        use_meta=bool(cfg.get("use_meta", True)),
        pooler_channels=POOLER_CHANNELS,
        pooler_size=POOLER_SIZE,
        stem_channels=int(cfg.get("gap_stem_channels", 96)),
        mid_channels=int(cfg.get("gap_mid_channels", 96)),
        dw_kernel=int(cfg.get("gap_dw_kernel", 3)),
        dropout=float(cfg.get("gap_dropout", cfg.get("dropout", 0.0))),
    )
    state_dict = checkpoint.get("model_state")
    if not isinstance(state_dict, Mapping):
        raise KeyError(f"checkpoint missing model_state: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint


__all__ = [
    "CLASS_IDS",
    "CLASS_NAMES",
    "INPUT_DIM",
    "POOLER_CHANNELS",
    "POOLER_SIZE",
    "classifier_from_checkpoint",
    "validate_checkpoint",
]
