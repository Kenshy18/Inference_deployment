"""Strict checkpoint loading for the EVA02 expanded classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .contracts import contract_from_checkpoint
from .model import RoiRichSpatialAttnFusionClassifier


def classifier_from_checkpoint(
    checkpoint_path: Path, map_location: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier checkpoint root must be a mapping")
    contract = contract_from_checkpoint(checkpoint)
    cfg = checkpoint.get("model_cfg") or {}
    model = RoiRichSpatialAttnFusionClassifier(
        input_dim=contract.input_dim,
        num_classes=contract.num_classes,
        use_meta=contract.use_meta,
        pooler_channels=int(cfg.get("pooler_channels", 256)),
        pooler_size=int(cfg.get("pooler_size", 7)),
        mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
        local_branch_dim=int(cfg.get("local_branch_dim", 128)),
        expanded_branch_dim=int(cfg.get("expanded_branch_dim", 128)),
        mask_branch_dim=int(cfg.get("mask_branch_dim", 128)),
        box_head_proj_dim=int(cfg.get("box_head_proj_dim", 320)),
        dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
        attn_token_dim=int(cfg.get("attn_token_dim", 192)),
        attn_num_heads=int(cfg.get("attn_num_heads", 4)),
        attn_dropout=float(cfg.get("attn_dropout", 0.0)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
        head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
        head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
    )
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise KeyError(f"checkpoint missing model_state: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint


__all__ = ["classifier_from_checkpoint"]
