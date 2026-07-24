"""Family-owned EVA02 expanded rich-spatial attention classifier."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


META_DIM = 5


def _build_dwconv_gap_branch(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    out_ch = int(out_ch)
    k = int(k)
    pad = int(k // 2)
    return nn.Sequential(
        nn.Conv2d(int(in_ch), out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=k,
            stride=1,
            padding=pad,
            groups=out_ch,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )


class RoiRichSpatialAttnFusionClassifier(nn.Module):
    """Cross-branch token attention on local/expanded/mask/box descriptors."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 128,
        expanded_branch_dim: int = 128,
        mask_branch_dim: int = 128,
        box_head_proj_dim: int = 320,
        dw_kernel: int = 3,
        attn_token_dim: int = 192,
        attn_num_heads: int = 4,
        attn_dropout: float = 0.0,
        token_pool_type: str = "mean",
        branch_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialAttnFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")
        token_dim = int(attn_token_dim)
        if token_dim <= 0:
            raise ValueError(f"attn_token_dim must be >0, got {token_dim}")
        num_heads = int(attn_num_heads)
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError(
                f"attn_num_heads must divide attn_token_dim: heads={num_heads}, dim={token_dim}"
            )
        self.token_pool_type = str(token_pool_type).strip().lower()
        if self.token_pool_type not in {"mean", "attn"}:
            raise ValueError(
                f"token_pool_type must be one of ['mean','attn'], got {token_pool_type}"
            )
        self.branch_dropout = float(branch_dropout)
        if not (0.0 <= self.branch_dropout < 1.0):
            raise ValueError(f"branch_dropout must be in [0,1), got {branch_dropout}")

        self.local_branch = _build_dwconv_gap_branch(
            self.pooler_channels, int(local_branch_dim), k=k
        )
        self.expanded_branch = _build_dwconv_gap_branch(
            self.pooler_channels, int(expanded_branch_dim), k=k
        )
        self.mask_branch = _build_dwconv_gap_branch(
            self.pooler_channels, int(mask_branch_dim), k=k
        )
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        self.local_token = nn.Linear(int(local_branch_dim), token_dim)
        self.expanded_token = nn.Linear(int(expanded_branch_dim), token_dim)
        self.mask_token = nn.Linear(int(mask_branch_dim), token_dim)
        self.box_token = nn.Linear(int(box_head_proj_dim), token_dim)

        self.pre_ln = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ffn_ln = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.token_pool = None
        if self.token_pool_type == "attn":
            self.token_pool = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 1, bias=False),
            )

        fused_dim = token_dim
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_feat_expanded: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(
                    f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}"
                )
            local = roi_feat.contiguous().view(
                -1, self.pooler_channels, self.pooler_size, self.pooler_size
            )
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(
                f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}"
            )

        if box_pooler_feat_expanded is None or box_pooler_feat_expanded.ndim != 4:
            raise ValueError("box_pooler_feat_expanded (rank-4) is required")
        if mask_pooler_feat is None or mask_pooler_feat.ndim != 4:
            raise ValueError("mask_pooler_feat (rank-4) is required")
        if (
            box_head_feat is None
            or box_head_feat.ndim != 2
            or int(box_head_feat.shape[1]) != 1024
        ):
            raise ValueError("box_head_feat [N,1024] is required")

        local_vec = self.local_branch(local).flatten(start_dim=1)
        expanded_vec = self.expanded_branch(box_pooler_feat_expanded).flatten(
            start_dim=1
        )
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        tokens = torch.stack(
            [
                self.local_token(local_vec),
                self.expanded_token(expanded_vec),
                self.mask_token(mask_vec),
                self.box_token(box_vec),
            ],
            dim=1,
        )
        if self.training and self.branch_dropout > 0.0:
            keep_prob = 1.0 - self.branch_dropout
            keep = (
                torch.rand(tokens.shape[:2], device=tokens.device) < keep_prob
            ).float()
            drop_all = keep.sum(dim=1) <= 0.0
            if bool(drop_all.any()):
                keep[drop_all, 0] = 1.0
            tokens = tokens * (keep.unsqueeze(-1) / keep_prob)

        attn_in = self.pre_ln(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_ln(tokens))
        if self.token_pool_type == "attn":
            assert self.token_pool is not None
            pool_scores = self.token_pool(tokens).squeeze(-1)
            pool_weights = torch.softmax(pool_scores, dim=1)
            pooled = torch.sum(tokens * pool_weights.unsqueeze(-1), dim=1)
        else:
            pooled = tokens.mean(dim=1)

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            pooled = torch.cat([pooled, meta], dim=1)
        return self.head(pooled)


__all__ = ["RoiRichSpatialAttnFusionClassifier"]
