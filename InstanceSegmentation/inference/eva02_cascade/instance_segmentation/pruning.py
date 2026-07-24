"""Structural EVA02 ViT block pruning owned by the model family."""

from __future__ import annotations

import torch.nn as nn


def drop_blocks(
    model, block_indices: tuple[int, ...]
) -> tuple[object, tuple[int, ...]]:
    if not block_indices:
        return model, ()
    blocks = getattr(getattr(model, "backbone", None), "net", None)
    blocks = getattr(blocks, "blocks", None)
    if blocks is None:
        raise RuntimeError("model.backbone.net.blocks not found; cannot prune blocks")
    total = len(blocks)
    requested = tuple(block_indices)
    if any(index < 0 or index >= total for index in requested):
        raise ValueError(
            f"invalid block indices: requested={requested}, total_blocks={total}"
        )
    if len(requested) >= total:
        raise ValueError(
            f"cannot drop all blocks: requested={requested}, total_blocks={total}"
        )
    dropped = set(requested)
    model.backbone.net.blocks = nn.ModuleList(
        [block for index, block in enumerate(blocks) if index not in dropped]
    )
    return model, requested


__all__ = ["drop_blocks"]
