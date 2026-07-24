"""Framework compatibility patches required before constructing EVA02."""

from __future__ import annotations

import torch


def install_legacy_checkpoint_loading() -> None:
    """Retain pre-PyTorch-2.6 checkpoint loading behavior."""

    try:

        def no_weights_only(_pickle_module=None):
            return False

        torch.serialization._default_to_weights_only = no_weights_only
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata

        torch.serialization.add_safe_globals(
            [DictConfig, ListConfig, ContainerMetadata]
        )
    except Exception:
        pass


def install_sdpa_attention() -> None:
    """Bind Detectron ViT attention to PyTorch SDPA without xFormers."""

    import detectron2.modeling.backbone.vit as vit_module
    import torch.nn.functional as functional

    class SdpaOps:
        @staticmethod
        def memory_efficient_attention(q, k, v):
            query = q.permute(0, 2, 1, 3)
            key = k.permute(0, 2, 1, 3)
            value = v.permute(0, 2, 1, 3)
            output = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            return output.permute(0, 2, 1, 3)

    vit_module.xops = SdpaOps


__all__ = ["install_legacy_checkpoint_loading", "install_sdpa_attention"]
