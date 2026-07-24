"""Exact mask evaluation."""

from .mask_iou import evaluate_mask_sqlites


def kfeval_main() -> None:
    """Run ellipse-union evaluation."""

    from .exact import kfeval_main as _main

    _main()


__all__ = ["evaluate_mask_sqlites", "kfeval_main"]
