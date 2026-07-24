"""Temporal mask gap filling."""

from .polygon import (
    LinearPolygonInterpolator,
    PolygonInterpolator,
    fill_keyframe_gaps_sqlite,
)


def kffill_main() -> None:
    """Run ellipse gap filling without loading it for polygon pipelines."""

    from .ellipse import kffill_main as _main

    _main()


__all__ = [
    "LinearPolygonInterpolator",
    "PolygonInterpolator",
    "fill_keyframe_gaps_sqlite",
    "kffill_main",
]
