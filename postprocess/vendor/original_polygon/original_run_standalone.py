#!/usr/bin/env python3
"""Historical import shim for pre-promotion research scripts.

The retired standalone implementation was removed when the adaptive polygon
pipeline was promoted. Deployed code imports the modular Production package
directly; this file only keeps old, non-packaged experiment entry points from
failing immediately.
"""

from __future__ import annotations

import sys

from production.polygon.runtime.optimizer_factory import build_optimizer_module


def _build_embedded_polygon_v22_module():
    """Return the promoted kernel for historical experiment compatibility."""
    return build_optimizer_module()


def dispatch_main() -> None:
    optimizer = build_optimizer_module()
    arguments = (
        sys.argv[2:]
        if len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize"
        else sys.argv[1:]
    )
    previous = sys.argv[:]
    try:
        sys.argv = [previous[0], *arguments]
        optimizer.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    dispatch_main()
