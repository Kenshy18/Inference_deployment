"""Fast topology checks for densely sampled Production curves.

Production's polygon checker is deliberately scalar because normal production
polygons contain only 14--20 vertices.  A sampled Catmull--Rom boundary has
typically 128--320 vertices, so the same exact strict-crossing predicate is
evaluated after an exact sweep-line bounding-box broad phase.  Collinear
contacts retain Production's current semantics: only proper crossings are
rejected.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _native_topology_module():
    """Load the CPU-only Production native extension when it is available."""
    try:
        module = importlib.import_module("native_interval_metrics")
    except ImportError:
        build = (
            Path(__file__).resolve().parents[2]
            / "polygon"
            / "runtime"
            / "native_interval"
            / "build"
        )
        if not build.exists():
            return None
        value = str(build)
        if value not in sys.path:
            sys.path.insert(0, value)
        try:
            module = importlib.import_module("native_interval_metrics")
        except ImportError:
            return None
    if not hasattr(module, "strict_self_intersection_batch"):
        return None
    return module


def has_strict_self_intersection(points: np.ndarray) -> bool:
    """Match Production's proper-segment-crossing test with an exact broad phase."""
    value = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = int(len(value))
    if count < 4:
        return False
    native = _native_topology_module()
    if native is not None:
        return bool(
            np.asarray(
                native.strict_self_intersection_batch(
                    np.ascontiguousarray(value[None], dtype=np.float64),
                    1,
                ),
                dtype=np.uint8,
            )[0]
        )
    starts = value
    ends = np.roll(value, -1, axis=0)

    # An exact sweep-line broad phase matters for sampled curves: a normal
    # 18-point curve is rendered as 288 short edges, while only neighbouring
    # spatial regions can possibly cross.  Sorting by minimum x and expiring
    # boxes whose maximum x is behind the sweep avoids constructing the full
    # O(E^2) pair arrays.  The final strict orientation test and its tolerance
    # remain byte-for-byte the same algebra as Production's predicate.
    lower = np.minimum(starts, ends)
    upper = np.maximum(starts, ends)
    order = np.argsort(lower[:, 0], kind="stable")
    active: list[int] = []
    for first_value in order:
        first = int(first_value)
        x_minimum = float(lower[first, 0])
        active = [second for second in active if float(upper[second, 0]) >= x_minimum]
        a_x, a_y = starts[first]
        b_x, b_y = ends[first]
        for second in active:
            if (second + 1) % count == first or (first + 1) % count == second:
                continue
            if max(float(lower[first, 1]), float(lower[second, 1])) > min(
                float(upper[first, 1]), float(upper[second, 1])
            ):
                continue
            c_x, c_y = starts[second]
            d_x, d_y = ends[second]
            ab_x = b_x - a_x
            ab_y = b_y - a_y
            cd_x = d_x - c_x
            cd_y = d_y - c_y
            ab_c = ab_x * (c_y - a_y) - ab_y * (c_x - a_x)
            ab_d = ab_x * (d_y - a_y) - ab_y * (d_x - a_x)
            cd_a = cd_x * (a_y - c_y) - cd_y * (a_x - c_x)
            cd_b = cd_x * (b_y - c_y) - cd_y * (b_x - c_x)
            if ab_c * ab_d < -1e-8 and cd_a * cd_b < -1e-8:
                return True
        active.append(first)
    return False


def strict_self_intersection_batch(
    boundaries: np.ndarray,
    *,
    threads: int = 1,
) -> np.ndarray:
    """Evaluate the exact same strict-crossing predicate for many boundaries.

    The native implementation uses double precision and the same stable
    x-sweep, adjacency exclusions and ``-1e-8`` proper-crossing threshold as
    :func:`has_strict_self_intersection`.  Falling back to the scalar Python
    implementation preserves portability without changing decisions.
    """
    values = np.asarray(boundaries, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("boundaries must have shape (cases, points, 2)")
    native = _native_topology_module()
    if native is None:
        return np.asarray(
            [has_strict_self_intersection(value) for value in values],
            dtype=bool,
        )
    return np.asarray(
        native.strict_self_intersection_batch(
            np.ascontiguousarray(values, dtype=np.float64),
            max(1, int(threads)),
        ),
        dtype=bool,
    )


__all__ = ("has_strict_self_intersection", "strict_self_intersection_batch")
