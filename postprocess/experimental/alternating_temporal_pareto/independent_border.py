"""Production-independent border constraints for temporal polygon fitting.

The hard reference remains the unmodified AI observation.  A mask touching a
video border additionally receives two side-local requirements:

* retain the visible mask strip at that border; and
* keep a small off-canvas extent so interpolation cannot pull the mask inward.

No Production keyframe or Production border-transform output is consulted.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from ..polygon_recall_optimizer.fixed_budget import RawMask, _polygonal
from ..polygon_recall_optimizer.superior import (
    BorderExpansionConfig,
    BorderFrameConstraint,
    BorderSideConstraint,
)


def _mask_points(raw: RawMask) -> np.ndarray:
    if raw.component_points:
        return np.concatenate(
            [np.asarray(value, dtype=np.float64) for value in raw.component_points],
            axis=0,
        )
    return np.asarray(raw.primary_points, dtype=np.float64)


def _expansion_amount(span: float, config: BorderExpansionConfig) -> float:
    return float(
        np.clip(
            max(float(span), 1.0) * float(config.expand_ratio),
            float(config.min_expand_px),
            float(config.max_expand_px),
        )
    )


def build_independent_border_constraints(
    raw_masks: dict[tuple[int, str], RawMask],
    *,
    width: int,
    height: int,
    config: BorderExpansionConfig,
    local_recall_floor: float,
) -> tuple[
    dict[tuple[int, str], RawMask],
    dict[tuple[int, str], BorderFrameConstraint],
    dict[str, object],
]:
    """Build side-local safeguards directly from raw observations."""

    if not 0.0 < float(local_recall_floor) <= 1.0:
        raise ValueError("local_recall_floor must be in (0, 1]")
    if not config.enabled:
        return dict(raw_masks), {}, {
            "enabled": False,
            "algorithm": "independent_raw_border_v1",
            "frame_count": 0,
        }

    strips = {
        "left": box(0.0, 0.0, float(config.influence_px), float(height)),
        "right": box(
            float(width) - float(config.influence_px),
            0.0,
            float(width),
            float(height),
        ),
        "top": box(0.0, 0.0, float(width), float(config.influence_px)),
        "bottom": box(
            0.0,
            float(height) - float(config.influence_px),
            float(width),
            float(height),
        ),
    }
    constraints: dict[tuple[int, str], BorderFrameConstraint] = {}
    side_counts = {side: 0 for side in strips}
    expansion_values: list[float] = []
    for identity, raw in raw_masks.items():
        points = _mask_points(raw)
        if len(points) < 3:
            continue
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span_x = float(maximum[0] - minimum[0] + 1.0)
        span_y = float(maximum[1] - minimum[1] + 1.0)
        amount_x = _expansion_amount(span_x, config)
        amount_y = _expansion_amount(span_y, config)
        touched: list[tuple[str, float]] = []
        if float(minimum[0]) <= float(config.trigger_px):
            touched.append(
                (
                    "left",
                    min(
                        float(minimum[0]) - amount_x,
                        -float(config.min_expand_px),
                    ),
                )
            )
        if float(maximum[0]) >= float(width - 1) - float(config.trigger_px):
            touched.append(
                (
                    "right",
                    max(
                        float(maximum[0]) + amount_x,
                        float(width - 1) + float(config.min_expand_px),
                    ),
                )
            )
        if float(minimum[1]) <= float(config.trigger_px):
            touched.append(
                (
                    "top",
                    min(
                        float(minimum[1]) - amount_y,
                        -float(config.min_expand_px),
                    ),
                )
            )
        if float(maximum[1]) >= float(height - 1) - float(config.trigger_px):
            touched.append(
                (
                    "bottom",
                    max(
                        float(maximum[1]) + amount_y,
                        float(height - 1) + float(config.min_expand_px),
                    ),
                )
            )
        sides: list[BorderSideConstraint] = []
        for side, required in touched:
            reference = _polygonal(raw.geometry.intersection(strips[side]))
            area = float(reference.area)
            if area <= 1e-8:
                continue
            sides.append(
                BorderSideConstraint(
                    side=side,
                    visible_reference=reference,
                    visible_area=area,
                    required_coordinate=float(required),
                )
            )
            side_counts[side] += 1
            if side in {"left", "right"}:
                expansion_values.append(amount_x)
            else:
                expansion_values.append(amount_y)
        if not sides:
            continue
        excluded = strips[sides[0].side]
        for side in sides[1:]:
            excluded = excluded.union(strips[side.side])
        constraints[identity] = BorderFrameConstraint(
            sides=tuple(sides),
            local_recall_floor=float(local_recall_floor),
            # Directional repair is not globally dilated by this value.  It is
            # retained for compatible audit/diagnostic helpers.
            max_repair_px=float(config.max_expand_px),
            quality_domain=box(0.0, 0.0, float(width), float(height)).difference(
                excluded
            ),
        )
    return dict(raw_masks), constraints, {
        "enabled": True,
        "algorithm": "independent_raw_border_v1",
        "production_transform_used": False,
        "constraint_reference": "unmodified_raw_mask",
        "frame_count": len(constraints),
        "side_counts": side_counts,
        "local_recall_floor": float(local_recall_floor),
        "mean_required_expansion_px": (
            float(np.mean(expansion_values)) if expansion_values else 0.0
        ),
        "max_required_expansion_px": (
            float(np.max(expansion_values)) if expansion_values else 0.0
        ),
        "parameters": {
            "trigger_px": float(config.trigger_px),
            "expand_ratio": float(config.expand_ratio),
            "min_expand_px": float(config.min_expand_px),
            "max_expand_px": float(config.max_expand_px),
            "influence_px": float(config.influence_px),
        },
    }
