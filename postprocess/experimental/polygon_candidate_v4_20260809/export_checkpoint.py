#!/usr/bin/env python3
"""Export a validated V4 keyframe checkpoint without rerunning optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental.polygon_candidate_v4_20260809.run_refinement_checkpoint import (
    _load_checkpoint,
)
from experimental.polygon_recall_optimizer.fixed_budget import (
    load_raw_masks,
    load_segments,
)
from experimental.polygon_recall_optimizer.sqlite_export import export_selected_sqlite
from experimental.polygon_recall_optimizer.superior import (
    supported_single_component_segments,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-sqlite", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=8681)
    parser.add_argument("--end-frame", type=int, default=20059)
    parser.add_argument("--label", default="男性器")
    parser.add_argument("--target-interval", type=float, required=True)
    parser.add_argument(
        "--allow-preserved-topology-fallbacks", action="store_true"
    )
    args = parser.parse_args()

    quality = load_raw_masks(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    loaded = load_segments(
        args.source_sqlite,
        label=args.label,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    supported, topology_fallbacks = supported_single_component_segments(loaded)
    if topology_fallbacks and not args.allow_preserved_topology_fallbacks:
        raise RuntimeError(
            f"refusing partial export: {len(topology_fallbacks)} topology fallbacks"
        )
    selected = _load_checkpoint(args.checkpoint, supported)
    quality = {item: raw for item, raw in quality.items() if item[1] in selected}
    result = export_selected_sqlite(
        args.source_sqlite,
        args.output_sqlite,
        selected,
        quality,
        label=args.label,
        target_mean_key_interval=args.target_interval,
        recall_floor=0.97,
        selection_reason="candidate_v4_safe_interior_checkpoint",
        algorithm="experimental.polygon_candidate_v4_20260809",
    )
    result["preserved_topology_fallbacks"] = topology_fallbacks
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
