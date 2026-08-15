#!/usr/bin/env python3
"""Run an isolated Production-v22 variant with minimum-frame Recall semantics.

The validated Production source is deliberately left untouched.  This module
loads that exact standalone implementation, applies the smallest semantic
patch needed to change its additive *mean Recall budget* into per-frame Recall
deficits, and then invokes the original entry point.

For a requested floor ``rho`` the original implementation accumulates
``1 - recall_t`` and permits a total budget ``T * (1 - rho)``.  That is an
average-Recall constraint.  This variant instead accumulates
``max(rho - recall_t, 0)`` and permits a total budget of zero.  Consequently a
zero-violation path exists iff every evaluated frame satisfies the floor.

The original candidate construction, frame pool, key-count penalty, polygon
border preparation, pair-vote refinement, interpolation, and export remain
unchanged.  Exact-recall repair is also unchanged except that its lexicographic
comparison receives the minimum per-frame Recall instead of the mean.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
POSTPROCESS_ROOT = HERE.parents[1]
PRODUCTION_RUNTIME = (
    POSTPROCESS_ROOT / "vendor" / "original_polygon" / "original_run_standalone.py"
)


def minimum_recall_deficit(recall: float, floor: float) -> float:
    """Return the non-negative per-frame violation of a minimum Recall floor."""

    return max(float(floor) - float(recall), 0.0)


def _load_production_runtime() -> ModuleType:
    module_name = "experimental_production_min_recall_source"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, PRODUCTION_RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Production runtime: {PRODUCTION_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_embedded_optimizer(module: ModuleType) -> ModuleType:
    if bool(getattr(module, "_minimum_recall_experiment_patched", False)):
        return module

    original_defaults = module.apply_fixed_practical_defaults
    original_exact_metrics = module.exact_interpolated_metrics

    def apply_fixed_practical_defaults(args: argparse.Namespace) -> argparse.Namespace:
        args = original_defaults(args)
        module._minimum_recall_floor = float(args.recall_min)
        module._minimum_recall_semantics = "per_frame_deficit_zero_budget"
        return args

    def recall_budget_from_metrics(metrics: dict[str, float]) -> float:
        floor = float(getattr(module, "_minimum_recall_floor", 0.97))
        return minimum_recall_deficit(float(metrics["recall"]), floor)

    def recall_budget_limit(_frame_count: int, _args: argparse.Namespace) -> float:
        return 0.0

    def recall_violation(total_budget: float, _frame_count: int, _args: argparse.Namespace) -> float:
        return max(float(total_budget), 0.0)

    def exact_interpolated_metrics(*args, **kwargs):
        result = original_exact_metrics(*args, **kwargs)
        rows = result[0]
        minimum = min((float(row["recall"]) for row in rows), default=1.0)
        # The fourth element was mean Recall.  Existing repair code uses that
        # element exclusively as the Recall criterion, so replacing it keeps
        # the repair implementation intact while changing its semantics.
        return result[0], result[1], result[2], minimum, result[4], result[5]

    module.apply_fixed_practical_defaults = apply_fixed_practical_defaults
    module.recall_budget_from_metrics = recall_budget_from_metrics
    module.recall_budget_limit = recall_budget_limit
    module.recall_violation = recall_violation
    module.exact_interpolated_metrics = exact_interpolated_metrics
    module._minimum_recall_experiment_patched = True
    return module


def _write_audit(output_dir: Path, recall_floor: float) -> dict[str, object]:
    metrics_path = output_dir / "exact" / "keyframe_exact_metrics.csv"
    recalls: list[float] = []
    ious: list[float] = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                recalls.append(float(row["recall"]))
                ious.append(float(row["iou"]))
    audit = {
        "schema_version": 1,
        "algorithm": "production_v22_minimum_per_frame_recall_experimental",
        "production_source": str(PRODUCTION_RUNTIME),
        "recall_floor": float(recall_floor),
        "evaluated_rows": len(recalls),
        "minimum_recall": min(recalls, default=1.0),
        "mean_recall": sum(recalls) / max(len(recalls), 1),
        "mean_iou": sum(ious) / max(len(ious), 1),
        "violations": sum(value + 1e-12 < recall_floor for value in recalls),
        "constraint_satisfied": all(
            value + 1e-12 >= recall_floor for value in recalls
        ),
        "semantic_patch": {
            "dp_budget_per_frame": "max(recall_floor - recall, 0)",
            "dp_total_budget_limit": 0.0,
            "exact_repair_criterion": "minimum per-frame recall",
        },
    }
    (output_dir / "minimum_recall_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    source = _load_production_runtime()
    original_builder = source._build_embedded_polygon_v22_module

    def build_patched_module() -> ModuleType:
        return _patch_embedded_optimizer(original_builder())

    source._build_embedded_polygon_v22_module = build_patched_module
    # The optimized Production repair has a delta-evaluation shortcut whose
    # local aggregator reports mean Recall.  Disable only that shortcut so all
    # repair trials pass through the patched exact minimum-Recall evaluator.
    os.environ["ATOSYORI_POLYGON_DISABLE_REPAIR_DELTA"] = "1"
    # The hidden-subcommand dispatch table captured the function, not the
    # builder, so replacing the builder is sufficient for the original CLI.
    source.dispatch_main()

    if len(sys.argv) >= 2 and sys.argv[1] == "__onefile_polygon_optimize":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--recall-min", type=float, default=0.97)
        known, _unknown = parser.parse_known_args(sys.argv[2:])
        audit = _write_audit(known.output_dir, known.recall_min)
        print(json.dumps({"minimum_recall_audit": audit}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
