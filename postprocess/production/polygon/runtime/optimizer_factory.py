#!/usr/bin/env python3
"""Composition root for the parity-frozen Production optimizer adapters."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from .optimizer_adapters import (
    install_artifact_adapters,
    install_geometry_adapters,
    install_native_dp_adapters,
    install_python_dp_adapter,
    install_resource_adapters,
)

POSTPROCESS_ROOT = Path(__file__).resolve().parents[3]


def _point_predictor_model_dir() -> Path:
    candidate = POSTPROCESS_ROOT / "models" / "polygon_point_predictor"
    if not (candidate / "best.pt").is_file():
        raise FileNotFoundError(
            f"Production polygon point predictor is missing: {candidate}"
        )
    return candidate


def build_optimizer_module() -> ModuleType:
    """Load the numerical kernel and install each compatibility concern once."""
    module = importlib.import_module("production.polygon.runtime.optimizer_kernel")
    if bool(getattr(module, "_production_runtime_defaults_applied", False)):
        return module

    try:
        module.cv2.setNumThreads(1)
    except Exception:
        pass

    # Capture every unmodified implementation before installing any adapter.
    original_resample_closed_contour = module.resample_closed_contour
    original_json_dumps = module.json.dumps
    original_get_context = module.multiprocessing.get_context
    original_apply_fixed_practical_defaults = module.apply_fixed_practical_defaults
    original_build_track_streams = module.build_track_streams
    original_load_rows = module.load_rows
    original_run_single_state_penalty_path = module.run_single_state_penalty_path
    original_repair_keyframe_vectors_for_exact_recall = (
        module.repair_keyframe_vectors_for_exact_recall
    )

    module.DEFAULT_ADAPTIVE_ANCHOR_COUNTS = True
    module.DEFAULT_ADAPTIVE_POINT_OFFSET = 10
    module.DEFAULT_MIN_ANCHORS_PER_CONTOUR = 8
    module.DEFAULT_POINT_PREDICTOR_MODEL_DIR = _point_predictor_model_dir()
    module.DEFAULT_PREDICTOR_DEVICE = "cuda"

    install_geometry_adapters(module, original_resample_closed_contour)
    install_python_dp_adapter(module)
    install_native_dp_adapters(
        module,
        original_run_single_state_penalty_path,
        original_repair_keyframe_vectors_for_exact_recall,
    )
    install_artifact_adapters(module, original_json_dumps, original_load_rows)
    install_resource_adapters(
        module,
        original_build_track_streams,
        original_apply_fixed_practical_defaults,
        original_get_context,
    )

    module._production_runtime_defaults_applied = True
    return module


__all__ = ("build_optimizer_module",)
