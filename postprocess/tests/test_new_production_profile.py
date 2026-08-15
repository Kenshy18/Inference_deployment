from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_new_production_profile_freezes_best_v4_for_release_intervals() -> None:
    experiment = Path(__file__).resolve().parents[1] / "experimental/0809"
    sys.path.insert(0, str(experiment))
    try:
        runtime = importlib.import_module("phase2_runtime")
        for interval in (1.0, 3.0, 6.0):
            for label in ("女性器", "男性器", "結合部分"):
                assert runtime._class_role_state_profile(
                    "new_production_v1", label, interval
                ) == runtime._class_role_state_profile(
                    "production_candidate_best_v4", label, interval
                )
    finally:
        sys.path.remove(str(experiment))
