from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experimental/0809/analyze_pair_vote_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("pair_vote_ablation_0809", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_area = MODULE._area
_flatten_matching = MODULE._flatten_matching
_metric_summary = MODULE._metric_summary


def test_pair_vote_geometry_helpers_preserve_vertex_correspondence() -> None:
    before = [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    after = [[[1.0, 0.0], [11.0, 0.0], [11.0, 10.0], [1.0, 10.0]]]
    left, right = _flatten_matching(before, after)

    assert _area(before) == 100.0
    assert _area(after) == 100.0
    assert np.allclose(np.linalg.norm(right - left, axis=1), 1.0)


def test_pair_vote_metric_summary_counts_recall_regressions() -> None:
    before = {
        ("1", 0, 0): {
            "has_keyframe": 1.0,
            "gt_area": 100.0,
            "pred_area": 110.0,
            "recall": 0.98,
            "precision": 0.90,
            "iou": 0.88,
        }
    }
    after = {
        ("1", 0, 0): {
            "has_keyframe": 1.0,
            "gt_area": 100.0,
            "pred_area": 90.0,
            "recall": 0.89,
            "precision": 0.99,
            "iou": 0.88,
        }
    }

    summary = _metric_summary(
        before,
        after,
        recall_floor=0.97,
        keyframes_only=False,
    )

    assert summary["recall_before_violations"] == 0
    assert summary["recall_after_violations"] == 1
    assert summary["area_ratio_before"]["mean"] == 1.1
    assert summary["area_ratio_after"]["mean"] == 0.9
