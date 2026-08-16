from __future__ import annotations

from experimental.production_min_recall.runtime import minimum_recall_deficit


def test_minimum_recall_deficit_is_zero_at_or_above_floor() -> None:
    assert minimum_recall_deficit(0.97, 0.97) == 0.0
    assert minimum_recall_deficit(0.99, 0.97) == 0.0


def test_minimum_recall_deficit_preserves_each_frame_violation() -> None:
    assert abs(minimum_recall_deficit(0.90, 0.97) - 0.07) < 1e-12


def test_zero_total_deficit_implies_every_frame_passes() -> None:
    recalls = [0.98, 0.97, 0.999]
    assert sum(minimum_recall_deficit(value, 0.97) for value in recalls) == 0.0


def test_average_recall_can_pass_while_minimum_recall_fails() -> None:
    recalls = [1.0, 1.0, 1.0, 0.90]
    assert sum(recalls) / len(recalls) >= 0.97
    assert sum(minimum_recall_deficit(value, 0.97) for value in recalls) > 0.0
